
import pytest

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from legrad.detect import detect_and_wrap


@pytest.fixture
def model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.eval()
    return model

@pytest.fixture
def preprocess():
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return transform


def test_timm_wrapper(model):
    wrapper = detect_and_wrap(model, prefer='timm', layer_indices=[2, 5, 8])
    device = wrapper._get_device_for_call()
    assert wrapper is not None
    assert isinstance(device, torch.device)


@pytest.mark.parametrize("batch_size, layer_indices", [
                                (10, [1, 3, 5]),
                                (2, [0, 4, 7, 11]),
                                (3, [])
                                ])
def test_feature_extraction(model, batch_size, layer_indices):
    # Test that we can extract features from a dummy input
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='timm', layer_indices=layer_indices)
    features = wrapper.forward_features(dummy_input)

    assert wrapper.embed_dim == 768  # ViT-B-16 embed dim
    assert features.shape == (batch_size, 197, 768)  # (B, N, D) with CLS token
    assert wrapper._requested_hook_indices == layer_indices


@pytest.mark.parametrize("batch_size, layer_indices", [
                                (10, [0, 11]),
                                ])
def test_hooks(model, batch_size, layer_indices):
    # Ensure that the wrapper works with the vision transformer architecture
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='timm', layer_indices=layer_indices)
    features = wrapper.forward_features(dummy_input)

    # block_outputs is transposed to (N, B, D) format in timm wrapper
    assert torch.stack(wrapper.block_outputs, dim=0).shape == (len(layer_indices), 197, batch_size, 768)
    # attn_weights shape: (num_layers, B*num_heads, N, N)
    assert torch.stack(wrapper.attn_weights, dim=0).shape == (len(layer_indices), batch_size*wrapper.num_heads, 197, 197)


@pytest.mark.parametrize("batch_size, layer_indices, num_concepts", [
                                (10, [0, 11], 10),
                                (5, [0, 5, 11], 3),
                                ])
def test_concept_vectors(model, batch_size, layer_indices, num_concepts):
    # Ensure that the wrapper works with the vision transformer architecture
    # For timm, we use random concept vectors in the embed_dim space
    concept_vectors = torch.randn(num_concepts, 768)
    concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=-1)

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='timm', layer_indices=layer_indices)
    features = wrapper.forward_features(dummy_input)
    # Check that attn_weights are captured
    attn_weights = torch.stack(wrapper.attn_weights, dim=0)
    assert attn_weights.shape == (len(layer_indices), batch_size*wrapper.num_heads, 197, 197)

    wrapper.dot_concept_vectors(concept_vectors)
    maps = torch.stack(wrapper.maps, dim=0)
    assert maps.shape == (len(layer_indices), 224, 224, batch_size, num_concepts)


@pytest.mark.parametrize("layer_indices, num_concepts", [
                                ([0, 11], 2),
                                ([0, 3, 6, 9, 11], 5),
                                ])
def test_aggregate_maps(model, layer_indices, num_concepts):
    # Test aggregation of layerwise maps
    concept_vectors = torch.randn(num_concepts, 768)
    concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=-1)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='timm', layer_indices=layer_indices)
    
    features = wrapper.forward_features(dummy_input)
    batch_size = 1
    # Check that attn_weights are captured
    attn_weights = torch.stack(wrapper.attn_weights, dim=0)
    assert attn_weights.shape == (len(layer_indices), batch_size*wrapper.num_heads, 197, 197)
    
    wrapper.dot_concept_vectors(concept_vectors)
    maps = torch.stack(wrapper.maps, dim=0)
    assert maps.shape == (len(layer_indices), 224, 224, batch_size, num_concepts)
    
    exp_map = wrapper.aggregate_layerwise_maps()
    # Aggregated maps should be (B, num_concepts, H*patch_size, W*patch_size)
    # With patch_size=16 and grid_size=14, output should be (1, num_concepts, 224, 224)
    assert exp_map.shape == (batch_size, num_concepts, 224, 224)
