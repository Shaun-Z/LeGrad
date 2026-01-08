
import pytest

import torch
from torchvision.models import vit_b_16
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from legrad.detect import detect_and_wrap


@pytest.fixture
def model():
    model = vit_b_16(weights=None)
    model.eval()
    return model

@pytest.fixture
def preprocess():
    # Standard ImageNet normalization for torchvision ViT
    return Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def test_torchvision_wrapper(model):
    wrapper = detect_and_wrap(model, prefer='torchvision', layer_indices=[2, 5, 8])
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
    wrapper = detect_and_wrap(model, prefer='torchvision', layer_indices=layer_indices)
    output = wrapper(dummy_input)

    assert wrapper.hidden_dim == 768  # ViT-B-16 hidden dim
    assert output.shape == (batch_size, 1000)  # Classification output
    assert wrapper._requested_hook_indices == layer_indices


@pytest.mark.parametrize("batch_size, layer_indices", [
                                (10, [0, 11]),
                                ])
def test_hooks(model, batch_size, layer_indices):
    # Ensure that the wrapper works with the vision transformer architecture
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='torchvision', layer_indices=layer_indices)
    output = wrapper(dummy_input)

    # result is transposed to (N, B, D) format in torchvision wrapper
    assert torch.stack(wrapper.result, dim=0).shape == (len(layer_indices), 197, batch_size, 768)


@pytest.mark.parametrize("batch_size, layer_indices, num_concepts", [
                                (10, [0, 11], 10),
                                (5, [0, 5, 11], 3),
                                ])
def test_concept_vectors(model, batch_size, layer_indices, num_concepts):
    # Ensure that the wrapper works with the vision transformer architecture
    # For torchvision, we use random concept vectors in the hidden_dim space
    concept_vectors = torch.randn(num_concepts, 768)
    concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=-1)

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='torchvision', layer_indices=layer_indices)
    output = wrapper(dummy_input)
    assert torch.stack(wrapper.result, dim=0).shape == (len(layer_indices), 197, batch_size, 768)

    wrapper.dot_concept_vectors(concept_vectors)
    assert torch.stack(wrapper.maps, dim=0).shape == (len(layer_indices), 14, 14, batch_size, num_concepts)


@pytest.mark.parametrize("layer_indices, num_concepts", [
                                ([0, 11], 2),
                                ([0, 3, 6, 9, 11], 5),
                                ])
def test_aggregate_maps(model, layer_indices, num_concepts):
    # Test aggregation of layerwise maps
    concept_vectors = torch.randn(num_concepts, 768)
    concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=-1)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    wrapper = detect_and_wrap(model, prefer='torchvision', layer_indices=layer_indices)
    
    output = wrapper(dummy_input)
    assert torch.stack(wrapper.result, dim=0).shape == (len(layer_indices), 197, 1, 768)
    
    wrapper.dot_concept_vectors(concept_vectors)
    assert torch.stack(wrapper.maps, dim=0).shape == (len(layer_indices), 14, 14, 1, num_concepts)
    
    maps = wrapper.aggregate_layerwise_maps()
    # Aggregated maps should be (B, num_concepts, H*patch_size, W*patch_size)
    # With patch_size=16 and grid_size=14, output should be (1, num_concepts, 224, 224)
    assert maps.shape == (1, num_concepts, 224, 224)
