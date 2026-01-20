
import pytest

import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models import vit_b_16
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from legrad.backends.timm.wrapper import TimmCVWrapper
from legrad.backends.torchvision.wrapper import TorchvisionCVWrapper
from legrad.detect import detect_and_wrap


# ==========================
# TimmCVWrapper Tests
# ==========================

@pytest.fixture
def timm_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.eval()
    return model


def test_timm_cv_wrapper_basic(timm_model):
    wrapper = TimmCVWrapper(timm_model, layer_indices=[2, 5, 8], include_private=False)
    device = wrapper._get_device_for_call()
    assert wrapper is not None
    assert isinstance(device, torch.device)
    assert hasattr(wrapper, 'input_tokens')


@pytest.mark.parametrize("batch_size, layer_indices", [
                                (10, [1, 3, 5]),
                                (2, [0, 4, 7, 11]),
                                ])
def test_timm_cv_feature_extraction(timm_model, batch_size, layer_indices):
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = TimmCVWrapper(timm_model, layer_indices=layer_indices, include_private=False)
    features = wrapper.forward_features(dummy_input)

    assert wrapper.embed_dim == 768
    assert features.shape == (batch_size, 197, 768)
    assert wrapper._requested_hook_indices == layer_indices
    # Check that input_tokens are captured
    assert len(wrapper.input_tokens) == len(layer_indices)


@pytest.mark.parametrize("batch_size, layer_indices, num_concepts", [
                                (2, [0, 11], 3),
                                (3, [0, 5, 11], 2),
                                ])
def test_timm_cv_concept_vectors(timm_model, batch_size, layer_indices, num_concepts):
    concept_vectors = torch.randn(num_concepts, 768)
    concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=-1)

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = TimmCVWrapper(timm_model, layer_indices=layer_indices, include_private=False)
    features = wrapper.forward_features(dummy_input)
    
    # Check that hooks captured data correctly
    assert len(wrapper.block_outputs) == len(layer_indices)
    assert len(wrapper.attn_weights) == len(layer_indices)
    assert len(wrapper.input_tokens) == len(layer_indices)

    wrapper.dot_concept_vectors(concept_vectors)
    maps = torch.stack(wrapper.maps, dim=0)
    assert maps.shape == (len(layer_indices), 224, 224, batch_size, num_concepts)


@pytest.mark.parametrize("layer_indices, num_concepts", [
                                ([0, 11], 2),
                                ([0, 5, 11], 3),
                                ])
def test_timm_cv_aggregate_maps(timm_model, layer_indices, num_concepts):
    concept_vectors = torch.randn(num_concepts, 768)
    concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=-1)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    wrapper = TimmCVWrapper(timm_model, layer_indices=layer_indices, include_private=False)
    
    features = wrapper.forward_features(dummy_input)
    batch_size = 1
    
    wrapper.dot_concept_vectors(concept_vectors)
    maps = torch.stack(wrapper.maps, dim=0)
    assert maps.shape == (len(layer_indices), 224, 224, batch_size, num_concepts)
    
    exp_map = wrapper.aggregate_layerwise_maps()
    assert exp_map.shape == (batch_size, num_concepts, 224, 224)


def test_timm_cv_detect_and_wrap(timm_model):
    """Test that detect_and_wrap works with use_cv_wrapper=True"""
    wrapper = detect_and_wrap(timm_model, prefer='timm', layer_indices=[0, 11], use_cv_wrapper=True)
    assert isinstance(wrapper, TimmCVWrapper)


# ==========================
# TorchvisionCVWrapper Tests
# ==========================

@pytest.fixture
def torchvision_model():
    model = vit_b_16(weights=None)
    model.eval()
    return model


def test_torchvision_cv_wrapper_basic(torchvision_model):
    wrapper = TorchvisionCVWrapper(torchvision_model, layer_indices=[2, 5, 8], include_private=False)
    device = wrapper._get_device_for_call()
    assert wrapper is not None
    assert isinstance(device, torch.device)
    assert hasattr(wrapper, 'input_tokens')


@pytest.mark.parametrize("batch_size, layer_indices", [
                                (10, [1, 3, 5]),
                                (2, [0, 4, 7, 11]),
                                ])
def test_torchvision_cv_feature_extraction(torchvision_model, batch_size, layer_indices):
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = TorchvisionCVWrapper(torchvision_model, layer_indices=layer_indices, include_private=False)
    output = wrapper(dummy_input)

    assert wrapper.hidden_dim == 768
    assert output.shape == (batch_size, 1000)
    assert wrapper._requested_hook_indices == layer_indices
    # Check that input_tokens are captured
    assert len(wrapper.input_tokens) == len(layer_indices)


@pytest.mark.parametrize("batch_size, layer_indices, num_concepts", [
                                (2, [0, 11], 3),
                                (3, [0, 5, 11], 2),
                                ])
def test_torchvision_cv_concept_vectors(torchvision_model, batch_size, layer_indices, num_concepts):
    concept_vectors = torch.randn(num_concepts, 768)
    concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=-1)

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    wrapper = TorchvisionCVWrapper(torchvision_model, layer_indices=layer_indices, include_private=False)
    output = wrapper(dummy_input)
    
    # Check that hooks captured data correctly
    assert len(wrapper.block_outputs) == len(layer_indices)
    assert len(wrapper.attn_weights) == len(layer_indices)
    assert len(wrapper.input_tokens) == len(layer_indices)

    wrapper.dot_concept_vectors(concept_vectors)
    maps = torch.stack(wrapper.maps, dim=0)
    assert maps.shape == (len(layer_indices), 224, 224, batch_size, num_concepts)


@pytest.mark.parametrize("layer_indices, num_concepts", [
                                ([0, 11], 2),
                                ([0, 5, 11], 3),
                                ])
def test_torchvision_cv_aggregate_maps(torchvision_model, layer_indices, num_concepts):
    concept_vectors = torch.randn(num_concepts, 768)
    concept_vectors = torch.nn.functional.normalize(concept_vectors, dim=-1)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    wrapper = TorchvisionCVWrapper(torchvision_model, layer_indices=layer_indices, include_private=False)
    
    output = wrapper(dummy_input)
    batch_size = 1
    
    wrapper.dot_concept_vectors(concept_vectors)
    maps = torch.stack(wrapper.maps, dim=0)
    assert maps.shape == (len(layer_indices), 224, 224, batch_size, num_concepts)
    
    exp_map = wrapper.aggregate_layerwise_maps()
    assert exp_map.shape == (batch_size, num_concepts, 224, 224)


def test_torchvision_cv_detect_and_wrap(torchvision_model):
    """Test that detect_and_wrap works with use_cv_wrapper=True"""
    wrapper = detect_and_wrap(torchvision_model, prefer='torchvision', layer_indices=[0, 11], use_cv_wrapper=True)
    assert isinstance(wrapper, TorchvisionCVWrapper)
