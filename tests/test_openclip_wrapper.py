
import pytest

import requests
from PIL import Image

import torch
import open_clip

from legrad.detect import detect_and_wrap


@pytest.fixture
def model():
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained=False)
    model.eval()
    return model

@pytest.fixture
def preprocess():
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=False)
    return preprocess

@pytest.fixture
def tokenizer():
    return open_clip.get_tokenizer(model_name='ViT-B-16')

def test_openclip_wrapper(model):
    wrapper = detect_and_wrap(model, prefer='openclip', layer_indices=[2, 5, 8])
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
    wrapper = detect_and_wrap(model, prefer='openclip', layer_indices=layer_indices)
    features = wrapper.encode_image(dummy_input)

    assert wrapper.visual.output_dim == 512  # ViT-B-16 output dim
    assert features.shape == (batch_size, 512)
    assert wrapper._requested_hook_indices == layer_indices

@pytest.mark.skipif(True, reason="Requires network access to download image")
@pytest.mark.parametrize("layer_indices, prompts", [
                                ([0,11],["a photo of a cat",
                                         "a photo of a remote controller"]),
                                ])
def test_single_image(model, preprocess, tokenizer, layer_indices, prompts):
    text = tokenizer(prompts)
    text_embeddings = model.encode_text(text, normalize=True)
    assert text_embeddings.shape == (len(prompts), 512)

    wrapper = detect_and_wrap(model, prefer='openclip', layer_indices=layer_indices)
    device = wrapper._get_device_for_call()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)
    assert image.shape == (1, 3, 224, 224)
    batch_size = image.shape[0]

    features = wrapper.encode_image(image)
    assert features.shape == (batch_size, 512)
    attn_weights = torch.stack(wrapper.attn_weights, dim=0)
    assert attn_weights.shape == (len(layer_indices), batch_size*wrapper.num_heads, 197, 197)
    wrapper.dot_concept_vectors(text_embeddings)
    maps = torch.stack(wrapper.maps, dim=0)
    assert maps.shape == (len(layer_indices), 224, 224, batch_size, len(prompts))
    exp_map = wrapper.aggregate_layerwise_maps()
    assert exp_map.shape == (batch_size, len(prompts), 224, 224)
    