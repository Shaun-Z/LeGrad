# %%
"""
OpenCLIP ViT-B-16 Batch Example Script

This script demonstrates how to use the legrad (Layerwise Concept Capture and Fusion)
library with an OpenCLIP ViT-B-16 model on a batch of COCO images.

The OpenCLIP backend uses a gradient-based approach:
- Captures attention weights from specified layers
- Computes gradients of similarity w.r.t. attention weights
- Generates explanation maps in (H, W, B, num_concepts) format
"""
import requests
from PIL import Image
import torch
import open_clip
from legrad.detect import detect_and_wrap, wrap_clip_preprocess
from legrad.utils import visualize, visualize_layerwise_maps
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

# %%
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
model.eval()
preprocess = wrap_clip_preprocess(preprocess, image_size=224)
tokenizer = open_clip.get_tokenizer(model_name='ViT-B-16')

# %%
layer_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# %%
prompts = ['a photo of a cat', 'a photo of a remote control', 'a photo of a laptop']

# %%
wrapper = detect_and_wrap(model, prefer='openclip', layer_indices=layer_indices)

# %%
device = wrapper._get_device_for_call()

# %%
text = tokenizer(prompts)
text_embeddings = model.encode_text(text, normalize=True)

# %%
urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000001675.jpg",
]
images = [preprocess(Image.open(requests.get(url, stream=True).raw)) for url in urls]
image_batch = torch.stack(images, dim=0).to(device)

# %%
features = wrapper.encode_image(image_batch)
print(f"Features shape: {features.shape}")

# %%
attn_weights = torch.stack(wrapper.attn_weights, dim=0)
print(f"Attention weights shape: {attn_weights.shape}")

# %%
wrapper.dot_concept_vectors(text_embeddings)

# %%
maps = torch.stack(wrapper.maps, dim=0)

# %%
visualize_layerwise_maps(image_batch, wrapper.maps, text_prompts=prompts, mean_std=(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD))

# %%
maps_aggregated = wrapper.aggregate_layerwise_maps()

# %%
visualize(image_batch, maps_aggregated, text_prompts=prompts, mean_std=(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD))

# %%
wrapper.reset()
