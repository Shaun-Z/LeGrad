# %%
"""
Timm ViT-B-16 Example Script

This script demonstrates how to use the legrad (Layerwise Concept Capture and Fusion) 
library with a timm-implemented ViT-B-16 model.

Unlike OpenCLIP/CLIP models which have text encoders for generating concept vectors,
timm models are pure vision models. This example uses concept vectors extracted from
the classification head weights corresponding to ImageNet classes (e.g., "tabby cat").

The timm backend now uses the same gradient-based approach as OpenCLIP:
- Captures attention weights from specified layers
- Computes gradients of similarity w.r.t. attention weights
- Generates explanation maps in (H, W, B, num_concepts) format
"""

# %%
import requests
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import matplotlib.pyplot as plt
from legrad.detect import detect_and_wrap, wrap_timm_preprocess
from legrad.utils import visualize, visualize_layerwise_maps

# %%
# Create timm ViT-B-16 model
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# Get the preprocessing transform for the model
config = resolve_data_config({}, model=model)
preprocess = create_transform(**config)
# Wrap the preprocess to accept arbitrary image size
preprocess = wrap_timm_preprocess(preprocess, image_size=224)

# %%
layer_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# %%
# Extract concept vectors from the classification head
# The head weight matrix has shape [num_classes, embed_dim] = [1000, 768]
# Each row is the weight vector for a class, which can be used as a concept vector

# ImageNet class indices for concepts of interest
# Class 281: tabby cat
# Class 285: Egyptian cat  
# Class 333: hamster
TABBY_CAT_IDX = 281

concept_names = ["tabby cat"]

# Extract concept vector for tabby cat from the classification head
# The weight vector for class i is model.head.weight[i]
concept_vectors = model.head.weight[TABBY_CAT_IDX].unsqueeze(0).detach()  # [1, 768]
concept_vectors = F.normalize(concept_vectors, dim=-1)

# %%
wrapper = detect_and_wrap(model, prefer='timm', layer_indices=layer_indices)

# %%
device = wrapper._get_device_for_call()

# %%
# Load a sample image from COCO dataset
urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000001675.jpg",
]
images = [preprocess(Image.open(requests.get(url, stream=True).raw)) for url in urls]
image_batch = torch.stack(images, dim=0).to(device)

# %%
# Forward pass to extract features
features = wrapper.forward_features(image_batch)
print(f"Features shape: {features.shape}")

# %%
# Check captured attention weights from specified layers
attn_weights = torch.stack(wrapper.attn_weights, dim=0)
print(f"Attention weights shape: {attn_weights.shape}")  # (num_layers, B*num_heads, N, N)

# %%
# Compute concept activation maps using gradient-based approach
wrapper.dot_concept_vectors(concept_vectors)

# %%
maps = torch.stack(wrapper.maps, dim=0)  # (num_layers, H, W, B, num_concepts)
print(f"Maps shape: {maps.shape}")

# %%
# Visualize layerwise maps
visualize_layerwise_maps(image_batch, wrapper.maps, text_prompts=concept_names, mean_std=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

# %%
# Aggregate maps across layers
maps_aggregated = wrapper.aggregate_layerwise_maps()
print(f"Aggregated maps shape: {maps_aggregated.shape}")

# %%
# Visualize aggregated maps
visualize(image_batch, maps_aggregated, text_prompts=concept_names, mean_std=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

# %%
wrapper.reset()  # Clear stored results and maps for next use
