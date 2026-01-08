# %%
"""
Torchvision ViT-B-16 Example Script

This script demonstrates how to use the legrad (Layerwise Concept Capture and Fusion) 
library with a torchvision-implemented ViT-B-16 model.

Unlike OpenCLIP/CLIP models which have text encoders for generating concept vectors,
torchvision models are pure vision models. This example uses concept vectors extracted from
the classification head weights corresponding to ImageNet classes (e.g., "tabby cat").

The torchvision backend now uses the same gradient-based approach as OpenCLIP:
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
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import matplotlib.pyplot as plt
from legrad.detect import detect_and_wrap, wrap_torchvision_preprocess
from legrad.utils import visualize, visualize_layerwise_maps

# %%
# Create torchvision ViT-B-16 model
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model.eval()

# Get the preprocessing transform for the model (standard ImageNet normalization)
preprocess = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Wrap the preprocess to accept arbitrary image size
preprocess = wrap_torchvision_preprocess(preprocess, image_size=224)

# %%
layer_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# %%
# Extract concept vectors from the classification head
# The head weight matrix has shape [num_classes, hidden_dim] = [1000, 768]
# Each row is the weight vector for a class, which can be used as a concept vector

# ImageNet class indices for concepts of interest
# Class 281: tabby cat
# Class 285: Egyptian cat  
# Class 333: hamster
TABBY_CAT_IDX = 281

concept_names = ["tabby cat"]

# Extract concept vector for tabby cat from the classification head
# The weight vector for class i is model.heads[0].weight[i]
concept_vectors = model.heads[0].weight[TABBY_CAT_IDX].unsqueeze(0).detach()  # [1, 768]
concept_vectors = F.normalize(concept_vectors, dim=-1)

# %%
wrapper = detect_and_wrap(model, prefer='torchvision', layer_indices=layer_indices)

# %%
device = wrapper._get_device_for_call()

# %%
# Load a sample image from COCO dataset
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = preprocess(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0).to(device)

# %%
# Forward pass to extract features
output = wrapper(image)
print(f"Model output shape: {output.shape}")

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
visualize_layerwise_maps(image, wrapper.maps, text_prompts=concept_names, mean_std=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

# %%
# Aggregate maps across layers
maps_aggregated = wrapper.aggregate_layerwise_maps()
print(f"Aggregated maps shape: {maps_aggregated.shape}")

# %%
# Visualize aggregated maps
visualize(image, maps_aggregated, text_prompts=concept_names, mean_std=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

# %%
wrapper.reset()  # Clear stored results and maps for next use
