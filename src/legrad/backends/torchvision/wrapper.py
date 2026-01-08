# src/legrad/backends/torchvision/wrapper.py
from typing import Any, List, Optional
import torch
import torch.nn.functional as F
import math
from einops import rearrange
import types

# CopyAttrWrapper is defined in legrad.wrap
from ...wrap import CopyAttrWrapper
from .functional import EncoderBlock_forward_with_attn_weights, MultiheadAttention_forward_batch_first

"""
Description:
 - The wrapper accepts an already constructed torchvision model (e.g., via torchvision.models.vit_b_16). Then pass in model and wrap)
 - If there is no preprocess, you can pass in tensor type inputs; if you give PIL.Image and there is no preprocess, it will report an error and give you a suggestion.
 """


class TorchvisionWrapper(CopyAttrWrapper):
    """
    A torchvision-specific derivative of CopyAttrWrapper that provides convenient methods for forward.
    Important: This wrapper assumes that you are passing in a torchvision model (typically the model returned by torchvision.models.vit_b_16).
    If you don't pass preprocess, make sure that the inputs you pass are already preprocessed tensors.
    This version captures gradients of attention maps, following the same approach as OpenCLIP.
    
    Key characteristics of torchvision ViT:
    - Tensor layout: (B, N, D) - batch first
    - Block access: model.encoder.layers[idx]
    - Normalization: Pre-norm (ln_1 before attn, ln_2 before MLP)
    - Attention: self_attention is MultiheadAttention (same as OpenCLIP)
    - Final layers: model.encoder.ln
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)

        # Get patch size and number of heads
        self.patch_size = model.patch_size
        self.num_heads = model.encoder.layers[0].self_attention.num_heads

        self.reset()
        
        for idx in layer_indices:
            block = self.encoder.layers[idx]
            # Override the MultiheadAttention forward to return attention weights (batch_first version)
            block.self_attention.forward = types.MethodType(MultiheadAttention_forward_batch_first, block.self_attention)
            # Override the EncoderBlock forward to call attention with need_weights=True
            block.forward = types.MethodType(EncoderBlock_forward_with_attn_weights, block)
            
            # Register hooks to save attention weights and block outputs
            block.self_attention.register_forward_hook(self._save_attn_hook)
            block.register_forward_hook(self._save_block_hook)
    
    def _save_attn_hook(self, module, input, output):
        # output is (attn_output, attn_weights)
        # attn_weights: (B*num_heads, N, N) in OpenCLIP format
        self.attn_weights.append(output[1])
    
    def _save_block_hook(self, module, input, output):
        # Block output: (B, N, D) - transpose to (N, B, D) to match OpenCLIP format
        self.block_outputs.append(output.permute(1, 0, 2))

    def dot_concept_vectors(self, concept_vectors: torch.Tensor):
        """Compute concept activation maps using gradient-based approach.
        
        Args:
            concept_vectors (torch.Tensor): [num_concepts, dim] - normalized concept vectors
        """
        w = h = int(math.sqrt(self.block_outputs[0].shape[0] - 1))  # Exclude CLS token
        for i, (block_output, attn_weight) in enumerate(zip(self.block_outputs, self.attn_weights)):
            # Zero gradients
            orig_model = self.original_model()
            orig_model.zero_grad()
            
            # block_output: (N, B, D) where N = 1 + H*W
            inter_feat = block_output.mean(dim=0)    # (B, D)
            # Apply final norm and normalize
            latent_feat = F.normalize(self.encoder.ln(inter_feat), dim=-1)  # (B, D)

            sim = torch.einsum('b d, m d -> b m', latent_feat, concept_vectors).sum(dim=0)  # (B, num_concepts) -> (num_concepts)
            # Compute gradients of sim w.r.t. attn_weight
            eye = torch.eye(sim.numel(), device=sim.device).view(sim.numel(), *sim.shape)

            grad = torch.autograd.grad(outputs=sim, inputs=attn_weight,
                                       grad_outputs=eye,
                                       retain_graph=True,
                                       create_graph=False,
                                       is_grads_batched=True)[0]  # (num_concepts, B*num_heads, N, N)
            grad = torch.clamp(grad, min=0.)
            grad = rearrange(grad, 'm (b h) n1 n2 -> (m b) h n1 n2', h=self.num_heads)  # (num_concepts*B, num_heads, N, N)

            expl_map = grad.mean(dim=1).mean(dim=1)[..., 1:]  # (num_concepts*B, N-1) Exclude CLS token
            expl_map = rearrange(expl_map, '(m b) (h w) -> b m h w', m=len(concept_vectors), h=h, w=w)
            expl_map = F.interpolate(expl_map, scale_factor=self.patch_size, mode='bilinear')
            self.maps.append(expl_map.permute(2, 3, 0, 1))  # (H, W, B, num_concepts)

    def aggregate_layerwise_maps(self):
        """Aggregate the stored maps across all requested layers.
        
        Returns:
            torch.Tensor: Aggregated attention maps of shape [B, num_concepts, H, W]
        """
        if not self.maps:
            raise ValueError("No attention maps stored. Please run a forward pass and compute dot_concept_vectors first.")
        maps = torch.stack(self.maps, dim=0)
        maps = torch.einsum('l h w b m -> b m h w', maps)

        maps_min = maps.amin(dim=(-2, -1), keepdim=True)
        maps_max = maps.amax(dim=(-2, -1), keepdim=True)
        maps = (maps - maps_min) / (maps_max - maps_min + 1e-8)
        return maps

    def reset(self):
        """Reset the stored results and maps."""
        self.attn_weights = []
        self.block_outputs = []
        self.maps = []

    def _get_device_for_call(self, device: Optional[str] = None):
        # Try to get the device from the original model's parameters, otherwise use the passed device or cpu
        orig = self.original_model()
        if device is not None:
            return torch.device(device)
        try:
            # Find the device of the first parameter
            for p in orig.parameters():
                return p.device
        except Exception:
            pass
        return torch.device("cpu")

    def to(self, *args, **kwargs):
        # Move the original model to the target device as well
        orig = self.original_model()
        try:
            if hasattr(orig, "to"):
                orig.to(*args, **kwargs)
        except Exception:
            # Ignore errors when moving the original model, but still try to call the parent class's to
            pass
        # CopyAttrWrapper has no tensor buffers of its own, still call the parent class (it will move parameters registered to the wrapper)
        return super().to(*args, **kwargs)