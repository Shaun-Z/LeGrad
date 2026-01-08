# src/legrad/backends/openclip/wrapper.py
from typing import Any, Iterable, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import types

# CopyAttrWrapper is defined in legrad.wrap
from ...wrap import CopyAttrWrapper
from .functional import MultiheadAttention_forward

"""
Description:
 - The wrapper accepts an already constructed open_clip model (e.g., via the model, preprocess = open_clip.create_model_and_transforms("ViT-B-32"). Then pass in model (or model.visual) and wrap)
 - If there is no preprocess, you can pass in tensor type inputs; if you give PIL.Image and there is no preprocess, it will report an error and give you a suggestion.
 """

class OpenCLIPWrapper(CopyAttrWrapper):
    """
    An open-clip-specific derivative of CopyAttrWrapper that provides convenient methods for encode_text / encode_image.
    Important: This wrapper assumes that you are passing in the model of open_clip (typically the model returned by open_clip.create_model_and_transforms).
    passed in. If you don't pass preprocess, make sure that the inputs you pass are already preprocessed tensors.
    This version captures gradients of attention maps.
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)

        # Get patch size
        self.patch_size = self.visual.patch_size[0]
        self.num_heads = self.visual.transformer.resblocks[0].attn.num_heads

        self.reset()
        
        for idx in layer_indices:
            block = self.visual.transformer.resblocks[idx]
            assert hasattr(block, 'attention'), "The block does not have attention attribute."
            block.attention = types.MethodType(OpenCLIPWrapper.__attention_with_weights, block) # Override attention method: `need_weights=True`
            block.attn.forward = types.MethodType(MultiheadAttention_forward, block.attn)   # Override MHA forward method: save attn maps

            block.attn.register_forward_hook(self._save_attn_hook)    # (n, b, d)
            block.register_forward_hook(self._save_block_hook)  # (n, b, d)
    
    def __attention_with_weights(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None

        attn_output, attn_weight = self.attn(
            q_x, k_x, v_x, need_weights=True, attn_mask=attn_mask, average_attn_weights=False
        )
        # self.attn_weight = attn_weight  # Save attention weights for gradient computation
        return attn_output

    def _save_attn_hook(self, module, input, output):   # attn_output: (n, b, d) ; attn_weights: (bsz*num_heads, n, n)
        self.attn_weights.append(output[1]) # gather attention weights (bsz*num_heads, n, n)
    def _save_block_hook(self, module, input, output):
        self.block_outputs.append(output)

    def dot_concept_vectors(self, concept_vectors: torch.Tensor):
        """_summary_
            Call this function before foward.
        Args:
            concept_vectors (torch.Tensor): [batch_size, dim]
        """
        w = h = int(math.sqrt(self.block_outputs[0].shape[0]-1))  # Exclude CLS token
        for i, (block_output, attn_weight) in enumerate(zip(self.block_outputs, self.attn_weights)):
            self.visual.zero_grad()
            inter_feat = block_output.mean(dim=0)    # (batch_size, 768)
            latent_feat = F.normalize(self.visual.ln_post(inter_feat) @ self.visual.proj, dim=-1) # (bsz, 512)

            sim = torch.einsum('b d, m d ->b m', latent_feat, concept_vectors).sum(dim=0)  # (bsz, num_concepts) -> (num_concepts)
            # Compute gradients of sim w.r.t. attn_weight
            eye = torch.eye(sim.numel(), device=sim.device).view(sim.numel(), *sim.shape)

            grad = torch.autograd.grad(outputs=sim, inputs=attn_weight,
                                       grad_outputs=eye,
                                       retain_graph=True,
                                       create_graph=False,
                                       is_grads_batched=True)[0]  # (num_concepts, bsz*num_heads, n, n)
            grad = torch.clamp(grad, min=0.)
            grad = rearrange(grad, 'm (b h) n1 n2 ->(m b) h n1 n2', h=self.num_heads)  # (num_concepts*bsz, num_heads, n, n)

            expl_map = grad.mean(dim=1).mean(dim=1)[...,1:]  # (num_concepts*bsz, n-1) Exclude CLS token
            expl_map = rearrange(expl_map, '(m b) (h w) -> b m h w', m=len(concept_vectors), h=h, w=w)
            expl_map = F.interpolate(expl_map, scale_factor=self.patch_size, mode='bilinear')
            self.maps.append(expl_map.permute(2,3,0,1))  # (H, W, batch_size, num_concepts)

    def aggregate_layerwise_maps(self):
        """Aggregate the stored maps across all requested layers.
        Returns:
            torch.Tensor: Aggregated attention maps of shape [H, W, batch_size, num_concepts]
        """
        if not self.maps:
            raise ValueError("No attention maps stored. Please run a forward pass and compute dot_concept_vectors first.")
        maps = torch.stack(self.maps, dim=0)
        maps = torch.einsum('l h w b m ->  b m h w', maps)

        maps_min = maps.amin(dim=(-2, -1), keepdim=True)
        maps_max = maps.amax(dim=(-2, -1), keepdim=True)
        maps = (maps - maps_min) / (maps_max - maps_min + 1e-8)
        return maps

    def reset(self):
        """Reset the stored results and maps."""
        self.attn_weights = []
        self.block_outputs = []
        self.maps = []