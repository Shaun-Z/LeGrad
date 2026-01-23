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
        maps = (maps - maps_min) / (maps_max - maps_min)
        return maps

    def reset(self):
        """Reset the stored results and maps."""
        self.attn_weights = []
        self.block_outputs = []
        self.maps = []


class OpenCLIPCVWrapper(CopyAttrWrapper):
    """
    Chained-Vector variant of OpenCLIPWrapper that propagates concept vectors through layers.
    
    In this variant:
    - The concept_vectors passed in are only used for the deepest (last) layer
    - For each layer, gradients are computed w.r.t. both attention weights and input tokens
    - The input tokens' gradient from layer i becomes the concept vectors for layer i-1
    - This creates a chain: deepest layer uses original concept_vectors → compute gradients 
      → tokens' grad becomes concept for previous layer → repeat
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)

        # Get patch size
        self.patch_size = self.visual.patch_size[0]
        self.num_heads = self.visual.transformer.resblocks[0].attn.num_heads

        self.reset()
        
        for idx in layer_indices:
            block = self.visual.transformer.resblocks[idx]
            assert hasattr(block, 'attention'), "The block does not have attention attribute."
            block.attention = types.MethodType(OpenCLIPCVWrapper.__attention_with_weights, block)
            block.attn.forward = types.MethodType(MultiheadAttention_forward, block.attn)

            block.attn.register_forward_hook(self._save_attn_hook)
            block.register_forward_hook(self._save_block_hook)
    
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
        return attn_output

    def _save_attn_hook(self, module, input, output):
        self.attn_weights.append(output[1])
    
    def _save_block_hook(self, module, input, output):
        self.block_outputs.append(output)
        # Save block input (not attention input) to form gradient chain from output to input
        # input[0] is the block input: (N, B, D) for OpenCLIP
        self.input_tokens.append(input[0])

    def _compute_deepest_layer_similarity(self, block_output: torch.Tensor, 
                                          concept_vectors: torch.Tensor) -> torch.Tensor:
        """Compute similarity for the deepest layer using projected 512 dim space.
        
        Args:
            block_output: (N, B, 768) - block output in internal space
            concept_vectors: (M, 512) - concept vectors in projected space
            
        Returns:
            sim: (M,) - similarity scores summed over batch
        """
        inter_feat = block_output.mean(dim=0)  # (B, 768)
        latent_feat = F.normalize(self.visual.ln_post(inter_feat) @ self.visual.proj, dim=-1)  # (B, 512)
        sim = torch.einsum('b d, m d -> b m', latent_feat, concept_vectors)
        return sim.sum(dim=0)  # (M,) - sum over batch dimension
    
    def _compute_shallow_layer_similarity(self, block_output: torch.Tensor,
                                          concept_vectors: torch.Tensor) -> torch.Tensor:
        """Compute similarity for shallower layers using 768 dim internal space.
        
        Args:
            block_output: (N, B, 768) - block output in internal space
            concept_vectors: (M, 768) - concept vectors in internal space
            
        Returns:
            sim: (M,) - similarity scores summed over batch and sequence
        """
        latent_feat = F.normalize(self.visual.ln_post(block_output), dim=-1)  # (N, B, 768)
        sim = torch.einsum('n b d, m d -> n b m', latent_feat, concept_vectors)
        sim = sim.sum(dim=0)  # (B, M) - sum over sequence dimension
        return sim.sum(dim=0)  # (M,) - sum over batch dimension

    def dot_concept_vectors(self, concept_vectors: torch.Tensor):
        """Compute concept activation maps with chained gradient flow.
        
        The concept_vectors are only used in the deepest layer. For shallower layers,
        the concept vectors are derived from the input tokens' gradient of the deeper layer.
        
        For OpenCLIP:
        - Deepest layer: concept_vectors should be in 512 dim (projected space)
        - After computing token gradients (768 dim), these become concepts for shallower layers
        - Shallower layers: concepts are in 768 dim (internal space)
        
        Args:
            concept_vectors (torch.Tensor): [num_concepts, dim] - normalized concept vectors (512 dim for CLIP)
        """
        w = h = int(math.sqrt(self.block_outputs[0].shape[0] - 1))  # Exclude CLS token
        
        # Process layers from deepest to shallowest (reverse order)
        num_layers = len(self.block_outputs)
        current_concept_vectors = concept_vectors  # Start with the provided concept vectors
        is_first_layer = True  # Deepest layer flag
        
        for i in range(num_layers - 1, -1, -1):
            block_output = self.block_outputs[i]
            attn_weight = self.attn_weights[i]
            input_token = self.input_tokens[i]
            
            self.visual.zero_grad()
            
            # Compute similarity using appropriate method based on layer depth
            if is_first_layer:
                sim = self._compute_deepest_layer_similarity(block_output, current_concept_vectors)
            else:
                sim = self._compute_shallow_layer_similarity(block_output, current_concept_vectors)
            
            # Compute gradients w.r.t. attn_weight and input_token
            eye = torch.eye(sim.numel(), device=sim.device).view(sim.numel(), *sim.shape)
            
            grads = torch.autograd.grad(
                outputs=sim, 
                inputs=[attn_weight, input_token],
                grad_outputs=eye,
                retain_graph=True,
                create_graph=False,
                is_grads_batched=True
            )
            
            attn_grad = grads[0]  # (num_concepts, bsz*num_heads, N, N)
            token_grad = grads[1]  # (num_concepts, N, B, D)
            
            # Process attention gradient for explanation map
            attn_grad = torch.clamp(attn_grad, min=0.)
            attn_grad = rearrange(attn_grad, 'm (b h) n1 n2 -> (m b) h n1 n2', h=self.num_heads)
            
            expl_map = attn_grad.mean(dim=1).mean(dim=1)[..., 1:]  # (num_concepts*bsz, n-1) Exclude CLS token
            expl_map = rearrange(expl_map, '(m b) (h w) -> b m h w', m=len(current_concept_vectors), h=h, w=w)
            expl_map = F.interpolate(expl_map, scale_factor=self.patch_size, mode='bilinear')
            
            # Insert at beginning since we're processing in reverse order
            self.maps.insert(0, expl_map.permute(2, 3, 0, 1))  # (H, W, batch_size, num_concepts)
            
            # Prepare concept vectors for the next (shallower) layer
            # token_grad: (num_concepts, N, B, D) where D=768 - use mean over batch and normalize
            if i > 0:
                # Average over batch and sequence, normalize for next layer
                next_concept = token_grad.mean(dim=1).mean(dim=1)  # (num_concepts, 768)
                current_concept_vectors = F.normalize(next_concept, dim=-1)
            
            is_first_layer = False

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
        maps = (maps - maps_min) / (maps_max - maps_min)
        return maps

    def reset(self):
        """Reset the stored results and maps."""
        self.attn_weights = []
        self.block_outputs = []
        self.input_tokens = []
        self.maps = []