# src/legrad/backends/timm/wrapper.py
from typing import Any, List, Optional
import torch
import torch.nn.functional as F
import math
from einops import rearrange
import types

# CopyAttrWrapper is defined in legrad.wrap
from ...wrap import CopyAttrWrapper
from .functional import Attention_forward_with_weights, Block_forward_with_attn_weights

"""
Description:
 - The wrapper accepts an already constructed timm model (e.g., via timm.create_model). Then pass in model and wrap)
 - If there is no preprocess, you can pass in tensor type inputs; if you give PIL.Image and there is no preprocess, it will report an error and give you a suggestion.
 """


class TimmWrapper(CopyAttrWrapper):
    """
    A timm-specific derivative of CopyAttrWrapper that provides convenient methods for forward_features.
    Important: This wrapper assumes that you are passing in a timm model (typically the model returned by timm.create_model).
    If you don't pass preprocess, make sure that the inputs you pass are already preprocessed tensors.
    This version captures gradients of attention maps, following the same approach as OpenCLIP.
    
    Key differences from OpenCLIP:
    - Tensor layout: timm uses (B, N, D) vs OpenCLIP uses (N, B, D)
    - Block access: model.blocks[idx] vs model.visual.transformer.resblocks[idx]
    - Normalization: Pre-norm (norm1 before attn, norm2 before MLP)
    - Attention: timm uses custom Attention class with combined qkv
    - Final layers: model.norm vs ln_post + proj
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)

        # Get patch size and number of heads
        self.patch_size = model.patch_embed.proj.kernel_size[0]
        self.num_heads = model.blocks[0].attn.num_heads

        self.reset()
        
        for idx in layer_indices:
            block = self.blocks[idx]
            # Override attention forward to return attention weights
            block.attn.forward = types.MethodType(Attention_forward_with_weights, block.attn)
            # Override block forward to handle tuple return from attention
            block.forward = types.MethodType(Block_forward_with_attn_weights, block)
            
            # Register hooks to save attention weights and block outputs
            block.attn.register_forward_hook(self._save_attn_hook)    # (B, N, D), attn_weights: (B*num_heads, N, N)
            block.register_forward_hook(self._save_block_hook)  # (B, N, D)
    
    def _save_attn_hook(self, module, input, output):
        # output is now a tuple: (attn_output, attn_weights)
        # attn_weights: (B*num_heads, N, N)
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
            latent_feat = F.normalize(self.norm(inter_feat), dim=-1)  # (B, D)

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


class TimmCVWrapper(CopyAttrWrapper):
    """
    Chained-Vector variant of TimmWrapper that propagates concept vectors through layers.
    
    In this variant:
    - The concept_vectors passed in are only used for the deepest (last) layer
    - For each layer, gradients are computed w.r.t. both attention weights and input tokens
    - The input tokens' gradient from layer i becomes the concept vectors for layer i-1
    - This creates a chain: deepest layer uses original concept_vectors → compute gradients 
      → tokens' grad becomes concept for previous layer → repeat
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)

        # Get patch size and number of heads
        self.patch_size = model.patch_embed.proj.kernel_size[0]
        self.num_heads = model.blocks[0].attn.num_heads

        self.reset()
        
        for idx in layer_indices:
            block = self.blocks[idx]
            # Override attention forward to return attention weights
            block.attn.forward = types.MethodType(Attention_forward_with_weights, block.attn)
            # Override block forward to handle tuple return from attention
            block.forward = types.MethodType(Block_forward_with_attn_weights, block)
            
            # Register hooks to save attention weights, input tokens, and block outputs
            block.attn.register_forward_hook(self._save_attn_hook)
            block.register_forward_hook(self._save_block_hook)
    
    def _save_attn_hook(self, module, input, output):
        # output is (attn_output, attn_weights)
        # attn_weights: (B*num_heads, N, N)
        self.attn_weights.append(output[1])
    
    def _save_block_hook(self, module, input, output):
        # Block output: (B, N, D) - transpose to (N, B, D) to match format
        self.block_outputs.append(output.permute(1, 0, 2))
        # Save block input (not attention input) to form gradient chain from output to input
        # input[0] is the block input: (B, N, D) - keep as-is to preserve gradient connection
        self.input_tokens.append(input[0])

    def dot_concept_vectors(self, concept_vectors: torch.Tensor):
        """Compute concept activation maps with chained gradient flow.
        
        The concept_vectors are only used in the deepest layer. For shallower layers,
        the concept vectors are derived from the input tokens' gradient of the deeper layer.
        
        Args:
            concept_vectors (torch.Tensor): [num_concepts, dim] - normalized concept vectors
        """
        w = h = int(math.sqrt(self.block_outputs[0].shape[0] - 1))  # Exclude CLS token
        
        # Process layers from deepest to shallowest (reverse order)
        num_layers = len(self.block_outputs)
        current_concept_vectors = concept_vectors
        
        for i in range(num_layers - 1, -1, -1):
            block_output = self.block_outputs[i]
            attn_weight = self.attn_weights[i]
            input_token = self.input_tokens[i]
            
            orig_model = self.original_model()
            orig_model.zero_grad()
            
            # block_output: (N, B, D) where N = 1 + H*W
            latent_feat = F.normalize(self.norm(block_output), dim=-1)  # (N, B, D)
            
            # Compute similarity
            sim = torch.einsum('n b d, m d -> n b m', latent_feat, current_concept_vectors)
            sim = sim.sum(dim=0)  # (B, M)
            sim = sim.sum(dim=0)  # (M,)
            
            eye = torch.eye(sim.numel(), device=sim.device).view(sim.numel(), *sim.shape)
            
            grads = torch.autograd.grad(
                outputs=sim,
                inputs=[attn_weight, input_token],
                grad_outputs=eye,
                retain_graph=True,
                create_graph=False,
                is_grads_batched=True
            )
            
            attn_grad = grads[0]  # (num_concepts, B*num_heads, N, N)
            token_grad = grads[1]  # (num_concepts, B, N, D) - timm uses (B, N, D) layout
            
            # Process attention gradient for explanation map
            attn_grad = torch.clamp(attn_grad, min=0.)
            attn_grad = rearrange(attn_grad, 'm (b h) n1 n2 -> (m b) h n1 n2', h=self.num_heads)
            
            expl_map = attn_grad.mean(dim=1).mean(dim=1)[..., 1:]  # Exclude CLS token
            expl_map = rearrange(expl_map, '(m b) (h w) -> b m h w', m=len(current_concept_vectors), h=h, w=w)
            expl_map = F.interpolate(expl_map, scale_factor=self.patch_size, mode='bilinear')
            
            # Insert at beginning since we're processing in reverse order
            self.maps.insert(0, expl_map.permute(2, 3, 0, 1))
            
            # Prepare concept vectors for the next (shallower) layer
            if i > 0:
                # token_grad: (num_concepts, B, N, D) - average over batch and sequence
                next_concept = token_grad.mean(dim=1).mean(dim=1)  # (num_concepts, D)
                current_concept_vectors = F.normalize(next_concept, dim=-1)

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
        self.input_tokens = []
        self.maps = []
