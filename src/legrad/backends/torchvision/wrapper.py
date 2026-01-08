# src/legrad/backends/torchvision/wrapper.py
from typing import Any, List, Optional
import torch
import torch.nn.functional as F
import math
from einops import rearrange

# CopyAttrWrapper is defined in legrad.wrap
from ...wrap import CopyAttrWrapper

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
    
    Key characteristics of torchvision ViT:
    - Tensor layout: (B, N, D) - same as timm
    - Block access: model.encoder.layers[idx]
    - Normalization: Pre-norm (ln_1 before attn, ln_2 before MLP)
    - MLP components: Sequential with [0]=fc1, [1]=GELU, [2]=Dropout, [3]=fc2
    - Attention: self_attention is MultiheadAttention with out_proj
    - Final layers: model.encoder.ln + model.heads[0] (classification head)
    - GELU: Uses exact GELU (approximate='none')
    """
    def __init__(self, model: Any, layer_indices: Optional[List[int]] = None, include_private: bool = False):
        # The copying behavior is done by the parent class CopyAttrWrapper (tiling the model's attributes)
        super().__init__(model, layer_indices=layer_indices, include_private=include_private)
        
        self.reset()
        
        # Store patch info for later use
        self._patch_size = model.patch_size
        self._hidden_dim = model.hidden_dim
        
        # Register hooks to the specified layers to capture attention outputs
        # torchvision uses pre-norm: x = x + attn(ln_1(x)), x = x + mlp(ln_2(x))
        for idx in layer_indices:
            block = self.encoder.layers[idx]
            
            # Hook on self_attention output (MultiheadAttention returns tuple(output, attn_weights))
            # Note: out_proj hook may not work due to PyTorch's fast path, so we hook on the full self_attention
            block.self_attention.register_forward_hook(self._save_attn_output)  # (b, n, d)
            # Hook on ln_2 to scale by LayerNorm
            block.ln_2.register_forward_hook(self._aggregate_ln2)  # (b, n, d)
            # Hook on MLP fc1 (mlp[0])
            block.mlp[0].register_forward_hook(self._aggregate_fc1)
            # Hook on MLP fc2 (mlp[3])
            block.mlp[3].register_forward_hook(self._aggregate_fc2)  # (b, n, d)
            # Final hook on block output
            block.register_forward_hook(self._finalize_hook)  # (b, n, d)

    def reset(self):
        """Reset the stored results and maps."""
        self.tmp = None
        self.result = []
        self.maps = []
        self.normed_clss = []

    def _save_attn_output(self, module, input, output):
        # torchvision MultiheadAttention returns (output, attn_weights)
        # output: (B, N, D)
        self.tmp = output[0].detach()
    
    def _aggregate_ln2(self, module, input, output):
        # input[0] is the input to ln_2: (B, N, D)
        # We need to account for the LayerNorm scaling
        std = input[0].std(dim=-1).detach()
        self.tmp /= rearrange(std, 'b n -> b n 1')
        self.tmp *= module.weight
    
    def _aggregate_fc1(self, module, input, output):
        # Apply fc1 weight transformation
        self.tmp = self.tmp @ module.weight.T
        # torchvision uses exact GELU: approximate='none'
        # Gradient of exact GELU: GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
        # d/dx GELU(x) = Φ(x) + x * φ(x) where φ is the PDF
        x_ = self.tmp
        # Standard normal CDF and PDF
        sqrt_2 = math.sqrt(2.0)
        phi = 0.5 * (1.0 + torch.erf(x_ / sqrt_2))  # CDF
        pdf = torch.exp(-0.5 * x_ * x_) / math.sqrt(2.0 * math.pi)  # PDF
        grad = phi + x_ * pdf
        self.tmp = self.tmp * grad
    
    def _aggregate_fc2(self, module, input, output):
        # Apply fc2 weight transformation
        self.tmp = self.tmp @ module.weight.T
    
    def _finalize_hook(self, module, input, output):
        # Block output: (B, N, D)
        # Extract CLS token (first token)
        cls = output[:, :1, ...]  # (B, 1, D)
        
        # Apply final LayerNorm scaling
        std = cls.std(dim=-1).detach()
        self.tmp /= rearrange(std, 'b 1 -> b 1 1')
        self.tmp *= self.encoder.ln.weight  # torchvision uses model.encoder.ln as final LayerNorm
        
        # For torchvision ViT classification models, the head is a classification head
        # For concept vectors, we work in the hidden_dim space
        cls_encoded = self.encoder.ln(cls)  # (B, 1, D)
        val = cls_encoded.norm(dim=-1, keepdim=True)  # (B, 1, 1)
        self.tmp /= val
        
        self.normed_clss.append(F.normalize(cls_encoded, dim=-1))
        
        # Transpose to (N, B, D) to match format for dot_concept_vectors
        self.result.append(rearrange(self.tmp, 'b n d -> n b d'))

    def dot_concept_vectors(self, concept_vectors: torch.Tensor):
        """Compute concept activation maps.
        
        Args:
            concept_vectors (torch.Tensor): [num_concepts, dim] - normalized concept vectors
        """
        w = h = int(math.sqrt(self.result[0].shape[0] - 1))  # Exclude CLS token
        for i, res in enumerate(self.result):
            # res: (N, B, D) where N = 1 + H*W (CLS + patches)
            prod = torch.einsum('n b d, m d -> n b m', res, concept_vectors)
            # normed_clss[i]: (B, 1, D) - transpose to (1, B, D)
            normed_cls = rearrange(self.normed_clss[i], 'b 1 d -> 1 b d')
            weight = torch.einsum('n b d, m d -> n b m', normed_cls, concept_vectors)
            prod = prod * weight
            map = torch.clamp(prod - prod.mean(dim=0, keepdim=True), min=0.)  # negative gradient
            map = rearrange(map[1:, ...], '(h w) b m -> h w b m', h=h, w=w)  # Exclude CLS token
            self.maps.append(map)

    def aggregate_layerwise_maps(self):
        """Aggregate the stored maps across all requested layers.
        
        Returns:
            torch.Tensor: Aggregated attention maps of shape [B, num_concepts, H, W]
        """
        if not self.maps:
            raise ValueError("No attention maps stored. Please run a forward pass and compute dot_concept_vectors first.")
        maps = torch.stack(self.maps, dim=0)
        maps = torch.einsum('l h w b m -> h w b m', maps)
        maps = rearrange(maps, 'h w b m -> b m h w')
        
        maps = (maps - maps.min()) / (maps.max() - maps.min() + 1e-8)
        maps = F.interpolate(maps, scale_factor=self._patch_size, mode='bilinear')
        return maps

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