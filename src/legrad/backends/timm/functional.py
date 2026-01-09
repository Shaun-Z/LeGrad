# src/legrad/backends/timm/functional.py
from typing import Optional
import torch
import torch.nn.functional as F

"""
Replace the forward method of timm.layers.attention.Attention with this implementation
to return attention weights for gradient computation.

This is similar to OpenCLIP's functional.py, but adapted for timm's Attention module.
"""


def Attention_forward_with_weights(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Modified forward method for timm's Attention module that returns attention weights.
    
    Args:
        x: Input tensor of shape (B, N, C)
        attn_mask: Optional attention mask
    
    Returns:
        tuple: (output tensor, attention weights)
            - output: (B, N, C) 
            - attn_weights: (B*num_heads, N, N) - same format as OpenCLIP for consistency
    """
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim) each
    q, k = self.q_norm(q), self.k_norm(k)

    # Reshape q, k, v to (B*num_heads, N, head_dim) format like OpenCLIP
    q = q.reshape(B * self.num_heads, N, self.head_dim)  # (B*num_heads, N, head_dim)
    k = k.reshape(B * self.num_heads, N, self.head_dim)  # (B*num_heads, N, head_dim)
    v = v.reshape(B * self.num_heads, N, self.head_dim)  # (B*num_heads, N, head_dim)

    # Compute attention in (B*num_heads, N, N) format
    q_scaled = q * self.scale
    attn = torch.bmm(q_scaled, k.transpose(-2, -1))  # (B*num_heads, N, N)
    
    # Apply attention mask if provided
    if attn_mask is not None:
        # Reshape mask to match
        if attn_mask.dim() == 2:
            attn = attn + attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            attn = attn + attn_mask.reshape(B * self.num_heads, N, N)
        else:
            attn = attn + attn_mask.reshape(B * self.num_heads, N, N)
    
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    # Save the attention weights BEFORE any reshape that would break gradient
    attn_weights = attn  # (B*num_heads, N, N) - already in the right format!
    
    attn_output = torch.bmm(attn, v)  # (B*num_heads, N, head_dim)

    # Reshape back to (B, N, embed_dim)
    # Note: use num_heads * head_dim instead of attn_dim for backward compatibility with older timm versions
    attn_output = attn_output.reshape(B, self.num_heads, N, self.head_dim)
    attn_output = attn_output.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
    attn_output = self.norm(attn_output)
    attn_output = self.proj(attn_output)
    attn_output = self.proj_drop(attn_output)
    
    return attn_output, attn_weights


def Block_forward_with_attn_weights(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Modified forward method for timm's Block module that handles tuple return from attention.
    
    The attention module returns (output, attn_weights), but we only use output for the residual.
    The attn_weights are captured by hooks.
    """
    # self.attn now returns (output, attn_weights)
    attn_output = self.attn(self.norm1(x), attn_mask=attn_mask)
    if isinstance(attn_output, tuple):
        attn_output = attn_output[0]  # Only use the output tensor
    x = x + self.drop_path1(self.ls1(attn_output))
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x
