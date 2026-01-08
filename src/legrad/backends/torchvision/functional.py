# src/legrad/backends/torchvision/functional.py
from typing import Optional
import torch
import torch.nn as nn


def EncoderBlock_forward_with_attn_weights(
        self,
        input: torch.Tensor,
) -> torch.Tensor:
    """
    Modified forward method for torchvision's EncoderBlock module that captures attention weights.
    
    The key change is to call self_attention with need_weights=True instead of False,
    and to use average_attn_weights=False to get per-head weights.
    """
    torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
    x = self.ln_1(input)
    # Call with need_weights=True and average_attn_weights=False to match OpenCLIP
    x, _ = self.self_attention(x, x, x, need_weights=True, average_attn_weights=False)
    x = self.dropout(x)
    x = x + input

    y = self.ln_2(x)
    y = self.mlp(y)
    return x + y


def MultiheadAttention_forward_batch_first(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Modified forward method for PyTorch's MultiheadAttention that always returns attention weights.
    This version is for batch_first=True (used by torchvision).
    
    Returns:
        tuple: (output, attn_weights)
            - output: (B, N, D) when batch_first=True
            - attn_weights: (B*num_heads, N, N) - consistent with OpenCLIP format
    """
    B, N, D = query.shape
    
    # Use the in_proj_weight and in_proj_bias to compute Q, K, V
    # For self-attention, query = key = value
    if self.in_proj_weight is not None:
        # Combined QKV projection
        w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)
        if self.in_proj_bias is not None:
            b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)
        else:
            b_q = b_k = b_v = None
    else:
        # Separate projections
        w_q = self.q_proj_weight
        w_k = self.k_proj_weight
        w_v = self.v_proj_weight
        b_q = b_k = b_v = None
    
    q = torch.nn.functional.linear(query, w_q, b_q)
    k = torch.nn.functional.linear(key, w_k, b_k)
    v = torch.nn.functional.linear(value, w_v, b_v)
    
    # Reshape for multi-head attention: (B, N, D) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
    head_dim = D // self.num_heads
    q = q.view(B, N, self.num_heads, head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
    k = k.view(B, N, self.num_heads, head_dim).transpose(1, 2)
    v = v.view(B, N, self.num_heads, head_dim).transpose(1, 2)
    
    # Reshape to (B*num_heads, N, head_dim) for bmm
    q = q.reshape(B * self.num_heads, N, head_dim)
    k = k.reshape(B * self.num_heads, N, head_dim)
    v = v.reshape(B * self.num_heads, N, head_dim)
    
    # Compute attention scores
    scale = head_dim ** -0.5
    attn = torch.bmm(q * scale, k.transpose(-2, -1))  # (B*num_heads, N, N)
    
    # Apply attention mask if provided
    if attn_mask is not None:
        attn = attn + attn_mask
    
    attn = attn.softmax(dim=-1)
    attn_weights = attn  # Save attention weights BEFORE dropout
    
    # Apply dropout
    if self.dropout > 0 and self.training:
        attn = torch.nn.functional.dropout(attn, p=self.dropout, training=self.training)
    
    # Compute attention output
    attn_output = torch.bmm(attn, v)  # (B*num_heads, N, head_dim)
    
    # Reshape back: (B*num_heads, N, head_dim) -> (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, D)
    attn_output = attn_output.view(B, self.num_heads, N, head_dim)
    attn_output = attn_output.transpose(1, 2).reshape(B, N, D)
    
    # Output projection
    attn_output = torch.nn.functional.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
    
    return attn_output, attn_weights
