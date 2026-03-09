import torch
import math
from test_utils import MockInferState
from kernel.flash_decoding import (
    token_decode_attention_flash_decoding, 
    token_decode_attention_flash_latent_per_head_decoding_rope,
    token_decode_attention_flash_int8_latent_per_head_decoding_rope
)
import torch.nn as nn

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm implementation (LLaMA style, no bias).

    Args:
        x: Input tensor with shape (..., hidden_size).
        weight: Learnable scale parameter with shape (hidden_size,).
        eps: Numerical stability term.

    Returns:
        Normalized and scaled tensor with the same shape as x.
    """
    original_dtype = x.dtype
    x_float = x.to(torch.float32)
    # Compute root mean square norm
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_float * torch.rsqrt(variance + eps)
    # Scale and cast back to the original dtype
    return (x_norm.to(weight.dtype) * weight).to(original_dtype)

def input_layernorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm before attention (pre-attention norm)."""
    return _rms_norm(x, weight, eps)

def post_attention_layernorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm after attention (pre-norm before FFN)."""
    return _rms_norm(x, weight, eps)

def create_test_data_layer(batch_size=2, q_head_num=40, head_dim=128, seq_len=2048, rank_ratio=0.9, kv_head_num=40, intermediate_dim=128*40*3):
    """Create test data."""
    device = "cuda"
    dtype = torch.float16
    
    q_len = 1
    
    rank_head = int(rank_ratio * head_dim)
    H_q = q_head_num
    H_kv = kv_head_num
    D = head_dim
    R = rank_head
    model_dim = H_q * D
    rank = H_kv * R
    
    # Create input (batch_size, q_len, q_head_num * head_dim)
    input = torch.randn(batch_size, q_len, model_dim, device=device, dtype=dtype)
    
    # Create latent cache (batch_size, seq_len, rank) - not split by head
    cache_latent_k = torch.randn(batch_size, seq_len, rank, device=device, dtype=dtype)
    cache_latent_v = torch.randn(batch_size, seq_len, rank, device=device, dtype=dtype)

    # Create weight matrices
    Wdq = torch.randn(model_dim, H_q * R, device=device, dtype=dtype) # (H_q, model_dim, rank_head)
    Wdk = torch.randn(model_dim, H_kv * R, device=device, dtype=dtype) # (H_kv, model_dim, rank_head)
    Wdv = torch.randn(model_dim, H_kv * R, device=device, dtype=dtype) # (H_kv, model_dim, rank_head)
    
    Wuq = torch.randn(rank_head, H_q * D, device=device, dtype=dtype) # (H_q, rank_head, head_dim)
    Wuk = torch.randn(rank_head, H_kv * D, device=device, dtype=dtype) # (H_kv, rank_head, head_dim)
    Wuv = torch.randn(rank_head, H_kv * D, device=device, dtype=dtype) # (H_kv, rank_head, head_dim)
    
    # Full attention
    Wq = torch.randn(model_dim, H_q * D, device=device, dtype=dtype)
    Wk = torch.randn(model_dim, H_kv * D, device=device, dtype=dtype)
    Wv = torch.randn(model_dim, H_kv * D, device=device, dtype=dtype)
    
    cache_k = torch.randn(batch_size, seq_len, H_kv * D, device=device, dtype=dtype)
    cache_v = torch.randn(batch_size, seq_len, H_kv * D, device=device, dtype=dtype)
    
    # Output and FFN
    Wout = torch.randn(model_dim, model_dim, device=device, dtype=dtype)
    
    Wup = torch.randn(model_dim, intermediate_dim, device=device, dtype=dtype) # 2.7 is the factor of LLaVA 13B
    Wgate = torch.randn(model_dim, intermediate_dim, device=device, dtype=dtype)
    Wdown = torch.randn(intermediate_dim, model_dim, device=device, dtype=dtype)

    # Create infer_state
    seq_lens = [seq_len] * batch_size
    infer_state = MockInferState(batch_size, seq_len, seq_lens)
    
    # Initialize weights
    for W in [Wq, Wk, Wv, Wout, Wdq, Wdk, Wdv, Wuq, Wuk, Wuv, Wdown]:
        torch.nn.init.xavier_uniform_(W)

    torch.nn.init.xavier_uniform_(Wup);   Wup.mul_(1 / math.sqrt(2))
    torch.nn.init.xavier_uniform_(Wgate); Wgate.mul_(1 / math.sqrt(2))
    
    return input, cache_latent_k, cache_latent_v, Wdq, Wdk, Wdv, Wuq, Wuk, Wuv, Wq, Wk, Wv, cache_k, cache_v, Wout, Wup, Wgate, Wdown, infer_state

def reference_eager_attention_layer(seq_len, head_dim, q_head_num, input, cache_k, cache_v, Wq, Wk, Wv, Wout, Wup, Wgate, Wdown):
    batch_size, seqq, model_dim = input.shape
    seq_len = seq_len
    nheads = q_head_num
    head_dim = head_dim
    sm_scale = 1 / math.sqrt(head_dim)
    causal = False

    Wnorm = torch.ones(model_dim, device=input.device, dtype=input.dtype)
    input = input_layernorm(input, Wnorm)

    # Run function
    q = input @ Wq
    k = input @ Wk
    v = input @ Wv
    
    q = q.view(batch_size, seqq, nheads, head_dim).permute(0, 2, 1, 3)  # [batch_size, nheads, seqq, head_dim]
    # k = cache_k.view(batch_size, seq_len, nheads, head_dim).permute(0, 2, 1, 3)
    # v = cache_v.view(batch_size, seq_len, nheads, head_dim).permute(0, 2, 1, 3)
    k = cache_k
    v = cache_v

    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,
            dropout_p=0.0, 
            is_causal=causal,
            scale=sm_scale,
            enable_gqa=True
        )
    
    out = out.transpose(1, 2).reshape(batch_size, seqq, model_dim)
    
    # 2. Output projection
    output = out @ Wout
    output = output.reshape(batch_size, seqq, model_dim)
    
    # 3. Residual connection
    res = input + output
    
    # 4. FFN (SwiGLU style)
    res = post_attention_layernorm(res, Wnorm)
    output = res @ Wup
    gate = res @ Wgate
    gate = torch.nn.functional.silu(gate)  # SwiGLU activation
    output = output * gate
    output = output @ Wdown

    # 5. Final residual connection
    final_output = res + output
    
    return final_output

def reference_attention_layer(seq_len, head_dim, q_head_num, input, cache_k, cache_v, Wq, Wk, Wv, Wout, Wup, Wgate, Wdown):
    batch_size, seqq, model_dim = input.shape
    seq_len = seq_len
    nheads = q_head_num
    head_dim = head_dim
    sm_scale = 1 / math.sqrt(head_dim)
    causal = False

    Wnorm = torch.ones(model_dim, device=input.device, dtype=input.dtype)
    input = input_layernorm(input, Wnorm)

    # Run function
    q = input @ Wq
    k = input @ Wk
    v = input @ Wv
    
    q = q.view(batch_size, seqq, nheads, head_dim).permute(0, 2, 1, 3)  # [batch_size, nheads, seqq, head_dim]
    # k = cache_k.view(batch_size, seq_len, nheads, head_dim).permute(0, 2, 1, 3)
    # v = cache_v.view(batch_size, seq_len, nheads, head_dim).permute(0, 2, 1, 3)
    k = cache_k
    v = cache_v
    
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=None,
        dropout_p=0.0, 
        is_causal=causal,
        scale=sm_scale,
        enable_gqa=True
    )
    
    out = out.transpose(1, 2).reshape(batch_size, seqq, model_dim)
    
    # 2. Output projection
    output = out @ Wout
    output = output.reshape(batch_size, seqq, model_dim)
    
    # 3. Residual connection
    res = input + output

    # 4. FFN (SwiGLU style)
    res = post_attention_layernorm(res, Wnorm)
    output = res @ Wup
    gate = res @ Wgate
    gate = torch.nn.functional.silu(gate)  # SwiGLU activation
    output = output * gate
    output = output @ Wdown

    # 5. Final residual connection
    final_output = res + output
    
    return final_output

def reference_attention_rope_layer(seq_len, head_dim, q_head_num, input, cache_k, cache_v, Wq, Wk, Wv, Wout, Wup, Wgate, Wdown):
    batch_size, seqq, model_dim = input.shape
    seq_len = seq_len
    nheads = q_head_num
    head_dim = head_dim
    sm_scale = 1 / math.sqrt(head_dim)
    causal = False

    Wnorm = torch.ones(model_dim, device=input.device, dtype=input.dtype)
    input = input_layernorm(input, Wnorm)

    # Run function
    q = input @ Wq
    k = input @ Wk
    v = input @ Wv
    
    q = q.view(batch_size, seqq, nheads, head_dim).permute(0, 2, 1, 3)  # [batch_size, nheads, seqq, head_dim]
    # k = cache_k.view(batch_size, seq_len, -1, head_dim).permute(0, 2, 1, 3) # [batch_size, seq_len, nheads_kv, head_dim]
    # v = cache_v.view(batch_size, seq_len, -1, head_dim).permute(0, 2, 1, 3) # [batch_size, seq_len, nheads_kv, head_dim]
    k = cache_k
    v = cache_v
    
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=None,
        dropout_p=0.0, 
        is_causal=causal,
        scale=sm_scale,
        enable_gqa=True
    )
    
    out = out.transpose(1, 2).reshape(batch_size, seqq, model_dim)
    
    # 2. Output projection
    output = out @ Wout
    output = output.reshape(batch_size, seqq, model_dim)
    
    # 3. Residual connection
    res = input + output
    
    # 4. FFN (SwiGLU style)
    res = post_attention_layernorm(res, Wnorm)
    output = res @ Wup
    gate = res @ Wgate
    gate = torch.nn.functional.silu(gate)  # SwiGLU activation
    output = output * gate
    output = output @ Wdown

    # 5. Final residual connection
    final_output = res + output
    
    return final_output

def run_flash_latent_per_head_decoding_rope_layer(seq_len, head_dim, q_head_num, input, cache_latent_k, cache_latent_v, Wdq, Wdk, Wdv, Wuq, Wuk, Wuvout, Wup, Wgate, Wdown, infer_state):
    batch_size, seqq, model_dim = input.shape
    seq_len = seq_len
    nheads = q_head_num
    head_dim = head_dim
    sm_scale = 1 / math.sqrt(head_dim)
    causal = False

    Wnorm = torch.ones(model_dim, device=input.device, dtype=input.dtype)
    input = input_layernorm(input, Wnorm)

    # Run function
    latent_q = input @ Wdq
    latent_k = input @ Wdk
    latent_v = input @ Wdv
    
    latent_q = latent_q.view(batch_size, seqq, nheads, -1)
    Wuq = Wuq.view(nheads, -1, head_dim)
    q = (latent_q.transpose(1, 2) @ Wuq).transpose(1, 2)
    
    q = q.view(batch_size*seqq, nheads, head_dim)
    # cache_latent_k = cache_latent_k.view(batch_size*seq_len, nheads, -1)
    # cache_latent_v = cache_latent_v.view(batch_size*seq_len, nheads, -1)
    
    out = token_decode_attention_flash_latent_per_head_decoding_rope(
        q, infer_state, q_head_num, head_dim, cache_latent_k, cache_latent_v, Wuk
    )
    out = out.view(batch_size, seqq, -1)
    
    # 2. Output projection
    Wuvout = Wuvout.view(-1, head_dim * nheads)
    output = out @ Wuvout
    output = output.reshape(batch_size, seqq, model_dim)
    
    # 3. Residual connection
    res = input + output
    
    # 4. FFN (SwiGLU style)
    res = post_attention_layernorm(res, Wnorm)
    output = res @ Wup
    gate = res @ Wgate
    gate = torch.nn.functional.silu(gate)  # SwiGLU activation
    output = output * gate
    output = output @ Wdown
    
    # 5. Final residual connection
    final_output = res + output
    
    return final_output

def run_flash_int8_latent_per_head_decoding_rope_layer(seq_len, head_dim, q_head_num, input, cache_latent_k, cache_latent_k_scale, cache_latent_v, cache_latent_v_scale, Wdq, Wdk, Wdv, Wuq, Wuk, Wuk_scale, Wuvout, Wup, Wgate, Wdown, infer_state):
    batch_size, seqq, model_dim = input.shape
    seq_len = seq_len
    nheads = q_head_num
    head_dim = head_dim
    sm_scale = 1 / math.sqrt(head_dim)
    causal = False

    Wnorm = torch.ones(model_dim, device=input.device, dtype=input.dtype)
    input = input_layernorm(input, Wnorm)

    # Run function
    latent_q = input @ Wdq
    latent_k = input @ Wdk
    latent_v = input @ Wdv
    
    latent_q = latent_q.view(batch_size, seqq, nheads, -1)
    Wuq = Wuq.view(nheads, -1, head_dim)
    q = (latent_q.transpose(1, 2) @ Wuq).transpose(1, 2)
    
    q = q.view(batch_size*seqq, nheads, head_dim)
    # cache_latent_k = cache_latent_k.view(batch_size*seq_len, nheads, -1)
    # cache_latent_v = cache_latent_v.view(batch_size*seq_len, nheads, -1)
    
    out = token_decode_attention_flash_int8_latent_per_head_decoding_rope(
        q, infer_state, q_head_num, head_dim, 
        cache_latent_k, cache_latent_k_scale, 
        cache_latent_v, cache_latent_v_scale, 
        Wuk, Wuk_scale
    )
    out = out.view(batch_size, seqq, -1)
    
    # 2. Output projection
    Wuvout = Wuvout.view(-1, head_dim * nheads)
    output = out @ Wuvout
    output = output.reshape(batch_size, seqq, model_dim)
    
    # 3. Residual connection
    res = input + output
    
    # 4. FFN (SwiGLU style)
    res = post_attention_layernorm(res, Wnorm)
    output = res @ Wup
    gate = res @ Wgate
    gate = torch.nn.functional.silu(gate)  # SwiGLU activation
    output = output * gate
    output = output @ Wdown
    
    # 5. Final residual connection
    final_output = res + output
    
    return final_output
