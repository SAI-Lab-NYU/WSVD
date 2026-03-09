import torch
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MockInferState:
    def __init__(self, batch_size, max_len_in_batch, seq_lens):
        self.batch_size = batch_size
        self.max_len_in_batch = max_len_in_batch
        self.b_seq_len = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
        self.b_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
        
        # Create req_manager
        class ReqManager:
            def __init__(self, batch_size, max_len):
                # self.req_to_token_indexs = torch.arange(max_len, dtype=torch.int32, device="cuda").unsqueeze(0).repeat(batch_size, 1)
                self.req_to_token_indexs = torch.arange(batch_size * max_len, dtype=torch.int32, device="cuda").view(batch_size, max_len)

        self.req_manager = ReqManager(batch_size, max_len_in_batch)

def create_test_data(batch_size=2, q_head_num=8, head_dim=64, seq_len=256, rank=128, rank_ratio=0.9):
    """Create test data."""
    device = "cuda"
    
    q_len = 1
    
    # Create query (batch_size, q_head_num, head_dim)
    q = torch.randn(batch_size, q_len, q_head_num, head_dim, device=device, dtype=torch.float16)
    rank_head = int(rank_ratio * head_dim)
    # Create latent cache (batch_size, seq_len, rank) - not split by head
    cache_latent = torch.randn(batch_size, seq_len, rank_head*q_head_num, device=device, dtype=torch.float16)
    rank = rank_head * q_head_num
    # Create weight matrices wk, wv (rank, head_dim * q_head_num)
    wk = torch.randn(rank_head, head_dim * q_head_num, device=device, dtype=torch.float16)
    wv = torch.randn(rank_head, head_dim * q_head_num, device=device, dtype=torch.float16)

    # Create infer_state
    seq_lens = [seq_len] * batch_size
    infer_state = MockInferState(batch_size, seq_len, seq_lens)
    
    return q, cache_latent, wk, wv, infer_state


def reference_attention(q, k, v, head_dim):
    """Reference implementation: simulate attention computation for Flash decoding."""
    batch_size, q_len, q_head_num, _ = q.shape
    seq_len = k.shape[1]
    
    # Transpose to (batch_size, q_head_num, seq_len, head_dim)
    k = k.view(batch_size, seq_len, q_head_num, head_dim).transpose(1, 2).to(torch.float16)
    v = v.view(batch_size, seq_len, q_head_num, head_dim).transpose(1, 2).to(torch.float16)
    
    # Reshape q dimensions
    q_reshaped = q.view(batch_size, q_len, q_head_num, head_dim).transpose(1, 2).to(torch.float16)  # (batch_size, q_head_num, q_len, head_dim)
    
    # Use SDPA while keeping logic consistent with Flash decoding
    # Flash decoding uses sequence-length masking instead of causal masking
    output = torch.nn.functional.scaled_dot_product_attention(q_reshaped, k, v, is_causal=False)
    
    # Transpose back to original dimension order
    output = output.transpose(1, 2)  # (batch_size, q_len, q_head_num, head_dim)

    return output

def reference_flash_decoding(q, k, v, head_dim, infer_state):
    """Reference implementation: simulate attention computation for Flash decoding."""
    batch_size, q_len, q_head_num, _ = q.shape
    seq_len = k.shape[1]
    
    # Reshape to (batch_size, q_head_num, seq_len, head_dim)
    k = k.view(batch_size*seq_len, q_head_num, head_dim).to(torch.float16)
    v = v.view(batch_size*seq_len, q_head_num, head_dim).to(torch.float16)
    
    # Reshape q dimensions
    q_reshaped = q.view(batch_size*q_len, q_head_num, head_dim).to(torch.float16)
    
    from kernel.flash_decoding import token_decode_attention_flash_decoding
    output = token_decode_attention_flash_decoding(
        q_reshaped, infer_state, q_head_num, head_dim, k, v
    )
    
    # Reshape back to original dimension order
    output = output.view(batch_size, q_len, q_head_num, head_dim)

    return output

def reference_latent_attention(q, cache_latent, wk, wv, head_dim):
    """Reference implementation: simulate attention computation for Flash latent decoding."""
    batch_size, q_len, q_head_num, _ = q.shape
    seq_len = cache_latent.shape[1]
    rank = cache_latent.shape[2]
    cache_latent = cache_latent.view(batch_size, seq_len, q_head_num, -1)
    wk = wk.view(q_head_num, -1, head_dim)
    wv = wv.view(q_head_num, -1, head_dim)
    k_full = (cache_latent.transpose(1, 2) @ wk).transpose(1, 2).to(torch.float16)
    v_full = (cache_latent.transpose(1, 2) @ wv).transpose(1, 2).to(torch.float16)
    
    # Reconstruct K and V
    k = k_full.view(batch_size, seq_len, q_head_num, head_dim)
    v = v_full.view(batch_size, seq_len, q_head_num, head_dim)
    
    # Transpose to (batch_size, q_head_num, seq_len, head_dim)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # Reshape q dimensions
    q_reshaped = q.transpose(1, 2).to(torch.float16)  # (batch_size, q_head_num, q_len, head_dim)
    
    # Use SDPA while keeping logic consistent with Flash decoding
    # Flash decoding uses sequence-length masking instead of causal masking
    output = torch.nn.functional.scaled_dot_product_attention(q_reshaped, k, v, is_causal=False)
    
    # Transpose back to original dimension order
    output = output.transpose(1, 2)  # (batch_size, q_len, q_head_num, head_dim)
    
    return output

def reference_flash_latent_decoding(q, cache_latent, wk, wv, head_dim, infer_state):
    """Reference implementation: simulate attention computation for Flash decoding."""
    batch_size, q_len, q_head_num, _ = q.shape
    seq_len = cache_latent.shape[1]
    rank = cache_latent.shape[2]
    
    # Reconstruct K and V
    k_full = (cache_latent @ wk).to(torch.float16)
    k = k_full.view(batch_size, seq_len, q_head_num, head_dim)
    
    v_full = (cache_latent @ wv).to(torch.float16)
    v = v_full.view(batch_size, seq_len, q_head_num, head_dim)
    
    # Reshape to (batch_size, q_head_num, seq_len, head_dim)
    k = k.view(batch_size*seq_len, q_head_num, head_dim)
    v = v.view(batch_size*seq_len, q_head_num, head_dim)
    
    # Reshape q dimensions
    q_reshaped = q.view(batch_size*q_len, q_head_num, head_dim).to(torch.float16)
    
    from kernel.flash_decoding import token_decode_attention_flash_decoding
    output = token_decode_attention_flash_decoding(
        q_reshaped, infer_state, q_head_num, head_dim, k, v
    )
    
    return output.view(batch_size, 1, q_head_num, head_dim)

def run_flash_latent_decoding(q, cache_latent, wk, wv, head_dim, infer_state, batch_size, q_head_num, seq_len, rank):
    from kernel.flash_decoding import token_decode_attention_flash_latent_decoding
    q_len = 1
    q = q.view(batch_size*q_len, q_head_num, head_dim).contiguous()
    cache_latent = cache_latent.view(batch_size*seq_len, rank).contiguous()
    breakpoint()
    wk = wk.view(rank, head_dim * q_head_num).contiguous()
    wv = wv.view(rank, head_dim * q_head_num).contiguous()
    output_flash = token_decode_attention_flash_latent_decoding(
        q, infer_state, q_head_num, head_dim, cache_latent, wk, wv
    )
    output_flash = output_flash.view(batch_size, q_len, q_head_num, head_dim)
    return output_flash

def run_flash_latent_per_head_decoding(q, cache_latent_k, cache_latent_v, wk, head_dim, infer_state, batch_size, q_head_num, seq_len, rank):
    from kernel.flash_decoding import token_decode_attention_flash_latent_per_head_decoding
    q_len = 1
    q = q.view(batch_size*q_len, q_head_num, head_dim).contiguous()
    head_num = q_head_num
    rank_head = rank // head_num
    
    rank = rank_head * head_num
    # cache_latent = cache_latent.view(batch_size*seq_len, rank).contiguous()
    cache_latent_k = cache_latent_k.view(batch_size*seq_len, head_num, rank_head).contiguous()
    cache_latent_v = cache_latent_v.view(batch_size*seq_len, head_num, rank_head).contiguous()
    wk = wk.view(rank_head, head_dim * q_head_num).contiguous()
    # wv = wv.view(rank, head_dim * q_head_num).contiguous()
    output_flash = token_decode_attention_flash_latent_per_head_decoding(
        q, infer_state, q_head_num, head_dim, cache_latent_k, cache_latent_v, wk
    )
    output_flash = output_flash.view(batch_size, q_len, q_head_num, -1)
    return output_flash

def run_flash_latent_per_head_decoding_rope(q, cache_latent_k, cache_latent_v, wk, head_dim, infer_state, batch_size, q_head_num, seq_len, rank):
    from kernel.flash_decoding import token_decode_attention_flash_latent_per_head_decoding_rope
    q_len = 1
    q = q.view(batch_size*q_len, q_head_num, head_dim).contiguous()
    head_num = q_head_num
    rank_head = rank // head_num
    
    rank = rank_head * head_num
    # cache_latent = cache_latent.view(batch_size*seq_len, rank).contiguous()
    cache_latent_k = cache_latent_k.view(batch_size*seq_len, head_num, rank_head).contiguous()
    cache_latent_v = cache_latent_v.view(batch_size*seq_len, head_num, rank_head).contiguous()
    wk = wk.view(rank_head, head_dim * q_head_num).contiguous()
    # wv = wv.view(rank, head_dim * q_head_num).contiguous()
    output_flash = token_decode_attention_flash_latent_per_head_decoding_rope(
        q, infer_state, q_head_num, head_dim, cache_latent_k, cache_latent_v, wk
    )
    output_flash = output_flash.view(batch_size, q_len, q_head_num, -1)
    return output_flash