import torch

def token_decode_attention_flash_int8_latent_per_head_decoding_rope(
    q,
    infer_state,
    q_head_num,
    head_dim,
    cache_latent_k,
    cache_latent_k_scale,
    cache_latent_v,
    cache_latent_v_scale,
    wk,
    wk_scale,
    out=None,
    alloc_tensor_func=torch.empty,
    block_seq: int = 256,
):
    BLOCK_SEQ = block_seq
    batch_size = infer_state.batch_size
    max_len_in_batch = infer_state.max_len_in_batch
    calcu_shape1 = (batch_size, q_head_num, head_dim)
    rank_dim_k = cache_latent_k.shape[-1] #// q_head_num
    rank_dim_v = cache_latent_v.shape[-1] #// q_head_num

    from kernel.flash_int8_latent_per_head_decoding_rope_stage1 import flash_int8_latent_per_head_decode_rope_stage1
    from kernel.flash_latent_per_head_decoding_stage2 import flash_decode_stage2
    
    o_rank_shape = (batch_size, q_head_num, rank_dim_v)
    o_tensor = alloc_tensor_func(o_rank_shape, dtype=q.dtype, device=q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, (max_len_in_batch + BLOCK_SEQ - 1) // BLOCK_SEQ, rank_dim_v], dtype=torch.float32, device="cuda"
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, (max_len_in_batch + BLOCK_SEQ - 1) // BLOCK_SEQ], dtype=torch.float32, device="cuda"
    )
    
    flash_int8_latent_per_head_decode_rope_stage1(
        q.view(calcu_shape1),
        cache_latent_k,
        cache_latent_k_scale,
        cache_latent_v,
        cache_latent_v_scale,
        wk,
        wk_scale,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        infer_state.max_len_in_batch,
        mid_o,
        mid_o_logexpsum,
        BLOCK_SEQ,
    )
    flash_decode_stage2(mid_o, mid_o_logexpsum, infer_state.b_seq_len, o_tensor.view(o_rank_shape), BLOCK_SEQ, rank_chunk=64)
    return o_tensor

def token_decode_attention_flash_latent_per_head_decoding_rope(
    q, infer_state, q_head_num, head_dim, cache_latent_k, cache_latent_v, wk, out=None, alloc_tensor_func=torch.empty
):
    BLOCK_SEQ = 256
    batch_size = infer_state.batch_size
    max_len_in_batch = infer_state.max_len_in_batch
    calcu_shape1 = (batch_size, q_head_num, head_dim)
    rank_dim_k = cache_latent_k.shape[-1] #// q_head_num
    rank_dim_v = cache_latent_v.shape[-1] #// q_head_num

    from kernel.flash_latent_per_head_decoding_rope_stage1 import flash_latent_per_head_decode_rope_stage1
    from kernel.flash_latent_per_head_decoding_stage2 import flash_decode_stage2
    
    o_rank_shape = (batch_size, q_head_num, rank_dim_v)
    o_tensor = alloc_tensor_func(o_rank_shape, dtype=q.dtype, device=q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, (max_len_in_batch + BLOCK_SEQ - 1) // BLOCK_SEQ, rank_dim_v], dtype=torch.float32, device="cuda"
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, (max_len_in_batch + BLOCK_SEQ - 1) // BLOCK_SEQ], dtype=torch.float32, device="cuda"
    )
    
    flash_latent_per_head_decode_rope_stage1(
        q.view(calcu_shape1),
        cache_latent_k,
        cache_latent_v,
        wk,
        # wv,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        infer_state.max_len_in_batch,
        mid_o,
        mid_o_logexpsum,
        BLOCK_SEQ,
    )
    flash_decode_stage2(mid_o, mid_o_logexpsum, infer_state.b_seq_len, o_tensor.view(o_rank_shape), BLOCK_SEQ, rank_chunk=64)
    return o_tensor


def token_decode_attention_flash_decoding(
    q, infer_state, q_head_num, head_dim, cache_k, cache_v, out=None, alloc_tensor_func=torch.empty
):
    BLOCK_SEQ = 256
    batch_size = infer_state.batch_size
    max_len_in_batch = infer_state.max_len_in_batch
    calcu_shape1 = (batch_size, q_head_num, head_dim)

    from kernel.flash_decoding_stage1 import flash_decode_stage1
    from kernel.flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q.shape, dtype=q.dtype, device=q.device) if out is None else out

    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, max_len_in_batch // BLOCK_SEQ + 1, head_dim], dtype=torch.float32, device="cuda"
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, max_len_in_batch // BLOCK_SEQ + 1], dtype=torch.float32, device="cuda"
    )

    flash_decode_stage1(
        q.view(calcu_shape1),
        cache_k,
        cache_v,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        infer_state.max_len_in_batch,
        mid_o,
        mid_o_logexpsum,
        BLOCK_SEQ,
    )
    flash_decode_stage2(mid_o, mid_o_logexpsum, infer_state.b_seq_len, o_tensor.view(calcu_shape1), BLOCK_SEQ)
    return o_tensor
