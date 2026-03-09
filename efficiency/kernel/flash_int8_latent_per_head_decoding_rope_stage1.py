import torch
import triton
import triton.language as tl
import os
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"


@triton.jit
def _get_freq_block_from_pos(pos_idx, theta: tl.constexpr, DModel: tl.constexpr):
    D2: tl.constexpr = DModel // 2
    offs_pair = tl.arange(0, D2) * 2
    inv_freq = tl.extra.cuda.libdevice.fast_powf(theta, (offs_pair.to(tl.float32) / DModel))
    angles = pos_idx[:, None].to(tl.float32) / inv_freq[None, :]
    return tl.extra.cuda.libdevice.fast_cosf(angles), tl.extra.cuda.libdevice.fast_sinf(angles)

def get_configs():
    configs = []
    for BLOCK_N in [16, 32, 64, 128]:
        for BLOCK_SEQ in [256, 512, 1024, 2048]:
            if BLOCK_SEQ % BLOCK_N != 0:
                continue
            for num_warps in [1, 2, 4]:
                for num_stages in [1, 2, 3]:
                    configs.append(triton.Config({"BLOCK_N": BLOCK_N, "BLOCK_SEQ": BLOCK_SEQ}, num_warps=num_warps, num_stages=num_stages))
    # configs.append(triton.Config({"BLOCK_N": 64, "BLOCK_SEQ": 256}, num_warps=4, num_stages=1))
    return configs

@triton.autotune(
    configs=get_configs(),
    key=["BLOCK_DMODEL", "RANK_K_PAD", "RANK_V_PAD"],
)
@triton.jit
def _fwd_kernel_flash_int8_latent_decode_rope_stage1(
    Q, latent_K, latent_K_scale, latent_V, latent_V_scale, wk, wk_scale, sm_scale, Req_to_tokens, B_req_idx, B_Seqlen,
    Mid_O, # [batch, head, seq_block_num, rank_v]
    Mid_O_LogExpSum, #[batch, head, seq_block_num]
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_qbs, stride_qh, stride_qd,
    stride_latent_k_bs, stride_latent_k_h, stride_latent_k_r,
    stride_latent_v_bs, stride_latent_v_h, stride_latent_v_r,
    stride_latent_k_scale_bs, stride_latent_k_scale_h,
    stride_latent_v_scale_bs, stride_latent_v_scale_h,
    stride_wk_r, stride_wk_d,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    gqa_group_size,
    THETA: tl.constexpr,
    BLOCK_SEQ: tl.constexpr, 
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    RANK_K: tl.constexpr,
    RANK_V: tl.constexpr,
    RANK_K_PAD: tl.constexpr,
    RANK_V_PAD: tl.constexpr,
    # RANK_CHUNK: tl.constexpr
):
    D2: tl.constexpr = BLOCK_DMODEL // 2
    
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)
    cur_kv_head = cur_head // gqa_group_size

    # offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_d0 = tl.arange(0, D2)
    offs_d1 = offs_d0 + D2
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)

    # off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    off_q0 = cur_batch * stride_qbs + cur_head * stride_qh + offs_d0
    off_q1 = cur_batch * stride_qbs + cur_head * stride_qh + offs_d1
    
    block_n_size = tl.where(cur_batch_end_index - cur_batch_start_index <= 0, 0, cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1) // BLOCK_N
    
    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
    
    # offs_r_chunk = tl.arange(0, RANK_CHUNK)   
    
    # q = tl.load(Q + off_q)
    q0 = tl.load(Q + off_q0)
    q1 = tl.load(Q + off_q1)

    sum_exp = 0.0
    max_logic = -float("inf")
    
    offs_v = tl.arange(0, RANK_V_PAD)
    mask_v = offs_v < RANK_V
    offs_r_full = tl.arange(0, RANK_K_PAD)  # [RANK_K]
    mask_r_full = offs_r_full < RANK_K
    
    acc = tl.zeros([RANK_V_PAD], dtype=tl.float32)
    
    off_wk0 = (cur_kv_head * BLOCK_DMODEL + offs_d0) * stride_wk_d  # [D2]
    off_wk1 = (cur_kv_head * BLOCK_DMODEL + offs_d1) * stride_wk_d  # [D2]
    
    # Wk: [RANK_K, BLOCK_DMODEL] -> split halves
    idxs_wk0 = offs_r_full[:, None] * stride_wk_r + off_wk0[None, :]
    idxs_wk1 = offs_r_full[:, None] * stride_wk_r + off_wk1[None, :]
    wk0 = tl.load(wk + idxs_wk0, mask=mask_r_full[:, None])
    wk1 = tl.load(wk + idxs_wk1, mask=mask_r_full[:, None])
    
    off_wk_scale0 = cur_kv_head * BLOCK_DMODEL + offs_d0
    off_wk_scale1 = cur_kv_head * BLOCK_DMODEL + offs_d1
    wk_scale0 = tl.load(wk_scale + off_wk_scale0) # [D2]
    wk_scale1 = tl.load(wk_scale + off_wk_scale1) # [D2]

    # # Pre-multiply wk_scale into q: wk_scale only depends on head/dim, not key position
    # # This avoids repeated per-element (k * wk_scale) multiplication for every key token
    # q0_w0 = q0 * wk_scale0
    # q1_w0 = q1 * wk_scale0
    # q0_w1 = q0 * wk_scale1
    # q1_w1 = q1 * wk_scale1
    
    for start_n in range(0, block_n_size, 1):  # pyright: ignore[reportUnreachable]
        offs_n_new = start_n * BLOCK_N + offs_n
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=offs_n_new < cur_batch_end_index,
            other=0,
        ).to(tl.int64)
        # Position mask for each request (computed in the loop)
        mask_n = offs_n_new < cur_batch_end_index                # [BLOCK_N] bool

        # Load latent: (batch*seqlen, head, rank) for current batch/head/position
        # Following the original LightLLM approach
        off_latent_k = k_loc[:, None] * stride_latent_k_bs + cur_kv_head * stride_latent_k_h + offs_r_full[None, :] * stride_latent_k_r
        off_latent_v = k_loc[:, None] * stride_latent_v_bs + cur_kv_head * stride_latent_v_h + offs_v[None, :] * stride_latent_v_r
        
        off_latent_k_scale = k_loc[:, None] * stride_latent_k_scale_bs + cur_kv_head * stride_latent_k_scale_h
        off_latent_v_scale = k_loc[:, None] * stride_latent_v_scale_bs + cur_kv_head * stride_latent_v_scale_h
        
        latent_k_scale = tl.load(latent_K_scale + off_latent_k_scale, mask=mask_n[:, None], other=1.0)
        latent_v_scale = tl.load(latent_V_scale + off_latent_v_scale, mask=mask_n[:, None], other=1.0)
        
        k0_acc = tl.zeros([BLOCK_N, D2], dtype=tl.float32)  # f32 accumulators for halves
        k1_acc = tl.zeros([BLOCK_N, D2], dtype=tl.float32)

        # No rank chunking: load full rank at once and compute latent_K @ Wk
        # latent_K: [BLOCK_N, RANK_K]
        latent_k = tl.load(
            latent_K + off_latent_k,
            mask=mask_n[:, None] & mask_r_full[None, :],
            other=0.0
        )
        
        k0_acc += tl.dot(latent_k, wk0)
        k1_acc += tl.dot(latent_k, wk1)
        # Do not compute V here
        k0_acc = k0_acc.to(tl.float16)
        k1_acc = k1_acc.to(tl.float16)
        
        # RoPE: position must use logical sequence position, not physical cache index k_loc
        pos_idx = offs_n_new  # [BLOCK_N]
        cos, sin = _get_freq_block_from_pos(pos_idx, THETA, BLOCK_DMODEL)  # [BLOCK_N, D2]
        cos = cos.to(tl.float16)
        sin = sin.to(tl.float16)

        # dequantize k with latent k scale and wk scale
        k0_acc = k0_acc * latent_k_scale * wk_scale0[None, :]
        k1_acc = k1_acc * latent_k_scale * wk_scale1[None, :]
        
        k0_rot = k0_acc * cos - k1_acc * sin
        k1_rot = k1_acc * cos + k0_acc * sin

        # Compute attention score without reconstructing full K block
        att_value = tl.sum(q0[None, :] * k0_rot, axis=1) + tl.sum(q1[None, :] * k1_rot, axis=1)  # [BLOCK_N]
        
        # # Fuse dequantize + RoPE + QK^T:
        # # - latent_k_scale is a per-token scalar, and RoPE/dot-product are linear,
        # #   so multiplication can be moved to the end to avoid per-dim elementwise multiply
        # # - wk_scale is pre-fused into q (done above), avoiding repeated multiply per key token
        # #
        # # Original form:
        # #   k0' = (k0 * wk0) * latent
        # #   k1' = (k1 * wk1) * latent
        # #   k_rot = RoPE(k')，att = q · k_rot
        # # Equivalent rewrite:
        # #   att = latent * [ sum(k0 * (q0*wk0*cos + q1*wk0*sin))
        # #                 + sum(k1 * (q1*wk1*cos - q0*wk1*sin)) ]
        # t0 = q0_w0[None, :] * cos + q1_w0[None, :] * sin  # [BLOCK_N, D2]
        # t1 = q1_w1[None, :] * cos - q0_w1[None, :] * sin  # [BLOCK_N, D2]
        # att_value = tl.sum(k0_acc * t0, axis=1) + tl.sum(k1_acc * t1, axis=1)  # [BLOCK_N]
        # att_value *= latent_k_scale[:, 0]
        
        att_value *= sm_scale
        att_value = tl.where(offs_n_new < cur_batch_end_index, att_value, float("-inf"))
        
        cur_max_logic = tl.max(att_value, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)
        
        latent_v = tl.load(
            latent_V + off_latent_v,
            mask=mask_n[:, None] & mask_v[None, :],
            other=0.0,
        )

        # dequantize v with latent v scale
        latent_v = latent_v * latent_v_scale
        
        exp_logic = tl.exp(att_value - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * latent_v, axis=0)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic
    
    need_store = tl.where(block_n_size == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid_o = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + seq_start_block * stride_mid_os + offs_v
        off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + seq_start_block
        tl.store(Mid_O + off_mid_o, acc / sum_exp, mask=mask_v)
        tl.store(Mid_O_LogExpSum + off_mid_o_logexpsum, max_logic + tl.log(sum_exp))
    return


@torch.no_grad()
def flash_int8_latent_per_head_decode_rope_stage1(q, latent_k, latent_k_scale, latent_v, latent_v_scale, wk, wk_scale, Req_to_tokens, B_req_idx, B_Seqlen, max_len_in_batch, mid_out, mid_out_logsumexp, block_seq):
    # BLOCK_SEQ = 256 # block_seq 256
    # BLOCK_N = 32 # 64 tokens per block 
    # assert BLOCK_SEQ % BLOCK_N == 0
    
    # shape constraints
    head_dim = q.shape[-1]
    head_num = q.shape[1]
    rank_k = latent_k.shape[-1] # latent_k: (batch*seqlen, head_num_kv, rank_k), rank_k per head
    rank_v = latent_v.shape[-1] # latent_v: (batch*seqlen, head_num_kv, rank_v), rank_v per head
    assert head_dim * latent_k.shape[1] == wk.shape[1]  # wk: (rank_k, headdim * head_num_kv)
    assert rank_k == wk.shape[0]
    
    # token-wise scale, (batch*seqlen, head_num_kv)
    assert latent_k_scale.shape[0] == latent_k.shape[0] # btach*seqlen
    assert latent_v_scale.shape[0] == latent_v.shape[0] # btach*seqlen
    assert latent_k_scale.shape[1] == latent_k.shape[1] # head_num_kv
    assert latent_v_scale.shape[1] == latent_v.shape[1] # head_num_kv

    # head-wise scale, (1, head_dim * head_num_kv)
    assert wk_scale.shape[1] == wk.shape[1] # head_dim * head_num_kv

    assert head_dim in {16, 32, 64, 128}
    sm_scale = 1.0 / (head_dim ** 0.5)
    batch = B_req_idx.shape[0]
    # grid = (batch, head_num, triton.cdiv(max_len_in_batch, BLOCK_SEQ))
    grid = lambda META: (batch, head_num, triton.cdiv(max_len_in_batch, META["BLOCK_SEQ"]))
    gqa_group_size = q.shape[1] // latent_k.shape[1]  # latent is split by head, so use q_heads // kv_heads
    
    _fwd_kernel_flash_int8_latent_decode_rope_stage1[grid](
        q, 
        latent_k, latent_k_scale, 
        latent_v, latent_v_scale, 
        wk, wk_scale, 
        sm_scale, 
        Req_to_tokens, B_req_idx, B_Seqlen,
        mid_out,
        mid_out_logsumexp,
        Req_to_tokens.stride(0), Req_to_tokens.stride(1),
        q.stride(0), q.stride(1), q.stride(2),
        latent_k.stride(0), latent_k.stride(1), latent_k.stride(2),
        latent_v.stride(0), latent_v.stride(1), latent_v.stride(2),
        latent_k_scale.stride(0), latent_k_scale.stride(1),
        latent_v_scale.stride(0), latent_v_scale.stride(1),
        wk.stride(0), wk.stride(1),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_out_logsumexp.stride(0), mid_out_logsumexp.stride(1), mid_out_logsumexp.stride(2),
        gqa_group_size,
        THETA=10000.0,
        # BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=head_dim,
        # BLOCK_N=BLOCK_N,
        RANK_K=rank_k,
        RANK_V=rank_v,
        RANK_K_PAD=next_pow2(rank_k),
        RANK_V_PAD=next_pow2(rank_v),
        # RANK_CHUNK=128, # 128 rank chunks per block, 64 for 0.5 ratio
        # num_warps=2, # 4 warps per block, 2 for 0.5 ratio
        # num_stages=1,
    )
    return

def next_pow2(x): 
    return 1 << (x - 1).bit_length()