import argparse
import math
import os
import sys

import torch

# Keep local import style consistent with existing scripts.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from layer_utils import (  # noqa: E402
    create_test_data_layer,
    run_flash_latent_per_head_decoding_rope_layer,
)


def _rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    x_f32 = x.float()
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_f32 * torch.rsqrt(var + eps)
    return (x_norm.to(weight.dtype) * weight).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x0 = x[..., ::2]
    x1 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = -x1
    out[..., 1::2] = x0
    return out


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    # x: [S, H, D] or [B, H, D], positions: [S] or [B]
    d = x.shape[-1]
    assert d % 2 == 0, "head_dim must be even for RoPE"
    half = d // 2
    idx = torch.arange(0, half, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (2.0 * idx / d))
    angle = positions.float()[:, None] * inv_freq[None, :]
    cos = torch.repeat_interleave(torch.cos(angle), 2, dim=-1).to(x.dtype)
    sin = torch.repeat_interleave(torch.sin(angle), 2, dim=-1).to(x.dtype)
    return x * cos[:, None, :] + _rotate_half(x) * sin[:, None, :]


def _prepare_inputs(batch_size, q_head_num, kv_head_num, head_dim, seq_len, rank_ratio, intermediate_dim):
    (
        x,
        cache_latent_k,
        cache_latent_v,
        Wdq,
        Wdk,
        Wdv,
        Wuq,
        Wuk,
        Wuv,
        _Wq,
        _Wk,
        _Wv,
        _cache_k,
        _cache_v,
        Wout,
        Wup,
        Wgate,
        Wdown,
        infer_state,
    ) = create_test_data_layer(
        batch_size=batch_size,
        q_head_num=q_head_num,
        head_dim=head_dim,
        seq_len=seq_len,
        rank_ratio=rank_ratio,
        kv_head_num=kv_head_num,
        intermediate_dim=intermediate_dim,
    )

    rank_head = int(rank_ratio * head_dim)
    # Same construction as timing script.
    Wuv_h = Wuv.view(kv_head_num, rank_head, head_dim)
    Wuv_broadcast = Wuv_h.repeat(q_head_num // kv_head_num, 1, 1)
    Wout_h = Wout.view(q_head_num, head_dim, q_head_num * head_dim)
    Wuvout = Wuv_broadcast @ Wout_h

    cache_latent_k = cache_latent_k.view(batch_size * seq_len, kv_head_num, rank_head)
    cache_latent_v = cache_latent_v.view(batch_size * seq_len, kv_head_num, rank_head)

    return {
        "x": x,
        "cache_latent_k": cache_latent_k,
        "cache_latent_v": cache_latent_v,
        "Wdq": Wdq,
        "Wdk": Wdk,
        "Wdv": Wdv,
        "Wuq": Wuq,
        "Wuk": Wuk,
        "Wuv": Wuv,
        "Wout": Wout,
        "Wuvout": Wuvout,
        "Wup": Wup,
        "Wgate": Wgate,
        "Wdown": Wdown,
        "infer_state": infer_state,
        "rank_head": rank_head,
    }


def reference_online_rope_kernel_equivalent_layer(
    seq_len,
    head_dim,
    q_head_num,
    kv_head_num,
    x,
    cache_latent_k,
    cache_latent_v,
    Wdq,
    Wuq,
    Wuk,
    Wuvout,
    Wup,
    Wgate,
    Wdown,
    infer_state,
):
    # This reference mirrors the custom decode kernel semantics:
    # - K is reconstructed from latent and RoPE is applied online.
    # - V stays in latent space during attention.
    # - Then latent output is projected by Wuvout.
    bsz, q_len, model_dim = x.shape
    rank_head = cache_latent_k.shape[-1]
    assert q_len == 1
    gqa_group = q_head_num // kv_head_num
    sm_scale = 1.0 / math.sqrt(head_dim)

    x_norm = _rms_norm(x)
    latent_q = (x_norm @ Wdq).view(bsz, q_len, q_head_num, rank_head)[:, 0]
    Wuq_h = Wuq.view(q_head_num, rank_head, head_dim)
    q = torch.einsum("bhr,hrd->bhd", latent_q, Wuq_h)  # [B, Hq, D]

    Wuk_h = Wuk.view(rank_head, kv_head_num, head_dim)
    req_to_tokens = infer_state.req_manager.req_to_token_indexs
    b_req_idx = infer_state.b_req_idx
    b_seq_len = infer_state.b_seq_len

    o_rank = torch.empty(bsz, q_head_num, rank_head, device=x.device, dtype=x.dtype)
    for b in range(bsz):
        req_id = int(b_req_idx[b].item())
        cur_len = int(b_seq_len[b].item())
        token_idx = req_to_tokens[req_id, :cur_len]

        latent_k_b = cache_latent_k[token_idx]  # [S, Hkv, R]
        latent_v_b = cache_latent_v[token_idx]  # [S, Hkv, R]

        k_full = torch.einsum("shr,rhd->shd", latent_k_b, Wuk_h)  # [S, Hkv, D]
        pos = torch.arange(cur_len, device=x.device, dtype=torch.float32)
        k_rope = _rope(k_full, pos)  # online RoPE on K

        k_expand = k_rope.repeat_interleave(gqa_group, dim=1)  # [S, Hq, D]
        v_expand = latent_v_b.repeat_interleave(gqa_group, dim=1)  # [S, Hq, R]

        score = torch.einsum("hd,shd->hs", q[b], k_expand) * sm_scale
        prob = torch.softmax(score, dim=-1)
        o_rank[b] = torch.einsum("hs,shr->hr", prob, v_expand)

    out = o_rank.view(bsz, q_len, q_head_num * rank_head)
    out = out @ Wuvout.view(-1, q_head_num * head_dim)
    out = out.view(bsz, q_len, model_dim)

    res = x_norm + out
    res = _rms_norm(res)
    ffn = (res @ Wup) * torch.nn.functional.silu(res @ Wgate)
    ffn = ffn @ Wdown
    return res + ffn


def reference_online_rope_attention_projected_only(
    head_dim,
    q_head_num,
    kv_head_num,
    x,
    cache_latent_k,
    cache_latent_v,
    Wdq,
    Wuq,
    Wuk,
    Wuvout,
    infer_state,
):
    bsz, q_len, model_dim = x.shape
    rank_head = cache_latent_k.shape[-1]
    assert q_len == 1
    gqa_group = q_head_num // kv_head_num
    sm_scale = 1.0 / math.sqrt(head_dim)

    x_norm = _rms_norm(x)
    latent_q = (x_norm @ Wdq).view(bsz, q_len, q_head_num, rank_head)[:, 0]
    Wuq_h = Wuq.view(q_head_num, rank_head, head_dim)
    q = torch.einsum("bhr,hrd->bhd", latent_q, Wuq_h)

    Wuk_h = Wuk.view(rank_head, kv_head_num, head_dim)
    req_to_tokens = infer_state.req_manager.req_to_token_indexs
    b_req_idx = infer_state.b_req_idx
    b_seq_len = infer_state.b_seq_len

    o_rank = torch.empty(bsz, q_head_num, rank_head, device=x.device, dtype=x.dtype)
    for b in range(bsz):
        req_id = int(b_req_idx[b].item())
        cur_len = int(b_seq_len[b].item())
        token_idx = req_to_tokens[req_id, :cur_len]

        latent_k_b = cache_latent_k[token_idx]
        latent_v_b = cache_latent_v[token_idx]
        k_full = torch.einsum("shr,rhd->shd", latent_k_b, Wuk_h)
        pos = torch.arange(cur_len, device=x.device, dtype=torch.float32)
        k_rope = _rope(k_full, pos)

        k_expand = k_rope.repeat_interleave(gqa_group, dim=1)
        v_expand = latent_v_b.repeat_interleave(gqa_group, dim=1)

        score = torch.einsum("hd,shd->hs", q[b], k_expand) * sm_scale
        prob = torch.softmax(score, dim=-1)
        o_rank[b] = torch.einsum("hs,shr->hr", prob, v_expand)

    out = o_rank.view(bsz, q_len, q_head_num * rank_head)
    out = out @ Wuvout.view(-1, q_head_num * head_dim)
    return out.view(bsz, q_len, model_dim)


def reference_llama_style_offline_rope_attention_only(
    seq_len,
    head_dim,
    q_head_num,
    kv_head_num,
    x,
    cache_latent_k,
    cache_latent_v,
    Wdq,
    Wuq,
    Wuk,
    Wuv,
):
    # LLaMA-style perspective:
    # - RoPE is usually applied before/when writing K cache (offline for decode kernel).
    # - Attention step then consumes already-rotated K cache.
    bsz, q_len, _ = x.shape
    rank_head = cache_latent_k.shape[-1]
    assert q_len == 1
    gqa_group = q_head_num // kv_head_num
    sm_scale = 1.0 / math.sqrt(head_dim)

    x_norm = _rms_norm(x)
    latent_q = (x_norm @ Wdq).view(bsz, q_len, q_head_num, rank_head)[:, 0]
    Wuq_h = Wuq.view(q_head_num, rank_head, head_dim)
    q = torch.einsum("bhr,hrd->bhd", latent_q, Wuq_h)

    Wuk_h = Wuk.view(rank_head, kv_head_num, head_dim)
    Wuv_h = Wuv.view(rank_head, kv_head_num, head_dim)

    # Decode with contiguous cache order in this mock setup.
    k_lat = cache_latent_k.view(bsz, seq_len, kv_head_num, rank_head)
    v_lat = cache_latent_v.view(bsz, seq_len, kv_head_num, rank_head)
    k_full = torch.einsum("bshr,rhd->bshd", k_lat, Wuk_h)
    v_full = torch.einsum("bshr,rhd->bshd", v_lat, Wuv_h)

    pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    k_cache_rope = torch.stack([_rope(k_full[b], pos) for b in range(bsz)], dim=0)

    k_expand = k_cache_rope.repeat_interleave(gqa_group, dim=2)  # [B, S, Hq, D]
    v_expand = v_full.repeat_interleave(gqa_group, dim=2)  # [B, S, Hq, D]

    score = torch.einsum("bhd,bshd->bhs", q, k_expand) * sm_scale
    prob = torch.softmax(score, dim=-1)
    out = torch.einsum("bhs,bshd->bhd", prob, v_expand)
    return out  # [B, Hq, D]


def reference_llama_style_offline_rope_attention_projected_only(
    head_dim,
    q_head_num,
    kv_head_num,
    x,
    cache_latent_k,
    cache_latent_v,
    Wdq,
    Wuq,
    Wuk,
    Wuvout,
    infer_state,
):
    # Strictly isomorphic offline reference:
    # 1) precompute and cache RoPE-applied K with logical positions (offline/prefill style)
    # 2) decode step only reads cached K_rope, no online RoPE
    # 3) V path and Wuvout projection are identical to online kernel path
    bsz, q_len, model_dim = x.shape
    rank_head = cache_latent_k.shape[-1]
    assert q_len == 1
    gqa_group = q_head_num // kv_head_num
    sm_scale = 1.0 / math.sqrt(head_dim)

    x_norm = _rms_norm(x)
    latent_q = (x_norm @ Wdq).view(bsz, q_len, q_head_num, rank_head)[:, 0]
    Wuq_h = Wuq.view(q_head_num, rank_head, head_dim)
    q = torch.einsum("bhr,hrd->bhd", latent_q, Wuq_h)

    Wuk_h = Wuk.view(rank_head, kv_head_num, head_dim)
    req_to_tokens = infer_state.req_manager.req_to_token_indexs
    b_req_idx = infer_state.b_req_idx
    b_seq_len = infer_state.b_seq_len

    k_cache_rope = torch.empty(
        cache_latent_k.shape[0],
        kv_head_num,
        head_dim,
        device=x.device,
        dtype=x.dtype,
    )
    for b in range(bsz):
        req_id = int(b_req_idx[b].item())
        cur_len = int(b_seq_len[b].item())
        token_idx = req_to_tokens[req_id, :cur_len]
        latent_k_b = cache_latent_k[token_idx]  # [S, Hkv, R]
        k_full = torch.einsum("shr,rhd->shd", latent_k_b, Wuk_h)  # [S, Hkv, D]
        pos = torch.arange(cur_len, device=x.device, dtype=torch.float32)
        k_cache_rope[token_idx] = _rope(k_full, pos)

    o_rank = torch.empty(bsz, q_head_num, rank_head, device=x.device, dtype=x.dtype)
    for b in range(bsz):
        req_id = int(b_req_idx[b].item())
        cur_len = int(b_seq_len[b].item())
        token_idx = req_to_tokens[req_id, :cur_len]

        k_rope_b = k_cache_rope[token_idx]  # [S, Hkv, D]
        latent_v_b = cache_latent_v[token_idx]  # [S, Hkv, R]

        k_expand = k_rope_b.repeat_interleave(gqa_group, dim=1)  # [S, Hq, D]
        v_expand = latent_v_b.repeat_interleave(gqa_group, dim=1)  # [S, Hq, R]

        score = torch.einsum("hd,shd->hs", q[b], k_expand) * sm_scale
        prob = torch.softmax(score, dim=-1)
        o_rank[b] = torch.einsum("hs,shr->hr", prob, v_expand)

    out = o_rank.view(bsz, q_len, q_head_num * rank_head)
    out = out @ Wuvout.view(-1, q_head_num * head_dim)
    return out.view(bsz, q_len, model_dim)


def check_fp16_path(args):
    data = _prepare_inputs(
        batch_size=args.batch_size,
        q_head_num=args.q_head_num,
        kv_head_num=args.kv_head_num,
        head_dim=args.head_dim,
        seq_len=args.seq_len,
        rank_ratio=args.rank_ratio,
        intermediate_dim=args.intermediate_dim,
    )

    y_kernel = run_flash_latent_per_head_decoding_rope_layer(
        args.seq_len,
        args.head_dim,
        args.q_head_num,
        data["x"],
        data["cache_latent_k"],
        data["cache_latent_v"],
        data["Wdq"],
        data["Wdk"],
        data["Wdv"],
        data["Wuq"],
        data["Wuk"],
        data["Wuvout"],
        data["Wup"],
        data["Wgate"],
        data["Wdown"],
        data["infer_state"],
    )

    y_ref = reference_online_rope_kernel_equivalent_layer(
        args.seq_len,
        args.head_dim,
        args.q_head_num,
        args.kv_head_num,
        data["x"],
        data["cache_latent_k"],
        data["cache_latent_v"],
        data["Wdq"],
        data["Wuq"],
        data["Wuk"],
        data["Wuvout"],
        data["Wup"],
        data["Wgate"],
        data["Wdown"],
        data["infer_state"],
    )

    atol = args.atol
    rtol = args.rtol
    torch.testing.assert_close(y_kernel, y_ref, atol=atol, rtol=rtol)
    max_abs = (y_kernel - y_ref).abs().max().item()
    print(f"[FP16] kernel vs online-reference passed. max_abs={max_abs:.6f}, atol={atol}, rtol={rtol}")

    # Default numeric comparison: offline LLaMA-style vs online-kernel-style attention path.
    out_online_proj = reference_online_rope_attention_projected_only(
        args.head_dim,
        args.q_head_num,
        args.kv_head_num,
        data["x"],
        data["cache_latent_k"],
        data["cache_latent_v"],
        data["Wdq"],
        data["Wuq"],
        data["Wuk"],
        data["Wuvout"],
        data["infer_state"],
    )
    out_offline_proj = reference_llama_style_offline_rope_attention_projected_only(
        args.head_dim,
        args.q_head_num,
        args.kv_head_num,
        data["x"],
        data["cache_latent_k"],
        data["cache_latent_v"],
        data["Wdq"],
        data["Wuq"],
        data["Wuk"],
        data["Wuvout"],
        data["infer_state"],
    )
    diff = (out_online_proj - out_offline_proj).abs()
    print(
        f"[Compare] offline(isomorphic) vs online-kernel attention-proj: "
        f"max_abs={diff.max().item():.6f}, mean_abs={diff.mean().item():.6f}"
    )

    attn_offline = reference_llama_style_offline_rope_attention_only(
        args.seq_len,
        args.head_dim,
        args.q_head_num,
        args.kv_head_num,
        data["x"],
        data["cache_latent_k"],
        data["cache_latent_v"],
        data["Wdq"],
        data["Wuq"],
        data["Wuk"],
        data["Wuv"],
    )
    print(f"[Info] offline-RoPE LLaMA-style attention output shape: {tuple(attn_offline.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Functional checks for latent decode layer (online RoPE kernel).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--q-head-num", type=int, default=32)
    parser.add_argument("--kv-head-num", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--rank-ratio", type=float, default=0.5)
    parser.add_argument("--intermediate-dim", type=int, default=11008)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this functional test.")

    torch.manual_seed(args.seed)
    print("Running functional checks for decode layer...")
    print(
        f"cfg: bs={args.batch_size}, qh={args.q_head_num}, kvh={args.kv_head_num}, "
        f"d={args.head_dim}, seq={args.seq_len}, rank_ratio={args.rank_ratio}"
    )

    check_fp16_path(args)
    print("All functional checks passed.")


if __name__ == "__main__":
    main()
