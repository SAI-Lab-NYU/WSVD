import torch
from quarot.nn import Linear4bit, Quantizer, OnlineHadamard
import time
import argparse
import numpy as np
import pprint
import flashinfer

model_sizes = [
    (4096, 4096), #llava-7b
    (5120, 5120), #llava-13b
]

mlp_sizes = [
    (4096, 11008), #llava-7b
    (5120, 13824), #llava-13b
]
benchmark_dtypes = [torch.float16]
num_warmup_steps = 5
num_bench_steps = 1 #100

kv_len = 64*1024
num_kv_heads = 40
head_dim = 128
num_qo_heads = 40

def decode_base_benchmark(rank_ratio, x):
    k = torch.randn(kv_len, num_kv_heads, head_dim).half().to('cpu')
    v = torch.randn(kv_len, num_kv_heads, head_dim).half().to('cpu')
    
    # decode attention
    
    q = torch.randn(num_qo_heads, head_dim).half().to(0)
    # o = flashinfer.single_decode_with_kv_cache(q, k, v) # decode attention without RoPE on-the-fly
    # o_rope_on_the_fly = flashinfer.single_decode_with_kv_cache(q, k, v, pos_encoding_mode="ROPE_LLAMA") # decode with LLaMA style RoPE on-the-fly

    # x = x.cuda()
    
    # warmup
    for i in range(num_warmup_steps):
        k_cuda = k.to('cuda')
        v_cuda = v.to('cuda')
        # out = module(x)
        o = flashinfer.single_decode_with_kv_cache(q, k_cuda, v_cuda) # decode attention without RoPE on-the-fly
    torch.cuda.synchronize()
    
    k.to('cpu')
    v.to('cpu')
    
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        k_cuda = k.to('cuda')
        v_cuda = v.to('cuda')
        # out = module(x)
        o = flashinfer.single_decode_with_kv_cache(q, k_cuda, v_cuda) # decode attention without RoPE on-the-fly
    torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    return (end_time - start_time) * 1000 / num_bench_steps

def decode_benchmark(rank_ratio, x):
    latent_dim = int(num_kv_heads*head_dim*rank_ratio)
    
    k = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
    v = torch.randn(kv_len, num_kv_heads, head_dim).half().to(0)
    
    latent = torch.randn(kv_len, latent_dim).half().to(0)
    quantizer = Quantizer(input_clip_ratio=1.0).cuda()
    latent_4bit = quantizer(latent)
    
    dtype = torch.float16
    base_reconstruct_proj = torch.nn.Linear(latent_dim,
                                        num_kv_heads*head_dim*2, # Reconstruct both K and V
                                        bias=False).cuda().to(dtype)
    base_reconstruct_proj.weight.data = torch.randint_like(base_reconstruct_proj.weight.data,
                                                    low=-8, high=7).to(dtype)
    s_up = torch.ones((num_kv_heads*head_dim*2, 1), dtype=torch.float16, device='cuda')
    reconstruct_proj = Linear4bit.from_float(base_reconstruct_proj, weight_scales=s_up).cuda() # 4bit reconstruct proj
    
    # decode attention
    q = torch.randn(num_qo_heads, head_dim).half().to(0)

    # o = flashinfer.single_decode_with_kv_cache(q, k, v) # decode attention without RoPE on-the-fly
    # o_rope_on_the_fly = flashinfer.single_decode_with_kv_cache(q, k, v, pos_encoding_mode="ROPE_LLAMA") # decode with LLaMA style RoPE on-the-fly

    # x = x.cuda()
    
    # warmup
    for i in range(num_warmup_steps):
        # out = module(x)
        # o = flashinfer.single_decode_with_kv_cache(q, k, v) # decode attention without RoPE on-the-fly
        result = reconstruct_proj(latent_4bit)
        total_dim = num_kv_heads * head_dim
        k = result[:, :total_dim].reshape(kv_len, num_kv_heads, head_dim).contiguous()
        v = result[:, total_dim:].reshape(kv_len, num_kv_heads, head_dim).contiguous()
        # breakpoint()
        o_rope_on_the_fly = flashinfer.single_decode_with_kv_cache(q, k, v, pos_encoding_mode="ROPE_LLAMA") # decode with LLaMA style RoPE on-the-fly
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        # out = module(x)
        # o = flashinfer.single_decode_with_kv_cache(q, k, v) # decode attention without RoPE on-the-fly
        result = reconstruct_proj(latent_4bit)
        total_dim = num_kv_heads * head_dim
        k = result[:, :total_dim].reshape(kv_len, num_kv_heads, head_dim).contiguous()
        v = result[:, total_dim:].reshape(kv_len, num_kv_heads, head_dim).contiguous()
        o_rope_on_the_fly = flashinfer.single_decode_with_kv_cache(q, k, v, pos_encoding_mode="ROPE_LLAMA") # decode with LLaMA style RoPE on-the-fly
    torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    return (end_time - start_time) * 1000 / num_bench_steps

def linear4bit_benchmark(args):
        
    bsz = args.bsz
    seq_len = args.seq_len
    
    layer_size = model_sizes
    rank_ratio = args.rank_ratio
    
    times_baseline = []  
    times_ours = []  
    for i in range(10):
        times_baseline.append(decode_base_benchmark(rank_ratio=1, x=None))
        times_ours.append(decode_benchmark(rank_ratio=rank_ratio, x=None))
    print(f"Basline Decode time: {np.mean(times_baseline):.3f} +- {1.96 * np.std(times_baseline):.3f}ms")
    print(f"Ours Decode time: {np.mean(times_ours):.3f} +- {1.96 * np.std(times_ours):.3f}ms")
    print(f"Speedup: {np.mean(times_baseline) / np.mean(times_ours):.3f}x")
    
    # for (feature_dim_in, feature_dim_out) in layer_size:
    #     for dtype in benchmark_dtypes:
    #         # down/up-projection of SVD 
    #         down_dim_in = feature_dim_in
    #         down_dim_out = int(feature_dim_in * rank_ratio)
    #         down_dim_out = (down_dim_out + 63) // 64 * 64  # 向上取整到最近的64的倍数，由于kernel限制
    #         up_dim_in = down_dim_out
    #         up_dim_out = feature_dim_out
    #         real_rank_ratio = down_dim_out/feature_dim_in
            
    #         x = torch.rand((bsz,
    #                         seq_len,
    #                         feature_dim_in)).cuda().to(dtype)
            
    #         # FP16 baseline without SVD
    #         baseline_mod = torch.nn.Linear(feature_dim_in,
    #                                        feature_dim_out,
    #                                        bias=False).cuda().to(dtype)
            
    #         baseline_mod.weight.data = torch.randint_like(baseline_mod.weight.data,
    #                                                       low=-8, high=7).to(dtype)
            
    #         s_w = torch.ones((feature_dim_out, 1), dtype=torch.float16, device='cuda')
            
    #         # INT4 baseline without SVD
    #         int4_mod = torch.nn.Sequential(
    #             Quantizer(input_clip_ratio=1.0),
    #             Linear4bit.from_float(baseline_mod, weight_scales=s_w)
    #         ).cuda()

    #         # FP16 baseline with SVD
    #         baseline_down_proj = torch.nn.Linear(down_dim_in,
    #                                        down_dim_out,
    #                                        bias=False).cuda().to(dtype)
            
    #         baseline_up_proj = torch.nn.Linear(up_dim_in,
    #                                        up_dim_out,
    #                                        bias=False).cuda().to(dtype)
            
    #         baseline_down_proj.weight.data = torch.randint_like(baseline_down_proj.weight.data,
    #                                                       low=-8, high=7).to(dtype)
            
    #         baseline_up_proj.weight.data = torch.randint_like(baseline_up_proj.weight.data,
    #                                                       low=-8, high=7).to(dtype)
            
    #         s_down = torch.ones((down_dim_out, 1), dtype=torch.float16, device='cuda')
    #         s_up = torch.ones((up_dim_out, 1), dtype=torch.float16, device='cuda')
            
    #         baseline_svd = torch.nn.Sequential(
    #             baseline_down_proj,
    #             baseline_up_proj
    #         ).cuda()
            
    #         # INT4 baseline with SVD
    #         int4_svd = torch.nn.Sequential(
    #             Quantizer(input_clip_ratio=1.0),
    #             Linear4bit.from_float(baseline_down_proj, weight_scales=s_down),
                
    #             Quantizer(input_clip_ratio=1.0),
    #             Linear4bit.from_float(baseline_up_proj, weight_scales=s_up)
    #         ).cuda()
            
    #         print(f"{dtype}. Sizes: {baseline_mod.weight.shape}")
    #         print(f"SVD down projection shape: {baseline_down_proj.weight.shape}, up projection shape: {baseline_up_proj.weight.shape}")
            
    #         times_4bit = []
    #         for i in range(10):
    #             times_4bit.append(module_benchmark(int4_mod, x))
    #         print(f"Int4 time: {np.mean(times_4bit):.3f} +- {1.96 * np.std(times_4bit):.3f}ms")
            
    #         times_baseline = []
    #         for i in range(10):
    #             times_baseline.append(module_benchmark(baseline_mod, x))
    #         print(f"FP16 time: {np.mean(times_baseline):.3f} +- {1.96 * np.std(times_baseline):.3f}ms")
            
    #         # 添加SVD模型的基准测试
    #         print(f"Real rank ratio: {real_rank_ratio:.3f} (requested: {rank_ratio:.3f})")
    #         times_baseline_svd = []
    #         for i in range(10):
    #             times_baseline_svd.append(module_benchmark(baseline_svd, x))
    #         print(f"FP16+SVD time: {np.mean(times_baseline_svd):.3f} +- {1.96 * np.std(times_baseline_svd):.3f}ms")
            
    #         times_4bit_svd = []
    #         for i in range(10):
    #             times_4bit_svd.append(module_benchmark(int4_svd, x))
    #         print(f"Int4+SVD time: {np.mean(times_4bit_svd):.3f} +- {1.96 * np.std(times_4bit_svd):.3f}ms")
            
    #         print(f"Speedup (no SVD): {np.mean(times_baseline) / np.mean(times_4bit):.3f}x")
    #         print(f"Speedup (with SVD): {np.mean(times_baseline_svd) / np.mean(times_4bit_svd):.3f}x")
    #         print(f"SVD vs no SVD (FP16): {np.mean(times_baseline) / np.mean(times_baseline_svd):.3f}x")
    #         print(f"SVD vs no SVD (Int4): {np.mean(times_4bit) / np.mean(times_4bit_svd):.3f}x")
    #         print(f"FP16 no SVD vs Int4 with SVD: {np.mean(times_baseline) / np.mean(times_4bit_svd):.3f}x")
    
    #         # 更新表格输出，包含SVD结果和real_rank_ratio
    #         print(f'{feature_dim_in}x{feature_dim_out} & {args.bsz} & {real_rank_ratio:.3f} & {np.mean(times_baseline):.3f} & {np.mean(times_4bit):.3f} & {np.mean(times_baseline_svd):.3f} & {np.mean(times_4bit_svd):.3f}\\\\')
    #         print('--------------')
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bsz', type=int,
        help='Batch size',
        default=1,
    )
    parser.add_argument(
        '--seq_len', type=int,
        help='Size of the input sequence',
        default=2048,
    )
    parser.add_argument(
        '--layer_type', type=str,
        help='Type of the layer in the model (qkvv_proj [default], down_proj)',
        default='qkv_proj',
        choices=['qkv_proj', 'down_proj']
    )
    parser.add_argument(
        '--rank_ratio', type=float,
        help='Rank ratio for joint SVD, equals rank/embedding_dim',
        default=0.35,
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    linear4bit_benchmark(args)