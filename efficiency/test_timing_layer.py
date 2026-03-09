import torch
import sys
import os
import time
import gc
from copy import deepcopy
import statistics
from layer_utils import (
    create_test_data_layer,
    reference_attention_rope_layer,
    reference_eager_attention_layer,
    run_flash_latent_per_head_decoding_rope_layer,
    run_flash_int8_latent_per_head_decoding_rope_layer,
)


def warmup_and_benchmark(func, warmup_runs=10, benchmark_runs=20, *args, **kwargs):
    """Run warmup iterations first, then benchmark and report average time."""
    # Clean memory and cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Warmup phase
    print(f"  Starting warmup ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        try:
            _ = func(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            print(f"    Warmup {i+1} failed: {e}")
            return None, None, None
    
    # Clean up state after warmup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Benchmark phase
    print(f"  Starting benchmark ({benchmark_runs} runs)...")
    times = []
    for i in range(benchmark_runs):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            start_time = time.perf_counter()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            result = func(*args, **kwargs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds
            times.append(elapsed_time)
            
        except Exception as e:
            print(f"    Benchmark run {i+1} failed: {e}")
            return None, None, None
    
    # Compute statistics
    if len(times) == 0:
        return None, None, None
    
    mean_time = statistics.mean(times)
    median_time = statistics.median(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    
    return result, {
        'mean': mean_time,
        'median': median_time,
        'std': std_time,
        'min': min_time,
        'max': max_time,
        'times': times
    }

def reset_gpu_state():
    """Reset GPU state to keep the test environment clean."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        # Force CUDA context reset
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
def test_eager_attention_timing_layer(batch_size, q_head_num, head_dim, seq_len, rank_ratio=1.0, kv_head_num=None, intermediate_dim=None):
    """Test timing performance of the reference Attention implementation."""
    print("\nTesting timing performance of reference Attention...")
    
    # Reset GPU state
    reset_gpu_state()
    
    # Create test data
    input, cache_latent_k, cache_latent_v, Wdq, Wdk, Wdv, Wuq, Wuk, Wuv, Wq, Wk, Wv, cache_k, cache_v, Wout, Wup, Wgate, Wdown, infer_state = create_test_data_layer(
        batch_size, q_head_num, head_dim, seq_len, rank_ratio, kv_head_num, intermediate_dim
    )
    cache_k = cache_k.view(batch_size, seq_len, kv_head_num, head_dim).permute(0, 2, 1, 3)
    cache_v = cache_v.view(batch_size, seq_len, kv_head_num, head_dim).permute(0, 2, 1, 3)
    del cache_latent_k, cache_latent_v, Wdq, Wdk, Wdv, Wuq, Wuk, Wuv, infer_state
    
    try:
        output_eager_attention, timing_eager_attention = warmup_and_benchmark(
            reference_eager_attention_layer, 10, 50, seq_len, head_dim, q_head_num, input, cache_k, cache_v, Wq, Wk, Wv, Wout, Wup, Wgate, Wdown
        )
        
        if timing_eager_attention:
            print("✅ Reference Eager Attention execution succeeded")
            print(f"  Mean time: {timing_eager_attention['mean']:.2f}ms")
            print(f"  Median time: {timing_eager_attention['median']:.2f}ms")
            print(f"  Std dev: {timing_eager_attention['std']:.2f}ms")
            print(f"  Min time: {timing_eager_attention['min']:.2f}ms")
            print(f"  Max time: {timing_eager_attention['max']:.2f}ms")
        else:
            print("❌ Reference Eager Attention execution failed")
            return None
            
        # Clean up data
        del input, cache_k, cache_v, Wq, Wk, Wv, Wout, Wup, Wgate, Wdown, output_eager_attention
        reset_gpu_state()
        
        return timing_eager_attention
        
    except Exception as e:
        print(f"❌ Reference Eager Attention execution failed: {e}")
        import traceback
        traceback.print_exc()
        # Clean up data
        del input, cache_k, cache_v, Wq, Wk, Wv, Wout, Wup, Wgate, Wdown
        reset_gpu_state()
        return None
    
def test_attention_reference_timing_layer_rope(batch_size, q_head_num, head_dim, seq_len, rank_ratio=1.0, kv_head_num=None, intermediate_dim=None):
    """Test timing performance of reference Attention with RoPE."""
    print("\nTesting timing performance of reference Attention with RoPE...")
    
    # Reset GPU state
    reset_gpu_state()
    
    # Create test data
    input, cache_latent_k, cache_latent_v, Wdq, Wdk, Wdv, Wuq, Wuk, Wuv, Wq, Wk, Wv, cache_k, cache_v, Wout, Wup, Wgate, Wdown, infer_state = create_test_data_layer(
        batch_size, q_head_num, head_dim, seq_len, rank_ratio, kv_head_num, intermediate_dim
    )
    cache_k = cache_k.view(batch_size, seq_len, kv_head_num, head_dim).permute(0, 2, 1, 3)
    cache_v = cache_v.view(batch_size, seq_len, kv_head_num, head_dim).permute(0, 2, 1, 3)
    del cache_latent_k, cache_latent_v, Wdq, Wdk, Wdv, Wuq, Wuk, Wuv, infer_state
    
    try:
        output_rope_attention, timing_rope_attention = warmup_and_benchmark(
            reference_attention_rope_layer, 10, 50, seq_len, head_dim, q_head_num, input, cache_k, cache_v, Wq, Wk, Wv, Wout, Wup, Wgate, Wdown
        )
        
        if timing_rope_attention:
            print("✅ Reference Attention with RoPE execution succeeded")
            print(f"  Mean time: {timing_rope_attention['mean']:.2f}ms")
            print(f"  Median time: {timing_rope_attention['median']:.2f}ms")
            print(f"  Std dev: {timing_rope_attention['std']:.2f}ms")
            print(f"  Min time: {timing_rope_attention['min']:.2f}ms")
            print(f"  Max time: {timing_rope_attention['max']:.2f}ms")
        else:
            print("❌ Reference Attention with RoPE execution failed")
            return None
            
        # Clean up data
        del input, cache_k, cache_v, Wq, Wk, Wv, Wout, Wup, Wgate, Wdown, output_rope_attention
        reset_gpu_state()
        
        return timing_rope_attention
        
    except Exception as e:
        print(f"❌ Reference Attention with RoPE execution failed: {e}")
        import traceback
        traceback.print_exc()
        # Clean up data
        del input, cache_k, cache_v, Wq, Wk, Wv, Wout, Wup, Wgate, Wdown
        reset_gpu_state()
        return None
    
def test_flash_latent_per_head_decoding_rope_timing_layer(batch_size, q_head_num, head_dim, seq_len, rank_ratio=1.0, kv_head_num=None, intermediate_dim=None):
    """Test timing performance of Flash latent per-head decoding with RoPE."""
    print("\nTesting timing performance of Flash latent per-head decoding with RoPE...")
    
    # Reset GPU state
    reset_gpu_state()
    
    # Create test data
    input, cache_latent_k, cache_latent_v, Wdq, Wdk, Wdv, Wuq, Wuk, Wuv, Wq, Wk, Wv, cache_k, cache_v, Wout, Wup, Wgate, Wdown, infer_state = create_test_data_layer(
        batch_size, q_head_num, head_dim, seq_len, rank_ratio, kv_head_num, intermediate_dim
    )

    Wuv = Wuv.view(kv_head_num, -1, head_dim)
    Wuv_broadcast = Wuv.repeat(q_head_num // kv_head_num, 1, 1)
    Wout = Wout.view(q_head_num, -1, head_dim * q_head_num)
    Wuvout = Wuv_broadcast @ Wout
    
    cache_latent_k = cache_latent_k.view(batch_size*seq_len, kv_head_num, -1)
    cache_latent_v = cache_latent_v.view(batch_size*seq_len, kv_head_num, -1)
    
    del cache_k, cache_v, Wuv, Wq, Wk, Wv, Wout, Wuv_broadcast
    
    try:
        output_flash_per_head, timing_flash_per_head = warmup_and_benchmark(
            run_flash_latent_per_head_decoding_rope_layer, 10, 50, seq_len, head_dim, q_head_num, input, cache_latent_k, cache_latent_v, Wdq, Wdk, Wdv, Wuq, Wuk, Wuvout, Wup, Wgate, Wdown, infer_state
        )
        
        if timing_flash_per_head:
            print("✅ Flash latent per-head decoding with RoPE succeeded")
            print(f"  Mean time: {timing_flash_per_head['mean']:.2f}ms")
            print(f"  Median time: {timing_flash_per_head['median']:.2f}ms")
            print(f"  Std dev: {timing_flash_per_head['std']:.2f}ms")
            print(f"  Min time: {timing_flash_per_head['min']:.2f}ms")
            print(f"  Max time: {timing_flash_per_head['max']:.2f}ms")
        else:
            print("❌ Flash latent per-head decoding with RoPE failed")
            return None
            
        # Clean up data
        del input, cache_latent_k, cache_latent_v, Wdq, Wdk, Wdv, Wuq, Wuk, Wuvout, Wup, Wgate, Wdown, output_flash_per_head
        reset_gpu_state()
        
        return timing_flash_per_head
        
    except Exception as e:
        print(f"❌ Flash latent per-head decoding with RoPE failed: {e}")
        import traceback
        traceback.print_exc()
        # Clean up data
        del input, cache_latent_k, cache_latent_v, Wdq, Wdk, Wdv, Wuq, Wuk, Wuvout, Wup, Wgate, Wdown
        reset_gpu_state()
        return None
    
def test_flash_int8_latent_per_head_decoding_rope_timing_layer(batch_size, q_head_num, head_dim, seq_len, rank_ratio=1.0, kv_head_num=None, intermediate_dim=None):
    """Test timing performance of Flash int8 latent per-head decoding with RoPE."""
    print("\nTesting timing performance of Flash int8 latent per-head decoding with RoPE...")
    
    # Reset GPU state
    reset_gpu_state()
    
    # Create test data
    input, cache_latent_k, cache_latent_v, Wdq, Wdk, Wdv, Wuq, Wuk, Wuv, Wq, Wk, Wv, cache_k, cache_v, Wout, Wup, Wgate, Wdown, infer_state = create_test_data_layer(
        batch_size, q_head_num, head_dim, seq_len, rank_ratio, kv_head_num, intermediate_dim
    )

    Wuv = Wuv.view(kv_head_num, -1, head_dim)
    Wuv_broadcast = Wuv.repeat(q_head_num // kv_head_num, 1, 1)
    Wout = Wout.view(q_head_num, -1, head_dim * q_head_num)
    Wuvout = Wuv_broadcast @ Wout
    
    cache_latent_k = cache_latent_k.view(batch_size*seq_len, kv_head_num, -1)
    cache_latent_v = cache_latent_v.view(batch_size*seq_len, kv_head_num, -1)
    
    # int8 quant scale
    cache_latent_k_scale = cache_latent_k.abs().max(dim=-1).values / 128 # [batch*seqlen, kv_head_num]
    cache_latent_v_scale = cache_latent_v.abs().max(dim=-1).values / 128 # [batch*seqlen, kv_head_num]
    
    Wuk_scale = Wuk.abs().max(dim=0).values.view(1, -1) / 128 # [1, head_dim * head_num_kv]
    
    cache_latent_k = (cache_latent_k/cache_latent_k_scale[:, :, None]).to(torch.int8)
    cache_latent_v = (cache_latent_v/cache_latent_v_scale[:, :, None]).to(torch.int8)
    Wuk = (Wuk/Wuk_scale).to(torch.int8)
    
    del cache_k, cache_v, Wuv, Wq, Wk, Wv, Wout, Wuv_broadcast
    
    try:
        output_flash_int8_per_head, timing_flash_int8_per_head = warmup_and_benchmark(
            run_flash_int8_latent_per_head_decoding_rope_layer, 10, 50, 
            seq_len, head_dim, q_head_num, 
            input, 
            cache_latent_k, cache_latent_k_scale, 
            cache_latent_v, cache_latent_v_scale, 
            Wdq, Wdk, Wdv, Wuq, 
            Wuk, Wuk_scale, 
            Wuvout, Wup, Wgate, Wdown, infer_state
        )
        
        if timing_flash_int8_per_head:
            print("✅ Flash int8 latent per-head decoding with RoPE succeeded")
            print(f"  Mean time: {timing_flash_int8_per_head['mean']:.2f}ms")
            print(f"  Median time: {timing_flash_int8_per_head['median']:.2f}ms")
            print(f"  Std dev: {timing_flash_int8_per_head['std']:.2f}ms")
            print(f"  Min time: {timing_flash_int8_per_head['min']:.2f}ms")
            print(f"  Max time: {timing_flash_int8_per_head['max']:.2f}ms")
        else:
            print("❌ Flash int8 latent per-head decoding with RoPE failed")
            return None
            
        # Clean up data
        del input, cache_latent_k, cache_latent_v, Wdq, Wdk, Wdv, Wuq, Wuk, Wuk_scale, cache_latent_k_scale, cache_latent_v_scale, Wuvout, Wup, Wgate, Wdown, output_flash_int8_per_head
        reset_gpu_state()
        
        return timing_flash_int8_per_head
        
    except Exception as e:
        print(f"❌ Flash int8 latent per-head decoding with RoPE failed: {e}")
        import traceback
        traceback.print_exc()
        # Clean up data
        del input, cache_latent_k, cache_latent_v, Wdq, Wdk, Wdv, Wuq, Wuk, Wuk_scale, cache_latent_k_scale, cache_latent_v_scale, Wuvout, Wup, Wgate, Wdown
        reset_gpu_state()
        return None
    
def test_timing():
    """Benchmark timing performance of different implementations."""
    print("Starting timing benchmark...")
    torch.manual_seed(42)
    
    # Test parameters
    q_head_num = 32 # llava 7B: 32, 40: llava 13B
    kv_head_num = 32 # GQA llama3 8B: 8
    intermediate_dim = 11008 # llama3 8B: 14336, llava 7B: 11008, llava 13B: 13824
    param_ratio = 0.5
    # rank_ratio = param_ratio / (1 + 1/q_head_num)
    rank_ratio = 0.5
    batch_size = 16 # 16
    
    head_dim = 128
    seq_len = 8192 # 8192 # 32768 // 2
    # rank = int(rank_ratio * head_dim) * kv_head_num
    rank_head = int(rank_ratio * head_dim)
    
    print("Test configuration:")
    print(f"  batch_size: {batch_size}")
    print(f"  q_head_num: {q_head_num}")
    print(f"  kv_head_num: {kv_head_num}")
    print(f"  head_dim: {head_dim}")
    print(f"  intermediate_dim: {intermediate_dim}")
    print(f"  seq_len: {seq_len}")
    # print(f"  rank: {rank}")
    print(f"  rank_head: {rank_head}")
    print(f"  rank_ratio: {rank_ratio}")
    print(f"  param_ratio: {param_ratio}")
    
    # Ensure initial state is clean
    reset_gpu_state()
    
    # Benchmark each implementation
    timing_eager_attention = test_eager_attention_timing_layer(
        batch_size, q_head_num, head_dim, seq_len, rank_ratio, kv_head_num, intermediate_dim
    )
    
    timing_ref_attention = test_attention_reference_timing_layer_rope(
        batch_size, q_head_num, head_dim, seq_len, rank_ratio, kv_head_num, intermediate_dim
    )
    
    timing_flash_per_head_rope = test_flash_latent_per_head_decoding_rope_timing_layer(
        batch_size, q_head_num, head_dim, seq_len, rank_ratio, kv_head_num, intermediate_dim
    )
    
    timing_flash_int8_per_head_rope = test_flash_int8_latent_per_head_decoding_rope_timing_layer(
        batch_size, q_head_num, head_dim, seq_len, rank_ratio, kv_head_num, intermediate_dim
    )

    
    rank_k_head = rank_head # Assume latent is per-head, so rank is per head
    print(f"  rank_k_head: {rank_k_head}")
    print(f"  speed_up_rope: {timing_ref_attention['mean'] / timing_flash_per_head_rope['mean']:.2f}x")
    print(f"  speed_up_int8_rope: {timing_ref_attention['mean'] / timing_flash_int8_per_head_rope['mean']:.2f}x")



if __name__ == "__main__":
    print("Latent-based Flash Decoding Timing Benchmark")
    print("="*50)
    
    # Basic timing benchmark
    test_timing()
    
    print("\nTiming benchmark completed!")
