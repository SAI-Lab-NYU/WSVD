import torch
from quarot.nn import Quantizer, OnlineHadamard # Linear4bit
import time
import argparse
import numpy as np
import pprint
import torch.nn as nn
import torch.nn.functional as F
# import flashinfer
import gemm_int8
import quarot

class Linear4bit(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        '''
        Symmetric 4-bit Linear Layer.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scales',
                             torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', (torch.randint(1, 7, (self.out_features, self.in_features // 2),
                                                             # SubByte weight
                                                             dtype=torch.uint8, requires_grad=False)))
        if bias:                                                        
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None
        
    def forward(self, x):
        #if torch.cuda.current_device() != x.device:
        #    torch.cuda.set_device(x.device)
        
        assert type(x) == quarot.PackedQuantizedTensor #Quantized input is given
        x, scales_x = x.quantized_x, x.scales_x
        #shape_handler = ShapeHandler(quantized_x)
        #quantized_x = shape_handler.flatten(quantized_x)
        x = quarot.matmul(x, self.weight)
        #out = shape_handler.unflatten(
        #    quarot.sym_dequant(int_result, scales_x, self.weight_scales))
        if self.bias is not None:
            return quarot.sym_dequant(x, scales_x, self.weight_scales) + self.bias
        else:
            return quarot.sym_dequant(x, scales_x, self.weight_scales)

    @staticmethod
    def from_float(module: torch.nn.Linear, weight_scales=None,):
        '''
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        weight_matrix = module.weight.data
        
        
        int_module = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, dtype=weight_matrix.dtype).to(weight_matrix.dtype)
        if weight_scales is not None:
            assert weight_scales.shape == (module.out_features, 1), 'weight_scales should have shape (out_features, 1)'
            weight_matrix = weight_matrix.cuda()
            int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
            int_rounded_weight = (weight_matrix/weight_scales.cuda()).round()
            int_module.weight.copy_(quarot.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())
        
            if module.bias is not None:
                int_module.bias.copy_(module.bias)
        
        return int_module

class Quantizer8bit(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio
    
    def forward(self, x):
        scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(-1)/128).to(torch.float16) * self.input_clip_ratio
        quantized_x = torch.clamp(x/scales_x, -128, 127).to(torch.int8)
        packed_tensor = quarot.PackedQuantizedTensor(quantized_x, scales_x)
        return packed_tensor
       
class Linear8bit(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        '''
        Symmetric 8-bit Linear Layer.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scales',
                             torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', (torch.randint(-128, 127, (self.out_features, self.in_features),
                                                             # SubByte weight
                                                             dtype=torch.int8, requires_grad=False)))
        if bias:                                                        
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None
        
    def forward(self, x):
        #if torch.cuda.current_device() != x.device:
        #    torch.cuda.set_device(x.device)
        
        assert type(x) == quarot.PackedQuantizedTensor #Quantized input is given
        x, scales_x = x.quantized_x, x.scales_x
        #shape_handler = ShapeHandler(quantized_x)
        #quantized_x = shape_handler.flatten(quantized_x)
        # x = quarot.matmul(x, self.weight)
        #breakpoint()
        B, M, K_in = x.shape
        N_out = self.weight.shape[0] # 等同于 self.out_features
        
        # 将输入激活 reshape 为 2D: (B*M, K_in)
        x = x.reshape(-1, K_in)
        x = gemm_int8.matmul(x, self.weight, alpha=1).to(torch.int32)
        x = x.reshape(B, M, N_out)
        #out = shape_handler.unflatten(
        #    quarot.sym_dequant(int_result, scales_x, self.weight_scales))
        if self.bias is not None:
            return quarot.sym_dequant(x, scales_x, self.weight_scales) + self.bias
        else:
            #breakpoint()
            return quarot.sym_dequant(x, scales_x, self.weight_scales)

    @staticmethod
    def from_float(module: torch.nn.Linear, weight_scales=None,):
        '''
        Generate a new Linear8bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        weight_matrix = module.weight.data
        
        
        int_module = Linear8bit(module.in_features, module.out_features, bias=module.bias is not None, dtype=weight_matrix.dtype).to(weight_matrix.dtype)
        if weight_scales is not None:
            assert weight_scales.shape == (module.out_features, 1), 'weight_scales should have shape (out_features, 1)'
            weight_matrix = weight_matrix.cuda()
            int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
            int_rounded_weight = (weight_matrix/weight_scales.cuda()).round()
            # int_module.weight.copy_(quarot.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())
            int_module.weight.copy_(int_rounded_weight.to(torch.int8).cpu())
        
            if module.bias is not None:
                int_module.bias.copy_(module.bias)
        
        return int_module
    
    
    
    
model_sizes = [
    (4096, 4096*3), #llava-7b
    (5120, 5120*3), #llava-13b
]

mlp_sizes = [
    (4096, 11008), #llava-7b
    (5120, 13824), #llava-13b
]
benchmark_dtypes = [torch.float16]
num_warmup_steps = 5
num_bench_steps = 100

class CausalSelfAttention(nn.Module):

    def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, dropout:float=0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        query_projected = self.c_attn(x)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        # if self.training:
        #     dropout = self.dropout
        #     is_causal = self.is_causal
        # else:
        #     dropout = 0.0
        #     is_causal = False

        y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0., is_causal=False)
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y

class SVDSelfAttention(nn.Module):

    def __init__(self, num_heads: int, embed_dimension: int, rank_ratio: float, bias: bool=False, is_causal: bool=False, dropout:float=0.0):
        super().__init__()
        assert embed_dimension % num_heads == 0
        self.rank = int(embed_dimension * rank_ratio)
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dimension, self.rank, bias=bias)
        self.up_proj = nn.Linear(self.rank, embed_dimension*3, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
        # regularization
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.embed_dimension = embed_dimension
        # Perform causal masking
        self.is_causal = is_causal

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        latent = self.c_attn(x)
        query_projected = self.up_proj(latent)

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        # if self.training:
        #     dropout = self.dropout
        #     is_causal = self.is_causal
        # else:
        #     dropout = 0.0
        #     is_causal = False

        y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        # y = flashinfer.batch_prefill_with_kv_cache(q, k, v, causal=True, pos_encoding_mode="ROPE_LLAMA")
        y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

        y = self.resid_dropout(self.c_proj(y))
        return y



def module_benchmark(module, x):
    x = x.cuda()
    
    # warmup
    for i in range(num_warmup_steps):
        out = module(x)
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for i in range(num_bench_steps):
        out = module(x)
    torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    
    return (end_time - start_time) * 1000 / num_bench_steps

def linear4bit_benchmark(args):
        
    bsz = args.bsz
    seq_len = args.seq_len
    
    layer_size = model_sizes
    rank_ratio = args.rank_ratio
        
    
    for (feature_dim_in, feature_dim_out) in layer_size:
        for dtype in benchmark_dtypes:
            # down/up-projection of SVD 
            down_dim_in = feature_dim_in
            down_dim_out = int(feature_dim_in * rank_ratio)
            down_dim_out = (down_dim_out + 63) // 64 * 64  # 向上取整到最近的64的倍数，由于kernel限制
            up_dim_in = down_dim_out
            up_dim_out = feature_dim_out
            real_rank_ratio = down_dim_out/feature_dim_in
            
            print(f"Feature dim in: {feature_dim_in}, out: {feature_dim_out}")
            print(f"Down projection dim in: {down_dim_in}, out: {down_dim_out}")
            print(f"Up projection dim in: {up_dim_in}, out: {up_dim_out}")
            x = torch.rand((bsz,
                            seq_len,
                            feature_dim_in)).cuda().to(dtype)
            
            # FP16 baseline without SVD
            baseline_mod = torch.nn.Linear(feature_dim_in,
                                           feature_dim_out,
                                           bias=False).cuda().to(dtype)
            
            baseline_mod.weight.data = torch.randint_like(baseline_mod.weight.data,
                                                          low=-8, high=7).to(dtype)
            
            s_w = torch.ones((feature_dim_out, 1), dtype=torch.float16, device='cuda')
            
            # INT8 baseline without SVD
            int8_mod = torch.nn.Sequential(
                Quantizer8bit(input_clip_ratio=1.0),
                Linear8bit.from_float(baseline_mod, weight_scales=s_w)
            ).cuda()
            
            # INT4 baseline without SVD
            int4_mod = torch.nn.Sequential(
                Quantizer(input_clip_ratio=1.0),
                Linear4bit.from_float(baseline_mod, weight_scales=s_w)
            ).cuda()

            # FP16 baseline with SVD
            baseline_down_proj = torch.nn.Linear(down_dim_in,
                                           down_dim_out,
                                           bias=False).cuda().to(dtype)
            
            baseline_up_proj = torch.nn.Linear(up_dim_in,
                                           up_dim_out,
                                           bias=False).cuda().to(dtype)
            
            baseline_down_proj.weight.data = torch.randint_like(baseline_down_proj.weight.data,
                                                          low=-8, high=7).to(dtype)
            
            baseline_up_proj.weight.data = torch.randint_like(baseline_up_proj.weight.data,
                                                          low=-8, high=7).to(dtype)
            
            s_down = torch.ones((down_dim_out, 1), dtype=torch.float16, device='cuda')
            s_up = torch.ones((up_dim_out, 1), dtype=torch.float16, device='cuda')
            
            baseline_svd = torch.nn.Sequential(
                baseline_down_proj,
                baseline_up_proj
            ).cuda()
            
            # INT8 baseline with SVD
            int8_svd = torch.nn.Sequential(
                Quantizer8bit(input_clip_ratio=1.0),
                Linear8bit.from_float(baseline_down_proj, weight_scales=s_down),
                
                Quantizer8bit(input_clip_ratio=1.0),
                Linear8bit.from_float(baseline_up_proj, weight_scales=s_up)
            ).cuda()
            
            # INT4 baseline with SVD
            int4_svd = torch.nn.Sequential(
                Quantizer(input_clip_ratio=1.0),
                Linear4bit.from_float(baseline_down_proj, weight_scales=s_down),
                
                Quantizer(input_clip_ratio=1.0),
                Linear4bit.from_float(baseline_up_proj, weight_scales=s_up)
            ).cuda()
            
            print(f"{dtype}. Sizes: {baseline_mod.weight.shape}")
            print(f"SVD down projection shape: {baseline_down_proj.weight.shape}, up projection shape: {baseline_up_proj.weight.shape}")
            
            
            times_baseline = []
            for i in range(10):
                times_baseline.append(module_benchmark(baseline_mod, x))
            print(f"FP16 time: {np.mean(times_baseline):.3f} +- {1.96 * np.std(times_baseline):.3f}ms")
            
            times_8bit = []
            for i in range(10):
                times_8bit.append(module_benchmark(int8_mod, x))
            print(f"Int8 time: {np.mean(times_8bit):.3f} +- {1.96 * np.std(times_8bit):.3f}ms")
            
            times_4bit = []
            for i in range(10):
                times_4bit.append(module_benchmark(int4_mod, x))
            print(f"Int4 time: {np.mean(times_4bit):.3f} +- {1.96 * np.std(times_4bit):.3f}ms")
                  
            # 添加SVD模型的基准测试
            print(f"Real rank ratio: {real_rank_ratio:.3f} (requested: {rank_ratio:.3f})")
            times_baseline_svd = []
            for i in range(10):
                times_baseline_svd.append(module_benchmark(baseline_svd, x))
            print(f"FP16+SVD time: {np.mean(times_baseline_svd):.3f} +- {1.96 * np.std(times_baseline_svd):.3f}ms")
            
            times_8bit_svd = []
            for i in range(10):
                times_8bit_svd.append(module_benchmark(int8_svd, x))
            print(f"Int8+SVD time: {np.mean(times_8bit_svd):.3f} +- {1.96 * np.std(times_8bit_svd):.3f}ms")
            
            times_4bit_svd = []
            for i in range(10):
                times_4bit_svd.append(module_benchmark(int4_svd, x))
            print(f"Int4+SVD time: {np.mean(times_4bit_svd):.3f} +- {1.96 * np.std(times_4bit_svd):.3f}ms")
            
            print(f"Speedup (no SVD): {np.mean(times_baseline) / np.mean(times_4bit):.3f}x")
            print(f"Speedup (with SVD): {np.mean(times_baseline_svd) / np.mean(times_4bit_svd):.3f}x")
            print(f"SVD vs no SVD (FP16): {np.mean(times_baseline) / np.mean(times_baseline_svd):.3f}x")
            print(f"SVD vs no SVD (Int4): {np.mean(times_4bit) / np.mean(times_4bit_svd):.3f}x")
            print(f"FP16 no SVD vs Int4 with SVD: {np.mean(times_baseline) / np.mean(times_4bit_svd):.3f}x")
    
            # 更新表格输出，包含SVD结果和real_rank_ratio
            print(f'{feature_dim_in}x{feature_dim_out} & {args.bsz} & {real_rank_ratio:.3f} & {np.mean(times_baseline):.3f} & {np.mean(times_4bit):.3f} & {np.mean(times_baseline_svd):.3f} & {np.mean(times_4bit_svd):.3f}\\\\')
            print('--------------')
            
def self_attention_benchmark(args):

    bsz = args.bsz
    seq_len = args.seq_len
    layer_type = args.layer_type
    rank_ratio = args.rank_ratio

    num_heads = 32
    heads_per_dim = 128
    embed_dimension = num_heads * heads_per_dim
    dtype = torch.float16
    
    base_model_7b = CausalSelfAttention(num_heads=32, embed_dimension=4096, bias=False, is_causal=False).to("cuda").to(dtype).eval()
    base_model_13b = CausalSelfAttention(num_heads=40, embed_dimension=5120, bias=False, is_causal=False).to("cuda").to(dtype).eval()
    svd_model_7b = SVDSelfAttention(num_heads=32, embed_dimension=4096, rank_ratio=rank_ratio, bias=False, is_causal=False).to("cuda").to(dtype).eval()
    svd_model_13b = SVDSelfAttention(num_heads=40, embed_dimension=5120, rank_ratio=rank_ratio,bias=False, is_causal=False).to("cuda").to(dtype).eval()
    
    x = torch.rand((bsz, seq_len, 4096)).cuda().to(dtype)
    times_baseline_7b = []
    for i in range(10):
        times_baseline_7b.append(module_benchmark(base_model_7b, x))
    print(f"FP16 time: {np.mean(times_baseline_7b):.3f} +- {1.96 * np.std(times_baseline_7b):.3f}ms")
    
    # 添加SVD模型的基准测试
    print(f"Rank ratio: {rank_ratio:.3f}")
    times_baseline_svd_7b = []
    for i in range(10):
        times_baseline_svd_7b.append(module_benchmark(svd_model_7b, x))
    print(f"FP16+SVD time: {np.mean(times_baseline_svd_7b):.3f} +- {1.96 * np.std(times_baseline_svd_7b):.3f}ms")
    
    x = torch.rand((bsz, seq_len, 5120)).cuda().to(dtype)
    times_baseline_13b = []
    for i in range(10):
        times_baseline_13b.append(module_benchmark(base_model_13b, x))
    print(f"FP16 time: {np.mean(times_baseline_13b):.3f} +- {1.96 * np.std(times_baseline_13b):.3f}ms")
    
    # 添加SVD模型的基准测试
    print(f"Rank ratio: {rank_ratio:.3f}")
    times_baseline_svd_13b = []
    for i in range(10):
        times_baseline_svd_13b.append(module_benchmark(svd_model_13b, x))
    print(f"FP16+SVD time: {np.mean(times_baseline_svd_13b):.3f} +- {1.96 * np.std(times_baseline_svd_13b):.3f}ms")
    
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
        default=0.4,
    )
    
    args = parser.parse_args()
    pprint.pprint(vars(args))
    # linear4bit_benchmark(args)
    self_attention_benchmark(args)