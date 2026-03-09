import torch
from e2e.quantized_llava.hadamard import *
import int4_inference
import logging
# import quarot

class Quantizer(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio
    
    def forward(self, x):
        x = x.reshape(-1, x.shape[-1]).contiguous().float()  # flatten (batch, token, embedding) to (batch*token, embedding)
        scales_x = (int4_inference.max_abs_per_token(x)[0]/7) * self.input_clip_ratio
        # pq_x_output = int4_inference.int4_quant_and_pack_per_tensor(x, scales_x)
        pq_x_output = int4_inference.int4_quant_and_pack_per_token(x, scales_x)#  2d, 1d input,
        pq_x = pq_x_output[0]
        return (pq_x, scales_x)


class Linear4bit(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        '''
        Symmetric 4-bit Linear Layer.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scales',
                             torch.zeros((self.out_features), requires_grad=False))
        self.register_buffer('weight', (torch.randint(1, 7, (self.out_features, self.in_features // 2),
                                                             # SubByte weight
                                                             dtype=torch.int8, requires_grad=False)))
        self.dtype = dtype                                                            
        if bias:                                                        
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None
        
    def forward(self, x):
        #if torch.cuda.current_device() != x.device:
        #    torch.cuda.set_device(x.device)
        
        pq_x, scales_x = x
        scales_weight = self.weight_scales.contiguous().to(torch.float32)
        #shape_handler = ShapeHandler(quantized_x)
        #quantized_x = shape_handler.flatten(quantized_x)
        x = int4_inference.int4_gemm_per_token_fuse_dequant(pq_x, self.weight, scales_x, scales_weight)
        #out = shape_handler.unflatten(
        #    quarot.sym_dequant(int_result, scales_x, self.weight_scales))
        #  pq_x: torch.Size([577, 512]), torch.int8
        #  weight: torch.Size([1024, 512]), torch.uint8
        #  scales_x: torch.Size([577]), torch.float32
        #  scales_weight: torch.Size([1024]), torch.float32
        self.acttime = x[2]
        x = x[0].to(self.dtype) # x[0] is fp32, make sure to convert to dtype
        
        if self.bias is not None:
            return x.unsqueeze(0) + self.bias
        else:
            return x.unsqueeze(0)

    @staticmethod
    def from_float(module: torch.nn.Linear, weight_scales=None,):
        '''
        Generate a new Linear4bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        weight_matrix = module.weight.data
        
        
        int_module = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, dtype=torch.float16)
        # # use quarot
        # weight_matrix = module.weight.data.cuda().float()
        # weight_scales = (int4_inference.max_abs_per_token(weight_matrix)[0]/7)
        # int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
        # int_rounded_weight = (weight_matrix/weight_scales.unsqueeze(1).cuda()).round()
        # int_module.weight.copy_(quarot.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())
        # if module.bias is not None:
        #     int_module.bias.copy_(module.bias)
        # # use haiyu
        weight_scales = (int4_inference.max_abs_per_token(weight_matrix)[0]/7)
        int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
        # do not know weight_scale here need flatten(1) or not
        pq_w_output = int4_inference.int4_quant_and_pack_per_token(weight_matrix, weight_scales)
        int_module.weight.copy_(pq_w_output[0])
        int_module.w_time = pq_w_output[1]
        if module.bias is not None:
            int_module.bias.copy_(module.bias)
        return int_module

# class Quantizer(torch.nn.Module):
#     def __init__(self, input_clip_ratio=1.0):
#         super().__init__()
#         self.input_clip_ratio = input_clip_ratio
    
#     def forward(self, x):
#         # scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(1)/7).to(torch.float32) * self.input_clip_ratio
#         scales_x = (torch.max(torch.abs(x), dim=-1)[0]/7).float() * self.input_clip_ratio
#         pq_x_output = int4_inference.int4_quant_and_pack_per_token(x[0], scales_x[0])#  2d, 1d input,
#         pq_x, time, _ = pq_x_output
#         self.x_time = time
#         logging.info(f"kernel time {time}")
#         return (pq_x, scales_x)


# class Linear4bit(torch.nn.Module):
#     def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
#         '''
#         Symmetric 4-bit Linear Layer.
#         '''
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.register_buffer('weight_scales',
#                              torch.zeros((self.out_features), requires_grad=False))
#         self.register_buffer('weight', (torch.randint(1, 7, (self.out_features, self.in_features // 2),
#                                                              # SubByte weight
#                                                              dtype=torch.uint8, requires_grad=False)))
#         if bias:                                                        
#             self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
#         else:
#             self.bias = None
        
#     def forward(self, x):
#         #if torch.cuda.current_device() != x.device:
#         #    torch.cuda.set_device(x.device)
        
#         pq_x, scales_x = x
#         scales_x = scales_x.contiguous().to(torch.float32)
#         scales_weight = self.weight_scales.contiguous().to(torch.float32)
#         #shape_handler = ShapeHandler(quantized_x)
#         #quantized_x = shape_handler.flatten(quantized_x)
#         x = int4_inference.int4_gemm_per_token_fuse_dequant(pq_x, self.weight, scales_x[0], scales_weight)
#         #out = shape_handler.unflatten(
#         #    quarot.sym_dequant(int_result, scales_x, self.weight_scales))
        
#         # x = x[0].to(dtype) # x[0] is fp32, make sure to convert to dtype
        
#         if self.bias is not None:
#             return x + self.bias
#         else:
#             return x

#     @staticmethod
#     def from_float(module: torch.nn.Linear, weight_scales=None,):
#         '''
#         Generate a new Linear4bit module from a FP16 Linear module.
#         The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
#         routine. We will convert it to subByte representation and save it in the int_weight buffer.
#         '''
#         weight_matrix = module.weight.data
        
        
#         int_module = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, dtype=weight_matrix.dtype).to(weight_matrix.dtype)
#         weight_matrix = weight_matrix.cuda()
#         scales_w = (torch.max(torch.abs(weight_matrix), dim=-1)[0]/7).float()
#         int_module.weight_scales.copy_(scales_w.to(weight_matrix.dtype))
#         pq_w_output = int4_inference.int4_quant_and_pack_per_token(weight_matrix, scales_w)
#         int_module.weight.copy_(pq_w_output[0])
#         int_module.w_time = pq_w_output[1]
#         if module.bias is not None:
#             int_module.bias.copy_(module.bias)
#         # if weight_scales is not None:
#         #     assert weight_scales.shape == (module.out_features, 1), 'weight_scales should have shape (out_features, 1)'
#         #     weight_matrix = weight_matrix.cuda()
#         #     int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
#         #     int_rounded_weight = (weight_matrix/weight_scales.cuda()).round()
#         #     int_module.weight.copy_(quarot.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())
        
#         #     if module.bias is not None:
#         #         int_module.bias.copy_(module.bias)
        
#         return int_module


class OnlineHadamard(torch.nn.Module):
    def __init__(self, hadamard_dim, force_fp32=False):
        super().__init__()
        self.fp32_had = force_fp32
        had_rem_dim, self.rem_dim = get_hadK(hadamard_dim)
        if had_rem_dim is not None:
            self.register_buffer("had_rem_dim", had_rem_dim)
            if not self.fp32_had:
                self.had_rem_dim = self.had_rem_dim.to(torch.float16)
        else:
            self.had_rem_dim = None       
    
    def forward(self, x):
        x_dtype = x.dtype
        if self.fp32_had:
            x = x.float()
        x = matmul_hadU_cuda(x, self.had_rem_dim, self.rem_dim)
        x = x.to(x_dtype)
        return x
