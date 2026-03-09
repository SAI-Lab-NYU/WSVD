import model_utils
import torch
import torch.nn as nn
import typing
import utils
import transformers
import tqdm, math
import quant_utils
import hadamard_utils
import gptq_utils
from hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2
from fast_hadamard_transform import hadamard_transform
from optimizer_utils import SGDG
import logging


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()

        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)

            
def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)

def bake_mean_into_linearmm(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and adds the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ + W_.mean(dim=-1, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)


def bake_mean_into_linear_return(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    W_mean = W_.mean(dim=-2, keepdim=True)
    linear.weight.data = W_ - W_mean
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        b_mean = b_.mean()
        linear.bias.data = b_ - b_mean
        linear.bias.data = linear.bias.data.to(linear_dtype)
    return (W_mean, b_mean)

def fuse_layer_norms_noeb(model):
    
    model_type = model_utils.get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # # Embedding fusion # why is this?
    # for W in model_utils.get_embeddings(**kwargs):
    #     W_ = W.weight.data.double()
    #     W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    layers = model_utils.get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
            
    
        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
    
    model_utils.replace_modules(
        model,
        transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == model_utils.LLAMA_MODEL else torch.nn.LayerNorm,
        lambda _: model_utils.RMSN(model.config.hidden_size),
        replace_layers=False,
    )
    
            
def fuse_layer_norms(model):
    
    model_type = model_utils.get_model_type(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion # why is this?
    for W in model_utils.get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
        
    layers = model_utils.get_transformer_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])    
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(layer.self_attn_layer_norm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
            
    
        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)
                    
    
    fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])
    
    model_utils.replace_modules(
        model,
        transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == model_utils.LLAMA_MODEL else torch.nn.LayerNorm,
        lambda _: model_utils.RMSN(model.config.hidden_size),
        replace_layers=False,
    )
    
def fuse_layer_normsvit(model):
    
    model_type = model_utils.get_model_typevit(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion # why is this?
    # for W in model_utils.get_embeddings(**kwargs):
    #     W_ = W.weight.data.double()
    #     W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
    ########## VIT no embedding replace this to pre-LN
    # Replace Pre-LN in vit with substract mean
    vit_model = model.model.vision_tower.vision_tower.vision_model
    vit_model.pre_layrnorm = model_utils.LN_(vit_model.pre_layrnorm)
    layers = model_utils.get_vit_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for idx, layer in enumerate(layers):
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAVA_MODEL:
            fuse_ln_linear(layer.layer_norm1, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.layer_norm2, [layer.mlp.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
        if model_type == model_utils.LLAVA_MODEL:
            # here do not include the second last layers
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.
            # if  idx < len(layers) - 2:
            #     bake_mean_into_linear(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.
            # else:
            #     print(f"skip mlp mean in layer{idx}")
            #     pass

    layer = model.model.mm_projector
    bake_mean_into_linearmm(layer[0])
    # fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)]) # 
    # no post ln, just add rotate 
    
    # model_utils.replace_modules(
    #     model.model.vision_tower.vision_tower.vision_model.encoder,
    #     torch.nn.LayerNorm,
    #     lambda _: model_utils.RMSN(model.model.vision_tower.config.hidden_size),
    #     replace_layers=False,
    # )
    model_utils.replace_modules(
        model.model.vision_tower.vision_tower.vision_model.encoder,
        torch.nn.LayerNorm,
        lambda _: model_utils.RMSLN(model.model.vision_tower.config.hidden_size),
        replace_layers=False,
    )


def fuse_layer_normsvit_return(model):
    
    model_type = model_utils.get_model_typevit(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion # why is this?
    # for W in model_utils.get_embeddings(**kwargs):
    #     W_ = W.weight.data.double()
    #     W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
    ########## VIT no embedding replace this to pre-LN
    # Replace Pre-LN in vit with substract mean
    vit_model = model.model.vision_tower.vision_tower.vision_model
    vit_model.pre_layrnorm = model_utils.LN_(vit_model.pre_layrnorm)
    layers = model_utils.get_vit_layers(**kwargs)
    Ws = []
    bs = []
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for idx, layer in enumerate(layers):
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAVA_MODEL:
            fuse_ln_linear(layer.layer_norm1, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.layer_norm2, [layer.mlp.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
        if model_type == model_utils.LLAVA_MODEL:
            # here do not include the second last layers
            w, b = bake_mean_into_linear_return(layer.self_attn.out_proj)
            if idx < len(layers) - 1:
                Ws.append(w+b)
                bs.append(b.unsqueeze(0))
            w, b = bake_mean_into_linear_return(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.
            if idx < len(layers) - 1:
                Ws.append((w+ b).reshape(-1, 1024) )
                bs.append(b.unsqueeze(0))
            # if  idx < len(layers) - 2:
            #     bake_mean_into_linear(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.
            # else:
            #     print(f"skip mlp mean in layer{idx}")
            #     pass
    # fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)]) # 
    # no post ln, just add rotate 
    
    # model_utils.replace_modules(
    #     model.model.vision_tower.vision_tower.vision_model.encoder,
    #     torch.nn.LayerNorm,
    #     lambda _: model_utils.RMSN(model.model.vision_tower.config.hidden_size),
    #     replace_layers=False,
    # )
    model_utils.replace_modules(
        model.model.vision_tower.vision_tower.vision_model.encoder,
        torch.nn.LayerNorm,
        lambda _: model_utils.RMSLN(model.model.vision_tower.config.hidden_size),
        replace_layers=False,
    )
    return torch.cat(Ws, dim=0), torch.cat(bs, dim=0)


def fuse_layer_normsvit_returnskip(model):
    
    model_type = model_utils.get_model_typevit(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion # why is this?
    # for W in model_utils.get_embeddings(**kwargs):
    #     W_ = W.weight.data.double()
    #     W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
    ########## VIT no embedding replace this to pre-LN
    # Replace Pre-LN in vit with substract mean
    vit_model = model.model.vision_tower.vision_tower.vision_model
    vit_model.pre_layrnorm = model_utils.LN_(vit_model.pre_layrnorm)
    layers = model_utils.get_vit_layers(**kwargs)
    Ws = []
    bs = []
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for idx, layer in enumerate(layers):
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAVA_MODEL:
            fuse_ln_linear(layer.layer_norm1, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.layer_norm2, [layer.mlp.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
        if model_type == model_utils.LLAVA_MODEL:
            # here do not include the second last layers
            w, b = bake_mean_into_linear_return(layer.self_attn.out_proj)
            if idx < len(layers) - 1:
                Ws.append(w+b)
                bs.append(b.unsqueeze(0))
            if  idx < len(layers) - 2:
                w, b = bake_mean_into_linear_return(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.
                if idx < len(layers) - 1:
                    Ws.append((w+ b).reshape(-1, 1024) )
                    bs.append(b.unsqueeze(0))
            else:
                logging.info(f"skip mlp mean in layer{idx}")
            # if  idx < len(layers) - 2:
            #     bake_mean_into_linear(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.
            # else:
            #     print(f"skip mlp mean in layer{idx}")
            #     pass
    # fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)]) # 
    # no post ln, just add rotate 
    
    # model_utils.replace_modules(
    #     model.model.vision_tower.vision_tower.vision_model.encoder,
    #     torch.nn.LayerNorm,
    #     lambda _: model_utils.RMSN(model.model.vision_tower.config.hidden_size),
    #     replace_layers=False,
    # )
    model_utils.replace_modules(
        model.model.vision_tower.vision_tower.vision_model.encoder,
        torch.nn.LayerNorm,
        lambda _: model_utils.RMSLN(model.model.vision_tower.config.hidden_size),
        replace_layers=False,
    )
    return torch.cat(Ws, dim=0), torch.cat(bs, dim=0)


def fuse_layer_normsvit_skip(model):
    
    model_type = model_utils.get_model_typevit(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    
    # Embedding fusion # why is this?
    # for W in model_utils.get_embeddings(**kwargs):
    #     W_ = W.weight.data.double()
    #     W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
    ########## VIT no embedding replace this to pre-LN
    # Replace Pre-LN in vit with substract mean
    vit_model = model.model.vision_tower.vision_tower.vision_model
    vit_model.pre_layrnorm = model_utils.LN_(vit_model.pre_layrnorm)
    layers = model_utils.get_vit_layers(**kwargs)
    
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for idx, layer in enumerate(layers):
        
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAVA_MODEL:
            fuse_ln_linear(layer.layer_norm1, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.layer_norm2, [layer.mlp.fc1])
        else:
            raise ValueError(f'Unknown model type {model_type}')
            
        if model_type == model_utils.LLAVA_MODEL:
            # here do not include the second last layers
            bake_mean_into_linear(layer.self_attn.out_proj)
            # bake_mean_into_linear(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.
            if  idx < len(layers) - 2:
                bake_mean_into_linear(layer.mlp.fc2) # here skip the second last one, won't work, with no ln following.
            else:
                logging.info(f"skip mlp mean in layer{idx}")
            #     pass

    
    # fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)]) # 
    # no post ln, just add rotate 
    
    # model_utils.replace_modules(
    #     model.model.vision_tower.vision_tower.vision_model.encoder,
    #     torch.nn.LayerNorm,
    #     lambda _: model_utils.RMSN(model.model.vision_tower.config.hidden_size),
    #     replace_layers=False,
    # )
    model_utils.replace_modules(
        model.model.vision_tower.vision_tower.vision_model.encoder,
        torch.nn.LayerNorm,
        lambda _: model_utils.RMSLN(model.model.vision_tower.config.hidden_size),
        replace_layers=False,
    )


def wrap_layer_normvit(model):
    model_type = model_utils.get_model_typevit(model)
    
    kwargs = {'model': model, 'model_type': model_type}
    layers = model_utils.get_vit_layers(**kwargs)
    for idx, layer in enumerate(layers):
        layer.layer_norm1 = model_utils.LNRotWrapper(layer.layer_norm1)
        layer.layer_norm2 = model_utils.LNRotWrapper(layer.layer_norm2)

    
def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_orthogonal_matrix(size, mode, device=utils.DEV):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')

    

def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_projectors(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for proj in model_utils.get_projector(model, model_type):
        if type(proj) in [nn.Linear]:
            W = proj
        else:
            W = proj[-1]
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
    
def rotate_attention_inputs(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_qk(layer, Q, model_type) -> None:
    # do not need to rotate vit q k matrix, as no kv cache needed.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL:
        W = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_input(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAMA_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    elif model_type == model_utils.LLAVA_MODEL:
        mlp_inputs = [layer.mlp.fc1]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    
def rotate_mlp_output(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    elif model_type == model_utils.LLAVA_MODEL:
        W = layer.mlp.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_outputvit(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    elif model_type == model_utils.LLAVA_MODEL:
        W = layer.mlp.fc2
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def matmul_hadU_cuda_had(X, hadK, transpose=False):
    '''
    Apply hadamard transformation. 
    It reshapes X and applies Walsh-Hadamard transform to the last dimension. 
    Then, it will multiply the retult by another hadamard matrix.
    '''
    from fast_hadamard_transform import hadamard_transform
    from hadamard_utils import get_had172
    n = X.shape[-1]
    K = hadK.shape[-1]

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().cuda().view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), scale=1/math.sqrt(n))
    input = hadK.to(input.device).to(input.dtype) @ input 
    return input.to(X.device).to(X.dtype).reshape(
        X.shape) 

def rotate_faster_down_proj(layer, model_type, hardK):
    from fast_hadamard_transform import hadamard_transform
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Faster MLP is onlu supported for LLaMa models!')
    
    dtype = W.weight.data.dtype
    W.weight.data = matmul_hadU_cuda_had(W.weight.data.float().cuda(), hardK)
    W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)

def rotate_projectorsvit(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for proj in model_utils.get_projector(model, model_type):
        if type(proj) in [nn.Linear]:
            W = proj
        else:
            W = proj[0]
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
  
def rotate_projectorsinput(model) -> None:
    model_type = model_utils.model_type_extractor(model)
    for proj in model_utils.get_projector(model, model_type):
        if type(proj) in [nn.Linear]:
            W = [proj]
        else:
            W = [proj[0], proj[2]]
        for W_ in W:
            apply_exact_had_to_linear(W_, had_dim=-1, output=False)


def rotate_projectorsinputllava_fused(model, Q: torch.Tensor) -> None:
    model_type = model_utils.model_type_extractor(model)
    for proj in model_utils.get_projector(model, model_type):
        W = proj[0]
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        W_ = proj[2]
        apply_exact_had_to_linear(W_, had_dim=-1, output=False)

def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_ov_proj(layer, model_type, head_num, head_dim):
    v_proj = layer.self_attn.v_proj
    if model_type == model_utils.LLAMA_MODEL:
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL:
        o_proj = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)
    # apply_exact_had_to_linear(v_proj, had_dim=-1, output=True)
    # apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

def rotate_ov_projvit(layer, model_type, head_num, head_dim):
    v_proj = layer.self_attn.v_proj
    if model_type == model_utils.LLAMA_MODEL:
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL:
        o_proj = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    # apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)
    # apply_exact_had_to_linear(v_proj, had_dim=-1, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

@torch.inference_mode()
def rotate_model(model, args):
    Q = get_orthogonal_matrix(model.config.hidden_size,
                                                args.rotate_mode)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q) # +rotate_projector
    rotate_head(model, Q) #
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type) # no in vit
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)

@torch.inference_mode()
def rotate_modelllava(model, args):
    Q = get_orthogonal_matrix(model.config.hidden_size,
                                                args.rotate_mode)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q) # +rotate_projector
    rotate_projectors(model, Q) # +rotate_projector
    rotate_head(model, Q) # is lm head
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_output(layers[idx], Q, model_type) # no in vit
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)

@torch.inference_mode()
def rotate_modelllavaWout(model, args, rotations):
    Q = get_orthogonal_matrix(model.config.hidden_size,
                                                args.rotate_mode)
    R1 = rotations['R1'].double()
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, R1) # +rotate_projector
    rotate_projectors(model, R1) # +rotate_projector
    rotate_head(model, R1) # is lm head
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, 
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], R1, model_type)
        rotate_attention_output(layers[idx], R1, model_type)
        rotate_mlp_input(layers[idx], R1, model_type)
        rotate_mlp_output(layers[idx], R1, model_type) # no in vit
        rotate_ov_proj(layers[idx], model_type, num_heads, head_dim)
        


@torch.inference_mode()
def rotate_modelvit(model, args):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    ln = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    ln.online_random_had = True
    ln.had_K = Q
    ln.K = 1
    ln.fp32_had = args.fp32_had #
    model_type = model_utils.get_model_typevit(model)
    # rotate_embeddings(model, Q) # add online rotate❓
    rotate_projectorsvit(model, Q) # rotate projector input 
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating vit layers")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_outputvit(layers[idx], Q, model_type) # no in vit
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)


@torch.inference_mode()
def rotate_modelvitmm(model, args):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    ln = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    ln.online_random_had = True
    ln.had_K = Q
    ln.K = 1
    ln.fp32_had = args.fp32_had #
    model_type = model_utils.get_model_typevit(model)
    # rotate_embeddings(model, Q) # add online rotate❓
    rotate_projectorsvit(model, Q) # +rotate_projector
    rotate_projectorsinput(model) # here have mm proj issue
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating vit mm")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_outputvit(layers[idx], Q, model_type) # no in vit
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
    

@torch.inference_mode()
def rotate_modelvitmmv2(model, args):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode)
    Q1 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                args.rotate_mode)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    ln = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    ln.online_random_had = True
    ln.had_K = Q
    ln.K = 1
    ln.fp32_had = args.fp32_had #
    model_type = model_utils.get_model_typevit(model)
    # rotate_embeddings(model, Q) # add online rotate❓
    rotate_projectorsinputllava_fused(model, Q)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating vit mm")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_outputvit(layers[idx], Q, model_type) # no mlpfc2 online in vit
        W = layer.mlp.fc2
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q1).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
    return Q1



@torch.inference_mode()
def rotate_modelvitmmv2svd(model, args):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode)
    Q1 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                args.rotate_mode)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    ln = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    ln.online_random_had = True
    ln.had_K = Q
    ln.K = 1
    ln.fp32_had = args.fp32_had #
    model_type = model_utils.get_model_typevit(model)
    # rotate_embeddings(model, Q) # add online rotate❓
    rotate_projectorsinputllava_fused(model, Q)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating vit mm")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_outputvit(layers[idx], Q, model_type) # no mlpfc2 online in vit
        W = layer.mlp.fc2.BLinear
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q1).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
    return Q1



@torch.inference_mode()
def rotate_modelvitmmv2learn(model, args, rotations):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode)
    Q1 = rotations['R4'].double()
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    ln = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    ln.online_random_had = True
    ln.had_K = Q
    ln.K = 1
    ln.fp32_had = args.fp32_had #
    model_type = model_utils.get_model_typevit(model)
    # rotate_embeddings(model, Q) # add online rotate❓
    rotate_projectorsinputllava_fused(model, Q)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating vit mm")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_outputvit(layers[idx], Q, model_type) # no mlpfc2 online in vit
        W = layer.mlp.fc2
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q1).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
    return Q1


@torch.inference_mode()
def rotate_modelvitmmv3(model, args, rotations):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode)
    R3, R4 = rotations['R3'].double(), rotations['R4'].double()
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    ln = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    ln.online_random_had = True
    ln.had_K = Q
    ln.K = 1
    ln.fp32_had = args.fp32_had #
    model_type = model_utils.get_model_typevit(model)
    # rotate_embeddings(model, Q) # add online rotate❓
    rotate_projectorsinputllava_fused(model, Q)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating vit mm")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type)
        rotate_mlp_outputvit(layers[idx], Q, model_type) # no mlpfc2 online in vit
        # Later fused to function
        W = layer.mlp.fc1
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(R3.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
            W.bias.data = torch.matmul(R3.T, b).to(device="cpu", dtype=dtype)
        W = layer.mlp.fc2
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R4).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)


@torch.inference_mode()
def rotate_modelvitmmv4(model, args, rotations):
    # learnt wout Q
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode)
    Q1 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                args.rotate_mode)
    R1 = rotations['R1'].double()
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    ln = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    ln.online_random_had = True
    ln.had_K = R1
    ln.K = 1
    ln.fp32_had = args.fp32_had #
    model_type = model_utils.get_model_typevit(model)
    # rotate_embeddings(model, Q) # add online rotate❓
    rotate_projectorsinputllava_fused(model, R1)
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating vit mm")):
        rotate_attention_inputs(layers[idx], R1, model_type)
        # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        rotate_attention_output(layers[idx], R1, model_type)
        rotate_mlp_input(layers[idx], R1, model_type)
        rotate_mlp_outputvit(layers[idx], R1, model_type) # no mlpfc2 online in vit
        W = layer.mlp.fc2
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q1).to(device="cpu", dtype=dtype)
        # Later fused to function
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
    return Q1

@torch.inference_mode()
def rotate_modelvitonline(model, args):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.get_model_typevit(model)
    # rotate_projectorsvit(model, Q) # +rotate_projector do not need for now, or wrap an online mmproj
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        # rotate_attention_inputs(layers[idx], Q, model_type)
        for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        # # rotate_attention_output(layers[idx], Q, model_type) # do not need
        # rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        W = layer.mlp.fc1
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # rotate_mlp_output(layers[idx], Q, model_type) # no in vit do not need 
        W = layer.mlp.fc2
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        had_K, K = hadamard_utils.get_hadK(model_dim)
        layer.layer_norm1.online_full_had = True
        layer.layer_norm1.had_K = had_K
        layer.layer_norm1.K = K
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_full_had = True
        layer.layer_norm2.had_K = had_K
        layer.layer_norm2.K = K
        layer.layer_norm2.fp32_had = args.fp32_had


@torch.inference_mode()
def rotate_modelvitonlineR(model, args):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.get_model_typevit(model)
    # rotate_projectorsvit(model, Q) # +rotate_projector do not need for now, or wrap an online mmproj
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        # for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        #     apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        # # rotate_attention_output(layers[idx], Q, model_type) # do not need
        rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        # W = layer.mlp.fc1
        # apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # rotate_mlp_output(layers[idx], Q, model_type) # no in vit do not need 
        W = layer.mlp.fc2
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        layer.layer_norm1.online_random_had = True
        layer.layer_norm1.had_K = Q
        layer.layer_norm1.K = 1
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_random_had = True
        layer.layer_norm2.had_K = Q
        layer.layer_norm2.K = 1
        layer.layer_norm2.fp32_had = args.fp32_had

@torch.inference_mode()
def rotate_modelvitmmonline(model, args):
    # Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
    #                                             args.rotate_mode)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.get_model_typevit(model)
    # rotate_projectorsvit(model, Q) # +rotate_projector do not need for now, or wrap an online mmproj
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        # rotate_attention_inputs(layers[idx], Q, model_type)
        for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        # # rotate_attention_output(layers[idx], Q, model_type) # do not need
        # rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        W = layer.mlp.fc1
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # rotate_mlp_output(layers[idx], Q, model_type) # no in vit do not need 
        W = layer.mlp.fc2
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        had_K, K = hadamard_utils.get_hadK(model_dim)
        layer.layer_norm1.online_full_had = True
        layer.layer_norm1.had_K = had_K
        layer.layer_norm1.K = K
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_full_had = True
        layer.layer_norm2.had_K = had_K
        layer.layer_norm2.K = K
        layer.layer_norm2.fp32_had = args.fp32_had
    # add rotate projector layers, should be online
    rotate_projectorsinput(model)


@torch.inference_mode()
def rotate_modelvitmmonlineR(model, args):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode)
    Q1 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                args.rotate_mode)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.get_model_typevit(model)
    # rotate_projectorsvit(model, Q) # +rotate_projector do not need for now, or wrap an online mmproj
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        # for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        #     apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        # # rotate_attention_output(layers[idx], Q, model_type) # do not need
        rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        # W = layer.mlp.fc1
        # apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # rotate_mlp_output(layers[idx], Q, model_type) # no in vit do not need 
        # W = layer.mlp.fc2
        # apply_exact_had_to_linear(W, had_dim=-1, output=False)
        W = layer.mlp.fc2
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q1).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        layer.layer_norm1.online_random_had = True
        layer.layer_norm1.had_K = Q
        layer.layer_norm1.K = 1
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_random_had = True
        layer.layer_norm2.had_K = Q
        layer.layer_norm2.K = 1
        layer.layer_norm2.fp32_had = args.fp32_had
    # add rotate projector layers, should be online
    rotate_projectorsinput(model)
    return Q1


@torch.inference_mode()
def rotate_modelvitmmonlinev3(model, args, rotations):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode)
    R3, R4 = rotations['R3'].double(), rotations['R4'].double()
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    had_dim = R3.shape[0]


    model_type = model_utils.get_model_typevit(model)
    # rotate_projectorsvit(model, Q) # +rotate_projector do not need for now, or wrap an online mmproj
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        # for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        #     apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        # # rotate_attention_output(layers[idx], Q, model_type) # do not need
        rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        # W = layer.mlp.fc1
        # apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # rotate_mlp_output(layers[idx], Q, model_type) # no in vit do not need 
        # W = layer.mlp.fc2
        # apply_exact_had_to_linear(W, had_dim=-1, output=False)
        if had_dim == model_dim * 4:
            W = layer.mlp.fc1
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(R3.T, W_).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
                W.bias.data = torch.matmul(R3.T, b).to(device="cpu", dtype=dtype)
            W = layer.mlp.fc2
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, R4).to(device="cpu", dtype=dtype)
        else:
            # W = layer.mlp.fc1
            # dtype = W.weight.data.dtype
            # init_shape = W.weight.shape
            # temp = W.weight.data.reshape(init_shape[0]//had_dim, had_dim, -1).to(device=utils.DEV, dtype=torch.float64)
            # W.weight.data = torch.matmul(R3.T, temp).reshape(init_shape).to(device="cpu", dtype=dtype)
            W = layer.mlp.fc1
            dtype = W.weight.data.dtype
            temp = W.weight.t()
            init_shape = temp.shape
            temp = temp.reshape(-1, init_shape[0]//had_dim, had_dim).to(device=utils.DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(temp, R3).reshape(init_shape).to(device="cpu", dtype=dtype).t()
            if W.bias is not None:
                init_shape = W.bias.shape
                temp = W.bias.data.reshape(had_dim, -1).to(device=utils.DEV, dtype=torch.float64)
                W.bias.data = torch.matmul(R3.T, temp).reshape(init_shape).to(device="cpu", dtype=dtype)
            W = layer.mlp.fc2
            dtype = W.weight.data.dtype
            init_shape = W.weight.shape
            temp = W.weight.data.reshape(-1, init_shape[0]//had_dim, had_dim).to(device=utils.DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(temp, R4).reshape(init_shape).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        layer.layer_norm1.online_random_had = True
        layer.layer_norm1.had_K = Q
        layer.layer_norm1.K = 1
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_random_had = True
        layer.layer_norm2.had_K = Q
        layer.layer_norm2.K = 1
        layer.layer_norm2.fp32_had = args.fp32_had
    # add rotate projector layers, should be online
    rotate_projectorsinput(model)


@torch.inference_mode()
def rotate_modelvitmmonlinev4(model, args, rotations):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode)
    R3s, R4s, SQs= rotations['R3'], rotations['R4'], rotations['sq']
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    had_dim = R3s[0].shape[0]


    model_type = model_utils.get_model_typevit(model)
    # rotate_projectorsvit(model, Q) # +rotate_projector do not need for now, or wrap an online mmproj
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        R3, R4 = R3s[idx].double(), R4s[idx].double()
        if len(SQs) > 0:
            sq = SQs[idx].double()
        else:
            sq = None
        rotate_attention_inputs(layers[idx], Q, model_type)
        # for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        #     apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        # # rotate_attention_output(layers[idx], Q, model_type) # do not need
        rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        # W = layer.mlp.fc1
        # apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # rotate_mlp_output(layers[idx], Q, model_type) # no in vit do not need 
        # W = layer.mlp.fc2
        # apply_exact_had_to_linear(W, had_dim=-1, output=False)
        if had_dim == model_dim * 4:
            W = layer.mlp.fc1
            dtype = W.weight.data.dtype
            
            W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
            if sq is not None:
                W_ *= sq[:, None].to(device=utils.DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(R3.T, W_).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
                if sq is not None:
                    b *= sq.to(device=utils.DEV, dtype=torch.float64)
                W.bias.data = torch.matmul(R3.T, b).to(device="cpu", dtype=dtype)
            W = layer.mlp.fc2
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
            if sq is not None:
                W_ /= sq.to(device=utils.DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, R4).to(device="cpu", dtype=dtype)
        else:
            W = layer.mlp.fc1
            dtype = W.weight.data.dtype
            init_shape = W.weight.shape
            if sq is not None:
                W.weight.data = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
                W.weight.data *= sq[:, None].to(device=utils.DEV, dtype=torch.float64)
            temp = W.weight.data.reshape(init_shape[0]//had_dim, had_dim, -1).to(device=utils.DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(R3.T, temp).reshape(init_shape).to(device="cpu", dtype=dtype)
            # if sq is not None:
            #     W.weight.data = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
            #     W.weight.data *= sq[:, None].to(device=utils.DEV, dtype=torch.float64)
            # init_shape = W.weight.t().shape
            # temp = W.weight.data.t().reshape(-1, init_shape[0]//had_dim, had_dim).to(device=utils.DEV, dtype=torch.float64)
            # W.weight.data = torch.matmul(temp, R3).reshape(init_shape).to(device="cpu", dtype=dtype).t()
            if W.bias is not None:
                init_shape = W.bias.shape
                if sq is not None:
                    W.bias.data=W.bias.data.to(device=utils.DEV, dtype=torch.float64)
                    W.bias.data *= sq.to(device=utils.DEV, dtype=torch.float64)
                temp = W.bias.data.reshape(had_dim, -1).to(device=utils.DEV, dtype=torch.float64)
                W.bias.data = torch.matmul(R3.T, temp).reshape(init_shape).to(device="cpu", dtype=dtype)
            W = layer.mlp.fc2
            dtype = W.weight.data.dtype
            init_shape = W.weight.shape
            if sq is not None:
                W.weight.data= W.weight.data.to(device=utils.DEV, dtype=torch.float64)
                W.weight.data /= sq.to(device=utils.DEV, dtype=torch.float64)
            temp = W.weight.data.reshape(-1, init_shape[0]//had_dim, had_dim).to(device=utils.DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(temp, R4).reshape(init_shape).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        layer.layer_norm1.online_random_had = True
        layer.layer_norm1.had_K = Q
        layer.layer_norm1.K = 1
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_random_had = True
        layer.layer_norm2.had_K = Q
        layer.layer_norm2.K = 1
        layer.layer_norm2.fp32_had = args.fp32_had
    # add rotate projector layers, should be online
    rotate_projectorsinput(model)



@torch.inference_mode()
def rotate_modelvitmmonlinev5(model, args, rotations):
    # v5, sq learn sperately for Win and Wout
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode)
    R3s, R4s, SQs= rotations['R3'], rotations['R4'], rotations['sq']
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    had_dim = R3s[0].shape[0]


    model_type = model_utils.get_model_typevit(model)
    # rotate_projectorsvit(model, Q) # +rotate_projector do not need for now, or wrap an online mmproj
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        R3, R4 = R3s[idx].double(), R4s[idx].double()
        if len(SQs) > 0:
            sq, sq1 = SQs[idx][0].double(), SQs[idx][1].double()
        else:
            sq = None
        rotate_attention_inputs(layers[idx], Q, model_type)
        # for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        #     apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        # # rotate_attention_output(layers[idx], Q, model_type) # do not need
        rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        # W = layer.mlp.fc1
        # apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # rotate_mlp_output(layers[idx], Q, model_type) # no in vit do not need 
        # W = layer.mlp.fc2
        # apply_exact_had_to_linear(W, had_dim=-1, output=False)
        if had_dim == model_dim * 4:
            W = layer.mlp.fc1
            dtype = W.weight.data.dtype
            
            W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
            if sq is not None:
                W_ /= sq1.to(device=utils.DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(R3.T, W_).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
                if sq is not None:
                    b /= sq1.to(device=utils.DEV, dtype=torch.float64)
                W.bias.data = torch.matmul(R3.T, b).to(device="cpu", dtype=dtype)
            W = layer.mlp.fc2
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
            if sq is not None:
                W_ *= sq.to(device=utils.DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, R4).to(device="cpu", dtype=dtype)
        else:
            W = layer.mlp.fc1
            dtype = W.weight.data.dtype
            init_shape = W.weight.shape
            if sq is not None:
                W.weight.data = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
                W.weight.data /= sq1.to(device=utils.DEV, dtype=torch.float64)
            temp = W.weight.data.reshape(init_shape[0]//had_dim, had_dim, -1).to(device=utils.DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(R3.T, temp).reshape(init_shape).to(device="cpu", dtype=dtype)
            # if sq is not None:
            #     W.weight.data = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
            #     W.weight.data *= sq[:, None].to(device=utils.DEV, dtype=torch.float64)
            # init_shape = W.weight.t().shape
            # temp = W.weight.data.t().reshape(-1, init_shape[0]//had_dim, had_dim).to(device=utils.DEV, dtype=torch.float64)
            # W.weight.data = torch.matmul(temp, R3).reshape(init_shape).to(device="cpu", dtype=dtype).t()
            if W.bias is not None:
                init_shape = W.bias.shape
                if sq is not None:
                    W.bias.data=W.bias.data.to(device=utils.DEV, dtype=torch.float64)
                    W.bias.data /= sq1.to(device=utils.DEV, dtype=torch.float64)
                temp = W.bias.data.reshape(had_dim, -1).to(device=utils.DEV, dtype=torch.float64)
                W.bias.data = torch.matmul(R3.T, temp).reshape(init_shape).to(device="cpu", dtype=dtype)
            W = layer.mlp.fc2
            dtype = W.weight.data.dtype
            init_shape = W.weight.shape
            if sq is not None:
                W.weight.data= W.weight.data.to(device=utils.DEV, dtype=torch.float64)
                W.weight.data *= sq.to(device=utils.DEV, dtype=torch.float64)
            temp = W.weight.data.reshape(-1, init_shape[0]//had_dim, had_dim).to(device=utils.DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(temp, R4).reshape(init_shape).to(device="cpu", dtype=dtype)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)
        layer.layer_norm1.online_random_had = True
        layer.layer_norm1.had_K = Q
        layer.layer_norm1.K = 1
        layer.layer_norm1.fp32_had = args.fp32_had
        layer.layer_norm2.online_random_had = True
        layer.layer_norm2.had_K = Q
        layer.layer_norm2.K = 1
        layer.layer_norm2.fp32_had = args.fp32_had
    # add rotate projector layers, should be online
    rotate_projectorsinput(model)



@torch.inference_mode()
def rotate_modelvitonlinev1(model, args):
    Q = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
                                                args.rotate_mode)
    config = model.model.vision_tower.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads


    model_type = model_utils.get_model_typevit(model)
    # rotate_projectorsvit(model, Q) # +rotate_projector do not need for now, or wrap an online mmproj
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        # rotate_attention_inputs(layers[idx], Q, model_type)
        for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # # rotate_attention_qk(layers[idx], Q, model_type) # no need? we do not need kv cache for VIT?
        # # rotate_attention_output(layers[idx], Q, model_type) # do not need
        # rotate_mlp_input(layers[idx], Q, model_type) # write new mlp_input for online rotation here
        W = layer.mlp.fc1
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
        # # rotate_mlp_output(layers[idx], Q, model_type) # no in vit do not need 
        W = layer.mlp.fc2
        apply_exact_had_to_linear(W, had_dim=-1, output=False)
        rotate_ov_projvit(layers[idx], model_type, num_heads, head_dim)

@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]

def register_online_rotation(module, Q:torch.Tensor):
    assert not hasattr(module, 'Q')
    module.register_buffer('Q', Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)


class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs['k_groupsize'] in [-1, head_dim], f'Only token-wise/{head_dim}g quantization is supported for K-cache'
            self.k_bits = kwargs['k_bits']
            self.k_groupsize = kwargs['k_groupsize']
            self.k_sym = kwargs['k_sym']
            self.k_clip_ratio = kwargs['k_clip_ratio']
            self.k_quantizer.configure(bits=self.k_bits, groupsize=-1, #we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                                   sym=self.k_sym, clip_ratio=self.k_clip_ratio)

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = hadamard_transform(q.float(), scale=1/math.sqrt(q.shape[-1])).to(dtype)
        k = hadamard_transform(k.float(), scale=1/math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape
        

        if self.k_groupsize == -1: #token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, self.config.hidden_size)
            self.k_quantizer.find_params(token_wise_k)
            k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        else: #head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)
        
        self.k_quantizer.free()
            
        return q, k



def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward. 
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    import monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name, functools.partial(QKRotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)





class QKfp8RotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.k_quantizer = quant_utils.Actfp8Quantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs['k_groupsize'] in [-1, head_dim], f'Only token-wise/{head_dim}g quantization is supported for K-cache'
            self.k_bits = kwargs['k_bits']
            self.k_groupsize = kwargs['k_groupsize']
            self.k_sym = kwargs['k_sym']
            self.k_clip_ratio = kwargs['k_clip_ratio']
            self.k_quantizer.configure(bits=self.k_bits, groupsize=-1, #we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                                   sym=self.k_sym, clip_ratio=self.k_clip_ratio)

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = hadamard_transform(q.float(), scale=1/math.sqrt(q.shape[-1])).to(dtype)
        k = hadamard_transform(k.float(), scale=1/math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape
        

        if self.k_groupsize == -1: #token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, self.config.hidden_size)
            self.k_quantizer.find_params(token_wise_k)
            k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        else: #head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)
        
        self.k_quantizer.free()
            
        return q, k

def addfp8_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward. 
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    import monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name, functools.partial(QKfp8RotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)


def rotate_per_layer(model, dataloader, dev, args, tokenizer=None, image_processor=None):
    logging.info('-----Learn Q vit -----')
    dataloader, _ = dataloader
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.model.vision_tower.vision_tower.vision_model.parameters())).dtype
    # R1 = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
    #                                             args.rotate_mode)
    # R2 = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
    #                                             args.rotate_mode)
    R1 = None
    R2 = None
    if args.rot_dim:
        R3 = get_orthogonal_matrix(args.rot_dim,
                                    args.rotate_mode) # applied after fc1
        R4 = get_orthogonal_matrix(args.rot_dim,
                                    args.rotate_mode) # applied after fc1
    else:
        R3 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                    args.rotate_mode) # applied after fc1
        R4 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                args.rotate_mode) # applied before fc2
    ## have
    R1 = nn.Parameter(R1.to(torch.float32).to(utils.DEV)) if R1 is not None else None
    R2 = nn.Parameter(R2.to(torch.float32).to(utils.DEV)) if R2 is not None else None
    R3 = nn.Parameter(R3.to(torch.float32).to(utils.DEV)) if R3 is not None else None
    R4 = nn.Parameter(R4.to(torch.float32).to(utils.DEV)) if R4 is not None else None
    inps = torch.zeros(
        (args.nsamples, model.model.vision_tower.num_patches+1, model.model.vision_tower.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps[cache['i']] = inp
            # inps.append(inp[0])
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # model(batch[0].to(dev))
            input_ids, images = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
            from llava.constants import IMAGE_TOKEN_INDEX
            # num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
            # print(num_images)
            model.generate(input_ids, images=images,
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=512,
                            use_cache=True,)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    outsfp16 = torch.zeros_like(inps)
    

    rots = {}
    for i in range(len(layers)):
        torch.set_grad_enabled(True)
        logging.info(f'\nLayer {i}:\n', flush=True, end=' ')
        layer = layers[i].to(dev)
        for param in layer.parameters():
            param.requires_grad = False
        # for j in range(args.nsamples):
        #     outsfp16[j] = layer(inps[j].unsqueeze(0), None, None)[0]
        outsfp16 = layer(inps, None, None)
        layer = model_utils.Layerwrapper(layer, R1=R1, R2=R2, R3=R3, R4=R4) # here add forward to fuse weight etc
        layer.init_rot()
        print("using rot")
        layer.init_quant(args)
        print("using quant")
        loss_fn = nn.MSELoss()
        trainable_parameters = [layer.R3, layer.R4]
        optimizer = SGDG(trainable_parameters, lr=args.rot_lr, stiefel=True)
        for epoch in range(args.rot_epochs):
            for j in range(args.nsamples):
                optimizer.zero_grad()
                out = layer(inps[j].unsqueeze(0), None, None)[0]
                loss = ((outsfp16[j]-out)**2).sum(-1).mean()
                # loss = loss_fn(outsfp16[j], out)
                loss.backward()
                optimizer.step()
                if j % 50 == 0:
                    logging.info(f"Epoch {epoch}, Iter {j}, Loss: {loss.item()}")
        layer.reset_rot()
        layers[i] = layer.module.cpu() # here delete the wrapper
        del layer
        torch.cuda.empty_cache()
        logging.info('now testing only 1 layer results')
        break
        inps, outsfp16 = outsfp16, inps
    rots = {'R1':R1, 'R2':R2, 'R3':R3, 'R4': R4}
    return rots



def rotate_per_layermlp(model, dataloader, dev, args, tokenizer=None, image_processor=None):
    logging.info('-----Learn Q vit mlp-----')
    dataloader, _ = dataloader
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.model.vision_tower.vision_tower.vision_model.parameters())).dtype
    # R1 = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
    #                                             args.rotate_mode)
    # R2 = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
    #                                             args.rotate_mode)
    def init_R():
        R1 = None
        R2 = None
        if args.rot_dim:
            R3 = get_orthogonal_matrix(args.rot_dim,
                                        args.rotate_mode) # applied after fc1
            R4 = get_orthogonal_matrix(args.rot_dim,
                                        args.rotate_mode) # applied after fc1
        else:
            R3 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                        args.rotate_mode) # applied after fc1
            R4 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                    args.rotate_mode) # applied before fc2
        R1 = nn.Parameter(R1.to(torch.float32).to(utils.DEV)) if R1 is not None else None
        R2 = nn.Parameter(R2.to(torch.float32).to(utils.DEV)) if R2 is not None else None
        R3 = nn.Parameter(R3.to(torch.float32).to(utils.DEV)) if R3 is not None else None
        R4 = nn.Parameter(R4.to(torch.float32).to(utils.DEV)) if R4 is not None else None
        return R1, R2, R3, R4
    def init_R_I():
        R1 = None
        R2 = None
        if args.rot_dim:
            R3 = torch.eye(args.rot_dim)
            R4 = torch.eye(args.rot_dim)
        else:
            R3 = torch.eye(model.model.vision_tower.config.intermediate_size)
            R4 = torch.eye(model.model.vision_tower.config.intermediate_size)
        R1 = nn.Parameter(R1.to(torch.float32).to(utils.DEV)) if R1 is not None else None
        R2 = nn.Parameter(R2.to(torch.float32).to(utils.DEV)) if R2 is not None else None
        R3 = nn.Parameter(R3.to(torch.float32).to(utils.DEV)) if R3 is not None else None
        R4 = nn.Parameter(R4.to(torch.float32).to(utils.DEV)) if R4 is not None else None
        return R1, R2, R3, R4
    def init_sq(sq = False, weight=None):
        if sq:
            # return nn.Parameter(torch.ones(model.model.vision_tower.config.intermediate_size).to(torch.float32).to(utils.DEV))
            # fc1_max = weight.fc1.weight.data.max(-1)[0]
            # # fc2_max = weight.fc2.weight.data.max(0)[0]
            # # ratio_max = fc2_max / fc1_max
            # ratio = torch.ones_like(fc1_max)
            # ratio1 = torch.ones_like(fc1_max) 
            # return (nn.Parameter(ratio.to(torch.float32).to(utils.DEV)* 10.), nn.Parameter(ratio1.to(torch.float32).to(utils.DEV)* 64.))
            # return (nn.Parameter(torch.tensor(10).to(torch.float32).to(utils.DEV)), nn.Parameter(torch.tensor(64).to(torch.float32).to(utils.DEV)))
            return (nn.Parameter(torch.tensor(1).to(torch.float32).to(utils.DEV)), nn.Parameter(torch.tensor(1).to(torch.float32).to(utils.DEV)))
            # fc1_mean = weight.fc1.weight.data.mean(-1)
            # fc2_mean = weight.fc2.weight.data.mean(0)
            # ratio_mean = fc2_mean / fc1_mean
            # return nn.Parameter(ratio_mean.to(torch.float32).to(utils.DEV))
            # fc1_amean = weight.fc1.weight.data.abs().mean(-1)
            # fc2_amean = weight.fc2.weight.data.abs().mean(0)
            # ratio_absmean = fc2_amean / fc1_amean
            # return nn.Parameter(ratio_absmean.to(torch.float32).to(utils.DEV))
        else:
            return None
    inps = torch.zeros(
        (args.nsamples, model.model.vision_tower.num_patches+1, model.model.vision_tower.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0}
    cacheresidual = {'i': 0}
    residual = torch.zeros_like(inps)
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps[cache['i']] = inp
            # inps.append(inp[0])
            cache['i'] += 1
            raise ValueError
    class residualCatcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            residual[[cacheresidual['i']]] = inp
            # inps.append(inp[0])
            cacheresidual['i'] += 1
            return self.module(inp) # pass through layernorm
    layers[0].mlp = Catcher(layers[0].mlp)
    layers[0].layer_norm2 = residualCatcher(layers[0].layer_norm2)
    for batch in dataloader:
        try:
            # model(batch[0].to(dev))
            input_ids, images = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
            from llava.constants import IMAGE_TOKEN_INDEX
            # num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
            # print(num_images)
            model.generate(input_ids, images=images,
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=512,
                            use_cache=True,)
        except ValueError:
            if cache['i'] == 1:
                print('valueerror')
            pass
    layers[0].mlp = layers[0].mlp.module
    layers[0].layer_norm2 = layers[0].layer_norm2.module
    layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()
    outsfp16 = torch.zeros_like(inps)
    

    rots = {}
    rots = {'R1':[], 'R2':[], 'R3':[], 'R4': [], 'sq': []}
    # return rots
    for i in range(len(layers)):
        torch.set_grad_enabled(True)
        print(f'\nLayer {i}:', flush=True, end=' ')

        layer = layers[i].mlp.to(dev)
        for param in layer.parameters():
            param.requires_grad = False
        # for j in range(args.nsamples):
        #     outsfp16[j] = layer(inps[j]) # need to always use catcher for next mlp 
        R1, R2, R3, R4 = init_R()
        # R1, R2, R3, R4 = init_R_I()
        sq = init_sq(args.sq, weight = layer)
        outsfp16 = layer(inps)
        layer = model_utils.mlpLayerwrapper(layer, R1=R1, R2=R2, R3=R3, R4=R4, sq=sq) # here add forward to fuse weight etc
        layer.init_rot()
        # layer.init_quant(args)
        loss_fn = nn.MSELoss()
        # loss_fn = nn.L1Loss()
        trainable_parameters = []
        trainable_parameters.append({"params": layer.R3, "lr": args.rot_lr})
        trainable_parameters.append({"params": layer.R4, "lr": args.rot_lr})
        # trainable_parameters = [layer.R4]
        # optimizer = torch.optim.Adam(trainable_parameters, lr=args.rot_lr)
        if sq is not None:
            trainable_parameters.append({"params": layer.sq, "lr": args.sq_lr})
            trainable_parameters.append({"params": layer.sq1, "lr": args.sq_lr})
        optimizer = torch.optim.SGD(trainable_parameters)
        loss_curve = []
        loss_curve_normabs = []
        # optimizer = SGDG(trainable_parameters, lr=args.rot_lr)
        for epoch in range(args.rot_epochs):
            for j in range(args.nsamples):
                optimizer.zero_grad()
                out = layer(inps[j])
                # loss = ((outsfp16[j]-out)**2).sum(-1).mean()
                loss = loss_fn(outsfp16[j], out) 
                if torch.isnan(loss):
                    # print("nan loss skip")
                    continue
                loss_curve.append(loss.item())
                # loss /= outsfp16[j].square().mean()
                loss /= outsfp16[j].abs().mean()
                loss.backward()
                optimizer.step()
                # y5_norm = outsfp16[j].mean()
                # y5_abs_norm = outsfp16[j].abs().mean()
                # loss_curve_norm.append(loss.item()/outsfp16[j].mean().cpu().abs())
                # loss_curve_normabs.append(loss.item()/outsfp16[j].abs().mean().cpu())
                loss_curve_normabs.append(loss.item())
                if j % 50 == 0 and epoch % 5 ==0:
                    logging.info(f"Epoch {epoch}, Iter {j}, Loss: {loss.item()}")
        logging.info(f"Epoch {epoch}, Iter {j}, Loss: {loss.item()}")
        def save_plot(loss_curve, name=None):
            import matplotlib.pyplot as plt
            import os
            save_path = args.save_path + '/save_loss/'
            os.makedirs(save_path, exist_ok=True)
            plt.figure(figsize=(10, 5))
            iters = [_ for _ in range(args.nsamples * args.rot_epochs)]
            plt.plot(iters, loss_curve, label = 'loss', linewidth=2)
            epochs = [_ for _ in range(args.rot_epochs)]
            epochs_iter = [_* args.nsamples for _ in range(args.rot_epochs)]
            plt.xlabel("Epochs")
            plt.xticks(epochs_iter, epochs)
            plt.ylabel("loss")
            # if i == 0:
            #     plt.ylim(0, 3) 
            plt.title(f"Loss curve-layer{i}")
            plt.savefig(f'{save_path}lr_{args.rot_lr}_sgd_r3_layer_{i}_{name}.png')
            plt.close()
        # save_plot(loss_curve, 'ori')
        save_plot(loss_curve_normabs,'normabs')
        layer.reset_rot()
        layers[i].mlp = layer.module.cpu() # here delete the wrapper
        del layer
        torch.cuda.empty_cache()
        # logging.info('now testing only 1 layer results')
        # break
        # inps, outsfp16 = outsfp16, inps
        if i < 23:
            class Catchers(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                def forward(self, inp, *args, **kwargs):
                    inps[:] = inp
                    # inps.append(inp[0])
                    cache['i'] += 1
                    raise ValueError
            class residualCatchers(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                def forward(self, inp, *args, **kwargs):
                    residual[:] = inp
                    # inps.append(inp[0])
                    cacheresidual['i'] += 1
                    return self.module(inp) # pass through layernorm
            torch.set_grad_enabled(False)
            # print()
            layer = layers[i+1].to(dev)
            cache = {'i': 0}
            cacheresidual = {'i': 0}
            layer.mlp = Catchers(layer.mlp)
            layer.layer_norm2 = residualCatchers(layer.layer_norm2)
            # for j in range(args.nsamples):
            #     try:
            #         layer(outsfp16[j].unsqueeze(0), None, None)
            #     except ValueError:
            #         pass
            try:
                layer(outsfp16 + residual, None, None)
            except ValueError:
                print('valueerror')
                pass
            layer.mlp = layer.mlp.module
            layer.layer_norm2 = layer.layer_norm2.module
            layers[i+1] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
        # rots['R1'].append(R1.clone().detach())
        # rots['R2'].append(R2.clone().detach())
        rots['R3'].append(R3.clone().detach())
        rots['R4'].append(R4.clone().detach())
        if sq is not None:
            rots['sq'].append(sq)

    # rots = {'R1':R1, 'R2':R2, 'R3':R3, 'R4': R4}
    return rots



def rotate_per_layermlponline(model, dataloader, dev, args, tokenizer=None, image_processor=None):
    logging.info('-----Learn Q vit mlp-----')
    dataloader, _ = dataloader
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.model.vision_tower.vision_tower.vision_model.parameters())).dtype
    # R1 = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
    #                                             args.rotate_mode)
    # R2 = get_orthogonal_matrix(model.model.vision_tower.config.hidden_size,
    #                                             args.rotate_mode)
    def init_R():
        R1 = None
        R2 = None
        if args.rot_dim:
            R3 = get_orthogonal_matrix(args.rot_dim,
                                        args.rotate_mode) # applied after fc1
            R4 = get_orthogonal_matrix(args.rot_dim,
                                        args.rotate_mode) # applied after fc1
        else:
            R3 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                        args.rotate_mode) # applied after fc1
            R4 = get_orthogonal_matrix(model.model.vision_tower.config.intermediate_size,
                                                    args.rotate_mode) # applied before fc2
        R1 = nn.Parameter(R1.to(torch.float32).to(utils.DEV)) if R1 is not None else None
        R2 = nn.Parameter(R2.to(torch.float32).to(utils.DEV)) if R2 is not None else None
        R3 = nn.Parameter(R3.to(torch.float32).to(utils.DEV)) if R3 is not None else None
        R4 = nn.Parameter(R4.to(torch.float32).to(utils.DEV)) if R4 is not None else None
        return R1, R2, R3, R4
    def init_sq(sq = False, weight=None):
        if sq:
            # return nn.Parameter(torch.ones(model.model.vision_tower.config.intermediate_size).to(torch.float32).to(utils.DEV))
            fc1_max = weight.fc1.weight.data.max(-1)[0]
            fc2_max = weight.fc2.weight.data.max(0)[0]
            ratio_max = fc2_max / fc1_max
            return nn.Parameter(ratio_max.to(torch.float32).to(utils.DEV))
            # fc1_mean = weight.fc1.weight.data.mean(-1)
            # fc2_mean = weight.fc2.weight.data.mean(0)
            # ratio_mean = fc2_mean / fc1_mean
            # return nn.Parameter(ratio_mean.to(torch.float32).to(utils.DEV))
            # fc1_amean = weight.fc1.weight.data.abs().mean(-1)
            # fc2_amean = weight.fc2.weight.data.abs().mean(0)
            # ratio_absmean = fc2_amean / fc1_amean
            # return nn.Parameter(ratio_absmean.to(torch.float32).to(utils.DEV))
        else:
            return None
    inps = torch.zeros(
        (args.nsamples, model.model.vision_tower.num_patches+1, model.model.vision_tower.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0}
    cacheresidual = {'i': 0}
    residual = torch.zeros_like(inps)
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps[cache['i']] = inp
            # inps.append(inp[0])
            cache['i'] += 1
            raise ValueError
    class residualCatcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            residual[[cacheresidual['i']]] = inp
            # inps.append(inp[0])
            cacheresidual['i'] += 1
            return self.module(inp) # pass through layernorm
    class Catchers(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps = inp
            # inps.append(inp[0])
            cache['i'] += 1
            raise ValueError
    class residualCatchers(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            residual = inp
            # inps.append(inp[0])
            cacheresidual['i'] += 1
            return self.module(inp) # pass through layernorm
    layers[0].mlp = Catcher(layers[0].mlp)
    layers[0] = residualCatchers(layers[0].layer_norm2)
    for batch in dataloader:
        try:
            # model(batch[0].to(dev))
            input_ids, images = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
            from llava.constants import IMAGE_TOKEN_INDEX
            # num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
            # print(num_images)
            model.generate(input_ids, images=images,
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=512,
                            use_cache=True,)
        except ValueError:
            pass
    layers[0].mlp = layers[0].mlp.module
    layers[0].layer_norm2 = layers[0].layer_norm2.module
    layers[0] = layers[0].cpu()
    torch.cuda.empty_cache()
    # outs = torch.zeros_like(inps)
    outsfp16 = torch.zeros_like(inps)
    
    

    rots = {}
    rots = {'R1':[], 'R2':[], 'R3':[], 'R4': [], 'sq': []}
    # return rots
    for i in range(len(layers)):
        torch.set_grad_enabled(True)
        print(f'\nLayer {i}:', flush=True, end=' ')

        layer = layers[i].mlp.to(dev)
        for param in layer.parameters():
            param.requires_grad = False
        # for j in range(args.nsamples):
        #     outsfp16[j] = layer(inps[j]) # need to always use catcher for next mlp 
        R1, R2, R3, R4 = init_R()
        sq = init_sq(args.sq, weight = layer)
        outsfp16 = layer(inps)
        layer = model_utils.mlpLayerwrapper(layer, R1=R1, R2=R2, R3=R3, R4=R4, sq=sq) # here add forward to fuse weight etc
        layer.init_onlinerot(args)
        layer.init_quant(args)
        loss_fn = nn.MSELoss()
        # loss_fn = nn.L1Loss()
        
        # trainable_parameters = [layer.R3, layer.R4] 
        trainable_parameters = [layer.R4]
        # optimizer = torch.optim.Adam(trainable_parameters, lr=args.rot_lr)
        if sq is not None:
            trainable_parameters += [layer.sq]
        optimizer = torch.optim.SGD(trainable_parameters, lr=args.rot_lr)
        loss_curve = []
        loss_curve_normabs = []
        # optimizer = SGDG(trainable_parameters, lr=args.rot_lr)
        for epoch in range(args.rot_epochs):
            for j in range(args.nsamples):
                optimizer.zero_grad()
                out = layer(inps[j])
                # loss = ((outsfp16[j]-out)**2).sum(-1).mean()
                loss = loss_fn(outsfp16[j], out) 
                if torch.isnan(loss):
                    # print("nan loss skip")
                    continue
                loss_curve.append(loss.item())
                # loss /= outsfp16[j].square().mean()
                loss /= outsfp16[j].abs().mean()
                loss.backward()
                optimizer.step()
                # y5_norm = outsfp16[j].mean()
                # y5_abs_norm = outsfp16[j].abs().mean()
                # loss_curve_norm.append(loss.item()/outsfp16[j].mean().cpu().abs())
                # loss_curve_normabs.append(loss.item()/outsfp16[j].abs().mean().cpu())
                loss_curve_normabs.append(loss.item())
                if j % 50 == 0 and epoch % 5 ==0:
                    logging.info(f"Epoch {epoch}, Iter {j}, Loss: {loss.item()}")
        logging.info(f"Epoch {epoch}, Iter {j}, Loss: {loss.item()}")
        def save_plot(loss_curve, name=None):
            import matplotlib.pyplot as plt
            import os
            save_path = args.save_path + '/save_loss/'
            os.makedirs(save_path, exist_ok=True)
            plt.figure(figsize=(10, 5))
            iters = [_ for _ in range(args.nsamples * args.rot_epochs)]
            plt.plot(iters, loss_curve, label = 'loss', linewidth=2)
            epochs = [_ for _ in range(args.rot_epochs)]
            epochs_iter = [_* args.nsamples for _ in range(args.rot_epochs)]
            plt.xlabel("Epochs")
            plt.xticks(epochs_iter, epochs)
            plt.ylabel("loss")
            # if i == 0:
            #     plt.ylim(0, 3) 
            plt.title(f"Loss curve-layer{i}")
            plt.savefig(f'{save_path}lr_{args.rot_lr}_sgd_r3_layer_{i}_{name}.png')
            plt.close()
        # save_plot(loss_curve, 'ori')
        save_plot(loss_curve_normabs,'normabs')
        layer.reset_onlinerot()
        layers[i].mlp = layer.module.cpu() # here delete the wrapper
        del layer
        torch.cuda.empty_cache()
        # logging.info('now testing only 1 layer results')
        # break
        # inps, outsfp16 = outsfp16, inps
        if i < 23:
            torch.set_grad_enabled(False)
            layer = layers[i+1].to(dev)
            cache = {'i': 0}
            cacheresidual = {'i': 0}
            layer.mlp = Catchers(layer.mlp)
            layer.layer_norm2 = residualCatcher(layer.layer_norm2)
            # for j in range(args.nsamples):
            #     try:
            #         layer(outsfp16[j].unsqueeze(0), None, None)
            #     except ValueError:
            #         pass
            try:
                layer(outsfp16 + residual, None, None)
            except ValueError:
                pass
            layer.mlp = layer.mlp.module
            layer.layer_norm2 = layer.layer_norm2.module
            layers[i+1] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
        # rots['R1'].append(R1.clone().detach())
        # rots['R2'].append(R2.clone().detach())
        rots['R3'].append(R3.clone().detach())
        rots['R4'].append(R4.clone().detach())
        if sq is not None:
            rots['sq'].append(sq.clone().detach())

    # rots = {'R1':R1, 'R2':R2, 'R3':R3, 'R4': R4}
    return rots



def rotate_projectorsinputllava_fusedsvd(model, Q: torch.Tensor) -> None:
    model_type = model_utils.model_type_extractor(model)
    for proj in model_utils.get_projector(model, model_type):
        W = proj[0].BLinear
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        W_ = proj[2].BLinear
        apply_exact_had_to_linear(W_, had_dim=-1, output=False)

def rotate_attention_inputssvd(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj.BLinear, layer.self_attn.k_proj.BLinear, layer.self_attn.v_proj.BLinear]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_outputsvd(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.self_attn.o_proj.ALinear
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL:
        W = layer.self_attn.out_proj.ALinear
    else:
        raise ValueError(f'Unknown model type {model_type}')

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_mlp_inputsvd(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAMA_MODEL:
        mlp_inputs = [layer.mlp.up_proj.BLinear, layer.mlp.gate_proj.BLinear]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1.BLinear]
    elif model_type == model_utils.LLAVA_MODEL:
        mlp_inputs = [layer.mlp.fc1.BLinear]
    else:
        raise ValueError(f'Unknown model type {model_type}')
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj.ALinear
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2.ALinear
    elif model_type == model_utils.LLAVA_MODEL:
        W = layer.mlp.fc2.ALinear
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj.BLinear
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2.BLinear
    elif model_type == model_utils.LLAVA_MODEL:
        W = layer.mlp.fc2.BLinear
    else:
        raise ValueError(f'Unknown model type {model_type}')
    apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output


def rotate_mlp_outputvitsvd(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj.ALinear
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2.ALinear
    elif model_type == model_utils.LLAVA_MODEL:
        W = layer.mlp.fc2.ALinear
    else:
        raise ValueError(f'Unknown model type {model_type}')
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_ov_projvitsvd(layer, model_type, head_num, head_dim):
    v_proj = layer.self_attn.v_proj.BLinear
    if model_type == model_utils.LLAMA_MODEL:
        o_proj = layer.self_attn.o_proj.ALinear
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL:
        o_proj = layer.self_attn.out_proj.ALinear
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    # apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)
    # apply_exact_had_to_linear(v_proj, had_dim=-1, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

def rotate_ov_projsvd(layer, model_type, head_num, head_dim):
    v_proj = layer.self_attn.v_proj.BLinear
    if model_type == model_utils.LLAMA_MODEL:
        o_proj = layer.self_attn.o_proj.ALinear
    elif model_type == model_utils.OPT_MODEL or model_type == model_utils.LLAVA_MODEL:
        o_proj = layer.self_attn.out_proj.ALinear
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)
    # apply_exact_had_to_linear(v_proj, had_dim=-1, output=True)
    # apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)

