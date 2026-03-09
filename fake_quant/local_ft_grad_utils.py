import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import utils
import gptq_utils
import data_utils
import quant_utils
import model_utils
import rotation_utils
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
import logging
import math
import os
import torch.distributed as dist
import datetime
import time

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

import torch


@torch.enable_grad()
def calib_localft_grad_info(model, dataloader, tokenizer, image_processor, args, use_cache=True, cache_file=None, stage_module=None, layer_indices_dict=None): 
    """
    Calculate Grad matrix for each layer of the model to evaluate parameter importance
    
    Args:
        model: Model to be calibrated
        tokenizer: Tokenizer
        image_processor: Image processor
        args: Parameter configuration
        use_cache: Whether to use cache
        cache_file: Cache file path, automatically generated if None
    """
    model_id = model.config._name_or_path


    
    if args.cache_file is None:
        cache_dir  = args.act_cache_dir + "/cache"
        if args.cache_in_log:
            cache_dir = args.act_cache_dir + "/cache"
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = args.cache_file
        
    # Add relevant information to cache file name
    rotate_info = "rotated" if hasattr(args, "rotate") and args.rotate else "norotate"
    calib_method_info = args.calib_method if hasattr(args, "act_aware") and args.act_aware else "no_act_aware"
    # cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_{rotate_info}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma_grad_info.pt")
    if 'all_' in args.svd_modules:
        sigma_modules = args.svd_modules
    else:
        if args.qkv_fuse:
            sigma_modules = args.svd_modules
        else:
            if args.svd_modules in ['qkv', 'attn', 'all']:
                sigma_modules = args.svd_modules + '_qkvseparate'
            else:
                sigma_modules = args.svd_modules
        if args.mlp_fuse:
            sigma_modules = sigma_modules
        else:
            if args.svd_modules in ['mlp', 'all', 'gaup']:
                sigma_modules = sigma_modules + '_mlpseparate'
            else:
                sigma_modules = sigma_modules
    if args.svd_ft_mode == 'output':
        sigma_modules = sigma_modules + '_output' # args.svd_modules + '_output'# add output grad notation, since now we use Y grad for output objective
    else:
        sigma_modules = sigma_modules # args.svd_modules
    if args.smooth_grad:
        sigma_modules = sigma_modules + f"_smooth{args.smooth_method}"
        if args.smooth_ma_only:
            sigma_modules = sigma_modules + f"_ma_only"
        if args.smooth_percentile > 0:
            sigma_modules = sigma_modules + f"_percentile{args.smooth_percentile}"
        elif args.smooth_outlier_std > 0:
            sigma_modules = sigma_modules + f"_outlier{args.smooth_outlier_std}"
        elif args.smooth_power_threshold > 0:
            sigma_modules = sigma_modules + f"_power{args.smooth_power_threshold}"
        else:
            sigma_modules = sigma_modules + f"_topk{args.topk}"
        if args.smooth_final:
            sigma_modules = sigma_modules + "_final"
        else:
            sigma_modules = sigma_modules + "_sample"
    if stage_module is not None:
        sigma_modules = stage_module + sigma_modules
    if args.is_rank_allocate_ft:
        sigma_modules = sigma_modules + "_rank_allocate"
    if args.is_per_head_svd:
        sigma_modules = sigma_modules + "_perhead"
    if args.is_quant_aware_ft:  
        if args.weighted_none_svd_qat:
            sigma_modules = sigma_modules + "_Wqat"
        else:
            sigma_modules = sigma_modules + "_qat"
    if args.a_clip_ratio == 1.0:
        if args.cache_file is not None:
            cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_{calib_method_info}_sigma{sigma_modules}_taylor{args.taylor_order}_lft_grad_info.pt")
        else:
            cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma{sigma_modules}_taylor{args.taylor_order}_lft_grad_info.pt")
    else:
        if args.cache_file is not None:
            cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_aclip{args.a_clip_ratio}_{calib_method_info}_sigma{sigma_modules}_taylor{args.taylor_order}_lft_grad_info.pt")
        else:
            cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_aclip{args.a_clip_ratio}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma{sigma_modules}_taylor{args.taylor_order}_lft_grad_info.pt")
    
    temp_svd_modules = args.svd_modules
    if 'all' in args.svd_modules:
        args.svd_modules = 'all'
    ### add svd initialization in advance
    device = utils.get_dev()
    for idx, layer in enumerate(model_utils.get_layers(model)):
        if args.svd_modules in ['qkv', 'attn', 'all']:
            q_linear = layer.self_attn.q_proj
            k_linear = layer.self_attn.k_proj
            v_linear = layer.self_attn.v_proj
            
            if args.is_per_head_svd:
                num_heads = layer.self_attn.config.num_key_value_heads if not args.is_q_headnum else layer.self_attn.config.num_attention_heads
                W = q_linear.weight.data.view(num_heads, -1, q_linear.in_features).float().to(device)
            else:
                W = q_linear.weight.data.float().to(device)
            if not hasattr(q_linear, "svd_info"): 
                # Apply activation-aware scaling (if enabled)
                if args.act_aware: # q has the same input as k, so we can use the same scaling matrix
                    scaling_diag_matrix = torch.ones(q_linear.in_features, device=utils.get_dev())  # avoid zero division
                    W, scaling_diag_matrix, scaling_matrix_inv = create_and_fuse_scalingmatrix_to_W(W, scaling_diag_matrix, args, k_linear, keys='q')
                
                # SVD decomposition
                U, S, V = create_svd_decomposition(W, args, layer_indices_dict, q_linear, idx, keys='q_proj', num_heads=num_heads if args.is_per_head_svd else None)
                if args.act_aware:
                    V = fuse_invariance_S_to_V(V, scaling_matrix_inv, scaling_diag_matrix, args, keys='q', layer_indices_dict=layer_indices_dict, num_heads=num_heads if args.is_per_head_svd else None)
                q_linear.svd_info_before_rot = {
                    'U': U,
                    'S': S,
                    'V': V
                }
            if args.is_per_head_svd:
                num_heads = layer.self_attn.config.num_key_value_heads
                W = k_linear.weight.data.float().view(num_heads, -1, k_linear.in_features).to(device)
            else:
                W = k_linear.weight.data.float().to(device)
            if not hasattr(k_linear, "svd_info"):
                # Apply activation-aware scaling (if enabled)
                if args.act_aware:
                    scaling_diag_matrix = torch.ones(k_linear.in_features, device=utils.get_dev())  # avoid zero division
                    W, scaling_diag_matrix, scaling_matrix_inv = create_and_fuse_scalingmatrix_to_W(W, scaling_diag_matrix, args, k_linear, keys='k')
                
                # SVD decomposition
                U, S, V = create_svd_decomposition(W, args, layer_indices_dict, k_linear, idx, keys='k_proj', num_heads=num_heads if args.is_per_head_svd else None)

                if args.act_aware:
                    V = fuse_invariance_S_to_V(V, scaling_matrix_inv, scaling_diag_matrix, args, layer_indices_dict=layer_indices_dict, num_heads=num_heads if args.is_per_head_svd else None)
                k_linear.svd_info_before_rot = {
                    'U': U,
                    'S': S,
                    'V': V
                }
            if args.is_per_head_svd:
                num_heads = layer.self_attn.config.num_key_value_heads
                W = v_linear.weight.data.float().view(num_heads, -1, v_linear.in_features).to(device)
            else:
                W = v_linear.weight.data.float().to(device)
            if not hasattr(v_linear, "svd_info"):
                # Apply activation-aware scaling (if enabled)
                if args.act_aware:
                    scaling_diag_matrix = torch.ones(v_linear.in_features, device=utils.get_dev())  # avoid zero division
                    W, scaling_diag_matrix, scaling_matrix_inv = create_and_fuse_scalingmatrix_to_W(W, scaling_diag_matrix, args, k_linear, keys='v')
                
                # SVD decomposition
                U, S, V = create_svd_decomposition(W, args, layer_indices_dict, v_linear, idx, keys='v_proj', num_heads=num_heads if args.is_per_head_svd else None)

                if args.act_aware:
                    V = fuse_invariance_S_to_V(V, scaling_matrix_inv, scaling_diag_matrix, args, layer_indices_dict=layer_indices_dict, num_heads=num_heads if args.is_per_head_svd else None)
                v_linear.svd_info_before_rot = {
                    'U': U,
                    'S': S,
                    'V': V
                }
        if args.svd_modules in ['attn', 'all', 'o']:
            o_linear = layer.self_attn.o_proj
            W = o_linear.weight.data.float().to(device)
            if not hasattr(o_linear, "svd_info"):
                # Apply activation-aware scaling (if enabled)
                if args.act_aware:
                    scaling_diag_matrix = torch.ones(o_linear.in_features, device=utils.get_dev())  # avoid zero division
                    W, scaling_diag_matrix, scaling_matrix_inv = create_and_fuse_scalingmatrix_to_W(W, scaling_diag_matrix, args, o_linear, keys='o')
                    
                # SVD decomposition
                U, S, V = create_svd_decomposition(W, args, layer_indices_dict, o_linear, idx, keys='o_proj')
                if args.act_aware:
                    V = fuse_invariance_S_to_V(V, scaling_matrix_inv, scaling_diag_matrix, args, keys='o')
                o_linear.svd_info_before_rot = {
                    'U': U,
                    'S': S,
                    'V': V
                }
        if args.svd_modules in ['all', 'mlp', 'gaup']:
            up_linear = layer.mlp.up_proj
            W = up_linear.weight.data.float().to(device)
            if not hasattr(up_linear, "svd_info"):
                if args.act_aware:
                    scaling_diag_matrix = torch.ones(up_linear.in_features, device=utils.get_dev())  # avoid zero division
                    W, scaling_diag_matrix, scaling_matrix_inv = create_and_fuse_scalingmatrix_to_W(W, scaling_diag_matrix, args, up_linear, keys='up')
                
                # SVD decomposition
                U, S, V = create_svd_decomposition(W, args, layer_indices_dict, up_linear, idx, keys='up_proj')
                if args.act_aware:
                    V = fuse_invariance_S_to_V(V, scaling_matrix_inv, scaling_diag_matrix, args, keys='up')
                up_linear.svd_info_before_rot = {
                    'U': U,
                    'S': S,
                    'V': V
                }
            gate_linear = layer.mlp.gate_proj
            W = gate_linear.weight.data.float().to(device)
            if not hasattr(gate_linear, "svd_info"):
                if args.act_aware:
                    scaling_diag_matrix = torch.ones(gate_proj.in_features, device=utils.get_dev())  # avoid zero division
                    W, scaling_diag_matrix, scaling_matrix_inv = create_and_fuse_scalingmatrix_to_W(W, scaling_diag_matrix, args, gate_linear, keys='gate')
                
                # SVD decomposition
                U, S, V = create_svd_decomposition(W, args, layer_indices_dict, gate_linear, idx, keys='gate_proj')
                if args.act_aware:
                    V = fuse_invariance_S_to_V(V, scaling_matrix_inv, scaling_diag_matrix, args, keys='gate')
                gate_linear.svd_info_before_rot = {
                    'U': U,
                    'S': S,
                    'V': V
                }
        if args.svd_modules in ['all', 'mlp', 'down']:
            down_linear = layer.mlp.down_proj
            W = down_linear.weight.data.float().to(device)
            if not hasattr(down_linear, "svd_info"):
                if args.act_aware:
                    scaling_diag_matrix = torch.ones(down_linear.in_features, device=utils.get_dev())  # avoid zero division
                    W, scaling_diag_matrix, scaling_matrix_inv = create_and_fuse_scalingmatrix_to_W(W, scaling_diag_matrix, args, down_linear, keys='down')
                
                # SVD decomposition
                U, S, V = create_svd_decomposition(W, args, layer_indices_dict, down_linear, idx, keys='down_proj')
                if args.act_aware:
                    V = fuse_invariance_S_to_V(V, scaling_matrix_inv, scaling_diag_matrix, args, keys='down')
                down_linear.svd_info_before_rot = {
                    'U': U,
                    'S': S,
                    'V': V
                }
    # ### add normfuse and rotation fuse here
    # if args.rotate and args.is_quant_aware_ft:
    #     set_rotation_norm_fuse_to_model(model, args)
    #     logging.info("Successfully set rotation and norm fuse to model!")
    
    if args.is_quant_aware_ft and not args.is_stage and args.weighted_none_svd_qat:
        # Here we skip the svd initialization and rotation for UV for selected svdmodules
        # But for weighted W, we need to store for all model
        args.svd_modules = 'all'

    if os.path.exists(cache_file) and use_cache:
        logging.info(f"Loading Grad information cache from {cache_file}...")

        all_grad_info = torch.load(cache_file, map_location="cpu")
        # Load gradient information into the self_attn.W_grad_info attribute of corresponding layers
        for idx, layer in enumerate(model_utils.get_layers(model)):
            layer_key = f"layer_{idx}"
            if layer_key in all_grad_info:
                if args.svd_modules in ['qkv', 'attn', 'all']:
                    if args.qkv_fuse:
                        layer.self_attn.k_proj.W_grad_info = all_grad_info[layer_key]['k_proj'].to(utils.get_dev())
                    else:
                        layer.self_attn.q_proj.W_grad_info = all_grad_info[layer_key]['q_proj'].to(utils.get_dev())
                        layer.self_attn.k_proj.W_grad_info = all_grad_info[layer_key]['k_proj'].to(utils.get_dev())
                        layer.self_attn.v_proj.W_grad_info = all_grad_info[layer_key]['v_proj'].to(utils.get_dev())
                if args.svd_modules in ['attn', 'all', 'o']: # only valid for not qkv-only SVD
                    layer.self_attn.o_proj.W_grad_info = all_grad_info[layer_key]['o_proj'].to(utils.get_dev())
                if args.svd_modules in ['all', 'mlp','gaup']: # only valid for all model SVD  
                    if args.mlp_fuse:
                        layer.mlp.up_proj.W_grad_info = all_grad_info[layer_key]['up_proj'].to(utils.get_dev())
                    else:
                        layer.mlp.up_proj.W_grad_info = all_grad_info[layer_key]['up_proj'].to(utils.get_dev())
                        layer.mlp.gate_proj.W_grad_info = all_grad_info[layer_key]['gate_proj'].to(utils.get_dev())
                if args.svd_modules in ['all', 'mlp', 'down']:
                    layer.mlp.down_proj.W_grad_info = all_grad_info[layer_key]['down_proj'].to(utils.get_dev())
        logging.info("Successfully loaded Grad information cache!")
        if args.is_quant_aware_ft:
            args.svd_modules = temp_svd_modules
        return
    
    print("Starting Grad information calculation...")
    logging.info('start grad computing')
    model.eval()

    # Ensure the entire model is on CUDA
    device = utils.get_dev()
    model = model.to(device)

    accumulation_steps = 1   # Number of accumulated batches
    batch_count = 0          # Accumulated batch counter
    
    # Set model to training mode and only allow gradient computation for svd_modules layers
    set_grad_for_svd_modules(model, args)

    # For output mode, register gradient hooks to capture ∂L/∂y
    if hasattr(args, 'svd_ft_mode') and args.svd_ft_mode == 'output':
        grad_hooks, grad_dict = register_gradient_hook_to_linear_layer(model, args)
    else:
        grad_hooks, grad_dict = None, None

    for batch in tqdm(dataloader, desc="Computing Gradient Information"):
        try:
            if tokenizer is None and image_processor is None:
                input_ids = batch[0].to(device)
                labels = input_ids.clone()
                if args.label_shift: # enable label shift like palu, or spinquant handle this in Trainer?
                    labels = labels[:, 1:]
                    input_ids = input_ids[:, :-1]
                with torch.enable_grad():
                    outputs = model(input_ids, labels=labels)
                    loss = outputs[0]
                    loss = loss / accumulation_steps  # Normalize loss
                    loss.backward()
            elif tokenizer is None: # SmolVLM
                    # Use message_to_prompt_train to process batch data
                inputs, _, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                # Define recursive function to move nested tensor structures to specified device
                def move_to_device(obj, target_device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(target_device)
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(move_to_device(item, target_device) for item in obj)
                    elif isinstance(obj, dict):
                        return {k: move_to_device(v, target_device) for k, v in obj.items()}
                    else:
                        return obj
                
                # Move data to corresponding device
                inputs = move_to_device(inputs, device)
                output_ids = move_to_device(output_ids, device)

                input_ids = inputs.get('input_ids')
                output_ids = output_ids 
                
                # Adjust input and label lengths to match
                if input_ids.size(1) != output_ids.size(1):
                    max_len = max(input_ids.size(1), output_ids.size(1))
                    if input_ids.size(1) < max_len:
                        padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                            dtype=input_ids.dtype, device=input_ids.device)
                        input_ids = torch.cat([input_ids, padding], dim=1)
                    else:
                        input_ids = input_ids[:, :max_len]
                    if output_ids.size(1) < max_len:
                        padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                            fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                        output_ids = torch.cat([output_ids, padding], dim=1)
                    else:
                        output_ids = output_ids[:, :max_len]
                    print(f"Adjusted input and label lengths to {max_len}")
                    input_ids = input_ids.to(device)
                    output_ids = output_ids.to(device)
                
                inputs['input_ids'] = input_ids
                inputs['attention_mask'] = input_ids.ne(0).to(device)

                # Calculate loss for current batch and perform gradient accumulation
                with torch.enable_grad():
                    outputs = model(**inputs, labels=output_ids)
                    loss = outputs[0]
                    loss = loss / accumulation_steps  # Normalize loss
                    loss.backward()
            elif tokenizer == 'hf_v16': # LLaVA Next/v1.6
                    # Use message_to_prompt_train to process batch data
                inputs, _, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                
                # Define recursive function to move nested tensor structures to specified device
                def move_to_device(obj, target_device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(target_device)
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(move_to_device(item, target_device) for item in obj)
                    elif isinstance(obj, dict):
                        return {k: move_to_device(v, target_device) for k, v in obj.items()}
                    else:
                        return obj
                
                # Move data to corresponding device
                inputs = move_to_device(inputs, device)
                output_ids = move_to_device(output_ids, device)
                input_ids = inputs.get('input_ids')
                output_ids = output_ids

                # 0 as pad token id, following many tokenizers
                if input_ids.size(1) != output_ids.size(1):
                    max_len = max(input_ids.size(1), output_ids.size(1))
                    if input_ids.size(1) < max_len:
                        padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                            dtype=input_ids.dtype, device=input_ids.device)
                        input_ids = torch.cat([input_ids, padding], dim=1)
                    else:
                        input_ids = input_ids[:, :max_len]
                    if output_ids.size(1) < max_len:
                        padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                            fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                        output_ids = torch.cat([output_ids, padding], dim=1)
                    else:
                        output_ids = output_ids[:, :max_len]
                    print(f"Adjusted input and label lengths to {max_len}")
                    input_ids = input_ids.to(device)
                    output_ids = output_ids.to(device)
                
                inputs['input_ids'] = input_ids
                inputs['attention_mask'] = input_ids.ne(0).to(device)


                # Calculate loss for current batch and perform gradient accumulation
                with torch.enable_grad():
                    outputs = model(**inputs, labels=output_ids)
                    loss = outputs[0]
                    loss = loss / accumulation_steps  # Normalize loss
                    loss.backward()
            else: # LLaVA v1.5
                # Use message_to_prompt_train to process batch data
                input_ids, images, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                
                # Define recursive function to move nested tensor structures to specified device
                def move_to_device(obj, target_device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(target_device)
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(move_to_device(item, target_device) for item in obj)
                    elif isinstance(obj, dict):
                        return {k: move_to_device(v, target_device) for k, v in obj.items()}
                    else:
                        return obj
                
                # Move data to corresponding device
                input_ids = move_to_device(input_ids, device)
                image_sizes = None
                if images is not None:
                    images, image_sizes = images
                    images = move_to_device(images, device)
                output_ids = move_to_device(output_ids, device)
                
                # Adjust input and label lengths to match
                if input_ids.size(1) != output_ids.size(1):
                    max_len = max(input_ids.size(1), output_ids.size(1))
                    if input_ids.size(1) < max_len:
                        padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                            dtype=input_ids.dtype, device=input_ids.device)
                        input_ids = torch.cat([input_ids, padding], dim=1)
                    else:
                        input_ids = input_ids[:, :max_len]
                    if output_ids.size(1) < max_len:
                        padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                            fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                        output_ids = torch.cat([output_ids, padding], dim=1)
                    else:
                        output_ids = output_ids[:, :max_len]
                    print(f"Adjusted input and label lengths to {max_len}")
                    input_ids = input_ids.to(device)
                    output_ids = output_ids.to(device)
                else:
                    print(f"Original input and label lengths: {input_ids.size(1)}")
                
                attention_mask = input_ids.ne(0).to(device) 

                # Calculate loss for current batch and perform gradient accumulation
                with torch.enable_grad():
                    outputs = model(input_ids=input_ids, images=images, labels=output_ids, attention_mask=attention_mask, image_sizes=image_sizes)
                    loss = outputs[0]
                    loss = loss / accumulation_steps  # Normalize loss
                    loss.backward()
            
            batch_count += 1  # Count each batch
            
            # Perform backpropagation when accumulated the set number of batches
            if batch_count % accumulation_steps == 0:
                # Update gradient information for S in each layer
                for idx, layer in enumerate(model_utils.get_layers(model)):
                    if args.svd_modules in ['qkv', 'attn', 'all']:
                        if args.qkv_fuse:
                            # if hasattr(layer.self_attn.k_proj, 'qkv_svd_info'): #[FIXME: now gradinfo only support qkv fuse]
                            #     # qkv only SVD grad info compute
                            #     svd_info = layer.self_attn.k_proj.qkv_svd_info
                            q_linear = layer.self_attn.q_proj
                            k_linear = layer.self_attn.k_proj
                            v_linear = layer.self_attn.v_proj
                            if args.svd_ft_mode == 'output' and grad_dict is not None:
                                if f"block_{idx}.attnq_grad" in grad_dict:
                                    # print('q_grad', grad_dict[f"block_{idx}.attnq_grad"].shape)
                                    # print('k_grad', grad_dict[f"block_{idx}.attnk_grad"].shape)
                                    # print('v_grad', grad_dict[f"block_{idx}.attnv_grad"].shape) # got shape [1, 1024, c1]
                                    output_grads = torch.cat([
                                            grad_dict[f"block_{idx}.attnq_grad"].to(device).to(torch.bfloat16),
                                            grad_dict[f"block_{idx}.attnk_grad"].to(device).to(torch.bfloat16),
                                            grad_dict[f"block_{idx}.attnv_grad"].to(device).to(torch.bfloat16),
                                            ],dim=-1)
                                    if not hasattr(layer.self_attn.k_proj, 'W_grad_info'):
                                        layer.self_attn.k_proj.W_grad_info = get_layer_importance(output_grads, args)
                                    else:
                                        layer.self_attn.k_proj.W_grad_info += get_layer_importance(output_grads, args)
                            else:
                                if (q_linear.weight.grad is not None and 
                                    k_linear.weight.grad is not None and 
                                    v_linear.weight.grad is not None):
                                    multiplier = args.group_ratio if args.group_ratio > 0 else 1.0
                                    if args.is_per_head_svd:
                                        num_heads = layer.self_attn.config.num_key_value_heads if not args.is_q_headnum else layer.self_attn.config.num_attention_heads
                                        grad_cat = torch.cat([
                                            q_linear.weight.grad.detach().view(num_heads, -1, q_linear.in_features).to(torch.bfloat16),
                                            k_linear.weight.grad.detach().view(num_heads, -1, k_linear.in_features).to(torch.bfloat16) * multiplier,
                                            v_linear.weight.grad.detach().view(num_heads, -1, v_linear.in_features).to(torch.bfloat16) * multiplier,
                                            ], dim=1).to(device) # n, 3c, C_in
                                    else:
                                        grad_cat = torch.cat([
                                            q_linear.weight.grad.detach().to(torch.bfloat16),
                                            k_linear.weight.grad.detach().to(torch.bfloat16) * multiplier,
                                            v_linear.weight.grad.detach().to(torch.bfloat16) * multiplier,
                                        ], dim=0).to(device)
                                    if not hasattr(layer.self_attn.k_proj, 'W_grad_info'): # 
                                        layer.self_attn.k_proj.W_grad_info = get_layer_importance(grad_cat, args) # [FIXEDME:]keep it as [c, c']?
                                    else:
                                        layer.self_attn.k_proj.W_grad_info += get_layer_importance(grad_cat, args)
                        elif args.kv_fuse:
                            logging.info(f"Layer {idx} is not supported for kv fuse, skipping")
                        else:
                            q_linear = layer.self_attn.q_proj
                            k_linear = layer.self_attn.k_proj
                            v_linear = layer.self_attn.v_proj
                            if args.svd_ft_mode == 'output' and grad_dict is not None:
                                # Q‐proj
                                key_q = f"block_{idx}.attnq_grad"
                                if key_q in grad_dict:
                                    q_grad = grad_dict[key_q].to(device).to(torch.bfloat16)
                                    q_imp  = get_layer_importance(q_grad, args)
                                    if not hasattr(q_linear, 'W_grad_info'):
                                        q_linear.W_grad_info = q_imp
                                    else:
                                        q_linear.W_grad_info += q_imp

                                # K‐proj
                                key_k = f"block_{idx}.attnk_grad"
                                if key_k in grad_dict:
                                    k_grad = grad_dict[key_k].to(device).to(torch.bfloat16)
                                    k_imp  = get_layer_importance(k_grad, args)
                                    if not hasattr(k_linear, 'W_grad_info'):
                                        k_linear.W_grad_info = k_imp
                                    else:
                                        k_linear.W_grad_info += k_imp

                                # V‐proj
                                key_v = f"block_{idx}.attnv_grad"
                                if key_v in grad_dict:
                                    v_grad = grad_dict[key_v].to(device).to(torch.bfloat16)
                                    v_imp  = get_layer_importance(v_grad, args)
                                    if not hasattr(v_linear, 'W_grad_info'):
                                        v_linear.W_grad_info = v_imp
                                    else:
                                        v_linear.W_grad_info += v_imp
                            else:
                                if q_linear.weight.grad is not None:
                                    grad_cat = q_linear.weight.grad.detach().to(torch.bfloat16)
                                    if args.is_per_head_svd:
                                        num_heads = layer.self_attn.config.num_attention_heads
                                        grad_cat = grad_cat.view(num_heads, -1, q_linear.in_features)
                                    if not hasattr(layer.self_attn.q_proj, 'W_grad_info'):
                                        layer.self_attn.q_proj.W_grad_info = get_layer_importance(grad_cat, args)
                                    else:
                                        layer.self_attn.q_proj.W_grad_info += get_layer_importance(grad_cat, args)
                                if k_linear.weight.grad is not None:
                                    grad_cat = k_linear.weight.grad.detach().to(torch.bfloat16)
                                    if args.is_per_head_svd:
                                        num_heads = layer.self_attn.config.num_key_value_heads
                                        grad_cat = grad_cat.view(num_heads, -1, k_linear.in_features)
                                    if not hasattr(layer.self_attn.k_proj, 'W_grad_info'):
                                        layer.self_attn.k_proj.W_grad_info = get_layer_importance(grad_cat, args)
                                    else:
                                        layer.self_attn.k_proj.W_grad_info += get_layer_importance(grad_cat, args)
                                if v_linear.weight.grad is not None:
                                    grad_cat = v_linear.weight.grad.detach().to(torch.bfloat16)
                                    if args.is_per_head_svd:
                                        num_heads = layer.self_attn.config.num_key_value_heads
                                        grad_cat = grad_cat.view(num_heads, -1, v_linear.in_features)
                                    if not hasattr(layer.self_attn.v_proj, 'W_grad_info'):
                                        layer.self_attn.v_proj.W_grad_info = get_layer_importance(grad_cat, args)
                                    else:
                                        layer.self_attn.v_proj.W_grad_info += get_layer_importance(grad_cat, args)
                    # [FIXME: add o-proj/FFN etc. grad info compute]
                    if args.svd_modules in ['attn', 'all', 'o']:
                        o_linear = layer.self_attn.o_proj
                        if o_linear.weight.grad is not None:
                            grad_cat = o_linear.weight.grad.detach().to(torch.bfloat16)
                            if not hasattr(layer.self_attn.o_proj, 'W_grad_info'):
                                layer.self_attn.o_proj.W_grad_info = get_layer_importance(grad_cat, args)
                            else:
                                layer.self_attn.o_proj.W_grad_info += get_layer_importance(grad_cat, args)
                    if args.svd_modules in ['all', 'mlp', 'gaup']:
                        if args.mlp_fuse:
                            up_proj = layer.mlp.up_proj
                            gate_proj = layer.mlp.gate_proj
                            if (up_proj.weight.grad is not None and 
                                gate_proj.weight.grad is not None):
                                grad_cat = torch.cat([
                                    up_proj.weight.grad.detach().to(torch.bfloat16),
                                    gate_proj.weight.grad.detach().to(torch.bfloat16)
                                ], dim=0).to(device)
                            if not hasattr(layer.mlp.up_proj, 'W_grad_info'):
                                layer.mlp.up_proj.W_grad_info = get_layer_importance(grad_cat, args)
                            else:
                                layer.mlp.up_proj.W_grad_info += get_layer_importance(grad_cat, args)
                        else:
                            up_proj = layer.mlp.up_proj
                            if up_proj.weight.grad is not None:
                                grad_cat = up_proj.weight.grad.detach().to(torch.bfloat16)
                                if not hasattr(layer.mlp.up_proj, 'W_grad_info'):
                                    layer.mlp.up_proj.W_grad_info = get_layer_importance(grad_cat, args)
                                else:
                                    layer.mlp.up_proj.W_grad_info += get_layer_importance(grad_cat, args)
                            gate_proj = layer.mlp.gate_proj
                            if gate_proj.weight.grad is not None:
                                grad_cat = gate_proj.weight.grad.detach().to(torch.bfloat16)
                                if not hasattr(layer.mlp.gate_proj, 'W_grad_info'):
                                    layer.mlp.gate_proj.W_grad_info = get_layer_importance(grad_cat, args)
                                else:
                                    layer.mlp.gate_proj.W_grad_info += get_layer_importance(grad_cat, args)
                    if args.svd_modules in ['mlp', 'all', 'down']:
                        down_proj = layer.mlp.down_proj
                        if args.svd_ft_mode == 'output' and grad_dict is not None:
                            # Use captured output gradients ∂L/∂y
                            if f"block_{idx}.mlpdown_grad" in grad_dict:
                                # print(f"block_{idx}.mlpdown_grad in grad_dict")
                                output_grads = grad_dict[f"block_{idx}.mlpdown_grad"].to(device).to(torch.bfloat16)
                                if not hasattr(layer.mlp.down_proj, 'W_grad_info'):
                                    layer.mlp.down_proj.W_grad_info = get_layer_importance(output_grads, args)
                                else:
                                    layer.mlp.down_proj.W_grad_info += get_layer_importance(output_grads, args)
                        else:
                            # Use weight gradients ∂L/∂W (original method)
                            if down_proj.weight.grad is not None:
                                grad_cat = down_proj.weight.grad.detach().to(torch.bfloat16)
                                if not hasattr(layer.mlp.down_proj, 'W_grad_info'):
                                    layer.mlp.down_proj.W_grad_info = get_layer_importance(grad_cat, args)
                                else:
                                    layer.mlp.down_proj.W_grad_info += get_layer_importance(grad_cat, args)
                    
                model.zero_grad()  # Clear gradients
        
        except Exception as e:
            print(f"Error occurred during Grad information calculation: {e}")
            import traceback
            print("Detailed error information:")
            traceback.print_exc()
            if isinstance(batch, dict):
                print(f"Batch data keys: {list(batch.keys())}")
            elif isinstance(batch, list) and len(batch) > 0:
                print(f"Type of first item in batch data: {type(batch[0])}")
            continue

    # Normalize S gradient information
    if batch_count > 0:
        layer_idx = 0
        for layer in model_utils.get_layers(model):
            if hasattr(layer.self_attn.k_proj, 'W_grad_info'):
                if args.svd_modules in ['qkv', 'attn', 'all']:
                    if args.qkv_fuse:
                        layer.self_attn.k_proj.W_grad_info = layer.self_attn.k_proj.W_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                        # Apply final smoothing if enabled
                        if hasattr(args, 'smooth_final') and args.smooth_final:
                            layer.self_attn.k_proj.W_grad_info = smooth_gradient_info(layer.self_attn.k_proj.W_grad_info, args, layer_idx)
                    else:
                        layer.self_attn.q_proj.W_grad_info = layer.self_attn.q_proj.W_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                        layer.self_attn.k_proj.W_grad_info = layer.self_attn.k_proj.W_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                        layer.self_attn.v_proj.W_grad_info = layer.self_attn.v_proj.W_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                        # Apply final smoothing if enabled
                        if hasattr(args, 'smooth_final') and args.smooth_final:
                            layer.self_attn.q_proj.W_grad_info = smooth_gradient_info(layer.self_attn.q_proj.W_grad_info, args, layer_idx)
                            layer.self_attn.k_proj.W_grad_info = smooth_gradient_info(layer.self_attn.k_proj.W_grad_info, args, layer_idx)
                            layer.self_attn.v_proj.W_grad_info = smooth_gradient_info(layer.self_attn.v_proj.W_grad_info, args, layer_idx)
            if args.svd_modules in ['attn', 'all', 'o']: # [FIXME: add sgrad info check]
                layer.self_attn.o_proj.W_grad_info = layer.self_attn.o_proj.W_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                # Apply final smoothing if enabled
                if hasattr(args, 'smooth_final') and args.smooth_final:
                    layer.self_attn.o_proj.W_grad_info = smooth_gradient_info(layer.self_attn.o_proj.W_grad_info, args, layer_idx)
            if args.svd_modules in ['all', 'mlp', 'gaup']:
                if args.mlp_fuse:
                    layer.mlp.up_proj.W_grad_info = layer.mlp.up_proj.W_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                    # Apply final smoothing if enabled
                    if hasattr(args, 'smooth_final') and args.smooth_final:
                        layer.mlp.up_proj.W_grad_info = smooth_gradient_info(layer.mlp.up_proj.W_grad_info, args, layer_idx)
                else:
                    layer.mlp.up_proj.W_grad_info = layer.mlp.up_proj.W_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                    layer.mlp.gate_proj.W_grad_info = layer.mlp.gate_proj.W_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                    # Apply final smoothing if enabled
                    if hasattr(args, 'smooth_final') and args.smooth_final:
                        layer.mlp.up_proj.W_grad_info = smooth_gradient_info(layer.mlp.up_proj.W_grad_info, args, layer_idx)
                        layer.mlp.gate_proj.W_grad_info = smooth_gradient_info(layer.mlp.gate_proj.W_grad_info, args, layer_idx)
            if args.svd_modules in ['all', 'mlp', 'down']:
                layer.mlp.down_proj.W_grad_info = layer.mlp.down_proj.W_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                # Apply final smoothing if enabled
                if hasattr(args, 'smooth_final') and args.smooth_final:
                    layer.mlp.down_proj.W_grad_info = smooth_gradient_info(layer.mlp.down_proj.W_grad_info, args, layer_idx)
            layer_idx += 1
    if args.smooth_grad and args.smooth_final:
        logging.info('smooth grad info done at final step')
    elif args.smooth_grad and not args.smooth_final:
        logging.info('smooth grad info done at sample step')
    else:
        logging.info('no smooth grad info')
    logging.info('finished grad computing')
    # Save S gradient information
    all_grad_info = {}
    for idx, layer in enumerate(model_utils.get_layers(model)):
        all_grad_info[f"layer_{idx}"] = {}
        print(f"Layer {idx} initialize dict")
        if args.svd_modules in ['qkv', 'attn', 'all']:
            if args.qkv_fuse:
                if hasattr(layer.self_attn.k_proj, 'W_grad_info'):
                    print(f"Layer {idx}: {layer.self_attn.k_proj.W_grad_info.shape}")
                    all_grad_info[f"layer_{idx}"]['k_proj'] = layer.self_attn.k_proj.W_grad_info.cpu()
            else:
                if hasattr(layer.self_attn.q_proj, 'W_grad_info'):
                    print(f"Layer {idx}: {layer.self_attn.q_proj.W_grad_info.shape}")
                    all_grad_info[f"layer_{idx}"]['q_proj'] = layer.self_attn.q_proj.W_grad_info.cpu()
                if hasattr(layer.self_attn.k_proj, 'W_grad_info'):
                    print(f"Layer {idx}: {layer.self_attn.k_proj.W_grad_info.shape}")
                    all_grad_info[f"layer_{idx}"]['k_proj'] = layer.self_attn.k_proj.W_grad_info.cpu()
                if hasattr(layer.self_attn.v_proj, 'W_grad_info'):
                    print(f"Layer {idx}: {layer.self_attn.v_proj.W_grad_info.shape}")
                    all_grad_info[f"layer_{idx}"]['v_proj'] = layer.self_attn.v_proj.W_grad_info.cpu()
        if args.svd_modules in ['attn', 'all', 'o']:
            print(f"Layer {idx}: {layer.self_attn.o_proj.W_grad_info.shape}")
            all_grad_info[f"layer_{idx}"]['o_proj'] = layer.self_attn.o_proj.W_grad_info.cpu()
        if args.svd_modules in ['all', 'mlp', 'gaup']:
            if args.mlp_fuse:
                print(f"Layer {idx}: {layer.mlp.up_proj.W_grad_info.shape}")
                all_grad_info[f"layer_{idx}"]['up_proj'] = layer.mlp.up_proj.W_grad_info.cpu()
            else:
                print(f"Layer {idx}: {layer.mlp.up_proj.W_grad_info.shape}")
                print(f"Layer {idx}: {layer.mlp.gate_proj.W_grad_info.shape}")
                all_grad_info[f"layer_{idx}"]['up_proj'] = layer.mlp.up_proj.W_grad_info.cpu()
                all_grad_info[f"layer_{idx}"]['gate_proj'] = layer.mlp.gate_proj.W_grad_info.cpu()
        if args.svd_modules in ['mlp', 'all', 'down']:
            print(f"Layer {idx}: {layer.mlp.down_proj.W_grad_info.shape}")
            all_grad_info[f"layer_{idx}"]['down_proj'] = layer.mlp.down_proj.W_grad_info.cpu()

    logging.info(f"Saving Grad information cache to {cache_file}...")
    torch.save(all_grad_info, cache_file)
    logging.info("Grad information cache saved successfully!")
    
    # Clean up gradient hooks if they were registered
    if grad_hooks is not None:
        for h in grad_hooks:
            h.remove()
        del grad_hooks, grad_dict
    if ('all' in temp_svd_modules) or args.is_quant_aware_ft:
        args.svd_modules = temp_svd_modules


def local_ft_fused_svd(model, dataloader, tokenizer, image_processor, args, use_cache=True, cache_file=None, layer_indices_dict=None):
    """
    Local FT fused SVD decomposition
    Here we need to call
        hook to catch input and output of qkv or target linear layer
        svd initialization based on ASVD/SVDLLM?
        inner loop to perform local ft
    """
    disable_grad_for_svd_modules(model, args)
    device = utils.get_dev()
    model = model.to(device)
    loss_dict = {}
    loss_curve = []
    bsz_counter = 0
    norm_loss = True
    in_outs = {}
    quantizers = {}
    in_outs["inp"] = []
    
    from rotation_utils import get_orthogonal_matrix
    Q = get_orthogonal_matrix(model_utils.get_lm_config(model).hidden_size, args.rotate_mode, args.seed)
    
    for batch in tqdm(dataloader, desc="Local FT fused SVD"):
        # hooks, in_outs = register_hook_to_linear_layer(model, args) # to ensure in_outs are re-initialized?
        # add extra args.bs and counter to bsz catch?
        if args.svd_ft_mode == 'output': # capture the corresponding input of the target modules
            if bsz_counter % args.nsamples == 0:
                in_outs = {} # or move this out loop to catch all batch
                in_outs["inp"] = [] # change this to forward hook and catcher to catch only the layer input
                class Catch_args_input(nn.Module):
                    def __init__(self, module):
                        super().__init__()
                        self.module = module
                    def forward(self, inp, *args, **kwargs):
                        in_outs['inp'].append(inp)
                        in_outs['args'] = args
                        in_outs['kwargs'] = kwargs
                        # out = self.module(inp, *args, **kwargs)
                        raise ValueError
                layers = model_utils.get_layers(model)
                layers[0] = Catch_args_input(layers[0])
                # model.model.layers[0] = Catch_args_input(model.model.layers[0])
            try:
                # forward pass to catch input output of linear layer
                if tokenizer is None and image_processor is None:
                    input_ids = batch[0].to(device)
                    labels = input_ids.clone()
                    if args.label_shift: # enable label shift like palu, or spinquant handle this in Trainer?
                        labels = labels[:, 1:]
                        input_ids = input_ids[:, :-1]
                    with torch.no_grad():
                        model(input_ids, labels=labels)
                elif tokenizer is None: # SmolVLM
                        # Use message_to_prompt_train to process batch data
                    inputs, _, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                    # Define recursive function to move nested tensor structures to specified device
                    def move_to_device(obj, target_device):
                        if isinstance(obj, torch.Tensor):
                            return obj.to(target_device)
                        elif isinstance(obj, (list, tuple)):
                            return type(obj)(move_to_device(item, target_device) for item in obj)
                        elif isinstance(obj, dict):
                            return {k: move_to_device(v, target_device) for k, v in obj.items()}
                        else:
                            return obj
                    
                    # Move data to corresponding device
                    inputs = move_to_device(inputs, device)
                    output_ids = move_to_device(output_ids, device)

                    input_ids = inputs.get('input_ids')
                    output_ids = output_ids 
                    
                    # Adjust input and label lengths to match

                    if input_ids.size(1) != output_ids.size(1):
                        max_len = max(input_ids.size(1), output_ids.size(1))
                        if input_ids.size(1) < max_len:
                            padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                                dtype=input_ids.dtype, device=input_ids.device)
                            input_ids = torch.cat([input_ids, padding], dim=1)
                        else:
                            input_ids = input_ids[:, :max_len]
                        if output_ids.size(1) < max_len:
                            padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                                fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                            output_ids = torch.cat([output_ids, padding], dim=1)
                        else:
                            output_ids = output_ids[:, :max_len]
                        print(f"Adjusted input and label lengths to {max_len}")
                        input_ids = input_ids.to(device)
                        output_ids = output_ids.to(device)
                    
                    inputs['input_ids'] = input_ids
                    inputs['attention_mask'] = input_ids.ne(0).to(device)

                    # Calculate loss for current batch and perform gradient accumulation
                    with torch.no_grad():
                        model(**inputs, labels=output_ids)
                elif tokenizer == 'hf_v16': # LLaVA Next/v1.6
                        # Use message_to_prompt_train to process batch data
                    inputs, _, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                    
                    # Define recursive function to move nested tensor structures to specified device
                    def move_to_device(obj, target_device):
                        if isinstance(obj, torch.Tensor):
                            return obj.to(target_device)
                        elif isinstance(obj, (list, tuple)):
                            return type(obj)(move_to_device(item, target_device) for item in obj)
                        elif isinstance(obj, dict):
                            return {k: move_to_device(v, target_device) for k, v in obj.items()}
                        else:
                            return obj
                    
                    # Move data to corresponding device
                    inputs = move_to_device(inputs, device)
                    output_ids = move_to_device(output_ids, device)
                    input_ids = inputs.get('input_ids')
                    output_ids = output_ids

                    # 0 as pad token id, following many tokenizers
                    if input_ids.size(1) != output_ids.size(1):
                        max_len = max(input_ids.size(1), output_ids.size(1))
                        if input_ids.size(1) < max_len:
                            padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                                dtype=input_ids.dtype, device=input_ids.device)
                            input_ids = torch.cat([input_ids, padding], dim=1)
                        else:
                            input_ids = input_ids[:, :max_len]
                        if output_ids.size(1) < max_len:
                            padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                                fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                            output_ids = torch.cat([output_ids, padding], dim=1)
                        else:
                            output_ids = output_ids[:, :max_len]
                        print(f"Adjusted input and label lengths to {max_len}")
                        input_ids = input_ids.to(device)
                        output_ids = output_ids.to(device)
                    
                    inputs['input_ids'] = input_ids
                    inputs['attention_mask'] = input_ids.ne(0).to(device)

                    # Calculate loss for current batch and perform gradient accumulation
                    with torch.no_grad():
                        model(**inputs, labels=output_ids)
                else: # LLaVA v1.5
                    # Use message_to_prompt_train to process batch data
                    input_ids, images, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                    
                    # Define recursive function to move nested tensor structures to specified device
                    def move_to_device(obj, target_device):
                        if isinstance(obj, torch.Tensor):
                            return obj.to(target_device)
                        elif isinstance(obj, (list, tuple)):
                            return type(obj)(move_to_device(item, target_device) for item in obj)
                        elif isinstance(obj, dict):
                            return {k: move_to_device(v, target_device) for k, v in obj.items()}
                        else:
                            return obj
                    
                    # Move data to corresponding device
                    input_ids = move_to_device(input_ids, device)
                    image_sizes = None
                    if images is not None:
                        images, image_sizes = images
                        images = move_to_device(images, device)
                    output_ids = move_to_device(output_ids, device)
                    
                    # Adjust input and label lengths to match
                    if input_ids.size(1) != output_ids.size(1):
                        max_len = max(input_ids.size(1), output_ids.size(1))
                        if input_ids.size(1) < max_len:
                            padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                                dtype=input_ids.dtype, device=input_ids.device)
                            input_ids = torch.cat([input_ids, padding], dim=1)
                        else:
                            input_ids = input_ids[:, :max_len]
                        if output_ids.size(1) < max_len:
                            padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                                fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                            output_ids = torch.cat([output_ids, padding], dim=1)
                        else:
                            output_ids = output_ids[:, :max_len]
                        print(f"Adjusted input and label lengths to {max_len}")
                        input_ids = input_ids.to(device)
                        output_ids = output_ids.to(device)
                    else:
                        print(f"Original input and label lengths: {input_ids.size(1)}")
                    
                    attention_mask = input_ids.ne(0).to(device) 

                    # Calculate loss for current batch and perform gradient accumulation
                    with torch.no_grad():
                        model(input_ids=input_ids, images=images, labels=output_ids, attention_mask=attention_mask, image_sizes=image_sizes)
            except ValueError:
                pass
            
        if (bsz_counter + 1) % args.nsamples == 0: # here we just catch the whole nsamples
            model = model.cpu()
            if args.svd_ft_mode == 'output':
                layers = model_utils.get_layers(model)
                layers[0] = layers[0].module
                # model.model.layers[0] = model.model.layers[0].module
                in_outs["inp"] = torch.cat(in_outs["inp"], dim=0)
                in_outs['out'] = torch.empty_like(in_outs["inp"])
                hooks, activations = register_hook_to_linear_layer(model, args)
            
            for idx, layer in enumerate(model_utils.get_layers(model)):
                optimizers = {}
                schedulers = {}
                if args.is_quant_aware_ft:
                    if idx not in quantizers:
                        quantizers[idx] = {}
                if args.svd_ft_mode == 'output':
                    layer = layer.to(device)
                    for bsz in range(0, args.nsamples, args.bs):
                        in_outs["out"][bsz:bsz+args.bs] = layer(in_outs["inp"][bsz:bsz+args.bs], **in_outs['kwargs'])[0]
                    layer = layer.cpu()
                    utils.cleanup_memory()
                    time.sleep(0.1)
                if args.is_quant_aware_ft and not args.is_stage:
                    temp_svd_modules = args.svd_modules
                    args.svd_modules = 'all'
                if args.svd_modules in ['qkv', 'attn', 'all']:
                    q_linear = layer.self_attn.q_proj
                    k_linear = layer.self_attn.k_proj
                    v_linear = layer.self_attn.v_proj
                    if args.qkv_fuse:
                        logging.info(f"Layer {idx} is not supported for qkv fuse, skipping")
                    elif args.kv_fuse:
                        logging.info(f"Layer {idx} is not supported for kv fuse, skipping")
                    else:
                        try:
                            if args.is_per_head_svd:
                                num_heads = layer.self_attn.config.num_key_value_heads if not args.is_q_headnum else layer.self_attn.config.num_attention_heads
                                W = q_linear.weight.data.view(num_heads, -1, q_linear.in_features).float().to(device)
                            else:
                                W = q_linear.weight.data.float().to(device)
                            if not hasattr(q_linear, "svd_info"):
                                try:
                                    U = q_linear.svd_info_before_rot['U']
                                    S = q_linear.svd_info_before_rot['S']
                                    V = q_linear.svd_info_before_rot['V']
                                except:
                                    U = q_linear.weight.data
                                    S = V = None
                                #### A Blinear initialization
                                Alinear, Blinear, R = create_ABlinear_and_quantizer(U, S, V, args, quantizers, idx, keys='q', num_heads=num_heads if args.is_per_head_svd else None)
                            W = W.to(torch.bfloat16).to(device)
                            
                            if args.svd_ft_mode == 'output':
                                __in = activations.pop(f"block_{idx}.attnq_in").to(device).to(torch.bfloat16)
                            else:
                                __in = None
                            
                            if args.weighted_svd:
                                if Blinear is None and not args.weighted_none_svd_qat:
                                    args.weighted_svd = False
                                    print("skip weighted svd")
                                else:
                                    weight = layer.self_attn.q_proj.W_grad_info.to(device).to(torch.bfloat16) # nb, override the actquantwrapper thing
                                    print(f"Weighted weight shape: {weight.shape}")
                            if args.qat_optim_R:
                                if args.is_per_head_svd:
                                    params_R = list(R.parameters())
                                else:
                                    params_R = [R]
                            else:
                                params_R = None
                            try:
                                if args.is_per_head_svd:
                                    params_ = list(Alinear.parameters()) + list(Blinear.parameters())
                                    if args.qat_uv_reg:
                                        reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]
                                else:
                                    params_ = [Alinear.weight, Blinear.weight]
                            except:
                                params_ = [Alinear.weight]
                                if args.qat_uv_reg:
                                    reg_UV = [W]
                           
                            optimizers, schedulers = create_optimizer(params_, params_R, args, optimizers, schedulers, key='q')
                            
                            if not args.qat_L_off or Blinear is not None:
                                if f'layer{idx}' not in loss_dict:
                                    loss_dict[f'layer{idx}'] = {}
                                if 'q' not in loss_dict[f'layer{idx}']:
                                    loss_dict[f'layer{idx}']['q'] = []
                            else:
                                temp_localft_iters = args.localft_iters
                                args.localft_iters = 0
                                logging.info(f"Layer {idx} Qproj Local FT disabled due to qat_L_off or Blinear is not None")
                            if args.is_quant_aware_ft:
                                qat_start_iter = int(args.localft_iters * args.qat_start_iter)
                            else:
                                qat_start_iter = None
                            for _ in range(args.localft_iters):
                                if qat_start_iter is not None:
                                    # add extra criteria to control qat start iter
                                    if _ < qat_start_iter:
                                        args.is_quant_aware_ft = False
                                        if _ == 0:
                                            print(f"Layer {idx} Qproj QAT Local QAT disabled before  qat start iter")
                                    elif _ == qat_start_iter:
                                        if args.qat_uv_reg:
                                            logging.info(f"Layer {idx} Qproj QAT Local QAT, update reg_UV at iter {_}")
                                            reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]
                                            
                                        args.is_quant_aware_ft = True
                                        
                                        for pg in optimizers['q'].param_groups:
                                            pg['lr'] = args.qat_lr_UV
                                        if 'q' in schedulers and schedulers['q'] is not None:
                                            schedulers['q'].base_lrs = [pg['lr'] for pg in optimizers['q'].param_groups]

                                        print(f"Layer {idx} Qproj QAT Local QAT enabled after qat start iter")
                                    else: # _ > qat_start_iter
                                        args.is_quant_aware_ft = True
                                        
                                if (_+1) % args.qat_param_update_freq == 0 or _ == qat_start_iter:
                                    qat_param_update = True
                                else:
                                    qat_param_update = False
                                    
                                with torch.enable_grad():
                                    for bsz in range(0, args.nsamples, args.bs):
                                        if args.is_quant_aware_ft: # Local QAT
                                            optimizers['q'].zero_grad()
                                            optimizers['q_R'].zero_grad()
                                            loss = iter_local_qat_loss(Alinear, Blinear, W, Q, bsz, weight, __in, args, quantizers, idx,  keys='q', reg_UV=reg_UV if args.qat_uv_reg else None, R=R, qat_param_update=qat_param_update)
                                            optimizers['q'].step()
                                            optimizers['q_R'].step()
                                            if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                                                schedulers['q'].step()
                                                schedulers['q_R'].step()
                                            if _ % getattr(args, 'scheduler_step_size', 30) == 0:
                                                print(f"Layer {idx} Qproj - Iter {_}, Loss: {loss.item():.6f}, U/V LR: {optimizers['q'].param_groups[0]['lr']:.6f}, R LR: {optimizers['q_R'].param_groups[0]['lr']:.6f}")
                                        else: # Local FT
                                            optimizers['q'].zero_grad()
                                            loss = iter_local_ft_loss(Alinear, Blinear, W, bsz, weight, __in, args, quantizers, idx,  keys='q')
                                            optimizers['q'].step()
                                            if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                                                schedulers['q'].step()
                                            if _ % getattr(args, 'scheduler_step_size', 30) == 0:
                                                print(f"Layer {idx} Qproj - Iter {_}, Loss: {loss.item():.6f}, LR: {optimizers['q'].param_groups[0]['lr']:.6f}")
                                                
                                        loss_curve.append(loss.item())
                                        loss_dict[f'layer{idx}']['q'].append(loss.item())
                                        
                                if qat_start_iter is not None and qat_start_iter >= args.localft_iters and _ == args.localft_iters - 1:
                                    args.is_quant_aware_ft = True
                            
                            if Blinear is None:
                                q_dtype = q_linear.weight.dtype
                                q_device = q_linear.weight.device
                                quantizers[idx]['q']['Alinear'].find_params(Alinear.weight.data)
                                q_linear.weight.data = quantizers[idx]['q']['Alinear'].quantize(Alinear.weight.data).to(q_dtype).to(q_device)
                                if not args.weighted_none_svd_qat:
                                    args.weighted_svd = True
                                if args.qat_L_off:
                                    args.localft_iters = temp_localft_iters
                                print(f"Layer {idx} Qproj Local Qat FT completed, Sigma shape: {S[0].shape if S is not None else 'None'}")
                            else:
                                # store SVD results
                                if args.is_per_head_svd:
                                    if args.is_quant_aware_ft:
                                        fuse_per_head_svd_rotated_weights(Alinear, Blinear, quantizers, idx, keys='q', num_heads=num_heads, R=R)
                                    layer.self_attn.q_proj.svd_info = {
                                        'U': Alinear.cpu(),
                                        'S': None,
                                        'V': Blinear.cpu()
                                    }
                                    print(f"Layer {idx} Qproj Local FT completed, Sigma shape: {S[0].shape if S is not None else 'None'}")
                                else:
                                    layer.self_attn.q_proj.svd_info = {
                                        'U': Alinear.weight.data.cpu(),
                                        'S': None,
                                        'V': Blinear.weight.data.cpu()
                                    }
                                    print(f"Layer {idx} Qproj Local FT completed, Sigma shape: {S.shape}")
                        except Exception as e:
                            print(f"Layer {idx} Qproj Local FT failed: {e}")
                            import traceback
                            traceback.print_exc()
                        try:
                            if args.is_per_head_svd:
                                num_heads = layer.self_attn.config.num_key_value_heads
                                W = k_linear.weight.data.float().view(num_heads, -1, k_linear.in_features).to(device)
                            else:
                                W = k_linear.weight.data.float().to(device)
                            if not hasattr(k_linear, "svd_info"):
                                try:
                                    U = k_linear.svd_info_before_rot['U']
                                    S = k_linear.svd_info_before_rot['S']
                                    V = k_linear.svd_info_before_rot['V']
                                except:
                                    U = k_linear.weight.data
                                    S = V = None
                                #### A Blinear initialization
                                Alinear, Blinear, R = create_ABlinear_and_quantizer(U, S, V, args, quantizers, idx, keys='k', num_heads=num_heads if args.is_per_head_svd else None)
                            W = W.to(torch.bfloat16)

                            if args.svd_ft_mode == 'output':
                                __in = activations.pop(f"block_{idx}.attnk_in").to(device).to(torch.bfloat16)
                            else:
                                __in = None

                            if args.weighted_svd:
                                if Blinear is None and not args.weighted_none_svd_qat:
                                    args.weighted_svd = False
                                    print("skip weighted svd")
                                else:
                                    weight = layer.self_attn.k_proj.W_grad_info.to(device).to(torch.bfloat16)
                            if args.qat_optim_R:
                                if args.is_per_head_svd:
                                    params_R = list(R.parameters())
                                else:
                                    params_R = [R]
                            else:
                                params_R = None
                            try:
                                if args.is_per_head_svd:
                                    params_ = list(Alinear.parameters()) + list(Blinear.parameters())
                                    if args.qat_uv_reg:
                                        reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]
                                else:
                                    params_ = [Alinear.weight, Blinear.weight]
                            except:
                                params_ = [Alinear.weight]
                                if args.qat_uv_reg:
                                    reg_UV = [W]

                            optimizers, schedulers = create_optimizer(params_, params_R, args, optimizers, schedulers, key='k')
                            
                            if not args.qat_L_off or Blinear is not None:
                                if f'layer{idx}' not in loss_dict:
                                    loss_dict[f'layer{idx}'] = {}
                                if 'k' not in loss_dict[f'layer{idx}']:
                                    loss_dict[f'layer{idx}']['k'] = []
                            else:
                                temp_localft_iters = args.localft_iters
                                args.localft_iters = 0
                                logging.info(f"Layer {idx} Kproj Local FT disabled due to qat_L_off or Blinear is not None")
                            if args.is_quant_aware_ft:
                                qat_start_iter = int(args.localft_iters * args.qat_start_iter)
                            else:
                                qat_start_iter = None
                            for _ in range(args.localft_iters):
                                if qat_start_iter is not None:
                                    # add extra criteria to control qat start iter
                                    if _ < qat_start_iter:
                                        args.is_quant_aware_ft = False
                                    elif _ == qat_start_iter:
                                        if args.qat_uv_reg:
                                            reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]

                                        args.is_quant_aware_ft = True
                                        for pg in optimizers['k'].param_groups:
                                            pg['lr'] = args.qat_lr_UV
                                        if 'k' in schedulers and schedulers['k'] is not None:
                                            schedulers['k'].base_lrs = [pg['lr'] for pg in optimizers['k'].param_groups]
                                    else: # _ > qat_start_iter
                                        args.is_quant_aware_ft = True
                                if (_+1) % args.qat_param_update_freq == 0 or _ == qat_start_iter:
                                    qat_param_update = True
                                else:
                                    qat_param_update = False
                                    
                                with torch.enable_grad():
                                    for bsz in range(0, args.nsamples, args.bs):
                                        if args.is_quant_aware_ft: # Local QAT
                                            optimizers['k'].zero_grad()
                                            optimizers['k_R'].zero_grad()
                                            loss = iter_local_qat_loss(Alinear, Blinear, W, Q, bsz, weight, __in, args, quantizers, idx,  keys='k', reg_UV=reg_UV if args.qat_uv_reg else None, R=R, qat_param_update=qat_param_update)
                                            optimizers['k'].step()
                                            optimizers['k_R'].step()
                                            if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                                                schedulers['k'].step()
                                                schedulers['k_R'].step()
                                            if _ % getattr(args, 'scheduler_step_size', 30) == 0:
                                                print(f"Layer {idx} Kproj - Iter {_}, Loss: {loss.item():.6f}, U/V LR: {optimizers['k'].param_groups[0]['lr']:.6f}, R LR: {optimizers['k_R'].param_groups[0]['lr']:.6f}")
                                        else: # Local FT
                                            optimizers['k'].zero_grad()
                                            loss = iter_local_ft_loss(Alinear, Blinear, W, bsz, weight, __in, args, quantizers, idx,  keys='k')
                                            optimizers['k'].step()
                                            if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                                                schedulers['k'].step()
                                            if _ % getattr(args, 'scheduler_step_size', 30) == 0:
                                                print(f"Layer {idx} Kproj - Iter {_}, Loss: {loss.item():.6f}, LR: {optimizers['k'].param_groups[0]['lr']:.6f}")
                                                
                                        loss_curve.append(loss.item())
                                        loss_dict[f'layer{idx}']['k'].append(loss.item())
                                if qat_start_iter is not None and qat_start_iter >= args.localft_iters and _ == args.localft_iters - 1:
                                    args.is_quant_aware_ft = True
                            
                            if Blinear is None:
                                k_dtype = k_linear.weight.dtype
                                k_device = k_linear.weight.device
                                quantizers[idx]['k']['Alinear'].find_params(Alinear.weight.data)
                                k_linear.weight.data = quantizers[idx]['k']['Alinear'].quantize(Alinear.weight.data).to(k_dtype).to(k_device)
                                print(f"Layer {idx} Kproj Local Qat FT completed, Sigma shape: {S[0].shape if S is not None else 'None'}")
                                if not args.weighted_none_svd_qat:
                                    args.weighted_svd = True
                                if args.qat_L_off:
                                    args.localft_iters = temp_localft_iters
                            else:
                                # store SVD results
                                if args.is_per_head_svd:
                                    if args.is_quant_aware_ft:
                                        fuse_per_head_svd_rotated_weights(Alinear, Blinear, quantizers, idx, keys='k', num_heads=num_heads, R=R)
                                    layer.self_attn.k_proj.svd_info = {
                                        'U': Alinear.cpu(),
                                        'S': None,
                                        'V': Blinear.cpu()
                                    }
                                    print(f"Layer {idx} Kproj Local FT completed, Sigma shape: {S[0].shape if S is not None else 'None'}")
                                else:
                                    layer.self_attn.k_proj.svd_info = {
                                        'U': Alinear.weight.data.cpu(),
                                        'S': None,
                                        'V': Blinear.weight.data.cpu()
                                    }
                                    print(f"Layer {idx} Kproj Local FT completed, Sigma shape: {S.shape if S is not None else 'None'}")
                        except Exception as e:
                            print(f"Layer {idx} Kproj Local FT failed: {e}")
                            import traceback
                            traceback.print_exc()
                        try:
                            if args.is_per_head_svd:
                                num_heads = layer.self_attn.config.num_key_value_heads if not args.is_q_headnum else layer.self_attn.config.num_attention_heads
                                W = v_linear.weight.data.float().view(num_heads, -1, v_linear.in_features).to(device)
                            else:
                                W = v_linear.weight.data.float().to(device)
                            if not hasattr(v_linear, "svd_info"):
                                try:
                                    U = v_linear.svd_info_before_rot['U']
                                    S = v_linear.svd_info_before_rot['S']
                                    V = v_linear.svd_info_before_rot['V']
                                except:
                                    U = v_linear.weight.data
                                    S = V = None

                                #### A Blinear initialization
                                Alinear, Blinear, R = create_ABlinear_and_quantizer(U, S, V, args, quantizers, idx, keys='v', num_heads=num_heads if args.is_per_head_svd else None)
                            W = W.to(torch.bfloat16)

                            if args.svd_ft_mode == 'output':
                                __in = activations.pop(f"block_{idx}.attnv_in").to(device).to(torch.bfloat16)
                            else:
                                __in = None

                            if args.weighted_svd:
                                if Blinear is None and not args.weighted_none_svd_qat:
                                    args.weighted_svd = False
                                    weight = None
                                    print("skip weighted svd")
                                else:
                                    weight = layer.self_attn.v_proj.W_grad_info.to(device).to(torch.bfloat16)
                            if args.qat_optim_R:
                                if args.is_per_head_svd:
                                    params_R = list(R.parameters())
                                else:
                                    params_R = [R]
                            else:
                                params_R = None
                            try:
                                if args.is_per_head_svd:
                                    params_ = list(Alinear.parameters()) + list(Blinear.parameters())
                                    if args.qat_uv_reg:
                                        reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]
                                else:
                                    params_ = [Alinear.weight, Blinear.weight]
                            except:
                                params_ = [Alinear.weight]
                                if args.qat_uv_reg:
                                    reg_UV = [W]                      
                            optimizers, schedulers = create_optimizer(params_, params_R, args, optimizers, schedulers, key='v')
                            
                            
                            if not args.qat_L_off or Blinear is not None:
                                if f'layer{idx}' not in loss_dict:
                                    loss_dict[f'layer{idx}'] = {}
                                if 'v' not in loss_dict[f'layer{idx}']:
                                    loss_dict[f'layer{idx}']['v'] = []
                            else:
                                temp_localft_iters = args.localft_iters
                                args.localft_iters = 0
                                logging.info(f"Layer {idx} Vproj Local FT disabled due to qat_L_off or Blinear is not None")
                            if args.is_quant_aware_ft:
                                qat_start_iter = int(args.localft_iters * args.qat_start_iter)
                            else:
                                qat_start_iter = None
                            for _ in range(args.localft_iters):
                                if qat_start_iter is not None:
                                    # add extra criteria to control qat start iter
                                    if _ < qat_start_iter:
                                        args.is_quant_aware_ft = False
                                    elif _ == qat_start_iter:
                                        if args.qat_uv_reg:
                                            reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]
                                        
                                        args.is_quant_aware_ft = True
                                        for pg in optimizers['v'].param_groups:
                                            pg['lr'] = args.qat_lr_UV
                                        if 'v' in schedulers and schedulers['v'] is not None:
                                            schedulers['v'].base_lrs = [pg['lr'] for pg in optimizers['v'].param_groups]
                                    else: # _ > qat_start_iter
                                        args.is_quant_aware_ft = True
                                if (_+1) % args.qat_param_update_freq == 0 or _ == qat_start_iter:
                                    qat_param_update = True
                                else:
                                    qat_param_update = False
                               
                                with torch.enable_grad():
                                    for bsz in range(0, args.nsamples, args.bs):
                                        if args.is_quant_aware_ft: # Local QAT
                                            optimizers['v'].zero_grad()
                                            optimizers['v_R'].zero_grad()
                                            loss = iter_local_qat_loss(Alinear, Blinear, W, Q, bsz, weight, __in, args, quantizers, idx,  keys='v', reg_UV=reg_UV if args.qat_uv_reg else None, R=R, qat_param_update=qat_param_update)
                                            optimizers['v'].step()
                                            optimizers['v_R'].step()
                                            if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                                                schedulers['v'].step()
                                                schedulers['v_R'].step()
                                            if _ % getattr(args, 'scheduler_step_size', 30) == 0:
                                                print(f"Layer {idx} Vproj - Iter {_}, Loss: {loss.item():.6f}, U/V LR: {optimizers['v'].param_groups[0]['lr']:.6f}, R LR: {optimizers['v_R'].param_groups[0]['lr']:.6f}")
                                        else: # Local FT
                                            optimizers['v'].zero_grad()
                                            loss = iter_local_ft_loss(Alinear, Blinear, W, bsz, weight, __in, args, quantizers, idx,  keys='v')
                                            optimizers['v'].step()
                                            if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                                                schedulers['v'].step()
                                            if _ % getattr(args, 'scheduler_step_size', 30) == 0:
                                                print(f"Layer {idx} Vproj - Iter {_}, Loss: {loss.item():.6f}, LR: {optimizers['v'].param_groups[0]['lr']:.6f}")
                                                
                                        loss_curve.append(loss.item())
                                        loss_dict[f'layer{idx}']['v'].append(loss.item())
                                        
                                if qat_start_iter is not None and qat_start_iter >= args.localft_iters and _ == args.localft_iters - 1:
                                    args.is_quant_aware_ft = True
                            if Blinear is None:
                                v_dtype = v_linear.weight.dtype
                                v_device = v_linear.weight.device
                                quantizers[idx]['v']['Alinear'].find_params(Alinear.weight.data)
                                v_linear.weight.data = quantizers[idx]['v']['Alinear'].quantize(Alinear.weight.data).to(v_dtype).to(v_device)
                                print(f"Layer {idx} Vproj Local Qat FT completed, Sigma shape: {S[0].shape if S is not None else 'None'}")
                                if not args.weighted_none_svd_qat:
                                    args.weighted_svd = True
                                if args.qat_L_off:
                                    args.localft_iters = temp_localft_iters
                            else:
                                # store SVD results
                                if args.is_per_head_svd:
                                    if args.is_quant_aware_ft:
                                        fuse_per_head_svd_rotated_weights(Alinear, Blinear, quantizers, idx, keys='v', num_heads=num_heads, R=R)
                                    layer.self_attn.v_proj.svd_info = {
                                        'U': Alinear.cpu(),
                                        'S': None,
                                        'V': Blinear.cpu()
                                    }
                                    print(f"Layer {idx} Vproj Local FT completed, Sigma shape: {S[0].shape if S is not None else 'None'}")
                                else:
                                    # if args.is_quant_aware_ft:
                                    #     layer.self_attn.v_proj.svd_info = {
                                    #         'U': quantizers[idx]['v']['Alinear'].quantize(Alinear.weight.data).cpu(),
                                    #         'S': None,
                                    #         'V': quantizers[idx]['v']['Blinear'].quantize(Blinear.weight.data).cpu()
                                    #     }
                                    # else:
                                    layer.self_attn.v_proj.svd_info = {
                                        'U': Alinear.weight.data.cpu(),
                                        'S': None,
                                        'V': Blinear.weight.data.cpu()
                                    }
                                    print(f"Layer {idx} Vproj Local FT completed, Sigma shape: {S.shape if S is not None else 'None'}")
                        except Exception as e:
                            print(f"Layer {idx} Vproj Local FT failed: {e}")
                            import traceback
                            traceback.print_exc()
                if args.svd_modules in ['attn', 'all', 'o']:
                    o_linear = layer.self_attn.o_proj
                    try:
                        W = o_linear.weight.data.float().to(device)
                        if not hasattr(o_linear, "svd_info"):
                            try:    
                                U = o_linear.svd_info_before_rot['U']
                                S = o_linear.svd_info_before_rot['S']
                                V = o_linear.svd_info_before_rot['V']
                            except:
                                U = o_linear.weight.data
                                S = V = None
                            
                            #### A Blinear initialization
                            Alinear, Blinear, R = create_ABlinear_and_quantizer(U, S, V, args, quantizers, idx, keys='o')
                            # del o_linear.svd_info_before_rot #[FIXME:]

                        W = W.to(torch.bfloat16)

                        if args.svd_ft_mode == 'output':
                            __in = activations.pop(f"block_{idx}.attno_in").to(device).to(torch.bfloat16)
                        else:
                            __in = None

                        if args.weighted_svd:
                            if Blinear is None and not args.weighted_none_svd_qat:
                                args.weighted_svd = False
                                weight = None
                                print("skip weighted svd")
                            else:
                                weight = layer.self_attn.o_proj.W_grad_info.to(device).to(torch.bfloat16)
                        
                        params_R = None
                        try:
                            if args.is_per_head_svd:
                                params_ = list(Alinear.parameters()) + list(Blinear.parameters())
                                if args.qat_uv_reg:
                                    reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]
                            else:
                                params_ = [Alinear.weight, Blinear.weight]
                        except:
                            params_ = [Alinear.weight]
                            if args.qat_uv_reg:
                                reg_UV = [W]

                        optimizers, schedulers = create_optimizer(params_, params_R, args, optimizers, schedulers, key='o')

                        if not args.qat_L_off or Blinear is not None:
                            if f'layer{idx}' not in loss_dict:
                                loss_dict[f'layer{idx}'] = {}
                            if 'o' not in loss_dict[f'layer{idx}']:
                                loss_dict[f'layer{idx}']['o'] = []
                        else:
                            temp_localft_iters = args.localft_iters
                            args.localft_iters = 0
                            logging.info(f"Layer {idx} Oproj Local FT disabled due to qat_L_off or Blinear is not None")
                        ### [FIXME: add extra function to compute loss iter, this should be the same for all svdmodules]
                        if args.is_quant_aware_ft:
                            qat_start_iter = int(args.localft_iters * args.qat_start_iter)
                        else:
                            qat_start_iter = None
                        for _ in range(args.localft_iters):
                            if qat_start_iter is not None:
                                # add extra criteria to control qat start iter
                                if _ < qat_start_iter:
                                    args.is_quant_aware_ft = False
                                elif _ == qat_start_iter:
                                    if args.qat_uv_reg:
                                        reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]
                                    
                                    args.is_quant_aware_ft = True
                                else:
                                    args.is_quant_aware_ft = True
                            for bsz in range(0, args.nsamples, args.bs):
                                optimizers['o'].zero_grad()
                                with torch.enable_grad():
                                    loss = iter_local_ft_loss(Alinear, Blinear, W, bsz, weight, __in, args, quantizers, idx,  keys='o', R=R if args.qat_optim_R else None)
                                    optimizers['o'].step()    
                                if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                                    schedulers['o'].step()
                                    if _ % getattr(args, 'scheduler_step_size', 30) == 0:
                                        print(f"Layer {idx} Oproj - Iter {_}, Loss: {loss.item():.6f}, LR: {optimizers['o'].param_groups[0]['lr']:.6f}")
                                loss_curve.append(loss.item())
                                loss_dict[f'layer{idx}']['o'].append(loss.item())
                            if qat_start_iter is not None and qat_start_iter >= args.localft_iters and _ == args.localft_iters - 1:
                                args.is_quant_aware_ft = True
                        if Blinear is None:
                            o_dtype = o_linear.weight.dtype
                            o_device = o_linear.weight.device
                            quantizers[idx]['o']['Alinear'].find_params(Alinear.weight.data)
                            o_linear.weight.data = quantizers[idx]['o']['Alinear'].quantize(Alinear.weight.data).to(o_dtype).to(o_device)   
                            print(f"Layer {idx} Oproj Local Qat FT completed, Sigma shape: {S[0].shape if S is not None else 'None'}")
                            if not args.weighted_none_svd_qat:
                                args.weighted_svd = True
                            if args.qat_L_off:
                                args.localft_iters = temp_localft_iters
                        else:
                            # store SVD results
                            o_linear.svd_info = {
                                'U': Alinear.weight.data.cpu(),
                                'S': None,
                                'V': Blinear.weight.data.cpu()
                            }
                            print(f"Layer {idx} Oproj Local FT completed, Sigma shape: {S.shape}")

                    except Exception as e:
                        print(f"Layer {idx} Oproj Local FT failed: {e}")
                        import traceback
                        traceback.print_exc()
                if args.svd_modules in ['all', 'mlp', 'gaup']:
                    gate_linear = layer.mlp.gate_proj
                    up_linear = layer.mlp.up_proj
                    if args.mlp_fuse:
                        try:
                            if args.is_quant_aware_ft:
                                logging.info('No support of QAT-ft, as mlp share is not performing well')
                            # for now only support concat version?
                            W = torch.cat([
                                gate_linear.weight.data.float(),
                                up_linear.weight.data.float()
                            ], dim=0).to(device)
                            w = W.clone()
                            if not hasattr(up_linear, "svd_info"):
                                # Apply activation-aware scaling (if enabled)
                                if args.act_aware:
                                    scaling_diag_matrix = torch.ones(up_linear.in_features, device=utils.get_dev())  # avoid zero division
                                    if hasattr(up_linear, "scaling_diag_matrix"):
                                        scaling_diag_matrix *= up_linear.scaling_diag_matrix.to(utils.get_dev())**args.act_alpha
                                        scaling_diag_matrix += 1e-6  # avoid zero division
                                        scaling_matrix_inv = None
                                        W = W * scaling_diag_matrix.view(1, -1)
                                    elif hasattr(up_linear, "scaling_diag_matrixS"):
                                        scaling_diag_matrix = up_linear.scaling_diag_matrixS.to(utils.get_dev())
                                        
                                        try:
                                            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
                                        except RuntimeError as e:
                                            logging.info("Warning: scaling_diag_matrix is not full rank, adding epsilon for stability.")
                                            eps = 1e-6
                                            scaling_diag_matrix += eps * torch.eye(scaling_diag_matrix.shape[0], device=device)
                                            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
                                        W = W @ scaling_diag_matrix.float()
                                    else:
                                        raise ValueError("No scaling_diag_matrix found")
                                
                                # SVD decomposition
                                U, S, Vt = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
                                V = Vt.T
                                # truncate rank:
                                n_params = up_linear.weight.numel() + gate_linear.weight.numel()
                                compressed_params = int(n_params * args.rank_ratio)
                                LocalFT_rank = compressed_params // (up_linear.in_features + up_linear.out_features + gate_linear.out_features)
                                if layer_indices_dict is not None:
                                    LocalFT_rank = layer_indices_dict[idx]['up_proj']
                                    U = U[:, LocalFT_rank]
                                    S = S[LocalFT_rank]
                                    V = V[:, LocalFT_rank]
                                else:
                                    U = U[:, :LocalFT_rank]
                                    S = S[:LocalFT_rank]
                                    V = V[:, :LocalFT_rank]
                                if args.act_aware:
                                    if scaling_matrix_inv is not None:
                                        V = scaling_matrix_inv.T @ V
                                    else:
                                        V = V / scaling_diag_matrix.view(-1, 1)
                                
                                Alinear = nn.Linear(U.size(1), U.size(0), bias=False)
                                Blinear = nn.Linear(V.size(1), V.size(0), bias=False)
                                Alinear.weight.data = U.mul(S.sqrt()).contiguous().to(torch.bfloat16)
                                Blinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous().to(torch.bfloat16)
                            else:
                                U = up_linear.svd_info['U']
                                V = up_linear.svd_info['V']
                                Alinear = nn.Linear(U.size(1), U.size(0), bias=False)
                                Blinear = nn.Linear(V.size(1), V.size(0), bias=False)
                                Alinear.weight.data = U.to(torch.bfloat16).to(device)
                                Blinear.weight.data = V.to(torch.bfloat16).to(device)
                            W = w.to(torch.bfloat16)
                            Alinear.weight.requires_grad = True
                            Blinear.weight.requires_grad = True
                            if args.svd_ft_mode == 'output':
                                __in = activations.pop(f"block_{idx}.mlpgaup_in").to(device).to(torch.bfloat16)
                            else:
                                __in = None

                            if args.weighted_svd:
                                weight = layer.mlp.up_proj.W_grad_info.to(device).to(torch.bfloat16)

                            
                            if args.svd_ft_optim == 'adam':
                                optimizers['gaup'] = torch.optim.Adam([Alinear.weight, Blinear.weight], lr=args.localft_lr)
                            elif args.svd_ft_optim == 'sgd':
                                optimizers['gaup'] = torch.optim.SGD([Alinear.weight, Blinear.weight], lr=args.localft_lr)
                            elif args.svd_ft_optim == 'adamw':
                                optimizers['gaup'] = torch.optim.AdamW([Alinear.weight, Blinear.weight], lr=args.localft_lr)
                            elif args.svd_ft_optim == 'sgdm':
                                optimizers['gaup'] = torch.optim.SGD([Alinear.weight, Blinear.weight], lr=args.localft_lr, momentum=0.9)
                            else:
                                raise ValueError(f"Invalid svd_ft_optim: {args.svd_ft_optim}")
                            
                            # Add learning rate scheduler for gaup projection
                            if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                                if args.scheduler_type == 'step':
                                    schedulers['gaup'] = torch.optim.lr_scheduler.StepLR(
                                        optimizers['gaup'], 
                                        step_size=getattr(args, 'scheduler_step_size', 30), 
                                        gamma=getattr(args, 'scheduler_gamma', 0.8)
                                    )
                                elif args.scheduler_type == 'cosine':
                                    schedulers['gaup'] = torch.optim.lr_scheduler.CosineAnnealingLR(
                                        optimizers['gaup'], 
                                        T_max=args.localft_iters * (args.nsamples // args.bs),
                                        eta_min=getattr(args, 'scheduler_eta_min', 1e-6)
                                    )
                                else:
                                    raise ValueError(f"Invalid scheduler_type: {args.scheduler_type}. Supported types: 'step', 'cosine'")
                            
                            if f'layer{idx}' not in loss_dict:
                                loss_dict[f'layer{idx}'] = {}
                            if 'gaup' not in loss_dict[f'layer{idx}']:
                                loss_dict[f'layer{idx}']['gaup'] = []
                            for _ in range(args.localft_iters):
                                for bsz in range(0, args.nsamples, args.bs):
                                    optimizers['gaup'].zero_grad()
                                    with torch.enable_grad():
                                        if args.svd_ft_mode == 'output':
                                            _in = __in[bsz:bsz+args.bs]
                                            if args.weighted_svd:
                                                _out = _in @ (weight * W).T # [c, c']
                                                y = _in @ (weight * (Alinear.weight @ Blinear.weight)).T # [c', r] @ [r, c] -> [c, c']
                                            else:
                                                _out = _in @ W.T # [c, c']
                                                y = _in @ (Alinear.weight @ Blinear.weight).T # [c', r] @ [r, c] -> [c, c']
                                        elif args.svd_ft_mode == 'weight':
                                            if args.weighted_svd:
                                                _out = weight * W
                                                y = weight * (Alinear.weight @ Blinear.weight)
                                            else:
                                                _out = W
                                                y = Alinear.weight @ Blinear.weight
                                        else:
                                            raise ValueError(f"Invalid svd_ft_mode: {args.svd_ft_mode}")
                                        if norm_loss:
                                            loss = (y - _out).pow(2).mean() / _out.pow(2).mean()
                                            # loss = ((y - _out)/_out).pow(2).mean()
                                        else:
                                            loss = (y - _out).pow(2).mean()
                                        loss.backward()
                                        optimizers['gaup'].step()
                                    if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                                        schedulers['gaup'].step()
                                        if _ % getattr(args, 'scheduler_step_size', 30) == 0:
                                            print(f"Layer {idx} MLP gaup - Iter {_}, Loss: {loss.item():.6f}, LR: {optimizers['gaup'].param_groups[0]['lr']:.6f}")
                                    loss_curve.append(loss.item())
                                    loss_dict[f'layer{idx}']['gaup'].append(loss.item())
                            
                            # store SVD results
                            up_linear.svd_info = {
                                'U': Alinear.weight.data.cpu(),
                                'S': None,
                                'V': Blinear.weight.data.cpu()
                            }
                            
                            print(f"Layer {idx} MLP gaup SVD Local FT completed, Sigma shape: {S.shape}")
                        except Exception as e:
                            print(f"Layer {idx} MLP gaup SVD Local FT failed: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        try:
                            W = up_linear.weight.data.float().to(device)
                            if not hasattr(up_linear, "svd_info"):
                                try:    
                                    U = up_linear.svd_info_before_rot['U']              
                                    S = up_linear.svd_info_before_rot['S']
                                    V = up_linear.svd_info_before_rot['V']
                                except:
                                    U = up_linear.weight.data
                                    S = V = None
                                
                                #### A Blinear initialization
                                Alinear, Blinear, R = create_ABlinear_and_quantizer(U, S, V, args, quantizers, idx, keys='up')
                            W = W.to(torch.bfloat16)

                            if args.svd_ft_mode == 'output':
                                __in = activations[f"block_{idx}.mlpgaup_in"].to(device).to(torch.bfloat16)
                            else:
                                __in = None

                            if args.weighted_svd:
                                if Blinear is None and not args.weighted_none_svd_qat:
                                    args.weighted_svd = False
                                    weight = None
                                    print("skip weighted svd")
                                else:
                                    weight = layer.mlp.up_proj.W_grad_info.to(device).to(torch.bfloat16)
                            params_R = None
                            try:
                                if args.is_per_head_svd:
                                    params_ = list(Alinear.parameters()) + list(Blinear.parameters())
                                    if args.qat_uv_reg:
                                        reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]
                                else:
                                    params_ = [Alinear.weight, Blinear.weight]
                            except:
                                params_ = [Alinear.weight]
                                if args.qat_uv_reg:
                                    reg_UV = [W]
                            
                            optimizers, schedulers = create_optimizer(params_, params_R, args, optimizers, schedulers, key='up')
                            if not args.qat_L_off or Blinear is not None:
                                if f'layer{idx}' not in loss_dict:
                                    loss_dict[f'layer{idx}'] = {}
                                if 'up' not in loss_dict[f'layer{idx}']:
                                    loss_dict[f'layer{idx}']['up'] = []
                            else:
                                temp_localft_iters = args.localft_iters
                                args.localft_iters = 0
                                logging.info(f"Layer {idx} Upproj Local FT disabled due to qat_L_off or Blinear is not None")
                            if args.is_quant_aware_ft:
                                qat_start_iter = int(args.localft_iters * args.qat_start_iter)
                            else:
                                qat_start_iter = None
                            for _ in range(args.localft_iters):
                                if qat_start_iter is not None:
                                    # add extra criteria to control qat start iter
                                    if _ < qat_start_iter:
                                        args.is_quant_aware_ft = False
                                    elif _ == qat_start_iter:
                                        if args.qat_uv_reg:
                                            reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]
                                        #     update_qat_params(Alinear, Blinear, args, quantizers, idx, keys='q', num_heads=num_heads if args.is_per_head_svd else None)
                                        args.is_quant_aware_ft = True
                                        disable_grad_for_ABlinear(Alinear, Blinear)
                                    else: # _ > qat_start_iter
                                        args.is_quant_aware_ft = True
                                torch.cuda.empty_cache()
                                for bsz in range(0, args.nsamples, args.bs):
                                    optimizers['up'].zero_grad()
                                    with torch.enable_grad():
                                        loss = iter_local_ft_loss(Alinear, Blinear, W, bsz, weight, __in, args, quantizers, idx,  keys='up', R=R if args.qat_optim_R else None)
                                        optimizers['up'].step()
                                    if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                                        schedulers['up'].step()
                                        if _ % getattr(args, 'scheduler_step_size', 30) == 0:
                                            print(f"Layer {idx} MLP Up - Iter {_}, Loss: {loss.item():.6f}, LR: {optimizers['up'].param_groups[0]['lr']:.6f}")
                                    loss_curve.append(loss.item())
                                    loss_dict[f'layer{idx}']['up'].append(loss.item())
                                if qat_start_iter is not None and qat_start_iter >= args.localft_iters and _ == args.localft_iters - 1:
                                    args.is_quant_aware_ft = True
                            if Blinear is None:
                                up_dtype = up_linear.weight.dtype
                                up_device = up_linear.weight.device
                                quantizers[idx]['up']['Alinear'].find_params(Alinear.weight.data)
                                up_linear.weight.data = quantizers[idx]['up']['Alinear'].quantize(Alinear.weight.data).to(up_dtype).to(up_device)
                                print(f"Layer {idx} MLP Up SVD Local FT completed, Sigma shape: {S[0].shape if S is not None else 'None'}")
                                if not args.weighted_none_svd_qat:
                                    args.weighted_svd = True
                                if args.qat_L_off:
                                    args.localft_iters = temp_localft_iters
                            else:
                                # store SVD results
                                up_linear.svd_info = {
                                    'U': Alinear.weight.data.cpu(),
                                    'S': None,
                                    'V': Blinear.weight.data.cpu()
                                }
                                print(f"Layer {idx} MLP Up SVD Local FT completed, Sigma shape: {S.shape}")
                        except Exception as e:
                            print(f"Layer {idx} MLP Up SVD Local FT failed: {e}")
                            import traceback
                            traceback.print_exc()
                        try:
                            W = gate_linear.weight.data.float().to(device)
                            w = W.clone()
                            if not hasattr(gate_linear, "svd_info"):
                                try:
                                    U = gate_linear.svd_info_before_rot['U']
                                    S = gate_linear.svd_info_before_rot['S']
                                    V = gate_linear.svd_info_before_rot['V']
                                except:
                                    U = gate_linear.weight.data
                                    S = V = None

                                #### A Blinear initialization
                                Alinear, Blinear, R = create_ABlinear_and_quantizer(U, S, V, args, quantizers, idx, keys='gate')
                                
                            W = w.to(torch.bfloat16)

                            if args.svd_ft_mode == 'output':
                                __in = activations.pop(f"block_{idx}.mlpgaup_in").to(device).to(torch.bfloat16)
                            else:
                                __in = None

                            if args.weighted_svd:
                                if Blinear is None and not args.weighted_none_svd_qat:
                                    args.weighted_svd = False
                                    weight = None
                                    print("skip weighted svd")
                                else:
                                    weight = layer.mlp.up_proj.W_grad_info.to(device).to(torch.bfloat16)
                            params_R = None
                            try:
                                params_ = [Alinear.weight, Blinear.weight]
                            except:
                                params_ = [Alinear.weight]
                                if args.qat_uv_reg:
                                    reg_UV = [W]
                            optimizers, schedulers = create_optimizer(params_, params_R, args, optimizers, schedulers, key='gate')
                            
                            if not args.qat_L_off or Blinear is not None:
                                if f'layer{idx}' not in loss_dict:
                                    loss_dict[f'layer{idx}'] = {}
                                if 'gate' not in loss_dict[f'layer{idx}']:
                                    loss_dict[f'layer{idx}']['gate'] = []
                            else:
                                temp_localft_iters = args.localft_iters
                                args.localft_iters = 0
                                logging.info(f"Layer {idx} Gateproj Local FT disabled due to qat_L_off or Blinear is not None")
                            if args.is_quant_aware_ft:
                                qat_start_iter = int(args.localft_iters * args.qat_start_iter)
                            else:
                                qat_start_iter = None
                            for _ in range(args.localft_iters):
                                if qat_start_iter is not None:
                                    # add extra criteria to control qat start iter
                                    if _ < qat_start_iter:
                                        args.is_quant_aware_ft = False
                                    elif _ == qat_start_iter:
                                        if args.qat_uv_reg:
                                            reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]
                                        #     update_qat_params(Alinear, Blinear, args, quantizers, idx, keys='q', num_heads=num_heads if args.is_per_head_svd else None)
                                        args.is_quant_aware_ft = True
                                        disable_grad_for_ABlinear(Alinear, Blinear)
                                    else: # _ > qat_start_iter
                                        args.is_quant_aware_ft = True
                                for bsz in range(0, args.nsamples, args.bs):
                                    optimizers['gate'].zero_grad()
                                    with torch.enable_grad():
                                        loss = iter_local_ft_loss(Alinear, Blinear, W, bsz, weight, __in, args, quantizers, idx,  keys='gate', R=R if args.qat_optim_R else None)
                                        optimizers['gate'].step()
                                    if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                                        schedulers['gate'].step()
                                        if _ % getattr(args, 'scheduler_step_size', 30) == 0:
                                            print(f"Layer {idx} MLP Gate - Iter {_}, Loss: {loss.item():.6f}, LR: {optimizers['gate'].param_groups[0]['lr']:.6f}")
                                    loss_curve.append(loss.item())
                                    loss_dict[f'layer{idx}']['gate'].append(loss.item())
                                if qat_start_iter is not None and qat_start_iter >= args.localft_iters and _ == args.localft_iters - 1:
                                    args.is_quant_aware_ft = True
                            
                            if Blinear is None:
                                gate_dtype = gate_linear.weight.dtype
                                gate_device = gate_linear.weight.device
                                gate_linear.weight.data = quantizers[idx]['gate']['Alinear'].quantize(Alinear.weight.data).to(gate_dtype).to(gate_device)
                                print(f"Layer {idx} MLP Gate SVD Local FT completed, Sigma shape: {S[0].shape if S is not None else 'None'}")
                                if not args.weighted_none_svd_qat:
                                    args.weighted_svd = True
                                if args.qat_L_off:
                                    args.localft_iters = temp_localft_iters
                            else:
                                # store SVD results
                                gate_linear.svd_info = {
                                    'U': Alinear.weight.data.cpu(),
                                    'S': None,
                                    'V': Blinear.weight.data.cpu()
                                }
                                print(f"Layer {idx} MLP Gate SVD Local FT completed, Sigma shape: {S.shape}")
                        except Exception as e:
                            print(f"Layer {idx} MLP Gate SVD Local FT failed: {e}")
                            import traceback
                            traceback.print_exc()
                if args.svd_modules in ['all', 'mlp', 'down']:
                    down_linear = layer.mlp.down_proj
                    try:
                        W = down_linear.weight.data.float().to(device)
                        w = W.clone()
                        if not hasattr(down_linear, "svd_info"):
                            try:
                                U = down_linear.svd_info_before_rot['U']
                                S = down_linear.svd_info_before_rot['S']
                                V = down_linear.svd_info_before_rot['V']
                            except:
                                U = down_linear.weight.data
                                S = V = None
                            
                            #### A Blinear initialization
                            Alinear, Blinear, R = create_ABlinear_and_quantizer(U, S, V, args, quantizers, idx, keys='down')
                            
                        W = w.to(torch.bfloat16)
                        
                        if args.svd_ft_mode == 'output':
                            __in = activations.pop(f"block_{idx}.mlpdown_in").to(device).to(torch.bfloat16)
                        else:
                            __in = None

                        if args.weighted_svd:
                            if Blinear is None and not args.weighted_none_svd_qat:
                                args.weighted_svd = False
                                weight = None
                                print("skip weighted svd")
                            else:
                                weight = layer.mlp.down_proj.W_grad_info.to(device).to(torch.bfloat16)
                        
                        try:
                            if args.is_per_head_svd:
                                params_ = list(Alinear.parameters()) + list(Blinear.parameters())
                                if args.qat_uv_reg:
                                    reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]
                            else:
                                params_ = [Alinear.weight, Blinear.weight]
                        except:
                            params_ = [Alinear.weight]
                            if args.qat_uv_reg:
                                    reg_UV = [W]                        
                        optimizers, schedulers = create_optimizer(params_, params_R, args, optimizers, schedulers, key='down')
                        
                        if not args.qat_L_off or Blinear is not None:
                            if f'layer{idx}' not in loss_dict:
                                loss_dict[f'layer{idx}'] = {}
                            if 'down' not in loss_dict[f'layer{idx}']:
                                loss_dict[f'layer{idx}']['down'] = []
                        else:
                            temp_localft_iters = args.localft_iters
                            args.localft_iters = 0
                            logging.info(f"Layer {idx} Downproj Local FT disabled due to qat_L_off or Blinear is not None")
                        if args.is_quant_aware_ft:
                            qat_start_iter = int(args.localft_iters * args.qat_start_iter)
                        else:
                            qat_start_iter = None
                        for _ in range(args.localft_iters):
                            if qat_start_iter is not None:
                                # add extra criteria to control qat start iter
                                if _ < qat_start_iter:
                                    args.is_quant_aware_ft = False
                                elif _ == qat_start_iter:
                                    if args.qat_uv_reg:
                                        reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]
                                    #     update_qat_params(Alinear, Blinear, args, quantizers, idx, keys='q', num_heads=num_heads if args.is_per_head_svd else None)
                                    args.is_quant_aware_ft = True
                                    disable_grad_for_ABlinear(Alinear, Blinear)
                                else:
                                    args.is_quant_aware_ft = True
                            for bsz in range(0, args.nsamples, args.bs):
                                optimizers['down'].zero_grad()
                                with torch.enable_grad():
                                    loss = iter_local_ft_loss(Alinear, Blinear, W, bsz, weight, __in, args, quantizers, idx,  keys='down', R=R if args.qat_optim_R else None)
                                    optimizers['down'].step()
                                if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                                    schedulers['down'].step()
                                    if _ % getattr(args, 'scheduler_step_size', 30) == 0:
                                        print(f"Layer {idx} Down - Iter {_}, Loss: {loss.item():.6f}, LR: {optimizers['down'].param_groups[0]['lr']:.6f}")
                                loss_curve.append(loss.item())
                                loss_dict[f'layer{idx}']['down'].append(loss.item())
                            if qat_start_iter is not None and qat_start_iter >= args.localft_iters and _ == args.localft_iters - 1:
                                args.is_quant_aware_ft = True
                        
                        if Blinear is None:
                            down_dtype = down_linear.weight.dtype
                            down_device = down_linear.weight.device
                            quantizers[idx]['down']['Alinear'].find_params(Alinear.weight.data)
                            down_linear.weight.data = quantizers[idx]['down']['Alinear'].quantize(Alinear.weight.data).to(down_dtype).to(down_device)
                            print(f"Layer {idx} MLP Down SVD Local FT completed, Sigma shape: {S[0].shape if S is not None else 'None'}")
                            if not args.weighted_none_svd_qat:
                                args.weighted_svd = True
                            if args.qat_L_off:
                                args.localft_iters = temp_localft_iters
                        else:
                            # store SVD results
                            down_linear.svd_info = {
                                'U': Alinear.weight.data.cpu(),
                                'S': None,
                                'V': Blinear.weight.data.cpu()
                            }
                            print(f"Layer {idx} MLP Down SVD Local FT completed, Sigma shape: {S.shape}")

                    except Exception as e:
                        print(f"Layer {idx} MLP Down SVD Local FT failed: {e}")
                        import traceback
                        traceback.print_exc()
                if args.svd_modules not in ['qkv', 'attn', 'all', 'o', 'mlp', 'gaup', 'down']:
                    raise ValueError(f"Invalid svd_modules: {args.svd_modules}")
                if args.is_quant_aware_ft and not args.is_stage:
                    args.svd_modules = temp_svd_modules
                if args.svd_ft_mode == 'output':
                    in_outs['inp'] = in_outs['out'] # update input to next layer
            if args.svd_ft_mode == 'output':
                for h in hooks:
                    h.remove()
                del activations
                del hooks
                del optimizers
                del in_outs
        bsz_counter += 1
    def save_plot(loss_curve, name=None):
        import matplotlib.pyplot as plt
        import os
        save_path = args.save_path + '/save_loss/'
        os.makedirs(save_path, exist_ok=True)
        plt.figure(figsize=(10, 5))
        iters = [_ for _ in range(len(loss_curve))]
        plt.plot(iters, loss_curve, label = 'loss', linewidth=2)
        plt.xlabel("iters")
        plt.ylabel("loss")
        plt.title(f"Loss curve-loss")
        plt.savefig(f'{save_path}_Iter{args.localft_iters}_Lr{args.localft_lr}_{name}.png')
        # plt.savefig(f'{save_path}lr_{lr}_ep_{num_epochs}sgd_{name}.png')
        plt.close()
        num_layeres = len(loss_dict.keys())
        for i in [0, 1, 15, 30, 31]: # [0, 1, 5, 10, 15, 25, 30, 31] # shrink layer to save png to save disk memory, there is too many pngs.
            for key in loss_dict[f'layer{i}'].keys():
                plt.figure(figsize=(10, 5))
                iters = [_ for _ in range(len(loss_dict[f'layer{i}'][key]))]
                plt.plot(iters, loss_dict[f'layer{i}'][key], label = f'layer{i}', linewidth=2)
                plt.xlabel("iters")
                plt.ylabel("loss")
                plt.title(f"Loss curve-layer{i}-{key}")
                plt.savefig(f'{save_path}/layer{i}_{key}_loss.png')
                plt.close()
        # TODO: add args to decide whether to save loss_dict
        # torch.save(loss_dict, f'{save_path}/loss_dict.pth')
    # del optimizers
    if norm_loss:
        save_plot(loss_curve, 'local_ft_grad_normloss')
    else:
        save_plot(loss_curve, 'local_ft_grad')


def register_hook_to_linear_layer(model, args):
    """
    Register hook to catch input and output of qkv or target linear layer
    """
    # [FIXME:]add catcher for concat version
    
    def forward_hook(name, in_key=None, out_key=None, activations={}):
        def hook(module, input, output):
            if in_key is not None:
                if name+in_key in activations.keys():
                    activations[name+in_key] = torch.cat([activations[name+in_key], input[0].detach().clone().cpu()], dim=0)
                else:
                    activations[name+in_key]=input[0].detach().clone().cpu()
            if out_key is not None:
                if name+out_key in activations.keys():
                    activations[name+out_key] = torch.cat([activations[name+out_key], output.detach().clone().cpu()], dim=0)
                else:
                    activations[name+out_key]=output.detach().clone().cpu()
        return hook
    hooks = []
    activations = {}
    def save_and_register(layer, name, *args, **kwargs):
        hook = layer.register_forward_hook(forward_hook(name, *args, activations=activations, **kwargs))
        hooks.append(hook)
    
    for idx, layer in enumerate(model_utils.get_layers(model)):
        if args.svd_modules in ['qkv', 'attn', 'all']:
            if args.qkv_fuse:
                save_and_register(layer.self_attn.q_proj, f"block_{idx}.attn", in_key='qkv_in')
            else:
                save_and_register(layer.self_attn.q_proj, f"block_{idx}.attn", in_key='q_in') 
                save_and_register(layer.self_attn.k_proj, f"block_{idx}.attn", in_key='k_in') 
                save_and_register(layer.self_attn.v_proj, f"block_{idx}.attn", in_key='v_in')
            # save_and_register(layer.self_attn.q_proj, f"block_{idx}.attn", in_key='qkv_in', out_key='q_proj')
            # save_and_register(layer.self_attn.k_proj, f"block_{idx}.attn", out_key='k_proj')
            # save_and_register(layer.self_attn.v_proj, f"block_{idx}.attn", out_key='v_proj')
        if args.svd_modules in ['attn', 'all', 'o']:
            save_and_register(layer.self_attn.o_proj, f"block_{idx}.attn", in_key='o_in')
            # save_and_register(layer.self_attn.o_proj, f"block_{idx}.attn", in_key='o_in', out_key='o_proj')
        if args.svd_modules in ['all', 'mlp', 'gaup']:
            save_and_register(layer.mlp.up_proj, f"block_{idx}.mlp", in_key='gaup_in')
            # save_and_register(layer.mlp.up_proj, f"block_{idx}.mlp", in_key='gaup_in', out_key='up_proj')
            # save_and_register(layer.mlp.gate_proj, f"block_{idx}.mlp", out_key='gate_proj')
        if args.svd_modules in ['all', 'mlp', 'down']:
            save_and_register(layer.mlp.down_proj, f"block_{idx}.mlp", in_key='down_in')
            # save_and_register(layer.mlp.down_proj, f"block_{idx}.mlp", in_key='down_in', out_key='down_proj')
    return hooks, activations

def set_grad_for_svd_modules(model, args):
    model.train()
    for name, param in model.named_parameters():
        if 'model.layers' in name:
            if ('q_proj' in name or 'k_proj' in name or 'v_proj' in name) and args.svd_modules in ['all', 'qkv', 'attn']:
                param.requires_grad = True
            elif 'o_proj' in name and args.svd_modules in ['attn', 'all', 'o']:
                param.requires_grad = True
            elif ('up_proj' in name or 'gate_proj' in name) and args.svd_modules in ['all', 'mlp', 'gaup']:
                param.requires_grad = True
            elif 'down_proj' in name and args.svd_modules in ['all', 'mlp', 'down']:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False

def disable_grad_for_svd_modules(model, args):
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False
        
def disable_grad_for_ABlinear(Alinear, Blinear):
    for i, param in enumerate(Alinear):
        param.requires_grad = False
    for i, param in enumerate(Blinear):
        param.requires_grad = False

def save_svd_info(model, args):
    qkv_svd_info = {}
    for idx, layer in enumerate(model_utils.get_layers(model)):
        if args.svd_modules in ['qkv', 'attn', 'all', 'all_sep_fine', 'all_sep', 'all_sep_fine_wo_down', 'all_sep_rank_allocate_qkvonly']:
            if args.qkv_fuse:
                if 'k_proj' not in qkv_svd_info:
                    qkv_svd_info['k_proj'] = {}
                if hasattr(layer.self_attn.k_proj, 'qkv_svd_info'):
                    logging.info(f"Layer {idx} k_proj svd info saved")
                    qkv_svd_info['k_proj'][f"layer_{idx}"] = layer.self_attn.k_proj.qkv_svd_info
            else:
                if 'q_proj' not in qkv_svd_info:
                    qkv_svd_info['q_proj'] = {}
                if 'k_proj' not in qkv_svd_info:
                    qkv_svd_info['k_proj'] = {}
                if 'v_proj' not in qkv_svd_info:
                    qkv_svd_info['v_proj'] = {}
                if hasattr(layer.self_attn.q_proj, 'svd_info'):
                    logging.info(f"Layer {idx} q_proj svd info saved")
                    qkv_svd_info['q_proj'][f"layer_{idx}"] = layer.self_attn.q_proj.svd_info
                if hasattr(layer.self_attn.k_proj, 'svd_info'):
                    logging.info(f"Layer {idx} k_proj svd info saved")
                    qkv_svd_info['k_proj'][f"layer_{idx}"] = layer.self_attn.k_proj.svd_info
                if hasattr(layer.self_attn.v_proj, 'svd_info'):
                    logging.info(f"Layer {idx} v_proj svd info saved")
                    qkv_svd_info['v_proj'][f"layer_{idx}"] = layer.self_attn.v_proj.svd_info
        if args.svd_modules in ['attn', 'all', 'o', 'all_sep_fine', 'all_sep', 'all_sep_fine_wo_down', 'all_sep_rank_allocate_qkvonly']:
            if 'o_proj' not in qkv_svd_info:
                qkv_svd_info['o_proj'] = {}
            if hasattr(layer.self_attn.o_proj, 'svd_info'):
                logging.info(f"Layer {idx} o_proj svd info saved")
                qkv_svd_info['o_proj'][f"layer_{idx}"] = layer.self_attn.o_proj.svd_info
        if args.svd_modules in ['all', 'mlp', 'gaup', 'all_sep_fine', 'all_sep', 'all_sep_fine_wo_down', 'all_sep_rank_allocate_qkvonly']:
            if args.mlp_fuse:
                if 'up_proj' not in qkv_svd_info:
                    qkv_svd_info['up_proj'] = {}
                if hasattr(layer.mlp.up_proj, 'svd_info'):
                    logging.info(f"Layer {idx} up_proj svd info saved")
                    qkv_svd_info['up_proj'][f"layer_{idx}"] = layer.mlp.up_proj.svd_info
            else:
                if 'up_proj' not in qkv_svd_info:
                    qkv_svd_info['up_proj'] = {}
                if 'gate_proj' not in qkv_svd_info:
                    qkv_svd_info['gate_proj'] = {}
                if hasattr(layer.mlp.up_proj, 'svd_info'):
                    logging.info(f"Layer {idx} up_proj svd info saved")
                    qkv_svd_info['up_proj'][f"layer_{idx}"] = layer.mlp.up_proj.svd_info
                if hasattr(layer.mlp.gate_proj, 'svd_info'):
                    logging.info(f"Layer {idx} gate_proj svd info saved")
                    qkv_svd_info['gate_proj'][f"layer_{idx}"] = layer.mlp.gate_proj.svd_info
        if args.svd_modules in ['all', 'mlp', 'down', 'all_sep_fine', 'all_sep', 'all_sep_fine_wo_down', 'all_sep_rank_allocate_qkvonly']:
            if 'down_proj' not in qkv_svd_info:
                qkv_svd_info['down_proj'] = {}
            if hasattr(layer.mlp.down_proj, 'svd_info'):
                logging.info(f"Layer {idx} down_proj svd info saved")
                qkv_svd_info['down_proj'][f"layer_{idx}"] = layer.mlp.down_proj.svd_info
    if args.is_rank_allocate_ft:
        torch.save(qkv_svd_info, f'{args.save_path}/qkv_svd_ft_info_rank_allocate.pth')
    else:
        torch.save(qkv_svd_info, f'{args.save_path}/qkv_svd_ft_info.pth')

def load_svd_info(model, args):
    def safe_load(path, args):
        import torch
        from torch.nn.modules.container import ParameterList
        from quant_utils import WeightQuantizer
        allowed = [ParameterList, WeightQuantizer]
        if args.is_per_head_svd:
            with torch.serialization.safe_globals(allowed):
                return torch.load(path)
        else:
            return torch.load(path)
        
    if args.cache_file is not None:
        load_path = args.cache_file
    else:
        load_path = args.save_path
    logging.info(f"Loading svd info from {load_path}")
    if args.is_rank_allocate_ft:
        qkv_svd_info = safe_load(f'{load_path}/qkv_svd_ft_info_rank_allocate.pth', args)
        logging.info(f"Rank allocate svd info loaded from {load_path}/qkv_svd_ft_info_rank_allocate.pth")
    else:
        qkv_svd_info = safe_load(f'{load_path}/qkv_svd_ft_info.pth', args)
        logging.info(f"Svd info loaded from {load_path}/qkv_svd_ft_info.pth")
    for idx, layer in enumerate(model_utils.get_layers(model)):
        if args.svd_modules in ['qkv', 'attn', 'all', 'all_sep_fine', 'all_sep', 'all_sep_fine_wo_down', 'all_sep_rank_allocate_qkvonly']:
            if args.qkv_fuse:
                try:
                    layer.self_attn.k_proj.qkv_svd_info = qkv_svd_info['k_proj'][f"layer_{idx}"]
                    print(f"Layer {idx} k_proj svd info loaded")
                except:
                    print(f"Layer {idx} k_proj svd info not found")
            else:
                try:
                    layer.self_attn.q_proj.svd_info = qkv_svd_info['q_proj'][f"layer_{idx}"]
                    layer.self_attn.k_proj.svd_info = qkv_svd_info['k_proj'][f"layer_{idx}"]
                    layer.self_attn.v_proj.svd_info = qkv_svd_info['v_proj'][f"layer_{idx}"]
                    print(f"Layer {idx} q_proj svd info loaded")
                    print(f"Layer {idx} k_proj svd info loaded")
                    print(f"Layer {idx} v_proj svd info loaded")
                except:
                    print(f"Layer {idx} q_proj svd info not found")
                    print(f"Layer {idx} k_proj svd info not found")
                    print(f"Layer {idx} v_proj svd info not found")
        if args.svd_modules in ['o', 'attn', 'all', 'all_sep_fine', 'all_sep', 'all_sep_fine_wo_down', 'all_sep_rank_allocate_qkvonly']:
            try:
                layer.self_attn.o_proj.svd_info = qkv_svd_info['o_proj'][f"layer_{idx}"]
                print(f"Layer {idx} o_proj svd info loaded")
            except:
                print(f"Layer {idx} o_proj svd info not found")
        if args.svd_modules in ['all', 'mlp', 'gaup', 'all_sep_fine', 'all_sep', 'all_sep_fine_wo_down', 'all_sep_rank_allocate_qkvonly']:
            if args.mlp_fuse:
                try:
                    layer.mlp.up_proj.svd_info = qkv_svd_info['up_proj'][f"layer_{idx}"]
                    print(f"Layer {idx} gaup svd info loaded")
                except:
                    print(f"Layer {idx} gaup svd info not found")
            else:
                try:
                    layer.mlp.up_proj.svd_info = qkv_svd_info['up_proj'][f"layer_{idx}"]
                    layer.mlp.gate_proj.svd_info = qkv_svd_info['gate_proj'][f"layer_{idx}"]
                    print(f"Layer {idx} up_proj svd info loaded")
                    print(f"Layer {idx} gate_proj svd info loaded")
                except:
                    print(f"Layer {idx} up_proj svd info not found")
                    print(f"Layer {idx} gate_proj svd info not found")
        if args.svd_modules in ['all', 'mlp', 'down', 'all_sep_fine', 'all_sep', 'all_sep_fine_wo_down', 'all_sep_rank_allocate_qkvonly']:
            try:
                layer.mlp.down_proj.svd_info = qkv_svd_info['down_proj'][f"layer_{idx}"]
                print(f"Layer {idx} down_proj svd info loaded")
            except:
                print(f"Layer {idx} down_proj svd info not found")
    
def smooth_gradient_info(grad_info, args, layer_idx=None):
    """
    Smooth gradient information to eliminate abnormal spikes
    
    This function provides multiple smoothing strategies to handle outliers in gradient information:
    
    1. Top-k smoothing: Replace top-k largest absolute values
    2. Percentile-based smoothing: Replace values above a certain percentile
    3. Statistical outlier detection: Replace values beyond N standard deviations
    
    Args:
        grad_info: Gradient information tensor
        args: Arguments containing smoothing parameters
    
    Smoothing Parameters (all optional):
        - smooth_grad (bool): Enable/disable smoothing (default: False)
        - smooth_method (str): Smoothing method ('mean', 'zero', 'percentile', 'outlier')
        - smooth_topk_ratio (float): Ratio of top values to smooth (default: 0.01)
        - smooth_percentile (float): Percentile threshold for percentile method (default: 99.0)
        - smooth_outlier_std (float): Standard deviation threshold for outlier method (default: 3.0)
        - smooth_percentile_replacement (str): Replacement for percentile method ('zero' or 'mean')
        - smooth_outlier_replacement (str): Replacement for outlier method ('zero' or 'mean')
        - smooth_final (bool): Apply smoothing at final normalization step
    
    Examples:
        # Enable top-k smoothing with mean replacement
        args.smooth_grad = True
        args.smooth_method = 'mean'
        args.smooth_topk_ratio = 0.01
        
        # Enable percentile-based smoothing
        args.smooth_grad = True
        args.smooth_method = 'percentile'
        args.smooth_percentile = 99.5
        
        # Enable statistical outlier detection
        args.smooth_grad = True
        args.smooth_method = 'outlier'
        args.smooth_outlier_std = 2.5
        
        # Apply smoothing only at final step
        args.smooth_final = True
        args.smooth_method = 'percentile'
    
    Returns:
        Smoothed gradient information tensor
    """
    if not hasattr(args, 'smooth_grad') or not args.smooth_grad:
        return grad_info
    if args.smooth_ma_only:
        if layer_idx not in [1, 31]:
            logging.info(f"Layer {layer_idx} is not MA layer, skip smoothing")
            return grad_info
    # Get smoothing parameters
    topk_ratio = getattr(args, 'topk', 10)  # Default: smooth top 1%
    smooth_method = getattr(args, 'smooth_method', 'mean')  # 'mean', 'zero', 'percentile', 'outlier'
    percentile_threshold = getattr(args, 'smooth_percentile', 99.0)  # For percentile method
    outlier_std_threshold = getattr(args, 'smooth_outlier_std', 3.0)  # For outlier method
    power_threshold = getattr(args, 'smooth_power_threshold', 1.0)  # For power scale method
    
    # Flatten the tensor for processing
    original_shape = grad_info.shape
    grad_flat = grad_info.flatten()
    
    if percentile_threshold > 0:
        # Use percentile-based smoothing
        threshold = torch.quantile(grad_flat.abs(), percentile_threshold / 100.0)
        mask = grad_flat.abs() > threshold
        if mask.sum() > 0:
            if args.smooth_method == 'zero':
                grad_flat[mask] = 0.0
            else:
                # Replace with mean of non-outlier values
                replacement_value = grad_flat[~mask].mean()
                grad_flat[mask] = replacement_value
    
    elif outlier_std_threshold > 0:
        # Use statistical outlier detection (larger than some mean + threshold * std)
        mean_val = grad_flat.mean()
        std_val = grad_flat.std()
        threshold = mean_val + outlier_std_threshold * std_val
        mask = grad_flat.abs() > threshold
        if mask.sum() > 0:
            if args.smooth_method == 'zero':
                grad_flat[mask] = 0.0
            else:
                # Replace with mean of non-outlier values
                replacement_value = grad_flat[~mask].mean()
                grad_flat[mask] = replacement_value
        logging.info(f"smooth_outlier_std: {outlier_std_threshold} replaced {mask.sum()} values")
    elif power_threshold > 0:
        # Use power scale method
        grad_flat = grad_flat**power_threshold
        logging.info(f"smooth_power_threshold: {power_threshold} scaled grad info")
    else:
        # Original topk-based smoothing
        num_elements = grad_flat.numel()
        num_to_smooth = max(1, int(num_elements * topk_ratio))
        
        # Find top-k values (largest absolute values)
        if num_to_smooth < num_elements:
            # Get indices of top-k largest absolute values
            _, topk_indices = torch.topk(grad_flat.abs(), k=num_to_smooth, largest=True)
            
            # Calculate replacement value
            if smooth_method == 'mean':
                # Calculate mean of non-topk values
                mask = torch.ones_like(grad_flat, dtype=torch.bool)
                mask[topk_indices] = False
                replacement_value = grad_flat[mask].mean()
            elif smooth_method == 'zero':
                replacement_value = 0.0
            else:
                raise ValueError(f"Invalid smooth_method: {smooth_method}. Supported: 'mean', 'zero'")
            
            # Replace top-k values
            grad_flat[topk_indices] = replacement_value
        logging.info(f"smooth_topk_ratio: {topk_ratio} replaced {topk_indices.sum()} values")
    # Reshape back to original shape
    return grad_flat.reshape(original_shape)

def get_layer_importance(grad, args):
    if args.smooth_grad and not args.smooth_final:
        grad = smooth_gradient_info(grad, args)
    if args.taylor_order == 1:
        return -1.0 * grad
    elif args.taylor_order == 2:
        return grad.pow(2)
    elif args.taylor_order == 12:
        return -2.0 * grad + grad.pow(2)

def register_gradient_hook_to_linear_layer(model, args):
    """
    Register backward hooks to catch output gradients ∂L/∂y of linear layers
    Similar to register_hook_to_linear_layer but captures gradients instead of activations
    """
    def backward_hook(name, grad_key=None, gradients={}):
        def hook(module, grad_input, grad_output):
            # grad_output[0] contains ∂L/∂y where y is the output of the linear layer
            if grad_output is not None:
                gradients[name+grad_key] = grad_output[0].detach().clone().cpu() # for now we just capture the gradient one sample a time.
        return hook
    hooks = []
    gradients = {}
    def save_and_register(layer, name, *args, **kwargs):
        hook = layer.register_full_backward_hook(backward_hook(name, *args, gradients=gradients, **kwargs))
        hooks.append(hook)
    
    for idx, layer in enumerate(model_utils.get_layers(model)):
        if args.svd_modules in ['qkv', 'attn', 'all']:
            save_and_register(layer.self_attn.q_proj, f"block_{idx}.attn", grad_key='q_grad')# as output differs for these three
            save_and_register(layer.self_attn.k_proj, f"block_{idx}.attn", grad_key='k_grad')
            save_and_register(layer.self_attn.v_proj, f"block_{idx}.attn", grad_key='v_grad')
        if args.svd_modules in ['attn', 'all', 'o']:
            save_and_register(layer.self_attn.o_proj, f"block_{idx}.attn", grad_key='o_grad')
        if args.svd_modules in ['all', 'mlp', 'gaup']:
            save_and_register(layer.mlp.up_proj, f"block_{idx}.mlp", grad_key='up_grad')
            save_and_register(layer.mlp.gate_proj, f"block_{idx}.mlp", grad_key='gate_grad')
        if args.svd_modules in ['all', 'mlp', 'down']:
            save_and_register(layer.mlp.down_proj, f"block_{idx}.mlp", grad_key='down_grad')
            
    return hooks, gradients

def create_and_fuse_scalingmatrix_to_W(W, scaling_diag_matrix, args, linear, keys='q'):
    device = utils.get_dev()
    if hasattr(linear, "scaling_diag_matrix"):
        scaling_diag_matrix *= linear.scaling_diag_matrix.to(device)**args.act_alpha
        scaling_diag_matrix += 1e-6  # avoid zero division
        scaling_matrix_inv = None
        if args.is_per_head_svd and keys in ['q', 'k', 'v']:
            W = W * scaling_diag_matrix.view(1, 1, -1)
        else:
            W = W * scaling_diag_matrix.view(1, -1)
    elif hasattr(linear, "scaling_diag_matrixS"):
        scaling_diag_matrix = linear.scaling_diag_matrixS.to(device)
        try:
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
        except RuntimeError as e:
            logging.info("Warning: scaling_diag_matrix is not full rank, adding epsilon for stability.")
            eps = 1e-6
            scaling_diag_matrix += eps * torch.eye(scaling_diag_matrix.shape[0], device=device)
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
        W = W @ scaling_diag_matrix.float()
    else:
        scaling_matrix_inv = None
        raise ValueError("No scaling_diag_matrix found")
    return W, scaling_diag_matrix, scaling_matrix_inv

def create_svd_decomposition(W, args, layer_indices_dict, linear, idx, num_heads=None, keys='q_proj'):
    U, S, Vt = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
    if args.is_per_head_svd and keys in ['q_proj', 'k_proj', 'v_proj']:
        V = Vt.transpose(1, 2) # n, C_in, c
        n_params = linear.weight.numel()
        compressed_params = int(n_params/num_heads * args.rank_ratio)
        LocalFT_rank = compressed_params // (linear.in_features + linear.out_features/num_heads) # update rank calculation
        LocalFT_rank = int(np.ceil(LocalFT_rank/1.0)*1.0)
        if layer_indices_dict is not None:
            U_ = []
            S_ = []
            V_ = []
            LocalFT_rank = layer_indices_dict[idx][keys]
            for i in range(num_heads):
                try:
                    rank_ = LocalFT_rank[i]
                except:
                    rank_ = [1]
                U_.append(U[i, :, rank_])
                S_.append(S[i, rank_])
                V_.append(V[i, :, rank_])
            U = U_
            S = S_
            V = V_
            del U_, S_, V_
        else:
            # U: n, c, c(r)
            # S: n, c(r)
            # V:  n, C_in, c(r); 
            # print(f"U shape: {U.shape}, S shape: {S.shape}, V shape: {V.shape}")
            U = U[:, :, :LocalFT_rank]
            S = S[:, :LocalFT_rank]
            V = V[:, :, :LocalFT_rank]

    else:
        V = Vt.T
        # truncate rank:
        n_params = linear.weight.numel()
        compressed_params = int(n_params * args.rank_ratio)
        LocalFT_rank = compressed_params // (linear.in_features + linear.out_features) # update rank calculation
        if layer_indices_dict is not None:
            LocalFT_rank = layer_indices_dict[idx][keys]
            
            U = U[:, LocalFT_rank]
            S = S[LocalFT_rank]
            V = V[:, LocalFT_rank]
        else:
            U = U[:, :LocalFT_rank]
            S = S[:LocalFT_rank]
            V = V[:, :LocalFT_rank]
    
    return U, S, V

def fuse_invariance_S_to_V(V, scaling_matrix_inv, scaling_diag_matrix, args, keys='q', layer_indices_dict=None, num_heads=None):
    if scaling_matrix_inv is not None:
        if args.is_per_head_svd and keys in ['q', 'k', 'v']:
            if layer_indices_dict is not None:
                V = [scaling_matrix_inv.T @ vv for vv in V]
            else:
                V = scaling_matrix_inv.T.float().unsqueeze(0).expand(num_heads, -1, -1) @ V # V: n, C_in, rank, scaling_matrix_inv: C_in, C_in
        else:
            V = scaling_matrix_inv.T @ V
    else:
        if args.is_per_head_svd and keys in ['q', 'k', 'v']:
            if layer_indices_dict is not None:
                V = [vv / scaling_diag_matrix.view(-1, 1) for vv in V]
            else:
                V = V/ scaling_diag_matrix.view(1, -1, 1)  # V: n, C_in, rank, scaling_diag_matrix: C_in, 1
        else:
            V = V / scaling_diag_matrix.view(-1, 1)
    return V

def create_ABlinear_and_quantizer(U, S, V, args, quantizers, idx, keys='q', num_heads=None):
    device = utils.get_dev()
    if S is None:
        Alinear = nn.Linear(U.size(1), U.size(0), bias=False, device=device)
        Alinear.weight.requires_grad = True
        Blinear = None
        with torch.no_grad():
            Alinear.weight.data.copy_(U.to(device).to(torch.bfloat16))
        if args.is_quant_aware_ft:
            if keys not in quantizers[idx]:
                quantizers[idx][keys] = {}
            if 'Alinear' not in quantizers[idx][keys]:
                quantizers[idx][keys]['Alinear'] = quant_utils.WeightQuantizer()
                quantizers[idx][keys]['Alinear'].configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
                quantizers[idx][keys]['Alinear'].find_params(Alinear.weight.data)
        Alinear.requires_grad = True
        return Alinear, Blinear, None

    if args.is_per_head_svd and keys in ['q', 'k', 'v']:
        Alinear = nn.ParameterList()
        Blinear = nn.ParameterList()
        R = nn.ParameterList()
        if args.is_quant_aware_ft:
            if keys not in quantizers[idx]:
                quantizers[idx][keys] = {}
            if 'Alinear' not in quantizers[idx][keys]:
                quantizers[idx][keys]['Alinear'] = []
                quantizers[idx][keys]['Blinear'] = []
        
        for i in range(num_heads):
            if args.rotate:
                rank_ = S[i].shape[-1]
                had_K = rotation_utils.get_orthogonal_matrix(rank_, 'random', args.seed) # do not think much now
                R.append(nn.Parameter(had_K.to(torch.bfloat16).to(device), requires_grad=True))
            Alinear.append(nn.Parameter(U[i].mul(S[i].sqrt()).contiguous().to(torch.bfloat16).to(device), requires_grad=True))
            Blinear.append(nn.Parameter(V[i].t().mul(S[i].sqrt().view(-1, 1)).contiguous().to(torch.bfloat16).to(device), requires_grad=True))

            if args.is_quant_aware_ft:
                if len(quantizers[idx][keys]['Alinear']) < num_heads:
                    quantizers[idx][keys]['Alinear'].append(quant_utils.WeightQuantizer())
                    quantizers[idx][keys]['Blinear'].append(quant_utils.WeightQuantizer())
                    quantizers[idx][keys]['Alinear'][-1].configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
                    quantizers[idx][keys]['Blinear'][-1].configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
                    # if idx == 0:
                    #     quantizers[idx][keys]['act_lat'].append(quant_utils.ActQuantizer())
                    #     quantizers[idx][keys]['act_lat'][-1].configure(args.a_bits, groupsize=-1, sym=not(args.a_asym), clip_ratio=args.a_clip_ratio)
                    
    else:
        Alinear = nn.Linear(U.size(1), U.size(0), bias=False, device=device)
        Blinear = nn.Linear(V.size(0), V.size(1), bias=False, device=device) # V: C_in, Rank
        R = None
        # Set requires_grad before assigning data
        Alinear.weight.requires_grad = True
        Blinear.weight.requires_grad = True
    
        # Use proper assignment that preserves gradients
        with torch.no_grad():
            if args.rotate:
                had_K = rotation_utils.get_orthogonal_matrix(S.shape[-1], 'random', args.seed)
                R = nn.Parameter(had_K.to(torch.bfloat16).to(device), requires_grad=True)
            Alinear.weight.data.copy_(U.mul(S.sqrt()).contiguous().to(device).to(torch.bfloat16))
            Blinear.weight.data.copy_(V.t().mul(S.sqrt().view(-1, 1)).contiguous().to(device).to(torch.bfloat16))
        if args.is_quant_aware_ft:
            if keys not in quantizers[idx]:
                quantizers[idx][keys] = {}
            if 'Alinear' not in quantizers[idx][keys]:
                quantizers[idx][keys]['Alinear'] = quant_utils.WeightQuantizer()
                quantizers[idx][keys]['Blinear'] = quant_utils.WeightQuantizer()
                quantizers[idx][keys]['Alinear'].configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
                quantizers[idx][keys]['Blinear'].configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
                # quantizers[idx][keys]['Alinear'].find_params(Alinear.weight.data)
                # quantizers[idx][keys]['Blinear'].find_params(Blinear.weight.data)
        Alinear.requires_grad = True
        Blinear.requires_grad = True

    return Alinear, Blinear, R


def update_qat_params(Alinear, Blinear, args, quantizers, idx, keys='q', num_heads=None):
    if Blinear is None:
        # qat linear is always fting
        pass
    if args.is_per_head_svd and keys in ['q', 'k', 'v']:
        for i in range(num_heads):
            quantizers[idx][keys]['Alinear'][i].find_params(Alinear[i].data)
            quantizers[idx][keys]['Blinear'][i].find_params(Blinear[i].data)
    else:
        quantizers[idx][keys]['Alinear'].find_params(Alinear.weight.data)
        quantizers[idx][keys]['Blinear'].find_params(Blinear.weight.data)

def create_optimizer(params_, params_R, args, optimizers, schedulers, key='q'):
    key_R = key + '_R'
    if args.qat_optim_R and params_R is not None:
        import optimizer_utils
        optimizers[key_R] = optimizer_utils.SGDG(params_R, lr=args.qat_lr_R, stiefel=True)
    
    if args.svd_ft_optim == 'adam':
        optimizers[key] = torch.optim.Adam(params_, lr=args.localft_lr)
    elif args.svd_ft_optim == 'sgd':
        optimizers[key] = torch.optim.SGD(params_, lr=args.localft_lr)
    elif args.svd_ft_optim == 'adamw':
        optimizers[key] = torch.optim.AdamW(params_, lr=args.localft_lr)
    elif args.svd_ft_optim == 'sgdm':
        optimizers[key] = torch.optim.SGD(params_, lr=args.localft_lr, momentum=0.9)
    else:
        raise ValueError(f"Invalid svd_ft_optim: {args.svd_ft_optim}")

    # Add learning rate scheduler for q projection
    if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
        if args.scheduler_type == 'step':
            schedulers[key] = torch.optim.lr_scheduler.StepLR(
                optimizers[key], 
                step_size=getattr(args, 'scheduler_step_size', 30), 
                gamma=getattr(args, 'scheduler_gamma', 0.8)
            )
            if args.qat_optim_R and params_R is not None:
                schedulers[key_R] = torch.optim.lr_scheduler.StepLR(
                    optimizers[key_R], 
                    step_size=getattr(args, 'scheduler_step_size', 30), 
                    gamma=getattr(args, 'scheduler_gamma', 0.8)
                )
        elif args.scheduler_type == 'cosine':
            schedulers[key] = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[key], 
                T_max=args.localft_iters * (args.nsamples // args.bs),
                eta_min=getattr(args, 'scheduler_eta_min', 1e-6)
            )
            if args.qat_optim_R and params_R is not None:
                schedulers[key_R] = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[key_R], 
                    T_max=args.localft_iters * (args.nsamples // args.bs),
                    eta_min=getattr(args, 'scheduler_eta_min', 1e-6)
                )
        else:
            raise ValueError(f"Invalid scheduler_type: {args.scheduler_type}. Supported types: 'step', 'cosine'")
    
    return optimizers, schedulers
 
def iter_local_ft_loss(Alinear, Blinear, W, bsz, weight, __in, args, quantizers, idx,  keys='q', reg_UV=None, R=None, qat_param_update=False):
    norm_loss = True
    reg = None
    # here within enable grad can abstract a function class
    if Blinear is None:
        if args.svd_ft_mode == 'output':
            _in = __in[bsz:bsz+args.bs]
            if args.weighted_svd:
                _out = (_in @ W.T) * weight
                y = (_in @ Alinear.weight) * weight
                # y = Alinear(_in) * weight
            else:
                _out = _in @ W.T
                y = _in @ Alinear.weight
        elif args.svd_ft_mode == 'weight':
            def regularization_W(Alinear, reg_UV, weight=None):
                if weight is not None:
                    loss = (weight * (Alinear - reg_UV[-1])).pow(2).mean()
                else:
                    loss = (Alinear - reg_UV[-1]).pow(2).mean()
                return loss
            if args.weighted_svd:
                _out = weight * W.detach()
                y = weight * Alinear.weight
                y = weight * (quantizers[idx][keys]['Alinear'].quantize(Alinear.weight))
                if reg_UV is not None:
                    reg = regularization_W(Alinear, reg_UV, None)
            else:
                _out = W.detach()
                y = quantizers[idx][keys]['Alinear'].quantize(Alinear.weight)
                if reg_UV is not None:
                    reg = regularization_W(Alinear.weight, reg_UV)
        else:
            raise ValueError(f"Invalid svd_ft_mode: {args.svd_ft_mode}")
    else:
        if args.svd_ft_mode == 'output':
            _in = __in[bsz:bsz+args.bs]
            if args.weighted_svd:
                _out = (_in @ W.T) * weight
                y = (_in @ (Alinear.weight @ Blinear.weight)).T * weight
            else:
                _out = _in @ W.T
                y = _in @ (Alinear.weight @ Blinear.weight).T
        elif args.svd_ft_mode == 'weight':
            if args.is_per_head_svd and keys in ['q', 'k', 'v']:
                def regularization_UV(Alinear, Blinear, reg_UV, weight):
                    loss = None
                    for A_i, B_i, A0_i, B0_i in zip(Alinear, Blinear, reg_UV[0], reg_UV[1]):
                        if loss is None:
                            loss = (A_i - A0_i).pow(2).sum() + (B_i - B0_i).pow(2).sum()
                        else:
                            loss += (A_i - A0_i).pow(2).sum() + (B_i - B0_i).pow(2).sum()
                    # else:
                    #     for A_i, B_i, A0_i, B0_i, w_i in zip(Alinear, Blinear, reg_UV[0], reg_UV[1], weight):
                    #         loss = (w_i *(A_i - A0_i)).pow(2).sum() + (w_i *(B_i - B0_i)).pow(2).sum()
                    return loss
                def per_head_low_rank_forward_W(Alinear, Blinear, Qs=None, R=None):
                    y_ = []
                    if args.is_quant_aware_ft:
                        if R is not None:
                            for A_i, B_i, Q_a_i, Q_b_i, R_i in zip(Alinear, Blinear, Qs[keys]['Alinear'], Qs[keys]['Blinear'], R):
                                A_r = A_i @ R_i
                                B_r = R_i.T @ B_i
                                if qat_param_update:
                                    Q_a_i.find_params(A_r.detach())
                                    Q_b_i.find_params(B_r.detach())
                                y_.append((Q_a_i.quantize(A_r) @ Q_b_i.quantize(B_r)).unsqueeze(0))
                        else:
                            for A_i, B_i, Q_a_i, Q_b_i in zip(Alinear, Blinear, Qs[keys]['Alinear'], Qs[keys]['Blinear']):
                                if qat_param_update:
                                    Q_a_i.find_params(A_i.detach())
                                    Q_b_i.find_params(B_i.detach())
                                y_.append((Q_a_i.quantize(A_i) @ Q_b_i.quantize(B_i)).unsqueeze(0))
                            # print("qat quantize")
                    else:
                        for A_i, B_i in zip(Alinear, Blinear):
                            y_.append((A_i @ B_i).unsqueeze(0))
                            # print("no qat quantize")
                    # print(y_[-1].shape)
                    return torch.cat(y_, dim=0)
                if args.weighted_svd:
                    _out = weight * W
                    if args.is_quant_aware_ft:
                        y = weight * per_head_low_rank_forward_W(Alinear, Blinear, Qs=quantizers[idx], R=R)
                    else:
                        y = weight * per_head_low_rank_forward_W(Alinear, Blinear)
                        # y = per_head_low_rank_forward_W(Alinear, Blinear)
                        # print(y.shape)
                        # y = weight * y
                    if reg_UV is not None and args.is_quant_aware_ft:
                        reg = regularization_UV(Alinear, Blinear, reg_UV, None)
                else:
                    _out = W
                    y = per_head_low_rank_forward_W(Alinear, Blinear)
                    if reg_UV is not None and args.is_quant_aware_ft:
                        reg = regularization_UV(Alinear, Blinear, reg_UV, None)
            else:
                if args.weighted_svd:
                    _out = weight * W
                    if args.is_quant_aware_ft:
                        y = weight * (quantizers[idx][keys]['Alinear'].quantize(Alinear.weight) @ quantizers[idx][keys]['Blinear'].quantize(Blinear.weight))
                    else:
                        y = weight * (Alinear.weight @ Blinear.weight)
                else:
                    _out = W
                    y = Alinear.weight @ Blinear.weight 
        else:
            raise ValueError(f"Invalid svd_ft_mode: {args.svd_ft_mode}")
    if reg_UV is not None and args.is_quant_aware_ft:
        if norm_loss:
            loss = (y - _out).pow(2).mean() / _out.pow(2).mean() 
            # print(loss.item()/reg.item())
            if args.qat_uv_reg_scale:
                scale = (loss.item()/(reg.item()+1e-5)) 
            # print(loss.item(), reg.item())
            # print(loss.item()/(reg.item()+1e-5))
            loss += args.qat_uv_reg_alpha * reg
        else:
            loss = (y - _out).pow(2).mean() + args.qat_uv_reg_alpha * reg
    else:
        if norm_loss:
            loss = (y - _out).pow(2).mean() / _out.pow(2).mean()
        else:
            loss = (y - _out).pow(2).mean()
    loss.backward()
    return loss


def iter_local_qat_loss(Alinear, Blinear, W, Q, bsz, weight, __in, args, quantizers, idx,  keys='q', reg_UV=None, R=None, qat_param_update=False):
    norm_loss = True
    reg = None
    # here within enable grad can abstract a function class
 
    if args.svd_ft_mode == 'output':
        raise ValueError(f"Invalid QAT mode for now: {args.svd_ft_mode}")
        # _in = __in[bsz:bsz+args.bs]
        # if args.weighted_svd:
        #     _out = (_in @ W.T) * weight
        #     y = (_in @ (Alinear.weight @ Blinear.weight)).T * weight
        # else:
        #     _out = _in @ W.T
        #     y = _in @ (Alinear.weight @ Blinear.weight).T
        
    elif args.svd_ft_mode == 'weight':
        if args.is_per_head_svd and keys in ['q', 'k', 'v']:
            W_had, Alinear_had, Blinear_had = get_had_W_and_ABlinear(W, Alinear, Blinear, Q, key=keys) # online hadamard rotation
            
            _out = W_had
            y = per_head_low_rank_forward_W(Alinear_had, Blinear_had, keys=keys, Qs=quantizers[idx], R=R, qat_param_update=qat_param_update) # further rotate in R, and quantize
            
            if reg_UV is not None and args.is_quant_aware_ft:
                reg = regularization_UV(Alinear_had, Blinear_had, reg_UV, None)
                
            if args.weighted_svd:
                _out = weight * _out.detach()
                y = weight * y
            else:
                _out = _out.detach()
                y = y
            
            # _out = _out.detach()
            # y = y
    else:
        raise ValueError(f"Invalid QAT mode: {args.svd_ft_mode}")

    if reg_UV is not None and args.is_quant_aware_ft:
        if norm_loss:
            loss = (y - _out).pow(2).mean() / _out.pow(2).mean() 
            if args.qat_uv_reg_scale:
                scale = (loss.item()/(reg.item()+1e-5)) 
            loss += args.qat_uv_reg_alpha * reg
        else:
            loss = (y - _out).pow(2).mean() + args.qat_uv_reg_alpha * reg
    else:
        if norm_loss:
            loss = (y - _out).pow(2).mean() / _out.pow(2).mean()
        else:
            loss = (y - _out).pow(2).mean()

    loss.backward()
    return loss

def regularization_UV(Alinear, Blinear, reg_UV, weight):
    loss = None
    for A_i, B_i, A0_i, B0_i in zip(Alinear, Blinear, reg_UV[0], reg_UV[1]):
        if loss is None:
            loss = (A_i - A0_i).pow(2).sum() + (B_i - B0_i).pow(2).sum()
        else:
            loss += (A_i - A0_i).pow(2).sum() + (B_i - B0_i).pow(2).sum()
    return loss

def per_head_low_rank_forward_W(Alinear, Blinear, keys='q', Qs=None, R=None, qat_param_update=True):
    y_ = []
    for A_i, B_i, Q_a_i, Q_b_i, R_i in zip(Alinear, Blinear, Qs[keys]['Alinear'], Qs[keys]['Blinear'], R):
        A_r = A_i @ R_i
        B_r = R_i.T @ B_i
        if qat_param_update:
            Q_a_i.find_params(A_r.detach())
            Q_b_i.find_params(B_r.detach())
        y_.append((Q_a_i.quantize(A_r) @ Q_b_i.quantize(B_r)).unsqueeze(0))
    return torch.cat(y_, dim=0)

def fuse_per_head_svd_rotated_weights(Alinear, Blinear, quantizers, idx, keys='q', num_heads=None, R=None):
    # Only rotate in-place, no quantization
    if R is None or num_heads is None:
        return
    import torch
    with torch.no_grad():
        for i in range(num_heads):
            # Promote to float32 for matmul stability, then cast back
            dev_a, dt_a = Alinear[i].device, Alinear[i].dtype
            dev_b, dt_b = Blinear[i].device, Blinear[i].dtype
            A_new = (Alinear[i].detach().to(torch.float32) @ R[i].to(torch.float32)).to(device=dev_a, dtype=dt_a)
            B_new = (R[i].T.to(torch.float32) @ Blinear[i].detach().to(torch.float32)).to(device=dev_b, dtype=dt_b)
            Alinear[i].copy_(A_new)
            Blinear[i].copy_(B_new)
            
def get_had_W_and_ABlinear(W, Alinear, Blinear, Q, key=None):  
    from rotation_utils import apply_exact_had_to_UV
    if key == 'v': # Only V proj need to apply hadamard
        a_dtype = Alinear[0].dtype
        device = Alinear[0].device
        Alinear_had = [apply_exact_had_to_UV(aa.to(device=utils.get_dev(), dtype=torch.float64), had_dim=-1, output=True).to(device=device, dtype=a_dtype) for aa in Alinear]

        W_had = [apply_exact_had_to_UV(ww.to(device=utils.get_dev(), dtype=torch.float64), had_dim=-1, output=True).to(device=device, dtype=W.dtype) for ww in W]
        W_had = torch.stack(W_had, dim=0)
    else:
        Alinear_had = Alinear
        W_had = W
    
    b_dtype = Blinear[0].dtype
    device = Blinear[0].device
    Blinear_had = [torch.matmul(bb.to(device=utils.get_dev(), dtype=torch.float64), Q).to(device=device, dtype=b_dtype) for bb in Blinear]
    W_had = [torch.matmul(ww.to(device=utils.get_dev(), dtype=torch.float64), Q).to(device=device, dtype=W.dtype) for ww in W_had]
    W_had = torch.stack(W_had, dim=0)

    return W_had, Alinear_had, Blinear_had

def apply_exact_had_to_A(Alinear):
    from hadamard_utils import get_hadK
    # note: should always pass UV as a single matrix (out, in)

    out_features = 128
    had_K, K = get_hadK(out_features)
    # K = 1, had_K = 128 hadamard matrix
    Alinear_had = matmul_had_U_128_cuda(Alinear.t()).t()
        
    return Alinear_had

def matmul_had_U_128_cuda(X):
    import fast_hadamard_transform
    n = X.shape[-1]
    return fast_hadamard_transform.hadamard_transform(X.contiguous(), 1.0/torch.tensor(n).sqrt()) 