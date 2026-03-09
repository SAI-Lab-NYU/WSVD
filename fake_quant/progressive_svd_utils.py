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
import local_ft_grad_utils
import grad_info_utils
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
import logging
import math
import os
import torch.distributed as dist
import datetime
import time
import re

def progressive_svd(model, dataloader, tokenizer, image_processor, args, use_cache=True, cache_file=None):
    """
    Progressive SVD to allocate rank for each layer
    """
    # Number of layers to truncate per round (default: 1)
    layers_per_truncate = getattr(args, 'layers_per_truncate', 1)
    logging.info(f"Truncating {layers_per_truncate} layer(s) per round")
    
    # 1. Perform one SVD decomposition for all layers
    grad_info_utils.prepare_fuse_svd(model, args)
    layers = model_utils.get_layers(model)
    num_layers = len(layers)
    
    q_linear = layers[0].self_attn.q_proj
    k_linear = layers[0].self_attn.k_proj
    v_linear = layers[0].self_attn.v_proj
    n_params = q_linear.weight.numel() + k_linear.weight.numel() + v_linear.weight.numel()
    compressed_params = int(n_params * args.rank_ratio)
    average_rank = compressed_params // (q_linear.in_features + q_linear.out_features + k_linear.out_features + v_linear.out_features)
    logging.info(f"Average rank: {average_rank}")
    total_rank = (num_layers * compressed_params) // (q_linear.in_features + q_linear.out_features + k_linear.out_features + v_linear.out_features)
    logging.info(f"Total rank: {total_rank}")

    # 2. Initialize each layer rank to full rank
    layer_ranks = [layer.self_attn.k_proj.qkv_svd_info['S'].numel() for layer in layers]
    truncated_layers = [False] * num_layers

    remaining_rank = total_rank
    all_svd_indices = {}
    
    # grad_info_utils.calib_grad_info(model, dataloader, tokenizer, image_processor, args)
    
    # Process layers in batches, each with layers_per_truncate layers
    for batch_start in range(0, num_layers, layers_per_truncate):
        batch_end = min(batch_start + layers_per_truncate, num_layers)
        current_batch_layers = list(range(batch_start, batch_end))
        
        logging.info(f"Processing layers {batch_start} to {batch_end-1}")
        
        # 3. Compute importance scores for untruncated layers using calibration set
        # grad_info_utils.calib_grad_info(model, dataloader, tokenizer, image_processor, args)
        calib_grad_info(model, batch_start, dataloader, tokenizer, image_processor, args)
        # 4. Compute global top-k sigmas for remaining rank budget
        top_indices, _, layer_indices_dict = svd_qkv_with_grad_info(layers, batch_start, remaining_rank, args)
        
        # 5. Count selected sigmas and truncate each layer in current batch
        batch_total_rank = 0
        for l in current_batch_layers:
            kept_indices = layer_indices_dict.get(l, {}).get('k_proj', [])
            all_svd_indices[l] = kept_indices.copy()
            k_l = len(kept_indices)
            batch_total_rank += k_l
            
            if k_l == 0:
                logging.info(f"Layer {l} was assigned no sigma; skipping")
                continue
                
            # 6. Truncate layer l and replace it in the model
            truncate_layer_by_sigma(layers[l], l, kept_indices, args)
            truncated_layers[l] = True
            logging.info(f"Layer {l} truncated to {k_l}")
        
        # 7. Update remaining rank budget
        remaining_rank -= batch_total_rank
        logging.info(f"Batch {batch_start}-{batch_end-1} truncated {batch_total_rank} ranks in total; remaining rank: {remaining_rank}")
        
        reset_grad_info(model)
        
        if remaining_rank <= 0:
            logging.info("Rank budget exhausted; stopping truncation")
            break
    # 8. Save SVD indices for all layers
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    torch.save(all_svd_indices, os.path.join(save_path, f"all_svd_indices.pt"))
    logging.info(f"Saved all_svd_indices to {save_path}")
    return


def truncate_layer_by_sigma(layer, idx, kept_indices, args):
    """
    Keep only sigmas at kept_indices and reconstruct QKV weights.
    """
    from svd_utils import SVDLinear, rsetattr
    device = utils.get_dev()
    try:
        qlinear, klinear, vlinear = SVDLinear.from_linearqkv_with_grad(
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            param_ratio=args.rank_ratio,
            alpha=args.act_alpha,
            act_aware=args.act_aware,
            rank_align=1.,
            sigma_fuse=args.svd_mode,
            had_rank=args.had_rank,
            had_mode='random', # ‘rh'
            singular_indices=kept_indices,
            seed=args.seed,
            module_name='k_proj'
        )

        rsetattr(layer, 'self_attn.q_proj', qlinear)
        rsetattr(layer, 'self_attn.k_proj', klinear)
        rsetattr(layer, 'self_attn.v_proj', vlinear)
        logging.info(f"Layer {idx} QKV fusion SVD completed")
    except Exception as e:
        logging.info(f"Layer {idx} QKV fusion SVD failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
def calib_grad_info(model, current_idx, dataloader, tokenizer, image_processor, args):
    """
    Calibrate grad info for layer from idx to end
    """
    print("Starting Grad information calculation...")
    logging.info('start grad computing')
    model_id = model.config._name_or_path
    
    model.eval()

    # Ensure the entire model is on CUDA
    device = utils.get_dev()
    model = model.to(device)

    accumulation_steps = 1   # Number of accumulated batches
    batch_count = 0          # Accumulated batch counter
    
    # Set model to training mode and only allow gradient computation for svd_modules layers
    model.train()
    for name, param in model.named_parameters():
        if 'model.layers' in name:
            match = re.search(r'layers\.(\d+)', name)
            layer_idx = int(match.group(1)) if match else -1
            if ('q_proj' in name or 'k_proj' in name or 'v_proj' in name) and layer_idx >= current_idx: 
                # only compute grad info for layers from current_idx to end, since we only need to compute grad info for layers that are not truncated
                # although truncated layers' q,k,v proj are substituted and not in name, we still skip them by layer_idx >= current_idx
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False
    
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
                # breakpoint()
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

                #breakpoint()

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

                # breakpoint()
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

                #breakpoint()

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
                # breakpoint()
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
                    if idx >= current_idx:
                        # qkv only SVD grad info compute
                        svd_info = layer.self_attn.k_proj.qkv_svd_info
                        q_linear = layer.self_attn.q_proj
                        k_linear = layer.self_attn.k_proj
                        v_linear = layer.self_attn.v_proj
                        
                        if (q_linear.weight.grad is not None and 
                            k_linear.weight.grad is not None and 
                            v_linear.weight.grad is not None):
                            grad_cat = torch.cat([
                                q_linear.weight.grad.detach().to(torch.bfloat16),
                                k_linear.weight.grad.detach().to(torch.bfloat16),
                                v_linear.weight.grad.detach().to(torch.bfloat16),
                            ], dim=0).to(device)
                            
                            if args.act_aware:
                                # scaling_diag_matrix = svd_info['scaling_diag_matrix'].to(device) #
                                if hasattr(k_linear, "scaling_diag_matrix"):
                                    scaling_diag_matrix = k_linear.scaling_diag_matrix.to(device)
                                elif hasattr(k_linear, "scaling_diag_matrixS"): # [FIXME: use scaling_diag_matrix for SVDLLM/ASVD, differentiate by ndim]
                                    scaling_diag_matrix = k_linear.scaling_diag_matrixS.to(device)
                                else:
                                    raise ValueError("No scaling_diag_matrix found")
                                if scaling_diag_matrix.ndim == 1: # [NOTE: here we have to multiply grad_cat with S since V'=VS-1 will not be orthognal, need compute another V'-1]
                                    # 1D vector representing diagonal matrix elements
                                    scaling_diag_matrix = scaling_diag_matrix**args.act_alpha
                                    scaling_diag_matrix += 1e-6  # avoid zero division
                                    grad_cat = grad_cat * scaling_diag_matrix.view(1, -1).to(torch.bfloat16)  # Scale each column
                                elif scaling_diag_matrix.ndim == 2:
                                    # 2D matrix representing full scaling matrix (possibly non-diagonal)
                                    grad_cat = grad_cat @ scaling_diag_matrix.to(torch.bfloat16)  # Right multiply matrix
                            U = svd_info['U'].to(device).to(torch.bfloat16)
                            V = svd_info['V'].to(device).to(torch.bfloat16)
                            S_grad = torch.diag(U.T @ grad_cat @ V)
                            if args.is_taylor:
                                S = svd_info['S'].to(device).to(torch.bfloat16)
                                S_grad_squared = grad_info_utils.taylor_like_s_grad_squared(S, S_grad, args.taylor_order, args.add_taylor_first) # will this be enough?
                            else:
                                S_grad_squared = S_grad.pow(2)
                            
                            if not hasattr(layer.self_attn.k_proj, 'S_grad_info'): # 
                                layer.self_attn.k_proj.S_grad_info = S_grad_squared
                            else:
                                layer.self_attn.k_proj.S_grad_info += S_grad_squared
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
        for layer in model_utils.get_layers(model):
            if hasattr(layer.self_attn.k_proj, 'S_grad_info'):
                layer.self_attn.k_proj.S_grad_info = layer.self_attn.k_proj.S_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
    logging.info('finished grad computing')
    
def svd_qkv_with_grad_info(layers, current_idx, remaining_rank, args):
    """
    Perform SVD decomposition, truncate and replace current layer's q,k,v proj with SVD result
    
    Args:
        layers: List of model layers
        args: Parameter configuration 
    Returns:
        grad_scores_dict: Dictionary containing gradient importance scores for S in each layer
    """
    grad_alpha = args.grad_alpha
        
    # Directly use pre-computed S gradient information
    grad_scores_dict = {}
    device = utils.get_dev()  # Get CUDA device
    for idx, layer in enumerate(layers):
        if idx >= current_idx:
            layer_key = f"layer_{idx}"
            grad_scores_dict[layer_key] = {}
            # QKV fusion case (existing code)
            if hasattr(layer.self_attn.k_proj, 'qkv_svd_info') and hasattr(layer.self_attn.k_proj, 'S_grad_info'):
                svd_info = layer.self_attn.k_proj.qkv_svd_info
                S = svd_info['S']
                S_grad = layer.self_attn.k_proj.S_grad_info
                
                # Ensure S and S_grad are on the same device (both moved to CUDA)
                S = svd_info['S'].to(device).to(torch.bfloat16)
                S_grad = layer.self_attn.k_proj.S_grad_info.to(device).to(torch.bfloat16)
                
                # Calculate importance score: |S| * |S_grad|
                importance_score = torch.abs(S) * (torch.abs(S_grad)**grad_alpha)
                
                # Move result back to CPU for saving
                grad_scores_dict[layer_key]['k_proj'] = importance_score.cpu()
                grad_scores_dict[layer_key]['k_proj_S'] = S.cpu()

                print(f"Layer {idx} QKV importance score computed, shape: {importance_score.shape}")
            else:
                print(f"Warning: Layer {idx} lacks necessary SVD information or gradient information, cannot compute importance score")
    
    # Get indices and scores of top k important singular values
    # qkv fuse
    k_value = remaining_rank
    top_indices, top_scores, layer_indices_dict = grad_info_utils.get_top_k_scores(grad_scores_dict, k_list=[k_value], keys=['k_proj'])
    
    for key, indices in top_indices.items():
        logging.info(f"Selected top {len(indices)} important singular values for {key}")
    return top_indices, top_scores, layer_indices_dict    

def reset_grad_info(model):
    for layer in model_utils.get_layers(model):
        if hasattr(layer.self_attn.k_proj, 'S_grad_info'):
            layer.self_attn.k_proj.S_grad_info = torch.zeros_like(layer.self_attn.k_proj.S_grad_info)