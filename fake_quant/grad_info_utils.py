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
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
import logging
import math
import os
import torch.distributed as dist
import datetime
import random

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

def create_misaligned_dataloader(dataloader, seed=None):
    """
    Create misaligned text-image pairs by shuffling images in the dataloader.
    
    Args:
        dataloader: Original dataloader with aligned text-image pairs
        seed: Random seed for reproducibility
    
    Returns:
        misaligned_dataloader: Dataloader with shuffled images
    """
    if seed is not None:
        random.seed(seed)
    
    # Extract all image paths from the dataloader
    all_image_paths = []
    
    # Handle different dataloader structures
    if isinstance(dataloader, list):
        # If dataloader is a list of samples
        for sample in dataloader:
            if isinstance(sample, list):
                # Each sample is a list of message items
                for item in sample:
                    if isinstance(item, dict) and item.get("type") == "image":
                        all_image_paths.append(item["value"])
            elif isinstance(sample, dict):
                # Each sample is a single message item
                if sample.get("type") == "image":
                    all_image_paths.append(sample["value"])
    else:
        # If dataloader is an iterable (like a DataLoader)
        for batch in dataloader:
            if isinstance(batch, list):
                for item in batch:
                    if isinstance(item, dict) and item.get("type") == "image":
                        all_image_paths.append(item["value"])
            elif isinstance(batch, dict):
                if batch.get("type") == "image":
                    all_image_paths.append(batch["value"])
    
    if not all_image_paths:
        logging.warning("No image items found in dataloader. Returning original dataloader.")
        return dataloader
    
    # Shuffle the image paths
    shuffled_image_paths = all_image_paths.copy()
    random.shuffle(shuffled_image_paths)
    
    # Create misaligned dataloader
    misaligned_dataloader = []
    image_idx = 0
    
    if isinstance(dataloader, list):
        # Handle list structure
        for sample in dataloader:
            if isinstance(sample, list):
                # Each sample is a list of message items
                misaligned_sample = []
                for item in sample:
                    if isinstance(item, dict) and item.get("type") == "image":
                        # Replace with shuffled image
                        misaligned_item = item.copy()
                        misaligned_item["value"] = shuffled_image_paths[image_idx]
                        misaligned_sample.append(misaligned_item)
                        image_idx += 1
                    else:
                        # Keep text items unchanged
                        misaligned_sample.append(item)
                misaligned_dataloader.append(misaligned_sample)
            elif isinstance(sample, dict):
                # Each sample is a single message item
                if sample.get("type") == "image":
                    misaligned_item = sample.copy()
                    misaligned_item["value"] = shuffled_image_paths[image_idx]
                    misaligned_dataloader.append(misaligned_item)
                    image_idx += 1
                else:
                    misaligned_dataloader.append(sample)
    else:
        # Handle iterable structure
        for batch in dataloader:
            if isinstance(batch, list):
                misaligned_batch = []
                for item in batch:
                    if isinstance(item, dict) and item.get("type") == "image":
                        # Replace with shuffled image
                        misaligned_item = item.copy()
                        misaligned_item["value"] = shuffled_image_paths[image_idx]
                        misaligned_batch.append(misaligned_item)
                        image_idx += 1
                    else:
                        # Keep text items unchanged
                        misaligned_batch.append(item)
                misaligned_dataloader.append(misaligned_batch)
            elif isinstance(batch, dict):
                if batch.get("type") == "image":
                    misaligned_item = batch.copy()
                    misaligned_item["value"] = shuffled_image_paths[image_idx]
                    misaligned_dataloader.append(misaligned_item)
                    image_idx += 1
                else:
                    misaligned_dataloader.append(batch)
    
    logging.info(f"Created misaligned dataloader with {len(misaligned_dataloader)} samples, shuffled {len(all_image_paths)} images")
    return misaligned_dataloader

@torch.enable_grad()
def calib_grad_info(model, dataloader, tokenizer, image_processor, args, use_cache=True, cache_file=None):
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
    

    cache_file = args.cache_file
    
    if args.cache_file is None:
        cache_dir  = args.act_cache_dir + "/cache"
        if args.cache_in_log:
            cache_dir = args.act_cache_dir + "/cache"
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = args.cache_file
        
    # Add relevant information to cache file name
    if args.qkv_fuse:
        fusion_mode = "qkv_fuse"
    elif args.kv_fuse:
        fusion_mode = "kv_fuse"
    else:
        fusion_mode = "separate"
    rotate_info = "rotated" if hasattr(args, "rotate") and args.rotate else "norotate"
    calib_method_info = args.calib_method if hasattr(args, "act_aware") and args.act_aware else "no_act_aware"
    # cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_{rotate_info}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma_grad_info.pt")
    if args.cal_dataset == 'COCO_CALIB':
        fusion_mode = fusion_mode + "_coco"
    if args.misalign_text_image:
        fusion_mode = fusion_mode + "_misalign"
    if args.is_per_head_svd:
        fusion_mode = fusion_mode + "_perhead"
    if args.taylor_order != 2:
        fusion_mode = fusion_mode + "_taylor" + str(args.taylor_order)
    if args.a_clip_ratio == 1.0:
        if args.cache_file is not None:
            cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_{calib_method_info}_sigma{args.svd_modules}_{fusion_mode}_sigma_grad_info.pt")
        else:
            cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma{args.svd_modules}_{fusion_mode}_sigma_grad_info.pt")
    else:
        if args.cache_file is not None:
            cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_aclip{args.a_clip_ratio}_{calib_method_info}_sigma{args.svd_modules}_{fusion_mode}_sigma_grad_info.pt")
        else:
            cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_aclip{args.a_clip_ratio}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma{args.svd_modules}_{fusion_mode}_sigma_grad_info.pt")
    
    # First perform QKV SVD decomposition and store
    logging.info('start qkv svd for grad')
    prepare_fuse_svd(model, args)
    logging.info('finish qkv svd for grad')

    if not args.qkv_fuse and not args.kv_fuse:
        for idx, layer in enumerate(model_utils.get_layers(model)):
            if hasattr(layer.self_attn.k_proj, "scaling_diag_matrix"):
                if not hasattr(layer.self_attn.q_proj, "scaling_diag_matrix"):
                    layer.self_attn.q_proj.scaling_diag_matrix = layer.self_attn.k_proj.scaling_diag_matrix
                if not hasattr(layer.self_attn.v_proj, "scaling_diag_matrix"):
                    layer.self_attn.v_proj.scaling_diag_matrix = layer.self_attn.k_proj.scaling_diag_matrix
            elif hasattr(layer.self_attn.k_proj, "scaling_diag_matrixS"):
                if not hasattr(layer.self_attn.q_proj, "scaling_diag_matrixS"):
                    layer.self_attn.q_proj.scaling_diag_matrixS = layer.self_attn.k_proj.scaling_diag_matrixS
                if not hasattr(layer.self_attn.v_proj, "scaling_diag_matrixS"):
                    layer.self_attn.v_proj.scaling_diag_matrixS = layer.self_attn.k_proj.scaling_diag_matrixS


    if os.path.exists(cache_file) and use_cache:
        logging.info(f"Loading Grad information cache from {cache_file}...")
        all_grad_info = torch.load(cache_file, map_location="cpu")
        # Load gradient information into the self_attn.S_grad_info attribute of corresponding layers
        for idx, layer in enumerate(model_utils.get_layers(model)):
            layer_key = f"layer_{idx}"
            if layer_key in all_grad_info:
                if args.svd_modules in ['qkv', 'attn', 'all']:
                    for proj_name in ['q_proj', 'v_proj', 'k_proj']:
                        if proj_name in all_grad_info[layer_key]:
                            proj = getattr(layer.self_attn, proj_name)
                            proj.S_grad_info = all_grad_info[layer_key][proj_name].to(utils.get_dev())                
                if args.svd_modules in ['attn', 'all', 'o']: # only valid for not qkv-only SVD
                    layer.self_attn.o_proj.S_grad_info = all_grad_info[layer_key]['o_proj'].to(utils.get_dev())
                if args.svd_modules in ['all', 'mlp','gaup']: # only valid for all model SVD  
                    if args.mlp_fuse:
                        layer.mlp.up_proj.S_grad_info = all_grad_info[layer_key]['up_proj'].to(utils.get_dev())
                    else:
                        layer.mlp.up_proj.S_grad_info = all_grad_info[layer_key]['up_proj'].to(utils.get_dev())
                        layer.mlp.gate_proj.S_grad_info = all_grad_info[layer_key]['gate_proj'].to(utils.get_dev())
                if args.svd_modules in ['all', 'mlp', 'down']:
                    layer.mlp.down_proj.S_grad_info = all_grad_info[layer_key]['down_proj'].to(utils.get_dev())
        logging.info("Successfully loaded Grad information cache!")
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
                inputs, _, outputs = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
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
                outputs = move_to_device(outputs, device)

                input_ids = inputs.get('input_ids')
                output_ids = outputs.get('input_ids')
                
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
            elif tokenizer == 'qwen':
                # Use message_to_prompt_train to process batch data
                inputs, _, outputs = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer)
                
                # Define a recursive helper to move nested tensor structures to target device
                def move_to_device(obj, target_device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(target_device)
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(move_to_device(item, target_device) for item in obj)
                    elif isinstance(obj, dict):
                        return {k: move_to_device(v, target_device) for k, v in obj.items()}
                    else:
                        return obj
                
                # Move data to the target device
                inputs = move_to_device(inputs, device)
                outputs = move_to_device(outputs, device)
                
                # Align input and label lengths
                input_ids = inputs.get('input_ids')
                output_ids = outputs.get('input_ids')
                
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
                # inputs['attention_mask'] = input_ids.ne(0).to(device)
                # breakpoint()
                # Calculate loss for current batch and perform gradient accumulation
                with torch.enable_grad():
                    outputs = model(**inputs, labels=output_ids)
                    loss = outputs[0]
                    loss = loss / accumulation_steps  # Normalize loss
                    loss.backward()
            elif tokenizer == 'intern':
                inputs = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer)
                ### here turn chat funct warp to here
                ### msg to prompt inputsid in dataloader, but we implment here in this repo
                IMG_START_TOKEN='<img>'
                IMG_END_TOKEN='</img>'
                IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
                pixel_values = inputs.get('pixel_values', None)
                num_patches_list = inputs.get('num_patches_list', None)
                prompt = inputs.get('question', None)
                label = inputs.get('label', None)
                generation_config = inputs.get('generation_config', None)
                if pixel_values is not None and '<image>' not in prompt:
                    prompt = '<image>\n' + prompt
                if num_patches_list is not None:
                    num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
                assert pixel_values is None or len(pixel_values) == sum(num_patches_list)
                img_context_token_id = image_processor.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
                # {'pixel_values': pixel_values, 
                # 'num_patches_list': num_patches_list, 
                # 'question': prompt, 
                # 'generation_config': kwargs_default}
                pass
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
                    if args.svd_modules in ['qkv', 'attn', 'all']:
                        if args.qkv_fuse: #[FIXME: now gradinfo only support qkv fuse]
                            # qkv only SVD grad info compute
                            svd_info = layer.self_attn.k_proj.qkv_svd_info
                            q_linear = layer.self_attn.q_proj
                            k_linear = layer.self_attn.k_proj
                            v_linear = layer.self_attn.v_proj
                            
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
                                        ], dim=1).to(device) # n, 3c, C
                                else:
                                    grad_cat = torch.cat([
                                        q_linear.weight.grad.detach().to(torch.bfloat16),
                                        k_linear.weight.grad.detach().to(torch.bfloat16) * multiplier,
                                        v_linear.weight.grad.detach().to(torch.bfloat16) * multiplier,
                                    ], dim=0).to(device)
                                
                                if args.act_aware:
                                    # scaling_diag_matrix = svd_info['scaling_diag_matrix'].to(device) #
                                    if hasattr(k_linear, "scaling_diag_matrix"):
                                        scaling_diag_matrix = k_linear.scaling_diag_matrix.to(device)
                                    elif hasattr(k_linear, "scaling_diag_matrixS"): # [FIXME: use scaling_diag_matrix for SVDLLM/ASVD, differentiate by ndim]
                                        scaling_diag_matrix = k_linear.scaling_diag_matrixS.to(device)
                                        assert scaling_diag_matrix.ndim == 2, "scaling_diag_matrixS should be a 2D matrix"
                                        if hasattr(k_linear, "scaling_matrix_inverse_transpose"):
                                            scaling_matrix_inverse_transpose = k_linear.scaling_matrix_inverse_transpose
                                        else:
                                            scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                                            k_linear.scaling_matrix_inverse_transpose = scaling_matrix_inverse_transpose
                                    else:
                                        raise ValueError("No scaling_diag_matrix found")
                                    if scaling_diag_matrix.ndim == 1: # [NOTE: here we have to multiply grad_cat with S since V'=VS-1 will not be orthognal, need compute another V'-1]
                                        # 1D vector representing diagonal matrix elements
                                        scaling_diag_matrix += 1e-6  # avoid zero division
                                        scaling_diag_matrix = scaling_diag_matrix**(-args.act_alpha)
                                        
                                        if args.is_per_head_svd:
                                            grad_cat = grad_cat * scaling_diag_matrix.view(1, 1, -1).to(torch.bfloat16)  # Scale each column
                                        else:
                                            grad_cat = grad_cat * scaling_diag_matrix.view(1, -1).to(torch.bfloat16)  # Scale each column
                                    elif scaling_diag_matrix.ndim == 2:
                                        # 2D matrix representing full scaling matrix (possibly non-diagonal)
                                        # G(WS) = G(W)*(S^T)^(-1)
                                        # old
                                        # grad_cat = grad_cat @ scaling_diag_matrix.to(torch.bfloat16)  # Right multiply matrix
                                        
                                        # new
                                        grad_cat = grad_cat @ scaling_matrix_inverse_transpose.to(torch.bfloat16)  # Right multiply matrix
                                        # grad_cat = grad_cat @ scaling_diag_matrix.to(torch.bfloat16)  # Right multiply matrix
                                U = svd_info['U'].to(device).to(torch.bfloat16)
                                if args.group_ratio > 0:
                                    U = U.clone()
                                    q_channel = q_linear.out_features
                                    U[q_channel:,...] *= args.group_ratio
                                V = svd_info['V'].to(device).to(torch.bfloat16)
                                if args.is_per_head_svd:
                                    # FIXME: need add extra 
                                    # U: n, c, c(r)
                                    # V:  n, C_in, c(r); 
                                    # grad_cat: Cout, Cin ; n, 3c, C_in
                                    # S_grad = torch.diagonal(U.transpose(1, 2) @ grad_cat @ V, dim1=1, dim2=2) # n, c(r), c(r)? -> diag (n, c(r))
                                    S_grad = torch.sum(U * (grad_cat @ V), dim=1) # n, r
                                else:
                                    # S_grad = torch.diag(U.T @ grad_cat @ V)
                                    # more efficient - compute only diagonal elements
                                    # U: Cout, r
                                    # grad_cat: Cout, Cin
                                    # V: Cin, r
                                    # S_grad: r
                                    S_grad = torch.sum(U * (grad_cat @ V), dim=0)
                                if args.is_taylor:
                                    S = svd_info['S'].to(device).to(torch.bfloat16)
                                    S_grad_squared = taylor_like_s_grad_squared(S, S_grad, args.taylor_order, args.add_taylor_first) # will this be enough?
                                else:
                                    S_grad_squared = get_layer_importance(S_grad,  args)
                                
                                if not hasattr(layer.self_attn.k_proj, 'S_grad_info'): # 
                                    layer.self_attn.k_proj.S_grad_info = S_grad_squared
                                else:
                                    layer.self_attn.k_proj.S_grad_info += S_grad_squared
                        if not args.qkv_fuse and not args.kv_fuse:
                            # Separate Q, K, V handling
                            for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                                proj = getattr(layer.self_attn, proj_name)
                                if hasattr(proj, 'svd_info') and proj.weight.grad is not None:
                                    svd_info = proj.svd_info
                                    if args.is_per_head_svd:
                                        if proj_name == 'q_proj':
                                            num_heads = layer.self_attn.config.num_attention_heads
                                        else:
                                            num_heads = layer.self_attn.config.num_key_value_heads
                                        grad_cat = proj.weight.grad.detach().view(num_heads, -1, proj.in_features).to(torch.bfloat16).to(device)
                                    else:
                                        grad_cat = proj.weight.grad.detach().to(torch.bfloat16).to(device)
                                    
                                    
                                    if args.act_aware:
                                        if hasattr(proj, "scaling_diag_matrix"):
                                            try:
                                                scaling_diag_matrix = proj.scaling_diag_matrix.to(device)
                                            except:
                                                proj.scaling_diag_matrix = layer.self_attn.k_proj.scaling_diag_matrix
                                                scaling_diag_matrix = layer.self_attn.k_proj.scaling_diag_matrix.to(device)
                                            scaling_diag_matrix += 1e-6
                                            scaling_diag_matrix = scaling_diag_matrix**(-args.act_alpha)
                                            if args.is_per_head_svd:
                                                grad_cat = grad_cat * scaling_diag_matrix.view(1, 1, -1).to(torch.bfloat16)
                                            else:
                                                grad_cat = grad_cat * scaling_diag_matrix.view(1, -1).to(torch.bfloat16)
                                        elif hasattr(proj, "scaling_diag_matrixS"):
                                            try:
                                                scaling_diag_matrix = proj.scaling_diag_matrixS.to(device)
                                            except:
                                                proj.scaling_diag_matrixS = layer.self_attn.k_proj.scaling_diag_matrixS
                                                scaling_diag_matrix = layer.self_attn.k_proj.scaling_diag_matrixS.to(device)
                                            if hasattr(layer.self_attn.k_proj, "scaling_matrix_inverse_transpose"):
                                                scaling_matrix_inverse_transpose = layer.self_attn.k_proj.scaling_matrix_inverse_transpose
                                            else:
                                                scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                                                layer.self_attn.k_proj.scaling_matrix_inverse_transpose = scaling_matrix_inverse_transpose
                                            grad_cat = grad_cat @ scaling_matrix_inverse_transpose.to(torch.bfloat16)
                                    
                                    U = svd_info['U'].to(device).to(torch.bfloat16)
                                    V = svd_info['V'].to(device).to(torch.bfloat16)
                                    # S_grad = torch.diag(U.T @ grad_cat @ V)
                                    if args.is_per_head_svd:
                                        S_grad = torch.sum(U * (grad_cat @ V), dim=1) # n, r
                                    else:
                                        S_grad = torch.sum(U * (grad_cat @ V), dim=0)
                                    
                                    if args.is_taylor:
                                        S = svd_info['S'].to(device).to(torch.bfloat16)
                                        S_grad_squared = taylor_like_s_grad_squared(S, S_grad, args.taylor_order, args.add_taylor_first)
                                    else:
                                        S_grad_squared = get_layer_importance(S_grad,  args)
                                    
                                    if not hasattr(proj, 'S_grad_info'):
                                        proj.S_grad_info = S_grad_squared
                                    else:
                                        proj.S_grad_info += S_grad_squared 
                        elif args.kv_fuse:
                            # Handle Q separately
                            q_proj = layer.self_attn.q_proj
                            if hasattr(q_proj, 'svd_info') and q_proj.weight.grad is not None:
                                svd_info = q_proj.svd_info
                                grad_cat = q_proj.weight.grad.detach().to(torch.bfloat16).to(device)
                                
                                if args.act_aware:
                                    if hasattr(q_proj, "scaling_diag_matrix"):
                                        scaling_diag_matrix = q_proj.scaling_diag_matrix.to(device)
                                        scaling_diag_matrix += 1e-6
                                        scaling_diag_matrix = scaling_diag_matrix**(-args.act_alpha)
                                        grad_cat = grad_cat * scaling_diag_matrix.view(1, -1).to(torch.bfloat16)
                                    elif hasattr(q_proj, "scaling_diag_matrixS"):
                                        scaling_diag_matrix = q_proj.scaling_diag_matrixS.to(device)
                                        grad_cat = grad_cat @ scaling_diag_matrix.to(torch.bfloat16)
                                
                                U = svd_info['U'].to(device).to(torch.bfloat16)
                                V = svd_info['V'].to(device).to(torch.bfloat16)
                                S_grad = torch.diag(U.T @ grad_cat @ V)
                                
                                if args.is_taylor:
                                    S = svd_info['S'].to(device).to(torch.bfloat16)
                                    S_grad_squared = taylor_like_s_grad_squared(S, S_grad, args.taylor_order, args.add_taylor_first)
                                else:
                                    S_grad_squared = get_layer_importance(S_grad,  args)
                                
                                if not hasattr(q_proj, 'S_grad_info'):
                                    q_proj.S_grad_info = S_grad_squared
                                else:
                                    q_proj.S_grad_info += S_grad_squared
                            
                            # Handle KV fusion
                            k_proj = layer.self_attn.k_proj
                            v_proj = layer.self_attn.v_proj
                            if (hasattr(k_proj, 'svd_info') and 
                                k_proj.weight.grad is not None and 
                                v_proj.weight.grad is not None):
                                
                                svd_info = k_proj.svd_info
                                grad_cat = torch.cat([
                                    k_proj.weight.grad.detach().to(torch.bfloat16),
                                    v_proj.weight.grad.detach().to(torch.bfloat16)
                                ], dim=0).to(device)
                                
                                if args.act_aware:
                                    if hasattr(k_proj, "scaling_diag_matrix"):
                                        scaling_diag_matrix = k_proj.scaling_diag_matrix.to(device)
                                        scaling_diag_matrix += 1e-6
                                        scaling_diag_matrix = scaling_diag_matrix**(-args.act_alpha)
                                        grad_cat = grad_cat * scaling_diag_matrix.view(1, -1).to(torch.bfloat16)
                                    elif hasattr(k_proj, "scaling_diag_matrixS"):
                                        scaling_diag_matrix = k_proj.scaling_diag_matrixS.to(device)
                                        grad_cat = grad_cat @ scaling_diag_matrix.to(torch.bfloat16)
                                
                                U = svd_info['U'].to(device).to(torch.bfloat16)
                                V = svd_info['V'].to(device).to(torch.bfloat16)
                                S_grad = torch.diag(U.T @ grad_cat @ V)
                                
                                if args.is_taylor:
                                    S = svd_info['S'].to(device).to(torch.bfloat16)
                                    S_grad_squared = taylor_like_s_grad_squared(S, S_grad, args.taylor_order, args.add_taylor_first)
                                else:
                                    S_grad_squared = get_layer_importance(S_grad,  args)
                                
                                if not hasattr(k_proj, 'S_grad_info'):
                                    k_proj.S_grad_info = S_grad_squared
                                else:
                                    k_proj.S_grad_info += S_grad_squared
                        
                    if args.svd_modules in ['attn', 'all', 'o']: # add o_proj grad info compute
                        svd_info = layer.self_attn.o_proj.svd_info
                        o_proj = layer.self_attn.o_proj
                        if (o_proj.weight.grad is not None):
                            grad_cat = o_proj.weight.grad.detach().to(torch.bfloat16).to(device)
                            if args.act_aware:
                                if hasattr(o_proj, "scaling_diag_matrix"):
                                    scaling_diag_matrix = o_proj.scaling_diag_matrix.to(device)
                                elif hasattr(o_proj, "scaling_diag_matrixS"):
                                    scaling_diag_matrix = o_proj.scaling_diag_matrixS.to(device)
                                    assert scaling_diag_matrix.ndim == 2, "scaling_diag_matrixS should be a 2D matrix"
                                    if hasattr(o_proj, "scaling_matrix_inverse_transpose"):
                                        scaling_matrix_inverse_transpose = o_proj.scaling_matrix_inverse_transpose
                                    else:
                                        scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                                        o_proj.scaling_matrix_inverse_transpose = scaling_matrix_inverse_transpose
                                else:
                                    raise ValueError("No scaling_diag_matrix found")
                                if scaling_diag_matrix.ndim == 1:
                                    scaling_diag_matrix += 1e-6  # avoid zero division
                                    scaling_diag_matrix = scaling_diag_matrix**(-args.act_alpha)
                                    grad_cat = grad_cat * scaling_diag_matrix.view(1, -1).to(torch.bfloat16)  # Scale each column
                                elif scaling_diag_matrix.ndim == 2:
                                    # grad_cat = grad_cat @ scaling_diag_matrix.to(torch.bfloat16)  # Right multiply matrix
                                    grad_cat = grad_cat @ scaling_matrix_inverse_transpose.to(torch.bfloat16)  # Right multiply matrix
                            U = svd_info['U'].to(device).to(torch.bfloat16)
                            V = svd_info['V'].to(device).to(torch.bfloat16)
                            # S_grad = torch.diag(U.T @ grad_cat @ V)
                            S_grad = torch.sum(U * (grad_cat @ V), dim=0)
                            if args.is_taylor:
                                S = svd_info['S'].to(device).to(torch.bfloat16)
                                S_grad_squared = taylor_like_s_grad_squared(S, S_grad, args.taylor_order, args.add_taylor_first) # will this be enough?
                            else:
                                S_grad_squared = get_layer_importance(S_grad,  args)
                            if not hasattr(layer.self_attn.o_proj, 'S_grad_info'): # 
                                layer.self_attn.o_proj.S_grad_info = S_grad_squared
                            else:
                                layer.self_attn.o_proj.S_grad_info += S_grad_squared
                    if args.svd_modules in ['mlp', 'all', 'gaup']: # add mlp grad info compute
                        if args.mlp_fuse:
                            svd_info = layer.mlp.up_proj.svd_info # add mlp shared gated up info compute
                            up_proj = layer.mlp.up_proj
                            gate_proj = layer.mlp.gate_proj #[FIXME: support share gate_proj for now]
                            if (up_proj.weight.grad is not None and 
                                gate_proj.weight.grad is not None):
                                grad_cat = torch.cat([
                                    up_proj.weight.grad.detach().to(torch.bfloat16),
                                    gate_proj.weight.grad.detach().to(torch.bfloat16)
                                ], dim=0).to(device)
                                if args.act_aware:
                                    if hasattr(up_proj, "scaling_diag_matrix"):
                                        scaling_diag_matrix = up_proj.scaling_diag_matrix.to(device)
                                    elif hasattr(up_proj, "scaling_diag_matrixS"):
                                        scaling_diag_matrix = up_proj.scaling_diag_matrixS.to(device)
                                        assert scaling_diag_matrix.ndim == 2, "scaling_diag_matrixS should be a 2D matrix"
                                        if hasattr(up_proj, "scaling_matrix_inverse_transpose"):
                                            scaling_matrix_inverse_transpose = up_proj.scaling_matrix_inverse_transpose
                                        else:
                                            scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                                            up_proj.scaling_matrix_inverse_transpose = scaling_matrix_inverse_transpose
                                    else:
                                        raise ValueError("No scaling_diag_matrix found")
                                    if scaling_diag_matrix.ndim == 1:
                                        scaling_diag_matrix += 1e-6  # avoid zero division
                                        scaling_diag_matrix = scaling_diag_matrix**(-args.act_alpha)
                                        grad_cat = grad_cat * scaling_diag_matrix.view(1, -1).to(torch.bfloat16)  # Scale each column
                                    elif scaling_diag_matrix.ndim == 2:
                                        # grad_cat = grad_cat @ scaling_diag_matrix.to(torch.bfloat16)  # Right multiply matrix
                                        grad_cat = grad_cat @ scaling_matrix_inverse_transpose.to(torch.bfloat16)  # Right multiply matrix
                                U = svd_info['U'].to(device).to(torch.bfloat16)
                                V = svd_info['V'].to(device).to(torch.bfloat16)
                                # S_grad = torch.diag(U.T @ grad_cat @ V)
                                S_grad = torch.sum(U * (grad_cat @ V), dim=0)
                                if args.is_taylor:
                                    S = svd_info['S'].to(device).to(torch.bfloat16)
                                    S_grad_squared = taylor_like_s_grad_squared(S, S_grad, args.taylor_order, args.add_taylor_first) # will this be enough?
                                else:
                                    S_grad_squared = get_layer_importance(S_grad,  args)

                                if not hasattr(layer.mlp.up_proj, 'S_grad_info'): # 
                                    layer.mlp.up_proj.S_grad_info = S_grad_squared
                                else:
                                    layer.mlp.up_proj.S_grad_info += S_grad_squared
                        else:
                            svd_info = layer.mlp.up_proj.svd_info # add mlp down grad info compute
                            up_proj = layer.mlp.up_proj
                            if (up_proj.weight.grad is not None):
                                grad_cat = up_proj.weight.grad.detach().to(torch.bfloat16).to(device)
                                if args.act_aware:
                                    if hasattr(up_proj, "scaling_diag_matrix"):
                                        scaling_diag_matrix = up_proj.scaling_diag_matrix.to(device)
                                    elif hasattr(up_proj, "scaling_diag_matrixS"):
                                        scaling_diag_matrix = up_proj.scaling_diag_matrixS.to(device)
                                        assert scaling_diag_matrix.ndim == 2, "scaling_diag_matrixS should be a 2D matrix"
                                        if hasattr(up_proj, "scaling_matrix_inverse_transpose"):
                                            scaling_matrix_inverse_transpose = up_proj.scaling_matrix_inverse_transpose
                                        else:
                                            scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                                            up_proj.scaling_matrix_inverse_transpose = scaling_matrix_inverse_transpose
                                    else:
                                        raise ValueError("No scaling_diag_matrix found")
                                    if scaling_diag_matrix.ndim == 1:
                                        scaling_diag_matrix += 1e-6  # avoid zero division
                                        scaling_diag_matrix = scaling_diag_matrix**(-args.act_alpha)
                                        grad_cat = grad_cat * scaling_diag_matrix.view(1, -1).to(torch.bfloat16)  # Scale each column
                                    elif scaling_diag_matrix.ndim == 2:
                                        # grad_cat = grad_cat @ scaling_diag_matrix.to(torch.bfloat16)  # Right multiply matrix
                                        grad_cat = grad_cat @ scaling_matrix_inverse_transpose.to(torch.bfloat16)  # Right multiply matrix
                                U = svd_info['U'].to(device).to(torch.bfloat16)
                                V = svd_info['V'].to(device).to(torch.bfloat16)
                                # S_grad = torch.diag(U.T @ grad_cat @ V)
                                S_grad = torch.sum(U * (grad_cat @ V), dim=0)
                                if args.is_taylor:
                                    S = svd_info['S'].to(device).to(torch.bfloat16)
                                    S_grad_squared = taylor_like_s_grad_squared(S, S_grad, args.taylor_order, args.add_taylor_first) # will this be enough?
                                else:
                                    S_grad_squared = get_layer_importance(S_grad,  args)
                                if not hasattr(layer.mlp.up_proj, 'S_grad_info'): # 
                                    layer.mlp.up_proj.S_grad_info = S_grad_squared
                                else:
                                    layer.mlp.up_proj.S_grad_info += S_grad_squared
                            svd_info = layer.mlp.gate_proj.svd_info # add mlp down grad info compute
                            gate_proj = layer.mlp.gate_proj
                            if (gate_proj.weight.grad is not None):
                                grad_cat = gate_proj.weight.grad.detach().to(torch.bfloat16).to(device)
                                if args.act_aware: # reuse up_proj scaling_diag_matrix
                                    if hasattr(up_proj, "scaling_diag_matrix"):
                                        scaling_diag_matrix = up_proj.scaling_diag_matrix.to(device)
                                    elif hasattr(up_proj, "scaling_diag_matrixS"):
                                        scaling_diag_matrix = up_proj.scaling_diag_matrixS.to(device)
                                        assert scaling_diag_matrix.ndim == 2, "scaling_diag_matrixS should be a 2D matrix"
                                        if hasattr(up_proj, "scaling_matrix_inverse_transpose"):
                                            scaling_matrix_inverse_transpose = up_proj.scaling_matrix_inverse_transpose
                                        else:
                                            scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                                            up_proj.scaling_matrix_inverse_transpose = scaling_matrix_inverse_transpose
                                    else:
                                        raise ValueError("No scaling_diag_matrix found")
                                    if scaling_diag_matrix.ndim == 1:
                                        scaling_diag_matrix += 1e-6  # avoid zero division
                                        scaling_diag_matrix = scaling_diag_matrix**(-args.act_alpha)
                                        grad_cat = grad_cat * scaling_diag_matrix.view(1, -1).to(torch.bfloat16)  # Scale each column
                                    elif scaling_diag_matrix.ndim == 2:
                                        # grad_cat = grad_cat @ scaling_diag_matrix.to(torch.bfloat16)  # Right multiply matrix
                                        grad_cat = grad_cat @ scaling_matrix_inverse_transpose.to(torch.bfloat16)  # Right multiply matrix
                                U = svd_info['U'].to(device).to(torch.bfloat16)
                                V = svd_info['V'].to(device).to(torch.bfloat16)
                                # S_grad = torch.diag(U.T @ grad_cat @ V)
                                S_grad = torch.sum(U * (grad_cat @ V), dim=0)
                                if args.is_taylor:
                                    S = svd_info['S'].to(device).to(torch.bfloat16)
                                    S_grad_squared = taylor_like_s_grad_squared(S, S_grad, args.taylor_order, args.add_taylor_first) # will this be enough?
                                else:
                                    S_grad_squared = get_layer_importance(S_grad,  args)
                                if not hasattr(layer.mlp.gate_proj, 'S_grad_info'): # 
                                    layer.mlp.gate_proj.S_grad_info = S_grad_squared
                                else:
                                    layer.mlp.gate_proj.S_grad_info += S_grad_squared
                    if args.svd_modules in ['mlp', 'all', 'down']:
                        svd_info = layer.mlp.down_proj.svd_info # add mlp down grad info compute
                        down_proj = layer.mlp.down_proj
                        if (down_proj.weight.grad is not None):
                            grad_cat = down_proj.weight.grad.detach().to(torch.bfloat16).to(device)
                            if args.act_aware:
                                if hasattr(down_proj, "scaling_diag_matrix"):
                                    scaling_diag_matrix = down_proj.scaling_diag_matrix.to(device)
                                elif hasattr(down_proj, "scaling_diag_matrixS"):
                                    scaling_diag_matrix = down_proj.scaling_diag_matrixS.to(device)
                                    assert scaling_diag_matrix.ndim == 2, "scaling_diag_matrixS should be a 2D matrix"
                                    if hasattr(down_proj, "scaling_matrix_inverse_transpose"):
                                        scaling_matrix_inverse_transpose = down_proj.scaling_matrix_inverse_transpose
                                    else:
                                        scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                                        down_proj.scaling_matrix_inverse_transpose = scaling_matrix_inverse_transpose
                                else:
                                    raise ValueError("No scaling_diag_matrix found")
                                if scaling_diag_matrix.ndim == 1:
                                    scaling_diag_matrix += 1e-6  # avoid zero division
                                    scaling_diag_matrix = scaling_diag_matrix**(-args.act_alpha)
                                    grad_cat = grad_cat * scaling_diag_matrix.view(1, -1).to(torch.bfloat16)  # Scale each column
                                elif scaling_diag_matrix.ndim == 2:
                                    # grad_cat = grad_cat @ scaling_diag_matrix.to(torch.bfloat16)  # Right multiply matrix
                                    grad_cat = grad_cat @ scaling_matrix_inverse_transpose.to(torch.bfloat16)  # Right multiply matrix
                            U = svd_info['U'].to(device).to(torch.bfloat16)
                            V = svd_info['V'].to(device).to(torch.bfloat16)
                            # S_grad = torch.diag(U.T @ grad_cat @ V)
                            S_grad = torch.sum(U * (grad_cat @ V), dim=0)
                            if args.is_taylor:
                                S = svd_info['S'].to(device).to(torch.bfloat16)
                                S_grad_squared = taylor_like_s_grad_squared(S, S_grad, args.taylor_order, args.add_taylor_first) # will this be enough?
                            else:
                                S_grad_squared = get_layer_importance(S_grad,  args)
                            if not hasattr(layer.mlp.down_proj, 'S_grad_info'): # 
                                layer.mlp.down_proj.S_grad_info = S_grad_squared
                            else:
                                layer.mlp.down_proj.S_grad_info += S_grad_squared

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
                if args.svd_modules in ['qkv', 'attn', 'all']:
                    layer.self_attn.k_proj.S_grad_info = layer.self_attn.k_proj.S_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                    if not args.qkv_fuse and not args.kv_fuse:
                        # Separate Q, K, V handling
                        for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                            proj = getattr(layer.self_attn, proj_name)
                            if hasattr(proj, 'S_grad_info'):
                                proj.S_grad_info = proj.S_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                    elif args.kv_fuse:
                        # Normalize Q separately for KV fusion
                        if hasattr(layer.self_attn.q_proj, 'S_grad_info'):
                            layer.self_attn.q_proj.S_grad_info = layer.self_attn.q_proj.S_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
            if args.svd_modules in ['attn', 'all', 'o']: # [FIXME: add sgrad info check]
                layer.self_attn.o_proj.S_grad_info = layer.self_attn.o_proj.S_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
            if args.svd_modules in ['all', 'mlp', 'gaup']:
                if args.mlp_fuse:
                    layer.mlp.up_proj.S_grad_info = layer.mlp.up_proj.S_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                else:
                    layer.mlp.up_proj.S_grad_info = layer.mlp.up_proj.S_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                    layer.mlp.gate_proj.S_grad_info = layer.mlp.gate_proj.S_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
            if args.svd_modules in ['all', 'mlp', 'down']:
                layer.mlp.down_proj.S_grad_info = layer.mlp.down_proj.S_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
    logging.info('finished grad computing')
    # Save S gradient information
    all_grad_info = {}
    for idx, layer in enumerate(model_utils.get_layers(model)):
        all_grad_info[f"layer_{idx}"] = {}
        print(f"Layer {idx} initialize dict")
        if args.svd_modules in ['qkv', 'attn', 'all']:
            if hasattr(layer.self_attn.k_proj, 'S_grad_info'):
                print(f"Layer {idx}: {layer.self_attn.k_proj.S_grad_info.shape}")
                all_grad_info[f"layer_{idx}"]['k_proj'] = layer.self_attn.k_proj.S_grad_info.cpu()
            
            for proj_name in ['q_proj', 'v_proj']:
                proj = getattr(layer.self_attn, proj_name)
                if hasattr(proj, 'S_grad_info'):
                    print(f"Layer {idx} {proj_name}: {proj.S_grad_info.shape}")
                    all_grad_info[f"layer_{idx}"][proj_name] = proj.S_grad_info.cpu()
        if args.svd_modules in ['attn', 'all', 'o']:
            print(f"Layer {idx}: {layer.self_attn.o_proj.S_grad_info.shape}")
            all_grad_info[f"layer_{idx}"]['o_proj'] = layer.self_attn.o_proj.S_grad_info.cpu()
        if args.svd_modules in ['all', 'mlp', 'gaup']:
            if args.mlp_fuse:
                print(f"Layer {idx}: {layer.mlp.up_proj.S_grad_info.shape}")
                all_grad_info[f"layer_{idx}"]['up_proj'] = layer.mlp.up_proj.S_grad_info.cpu()
            else:
                print(f"Layer {idx}: {layer.mlp.up_proj.S_grad_info.shape}")
                print(f"Layer {idx}: {layer.mlp.gate_proj.S_grad_info.shape}")
                all_grad_info[f"layer_{idx}"]['up_proj'] = layer.mlp.up_proj.S_grad_info.cpu()
                all_grad_info[f"layer_{idx}"]['gate_proj'] = layer.mlp.gate_proj.S_grad_info.cpu()
        if args.svd_modules in ['mlp', 'all', 'down']:
            print(f"Layer {idx}: {layer.mlp.down_proj.S_grad_info.shape}")
            all_grad_info[f"layer_{idx}"]['down_proj'] = layer.mlp.down_proj.S_grad_info.cpu()

    logging.info(f"Saving Grad information cache to {cache_file}...")
    torch.save(all_grad_info, cache_file)
    logging.info("Grad information cache saved successfully!")

def prepare_fuse_svd(model, args):
    """
    Pre-process QKV layers with SVD decomposition and store results in attention modules
    
    Args:
        model: Model to be processed
        args: Parameter configuration
    """
    print("Preprocessing QKV layer SVD decomposition...")
    device = utils.get_dev()
    alpha = args.act_alpha
    # model_utils.get_layers(model)
    
    for idx, layer in enumerate(tqdm(model_utils.get_layers(model), desc="Preparing QKV SVD")):
        if args.svd_modules in ['qkv', 'attn', 'all']:
            q_linear = layer.self_attn.q_proj
            k_linear = layer.self_attn.k_proj
            v_linear = layer.self_attn.v_proj
            if args.qkv_fuse:
                try:
                    # Move weights to CUDA device
                    if args.is_per_head_svd:
                        num_heads = layer.self_attn.config.num_key_value_heads if not args.is_q_headnum else layer.self_attn.config.num_attention_heads
                        w = torch.cat([
                            q_linear.weight.data.view(num_heads, -1, q_linear.in_features).float(), 
                            k_linear.weight.data.view(num_heads, -1, k_linear.in_features).float(), 
                            v_linear.weight.data.view(num_heads, -1, v_linear.in_features).float()
                        ], dim=1).to(device)
                    else:
                        w = torch.cat([
                            q_linear.weight.data.float(), 
                            k_linear.weight.data.float(), 
                            v_linear.weight.data.float()
                        ], dim=0).to(device)
                    
                    # Apply activation-aware scaling (if enabled)
                    if args.act_aware:
                        scaling_diag_matrix = torch.ones(k_linear.in_features, device=utils.get_dev())  # avoid zero division
                        if hasattr(k_linear, "scaling_diag_matrix"):
                            # print("WARNING: scaling_diag_matrix is used")
                            scaling_diag_matrix *= k_linear.scaling_diag_matrix.to(utils.get_dev())**alpha
                            scaling_diag_matrix += 1e-6  # avoid zero division
                            scaling_matrix_inv = None
                            if args.is_per_head_svd:
                                w = w * scaling_diag_matrix.view(1, 1, -1)
                            else:
                                w = w * scaling_diag_matrix.view(1, -1)
                        elif hasattr(k_linear, "scaling_diag_matrixS"):
                            scaling_diag_matrix = k_linear.scaling_diag_matrixS.to(utils.get_dev())
                            try:
                                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                            except Exception as e:
                                logging.info("Warning: scaling_diag_matrix is not full rank!")
                                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                                scaling_matrix_inv = None
                            w = w @ scaling_diag_matrix.float()

                    # SVD decomposition
                    U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False) # SVD decomposition of WS
                    if args.is_per_head_svd:
                        V = Vt.transpose(1, 2) # n, C_in, c
                    else:
                        V = Vt.T
                    
                    # Store SVD results
                    layer.self_attn.k_proj.qkv_svd_info = {
                        'U': U.cpu(),
                        'S': S.cpu(),
                        'V': V.cpu()
                    } 
                    print(f"Layer {idx} QKV SVD completed, S shape: {S.shape}")
                    
                except Exception as e:
                    print(f"Layer {idx} QKV SVD failed: {e}")
                    import traceback
                    traceback.print_exc()
            elif args.kv_fuse:
                # Handle Q separately
                try:
                    w_q = q_linear.weight.data.float().to(utils.get_dev())
                    if args.act_aware:
                        scaling_diag_matrix = torch.ones(q_linear.in_features, device=device)
                        if hasattr(q_linear, "scaling_diag_matrix"):
                            scaling_diag_matrix *= q_linear.scaling_diag_matrix.to(device)**alpha
                            scaling_diag_matrix += 1e-6
                            w_q = w_q * scaling_diag_matrix.view(1, -1)
                        elif hasattr(q_linear, "scaling_diag_matrixS"):
                            scaling_diag_matrix = q_linear.scaling_diag_matrixS.to(device)
                            try:
                                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                            except Exception as e:
                                logging.info("Warning: scaling_diag_matrix is not full rank!")
                                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                                scaling_matrix_inv = None
                            w_q = w_q @ scaling_diag_matrix.float()
                    
                    U_q, S_q, Vt_q = torch.linalg.svd(w_q.to(torch.float32), full_matrices=False)
                    V_q = Vt_q.T
                    
                    layer.self_attn.q_proj.svd_info = {
                        'U': U_q.cpu(),
                        'S': S_q.cpu(),
                        'V': V_q.cpu()
                    }
                    print(f"Layer {idx} Q SVD completed, S shape: {S_q.shape}")
                except Exception as e:
                    print(f"Layer {idx} Q SVD failed: {e}")
                
                # Handle KV fusion
                try:
                    w_kv = torch.cat([
                        k_linear.weight.data.float(),
                        v_linear.weight.data.float()
                    ], dim=0).to(utils.get_dev())
                    
                    if args.act_aware:
                        scaling_diag_matrix = torch.ones(k_linear.in_features, device=device)
                        if hasattr(k_linear, "scaling_diag_matrix"):
                            scaling_diag_matrix *= k_linear.scaling_diag_matrix.to(device)**alpha
                            scaling_diag_matrix += 1e-6
                            w_kv = w_kv * scaling_diag_matrix.view(1, -1)
                        elif hasattr(k_linear, "scaling_diag_matrixS"):
                            scaling_diag_matrix = k_linear.scaling_diag_matrixS.to(device)
                            try:
                                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                            except Exception as e:
                                logging.info("Warning: scaling_diag_matrix is not full rank!")
                                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                                scaling_matrix_inv = None
                            w_kv = w_kv @ scaling_diag_matrix.float()
                    
                    U_kv, S_kv, Vt_kv = torch.linalg.svd(w_kv.to(torch.float32), full_matrices=False)
                    V_kv = Vt_kv.T
                    
                    layer.self_attn.k_proj.svd_info = {
                        'U': U_kv.cpu(),
                        'S': S_kv.cpu(),
                        'V': V_kv.cpu()
                    }
                    print(f"Layer {idx} KV SVD completed, S shape: {S_kv.shape}")
                except Exception as e:
                    print(f"Layer {idx} KV SVD failed: {e}")
            else:
                # Handle separate Q, K, V
                for proj_name, proj_layer in [('q_proj', q_linear), ('k_proj', k_linear), ('v_proj', v_linear)]:
                    try:
                        if args.is_per_head_svd:
                            if proj_name == 'q_proj':
                                num_heads = layer.self_attn.config.num_attention_heads
                            else:
                                num_heads = layer.self_attn.config.num_key_value_heads
                            w = proj_layer.weight.data.view(num_heads, -1, proj_layer.in_features).float().to(utils.get_dev())
                        else:
                            w = proj_layer.weight.data.float().to(utils.get_dev())
                        if args.act_aware:
                            scaling_diag_matrix = torch.ones(proj_layer.in_features, device=device)
                            if hasattr(k_linear, "scaling_diag_matrix") and k_linear.scaling_diag_matrix is not None:
                                scaling_diag_matrix *= k_linear.scaling_diag_matrix.to(device)**alpha
                                scaling_diag_matrix += 1e-6
                                if args.is_per_head_svd:
                                    w = w * scaling_diag_matrix.view(1, 1, -1)
                                else:
                                    w = w * scaling_diag_matrix.view(1, -1)
                            elif hasattr(k_linear, "scaling_diag_matrixS") and k_linear.scaling_diag_matrixS is not None:
                                scaling_diag_matrix = k_linear.scaling_diag_matrixS.to(device)
                                try:
                                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                                except Exception as e:
                                    logging.info("Warning: scaling_diag_matrix is not full rank!")
                                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                                    scaling_matrix_inv = None
                                w = w @ scaling_diag_matrix.float()
                        
                        U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False)
                        if args.is_per_head_svd:
                            V = Vt.transpose(1, 2)
                        else:
                            V = Vt.T
                        
                        proj_layer.svd_info = {
                            'U': U.cpu(),
                            'S': S.cpu(),
                            'V': V.cpu()
                        }
                        print(f"Layer {idx} {proj_name} SVD completed, S shape: {S.shape}")
                    except Exception as e:
                        print(f"Layer {idx} {proj_name} SVD failed: {e}")
                        import traceback
                        traceback.print_exc()
        if args.svd_modules in ['attn', 'all', 'o']:
            o_proj = layer.self_attn.o_proj
            try:
                # Move weights to CUDA device
                w = o_proj.weight.data.float().to(utils.get_dev())

                # Apply activation-aware scaling (if enabled)
                if args.act_aware:
                    scaling_diag_matrix = torch.ones(o_proj.in_features, device=utils.get_dev())  # avoid zero division
                    if hasattr(o_proj, "scaling_diag_matrix"):
                        scaling_diag_matrix *= o_proj.scaling_diag_matrix.to(utils.get_dev())**alpha
                        scaling_diag_matrix += 1e-6  # avoid zero division
                        scaling_matrix_inv = None
                        w = w * scaling_diag_matrix.view(1, -1)
                    elif hasattr(o_proj, "scaling_diag_matrixS"):
                        scaling_diag_matrix = o_proj.scaling_diag_matrixS.to(utils.get_dev())
                        try:
                            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                        except Exception as e:
                            logging.info("Warning: scaling_diag_matrix is not full rank!")
                            scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                            scaling_matrix_inv = None
                        w = w @ scaling_diag_matrix.float()
                    else:
                        raise ValueError("No scaling_diag_matrix found")
                
                # SVD decomposition
                U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False) # SVD decomposition of WS
                V = Vt.T
                
                # Store SVD results
                layer.self_attn.o_proj.svd_info = {
                    'U': U.cpu(),
                    'S': S.cpu(),
                    'V': V.cpu()
                }
                print(f"Layer {idx} O-proj SVD completed, S shape: {S.shape}")
            except Exception as e:
                print(f"Layer {idx} O-proj SVD failed: {e}")
                import traceback
                traceback.print_exc()

        if args.svd_modules in ['all', 'mlp', 'gaup']:
            up_proj = layer.mlp.up_proj
            gate_proj = layer.mlp.gate_proj
            if args.mlp_fuse:
                try:
                    # Move weights to CUDA device
                    w = torch.cat([
                        up_proj.weight.data.float(),
                        gate_proj.weight.data.float()
                    ], dim=0).to(utils.get_dev())
                    
                    # Apply activation-aware scaling (if enabled)
                    if args.act_aware:
                        scaling_diag_matrix = torch.ones(up_proj.in_features, device=utils.get_dev())  # avoid zero division
                        if hasattr(up_proj, "scaling_diag_matrix"):
                            scaling_diag_matrix *= up_proj.scaling_diag_matrix.to(utils.get_dev())**alpha
                            scaling_diag_matrix += 1e-6  # avoid zero division
                            scaling_matrix_inv = None
                            w = w * scaling_diag_matrix.view(1, -1)
                        elif hasattr(up_proj, "scaling_diag_matrixS"):
                            scaling_diag_matrix = up_proj.scaling_diag_matrixS.to(utils.get_dev())
                            try:
                                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                            except Exception as e:
                                logging.info("Warning: scaling_diag_matrix is not full rank!")
                                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                                scaling_matrix_inv = None
                            w = w @ scaling_diag_matrix.float()
                    
                    # SVD decomposition
                    U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False) # SVD decomposition of WS
                    V = Vt.T
                    
                    # Store SVD results
                    layer.mlp.up_proj.svd_info = {
                        'U': U.cpu(),
                        'S': S.cpu(),
                        'V': V.cpu()
                    }
                    print(f"Layer {idx} Up-proj SVD completed, S shape: {S.shape}")
                    
                except Exception as e:
                    print(f"Layer {idx} Up-proj SVD failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                try:
                    w = up_proj.weight.data.float().to(utils.get_dev())
                    if args.act_aware:
                        scaling_diag_matrix = torch.ones(up_proj.in_features, device=utils.get_dev())  # avoid zero division
                        if hasattr(up_proj, "scaling_diag_matrix"):
                            scaling_diag_matrix *= up_proj.scaling_diag_matrix.to(utils.get_dev())**alpha
                            scaling_diag_matrix += 1e-6  # avoid zero division
                            scaling_matrix_inv = None
                            w = w * scaling_diag_matrix.view(1, -1)
                        elif hasattr(up_proj, "scaling_diag_matrixS"):
                            scaling_diag_matrix = up_proj.scaling_diag_matrixS.to(utils.get_dev())
                            try:
                                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                            except Exception as e:
                                logging.info("Warning: scaling_diag_matrix is not full rank!")
                                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                                scaling_matrix_inv = None
                            w = w @ scaling_diag_matrix.float()
                    
                    U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False) # SVD decomposition of WS
                    V = Vt.T
                    
                    layer.mlp.up_proj.svd_info = {
                        'U': U.cpu(),
                        'S': S.cpu(),
                        'V': V.cpu()
                    }
                    print(f"Layer {idx} Up-proj SVD completed, S shape: {S.shape}")
                except Exception as e:
                    print(f"Layer {idx} Up-proj SVD failed: {e}")
                    import traceback
                    traceback.print_exc()
                try:
                    w = gate_proj.weight.data.float().to(utils.get_dev())
                    if args.act_aware:
                        scaling_diag_matrix = torch.ones(up_proj.in_features, device=utils.get_dev())  # avoid zero division
                        if hasattr(up_proj, "scaling_diag_matrix"):
                            scaling_diag_matrix *= up_proj.scaling_diag_matrix.to(utils.get_dev())**alpha
                            scaling_diag_matrix += 1e-6  # avoid zero division
                            scaling_matrix_inv = None
                            w = w * scaling_diag_matrix.view(1, -1)
                        elif hasattr(up_proj, "scaling_diag_matrixS"):
                            scaling_diag_matrix = up_proj.scaling_diag_matrixS.to(utils.get_dev())
                            try:
                                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                            except Exception as e:
                                logging.info("Warning: scaling_diag_matrix is not full rank!")
                                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                                scaling_matrix_inv = None
                            w = w @ scaling_diag_matrix.float()
                            
                    U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False) # SVD decomposition of WS
                    V = Vt.T
                    
                    layer.mlp.gate_proj.svd_info = {
                        'U': U.cpu(),
                        'S': S.cpu(),
                        'V': V.cpu()
                    }
                    print(f"Layer {idx} Gate-proj SVD completed, S shape: {S.shape}")
                except Exception as e:
                    print(f"Layer {idx} Gate-proj SVD failed: {e}")
                    import traceback
                    traceback.print_exc()

        if args.svd_modules in ['all', 'mlp', 'down']:
            down_proj = layer.mlp.down_proj
            try:
                w = down_proj.weight.data.float().to(utils.get_dev())
                if args.had_svd:
                    from hadamard_utils import get_hadK, matmul_hadU_cuda
                    
                    hadK_left, K_left = get_hadK(w.size(0))    
                    hadK_right, K_right = get_hadK(w.size(1))  
                    
                    w = matmul_hadU_cuda(w, hadK_right, K_right)           # W @ H_right^T
                    w = matmul_hadU_cuda(w.T, hadK_left, K_left).T        # H_left @ W (via transpose trick)

                if args.act_aware:
                    scaling_diag_matrix = torch.ones(down_proj.in_features, device=utils.get_dev())  # avoid zero division
                    if hasattr(down_proj, "scaling_diag_matrix"):
                        scaling_diag_matrix *= down_proj.scaling_diag_matrix.to(utils.get_dev())**alpha
                        scaling_diag_matrix += 1e-6  # avoid zero division 
                        scaling_matrix_inv = None
                        w = w * scaling_diag_matrix.view(1, -1)
                    elif hasattr(down_proj, "scaling_diag_matrixS"):
                        scaling_diag_matrix = down_proj.scaling_diag_matrixS.to(utils.get_dev())
                        try:
                            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                        except Exception as e:
                            logging.info("Warning: scaling_diag_matrix is not full rank!")
                            scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                            scaling_matrix_inv = None
                        w = w @ scaling_diag_matrix.float()
                    
                    U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False) # SVD decomposition of WS
                    V = Vt.T
                    
                    layer.mlp.down_proj.svd_info = {
                        'U': U.cpu(),
                        'S': S.cpu(),
                        'V': V.cpu()
                    }
                    print(f"Layer {idx} Down-proj SVD completed, S shape: {S.shape}")
            except Exception as e:
                print(f"Layer {idx} Down-proj SVD failed: {e}")
                import traceback
                traceback.print_exc()
    
    if args.svd_modules in ['qkv', 'attn', 'all']:
        print("QKV layer SVD preprocessing completed")
    if args.svd_modules in ['attn', 'all', 'o']:
        print("O-proj layer SVD preprocessing completed")
    if args.svd_modules in ['all', 'mlp', 'gaup', 'down']:
        print("MLP layer SVD preprocessing completed")

def svd_qkv_with_grad_info(layers, args, use_cache=True, cache_file=None):
    """
    Perform SVD decomposition on QKV layer fusion and utilize gradient information to construct S importance scores
    
    Args:
        layers: List of model layers
        args: Parameter configuration
        use_cache: Whether to use cache
        cache_file: Cache file path, automatically generated if None
    
    Returns:
        grad_scores_dict: Dictionary containing gradient importance scores for S in each layer
    """
    grad_alpha = args.grad_alpha
    # Automatically generate cache file path
    if args.cache_file is None:
        cache_dir  = args.act_cache_dir + "/cache"
        if hasattr(args, "cache_in_log") and args.cache_in_log:
            cache_dir = args.save_path + "/cache"
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = args.cache_file
        
    if args.qkv_fuse:
        fusion_mode = "qkv_fuse"
    elif args.kv_fuse:
        fusion_mode = "kv_fuse"
    else:
        fusion_mode = "separate"
    if args.cal_dataset == 'COCO_CALIB':
        fusion_mode = fusion_mode + "_coco"
    if args.misalign_text_image:
        fusion_mode = fusion_mode + "_misalign"
    if args.is_per_head_svd:
        fusion_mode = fusion_mode + "_per_head"
    # Add relevant information to cache file name
    calib_method_info = args.calib_method if hasattr(args, "act_aware") and args.act_aware else "no_act_aware"
    if args.cache_file is not None:
        cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_{calib_method_info}_{grad_alpha}_sigma{args.svd_modules}_{fusion_mode}_grad_scores.pt")
    else:
        cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_{args.nsamples}_{args.seed}_{calib_method_info}_{grad_alpha}_sigma{args.svd_modules}_{fusion_mode}_grad_scores.pt")

    # If cache exists and cache is enabled, load directly
    if os.path.exists(cache_file) and use_cache:
        logging.info(f"Loading gradient importance score cache from {cache_file}...")
        grad_scores_dict = torch.load(cache_file, map_location="cpu")
        logging.info("Successfully loaded gradient importance score cache!")
        # [FIXME: add clean svd_info here, since no computation needed]
        if args.svd_lm_localft:
            for idx, layer in enumerate(layers):
                if args.svd_modules in ['qkv', 'attn', 'all']:
                    if args.qkv_fuse:
                        del layer.self_attn.k_proj.qkv_svd_info
                    elif args.kv_fuse:
                        del layer.self_attn.q_proj.svd_info
                        del layer.self_attn.k_proj.svd_info
                    else:
                        del layer.self_attn.q_proj.svd_info
                        del layer.self_attn.k_proj.svd_info
                        del layer.self_attn.v_proj.svd_info
                if args.svd_modules in ['attn', 'all', 'o']:
                    del layer.self_attn.o_proj.svd_info
                if args.svd_modules in ['all', 'mlp', 'gaup']:
                    if args.mlp_fuse:
                        del layer.mlp.up_proj.svd_info
                    else:
                        del layer.mlp.up_proj.svd_info
                        del layer.mlp.gate_proj.svd_info
                if args.svd_modules in ['all', 'mlp', 'down']:
                    del layer.mlp.down_proj.svd_info
                logging.info(f"Successfully cleaned svd_info for layer {idx}!, quick fix for grad score cache existing")

    else:
        # Load gradient information cache file
        grad_info_cache_dir  = args.model + "/cache" # still have differenciate issue with VIT different quant configuration
        if hasattr(args, "cache_in_log") and args.cache_in_log:
            grad_info_cache_dir = args.save_path + "/cache"
        if args.qkv_fuse:
            fusion_mode = "qkv_fuse"
        elif args.kv_fuse:
            fusion_mode = "kv_fuse"
        else:
            fusion_mode = "qkv_separate"
        if args.use_S_gradinfo_init:
            fusion_mode += "_S_gradinfo_init"
        if args.cal_dataset == 'COCO_CALIB':
            fusion_mode = fusion_mode + "_coco"
        if args.misalign_text_image:
            fusion_mode = fusion_mode + "_misalign"
        if args.is_per_head_svd:
            fusion_mode = fusion_mode + "_per_head"
        # Build gradient information cache file path
        if hasattr(args, "a_clip_ratio") and args.a_clip_ratio == 1.0:
            grad_info_cache = os.path.join(grad_info_cache_dir, f"{args.model.replace('/','_')}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma{args.svd_modules}_{fusion_mode}_sigma_grad_info.pt")
        else:
            grad_info_cache = os.path.join(grad_info_cache_dir, f"{args.model.replace('/','_')}_aclip{args.a_clip_ratio}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma{args.svd_modules}_{fusion_mode}_sigma_grad_info.pt")
        
        # Check if gradient information cache exists
        if os.path.exists(grad_info_cache):
            logging.info(f"Loading gradient information from {grad_info_cache}...")
            all_grad_info = torch.load(grad_info_cache, map_location="cpu")
            
            # Load gradient information into corresponding layers
            for idx, layer in enumerate(layers):
                layer_key = f"layer_{idx}"
                if layer_key in all_grad_info:
                    if not hasattr(layer.self_attn.k_proj, 'S_grad_info'):
                        if args.svd_modules in ['qkv', 'attn', 'all', 'all_sep_fine', 'all_sep_fine_wo_down', 'all_sep']:
                            if args.qkv_fuse:
                                layer.self_attn.k_proj.S_grad_info = all_grad_info[layer_key]['k_proj'].to(utils.get_dev())
                            elif args.kv_fuse:
                                # Load Q separately and KV together
                                if 'q_proj' in all_grad_info[layer_key]:
                                    layer.self_attn.q_proj.S_grad_info = all_grad_info[layer_key]['q_proj'].to(utils.get_dev())
                                if 'k_proj' in all_grad_info[layer_key]:
                                    layer.self_attn.k_proj.S_grad_info = all_grad_info[layer_key]['k_proj'].to(utils.get_dev())
                            else:
                                # Separate Q, K, V
                                for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                                    if proj_name in all_grad_info[layer_key]:
                                        proj = getattr(layer.self_attn, proj_name)
                                        proj.S_grad_info = all_grad_info[layer_key][proj_name].to(utils.get_dev())
                        if args.svd_modules in ['attn', 'all', 'o', 'all_sep_fine', 'all_sep_fine_wo_down', 'all_sep']:
                            layer.self_attn.o_proj.S_grad_info = all_grad_info[layer_key]['o_proj'].to(utils.get_dev())
                        if args.svd_modules in ['all', 'mlp', 'gaup', 'all_sep_fine', 'all_sep_fine_wo_down', 'all_sep']:
                            if args.mlp_fuse:
                                layer.mlp.up_proj.S_grad_info = all_grad_info[layer_key]['up_proj'].to(utils.get_dev())
                            else:
                                layer.mlp.up_proj.S_grad_info = all_grad_info[layer_key]['up_proj'].to(utils.get_dev())
                                layer.mlp.gate_proj.S_grad_info = all_grad_info[layer_key]['gate_proj'].to(utils.get_dev())
                        if args.svd_modules in ['mlp', 'all', 'down', 'all_sep_fine', 'all_sep_fine_wo_down', 'all_sep']:
                            layer.mlp.down_proj.S_grad_info = all_grad_info[layer_key]['down_proj'].to(utils.get_dev())
                            # here we still apply rank allocation to down proj, just no ft? in all_sep_fine_wo_down mode
            logging.info("Successfully loaded gradient information!")
        
        # Directly use pre-computed S gradient information
        grad_scores_dict = {}
        device = utils.get_dev()  # Get CUDA device
        for idx, layer in enumerate(layers):
            layer_key = f"layer_{idx}"
            grad_scores_dict[layer_key] = {}
            if args.svd_modules in ['qkv', 'attn', 'all']:
                if args.qkv_fuse:
                    # QKV fusion case (existing code)
                    if hasattr(layer.self_attn.k_proj, 'qkv_svd_info') and hasattr(layer.self_attn.k_proj, 'S_grad_info'):
                        svd_info = layer.self_attn.k_proj.qkv_svd_info
                        S = svd_info['S']
                        S_grad = layer.self_attn.k_proj.S_grad_info
                        
                        # Ensure S and S_grad are on the same device (both moved to CUDA)
                        S = svd_info['S'].to(device).to(torch.bfloat16)
                        S_grad = layer.self_attn.k_proj.S_grad_info.to(device).to(torch.bfloat16)
                        
                        # Calculate importance score: |S| * |S_grad|
                        # if args.is_per_head_svd:
                        #     # S = n, c
                        #     # s_grad = n,c
                        #     importance_score = torch.abs(S) * (torch.abs(S_grad)**grad_alpha) # n, c
                        # else:
                        importance_score = torch.abs(S) * (torch.abs(S_grad)**grad_alpha)
                        
                        # Move result back to CPU for saving
                        grad_scores_dict[layer_key]['k_proj'] = importance_score.cpu()
                        grad_scores_dict[layer_key]['k_proj_S'] = S.cpu()
                        if args.use_S_gradinfo_init:
                            layer.self_attn.k_proj.S_grad_info = importance_score.cpu()
                        logging.info(f"Layer {idx} QKV importance score computed, shape: {importance_score.shape}")
                        if args.is_rank_allocate_ft:
                            del layer.self_attn.k_proj.qkv_svd_info
                    else:
                        logging.info(f"Warning: Layer {idx} lacks necessary SVD information or gradient information, cannot compute importance score")
                
                elif args.kv_fuse:
                    # KV fusion case (NEW)
                    # Handle Q separately
                    if hasattr(layer.self_attn.q_proj, 'svd_info') and hasattr(layer.self_attn.q_proj, 'S_grad_info'):
                        svd_info = layer.self_attn.q_proj.svd_info
                        S = svd_info['S'].to(device).to(torch.bfloat16)
                        S_grad = layer.self_attn.q_proj.S_grad_info.to(device).to(torch.bfloat16)
                        
                        # Calculate importance score for Q
                        importance_score = torch.abs(S) * (torch.abs(S_grad)**grad_alpha)
                        grad_scores_dict[layer_key]['q_proj'] = importance_score.cpu()
                        grad_scores_dict[layer_key]['q_proj_S'] = S.cpu()
                        if args.use_S_gradinfo_init:
                            layer.self_attn.q_proj.S_grad_info = importance_score.cpu()
                        logging.info(f"Layer {idx} Q importance score computed, shape: {importance_score.shape}")
                        if args.is_rank_allocate_ft:
                            del layer.self_attn.q_proj.svd_info
                    else:
                        logging.info(f"Warning: Layer {idx} Q lacks necessary SVD information or gradient information, cannot compute importance score")
                    
                    # Handle KV fusion
                    if hasattr(layer.self_attn.k_proj, 'svd_info') and hasattr(layer.self_attn.k_proj, 'S_grad_info'):
                        svd_info = layer.self_attn.k_proj.svd_info
                        S = svd_info['S'].to(device).to(torch.bfloat16)
                        S_grad = layer.self_attn.k_proj.S_grad_info.to(device).to(torch.bfloat16)
                        
                        # Calculate importance score for KV
                        importance_score = torch.abs(S) * (torch.abs(S_grad)**grad_alpha)
                        grad_scores_dict[layer_key]['k_proj'] = importance_score.cpu()
                        grad_scores_dict[layer_key]['k_proj_S'] = S.cpu()
                        if args.use_S_gradinfo_init:
                            layer.self_attn.k_proj.S_grad_info = importance_score.cpu()
                        logging.info(f"Layer {idx} KV importance score computed, shape: {importance_score.shape}")
                        if args.is_rank_allocate_ft:
                            del layer.self_attn.k_proj.svd_info
                    else:
                        logging.info(f"Warning: Layer {idx} KV lacks necessary SVD information or gradient information, cannot compute importance score")
                
                else:
                    # Separate Q, K, V case (NEW)
                    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                        proj = getattr(layer.self_attn, proj_name)
                        if hasattr(proj, 'svd_info') and hasattr(proj, 'S_grad_info'):
                            svd_info = proj.svd_info
                            S = svd_info['S'].to(device).to(torch.bfloat16)
                            S_grad = proj.S_grad_info.to(device).to(torch.bfloat16)
                            
                            # Calculate importance score
                            importance_score = torch.abs(S) * (torch.abs(S_grad)**grad_alpha)
                            grad_scores_dict[layer_key][proj_name] = importance_score.cpu()
                            grad_scores_dict[layer_key][f'{proj_name}_S'] = S.cpu()
                            if args.use_S_gradinfo_init:
                                proj.S_grad_info = importance_score.cpu()
                            print(f"Layer {idx} {proj_name} importance score computed, shape: {importance_score.shape}")
                            if args.is_rank_allocate_ft:
                                del proj.svd_info
                        else:
                            print(f"Warning: Layer {idx} {proj_name} lacks necessary SVD information or gradient information, cannot compute importance score")
            if args.svd_modules in ['attn', 'all', 'o']: # [FIXME: add exist attr check as did in qkv]
                svd_info = layer.self_attn.o_proj.svd_info
                S = svd_info['S'].to(device).to(torch.bfloat16)
                S_grad = layer.self_attn.o_proj.S_grad_info.to(device).to(torch.bfloat16)
                importance_score = (torch.abs(S_grad)**grad_alpha) #importance_score = torch.abs(S) * (torch.abs(S_grad)**grad_alpha)
                grad_scores_dict[layer_key]['o_proj'] = importance_score.cpu()
                grad_scores_dict[layer_key]['o_proj_S'] = S.cpu()
                if args.use_S_gradinfo_init:
                    layer.self_attn.o_proj.S_grad_info = importance_score.cpu()
                if args.is_rank_allocate_ft:
                    del layer.self_attn.o_proj.svd_info
                print(f"Layer {idx} O-proj importance score computed, shape: {importance_score.shape}")

            if args.svd_modules in ['all', 'mlp', 'gaup']:
                if args.mlp_fuse:
                    svd_info = layer.mlp.up_proj.svd_info
                    S = svd_info['S'].to(device).to(torch.bfloat16)
                    S_grad = layer.mlp.up_proj.S_grad_info.to(device).to(torch.bfloat16)
                    importance_score = (torch.abs(S_grad)**grad_alpha) #importance_score = torch.abs(S) * (torch.abs(S_grad)**grad_alpha)
                    grad_scores_dict[layer_key]['up_proj'] = importance_score.cpu()
                    grad_scores_dict[layer_key]['up_proj_S'] = S.cpu()
                    if args.use_S_gradinfo_init:
                        layer.mlp.up_proj.S_grad_info = importance_score.cpu()
                    print(f"Layer {idx} Up-proj importance score computed, shape: {importance_score.shape}")
                    if args.is_rank_allocate_ft:
                        del layer.mlp.up_proj.svd_info
                else:
                    svd_info = layer.mlp.up_proj.svd_info
                    S = svd_info['S'].to(device).to(torch.bfloat16)
                    S_grad = layer.mlp.up_proj.S_grad_info.to(device).to(torch.bfloat16)
                    importance_score = (torch.abs(S_grad)**grad_alpha) #importance_score = torch.abs(S) * (torch.abs(S_grad)**grad_alpha)
                    grad_scores_dict[layer_key]['up_proj'] = importance_score.cpu()
                    grad_scores_dict[layer_key]['up_proj_S'] = S.cpu()
                    if args.use_S_gradinfo_init:
                        layer.mlp.up_proj.S_grad_info = importance_score.cpu()
                    print(f"Layer {idx} Up-proj importance score computed, shape: {importance_score.shape}")
                    if args.is_rank_allocate_ft:
                        del layer.mlp.up_proj.svd_info
                    svd_info = layer.mlp.gate_proj.svd_info
                    S = svd_info['S'].to(device).to(torch.bfloat16)
                    S_grad = layer.mlp.gate_proj.S_grad_info.to(device).to(torch.bfloat16)
                    importance_score = (torch.abs(S_grad)**grad_alpha) #importance_score = torch.abs(S) * (torch.abs(S_grad)**grad_alpha)
                    grad_scores_dict[layer_key]['gate_proj'] = importance_score.cpu()
                    grad_scores_dict[layer_key]['gate_proj_S'] = S.cpu()
                    if args.use_S_gradinfo_init:
                        layer.mlp.gate_proj.S_grad_info = importance_score.cpu()
                    print(f"Layer {idx} Gate-proj importance score computed, shape: {importance_score.shape}")
                    if args.is_rank_allocate_ft:
                        del layer.mlp.gate_proj.svd_info
            if args.svd_modules in ['mlp', 'all', 'down']:
                svd_info = layer.mlp.down_proj.svd_info
                S = svd_info['S'].to(device).to(torch.bfloat16)
                S_grad = layer.mlp.down_proj.S_grad_info.to(device).to(torch.bfloat16)
                importance_score = (torch.abs(S_grad)**grad_alpha) #importance_score = torch.abs(S) * (torch.abs(S_grad)**grad_alpha)
                grad_scores_dict[layer_key]['down_proj'] = importance_score.cpu()
                grad_scores_dict[layer_key]['down_proj_S'] = S.cpu()
                if args.use_S_gradinfo_init:
                    layer.mlp.down_proj.S_grad_info = importance_score.cpu()
                if args.is_rank_allocate_ft:
                    del layer.mlp.down_proj.svd_info
                print(f"Layer {idx} Down-proj importance score computed, shape: {importance_score.shape}")
                    

        # # Visualize gradient importance score distribution
        # visualize_score_distribution(grad_scores_dict, 
        #                             save_path=os.path.join(cache_dir, f"{args.model.replace('/','_')}_sigma*grad_scores_dist.png"),
        #                             plot_type='boxplot')
        
        # # Visualize histogram of scores for each layer
        # visualize_layer_score_histograms(grad_scores_dict,
        #                                save_path=os.path.join(cache_dir, f"{args.model.replace('/','_')}_sigma*grad_scores_hist.png"))
        
        # Save gradient importance score cache
        logging.info(f"Saving gradient importance score cache to {cache_file}...")
        torch.save(grad_scores_dict, cache_file)
        logging.info("Gradient importance score cache saved successfully!")
    
    if args.use_S_gradinfo_init:
        return # only log the S_grad_info as importance score for pruning
    # Get indices and scores of top k important singular values
    num_layers = len(layers)
        
    if args.svd_modules == 'attn':
        if args.use_true_param_ratio:
            if args.is_per_head_svd:
                num_heads = layers[0].self_attn.config.num_key_value_heads if not args.is_q_headnum else layers[0].self_attn.config.num_attention_heads
                hidden_size =[(layers[0].self_attn.q_proj.weight.numel()/num_heads + layers[0].self_attn.k_proj.weight.numel()/num_heads  + layers[0].self_attn.v_proj.weight.numel()/num_heads)/ (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.q_proj.out_features/num_heads  + layers[0].self_attn.k_proj.out_features/num_heads + layers[0].self_attn.v_proj.out_features/num_heads ),\
                            layers[0].self_attn.o_proj.in_features * layers[0].self_attn.o_proj.out_features / (layers[0].self_attn.o_proj.in_features + layers[0].self_attn.o_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            else:
                hidden_size =[(layers[0].self_attn.q_proj.weight.numel() + layers[0].self_attn.k_proj.weight.numel() + layers[0].self_attn.v_proj.weight.numel())/ (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.q_proj.out_features + layers[0].self_attn.k_proj.out_features+ layers[0].self_attn.v_proj.out_features),\
                            layers[0].self_attn.o_proj.in_features * layers[0].self_attn.o_proj.out_features / (layers[0].self_attn.o_proj.in_features + layers[0].self_attn.o_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
        else:
            hidden_size = [layers[0].self_attn.k_proj.in_features, layers[0].self_attn.o_proj.in_features]
            total_rank = [num_layers * hsz for hsz in hidden_size]
            k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
        top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['k_proj','o_proj'], is_per_head_svd=args.is_per_head_svd)
    elif args.svd_modules == 'all':
        if args.mlp_fuse:
            if args.use_param_ratio:
                # hidden_size =[max(layers[0].self_attn.k_proj.in_features , layers[0].self_attn.k_proj.out_features) / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features), \
                #     max(layers[0].self_attn.o_proj.in_features , layers[0].self_attn.o_proj.out_features) / (layers[0].self_attn.o_proj.in_features + layers[0].self_attn.o_proj.out_features), \
                #     max(layers[0].mlp.up_proj.in_features , layers[0].mlp.up_proj.out_features) / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features), \
                #     max(layers[0].mlp.down_proj.in_features , layers[0].mlp.down_proj.out_features) / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
                hidden_size =[layers[0].self_attn.k_proj.in_features * layers[0].self_attn.k_proj.out_features / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features), \
                    layers[0].self_attn.o_proj.in_features * layers[0].self_attn.o_proj.out_features / (layers[0].self_attn.o_proj.in_features + layers[0].self_attn.o_proj.out_features), \
                    layers[0].mlp.up_proj.in_features * layers[0].mlp.up_proj.out_features / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features), \
                    layers[0].mlp.down_proj.in_features * layers[0].mlp.down_proj.out_features / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            elif args.use_true_param_ratio:
                if args.is_per_head_svd:
                    logging.info("per head svd is not supported for all module for now")
                hidden_size =[(layers[0].self_attn.q_proj.weight.numel() + layers[0].self_attn.k_proj.weight.numel() + layers[0].self_attn.v_proj.weight.numel())/ (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.q_proj.out_features + layers[0].self_attn.k_proj.out_features+ layers[0].self_attn.v_proj.out_features), \
                    layers[0].self_attn.o_proj.in_features * layers[0].self_attn.o_proj.out_features / (layers[0].self_attn.o_proj.in_features + layers[0].self_attn.o_proj.out_features), \
                    layers[0].mlp.up_proj.weight.numel() * 2 / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features + layers[0].mlp.gate_proj.out_features), \
                    layers[0].mlp.down_proj.in_features * layers[0].mlp.down_proj.out_features / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            else:
                hidden_size = [layers[0].self_attn.k_proj.in_features, layers[0].self_attn.o_proj.in_features, layers[0].mlp.up_proj.in_features, layers[0].mlp.down_proj.out_features]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
            top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['k_proj','o_proj','up_proj','down_proj'])
        else:
            if args.use_param_ratio or args.use_true_param_ratio:
                hidden_size = [layers[0].self_attn.k_proj.in_features * layers[0].self_attn.k_proj.out_features / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features), \
                    layers[0].self_attn.o_proj.in_features * layers[0].self_attn.o_proj.out_features / (layers[0].self_attn.o_proj.in_features + layers[0].self_attn.o_proj.out_features), \
                    layers[0].mlp.up_proj.in_features * layers[0].mlp.up_proj.out_features / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features), \
                    layers[0].mlp.gate_proj.in_features * layers[0].mlp.gate_proj.out_features / (layers[0].mlp.gate_proj.in_features + layers[0].mlp.gate_proj.out_features), \
                    layers[0].mlp.down_proj.in_features * layers[0].mlp.down_proj.out_features / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            else:
                hidden_size = [layers[0].self_attn.k_proj.in_features, layers[0].self_attn.o_proj.in_features, layers[0].mlp.up_proj.in_features, layers[0].mlp.gate_proj.in_features, layers[0].mlp.down_proj.out_features]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
            top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['k_proj','o_proj','up_proj','gate_proj','down_proj'])
    elif args.svd_modules == 'mlp':
        if args.mlp_fuse:
            if args.use_param_ratio:
                hidden_size = [layers[0].mlp.up_proj.in_features * layers[0].mlp.up_proj.out_features / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features), \
                    layers[0].mlp.down_proj.in_features * layers[0].mlp.down_proj.out_features / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            elif args.use_true_param_ratio:
                hidden_size = [layers[0].mlp.up_proj.weight.numel() * 2 / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features + layers[0].mlp.gate_proj.out_features), \
                    layers[0].mlp.down_proj.in_features * layers[0].mlp.down_proj.out_features / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            else:
                hidden_size = [layers[0].mlp.up_proj.in_features, layers[0].mlp.down_proj.out_features]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
            top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['up_proj','down_proj'])
        else:
            if args.use_param_ratio or args.use_true_param_ratio:
                hidden_size = [layers[0].mlp.up_proj.in_features * layers[0].mlp.up_proj.out_features / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features), \
                    layers[0].mlp.gate_proj.in_features * layers[0].mlp.gate_proj.out_features / (layers[0].mlp.gate_proj.in_features + layers[0].mlp.gate_proj.out_features), \
                    layers[0].mlp.down_proj.in_features * layers[0].mlp.down_proj.out_features / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            else:
                hidden_size = [layers[0].mlp.up_proj.in_features, layers[0].mlp.gate_proj.in_features, layers[0].mlp.down_proj.out_features]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
            top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['up_proj','gate_proj','down_proj'])
    elif args.svd_modules == 'gaup':
        if args.mlp_fuse:
            if args.use_param_ratio:
                hidden_size = [layers[0].mlp.up_proj.in_features * layers[0].mlp.up_proj.out_features / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            elif args.use_true_param_ratio:
                hidden_size = [layers[0].mlp.up_proj.weight.numel() * 2 / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features + layers[0].mlp.gate_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            else:
                hidden_size = [layers[0].mlp.up_proj.in_features]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
            top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['up_proj'])
        else:
            if args.use_param_ratio:
                hidden_size = [layers[0].mlp.up_proj.in_features * layers[0].mlp.up_proj.out_features / (layers[0].mlp.up_proj.in_features + layers[0].mlp.up_proj.out_features), \
                    layers[0].mlp.gate_proj.in_features * layers[0].mlp.gate_proj.out_features / (layers[0].mlp.gate_proj.in_features + layers[0].mlp.gate_proj.out_features)]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            else:
                hidden_size = [layers[0].mlp.up_proj.in_features, layers[0].mlp.gate_proj.in_features]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
            top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['up_proj','gate_proj'])
    elif args.svd_modules == 'o':
        hidden_size = [layers[0].self_attn.o_proj.in_features]
        total_rank = [num_layers * hsz for hsz in hidden_size]
        k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
        top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['o_proj'])
    elif args.svd_modules == 'down':
        if args.use_param_ratio or args.use_true_param_ratio:
            hidden_size = [layers[0].mlp.down_proj.in_features * layers[0].mlp.down_proj.out_features / (layers[0].mlp.down_proj.in_features + layers[0].mlp.down_proj.out_features)]
            total_rank = [num_layers * hsz for hsz in hidden_size]
            k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
        else:
            hidden_size = [layers[0].mlp.down_proj.out_features]
            total_rank = [num_layers * hsz for hsz in hidden_size]
            k_value = [int(args.rank_ratio/2 * t_rank) for t_rank in total_rank]
        top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['down_proj'])
    else: # qkv fuse
        if args.qkv_fuse:
            if args.use_true_param_ratio:
                if args.is_per_head_svd:
                    logging.info("added per head ratio compute here")
                    num_heads = layers[0].self_attn.config.num_key_value_heads  if not args.is_q_headnum else layers[0].self_attn.config.num_attention_heads
                    hidden_size =[(layers[0].self_attn.q_proj.weight.numel() + layers[0].self_attn.k_proj.weight.numel() + layers[0].self_attn.v_proj.weight.numel())/ (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.q_proj.out_features/num_heads + layers[0].self_attn.k_proj.out_features/num_heads + layers[0].self_attn.v_proj.out_features/num_heads)/num_heads]
                    total_rank = [num_layers * hsz for hsz in hidden_size]
                    k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
                else:
                    hidden_size =[(layers[0].self_attn.q_proj.weight.numel() + layers[0].self_attn.k_proj.weight.numel() + layers[0].self_attn.v_proj.weight.numel())/ (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.q_proj.out_features + layers[0].self_attn.k_proj.out_features+ layers[0].self_attn.v_proj.out_features)]
                    total_rank = [num_layers * hsz for hsz in hidden_size]
                    k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
                top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['k_proj'], is_per_head_svd=args.is_per_head_svd)
            elif args.use_all_true_param_ratio:
                if args.is_per_head_svd:
                    logging.info('adding all true p ratio then assign to each head')
                    num_heads = layers[0].self_attn.config.num_key_value_heads
                    hidden_size =[(layers[0].self_attn.q_proj.weight.numel() + layers[0].self_attn.k_proj.weight.numel() + layers[0].self_attn.v_proj.weight.numel())/ (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.q_proj.out_features + layers[0].self_attn.k_proj.out_features + layers[0].self_attn.v_proj.out_features)/num_heads]
                    total_rank = [num_layers * hsz for hsz in hidden_size]
                    k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
                else:
                    hidden_size =[(layers[0].self_attn.q_proj.weight.numel() + layers[0].self_attn.k_proj.weight.numel() + layers[0].self_attn.v_proj.weight.numel())/ (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.q_proj.out_features + layers[0].self_attn.k_proj.out_features+ layers[0].self_attn.v_proj.out_features)]
                    total_rank = [num_layers * hsz for hsz in hidden_size]
                    k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
                top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['k_proj'], is_per_head_svd=args.is_per_head_svd)

            else:
                hidden_size = layers[0].self_attn.k_proj.in_features
                total_rank = num_layers * hidden_size
                k_value = int(args.rank_ratio/2 * total_rank)
                top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=[k_value], keys=['k_proj'])
        
        elif args.kv_fuse:
            if args.use_true_param_ratio:
                if args.is_per_head_svd:
                    logging.info("per head ratio is not supported for kv fuse for now")
                # Calculate parameter ratios for Q and KV separately
                q_param_ratio = layers[0].self_attn.q_proj.weight.numel() / (layers[0].self_attn.q_proj.in_features + layers[0].self_attn.q_proj.out_features)
                kv_param_ratio = (layers[0].self_attn.k_proj.weight.numel() + layers[0].self_attn.v_proj.weight.numel()) / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features + layers[0].self_attn.v_proj.out_features)
                hidden_size = [q_param_ratio, kv_param_ratio]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
                top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['q_proj', 'k_proj'])
            else:
                hidden_size = layers[0].self_attn.k_proj.in_features
                total_rank = num_layers * hidden_size
                q_k_value = int(args.rank_ratio/2 * total_rank)  
                kv_k_value = int(args.rank_ratio/2 * total_rank)  
                k_value = [q_k_value, kv_k_value]
                top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['q_proj', 'k_proj'])
        
        else:  # Separate Q, K, V processing
            if args.use_true_param_ratio:
                # Calculate parameter ratios for Q, K, V separately
                if args.is_per_head_svd:
                    num_q_heads = layers[0].self_attn.config.num_attention_heads
                    num_heads = layers[0].self_attn.config.num_key_value_heads
                    q_param_ratio = layers[0].self_attn.q_proj.weight.numel() / (layers[0].self_attn.q_proj.in_features + layers[0].self_attn.q_proj.out_features/num_q_heads)/num_q_heads
                    k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features/num_heads)/num_heads
                    v_param_ratio = layers[0].self_attn.v_proj.weight.numel() / (layers[0].self_attn.v_proj.in_features + layers[0].self_attn.v_proj.out_features/num_heads)/num_heads
                else:
                    q_param_ratio = layers[0].self_attn.q_proj.weight.numel() / (layers[0].self_attn.q_proj.in_features + layers[0].self_attn.q_proj.out_features)
                    k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features)
                    v_param_ratio = layers[0].self_attn.v_proj.weight.numel() / (layers[0].self_attn.v_proj.in_features + layers[0].self_attn.v_proj.out_features)
                hidden_size = [q_param_ratio, k_param_ratio, v_param_ratio]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
                top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['q_proj', 'k_proj', 'v_proj'], is_per_head_svd=args.is_per_head_svd)
            elif args.use_all_true_param_ratio:
                # Calculate parameter ratios for Q, K, V separately
                if args.is_per_head_svd:
                    num_q_heads = layers[0].self_attn.config.num_attention_heads
                    num_heads = layers[0].self_attn.config.num_key_value_heads
                    q_param_ratio = layers[0].self_attn.q_proj.weight.numel() / (layers[0].self_attn.q_proj.in_features + layers[0].self_attn.q_proj.out_features)/num_q_heads
                    k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features)/num_heads
                    v_param_ratio = layers[0].self_attn.v_proj.weight.numel() / (layers[0].self_attn.v_proj.in_features + layers[0].self_attn.v_proj.out_features)/num_heads
                else:
                    q_param_ratio = layers[0].self_attn.q_proj.weight.numel() / (layers[0].self_attn.q_proj.in_features + layers[0].self_attn.q_proj.out_features)
                    k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features)
                    v_param_ratio = layers[0].self_attn.v_proj.weight.numel() / (layers[0].self_attn.v_proj.in_features + layers[0].self_attn.v_proj.out_features)
                hidden_size = [q_param_ratio, k_param_ratio, v_param_ratio]
                total_rank = [num_layers * hsz for hsz in hidden_size]
                k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
                top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['q_proj', 'k_proj', 'v_proj'], is_per_head_svd=args.is_per_head_svd)
            else:
                hidden_size = layers[0].self_attn.k_proj.in_features
                total_rank = num_layers * hidden_size
                qkv_k_value = int(args.rank_ratio/3 * total_rank)
                k_value = [qkv_k_value, qkv_k_value, qkv_k_value]
                top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['q_proj', 'k_proj', 'v_proj'])
    
    for key, indices in top_indices.items():
        logging.info(f"Selected top {len(indices)} important singular values for {key}")
    return top_indices, top_scores, layer_indices_dict

# Add the following functions at the end of the file

def get_top_k_scores(grad_scores_dict, k_list, keys=['k_proj'], is_per_head_svd=False):
    """
    Get indices and scores of top k important singular values for each key across all layers.
    
    Args:
        grad_scores_dict: dict of {layer_idx: {key: tensor of scores}}
        k_list: list of integers, top-k values for each key
        keys: list of keys to consider (e.g., ['k_proj', 'o_proj'])
    
    Returns:
        top_indices: dict of {key: list of (layer_idx, sv_idx)}
        top_scores: dict of {key: list of scores}
        layer_indices_dict: dict of {layer_idx: {key: [sv_idx, ...]}}
    """
    top_indices = {}
    top_scores = {}
    layer_indices_dict = {}
    assert len(keys) == len(k_list), "keys and k_list must have the same length" # [FIXED: num * [v] -> [num * v]]
    for key, k in zip(keys, k_list):
        all_scores = []
        if is_per_head_svd and key in ['k_proj', 'q_proj', 'v_proj']:
            logging.info("added per head score here")
            for layer_idx, scores in grad_scores_dict.items():
                if key not in scores:
                    print(f"Layer {layer_idx} has no {key} grad scores")
                    continue
                layer_num = int(layer_idx.split('_')[1])
                score = scores[key]
                # score_ = score # n, c
                # logging.info(f'score shape {score.shape}') # [384?]
                head_num = score.shape[0]
                for head_idx in range(score.shape[0]):
                    for i, score_ in enumerate(score[head_idx]):
                        all_scores.append((layer_num, i, head_idx, score_.item()))
        else:
            for layer_idx, scores in grad_scores_dict.items():
                if key not in scores:
                    print(f"Layer {layer_idx} has no {key} grad scores")
                    continue
                layer_num = int(layer_idx.split('_')[1])
                for i, score in enumerate(scores[key]):
                    all_scores.append((layer_num, i, score.item()))
        
        # Sort and get top-k
        all_scores.sort(key=lambda x: x[-1], reverse=True)
        # [TODO: only support heterogenous per head svd rank allocation for now]
        if is_per_head_svd and key in ['k_proj', 'q_proj', 'v_proj']:
            logging.info(f'{key} original for per head svd is {k}')
            k = k * head_num # [FIXME: check head_num use kv_headnum to accomodate GQA]
            logging.info(f'scale {key} by head_num: {head_num} to k: {k}')
        top_k = all_scores[:k]

        # Store top indices and scores
        if is_per_head_svd and key in ['k_proj', 'q_proj', 'v_proj']:
            top_indices[key] = [(layer, idx, head_idx) for layer, idx, head_idx, _ in top_k] # [TODO:]here need some quick fix to support perhead and other global svd rank allocation
            top_scores[key] = [score for _, _, _, score in top_k]
            # Fill per-layer per-head dictionary
            for layer_idx, sv_idx, head_idx in top_indices[key]:
                if layer_idx not in layer_indices_dict:
                    layer_indices_dict[layer_idx] = {}
                if key not in layer_indices_dict[layer_idx]:
                    layer_indices_dict[layer_idx][key] = {}
                if head_idx not in layer_indices_dict[layer_idx][key]:
                    layer_indices_dict[layer_idx][key][head_idx] = []
                layer_indices_dict[layer_idx][key][head_idx].append(sv_idx)
        else:
            top_indices[key] = [(layer, idx) for layer, idx, _ in top_k] 
            top_scores[key] = [score for _, _, score in top_k]

            # Fill per-layer dictionary
            for layer_idx, sv_idx in top_indices[key]:
                if layer_idx not in layer_indices_dict:
                    layer_indices_dict[layer_idx] = {}
                if key not in layer_indices_dict[layer_idx]:
                    layer_indices_dict[layer_idx][key] = []
                layer_indices_dict[layer_idx][key].append(sv_idx)

    return top_indices, top_scores, layer_indices_dict



def visualize_score_distribution(grad_scores_dict, save_path=None, plot_type='boxplot'):
    """
    Visualize the distribution of gradient importance scores
    
    Args:
        grad_scores_dict: Dictionary containing gradient importance scores for each layer
        save_path: Path to save the image
        plot_type: Plot type, 'boxplot' or 'violin'
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Collect scores from all layers
        layer_scores = []
        layer_names = []
        
        for layer_name, scores in grad_scores_dict.items():
            layer_scores.append(scores.cpu().numpy())
            layer_names.append(layer_name)
        
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'boxplot':
            plt.boxplot(layer_scores, labels=layer_names)
            plt.title('Gradient Importance Score Distribution (Box Plot)')
        elif plot_type == 'violin':
            sns.violinplot(data=layer_scores)
            plt.xticks(range(len(layer_names)), layer_names)
            plt.title('Gradient Importance Score Distribution (Violin Plot)')
        
        plt.xlabel('Layer')
        plt.ylabel('Importance Score')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Distribution plot saved to {save_path}")
        
        plt.close()
        
    except ImportError:
        print("Cannot import matplotlib or seaborn, skipping visualization")
    except Exception as e:
        print(f"Error during visualization: {e}")

def visualize_layer_score_histograms(grad_scores_dict, save_path=None, max_layers=16):
    """
    Draw histograms of importance scores for each layer
    
    Args:
        grad_scores_dict: Dictionary containing gradient importance scores for each layer
        save_path: Path to save the image
        max_layers: Maximum number of layers to display
    """
    try:
        import matplotlib.pyplot as plt
        
        # Limit the number of layers to display
        layer_names = list(grad_scores_dict.keys())[:max_layers]
        n_layers = len(layer_names)
        
        # Calculate subplot layout
        n_cols = min(4, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 3 * n_rows))
        
        for i, layer_name in enumerate(layer_names):
            scores = grad_scores_dict[layer_name].cpu().numpy()
            
            plt.subplot(n_rows, n_cols, i + 1)
            plt.hist(scores, bins=50)
            plt.title(f'{layer_name}')
            plt.xlabel('Importance Score')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Histogram saved to {save_path}")
        
        plt.close()
        
    except ImportError:
        print("Cannot import matplotlib, skipping visualization")
    except Exception as e:
        print(f"Error during visualization: {e}")


def insert_ignore_index_after_prompt(input_ids, output_ids, image_token_id=32000, ignore_index=-100):
    """
    In output_ids, after the prompt part and before the image token part,
    insert the corresponding number of ignore_index (-100) for masking during loss calculation.

    Args:
        input_ids (torch.Tensor): shape (seq_len,)
        output_ids (torch.Tensor): shape (seq_len,)
        image_token_id (int): image placeholder token id, default 32000
        ignore_index (int): marker to be ignored by CrossEntropyLoss, default -100

    Returns:
        torch.Tensor: processed output_ids with ignore_index segment
    """
    # Find the position of the first <image>
    image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)
    if len(image_positions[0]) == 0:
        # No image token, return original output_ids
        return output_ids.clone()

    first_image_idx = image_positions[0][0].item()
    num_image_tokens = (input_ids == image_token_id).sum().item()

    # Split prompt and remaining parts
    prompt_output_ids = output_ids[:first_image_idx]
    rest_output_ids = output_ids[first_image_idx:]

    # Construct ignore_index segment
    ignore_prefix = torch.full((num_image_tokens,), ignore_index, dtype=output_ids.dtype, device=output_ids.device)

    # Concatenate
    final_output_ids = torch.cat([prompt_output_ids, ignore_prefix, rest_output_ids], dim=0)

    return final_output_ids

def taylor_like_s_grad_squared(S, S_grad, n, add_taylor_first=False):
    """
    Compute generalized S_grad_squared up to order n using:
    sum_{k=0}^{n} [2 * (-1)^k / (k+2)!] * S^k * S_grad^{k+2}
    """
    n = n - 2
    result = torch.zeros_like(S)
    if add_taylor_first:
        result = - 2 * S_grad / S
    for k in range(n + 1):
        coeff = 2 * ((-1) ** k) / math.factorial(k + 2)
        term = coeff * S.pow(k) * S_grad.pow(k + 2)
        result = result + term
    return result


###### [Urgent FIXME:]ADD S in Importance Score computation
def get_layer_importance(grad, args):
    if args.taylor_order == 1:
        return -1.0 * grad
    elif args.taylor_order == 2:
        return grad.pow(2)
    elif args.taylor_order == 12:
        return -2.0 * grad + grad.pow(2)

def get_layer_importance_S(grad, args, S):
    if args.taylor_order == 1:
        return -1.0 * grad / S
    elif args.taylor_order == 2:
        return grad.pow(2)
    elif args.taylor_order == 12:
        return -2.0 * grad / S + grad.pow(2)