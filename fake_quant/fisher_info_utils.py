import os
import torch
import torch.nn as nn
from tqdm import tqdm
import gptq_utils
import data_utils

@torch.enable_grad()
def calib_fisher_info(model, dataloader, tokenizer, image_processor, args, use_cache=True, cache_file=None):
    """
    Compute Fisher information for model layers to estimate parameter importance.
    
    Args:
        model: Model to calibrate.
        tokenizer: Tokenizer.
        image_processor: Image processor.
        args: Configuration arguments.
        use_cache: Whether to use cache.
        cache_file: Cache file path. If None, it is generated automatically.
    """
    model_id = model.config._name_or_path
    
    if args.cache_file is None:
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = args.cache_file
        
    # Include rotate info in cache filename
    rotate_info = "rotated" if hasattr(args, "rotate") and args.rotate else "norotate"
    if args.cache_file is not None:
        cache_file = os.path.join(cache_dir, f"{model_id.replace('/','_')}_{rotate_info}_calib_fisher_info.pt")
    else:
        cache_file = os.path.join(cache_dir, f"{model_id.replace('/','_')}_{rotate_info}_{args.nsamples}_calib_fisher_info.pt")
    
    if os.path.exists(cache_file) and use_cache:
        print(f"Loading Fisher information cache from {cache_file}...")
        all_fisher_info = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info = all_fisher_info[name].to(module.weight.device)
        print("Fisher information cache loaded successfully!")
        return
    
    print("Starting Fisher information computation...")
    model.eval()

    # Initialize Fisher information
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Ensure Fisher info is initialized on the same device as weights
            module.fisher_info = torch.zeros_like(module.weight[0], device=module.weight.device)

    # Ensure the whole model is on CUDA
    device = 'cuda'
    model = model.to(device)
    
    # Collect Fisher information
    batch_count = 0
    for batch in tqdm(dataloader, desc="Computing Fisher information"):
        try:
            # Process batch data via message_to_prompt
            input_ids, images, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer)
            
            # Ensure input_ids and images are on the correct device
            # input_ids = input_ids.to(device)
            # if images is not None:
            #     images = images.to(device)
            # output_ids = output_ids.to(device)
            # Recursively move nested tensor structures to the target device
            def move_to_device(obj, target_device):
                if isinstance(obj, torch.Tensor):
                    return obj.to(target_device)
                elif isinstance(obj, (list, tuple)):
                    return type(obj)(move_to_device(item, target_device) for item in obj)
                elif isinstance(obj, dict):
                    return {k: move_to_device(v, target_device) for k, v in obj.items()}
                else:
                    return obj
            
            # Ensure input_ids and images are on the correct device
            input_ids = move_to_device(input_ids, device)
            if images is not None:
                images = move_to_device(images, device)
            output_ids = move_to_device(output_ids, device)
            
             # Align input and label lengths
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
                print(f"Input and label lengths adjusted to {max_len}")
                input_ids = input_ids.to(device)
                output_ids = output_ids.to(device)
            
            attention_mask = input_ids.ne(0).to(device) 
            
            # Ensure training mode for gradient computation
            model.train()
            
            # Enable gradients only for q_proj/k_proj/v_proj and disable others
            for name, param in model.named_parameters():
                # Only enable gradients for q_proj, k_proj, and v_proj
                if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    
            # Ensure outputs require gradients
            with torch.enable_grad():
                # Compute loss and backpropagate
                outputs = model(input_ids=input_ids, images=images, labels=output_ids, attention_mask=attention_mask)
                loss = outputs[0]
                loss.backward()
            
            # Restore eval mode
            model.eval()
                
            # Accumulate Fisher information
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and module.weight.grad is not None:
                    # Ensure Fisher info and gradients are on the same device
                    fisher_update = module.weight.grad.detach().pow(2).mean(0).to(device)
                    # Ensure module.fisher_info is also on the same device
                    module.fisher_info = module.fisher_info.to(device)
                    module.fisher_info += fisher_update
            
            model.zero_grad()
            batch_count += 1
            
        except Exception as e:
            print(f"Error while computing Fisher information: {e}")
            # Print detailed traceback for debugging
            import traceback
            print("Detailed error traceback:")
            traceback.print_exc()
            # Print batch data type and structure to help debugging
            print(f"Batch data type: {type(batch)}")
            if isinstance(batch, dict):
                print(f"Batch data keys: {list(batch.keys())}")
            elif isinstance(batch, list) and len(batch) > 0:
                print(f"Type of first item in batch data: {type(batch[0])}")
            continue
    
    # Normalize Fisher information
    if batch_count > 0:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info = module.fisher_info.div(batch_count).sqrt()
    
    # Save Fisher information
    all_fisher_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            all_fisher_info[name] = module.fisher_info.cpu()
    
    print(f"Saving Fisher information cache to {cache_file}...")
    torch.save(all_fisher_info, cache_file)
    print("Fisher information cache saved successfully!")