import utils
import model_utils
import quant_utils
import torch
import os
import logging
import json
from tqdm import tqdm
from vlmeval.smp import *
import torch.distributed as dist
import datetime
@torch.no_grad()
def evaluator(model, testenc, dev, args, tokenizer, image_processor):
    from vlmeval.inference import infer_data
    from vlmeval.smp import dump, get_rank_and_world_size, string, load
    from vlmeval.dataset import DATASET_TYPE
    import pandas as pd
    rank, world_size = get_rank_and_world_size()
    # if world_size > 1:
    #     local_rank = os.environ.get('LOCAL_RANK', 0)
    #     torch.cuda.set_device(int(local_rank))
    #     dist.init_process_group(
    #         backend='nccl',
    #         timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
    #     )
    _, testenc = testenc
    # torch.cuda.synchronize()
    model = model.to(rank)
    logging.info("moving model to dev")


    def build_prompt_llava(line, dataset):
        tgt_path = dataset.dump_image(line)
        question = line['question']
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question 
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        if not args.case_study:
            if len(options):
                question += "\nAnswer with the option's letter from the given choices directly."
            else:
                question += "\nAnswer the question directly."
        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=question))
        return message

    def ensure_image_url(image: str) -> str:
        prefixes = ['http://', 'https://', 'file://', 'data:image;']
        if any(image.startswith(prefix) for prefix in prefixes):
            return image
        if os.path.exists(image):
            return 'file://' + image
        raise ValueError(f'Invalid image: {image}')

    def output_process(answer):
        if "<s>" in answer:
            answer = answer.replace("<s>", "").strip()
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[1].strip()
        elif "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[1].strip()
        elif "assistant\n" in answer:
            answer = answer.split("assistant\n")[1].strip()
        elif "<|end_header_id|>\n\n" in answer:
            answer = answer.split("<|end_header_id|>\n\n")[2].strip()

        if "</s>" in answer:
            answer = answer.split("</s>")[0].strip()
        elif "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        elif "<|eot_id|>" in answer:
            answer = answer.split("<|eot_id|>")[0].strip()
        return answer

    def message_to_prompt(data, image_processor, model, tokenizer, dataset=None):
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from llava.constants import IMAGE_TOKEN_INDEX
        from PIL import Image
        from abc import abstractproperty
        if 'hf_v16' in str(tokenizer):
            content, images = [], []
            for msg in data:
                if msg["type"] == "text":
                    content.append({"type": msg["type"], "text": msg["value"]})
                else:
                    content.append({"type": "image"})
                    images.append(Image.open(msg["value"]).convert("RGB"))
            conversation = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            prompt = image_processor.apply_chat_template(
            conversation, add_generation_prompt=True
            )
            inputs = image_processor(prompt, images, return_tensors="pt").to(
                        "cuda", torch.float16)
            return inputs
        elif 'qwen' in str(tokenizer):
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
                raise err
            content = []
            for s in data:
                if s['type'] == 'image':
                    item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                    item['max_pixels'] = 16384 * 28 * 28
                    item['min_pixels'] = 1280 * 28 * 28
                elif s['type'] == 'text':
                    item = {'type': 'text', 'text': s['value']}
                else:
                    raise ValueError(f"Invalid message type: {s['type']}, {s}")
                content.append(item)
            messages = []
            messages.append({'role': 'user', 'content': content})
            text = image_processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info([messages])
            inputs = image_processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt').to("cuda")
            return inputs
        elif 'intern' in str(tokenizer):
            from intern_utils import load_image
            message = data
            image_num = len([x for x in message if x['type'] == 'image'])
            max_num = 6
            total_max_num = 64
            max_num = max(1, min(max_num, total_max_num // image_num))
            device = utils.get_dev()
            prompt = reorganize_prompt(message, image_num, dataset) # we do not need dataset here?
            

            if image_num > 1:
                image_path = [x['value'] for x in message if x['type'] == 'image']
                num_patches_list, pixel_values_list = [], []
                for image_idx, file_name in enumerate(image_path):
                    upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                    curr_pixel_values = load_image(
                        file_name, max_num=max_num, upscale=upscale_flag).to(device).to(torch.bfloat16)
                    num_patches_list.append(curr_pixel_values.size(0))
                    pixel_values_list.append(curr_pixel_values)
                pixel_values = torch.cat(pixel_values_list, dim=0)
            elif image_num == 1:
                image_path = [x['value'] for x in message if x['type'] == 'image'][0]
                upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
                pixel_values = load_image(
                    image_path, max_num=max_num, upscale=upscale_flag).to(device).to(torch.bfloat16)
                num_patches_list = [pixel_values.size(0)]
            else:
                pixel_values = None
                num_patches_list = []
            kwargs_default = dict(
                do_sample = False,
                max_new_tokens = 512,
                temperature = 0.6,
                top_p = 0.95
            )
            return {'pixel_values': pixel_values, 'num_patches_list': num_patches_list, 'question': prompt, 'generation_config': kwargs_default}
        system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )

        def concat_tilist(message):
            text, images = "", []
            for item in message:
                if item["type"] == "text":
                    text += item["value"]
                elif item["type"] == "image":
                    text += " <image> "
                    images.append(item["value"])
            return text, images

        content, images = concat_tilist(data)
        images = [Image.open(s).convert("RGB") for s in images]
        image_sizes = [img.size for img in images]
        args = abstractproperty()
        args.image_aspect_ratio = "pad"
        image_tensor = process_images(images, image_processor, args).to(
            "cuda", dtype=torch.float16
        )

        prompt = system_prompt + "USER: " + content + " ASSISTANT: "
        # ADD conv templets
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stopping_criteria = KeywordsStoppingCriteria(
            ["</s>"], tokenizer, input_ids
        )
        return input_ids, (image_tensor, image_sizes), stopping_criteria


    def reorganize_prompt(message, image_num, dataset=None):
        if dataset is not None and listinstr(['MUIRBench'], dataset):
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            images_to_remove = ' '.join(['<image>'] * image_num)
            prompt = prompt.replace(images_to_remove, '')
            for i in range(image_num):
                prompt = prompt.replace('<image>', f'<Image-{i + 1}>', 1)
            prompt = ''.join([f'Image-{i + 1}: <image>\n' for i in range(image_num)]) + prompt
        elif dataset is not None and listinstr(["bmmr"], dataset.lower()):
            if image_num == 1:
                prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])
            else:
                prompt, image_idx = "", 1
                for x in message:
                    if x["type"] == "text":
                        prompt += x["value"]
                    elif x["type"] == "image":
                        image_idx += 1
        elif image_num == 1:
            prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        else:
            prompt, image_idx = '', 1
            for x in message:
                if x['type'] == 'text':
                    prompt += x['value']
                elif x['type'] == 'image':
                    prompt += f'<Image-{image_idx}>'
                    image_idx += 1
            prompt = ''.join([f'Image-{i + 1}: <image>\n' for i in range(image_num)]) + prompt
            images_to_remove = ''.join([f'<Image-{i + 1}>' for i in range(image_num)])
            prompt = prompt.replace(images_to_remove, '')
        return prompt


    
    def infer_data_llava(lm, args, verbose=False):

        dataset = testenc
        dataset_name = dataset.dataset_name
        rank, world_size = get_rank_and_world_size()
        if rank == 0:
            logging.info("Start selecting split data!")
        # Each rank writes to a unique output file
        sheet_indices = list(range(rank, len(dataset), world_size))
        data = dataset.data.iloc[sheet_indices]
        if args.case_study:
            case_file = args.save_path + '/case_study.jsonl'
            os.makedirs(os.path.dirname(case_file), exist_ok=True)
            case_anw = open(case_file, "w")
            res = {}
            
        else:
            out_file = args.save_path + f'/vlm_eval_rank{rank}.pkl'
            data_indices = [i for i in data['index']]
            res = {}

            if os.path.exists(out_file):
                res.update(load(out_file))

            # Check if all results are already computed
            all_finished = all(idx in res for idx in data['index'])
            # if all_finished:
            #     dump({k: res[k] for k in data_indices}, out_file)
            #     return

        data = data[~data['index'].isin(res)]
        if world_size > 1:
            dist.barrier()
        
        logging.info("finish selecting split data!")
        for i in tqdm(range(len(data))):
            if args.case_study:
                if i not in [40, 141, 1384]:#[314, 1994, 200, 1733, 1809, 1509, 1684, 904, 1912, 1465]:
                    continue
            idx = data.iloc[i]['index']
            if 'qwen' in str(tokenizer) or 'math' in dataset_name.lower():
                message = dataset.build_prompt(data.iloc[i])
            else:
                message = build_prompt_llava(data.iloc[i], dataset)
            # if isinstance(tokenizer, string):
            if 'hf_v16' in str(tokenizer):
                inputs = message_to_prompt(
                message, image_processor, model, tokenizer
                )
                output = model.generate(
                    **inputs,
                    do_sample=False, 
                    temperature=0,
                    max_new_tokens=16 if not args.case_study else 512,
                    top_p=None,
                    num_beams=1
                )
                response = image_processor.decode(output[0], skip_special_tokens=True)
                response = output_process(response)
            elif 'qwen' in str(tokenizer):
                inputs = message_to_prompt(message, image_processor, model, tokenizer)
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **inputs, 
                        top_p=0.001,
                        top_k=1,
                        temperature=0.01,
                        repetition_penalty=1.0,
                        max_new_tokens=16 if not args.case_study else 512)
                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                    ]
                torch.cuda.empty_cache()
                out = image_processor.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                response = out[0]
            elif 'intern' in str(tokenizer):
                inputs = message_to_prompt(message, image_processor, model, tokenizer, dataset_name)
                with torch.inference_mode():
                    response = model.chat(
                        image_processor,
                        **inputs,
                    )
            else:
                inputs_id, image, stopping_criteria = message_to_prompt(
                    message, image_processor, model, tokenizer
                ) 
                image, image_sizes = image
                with torch.inference_mode():
                    if not args.dosample:
                        response = model.generate(
                            inputs_id, images=image,
                            image_sizes = image_sizes,
                            do_sample=False,
                            max_new_tokens=16 if not args.case_study else 512,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                        )
                    else:
                        response = model.generate(
                            inputs_id, images=image,
                            image_sizes = image_sizes,
                            do_sample=True, temperature=0.1,
                            max_new_tokens=16, 
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                        )
                torch.cuda.empty_cache()
                response = tokenizer.batch_decode(response, skip_special_tokens=True)[0].strip()
            if args.verbose:
                print(response, flush=True)
            if args.case_study:
                # pass # add early exit, just need test 10 samples
                import json
                case_anw.write(json.dumps(
                    {"case_id": i,
                    "input":message,
                    "output":response,}) + "\n")
            else:
                res[idx] = response
                if (i + 1) % 10 == 0:
                    dump(res, out_file)

        if args.case_study:
            logging.info("finish case study!")
            return
        dump({k: res[k] for k in data_indices}, out_file)
        if world_size > 1:
            dist.barrier()
        # Merge results on rank 0 after inference
        if rank == 0:
            merged_results = {}
            for r in range(world_size):
                rank_file = args.save_path + f'/vlm_eval_rank{r}.pkl'
                if os.path.exists(rank_file):
                    merged_results.update(load(rank_file))

            # Dump merged results
            merged_out_file = args.save_path + '/vlm_eval.pkl'
            dump(merged_results, merged_out_file)

            # Clean up per-GPU files
            for r in range(world_size):
                rank_file = args.save_path + f'/vlm_eval_rank{r}.pkl'
                if os.path.exists(rank_file):
                    os.remove(rank_file)

            logging.info("Merged and cleaned up all per-GPU outputs.")

        return
    if world_size > 1:
        dist.barrier()
    # Run inference
    infer_data_llava(model, args, True)
    if args.case_study:
        return # early exit without evaluation
    # Rank 0 handles evaluation
    rank, _ = get_rank_and_world_size()
    if rank == 0:
        logging.info("Start evaluation!")
        result_file = args.save_path + '/vlm_result.xlsx'
        data_all = load(args.save_path + '/vlm_eval.pkl')
        dataset = testenc.data

        # Ensure all indices are covered
        for x in dataset['index']:
            assert x in data_all

        dataset['prediction'] = [str(data_all[x]) for x in dataset['index']]
        if 'image' in dataset:
            dataset.pop('image')

        dump(dataset, result_file)

        judge_kwargs = {'nproc': 4, 'verbose': True, 'retry': 3}
        eval_results = testenc.evaluate(result_file, **judge_kwargs)
        import json
        logging.info(eval_results)
        if isinstance(eval_results, dict):
            logging.info('\n' + json.dumps(eval_results, indent=4))
        elif isinstance(eval_results, pd.DataFrame):
            if len(eval_results) < len(eval_results.columns):
                eval_results = eval_results.T
            logging.info('\n' + tabulate(eval_results))
        return eval_results
