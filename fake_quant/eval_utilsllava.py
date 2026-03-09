import utils
import model_utils
import quant_utils
import torch
import os
import logging
from tqdm import tqdm


@torch.no_grad()
def evaluator(model, testenc, dev, args, tokenizer, image_processor):
    _, testenc = testenc
    torch.cuda.synchronize()
    model = model.to(dev)
    
    # model.model.to(dev)
    # model.lm_head.to(dev)
    # model.model.mm_projector.to(dev)
    logging.info("moving model to dev")
    from vlmeval.inference import infer_data
    from vlmeval.smp import dump, get_rank_and_world_size,string,load
    from vlmeval.dataset import DATASET_TYPE
    import pandas as pd
    # llava use custom prompt, llamav pixtral use dataset prompt
    def build_prompt_llava(line, dataset):
        # assert DATASET_TYPE(dataset.dataset_name) == "MCQ"
        # assert dataset is None or isinstance(dataset, str)
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
        if len(options):
            question += "\nAnswer with the option's letter from the given choices directly."
        else:
            question += "\nAnswer the question directly."
        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=question))
        return message

    def message_to_prompt(data, image_processor, model, tokenizer):
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from llava.constants import IMAGE_TOKEN_INDEX
        from PIL import Image
        from abc import abstractproperty
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
        args = abstractproperty()
        args.image_aspect_ratio = "pad"
        image_tensor = process_images(images, image_processor, args).to(
            "cuda", dtype=torch.float16
        )

        prompt = system_prompt + "USER: " + content + " ASSISTANT: "
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stopping_criteria = KeywordsStoppingCriteria(
            ["</s>"], tokenizer, input_ids
        )
        return input_ids, image_tensor, stopping_criteria

    
    def infer_data_llava(lm, out_file, verbose=False):
        logging.info("start select split data!")
        dataset = testenc
        dataset_name = dataset.dataset_name
        rank, world_size = get_rank_and_world_size()
        sheet_indices = list(range(rank, len(dataset), world_size))
        lt = len(sheet_indices)
        data = dataset.data.iloc[sheet_indices]
        data_indices = [i for i in data['index']]
        res = {}
        if os.path.exists(out_file):
            # save data to 
            res.update(load(out_file))
            # os.remove(out_file)# to avoid conflict, for now we do not reload
        
        # If finished, will exit without building the model
        all_finished = True
        for i in range(lt):
            idx = data.iloc[i]['index']
            if idx not in res:
                all_finished = False
        if all_finished:
            res = {k: res[k] for k in data_indices}
            dump(res, out_file)
        data = data[~data['index'].isin(res)]
        lt = len(data)
        logging.info("start inference!")
        for i in tqdm(range(lt)):
            idx = data.iloc[i]['index']
            # if idx in res:
            #     continue

            # if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            #     struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
            # else:
            #     struct = dataset.build_prompt(data.iloc[i])
            message = build_prompt_llava(data.iloc[i], dataset)
            inputs_id, image, stopping_criteria= message_to_prompt(message, image_processor,model, tokenizer)
            with torch.inference_mode():
                response = model.generate(inputs_id, images=image,
                                            do_sample=False,
                                            temperature=0,
                                            max_new_tokens=32, #512
                                            top_p=None,
                                            num_beams=1,
                                            use_cache=True,
                                            stopping_criteria = [stopping_criteria],
                                            )
            torch.cuda.empty_cache()
            response = tokenizer.batch_decode(response, skip_special_tokens=True)[0].strip()
            if verbose:
                print(response, flush=True)

            res[idx] = response
            if (i + 1) % 10 == 0:
                dump(res, out_file)

        res = {k: res[k] for k in data_indices}
        # logger.info(res)
        dump(res, out_file)
        # eval llava?
        data_all = {}
        data_all.update(load(out_file))
        data = dataset.data
        for x in data['index']:
            assert x in data_all
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')
        result_file = args.save_path + '/vlm_result.xlsx'
        dump(data, result_file)
        judge_kwargs = {
            'nproc': 4,
            'verbose': True,
            'retry':  3
        }
        logging.info("start evaluation!")
        eval_results = dataset.evaluate(result_file,**judge_kwargs)
        # acc_file =  args.save_path + '/vlm_acc.xlsx'
        logging.info(eval_results)
        return eval_results
        
    eval_results = infer_data_llava(model, args.save_path+'/vlm_eval.pkl', True)
    # _, dataset = get_loaders(
    #     args.vlm_task,
    #     nsamples=0,
    #     seed=args.seed,
    #     model=args.model,
    #     seqlen=model.seqlen,
    # )
    # result_file = args.save_path + '/vlm_result.xlsx'
    # # data['prediction'] = 
    # judge_kwargs = {
    #     'nproc': 4,
    #     'verbose': True,
    #     'retry':  3
    # }
    # eval_results = dataset.evaluate(result_file,**judge_kwargs)
    # # acc_file =  args.save_path + '/vlm_acc.xlsx'
    # logger.info(eval_results)
    # eval_data_llava()
    return eval_results
