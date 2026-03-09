
import os
import torch
import time
import math
import json
os.environ['HF_HOME'] = '/vast/yw6594/log'
os.chdir("/scratch/yw6594/cf/vlm/quant/QuaRot/e2e")

from PIL import Image
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from e2e.quantized_llava.llava_builder import load_pretrained_model
model_path = 'liuhaotian/llava-v1.5-13b'
disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, torch_dtype=torch.float16)
# print(model)
# stime = time.time()
model.init_quant()
layer_lm = model.model.layers[0]
# layer_vit = model.model.vision_tower.vision_tower.vision_model.encoder.layers[0]
# ttime = time.time() - stime
# print(ttime)
# print(model)


# generate e2e
# load data
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
question_file = './llava_test_CQM-A.json'
questions = json.load(open(os.path.expanduser(question_file), "r"))
num_chunks = 1
chunk_idx = 0
qidx = 5
questions = get_chunk(questions, num_chunks, chunk_idx)
prompt = questions[qidx]['conversations'][0]
if getattr(model.config, 'mm_use_im_start_end', False):
    prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
else:
    prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt # use here
prompt = prompt +  '\n' + "Answer with the option's letter from the given choices directly."
conv = conv_templates['vicuna_v1'].copy()
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


img_path = '/scratch/yw6594/cf/vlm/quant/QuaRot/e2e/image.png'
image = Image.open(img_path)
image_tensor = process_images([image], image_processor, model.config)[0]
images = image_tensor.unsqueeze(0).half().cuda()


with torch.inference_mode():
    model.generate(input_ids, images=image,
                                do_sample=False,
                                temperature=0,
                                max_new_tokens=512,
                                top_p=None,
                                num_beams=1,
                                use_cache=True,
                                )