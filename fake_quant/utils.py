import argparse
import pprint
import torch
import random
import numpy as np
import os
from datetime import datetime
import logging


from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

supported_models = [
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-13b-hf',
            'meta-llama/Llama-2-70b-hf',
            'meta-llama/Meta-Llama-3-8B',
            'meta-llama/Meta-Llama-3-70B',
            'facebook/opt-125m',
            'liuhaotian/llava-v1.5-7b',
            'liuhaotian/llava-v1.5-13b',
            'liuhaotian/llava-v1.6-vicuna-7b',
            'liuhaotian/llava-v1.6-vicuna-13b',
            'llava-hf/llava-v1.6-vicuna-7b-hf',
            'llava-hf/llava-v1.6-vicuna-13b-hf',
            "HuggingFaceTB/SmolVLM-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "OpenGVLab/InternVL3-8B",
            "OpenGVLab/InternVL3-14B",
            "OpenGVLab/InternVL3_5-8B-HF",
            "OpenGVLab/InternVL3_5-14B-HF",
            "Qwen/Qwen2.5-VL-32B-Instruct",
            "OpenGVLab/InternVL2-8B",
            "OpenGVLab/InternVL2_5-8B",
            "OpenGVLab/InternVL3-8B",

            ]
supported_datasets = ['wikitext2', 'ptb', 'c4','ScienceQA_Train','ScienceQA_TEST', 'VizWiz', 'SEEDBench', 'SEEDBench_IMG', 'COCO_CALIB', 'OCRBench','MathVista_MINI']

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import torch.distributed as dist
def get_dev():
    return (
        torch.device(f'cuda:{dist.get_rank()}') if dist.is_initialized() else 
        torch.device('cuda') if torch.cuda.is_available() else 
        torch.device('cpu'))
DEV = get_dev()


def llama_down_proj_groupsize(model, groupsize):
    
    assert groupsize > 1, 'groupsize should be greater than 1!'
    
    if model.config.intermediate_size % groupsize == 0:
        logging.info(f'(Act.) Groupsiz = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size/groupsize)
    assert groupsize*group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size//group_num
    assert down_proj_groupsize*group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logging.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize


def set_seed(seed):
    np.random.seed(seed)  # Set NumPy seed
    random.seed(seed)  # Set Python's built-in random seed
    torch.manual_seed(seed)  # Set PyTorch seed for CPU & CUDA
    torch.cuda.manual_seed(seed)  # Set seed for CUDA (if using GPU)
    torch.cuda.manual_seed_all(seed)  # If multi-GPU, set for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-optimizations for determinism
    # random.seed(seed)
    # np.random.seed(seed)
    # if is_torch_available():
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     # ^^ safe to call this function even if cuda is not available
    #     if deterministic:
    #         torch.use_deterministic_algorithms(True)

# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     level=logging.INFO,
#     datefmt="%Y-%m-%d %H:%M:%S"
# )

# Dump the log both to console and a log file.
def config_logging(log_file, level=logging.INFO):
    class LogFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                self._style._fmt = "%(asctime)s - %(message)s"
            else:
                self._style._fmt = "%(asctime)s - %(levelname)s - %(message)s"
            return super().format(record)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LogFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(LogFormatter())

    logging.basicConfig(level=level, handlers=[console_handler, file_handler])


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='Model to load;', choices=supported_models)
    parser.add_argument('--seed', type=int, default=0, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                        help='Dataset for Evaluation (default: wikitext2)', choices=supported_datasets,)
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--bsz', type=int, default=32,
                        help='Batch-size for PPL evaluation (default:32)')


    # Rotation Arguments
    parser.add_argument('--rotate', action=argparse.BooleanOptionalAction, default=False, 
                        help='''Rotate the moodel. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys''')
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--rotation_seed', type=int, default=-1,
                        help='Random Seed for generating random matrix!!')
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply Hadamard rotation in FP32 (default: False)')
    parser.add_argument('--rot_lr', type=float, default=1,
                        help='learning rate for rotation!!')
    parser.add_argument('--sq_lr', type=float, default=1,
                        help='learning rate for scaling factor!!')
    parser.add_argument('--rot_epochs', type=int, default=100,
                        help='learning rate for rotation!!')
    parser.add_argument('--rot_dim', type=int, default=0,
                        help='dimension for rotation matrix!!')
    parser.add_argument('--sq', action=argparse.BooleanOptionalAction, default=False, 
                        help='weather use learnt smooothquant for learning rotation matrix!!')
    parser.add_argument('--beta_lr', type=float, default=1,
                        help='learning rate for beta!!')
    parser.add_argument('--beta_epochs', type=int, default=100,
                        help='learning rate for beta!!')
    parser.add_argument('--bs', type=int, default=512,
                        help='mini batchsize for beta!!')
    parser.add_argument('--nobeta',action=argparse.BooleanOptionalAction, default=False,  
        help='whether use beta learning/bias')
    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)''')
    parser.add_argument('--a_groupsize', type=int, default=-1, 
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize')
    parser.add_argument('--a_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric Activation quantization (default: False)')
    parser.add_argument('--a_clip_ratio', type=float, default=1.0,
        help='Clip ratio for activation quantization. new_max = max * clip_ratio')
    parser.add_argument('--lma_clip_ratio', type=float, default=1.0,
        help='Clip ratio for lm activation quantization. new_max = max * clip_ratio')
    parser.add_argument('--vita_clip_ratio', type=float, default=1.0,
        help='Clip ratio for vit activation quantization. new_max = max * clip_ratio')
    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16, 
                        help='Number of bits for weights of the Linear layers')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize')
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric weight quantization (default: False)')
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False,
                        help='Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ')
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False,
                        help='''Clipping the weight quantization! 
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization''')
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for GPTQ.')
    parser.add_argument('--vitnsamples', type=int, default=0,
                        help='Number of calibration data samples for vit learn beta.')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
                        help='calibration data samples for GPTQ.', choices=supported_datasets)
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False,
                        help='act-order in GPTQ')
    parser.add_argument('--dosample', action=argparse.BooleanOptionalAction, default=False\
        , help='in llava, whether use do_sample')

    # General Quantization Arguments
    parser.add_argument('--int8_down_proj', action=argparse.BooleanOptionalAction, default=False,
                        help='Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8')

    # KV-Cache Quantization Arguments
    parser.add_argument('--v_bits', type=int, default=16,
                        help='''Number of bits for V-cache quantization. 
                        Note that quantizing the V-cache does not need any other rotation''')
    parser.add_argument('--v_groupsize', type=int, default=-1)
    parser.add_argument('--v_asym', action=argparse.BooleanOptionalAction, default=False,
                        help='ASymmetric V-cache quantization')
    parser.add_argument('--v_clip_ratio', type=float, default=1.0,
        help='Clip ratio for v-cache quantization. new_max = max * clip_ratio')
    
    parser.add_argument('--k_bits', type=int, default=16,
                        help='''Number of bits for K-cache quantization. 
                        Note that quantizing the K-cache needs another rotation for the keys/queries''')
    parser.add_argument('--k_groupsize', type=int, default=-1)
    parser.add_argument('--k_asym', action=argparse.BooleanOptionalAction, default=False, 
                        help='ASymmetric K-cache quantization')
    parser.add_argument('--k_pre_rope', action=argparse.BooleanOptionalAction, default=False, 
                        help='Pre-RoPE quantization for K-cache (not Supported yet!)')
    parser.add_argument('--k_clip_ratio', type=float, default=1.0,
        help='Clip ratio for k-cache quantization. new_max = max * clip_ratio')

    # low rank parameters
    ### low rank experiments
    parser.add_argument('--is_ma_hack', action=argparse.BooleanOptionalAction, default=False,
        help='whether use MA hack for local ft')
    parser.add_argument('--topk', type=int, default=1,
        help='topk for super weight analysis')
    parser.add_argument('--smooth_grad', action=argparse.BooleanOptionalAction, default=False,
        help='whether smooth grad info')
    parser.add_argument('--smooth_method', type=str, default='mean', choices=['mean', 'zero'],
        help='smooth method for grad info')
    parser.add_argument('--smooth_percentile', type=float, default=-1,
        help='percentile for grad info')
    parser.add_argument('--smooth_outlier_std', type=float, default=-1,
        help='outlier std for grad info')
    parser.add_argument('--smooth_power_threshold', type=float, default=-1,
        help='power threshold for grad info')
    parser.add_argument('--smooth_ma_only', action=argparse.BooleanOptionalAction, default=False,
        help='whether use ma layer only smoothfor grad info')
    parser.add_argument('--smooth_final', action=argparse.BooleanOptionalAction, default=False,
        help='whether smooth grad info at final step or per get_layer_importance step')
    parser.add_argument('--fix_loss', action=argparse.BooleanOptionalAction, default=False,
        help='whether fix zigzag loss for local ft')
    parser.add_argument('--sw_analyze', action=argparse.BooleanOptionalAction, default=False,
        help='whether analyze super weight')
    parser.add_argument('--scheduler_type', type=str, default=None, choices=[None, 'cosine','step'],
        help='scheduler type for local ft')
    parser.add_argument('--scheduler_step_size', type=int, default=30,
        help='step size for scheduler')
    parser.add_argument('--scheduler_gamma', type=float, default=0.8,
        help='gamma for scheduler')
    parser.add_argument('--svd_ft_optim', type=str, default="adam", choices=['adam', 'adamw', 'sgd', 'sgdm'],
        help='svd ft optimizer')
    parser.add_argument('--svd_emb', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use svd on embedding layer')
    parser.add_argument('--use_S_gradinfo_init', action=argparse.BooleanOptionalAction, default=False,
        help='whether use S_gradinfo_init')
    ### low rank experiments
    # parser.add_argument('--ft_modules', type=str, default=None, choices=[None, 'qkv', 'attn', 'mlp', 'gaup', 'down', 'o'],
    #     help='Apply SVD on which modules for local ft(, add extra for loop as did in svd stage modules)') # maybe we can just depracate this, use svd moduels for all only
    parser.add_argument('--svd_layeridx', type=int, default=99, 
        help='svd layer index for local ft')
    parser.add_argument('--is_quant_aware_ft', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use quant aware ft')
    parser.add_argument('--is_per_head_svd', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use per head svd in QKV')
    parser.add_argument('--is_q_headnum', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use per q head svd in GQA')
    parser.add_argument('--misalign_text_image', action=argparse.BooleanOptionalAction, default=False, 
        help='whether misalign text and image for grad info')
    parser.add_argument('--is_taylor', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use taylor expansion to compute grad info')
    parser.add_argument('--taylor_order', type=int, default=2, 
        help='the taylor expansion order for S_grad_squared')
    parser.add_argument('--add_taylor_first', action=argparse.BooleanOptionalAction, default=False, 
        help='whether add taylor first order to grad info')
    parser.add_argument('--label_shift', action=argparse.BooleanOptionalAction, default=False,  
        help='shift llm label to the left')
    parser.add_argument('--is_rank_allocate_ft', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use rank allocate ft')
    parser.add_argument('--svd_modules', type=str, default=None, choices=[None,'qkv','attn', 'all', 'mlp','gaup', 'all_sep','all_sep_mlpfine','all_sep_fine', 'down', 'o', 'all_sep_fine_wo_down', 'all_sep_rank_allocate_qkvonly'],
        help='Apply SVD on which modules')
    parser.add_argument('--use_true_param_ratio', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use rank_ratio as param_ratio as did in asvd/svdllm for share svd')
    parser.add_argument('--use_all_true_param_ratio', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use all matrix true param ratio(just final /num_head)')
    parser.add_argument('--use_param_ratio', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use rank_ratio as param_ratio in gradinfo as did in asvd/svdllm')
    parser.add_argument('--use_group_svd', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use rank_ratio as param_ratio in GQA specifically, rank_Q = group_ratio * rank_KV')
    parser.add_argument('--group_ratio', type=float, default=0.0,
        help='group ratio for GQA specifically, rank_Q = group_ratio * rank_KV, also control grad multiplier for qkv')
    parser.add_argument('--rank_ratio', type=float, default=1.0,
        help='rank ratio for SVD compression, lowrank param = linear param * rank_ratio')
    parser.add_argument('--had_svd', action=argparse.BooleanOptionalAction, default=False, 
        help='had then svd')
    parser.add_argument('--had_rank', action=argparse.BooleanOptionalAction, default=False, 
        help='had on rank dimension')
    parser.add_argument('--svd_lm', action=argparse.BooleanOptionalAction, default=False,  
        help='Apply SVD compression to language model')
    parser.add_argument('--svd_lm_localft', action=argparse.BooleanOptionalAction, default=False,  
        help='Apply SVD ft to language model')
    parser.add_argument('--weighted_svd', action=argparse.BooleanOptionalAction, default=False,
        help='weighted svd using grad info')
    parser.add_argument('--weighted_none_svd_qat', action=argparse.BooleanOptionalAction, default=False,
        help='weighted qat using grad info for none svd linears')
    parser.add_argument('--qat_L_off', action=argparse.BooleanOptionalAction, default=False,
        help='disable Linear ft in qat (for linear layers without SVD compression)')
    parser.add_argument('--qat_uv_reg', action=argparse.BooleanOptionalAction, default=False,
        help='regularization for U/V in qat')
    parser.add_argument('--qat_uv_reg_scale', action=argparse.BooleanOptionalAction, default=False,
        help='scale regularization for U/V in qat')
    parser.add_argument('--qat_uv_reg_alpha', type=float, default=0.1,
        help='regularization alpha for U/V in qat')
    parser.add_argument('--svd_ft_mode', type=str, default="weight", choices=['output', 'weight'],
        help='local ft svd mode')
    parser.add_argument('--qat_start_iter', type=float, default=0.0,
        help='start iteration ratio for qat [0.0 - 1.0]')
    parser.add_argument('--qat_optim_R', action=argparse.BooleanOptionalAction, default=False,
        help='use R optimizer for qat')
    parser.add_argument('--qat_param_update_freq', type=int, default=1,
        help='frequency of parameter update for qat')
    parser.add_argument('--localft_iters', type=int, default=100,
        help='number of iterations for local ft')
    parser.add_argument('--localft_lr', type=float, default=1e-4,
        help='learning rate for local ft')
    parser.add_argument('--localft_lr_R', type=float, default=None,
        help='learning rate for QAT rotation matrix, deprecated')
    parser.add_argument('--qat_lr', type=float, default=None,
        help='learning rate for QAT rotation matrix, deprecated')
    parser.add_argument('--qat_lr_UV', type=float, default=None,
        help='learning rate for QAT UV matrix')
    parser.add_argument('--qat_lr_R', type=float, default=None,
        help='learning rate for QAT rotation matrix')
    parser.add_argument('--svd_vit',action=argparse.BooleanOptionalAction, default=False,  
        help='Apply SVD compression to vision model')
    parser.add_argument('--set_seedrot',action=argparse.BooleanOptionalAction, default=False,  
        help='whether apply same seed for rotation')
    parser.add_argument('--svd_mode', type=str, default="UV", 
        help='how to fold in Sigma into UV')
    parser.add_argument('--kv_fuse', action=argparse.BooleanOptionalAction, default=False, 
        help='use concat KV proj to have share V')
    parser.add_argument('--qkv_fuse', action=argparse.BooleanOptionalAction, default=False, 
        help='use concat QKV proj to have share V')
    parser.add_argument('--mlp_fuse', action=argparse.BooleanOptionalAction, default=False, 
        help='use concat up/gated to share Mlp up proj')
    parser.add_argument('--act_aware',action=argparse.BooleanOptionalAction, default=False, 
        help='use act aware SVD with calibration')
    parser.add_argument('--act_alpha', type=float, default=0.5,
        help='sensitivity of ASVD S construction')
    parser.add_argument('--calib_method', type=str, default='abs_mean', choices=['abs_mean', 'abs_max', 'cholesky'], 
        help='calibation method for act aware SVD')
    parser.add_argument('--fisher_info', action=argparse.BooleanOptionalAction, default=False,
        help='calculate Fisher Information for rank allocations for each layer')
    parser.add_argument('--grad_info', action=argparse.BooleanOptionalAction, default=False,
        help='calculate Grad Information for rank allocations for each layer')
    parser.add_argument('--use_cache', type=lambda x: x.lower() in ['true', '1'], default=True, 
                        help='Use previous cached calibration results, default is True!') 
    parser.add_argument('--profile_method', action=argparse.BooleanOptionalAction, default=False, 
        help='whether to skip the eval part, just to profile the model using calibrated dataset')
    parser.add_argument('--beta_then_svd', action=argparse.BooleanOptionalAction, default=False, 
        help='lm svd depend on vit output, so this decides whether svd happens before or after vit module')
    parser.add_argument('--vit_module', action=argparse.BooleanOptionalAction, default=False, 
        help='whether we quantize/rotate vit model')
    parser.add_argument('--vit_online', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use online rotation vit model')
    parser.add_argument('--vit_mmoff', action=argparse.BooleanOptionalAction, default=False, 
        help='whether skip quantize  mmprojector')
    parser.add_argument('--mm_rh', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use rh after gelu in mmprojector')
    parser.add_argument('--lm_off', action=argparse.BooleanOptionalAction, default=False, 
        help='whether skip language model rotate/quantize')
    parser.add_argument('--cache_in_log', action=argparse.BooleanOptionalAction, default=False, 
        help='whether cache calibration results in log file')
    parser.add_argument('--cache_file', type=str, default=None,
        help='path to cached calibration results, default is None!')
    parser.add_argument('--grad_alpha', type=float, default=1.0,
        help='sensitivity of grad score construction')
    parser.add_argument('--bs_to_nsamples', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use full sample size in llava v1.6 beta learning')
    parser.add_argument('--token_length', type=int, default=-1, 
        help='truncate grad info input token length')
    parser.add_argument('--label_mode', type=str, default="q-a", 
        help='how to set input and label for loss and grad')
    parser.add_argument('--case_study', action=argparse.BooleanOptionalAction, default=False, 
        help='whether use case study for llava evaluation')
    
    # Save/Load Quantized Model Arguments
    parser.add_argument('--load_qmodel_path', type=str, default=None,
                        help='Load the quantized model from the specified path!')
    parser.add_argument('--save_qmodel_path', type=str, default=None, 
                        help='Save the quantized model to the specified path!')

    # WandB Arguments
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)



    #Experiments Arguments
    parser.add_argument('--setting', type=str, default="", help='The path to save experiment data, '
                                                                    'including quantized models, dumped layer inputs, etc. The data will be saved in experiments/[model]/save_name. Default: [datetime].')
    parser.add_argument('--save_name', type=str, default=None, help='The path to save experiment data, '
                                                                    'including quantized models, dumped layer inputs, etc. The data will be saved in experiments/[model]/save_name. Default: [datetime].')
    parser.add_argument('--capture_layer_io', action=argparse.BooleanOptionalAction, default=False,
                        help='Capture the input and output of the specified decoder layer and dump into a file')
    parser.add_argument('--layer_idx', type=int, default=10, help='Which decoder layer to capture')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False,
                        help='whether to print the response of the model')
    # LM Eval Arguments
    parser.add_argument("--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    # parser.add_argument(
    #     '--tasks',
    #     nargs='+',
    #     default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada", "boolq"],
    # )
    parser.add_argument(
        '--tasks',
        type=lambda s: s.split(','),
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada", "boolq"],
    )
    parser.add_argument(
        '--vlmtasks',
        nargs='+',
        default=["ScienceQA_TEST"],
    )
    parser.add_argument('--lm_eval_batch_size', type=int, default=16, help='Batch size for evaluating with lm eval harness.')
    parser.add_argument(
        "--distribute",
        action="store_true",
        help="Distribute the model on multiple GPUs for evaluation.",
    )

    args = parser.parse_args()
    if args.localft_lr_R is None:
        args.localft_lr_R = args.localft_lr
    # quant_type = f'w{args.w_bits}a{args.a_bits}_{args.rotate_mode}'
    if args.save_name is None:
        args.save_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.setting = f'W{args.w_bits}A{args.a_bits}K{args.k_bits}V{args.v_bits}'+args.setting
    setattr(args, 'save_path',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', args.model, args.setting))
    os.makedirs(args.save_path, exist_ok=True)
    args.act_cache_dir = f'W{args.w_bits}A{args.a_bits}K{args.k_bits}V{args.v_bits}' + 'wsvd' # for now just use this
    setattr(args, 'act_cache_dir',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', args.model, args.act_cache_dir))
    config_logging(os.path.join(args.save_path, f'{args.save_name}.log'))
    if args.save_qmodel_path:
        args.save_qmodel_path = args.save_path + f'{args.save_qmodel_path}.pth'
    
    assert args.a_groupsize == args.w_groupsize, 'a_groupsize should be the same as w_groupsize!'
    assert args.k_pre_rope == False, 'Pre-RoPE quantization is not supported yet!'

    if args.model == 'facebook/opt-125m' or args.model == 'facebook/opt-1.3b':
        logging.warning('Warning: OPT-125M/1.3B is only for debugging purposes!!')


    if args.wandb:
        assert args.wandb_id is not None and args.wandb_project is not None, 'WandB ID/project is not provided!'
        
    logging.info('Arguments: ')
    logging.info(pprint.pformat(vars(args)))
    logging.info('--' * 30)
    if args.lm_eval:
        from lm_eval import tasks
        from lm_eval import utils as lm_eval_utils
        from lm_eval.tasks import initialize_tasks
        initialize_tasks()
        for task in args.tasks:
            if task not in lm_eval_utils.MultiChoice(tasks.ALL_TASKS):
                raise ValueError(f"Invalid task: {task}")
    return args


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

def distribute_model(model) -> None:
    """Distribute the model across available GPUs. NB: only implemented for Llama-2."""
    no_split_module_classes = ['LlamaDecoderLayer']
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    cleanup_memory()