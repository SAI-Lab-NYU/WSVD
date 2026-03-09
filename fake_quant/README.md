## VLM Evaluations

In this directory, we provide the source code for WSVD. 


Currently, we only support **LLaVA-v1.5 7B, 13B** and **LLaVA-Next 7B, 13B** models. You can simply run the `main_llava.py` or `main_llava_next.py` to reproduce the results in the paper. The most important arguments are:

- `--model`: the model name (or path to the weights)
- `--bsz`: the batch size for PPL evaluation
- `--rotate`: whether we want to rotate the model
- `--lm_eval`: whether we want to run LM-Eval for Zero-Shot tasks
- `--tasks`: the tasks for LM-Eval
- `--cal_dataset`: the calibration dataset for GPTQ quantization
- `--a_bits`: the number of bits for activation quantization
- `--w_bits`: the number of bits for weight quantization
- `--v_bits`: the number of bits for value quantization
- `--k_bits`: the number of bits for key quantization
- `--w_clip`: Whether we want to clip the weights
- `--a_clip_ratio`: The ratio of clipping for activation
- `--k_clip_ratio`: The ratio of clipping for key
- `--v_clip_ratio`: The ratio of clipping for value
- `--w_asym`: Whether we want to use asymmetric quantization for weights
- `--a_asym`: Whether we want to use asymmetric quantization for activation
- `--v_asym`: Whether we want to use asymmetric quantization for value
- `--k_asym`: Whether we want to use asymmetric quantization for key
- `--a_groupsize`: The group size for activation quantization
- `--w_groupsize`: The group size for weight quantization
- `--v_groupsize`: The group size for value quantization
- `--k_groupsize`: The group size for key quantization
- `--weighted_svd`: Enable WSVD local FT and QAT (with `--is_quant_aware_ft`).
- `--rank_ratio`: When `--use_true_param_ratio` is provided, this is interpreted as the parameter ratio. 
- `--localft_iters`: Total iterations of local fine-tuning plus QAT.
- `--localft_lr`: Learning rate for local fine-tuning.
- `--qat_start_iter`: Fraction in [0, 1] indicating when QAT starts within the total iterations; use this to split FT and QAT iteration counts. Iterations for FT: localft\_iters $\times$ qat\_start\_iter; for QAT: localft\_iters $\times$ (1 - qat\_start\_iter).
- `--qat_lr_UV`: Learning rate for low-rank factors (`U/V`) during local QAT.
- `--qat_lr_R`: Learning rate for rotation matrix `R` during local QAT.
