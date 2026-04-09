To reproduce Table 2 results with cache files, run:
```bash
# Step 1: Download cache files to wsvd/cache_file
bash use_cache/download_cache.sh

# Step 2: Run WSVD QAT scripts that use downloaded cache
#
# In each shell script, replace the placeholders at the top:
# - Set export HF_HOME='path_to_huggingface'
# - Set cd path_to_wsvd/fake_quant
# Then run:
bash use_cache/wsvd_qat_llava_1.5_7b.sh 0 64 0.5 8 8 150 1e-4 2 1.0 1e-5 0.66666667 
bash use_cache/wsvd_qat_llava_1.5_13b.sh 0 64 0.5 8 8 150 1e-4 2 1.0 1e-5 0.66666667
bash use_cache/wsvd_qat_llava_next_7b.sh 0 64 0.5 8 8 150 1e-4 2 1.0 1e-5 0.66666667 
bash use_cache/wsvd_qat_llava_next_13b.sh 0 64 0.5 8 8 150 1e-4 2 1.0 1e-5 0.66666667
```

This script will automatically:

1. Load the pre-computed calibration cache from [`cache_file/llava-next-13b`](cache_file/llava-next-13b).  
2. Compress the model with WSVD under the default quantized configuration (SVD compress 50% + W8A8).
3. Evaluate the compressed model.

Notes:
- The argument `0.5` in the command specifies the **retained parameter ratio $\rho_1$** used for overall SVD compression.
- The number of iteration for local FT is 100 ($150\times 0.66667$), for QAT is 50 ($150\times 0.33333$).
- Results may still vary slightly, because the same random seed can produce different local FT/QAT behavior on different GPU models.

To test quantized models without cache files, run:
```bash
bash wsvd_qat_llava_1.5_7b.sh 0 64 0.5 8 8 150 1e-4 2 1.0 1e-5 0.66666667 
bash wsvd_qat_llava_1.5_13b.sh 0 64 0.5 8 8 150 1e-4 2 1.0 1e-5 0.66666667
bash wsvd_qat_llava_next_7b.sh 0 64 0.5 8 8 150 1e-4 2 1.0 1e-5 0.66666667 
bash wsvd_qat_llava_next_13b.sh 0 64 0.5 8 8 150 1e-4 2 1.0 1e-5 0.66666667
```