export HF_HOME='path_to_huggingface'
cd path_to_wsvd/fake_quant


SEEDS=(${1:-0})
BSs=(${2:-64})
RANKRATIOs=(${3:-0.9})
wbits=${4:-16}
bits=${5:-16}
localft_iters=(${6:-100})
localft_lrs=(${7:-1e-4})
taylor_order=(${8:-2})
svd_mode="U"
bsz=64
for localft_iter in "${localft_iters[@]}"; do
    for localft_lr in "${localft_lrs[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for rank_ratio in "${RANKRATIOs[@]}"; do
                for bs in "${BSs[@]}"; do
                    echo "Running experiment with bs=${bs}, seed=${seed}"
                    echo "svd_mode=${svd_mode}, rank_ratio=${rank_ratio}"
                    python main_llava.py \
                        --model liuhaotian/llava-v1.5-7b  \
                        --a_bits "$bits" \
                        --w_bits "$wbits" \
                        --k_bits 16 \
                        --v_bits 16 \
                        --cal_dataset ScienceQA_Train \
                        --eval_dataset ScienceQA_TEST \
                        --tasks None \
                        --w_clip \
                        --nsamples "$bs" \
                        --bs "$bsz" \
                        --seed "$seed" \
                        --svd_mode "$svd_mode" \
                        --svd_modules "qkv" \
                        --svd_lm_localft \
                        --localft_iters "$localft_iter" \
                        --localft_lr "$localft_lr" \
                        --svd_lm \
                        --act_aware \
                        --act_alpha 0.5 \
                        --taylor_order "$taylor_order" \
                        --svd_ft_mode "weight" \
                        --svd_ft_optim "adam" \
                        --calib_method 'cholesky' \
                        --rank_ratio "$rank_ratio" \
                        --weighted_svd \
                        --is_per_head_svd \
                        --is_rank_allocate_ft \
                        --setting "wsvd/sqa/noq/paramRatio${rank_ratio}${svd_mode}_mean${bs}_${bsz}/ftIter${localft_iter}_ftLr${localft_lr}/seed${seed}" \
                        --use_true_param_ratio \
                        --grad_info \
                        --cache_in_log 
                done
            done
        done
    done
done