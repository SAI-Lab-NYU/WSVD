export HF_HOME='/vast/yw6594/log'
export CUDA_VISIBLE_DEVICES=0
cd /scratch/yw6594/cf/vlm/quant/QuaRot/e2e
python benchmark_llava.py \
    --prefill_seq_len 1024 \
    --decode_steps 1
