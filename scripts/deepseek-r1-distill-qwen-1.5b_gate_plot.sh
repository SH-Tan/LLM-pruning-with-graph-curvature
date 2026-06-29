#!/bin/sh
set -e

# Gate distribution plot only for DeepSeek-R1-Distill-Qwen-1.5B.
# This collects layer activations and skips curvature calculation.
model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
python_bin="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
sparsity_ratios="0"
nsamples="${NSAMPLES:-2}"
seed="${SEED:-13}"
alpha="${ALPHA:-0.9}"
model_device="${MODEL_DEVICE:-cuda:0}"
compute_device="${COMPUTE_DEVICE:-cuda:1}"
seq_len="${SEQ_LEN:-512}"
sample_edge_ratio="0.1"
sample_edge_num="-1"
calib_data="${CALIB_DATA:-c4_independent}"
curvature_dir="${CURVATURE_DIR:-out/deepseek_r1_distill_qwen_1.5b/gate_plot/}"
curvature_dtype="${CURVATURE_DTYPE:-float32}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
top_k_seq="${TOP_K_SEQ:-10}"
seq_select="${SEQ_SELECT:-top}"
curvature_lpf_window="${CURVATURE_LPF_WINDOW:-0}"

cuda_device=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
export CUDA_VISIBLE_DEVICES=$cuda_device
export CURV_GATE_PLOT_ENABLED=1
export CURV_COLLECT_LAYER_DATA_ONLY=1
export CURV_GATE_PLOT_MAX_POINTS="${CURV_GATE_PLOT_MAX_POINTS:-200000}"

echo "Running gate plot only: model=$model, nsamples=$nsamples, seq_len=$seq_len"
"$python_bin" src/llm_main.py \
    --model $model \
    --prune_method curvature \
    --sparsity_ratio $sparsity_ratios \
    --sparsity_type unstructured \
    --save $curvature_dir \
    --nsamples $nsamples \
    --seed $seed \
    --model_device $model_device \
    --compute_device $compute_device \
    --alpha $alpha \
    --calib_data $calib_data \
    --sample_edge_ratio $sample_edge_ratio \
    --sample_edge_num $sample_edge_num \
    --seqlen $seq_len \
    --save_curvature_dir $curvature_dir \
    --shared_top_k $top_k_seq \
    --shared_seq_select $seq_select \
    --curvature_lpf_window $curvature_lpf_window \
    --curvature_dtype $curvature_dtype \
    --model_dtype $model_dtype
