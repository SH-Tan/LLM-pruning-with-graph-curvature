#!/bin/sh
set -e

# Curvature calculation only for Llama-3-8B. No pruning/eval is run.
model="meta-llama/Meta-Llama-3-8B"
python_bin="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
sparsity_ratios="0"
nsamples=5
seed=13
alpha=0.9
model_device="cuda:0"
compute_device="cuda:1"
seq_len=512
sample_edge_ratio=0.2
sample_edge_num=-1
calib_data="c4_independent"
curvature_dir="out/llama_8b/unstructured/curvature/Q_0.2/"

cuda_device=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
export CUDA_VISIBLE_DEVICES=$cuda_device

run_curvature_calculation() {
    use_l2_norm=$1
    l2_mode=$2
    l2_flag=""
    if [ "$use_l2_norm" = "1" ]; then
        l2_flag="--L2-norm --l2_norm_mode $l2_mode"
    fi

    echo "Running curvature calculation: use_l2_norm=$use_l2_norm, l2_norm_mode=$l2_mode, top_k_seq=$top_k_seq, seq_select=$seq_select"
    "$python_bin" llm_main.py \
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
        $l2_flag
}

# Run list. Keep each setting explicit so it is easy to comment out or add variants.

# 1. L2 norm type 1: current behavior, L2 per example over all sequence positions.
top_k_seq=-1
seq_select="top"
curvature_lpf_window=0
run_curvature_calculation 1 "per_example"

# 2. L2 norm type 2: Wanda-style, L2 over all examples and all sequence positions.
top_k_seq=-1
seq_select="top"
curvature_lpf_window=0
run_curvature_calculation 1 "all_examples"

# 3. No L2: select top 10 sequence positions.
top_k_seq=10
seq_select="top"
curvature_lpf_window=0
run_curvature_calculation 0 "per_example"

# Future examples:
# top_k_seq=10
# seq_select="median"
# curvature_lpf_window=0
# run_curvature_calculation 0 "per_example"
