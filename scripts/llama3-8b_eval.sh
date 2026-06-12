#!/bin/sh
set -e

# All-layer pruning/eval for Llama-3-8B.
model="meta-llama/Meta-Llama-3-8B"
python_bin="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
sparsity_ratios="0 0.3 0.5 0.7 0.9 1"
nsamples=5
seed=13
alpha=0.9
model_device="cuda:0"
compute_device="${COMPUTE_DEVICE:-cuda:1}"
seq_len=512
sample_edge_ratio=0.2
sample_edge_num=-1
pp_seqlen="$seq_len 1024"
calib_data="c4_independent"

curvature_dir="out/llama_8b/unstructured/curvature/noresidual/"
wanda_dir="out/llama_8b/unstructured/wanda/all_seq_compare/"
magnitude_dir="out/llama_8b/unstructured/magnitude/all_seq_compare/"
compare_dir="out/llama_8b/unstructured/all_layer_compare/noresidual/"

cuda_device=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
export CUDA_VISIBLE_DEVICES=$cuda_device

run_python_command() {
    prune_method=$1
    save_dir=$2
    top_k_seq=$3
    seq_select=$4
    lpf_window=$5
    prunescore_order=$6
    prune_score_order=$7
    compare_dir=$8

    "$python_bin" llm_main.py \
        --model $model \
        --prune_method $prune_method \
        --sparsity_ratio $sparsity_ratios \
        --sparsity_type unstructured \
        --save $save_dir \
        --nsamples $nsamples \
        --seed $seed \
        --model_device $model_device \
        --compute_device $compute_device \
        --alpha $alpha \
        --calib_data $calib_data \
        --sample_edge_ratio $sample_edge_ratio \
        --sample_edge_num $sample_edge_num \
        --seqlen $seq_len \
        --pp_seqlen $pp_seqlen \
        --prune_score_order $prune_score_order \
        --sparsity_schedule input \
        --save_curvature_dir $curvature_dir \
        --load_curvature_dir $curvature_dir \
        --prunescore_order $prunescore_order \
        --shared_top_k $top_k_seq \
        --shared_seq_select $seq_select \
        --curvature_lpf_window $lpf_window \
        --per_layer_compare_dir $compare_dir \
        --run_pp_eval
}

echo "Running all-layer curvature local"
run_python_command "curvature" "$curvature_dir" 10 "top" 0 "locally" "high_to_low" "$compare_dir"
echo "Finished all-layer curvature local"

# echo "Running all-layer curvature global"
# run_python_command "curvature" "$curvature_dir" 10 "top" 0 "globally" "high_to_low" "$compare_dir"
# echo "Finished all-layer curvature global"

echo "Running all-layer WANDA local"
run_python_command "wanda" "$wanda_dir" 10 "top" 0 "locally" "low_to_high" "$compare_dir"
echo "Finished all-layer WANDA local"

# echo "Running all-layer WANDA global"
# run_python_command "wanda" "$wanda_dir" 10 "top" 0 "globally" "low_to_high" "$compare_dir"
# echo "Finished all-layer WANDA global"

echo "Running all-layer magnitude local"
run_python_command "magnitude" "$magnitude_dir" 10 "top" 0 "locally" "low_to_high" "$compare_dir"
echo "Finished all-layer magnitude local"

# echo "Running all-layer magnitude global"
# run_python_command "magnitude" "$magnitude_dir" 10 "top" 0 "globally" "low_to_high" "$compare_dir"
# echo "Finished all-layer magnitude global"
