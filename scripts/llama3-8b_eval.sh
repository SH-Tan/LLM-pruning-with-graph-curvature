#!/bin/sh
set -e

# All-layer pruning/eval for Llama-3-8B. Per-layer eval is intentionally off.
model="meta-llama/Meta-Llama-3-8B"
python_bin="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
sparsity_ratios="0 0.1 0.3 0.5 0.7 0.9 1.0"
nsamples=3
seed=13
alpha=0.9
model_device="cuda:0"
compute_device="cuda:1"
seq_len=512
sample_edge_ratio=0.01
sample_edge_num=-1
pp_seqlen="$seq_len 1024 2048"
sparsity_schedule="input"

curvature_dir="out/llama_8b/unstructured/curvature/small_set_p_test/"
wanda_dir="out/llama_8b/unstructured/wanda/small_set_p_test/"
magnitude_dir="out/llama_8b/unstructured/magnitude/small_set_p_test/"
all_layer_compare_dir="out/llama_8b/unstructured/all_layer_compare/small_set_p_test/"

top_k_seq=10
seq_select="median"
curvature_lpf_window=0
l2_norm_mode="${L2_NORM_MODE:-per_example}"

cuda_device=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
export CUDA_VISIBLE_DEVICES=$cuda_device

run_python_command() {
    prune_method=$1
    sparsity_type=$2
    save_dir=$3
    calib_data=$4
    prune_score_order=$5
    shared_top_k=${6:-$top_k_seq}
    seq_select_arg=${7:-$seq_select}
    lpf_window=${8:-$curvature_lpf_window}
    use_l2_norm=${9:-0}
    prune_scope=${10:-global}
    prunescore_scope=${11:-globally}

    l2_flag=""
    if [ "$use_l2_norm" = "1" ]; then
        l2_flag="--L2-norm --l2_norm_mode $l2_norm_mode"
    fi

    "$python_bin" llm_main.py \
        --model $model \
        --prune_method $prune_method \
        --sparsity_ratio $sparsity_ratios \
        --sparsity_type $sparsity_type \
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
        --sparsity_schedule $sparsity_schedule \
        --save_curvature_dir $curvature_dir \
        --load_curvature_dir $curvature_dir \
        --curvature_prune_scope $prune_scope \
        --prunescore_order $prunescore_scope \
        --shared_top_k $shared_top_k \
        --shared_seq_select $seq_select_arg \
        --curvature_lpf_window $lpf_window \
        --per_layer_compare_dir $all_layer_compare_dir \
        --run_pp_eval \
        $l2_flag
}

run_curvature_pair() {
    label=$1
    shared_top_k=$2
    seq_select_arg=$3
    lpf_window=$4
    use_l2_norm=$5

    echo "Running all-layer curvature $label with global score ordering"
    run_python_command "curvature" "unstructured" "$curvature_dir" "c4_independent" "high_to_low" "$shared_top_k" "$seq_select_arg" "$lpf_window" "$use_l2_norm" "global" "globally"

    echo "Running all-layer curvature $label with local per-layer score ordering"
    run_python_command "curvature" "unstructured" "$curvature_dir" "c4_independent" "high_to_low" "$shared_top_k" "$seq_select_arg" "$lpf_window" "$use_l2_norm" "per_layer" "locally"

    echo "Finished all-layer curvature $label"
}

# Curvature: five score variants x two prune-score scopes = 10 curves.
run_curvature_pair "all_seq_no_L2" -1 "top" 0 0
run_curvature_pair "all_seq_lpf" -1 "top" 5 0
run_curvature_pair "top_10_seq" 10 "top" 0 0
run_curvature_pair "median_10_seq" 10 "median" 0 0
run_curvature_pair "all_seq_L2" -1 "top" 0 1

echo "Running all-layer WANDA pruning/eval"
run_python_command "wanda" "unstructured" "$wanda_dir" "c4_independent" "low_to_high" "$top_k_seq" "$seq_select" "$curvature_lpf_window" 0 "global" "globally"
echo "Finished all-layer WANDA pruning/eval"

echo "Running all-layer magnitude pruning/eval"
run_python_command "magnitude" "unstructured" "$magnitude_dir" "c4_independent" "low_to_high" "$top_k_seq" "$seq_select" "$curvature_lpf_window" 0 "global" "globally"
echo "Finished all-layer magnitude pruning/eval"
