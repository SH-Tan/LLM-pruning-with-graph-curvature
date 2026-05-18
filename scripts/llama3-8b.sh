#!/bin/sh

# Set common variables
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
prune_score_orders="high_to_low low_to_high"
sparsity_schedule="input"
# pp_seqlen="1024"
pp_seqlen="$seq_len 1024 2048"
curvature_dir="out/llama_8b/unstructured/curvature/debug/"
wanda_dir="out/llama_8b/unstructured/wanda/debug/"
magnitude_dir="out/llama_8b/unstructured/magnitude/debug/"
load_curvature_non_curvature=1
load_curvature_curvature=0
curvature_prune_scope="per_layer"
top_k_seq=10
seq_select="median"
curvature_lpf_window=0
run_pp_eval=0
run_per_layer_eval=0
# cuda_device=0
cuda_device=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command() {
    prune_score_order=${5:-low_to_high}
    shared_top_k=${6:-$top_k_seq}
    seq_select_arg=${7:-$seq_select}
    lpf_window=${8:-$curvature_lpf_window}
    use_l2_norm=${9:-0}
    l2_flag=""
    pp_eval_flag=""
    per_layer_eval_flag=""
    load_curvature_flag=""
    if [ "$use_l2_norm" = "1" ]; then
        l2_flag="--L2-norm"
    fi
    if [ "$run_pp_eval" = "1" ]; then
        pp_eval_flag="--run_pp_eval"
    fi
    if [ "$run_per_layer_eval" = "1" ]; then
        per_layer_eval_flag="--run_per_layer_eval"
    fi
    if [ "$1" = "curvature" ] && [ "$load_curvature_curvature" = "1" ]; then
        load_curvature_flag="--load_curvature_dir $curvature_dir"
    elif [ "$1" != "curvature" ] && [ "$load_curvature_non_curvature" = "1" ]; then
        load_curvature_flag="--load_curvature_dir $curvature_dir"
    fi
    "$python_bin" llm_main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratios \
    --sparsity_type $2 \
    --save $3 \
    --nsamples $nsamples \
    --seed $seed \
    --model_device $model_device \
    --compute_device $compute_device \
    --alpha $alpha \
    --calib_data $4 \
    --sample_edge_ratio $sample_edge_ratio \
    --sample_edge_num $sample_edge_num \
    --seqlen $seq_len \
    --pp_seqlen $pp_seqlen \
    --prune_score_order $prune_score_order \
    --sparsity_schedule $sparsity_schedule \
    --save_curvature_dir $curvature_dir \
    --curvature_prune_scope $curvature_prune_scope \
    --shared_top_k $shared_top_k \
    --shared_seq_select $seq_select_arg \
    --curvature_lpf_window $lpf_window \
    --save_parameter_metric_logs \
    $pp_eval_flag \
    $per_layer_eval_flag \
    $l2_flag \
    $load_curvature_flag
}

echo "Running graph curvature calculation with per-parameter logs"
# run_python_command "curvature" "unstructured" "$curvature_dir" "c4_independent" "$prune_score_orders" -1 "top" 0
# run_python_command "curvature" "unstructured" "$curvature_dir" "c4_independent" "$prune_score_orders" -1 "top" 5
run_python_command "curvature" "unstructured" "$curvature_dir" "c4_independent" "$prune_score_orders" 10 "top" 0
# run_python_command "curvature" "unstructured" "$curvature_dir" "c4_independent" "$prune_score_orders" 10 "median" 0
# run_python_command "curvature" "unstructured" "$curvature_dir" "c4_independent" "$prune_score_orders" -1 "top" 0 1
echo "Finished graph curvature calculation"


# echo "Running per-layer WANDA pruning eval"
# run_python_command "wanda" "unstructured" "$wanda_dir" "c4_independent"
# echo "Finished per-layer WANDA pruning eval"


# echo "Running per-layer magnitude pruning eval"
# run_python_command "magnitude" "unstructured" "$magnitude_dir" "c4_independent"
# echo "Finished per-layer magnitude pruning eval"
