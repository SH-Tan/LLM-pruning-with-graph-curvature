#!/bin/sh

# Set common variables
model="meta-llama/Meta-Llama-3-8B"
python_bin="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
sparsity_ratios="0.1 0.5 0.7 1.0"
nsamples=3
seed=13
alpha=0.9
model_device="cuda:0"
compute_device="cuda:1"
seq_len=512
sample_edge_ratio=0.1
sample_edge_num=-1
prune_score_orders="high_to_low low_to_high"
sparsity_schedule="input"
# pp_seqlen="1024"
pp_seqlen="$seq_len 1024 2048"
curvature_dir="out/llama_8b/unstructured/curvature/"
load_curvature_non_curvature=1
load_curvature_curvature=0
# cuda_device=0
cuda_device=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command() {
    prune_score_order=${5:-low_to_high}
    use_l2_norm=${6:-0}
    l2_flag=""
    load_curvature_flag=""
    if [ "$use_l2_norm" = "1" ]; then
        l2_flag="--L2-norm"
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
    $load_curvature_flag
}

# llama-7b with magnitude pruning method
echo "Running with graph curvature pruning method"
run_python_command "curvature" "unstructured" "out/llama_8b/unstructured/curvature/" "c4_independent" "$prune_score_orders"
# run_python_command "curvature" "unstructured" "out/llama_8b/unstructured/curvature/" "c4_independent" "$prune_score_orders" 1
# echo "Finished graph curvature pruning method"


# llama-7b with wanda pruning method
echo "Running with wanda pruning method"
run_python_command "wanda" "unstructured" "out/llama_8b/unstructured/wanda/" "c4_independent"
# run_python_command "wanda" "2:4" "out/llama_8b/2-4/wanda/"
# run_python_command "wanda" "4:8" "out/llama_8b/4-8/wanda/"
# echo "Finished wanda pruning method"


# llama-7b with magnitude pruning method
echo "Running with magnitude pruning method"
run_python_command "magnitude" "unstructured" "out/llama_8b/unstructured/magnitude/" "c4_independent"
# run_python_command "magnitude" "2:4" "out/llama_8b/2-4/magnitude/"
# run_python_command "magnitude" "4:8" "out/llama_8b/4-8/magnitude/"
# echo "Finished magnitude pruning method"
