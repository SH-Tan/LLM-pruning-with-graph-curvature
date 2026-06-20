#!/bin/sh
set -e

# All-layer pruning/eval for Llama-3-8B.
model="meta-llama/Meta-Llama-3-8B"
python_bin="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
sparsity_ratios="0 0.3 0.4 0.5 0.6 0.7 0.9 1"
nsamples=5
seed=13
alpha=0.9
model_device="cuda:0"
compute_device="${COMPUTE_DEVICE:-cuda:1}"
seq_len=512
sample_edge_ratio=0.1
sample_edge_num=-1
pp_seqlen="$seq_len 1024"
calib_data="c4_independent"
prune_ops="${PRUNE_OPS:-gate_proj}"

curvature_dir="${CURVATURE_DIR:-out/llama_8b/unstructured/curvature/gate1/}"
wanda_dir="${WANDA_DIR:-out/llama_8b/unstructured/wanda/all_seq_compare/}"
magnitude_dir="${MAGNITUDE_DIR:-out/llama_8b/unstructured/magnitude/all_seq_compare/}"
compare_dir="${COMPARE_DIR:-out/llama_8b/unstructured/all_layer_compare/gate1/}"

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
    eval_flag=$9
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
        --prune_ops $prune_ops \
        --per_layer_compare_dir $compare_dir \
        $eval_flag
}

echo "Running all-layer curvature globally"
run_python_command "curvature" "$curvature_dir" 10 "top" 0 "globally" "high_to_low" "$compare_dir" "--run_pp_eval"
echo "Finished all-layer curvature globally"

echo "Running all-layer WANDA globally"
run_python_command "wanda" "$wanda_dir" 10 "top" 0 "globally" "low_to_high" "$compare_dir" "--run_pp_eval"
echo "Finished all-layer WANDA globally"

echo "Running all-layer magnitude globally"
run_python_command "magnitude" "$magnitude_dir" 10 "top" 0 "globally" "low_to_high" "$compare_dir" "--run_pp_eval"
echo "Finished all-layer magnitude globally"

echo "Running all-layer curvature locally"
run_python_command "curvature" "$curvature_dir" 10 "top" 0 "locally" "high_to_low" "$compare_dir" "--run_pp_eval"
echo "Finished all-layer curvature locally"

echo "Running all-layer WANDA locally"
run_python_command "wanda" "$wanda_dir" 10 "top" 0 "locally" "low_to_high" "$compare_dir" "--run_pp_eval"
echo "Finished all-layer WANDA locally"

echo "Running all-layer magnitude locally"
run_python_command "magnitude" "$magnitude_dir" 10 "top" 0 "locally" "low_to_high" "$compare_dir" "--run_pp_eval"
echo "Finished all-layer magnitude locally"

# echo "Running all-layer curvature per_op"
# run_python_command "curvature" "$curvature_dir" 10 "top" 0 "per_op" "high_to_low" "$compare_dir" "--run_pp_eval"
# echo "Finished all-layer curvature per_op"

# echo "Running all-layer WANDA per_op"
# run_python_command "wanda" "$wanda_dir" 10 "top" 0 "per_op" "low_to_high" "$compare_dir" "--run_pp_eval"
# echo "Finished all-layer WANDA per_op"

# echo "Running all-layer magnitude per_op"
# run_python_command "magnitude" "$magnitude_dir" 10 "top" 0 "per_op" "low_to_high" "$compare_dir" "--run_pp_eval"
# echo "Finished all-layer magnitude per_op"
