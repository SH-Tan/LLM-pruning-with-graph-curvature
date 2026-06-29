#!/bin/sh
set -e

# PPL eval from saved curvature PKLs for Llama-3-8B.
# Uses the Transformers model path in src/llm_main.py; no downstream/vLLM eval.
model="${MODEL:-meta-llama/Meta-Llama-3-8B}"
python_bin="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
sparsity_ratios="${SPARSITY_RATIOS:-0 0.3 0.4 0.5 0.6 0.7 0.9 1}"
nsamples="${NSAMPLES:-5}"
seed="${SEED:-13}"
alpha="${ALPHA:-0.9}"
model_device="${MODEL_DEVICE:-cuda:0}"
compute_device="${COMPUTE_DEVICE:-cuda:1}"
seq_len="${SEQ_LEN:-512}"
sample_edge_ratio="${SAMPLE_EDGE_RATIO:-0.1}"
sample_edge_num="${SAMPLE_EDGE_NUM:--1}"
pp_seqlen="${PP_SEQLEN:-$seq_len 1024}"
calib_data="${CALIB_DATA:-c4_independent}"
prune_ops="${PRUNE_OPS:-gate_proj up_proj}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
top_k_seq="${TOP_K_SEQ:-10}"
seq_select="${SEQ_SELECT:-top}"
curvature_lpf_window="${CURVATURE_LPF_WINDOW:-0}"
run_all_layer_eval="${RUN_ALL_LAYER_EVAL:-${RUN_PP_EVAL:-1}}"
run_per_layer_eval="${RUN_PER_LAYER_EVAL:-0}"

curvature_dir="${CURVATURE_DIR:-out/llama_8b/unstructured/curvature/gate_up_resid/}"
wanda_dir="${WANDA_DIR:-out/llama_8b/unstructured/wanda/all_seq_compare/}"
magnitude_dir="${MAGNITUDE_DIR:-out/llama_8b/unstructured/magnitude/all_seq_compare/}"
compare_dir_root="${COMPARE_DIR_ROOT:-out/llama_8b/unstructured/all_layer_compare}"
mlp_activation="${MLP_ACTIVATION:-model}"
echo "Eval MLP activation: $mlp_activation"

cuda_device=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
export CUDA_VISIBLE_DEVICES=$cuda_device

run_python_command() {
    prune_method=$1
    save_dir=$2
    prunescore_order=$3
    prune_score_order=$4
    eval_flag=$5

    "$python_bin" src/llm_main.py \
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
        --curvature_lpf_window $curvature_lpf_window \
        --prune_ops $prune_ops \
        --per_layer_compare_dir $compare_dir \
        --model_dtype $model_dtype \
        --mlp_activation $mlp_activation \
        $eval_flag
}

run_scope_all_methods() {
    prunescore_order=$1
    scope_name=$2

    echo "Running all-layer curvature $scope_name"
    run_python_command "curvature" "$curvature_dir" "$prunescore_order" "high_to_low" "--run_pp_eval"
    echo "Finished all-layer curvature $scope_name"

    echo "Running all-layer WANDA $scope_name"
    run_python_command "wanda" "$wanda_dir" "$prunescore_order" "low_to_high" "--run_pp_eval"
    echo "Finished all-layer WANDA $scope_name"

    echo "Running all-layer magnitude $scope_name"
    run_python_command "magnitude" "$magnitude_dir" "$prunescore_order" "low_to_high" "--run_pp_eval"
    echo "Finished all-layer magnitude $scope_name"
}

run_eval_for_prune_ops() {
    prune_ops=$1
    compare_dir=$2
    echo "Eval prune ops: $prune_ops"
    echo "Eval compare dir: $compare_dir"

    if [ "$run_all_layer_eval" = "1" ]; then
        run_scope_all_methods "globally" "globally"
        run_scope_all_methods "locally" "locally"
        run_scope_all_methods "per_op" "per-op"
    fi

    if [ "$run_per_layer_eval" = "1" ]; then
        echo "Running per-layer curvature locally"
        run_python_command "curvature" "$curvature_dir" "locally" "high_to_low" "--run_per_layer_eval"
        echo "Finished per-layer curvature locally"

        echo "Running per-layer WANDA locally"
        run_python_command "wanda" "$wanda_dir" "locally" "low_to_high" "--run_per_layer_eval"
        echo "Finished per-layer WANDA locally"

        echo "Running per-layer magnitude locally"
        run_python_command "magnitude" "$magnitude_dir" "locally" "low_to_high" "--run_per_layer_eval"
        echo "Finished per-layer magnitude locally"
    fi
}

run_eval_for_prune_ops "gate_proj up_proj" "$compare_dir_root/gate_up_resid/"
run_eval_for_prune_ops "up_proj" "$compare_dir_root/up_resid/"
run_eval_for_prune_ops "gate_proj" "$compare_dir_root/gate_resid/"
