#!/bin/sh
set -e

# Eval from saved curvature PKLs for Llama-3.2-1B-Instruct.
model="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
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
prune_ops="${PRUNE_OPS:-gate_proj}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
top_k_seq="${TOP_K_SEQ:-10}"
seq_select="${SEQ_SELECT:-top}"
curvature_lpf_window="${CURVATURE_LPF_WINDOW:-0}"
run_all_layer_eval="${RUN_ALL_LAYER_EVAL:-${RUN_PP_EVAL:-1}}"
run_per_layer_eval="${RUN_PER_LAYER_EVAL:-0}"
run_downstream_test="${RUN_DOWNSTREAM_TEST:-1}"
downstream_only="${DOWNSTREAM_ONLY:-1}"
downstream_task_data="${DOWNSTREAM_TASK_DATA:-downstream_test/dataset/mathqa500/test.parquet}"
downstream_output_dir="${DOWNSTREAM_OUTPUT_DIR:-}"
downstream_vllm_python="${DOWNSTREAM_VLLM_PYTHON:-${VLLM_PYTHON:-/home/tans5/anaconda3/envs/vllm/bin/python}}"
downstream_batch_size="${DOWNSTREAM_BATCH_SIZE:-1}"
downstream_generation_max_batch_tokens="${DOWNSTREAM_GENERATION_MAX_BATCH_TOKENS:-32768}"
downstream_max_prompt_length="${DOWNSTREAM_MAX_PROMPT_LENGTH:-2048}"
downstream_max_new_tokens="${DOWNSTREAM_MAX_NEW_TOKENS:-2048}"
downstream_min_tokens="${DOWNSTREAM_MIN_TOKENS:-16}"
downstream_temperature="${DOWNSTREAM_TEMPERATURE:-0.0}"
downstream_top_p="${DOWNSTREAM_TOP_P:-1.0}"
downstream_top_k="${DOWNSTREAM_TOP_K:-0}"
downstream_response_log_max="${DOWNSTREAM_RESPONSE_LOG_MAX:-50}"

curvature_dir="${CURVATURE_DIR:-out/llama3.2_1b_instruct/unstructured/curvature/gate_wores/}"
wanda_dir="${WANDA_DIR:-out/llama3.2_1b_instruct/unstructured/wanda/all_seq_compare/}"
magnitude_dir="${MAGNITUDE_DIR:-out/llama3.2_1b_instruct/unstructured/magnitude/all_seq_compare/}"
compare_dir="${COMPARE_DIR:-out/llama3.2_1b_instruct/unstructured/all_layer_compare/gate_wores/}"

cuda_device=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
export CUDA_VISIBLE_DEVICES=$cuda_device

run_python_command() {
    prune_method=$1
    save_dir=$2
    prunescore_order=$3
    prune_score_order=$4
    eval_flag=$5

    downstream_flag=""
    if [ "$run_downstream_test" = "1" ] && [ "$eval_flag" = "--run_pp_eval" ]; then
        downstream_flag="--run_downstream_eval"
        if [ "$downstream_only" = "1" ]; then
            downstream_flag="$downstream_flag --downstream_only"
        fi
    fi

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
        --downstream_task_data "$downstream_task_data" \
        --downstream_output_dir "$downstream_output_dir" \
        --downstream_vllm_python "$downstream_vllm_python" \
        --downstream_batch_size "$downstream_batch_size" \
        --downstream_generation_max_batch_tokens "$downstream_generation_max_batch_tokens" \
        --downstream_max_prompt_length "$downstream_max_prompt_length" \
        --downstream_max_new_tokens "$downstream_max_new_tokens" \
        --downstream_min_tokens "$downstream_min_tokens" \
        --downstream_temperature "$downstream_temperature" \
        --downstream_top_p "$downstream_top_p" \
        --downstream_top_k "$downstream_top_k" \
        --downstream_response_log_max "$downstream_response_log_max" \
        $downstream_flag \
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

if [ "$run_all_layer_eval" = "1" ]; then
    run_scope_all_methods "globally" "globally"
    run_scope_all_methods "locally" "locally"
fi


    # run_scope_all_methods "per_op" "per-op"

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
