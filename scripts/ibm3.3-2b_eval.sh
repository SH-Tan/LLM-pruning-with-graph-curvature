#!/bin/sh
set -e

# PPL and lm-eval downstream eval from saved curvature PKLs for IBM Granite 3.3 2B.
model="${MODEL:-ibm-granite/granite-3.3-2b-instruct}"
python_bin="${PYTHON_BIN:-python}"
sparsity_ratios="${SPARSITY_RATIOS:-0 0.3 0.4 0.5 0.6 0.7 0.9 1}"
nsamples="${NSAMPLES:-128}"
seed="${SEED:-13}"
alpha="${ALPHA:-0.9}"
model_device="${MODEL_DEVICE:-cuda:0}"
compute_device="${COMPUTE_DEVICE:-cuda:1}"
seq_len="${SEQ_LEN:-512}"
sample_edge_ratio="${SAMPLE_EDGE_RATIO:-0.5}"
sample_edge_num="${SAMPLE_EDGE_NUM:--1}"
pp_seqlen="${PP_SEQLEN:-$seq_len 1024}"
calib_data="${CALIB_DATA:-c4_independent}"
prune_ops="${PRUNE_OPS:-gate_proj}"
skip_prune_layer_ids="${SKIP_PRUNE_LAYER_IDS:-}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
use_l2_norm="${USE_L2_NORM:-0}"
l2_norm_mode="${L2_NORM_MODE:-all_examples}"
top_k_seq="${TOP_K_SEQ:-10}"
seq_select="${SEQ_SELECT:-top}"
curvature_lpf_window="${CURVATURE_LPF_WINDOW:-0}"
run_all_layer_eval="${RUN_ALL_LAYER_EVAL:-${RUN_PP_EVAL:-1}}"
run_per_layer_eval="${RUN_PER_LAYER_EVAL:-0}"
run_downstream_test="${RUN_DOWNSTREAM_TEST:-1}"
downstream_only="${DOWNSTREAM_ONLY:-1}"
downstream_tasks="${DOWNSTREAM_TASKS:-gsm8k}"
downstream_batch_size="${DOWNSTREAM_BATCH_SIZE:-auto}"
downstream_hf_batch_size="${DOWNSTREAM_HF_BATCH_SIZE:-auto}"
downstream_hf_max_batch_size="${DOWNSTREAM_HF_MAX_BATCH_SIZE:-8}"
downstream_hf_gpu_memory_utilization="${DOWNSTREAM_HF_GPU_MEMORY_UTILIZATION:-0.7}"
downstream_output_dir="${DOWNSTREAM_OUTPUT_DIR:-}"
downstream_summary_csv="${DOWNSTREAM_SUMMARY_CSV:-eval_results/summary.csv}"
downstream_suite="${DOWNSTREAM_SUITE:-core}"
downstream_suite_benchmarks="${DOWNSTREAM_SUITE_BENCHMARKS:-mmlu;winogrande;truthfulqa;gsm8k;ifeval}"
downstream_suite_tasks="${DOWNSTREAM_SUITE_TASKS:-mmlu_stem,mmlu_social_sciences;winogrande;truthfulqa_mc1,truthfulqa_mc2;gsm8k;ifeval}"
downstream_suite_backends="${DOWNSTREAM_SUITE_BACKENDS:-hf;hf;hf;vllm;vllm}"
downstream_suite_fewshots="${DOWNSTREAM_SUITE_FEWSHOTS:-5;5;0;8;0}"
downstream_suite_limits="${DOWNSTREAM_SUITE_LIMITS:-0.25;1000;500;500;500}"
downstream_num_fewshot="${DOWNSTREAM_NUM_FEWSHOT:-5}"
downstream_apply_chat_template="${DOWNSTREAM_APPLY_CHAT_TEMPLATE:-0}"
downstream_fewshot_as_multiturn="${DOWNSTREAM_FEWSHOT_AS_MULTITURN:-0}"
downstream_chat_template_args="${DOWNSTREAM_CHAT_TEMPLATE_ARGS:-}"
downstream_gen_kwargs="${DOWNSTREAM_GEN_KWARGS:-max_gen_toks=2048}"
downstream_limit="${DOWNSTREAM_LIMIT:-}"
downstream_lm_eval_backend="${DOWNSTREAM_LM_EVAL_BACKEND:-vllm}"
downstream_vllm_python="${DOWNSTREAM_VLLM_PYTHON:-/home/tans5/anaconda3/envs/vllm/bin/python}"
downstream_gpu_memory_utilization="${DOWNSTREAM_GPU_MEMORY_UTILIZATION:-0.8}"
downstream_tensor_parallel_size="${DOWNSTREAM_TENSOR_PARALLEL_SIZE:-1}"
downstream_data_parallel_size="${DOWNSTREAM_DATA_PARALLEL_SIZE:-1}"
downstream_dtype="${DOWNSTREAM_DTYPE:-bfloat16}"
downstream_max_model_len="${DOWNSTREAM_MAX_MODEL_LEN:-16384}"
downstream_max_num_batched_tokens="${DOWNSTREAM_MAX_NUM_BATCHED_TOKENS:-49152}"
downstream_max_num_seqs="${DOWNSTREAM_MAX_NUM_SEQS:-24}"
downstream_save_shard_size="${DOWNSTREAM_SAVE_SHARD_SIZE:-2GB}"
downstream_cache_requests="${DOWNSTREAM_CACHE_REQUESTS:-true}"
downstream_request_cache_path="${DOWNSTREAM_REQUEST_CACHE_PATH:-eval_results/lm_eval_request_cache}"
downstream_include_path="${DOWNSTREAM_INCLUDE_PATH:-lm_eval_tasks}"
downstream_log_samples="${DOWNSTREAM_LOG_SAMPLES:-1}"
downstream_log_samples_limit="${DOWNSTREAM_LOG_SAMPLES_LIMIT:-20}"
downstream_task_data="${DOWNSTREAM_TASK_DATA:-downstream_test/dataset/mathqa500/test.parquet}"
downstream_local_batch_size="${DOWNSTREAM_LOCAL_BATCH_SIZE:-1}"
downstream_generation_max_batch_tokens="${DOWNSTREAM_GENERATION_MAX_BATCH_TOKENS:-65536}"
downstream_max_prompt_length="${DOWNSTREAM_MAX_PROMPT_LENGTH:-2048}"
downstream_max_new_tokens="${DOWNSTREAM_MAX_NEW_TOKENS:-2048}"

curvature_dir="${CURVATURE_DIR:-out/ibm_2b_instruct/unstructured/curvature/gate_up/}"
wanda_dir="${WANDA_DIR:-out/ibm_2b_instruct/unstructured/wanda/all_seq_compare/}"
magnitude_dir="${MAGNITUDE_DIR:-out/ibm_2b_instruct/unstructured/magnitude/all_seq_compare/}"
compare_dir_root="${COMPARE_DIR_ROOT:-out/ibm_2b_instruct/unstructured/all_layer_compare}"
mlp_activation="${MLP_ACTIVATION:-model}"
echo "Eval MLP activation: $mlp_activation"

cuda_device=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
export CUDA_VISIBLE_DEVICES=$cuda_device

skip_prune_layer_flag=""
if [ -n "$skip_prune_layer_ids" ]; then
    skip_prune_layer_flag="--skip_prune_layer_ids $skip_prune_layer_ids"
fi

downstream_log_samples_flag=""
if [ "$downstream_log_samples" = "1" ]; then
    downstream_log_samples_flag="--downstream_log_samples"
fi

run_python_command() {
    prune_method=$1
    save_dir=$2
    prunescore_order=$3
    prune_score_order=$4
    eval_flag=$5

    prune_ops_flag=""
    if [ -n "$prune_ops" ]; then
        prune_ops_flag="--prune_ops $prune_ops"
    fi
    l2_flag=""
    if [ "$use_l2_norm" = "1" ]; then
        l2_flag="--L2-norm --l2_norm_mode $l2_norm_mode"
    fi

    downstream_flag=""
    if [ "$run_downstream_test" = "1" ] && [ "$eval_flag" = "--run_pp_eval" ]; then
        downstream_flag="--run_downstream_eval"
        if [ "$downstream_only" = "1" ]; then
            downstream_flag="$downstream_flag --downstream_only"
        fi
        if [ "$downstream_apply_chat_template" = "1" ]; then
            downstream_flag="$downstream_flag --downstream_apply_chat_template"
        fi
        if [ "$downstream_fewshot_as_multiturn" = "1" ]; then
            downstream_flag="$downstream_flag --downstream_fewshot_as_multiturn"
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
        --per_layer_compare_dir $compare_dir \
        --model_dtype $model_dtype \
        --mlp_activation $mlp_activation \
        $l2_flag \
        --downstream_tasks $downstream_tasks \
        --downstream_batch_size "$downstream_batch_size" \
        --downstream_hf_batch_size "$downstream_hf_batch_size" \
        --downstream_hf_max_batch_size "$downstream_hf_max_batch_size" \
        --downstream_hf_gpu_memory_utilization "$downstream_hf_gpu_memory_utilization" \
        --downstream_output_dir "$downstream_output_dir" \
        --downstream_summary_csv "$downstream_summary_csv" \
        --downstream_suite "$downstream_suite" \
        --downstream_suite_benchmarks "$downstream_suite_benchmarks" \
        --downstream_suite_tasks "$downstream_suite_tasks" \
        --downstream_suite_backends "$downstream_suite_backends" \
        --downstream_suite_fewshots "$downstream_suite_fewshots" \
        --downstream_suite_limits "$downstream_suite_limits" \
        --downstream_num_fewshot "$downstream_num_fewshot" \
        --downstream_chat_template_args "$downstream_chat_template_args" \
        --downstream_gen_kwargs "$downstream_gen_kwargs" \
        --downstream_limit "$downstream_limit" \
        --downstream_lm_eval_backend "$downstream_lm_eval_backend" \
        --downstream_vllm_python "$downstream_vllm_python" \
        --downstream_gpu_memory_utilization "$downstream_gpu_memory_utilization" \
        --downstream_tensor_parallel_size "$downstream_tensor_parallel_size" \
        --downstream_data_parallel_size "$downstream_data_parallel_size" \
        --downstream_dtype "$downstream_dtype" \
        --downstream_max_model_len "$downstream_max_model_len" \
        --downstream_max_num_batched_tokens "$downstream_max_num_batched_tokens" \
        --downstream_max_num_seqs "$downstream_max_num_seqs" \
        --downstream_save_shard_size "$downstream_save_shard_size" \
        --downstream_cache_requests "$downstream_cache_requests" \
        --downstream_request_cache_path "$downstream_request_cache_path" \
        --downstream_include_path "$downstream_include_path" \
        --downstream_log_samples_limit "$downstream_log_samples_limit" \
        --downstream_task_data "$downstream_task_data" \
        --downstream_local_batch_size "$downstream_local_batch_size" \
        --downstream_generation_max_batch_tokens "$downstream_generation_max_batch_tokens" \
        --downstream_max_prompt_length "$downstream_max_prompt_length" \
        --downstream_max_new_tokens "$downstream_max_new_tokens" \
        $prune_ops_flag \
        $skip_prune_layer_flag \
        $downstream_log_samples_flag \
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

run_eval_for_prune_ops() {
    prune_ops=$1
    compare_dir=$2
    echo "Eval prune ops: $prune_ops"
    echo "Eval compare dir: $compare_dir"

    if [ "$run_all_layer_eval" = "1" ]; then
        run_scope_all_methods "per_op" "per-op"
        run_scope_all_methods "locally" "locally"
        # run_scope_all_methods "globally" "globally"
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

run_eval_for_prune_ops "gate_proj" "$compare_dir_root/gate_l2_allexamples/"
