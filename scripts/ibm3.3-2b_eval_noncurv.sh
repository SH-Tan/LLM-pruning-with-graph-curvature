#!/bin/sh
set -e

# All-layer pruning + downstream eval for non-curvature methods.
# Default runs magnitude only. Use PRUNE_METHODS="magnitude wanda" to include WANDA.
model="${MODEL:-ibm-granite/granite-3.3-2b-instruct}"
python_bin="${PYTHON_BIN:-python}"
sparsity_ratios="${SPARSITY_RATIOS:-0 0.3 0.4 0.5 0.6 0.7 0.9 1}"
nsamples="${NSAMPLES:-5}"
seed="${SEED:-13}"
alpha="${ALPHA:-0.9}"
model_device="${MODEL_DEVICE:-cuda:0}"
compute_device="${COMPUTE_DEVICE:-cuda:1}"
seq_len="${SEQ_LEN:-512}"
pp_seqlen="${PP_SEQLEN:-$seq_len 1024}"
calib_data="${CALIB_DATA:-c4_independent}"
prune_ops="${PRUNE_OPS:-}"
skip_prune_layer_ids="${SKIP_PRUNE_LAYER_IDS:-}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
run_downstream_test="${RUN_DOWNSTREAM_TEST:-1}"
downstream_only="${DOWNSTREAM_ONLY:-1}"
downstream_tasks="${DOWNSTREAM_TASKS:-gsm8k}"
downstream_batch_size="${DOWNSTREAM_BATCH_SIZE:-auto}"
downstream_hf_batch_size="${DOWNSTREAM_HF_BATCH_SIZE:-auto}"
downstream_hf_max_batch_size="${DOWNSTREAM_HF_MAX_BATCH_SIZE:-8}"
downstream_hf_gpu_memory_utilization="${DOWNSTREAM_HF_GPU_MEMORY_UTILIZATION:-0.6}"
downstream_output_dir="${DOWNSTREAM_OUTPUT_DIR:-}"
downstream_summary_csv="${DOWNSTREAM_SUMMARY_CSV:-eval_results/summary.csv}"
downstream_suite="${DOWNSTREAM_SUITE:-core}"
downstream_suite_benchmarks="${DOWNSTREAM_SUITE_BENCHMARKS:-mmlu;commonsense_qa;winogrande;boolq;truthfulqa;gsm8k;humaneval;math500}"
downstream_suite_tasks="${DOWNSTREAM_SUITE_TASKS:-mmlu_stem,mmlu_social_sciences;commonsense_qa;winogrande;boolq;truthfulqa_mc1,truthfulqa_mc2;gsm8k;humaneval;minerva_math500}"
downstream_suite_backends="${DOWNSTREAM_SUITE_BACKENDS:-hf;hf;hf;hf;hf;vllm;vllm;vllm}"
downstream_suite_fewshots="${DOWNSTREAM_SUITE_FEWSHOTS:-5;7;5;0;0;5;0;4}"
downstream_suite_limits="${DOWNSTREAM_SUITE_LIMITS:-0.25;1000;1000;1000;500;500;164;500}"
downstream_num_fewshot="${DOWNSTREAM_NUM_FEWSHOT:-5}"
downstream_apply_chat_template="${DOWNSTREAM_APPLY_CHAT_TEMPLATE:-0}"
downstream_fewshot_as_multiturn="${DOWNSTREAM_FEWSHOT_AS_MULTITURN:-0}"
downstream_gen_kwargs="${DOWNSTREAM_GEN_KWARGS:-max_gen_toks=2048}"
downstream_limit="${DOWNSTREAM_LIMIT:-}"
downstream_lm_eval_backend="${DOWNSTREAM_LM_EVAL_BACKEND:-vllm}"
downstream_vllm_python="${DOWNSTREAM_VLLM_PYTHON:-/home/tans5/anaconda3/envs/vllm/bin/python}"
downstream_gpu_memory_utilization="${DOWNSTREAM_GPU_MEMORY_UTILIZATION:-0.6}"
downstream_tensor_parallel_size="${DOWNSTREAM_TENSOR_PARALLEL_SIZE:-1}"
downstream_data_parallel_size="${DOWNSTREAM_DATA_PARALLEL_SIZE:-1}"
downstream_dtype="${DOWNSTREAM_DTYPE:-bfloat16}"
downstream_max_model_len="${DOWNSTREAM_MAX_MODEL_LEN:-2048}"
downstream_max_num_batched_tokens="${DOWNSTREAM_MAX_NUM_BATCHED_TOKENS:-8192}"
downstream_max_num_seqs="${DOWNSTREAM_MAX_NUM_SEQS:-64}"
downstream_save_shard_size="${DOWNSTREAM_SAVE_SHARD_SIZE:-2GB}"
downstream_cache_requests="${DOWNSTREAM_CACHE_REQUESTS:-true}"
downstream_request_cache_path="${DOWNSTREAM_REQUEST_CACHE_PATH:-eval_results/lm_eval_request_cache}"
downstream_include_path="${DOWNSTREAM_INCLUDE_PATH:-lm_eval_tasks}"
downstream_log_samples="${DOWNSTREAM_LOG_SAMPLES:-0}"

prune_methods="${PRUNE_METHODS:-magnitude}"
prune_scopes="${PRUNE_SCOPES:-globally locally per_op}"
wanda_dir="${WANDA_DIR:-out/ibm_2b_instruct/unstructured/wanda/noncurv_eval/}"
magnitude_dir="${MAGNITUDE_DIR:-out/ibm_2b_instruct/unstructured/magnitude/noncurv_eval/}"
compare_dir_root="${COMPARE_DIR_ROOT:-out/ibm_2b_instruct/unstructured/noncurv_all_layer_compare}"
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
    method_summary_csv=$5

    prune_ops_flag=""
    if [ -n "$prune_ops" ]; then
        prune_ops_flag="--prune_ops $prune_ops"
    fi

    downstream_flag=""
    if [ "$run_downstream_test" = "1" ]; then
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
        --seqlen $seq_len \
        --pp_seqlen $pp_seqlen \
        --prune_score_order $prune_score_order \
        --sparsity_schedule input \
        --prunescore_order $prunescore_order \
        --per_layer_compare_dir $compare_dir \
        --model_dtype $model_dtype \
        --mlp_activation $mlp_activation \
        --downstream_tasks $downstream_tasks \
        --downstream_batch_size "$downstream_batch_size" \
        --downstream_hf_batch_size "$downstream_hf_batch_size" \
        --downstream_hf_max_batch_size "$downstream_hf_max_batch_size" \
        --downstream_hf_gpu_memory_utilization "$downstream_hf_gpu_memory_utilization" \
        --downstream_output_dir "$downstream_output_dir" \
        --downstream_summary_csv "$method_summary_csv" \
        --downstream_suite "$downstream_suite" \
        --downstream_suite_benchmarks "$downstream_suite_benchmarks" \
        --downstream_suite_tasks "$downstream_suite_tasks" \
        --downstream_suite_backends "$downstream_suite_backends" \
        --downstream_suite_fewshots "$downstream_suite_fewshots" \
        --downstream_suite_limits "$downstream_suite_limits" \
        --downstream_num_fewshot "$downstream_num_fewshot" \
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
        $prune_ops_flag \
        $skip_prune_layer_flag \
        $downstream_log_samples_flag \
        $downstream_flag \
        --run_pp_eval
}

run_method_scope() {
    prune_method=$1
    prunescore_order=$2

    if [ "$prune_method" = "wanda" ]; then
        save_dir=$wanda_dir
    else
        save_dir=$magnitude_dir
    fi
    summary_base=${downstream_summary_csv%.csv}
    method_summary_csv="${summary_base}_${prune_method}.csv"

    echo "Running all-layer $prune_method $prunescore_order"
    echo "Downstream summary CSV: $method_summary_csv"
    run_python_command "$prune_method" "$save_dir" "$prunescore_order" "low_to_high" "$method_summary_csv"
    echo "Finished all-layer $prune_method $prunescore_order"
}

run_eval_for_prune_ops() {
    prune_ops=$1
    compare_dir=$2
    echo "Eval prune ops: $prune_ops"
    echo "Eval compare dir: $compare_dir"

    for prune_method in $prune_methods; do
        case "$prune_method" in
            magnitude|wanda)
                ;;
            *)
                echo "Unsupported non-curvature method: $prune_method" >&2
                exit 1
                ;;
        esac

        for scope in $prune_scopes; do
            run_method_scope "$prune_method" "$scope"
        done
    done
}

run_eval_for_prune_ops "$prune_ops" "$compare_dir_root/gate_up_resid/"
