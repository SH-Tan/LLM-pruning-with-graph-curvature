#!/bin/sh
set -e

# All-layer WANDA pruning + downstream eval using C4 and generated calibration data.
model="${MODEL:-ibm-granite/granite-3.3-2b-instruct}"
python_bin="${PYTHON_BIN:-python}"
sparsity_ratios="${SPARSITY_RATIOS:-0 0.1 0.3 0.4 0.5 0.6 0.8 1}"
nsamples="${NSAMPLES:-64}"
seed="${SEED:-13}"
alpha="${ALPHA:-0.9}"
model_device="${MODEL_DEVICE:-cuda:0}"
compute_device="${COMPUTE_DEVICE:-cuda:1}"
seq_len="${SEQ_LEN:-650}"
pp_seqlen="${PP_SEQLEN:-$seq_len 1024}"
calib_data_list="${CALIB_DATA_LIST:-${CALIB_DATA:-generated_downstream}}"
generated_calib_data_paths="${GENERATED_CALIB_DATA_PATHS:-${GENERATED_CALIB_DATA_PATH:-downstream_calib_data/generated/mmlu_stem.jsonl downstream_calib_data/generated/truthfulqa_mc1.jsonl downstream_calib_data/generated/gsm8k.jsonl}}"
generated_calib_text_mode="${GENERATED_CALIB_TEXT_MODE:-answer}"
prune_ops="${PRUNE_OPS:-up_proj gate_proj down_proj}"
prune_ops_tag=$(printf "%s" "$prune_ops" | tr ' /' '__')
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
downstream_suite="${DOWNSTREAM_SUITE:-core}"
downstream_suite_benchmarks="${DOWNSTREAM_SUITE_BENCHMARKS:-mmlu;agieval_cn;winogrande;truthfulqa;gsm8k}"
downstream_suite_tasks="${DOWNSTREAM_SUITE_TASKS:-mmlu_stem,mmlu_social_sciences;agieval_gaokao_chinese,agieval_logiqa_zh,agieval_jec_qa_kd;winogrande;truthfulqa_mc1,truthfulqa_mc2;gsm8k}"
downstream_suite_backends="${DOWNSTREAM_SUITE_BACKENDS:-hf;hf;hf;hf;vllm}"
downstream_suite_fewshots="${DOWNSTREAM_SUITE_FEWSHOTS:-5;0;5;0;8}"
downstream_suite_limits="${DOWNSTREAM_SUITE_LIMITS:-0.25;0.25;1000;500;500}"
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
downstream_max_model_len="${DOWNSTREAM_MAX_MODEL_LEN:-4096}"
downstream_max_num_batched_tokens="${DOWNSTREAM_MAX_NUM_BATCHED_TOKENS:-8192}"
downstream_max_num_seqs="${DOWNSTREAM_MAX_NUM_SEQS:-4}"
downstream_save_shard_size="${DOWNSTREAM_SAVE_SHARD_SIZE:-2GB}"
downstream_cache_requests="${DOWNSTREAM_CACHE_REQUESTS:-true}"
downstream_request_cache_path="${DOWNSTREAM_REQUEST_CACHE_PATH:-eval_results/lm_eval_request_cache}"
downstream_include_path="${DOWNSTREAM_INCLUDE_PATH:-lm_eval_tasks}"
downstream_log_samples="${DOWNSTREAM_LOG_SAMPLES:-1}"
downstream_log_samples_limit="${DOWNSTREAM_LOG_SAMPLES_LIMIT:-20}"
downstream_task_data="${DOWNSTREAM_TASK_DATA:-downstream_test/dataset/mathqa500/test.parquet}"
downstream_local_batch_size="${DOWNSTREAM_LOCAL_BATCH_SIZE:-1}"
downstream_generation_max_batch_tokens="${DOWNSTREAM_GENERATION_MAX_BATCH_TOKENS:-2048}"
downstream_max_prompt_length="${DOWNSTREAM_MAX_PROMPT_LENGTH:-2048}"
downstream_max_new_tokens="${DOWNSTREAM_MAX_NEW_TOKENS:-4096}"

prune_scopes="${PRUNE_SCOPES:-per_op}"
wanda_dir_root="${WANDA_DIR_ROOT:-out/ibm_2b_instruct/unstructured/wanda/noncurv_eval}"
compare_dir_base="${COMPARE_DIR_ROOT:-out/ibm_2b_instruct/unstructured/noncurv_all_layer_compare}"
mlp_activation="${MLP_ACTIVATION:-model}"
echo "Eval MLP activation: $mlp_activation"

cuda_device="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES=$cuda_device
export GENERATED_CALIB_TEXT_MODE="$generated_calib_text_mode"

skip_prune_layer_flag=""
if [ -n "$skip_prune_layer_ids" ]; then
    skip_prune_layer_flag="--skip_prune_layer_ids $skip_prune_layer_ids"
fi

downstream_log_samples_flag=""
if [ "$downstream_log_samples" = "1" ]; then
    downstream_log_samples_flag="--downstream_log_samples"
fi

set_calib_run_paths() {
    calib_data=$1
    calib_tag="$calib_data"
    if [ "$calib_data" = "generated_downstream" ]; then
        generated_calib_file="$(basename "$generated_calib_data_path")"
        generated_calib_tag="${generated_calib_file%.*}"
        calib_tag="${generated_calib_tag}_${generated_calib_text_mode}_seq_len_${seq_len}"
        export GENERATED_CALIB_DATA_PATH="$generated_calib_data_path"
    fi

    save_tag="${calib_tag}_${prune_ops_tag}"
    downstream_summary_csv="${DOWNSTREAM_SUMMARY_CSV:-eval_results/summary_${save_tag}.csv}"
    wanda_dir="${WANDA_DIR:-$wanda_dir_root/${save_tag}/}"
    compare_dir_root="$compare_dir_base/${save_tag}"
}

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
        --run_pp_eval
}

run_method_scope() {
    prunescore_order=$1
    prune_method="wanda"
    save_dir=$wanda_dir
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

    for scope in $prune_scopes; do
        run_method_scope "$scope"
    done
}

for calib_data in $calib_data_list; do
    if [ "$calib_data" = "generated_downstream" ]; then
        for generated_calib_data_path in $generated_calib_data_paths; do
            set_calib_run_paths "$calib_data"
            echo "Eval calibration data: $calib_data"
            echo "Generated calibration data path: $generated_calib_data_path"
            run_eval_for_prune_ops "$prune_ops" "$compare_dir_root/mlp_ops/"
        done
    else
        set_calib_run_paths "$calib_data"
        echo "Eval calibration data: $calib_data"
        run_eval_for_prune_ops "$prune_ops" "$compare_dir_root/mlp_ops/"
    fi
done
