#!/bin/sh
set -e

# PPL and lm-eval downstream eval from saved curvature PKLs for Llama-3-8B.
model="${MODEL:-meta-llama/Meta-Llama-3-8B}"
python_bin="${PYTHON_BIN:-python}"
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
run_downstream_test="${RUN_DOWNSTREAM_TEST:-1}"
downstream_only="${DOWNSTREAM_ONLY:-1}"
downstream_tasks="${DOWNSTREAM_TASKS:-mmlu}"
downstream_batch_size="${DOWNSTREAM_BATCH_SIZE:-auto}"
downstream_output_dir="${DOWNSTREAM_OUTPUT_DIR:-}"
downstream_summary_csv="${DOWNSTREAM_SUMMARY_CSV:-eval_results/summary.csv}"
downstream_suite="${DOWNSTREAM_SUITE:-core}"
downstream_suite_benchmarks="${DOWNSTREAM_SUITE_BENCHMARKS:-mmlu;agieval_en;commonsense_qa;winogrande;triviaqa;boolq;squadv2}"
downstream_suite_tasks="${DOWNSTREAM_SUITE_TASKS:-mmlu_stem,mmlu_social_sciences;agieval_en;commonsense_qa;winogrande;triviaqa;boolq;squadv2}"
downstream_suite_backends="${DOWNSTREAM_SUITE_BACKENDS:-hf;hf;hf;hf;vllm;hf;vllm}"
downstream_suite_fewshots="${DOWNSTREAM_SUITE_FEWSHOTS:-5;0;7;5;5;0;1}"
downstream_suite_limits="${DOWNSTREAM_SUITE_LIMITS:-0.25;0.5;1000;1000;500;1000;500}"
downstream_num_fewshot="${DOWNSTREAM_NUM_FEWSHOT:-5}"
downstream_apply_chat_template="${DOWNSTREAM_APPLY_CHAT_TEMPLATE:-0}"
downstream_fewshot_as_multiturn="${DOWNSTREAM_FEWSHOT_AS_MULTITURN:-0}"
downstream_gen_kwargs="${DOWNSTREAM_GEN_KWARGS:-}"
downstream_limit="${DOWNSTREAM_LIMIT:-}"
downstream_lm_eval_backend="${DOWNSTREAM_LM_EVAL_BACKEND:-hf}"
downstream_vllm_python="${DOWNSTREAM_VLLM_PYTHON:-/home/tans5/anaconda3/envs/vllm/bin/python}"
downstream_gpu_memory_utilization="${DOWNSTREAM_GPU_MEMORY_UTILIZATION:-0.7}"
downstream_tensor_parallel_size="${DOWNSTREAM_TENSOR_PARALLEL_SIZE:-1}"
downstream_data_parallel_size="${DOWNSTREAM_DATA_PARALLEL_SIZE:-1}"
downstream_dtype="${DOWNSTREAM_DTYPE:-bfloat16}"
downstream_max_model_len="${DOWNSTREAM_MAX_MODEL_LEN:-2048}"
downstream_max_num_batched_tokens="${DOWNSTREAM_MAX_NUM_BATCHED_TOKENS:-8192}"
downstream_max_num_seqs="${DOWNSTREAM_MAX_NUM_SEQS:-64}"
downstream_save_shard_size="${DOWNSTREAM_SAVE_SHARD_SIZE:-2GB}"
downstream_cache_requests="${DOWNSTREAM_CACHE_REQUESTS:-true}"
downstream_request_cache_path="${DOWNSTREAM_REQUEST_CACHE_PATH:-eval_results/lm_eval_request_cache}"

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
        --prune_ops $prune_ops \
        --per_layer_compare_dir $compare_dir \
        --model_dtype $model_dtype \
        --mlp_activation $mlp_activation \
        --downstream_tasks $downstream_tasks \
        --downstream_batch_size "$downstream_batch_size" \
        --downstream_output_dir "$downstream_output_dir" \
        --downstream_summary_csv "$downstream_summary_csv" \
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
