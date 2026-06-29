#!/bin/sh
set -e

# Dense downstream eval only for DeepSeek-R1-Distill-Qwen-1.5B.
# This runs the base model directly in vLLM and writes results as sparsity 0.
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

model="${model:-${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}}"
dataset_path="${downstream_task_data:-${DOWNSTREAM_TASK_DATA:-downstream_test/dataset/mathqa500/test.parquet}}"
prompt_key="${downstream_prompt_key:-${DOWNSTREAM_PROMPT_KEY:-prompt}}"
response_key="${downstream_response_key:-${DOWNSTREAM_RESPONSE_KEY:-}}"
reward_score_dir="${downstream_reward_score_dir:-${DOWNSTREAM_REWARD_SCORE_DIR:-}}"
output_dir="${output_dir:-${DOWNSTREAM_OUTPUT_DIR:-out/deepseek_r1_distill_qwen_1.5b/downstream_results}}"
score_order="${score_order:-dense}"
target_sparsity="0.000000"

vllm_python="${vllm_python:-${DOWNSTREAM_VLLM_PYTHON:-${VLLM_PYTHON:-/home/tans5/anaconda3/envs/vllm/bin/python}}}"
if [ ! -x "$vllm_python" ]; then
    vllm_python="python"
fi

seed="${seed:-${SEED:-13}}"
max_examples="${downstream_max_examples:-${DOWNSTREAM_MAX_EXAMPLES:-500}}"
start_index="${downstream_start_index:-${DOWNSTREAM_START_INDEX:-0}}"
shuffle="${downstream_shuffle:-${DOWNSTREAM_SHUFFLE:-0}}"
batch_size="${downstream_batch_size:-${DOWNSTREAM_BATCH_SIZE:-64}}"
generation_max_batch_tokens="${downstream_generation_max_batch_tokens:-${DOWNSTREAM_GENERATION_MAX_BATCH_TOKENS:-491520}}"
max_prompt_length="${downstream_max_prompt_length:-${DOWNSTREAM_MAX_PROMPT_LENGTH:-2048}}"
max_new_tokens="${downstream_max_new_tokens:-${DOWNSTREAM_MAX_NEW_TOKENS:-8192}}"
min_tokens="${downstream_min_tokens:-${DOWNSTREAM_MIN_TOKENS:-0}}"
temperature="${downstream_temperature:-${DOWNSTREAM_TEMPERATURE:-0.0}}"
top_p="${downstream_top_p:-${DOWNSTREAM_TOP_P:-1.0}}"
top_k="${downstream_top_k:-${DOWNSTREAM_TOP_K:-0}}"
response_log_max="${downstream_response_log_max:-${DOWNSTREAM_RESPONSE_LOG_MAX:-50}}"
tensor_parallel_size="${vllm_tensor_parallel_size:-${VLLM_TENSOR_PARALLEL_SIZE:-${DOWNSTREAM_TENSOR_PARALLEL_SIZE:-1}}}"
gpu_memory_utilization="${vllm_gpu_memory_utilization:-${VLLM_GPU_MEMORY_UTILIZATION:-${DOWNSTREAM_GPU_MEMORY_UTILIZATION:-0.7}}}"
vllm_dtype="${vllm_dtype:-${VLLM_DTYPE:-${DOWNSTREAM_DTYPE:-auto}}}"

best_free_gpu_ids() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2,2nr \
        | awk -F, -v count="$1" 'NR <= count {gsub(/ /, "", $1); ids = ids sep $1; sep = ","} END {print ids}'
}

if [ -n "${DOWNSTREAM_CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$DOWNSTREAM_CUDA_VISIBLE_DEVICES"
else
    cuda_device="$(best_free_gpu_ids "$tensor_parallel_size")"
    if [ -n "$cuda_device" ]; then
        export CUDA_VISIBLE_DEVICES="$cuda_device"
    fi
fi

mkdir -p "$output_dir"
output_path="$output_dir/downstream_task_responses_${score_order}_sparsity_${target_sparsity}.jsonl"
metrics_path="$output_dir/downstream_task_metrics_${score_order}_sparsity_${target_sparsity}.json"

set -- "$vllm_python" -m downstream_test.vllm_accuracy_runner \
    --model_path "$model" \
    --dataset_path "$dataset_path" \
    --output_path "$output_path" \
    --metrics_path "$metrics_path" \
    --prompt_key "$prompt_key" \
    --start_index "$start_index" \
    --seed "$seed" \
    --max_examples "$max_examples" \
    --batch_size "$batch_size" \
    --generation_max_batch_tokens "$generation_max_batch_tokens" \
    --max_prompt_length "$max_prompt_length" \
    --max_new_tokens "$max_new_tokens" \
    --min_tokens "$min_tokens" \
    --temperature "$temperature" \
    --top_p "$top_p" \
    --top_k "$top_k" \
    --response_log_max "$response_log_max" \
    --tensor_parallel_size "$tensor_parallel_size" \
    --gpu_memory_utilization "$gpu_memory_utilization" \
    --dtype "$vllm_dtype"

if [ -n "$response_key" ]; then
    set -- "$@" --response_key "$response_key"
fi
if [ -n "$reward_score_dir" ]; then
    set -- "$@" --reward_score_dir "$reward_score_dir"
fi
if [ "$shuffle" = "1" ]; then
    set -- "$@" --shuffle
fi

echo "Running dense sparsity 0 downstream vLLM eval"
echo "score_order=$score_order"
echo "model=$model"
echo "dataset_path=$dataset_path"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "batch_size=$batch_size"
echo "generation_max_batch_tokens=$generation_max_batch_tokens"
echo "max_new_tokens=$max_new_tokens"
echo "output_path=$output_path"
echo "metrics_path=$metrics_path"

"$@"
