#!/bin/sh
set -e

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

model="${MODEL:-ibm-granite/granite-3.3-2b-instruct}"
vllm_python="${VLLM_PYTHON:-/home/tans5/anaconda3/envs/vllm/bin/python}"
python_bin="${PYTHON_BIN:-$vllm_python}"
tasks="${DOWNSTREAM_CALIB_TASKS:-mmlu_stem,mmlu_social_sciences;winogrande;truthfulqa_mc1,truthfulqa_mc2;gsm8k;math500}"
seed="${SEED:-13}"
nsamples_per_task="${NSAMPLES_PER_TASK:-64}"
candidates_per_round="${CANDIDATES_PER_ROUND:-128}"
max_attempts_per_task="${MAX_ATTEMPTS_PER_TASK:-0}"
batch_size="${BATCH_SIZE:-8}"
max_batch_tokens="${MAX_BATCH_TOKENS:-32768}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_new_tokens="${MAX_NEW_TOKENS:-2048}"
temperature="${TEMPERATURE:-0.0}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:-0}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-1}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.6}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
output_dir="${OUTPUT_DIR:-downstream_calib_data/generated}"

cuda_device=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$cuda_device}"

"$python_bin" -m downstream_calib_data.generate_downstream_calib \
    --model_path "$model" \
    --tasks "$tasks" \
    --output_dir "$output_dir" \
    --seed "$seed" \
    --nsamples_per_task "$nsamples_per_task" \
    --candidates_per_round "$candidates_per_round" \
    --max_attempts_per_task "$max_attempts_per_task" \
    --batch_size "$batch_size" \
    --max_batch_tokens "$max_batch_tokens" \
    --max_prompt_length "$max_prompt_length" \
    --max_new_tokens "$max_new_tokens" \
    --temperature "$temperature" \
    --top_p "$top_p" \
    --top_k "$top_k" \
    --tensor_parallel_size "$tensor_parallel_size" \
    --gpu_memory_utilization "$gpu_memory_utilization" \
    --dtype "$model_dtype"

echo "Saved generated downstream calibration data to $output_dir"
# echo "Use with pruning:"
# echo "  CALIB_DATA=generated_downstream GENERATED_CALIB_DATA_PATH=$output_dir GENERATED_CALIB_TEXT_MODE=answer sh scripts/ibm3.3-2b_curv_cal.sh"
# echo "  CALIB_DATA=generated_downstream GENERATED_CALIB_DATA_PATH=$output_dir GENERATED_CALIB_TEXT_MODE=prompt_answer sh scripts/ibm3.3-2b_curv_cal.sh"
