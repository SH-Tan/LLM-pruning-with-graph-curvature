#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/tans5/anaconda3/envs/vllm/bin/python}"

IBM_MODEL_DIR="${IBM_MODEL_DIR:-$REPO_DIR/llm_weights/models--ibm-granite--granite-3.3-2b-instruct/snapshots/707f574c62054322f6b5b04b6d075f0a8f05e0f0}"
MODEL="${MODEL:-vllm/ibm-granite-local}"
GPU="${GPU:-0}"
DTYPE="${DTYPE:-bfloat16}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/inspect_evals/logs/downstream}"
LIMIT="${LIMIT:-}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_CONNECTIONS="${MAX_CONNECTIONS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
INSPECT_DISPLAY="${INSPECT_DISPLAY:-plain}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

RUN_PRUNED_SWEEP="${RUN_PRUNED_SWEEP:-1}"
SPARSITY_RATIOS="${SPARSITY_RATIOS:-0 0.3 0.4 0.5 0.6 0.7}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUNTIME_CHECKPOINT_ROOT="${RUNTIME_CHECKPOINT_ROOT:-$REPO_DIR/inspect_evals/runtime_checkpoints/$RUN_ID}"

PRUNE_PYTHON_BIN="${PRUNE_PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
PRUNE_CUDA_VISIBLE_DEVICES="${PRUNE_CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)}"
PRUNE_MODEL="${PRUNE_MODEL:-$IBM_MODEL_DIR}"
PRUNE_METHODS=(
    # curvature
    wanda
    magnitude
)
PRUNE_OP_GROUPS=(
    "gate_proj"
    "gate_proj up_proj"
    "up_proj"
)
PRUNESCORE_ORDER="${PRUNESCORE_ORDER:-per_op}"
PRUNE_SCORE_ORDER="${PRUNE_SCORE_ORDER:-high_to_low}"
PRUNE_CALIB_DATA="${PRUNE_CALIB_DATA:-generated_downstream}"
PRUNE_GENERATED_CALIB_DATA_PATH="${PRUNE_GENERATED_CALIB_DATA_PATH:-$REPO_DIR/downstream_calib_data/generated/mmlu_stem.jsonl}"
PRUNE_GENERATED_CALIB_TEXT_MODE="${PRUNE_GENERATED_CALIB_TEXT_MODE:-prompt_answer}"
PRUNE_NSAMPLES="${PRUNE_NSAMPLES:-5}"
PRUNE_SEED="${PRUNE_SEED:-13}"
PRUNE_ALPHA="${PRUNE_ALPHA:-0.9}"
PRUNE_SEQ_LEN="${PRUNE_SEQ_LEN:-512}"
PRUNE_PP_SEQLEN="${PRUNE_PP_SEQLEN:-$PRUNE_SEQ_LEN 1024}"
PRUNE_SAMPLE_EDGE_RATIO="${PRUNE_SAMPLE_EDGE_RATIO:-1}"
PRUNE_SAMPLE_EDGE_NUM="${PRUNE_SAMPLE_EDGE_NUM:--1}"
PRUNE_MODEL_DEVICE="${PRUNE_MODEL_DEVICE:-cuda:0}"
PRUNE_COMPUTE_DEVICE="${PRUNE_COMPUTE_DEVICE:-cuda:1}"
PRUNE_MODEL_DTYPE="${PRUNE_MODEL_DTYPE:-bfloat16}"
PRUNE_TOP_K_SEQ="${PRUNE_TOP_K_SEQ:-1}"
PRUNE_SEQ_SELECT="${PRUNE_SEQ_SELECT:-top}"
PRUNE_CURVATURE_LPF_WINDOW="${PRUNE_CURVATURE_LPF_WINDOW:-0}"
PRUNE_CURVATURE_DIR="${PRUNE_CURVATURE_DIR:-$REPO_DIR/out/ibm_2b_instruct/unstructured/curvature/gate_mmlu}"
RUNTIME_CHECKPOINT_PREFIX_BASE="${RUNTIME_CHECKPOINT_PREFIX_BASE:-ibm_granite_3p3_2b_inspect}"
PRUNE_COMPARE_ROOT="${PRUNE_COMPARE_ROOT:-$RUNTIME_CHECKPOINT_ROOT/compare}"
PRUNE_METHOD=""
PRUNE_OPS=""
PRUNE_OPS_TAG=""
PRUNE_RUN_TAG=""
RUNTIME_CHECKPOINT_PREFIX=""
PRUNE_COMPARE_DIR=""

RUN_CURRENT_DOWNSTREAM="${RUN_CURRENT_DOWNSTREAM:-1}"
CURRENT_DOWNSTREAM_TASKS="${CURRENT_DOWNSTREAM_TASKS:-mmlu_0_shot winogrande truthfulqa gsm8k}"
CURRENT_MMLU_LIMIT="${CURRENT_MMLU_LIMIT:-1000}"  # full: 14042
WINOGRANDE_LIMIT="${WINOGRANDE_LIMIT:-250}"  # full: 1267
TRUTHFULQA_LIMIT="${TRUTHFULQA_LIMIT:-500}"  # full: 817
GSM8K_LIMIT="${GSM8K_LIMIT:-500}"  # full: 1319
MATH500_LIMIT="${MATH500_LIMIT:-500}"  # full: 500
IFEVAL_LIMIT="${IFEVAL_LIMIT:-500}"  # full: 541

RUN_MMMLU="${RUN_MMMLU:-1}"
MMMLU_TASK="${MMMLU_TASK:-mmlu_0_shot}"
MMMLU_LANGUAGES="${MMMLU_LANGUAGES:-ZH_CN DE_DE}"
MMMLU_LIMIT="${MMMLU_LIMIT:-500}"  # full: 14042 per language

RUN_MGSM="${RUN_MGSM:-1}"
MGSM_LANGUAGES="${MGSM_LANGUAGES:-zh}"
MGSM_LIMIT_SAMPLES_PER_LANG="${MGSM_LIMIT_SAMPLES_PER_LANG:-100}"  # full: 250 per language
MGSM_USE_COT="${MGSM_USE_COT:-true}"

RUN_SEVENLLM="${RUN_SEVENLLM:-1}"
SEVENLLM_TASKS="${SEVENLLM_TASKS:-sevenllm_mcq_zh}"
SEVENLLM_MCQ_LIMIT="${SEVENLLM_MCQ_LIMIT:-50}"  # full: 50
SEVENLLM_QA_LIMIT="${SEVENLLM_QA_LIMIT:-100}"  # full: 600

RUN_ADDITIONAL_NON_ENGLISH="${RUN_ADDITIONAL_NON_ENGLISH:-0}"
ADDITIONAL_NON_ENGLISH_TASKS="${ADDITIONAL_NON_ENGLISH_TASKS:-onet_m6 race_h}"
ONET_LIMIT="${ONET_LIMIT:-100}"  # full: 397
RACE_H_LIMIT="${RACE_H_LIMIT:-500}"  # full: 3498

EXTRA_TASKS="${EXTRA_TASKS:-}"

export INSPECT_NO_SPAN=1
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO_DIR/inspect_evals/src${PYTHONPATH:+:$PYTHONPATH}"
export GENERATED_CALIB_DATA_PATH="$PRUNE_GENERATED_CALIB_DATA_PATH"
export GENERATED_CALIB_TEXT_MODE="$PRUNE_GENERATED_CALIB_TEXT_MODE"

read -r -a PRUNE_PP_SEQLEN_ARGS <<< "$PRUNE_PP_SEQLEN"

ACTIVE_MODEL_DIR=""
ACTIVE_LOG_DIR=""
ACTIVE_TAGS=""

configure_prune_run() {
    PRUNE_METHOD=$1
    PRUNE_OPS=$2
    PRUNE_OPS_TAG="${PRUNE_OPS// /-}"
    PRUNE_RUN_TAG="${PRUNE_METHOD}_${PRUNESCORE_ORDER}_${PRUNE_SCORE_ORDER}_${PRUNE_OPS_TAG}"
    RUNTIME_CHECKPOINT_PREFIX="${RUNTIME_CHECKPOINT_PREFIX_BASE}_${PRUNE_RUN_TAG}"
    PRUNE_COMPARE_DIR="$PRUNE_COMPARE_ROOT/$PRUNE_RUN_TAG"
}

sparsity_tag() {
    "$PYTHON_BIN" -c 'import sys; r=float(sys.argv[1]); print((f"{r:.4f}".rstrip("0").rstrip(".") or "0").replace(".", "p"))' "$1"
}

checkpoint_path_for_sparsity() {
    sparsity=$1
    tag=$(sparsity_tag "$sparsity")

    printf "%s/%s_sparsity_%s\n" "$RUNTIME_CHECKPOINT_ROOT" "$RUNTIME_CHECKPOINT_PREFIX" "$tag"
}

checkpoint_complete() {
    path=$1
    [ -d "$path" ] || return 1
    [ -f "$path/config.json" ] || return 1
    [ -f "$path/tokenizer_config.json" ] || return 1
    [ -f "$path/tokenizer.json" ] || return 1
    [ -f "$path/model.safetensors" ] && return 0
    [ -f "$path/model.safetensors.index.json" ] && return 0
    [ -f "$path/pytorch_model.bin" ] && return 0
    [ -f "$path/pytorch_model.bin.index.json" ] && return 0
    find "$path" -maxdepth 1 -name '*.safetensors' -print -quit | grep -q .
}

save_runtime_checkpoint() {
    sparsity=$1
    checkpoint_dir=$2

    prune_args=(
        "$REPO_DIR/src/llm_main.py"
        --model "$PRUNE_MODEL"
        --prune_method "$PRUNE_METHOD"
        --sparsity_ratio "$sparsity"
        --sparsity_type unstructured
        --save "$PRUNE_CURVATURE_DIR"
        --save_model "$checkpoint_dir"
        --nsamples "$PRUNE_NSAMPLES"
        --seed "$PRUNE_SEED"
        --model_device "$PRUNE_MODEL_DEVICE"
        --compute_device "$PRUNE_COMPUTE_DEVICE"
        --alpha "$PRUNE_ALPHA"
        --calib_data "$PRUNE_CALIB_DATA"
        --sample_edge_ratio "$PRUNE_SAMPLE_EDGE_RATIO"
        --sample_edge_num "$PRUNE_SAMPLE_EDGE_NUM"
        --seqlen "$PRUNE_SEQ_LEN"
        --pp_seqlen "${PRUNE_PP_SEQLEN_ARGS[@]}"
        --prune_score_order "$PRUNE_SCORE_ORDER"
        --sparsity_schedule input
        --save_curvature_dir "$PRUNE_CURVATURE_DIR"
        --load_curvature_dir "$PRUNE_CURVATURE_DIR"
        --prunescore_order "$PRUNESCORE_ORDER"
        --shared_top_k "$PRUNE_TOP_K_SEQ"
        --shared_seq_select "$PRUNE_SEQ_SELECT"
        --curvature_lpf_window "$PRUNE_CURVATURE_LPF_WINDOW"
        --per_layer_compare_dir "$PRUNE_COMPARE_DIR"
        --model_dtype "$PRUNE_MODEL_DTYPE"
        --mlp_activation model
        --downstream_only
        --run_pp_eval
    )
    if [ -n "$PRUNE_OPS" ]; then
        prune_args+=(--prune_ops $PRUNE_OPS)
    fi

    echo "Saving fresh runtime checkpoint for sparsity=$sparsity"
    echo "  checkpoint dir: $checkpoint_dir"
    CUDA_VISIBLE_DEVICES="$PRUNE_CUDA_VISIBLE_DEVICES" "$PRUNE_PYTHON_BIN" "${prune_args[@]}"

    if ! checkpoint_complete "$checkpoint_dir"; then
        echo "Runtime checkpoint is incomplete after pruning: $checkpoint_dir" >&2
        exit 1
    fi
}

run_inspect_eval() {
    task=$1
    shift
    task_path=$(inspect_task_path "$task")

    base_args=(
        --model "$MODEL"
        -M "model_path=$ACTIVE_MODEL_DIR"
        -M "tokenizer_path=$ACTIVE_MODEL_DIR"
        -M "dtype=$DTYPE"
        -M "trust_remote_code=true"
        -M "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
        -M "max_model_len=$MAX_MODEL_LEN"
        -M "max_num_seqs=$MAX_NUM_SEQS"
        -M "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
        --log-dir "$ACTIVE_LOG_DIR"
        --display "$INSPECT_DISPLAY"
        --max-connections "$MAX_CONNECTIONS"
        --max-samples "$MAX_SAMPLES"
        --max-tokens "$MAX_TOKENS"
    )
    if [ -n "$ACTIVE_TAGS" ]; then
        base_args+=(--tags "$ACTIVE_TAGS")
    fi

    echo "Running inspect eval: $task $*"
    "$PYTHON_BIN" -m inspect_ai eval "$task_path" "${base_args[@]}" "$@"
}

inspect_task_path() {
    task=$1
    case "$task" in
        mmlu_0_shot|mmlu_5_shot) printf "%s\n" "$REPO_DIR/inspect_evals/src/inspect_evals/mmlu/mmlu.py@$task" ;;
        gsm8k) printf "%s\n" "$REPO_DIR/inspect_evals/src/inspect_evals/gsm8k/gsm8k.py@gsm8k" ;;
        winogrande) printf "%s\n" "$REPO_DIR/inspect_evals/src/inspect_evals/winogrande/winogrande.py@winogrande" ;;
        truthfulqa) printf "%s\n" "$REPO_DIR/inspect_evals/src/inspect_evals/truthfulqa/truthfulqa.py@truthfulqa" ;;
        math500) printf "%s\n" "$REPO_DIR/inspect_evals/src/inspect_evals/math500/math500.py@math500" ;;
        ifeval) printf "%s\n" "$REPO_DIR/inspect_evals/src/inspect_evals/ifeval/ifeval.py@ifeval" ;;
        mgsm) printf "%s\n" "$REPO_DIR/inspect_evals/src/inspect_evals/mgsm/mgsm.py@mgsm" ;;
        sevenllm_*) printf "%s\n" "$REPO_DIR/inspect_evals/src/inspect_evals/sevenllm/sevenllm.py@$task" ;;
        onet_*) printf "%s\n" "$REPO_DIR/inspect_evals/src/inspect_evals/onet/onet.py@$task" ;;
        race_h) printf "%s\n" "$REPO_DIR/inspect_evals/src/inspect_evals/race_h/race_h.py@race_h" ;;
        *) printf "%s\n" "$task" ;;
    esac
}

run_task_suite() {
    if [ "$RUN_CURRENT_DOWNSTREAM" = "1" ]; then
        for task in $CURRENT_DOWNSTREAM_TASKS; do
            current_args=()
            current_limit="$LIMIT"
            if [ -z "$current_limit" ]; then
                case "$task" in
                    mmlu_0_shot|mmlu_5_shot) current_limit="$CURRENT_MMLU_LIMIT" ;;
                    winogrande) current_limit="$WINOGRANDE_LIMIT" ;;
                    truthfulqa) current_limit="$TRUTHFULQA_LIMIT" ;;
                    gsm8k) current_limit="$GSM8K_LIMIT" ;;
                    math500) current_limit="$MATH500_LIMIT" ;;
                    ifeval) current_limit="$IFEVAL_LIMIT" ;;
                esac
            fi
            if [ -n "$current_limit" ]; then
                current_args+=(--limit "$current_limit")
            fi
            run_inspect_eval "$task" "${current_args[@]}"
        done
    fi

    if [ "$RUN_MMMLU" = "1" ]; then
        for lang in $MMMLU_LANGUAGES; do
            mmmlu_args=(-T "language=$lang")
            mmmlu_limit="${LIMIT:-$MMMLU_LIMIT}"
            if [ -n "$mmmlu_limit" ]; then
                mmmlu_args+=(--limit "$mmmlu_limit")
            fi
            run_inspect_eval "$MMMLU_TASK" "${mmmlu_args[@]}"
        done
    fi

    if [ "$RUN_MGSM" = "1" ]; then
        for lang in $MGSM_LANGUAGES; do
            mgsm_args=(-T "languages=$lang" -T "use_cot=$MGSM_USE_COT")
            if [ -n "$MGSM_LIMIT_SAMPLES_PER_LANG" ]; then
                mgsm_args+=(-T "limit_samples_per_lang=$MGSM_LIMIT_SAMPLES_PER_LANG")
            fi
            if [ -n "$LIMIT" ]; then
                mgsm_args+=(--limit "$LIMIT")
            fi
            run_inspect_eval mgsm "${mgsm_args[@]}"
        done
    fi

    if [ "$RUN_SEVENLLM" = "1" ]; then
        for task in $SEVENLLM_TASKS; do
            sevenllm_args=()
            sevenllm_limit="$LIMIT"
            if [ -z "$sevenllm_limit" ]; then
                case "$task" in
                    *mcq*) sevenllm_limit="$SEVENLLM_MCQ_LIMIT" ;;
                    *qa*) sevenllm_limit="$SEVENLLM_QA_LIMIT" ;;
                esac
            fi
            if [ -n "$sevenllm_limit" ]; then
                sevenllm_args+=(--limit "$sevenllm_limit")
            fi
            run_inspect_eval "$task" "${sevenllm_args[@]}"
        done
    fi

    if [ "$RUN_ADDITIONAL_NON_ENGLISH" = "1" ]; then
        for task in $ADDITIONAL_NON_ENGLISH_TASKS; do
            additional_args=()
            additional_limit="$LIMIT"
            if [ -z "$additional_limit" ]; then
                case "$task" in
                    onet_m6) additional_limit="$ONET_LIMIT" ;;
                    race_h) additional_limit="$RACE_H_LIMIT" ;;
                esac
            fi
            if [ -n "$additional_limit" ]; then
                additional_args+=(--limit "$additional_limit")
            fi
            run_inspect_eval "$task" "${additional_args[@]}"
        done
    fi

    for task in $EXTRA_TASKS; do
        extra_args=()
        if [ -n "$LIMIT" ]; then
            extra_args+=(--limit "$LIMIT")
        fi
        run_inspect_eval "$task" "${extra_args[@]}"
    done
}

run_for_model_dir() {
    model_dir=$1
    run_name=$2
    sparsity=$3

    ACTIVE_MODEL_DIR="$model_dir"
    ACTIVE_LOG_DIR="$LOG_DIR/$run_name"
    ACTIVE_TAGS="method=$PRUNE_METHOD,prunescore_order=$PRUNESCORE_ORDER,prune_score_order=$PRUNE_SCORE_ORDER,prune_ops=$PRUNE_OPS_TAG,sparsity=$sparsity,model_dir=$model_dir"

    echo "Running Inspect downstream eval"
    echo "  model: $MODEL"
    echo "  model dir: $ACTIVE_MODEL_DIR"
    echo "  sparsity: $sparsity"
    echo "  gpu: $GPU"
    echo "  log dir: $ACTIVE_LOG_DIR"

    run_task_suite
}

for method in "${PRUNE_METHODS[@]}"; do
    for ops in "${PRUNE_OP_GROUPS[@]}"; do
        configure_prune_run "$method" "$ops"

        if [ "$RUN_PRUNED_SWEEP" = "1" ]; then
            for sparsity in $SPARSITY_RATIOS; do
                tag=$(sparsity_tag "$sparsity")
                checkpoint_dir=$(checkpoint_path_for_sparsity "$sparsity")

                save_runtime_checkpoint "$sparsity" "$checkpoint_dir"
                run_for_model_dir "$checkpoint_dir" "pruned_${PRUNE_RUN_TAG}_sparsity_$tag" "$sparsity"
            done
        else
            run_for_model_dir "$IBM_MODEL_DIR" "base_${PRUNE_RUN_TAG}" "base"
        fi
    done
done
