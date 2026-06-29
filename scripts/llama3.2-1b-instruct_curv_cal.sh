#!/bin/sh
set -e

# Curvature calculation for Llama-3.2-1B-Instruct, with optional eval after it finishes.
model="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
python_bin="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
sparsity_ratios="${SPARSITY_RATIOS:-0 0.3 0.4 0.5 0.6 0.7 0.9 1}"
curv_sparsity_ratios="${CURV_SPARSITY_RATIOS:-0}"
nsamples="${NSAMPLES:-5}"
seed="${SEED:-13}"
alpha="${ALPHA:-0.9}"
model_device="${MODEL_DEVICE:-cuda:0}"
compute_device="${COMPUTE_DEVICE:-cuda:1}"
seq_len="${SEQ_LEN:-512}"
sample_edge_ratio="${SAMPLE_EDGE_RATIO:-0.05}"
sample_edge_num="${SAMPLE_EDGE_NUM:--1}"
calib_data="${CALIB_DATA:-c4_independent}"
curvature_dir="${CURVATURE_DIR:-out/llama3.2_1b_instruct/unstructured/curvature/gate_wores_relu/}"
curvature_dtype="${CURVATURE_DTYPE:-float32}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
save_parameter_metric_logs="${SAVE_PARAMETER_METRIC_LOGS:-0}"
run_eval_after_curv="${RUN_EVAL_AFTER_CURV:-1}"
run_all_layer_eval="${RUN_ALL_LAYER_EVAL:-${RUN_PP_EVAL:-1}}"
run_per_layer_eval="${RUN_PER_LAYER_EVAL:-0}"
run_downstream_test="${RUN_DOWNSTREAM_TEST:-0}"

cuda_device=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
export CUDA_VISIBLE_DEVICES=$cuda_device
export CURV_GATE_PLOT_ENABLED=0
export CURV_COLLECT_LAYER_DATA_ONLY=0
export CURV_GATE_PLOT_WANDA_L2=0
export CURV_GATE_PLOT_PER_SAMPLE=0
export CURV_GATE_PLOT_MAX_POINTS="${CURV_GATE_PLOT_MAX_POINTS:-200000}"

run_curvature_calculation() {
    use_l2_norm=$1
    l2_mode=$2
    l2_flag=""
    if [ "$use_l2_norm" = "1" ]; then
        l2_flag="--L2-norm --l2_norm_mode $l2_mode"
    fi
    parameter_log_flag=""
    if [ "$save_parameter_metric_logs" = "1" ]; then
        parameter_log_flag="--save_parameter_metric_logs"
    fi

    echo "Running curvature calculation: use_l2_norm=$use_l2_norm, l2_norm_mode=$l2_mode, top_k_seq=$top_k_seq, seq_select=$seq_select, lpf_window=$curvature_lpf_window"
    "$python_bin" src/llm_main.py \
        --model $model \
        --prune_method curvature \
        --sparsity_ratio $curv_sparsity_ratios \
        --sparsity_type unstructured \
        --save $curvature_dir \
        --nsamples $nsamples \
        --seed $seed \
        --model_device $model_device \
        --compute_device $compute_device \
        --alpha $alpha \
        --calib_data $calib_data \
        --sample_edge_ratio $sample_edge_ratio \
        --sample_edge_num $sample_edge_num \
        --seqlen $seq_len \
        --save_curvature_dir $curvature_dir \
        --shared_top_k $top_k_seq \
        --shared_seq_select $seq_select \
        --curvature_lpf_window $curvature_lpf_window \
        --curvature_dtype $curvature_dtype \
        --model_dtype $model_dtype \
        $parameter_log_flag \
        $l2_flag
}

# Run list. Keep each setting explicit so it is easy to comment out or add variants.

# 1. L2 norm type 1: current behavior, L2 per example over all sequence positions.
# top_k_seq=-1
# seq_select="top"
# curvature_lpf_window=0
# run_curvature_calculation 1 "per_example"

# 2. No L2: evaluate seq positions 0, 10, 20, ... and save LPF curvature.
#    Set curvature_lpf_window=0 to disable LPF, or change seq_select back to "top".
# top_k_seq=-1
# seq_select="stride10"
# curvature_lpf_window="${CURVATURE_LPF_WINDOW:-5}"
# run_curvature_calculation 0 "per_example"

# 3. L2 norm type 2: Wanda-style, L2 over all examples and all sequence positions.
# top_k_seq=-1
# seq_select="top"
# curvature_lpf_window=0
# run_curvature_calculation 1 "all_examples"

top_k_seq="${TOP_K_SEQ:-10}"
seq_select="${SEQ_SELECT:-top}"
curvature_lpf_window="${CURVATURE_LPF_WINDOW:-0}"
run_curvature_calculation 0 "per_example"

if [ "$run_eval_after_curv" = "1" ]; then
    echo "Curvature calculation finished; running eval script."
    MODEL="$model" \
    PYTHON_BIN="$python_bin" \
    SPARSITY_RATIOS="$sparsity_ratios" \
    NSAMPLES="$nsamples" \
    SEED="$seed" \
    ALPHA="$alpha" \
    MODEL_DEVICE="$model_device" \
    COMPUTE_DEVICE="$compute_device" \
    SEQ_LEN="$seq_len" \
    SAMPLE_EDGE_RATIO="$sample_edge_ratio" \
    SAMPLE_EDGE_NUM="$sample_edge_num" \
    CALIB_DATA="$calib_data" \
    CURVATURE_DIR="$curvature_dir" \
    MODEL_DTYPE="$model_dtype" \
    TOP_K_SEQ="$top_k_seq" \
    SEQ_SELECT="$seq_select" \
    CURVATURE_LPF_WINDOW="$curvature_lpf_window" \
    RUN_ALL_LAYER_EVAL="$run_all_layer_eval" \
    RUN_PER_LAYER_EVAL="$run_per_layer_eval" \
    RUN_DOWNSTREAM_TEST="$run_downstream_test" \
    sh scripts/llama3.2-1b-instruct_eval.sh
fi
