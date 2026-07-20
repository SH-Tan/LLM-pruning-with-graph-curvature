#!/bin/sh
set -e

# Curvature/WANDA and curvature/magnitude pruning-overlap logs only.
model="${MODEL:-ibm-granite/granite-3.3-2b-instruct}"
python_bin="${PYTHON_BIN:-/home/tans5/anaconda3/envs/prune_llm/bin/python}"
sparsity_ratios="${SPARSITY_RATIOS:-0 0.3 0.4 0.5 0.6 0.7 0.9 1}"
nsamples="${NSAMPLES:-5}"
seed="${SEED:-13}"
alpha="${ALPHA:-0.9}"
model_device="${MODEL_DEVICE:-cuda:0}"
compute_device="${COMPUTE_DEVICE:-cuda:1}"
seq_len="${SEQ_LEN:-512}"
sample_edge_ratio="${SAMPLE_EDGE_RATIO:-0.5}"
sample_edge_num="${SAMPLE_EDGE_NUM:--1}"
calib_data="${CALIB_DATA:-c4_independent}"
generated_calib_data_path="${GENERATED_CALIB_DATA_PATH:-downstream_calib_data/generated/gsm8k.jsonl}"
generated_calib_text_mode="${GENERATED_CALIB_TEXT_MODE:-prompt_answer}"
prune_ops="${PRUNE_OPS:-gate_proj}"
skip_prune_layer_ids="${SKIP_PRUNE_LAYER_IDS:-}"
curvature_dtype="${CURVATURE_DTYPE:-float32}"
model_dtype="${MODEL_DTYPE:-bfloat16}"
use_l2_norm="${USE_L2_NORM:-0}"
l2_norm_mode="${L2_NORM_MODE:-per_example}"
top_k_seq="${TOP_K_SEQ:-10}"
seq_select="${SEQ_SELECT:-top}"
curvature_lpf_window="${CURVATURE_LPF_WINDOW:-0}"
prune_scopes="${PRUNE_SCOPES:-per_op locally}"

curvature_dir="${CURVATURE_DIR:-out/ibm_2b_instruct/unstructured/curvature/gate_up/}"
compare_dir_root="${COMPARE_DIR_ROOT:-out/ibm_2b_instruct/unstructured/prune_overlap_compare_c4}"
mlp_activation="${MLP_ACTIVATION:-model}"

echo "Overlap log MLP activation: $mlp_activation"
echo "Overlap compares curvature pruning masks against WANDA and magnitude masks."

cuda_device=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)
export CUDA_VISIBLE_DEVICES=$cuda_device
export GENERATED_CALIB_DATA_PATH="$generated_calib_data_path"
export GENERATED_CALIB_TEXT_MODE="$generated_calib_text_mode"

skip_prune_layer_flag=""
if [ -n "$skip_prune_layer_ids" ]; then
    skip_prune_layer_flag="--skip_prune_layer_ids $skip_prune_layer_ids"
fi

prune_ops_flag=""
if [ -n "$prune_ops" ]; then
    prune_ops_flag="--prune_ops $prune_ops"
fi

l2_flag=""
if [ "$use_l2_norm" = "1" ]; then
    l2_flag="--L2-norm --l2_norm_mode $l2_norm_mode"
fi

curvature_l2_tag="no_L2_norm"
if [ "$use_l2_norm" = "1" ]; then
    if [ "$l2_norm_mode" = "per_example" ]; then
        curvature_l2_tag="L2_norm"
    else
        curvature_l2_tag="L2_norm_${l2_norm_mode}"
    fi
fi

curvature_seq_tag="curv_topseq_${top_k_seq}_pkl"
if [ "$seq_select" != "top" ] || [ "$curvature_lpf_window" -gt 1 ]; then
    curvature_seq_tag="curv_${seq_select}_seq_${top_k_seq}"
    if [ "$curvature_lpf_window" -gt 1 ]; then
        curvature_seq_tag="${curvature_seq_tag}_lpf_${curvature_lpf_window}"
    fi
    curvature_seq_tag="${curvature_seq_tag}_pkl"
fi

resolved_curvature_dir=""
if ls "$curvature_dir/$curvature_seq_tag"/layer_*_curvature.pkl >/dev/null 2>&1; then
    resolved_curvature_dir="$curvature_dir"
elif ls "$curvature_dir/curvature_pkl"/layer_*_curvature.pkl >/dev/null 2>&1; then
    resolved_curvature_dir="$curvature_dir"
elif ls "$curvature_dir/$calib_data/$curvature_l2_tag/seq_len_${seq_len}/$curvature_seq_tag"/layer_*_curvature.pkl >/dev/null 2>&1; then
    resolved_curvature_dir="$curvature_dir/$calib_data/$curvature_l2_tag/seq_len_${seq_len}"
fi

if [ -z "$resolved_curvature_dir" ]; then
    echo "No curvature PKLs found for overlap log." >&2
    echo "Checked:" >&2
    echo "  $curvature_dir/$curvature_seq_tag" >&2
    echo "  $curvature_dir/curvature_pkl" >&2
    echo "  $curvature_dir/$calib_data/$curvature_l2_tag/seq_len_${seq_len}/$curvature_seq_tag" >&2
    exit 1
fi

echo "Using curvature PKLs from: $resolved_curvature_dir"

run_overlap_scope() {
    prunescore_order=$1
    compare_dir=$2

    echo "Running overlap log scope=$prunescore_order"
    "$python_bin" src/llm_main.py \
        --model $model \
        --prune_method curvature \
        --sparsity_ratio $sparsity_ratios \
        --sparsity_type unstructured \
        --save $resolved_curvature_dir \
        --nsamples $nsamples \
        --seed $seed \
        --model_device $model_device \
        --compute_device $compute_device \
        --alpha $alpha \
        --calib_data $calib_data \
        --seqlen $seq_len \
        --sample_edge_ratio $sample_edge_ratio \
        --sample_edge_num $sample_edge_num \
        --prune_score_order high_to_low \
        --sparsity_schedule input \
        --save_curvature_dir $resolved_curvature_dir \
        --load_curvature_dir $resolved_curvature_dir \
        --prunescore_order $prunescore_order \
        --shared_top_k $top_k_seq \
        --shared_seq_select $seq_select \
        --curvature_lpf_window $curvature_lpf_window \
        --curvature_dtype $curvature_dtype \
        --per_layer_compare_dir $compare_dir \
        --model_dtype $model_dtype \
        --mlp_activation $mlp_activation \
        $l2_flag \
        $prune_ops_flag \
        $skip_prune_layer_flag \
        --run_prune_overlap_log
    echo "Finished overlap log scope=$prunescore_order"
}

for scope in $prune_scopes; do
    run_overlap_scope "$scope" "$compare_dir_root/$scope/"
done
