import argparse
import os
import random

import numpy as np
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_main_utils import (
    contains_curvature_pkls,
    curvature_dir,
    log_path,
    resolve_prune_score_orders,
    resolve_sparsity_ratios,
    run_per_layer_eval,
    run_pp_eval,
)
from prune import load_curvature_pkls
from prune_curvature import prune_curvature
from prune_wanda import compute_wanda_scores

def enable_hf_offline_mode():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def safe_hf_login(token):
    if not token:
        print("Hugging Face login skipped: HF_TOKEN is not set")
        return
    try:
        login(token)
        print("Hugging Face login succeeded")
    except Exception as exc:
        enable_hf_offline_mode()
        print(f"Hugging Face login skipped: {exc}")


def _is_network_error(exc):
    msg = str(exc).lower()
    return (
        "name resolution" in msg
        or "connecterror" in msg
        or "connection error" in msg
        or "temporary failure" in msg
        or "offline" in msg
    )


safe_hf_login(os.environ.get("HF_TOKEN"))
print("# of gpus: ", torch.cuda.device_count())


def get_llm(model_name, cache_dir="llm_weights", device="cpu", seqlen="1024"):
    print("Loading model:", model_name)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map=device,
        )
    except Exception as exc:
        if not _is_network_error(exc):
            raise
        enable_hf_offline_mode()
        print(f"Falling back to local cached model files: {exc}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map=device,
            local_files_only=True,
        )

    if hasattr(model, "hf_device_map"):
        print("hf_device_map = ", model.hf_device_map)
    else:
        print("Model loaded with device_map, but hf_device_map not directly accessible")

    model.seqlen = seqlen
    return model


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="LLaMA model")
    parser.add_argument("--seed", type=int, default=13, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples.")
    parser.add_argument(
        "--sparsity_ratio",
        type=float,
        nargs="+",
        default=[0.0],
        help="One or more sparsity levels, e.g. --sparsity_ratio 0.1 0.3 0.5",
    )
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["curvature", "wanda", "magnitude"])
    parser.add_argument("--use_variant", action="store_true", help="Use the WANDA variant pruning threshold search.")
    parser.add_argument(
        "--prune_score_order",
        type=str,
        nargs="+",
        choices=["high_to_low", "low_to_high"],
        default=["high_to_low", "low_to_high"],
        help="For score-based pruning, remove highest scores first or lowest scores first.",
    )
    parser.add_argument(
        "--sparsity_schedule",
        type=str,
        choices=["input", "low_to_high", "high_to_low"],
        default="input",
        help="Order used when sweeping multiple sparsity ratios.",
    )
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument("--save", type=str, default=None, help="Path to save results.")
    parser.add_argument("--save_model", type=str, default=None, help="Path to save the pruned model.")
    parser.add_argument("--model_device", type=str, default="cuda:0", help="Device for model load.")
    parser.add_argument("--compute_device", type=str, default="cuda:1", help="Device for curvature computing.")
    parser.add_argument("--alpha", type=float, default=0.0, required=False, help="Alpha used for distribution")
    parser.add_argument(
        "--L2-norm",
        dest="l2_norm",
        action="store_true",
        help="Reduce node tensors of shape [seq, d] to [1, d] using L2 norm across seq before node normalization.",
    )
    parser.add_argument(
        "--l2_norm_mode",
        type=str,
        choices=["per_example", "all_examples"],
        default="per_example",
        help="L2 reduction mode for curvature: per example, or all calibration examples and seqs.",
    )
    parser.add_argument("--save_curvature_dir", type=str, default=None, help="Directory to save per-layer curvature pkl files.")
    parser.add_argument("--load_curvature_dir", type=str, default=None, help="Directory to load previously saved per-layer curvature pkl files.")
    parser.add_argument(
        "--calib_data",
        type=str,
        default="c4_independent",
        choices=["c4_independent", "c4_dependent"],
        help="Calibration data for pruning [c4_dependent, c4_independent].",
    )
    parser.add_argument("--sample_edge_num", type=int, default=-1, help="Number of edge samples for curvature calculation.")
    parser.add_argument("--sample_edge_ratio", type=float, default=1.0, help="Ratio of edge samples for curvature calculation.")
    parser.add_argument(
        "--curvature_prune_scope",
        type=str,
        choices=["global", "per_layer", "per_layer_op"],
        default="global",
        help="Curvature pruning scope: global, per layer over all ops, or per layer per op.",
    )
    parser.add_argument(
        "--prunescore_order",
        "--prunescaore_order",
        "--prunescore",
        "--prunescaore",
        "--prune_score_scope",
        dest="prunescore_order",
        type=str,
        choices=["globally", "locally"],
        default=None,
        help="Alias for curvature pruning score scope: globally over full model, or locally per block/layer op.",
    )
    parser.add_argument("--shared_top_k", type=int, default=10, help="Top score-ranked seq positions per edge for curvature; -1 evaluates all seq positions.")
    parser.add_argument(
        "--shared_seq_select",
        type=str,
        choices=["top", "median"],
        default="top",
        help="Seq selection mode when shared_top_k > 0: top score-ranked seqs or seqs closest to the median score.",
    )
    parser.add_argument(
        "--curvature_lpf_window",
        type=int,
        default=0,
        help="Sliding median low-pass window for all-seq curvature; active when shared_top_k is -1 and window > 1.",
    )
    parser.add_argument(
        "--save_parameter_metric_logs",
        action="store_true",
        help="Save per-parameter curvature/metric logs during curvature analysis.",
    )
    parser.add_argument(
        "--draw_parameter_metric_plots",
        action="store_true",
        help="Draw per-parameter curvature/metric comparison plots after curvature analysis.",
    )
    parser.add_argument(
        "--run_per_layer_eval",
        action="store_true",
        help="Run one-layer-at-a-time pruning sweeps and save per-layer PPL results and plots.",
    )
    parser.add_argument(
        "--per_layer_ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional layer indices for per-layer eval. If omitted, all available layers are evaluated.",
    )
    parser.add_argument(
        "--per_layer_compare_dir",
        type=str,
        default=None,
        help="Shared directory for per-layer method comparison CSVs and plots.",
    )
    parser.add_argument(
        "--run_pp_eval",
        action="store_true",
        help="Run perplexity evaluation after score/curvature calculation.",
    )
    parser.add_argument("--seqlen", type=int, default=32, help="Input seq len.")
    parser.add_argument(
        "--pp_seqlen",
        type=int,
        nargs="*",
        default=[],
        help="Perplexity eval sequence lengths, e.g. --pp_seqlen 32 128 256.",
    )
    parser.add_argument("--eval_zero_shot", type=int, default=0, help="evaluate on downsteam zero shot tasks")
    return parser


def _set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.prunescore_order == "globally":
        args.curvature_prune_scope = "global"
    elif args.prunescore_order == "locally":
        args.curvature_prune_scope = "per_layer"

    sparsity_ratios = resolve_sparsity_ratios(args)
    prune_score_orders = resolve_prune_score_orders(args)

    if args.prune_method == "curvature":
        if args.save_curvature_dir is None:
            args.save_curvature_dir = args.save
        args.save_curvature_dir = curvature_dir(args, args.save_curvature_dir, create=True)
    else:
        args.save_curvature_dir = curvature_dir(args, args.save_curvature_dir, create=False)
    args.load_curvature_dir = curvature_dir(args, args.load_curvature_dir, create=False)

    if (
        args.prune_method == "curvature"
        and (args.save_parameter_metric_logs or args.draw_parameter_metric_plots)
    ):
        args.load_curvature_dir = None

    _set_seed(args.seed)

    if torch.cuda.is_available():
        model_device = args.model_device
        compute_device = args.compute_device
    else:
        model_device = "cpu"
        compute_device = "cpu"

    print(f"Using {model_device} model device, {compute_device} for computing")

    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert all(ratio == 0.5 for ratio in sparsity_ratios), (
            "sparsity ratio must be 0.5 for structured N:M sparsity"
        )
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    save_filepath = log_path(args)

    needs_pruning = any(ratio != 0 for ratio in sparsity_ratios)
    base_curvature_scores = None
    base_wanda_scores = None

    if args.load_curvature_dir is not None:
        if contains_curvature_pkls(
            args.load_curvature_dir,
            args.shared_top_k,
            args.shared_seq_select,
            args.curvature_lpf_window,
        ):
            print(f"using curvature pkls from {args.load_curvature_dir}")
        else:
            print(f"no curvature pkls found at {args.load_curvature_dir}; curvature will be computed if needed")
            args.load_curvature_dir = None

    if (
        args.prune_method == "curvature"
        and (
            (args.run_pp_eval and needs_pruning)
            or (not args.run_pp_eval and not args.run_per_layer_eval)
            or (args.run_per_layer_eval and args.load_curvature_dir is None)
        )
    ):
        print(f"loading llm model {args.model} for curvature precomputation")
        model = get_llm(args.model, args.cache_dir, model_device, args.seqlen)
        model.eval()
        if args.load_curvature_dir is not None:
            print(f"loading curvature scores from {args.load_curvature_dir}")
            model.curvature_scores = load_curvature_pkls(
                args.load_curvature_dir,
                len(model.model.layers),
                shared_top_k=args.shared_top_k,
                shared_seq_select=args.shared_seq_select,
                curvature_lpf_window=args.curvature_lpf_window,
            )
            loaded = sum(len(layer_scores) for layer_scores in model.curvature_scores)
            if loaded > 0:
                print(f"Loaded {loaded} curvature tensors from {args.load_curvature_dir}")
            else:
                print("No curvature tensors loaded; recomputing curvature scores")
                args.curvature_timing_log_path = save_filepath
                prune_curvature(args, model, tokenizer, compute_device, prune_n, prune_m)
        else:
            print("precomputing curvature scores")
            args.curvature_timing_log_path = save_filepath
            prune_curvature(args, model, tokenizer, compute_device, prune_n, prune_m)
        base_curvature_scores = model.curvature_scores
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.prune_method == "curvature" and not args.run_pp_eval and not args.run_per_layer_eval:
        print("Skipping PP eval; curvature calculation is complete.")
        return

    if needs_pruning and args.prune_method == "wanda":
        print(f"loading llm model {args.model} for WANDA score precomputation")
        model = get_llm(args.model, args.cache_dir, model_device, args.seqlen)
        model.eval()
        model.seqlen = args.seqlen
        print(f"precomputing WANDA scores with seqlen={args.seqlen}")
        base_wanda_scores = compute_wanda_scores(args, model, tokenizer, model_device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eval_seq_lens = args.pp_seqlen if len(args.pp_seqlen) >= 1 else [args.seqlen]

    if args.run_per_layer_eval:
        run_per_layer_eval(
            args,
            get_llm,
            tokenizer,
            model_device,
            sparsity_ratios,
            prune_score_orders,
            eval_seq_lens,
            prune_n,
            prune_m,
            save_filepath,
            base_wanda_scores=base_wanda_scores,
        )
        if not args.run_pp_eval:
            return

    if not args.run_pp_eval:
        print("Skipping PP eval.")
        return

    run_pp_eval(
        args,
        get_llm,
        tokenizer,
        model_device,
        sparsity_ratios,
        prune_score_orders,
        eval_seq_lens,
        prune_n,
        prune_m,
        save_filepath,
        base_curvature_scores=base_curvature_scores,
        base_wanda_scores=base_wanda_scores,
    )


if __name__ == "__main__":
    main()
