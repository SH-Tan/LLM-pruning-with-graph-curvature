import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import random
import os
import argparse
from prune import check_sparsity
from prune_curvature import prune_curvature
from prune_magnitude import prune_magnitude
from prune_wanda import compute_wanda_scores, prune_wanda
from eval import eval_ppl
from curv_prune_utils import prune_global_curvature


from huggingface_hub import login


def enable_hf_offline_mode():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def safe_hf_login(token):
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


safe_hf_login(token)

print('# of gpus: ', torch.cuda.device_count())


def _log_path(args):
    save_dir = os.path.join(args.save, args.calib_data, _l2_path_tag(args), f"seq_len_{args.seqlen}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = f"eval_out_{args.prune_method}.txt"
    if args.prune_method == "curvature":
        setting_tag = _curvature_seq_tag(
            args.shared_top_k,
            args.shared_seq_select,
            args.curvature_lpf_window,
        ).removesuffix("_pkl")
        file_name = f"eval_out_{args.prune_method}_{setting_tag}.txt"
    return os.path.join(save_dir, file_name)


def _curvature_seq_tag(shared_top_k=None, shared_seq_select="top", curvature_lpf_window=0):
    if shared_top_k is None:
        return "curvature_pkl"
    if shared_seq_select == "top" and int(curvature_lpf_window) <= 1:
        return f"curv_topseq_{int(shared_top_k)}_pkl"
    tag = f"curv_{shared_seq_select}_seq_{int(shared_top_k)}"
    if int(curvature_lpf_window) > 1:
        tag += f"_lpf_{int(curvature_lpf_window)}"
    return f"{tag}_pkl"


def _contains_curvature_pkls(path, shared_top_k=None, shared_seq_select="top", curvature_lpf_window=0):
    if path is None or not os.path.isdir(path):
        return False

    search_dirs = [path]
    if shared_top_k is not None:
        search_dirs.append(os.path.join(
            path,
            _curvature_seq_tag(shared_top_k, shared_seq_select, curvature_lpf_window),
        ))
    search_dirs.append(os.path.join(path, "curvature_pkl"))

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        if any(
            file_name.startswith("layer_") and file_name.endswith("_curvature.pkl")
            for file_name in os.listdir(search_dir)
        ):
            return True
    return False


def _curvature_dir(args, base_dir, create=False):
    if base_dir is None:
        return None

    expected_seq_dir = f"seq_len_{args.seqlen}"
    if os.path.basename(os.path.normpath(base_dir)) == expected_seq_dir or _contains_curvature_pkls(
        base_dir,
        getattr(args, "shared_top_k", None),
        getattr(args, "shared_seq_select", "top"),
        getattr(args, "curvature_lpf_window", 0),
    ):
        save_dir = base_dir
    else:
        save_dir = os.path.join(base_dir, args.calib_data, _l2_path_tag(args), expected_seq_dir)

    if create:
        os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _l2_path_tag(args):
    return "L2_norm" if getattr(args, "l2_norm", False) else "no_L2_norm"


def _append_curvature_prune_summary(log_path, prune_summary, target_ratio=None, score_order=None):
    if not prune_summary:
        return

    total_pruned = sum(row["pruned_params"] for row in prune_summary)

    with open(log_path, "a+") as f:
        summary_header = "\nPruned parameters total"
        if target_ratio is not None:
            summary_header += f" (target_sparsity={target_ratio:.4f}"
            if score_order is not None:
                summary_header += f", score_order={score_order}"
            summary_header += ")"
        print(summary_header, file=f, flush=True)
        print(
            f"pruned_params={total_pruned}",
            file=f,
            flush=True,
        )


def _append_eval_run_header(log_path, args, target_ratio, score_order):
    with open(log_path, "a+") as f:
        print(
            "\n"
            f"Prune run: method={args.prune_method}, "
            f"score_order={score_order}, "
            f"target_sparsity={target_ratio:.4f}, "
            f"score_seq_len={args.seqlen}, "
            f"calib_data={args.calib_data}, "
            f"l2_norm={args.l2_norm}, "
            f"shared_top_k={args.shared_top_k}, "
            f"shared_seq_select={args.shared_seq_select}, "
            f"curvature_lpf_window={args.curvature_lpf_window}",
            file=f,
            flush=True,
        )
        if args.prune_method == "wanda":
            print(
                "WANDA eval modes: "
                "fixed_score_seq_len=precompute scores once at score_seq_len and sweep pp_seq_len; "
                "recompute_score_at_pp_len=recompute WANDA scores with score_seq_len=pp_seq_len before pruning/eval",
                file=f,
                flush=True,
            )


def _append_eval_result(
    log_path,
    args,
    score_order,
    target_ratio,
    actual_sparsity_ratio,
    eval_mode,
    score_seq_len,
    pp_seq_len,
    ppl_test,
):
    with open(log_path, "a+") as f:
        print(
            f"{args.prune_method:<15}{score_order:<15}{str(args.l2_norm):<10}{target_ratio:<18.4f}"
            f"{actual_sparsity_ratio:<18.4f}{args.calib_data:<20}{eval_mode:<28}"
            f"{score_seq_len:<16d}{pp_seq_len:<12d}{ppl_test:<12.4f}",
            file=f,
            flush=True,
        )
        print("", file=f, flush=True)


def _resolve_sparsity_ratios(args):
    ratios = [float(ratio) for ratio in args.sparsity_ratio]
    if not ratios:
        ratios = [0.0]

    deduped = []
    seen = set()
    for ratio in ratios:
        if ratio < 0 or ratio > 1:
            raise ValueError(f"sparsity_ratio must be in [0, 1], got {ratio}")
        if ratio not in seen:
            deduped.append(ratio)
            seen.add(ratio)

    if args.sparsity_schedule == "low_to_high":
        deduped.sort()
    elif args.sparsity_schedule == "high_to_low":
        deduped.sort(reverse=True)

    return deduped


def _format_sparsity_tag(ratio):
    return f"{ratio:.4f}".rstrip("0").rstrip(".").replace(".", "p") or "0"


def _save_model_path(base_path, ratio, total_runs):
    if not base_path or total_runs <= 1:
        return base_path
    return f"{base_path}_sparsity_{_format_sparsity_tag(ratio)}"


def _resolve_prune_score_orders(args):
    if args.prune_method in {"wanda", "magnitude"}:
        return ["low_to_high"]

    return list(dict.fromkeys(args.prune_score_order))


def get_llm(model_name, cache_dir="llm_weights", device = "cpu", seqlen = "1024"):
    print("Loading model:", model_name)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map=device
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
    
    if hasattr(model, 'hf_device_map'):
        print('hf_device_map = ', model.hf_device_map)
    else:
        # The device map is handled by accelerate under the hood
        print("Model loaded with device_map, but hf_device_map not directly accessible")

    model.seqlen = seqlen # model.config.max_position_embeddings
    # print(model.seqlen)
    return model



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=13, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument(
        '--sparsity_ratio',
        type=float,
        nargs='+',
        default=[0.0],
        help='One or more sparsity levels, e.g. --sparsity_ratio 0.1 0.3 0.5',
    )
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["curvature", "wanda", "magnitude"])
    parser.add_argument("--use_variant", action="store_true", help="Use the WANDA variant pruning threshold search.")
    parser.add_argument(
        "--prune_score_order",
        type=str,
        nargs='+',
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
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--model_device', type=str, default="cuda:0", help='Device for model load.')
    parser.add_argument('--compute_device', type=str, default="cuda:1", help='Device for curvature computing.')
    parser.add_argument('--alpha', type=float, default=0., required=False, help='Alpha used for distribution')
    parser.add_argument(
        '--L2-norm',
        dest='l2_norm',
        action='store_true',
        help='Reduce node tensors of shape [seq, d] to [1, d] using L2 norm across seq before node normalization.',
    )
    parser.add_argument('--save_curvature_dir',type=str,default=None,help='Directory to save per-layer curvature pkl files.')
    parser.add_argument('--load_curvature_dir',type=str,default=None,help='Directory to load previously saved per-layer curvature pkl files.')
    parser.add_argument('--calib_data',type=str,default="c4_independent",choices=["c4_independent", "c4_dependent"], help='Calibration data for pruning [c4_dependent, c4_independent].')
    parser.add_argument('--sample_edge_num', type=int, default=-1, help='Number of edge samples for curvature calculation.')
    parser.add_argument('--sample_edge_ratio', type=float, default=1.0, help='Ratio of edge samples for curvature calculation.')
    parser.add_argument('--shared_top_k', type=int, default=10, help='Top score-ranked seq positions per edge for curvature; -1 evaluates all seq positions.')
    parser.add_argument(
        '--shared_seq_select',
        type=str,
        choices=["top", "median"],
        default="top",
        help='Seq selection mode when shared_top_k > 0: top score-ranked seqs or seqs closest to the median score.',
    )
    parser.add_argument(
        '--curvature_lpf_window',
        type=int,
        default=0,
        help='Sliding median low-pass window for all-seq curvature; active when shared_top_k is -1 and window > 1.',
    )
    parser.add_argument(
        '--save_parameter_metric_logs',
        action='store_true',
        help='Save per-parameter curvature/metric logs during curvature analysis.',
    )
    parser.add_argument(
        '--draw_parameter_metric_plots',
        action='store_true',
        help='Draw per-parameter curvature/metric comparison plots after curvature analysis.',
    )
    parser.add_argument('--seqlen', type=int, default=32, help='Input seq len.')
    parser.add_argument(
        '--pp_seqlen',
        type=int,
        nargs='*',
        default=[],
        help='Perplexity eval sequence lengths, e.g. --pp_seqlen 32 128 256.',
    )

    parser.add_argument("--eval_zero_shot", type=int, default=0, help='evaluate on downsteam zero shot tasks')
    args = parser.parse_args()
    # if args.prune_method == "curvature":
    #     args.save_parameter_metric_logs = True
    #     args.draw_parameter_metric_plots = True
    sparsity_ratios = _resolve_sparsity_ratios(args)
    prune_score_orders = _resolve_prune_score_orders(args)

    if args.prune_method == "curvature":
        if args.save_curvature_dir is None:
            args.save_curvature_dir = args.save
        args.save_curvature_dir = _curvature_dir(args, args.save_curvature_dir, create=True)
    else:
        args.save_curvature_dir = _curvature_dir(args, args.save_curvature_dir, create=False)
    args.load_curvature_dir = _curvature_dir(args, args.load_curvature_dir, create=False)
    
    # set random seed
    seed = args.seed
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if torch.cuda.is_available():
        model_device = args.model_device
        compute_device = args.compute_device
    else:
        model_device = "cpu"
        compute_device = "cpu"
        
    print(f"Using {model_device} model device, {compute_device} for computing")

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert all(ratio == 0.5 for ratio in sparsity_ratios), (
            "sparsity ratio must be 0.5 for structured N:M sparsity"
        )
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    save_filepath = _log_path(args)

    needs_pruning = any(ratio != 0 for ratio in sparsity_ratios)
    base_curvature_scores = None
    base_wanda_scores = None
    
    if args.load_curvature_dir is not None:
        if _contains_curvature_pkls(
            args.load_curvature_dir,
            args.shared_top_k,
            args.shared_seq_select,
            args.curvature_lpf_window,
        ):
            print(f"using curvature pkls from {args.load_curvature_dir}")
        else:
            print(f"no curvature pkls found at {args.load_curvature_dir}; curvature will be computed if needed")
            args.load_curvature_dir = None

    if needs_pruning and args.prune_method == "curvature":
        print(f"loading llm model {args.model} for curvature precomputation")
        model = get_llm(args.model, args.cache_dir, model_device, args.seqlen)
        model.eval()
        print("loading curvature scores" if args.load_curvature_dir else "precomputing curvature scores")
        args.curvature_timing_log_path = save_filepath
        prune_curvature(args, model, tokenizer, compute_device, prune_n, prune_m)
        base_curvature_scores = model.curvature_scores
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    with open(save_filepath, "a+") as f:
        print(
            f"{'method':<15}{'score_order':<15}{'l2_norm':<10}{'target_sparsity':<18}"
            f"{'actual_sparsity':<18}{'calib_data':<20}{'eval_mode':<28}"
            f"{'score_seq_len':<16}{'pp_seq_len':<12}{'ppl_test':<12}",
            file=f,
            flush=True,
        )

    for score_order in prune_score_orders:
        args.prune_score_order = score_order
        for run_idx, target_ratio in enumerate(sparsity_ratios):
            print(
                f"starting sweep run {run_idx + 1}/{len(sparsity_ratios)} "
                f"with sparsity={target_ratio:.4f}, score_order={score_order}"
            )
            _append_eval_run_header(save_filepath, args, target_ratio, score_order)
            current_model = get_llm(args.model, args.cache_dir, model_device, args.seqlen)
            current_model.eval()

            current_model.seqlen = args.seqlen
            args.sparsity_ratio = target_ratio

            if target_ratio != 0:
                print("pruning starts")
                if args.prune_method == "curvature":
                    current_model.curvature_scores = base_curvature_scores
                    prune_summary = prune_global_curvature(args, current_model)
                    _append_curvature_prune_summary(
                        save_filepath,
                        prune_summary,
                        target_ratio=target_ratio,
                        score_order=score_order,
                    )
                elif args.prune_method == "wanda":
                    current_model.wanda_scores = base_wanda_scores
                    prune_wanda(args, current_model, tokenizer, model_device, prune_n, prune_m)
                elif args.prune_method == "magnitude":
                    prune_magnitude(args, current_model, tokenizer, model_device, prune_n, prune_m)

            print("*" * 30)
            actual_sparsity_ratio = check_sparsity(current_model)
            print(f"sparsity sanity check {actual_sparsity_ratio:.4f}")
            print("*" * 30)

            for seq in eval_seq_lens:
                current_model.seqlen = seq
                ppl_test = eval_ppl(args, current_model, tokenizer, model_device)
                eval_mode = "fixed_score_seq_len" if args.prune_method == "wanda" else "standard_eval"
                print(f"wikitext perplexity {ppl_test} using pp_seqlen = {seq}")
                _append_eval_result(
                    save_filepath,
                    args,
                    score_order,
                    target_ratio,
                    actual_sparsity_ratio,
                    eval_mode,
                    args.seqlen,
                    seq,
                    ppl_test,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if args.prune_method == "wanda" and target_ratio != 0:
                for seq in eval_seq_lens:
                    print(
                        f"recomputing WANDA scores with score/eval seqlen = {seq} "
                        "for an additional perplexity measurement"
                    )
                    seq_model = get_llm(args.model, args.cache_dir, model_device, seq)
                    seq_model.eval()
                    seq_model.seqlen = seq
                    seq_model.wanda_scores = compute_wanda_scores(args, seq_model, tokenizer, model_device)
                    prune_wanda(args, seq_model, tokenizer, model_device, prune_n, prune_m)

                    seq_actual_sparsity_ratio = check_sparsity(seq_model)
                    print(f"sparsity sanity check {seq_actual_sparsity_ratio:.4f} after recomputing scores")

                    ppl_test = eval_ppl(args, seq_model, tokenizer, model_device)
                    print(
                        f"wikitext perplexity {ppl_test} using pp_seqlen = {seq} "
                        f"with recomputed WANDA score seqlen = {seq}"
                    )
                    _append_eval_result(
                        save_filepath,
                        args,
                        score_order,
                        target_ratio,
                        seq_actual_sparsity_ratio,
                        "recompute_score_at_pp_len",
                        seq,
                        seq,
                        ppl_test,
                    )

                    del seq_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if args.save_model:
                save_model_path = _save_model_path(args.save_model, target_ratio, len(sparsity_ratios))
                if len(prune_score_orders) > 1:
                    save_model_path = os.path.join(save_model_path, score_order)
                current_model.save_pretrained(save_model_path)
                tokenizer.save_pretrained(save_model_path)

            del current_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # if args.eval_zero_shot:
    #     accelerate=False
    #     if "30b" in args.model or "65b" in args.model or "70b" in args.model:
    #         accelerate=True

    #     task_list=["hellaswag","winogrande","openbookqa","arc_easy"]
    #     num_shot = 0
    #     results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate, "cuda:0")
    #     with open(save_filepath, "a+") as f:
    #         print("\n********************************\n", file=f, flush=True)
    #         print("\nzero_shot evaluation results\n\n", file=f, flush=True)
    #         print(results, file=f, flush=True)

if __name__ == '__main__':
    main()
