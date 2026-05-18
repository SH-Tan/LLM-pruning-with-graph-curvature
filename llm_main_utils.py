import os

import torch

from curv_layer_prune_utils import (
    draw_ppl_vs_sparsity,
    prune_scoped_curvature,
    save_eval_records_csv,
)
from curv_prune_utils import prune_global_curvature
from eval import eval_ppl
from per_layer_eval_utils import (
    draw_per_layer_ppl_vs_sparsity,
    layer_sparsity,
    list_curvature_pkl_layers,
    load_curvature_scores_for_layer,
    prune_curvature_layer,
    prune_magnitude_layer,
    prune_wanda_layer,
)
from prune import check_sparsity
from prune_magnitude import prune_magnitude
from prune_wanda import compute_wanda_scores, prune_wanda


def l2_path_tag(args):
    return "L2_norm" if getattr(args, "l2_norm", False) else "no_L2_norm"


def curvature_seq_tag(shared_top_k=None, shared_seq_select="top", curvature_lpf_window=0):
    if shared_top_k is None:
        return "curvature_pkl"
    if shared_seq_select == "top" and int(curvature_lpf_window) <= 1:
        return f"curv_topseq_{int(shared_top_k)}_pkl"
    tag = f"curv_{shared_seq_select}_seq_{int(shared_top_k)}"
    if int(curvature_lpf_window) > 1:
        tag += f"_lpf_{int(curvature_lpf_window)}"
    return f"{tag}_pkl"


def log_path(args):
    save_dir = os.path.join(args.save, args.calib_data, l2_path_tag(args), f"seq_len_{args.seqlen}")
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"eval_out_{args.prune_method}.txt"
    if args.prune_method == "curvature":
        setting_tag = curvature_seq_tag(
            args.shared_top_k,
            args.shared_seq_select,
            args.curvature_lpf_window,
        ).removesuffix("_pkl")
        file_name = f"eval_out_{args.prune_method}_{setting_tag}.txt"
    return os.path.join(save_dir, file_name)


def contains_curvature_pkls(path, shared_top_k=None, shared_seq_select="top", curvature_lpf_window=0):
    if path is None or not os.path.isdir(path):
        return False

    search_dirs = [path]
    if shared_top_k is not None:
        search_dirs.append(os.path.join(
            path,
            curvature_seq_tag(shared_top_k, shared_seq_select, curvature_lpf_window),
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


def curvature_dir(args, base_dir, create=False):
    if base_dir is None:
        return None

    expected_seq_dir = f"seq_len_{args.seqlen}"
    if os.path.basename(os.path.normpath(base_dir)) == expected_seq_dir or contains_curvature_pkls(
        base_dir,
        getattr(args, "shared_top_k", None),
        getattr(args, "shared_seq_select", "top"),
        getattr(args, "curvature_lpf_window", 0),
    ):
        save_dir = base_dir
    else:
        save_dir = os.path.join(base_dir, args.calib_data, l2_path_tag(args), expected_seq_dir)

    if create:
        os.makedirs(save_dir, exist_ok=True)
    return save_dir


def append_curvature_prune_summary(log_file, prune_summary, target_ratio=None, score_order=None):
    if not prune_summary:
        return

    total_pruned = sum(row["pruned_params"] for row in prune_summary)
    with open(log_file, "a+") as f:
        summary_header = "\nPruned parameters total"
        if target_ratio is not None:
            summary_header += f" (target_sparsity={target_ratio:.4f}"
            if score_order is not None:
                summary_header += f", score_order={score_order}"
            summary_header += ")"
        print(summary_header, file=f, flush=True)
        print(f"pruned_params={total_pruned}", file=f, flush=True)


def append_eval_run_header(log_file, args, target_ratio, score_order):
    with open(log_file, "a+") as f:
        print(
            "\n"
            f"Prune run: method={args.prune_method}, "
            f"score_order={score_order}, "
            f"target_sparsity={target_ratio:.4f}, "
            f"score_seq_len={args.seqlen}, "
            f"calib_data={args.calib_data}, "
            f"l2_norm={args.l2_norm}, "
            f"curvature_prune_scope={getattr(args, 'curvature_prune_scope', 'global')}, "
            f"shared_top_k={args.shared_top_k}, "
            f"shared_seq_select={args.shared_seq_select}, "
            f"curvature_lpf_window={args.curvature_lpf_window}",
            file=f,
            flush=True,
        )
        if args.prune_method == "wanda":
            print(
                "WANDA eval mode: fixed_score_seq_len=precompute scores once at score_seq_len and sweep pp_seq_len",
                file=f,
                flush=True,
            )


def append_eval_result(
    log_file,
    args,
    score_order,
    target_ratio,
    actual_sparsity_ratio,
    eval_mode,
    score_seq_len,
    pp_seq_len,
    ppl_test,
):
    with open(log_file, "a+") as f:
        print(
            f"{args.prune_method:<15}{score_order:<15}{str(args.l2_norm):<10}{target_ratio:<18.4f}"
            f"{actual_sparsity_ratio:<18.4f}{args.calib_data:<20}{eval_mode:<28}"
            f"{score_seq_len:<16d}{pp_seq_len:<12d}{ppl_test:<12.4f}",
            file=f,
            flush=True,
        )
        print("", file=f, flush=True)


def append_per_layer_eval_result(
    log_file,
    args,
    layer_idx,
    score_order,
    target_ratio,
    layer_actual_sparsity,
    model_actual_sparsity,
    pp_seq_len,
    ppl_test,
    score_cutoff,
):
    with open(log_file, "a+") as f:
        print(
            f"per-layer method={args.prune_method}, layer={layer_idx}, score_order={score_order}, "
            f"target_sparsity={target_ratio:.4f}, layer_actual_sparsity={layer_actual_sparsity:.4f}, "
            f"model_actual_sparsity={model_actual_sparsity:.4f}, pp_seq_len={pp_seq_len}, "
            f"ppl_test={ppl_test:.4f}, score_cutoff={score_cutoff}",
            file=f,
            flush=True,
        )


def resolve_sparsity_ratios(args):
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


def save_model_path(base_path, ratio, total_runs):
    if not base_path or total_runs <= 1:
        return base_path
    tag = f"{ratio:.4f}".rstrip("0").rstrip(".").replace(".", "p") or "0"
    return f"{base_path}_sparsity_{tag}"


def resolve_prune_score_orders(args):
    if args.prune_method in {"wanda", "magnitude"}:
        return ["low_to_high"]
    return list(dict.fromkeys(args.prune_score_order))


def per_layer_result_tag(args):
    if args.prune_method != "curvature":
        return args.prune_method

    setting_tag = curvature_seq_tag(
        args.shared_top_k,
        args.shared_seq_select,
        args.curvature_lpf_window,
    ).removesuffix("_pkl")
    return f"{args.prune_method}_{setting_tag}"


def reference_layer_indices(args, get_llm_fn, model_device, base_wanda_scores):
    if args.prune_method == "curvature":
        layer_ids = list_curvature_pkl_layers(
            args.load_curvature_dir or args.save_curvature_dir,
            args.shared_top_k,
            args.shared_seq_select,
            args.curvature_lpf_window,
        )
        if not layer_ids:
            raise ValueError("No curvature layer PKLs found for per-layer evaluation")
        return layer_ids

    if args.load_curvature_dir is not None:
        layer_ids = list_curvature_pkl_layers(
            args.load_curvature_dir,
            args.shared_top_k,
            args.shared_seq_select,
            args.curvature_lpf_window,
        )
        if layer_ids:
            return layer_ids

    if args.prune_method == "wanda" and base_wanda_scores is not None:
        return list(range(len(base_wanda_scores)))

    ref_model = get_llm_fn(args.model, args.cache_dir, model_device, args.seqlen)
    try:
        return list(range(len(ref_model.model.layers)))
    finally:
        del ref_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_per_layer_eval(
    args,
    get_llm_fn,
    tokenizer,
    model_device,
    sparsity_ratios,
    prune_score_orders,
    eval_seq_lens,
    prune_n,
    prune_m,
    save_filepath,
    base_wanda_scores=None,
):
    layer_ids = reference_layer_indices(args, get_llm_fn, model_device, base_wanda_scores)
    result_dir = os.path.dirname(save_filepath)
    result_tag = per_layer_result_tag(args)
    plot_dir = os.path.join(result_dir, f"per_layer_plots_{result_tag}")
    if args.prune_method == "curvature":
        prune_score_orders = ["high_to_low"]

    for score_order in prune_score_orders:
        args.prune_score_order = score_order
        for layer_idx in layer_ids:
            layer_records = []
            layer_curvature_scores = None
            if args.load_curvature_dir is not None:
                layer_curvature_scores = load_curvature_scores_for_layer(
                    args.load_curvature_dir,
                    layer_idx,
                    args.shared_top_k,
                    args.shared_seq_select,
                    args.curvature_lpf_window,
                )
            elif args.prune_method == "curvature":
                layer_curvature_scores = load_curvature_scores_for_layer(
                    args.save_curvature_dir,
                    layer_idx,
                    args.shared_top_k,
                    args.shared_seq_select,
                    args.curvature_lpf_window,
                )

            for target_ratio in sparsity_ratios:
                print(
                    f"per-layer eval: layer={layer_idx} sparsity={target_ratio:.4f} "
                    f"score_order={score_order}"
                )
                current_model = get_llm_fn(args.model, args.cache_dir, model_device, args.seqlen)
                current_model.eval()
                current_model.seqlen = args.seqlen
                args.sparsity_ratio = target_ratio

                score_cutoff = None
                if target_ratio != 0:
                    if args.prune_method == "curvature":
                        if layer_curvature_scores is None:
                            raise ValueError(
                                f"Missing curvature PKL for layer {layer_idx} in per-layer eval"
                            )
                        _, score_cutoff = prune_curvature_layer(
                            args,
                            current_model,
                            layer_idx,
                            layer_curvature_scores,
                            edge_log_path=save_filepath,
                        )
                    elif args.prune_method == "wanda":
                        _, score_cutoff = prune_wanda_layer(
                            args,
                            current_model,
                            layer_idx,
                            base_wanda_scores[layer_idx],
                            layer_curvature_scores=layer_curvature_scores,
                            prune_n=prune_n,
                            prune_m=prune_m,
                            edge_log_path=save_filepath,
                        )
                    elif args.prune_method == "magnitude":
                        _, score_cutoff = prune_magnitude_layer(
                            args,
                            current_model,
                            layer_idx,
                            layer_curvature_scores=layer_curvature_scores,
                            prune_n=prune_n,
                            prune_m=prune_m,
                            edge_log_path=save_filepath,
                        )

                model_actual_sparsity = check_sparsity(current_model)
                layer_actual = layer_sparsity(current_model, layer_idx)

                for seq in eval_seq_lens:
                    current_model.seqlen = seq
                    ppl_test = eval_ppl(args, current_model, tokenizer, model_device)
                    append_per_layer_eval_result(
                        save_filepath,
                        args,
                        layer_idx,
                        score_order,
                        target_ratio,
                        layer_actual,
                        model_actual_sparsity,
                        seq,
                        ppl_test,
                        score_cutoff,
                    )
                    record = {
                        "method": args.prune_method,
                        "layer_idx": int(layer_idx),
                        "score_order": score_order,
                        "target_sparsity": float(target_ratio),
                        "layer_actual_sparsity": float(layer_actual),
                        "model_actual_sparsity": float(model_actual_sparsity),
                        "pp_seq_len": int(seq),
                        "ppl_test": float(ppl_test),
                        "score_cutoff": None if score_cutoff is None else float(score_cutoff),
                        "cutoff_nonpositive": (
                            score_cutoff is not None and float(score_cutoff) <= 0.0
                        ),
                    }
                    layer_records.append(record)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                with open(save_filepath, "a+", encoding="utf-8") as f:
                    print("", file=f, flush=True)

                del current_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            plot_paths = draw_per_layer_ppl_vs_sparsity(
                layer_records,
                plot_dir,
                annotate_cutoff=(args.prune_method == "curvature"),
            )
            if plot_paths:
                print(f"Saved layer {layer_idx} plot to {plot_paths[-1]}")


def run_pp_eval(
    args,
    get_llm_fn,
    tokenizer,
    model_device,
    sparsity_ratios,
    prune_score_orders,
    eval_seq_lens,
    prune_n,
    prune_m,
    save_filepath,
    base_curvature_scores=None,
    base_wanda_scores=None,
):
    with open(save_filepath, "a+") as f:
        print(
            f"{'method':<15}{'score_order':<15}{'l2_norm':<10}{'target_sparsity':<18}"
            f"{'actual_sparsity':<18}{'calib_data':<20}{'eval_mode':<28}"
            f"{'score_seq_len':<16}{'pp_seq_len':<12}{'ppl_test':<12}",
            file=f,
            flush=True,
        )

    eval_records = []
    for score_order in prune_score_orders:
        args.prune_score_order = score_order
        for run_idx, target_ratio in enumerate(sparsity_ratios):
            print(
                f"starting sweep run {run_idx + 1}/{len(sparsity_ratios)} "
                f"with sparsity={target_ratio:.4f}, score_order={score_order}"
            )
            append_eval_run_header(save_filepath, args, target_ratio, score_order)
            current_model = get_llm_fn(args.model, args.cache_dir, model_device, args.seqlen)
            current_model.eval()
            current_model.seqlen = args.seqlen
            args.sparsity_ratio = target_ratio

            if target_ratio != 0:
                print("pruning starts")
                if args.prune_method == "curvature":
                    current_model.curvature_scores = base_curvature_scores
                    if args.curvature_prune_scope == "global":
                        prune_summary = prune_global_curvature(args, current_model)
                    else:
                        prune_summary = prune_scoped_curvature(args, current_model)
                    append_curvature_prune_summary(
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
                append_eval_result(
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
                eval_records.append(
                    {
                        "method": args.prune_method,
                        "prune_scope": (
                            args.curvature_prune_scope
                            if args.prune_method == "curvature"
                            else "global"
                        ),
                        "score_order": score_order,
                        "target_sparsity": float(target_ratio),
                        "actual_sparsity": float(actual_sparsity_ratio),
                        "pp_seq_len": int(seq),
                        "ppl_test": float(ppl_test),
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if args.save_model:
                model_save_path = save_model_path(args.save_model, target_ratio, len(sparsity_ratios))
                if len(prune_score_orders) > 1:
                    model_save_path = os.path.join(model_save_path, score_order)
                current_model.save_pretrained(model_save_path)
                tokenizer.save_pretrained(model_save_path)

            del current_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if args.prune_method == "wanda" and target_ratio != 0:
                for seq in eval_seq_lens:
                    print(
                        f"recomputing WANDA scores with score/eval seqlen = {seq} "
                        "for an additional perplexity measurement"
                    )
                    seq_model = get_llm_fn(args.model, args.cache_dir, model_device, seq)
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
                    append_eval_result(
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

    if eval_records:
        result_dir = os.path.dirname(save_filepath)
        scope = args.curvature_prune_scope if args.prune_method == "curvature" else "global"
        result_tag = f"{args.prune_method}_{scope}"
        csv_path = os.path.join(result_dir, f"ppl_vs_sparsity_{result_tag}.csv")
        plot_path = os.path.join(result_dir, f"ppl_vs_sparsity_{result_tag}.png")
        save_eval_records_csv(eval_records, csv_path)
        drawn_path = draw_ppl_vs_sparsity(eval_records, plot_path)
        if drawn_path is not None:
            print(f"Saved PPL vs sparsity plot: {drawn_path}")
