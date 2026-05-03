import os
import time

import torch

import curv_analysis_utils as analysis_utils
from cal_curvature import compute_op_curvature
from curv_shortest_path_utils import build_shortest_path_cache
from curv_tensor_utils import build_layer_cache
from data_c4 import get_loaders_c4
from layerwrapper_curv import collect_layer_data, _make_lm_head_op
from prune import (
    find_layers,
    load_curvature_pkls,
    prepare_calibration_input,
    save_layer_curvature_pkl,
)


def _append_curvature_timing_header(log_path):
    if log_path is None:
        return

    with open(log_path, "a+") as f:
        print("\nCurvature calculation timing by example", file=f, flush=True)
        print(
            f"{'event':<22}{'layer':<8}{'sample':<10}{'nsamples':<10}"
            f"{'seq_len':<10}{'l2_norm':<10}{'elapsed_sec':<14}",
            file=f,
            flush=True,
        )


def _append_curvature_timing(log_path, layer_idx, sample_idx, elapsed_sec, nsamples, seq_len, l2_norm):
    if log_path is None:
        return

    with open(log_path, "a+") as f:
        print(
            f"{'curvature_example_time':<22}{layer_idx:<8d}{sample_idx:<10d}"
            f"{nsamples:<10d}{seq_len:<10d}{str(l2_norm):<10}{elapsed_sec:<14.6f}",
            file=f,
            flush=True,
        )


def _curvature_save_metadata(args):
    return {
        "model": getattr(args, "model", None),
        "seed": getattr(args, "seed", None),
        "nsamples": getattr(args, "nsamples", None),
        "alpha": getattr(args, "alpha", None),
        "l2_norm": getattr(args, "l2_norm", False),
        "prune_method": getattr(args, "prune_method", None),
    }


def _curvature_pkl_dir(base_dir):
    if base_dir is None:
        return None
    return os.path.join(base_dir, "curvature_pkl")


def _parameter_metric_log_root(base_dir, seq_len, dataset_name):
    if base_dir is None:
        return None
    log_root = os.path.join(
        base_dir,
        "parameter_metric_logs",
        f"seq_len_{int(seq_len)}",
        str(dataset_name),
    )
    os.makedirs(log_root, exist_ok=True)
    return log_root


def _maybe_draw_parameter_metric_plots(args, log_root):
    if not getattr(args, "draw_parameter_metric_plots", False):
        return
    if log_root is None:
        return
    # Per-parameter plots are finalized inside cal_curvature.py once all seq
    # for a parameter have been processed, so skip the end-of-run redraw here.


def prune_curvature(args, model, tokenizer, device="cuda:0", prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    model_device = args.model_device
    compute_device = args.compute_device

    print("loading calibration data")
    dataloader, _ = get_loaders_c4(
        args.calib_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")

    model.eval()

    with torch.no_grad():
        inps, _, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, model_device, args.nsamples
        )
    del dataloader

    inps = inps.cpu()

    if attention_mask is not None:
        attention_mask = attention_mask.to(model_device)
    if position_ids is not None:
        position_ids = position_ids.to(model_device)

    layers = model.model.layers
    target_ops = ["q_proj", "k_proj"]
    last_layer_idx = len(layers) - 1

    model.curvature_scores = [{} for _ in range(len(layers))]
    curvature_save_dir = getattr(args, "save_curvature_dir", None)
    curvature_load_dir = getattr(args, "load_curvature_dir", None)
    curvature_pkl_save_dir = _curvature_pkl_dir(curvature_save_dir)
    curvature_timing_log_path = getattr(args, "curvature_timing_log_path", None)
    curvature_metadata = _curvature_save_metadata(args)
    parameter_metric_log_root = None
    if getattr(args, "save_parameter_metric_logs", False) or getattr(args, "draw_parameter_metric_plots", False):
        parameter_metric_log_root = _parameter_metric_log_root(
            curvature_save_dir or os.path.dirname(__file__),
            model.seqlen,
            args.calib_data,
        )

    if curvature_load_dir is not None:
        model.curvature_scores = load_curvature_pkls(curvature_load_dir, len(layers))
        print(f"Loaded curvature pkls from {curvature_load_dir}")
        model.config.use_cache = use_cache
        return

    _append_curvature_timing_header(curvature_timing_log_path)

    uses_prev_layer_context = any(short in {"q_proj", "k_proj", "v_proj", "down_proj"} for short in target_ops)
    prev_layer_outputs = [None] * args.nsamples if uses_prev_layer_context else None
    layer_cache = {}
    sp_cache = {}

    try:
        for i, layer in enumerate(layers):
            print(f"Processing layer {i}")

            layer_cache = {}
            sp_cache = {}

            layer = layer.to(model_device)
            layer_subset = find_layers(layer)

            op_modules = {
                short: next(m for n, m in layer_subset.items() if n.endswith(short))
                for short in target_ops
            }
            modules_items = list(op_modules.items())
            has_down_proj_target = "down_proj" in op_modules

            for short, module in modules_items:
                W = module.weight
                module.min_curvature = torch.full_like(W, float("inf"), device="cpu")

            new_inps = torch.empty_like(inps, device="cpu")

            for j in range(args.nsamples):
                if torch.cuda.is_available():
                    torch.cuda.synchronize(model_device)
                    torch.cuda.synchronize(compute_device)
                sample_start_time = time.perf_counter()
                
                x = inps[j:j + 1].to(model_device, non_blocking=True)
                next_layer = layers[i + 1] if i < last_layer_idx else None

                with torch.no_grad():
                    x_out, operations, num_q_heads, num_kv_heads, repeat, head_dim = collect_layer_data(
                        layer,
                        x,
                        attention_mask,
                        position_ids,
                        model,
                        next_layer=next_layer,
                    )

                x_out = x_out.detach().cpu()

                print("Finish getting layer data!!!")

                # prev_* tensors come from layer i-1 and are used as context for layer i curvature.
                if prev_layer_outputs is not None and prev_layer_outputs[j] is not None:
                    for name in ["o_proj", "gate_up_out", "down_proj"]:
                        if name in prev_layer_outputs[j]:
                            operations[f"prev_{name}"] = prev_layer_outputs[j][name]

                if i == last_layer_idx:
                    operations.update(_make_lm_head_op(model, x_out))

                if j == 0:
                    layer_cache = build_layer_cache(model, operations, i, layer_cache, device=compute_device)

                    for short, _ in modules_items:
                        if short in {"q_proj", "k_proj"}:
                            continue

                        build_shortest_path_cache(
                            operations=operations,
                            layer_cache=layer_cache,
                            short_name=short,
                            sp_cache=sp_cache,
                            device=compute_device,
                        )

                    if prev_layer_outputs is not None and prev_layer_outputs[j] is not None and has_down_proj_target:
                        build_shortest_path_cache(
                            operations=operations,
                            layer_cache=layer_cache,
                            short_name="prev_down_proj",
                            sp_cache=sp_cache,
                            device=compute_device,
                        )

                    if i == last_layer_idx:
                        build_shortest_path_cache(
                            operations=operations,
                            layer_cache=layer_cache,
                            short_name="lm_head",
                            sp_cache=sp_cache,
                            device=compute_device,
                        )
                        layer_cache.pop("lm_head__dist", None)
                        torch.cuda.empty_cache()

                for short, module in modules_items:
                    if short == "down_proj" and i != last_layer_idx:
                        continue

                    print(f"Layer {i}, op name = {short}")

                    curv = compute_op_curvature(
                        operations=operations,
                        short_name=short,
                        layer_id=i,
                        sample_idx=j,
                        layer_cache=layer_cache,
                        sp_cache=sp_cache,
                        device=compute_device,
                        alpha=args.alpha,
                        seq_len=model.seqlen,
                        num_q_heads=num_q_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        repeat=repeat,
                        sample_edge_num=args.sample_edge_num,
                        sample_edge_ratio=args.sample_edge_ratio,
                        dataset_name=args.calib_data,
                        l2_norm=args.l2_norm,
                    )

                    assert curv is not None, f"{short} curv is None"

                    param_curv = curv
                    assert (
                        param_curv.shape == module.weight.shape
                        or param_curv.T.shape == module.weight.shape
                    ), f"Unexpected curvature shape for {short}: {tuple(param_curv.shape)} vs {tuple(module.weight.shape)}"

                    if param_curv.T.shape == module.weight.shape:
                        param_curv = param_curv.T

                    torch.minimum(module.min_curvature, param_curv, out=module.min_curvature)

                if prev_layer_outputs is not None and prev_layer_outputs[j] is not None and has_down_proj_target:
                    prev_i = i - 1
                    curv = compute_op_curvature(
                        operations=operations,
                        short_name="prev_down_proj",
                        layer_id=i,
                        sample_idx=j,
                        layer_cache=layer_cache,
                        sp_cache=sp_cache,
                        device=compute_device,
                        alpha=args.alpha,
                        seq_len=model.seqlen,
                        num_q_heads=num_q_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        repeat=repeat,
                        sample_edge_num=args.sample_edge_num,
                        sample_edge_ratio=args.sample_edge_ratio,
                        dataset_name=args.calib_data,
                        l2_norm=args.l2_norm,
                    )

                    assert curv is not None, "prev_down_proj curv is None"

                    if prev_i >= 0:
                        prev_weight = model.model.layers[prev_i].mlp.down_proj.weight.detach().cpu()
                        param_curv = curv
                        assert (
                            param_curv.shape == prev_weight.shape
                            or param_curv.T.shape == prev_weight.shape
                        ), f"Unexpected curvature shape for prev down_proj: {tuple(param_curv.shape)} vs {tuple(prev_weight.shape)}"

                        if param_curv.T.shape == prev_weight.shape:
                            param_curv = param_curv.T

                        if "down_proj" not in model.curvature_scores[prev_i]:
                            model.curvature_scores[prev_i]["down_proj"] = param_curv
                        else:
                            torch.minimum(
                                model.curvature_scores[prev_i]["down_proj"],
                                param_curv,
                                out=model.curvature_scores[prev_i]["down_proj"],
                            )

                        save_path = save_layer_curvature_pkl(
                            layer_idx=prev_i,
                            curvature_scores=model.curvature_scores[prev_i],
                            save_dir=curvature_pkl_save_dir,
                            metadata=curvature_metadata,
                        )
                        if save_path is not None:
                            print(f"Saved curvature pkl: {save_path}")
                            
                        analysis_utils.append_final_curvature_overall(
                            layer_id=prev_i,
                            short_name="down_proj",
                            curvature=model.curvature_scores[prev_i]["down_proj"],
                            seq_len=model.seqlen,
                            dataset_name=args.calib_data,
                        )

                if (i == last_layer_idx) and ("lm_head" in target_ops):
                    lm_curv = compute_op_curvature(
                        operations=operations,
                        short_name="lm_head",
                        layer_id=i,
                        sample_idx=j,
                        layer_cache=layer_cache,
                        sp_cache=sp_cache,
                        device=compute_device,
                        alpha=args.alpha,
                        seq_len=model.seqlen,
                        num_q_heads=num_q_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        repeat=repeat,
                        sample_edge_num=args.sample_edge_num,
                        sample_edge_ratio=args.sample_edge_ratio,
                        dataset_name=args.calib_data,
                        l2_norm=args.l2_norm,
                    )
                    assert lm_curv is not None, "lm_head curv is None"

                    if lm_curv.T.shape == model.lm_head.weight.shape:
                        lm_curv = lm_curv.T
                    model.curvature_scores[i]["lm_head"] = lm_curv

                if prev_layer_outputs is not None:
                    prev_layer_outputs[j] = {
                        name: operations[name]
                        for name in ["o_proj", "gate_up_out", "down_proj"]
                        if name in operations
                    }

                new_inps[j] = x_out.squeeze(0)

                del operations, x, x_out

                if torch.cuda.is_available():
                    torch.cuda.synchronize(model_device)
                    torch.cuda.synchronize(compute_device)
                sample_elapsed_sec = time.perf_counter() - sample_start_time
                _append_curvature_timing(
                    curvature_timing_log_path,
                    layer_idx=i,
                    sample_idx=j,
                    elapsed_sec=sample_elapsed_sec,
                    nsamples=args.nsamples,
                    seq_len=model.seqlen,
                    l2_norm=args.l2_norm,
                )
                print(
                    f"Curvature example time layer={i} sample={j}: "
                    f"{sample_elapsed_sec:.6f}s"
                )

                if j % 8 == 0:
                    torch.cuda.empty_cache()

            inps = new_inps

            for short, module in modules_items:
                model.curvature_scores[i][short] = module.min_curvature
                analysis_utils.append_final_curvature_overall(
                    layer_id=i,
                    short_name=short,
                    curvature=module.min_curvature,
                    seq_len=model.seqlen,
                    dataset_name=args.calib_data,
                )
                del module.min_curvature

            save_path = save_layer_curvature_pkl(
                layer_idx=i,
                curvature_scores=model.curvature_scores[i],
                save_dir=curvature_pkl_save_dir,
                metadata=curvature_metadata,
            )
            if save_path is not None:
                print(f"Saved curvature pkl: {save_path}")

            if i % 2 == 0:
                torch.cuda.empty_cache()
    finally:
        model.config.use_cache = use_cache
        if 'inps' in locals():
            del inps
        if 'prev_layer_outputs' in locals():
            del prev_layer_outputs
        if 'layer_cache' in locals():
            del layer_cache
        if 'sp_cache' in locals():
            del sp_cache
        torch.cuda.empty_cache()
