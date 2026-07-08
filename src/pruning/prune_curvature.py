import os
import time

import torch

import curvature_utils.curv_analysis_utils as analysis_utils
from curvature_utils.curv_dtype_utils import curvature_torch_dtype, set_curvature_dtype
from curvature_utils.cal_curvature import compute_op_curvature
from curvature_utils.curv_shortest_path_utils import build_shortest_path_cache
from curvature_utils.curv_tensor_utils import build_layer_cache
from data_loaders.data_c4 import get_loaders_c4
from curvature_utils.graph_relation import GRAPH
from curvature_utils.layerwrapper_curv import collect_layer_data
from pruning.prune import (
    align_curvature_to_weight_shape,
    find_layers,
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


def _sync_cuda_device(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _required_layer_cache_names(target_ops, operations, include_prev_down=False):
    required = set()
    for short in target_ops:
        rel = GRAPH.get(short, {})
        required.add(short)
        required.update(rel.get("prev", []))
        required.update(rel.get("residual", []))
        if short not in {"q_proj", "k_proj"}:
            required.update(rel.get("next", []))
    if include_prev_down:
        required.add("prev_down_proj")
        required.update(GRAPH.get("prev_down_proj", {}).get("prev", []))
        required.update(GRAPH.get("prev_down_proj", {}).get("next", []))
    return {name for name in required if name in operations}


def _curvature_save_metadata(args):
    return {
        "model": getattr(args, "model", None),
        "seed": getattr(args, "seed", None),
        "nsamples": getattr(args, "nsamples", None),
        "alpha": getattr(args, "alpha", None),
        "l2_norm": getattr(args, "l2_norm", False),
        "l2_norm_mode": getattr(args, "l2_norm_mode", "per_example"),
        "prune_method": getattr(args, "prune_method", None),
        "shared_top_k": getattr(args, "shared_top_k", 10),
        "shared_seq_select": getattr(args, "shared_seq_select", "top"),
        "curvature_lpf_window": getattr(args, "curvature_lpf_window", 0),
        "curvature_dtype": getattr(args, "curvature_dtype", "float32"),
        "curvature_layout": "weight_out_in",
    }


def _curvature_pkl_dir(base_dir, shared_top_k=None, shared_seq_select="top", curvature_lpf_window=0):
    if base_dir is None:
        return None
    if shared_top_k is None:
        return os.path.join(base_dir, "curvature_pkl")
    if shared_seq_select == "top" and int(curvature_lpf_window) <= 1:
        return os.path.join(base_dir, f"curv_topseq_{int(shared_top_k)}_pkl")
    tag = f"curv_{shared_seq_select}_seq_{int(shared_top_k)}"
    if int(curvature_lpf_window) > 1:
        tag += f"_lpf_{int(curvature_lpf_window)}"
    return os.path.join(base_dir, f"{tag}_pkl")


def _parameter_metric_log_root(
    base_dir,
    seq_len,
    dataset_name,
    shared_top_k=None,
    shared_seq_select="top",
    curvature_lpf_window=0,
):
    if base_dir is None:
        return None
    log_root = os.path.join(
        base_dir,
        "parameter_metric_logs",
        f"seq_len_{int(seq_len)}",
        str(dataset_name),
        _curvature_pkl_dir(
            "",
            shared_top_k=shared_top_k,
            shared_seq_select=shared_seq_select,
            curvature_lpf_window=curvature_lpf_window,
        ).lstrip(os.sep),
    )
    os.makedirs(log_root, exist_ok=True)
    return log_root


def prune_curvature(args, model, tokenizer, device="cuda:0", prune_n=0, prune_m=0):
    set_curvature_dtype(getattr(args, "curvature_dtype", "float32"))
    curv_torch_dtype = curvature_torch_dtype()
    print(f"Using curvature_dtype={getattr(args, 'curvature_dtype', 'float32')}")

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
    for layer in layers:
        layer.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_ops = list(getattr(args, "prune_ops", None) or ["gate_proj", "up_proj"])
    last_layer_idx = len(layers) - 1
    layer_start = 0
    layer_end = min(last_layer_idx, last_layer_idx)

    model.curvature_scores = [{} for _ in range(len(layers))]
    model.curvature_magnitude_fallbacks = [{} for _ in range(len(layers))]
    save_lpf_curvature = (
        int(getattr(args, "shared_top_k", 10)) == -1
        and int(getattr(args, "curvature_lpf_window", 0)) > 1
    )
    if save_lpf_curvature:
        model.lpf_curvature_scores = [{} for _ in range(len(layers))]
    curvature_save_dir = getattr(args, "save_curvature_dir", None)
    save_curvature_pkls = not (
        bool(getattr(args, "save_parameter_metric_logs", False))
        or bool(getattr(args, "draw_parameter_metric_plots", False))
    )
    curvature_pkl_save_dir = _curvature_pkl_dir(
        curvature_save_dir,
        getattr(args, "shared_top_k", 10),
        getattr(args, "shared_seq_select", "top"),
        0,
    )
    curvature_lpf_pkl_save_dir = None
    if save_lpf_curvature:
        curvature_lpf_pkl_save_dir = _curvature_pkl_dir(
            curvature_save_dir,
            getattr(args, "shared_top_k", 10),
            getattr(args, "shared_seq_select", "top"),
            getattr(args, "curvature_lpf_window", 0),
        )
    curvature_analysis_dir = curvature_pkl_save_dir
    gate_plot_dir = os.path.join(curvature_save_dir or os.path.dirname(__file__), "plots")
    prev_gate_plot_enabled = os.environ.get("CURV_GATE_PLOT_ENABLED")
    prev_gate_plot_dir = os.environ.get("CURV_GATE_PLOT_DIR")
    prev_gate_plot_tag = os.environ.get("CURV_GATE_PLOT_TAG")
    os.environ["CURV_GATE_PLOT_ENABLED"] = os.environ.get("CURV_GATE_PLOT_ENABLED", "0")
    os.environ["CURV_GATE_PLOT_DIR"] = gate_plot_dir
    collect_layer_data_only = os.environ.get("CURV_COLLECT_LAYER_DATA_ONLY") == "1"
    if os.environ.get("CURV_GATE_PLOT_ENABLED") == "1" and os.environ.get("CURV_GATE_PLOT_DIR"):
        print(f"Saving gate distribution plots to {gate_plot_dir}")
    if collect_layer_data_only:
        print("Collecting layer data only; skipping curvature calculation.")
    curvature_timing_log_path = getattr(args, "curvature_timing_log_path", None)
    curvature_metadata = _curvature_save_metadata(args)
    raw_curvature_metadata = dict(curvature_metadata)
    raw_curvature_metadata["curvature_lpf_window"] = 0
    parameter_metric_log_root = None
    if getattr(args, "save_parameter_metric_logs", False) or getattr(args, "draw_parameter_metric_plots", False):
        parameter_metric_log_root = _parameter_metric_log_root(
            curvature_save_dir or os.path.dirname(__file__),
            model.seqlen,
            args.calib_data,
            shared_top_k=getattr(args, "shared_top_k", 10),
            shared_seq_select=getattr(args, "shared_seq_select", "top"),
            curvature_lpf_window=getattr(args, "curvature_lpf_window", 0),
        )

    _append_curvature_timing_header(curvature_timing_log_path)

    uses_prev_layer_context = any(short in {"q_proj", "k_proj", "v_proj", "down_proj"} for short in target_ops)
    prev_layer_outputs = [None] * args.nsamples if uses_prev_layer_context else None

    try:
        for i, layer in enumerate(layers):
            if i > layer_end:
                break
            print(f"Processing layer {i}")

            layer = layer.to(model_device)

            if i < layer_start:
                print(f"Advancing layer {i} without curvature")
                next_inps = torch.empty_like(inps, device="cpu")
                next_prev_layer_outputs = [None] * args.nsamples if prev_layer_outputs is not None else None
                for j in range(args.nsamples):
                    x = inps[j:j + 1].to(model_device, non_blocking=True)
                    os.environ["CURV_GATE_PLOT_TAG"] = f"layer_{i:03d}/sample_{j:03d}"
                    with torch.no_grad():
                        x_out, operations, _, _, _, _ = collect_layer_data(
                            layer,
                            x,
                            attention_mask,
                            position_ids,
                            model,
                            next_layer=layers[i + 1] if i < last_layer_idx else None,
                            operation_dtype=curv_torch_dtype,
                        )
                    x_out = x_out.detach().cpu()
                    if next_prev_layer_outputs is not None:
                        next_prev_layer_outputs[j] = {
                            name: operations[name]
                            for name in ["o_proj", "gate_up_out", "down_proj", "qkv_residual"]
                            if name in operations
                        }
                    next_inps[j].copy_(x_out.squeeze(0))
                    del operations, x, x_out
                    if j % 8 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                inps = next_inps
                prev_layer_outputs = next_prev_layer_outputs
                layer.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            layer_subset = find_layers(layer)
            layer_target_ops = list(target_ops)

            op_modules = {
                short: next(m for n, m in layer_subset.items() if n.endswith(short))
                for short in layer_target_ops
            }
            modules_items = list(op_modules.items())
            has_down_proj_target = "down_proj" in op_modules

            for short, module in modules_items:
                if collect_layer_data_only:
                    continue
                W = module.weight
                module.min_curvature = torch.full(
                    W.shape,
                    float("inf"),
                    dtype=curv_torch_dtype,
                    device="cpu",
                )
                if save_lpf_curvature:
                    module.min_lpf_curvature = torch.full(
                        W.shape,
                        float("inf"),
                        dtype=curv_torch_dtype,
                        device="cpu",
                    )

            for j in range(args.nsamples):
                _sync_cuda_device(model_device)
                _sync_cuda_device(compute_device)
                sample_start_time = time.perf_counter()

                x = inps[j:j + 1].to(model_device, non_blocking=True)
                prev_outputs = prev_layer_outputs[j] if prev_layer_outputs is not None else None
                next_layer = layers[i + 1] if i < last_layer_idx else None

                os.environ["CURV_GATE_PLOT_TAG"] = f"layer_{i:03d}/sample_{j:03d}"
                with torch.no_grad():
                    x_out, operations, num_q_heads, num_kv_heads, repeat, head_dim = collect_layer_data(
                        layer,
                        x,
                        attention_mask,
                        position_ids,
                        model,
                        next_layer=next_layer,
                        operation_dtype=curv_torch_dtype,
                    )

                x_out = x_out.detach().cpu()

                print("Finish getting layer data!!!")

                # prev_* tensors come from layer i-1 and are used as context for layer i curvature_utils.
                if prev_outputs is not None:
                    for name in ["o_proj", "gate_up_out", "down_proj", "qkv_residual"]:
                        if name in prev_outputs:
                            operations[f"prev_{name}"] = prev_outputs[name]

                if prev_layer_outputs is not None:
                    prev_layer_outputs[j] = {
                        name: operations[name]
                        for name in ["o_proj", "gate_up_out", "down_proj", "qkv_residual"]
                        if name in operations
                    }

                inps[j].copy_(x_out.squeeze(0))
                del x_out
                del prev_outputs
                if collect_layer_data_only:
                    del operations, x
                    if j % 8 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                layer_cache = {}
                sp_cache = {}
                required_cache_names = _required_layer_cache_names(
                    target_ops,
                    operations,
                    include_prev_down=("prev_down_proj" in operations and has_down_proj_target),
                )
                print(f"Building layer cache for: {sorted(required_cache_names)}")
                
                layer_cache = build_layer_cache(
                    model,
                    operations,
                    i,
                    layer_cache,
                    device=compute_device,
                    required_names=required_cache_names,
                    l2_norm=args.l2_norm,
                    l2_norm_mode=getattr(args, "l2_norm_mode", "per_example"),
                )

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

                if "prev_down_proj" in operations and has_down_proj_target:
                    build_shortest_path_cache(
                        operations=operations,
                        layer_cache=layer_cache,
                        short_name="prev_down_proj",
                        sp_cache=sp_cache,
                        device=compute_device,
                    )

                for short, module in modules_items:
                    if short == "down_proj" and i != last_layer_idx:
                        continue

                    print(f"Layer {i}, op name = {short}")

                    curv_result = compute_op_curvature(
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
                        l2_norm_mode=getattr(args, "l2_norm_mode", "per_example"),
                        l2_norm_stats=None,
                        shared_top_k=getattr(args, "shared_top_k", 10),
                        shared_seq_select=getattr(args, "shared_seq_select", "top"),
                        curvature_lpf_window=getattr(args, "curvature_lpf_window", 0),
                        analysis_dir=curvature_analysis_dir,
                        parameter_log_root=parameter_metric_log_root,
                    )

                    curv = curv_result["curvature"] if isinstance(curv_result, dict) else curv_result
                    lpf_curv = curv_result.get("lpf_curvature") if isinstance(curv_result, dict) else None
                    magnitude_fallback = bool(
                        curv_result.get("magnitude_fallback", False)
                        if isinstance(curv_result, dict)
                        else False
                    )
                    assert curv is not None, f"{short} curv is None"

                    param_curv = align_curvature_to_weight_shape(
                        curv,
                        module.weight.shape,
                        context=f"layer {i} {short} curvature",
                    )

                    torch.minimum(module.min_curvature, param_curv, out=module.min_curvature)
                    if magnitude_fallback:
                        model.curvature_magnitude_fallbacks[i][short] = True

                    if save_lpf_curvature and lpf_curv is not None:
                        param_lpf_curv = align_curvature_to_weight_shape(
                            lpf_curv,
                            module.weight.shape,
                            context=f"layer {i} {short} lpf_curvature",
                        )
                        torch.minimum(module.min_lpf_curvature, param_lpf_curv, out=module.min_lpf_curvature)

                if "prev_down_proj" in operations and has_down_proj_target:
                    prev_i = i - 1
                    curv_result = compute_op_curvature(
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
                        l2_norm_mode=getattr(args, "l2_norm_mode", "per_example"),
                        l2_norm_stats=None,
                        shared_top_k=getattr(args, "shared_top_k", 10),
                        shared_seq_select=getattr(args, "shared_seq_select", "top"),
                        curvature_lpf_window=getattr(args, "curvature_lpf_window", 0),
                        analysis_dir=curvature_analysis_dir,
                        parameter_log_root=parameter_metric_log_root,
                    )

                    curv = curv_result["curvature"] if isinstance(curv_result, dict) else curv_result
                    lpf_curv = curv_result.get("lpf_curvature") if isinstance(curv_result, dict) else None
                    magnitude_fallback = bool(
                        curv_result.get("magnitude_fallback", False)
                        if isinstance(curv_result, dict)
                        else False
                    )
                    assert curv is not None, "prev_down_proj curv is None"

                    if prev_i >= 0:
                        prev_weight = model.model.layers[prev_i].mlp.down_proj.weight.detach().cpu()
                        param_curv = align_curvature_to_weight_shape(
                            curv,
                            prev_weight.shape,
                            context=f"layer {prev_i} down_proj curvature",
                        )

                        if "down_proj" not in model.curvature_scores[prev_i]:
                            model.curvature_scores[prev_i]["down_proj"] = param_curv
                        else:
                            torch.minimum(
                                model.curvature_scores[prev_i]["down_proj"],
                                param_curv,
                                out=model.curvature_scores[prev_i]["down_proj"],
                            )
                        if magnitude_fallback:
                            model.curvature_magnitude_fallbacks[prev_i]["down_proj"] = True

                        if save_lpf_curvature and lpf_curv is not None:
                            param_lpf_curv = align_curvature_to_weight_shape(
                                lpf_curv,
                                prev_weight.shape,
                                context=f"layer {prev_i} down_proj lpf_curvature",
                            )
                            if "down_proj" not in model.lpf_curvature_scores[prev_i]:
                                model.lpf_curvature_scores[prev_i]["down_proj"] = param_lpf_curv
                            else:
                                torch.minimum(
                                    model.lpf_curvature_scores[prev_i]["down_proj"],
                                    param_lpf_curv,
                                    out=model.lpf_curvature_scores[prev_i]["down_proj"],
                                )

                        if save_curvature_pkls:
                            save_path = save_layer_curvature_pkl(
                                layer_idx=prev_i,
                                curvature_scores=model.curvature_scores[prev_i],
                                save_dir=curvature_pkl_save_dir,
                                metadata=raw_curvature_metadata,
                            )
                            if save_path is not None:
                                print(f"Saved curvature pkl: {save_path}")
                            if save_lpf_curvature:
                                save_path = save_layer_curvature_pkl(
                                    layer_idx=prev_i,
                                    curvature_scores=model.lpf_curvature_scores[prev_i],
                                    save_dir=curvature_lpf_pkl_save_dir,
                                    metadata=curvature_metadata,
                                )
                                if save_path is not None:
                                    print(f"Saved LPF curvature pkl: {save_path}")

                        analysis_utils.append_final_curvature_overall(
                            layer_id=prev_i,
                            short_name="down_proj",
                            curvature=model.curvature_scores[prev_i]["down_proj"],
                            seq_len=model.seqlen,
                            dataset_name=args.calib_data,
                            analysis_dir=curvature_analysis_dir,
                        )
                        if save_lpf_curvature:
                            analysis_utils.append_final_curvature_overall(
                                layer_id=prev_i,
                                short_name="down_proj",
                                curvature=model.lpf_curvature_scores[prev_i]["down_proj"],
                                seq_len=model.seqlen,
                                dataset_name=args.calib_data,
                                analysis_dir=curvature_lpf_pkl_save_dir,
                            )

                del operations, x
                del layer_cache, sp_cache

                _sync_cuda_device(model_device)
                _sync_cuda_device(compute_device)
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

            if collect_layer_data_only:
                del layer_subset, op_modules, modules_items
                layer.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            for short, module in modules_items:
                if short == "down_proj" and i != last_layer_idx:
                    del module.min_curvature
                    if save_lpf_curvature:
                        del module.min_lpf_curvature
                    continue
                model.curvature_scores[i][short] = module.min_curvature
                if save_lpf_curvature:
                    model.lpf_curvature_scores[i][short] = module.min_lpf_curvature
                analysis_utils.append_final_curvature_overall(
                    layer_id=i,
                    short_name=short,
                    curvature=module.min_curvature,
                    seq_len=model.seqlen,
                    dataset_name=args.calib_data,
                    analysis_dir=curvature_analysis_dir,
                )
                if save_lpf_curvature:
                    analysis_utils.append_final_curvature_overall(
                        layer_id=i,
                        short_name=short,
                        curvature=module.min_lpf_curvature,
                        seq_len=model.seqlen,
                        dataset_name=args.calib_data,
                        analysis_dir=curvature_lpf_pkl_save_dir,
                    )
                del module.min_curvature
                if save_lpf_curvature:
                    del module.min_lpf_curvature

            if save_curvature_pkls:
                save_path = save_layer_curvature_pkl(
                    layer_idx=i,
                    curvature_scores=model.curvature_scores[i],
                    save_dir=curvature_pkl_save_dir,
                    metadata=raw_curvature_metadata,
                )
                if save_path is not None:
                    print(f"Saved curvature pkl: {save_path}")
                if save_lpf_curvature:
                    save_path = save_layer_curvature_pkl(
                        layer_idx=i,
                        curvature_scores=model.lpf_curvature_scores[i],
                        save_dir=curvature_lpf_pkl_save_dir,
                        metadata=curvature_metadata,
                    )
                    if save_path is not None:
                        print(f"Saved LPF curvature pkl: {save_path}")

            del layer_subset, op_modules, modules_items
            layer.to("cpu")
            if i % 2 == 0:
                torch.cuda.empty_cache()
    finally:
        if prev_gate_plot_enabled is None:
            os.environ.pop("CURV_GATE_PLOT_ENABLED", None)
        else:
            os.environ["CURV_GATE_PLOT_ENABLED"] = prev_gate_plot_enabled
        if prev_gate_plot_dir is None:
            os.environ.pop("CURV_GATE_PLOT_DIR", None)
        else:
            os.environ["CURV_GATE_PLOT_DIR"] = prev_gate_plot_dir
        if prev_gate_plot_tag is None:
            os.environ.pop("CURV_GATE_PLOT_TAG", None)
        else:
            os.environ["CURV_GATE_PLOT_TAG"] = prev_gate_plot_tag
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
