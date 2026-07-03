import torch

from utils.cuda_memory_utils import release_cuda_memory
from data_loaders.data_c4 import get_loaders_c4
from pruning.layerwrapper import WrappedGPT
from pruning.prune import (
    find_layers,
    prepare_calibration_input,
    _curvature_candidate_mask,
    prune_scope_from_args,
    score_order_largest,
    select_prune_masks_by_score,
    should_prune_op,
    skip_prune_layer,
)
from pruning.prune_log_utils import (
    append_all_layer_pruned_parameter_log,
    append_layer_pruned_parameter_log,
    collect_pruned_parameter_rows,
)


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def compute_wanda_scores(args, model, tokenizer, device=torch.device("cuda:0")):
    print(f"precomputing WANDA scores with seqlen={model.seqlen}")
    input_scalers = compute_wanda_input_scalers(args, model, tokenizer, device)
    layers = model.model.layers
    model.wanda_scores = [{} for _ in range(len(layers))]
    for i, layer_scalers in enumerate(input_scalers):
        subset = find_layers(layers[i])
        for name, scaler in layer_scalers.items():
            if name not in subset:
                continue
            print(f"collecting WANDA scores layer {i} name {name}")
            model.wanda_scores[i][name] = _wanda_metric_from_scaler(subset[name], scaler)
    if hasattr(model, "wanda_input_scalers"):
        del model.wanda_input_scalers
    release_cuda_memory()
    return model.wanda_scores


def compute_wanda_input_scalers(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print(f"loading calibration data for WANDA input scalers with seqlen={model.seqlen}")
    dataloader, _ = get_loaders_c4(
        args.calib_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )
    del dataloader

    layers = model.model.layers
    model.wanda_input_scalers = [{} for _ in range(len(layers))]

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        subset = {
            name: module
            for name, module in subset.items()
            if should_prune_op(args, name)
        }

        if hasattr(model, "hf_device_map") and (f"model.layers.{i}" in model.hf_device_map):
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                cos, sin = model.model.rotary_emb(inps[j].unsqueeze(0), position_ids)
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=(cos, sin),
                )[0]
                del cos, sin

        for h in handles:
            h.remove()

        for name in subset:
            print(f"collecting WANDA input scaler layer {i} name {name}")
            model.wanda_input_scalers[i][name] = wrapped_layers[name].scaler_row.detach().cpu()

        inps, outs = outs, inps
        del wrapped_layers, handles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    del inps, outs, attention_mask, position_ids
    release_cuda_memory()
    return model.wanda_input_scalers


def _wanda_metric_from_scaler(module, scaler):
    weight_cpu = module.weight.detach().cpu()
    scaler_cpu = scaler.detach().cpu()
    return torch.abs(weight_cpu) * torch.sqrt(scaler_cpu.reshape((1, -1)))


def _apply_wanda_input_scalers(args, model, wanda_input_scalers, prune_n=0, prune_m=0):
    layers = model.model.layers
    report_limit = int(getattr(args, "all_layer_report_rank_offset", 0)) + 25
    report_rows = []
    scope = prune_scope_from_args(args)
    largest = score_order_largest(args, default=False)

    if scope == "global":
        raise ValueError("WANDA input scalers are only for local/per-op pruning")

    for i in range(len(layers)):
        if skip_prune_layer(args, i):
            print(f"Skipping layer {i}: requested by skip_prune_layer_ids")
            continue
        layer = layers[i]
        subset = find_layers(layer)
        layer_report_rows = []
        layer_scalers = wanda_input_scalers[i] if i < len(wanda_input_scalers) else {}

        if prune_n == 0 and not args.use_variant and scope == "per_layer":
            layer_entries = []
            for name, module in subset.items():
                if not should_prune_op(args, name):
                    continue
                print(f"pruning layer {i} name {name}")
                if name not in layer_scalers:
                    raise KeyError(f"Missing precomputed WANDA input scaler for layer {i} name {name}")

                W = module.weight.data
                W_metric = _wanda_metric_from_scaler(module, layer_scalers[name])
                if W_metric.shape != W.shape:
                    raise ValueError(
                        f"WANDA score shape mismatch for layer {i} {name}: "
                        f"{tuple(W_metric.shape)} vs {tuple(W.shape)}"
                    )
                candidate_mask = _curvature_candidate_mask(args, model, i, name, W)
                if candidate_mask is not None:
                    candidate_mask = candidate_mask.cpu()
                layer_entries.append(
                    {
                        "name": name,
                        "module": module,
                        "metric": W_metric,
                        "candidate_mask": candidate_mask,
                    }
                )

            prune_masks, _ = select_prune_masks_by_score(
                layer_entries,
                args.sparsity_ratio,
                largest=largest,
            )
            for entry, W_mask in zip(layer_entries, prune_masks):
                name = entry["name"]
                module = entry["module"]
                report_rows.extend(
                    collect_pruned_parameter_rows(
                        i,
                        name,
                        module,
                        entry["metric"],
                        W_mask,
                        largest=largest,
                        limit=report_limit,
                        include_input_scale=True,
                    )
                )
                layer_report_rows.extend(
                    collect_pruned_parameter_rows(
                        i,
                        name,
                        module,
                        entry["metric"],
                        W_mask,
                        largest=largest,
                        limit=25,
                        include_input_scale=True,
                    )
                )
                module.weight.data[W_mask.to(device=module.weight.data.device)] = 0

            append_layer_pruned_parameter_log(
                getattr(args, "all_layer_parameter_log_path", None),
                args,
                "wanda",
                i,
                getattr(args, "prune_score_order", "low_to_high"),
                "wanda",
                layer_report_rows,
                largest=largest,
            )
            del layer_entries, prune_masks
            continue

        for name, module in subset.items():
            if not should_prune_op(args, name):
                continue
            print(f"pruning layer {i} name {name}")
            if name not in layer_scalers:
                raise KeyError(f"Missing precomputed WANDA input scaler for layer {i} name {name}")

            W = module.weight.data
            W_metric = _wanda_metric_from_scaler(module, layer_scalers[name])
            if W_metric.shape != W.shape:
                raise ValueError(
                    f"WANDA score shape mismatch for layer {i} {name}: "
                    f"{tuple(W_metric.shape)} vs {tuple(W.shape)}"
                )

            candidate_mask = _curvature_candidate_mask(args, model, i, name, W)
            if candidate_mask is not None:
                candidate_mask = candidate_mask.cpu()

            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        group_candidate = (
                            None if candidate_mask is None else candidate_mask[:, ii:(ii + prune_m)]
                        )
                        if group_candidate is not None:
                            tmp = tmp.masked_fill(~group_candidate, float("inf"))
                        selected = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                        if group_candidate is not None:
                            selected_mask = torch.gather(group_candidate, 1, selected)
                            selected = selected.masked_fill(~selected_mask, 0)
                        W_mask.scatter_(1, ii + selected, True)
                if candidate_mask is not None:
                    W_mask &= candidate_mask
            else:
                if candidate_mask is not None and args.use_variant:
                    print("WANDA variant with loaded curvature uses finite-curvature candidates only")
                    W_metric = W_metric.masked_fill(~candidate_mask, float("inf"))

                if args.use_variant:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    W_mask = select_prune_masks_by_score(
                        [
                            {
                                "metric": W_metric,
                                "candidate_mask": candidate_mask,
                            }
                        ],
                        args.sparsity_ratio,
                        largest=largest,
                    )[0][0]

            report_rows.extend(
                collect_pruned_parameter_rows(
                    i,
                    name,
                    module,
                    W_metric,
                    W_mask,
                    largest=largest,
                    limit=report_limit,
                    include_input_scale=True,
                )
            )
            layer_report_rows.extend(
                collect_pruned_parameter_rows(
                    i,
                    name,
                    module,
                    W_metric,
                    W_mask,
                    largest=largest,
                    limit=25,
                    include_input_scale=True,
                )
            )
            W[W_mask.to(device=W.device)] = 0
            del W_metric, W_mask

        append_layer_pruned_parameter_log(
            getattr(args, "all_layer_parameter_log_path", None),
            args,
            "wanda",
            i,
            getattr(args, "prune_score_order", "low_to_high"),
            "wanda",
            layer_report_rows,
            largest=largest,
        )

    append_all_layer_pruned_parameter_log(
        getattr(args, "all_layer_parameter_log_path", None),
        args,
        "wanda",
        getattr(args, "prune_score_order", "low_to_high"),
        "wanda",
        report_rows,
        largest=largest,
        rank_offset=getattr(args, "all_layer_report_rank_offset", 0),
    )


def _apply_wanda_scores(args, model, wanda_scores, prune_n=0, prune_m=0):
    layers = model.model.layers
    report_limit = int(getattr(args, "all_layer_report_rank_offset", 0)) + 25
    report_rows = []
    scope = prune_scope_from_args(args)

    if prune_n == 0 and not args.use_variant and scope == "global":
        entries = []
        for i, layer in enumerate(layers):
            if skip_prune_layer(args, i):
                print(f"Skipping layer {i}: requested by skip_prune_layer_ids")
                continue
            subset = find_layers(layer)
            layer_scores = wanda_scores[i] if i < len(wanda_scores) else {}
            for name, module in subset.items():
                if not should_prune_op(args, name):
                    continue
                print(f"pruning layer {i} name {name}")
                if name not in layer_scores:
                    raise KeyError(f"Missing precomputed WANDA scores for layer {i} name {name}")

                W = module.weight.data
                W_metric = layer_scores[name].detach().cpu()
                if W_metric.shape != W.shape:
                    raise ValueError(
                        f"WANDA score shape mismatch for layer {i} {name}: "
                        f"{tuple(W_metric.shape)} vs {tuple(W.shape)}"
                    )

                candidate_mask = _curvature_candidate_mask(args, model, i, name, W)
                if candidate_mask is not None:
                    candidate_mask = candidate_mask.cpu()
                entries.append(
                    {
                        "layer_idx": i,
                        "name": name,
                        "module": module,
                        "metric": W_metric,
                        "candidate_mask": candidate_mask,
                    }
                )

        largest = score_order_largest(args, default=False)
        prune_masks, _ = select_prune_masks_by_score(entries, args.sparsity_ratio, largest=largest)
        layer_report_rows_by_idx = {}
        for entry, W_mask in zip(entries, prune_masks):
            layer_idx = entry["layer_idx"]
            name = entry["name"]
            module = entry["module"]
            report_rows.extend(
                collect_pruned_parameter_rows(
                    layer_idx,
                    name,
                    module,
                    entry["metric"],
                    W_mask,
                    largest=largest,
                    limit=report_limit,
                    include_input_scale=True,
                )
            )
            layer_report_rows_by_idx.setdefault(layer_idx, []).extend(
                collect_pruned_parameter_rows(
                    layer_idx,
                    name,
                    module,
                    entry["metric"],
                    W_mask,
                    largest=largest,
                    limit=25,
                    include_input_scale=True,
                )
            )
            module.weight.data[W_mask.to(device=module.weight.data.device)] = 0

        for layer_idx, layer_report_rows in sorted(layer_report_rows_by_idx.items()):
            append_layer_pruned_parameter_log(
                getattr(args, "all_layer_parameter_log_path", None),
                args,
                "wanda",
                layer_idx,
                getattr(args, "prune_score_order", "low_to_high"),
                "wanda",
                layer_report_rows,
                largest=largest,
            )

        append_all_layer_pruned_parameter_log(
            getattr(args, "all_layer_parameter_log_path", None),
            args,
            "wanda",
            getattr(args, "prune_score_order", "low_to_high"),
            "wanda",
            report_rows,
            largest=largest,
            rank_offset=getattr(args, "all_layer_report_rank_offset", 0),
        )
        return

    for i in range(len(layers)):
        if skip_prune_layer(args, i):
            print(f"Skipping layer {i}: requested by skip_prune_layer_ids")
            continue
        layer = layers[i]
        subset = find_layers(layer)
        layer_report_rows = []

        layer_scores = wanda_scores[i] if i < len(wanda_scores) else {}

        if prune_n == 0 and not args.use_variant and scope == "per_layer":
            layer_entries = []
            for name, module in subset.items():
                if not should_prune_op(args, name):
                    continue
                print(f"pruning layer {i} name {name}")
                if name not in layer_scores:
                    raise KeyError(f"Missing precomputed WANDA scores for layer {i} name {name}")

                W = module.weight.data
                W_metric = layer_scores[name].detach().cpu()
                if W_metric.shape != W.shape:
                    raise ValueError(
                        f"WANDA score shape mismatch for layer {i} {name}: "
                        f"{tuple(W_metric.shape)} vs {tuple(W.shape)}"
                    )

                candidate_mask = _curvature_candidate_mask(args, model, i, name, W)
                if candidate_mask is not None:
                    candidate_mask = candidate_mask.cpu()
                layer_entries.append(
                    {
                        "name": name,
                        "module": module,
                        "metric": W_metric,
                        "candidate_mask": candidate_mask,
                    }
                )

            largest = score_order_largest(args, default=False)
            prune_masks, _ = select_prune_masks_by_score(
                layer_entries,
                args.sparsity_ratio,
                largest=largest,
            )
            for entry, W_mask in zip(layer_entries, prune_masks):
                name = entry["name"]
                module = entry["module"]
                report_rows.extend(
                    collect_pruned_parameter_rows(
                        i,
                        name,
                        module,
                        entry["metric"],
                        W_mask,
                        largest=largest,
                        limit=report_limit,
                        include_input_scale=True,
                    )
                )
                layer_report_rows.extend(
                    collect_pruned_parameter_rows(
                        i,
                        name,
                        module,
                        entry["metric"],
                        W_mask,
                        largest=largest,
                        limit=25,
                        include_input_scale=True,
                    )
                )
                module.weight.data[W_mask.to(device=module.weight.data.device)] = 0

            append_layer_pruned_parameter_log(
                getattr(args, "all_layer_parameter_log_path", None),
                args,
                "wanda",
                i,
                getattr(args, "prune_score_order", "low_to_high"),
                "wanda",
                layer_report_rows,
                largest=largest,
            )
            continue

        for name in subset:
            if not should_prune_op(args, name):
                continue
            print(f"pruning layer {i} name {name}")
            if name not in layer_scores:
                raise KeyError(f"Missing precomputed WANDA scores for layer {i} name {name}")

            W = subset[name].weight.data
            W_metric = layer_scores[name].detach().cpu()
            if W_metric.shape != W.shape:
                raise ValueError(
                    f"WANDA score shape mismatch for layer {i} {name}: "
                    f"{tuple(W_metric.shape)} vs {tuple(W.shape)}"
                )

            candidate_mask = _curvature_candidate_mask(args, model, i, name, W)
            if candidate_mask is not None:
                candidate_mask = candidate_mask.cpu()

            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        group_candidate = (
                            None if candidate_mask is None else candidate_mask[:, ii:(ii + prune_m)]
                        )
                        if group_candidate is not None:
                            tmp = tmp.masked_fill(~group_candidate, float("inf"))
                        selected = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                        if group_candidate is not None:
                            selected_mask = torch.gather(group_candidate, 1, selected)
                            selected = selected.masked_fill(~selected_mask, 0)
                        W_mask.scatter_(1, ii + selected, True)
                if candidate_mask is not None:
                    W_mask &= candidate_mask
            else:
                if candidate_mask is not None and args.use_variant:
                    print("WANDA variant with loaded curvature uses finite-curvature candidates only")
                    W_metric = W_metric.masked_fill(~candidate_mask, float("inf"))

                if args.use_variant:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    W_mask = select_prune_masks_by_score(
                        [
                            {
                                "metric": W_metric,
                                "candidate_mask": candidate_mask,
                            }
                        ],
                        args.sparsity_ratio,
                        largest=score_order_largest(args, default=False),
                    )[0][0]

            report_rows.extend(
                collect_pruned_parameter_rows(
                    i,
                    name,
                    subset[name],
                    W_metric,
                    W_mask,
                    largest=score_order_largest(args, default=False),
                    limit=report_limit,
                    include_input_scale=True,
                )
            )
            layer_report_rows.extend(
                collect_pruned_parameter_rows(
                    i,
                    name,
                    subset[name],
                    W_metric,
                    W_mask,
                    largest=score_order_largest(args, default=False),
                    limit=25,
                    include_input_scale=True,
                )
            )
            W[W_mask.to(device=W.device)] = 0
            del W_metric, W_mask

        append_layer_pruned_parameter_log(
            getattr(args, "all_layer_parameter_log_path", None),
            args,
            "wanda",
            i,
            getattr(args, "prune_score_order", "low_to_high"),
            "wanda",
            layer_report_rows,
            largest=score_order_largest(args, default=False),
        )

    append_all_layer_pruned_parameter_log(
        getattr(args, "all_layer_parameter_log_path", None),
        args,
        "wanda",
        getattr(args, "prune_score_order", "low_to_high"),
        "wanda",
        report_rows,
        largest=score_order_largest(args, default=False),
        rank_offset=getattr(args, "all_layer_report_rank_offset", 0),
    )


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if hasattr(model, "wanda_input_scalers"):
        _apply_wanda_input_scalers(args, model, model.wanda_input_scalers, prune_n, prune_m)
    elif not hasattr(model, "wanda_scores"):
        compute_wanda_scores(args, model, tokenizer, device)
        _apply_wanda_scores(args, model, model.wanda_scores, prune_n, prune_m)
    else:
        _apply_wanda_scores(args, model, model.wanda_scores, prune_n, prune_m)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
