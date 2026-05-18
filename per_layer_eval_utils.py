import os

import numpy as np
import torch

from curv_prune_utils import _get_prunable_module
from prune import find_layers, load_layer_curvature_pkl


def _curvature_seq_tag(shared_top_k=None, shared_seq_select="top", curvature_lpf_window=0):
    if shared_top_k is None:
        return "curvature_pkl"
    if shared_seq_select == "top" and int(curvature_lpf_window) <= 1:
        return f"curv_topseq_{int(shared_top_k)}_pkl"
    tag = f"curv_{shared_seq_select}_seq_{int(shared_top_k)}"
    if int(curvature_lpf_window) > 1:
        tag += f"_lpf_{int(curvature_lpf_window)}"
    return f"{tag}_pkl"


def resolve_curvature_pkl_dir(
    base_dir,
    shared_top_k=None,
    shared_seq_select="top",
    curvature_lpf_window=0,
):
    if base_dir is None or not os.path.isdir(base_dir):
        return None

    search_dirs = [base_dir]
    if shared_top_k is not None:
        search_dirs.append(
            os.path.join(
                base_dir,
                _curvature_seq_tag(shared_top_k, shared_seq_select, curvature_lpf_window),
            )
        )
    search_dirs.append(os.path.join(base_dir, "curvature_pkl"))

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        if any(
            file_name.startswith("layer_") and file_name.endswith("_curvature.pkl")
            for file_name in os.listdir(search_dir)
        ):
            return search_dir
    return None


def list_curvature_pkl_layers(
    base_dir,
    shared_top_k=None,
    shared_seq_select="top",
    curvature_lpf_window=0,
):
    pkl_dir = resolve_curvature_pkl_dir(
        base_dir,
        shared_top_k=shared_top_k,
        shared_seq_select=shared_seq_select,
        curvature_lpf_window=curvature_lpf_window,
    )
    if pkl_dir is None:
        return []

    layer_ids = []
    for file_name in sorted(os.listdir(pkl_dir)):
        if not (file_name.startswith("layer_") and file_name.endswith("_curvature.pkl")):
            continue
        try:
            layer_ids.append(int(file_name[len("layer_"):].split("_", 1)[0]))
        except ValueError:
            continue
    return layer_ids


def load_curvature_scores_for_layer(
    base_dir,
    layer_idx,
    shared_top_k=None,
    shared_seq_select="top",
    curvature_lpf_window=0,
):
    pkl_dir = resolve_curvature_pkl_dir(
        base_dir,
        shared_top_k=shared_top_k,
        shared_seq_select=shared_seq_select,
        curvature_lpf_window=curvature_lpf_window,
    )
    if pkl_dir is None:
        return None

    pkl_path = os.path.join(pkl_dir, f"layer_{int(layer_idx):03d}_curvature.pkl")
    if not os.path.isfile(pkl_path):
        return None

    _, layer_scores, _ = load_layer_curvature_pkl(pkl_path)
    return layer_scores


def layer_sparsity(model, layer_idx):
    subset = find_layers(model.model.layers[layer_idx])
    zero_count = 0
    total_count = 0
    for module in subset.values():
        weight = module.weight.data
        zero_count += int((weight == 0).sum().item())
        total_count += weight.numel()
    if total_count == 0:
        return 0.0
    return float(zero_count) / float(total_count)


def _score_order_is_descending(args):
    return getattr(args, "prune_score_order", "high_to_low") == "high_to_low"


def _candidate_mask_from_curvature(layer_scores, name, weight):
    if layer_scores is None:
        return None

    short_name = name.split(".")[-1]
    curv = layer_scores.get(short_name)
    if curv is None:
        return torch.zeros_like(weight, dtype=torch.bool)

    curv = curv.detach() if torch.is_tensor(curv) else torch.as_tensor(curv)
    if curv.shape != weight.shape:
        if curv.T.shape == weight.shape:
            curv = curv.T
        else:
            raise ValueError(
                f"Curvature shape mismatch for {name}: {tuple(curv.shape)} vs {tuple(weight.shape)}"
            )
    return torch.isfinite(curv).to(device=weight.device)


def _select_lowest_mask(metric, candidate_mask, ratio):
    prune_mask = torch.zeros_like(metric, dtype=torch.bool)
    eligible_count = int(candidate_mask.sum().item())
    prune_count = int(eligible_count * ratio)
    if prune_count <= 0:
        return prune_mask, None

    candidate_scores = metric[candidate_mask].float()
    prune_count = min(prune_count, candidate_scores.numel())
    selected = torch.topk(candidate_scores, k=prune_count, largest=False, sorted=False).indices
    selected_scores = candidate_scores[selected]
    cutoff = float(selected_scores.max().item()) if selected_scores.numel() > 0 else None

    flat_positions = candidate_mask.reshape(-1).nonzero(as_tuple=False).flatten()
    prune_mask.reshape(-1)[flat_positions[selected]] = True
    return prune_mask, cutoff


def _selected_score_cutoff(selected_scores, prune_high_scores):
    if selected_scores.numel() == 0:
        return None
    if prune_high_scores:
        return float(selected_scores.min().item())
    return float(selected_scores.max().item())


def _append_first_pruned_edges(log_path, args, method, layer_idx, score_order, score_name, edge_rows):
    if log_path is None or not edge_rows:
        return

    with open(log_path, "a+", encoding="utf-8") as f:
        print(
            f"first_pruned_edges method={method}, layer={layer_idx}, "
            f"target_sparsity={float(args.sparsity_ratio):.4f}, score_order={score_order}",
            file=f,
            flush=True,
        )
        print(
            f"{'rank':<6}{'op_name':<12}{'index':<16}{'weight_magnitude':<20}{score_name:<20}",
            file=f,
            flush=True,
        )
        for rank, row in enumerate(edge_rows[:25], start=1):
            print(
                f"{rank:<6}{row['op_name']:<12}"
                f"({row['i']},{row['j']})".ljust(16)
                + f"{row['weight_magnitude']:<20.8g}{row['score']:<20.8g}",
                file=f,
                flush=True,
            )
        print("", file=f, flush=True)


def _append_score_zero_summary(
    log_path,
    args,
    method,
    layer_idx,
    score_name,
    zero_count,
    eligible_count,
    rows,
):
    if log_path is None:
        return

    with open(log_path, "a+", encoding="utf-8") as f:
        print(
            f"score_zero_summary method={method}, layer={layer_idx}, "
            f"target_sparsity={float(args.sparsity_ratio):.4f}, "
            f"total_{score_name}_zero_count={zero_count}, eligible_count={eligible_count}",
            file=f,
            flush=True,
        )
        if rows:
            print(
                f"top_25_nonzero_parameters_by_low_score {score_name}",
                file=f,
                flush=True,
            )
            print(
                f"{'rank':<6}{'op_name':<12}{'index':<16}{'weight_magnitude':<20}{score_name:<20}",
                file=f,
                flush=True,
            )
            for rank, row in enumerate(rows[:25], start=1):
                print(
                    f"{rank:<6}{row['op_name']:<12}"
                    f"({row['i']},{row['j']})".ljust(16)
                    + f"{row['weight_magnitude']:<20.8g}{row['score']:<20.8g}",
                    file=f,
                    flush=True,
                )
        print("", file=f, flush=True)


def _weight_magnitude_at(module, row_idx, col_idx):
    weight = module.weight.data
    rows = torch.as_tensor([row_idx], device=weight.device)
    cols = torch.as_tensor([col_idx], device=weight.device)
    return float(weight[rows, cols].abs().detach().cpu().item())


def _top_pruned_edges_for_mask(op_name, module, metric, prune_mask, score_name, largest, limit=25):
    del score_name
    selected_flat = prune_mask.reshape(-1).nonzero(as_tuple=False).flatten()
    if selected_flat.numel() == 0:
        return []

    metric_flat = metric.reshape(-1)
    selected_scores = metric_flat[selected_flat].float()
    keep = min(limit, selected_scores.numel())
    top_order = torch.topk(selected_scores, k=keep, largest=largest, sorted=True).indices
    top_flat = selected_flat[top_order].detach().cpu()
    top_scores = selected_scores[top_order].detach().cpu()

    cols = metric.shape[1]
    rows = []
    for flat_idx, score in zip(top_flat.tolist(), top_scores.tolist()):
        row_idx = int(flat_idx // cols)
        col_idx = int(flat_idx % cols)
        rows.append(
            {
                "op_name": op_name,
                "i": row_idx,
                "j": col_idx,
                "weight_magnitude": _weight_magnitude_at(module, row_idx, col_idx),
                "score": float(score),
            }
        )
    return rows


def _nonzero_score_rows(op_name, module, metric, candidate_mask=None, limit=25):
    if candidate_mask is None:
        zero_count = int((metric == 0).sum().item())
        eligible_count = metric.numel()
        nonzero_mask = (metric != 0) & torch.isfinite(metric)
    else:
        eligible_mask = candidate_mask.to(device=metric.device)
        zero_count = int(((metric == 0) & eligible_mask).sum().item())
        eligible_count = int(eligible_mask.sum().item())
        nonzero_mask = (metric != 0) & torch.isfinite(metric) & eligible_mask

    selected_flat = nonzero_mask.reshape(-1).nonzero(as_tuple=False).flatten()
    if selected_flat.numel() == 0:
        return zero_count, eligible_count, []

    metric_flat = metric.reshape(-1)
    selected_scores = metric_flat[selected_flat].float()
    keep = min(limit, selected_scores.numel())
    top_order = torch.topk(selected_scores, k=keep, largest=False, sorted=True).indices
    top_flat = selected_flat[top_order].detach().cpu()
    top_scores = selected_scores[top_order].detach().cpu()

    cols = metric.shape[1]
    rows = []
    for flat_idx, score in zip(top_flat.tolist(), top_scores.tolist()):
        row_idx = int(flat_idx // cols)
        col_idx = int(flat_idx % cols)
        rows.append(
            {
                "op_name": op_name,
                "i": row_idx,
                "j": col_idx,
                "weight_magnitude": _weight_magnitude_at(module, row_idx, col_idx),
                "score": float(score),
            }
        )
    return zero_count, eligible_count, rows


def prune_curvature_layer(args, model, layer_idx, layer_scores, edge_log_path=None):
    group_entries = []
    for op_name, curv in layer_scores.items():
        module = _get_prunable_module(model, layer_idx, op_name)
        curv_cpu = curv.detach().cpu() if torch.is_tensor(curv) else torch.as_tensor(curv)
        if curv_cpu.shape != module.weight.data.shape:
            if curv_cpu.T.shape == module.weight.data.shape:
                curv_cpu = curv_cpu.T
            else:
                raise ValueError(
                    f"Curvature shape mismatch for layer {layer_idx} {op_name}: "
                    f"{tuple(curv_cpu.shape)} vs {tuple(module.weight.data.shape)}"
                )
        finite_mask = torch.isfinite(curv_cpu)
        if int(finite_mask.sum().item()) == 0:
            continue
        group_entries.append(
            {
                "op_name": op_name,
                "module": module,
                "curv": curv_cpu,
                "finite_mask": finite_mask,
            }
        )

    total_finite = sum(int(entry["finite_mask"].sum().item()) for entry in group_entries)
    prune_count = int(total_finite * float(args.sparsity_ratio))
    if prune_count <= 0 or total_finite == 0:
        return {"layer_idx": layer_idx, "pruned_params": 0, "total_params": total_finite}, None

    all_scores = torch.cat(
        [entry["curv"][entry["finite_mask"]].reshape(-1) for entry in group_entries]
    )
    prune_count = min(prune_count, all_scores.numel())
    prune_high_scores = _score_order_is_descending(args)
    selected = torch.topk(
        all_scores,
        k=prune_count,
        largest=prune_high_scores,
        sorted=True,
    ).indices
    selection_mask = torch.zeros(all_scores.numel(), dtype=torch.bool)
    selection_mask[selected] = True
    cutoff = _selected_score_cutoff(all_scores[selected], prune_high_scores)

    edge_rows = []
    for selected_idx in selected[:25].tolist():
        offset = 0
        for entry in group_entries:
            finite_count = int(entry["finite_mask"].sum().item())
            if selected_idx >= offset + finite_count:
                offset += finite_count
                continue

            local_idx = selected_idx - offset
            flat_positions = entry["finite_mask"].reshape(-1).nonzero(as_tuple=False).flatten()
            flat_idx = int(flat_positions[local_idx].item())
            col_count = entry["finite_mask"].shape[1]
            row_idx = int(flat_idx // col_count)
            col_idx = int(flat_idx % col_count)
            edge_rows.append(
                {
                    "op_name": entry["op_name"],
                    "i": row_idx,
                    "j": col_idx,
                    "weight_magnitude": _weight_magnitude_at(entry["module"], row_idx, col_idx),
                    "score": float(all_scores[selected_idx].item()),
                }
            )
            break
    _append_first_pruned_edges(
        edge_log_path,
        args,
        "curvature",
        layer_idx,
        getattr(args, "prune_score_order", "high_to_low"),
        "curvature",
        edge_rows,
    )

    offset = 0
    total_pruned = 0
    with torch.no_grad():
        for entry in group_entries:
            finite_mask = entry["finite_mask"]
            entry_count = int(finite_mask.sum().item())
            entry_selection = selection_mask[offset:offset + entry_count]
            offset += entry_count

            prune_mask_cpu = torch.zeros_like(finite_mask, dtype=torch.bool)
            prune_mask_cpu[finite_mask] = entry_selection
            total_pruned += int(prune_mask_cpu.sum().item())

            prune_mask = prune_mask_cpu.to(device=entry["module"].weight.data.device)
            entry["module"].weight.data[prune_mask] = 0

    return {
        "layer_idx": layer_idx,
        "pruned_params": total_pruned,
        "total_params": total_finite,
    }, cutoff


def prune_magnitude_layer(
    args,
    model,
    layer_idx,
    layer_curvature_scores=None,
    prune_n=0,
    prune_m=0,
    edge_log_path=None,
):
    subset = find_layers(model.model.layers[layer_idx])
    layer_pruned = 0
    layer_total = 0
    cutoffs = []
    score_rows = []
    zero_score_count = 0
    eligible_score_count = 0

    for name, module in subset.items():
        weight = module.weight.data
        metric = torch.abs(weight)
        candidate_mask = _candidate_mask_from_curvature(layer_curvature_scores, name, weight)
        op_zero_count, op_eligible_count, op_score_rows = _nonzero_score_rows(
            name,
            module,
            metric,
            candidate_mask=candidate_mask,
        )
        zero_score_count += op_zero_count
        eligible_score_count += op_eligible_count
        score_rows.extend(op_score_rows)
        if prune_n != 0:
            prune_mask = (torch.zeros_like(weight) == 1)
            selected_scores = []
            for col_idx in range(metric.shape[1]):
                if col_idx % prune_m != 0:
                    continue
                group_metric = metric[:, col_idx:(col_idx + prune_m)].float()
                group_candidate = (
                    None if candidate_mask is None else candidate_mask[:, col_idx:(col_idx + prune_m)]
                )
                if group_candidate is not None:
                    group_metric = group_metric.masked_fill(~group_candidate, float("inf"))
                selected = torch.topk(group_metric, prune_n, dim=1, largest=False)[1]
                if group_candidate is not None:
                    selected_mask = torch.gather(group_candidate, 1, selected)
                    valid_scores = torch.gather(group_metric, 1, selected)[selected_mask]
                    selected = selected.masked_fill(~selected_mask, 0)
                else:
                    valid_scores = torch.gather(group_metric, 1, selected).reshape(-1)
                if valid_scores.numel() > 0:
                    selected_scores.append(valid_scores.detach().cpu())
                prune_mask.scatter_(1, col_idx + selected, True)
            if candidate_mask is not None:
                prune_mask &= candidate_mask
            if selected_scores:
                cutoffs.append(float(torch.cat(selected_scores).max().item()))
        else:
            if candidate_mask is None:
                candidate_mask = torch.ones_like(metric, dtype=torch.bool, device=metric.device)
            prune_mask, cutoff = _select_lowest_mask(metric, candidate_mask, args.sparsity_ratio)
            if cutoff is not None:
                cutoffs.append(cutoff)

        weight[prune_mask] = 0
        layer_pruned += int(prune_mask.sum().item())
        layer_total += int(candidate_mask.sum().item()) if candidate_mask is not None else weight.numel()

    score_rows = sorted(score_rows, key=lambda row: row["score"])[:25]
    _append_score_zero_summary(
        edge_log_path,
        args,
        "magnitude",
        layer_idx,
        "magnitude",
        zero_score_count,
        eligible_score_count,
        score_rows,
    )

    cutoff = max(cutoffs) if cutoffs else None
    return {"layer_idx": layer_idx, "pruned_params": layer_pruned, "total_params": layer_total}, cutoff


def prune_wanda_layer(
    args,
    model,
    layer_idx,
    layer_wanda_scores,
    layer_curvature_scores=None,
    prune_n=0,
    prune_m=0,
    edge_log_path=None,
):
    subset = find_layers(model.model.layers[layer_idx])
    layer_pruned = 0
    layer_total = 0
    cutoffs = []
    score_rows = []
    zero_score_count = 0
    eligible_score_count = 0

    for name, module in subset.items():
        if name not in layer_wanda_scores:
            raise KeyError(f"Missing precomputed WANDA scores for layer {layer_idx} name {name}")

        weight = module.weight.data
        metric = layer_wanda_scores[name].detach().cpu()
        if metric.shape != weight.shape:
            raise ValueError(
                f"WANDA score shape mismatch for layer {layer_idx} {name}: "
                f"{tuple(metric.shape)} vs {tuple(weight.shape)}"
            )

        candidate_mask = _candidate_mask_from_curvature(layer_curvature_scores, name, weight)
        if candidate_mask is not None:
            candidate_mask = candidate_mask.cpu()
        op_zero_count, op_eligible_count, op_score_rows = _nonzero_score_rows(
            name,
            module,
            metric,
            candidate_mask=candidate_mask,
        )
        zero_score_count += op_zero_count
        eligible_score_count += op_eligible_count
        score_rows.extend(op_score_rows)

        prune_mask = (torch.zeros_like(metric) == 1)
        if prune_n != 0:
            selected_scores = []
            for col_idx in range(metric.shape[1]):
                if col_idx % prune_m != 0:
                    continue
                group_metric = metric[:, col_idx:(col_idx + prune_m)].float()
                group_candidate = (
                    None if candidate_mask is None else candidate_mask[:, col_idx:(col_idx + prune_m)]
                )
                if group_candidate is not None:
                    group_metric = group_metric.masked_fill(~group_candidate, float("inf"))
                selected = torch.topk(group_metric, prune_n, dim=1, largest=False)[1]
                if group_candidate is not None:
                    selected_mask = torch.gather(group_candidate, 1, selected)
                    valid_scores = torch.gather(group_metric, 1, selected)[selected_mask]
                    selected = selected.masked_fill(~selected_mask, 0)
                else:
                    valid_scores = torch.gather(group_metric, 1, selected).reshape(-1)
                if valid_scores.numel() > 0:
                    selected_scores.append(valid_scores.detach().cpu())
                prune_mask.scatter_(1, col_idx + selected, True)
            if candidate_mask is not None:
                prune_mask &= candidate_mask
            if selected_scores:
                cutoffs.append(float(torch.cat(selected_scores).max().item()))
        else:
            if candidate_mask is not None and not args.use_variant:
                prune_mask, cutoff = _select_lowest_mask(metric, candidate_mask, args.sparsity_ratio)
                if cutoff is not None:
                    cutoffs.append(cutoff)
            else:
                if candidate_mask is not None and args.use_variant:
                    metric = metric.masked_fill(~candidate_mask, float("inf"))

                if args.use_variant:
                    sort_res = torch.sort(metric, dim=-1, stable=True)
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    prune_mask, cur_sparsity = _return_given_alpha(alpha, sort_res, metric, tmp_metric, sum_before)
                    while (
                        torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                        and (alpha_hist[1] - alpha_hist[0] >= 0.001)
                    ):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        alpha = alpha_new
                        prune_mask, cur_sparsity = _return_given_alpha(
                            alpha, sort_res, metric, tmp_metric, sum_before
                        )
                else:
                    prune_mask = _row_lowest_mask(metric, args.sparsity_ratio)
                    if candidate_mask is not None:
                        prune_mask &= candidate_mask

                selected_scores = metric[prune_mask]
                if selected_scores.numel() > 0:
                    cutoffs.append(float(selected_scores.max().item()))

        weight[prune_mask.to(device=weight.device)] = 0
        layer_pruned += int(prune_mask.sum().item())
        layer_total += int(candidate_mask.sum().item()) if candidate_mask is not None else weight.numel()

    score_rows = sorted(score_rows, key=lambda row: row["score"])[:25]
    _append_score_zero_summary(
        edge_log_path,
        args,
        "wanda",
        layer_idx,
        "wanda_score",
        zero_score_count,
        eligible_score_count,
        score_rows,
    )

    cutoff = max(cutoffs) if cutoffs else None
    return {"layer_idx": layer_idx, "pruned_params": layer_pruned, "total_params": layer_total}, cutoff


def _row_lowest_mask(metric, ratio):
    prune_mask = torch.zeros_like(metric, dtype=torch.bool)
    prune_count = int(metric.shape[1] * ratio)
    if prune_count <= 0:
        return prune_mask
    if prune_count >= metric.shape[1]:
        return torch.ones_like(metric, dtype=torch.bool)
    threshold = torch.kthvalue(metric, k=prune_count, dim=1).values.reshape(-1, 1)
    return metric <= threshold


def _return_given_alpha(alpha, sort_res, metric, tmp_metric, sum_before):
    threshold_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= threshold_cumsum.reshape((-1, 1))
    threshold = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    prune_mask = metric <= threshold
    cur_sparsity = (prune_mask == True).sum() / prune_mask.numel()
    return prune_mask, cur_sparsity


def draw_per_layer_ppl_vs_sparsity(records, plot_dir, annotate_cutoff=False):
    if not records:
        return []

    try:
        mpl_config_dir = os.path.join("/tmp", "matplotlib")
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        os.makedirs(plot_dir, exist_ok=True)
        marker_path = os.path.join(plot_dir, "plot_error.txt")
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(f"Could not draw per-layer plots: {exc}\n")
        return []

    os.makedirs(plot_dir, exist_ok=True)
    layer_groups = {}
    for record in records:
        layer_groups.setdefault(int(record["layer_idx"]), []).append(record)

    saved_paths = []
    for layer_idx, layer_records in sorted(layer_groups.items()):
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        grouped = {}
        score_orders = sorted({record["score_order"] for record in layer_records})
        for record in layer_records:
            key = (record["score_order"], int(record["pp_seq_len"]))
            grouped.setdefault(key, []).append(record)

        for (score_order, pp_seq_len), group_records in sorted(grouped.items()):
            group_records = sorted(group_records, key=lambda item: item["target_sparsity"])
            xs = np.asarray([item["target_sparsity"] for item in group_records], dtype=np.float64)
            ys = np.asarray([item["ppl_test"] for item in group_records], dtype=np.float64)
            finite = np.isfinite(xs) & np.isfinite(ys)
            if not finite.any():
                continue
            ax.plot(
                xs[finite],
                ys[finite],
                marker="o",
                linewidth=1.0,
                markersize=2.6,
                label=(
                    f"pp={pp_seq_len}"
                    if len(score_orders) == 1
                    else f"{score_order}, pp={pp_seq_len}"
                ),
            )

        if annotate_cutoff:
            for score_order in score_orders:
                pp_lens = sorted(
                    int(record["pp_seq_len"])
                    for record in layer_records
                    if record["score_order"] == score_order
                )
                if not pp_lens:
                    continue
                anchor_records = sorted(
                    grouped[(score_order, pp_lens[0])],
                    key=lambda item: item["target_sparsity"],
                )
                for record in anchor_records:
                    cutoff = record.get("score_cutoff")
                    if cutoff is None or not np.isfinite(cutoff):
                        continue
                    ax.annotate(
                        f"{cutoff:.3g}",
                        (record["target_sparsity"], record["ppl_test"]),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha="center",
                        fontsize=6,
                    )

                nonpositive = sorted(
                    float(record["target_sparsity"])
                    for record in anchor_records
                    if bool(record.get("cutoff_nonpositive", False))
                )
                if nonpositive:
                    ax.axvline(
                        nonpositive[0],
                        color="tab:red",
                        linestyle="--",
                        linewidth=0.9,
                        label=f"{score_order} non-positive cutoff",
                    )

        ax.set_title(f"Layer {layer_idx}")
        ax.set_xlabel("target sparsity")
        ax.set_ylabel("perplexity")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        plot_path = os.path.join(plot_dir, f"layer_{layer_idx:03d}.png")
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        saved_paths.append(plot_path)

    return saved_paths
