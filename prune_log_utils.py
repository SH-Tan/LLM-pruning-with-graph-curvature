import os

import torch


def _weight_magnitude_at(module, row_idx, col_idx):
    weight = module.weight.data
    rows = torch.as_tensor([row_idx], device=weight.device)
    cols = torch.as_tensor([col_idx], device=weight.device)
    return float(weight[rows, cols].abs().detach().cpu().item())


def collect_pruned_parameter_rows(
    layer_idx,
    op_name,
    module,
    metric,
    prune_mask,
    largest,
    limit,
    include_input_scale=False,
):
    selected_flat = prune_mask.reshape(-1).nonzero(as_tuple=False).flatten()
    if selected_flat.numel() == 0:
        return []

    metric_cpu = metric.detach().cpu() if torch.is_tensor(metric) else torch.as_tensor(metric)
    selected_flat = selected_flat.cpu()
    metric_flat = metric_cpu.reshape(-1)
    selected_scores = metric_flat[selected_flat].float()
    keep = min(int(limit), selected_scores.numel())
    if keep <= 0:
        return []

    top_order = torch.topk(selected_scores, k=keep, largest=largest, sorted=True).indices
    top_flat = selected_flat[top_order]
    top_scores = selected_scores[top_order]

    cols = metric_cpu.shape[1]
    rows = []
    for flat_idx, score in zip(top_flat.tolist(), top_scores.tolist()):
        row_idx = int(flat_idx // cols)
        col_idx = int(flat_idx % cols)
        weight_magnitude = _weight_magnitude_at(module, row_idx, col_idx)
        row = {
            "layer_idx": int(layer_idx),
            "op_name": op_name,
            "row_idx": row_idx,
            "col_idx": col_idx,
            "weight_magnitude": weight_magnitude,
            "score": float(score),
        }
        if include_input_scale:
            row["input_scale"] = float(score) / weight_magnitude if weight_magnitude != 0 else 0.0
        rows.append(row)
    return rows


def append_all_layer_pruned_parameter_log(
    log_path,
    args,
    method,
    score_order,
    score_name,
    rows,
    largest,
    rank_offset=0,
):
    if log_path is None or not rows:
        return

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    rank_offset = int(rank_offset)
    rows = sorted(rows, key=lambda row: row["score"], reverse=largest)
    rows = rows[rank_offset:rank_offset + 25]
    if not rows:
        return

    has_input_scale = any("input_scale" in row for row in rows)
    with open(log_path, "a+", encoding="utf-8") as f:
        print(
            f"all_layer_pruned_parameters method={method}, "
            f"target_sparsity={float(args.sparsity_ratio):.4f}, "
            f"score_order={score_order}",
            file=f,
            flush=True,
        )
        header = (
            f"{'rank':<6}{'layer':<8}{'op_name':<12}{'index':<16}"
            f"{'weight_magnitude':<20}"
        )
        if has_input_scale:
            header += f"{'input_scale':<20}"
        header += f"{score_name:<20}"
        print(header, file=f, flush=True)
        for rank, row in enumerate(rows, start=rank_offset + 1):
            line = (
                f"{rank:<6}{row['layer_idx']:<8}{row['op_name']:<12}"
                f"({row['row_idx']},{row['col_idx']})".ljust(16)
                + f"{row['weight_magnitude']:<20.8g}"
            )
            if has_input_scale:
                line += f"{row.get('input_scale', 0.0):<20.8g}"
            line += f"{row['score']:<20.8g}"
            print(line, file=f, flush=True)
        print("", file=f, flush=True)


def append_layer_pruned_parameter_log(
    log_path,
    args,
    method,
    layer_idx,
    score_order,
    score_name,
    rows,
    largest,
):
    if log_path is None or not rows:
        return

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    rows = sorted(rows, key=lambda row: row["score"], reverse=largest)[:25]
    if not rows:
        return

    has_input_scale = any("input_scale" in row for row in rows)
    with open(log_path, "a+", encoding="utf-8") as f:
        print(
            f"per_layer_top_pruned_parameters method={method}, layer={int(layer_idx)}, "
            f"target_sparsity={float(args.sparsity_ratio):.4f}, "
            f"score_order={score_order}",
            file=f,
            flush=True,
        )
        header = (
            f"{'rank':<6}{'op_name':<12}{'index':<16}"
            f"{'weight_magnitude':<20}"
        )
        if has_input_scale:
            header += f"{'input_scale':<20}"
        header += f"{score_name:<20}"
        print(header, file=f, flush=True)

        for rank, row in enumerate(rows, start=1):
            line = (
                f"{rank:<6}{row['op_name']:<12}"
                f"({row['row_idx']},{row['col_idx']})".ljust(16)
                + f"{row['weight_magnitude']:<20.8g}"
            )
            if has_input_scale:
                line += f"{row.get('input_scale', 0.0):<20.8g}"
            line += f"{row['score']:<20.8g}"
            print(line, file=f, flush=True)
        print("", file=f, flush=True)
