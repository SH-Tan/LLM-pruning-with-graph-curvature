import csv
import os

import numpy as np
import torch

from curv_prune_utils import _get_prunable_module


def _curvature_score_order(args):
    return getattr(args, "prune_score_order", "high_to_low") == "high_to_low"


def _as_cpu_curvature(curv):
    return curv.detach().cpu() if torch.is_tensor(curv) else torch.as_tensor(curv)


def _prune_curvature_group(args, group_entries):
    finite_count = sum(int(entry["finite_mask"].sum().item()) for entry in group_entries)
    prune_count = int(finite_count * float(args.sparsity_ratio))
    if prune_count <= 0:
        return 0

    prune_count = min(prune_count, finite_count)
    all_scores = torch.cat([
        entry["curv"][entry["finite_mask"]].reshape(-1)
        for entry in group_entries
    ])
    selected = torch.topk(
        all_scores,
        k=prune_count,
        largest=_curvature_score_order(args),
        sorted=False,
    ).indices

    group_mask = torch.zeros(all_scores.numel(), dtype=torch.bool)
    group_mask[selected] = True

    offset = 0
    pruned_count = 0
    with torch.no_grad():
        for entry in group_entries:
            finite_mask = entry["finite_mask"]
            entry_count = int(finite_mask.sum().item())
            entry_selection = group_mask[offset:offset + entry_count]
            offset += entry_count

            prune_mask_cpu = torch.zeros_like(finite_mask, dtype=torch.bool)
            prune_mask_cpu[finite_mask] = entry_selection
            pruned_count += int(prune_mask_cpu.sum().item())

            prune_mask = prune_mask_cpu.to(device=entry["module"].weight.data.device)
            entry["module"].weight.data[prune_mask] = 0

            del prune_mask, prune_mask_cpu

    return pruned_count


def _iter_curvature_entries(model):
    for layer_idx, layer_scores in enumerate(getattr(model, "curvature_scores", [])):
        for op_name, curv in layer_scores.items():
            module = _get_prunable_module(model, layer_idx, op_name)
            curv_cpu = _as_cpu_curvature(curv)
            if curv_cpu.shape != module.weight.data.shape:
                raise ValueError(
                    f"Curvature shape mismatch for layer {layer_idx} {op_name}: "
                    f"{tuple(curv_cpu.shape)} vs {tuple(module.weight.data.shape)}"
                )

            finite_mask = torch.isfinite(curv_cpu)
            finite_count = int(finite_mask.sum().item())
            if finite_count == 0:
                print(f"Skipping layer {layer_idx} {op_name}: no finite curvature scores")
                continue

            yield {
                "layer_idx": layer_idx,
                "op_name": op_name,
                "module": module,
                "curv": curv_cpu,
                "finite_mask": finite_mask,
                "finite_count": finite_count,
            }


def prune_layer_curvature(args, model):
    model.eval()
    layer_groups = {}
    for entry in _iter_curvature_entries(model):
        layer_groups.setdefault(entry["layer_idx"], []).append(entry)

    prune_summary = []
    total_pruned = 0
    for layer_idx, group_entries in sorted(layer_groups.items()):
        layer_pruned = _prune_curvature_group(args, group_entries)
        layer_total = sum(entry["finite_count"] for entry in group_entries)
        total_pruned += layer_pruned
        prune_summary.append(
            {
                "layer_idx": layer_idx,
                "op_name": "all_ops",
                "pruned_edges": layer_pruned,
                "total_edges": layer_total,
                "pruned_params": layer_pruned,
                "total_params": layer_total,
            }
        )

    print(f"Layer-wise curvature pruning complete: pruned={total_pruned}")
    return prune_summary


def prune_layer_op_curvature(args, model):
    model.eval()
    prune_summary = []
    total_pruned = 0

    for entry in _iter_curvature_entries(model):
        op_pruned = _prune_curvature_group(args, [entry])
        total_pruned += op_pruned
        prune_summary.append(
            {
                "layer_idx": entry["layer_idx"],
                "op_name": entry["op_name"],
                "pruned_edges": op_pruned,
                "total_edges": entry["finite_count"],
                "pruned_params": op_pruned,
                "total_params": entry["finite_count"],
            }
        )

    print(f"Layer-op curvature pruning complete: pruned={total_pruned}")
    return prune_summary


def prune_scoped_curvature(args, model):
    scope = getattr(args, "curvature_prune_scope", "global")
    if scope == "per_layer":
        return prune_layer_curvature(args, model)
    if scope == "per_layer_op":
        return prune_layer_op_curvature(args, model)
    raise ValueError(f"Unsupported curvature_prune_scope for scoped pruning: {scope}")


def save_eval_records_csv(records, csv_path):
    if not records:
        return None

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "method",
        "prune_scope",
        "score_order",
        "target_sparsity",
        "actual_sparsity",
        "pp_seq_len",
        "ppl_test",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in fieldnames})

    return csv_path


def draw_ppl_vs_sparsity(records, plot_path):
    if not records:
        return None

    try:
        mpl_config_dir = os.path.join("/tmp", "matplotlib")
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        marker_path = os.path.splitext(plot_path)[0] + "_plot_error.txt"
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(f"Could not draw ppl vs sparsity plot: {exc}\n")
        return None

    grouped = {}
    for record in records:
        key = (record["score_order"], int(record["pp_seq_len"]))
        grouped.setdefault(key, []).append(record)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for (score_order, pp_seq_len), group_records in sorted(grouped.items()):
        group_records = sorted(group_records, key=lambda item: item["target_sparsity"])
        xs = np.asarray([item["target_sparsity"] for item in group_records], dtype=np.float64)
        ys = np.asarray([item["ppl_test"] for item in group_records], dtype=np.float64)
        finite = np.isfinite(xs) & np.isfinite(ys)
        if finite.any():
            ax.plot(
                xs[finite],
                ys[finite],
                marker="o",
                linewidth=1.0,
                markersize=2.4,
                label=f"{score_order}, pp={pp_seq_len}",
            )

    ax.set_xlabel("target sparsity")
    ax.set_ylabel("perplexity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    return plot_path
