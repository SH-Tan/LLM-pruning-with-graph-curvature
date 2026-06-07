import csv
import glob
import os

import numpy as np
import torch

from prune import align_curvature_to_weight_shape, skip_prune_layer
from curv_prune_utils import _get_prunable_module, _is_magnitude_fallback
from prune_log_utils import append_layer_pruned_parameter_log, collect_pruned_parameter_rows


def _curvature_score_order(args):
    return getattr(args, "prune_score_order", "high_to_low") == "high_to_low"


def _as_cpu_curvature(curv):
    return curv.detach().cpu() if torch.is_tensor(curv) else torch.as_tensor(curv)


def _prune_curvature_group(args, group_entries, collect_report_rows=False):
    finite_count = sum(int(entry["finite_mask"].sum().item()) for entry in group_entries)
    prune_count = int(finite_count * float(args.sparsity_ratio))
    if prune_count <= 0:
        return 0, []

    prune_count = min(prune_count, finite_count)
    entry_selections = [
        torch.zeros(int(entry["finite_mask"].sum().item()), dtype=torch.bool)
        for entry in group_entries
    ]
    for fallback, largest in ((False, _curvature_score_order(args)), (True, False)):
        remaining = prune_count - sum(int(mask.sum().item()) for mask in entry_selections)
        if remaining <= 0:
            break
        group_indices = [
            idx for idx, entry in enumerate(group_entries)
            if bool(entry["magnitude_fallback"]) == fallback
        ]
        if not group_indices:
            continue
        all_scores = torch.cat([
            group_entries[idx]["curv"][group_entries[idx]["finite_mask"]].reshape(-1)
            for idx in group_indices
        ])
        selected = torch.topk(
            all_scores,
            k=min(remaining, all_scores.numel()),
            largest=largest,
            sorted=False,
        ).indices

        group_mask = torch.zeros(all_scores.numel(), dtype=torch.bool)
        group_mask[selected] = True
        group_offset = 0
        for idx in group_indices:
            entry_count = int(group_entries[idx]["finite_mask"].sum().item())
            entry_selections[idx] = group_mask[group_offset:group_offset + entry_count]
            group_offset += entry_count

    pruned_count = 0
    report_rows = []
    with torch.no_grad():
        for entry, entry_selection in zip(group_entries, entry_selections):
            finite_mask = entry["finite_mask"]
            entry_count = int(finite_mask.sum().item())

            prune_mask_cpu = torch.zeros_like(finite_mask, dtype=torch.bool)
            prune_mask_cpu[finite_mask] = entry_selection
            pruned_count += int(prune_mask_cpu.sum().item())
            if collect_report_rows:
                report_rows.extend(
                    collect_pruned_parameter_rows(
                        entry["layer_idx"],
                        entry["op_name"],
                        entry["module"],
                        entry["curv"],
                        prune_mask_cpu,
                        largest=_curvature_score_order(args),
                        limit=25,
                    )
                )

            prune_mask = prune_mask_cpu.to(device=entry["module"].weight.data.device)
            entry["module"].weight.data[prune_mask] = 0

            del prune_mask, prune_mask_cpu

    return pruned_count, report_rows


def _iter_curvature_entries(args, model):
    for layer_idx, layer_scores in enumerate(getattr(model, "curvature_scores", [])):
        if skip_prune_layer(args, layer_idx):
            print(f"Skipping layer {layer_idx}: requested by skip_prune_layer_ids")
            continue
        for op_name, curv in layer_scores.items():
            module = _get_prunable_module(model, layer_idx, op_name)
            curv_cpu = align_curvature_to_weight_shape(
                curv,
                module.weight.data.shape,
                context=f"layer {layer_idx} {op_name} scoped curvature",
            ).cpu()
            magnitude_fallback = _is_magnitude_fallback(model, layer_idx, op_name)
            if magnitude_fallback:
                curv_cpu = torch.abs(module.weight.data.detach()).cpu()

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
                "magnitude_fallback": magnitude_fallback,
            }


def prune_layer_curvature(args, model):
    model.eval()
    layer_groups = {}
    for entry in _iter_curvature_entries(args, model):
        layer_groups.setdefault(entry["layer_idx"], []).append(entry)

    prune_summary = []
    total_pruned = 0
    for layer_idx, group_entries in sorted(layer_groups.items()):
        layer_pruned, report_rows = _prune_curvature_group(
            args,
            group_entries,
            collect_report_rows=True,
        )
        append_layer_pruned_parameter_log(
            getattr(args, "all_layer_parameter_log_path", None),
            args,
            "curvature",
            layer_idx,
            getattr(args, "prune_score_order", "high_to_low"),
            "curvature",
            report_rows,
            largest=_curvature_score_order(args),
        )
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

    for entry in _iter_curvature_entries(args, model):
        op_pruned, _ = _prune_curvature_group(args, [entry])
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
        "method_tag",
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


def _read_eval_records(csv_paths):
    records_by_key = {}
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    record = {
                        "method": row["method"],
                        "method_tag": row.get("method_tag") or row["method"],
                        "prune_scope": row.get("prune_scope", "global"),
                        "score_order": row["score_order"],
                        "target_sparsity": float(row["target_sparsity"]),
                        "pp_seq_len": int(row["pp_seq_len"]),
                        "ppl_test": float(row["ppl_test"]),
                    }
                except (KeyError, TypeError, ValueError):
                    continue
                key = (
                    record["method_tag"],
                    record["prune_scope"],
                    record["score_order"],
                    record["target_sparsity"],
                    record["pp_seq_len"],
                )
                records_by_key[key] = record
    return list(records_by_key.values())


def _collect_legend_handles_labels(axes):
    handles_by_label = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)
    return list(handles_by_label.values()), list(handles_by_label.keys())


def draw_method_comparison(compare_dir, plot_dir, eval_seq_lens=None):
    csv_paths = sorted(glob.glob(os.path.join(compare_dir, "pp_records_*.csv")))
    records = _read_eval_records(csv_paths)
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
        os.makedirs(plot_dir, exist_ok=True)
        marker_path = os.path.join(plot_dir, "method_compare_plot_error.txt")
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(f"Could not draw method comparison plot: {exc}\n")
        return None

    os.makedirs(plot_dir, exist_ok=True)
    pp_lens = [int(seq) for seq in eval_seq_lens] if eval_seq_lens else []
    if not pp_lens:
        pp_lens = sorted({int(record["pp_seq_len"]) for record in records})
    pp_lens = [seq for seq in pp_lens if any(int(record["pp_seq_len"]) == seq for record in records)]
    if not pp_lens:
        return None

    def draw_scope_plot(scope_label, curvature_scope):
        scope_records = [
            record for record in records
            if record["prune_scope"] == curvature_scope
        ]
        if not scope_records:
            return None

        scope_pp_lens = [
            seq for seq in pp_lens
            if any(int(record["pp_seq_len"]) == seq for record in scope_records)
        ]
        if not scope_pp_lens:
            return None

        fig, axes = plt.subplots(
            1,
            len(scope_pp_lens),
            figsize=(5.0 * len(scope_pp_lens), 3.6),
            squeeze=False,
        )
        for ax, pp_seq_len in zip(axes[0], scope_pp_lens):
            seq_records = [
                record for record in scope_records
                if int(record["pp_seq_len"]) == pp_seq_len
            ]
            grouped = {}
            for record in seq_records:
                label = record["method_tag"]
                if record["method"] == "curvature":
                    label = label.replace("curvature_", "curv_")
                    label = f"{label}_{scope_label}_{record['score_order']}"
                grouped.setdefault(label, []).append(record)

            y_values = []
            for label, group_records in sorted(grouped.items()):
                group_records = sorted(group_records, key=lambda item: item["target_sparsity"])
                xs = np.asarray([item["target_sparsity"] for item in group_records], dtype=np.float64)
                ys = np.asarray([item["ppl_test"] for item in group_records], dtype=np.float64)
                finite = np.isfinite(xs) & np.isfinite(ys)
                if not finite.any():
                    continue
                y_values.extend(ys[finite].tolist())
                ax.plot(
                    xs[finite],
                    ys[finite],
                    marker="o",
                    linewidth=1.1,
                    markersize=2.8,
                    label=label,
                )

            if y_values:
                ymin = min(y_values)
                ymax = max(y_values)
                pad = max((ymax - ymin) * 0.08, 1e-6)
                ax.set_ylim(ymin - pad, ymax + pad)

            ax.set_title(f"pp_seqlen={pp_seq_len}")
            ax.set_xlabel("target sparsity")
            ax.grid(True, alpha=0.3)

        axes[0][0].set_ylabel("perplexity")
        handles, labels = _collect_legend_handles_labels(axes[0])
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), fontsize=7)
        fig.suptitle(f"All-layer method comparison ({scope_label})", y=1.02)
        fig.tight_layout()
        plot_path = os.path.join(plot_dir, f"all_layers_method_compare_{scope_label}.png")
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    plot_paths = [
        path for path in [
            draw_scope_plot("global", "global"),
            draw_scope_plot("local", "per_layer"),
            draw_scope_plot("per_op", "per_layer_op"),
        ]
        if path is not None
    ]
    return plot_paths or None


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
