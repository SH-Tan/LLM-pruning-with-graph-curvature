#!/usr/bin/env python
import argparse
import json
import os
import re
from pathlib import Path
from zipfile import BadZipFile, ZipFile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import pandas as pd

from plot_eval_results import plot_summary_csv


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def _tag_dict(tags):
    pairs = []
    for tag_group in tags:
        for tag in str(tag_group).split(","):
            if "=" in tag:
                pairs.append(tag.split("=", 1))
    return dict(pairs)


def _prune_ops_from_folder(path):
    for parent in path.parents:
        match = re.match(
            r"^pruned_[^_]+_(?:per_op|globally|locally|global|local|per_layer|per_layer_op)_(?:high_to_low|low_to_high)_(.+)_sparsity_",
            parent.name,
        )
        if match:
            return match.group(1)
    return ""


def _normalize_prune_ops(prune_ops):
    return str(prune_ops).replace(" ", "_").replace("-", "_")


def _task_label(eval_info, tags):
    task = eval_info.get("task", "")
    language = tags.get("language", "")
    if language:
        return f"{task}_{language}"
    return task


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_from_header(header):
    scores = (header.get("results") or {}).get("scores") or []
    for score in scores:
        metrics = score.get("metrics") or {}
        if "accuracy" in metrics:
            return score, "accuracy", _float_or_none(metrics["accuracy"].get("value"))
    for score in scores:
        for metric_name, metric in (score.get("metrics") or {}).items():
            value = _float_or_none(metric.get("value"))
            if value is not None:
                return score, metric_name, value
    return None, None, None


def _read_header(path):
    try:
        with ZipFile(path) as zf:
            if "header.json" not in zf.namelist():
                return None, "missing_header"
            return json.loads(zf.read("header.json")), None
    except (BadZipFile, json.JSONDecodeError):
        return None, "bad_eval"


def build_inspect_log_summary(log_dir):
    rows = []
    skipped = []
    for path in sorted(Path(log_dir).glob("**/*.eval")):
        header, skip_reason = _read_header(path)
        if header is None:
            skipped.append({"path": str(path), "reason": skip_reason})
            continue

        score_info, metric_name, score = _score_from_header(header)
        if score is None:
            skipped.append({"path": str(path), "reason": "missing_numeric_metric"})
            continue

        eval_info = header.get("eval") or {}
        tags = _tag_dict(eval_info.get("tags") or [])
        prune_ops = _prune_ops_from_folder(path) or tags.get("prune_ops", "")
        prune_ops = _normalize_prune_ops(prune_ops)
        rows.append(
            {
                "sparsity": _float_or_none(tags.get("sparsity")),
                "prune_method": tags.get("method", ""),
                "prune_scope": tags.get("prunescore_order", ""),
                "score_order": tags.get("prune_score_order", ""),
                "prune_ops": prune_ops,
                "task": _task_label(eval_info, tags),
                "raw_task": eval_info.get("task", ""),
                "benchmark": tags.get("benchmark", ""),
                "language": tags.get("language", ""),
                "score": score,
                "metric": metric_name,
                "scorer": (score_info or {}).get("scorer", ""),
                "status": header.get("status", ""),
                "created": eval_info.get("created", ""),
                "run_id": eval_info.get("run_id", ""),
                "task_id": eval_info.get("task_id", ""),
                "completed_samples": (header.get("results") or {}).get("completed_samples"),
                "total_samples": (header.get("results") or {}).get("total_samples"),
                "log_path": str(path),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(skipped)


def aggregate_for_plots(df):
    required = ["sparsity", "prune_method", "prune_scope", "score_order", "task", "score"]
    df = df.dropna(subset=required).copy()
    group_cols = ["sparsity", "prune_method", "prune_scope", "score_order", "prune_ops", "task"]
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            score=("score", "mean"),
            runs=("score", "size"),
            completed_samples=("completed_samples", "mean"),
            total_samples=("total_samples", "mean"),
        )
        .sort_values(["prune_method", "prune_scope", "score_order", "prune_ops", "sparsity", "task"])
    )


def plot_by_prune_ops(plot_df, output_dir):
    saved = []
    for prune_ops, ops_df in plot_df.groupby("prune_ops", sort=True):
        ops_name = _safe_name(prune_ops)
        ops_dir = output_dir / "by_prune_ops" / ops_name
        ops_dir.mkdir(parents=True, exist_ok=True)
        ops_summary_path = ops_dir / "inspect_log_scores_summary.csv"
        ops_df.to_csv(ops_summary_path, index=False)
        print(f"saved {ops_summary_path}")
        saved.extend(plot_summary_csv(ops_summary_path, ops_dir))
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="inspect_evals/logs/downstream")
    parser.add_argument("--output-dir", default="inspect_evals/plots/downstream")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df, skipped_df = build_inspect_log_summary(args.log_dir)
    raw_path = output_dir / "inspect_log_scores_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"saved {raw_path}")

    if not skipped_df.empty:
        skipped_path = output_dir / "inspect_log_scores_skipped.csv"
        skipped_df.to_csv(skipped_path, index=False)
        print(f"saved {skipped_path}")

    plot_df = aggregate_for_plots(raw_df)
    summary_path = output_dir / "inspect_log_scores_summary.csv"
    plot_df.to_csv(summary_path, index=False)
    print(f"saved {summary_path}")

    for plot_path in plot_by_prune_ops(plot_df, output_dir):
        print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
