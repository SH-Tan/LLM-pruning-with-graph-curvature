#!/usr/bin/env python
import argparse
import glob
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import pandas as pd


RUN_LABELS = [
    ("gate_up", "global", "curvature"),
    ("gate_up", "global", "wanda"),
    ("gate_up", "global", "magnitude"),
    ("gate_up", "local", "curvature"),
    ("gate_up", "local", "wanda"),
    ("gate_up", "local", "magnitude"),
    ("gate_up", "per_op", "curvature"),
    ("gate_up", "per_op", "wanda"),
    ("gate_up", "per_op", "magnitude"),
    ("up", "global", "curvature"),
    ("up", "global", "wanda"),
    ("up", "global", "magnitude"),
    ("up", "local", "curvature"),
    ("up", "local", "wanda"),
    ("up", "local", "magnitude"),
    ("up", "per_op", "curvature"),
    ("up", "per_op", "wanda"),
    ("up", "per_op", "magnitude"),
    ("gate", "global", "curvature"),
    ("gate", "global", "wanda"),
    ("gate", "global", "magnitude"),
    ("gate", "local", "curvature"),
    ("gate", "local", "wanda"),
    ("gate", "local", "magnitude"),
    ("gate", "per_op", "curvature"),
    ("gate", "per_op", "wanda"),
    ("gate", "per_op", "magnitude"),
]


def build_downstream_records(summary_csv):
    df = pd.read_csv(summary_csv)
    if df.empty:
        return df, []

    first_sparsity = df.iloc[0]["sparsity"]
    per_sparsity_rows = 0
    for sparsity in df["sparsity"]:
        if sparsity != first_sparsity:
            break
        per_sparsity_rows += 1

    sparsities = sorted(df["sparsity"].unique())
    rows_per_run = len(sparsities) * per_sparsity_rows
    run_count = min((len(df) + rows_per_run - 1) // rows_per_run, len(RUN_LABELS))

    records = []
    run_labels = []
    for run_idx in range(run_count):
        start = run_idx * rows_per_run
        end = min((run_idx + 1) * rows_per_run, len(df))
        if start >= end:
            break

        ops, scope, method = RUN_LABELS[run_idx]
        chunk = df.iloc[start:end].copy()
        chunk["row_in_sparsity"] = chunk.groupby("sparsity").cumcount()
        chunk = chunk[chunk["row_in_sparsity"] > 0].copy()
        if chunk.empty:
            continue

        chunk.insert(0, "ops", ops)
        chunk.insert(1, "prune_scope", scope)
        chunk.insert(2, "prune_method", method)
        chunk.insert(3, "run_index", run_idx + 1)
        chunk["is_complete_run"] = (end - start) == rows_per_run
        chunk = chunk.drop(columns=["row_in_sparsity"])
        records.append(chunk)
        run_labels.append((ops, scope, method, chunk["is_complete_run"].all()))

    if not records:
        return pd.DataFrame(), []
    return pd.concat(records, ignore_index=True), run_labels


def export_downstream_csvs(summary_csv, output_dir):
    records, run_labels = build_downstream_records(summary_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if records.empty:
        return [], run_labels

    saved = []
    all_path = output_dir / "all_downstream_scores.csv"
    records.to_csv(all_path, index=False)
    saved.append(all_path)

    method_dir = output_dir / "by_method"
    task_dir = output_dir / "by_task"
    sparsity_dir = output_dir / "by_sparsity"
    method_dir.mkdir(exist_ok=True)
    task_dir.mkdir(exist_ok=True)
    sparsity_dir.mkdir(exist_ok=True)

    for (ops, scope, method), sub in records.groupby(["ops", "prune_scope", "prune_method"], sort=False):
        path = method_dir / f"{ops}_{scope}_{method}.csv"
        sub.sort_values(["sparsity", "task"]).to_csv(path, index=False)
        saved.append(path)

    for task, sub in records.groupby("task", sort=True):
        path = task_dir / f"{task}.csv"
        sub.sort_values(["sparsity", "ops", "prune_scope", "prune_method"]).to_csv(path, index=False)
        saved.append(path)

    for sparsity, sub in records.groupby("sparsity", sort=True):
        sparsity_tag = str(sparsity).replace(".", "p")
        path = sparsity_dir / f"sparsity_{sparsity_tag}.csv"
        sub.sort_values(["task", "ops", "prune_scope", "prune_method"]).to_csv(path, index=False)
        saved.append(path)

    return saved, run_labels


def plot_downstream_summary(summary_csv, output_dir):
    df = pd.read_csv(summary_csv)
    if df.empty:
        return None, []

    sparsities = sorted(df["sparsity"].unique())
    first_sparsity = df.iloc[0]["sparsity"]
    per_sparsity_rows = 0
    for sparsity in df["sparsity"]:
        if sparsity != first_sparsity:
            break
        per_sparsity_rows += 1
    rows_per_run = len(sparsities) * per_sparsity_rows
    completed_runs = min(len(df) // rows_per_run, len(RUN_LABELS))
    if completed_runs == 0:
        return None, []

    records = []
    for run_idx in range(completed_runs):
        op, scope, method = RUN_LABELS[run_idx]
        chunk = df.iloc[run_idx * rows_per_run : (run_idx + 1) * rows_per_run].copy()
        chunk["row_in_sparsity"] = chunk.groupby("sparsity").cumcount()
        chunk = chunk[chunk["row_in_sparsity"] > 0]
        mean_scores = chunk.groupby("sparsity", as_index=False)["score"].mean()
        mean_scores["label"] = f"{method}/{scope}"
        mean_scores["ops"] = op
        records.append(mean_scores)

    plot_df = pd.concat(records, ignore_index=True)
    ops = plot_df["ops"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, sub in plot_df.groupby("label", sort=False):
        sub = sub.sort_values("sparsity")
        ax.plot(sub["sparsity"], sub["score"], marker="o", linewidth=2, label=label)
    ax.set_title(f"Downstream mean accuracy ({ops}, completed runs)")
    ax.set_xlabel("Target sparsity")
    ax.set_ylabel("Mean accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "downstream_mean_accuracy_gate_up_completed.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path, RUN_LABELS[:completed_runs]


def _method_label(method_tag):
    if "curvature" in method_tag:
        return "curvature"
    if "wanda" in method_tag:
        return "wanda"
    if "magnitude" in method_tag:
        return "magnitude"
    return method_tag


def _scope_label(scope):
    return {
        "global": "global",
        "per_layer": "local",
        "per_layer_op": "per_op",
    }.get(scope, scope)


def plot_pp_records(compare_dir, pp_seqlen, output_dir):
    paths = sorted(glob.glob(os.path.join(compare_dir, "pp_records_*.csv")))
    if not paths:
        return None

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source"] = os.path.basename(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["pp_seq_len"] == pp_seqlen].copy()
    if df.empty:
        return None
    df["method_label"] = df["method_tag"].map(_method_label)
    df["scope_label"] = df["prune_scope"].map(_scope_label)

    scopes = [scope for scope in ("global", "local", "per_op") if scope in set(df["scope_label"])]
    fig, axes = plt.subplots(1, len(scopes), figsize=(5 * len(scopes), 4), sharey=True)
    if len(scopes) == 1:
        axes = [axes]

    for ax, scope in zip(axes, scopes):
        scoped = df[df["scope_label"] == scope]
        for method, sub in scoped.groupby("method_label", sort=False):
            sub = sub.sort_values("target_sparsity")
            ax.plot(sub["target_sparsity"], sub["ppl_test"], marker="o", linewidth=2, label=method)
        ax.set_title(scope)
        ax.set_xlabel("Target sparsity")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(f"PPL test, seq_len={pp_seqlen}")
    axes[-1].legend()
    fig.suptitle(f"Perplexity comparison: {Path(compare_dir).name}")
    fig.tight_layout()

    output_path = output_dir / f"ppl_{Path(compare_dir).name}_seq{pp_seqlen}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default="eval_results/summary.csv")
    parser.add_argument(
        "--compare-dir",
        default="out/llama_8b/unstructured/all_layer_compare/gate_up_resid",
    )
    parser.add_argument("--pp-seqlen", type=int, default=1024)
    parser.add_argument("--output-dir", default="eval_results/plots")
    parser.add_argument("--split-output-dir", default="eval_results/downstream_split")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_paths, run_labels = export_downstream_csvs(args.summary_csv, args.split_output_dir)
    downstream_plot, completed = plot_downstream_summary(args.summary_csv, output_dir)
    pp_plot = plot_pp_records(args.compare_dir, args.pp_seqlen, output_dir)

    if run_labels:
        print("downstream runs in split csvs:")
        for idx, (ops, scope, method, complete) in enumerate(run_labels, start=1):
            status = "complete" if complete else "partial"
            print(f"  {idx}: ops={ops}, scope={scope}, method={method}, status={status}")
    if completed:
        print("completed downstream runs:")
        for idx, (ops, scope, method) in enumerate(completed, start=1):
            print(f"  {idx}: ops={ops}, scope={scope}, method={method}")
    if split_paths:
        print(f"saved {len(split_paths)} split csvs under {args.split_output_dir}")
    if downstream_plot:
        print(f"saved {downstream_plot}")
    if pp_plot:
        print(f"saved {pp_plot}")


if __name__ == "__main__":
    main()
