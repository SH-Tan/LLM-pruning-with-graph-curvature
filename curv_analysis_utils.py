import os
import numpy as np
import torch


_AGGREGATE_MARKER = "overall:\n"
_PARAMETER_CURVATURE_POINTS = {}
_PARAMETER_NEIGHBOR_COST_POINTS = {}
_PARAMETER_METRIC_POINTS = {}
_ACTIVE_PARAMETER_LOGS = set()
_PYPLOT = None


def _analysis_file_path(layer_id, short_name, seq_len, dataset_name, analysis_dir=None):
    if analysis_dir is None:
        analysis_dir = os.path.join(
            os.path.dirname(__file__),
            "curv_analysis",
            f"seq_len_{int(seq_len)}",
            str(dataset_name),
        )
    os.makedirs(analysis_dir, exist_ok=True)
    safe_dataset_name = str(dataset_name).replace("/", "_").replace(" ", "_")
    return os.path.join(
        analysis_dir,
        f"layer_{int(layer_id):03d}_{short_name}_"
        f"{safe_dataset_name}_seq_{int(seq_len):04d}_"
        "min_curv_summary.txt",
    )


def _summarize_curvatures(curvature):
    curvature = np.asarray(curvature, dtype=np.float32)
    finite_mask = np.isfinite(curvature)
    finite_vals = curvature[finite_mask]

    if finite_vals.size == 0:
        return {
            "finite_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "zero_count": 0,
            "min_curv": float("nan"),
            "max_curv": float("nan"),
        }

    return {
        "finite_count": int(finite_vals.size),
        "positive_count": int((finite_vals > 0).sum()),
        "negative_count": int((finite_vals < 0).sum()),
        "zero_count": int((finite_vals == 0).sum()),
        "min_curv": float(finite_vals.min()),
        "max_curv": float(finite_vals.max()),
    }


def _strip_aggregate_footer(content):
    marker_idx = content.find(_AGGREGATE_MARKER)
    if marker_idx != -1:
        return content[:marker_idx].rstrip() + "\n"
    return content


def _write_final_curvature_footer(analysis_path, summary):
    with open(analysis_path, "r", encoding="utf-8") as f:
        content = _strip_aggregate_footer(f.read())

    footer = (
        "\n"
        f"{_AGGREGATE_MARKER}"
        + (
            "source=final_min_curvature_matrix, "
            f"finite_entries={summary['finite_count']}, "
            f"positive_edges={summary['positive_count']}, "
            f"negative_edges={summary['negative_count']}, "
            f"zero_edges={summary['zero_count']}, "
            f"min_curv={summary['min_curv']:.8f}, "
            f"max_curv={summary['max_curv']:.8f}\n"
        )
    )

    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.write(footer)


def start_curvature_analysis(
    layer_id,
    short_name,
    sample_idx,
    curvature_shape,
    seq_len,
    dataset_name,
    analysis_dir=None,
):
    analysis_path = _analysis_file_path(
        layer_id,
        short_name,
        seq_len,
        dataset_name,
        analysis_dir=analysis_dir,
    )
    header = (
        f"layer_id: {int(layer_id)}\n"
        f"op_name: {short_name}\n"
        f"dataset_name: {dataset_name}\n"
        f"curvature_shape: {tuple(curvature_shape)}\n"
        f"seq_len: {int(seq_len)}\n"
        "\n"
    )

    if not os.path.exists(analysis_path):
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(header)
    return analysis_path


def _parameter_log_path(log_root, layer_id, short_name, sample_idx, v_idx, u_idx):
    param_name = "down_proj" if short_name == "prev_down_proj" else short_name
    layer_dir = os.path.join(log_root, f"layer_{int(layer_id):03d}", param_name)
    example_dir = os.path.join(layer_dir, f"example_{int(sample_idx):03d}")
    param_dir = os.path.join(example_dir, f"param_v{int(v_idx)}_u{int(u_idx)}")
    os.makedirs(param_dir, exist_ok=True)
    log_path = os.path.join(
        param_dir,
        f"parameter_log_example_{int(sample_idx):03d}.txt",
    )
    return log_path


def prepare_parameter_detail_log(log_root, layer_id, short_name, sample_idx, v_idx, u_idx):
    if log_root is None:
        return None

    log_path = _parameter_log_path(log_root, layer_id, short_name, sample_idx, v_idx, u_idx)
    if log_path in _ACTIVE_PARAMETER_LOGS:
        return log_path

    with open(log_path, "w", encoding="utf-8"):
        pass

    _ACTIVE_PARAMETER_LOGS.add(log_path)
    _PARAMETER_CURVATURE_POINTS.pop(log_path, None)
    _PARAMETER_METRIC_POINTS.pop(log_path, None)
    _PARAMETER_NEIGHBOR_COST_POINTS.pop(log_path, None)
    return log_path


def _write_array_block(f, name, value):
    if value is None:
        return
    arr = np.asarray(value, dtype=np.float64)
    with np.printoptions(threshold=np.inf, linewidth=200, precision=8, suppress=False):
        f.write(f"{name}: {arr.tolist()}\n")


def append_parameter_detail_log(log_root, layer_id, short_name, edge_res):
    if log_root is None or not edge_res:
        return None

    v_idx = int(edge_res["v_idx"])
    u_idx = int(edge_res["u_idx"])
    sample_idx = int(edge_res["sample_idx"])
    log_path = _parameter_log_path(log_root, layer_id, short_name, sample_idx, v_idx, u_idx)
    seq_idx = int(edge_res["seq_idx"])
    curv = float(edge_res["curv"])
    lpf_curv = edge_res.get("lpf_curv")
    metric_score = edge_res.get("metric_score")
    metric_prev_score = edge_res.get("metric_prev_score")
    metric_next_score = edge_res.get("metric_next_score")
    weight_magnitude = edge_res.get("weight_magnitude")
    prev_cost_sum = edge_res.get("top_prev_to_target_sum")
    next_cost_sum = edge_res.get("top_next_from_source_sum")
    neighbor_cost_sum = edge_res.get("top_neighbor_cost_sum")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"parameter: u={u_idx}, v={v_idx}\n"
            f"parameter_seq: u={u_idx}, v={v_idx}, seq_idx={seq_idx}\n"
            f"sample_idx: {sample_idx}\n"
            f"seq_idx: {seq_idx}\n"
            f"edge_index: ({u_idx}, {v_idx})\n"
            f"len(mu): {edge_res['mu_len']}\n"
            f"len(nu): {edge_res['nu_len']}\n"
            "neighbor_note: mu: in neighbor, nu: out neighbor\n"
            f"W_dist: {edge_res['w_dist']}\n"
            f"sp_uv: {edge_res['sp_uv']}\n"
            f"curv: {curv}\n"
        )
        _write_array_block(f, "in_neighbors", edge_res.get("in_neighbors"))
        _write_array_block(f, "out_neighbors", edge_res.get("out_neighbors"))
        _write_array_block(f, "mu", edge_res.get("mu"))
        _write_array_block(f, "nu", edge_res.get("nu"))
        _write_array_block(f, "cost_matrix", edge_res.get("cost_matrix"))
        _write_array_block(f, "prev_neighbors_to_u_cost", edge_res.get("prev_neighbors_to_u_cost"))
        _write_array_block(f, "v_to_out_neighbors_cost", edge_res.get("v_to_out_neighbors_cost"))
        _write_array_block(f, "prev_neighbors_to_v_cost", edge_res.get("prev_neighbors_to_v_cost"))
        _write_array_block(f, "u_to_out_neighbors_cost", edge_res.get("u_to_out_neighbors_cost"))
        _write_array_block(
            f,
            "prev_neighbors_to_out_neighbors_cost",
            edge_res.get("prev_neighbors_to_out_neighbors_cost"),
        )
        if lpf_curv is not None:
            f.write(f"lpf_curv: {float(lpf_curv)}\n")
        if metric_score is not None:
            f.write(
                f"metric_score: {float(metric_score)}\n"
                f"metric_prev_score: {float(metric_prev_score)}\n"
                f"metric_next_score: {float(metric_next_score)}\n"
            )
        if weight_magnitude is not None:
            f.write(f"weight_magnitude: {float(weight_magnitude)}\n")
        if neighbor_cost_sum is not None:
            f.write(
                f"top_prev_to_target_sum: {float(prev_cost_sum)}\n"
                f"top_next_from_source_sum: {float(next_cost_sum)}\n"
                f"top_neighbor_cost_sum: {float(neighbor_cost_sum)}\n"
                "top_prev_to_target_nodes:\n"
            )
            for item in edge_res.get("top_prev_to_target_nodes", []):
                f.write(
                    "  "
                    f"node_idx={int(item['node_idx'])}, "
                    f"probability={float(item['probability'])}, "
                    f"cost_to_target={float(item['cost_to_target'])}, "
                    f"weighted_cost={float(item['weighted_cost'])}\n"
                )
            f.write("top_next_from_source_nodes:\n")
            for item in edge_res.get("top_next_from_source_nodes", []):
                f.write(
                    "  "
                    f"node_idx={int(item['node_idx'])}, "
                    f"probability={float(item['probability'])}, "
                    f"cost_from_source={float(item['cost_from_source'])}, "
                    f"weighted_cost={float(item['weighted_cost'])}\n"
                )
        f.write("\n")

    points = _PARAMETER_CURVATURE_POINTS.setdefault(log_path, [])
    points.append((seq_idx, curv))

    if metric_score is not None:
        metric_points = _PARAMETER_METRIC_POINTS.setdefault(log_path, [])
        metric_points.append(
            (
                seq_idx,
                float(metric_score),
                float(metric_prev_score),
                float(metric_next_score),
                curv,
            )
        )

    if neighbor_cost_sum is not None:
        neighbor_points = _PARAMETER_NEIGHBOR_COST_POINTS.setdefault(log_path, [])
        neighbor_points.append(
            (
                seq_idx,
                float(prev_cost_sum),
                float(next_cost_sum),
                float(neighbor_cost_sum),
                curv,
            )
        )
    return log_path


def _refresh_parameter_artifacts(log_path):
    if log_path is None or not os.path.exists(log_path):
        return

    curvature_points = _PARAMETER_CURVATURE_POINTS.get(log_path)
    metric_points = _PARAMETER_METRIC_POINTS.get(log_path)
    neighbor_points = _PARAMETER_NEIGHBOR_COST_POINTS.get(log_path)

    write_parameter_metric_statistics(
        log_path,
        metric_points=metric_points,
        curvature_points=curvature_points,
    )
    draw_parameter_curvature(log_path, points=curvature_points)
    draw_parameter_curvature_lpf_comparison(log_path)
    draw_parameter_metric_curvature_comparison(log_path, points=metric_points)
    draw_parameter_neighbor_cost_comparison(log_path, points=neighbor_points)


def _sort_parameter_log_entries(log_path):
    if log_path is None or not os.path.exists(log_path):
        return

    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_blocks = [block.strip() for block in content.split("\n\n") if block.strip()]
    if len(raw_blocks) <= 1:
        return

    parsed_blocks = []
    for block in raw_blocks:
        seq_idx = None
        for line in block.splitlines():
            if line.startswith("seq_idx:"):
                seq_idx = int(line.split(":", 1)[1].strip())
                break
        if seq_idx is None:
            return
        parsed_blocks.append((seq_idx, block))

    parsed_blocks.sort(key=lambda item: item[0])

    with open(log_path, "w", encoding="utf-8") as f:
        for _, block in parsed_blocks:
            f.write(block)
            f.write("\n\n")


def finalize_parameter_detail_log(log_path):
    _sort_parameter_log_entries(log_path)
    _refresh_parameter_artifacts(log_path)
    _ACTIVE_PARAMETER_LOGS.discard(log_path)


def _read_parameter_curvature_points(log_path):
    points = []
    seq_idx = None

    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("seq_idx:"):
                seq_idx = int(line.split(":", 1)[1].strip())
            elif line.startswith("curv:") and seq_idx is not None:
                curv = float(line.split(":", 1)[1].strip())
                points.append((seq_idx, curv))
                seq_idx = None

    points.sort(key=lambda item: item[0])
    return points


def _read_parameter_neighbor_cost_points(log_path):
    points = []
    seq_idx = None
    curv = None
    prev_cost_sum = None
    next_cost_sum = None
    neighbor_cost_sum = None

    def maybe_append():
        if (
            seq_idx is not None
            and curv is not None
            and prev_cost_sum is not None
            and next_cost_sum is not None
            and neighbor_cost_sum is not None
        ):
            points.append((seq_idx, prev_cost_sum, next_cost_sum, neighbor_cost_sum, curv))

    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                maybe_append()
                seq_idx = None
                curv = None
                prev_cost_sum = None
                next_cost_sum = None
                neighbor_cost_sum = None
            elif line.startswith("seq_idx:"):
                seq_idx = int(line.split(":", 1)[1].strip())
            elif line.startswith("curv:"):
                curv = float(line.split(":", 1)[1].strip())
            elif line.startswith("top_prev_to_target_sum:"):
                prev_cost_sum = float(line.split(":", 1)[1].strip())
            elif line.startswith("top_next_from_source_sum:"):
                next_cost_sum = float(line.split(":", 1)[1].strip())
            elif line.startswith("top_neighbor_cost_sum:"):
                neighbor_cost_sum = float(line.split(":", 1)[1].strip())

    maybe_append()
    points.sort(key=lambda item: item[0])
    return points


def _read_parameter_metric_points(log_path):
    points = []
    seq_idx = None
    curv = None
    metric_score = None
    metric_prev_score = None
    metric_next_score = None

    def maybe_append():
        if (
            seq_idx is not None
            and curv is not None
            and metric_score is not None
            and metric_prev_score is not None
            and metric_next_score is not None
        ):
            points.append((seq_idx, metric_score, metric_prev_score, metric_next_score, curv))

    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                maybe_append()
                seq_idx = None
                curv = None
                metric_score = None
                metric_prev_score = None
                metric_next_score = None
            elif line.startswith("seq_idx:"):
                seq_idx = int(line.split(":", 1)[1].strip())
            elif line.startswith("curv:"):
                curv = float(line.split(":", 1)[1].strip())
            elif line.startswith("metric_score:"):
                metric_score = float(line.split(":", 1)[1].strip())
            elif line.startswith("metric_prev_score:"):
                metric_prev_score = float(line.split(":", 1)[1].strip())
            elif line.startswith("metric_next_score:"):
                metric_next_score = float(line.split(":", 1)[1].strip())

    maybe_append()
    points.sort(key=lambda item: item[0])
    return points


def _read_parameter_lpf_points(log_path):
    points = []
    seq_idx = None
    curv = None
    lpf_curv = None

    def maybe_append():
        if seq_idx is not None and curv is not None and lpf_curv is not None:
            points.append((seq_idx, curv, lpf_curv))

    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                maybe_append()
                seq_idx = None
                curv = None
                lpf_curv = None
            elif line.startswith("seq_idx:"):
                seq_idx = int(line.split(":", 1)[1].strip())
            elif line.startswith("curv:"):
                curv = float(line.split(":", 1)[1].strip())
            elif line.startswith("lpf_curv:"):
                lpf_curv = float(line.split(":", 1)[1].strip())

    maybe_append()
    points.sort(key=lambda item: item[0])
    return points


def _safe_pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() <= 1:
        return float("nan")
    x = x[finite]
    y = y[finite]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def write_parameter_metric_statistics(log_path, metric_points=None, curvature_points=None):
    if log_path is None or not os.path.exists(log_path):
        return None

    if metric_points is None:
        metric_points = _PARAMETER_METRIC_POINTS.get(log_path)
    if metric_points is None:
        metric_points = _read_parameter_metric_points(log_path)

    if curvature_points is None:
        curvature_points = _PARAMETER_CURVATURE_POINTS.get(log_path)
    if curvature_points is None:
        curvature_points = _read_parameter_curvature_points(log_path)

    if not metric_points and not curvature_points:
        return None

    stats_path = os.path.splitext(log_path)[0] + "_statistics.txt"

    metric_seq = np.asarray([item[0] for item in metric_points], dtype=np.int64) if metric_points else np.asarray([], dtype=np.int64)
    metric_total = np.asarray([item[1] for item in metric_points], dtype=np.float64) if metric_points else np.asarray([], dtype=np.float64)
    metric_prev = np.asarray([item[2] for item in metric_points], dtype=np.float64) if metric_points else np.asarray([], dtype=np.float64)
    metric_next = np.asarray([item[3] for item in metric_points], dtype=np.float64) if metric_points else np.asarray([], dtype=np.float64)
    metric_curv = np.asarray([item[4] for item in metric_points], dtype=np.float64) if metric_points else np.asarray([], dtype=np.float64)
    curv_only = np.asarray([item[1] for item in curvature_points], dtype=np.float64) if curvature_points else np.asarray([], dtype=np.float64)

    def finite_summary(name, values):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return [
                f"{name}_count: 0",
                f"{name}_min: nan",
                f"{name}_max: nan",
                f"{name}_mean: nan",
            ]
        return [
            f"{name}_count: {int(finite.size)}",
            f"{name}_min: {float(finite.min()):.8f}",
            f"{name}_max: {float(finite.max()):.8f}",
            f"{name}_mean: {float(finite.mean()):.8f}",
        ]

    lines = [
        f"log_path: {log_path}",
        f"unique_seq_count: {int(len(np.unique(metric_seq)) if metric_seq.size else 0)}",
    ]
    lines.extend(finite_summary("curvature", curv_only))
    lines.extend(finite_summary("metric_score", metric_total))
    lines.extend(finite_summary("metric_prev_score", metric_prev))
    lines.extend(finite_summary("metric_next_score", metric_next))
    lines.append(f"pearson_metric_score_curvature: {_safe_pearson(metric_total, metric_curv):.8f}")
    lines.append(f"pearson_metric_prev_curvature: {_safe_pearson(metric_prev, metric_curv):.8f}")
    lines.append(f"pearson_metric_next_curvature: {_safe_pearson(metric_next, metric_curv):.8f}")

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return stats_path


def _get_pyplot(log_path):
    global _PYPLOT
    if _PYPLOT is not None:
        return _PYPLOT

    try:
        mpl_config_dir = os.path.join("/tmp", "matplotlib")
        os.makedirs(mpl_config_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        marker_path = os.path.splitext(log_path)[0] + "_curvature_plot_error.txt"
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(f"Could not draw curvature plot: {exc}\n")
        return None

    _PYPLOT = plt
    return _PYPLOT


def draw_parameter_curvature(log_path, points=None):
    if log_path is None or not os.path.exists(log_path):
        return None

    if points is None:
        points = _PARAMETER_CURVATURE_POINTS.get(log_path)
    if points is None:
        points = _read_parameter_curvature_points(log_path)
    if not points:
        return None

    plt = _get_pyplot(log_path)
    if plt is None:
        return None

    ordered_points = sorted(points, key=lambda item: item[0])
    xs = np.asarray([seq_idx for seq_idx, _ in ordered_points], dtype=np.int64)
    ys = np.asarray([curv for _, curv in ordered_points], dtype=np.float64)
    finite = np.isfinite(ys)
    if not finite.any():
        return None

    plot_path = os.path.splitext(log_path)[0] + "_curvature.png"
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    ax.plot(xs[finite], ys[finite], marker="o", linewidth=0.9, markersize=1.8)
    ax.set_xlabel("seq id")
    ax.set_ylabel("curvature")
    ax.set_title(os.path.basename(os.path.dirname(log_path)), fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    return plot_path


def draw_parameter_neighbor_cost_comparison(log_path, points=None):
    if log_path is None or not os.path.exists(log_path):
        return None

    if points is None:
        points = _PARAMETER_NEIGHBOR_COST_POINTS.get(log_path)
    if points is None:
        points = _read_parameter_neighbor_cost_points(log_path)
    if not points:
        return None

    plt = _get_pyplot(log_path)
    if plt is None:
        return None

    ordered_points = sorted(points, key=lambda item: item[0])
    xs = np.asarray([item[0] for item in ordered_points], dtype=np.int64)
    prev_sums = np.asarray([item[1] for item in ordered_points], dtype=np.float64)
    next_sums = np.asarray([item[2] for item in ordered_points], dtype=np.float64)
    total_sums = np.asarray([item[3] for item in ordered_points], dtype=np.float64)
    curv = np.asarray([item[4] for item in ordered_points], dtype=np.float64)

    finite_seq = np.isfinite(prev_sums) & np.isfinite(next_sums) & np.isfinite(total_sums)
    finite_corr = finite_seq & np.isfinite(curv)
    if not finite_seq.any():
        return None

    corr_text = ""
    if finite_corr.sum() > 1:
        finite_total = total_sums[finite_corr]
        finite_curv = curv[finite_corr]
        if np.std(finite_total) > 0 and np.std(finite_curv) > 0:
            corr = float(np.corrcoef(finite_total, finite_curv)[0, 1])
            corr_text = f", Pearson r={corr:.3f}"

    plot_path = os.path.splitext(log_path)[0] + "_neighbor_cost_curvature_compare.png"
    fig, (ax_seq, ax_scatter) = plt.subplots(2, 1, figsize=(4.6, 4.8))

    ax_seq.plot(
        xs[finite_seq],
        prev_sums[finite_seq],
        marker="o",
        linewidth=0.9,
        markersize=1.8,
        label="prev top10",
    )
    ax_seq.plot(
        xs[finite_seq],
        next_sums[finite_seq],
        marker="s",
        linewidth=0.9,
        markersize=1.8,
        label="next top10",
    )
    ax_seq.plot(
        xs[finite_seq],
        total_sums[finite_seq],
        marker="^",
        linewidth=1.1,
        markersize=2.0,
        label="sum",
    )
    ax_seq.set_xlabel("seq id")
    ax_seq.set_ylabel("top10 prob * cost")
    ax_seq.grid(True, alpha=0.3)
    ax_seq.legend(fontsize=6, loc="best")

    finite_curv = np.isfinite(curv)
    if finite_curv.any():
        ax_curv = ax_seq.twinx()
        ax_curv.plot(
            xs[finite_curv],
            curv[finite_curv],
            color="tab:red",
            marker="x",
            linewidth=0.9,
            markersize=2.2,
            label="curvature",
        )
        ax_curv.set_ylabel("curvature", color="tab:red")
        ax_curv.tick_params(axis="y", labelcolor="tab:red")

    ax_seq.set_title(
        f"{os.path.basename(os.path.dirname(log_path))}{corr_text}",
        fontsize=8,
    )

    if finite_corr.any():
        ax_scatter.scatter(total_sums[finite_corr], curv[finite_corr], s=10, alpha=0.8)
    ax_scatter.set_xlabel("top10 summed prob * cost")
    ax_scatter.set_ylabel("curvature")
    ax_scatter.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def draw_parameter_metric_curvature_comparison(log_path, points=None):
    if log_path is None or not os.path.exists(log_path):
        return None

    if points is None:
        points = _PARAMETER_METRIC_POINTS.get(log_path)
    if points is None:
        points = _read_parameter_metric_points(log_path)
    if not points:
        return None

    plt = _get_pyplot(log_path)
    if plt is None:
        return None

    ordered_points = sorted(points, key=lambda item: item[0])
    xs = np.asarray([item[0] for item in ordered_points], dtype=np.int64)
    metric_score = np.asarray([item[1] for item in ordered_points], dtype=np.float64)
    metric_prev_score = np.asarray([item[2] for item in ordered_points], dtype=np.float64)
    metric_next_score = np.asarray([item[3] for item in ordered_points], dtype=np.float64)
    curv = np.asarray([item[4] for item in ordered_points], dtype=np.float64)

    finite_metric = (
        np.isfinite(metric_score)
        & np.isfinite(metric_prev_score)
        & np.isfinite(metric_next_score)
    )
    finite_corr = finite_metric & np.isfinite(curv)
    if not finite_metric.any():
        return None

    corr_text = ""
    if finite_corr.sum() > 1:
        finite_metric_score = metric_score[finite_corr]
        finite_curv = curv[finite_corr]
        if np.std(finite_metric_score) > 0 and np.std(finite_curv) > 0:
            corr = float(np.corrcoef(finite_metric_score, finite_curv)[0, 1])
            corr_text = f", Pearson r={corr:.3f}"

    plot_path = os.path.splitext(log_path)[0] + "_metric_curvature_compare.png"
    fig, (ax_seq, ax_scatter) = plt.subplots(2, 1, figsize=(4.6, 4.8))

    ax_seq.plot(
        xs[finite_metric],
        metric_prev_score[finite_metric],
        marker="o",
        linewidth=0.9,
        markersize=1.8,
        label="metric prev",
    )
    ax_seq.plot(
        xs[finite_metric],
        metric_next_score[finite_metric],
        marker="s",
        linewidth=0.9,
        markersize=1.8,
        label="metric next",
    )
    ax_seq.plot(
        xs[finite_metric],
        metric_score[finite_metric],
        marker="^",
        linewidth=1.1,
        markersize=2.0,
        label="metric sum",
    )
    ax_seq.set_xlabel("seq id")
    ax_seq.set_ylabel("metric score")
    ax_seq.grid(True, alpha=0.3)
    ax_seq.legend(fontsize=6, loc="best")

    finite_curv = np.isfinite(curv)
    if finite_curv.any():
        ax_curv = ax_seq.twinx()
        ax_curv.plot(
            xs[finite_curv],
            curv[finite_curv],
            color="tab:red",
            marker="x",
            linewidth=0.9,
            markersize=2.2,
            label="curvature",
        )
        ax_curv.set_ylabel("curvature", color="tab:red")
        ax_curv.tick_params(axis="y", labelcolor="tab:red")

    ax_seq.set_title(
        f"{os.path.basename(os.path.dirname(log_path))}{corr_text}",
        fontsize=8,
    )

    if finite_corr.any():
        ax_scatter.scatter(metric_score[finite_corr], curv[finite_corr], s=10, alpha=0.8)
    ax_scatter.set_xlabel("metric score")
    ax_scatter.set_ylabel("curvature")
    ax_scatter.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def draw_parameter_curvature_lpf_comparison(log_path, points=None):
    if log_path is None or not os.path.exists(log_path):
        return None

    if points is None:
        points = _read_parameter_lpf_points(log_path)
    if not points:
        return None

    plt = _get_pyplot(log_path)
    if plt is None:
        return None

    ordered_points = sorted(points, key=lambda item: item[0])
    xs = np.asarray([item[0] for item in ordered_points], dtype=np.int64)
    raw_curv = np.asarray([item[1] for item in ordered_points], dtype=np.float64)
    lpf_curv = np.asarray([item[2] for item in ordered_points], dtype=np.float64)

    finite_raw = np.isfinite(raw_curv)
    finite_lpf = np.isfinite(lpf_curv)
    if not finite_raw.any() and not finite_lpf.any():
        return None

    plot_path = os.path.splitext(log_path)[0] + "_lpf_curvature_compare.png"
    fig, ax = plt.subplots(figsize=(4.2, 2.6))

    if finite_raw.any():
        ax.plot(
            xs[finite_raw],
            raw_curv[finite_raw],
            marker="o",
            linewidth=0.9,
            markersize=1.8,
            label="raw",
        )
    if finite_lpf.any():
        ax.plot(
            xs[finite_lpf],
            lpf_curv[finite_lpf],
            marker="s",
            linewidth=0.9,
            markersize=1.8,
            label="lpf",
        )

    ax.set_xlabel("seq id")
    ax.set_ylabel("curvature")
    ax.set_title(os.path.basename(os.path.dirname(log_path)), fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="best")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def draw_cached_parameter_curvatures(log_root=None):
    drawn_paths = []
    root_path = os.path.abspath(log_root) if log_root is not None else None
    seen_log_paths = set()

    for log_path, points in list(_PARAMETER_CURVATURE_POINTS.items()):
        abs_log_path = os.path.abspath(log_path)
        if root_path is not None and os.path.commonpath([root_path, abs_log_path]) != root_path:
            continue

        seen_log_paths.add(abs_log_path)
        plot_path = draw_parameter_curvature(log_path, points=points)
        if plot_path is not None:
            drawn_paths.append(plot_path)
        stats_path = write_parameter_metric_statistics(
            log_path,
            metric_points=_PARAMETER_METRIC_POINTS.get(log_path),
            curvature_points=points,
        )
        if stats_path is not None:
            drawn_paths.append(stats_path)
        metric_plot_path = draw_parameter_metric_curvature_comparison(
            log_path,
            points=_PARAMETER_METRIC_POINTS.get(log_path),
        )
        if metric_plot_path is not None:
            drawn_paths.append(metric_plot_path)
        neighbor_plot_path = draw_parameter_neighbor_cost_comparison(
            log_path,
            points=_PARAMETER_NEIGHBOR_COST_POINTS.get(log_path),
        )
        if neighbor_plot_path is not None:
            drawn_paths.append(neighbor_plot_path)
        _PARAMETER_CURVATURE_POINTS.pop(log_path, None)
        _PARAMETER_METRIC_POINTS.pop(log_path, None)
        _PARAMETER_NEIGHBOR_COST_POINTS.pop(log_path, None)

    if root_path is not None and os.path.isdir(root_path):
        for current_root, _, file_names in os.walk(root_path):
            for file_name in file_names:
                is_old_log_name = file_name == "parameter_log.txt"
                is_new_log_name = (
                    file_name.startswith("parameter_log_example_")
                    and file_name.endswith(".txt")
                    and not file_name.endswith("_statistics.txt")
                )
                if not (is_old_log_name or is_new_log_name):
                    continue

                log_path = os.path.join(current_root, file_name)
                abs_log_path = os.path.abspath(log_path)
                if abs_log_path in seen_log_paths:
                    continue

                plot_path = draw_parameter_curvature(log_path)
                if plot_path is not None:
                    drawn_paths.append(plot_path)
                stats_path = write_parameter_metric_statistics(log_path)
                if stats_path is not None:
                    drawn_paths.append(stats_path)
                metric_plot_path = draw_parameter_metric_curvature_comparison(log_path)
                if metric_plot_path is not None:
                    drawn_paths.append(metric_plot_path)
                lpf_plot_path = draw_parameter_curvature_lpf_comparison(log_path)
                if lpf_plot_path is not None:
                    drawn_paths.append(lpf_plot_path)
                neighbor_plot_path = draw_parameter_neighbor_cost_comparison(log_path)
                if neighbor_plot_path is not None:
                    drawn_paths.append(neighbor_plot_path)

    return drawn_paths


def append_final_min_curvature_summary(
    analysis_path,
    sample_idx,
    curvature,
    avg_mu_len,
    avg_nu_len,
    curr_dist_finite_edges,
    curr_dist_infinite_edges,
    cost_has_inf,
    cost_inf_count,
    runtime_sec=None,
):
    if torch.is_tensor(curvature):
        curvature = curvature.detach().cpu().numpy()
    else:
        curvature = np.asarray(curvature)

    summary = _summarize_curvatures(curvature)
    runtime_sec = float(runtime_sec) if runtime_sec is not None else float("nan")

    with open(analysis_path, "r", encoding="utf-8") as f:
        content = _strip_aggregate_footer(f.read())

    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(content)

    with open(analysis_path, "a", encoding="utf-8") as f:
        f.write(
            f"example {int(sample_idx)}: "
            f"curr_dist_finite_edges={int(curr_dist_finite_edges)}, "
            f"curr_dist_infinite_edges={int(curr_dist_infinite_edges)}, "
            f"finite_entries={summary['finite_count']}, "
            f"positive_edges={summary['positive_count']}, "
            f"negative_edges={summary['negative_count']}, "
            f"zero_edges={summary['zero_count']}, "
            f"avg_len(mu)={float(avg_mu_len):.8f}, "
            f"avg_len(nu)={float(avg_nu_len):.8f}, "
            f"cost_has_inf={bool(cost_has_inf)}, "
            f"cost_inf_count={int(cost_inf_count)}, "
            f"runtime_sec={float(runtime_sec):.6f}, "
            f"min_curv={summary['min_curv']:.8f}, "
            f"max_curv={summary['max_curv']:.8f}\n"
        )


def append_final_curvature_overall(layer_id, short_name, curvature, seq_len, dataset_name, analysis_dir=None):
    if torch.is_tensor(curvature):
        curvature = curvature.detach().cpu().numpy()
    else:
        curvature = np.asarray(curvature)

    analysis_path = _analysis_file_path(
        layer_id,
        short_name,
        seq_len,
        dataset_name,
        analysis_dir=analysis_dir,
    )
    if not os.path.exists(analysis_path):
        header = (
            f"layer_id: {int(layer_id)}\n"
            f"op_name: {short_name}\n"
            f"dataset_name: {dataset_name}\n"
            f"curvature_shape: {tuple(curvature.shape)}\n"
            f"seq_len: {int(seq_len)}\n"
            "\n"
        )
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(header)

    summary = _summarize_curvatures(curvature)
    _write_final_curvature_footer(analysis_path, summary)
