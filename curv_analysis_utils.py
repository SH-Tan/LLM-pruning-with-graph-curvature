import os
import numpy as np
import torch


_AGGREGATE_MARKER = "overall:\n"


def _analysis_file_path(layer_id, short_name, seq_len, dataset_name):
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


def start_curvature_analysis(layer_id, short_name, sample_idx, curvature_shape, seq_len, dataset_name):
    analysis_path = _analysis_file_path(layer_id, short_name, seq_len, dataset_name)
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
):
    if torch.is_tensor(curvature):
        curvature = curvature.detach().cpu().numpy()
    else:
        curvature = np.asarray(curvature)

    summary = _summarize_curvatures(curvature)

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
            f"min_curv={summary['min_curv']:.8f}, "
            f"max_curv={summary['max_curv']:.8f}\n"
        )


def append_final_curvature_overall(layer_id, short_name, curvature, seq_len, dataset_name):
    if torch.is_tensor(curvature):
        curvature = curvature.detach().cpu().numpy()
    else:
        curvature = np.asarray(curvature)

    analysis_path = _analysis_file_path(layer_id, short_name, seq_len, dataset_name)
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
