import numpy as np

from curv_dtype_utils import curvature_np_dtype
from curv_distribution_utils import _build_node_distribution_row_from_values

_SHARED_PREV_SCORE = None
_SHARED_NEXT_SCORE = None
_SHARED_SEQ_LEN = 1
_SHARED_TOP_K = 5
_SHARED_SEQ_SELECT = "top"
_OT_NEIGHBOR_TOP_K = 10


class QKOTNextScore:
    __slots__ = ("head_dim", "head_count", "codes", "values")

    def __init__(self, head_dim, head_count, codes, values):
        self.head_dim = int(head_dim)
        self.head_count = int(head_count)
        self.codes = np.asarray(codes, dtype=np.int64)
        self.values = np.asarray(values, dtype=curvature_np_dtype())

    def get_for_edge(self, u_idx, v_idx):
        head_idx = int(v_idx) // self.head_dim
        code = int(u_idx) * self.head_count + head_idx
        pos = int(np.searchsorted(self.codes, code))
        if pos < self.codes.size and int(self.codes[pos]) == code:
            return self.values[pos]
        return None


def is_compact_score(score):
    return isinstance(score, QKOTNextScore)


def set_shared_metric_state(prev_score, next_score, seq_len, top_k, seq_select="top"):
    global _SHARED_PREV_SCORE, _SHARED_NEXT_SCORE
    global _SHARED_SEQ_LEN, _SHARED_TOP_K, _SHARED_SEQ_SELECT

    _SHARED_PREV_SCORE = prev_score
    _SHARED_NEXT_SCORE = next_score
    _SHARED_SEQ_LEN = int(seq_len)
    _SHARED_TOP_K = int(top_k)
    _SHARED_SEQ_SELECT = seq_select


def reset_shared_metric_state():
    set_shared_metric_state(None, None, 1, 5, "top")


def _as_seq_distribution_matrix(distribution, seq_len, short_name=None):
    if distribution is None:
        return None
    if type(distribution) is list:
        return None

    arr = np.asarray(distribution, dtype=curvature_np_dtype())
    if short_name in {"q_proj", "k_proj"} and arr.ndim == 3:
        return arr.transpose(1, 0, 2).reshape(arr.shape[1], -1)[:seq_len]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr.transpose(1, 0, 2).reshape(arr.shape[1], -1)
    if arr.ndim != 2:
        return None
    return arr[:seq_len]


def _fill_negative_distribution_entries(distribution, alpha):
    if distribution is None:
        return None

    distribution = np.asarray(distribution, dtype=curvature_np_dtype()).copy()
    neg_mask = distribution == -1.0
    if not np.any(neg_mask):
        return distribution

    non_zero_count = np.count_nonzero(distribution, axis=1)
    fill_values = np.zeros((distribution.shape[0], 1), dtype=curvature_np_dtype())
    valid_rows = non_zero_count > 0
    fill_values[valid_rows, 0] = (1.0 - alpha) / non_zero_count[valid_rows]
    distribution[neg_mask] = np.broadcast_to(fill_values, distribution.shape)[neg_mask]
    return distribution


def _min_reduce_blocks(blocks):
    blocks = [np.asarray(block, dtype=curvature_np_dtype()) for block in blocks]
    if not blocks:
        return None
    if len(blocks) == 1:
        return blocks[0]
    return np.minimum.reduce(blocks)


def _distribution_cost_score(
    distribution,
    cost,
    top_k=_OT_NEIGHBOR_TOP_K,
    transpose_cost=False,
    columns=None,
):
    if distribution is None and cost is None:
        return 0.0
    if distribution is None or cost is None:
        print("Error: both distribution and cost must be present")
        return None

    distribution = np.asarray(distribution, dtype=curvature_np_dtype())
    cost = np.asarray(cost, dtype=curvature_np_dtype())

    if transpose_cost:
        cost = cost.T

    if distribution.shape[1] != cost.shape[0]:
        return None
    if columns is not None:
        columns = np.asarray(columns, dtype=np.int64).reshape(-1)
        if columns.size == 0:
            return {}
        cost = cost[:, columns]

    distribution = np.where(np.isfinite(distribution), distribution, 0.0)
    cost = np.where(np.isfinite(cost), cost, 0.0)

    seq_count, node_count = distribution.shape
    out_count = cost.shape[1]

    top_k = min(top_k, node_count)
    if top_k <= 0:
        score = np.zeros((seq_count, out_count), dtype=curvature_np_dtype())
        return score if columns is None else {
            int(col): score[:, idx] for idx, col in enumerate(columns)
        }

    top_idx = np.argpartition(-distribution, top_k - 1, axis=1)[:, :top_k]
    top_weights = np.take_along_axis(distribution, top_idx, axis=1)
    gathered_cost = cost[top_idx]
    score = np.sum(top_weights[:, :, None] * gathered_cost, axis=1)
    return score if columns is None else {
        int(col): score[:, idx] for idx, col in enumerate(columns)
    }


def _per_u_residual_distribution(prev_node_values, seq_len, u_idx, residual_start, alpha):
    values = np.asarray(prev_node_values, dtype=curvature_np_dtype())
    residual_idx = int(residual_start) + int(u_idx)
    if values.ndim != 2 or residual_idx >= values.shape[1]:
        return None
    rows = [
        _build_node_distribution_row_from_values(
            np.concatenate((
                values[seq_idx, :residual_start],
                values[seq_idx, residual_idx:residual_idx + 1],
            )),
            alpha,
        )
        for seq_idx in range(min(int(seq_len), values.shape[0]))
    ]
    return np.asarray(rows, dtype=curvature_np_dtype())


def _distribution_cost_score_per_u_residual(
    seq_len,
    prev_node_values,
    cost,
    edge_indices,
    alpha,
    residual_start,
):
    if prev_node_values is None or cost is None or edge_indices is None:
        return None

    cost = np.asarray(cost, dtype=curvature_np_dtype())
    edges = np.asarray(edge_indices, dtype=np.int64).reshape(-1, 2)
    scores = {}
    dist_cache = {}
    for u_idx, v_idx in edges:
        u_idx = int(u_idx)
        v_idx = int(v_idx)
        dist = dist_cache.get(u_idx)
        if dist is None:
            dist = _per_u_residual_distribution(prev_node_values, seq_len, u_idx, residual_start, alpha)
            dist_cache[u_idx] = dist
        if dist is None:
            continue

        residual_idx = int(residual_start) + u_idx
        col_cost = cost[:residual_start, v_idx].reshape(-1, 1)
        if residual_idx < cost.shape[0]:
            col_cost = np.concatenate((col_cost, cost[residual_idx, v_idx].reshape(1, 1)), axis=0)

        score = _distribution_cost_score(dist, col_cost)
        score = _sanitize_score_values(score)
        if score is not None and not np.isscalar(score):
            score = np.asarray(score, dtype=curvature_np_dtype()).reshape(dist.shape[0], -1)[:, 0]
        scores[(v_idx, u_idx)] = score
    return scores


def _sanitize_score_values(score, seq_len=None):
    if score is None:
        return None
    if is_compact_score(score):
        return score
    if isinstance(score, dict):
        return {
            key: _sanitize_score_values(values, seq_len=seq_len)
            for key, values in score.items()
        }
    if np.isscalar(score):
        value = float(score)
        if not np.isfinite(value):
            value = 0.0
        if seq_len is None:
            return value
        return np.full((int(seq_len),), value, dtype=curvature_np_dtype())

    score = np.asarray(score, dtype=curvature_np_dtype())
    np.nan_to_num(score, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return score


def _build_qk_ot_next_top_distribution(next_out_distribution, alpha):
    if next_out_distribution is None:
        return None

    dist = np.asarray(next_out_distribution, dtype=curvature_np_dtype())
    if dist.ndim != 3:
        return None

    head_count, seq_len, block_width = dist.shape
    dist_2d = _fill_negative_distribution_entries(dist.reshape(-1, block_width), alpha)
    dist_2d = np.where(np.isfinite(dist_2d), dist_2d, 0.0)
    dist = dist_2d.reshape(head_count, seq_len, block_width)

    top_k = min(_OT_NEIGHBOR_TOP_K, block_width)
    if top_k <= 0:
        return None

    top_idx = np.argpartition(-dist, top_k - 1, axis=2)[:, :, :top_k]
    top_weights = np.take_along_axis(dist, top_idx, axis=2)
    return top_idx, top_weights, block_width


def _build_qk_ot_next_score(qk_top_distribution, next_cost, edge_indices, curr_out_dim):
    if qk_top_distribution is None or next_cost is None or edge_indices is None:
        return None

    dtype = curvature_np_dtype()

    top_idx, top_weights, block_width = qk_top_distribution

    top_idx = np.asarray(top_idx)
    top_weights = np.asarray(top_weights, dtype=dtype)
    next_cost = np.asarray(next_cost, dtype=dtype)
    edges = np.asarray(edge_indices, dtype=np.int64).reshape(-1, 2)

    if (
        top_idx.ndim != 3
        or top_weights.ndim != 3
        or next_cost.ndim != 2
        or edges.size == 0
    ):
        return None

    head_count, seq_len, _ = top_idx.shape

    curr_out_dim = int(curr_out_dim)
    if curr_out_dim <= 0 or curr_out_dim % head_count != 0:
        return None

    head_dim = curr_out_dim // head_count

    u_all = edges[:, 0]
    v_all = edges[:, 1]
    heads = v_all // head_dim

    valid = (
        (heads >= 0)
        & (heads < head_count)
        & (u_all >= 0)
        & (u_all < next_cost.shape[0])
        & ((heads + 1) * block_width <= next_cost.shape[1])
    )

    if not np.any(valid):
        return QKOTNextScore(
            head_dim,
            head_count,
            np.empty((0,), dtype=np.int64),
            np.empty((0, seq_len), dtype=dtype),
        )

    u_valid = u_all[valid]
    heads_valid = heads[valid]

    # Faster replacement for:
    # np.unique(np.stack((u_valid, heads_valid), axis=1), axis=0)
    #
    # Since head is in [0, head_count), encode pair as:
    # code = u * head_count + head
    pair_codes = u_valid * head_count + heads_valid
    unique_codes = np.unique(pair_codes)

    unique_u = unique_codes // head_count
    unique_heads = unique_codes % head_count

    # Sort unique pairs by head once.
    order = np.argsort(unique_heads, kind="stable")
    score_values = np.zeros((unique_codes.shape[0], seq_len), dtype=dtype)

    # Tune this. Smaller = less memory, larger = faster.
    row_chunk_size = 256

    start = 0
    num_pairs = order.shape[0]

    while start < num_pairs:
        head_idx = int(unique_heads[order[start]])

        end = start + 1
        while end < num_pairs and int(unique_heads[order[end]]) == head_idx:
            end += 1

        pair_positions = order[start:end]
        u_indices_for_head = unique_u[pair_positions]

        cols = top_idx[head_idx] + head_idx * block_width
        weights = top_weights[head_idx]

        # Process rows in chunks to avoid huge temporary arrays.
        for chunk_start in range(0, u_indices_for_head.shape[0], row_chunk_size):
            chunk_end = min(chunk_start + row_chunk_size, u_indices_for_head.shape[0])
            u_chunk = u_indices_for_head[chunk_start:chunk_end]
            position_chunk = pair_positions[chunk_start:chunk_end]

            # Shape: (chunk_size, seq_len, top_k)
            costs = next_cost[u_chunk[:, None, None], cols[None, :, :]]

            # In-place, avoids an extra allocation from np.where.
            np.nan_to_num(costs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # Shape: (chunk_size, seq_len)
            chunk_scores = np.einsum(
                "usk,sk->us",
                costs,
                weights,
                optimize=True,
            ).astype(dtype, copy=False)

            np.nan_to_num(
                chunk_scores,
                copy=False,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            score_values[position_chunk] = chunk_scores

        start = end

    return QKOTNextScore(head_dim, head_count, unique_codes, score_values)


def build_ot_neighbor_score_matrices(
    seq_len,
    sp,
    alpha,
    prev_in_distribution,
    next_out_distribution,
    short_name=None,
    edge_indices=None,
    residual_start=None,
    prev_node_values=None,
):
    prev_columns = None
    next_columns = None
    if edge_indices is not None:
        edges = np.asarray(edge_indices, dtype=np.int64).reshape(-1, 2)
        prev_columns = np.unique(edges[:, 1])
        next_columns = np.unique(edges[:, 0])

    prev_cost = _min_reduce_blocks(sp.get("prev_to_curr_out_all", {}).values())
    if prev_node_values is not None and residual_start is not None and edge_indices is not None:
        prev_score = _distribution_cost_score_per_u_residual(
            seq_len,
            prev_node_values,
            prev_cost,
            edge_indices,
            alpha,
            int(residual_start),
        )
    else:
        prev_dist = _as_seq_distribution_matrix(prev_in_distribution, seq_len)
        prev_dist = _fill_negative_distribution_entries(prev_dist, alpha)
        prev_score = _distribution_cost_score(
            prev_dist,
            prev_cost,
            columns=prev_columns,
        )
        del prev_dist
    prev_score = _sanitize_score_values(prev_score)
    del prev_cost

    next_cost = _min_reduce_blocks(sp.get("curr_in_to_next_all", {}).values())
    if short_name in {"q_proj", "k_proj"}:
        curr_dist = sp.get("curr_dist")
        curr_out_dim = np.asarray(curr_dist).shape[1] if curr_dist is not None else 0
        qk_top_distribution = _build_qk_ot_next_top_distribution(next_out_distribution, alpha)
        next_score = _build_qk_ot_next_score(
            qk_top_distribution,
            next_cost,
            edge_indices,
            curr_out_dim,
        )
        next_score = _sanitize_score_values(next_score)
    else:
        next_dist = _as_seq_distribution_matrix(next_out_distribution, seq_len, short_name=short_name)
        next_dist = _fill_negative_distribution_entries(next_dist, alpha)
        next_score = _distribution_cost_score(
            next_dist,
            next_cost,
            transpose_cost=True,
            columns=next_columns,
        )
        next_score = _sanitize_score_values(next_score)
        del next_dist
    del next_cost

    return prev_score, next_score


def _score_column(score, idx, edge_key=None):
    if score is None:
        return None
    if is_compact_score(score):
        if edge_key is None:
            return None
        values = score.get_for_edge(edge_key[0], edge_key[1])
        return _sanitize_score_values(values, seq_len=_SHARED_SEQ_LEN)
    if isinstance(score, dict):
        values = score.get(edge_key) if edge_key is not None else None
        if values is None:
            values = score.get(int(idx))
        return _sanitize_score_values(values, seq_len=_SHARED_SEQ_LEN)
    if np.isscalar(score):
        return _sanitize_score_values(score, seq_len=_SHARED_SEQ_LEN)
    if getattr(score, "ndim", 0) < 2:
        return None
    if score.shape[1] == 0:
        return None
    if idx < score.shape[1]:
        return _sanitize_score_values(score[:, idx])
    return None


def top_seq_for_edge(edge):
    metric, _, _ = score_components_for_edge(edge)

    np.nan_to_num(metric, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    total_count = _SHARED_SEQ_LEN if _SHARED_TOP_K == -1 else min(_SHARED_TOP_K, _SHARED_SEQ_LEN)
    if total_count <= 0:
        return []

    if _SHARED_TOP_K == -1:
        return [int(seq_idx) for seq_idx in range(_SHARED_SEQ_LEN)]

    if _SHARED_SEQ_SELECT == "median":
        median_value = float(np.median(metric))
        ordered = np.argsort(np.abs(metric - median_value), kind="stable")
    else:
        ordered = np.argsort(-metric, kind="stable")
    return [int(seq_idx) for seq_idx in ordered[:total_count]]


def selected_seq_count():
    total_count = _SHARED_SEQ_LEN if _SHARED_TOP_K == -1 else min(_SHARED_TOP_K, _SHARED_SEQ_LEN)
    return max(int(total_count), 0)


def score_components_for_edge(edge):
    u_idx, v_idx = (int(edge[0]), int(edge[1]))
    metric = np.zeros((_SHARED_SEQ_LEN,), dtype=curvature_np_dtype())

    prev_col = _score_column(_SHARED_PREV_SCORE, v_idx, edge_key=(v_idx, u_idx))
    next_col = _score_column(_SHARED_NEXT_SCORE, u_idx, edge_key=(u_idx, v_idx))
    if prev_col is not None:
        metric += prev_col
    if next_col is not None:
        metric += next_col

    metric = _sanitize_score_values(metric)

    return metric, prev_col, next_col
