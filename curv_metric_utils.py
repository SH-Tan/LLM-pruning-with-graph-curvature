import numpy as np

_SHARED_PREV_SCORE = None
_SHARED_NEXT_SCORE = None
_SHARED_SEQ_LEN = 1
_SHARED_TOP_K = 5


def set_shared_metric_state(prev_score, next_score, seq_len, top_k):
    global _SHARED_PREV_SCORE, _SHARED_NEXT_SCORE
    global _SHARED_SEQ_LEN, _SHARED_TOP_K

    _SHARED_PREV_SCORE = prev_score
    _SHARED_NEXT_SCORE = next_score
    _SHARED_SEQ_LEN = int(seq_len)
    _SHARED_TOP_K = int(top_k)


def reset_shared_metric_state():
    set_shared_metric_state(None, None, 1, 5)


def _as_seq_distribution_matrix(distribution, seq_len):
    if distribution is None:
        return None
    if type(distribution) is list:
        return None

    arr = np.asarray(distribution, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    return arr[:seq_len]


def _fill_negative_distribution_entries(distribution, alpha):
    if distribution is None:
        return None

    distribution = np.asarray(distribution, dtype=np.float64).copy()
    neg_mask = distribution == -1.0
    if not np.any(neg_mask):
        return distribution

    non_zero_count = np.count_nonzero(distribution, axis=1)
    fill_values = np.zeros((distribution.shape[0], 1), dtype=np.float64)
    valid_rows = non_zero_count > 0
    fill_values[valid_rows, 0] = (1.0 - alpha) / non_zero_count[valid_rows]
    distribution[neg_mask] = np.broadcast_to(fill_values, distribution.shape)[neg_mask]
    return distribution


def _distribution_cost_score(distribution, cost, top_k=10, transpose_cost=False):
    if distribution is None and cost is None:
        return 0.0
    if distribution is None or cost is None:
        print("Error: both distribution and cost must be present")
        return None

    distribution = np.asarray(distribution, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    if transpose_cost:
        cost = cost.T

    if distribution.shape[1] != cost.shape[0]:
        return None
    
    # sanitize
    distribution = np.where(np.isfinite(distribution), distribution, 0.0)
    cost = np.where(np.isfinite(cost), cost, 0.0)

    seq_count, node_count = distribution.shape
    out_count = cost.shape[1]

    top_k = min(top_k, node_count)
    if top_k <= 0:
        return np.zeros((seq_count, out_count), dtype=np.float64)

    # Step 1: get top-k indices per row
    top_idx = np.argpartition(-distribution, top_k - 1, axis=1)[:, :top_k]

    # Step 2: gather weights
    top_weights = np.take_along_axis(distribution, top_idx, axis=1)

    # Step 3: gather corresponding cost rows with shape [seq, k, out_node].
    gathered_cost = cost[top_idx]

    # Step 4: weighted sum across the selected node dimension.
    score = np.sum(top_weights[:, :, None] * gathered_cost, axis=1)
    return score


def _sanitize_score_values(score, seq_len=None):
    if score is None:
        return None
    if np.isscalar(score):
        value = float(score)
        if not np.isfinite(value):
            value = 0.0
        if seq_len is None:
            return value
        return np.full((int(seq_len),), value, dtype=np.float64)

    score = np.asarray(score, dtype=np.float64)
    np.nan_to_num(score, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return score


def build_neighbor_score_matrices(
    seq_len,
    sp,
    alpha,
    prev_in_distribution,
    next_out_distribution,
):
    prev_dist = _as_seq_distribution_matrix(prev_in_distribution, seq_len)
    prev_dist = _fill_negative_distribution_entries(prev_dist, alpha)
    prev_cost = sp.get("prev_to_curr_in")
    if prev_cost is not None:
        prev_cost = np.asarray(prev_cost, dtype=np.float64)
    prev_score = _distribution_cost_score(prev_dist, prev_cost)
    prev_score = _sanitize_score_values(prev_score)
    del prev_dist, prev_cost

    next_dist = _as_seq_distribution_matrix(next_out_distribution, seq_len)
    next_dist = _fill_negative_distribution_entries(next_dist, alpha)
    next_cost = sp.get("curr_out_to_next")
    if next_cost is not None:
        next_cost = np.asarray(next_cost, dtype=np.float64)
    next_score = _distribution_cost_score(next_dist, next_cost, transpose_cost=True)
    next_score = _sanitize_score_values(next_score)
    del next_dist, next_cost

    return prev_score, next_score


def _score_column(score, idx):
    if score is None:
        return None
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
    total_count = min(_SHARED_TOP_K, _SHARED_SEQ_LEN)
    if total_count <= 0:
        return []

    ordered = np.argsort(-metric, kind="stable")
    return [int(seq_idx) for seq_idx in ordered[:total_count]]


def score_components_for_edge(edge):
    u_idx, v_idx = (int(edge[0]), int(edge[1]))
    metric = np.zeros((_SHARED_SEQ_LEN,), dtype=np.float64)

    prev_col = _score_column(_SHARED_PREV_SCORE, u_idx)
    next_col = _score_column(_SHARED_NEXT_SCORE, v_idx)
    if prev_col is not None:
        metric += prev_col
    if next_col is not None:
        metric += next_col

    metric = _sanitize_score_values(metric)

    return metric, prev_col, next_col
