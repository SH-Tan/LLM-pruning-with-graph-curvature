import numpy as np
import torch
import ot

import curv_analysis_utils as analysis_utils
from curv_filter_utils import sliding_median_low_pass
from curv_distribution_utils import (
    _build_node_distribution,
    _edge_distribution,
    _min_reduce_blocks,
    _build_qk_out_node_distribution
)

from curv_sequence_utils import (
    _build_att_out_to_o_cost,
    _build_oproj_to_att_in_value_map,
    _build_vproj_to_att_out_cost,
    _build_vproj_to_att_out_value_map,
    _precompute_oproj_prev_distributions,
    _precompute_vproj_next_distributions,
)
from curv_shortest_path_utils import build_shortest_path_cache
from curv_shared_utils import _from_shared_numpy, _to_shared_numpy, _load_worker_seq_distribution, _to_shared_seq_metas
import curv_metric_utils as metric_utils
from curv_tensor_utils import _build_v_to_att_out_template, _build_x_to_out_cost

from multiprocessing import get_context
import multiprocessing as mp
import time

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

proc = mp.cpu_count()

_SHARED_CURR_DIST = None
_SHARED_PREV_IN = None
_SHARED_NEXT_OUT = None
_SHARED_SP = None
_SHARED_ALPHA = 0.0
_A = None
_Q_to_A = None
_V_COST = None
_SPK = None

_SHARED_SHORT_NAME = None
_SHARED_SAMPLE_IDX = None
_SHARED_MODEL_META = None
_SHARED_SEQ_LEN = 1
_SHARED_TOP_K = 5
_SHARED_SEQ_SELECT = "top"
_SHARED_LPF_WINDOW = 0
_SHARED_SAVE_PARAMETER_LOGS = False
_WORKER_CURR_DIST_SHM = None
_WORKER_PREV_IN_SHM = None
_WORKER_NEXT_OUT_SHM = None
_WORKER_SCORE_SHMS = []
_WORKER_PREV_SEQ_METAS = None
_WORKER_NEXT_SEQ_METAS = None
_WORKER_PREV_SEQ_SHMS = {}
_WORKER_NEXT_SEQ_SHMS = {}
_WORKER_PREV_SEQ_CACHE = {}
_WORKER_NEXT_SEQ_CACHE = {}


def _as_index_array(active):
    if active is None:
        return np.empty((0,), dtype=np.int64)
    return np.asarray(active, dtype=np.int64).reshape(-1)


def _edge_cost_matrix_base(u_idx, v_idx, prev_active, next_active, sp_uv):
    """
    Static seq-local cost matrix.
    """
    prev_active = _as_index_array(prev_active)
    next_active = _as_index_array(next_active)

    prev_count = int(len(prev_active))
    next_count = int(len(next_active))

    cost = np.full((prev_count + 1, next_count + 1), np.inf, dtype=np.float64)
    sp = _SHARED_SP

    prev_to_next_all = sp.get("prev_to_next_all", {})
    prev_to_curr_out_all = sp.get("prev_to_curr_out_all", {})
    curr_in_to_next_all = sp.get("curr_in_to_next_all", {})

    if prev_count > 0 and next_count > 0 and prev_to_next_all:
        block = _min_reduce_blocks([
            matrix[np.ix_(prev_active, next_active)]
            for matrix in prev_to_next_all.values()
        ])
        if block is not None:
            cost[:-1, :-1] = block

    if prev_count > 0 and prev_to_curr_out_all:
        block = _min_reduce_blocks([
            matrix[prev_active, v_idx]
            for matrix in prev_to_curr_out_all.values()
        ])
        if block is not None:
            cost[:-1, -1] = block

    if next_count > 0 and curr_in_to_next_all:
        block = _min_reduce_blocks([
            matrix[u_idx, next_active]
            for matrix in curr_in_to_next_all.values()
        ])
        if block is not None:
            cost[-1, :-1] = block

    cost[-1, -1] = sp_uv
    return cost


def _get_min_QK_A_cost(v_idx, u_idx, s, prev_active):
    prev_active = _as_index_array(prev_active)

    head_dim = _SHARED_MODEL_META["head_dim"]
    repeat = _SHARED_MODEL_META["repeat"]
    seq_len = _SHARED_SEQ_LEN

    d = v_idx % head_dim
    kv_head = v_idx // head_dim
    
    # repeated q-heads that share this kv head
    q_start = kv_head * repeat
    q_end = (kv_head + 1) * repeat

    shared_q_heads = np.arange(q_start, q_end)
    # seq @ repeat
    shared_out_idx = shared_q_heads * head_dim + d   # shape [repeat]
    
    # k cost, _SPK["prev_to_next_all"] = [input, seq * q head]
    k_start = kv_head * seq_len * repeat
    k_end = k_start + repeat * seq_len
    shared_k_heads = np.arange(k_start, k_end)    # [repeat * seq_len]
    
    # all seq @ repeat
    k_down = _V_COST[q_start:q_end, s, d]
    
    # Q only overlaps K on out_seq = s
    shared_start = s * repeat
    shared_end = (s + 1) * repeat
    merged_width = seq_len * repeat

    prev_merged = np.empty((0, merged_width), dtype=np.float64)

    if len(prev_active) > 0:
        # previous input -> Q-local-out
        q_prev = _Q_to_A["prev_to_next_all"][np.ix_(prev_active, shared_out_idx)]     # [P, repeat]

        # previous input -> K-local-node over all seq/repeated heads
        k_prev_prefix = _SPK["prev_to_next_all"][np.ix_(prev_active, shared_k_heads)]   # [P, seq*repeat]
        # Reorder from [r0_s0, ..., r0_sN, r1_s0, ..., rM_sN] to [s0_r0, ..., sN_rM].
        k_prev_prefix = k_prev_prefix.reshape(len(prev_active), repeat, seq_len).transpose(0, 2, 1)

        # previous input -> K -> V-out, broadcast repeated-head cost across all seq
        k_prev = k_prev_prefix + k_down[None, None, :]                             # [P, seq, repeat]
        prev_merged = k_prev.reshape(len(prev_active), merged_width)
        
        prev_merged[:, shared_start:shared_end] = np.minimum(
            q_prev,
            prev_merged[:, shared_start:shared_end]
        )

        
    # current input -> Q-local-out
    q_curr = _Q_to_A["curr_in_to_next_all"][u_idx, shared_out_idx]             # [repeat]

    # current input -> K-local-node over all seq/repeated heads
    k_curr_prefix = _SPK["curr_in_to_next_all"][u_idx, shared_k_heads]         # [seq*repeat]
    # Reorder from [r0_s0, ..., r0_sN, r1_s0, ..., rM_sN] to [s0_r0, ..., sN_rM].
    k_curr_prefix = k_curr_prefix.reshape(repeat, seq_len).transpose(1, 0)     # [seq, repeat]
    
    # current input -> K -> V-out, broadcast repeated-head cost across all seq
    k_curr = k_curr_prefix + k_down[None, :]                                     # [seq, repeat]
    
    curr_merged = k_curr.reshape(merged_width)
    curr_merged[shared_start:shared_end] = np.minimum(
        q_curr,
        curr_merged[shared_start:shared_end]
    )

    return {
        "prev_to_out": prev_merged,
        "curr_to_out": curr_merged,
    }



def _edge_cost_matrix_seq_aware(u_idx, v_idx, prev_active, next_active, sp_uv, seq = 0):
    prev_active = _as_index_array(prev_active)
    next_active = _as_index_array(next_active)

    short_name = _SHARED_SHORT_NAME
    if short_name not in {"v_proj", "o_proj"}:
        return _edge_cost_matrix_base(u_idx, v_idx, prev_active, next_active, sp_uv)

    prev_count = int(len(prev_active))
    next_count = int(len(next_active))
    cost = np.full((prev_count + 1, next_count + 1), np.inf, dtype=np.float64)

    sp = _SHARED_SP
    A = _A
    meta = _SHARED_MODEL_META
    head_dim = meta["head_dim"]
    repeat = meta["repeat"]
    
    prev_to_curr_out_all = sp.get("prev_to_curr_out_all", {})
    curr_in_to_next_all = sp.get("curr_in_to_next_all", {})

    # For seq-aware ops, both attention weights and attention metadata must exist.
    assert A is not None and meta is not None

    cost[-1, -1] = sp_uv

    # Static prev -> current output-node column
    if prev_count > 0 and prev_to_curr_out_all:
        block = _min_reduce_blocks([
            matrix[prev_active, v_idx]
            for matrix in prev_to_curr_out_all.values()
        ])
        if block is not None:
            cost[:-1, -1] = block
            
    # Static current input-node row -> next
    if next_count > 0 and curr_in_to_next_all:
        block = _min_reduce_blocks([
            matrix[u_idx, next_active]
            for matrix in curr_in_to_next_all.values()
        ])
        if block is not None:
            cost[-1, :-1] = block


    # Dynamic attention-coupled part
    if short_name == "v_proj" and next_count > 0:
        merge_cost = _get_min_QK_A_cost(v_idx, u_idx, seq, prev_active)
        
        dynamic_next = _build_vproj_to_att_out_cost(
            a=A,
            seq_len=_SHARED_SEQ_LEN,
            s_in=seq,
            v_idx=v_idx,
            head_dim=head_dim,
            repeat=repeat,
        )
        
        # if next_active is a subset, select aligned entries first
        if dynamic_next.shape[0] != next_count:
            dynamic_next = dynamic_next[next_active]
        if dynamic_next.shape[0] != next_count:
            raise ValueError(
                f"dynamic_next shape mismatch: got {dynamic_next.shape}, expected ({next_count},)"
            )

        merge_prev = merge_cost["prev_to_out"]
        merge_curr = merge_cost["curr_to_out"]
        
        if merge_prev.shape[1] != next_count and merge_prev.shape[1] > 0:
            merge_prev = merge_prev[:, next_active]
        if merge_curr.shape[0] != next_count:
            merge_curr = merge_curr[next_active]

        # Base dynamic path uses current v_proj edge then attention-to-output.
        base_prev = cost[:-1, -1][:, None] + dynamic_next[None, :]
        base_curr = cost[-1, -1] + dynamic_next

        if prev_count > 0 and merge_prev.shape[1] == next_count:
            cost[:-1, :-1] = np.minimum(base_prev, merge_prev)
        else:
            cost[:-1, :-1] = base_prev

        # Current input row uses the better of direct v->A->out and merged Q/K->A->out.
        cost[-1, :-1] = np.minimum(base_curr, merge_curr)

    elif short_name == "o_proj" and prev_count > 0:
        dynamic_prev = _build_att_out_to_o_cost(
            a=A,
            s_out=seq,
            out_idx=u_idx,
            head_dim=head_dim,
        )
        
        # if prev_active is a subset, select aligned entries first
        if dynamic_prev.shape[0] != prev_count:
            dynamic_prev = dynamic_prev[prev_active]
        if dynamic_prev.shape[0] != prev_count:
            raise ValueError(
                f"dynamic_prev shape mismatch: got {dynamic_prev.shape}, expected ({prev_count},)"
            )
            
        # prev nodes -> next nodes through current endpoint
        cost[:-1, :] = dynamic_prev[:, None] + cost[-1, :][None, :]

    return cost


def _edge_seq_distributions_and_cost(edge_info, seq_info):
    u_idx, v_idx = edge_info

    sp_uv = float(_SHARED_CURR_DIST[u_idx, v_idx])

    # For seq-aware nodes, pick the row for this sequence.
    if _SHARED_SHORT_NAME == "v_proj":
        next_row = None if _SHARED_NEXT_OUT is None else _SHARED_NEXT_OUT[v_idx]
    else:
        next_row = None if _SHARED_NEXT_OUT is None else _SHARED_NEXT_OUT[seq_info]

    if _SHARED_SHORT_NAME == "o_proj":
        prev_row = None if _SHARED_PREV_IN is None else _SHARED_PREV_IN[u_idx]
    else:
        prev_row = None if _SHARED_PREV_IN is None else _SHARED_PREV_IN[seq_info]

    mu, prev_active = _edge_distribution(prev_row, _SHARED_ALPHA)
    nu, next_active = _edge_distribution(next_row, _SHARED_ALPHA)

    cost = _edge_cost_matrix_seq_aware(
        u_idx=u_idx,
        v_idx=v_idx,
        prev_active=prev_active,
        next_active=next_active,
        sp_uv=sp_uv,
        seq=seq_info,
    )

    return u_idx, v_idx, sp_uv, mu, prev_active, nu, next_active, cost


def _parameter_log_detail(edge, seq_idx, sp_uv, mu, prev_active, nu, next_active, cost):
    if not _SHARED_SAVE_PARAMETER_LOGS:
        return {}

    u_idx, v_idx = (int(edge[0]), int(edge[1]))
    metric_score, metric_prev_score, metric_next_score = metric_utils.score_components_for_edge(edge)

    detail = {
        "in_neighbors": [int(idx) for idx in prev_active.tolist()],
        "out_neighbors": [int(idx) for idx in next_active.tolist()],
        "mu": np.asarray(mu, dtype=np.float64).tolist(),
        "nu": np.asarray(nu, dtype=np.float64).tolist(),
        "cost_matrix": np.asarray(cost, dtype=np.float64).tolist(),
        "prev_neighbors_to_v_cost": np.asarray(cost[:-1, -1], dtype=np.float64).tolist(),
        "u_to_out_neighbors_cost": np.asarray(cost[-1, :-1], dtype=np.float64).tolist(),
        "prev_neighbors_to_out_neighbors_cost": np.asarray(cost[:-1, :-1], dtype=np.float64).tolist(),
        "weight_magnitude": float(1.0 / sp_uv) if sp_uv != 0.0 and np.isfinite(sp_uv) else float("inf"),
    }
    prev_to_curr_in = _SHARED_SP.get("prev_to_curr_in") if _SHARED_SP is not None else None
    if prev_to_curr_in is not None and len(prev_active) > 0:
        detail["prev_neighbors_to_u_cost"] = np.asarray(
            prev_to_curr_in[np.asarray(prev_active, dtype=np.int64), u_idx],
            dtype=np.float64,
        ).tolist()

    curr_out_to_next = _SHARED_SP.get("curr_out_to_next") if _SHARED_SP is not None else None
    if curr_out_to_next is not None and len(next_active) > 0:
        detail["v_to_out_neighbors_cost"] = np.asarray(
            curr_out_to_next[v_idx, np.asarray(next_active, dtype=np.int64)],
            dtype=np.float64,
        ).tolist()

    if metric_score is not None:
        detail["metric_score"] = float(metric_score[int(seq_idx)])
    if metric_prev_score is not None:
        detail["metric_prev_score"] = float(metric_prev_score[int(seq_idx)])
    if metric_next_score is not None:
        detail["metric_next_score"] = float(metric_next_score[int(seq_idx)])

    if len(prev_active) > 0:
        prev_cost = np.asarray(cost[:-1, -1], dtype=np.float64)
        prev_probs = np.asarray(mu[:-1], dtype=np.float64)
        detail["top_prev_to_target_sum"] = float(np.sum(prev_probs * np.nan_to_num(prev_cost, nan=0.0, posinf=0.0, neginf=0.0)))
        detail["top_prev_to_target_nodes"] = [
            {
                "node_idx": int(node_idx),
                "probability": float(probability),
                "cost_to_target": float(cost_to_target),
                "weighted_cost": float(probability * (cost_to_target if np.isfinite(cost_to_target) else 0.0)),
            }
            for node_idx, probability, cost_to_target in zip(prev_active, prev_probs, prev_cost)
        ]

    if len(next_active) > 0:
        next_cost = np.asarray(cost[-1, :-1], dtype=np.float64)
        next_probs = np.asarray(nu[:-1], dtype=np.float64)
        detail["top_next_from_source_sum"] = float(np.sum(next_probs * np.nan_to_num(next_cost, nan=0.0, posinf=0.0, neginf=0.0)))
        detail["top_next_from_source_nodes"] = [
            {
                "node_idx": int(node_idx),
                "probability": float(probability),
                "cost_from_source": float(cost_from_source),
                "weighted_cost": float(probability * (cost_from_source if np.isfinite(cost_from_source) else 0.0)),
            }
            for node_idx, probability, cost_from_source in zip(next_active, next_probs, next_cost)
        ]

    if "top_prev_to_target_sum" in detail or "top_next_from_source_sum" in detail:
        detail["top_neighbor_cost_sum"] = (
            float(detail.get("top_prev_to_target_sum", 0.0))
            + float(detail.get("top_next_from_source_sum", 0.0))
        )

    return detail


def _compute_single_edge_seq_global(edge_info, seq_info):
    u_idx, v_idx, sp_uv, mu, prev_active, nu, next_active, cost = (
        _edge_seq_distributions_and_cost(edge_info, seq_info)
    )
    cost_inf_count = int(np.isinf(cost).sum())

    try:
        w_dist = float(ot.emd2(mu, nu, cost))
    except Exception:
        detail = _parameter_log_detail(edge_info, seq_info, sp_uv, mu, prev_active, nu, next_active, cost)
        return {
            "seq_idx": seq_info,
            "v_idx": v_idx,
            "u_idx": u_idx,
            "curv": float("inf"),
            "mu_len": int(len(mu)),
            "nu_len": int(len(nu)),
            "w_dist": float("inf"),
            "sp_uv": sp_uv,
            "cost_has_inf": bool(cost_inf_count > 0),
            "cost_inf_count": cost_inf_count,
            **detail,
        }

    curv = 1.0 - (w_dist / sp_uv)
    curv = curv / (1-_SHARED_ALPHA)
    detail = _parameter_log_detail(edge_info, seq_info, sp_uv, mu, prev_active, nu, next_active, cost)

    return {
        "seq_idx": seq_info,
        "v_idx": v_idx,
        "u_idx": u_idx,
        "curv": np.float64(curv),
        "mu_len": int(len(mu)),
        "nu_len": int(len(nu)),
        "w_dist": w_dist,
        "sp_uv": sp_uv,
        "cost_has_inf": bool(cost_inf_count > 0),
        "cost_inf_count": cost_inf_count,
        **detail,
    }





def _init_worker(
    curr_dist_meta,
    prev_meta,
    next_meta,
    prev_seq_metas=None,
    next_seq_metas=None,
    prev_score_meta=None,
    next_score_meta=None,
):
    global _SHARED_CURR_DIST, _SHARED_PREV_IN, _SHARED_NEXT_OUT
    global _WORKER_CURR_DIST_SHM, _WORKER_PREV_IN_SHM, _WORKER_NEXT_OUT_SHM
    global _WORKER_SCORE_SHMS
    global _WORKER_PREV_SEQ_METAS, _WORKER_NEXT_SEQ_METAS
    global _WORKER_PREV_SEQ_SHMS, _WORKER_NEXT_SEQ_SHMS
    global _WORKER_PREV_SEQ_CACHE, _WORKER_NEXT_SEQ_CACHE

    _WORKER_CURR_DIST_SHM, _SHARED_CURR_DIST = _from_shared_numpy(curr_dist_meta)

    _WORKER_PREV_IN_SHM = None
    _SHARED_PREV_IN = None
    if prev_meta is not None:
        _WORKER_PREV_IN_SHM, _SHARED_PREV_IN = _from_shared_numpy(prev_meta)

    _WORKER_NEXT_OUT_SHM = None
    _SHARED_NEXT_OUT = None
    if next_meta is not None:
        _WORKER_NEXT_OUT_SHM, _SHARED_NEXT_OUT = _from_shared_numpy(next_meta)

    _WORKER_SCORE_SHMS = []
    prev_score = None
    if prev_score_meta is not None:
        shm, prev_score = _from_shared_numpy(prev_score_meta)
        _WORKER_SCORE_SHMS.append(shm)

    next_score = None
    if next_score_meta is not None:
        shm, next_score = _from_shared_numpy(next_score_meta)
        _WORKER_SCORE_SHMS.append(shm)
    metric_utils.set_shared_metric_state(
        prev_score,
        next_score,
        seq_len=_SHARED_SEQ_LEN,
        top_k=_SHARED_TOP_K,
        seq_select=_SHARED_SEQ_SELECT,
    )

    _WORKER_PREV_SEQ_METAS = prev_seq_metas
    _WORKER_NEXT_SEQ_METAS = next_seq_metas
    _WORKER_PREV_SEQ_SHMS = {}
    _WORKER_NEXT_SEQ_SHMS = {}
    _WORKER_PREV_SEQ_CACHE = {}
    _WORKER_NEXT_SEQ_CACHE = {}


def _compute_edge_with_seq(task):
    seq_idx, edge = task

    global _SHARED_PREV_IN, _SHARED_NEXT_OUT

    prev_in_distribution = _SHARED_PREV_IN
    next_out_distribution = _SHARED_NEXT_OUT

    if _SHARED_SHORT_NAME == "o_proj":
        prev_in_distribution = _load_worker_seq_distribution(
            seq_idx,
            _WORKER_PREV_SEQ_METAS,
            _WORKER_PREV_SEQ_SHMS,
            _WORKER_PREV_SEQ_CACHE,
        )
    elif _SHARED_SHORT_NAME == "v_proj":
        next_out_distribution = _load_worker_seq_distribution(
            seq_idx,
            _WORKER_NEXT_SEQ_METAS,
            _WORKER_NEXT_SEQ_SHMS,
            _WORKER_NEXT_SEQ_CACHE,
        )

    old_prev = _SHARED_PREV_IN
    old_next = _SHARED_NEXT_OUT
    _SHARED_PREV_IN = prev_in_distribution
    _SHARED_NEXT_OUT = next_out_distribution

    try:
        return _compute_single_edge_seq_global(edge, seq_idx)
    finally:
        _SHARED_PREV_IN = old_prev
        _SHARED_NEXT_OUT = old_next


def _compute_edge_with_top_seq(edge):
    results = []
    for seq_idx in metric_utils.top_seq_for_edge(edge):
        edge_res = _compute_edge_with_seq((seq_idx, edge))
        if edge_res:
            results.append(edge_res)
    if _SHARED_TOP_K == -1 and _SHARED_LPF_WINDOW > 1 and results:
        curvs = np.asarray([float(edge_res["curv"]) for edge_res in results], dtype=np.float64)
        raw_idx = int(np.argmin(curvs))
        smoothed_curvs = sliding_median_low_pass(curvs, _SHARED_LPF_WINDOW)
        if _SHARED_SAVE_PARAMETER_LOGS:
            detailed_results = []
            for idx, edge_res in enumerate(results):
                detailed_res = dict(edge_res)
                detailed_res["curv"] = np.float64(curvs[idx])
                detailed_res["lpf_curv"] = np.float64(smoothed_curvs[idx])
                detailed_results.append(detailed_res)
            return detailed_results

        lpf_idx = int(np.argmin(smoothed_curvs))
        best_res = dict(results[raw_idx])
        best_res["curv"] = np.float64(curvs[raw_idx])
        best_res["lpf_curv"] = np.float64(smoothed_curvs[lpf_idx])
        return [best_res]
    return results


def _reset_shared_state():
    global _SHARED_CURR_DIST, _SHARED_PREV_IN, _SHARED_NEXT_OUT
    global _SHARED_SP, _SHARED_ALPHA, _A, _Q_to_A, _V_COST, _SPK
    global _SHARED_SHORT_NAME, _SHARED_SAMPLE_IDX, _SHARED_MODEL_META, _SHARED_SEQ_LEN
    global _SHARED_TOP_K, _SHARED_SEQ_SELECT, _SHARED_LPF_WINDOW, _SHARED_SAVE_PARAMETER_LOGS

    _SHARED_CURR_DIST = None
    _SHARED_PREV_IN = None
    _SHARED_NEXT_OUT = None
    _SHARED_SP = None
    _SHARED_ALPHA = 0.0
    _A = None
    _Q_to_A = None
    _V_COST = None
    _SPK = None
    _SHARED_SHORT_NAME = None
    _SHARED_SAMPLE_IDX = None
    _SHARED_MODEL_META = None
    _SHARED_SEQ_LEN = 1
    _SHARED_TOP_K = 5
    _SHARED_SEQ_SELECT = "top"
    _SHARED_LPF_WINDOW = 0
    _SHARED_SAVE_PARAMETER_LOGS = False
    metric_utils.reset_shared_metric_state()


def _get_vproj_aux_shortest_paths(operations, layer_cache, sp_cache, device):
    sp_q, _ = build_shortest_path_cache(
        operations=operations,
        layer_cache=layer_cache,
        short_name="q_proj",
        sp_cache=sp_cache,
        device=device,
        model_meta=_SHARED_MODEL_META,
    )
    sp_k, _ = build_shortest_path_cache(
        operations=operations,
        layer_cache=layer_cache,
        short_name="k_proj",
        sp_cache=sp_cache,
        device=device,
        model_meta=_SHARED_MODEL_META,
    )
    return sp_q, sp_k



def compute_op_curvature(
    operations,
    short_name,
    layer_id,
    sample_idx,
    layer_cache,
    sp_cache=None,
    alpha=0.0,
    device="cpu",
    seq_len = 1,
    num_q_heads=0, num_kv_heads=0, head_dim=0, repeat=0,
    sample_edge_num = -1,
    sample_edge_ratio = 1.,
    dataset_name="unknown_dataset",
    l2_norm=False,
    shared_top_k=10,
    shared_seq_select="top",
    curvature_lpf_window=0,
    analysis_dir=None,
    parameter_log_root=None,
):
    global _SHARED_CURR_DIST, _SHARED_PREV_IN, _SHARED_NEXT_OUT
    global _SHARED_SP, _SHARED_ALPHA, _A, _Q_to_A, _V_COST, _SPK
    global _SHARED_SHORT_NAME, _SHARED_SAMPLE_IDX, _SHARED_MODEL_META, _SHARED_SEQ_LEN
    global _SHARED_TOP_K, _SHARED_SEQ_SELECT, _SHARED_LPF_WINDOW, _SHARED_SAVE_PARAMETER_LOGS

    if operations is None or layer_cache is None:
        return None

    _reset_shared_state()
    _SHARED_MODEL_META = {
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "repeat": repeat,
    }

    sp, graph_data = build_shortest_path_cache(
        operations=operations,
        layer_cache=layer_cache,
        short_name=short_name,
        sp_cache=sp_cache,
        device=device,
        model_meta=_SHARED_MODEL_META,
    )
    if sp is None:
        return None
    
    if short_name == "v_proj":
        sp_q, sp_k = _get_vproj_aux_shortest_paths(
            operations=operations,
            layer_cache=layer_cache,
            sp_cache=sp_cache,
            device=device,
        )
        if sp_q is None or sp_k is None:
            raise ValueError("q_proj and k_proj shortest-path caches are required for v_proj")
        
        _SPK = {}
        
        if sp_k["prev_to_next_all"]:
            _SPK["prev_to_next_all"] = next(iter(sp_k["prev_to_next_all"].values()))
        _SPK["curr_in_to_next_all"] = sp_k["curr_in_to_next_all"]["q_proj"]  # [input, seq * q_head]

    curr_dist = sp["curr_dist"]
    in_dim, out_dim = curr_dist.shape
    
    prev_in_distribution = None
    next_out_distribution = None
    
    # if is o_proj, prev is A
    if (short_name not in ["o_proj"]) and graph_data["prev_in"] is not None:
        prev_in_distribution = _build_node_distribution(
            graph_data["prev_in"],
            graph_data["prev_in_name"],
            alpha,
            l2_norm=l2_norm,
        )
        
    # if is q, k, v, the next is A or attention out     
    if (short_name not in ["q_proj", "k_proj", "v_proj"]) and graph_data["next_out"] is not None:
        next_out_distribution = _build_node_distribution(
            graph_data["next_out"],
            graph_data["next_out_name"],
            alpha,
            l2_norm=l2_norm,
        ) # [batch, seq, hidden size]
        
    
    # build q_proj,k_proj next distribution
    if (short_name in ["q_proj", "k_proj"]) and (graph_data["next_out"] is not None):
        next_out_distribution = _build_qk_out_node_distribution(
            short_name,
            graph_data["next_out"],
            alpha,
            l2_norm=l2_norm,
        )

  
    _SHARED_CURR_DIST = curr_dist
    _SHARED_SP = sp
    _SHARED_ALPHA = alpha
    _SHARED_SHORT_NAME = short_name
    _SHARED_SAMPLE_IDX = sample_idx
    _SHARED_SEQ_LEN = seq_len

    _SHARED_PREV_IN = prev_in_distribution
    _SHARED_NEXT_OUT = next_out_distribution
    
    curvature = torch.full((out_dim, in_dim), float("inf"), dtype=torch.float64)
    lpf_curvature = None

    curr_dist_np = np.asarray(curr_dist, dtype=np.float64)
    curr_dist_finite_edges = int(np.isfinite(curr_dist_np).sum())
    curr_dist_infinite_edges = int(np.isinf(curr_dist_np).sum())
    finite_edges = np.argwhere(np.isfinite(curr_dist_np) & (curr_dist_np > 0))

    analysis_path = analysis_utils.start_curvature_analysis(
        layer_id=layer_id,
        short_name=short_name,
        sample_idx=sample_idx,
        curvature_shape=curvature.shape,
        seq_len=seq_len,
        dataset_name=dataset_name,
        analysis_dir=analysis_dir,
    )
    precomputed_prev_dists = None
    precomputed_next_dists = None
    
    
    # build neighbor distribution for v_proj or o_proj with A
    if short_name in {"v_proj", "o_proj"}:
        _A = operations.get("A", None)
        
        if short_name == "v_proj":
            value_map = _build_vproj_to_att_out_value_map(
                graph_data["next_out"], out_dim, seq_len, head_dim, repeat
            )
            
            # # node distribution for all seq w/ mask
            # precomputed_next_dists = _precompute_vproj_next_distributions(
            #     value_map, seq_len, repeat, graph_data["next_out_name"], alpha
            # )
            
            # node distribution for all seq w/o mask
            precomputed_next_dists = _precompute_vproj_next_distributions(
                value_map,
                graph_data["next_out_name"],
                alpha,
                l2_norm=l2_norm,
            )
      
            # x -> Q -> A -> out
            v_cost = operations.get("v_proj", None)
            if v_cost is not None:                
                # [q_heads, seq, head_dim]
                _V_COST = _build_v_to_att_out_template(v_cost, _SHARED_MODEL_META) 
                
                # cost matrix from inout to attention out via q_proj
                _Q_to_A = _build_x_to_out_cost(
                    _V_COST,
                    sp_q,
                    _SHARED_MODEL_META,
                    device,
                ) # [input, head_dim * q_head]
                
                _V_COST = _V_COST.cpu().contiguous().numpy()
                
        elif short_name == "o_proj":
            value_map = _build_oproj_to_att_in_value_map(
                graph_data["prev_in"], in_dim, seq_len, head_dim, repeat
            )
            precomputed_prev_dists = _precompute_oproj_prev_distributions(
                value_map, seq_len, graph_data["prev_in_name"], alpha, l2_norm=l2_norm
            )
    
    ctx = get_context("fork")

    base_owned_shms = []
    curr_shm, curr_meta = _to_shared_numpy(curr_dist_np)
    base_owned_shms.append(curr_shm)

    base_prev_meta = None
    base_next_meta = None

    if _SHARED_PREV_IN is not None:
        shm, base_prev_meta = _to_shared_numpy(np.asarray(_SHARED_PREV_IN, dtype=np.float64))
        base_owned_shms.append(shm)

    if _SHARED_NEXT_OUT is not None:
        shm, base_next_meta = _to_shared_numpy(np.asarray(_SHARED_NEXT_OUT, dtype=np.float64))
        base_owned_shms.append(shm)

    seq_prev_metas, prev_seq_shms = _to_shared_seq_metas(precomputed_prev_dists)
    seq_next_metas, next_seq_shms = _to_shared_seq_metas(precomputed_next_dists)
    seq_owned_shms = prev_seq_shms + next_seq_shms
    
    # Select by ratio first, then optionally cap with sample_edge_num.
    if len(finite_edges) > 0:
        seed = 13
        rng = np.random.default_rng(seed)

        if sample_edge_ratio < 1.0:
            ratio_edge_num = max(1, int(len(finite_edges) * sample_edge_ratio))
            ratio_edge_num = min(ratio_edge_num, len(finite_edges))
            selected_idx = rng.choice(len(finite_edges), size=ratio_edge_num, replace=False)
            finite_edges = finite_edges[np.sort(selected_idx)]

        if sample_edge_num > 0 and sample_edge_num < len(finite_edges):
            selected_idx = rng.choice(len(finite_edges), size=sample_edge_num, replace=False)
            finite_edges = finite_edges[np.sort(selected_idx)]
    finite_edges = np.asarray(finite_edges, dtype=np.int64).reshape(-1, 2)
    
    finite_edges = [(379,523),(2220,466),(2811,1359),(2449,2816),(2759,4008),(3880,1552)]
    
    print(
        f'op = {short_name}, sample = {sample_idx}, seq = {seq_len}, '
        f'total edges = {len(finite_edges)} per seq, cur dist shape = {curr_dist.shape}'
    )

    mu_len_total = 0.0
    nu_len_total = 0.0
    mu_nu_count = 0
    cost_has_inf = False
    cost_inf_count = 0
    t1 = time.time()
    
    if l2_norm:
        print(_SHARED_NEXT_OUT.shape)
        seq_len = 1
        
    _SHARED_SEQ_LEN = seq_len
    _SHARED_TOP_K = int(shared_top_k)
    _SHARED_SEQ_SELECT = shared_seq_select
    _SHARED_LPF_WINDOW = int(curvature_lpf_window)
    _SHARED_SAVE_PARAMETER_LOGS = bool(parameter_log_root)
    if _SHARED_TOP_K == -1 and _SHARED_LPF_WINDOW > 1:
        lpf_curvature = torch.full((out_dim, in_dim), float("inf"), dtype=torch.float64)

    prev_score = None
    next_score = None
    if _SHARED_TOP_K != -1 and not l2_norm and seq_len > 1:
        prev_score, next_score = metric_utils.build_neighbor_score_matrices(
            seq_len=seq_len,
            sp=sp,
            alpha=alpha,
            prev_in_distribution=prev_in_distribution,
            next_out_distribution=next_out_distribution,
        )

    prev_score_meta = None
    next_score_meta = None
    if prev_score is not None and not np.isscalar(prev_score):
        shm, prev_score_meta = _to_shared_numpy(np.asarray(prev_score, dtype=np.float64))
        base_owned_shms.append(shm)
    if next_score is not None and not np.isscalar(next_score):
        shm, next_score_meta = _to_shared_numpy(np.asarray(next_score, dtype=np.float64))
        base_owned_shms.append(shm)

    try:
        effective_top_k = _SHARED_SEQ_LEN if _SHARED_TOP_K == -1 else min(_SHARED_TOP_K, _SHARED_SEQ_LEN)
        total_edge_seq_tasks = len(finite_edges) * max(effective_top_k, 0)
        print(
            f"Will evaluate {effective_top_k} seq positions with {_SHARED_SEQ_SELECT} selection "
            f"for about {total_edge_seq_tasks} edge/seq tasks from {len(finite_edges)} edges."
        )
        if _SHARED_TOP_K == -1 and _SHARED_LPF_WINDOW > 1:
            print(f"Using sliding median low-pass filter with window={_SHARED_LPF_WINDOW}.")
        print(f'Start creating Pool....')

        with ctx.Pool(processes=proc, initializer=_init_worker,
            initargs=(
                curr_meta,
                base_prev_meta,
                base_next_meta,
                seq_prev_metas,
                seq_next_metas,
                prev_score_meta,
                next_score_meta,
            ),) as pool:
            for edge_results in pool.imap_unordered(_compute_edge_with_top_seq, finite_edges, chunksize=1):
                if not edge_results:
                    continue
                parameter_log_path = None
                for edge_res in edge_results:
                    if not edge_res:
                        continue
                    edge_res["sample_idx"] = sample_idx
                    if parameter_log_root is not None:
                        parameter_log_path = analysis_utils.prepare_parameter_detail_log(
                            parameter_log_root,
                            layer_id,
                            short_name,
                            sample_idx,
                            edge_res["v_idx"],
                            edge_res["u_idx"],
                        )
                        analysis_utils.append_parameter_detail_log(
                            parameter_log_root,
                            layer_id,
                            short_name,
                            edge_res,
                        )
                    v_idx = edge_res["v_idx"]
                    u_idx = edge_res["u_idx"]
                    curv = float(edge_res["curv"])
                    if curv < float(curvature[v_idx, u_idx]):
                        curvature[v_idx, u_idx] = curv
                    if lpf_curvature is not None and "lpf_curv" in edge_res:
                        lpf_curv = float(edge_res["lpf_curv"])
                        if lpf_curv < float(lpf_curvature[v_idx, u_idx]):
                            lpf_curvature[v_idx, u_idx] = lpf_curv
                    mu_len_total += float(edge_res["mu_len"])
                    nu_len_total += float(edge_res["nu_len"])
                    mu_nu_count += 1
                    cost_has_inf = cost_has_inf or bool(edge_res["cost_has_inf"])
                    cost_inf_count += int(edge_res["cost_inf_count"])
                if parameter_log_path is not None:
                    analysis_utils.finalize_parameter_detail_log(parameter_log_path)

    finally:
        for shm in seq_owned_shms:
            shm.close()
            shm.unlink()

        for shm in base_owned_shms:
            shm.close()
            shm.unlink()

    runtime_sec = time.time() - t1
    print(f"sample_idx = {sample_idx}, time = {runtime_sec} s, ")

    _reset_shared_state()

    avg_mu_len = mu_len_total / mu_nu_count if mu_nu_count > 0 else float("nan")
    avg_nu_len = nu_len_total / mu_nu_count if mu_nu_count > 0 else float("nan")

    analysis_utils.append_final_min_curvature_summary(
        analysis_path=analysis_path,
        sample_idx=sample_idx,
        curvature=curvature,
        avg_mu_len=avg_mu_len,
        avg_nu_len=avg_nu_len,
        curr_dist_finite_edges=curr_dist_finite_edges,
        curr_dist_infinite_edges=curr_dist_infinite_edges,
        cost_has_inf=cost_has_inf,
        cost_inf_count=cost_inf_count,
        runtime_sec=runtime_sec,
    )

    # curvature = _merge_gqa_curvature(curvature, model, layer_id, short_name)
    if lpf_curvature is not None:
        return {
            "curvature": curvature,
            "lpf_curvature": lpf_curvature,
        }
    return curvature
