import numpy as np
import torch

from curv_dtype_utils import curvature_np_dtype, curvature_torch_dtype


def minmax_per_batch_nonzero_nozero(x, eps=1e-4, dim=-1):
    mask = x != 0

    xmin = x.masked_fill(~mask, float("inf")).amin(dim=dim, keepdim=True)
    xmax = x.masked_fill(~mask, float("-inf")).amax(dim=dim, keepdim=True)

    has_nonzero = mask.any(dim=dim, keepdim=True)
    xmin = torch.where(has_nonzero, xmin, torch.zeros_like(xmin))
    xmax = torch.where(has_nonzero, xmax, torch.ones_like(xmax))

    denom = (xmax - xmin).clamp_min(eps)

    # normalize nonzero values
    norm = torch.zeros_like(x)
    norm = torch.where(mask, (x - xmin) / denom, norm)

    # push nonzero zeros upward a little so smallest nonzero won't be 0
    norm = torch.where(mask & (norm == 0), torch.full_like(norm, eps), norm)
    return norm


def _normalize_node_value_per_sequence(node_val, dim=-1):
    node_val = node_val.abs()
    node_norm = minmax_per_batch_nonzero_nozero(node_val, dim=dim)
    return 1.0 / node_norm


def _resolve_node_name(operations, names):
    for name in names:
        if not name:
            continue
        if name in operations:
            return name
        # if name in {"q_proj", "k_proj", "v_proj"} and "layer_input" in operations:
        #     return "layer_input"
    return None


def _edge_distribution(dist_row_or_col, alpha):
    if dist_row_or_col is None:
        return np.array([1.0], dtype=curvature_np_dtype()), np.empty((0,), dtype=np.int64)

    probs = np.asarray(dist_row_or_col, dtype=curvature_np_dtype()).reshape(-1).copy()

    non_zero = np.nonzero(probs)[0]
    if non_zero.size == 0:
        return np.array([1.0], dtype=curvature_np_dtype()), np.empty((0,), dtype=np.int64)

    if np.any(probs == -1.0):
        tmp = (1.0 - alpha) / non_zero.size
        probs[probs == -1.0] = tmp

    return np.hstack((probs[non_zero], np.array([alpha], dtype=curvature_np_dtype()))), non_zero


def _min_reduce_blocks(blocks):
    """
    Faster and cleaner helper than repeating np.minimum.reduce([...]) inline.
    """
    if not blocks:
        return None
    if len(blocks) == 1:
        return np.asarray(blocks[0], dtype=curvature_np_dtype())
    return np.minimum.reduce([np.asarray(b, dtype=curvature_np_dtype()) for b in blocks])


def _build_node_distribution(
    node_tensor,
    node_name,
    alpha,
    eps=1e-7,
    l2_norm=False,
    l2_norm_mode="per_example",
    l2_reference=None,
):
    if node_tensor is None or node_tensor.numel() == 0:
        return None

    if l2_norm and l2_norm_mode == "all_examples" and l2_reference is not None:
        ref_tensor = torch.as_tensor(l2_reference, dtype=curvature_torch_dtype()).reshape(1, -1)
        if ref_tensor.shape[-1] == node_tensor.shape[-1]:
            node_tensor = ref_tensor
        else:
            node_tensor = node_tensor.to(dtype=curvature_torch_dtype())
    else:
        node_tensor = node_tensor.to(dtype=curvature_torch_dtype())

    if node_tensor.dim() == 3:
        if node_tensor.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 for node tensor, got shape {tuple(node_tensor.shape)}"
            )
        node_tensor = node_tensor.squeeze(0)

    if l2_norm and node_tensor.dim() == 2:
        node_tensor = torch.norm(node_tensor, p=2, dim=0, keepdim=True)
        
    node_tensor = _normalize_node_value_per_sequence(node_tensor)
    
    valid_mask = torch.isfinite(node_tensor) & (node_tensor != 0)
    # weights = torch.exp(-(node_tensor)) * valid_mask
    weights = torch.exp(-(node_tensor ** 2)) * valid_mask

    sum_weights = weights.sum(dim=-1, keepdim=True)
    dist = torch.where(
        sum_weights > eps,
        ((1.0 - alpha) * weights) / sum_weights,
        torch.zeros_like(weights),
    )

    empty_mask = (sum_weights <= eps).expand_as(valid_mask)
    dist = torch.where(empty_mask & valid_mask, torch.full_like(dist, -1.0), dist)

    dist = dist * valid_mask
    
    return dist.detach().cpu().numpy().astype(curvature_np_dtype(), copy=False)






def _build_block_row_node_distribution(
    node_tensor,
    alpha,
    eps=1e-7,
    l2_norm=False,
    l2_reference=None,
):
    if node_tensor is None or node_tensor.numel() == 0:
        return None

    node_tensor = node_tensor.to(dtype=curvature_torch_dtype())

    if l2_norm and l2_reference is not None:
        ref_tensor = torch.as_tensor(
            l2_reference,
            dtype=curvature_torch_dtype(),
            device=node_tensor.device,
        )

        expected_shape = (node_tensor.shape[0], node_tensor.shape[2])

        if ref_tensor.shape == expected_shape:
            node_tensor = ref_tensor.unsqueeze(1)
        elif ref_tensor.numel() == int(np.prod(expected_shape)):
            node_tensor = ref_tensor.reshape(*expected_shape).unsqueeze(1)
        else:
            raise ValueError(
                f"Invalid l2_reference shape {tuple(ref_tensor.shape)}, "
                f"expected {expected_shape} or numel={int(np.prod(expected_shape))}"
            )

    if l2_norm:
        node_tensor = torch.norm(node_tensor, p=2, dim=1, keepdim=True)

    # Important:
    # node_tensor shape is always [heads, seq_axis, feature_axis].
    #
    # For q_proj:
    #   [q_heads, seq_q, seq_k]
    #   dim=1 means normalize across seq_q.
    #
    # For k_proj:
    #   [k_heads, seq_k, seq_q * repeat]
    #   dim=1 means normalize across seq_k.
    node_tensor = _normalize_node_value_per_sequence(node_tensor, dim=1)

    valid_mask = torch.isfinite(node_tensor) & (node_tensor != 0)

    weights = torch.exp(-(node_tensor ** 2)) * valid_mask

    # Distribution over the feature axis for each node row.
    sum_weights = weights.sum(dim=-1, keepdim=True)

    dist = torch.where(
        sum_weights > eps,
        ((1.0 - alpha) * weights) / sum_weights,
        torch.zeros_like(weights),
    )

    empty_mask = (sum_weights <= eps).expand_as(valid_mask)
    dist = torch.where(
        empty_mask & valid_mask,
        torch.full_like(dist, -1.0),
        dist,
    )

    dist = dist * valid_mask

    return dist.detach().cpu().numpy().astype(
        curvature_np_dtype(),
        copy=False,
    )



def _build_qk_out_node_distribution(
    l_name,
    node,
    alpha,
    l2_norm=False,
    l2_norm_mode="per_example",
    l2_reference=None,
    repeat=1,
):
    if node is None or node.numel() == 0:
        return None

    if node.dim() != 4:
        raise ValueError(
            f"Expected A node shape [B, q_heads, seq, seq], got {tuple(node.shape)}"
        )

    if node.shape[0] != 1:
        raise ValueError(
            f"Expected batch size 1 for A node, got shape {tuple(node.shape)}"
        )

    # [1, q_heads, seq_q, seq_k] -> [q_heads, seq_q, seq_k]
    A = node.squeeze(0).contiguous()
    q_heads, seq_q, seq_k = A.shape
    repeat = max(int(repeat), 1)

    if l_name == "q_proj":
        # Q node:
        # For each q head h and query token i:
        # node = A[h, i, :]
        #
        # Shape: [q_heads, seq_q, seq_k]
        # Normalize across seq_q, i.e. dim=1.
        out_node = A

    elif l_name == "k_proj":
        # K node:
        # q_heads = k_heads * repeat
        #
        # Original:
        #   A: [q_heads, seq_q, seq_k]
        #
        # Reshape repeated Q heads into K-head groups:
        #   [k_heads, repeat, seq_q, seq_k]
        #
        # Then for each K head and key token j, collect all repeated
        # Q-head/query-token attention values:
        #   [k_heads, seq_k, seq_q * repeat]
        # Feature axis order is [repeat, seq_q], matching _get_qk_next_cost.
        #
        # Normalize across seq_k, i.e. dim=1.
        if q_heads % repeat != 0:
            raise ValueError(
                f"q_heads={q_heads} is not divisible by repeat={repeat}"
            )

        k_heads = q_heads // repeat

        out_node = (
            A.reshape(k_heads, repeat, seq_q, seq_k)
             .permute(0, 3, 1, 2)          # [k_heads, seq_k, repeat, seq_q]
             .reshape(k_heads, seq_k, seq_q * repeat)
             .contiguous()
        )

    else:
        raise ValueError(f"Unsupported l_name={l_name}")

    return _build_block_row_node_distribution(
        out_node,
        alpha=alpha,
        l2_norm=l2_norm,
        l2_reference=l2_reference,
    )
