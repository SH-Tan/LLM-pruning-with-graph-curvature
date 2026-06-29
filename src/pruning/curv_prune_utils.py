import torch
from pruning.prune import align_curvature_to_weight_shape, should_prune_op, skip_prune_layer
from pruning.prune_log_utils import (
    append_all_layer_pruned_parameter_log,
    collect_pruned_parameter_rows,
)


def _get_prunable_module(model, layer_idx, op_name):
    layer = model.model.layers[layer_idx]

    if op_name == "lm_head":
        return model.lm_head
    if op_name == "down_proj":
        return layer.mlp.down_proj
    if op_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return getattr(layer.self_attn, op_name)
    if op_name in {"gate_proj", "up_proj"}:
        return getattr(layer.mlp, op_name)

    raise KeyError(f"Unsupported op_name for curvature pruning: {op_name}")


def _is_magnitude_fallback(model, layer_idx, op_name):
    fallbacks = getattr(model, "curvature_magnitude_fallbacks", None)
    if fallbacks is None or layer_idx >= len(fallbacks):
        return False
    return bool(fallbacks[layer_idx].get(op_name, False))


def prune_global_curvature(args, model):
    if not hasattr(model, "curvature_scores"):
        raise AttributeError("Model does not have curvature_scores. Run prune_curvature first.")

    model.eval()

    score_refs = []
    total_finite_params = 0

    for layer_idx, layer_scores in enumerate(getattr(model, "curvature_scores", [])):
        if skip_prune_layer(args, layer_idx):
            print(f"Skipping layer {layer_idx}: requested by skip_prune_layer_ids")
            continue
        for op_name, curv in layer_scores.items():
            if not should_prune_op(args, op_name):
                continue
            module = _get_prunable_module(model, layer_idx, op_name)
            weight = module.weight.data
            curv_cpu = align_curvature_to_weight_shape(
                curv,
                weight.shape,
                context=f"layer {layer_idx} {op_name} global curvature",
            ).cpu()
            magnitude_fallback = _is_magnitude_fallback(model, layer_idx, op_name)
            if magnitude_fallback:
                curv_cpu = torch.abs(weight.detach()).cpu()

            finite_mask = torch.isfinite(curv_cpu)
            finite_count = int(finite_mask.sum().item())
            if finite_count == 0:
                print(f"Skipping layer {layer_idx} {op_name}: no finite curvature scores")
                continue

            score_refs.append(
                {
                    "layer_idx": layer_idx,
                    "op_name": op_name,
                    "module": module,
                    "curv": curv_cpu,
                    "finite_mask": finite_mask,
                    "magnitude_fallback": magnitude_fallback,
                }
            )
            total_finite_params += finite_count

    if total_finite_params == 0:
        print("No finite curvature scores found for global curvature pruning")
        return []

    # Only parameters with finite curvature scores participate in global pruning.
    prune_count = int(total_finite_params * args.sparsity_ratio)
    if prune_count <= 0:
        print("Global curvature pruning skipped because prune_count is 0")
        return []

    prune_count = min(prune_count, total_finite_params)
    prune_high_scores = getattr(args, "prune_score_order", "high_to_low") == "high_to_low"
    entry_selections = [
        torch.zeros(int(entry["finite_mask"].sum().item()), dtype=torch.bool)
        for entry in score_refs
    ]

    for fallback, largest in ((False, prune_high_scores), (True, False)):
        remaining = prune_count - sum(int(mask.sum().item()) for mask in entry_selections)
        if remaining <= 0:
            break
        group_indices = [
            idx for idx, entry in enumerate(score_refs)
            if bool(entry["magnitude_fallback"]) == fallback
        ]
        if not group_indices:
            continue
        group_scores = torch.cat([
            score_refs[idx]["curv"][score_refs[idx]["finite_mask"]].reshape(-1)
            for idx in group_indices
        ])
        if group_scores.numel() == 0:
            continue
        group_prune_count = min(remaining, group_scores.numel())
        topk_indices = torch.topk(
            group_scores,
            k=group_prune_count,
            largest=largest,
            sorted=False,
        ).indices
        group_mask = torch.zeros(group_scores.numel(), dtype=torch.bool)
        group_mask[topk_indices] = True
        group_offset = 0
        for idx in group_indices:
            count = int(score_refs[idx]["finite_mask"].sum().item())
            entry_selections[idx] = group_mask[group_offset:group_offset + count]
            group_offset += count

    total_pruned = 0
    prune_summary = []
    report_limit = int(getattr(args, "all_layer_report_rank_offset", 0)) + 25
    report_rows = []
    with torch.no_grad():
        for entry, flat_selection in zip(score_refs, entry_selections):
            module = entry["module"]
            finite_mask = entry["finite_mask"]
            flat_finite_count = int(finite_mask.sum().item())

            prune_mask_cpu = torch.zeros_like(finite_mask, dtype=torch.bool)
            prune_mask_cpu[finite_mask] = flat_selection
            layer_pruned = int(prune_mask_cpu.sum().item())
            total_pruned += layer_pruned
            report_rows.extend(
                collect_pruned_parameter_rows(
                    entry["layer_idx"],
                    entry["op_name"],
                    module,
                    entry["curv"],
                    prune_mask_cpu,
                    largest=prune_high_scores,
                    limit=report_limit,
                )
            )

            prune_mask = prune_mask_cpu.to(device=module.weight.data.device)
            module.weight.data[prune_mask] = 0

            prune_summary.append(
                {
                    "layer_idx": entry["layer_idx"],
                    "op_name": entry["op_name"],
                    "pruned_edges": layer_pruned,
                    "total_edges": flat_finite_count,
                    "pruned_params": layer_pruned,
                    "total_params": flat_finite_count,
                }
            )

            del prune_mask, prune_mask_cpu

    append_all_layer_pruned_parameter_log(
        getattr(args, "all_layer_parameter_log_path", None),
        args,
        "curvature",
        getattr(args, "prune_score_order", "high_to_low"),
        "curvature",
        report_rows,
        largest=prune_high_scores,
        rank_offset=getattr(args, "all_layer_report_rank_offset", 0),
    )

    print(
        f"Global curvature pruning complete: pruned={total_pruned}"
    )
    return prune_summary
