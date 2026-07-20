import os

import torch

from pruning.curv_prune_utils import _get_prunable_module, _is_magnitude_fallback
from pruning.prune import (
    align_curvature_to_weight_shape,
    prune_scope_from_args,
    should_prune_op,
    skip_prune_layer,
)


def _entry_metric(entry, metric_name):
    if metric_name == "magnitude":
        return torch.abs(entry["module"].weight.detach()).cpu()
    return entry[metric_name]


def _candidate_mask(entry):
    return torch.isfinite(entry["curv"])


def _lookup_score_by_short_name(scores, short_name):
    if short_name in scores:
        return scores[short_name]
    for name, score in scores.items():
        if str(name).split(".")[-1] == short_name:
            return score
    return None


def _masks_for_metric(entries, metric_name, ratio, largest):
    entries = list(entries)
    if not entries:
        return []

    chunks = []
    counts = []
    for entry in entries:
        metric = _entry_metric(entry, metric_name)
        mask = _candidate_mask(entry) & torch.isfinite(metric)
        count = int(mask.sum().item())
        counts.append(count)
        if count > 0:
            chunks.append(metric[mask].detach().float().cpu())
        if metric_name == "magnitude":
            del metric

    total = sum(counts)
    prune_count = int(total * float(ratio))
    if prune_count <= 0 or total == 0:
        return [
            torch.zeros_like(entry["curv"], dtype=torch.bool)
            for entry in entries
        ]

    scores = torch.cat(chunks)
    prune_count = min(prune_count, scores.numel())
    selected = torch.topk(scores, k=prune_count, largest=largest, sorted=False).indices
    selected_mask = torch.zeros(scores.numel(), dtype=torch.bool)
    selected_mask[selected] = True

    masks = []
    offset = 0
    for entry, count in zip(entries, counts):
        metric = _entry_metric(entry, metric_name)
        prune_mask = torch.zeros_like(metric, dtype=torch.bool)
        if count > 0:
            eligible = _candidate_mask(entry) & torch.isfinite(metric)
            positions = eligible.reshape(-1).nonzero(as_tuple=False).flatten()
            entry_selection = selected_mask[offset:offset + count]
            prune_mask.reshape(-1)[positions[entry_selection]] = True
            offset += count
        masks.append(prune_mask)
        if metric_name == "magnitude":
            del metric
    return masks


def _masks_for_curvature(entries, ratio):
    entries = list(entries)
    if not entries:
        return []

    counts = [int(_candidate_mask(entry).sum().item()) for entry in entries]
    total = sum(counts)
    prune_count = int(total * float(ratio))
    if prune_count <= 0 or total == 0:
        return [
            torch.zeros_like(entry["curv"], dtype=torch.bool)
            for entry in entries
        ]

    prune_count = min(prune_count, total)
    selections = [torch.zeros(count, dtype=torch.bool) for count in counts]
    for fallback, largest in ((False, entries[0]["curv_largest"]), (True, False)):
        selected_count = sum(int(mask.sum().item()) for mask in selections)
        remaining = prune_count - selected_count
        if remaining <= 0:
            break

        group_indices = [
            idx for idx, entry in enumerate(entries)
            if bool(entry["magnitude_fallback"]) == fallback
        ]
        if not group_indices:
            continue

        chunks = [
            entries[idx]["curv"][_candidate_mask(entries[idx])].reshape(-1).detach().float().cpu()
            for idx in group_indices
            if counts[idx] > 0
        ]
        if not chunks:
            continue
        scores = torch.cat(chunks)
        if scores.numel() == 0:
            continue

        selected = torch.topk(
            scores,
            k=min(remaining, scores.numel()),
            largest=largest,
            sorted=False,
        ).indices
        selected_mask = torch.zeros(scores.numel(), dtype=torch.bool)
        selected_mask[selected] = True

        offset = 0
        for idx in group_indices:
            count = counts[idx]
            if count > 0:
                selections[idx] = selected_mask[offset:offset + count]
                offset += count

    masks = []
    for entry, selection in zip(entries, selections):
        mask = _candidate_mask(entry)
        prune_mask = torch.zeros_like(mask, dtype=torch.bool)
        if selection.numel() > 0:
            positions = mask.reshape(-1).nonzero(as_tuple=False).flatten()
            prune_mask.reshape(-1)[positions[selection]] = True
        masks.append(prune_mask)
    return masks


def _iter_overlap_entries(args, model, wanda_scores):
    for layer_idx, layer_scores in enumerate(getattr(model, "curvature_scores", [])):
        if skip_prune_layer(args, layer_idx):
            continue
        wanda_layer_scores = wanda_scores[layer_idx] if layer_idx < len(wanda_scores) else {}
        for op_name, curv in layer_scores.items():
            wanda_score = _lookup_score_by_short_name(wanda_layer_scores, op_name)
            if not should_prune_op(args, op_name) or wanda_score is None:
                continue

            module = _get_prunable_module(model, layer_idx, op_name)
            weight = module.weight.data
            curv_cpu = align_curvature_to_weight_shape(
                curv,
                weight.shape,
                context=f"layer {layer_idx} {op_name} overlap curvature",
            ).cpu()
            magnitude_fallback = _is_magnitude_fallback(model, layer_idx, op_name)
            if magnitude_fallback:
                curv_cpu = torch.abs(weight.detach()).cpu()

            wanda_cpu = wanda_score.detach().cpu()
            if wanda_cpu.shape != curv_cpu.shape:
                wanda_cpu = align_curvature_to_weight_shape(
                    wanda_cpu,
                    curv_cpu.shape,
                    context=f"layer {layer_idx} {op_name} overlap wanda",
                ).cpu()

            yield {
                "layer_idx": layer_idx,
                "op_name": op_name,
                "module": module,
                "curv": curv_cpu,
                "wanda": wanda_cpu,
                "magnitude_fallback": magnitude_fallback,
                "curv_largest": getattr(args, "prune_score_order", "high_to_low") == "high_to_low",
            }


def _entry_groups(args, entries):
    scope = prune_scope_from_args(args)
    if scope == "global":
        return [("all_layers", entries)]
    if scope == "per_layer":
        groups = {}
        for entry in entries:
            groups.setdefault(entry["layer_idx"], []).append(entry)
        return [(f"layer_{layer_idx}", group) for layer_idx, group in sorted(groups.items())]
    return [
        (f"layer_{entry['layer_idx']}_{entry['op_name']}", [entry])
        for entry in entries
    ]


def append_prune_overlap_log(log_path, args, model, wanda_scores):
    if log_path is None or wanda_scores is None:
        return

    entries = list(_iter_overlap_entries(args, model, wanda_scores))
    if not entries:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a+", encoding="utf-8") as f:
            print(
                "pruned_parameter_overlap skipped=no_matching_curvature_wanda_scores",
                file=f,
                flush=True,
            )
            print("", file=f, flush=True)
        return

    totals = {
        "curv": 0,
        "wanda": 0,
        "magnitude": 0,
        "curv_wanda": 0,
        "curv_magnitude": 0,
    }
    for _, group in _entry_groups(args, entries):
        curv_masks = _masks_for_curvature(
            group,
            args.sparsity_ratio,
        )
        for curv_mask in curv_masks:
            totals["curv"] += int(curv_mask.sum().item())

        wanda_masks = _masks_for_metric(
            group,
            "wanda",
            args.sparsity_ratio,
            largest=False,
        )
        for curv_mask, wanda_mask in zip(curv_masks, wanda_masks):
            totals["wanda"] += int(wanda_mask.sum().item())
            totals["curv_wanda"] += int((curv_mask & wanda_mask).sum().item())
        del wanda_masks

        magnitude_masks = _masks_for_metric(
            group,
            "magnitude",
            args.sparsity_ratio,
            largest=False,
        )
        for curv_mask, magnitude_mask in zip(curv_masks, magnitude_masks):
            totals["magnitude"] += int(magnitude_mask.sum().item())
            totals["curv_magnitude"] += int((curv_mask & magnitude_mask).sum().item())
        del curv_masks, magnitude_masks

    curv_count = totals["curv"]
    wanda_ratio = totals["curv_wanda"] / curv_count if curv_count else 0.0
    magnitude_ratio = totals["curv_magnitude"] / curv_count if curv_count else 0.0

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a+", encoding="utf-8") as f:
        print(
            f"pruned_parameter_overlap target_sparsity={float(args.sparsity_ratio):.4f}, "
            f"prune_scope={getattr(args, 'curvature_prune_scope', 'global')}, "
            f"curvature_score_order={getattr(args, 'prune_score_order', 'high_to_low')}, "
            "compare_methods=wanda,magnitude",
            file=f,
            flush=True,
        )
        print(
            f"curvature_pruned={totals['curv']} wanda_pruned={totals['wanda']} "
            f"magnitude_pruned={totals['magnitude']} "
            f"curv_wanda_overlap={totals['curv_wanda']} "
            f"curv_magnitude_overlap={totals['curv_magnitude']} "
            f"curv_wanda_overlap_over_curv={wanda_ratio:.6f} "
            f"curv_magnitude_overlap_over_curv={magnitude_ratio:.6f}",
            file=f,
            flush=True,
        )
        print("", file=f, flush=True)
