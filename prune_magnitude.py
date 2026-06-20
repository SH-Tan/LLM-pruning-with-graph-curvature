import torch

from prune import (
    find_layers,
    _curvature_candidate_mask,
    prune_scope_from_args,
    score_order_largest,
    select_prune_masks_by_score,
    should_prune_op,
    skip_prune_layer,
)
from prune_log_utils import (
    append_all_layer_pruned_parameter_log,
    append_layer_pruned_parameter_log,
    collect_pruned_parameter_rows,
)


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers
    report_limit = int(getattr(args, "all_layer_report_rank_offset", 0)) + 25
    report_rows = []
    scope = prune_scope_from_args(args)

    if prune_n == 0 and scope == "global":
        entries = []
        for i, layer in enumerate(layers):
            if skip_prune_layer(args, i):
                print(f"Skipping layer {i}: requested by skip_prune_layer_ids")
                continue
            subset = find_layers(layer)
            for name, module in subset.items():
                if not should_prune_op(args, name):
                    continue
                W = module.weight.data
                W_metric = torch.abs(W).detach().cpu()
                candidate_mask = _curvature_candidate_mask(args, model, i, name, W)
                if candidate_mask is not None:
                    candidate_mask = candidate_mask.cpu()
                entries.append(
                    {
                        "layer_idx": i,
                        "name": name,
                        "module": module,
                        "metric": W_metric,
                        "candidate_mask": candidate_mask,
                    }
                )

        largest = score_order_largest(args, default=False)
        prune_masks, _ = select_prune_masks_by_score(entries, args.sparsity_ratio, largest=largest)
        layer_report_rows_by_idx = {}
        for entry, W_mask in zip(entries, prune_masks):
            layer_idx = entry["layer_idx"]
            name = entry["name"]
            module = entry["module"]
            report_rows.extend(
                collect_pruned_parameter_rows(
                    layer_idx,
                    name,
                    module,
                    entry["metric"],
                    W_mask,
                    largest=largest,
                    limit=report_limit,
                )
            )
            layer_report_rows_by_idx.setdefault(layer_idx, []).extend(
                collect_pruned_parameter_rows(
                    layer_idx,
                    name,
                    module,
                    entry["metric"],
                    W_mask,
                    largest=largest,
                    limit=25,
                )
            )
            module.weight.data[W_mask.to(device=module.weight.data.device)] = 0

        for layer_idx, layer_report_rows in sorted(layer_report_rows_by_idx.items()):
            append_layer_pruned_parameter_log(
                getattr(args, "all_layer_parameter_log_path", None),
                args,
                "magnitude",
                layer_idx,
                getattr(args, "prune_score_order", "low_to_high"),
                "magnitude",
                layer_report_rows,
                largest=largest,
            )

        append_all_layer_pruned_parameter_log(
            getattr(args, "all_layer_parameter_log_path", None),
            args,
            "magnitude",
            getattr(args, "prune_score_order", "low_to_high"),
            "magnitude",
            report_rows,
            largest=largest,
            rank_offset=getattr(args, "all_layer_report_rank_offset", 0),
        )
        return

    for i in range(len(layers)):
        if skip_prune_layer(args, i):
            print(f"Skipping layer {i}: requested by skip_prune_layer_ids")
            continue
        layer = layers[i]
        subset = find_layers(layer)
        layer_report_rows = []

        layer_entries = []
        if prune_n == 0 and scope == "per_layer":
            for name, module in subset.items():
                if not should_prune_op(args, name):
                    continue
                W = module.weight.data
                W_metric = torch.abs(W)
                candidate_mask = _curvature_candidate_mask(args, model, i, name, W)
                layer_entries.append(
                    {
                        "name": name,
                        "module": module,
                        "metric": W_metric,
                        "candidate_mask": candidate_mask,
                    }
                )
            largest = score_order_largest(args, default=False)
            prune_masks, _ = select_prune_masks_by_score(
                layer_entries,
                args.sparsity_ratio,
                largest=largest,
            )
            for entry, W_mask in zip(layer_entries, prune_masks):
                name = entry["name"]
                module = entry["module"]
                report_rows.extend(
                    collect_pruned_parameter_rows(
                        i,
                        name,
                        module,
                        entry["metric"],
                        W_mask,
                        largest=largest,
                        limit=report_limit,
                    )
                )
                layer_report_rows.extend(
                    collect_pruned_parameter_rows(
                        i,
                        name,
                        module,
                        entry["metric"],
                        W_mask,
                        largest=largest,
                        limit=25,
                    )
                )
                module.weight.data[W_mask.to(device=module.weight.data.device)] = 0

            append_layer_pruned_parameter_log(
                getattr(args, "all_layer_parameter_log_path", None),
                args,
                "magnitude",
                i,
                getattr(args, "prune_score_order", "low_to_high"),
                "magnitude",
                layer_report_rows,
                largest=largest,
            )
            continue

        for name in subset:
            if not should_prune_op(args, name):
                continue
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            candidate_mask = _curvature_candidate_mask(args, model, i, name, W)
            largest = score_order_largest(args, default=False)
            if prune_n != 0:
                W_mask = torch.zeros_like(W, dtype=torch.bool)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        group_candidate = (
                            None if candidate_mask is None else candidate_mask[:, ii:(ii + prune_m)]
                        )
                        if group_candidate is not None:
                            tmp = tmp.masked_fill(~group_candidate, float("inf"))
                        selected = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                        if group_candidate is not None:
                            selected_mask = torch.gather(group_candidate, 1, selected)
                            selected = selected.masked_fill(~selected_mask, 0)
                        W_mask.scatter_(1, ii + selected, True)
                if candidate_mask is not None:
                    W_mask &= candidate_mask
            else:
                W_mask = select_prune_masks_by_score(
                    [
                        {
                            "metric": W_metric,
                            "candidate_mask": candidate_mask,
                        }
                    ],
                    args.sparsity_ratio,
                    largest=largest,
                )[0][0]

            report_rows.extend(
                collect_pruned_parameter_rows(
                    i,
                    name,
                    subset[name],
                    W_metric,
                    W_mask,
                    largest=largest,
                    limit=report_limit,
                )
            )
            layer_report_rows.extend(
                collect_pruned_parameter_rows(
                    i,
                    name,
                    subset[name],
                    W_metric,
                    W_mask,
                    largest=largest,
                    limit=25,
                )
            )
            W[W_mask] = 0

        append_layer_pruned_parameter_log(
            getattr(args, "all_layer_parameter_log_path", None),
            args,
            "magnitude",
            i,
            getattr(args, "prune_score_order", "low_to_high"),
            "magnitude",
            layer_report_rows,
            largest=score_order_largest(args, default=False),
        )

    append_all_layer_pruned_parameter_log(
        getattr(args, "all_layer_parameter_log_path", None),
        args,
        "magnitude",
        getattr(args, "prune_score_order", "low_to_high"),
        "magnitude",
        report_rows,
        largest=score_order_largest(args, default=False),
        rank_offset=getattr(args, "all_layer_report_rank_offset", 0),
    )
