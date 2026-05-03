import torch

from prune import find_layers, _curvature_candidate_mask, _select_lowest_mask


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            candidate_mask = _curvature_candidate_mask(args, model, i, name, W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W) == 1)
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
                if candidate_mask is not None:
                    W_mask = _select_lowest_mask(W_metric, candidate_mask, args.sparsity_ratio)
                else:
                    W_mask = _select_lowest_mask(
                        W_metric,
                        torch.ones_like(W_metric, dtype=torch.bool),
                        args.sparsity_ratio,
                    )

            W[W_mask] = 0
