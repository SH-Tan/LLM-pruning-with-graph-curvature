import torch

from data_c4 import get_loaders_c4
from layerwrapper import WrappedGPT
from prune import (
    find_layers,
    prepare_calibration_input,
    _curvature_candidate_mask,
    _select_lowest_mask,
)


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def _row_lowest_mask(metric, ratio):
    W_mask = torch.zeros_like(metric, dtype=torch.bool)
    prune_count = int(metric.shape[1] * ratio)
    if prune_count <= 0:
        return W_mask
    if prune_count >= metric.shape[1]:
        return torch.ones_like(metric, dtype=torch.bool)

    threshold = torch.kthvalue(metric, k=prune_count, dim=1).values.reshape(-1, 1)
    return metric <= threshold


def compute_wanda_scores(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print(f"loading calibration data for WANDA scores with seqlen={model.seqlen}")
    dataloader, _ = get_loaders_c4(
        args.calib_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )
    del dataloader

    layers = model.model.layers
    model.wanda_scores = [{} for _ in range(len(layers))]

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if hasattr(model, "hf_device_map") and (f"model.layers.{i}" in model.hf_device_map):
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dev)
            if position_ids is not None:
                position_ids = position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                cos, sin = model.model.rotary_emb(inps[j].unsqueeze(0), position_ids)
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=(cos, sin),
                )[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"collecting WANDA scores layer {i} name {name}")
            weight_cpu = subset[name].weight.detach().cpu()
            scaler_cpu = wrapped_layers[name].scaler_row.detach().cpu()
            W_metric = torch.abs(weight_cpu) * torch.sqrt(scaler_cpu.reshape((1, -1)))
            model.wanda_scores[i][name] = W_metric.clone()
            del weight_cpu, scaler_cpu
            del W_metric

        inps, outs = outs, inps
        del wrapped_layers, handles

    model.config.use_cache = use_cache
    del inps, outs
    torch.cuda.empty_cache()
    return model.wanda_scores


def _apply_wanda_scores(args, model, wanda_scores, prune_n=0, prune_m=0):
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        layer_scores = wanda_scores[i] if i < len(wanda_scores) else {}

        for name in subset:
            print(f"pruning layer {i} name {name}")
            if name not in layer_scores:
                raise KeyError(f"Missing precomputed WANDA scores for layer {i} name {name}")

            W = subset[name].weight.data
            W_metric = layer_scores[name].detach().cpu()
            if W_metric.shape != W.shape:
                raise ValueError(
                    f"WANDA score shape mismatch for layer {i} {name}: "
                    f"{tuple(W_metric.shape)} vs {tuple(W.shape)}"
                )

            candidate_mask = _curvature_candidate_mask(args, model, i, name, W)
            if candidate_mask is not None:
                candidate_mask = candidate_mask.cpu()

            W_mask = (torch.zeros_like(W_metric) == 1)
            if prune_n != 0:
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
                if candidate_mask is not None and not args.use_variant:
                    W_mask = _select_lowest_mask(W_metric, candidate_mask, args.sparsity_ratio)
                    W[W_mask.to(device=W.device)] = 0
                    del W_metric, W_mask
                    continue

                if candidate_mask is not None and args.use_variant:
                    print("WANDA variant with loaded curvature uses finite-curvature candidates only")
                    W_metric = W_metric.masked_fill(~candidate_mask, float("inf"))

                if args.use_variant:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    W_mask = _row_lowest_mask(W_metric, args.sparsity_ratio)
                    if candidate_mask is not None:
                        W_mask &= candidate_mask

            W[W_mask.to(device=W.device)] = 0
            del W_metric, W_mask


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if not hasattr(model, "wanda_scores"):
        compute_wanda_scores(args, model, tokenizer, device)

    _apply_wanda_scores(args, model, model.wanda_scores, prune_n, prune_m)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
