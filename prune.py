import os
import pickle
import torch 
import torch.nn as nn 


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 


def prepare_calibration_input(model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    # ===== device handling =====
    if isinstance(device, str):
        device = torch.device(device)

    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]
        device = torch.device(f"cuda:{dev}") if isinstance(dev, int) else dev

    # ===== allocate =====
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device
    )

    cache = {
        "i": 0,
        "attention_mask": None,
        "position_ids": None
    }

    # ===== catcher =====
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            i = cache["i"]
            # ✅ stop if enough samples collected
            if i >= nsamples:
                raise StopIteration

            if inp.dim() == 3:
                if inp.shape[0] != 1:
                    raise ValueError(
                        f"Expected calibration batch size 1, got input shape {tuple(inp.shape)}"
                    )
                inp = inp.squeeze(0)
            elif inp.dim() != 2:
                raise ValueError(f"Unexpected calibration input shape {tuple(inp.shape)}")

            inps[i].copy_(inp)   # faster + safer than assignment
            cache["i"] += 1

            # only save once (they're usually same shape)
            if cache["attention_mask"] is None:
                cache["attention_mask"] = kwargs.get("attention_mask", None)

            if cache["position_ids"] is None:
                cache["position_ids"] = kwargs.get("position_ids", None)

            raise ValueError  # stop this forward

    # ===== replace first layer =====
    layers[0] = Catcher(layers[0])

    # ===== run dataloader =====
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
        except StopIteration:
            break  # ✅ stop early when enough samples collected

    # ===== restore layer =====
    layers[0] = layers[0].module

    # ===== outputs placeholder =====
    outs = torch.zeros_like(inps)

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    # ===== restore config =====
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def _clone_pickleable_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _clone_pickleable_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_pickleable_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_pickleable_to_cpu(v) for v in value)
    return value


def align_curvature_to_weight_shape(curv, weight_shape, context="curvature"):
    curv = curv.detach() if torch.is_tensor(curv) else torch.as_tensor(curv)
    if tuple(curv.shape) == tuple(weight_shape):
        return curv
    if tuple(curv.T.shape) == tuple(weight_shape):
        return curv.T
    raise ValueError(
        f"{context} shape mismatch: got {tuple(curv.shape)}, expected {tuple(weight_shape)} "
        f"or its transpose"
    )


def save_layer_curvature_pkl(layer_idx, curvature_scores, save_dir, metadata=None):
    if save_dir is None:
        return None

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"layer_{layer_idx:03d}_curvature.pkl")
    payload = {
        "layer_idx": int(layer_idx),
        "curvature_scores": {
            op_name: _clone_pickleable_to_cpu(curv)
            for op_name, curv in curvature_scores.items()
        },
    }
    if metadata is not None:
        payload["metadata"] = _clone_pickleable_to_cpu(metadata)

    with open(save_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    return save_path


def load_layer_curvature_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    layer_idx = int(payload["layer_idx"])
    metadata = payload.get("metadata", None)
    legacy_square_layout = not (
        isinstance(metadata, dict)
        and metadata.get("curvature_layout") == "weight_out_in"
    )
    curvature_scores = {
        op_name: (
            _clone_pickleable_to_cpu(curv.T)
            if (
                legacy_square_layout
                and torch.is_tensor(curv)
                and curv.ndim == 2
                and curv.shape[0] == curv.shape[1]
            )
            else _clone_pickleable_to_cpu(curv)
        )
        for op_name, curv in payload.get("curvature_scores", {}).items()
    }
    return layer_idx, curvature_scores, metadata


def _curvature_seq_tag(shared_top_k=None, shared_seq_select="top", curvature_lpf_window=0):
    if shared_top_k is None:
        return "curvature_pkl"
    if shared_seq_select == "top" and int(curvature_lpf_window) <= 1:
        return f"curv_topseq_{int(shared_top_k)}_pkl"
    tag = f"curv_{shared_seq_select}_seq_{int(shared_top_k)}"
    if int(curvature_lpf_window) > 1:
        tag += f"_lpf_{int(curvature_lpf_window)}"
    return f"{tag}_pkl"


def load_curvature_pkls(
    save_dir,
    num_layers,
    shared_top_k=None,
    shared_seq_select="top",
    curvature_lpf_window=0,
):
    curvature_scores = [{} for _ in range(num_layers)]
    if save_dir is None or not os.path.isdir(save_dir):
        return curvature_scores

    nested_dirs = []
    if shared_top_k is not None:
        nested_dirs.append(os.path.join(
            save_dir,
            _curvature_seq_tag(shared_top_k, shared_seq_select, curvature_lpf_window),
        ))
    nested_dirs.append(os.path.join(save_dir, "curvature_pkl"))
    for nested_save_dir in nested_dirs:
        if os.path.isdir(nested_save_dir):
            save_dir = nested_save_dir
            break

    for file_name in sorted(os.listdir(save_dir)):
        if not (file_name.startswith("layer_") and file_name.endswith("_curvature.pkl")):
            continue

        layer_idx, layer_scores, _ = load_layer_curvature_pkl(os.path.join(save_dir, file_name))
        if 0 <= layer_idx < num_layers:
            curvature_scores[layer_idx] = layer_scores

    return curvature_scores


def _ensure_loaded_curvature_scores(args, model):
    if hasattr(model, "curvature_scores"):
        return model.curvature_scores

    load_dir = getattr(args, "load_curvature_dir", None)
    if load_dir is None:
        return None

    model.curvature_scores = load_curvature_pkls(
        load_dir,
        len(model.model.layers),
        shared_top_k=getattr(args, "shared_top_k", 10),
        shared_seq_select=getattr(args, "shared_seq_select", "top"),
        curvature_lpf_window=getattr(args, "curvature_lpf_window", 0),
    )
    loaded = sum(len(layer_scores) for layer_scores in model.curvature_scores)
    print(f"Loaded {loaded} curvature tensors from {load_dir}")
    return model.curvature_scores


def _curvature_candidate_mask(args, model, layer_idx, name, weight):
    curvature_scores = _ensure_loaded_curvature_scores(args, model)
    if curvature_scores is None:
        return None

    short_name = name.split(".")[-1]
    layer_scores = curvature_scores[layer_idx] if layer_idx < len(curvature_scores) else {}
    curv = layer_scores.get(short_name)
    if curv is None:
        print(f"Skipping layer {layer_idx} {name}: no curvature values were loaded")
        return torch.zeros_like(weight, dtype=torch.bool, device=weight.device)

    curv = align_curvature_to_weight_shape(
        curv,
        weight.shape,
        context=f"layer {layer_idx} {name} loaded curvature",
    )

    return torch.isfinite(curv).to(device=weight.device)


def _select_lowest_mask(metric, candidate_mask, ratio):
    W_mask = torch.zeros_like(metric, dtype=torch.bool)
    eligible_count = int(candidate_mask.sum().item())
    prune_count = int(eligible_count * ratio)
    if prune_count <= 0:
        return W_mask

    candidate_scores = metric[candidate_mask].float()
    prune_count = min(prune_count, candidate_scores.numel())
    selected = torch.topk(candidate_scores, k=prune_count, largest=False, sorted=False).indices
    flat_candidate_positions = candidate_mask.reshape(-1).nonzero(as_tuple=False).flatten()
    W_mask.reshape(-1)[flat_candidate_positions[selected]] = True
    return W_mask
