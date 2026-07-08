import torch

from curvature_utils.curv_dtype_utils import curvature_torch_dtype


def _source_tensor_for_cost(operations, short_name):
    if short_name.startswith("prev_"):
        real_name = short_name.replace("prev_", "")
        source_name = f"prev_{_source_name_for_cost(real_name)}"
    else:
        source_name = _source_name_for_cost(short_name)
    source = operations.get(source_name)
    if source is None:
        raise ValueError(f"Missing source activation {source_name} for {short_name} edge cost")
    return source


def _source_name_for_cost(short_name):
    if short_name in {"q_proj", "k_proj", "v_proj"}:
        return "layer_input"
    if short_name == "o_proj":
        return "Att_out"
    if short_name in {"gate_proj", "up_proj"}:
        return "o_proj"
    if short_name == "down_proj":
        return "gate_up_out"
    return short_name


def _input_magnitude_for_cost(
    operations,
    short_name,
    in_dim,
    device,
    l2_norm=False,
    l2_norm_mode="per_example",
):
    source = _source_tensor_for_cost(operations, short_name)
    if not torch.is_tensor(source):
        source = torch.as_tensor(source)
    source = source.detach()
    if source.dim() == 1:
        mag = source.abs().to(dtype=curvature_torch_dtype())
    elif l2_norm and l2_norm_mode == "per_example":
        mag = torch.norm(
            source.to(dtype=curvature_torch_dtype()).reshape(-1, source.shape[-1]),
            p=2,
            dim=0,
        )
    else:
        mag = source.abs().to(dtype=curvature_torch_dtype()).reshape(-1, source.shape[-1]).mean(dim=0)
    if mag.numel() != int(in_dim):
        raise ValueError(
            f"Source activation width mismatch for {short_name}: "
            f"got {mag.numel()}, expected {int(in_dim)}"
        )
    return mag.to(device=device, non_blocking=True)


def _weight_from_model(model, short_name, layer_id, device=None):
    if short_name.startswith("prev_"):
        real_name = short_name.replace("prev_", "")
        if layer_id == 0:
            return None
        return _weight_from_model(model, real_name, layer_id - 1, device=device)

    layer = model.model.layers[layer_id]
    if short_name == "q_proj":
        w = layer.self_attn.q_proj.weight.detach()
    elif short_name == "k_proj":
        w = layer.self_attn.k_proj.weight.detach()
    elif short_name == "v_proj":
        w = layer.self_attn.v_proj.weight.detach()
    elif short_name == "o_proj":
        w = layer.self_attn.o_proj.weight.detach()
    elif short_name == "gate_proj":
        w = layer.mlp.gate_proj.weight.detach()
    elif short_name == "up_proj":
        w = layer.mlp.up_proj.weight.detach()
    elif short_name == "down_proj":
        w = layer.mlp.down_proj.weight.detach()
    else:
        raise KeyError(short_name)

    return w if device is None else w.to(device, non_blocking=True)


def _operation_distance_matrix_torch(
    model,
    operations,
    short_name,
    layer_id,
    device,
    l2_norm=False,
    l2_norm_mode="per_example",
):
    weight = _weight_from_model(model, short_name, layer_id, device=device)
    abs_w = weight.abs().to(dtype=curvature_torch_dtype())
    input_mag = _input_magnitude_for_cost(
        operations,
        short_name,
        weight.shape[1],
        device,
        l2_norm=l2_norm,
        l2_norm_mode=l2_norm_mode,
    )
    abs_w.mul_(input_mag.view(1, -1))
    cost = 1.0 / abs_w
    dist = cost.transpose(0, 1).contiguous()
    expected_shape = (weight.shape[1], weight.shape[0])
    del weight, abs_w, cost, input_mag
    if dist.shape != expected_shape:
        raise ValueError(
            f"Unexpected cost matrix shape for {short_name}: "
            f"got {tuple(dist.shape)}, expected {expected_shape}"
        )
    return dist
