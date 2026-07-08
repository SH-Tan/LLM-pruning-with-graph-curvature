import torch
import math
import numpy as np
import sys
import os


np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=sys.maxsize)

# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask.to(dtype=query.dtype) + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    attn_weight = attn_weight + attn_bias

    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)

    return attn_weight @ value, attn_weight



def _reshape_for_heads(x, num_heads, head_dim):
    batch, seq_len, _ = x.shape
    return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()


def _merge_heads(x):
    batch, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq_len, num_heads * head_dim)


def _repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _resolve_attention_dims(layer, model):
    attn = layer.self_attn
    config = getattr(attn, "config", None) or getattr(model, "config", None)

    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None and config is not None:
        head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = attn.q_proj.out_features // getattr(config, "num_attention_heads", 1)

    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None and config is not None:
        num_heads = getattr(config, "num_attention_heads", None)
    if num_heads is None:
        num_heads = attn.q_proj.out_features // head_dim

    num_kv_heads = getattr(attn, "num_key_value_heads", None)
    if num_kv_heads is None and config is not None:
        num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = attn.k_proj.out_features // head_dim

    num_kv_groups = getattr(attn, "num_key_value_groups", None)
    if num_kv_groups is None:
        num_kv_groups = max(num_heads // num_kv_heads, 1)

    return num_heads, num_kv_heads, num_kv_groups, head_dim


def _store_operation(op_bank, name, node, dtype=None):
    node = node.detach().cpu()
    if dtype is not None and torch.is_floating_point(node):
        node = node.to(dtype=dtype)
    op_bank[name] = node # _normalize_node_value_per_sequence(node, name)


def _plot_sorted_input_distribution(name, tensor, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    try:
        os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        marker_path = os.path.join(plot_dir, "plot_error.txt")
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(f"Could not draw {name} distribution plot: {exc}\n")
        return

    def _plot_values(ax, values, title, ylabel):
        sorted_values = torch.sort(values).values.cpu().numpy()
        max_points = int(os.environ.get("CURV_GATE_PLOT_MAX_POINTS", "200000"))
        if sorted_values.size > max_points:
            idx = np.linspace(0, sorted_values.size - 1, max_points, dtype=np.int64)
            sorted_values = sorted_values[idx]
        ax.plot(sorted_values, linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("sorted index")
        ax.set_ylabel(ylabel)

    marker_path = os.path.join(plot_dir, "plot_error.txt")
    try:
        values = tensor.detach()
        if values.dim() > 1 and values.shape[0] == 1:
            values = values.squeeze(0)
        if values.dim() != 2:
            values = values.reshape(-1, values.shape[-1])

        chunk_size = int(os.environ.get("CURV_GATE_PLOT_SEQ_CHUNK", "256"))
        if os.environ.get("CURV_GATE_PLOT_PER_SAMPLE", "1") == "1":
            seq_l2_chunks = []
            for start in range(0, values.shape[0], chunk_size):
                chunk = values[start:start + chunk_size].float()
                chunk = torch.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
                seq_l2_chunks.append(torch.linalg.vector_norm(chunk, ord=2, dim=-1).cpu())
                del chunk
            seq_l2 = torch.cat(seq_l2_chunks)
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            _plot_values(ax, seq_l2, f"{name} seq L2", "L2 norm")
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"{name}_seq_l2.png"), dpi=140)
            plt.close(fig)
            del seq_l2

            sample_count = min(10, values.shape[0])
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(os.environ.get("CURV_GATE_PLOT_SEED", "0")))
            seq_idx = torch.randperm(values.shape[0], generator=generator)[:sample_count]
            sampled = values.index_select(0, seq_idx.to(values.device)).float().abs().cpu()
            sampled = torch.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            for row, idx in zip(sampled, seq_idx.tolist()):
                sorted_values = torch.sort(row).values.numpy()
                ax.plot(sorted_values, linewidth=0.8, label=f"seq {idx}")
            ax.set_title(f"{name} sampled seq abs")
            ax.set_xlabel("sorted hidden index")
            ax.set_ylabel("abs(value)")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"{name}_sampled_seq_abs.png"), dpi=140)
            plt.close(fig)
            del sampled

        if os.environ.get("CURV_GATE_PLOT_WANDA_L2") == "1":
            layer_dir = os.path.dirname(plot_dir) if os.path.basename(plot_dir).startswith("sample_") else plot_dir
            wanda_dir = os.path.join(layer_dir, "wanda_l2")
            os.makedirs(wanda_dir, exist_ok=True)
            acc_path = os.path.join(wanda_dir, f"{name}_sumsq.npz")
            reset_acc = os.path.basename(plot_dir) == "sample_000"

            sumsq = torch.zeros(values.shape[-1], dtype=torch.float64, device="cpu")
            for start in range(0, values.shape[0], chunk_size):
                chunk = values[start:start + chunk_size].float()
                chunk = torch.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
                sumsq += chunk.square().sum(dim=0).double().cpu()
                del chunk
            count = values.shape[0]
            if os.path.exists(acc_path) and not reset_acc:
                acc = np.load(acc_path)
                sumsq += torch.from_numpy(acc["sumsq"])
                count += int(acc["count"])
            np.savez(acc_path, sumsq=sumsq.numpy(), count=np.array(count, dtype=np.int64))

            wanda_l2 = torch.sqrt(sumsq)
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            _plot_values(ax, wanda_l2, f"{name} Wanda L2", "L2 norm over examples and seq")
            fig.tight_layout()
            fig.savefig(os.path.join(wanda_dir, f"{name}_wanda_l2.png"), dpi=140)
            plt.close(fig)
            del wanda_l2, sumsq
        del values
    except Exception as exc:
        with open(marker_path, "a", encoding="utf-8") as f:
            f.write(f"Could not draw {name} distribution plot: {exc}\n")
        try:
            plt.close("all")
        except Exception:
            pass


def _maybe_plot_mlp_gate_distributions(x_norm, attn_context, x_norm2, mlp_hidden):
    if os.environ.get("CURV_GATE_PLOT_ENABLED") != "1":
        return

    plot_dir = os.environ.get("CURV_GATE_PLOT_DIR")
    if not plot_dir:
        return

    plot_tag = os.environ.get("CURV_GATE_PLOT_TAG")
    if plot_tag:
        step_dir = os.path.join(plot_dir, plot_tag)
    else:
        plot_idx = getattr(_maybe_plot_mlp_gate_distributions, "_plot_idx", 0)
        step_dir = os.path.join(plot_dir, f"mlp_gate_{plot_idx:03d}")
        _maybe_plot_mlp_gate_distributions._plot_idx = plot_idx + 1
    _plot_sorted_input_distribution("x_norm", x_norm, step_dir)
    _plot_sorted_input_distribution("attn_context", attn_context, step_dir)
    _plot_sorted_input_distribution("x_norm2", x_norm2, step_dir)
    _plot_sorted_input_distribution("mlp_hidden", mlp_hidden, step_dir)


def _print_finite_stats(name, tensor):
    if os.environ.get("CURV_GATE_PRINT_STATS") != "1":
        return
    values = tensor.detach()
    finite = torch.isfinite(values)
    finite_count = int(finite.sum().item())
    total_count = values.numel()
    if finite_count > 0:
        finite_values = values[finite].float()
        print(
            f"{name}: shape={tuple(values.shape)}, finite={finite_count}/{total_count}, "
            f"min={float(finite_values.min().item()):.6g}, "
            f"max={float(finite_values.max().item()):.6g}, "
            f"mean={float(finite_values.mean().item()):.6g}"
        )
    else:
        print(f"{name}: shape={tuple(values.shape)}, finite=0/{total_count}")



def collect_layer_data(layer, x, attention_mask, position_ids, model, next_layer=None, operation_dtype=None):
    operations = {}

    with torch.no_grad():
        x_in = x
        x_norm = layer.input_layernorm(x_in)
        
        # ---- store shared layer input ----
        _store_operation(operations, "layer_input", x_norm, operation_dtype)

        if not hasattr(layer, "_cached_dims"):
            layer._cached_dims = _resolve_attention_dims(layer, model)

        num_heads, num_kv_heads, num_kv_groups, head_dim = layer._cached_dims

        q_linear = layer.self_attn.q_proj(x_norm)
        k_linear = layer.self_attn.k_proj(x_norm)
        v_linear = layer.self_attn.v_proj(x_norm)

        q = _reshape_for_heads(q_linear, num_heads, head_dim) # [1, 32, 8192, 128]
        k = _reshape_for_heads(k_linear, num_kv_heads, head_dim) # [1, 8, 8192, 128]
        v = _reshape_for_heads(v_linear, num_kv_heads, head_dim) # [1, 8, 8192, 128]

        # ---- RoPE ----
        if position_ids is not None:
            cos, sin = model.model.rotary_emb(x_norm, position_ids)
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)
            del cos, sin
        
        # Keep GQA-expanded K/V aligned with the expanded curvature weight layout.
        _store_operation(operations, "q_proj", _merge_heads(q), operation_dtype)
        _store_operation(operations, "k_proj", _merge_heads(k), operation_dtype)
        _store_operation(operations, "v_proj", v_linear, operation_dtype)
        
        k = _repeat_kv(k, num_kv_groups) # [1, 32, 8192, 128]
        v = _repeat_kv(v, num_kv_groups) # [1, 32, 8192, 128]

        attn_output, A = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attention_mask is None,
        )  # [1, 32, 8192, 128]
        
        attn_context = _merge_heads(attn_output) # [1, 8192, 4096]
        
        # ---- store Attention weight and out ----
        A = torch.where(torch.isinf(A), torch.zeros_like(A), A)
        _store_operation(operations, "A", A, operation_dtype)
        _store_operation(operations, "Att_out", attn_context, operation_dtype)

        # ===== O PROJ =====
        o = layer.self_attn.o_proj(attn_context)
        x_res1 = x_in + o
        x_norm2 = layer.post_attention_layernorm(x_res1)
        _print_finite_stats("attn_context", attn_context)
        _print_finite_stats("x_norm2", x_norm2)
        
        # ---- residual value for qkv output ----
        # _store_operation(operations, "qkv_residual", x_res1, operation_dtype)
        _store_operation(operations, "o_residual", x_in, operation_dtype)
        
        # ---- attention output ----
        _store_operation(operations, "o_proj", x_norm2, operation_dtype)

        # ===== MLP =====
        gate = layer.mlp.gate_proj(x_norm2)
        up = layer.mlp.up_proj(x_norm2)
        act = layer.mlp.act_fn(gate)
        mlp_hidden = act * up
        
        gate_beta = torch.ones_like(gate)
        
        # gate_beta = torch.sigmoid(gate)
        # gate_beta.ge_(0.2)
        # gate_beta = gate_beta.to(dtype=gate.dtype)
        # _maybe_plot_mlp_gate_distributions(x_norm, attn_context, x_norm2, mlp_hidden)

        down = layer.mlp.down_proj(mlp_hidden)
        x_out = x_res1 + down
        
        _store_operation(operations, "gate_beta", gate_beta, operation_dtype)
        _store_operation(operations, "gate_proj", act, operation_dtype)
        _store_operation(operations, "up_proj", up, operation_dtype)
        _store_operation(operations, "gate_up_out", mlp_hidden, operation_dtype)

        # if next_layer is not None and hasattr(next_layer, "input_layernorm"):
        #     next_ln = next_layer.input_layernorm
        #     next_ln_device = next_ln.weight.device
        #     next_input = x_out.to(next_ln_device)
        #     next_input_norm = next_ln(next_input)
        #     _store_operation(operations, "down_proj", next_input_norm, operation_dtype)
        #     del next_input, next_input_norm

        # else:
            # _store_operation(operations, "down_proj", x_out, operation_dtype)
        
        _store_operation(operations, "down_proj", down, operation_dtype)
            
        # ---- cleanup (GPU memory critical) ----
        del q, k, v, A, attn_output
        del q_linear, k_linear, v_linear
        del attn_context, o, x_res1, x_norm2
        del gate, gate_beta, up, act, mlp_hidden, down

    return x_out, operations, num_heads, num_kv_heads, num_kv_groups, head_dim
