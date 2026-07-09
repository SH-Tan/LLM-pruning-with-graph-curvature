import json
import os
import shutil
import subprocess
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.cuda_memory_utils import release_cuda_memory
from pruning.curv_layer_prune_utils import (
    draw_method_comparison,
    draw_ppl_vs_sparsity,
    prune_scoped_curvature,
    save_eval_records_csv,
)
from pruning.curv_prune_utils import prune_global_curvature
from evaluation.eval import eval_ppl
from evaluation.lm_eval_pipeline import (
    DEFAULT_FINAL_TASKS,
    MMLU_CATEGORY_TASKS,
    MMLU_SUMMARY_TASKS,
    append_scores_to_csv,
    extract_main_scores,
    find_latest_result_json,
    run_lm_eval,
    save_pruned_model,
)
from pruning.per_layer_eval_utils import (
    draw_per_layer_method_comparison,
    draw_per_layer_ppl_vs_sparsity,
    layer_sparsity,
    list_curvature_pkl_layers,
    load_curvature_scores_for_layer,
    prune_curvature_layer,
    prune_magnitude_layer,
    prune_wanda_layer,
    save_per_layer_records_csv,
)
from pruning.prune import check_sparsity
from pruning.prune_magnitude import prune_magnitude
from pruning.prune_wanda import prune_wanda
from downstream_test.local_task_utils import resolve_local_task


def l2_path_tag(args):
    if not getattr(args, "l2_norm", False):
        return "no_L2_norm"
    l2_norm_mode = getattr(args, "l2_norm_mode", "per_example")
    return "L2_norm" if l2_norm_mode == "per_example" else f"L2_norm_{l2_norm_mode}"


def curvature_seq_tag(shared_top_k=None, shared_seq_select="top", curvature_lpf_window=0):
    if shared_top_k is None:
        return "curvature_pkl"
    if shared_seq_select == "top" and int(curvature_lpf_window) <= 1:
        return f"curv_topseq_{int(shared_top_k)}_pkl"
    tag = f"curv_{shared_seq_select}_seq_{int(shared_top_k)}"
    if int(curvature_lpf_window) > 1:
        tag += f"_lpf_{int(curvature_lpf_window)}"
    return f"{tag}_pkl"


def log_path(args):
    save_dir = os.path.join(args.save, args.calib_data, l2_path_tag(args), f"seq_len_{args.seqlen}")
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"eval_out_{args.prune_method}.txt"
    if args.prune_method == "curvature":
        setting_tag = curvature_seq_tag(
            args.shared_top_k,
            args.shared_seq_select,
            args.curvature_lpf_window,
        ).removesuffix("_pkl")
        file_name = f"eval_out_{args.prune_method}_{setting_tag}.txt"
    return os.path.join(save_dir, file_name)


def contains_curvature_pkls(path, shared_top_k=None, shared_seq_select="top", curvature_lpf_window=0):
    if path is None or not os.path.isdir(path):
        return False

    search_dirs = [path]
    if shared_top_k is not None:
        search_dirs.append(os.path.join(
            path,
            curvature_seq_tag(shared_top_k, shared_seq_select, curvature_lpf_window),
        ))
    search_dirs.append(os.path.join(path, "curvature_pkl"))

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        if any(
            file_name.startswith("layer_") and file_name.endswith("_curvature.pkl")
            for file_name in os.listdir(search_dir)
        ):
            return True
    return False


def curvature_dir(args, base_dir, create=False):
    if base_dir is None:
        return None

    expected_seq_dir = f"seq_len_{args.seqlen}"
    if os.path.basename(os.path.normpath(base_dir)) == expected_seq_dir or contains_curvature_pkls(
        base_dir,
        getattr(args, "shared_top_k", None),
        getattr(args, "shared_seq_select", "top"),
        getattr(args, "curvature_lpf_window", 0),
    ):
        save_dir = base_dir
    else:
        save_dir = os.path.join(base_dir, args.calib_data, l2_path_tag(args), expected_seq_dir)

    if create:
        os.makedirs(save_dir, exist_ok=True)
    return save_dir


def append_curvature_prune_summary(log_file, prune_summary, target_ratio=None, score_order=None):
    if not prune_summary:
        return

    total_pruned = sum(row["pruned_params"] for row in prune_summary)
    with open(log_file, "a+") as f:
        summary_header = "\nPruned parameters total"
        if target_ratio is not None:
            summary_header += f" (target_sparsity={target_ratio:.4f}"
            if score_order is not None:
                summary_header += f", score_order={score_order}"
            summary_header += ")"
        print(summary_header, file=f, flush=True)
        print(f"pruned_params={total_pruned}", file=f, flush=True)


def append_eval_run_header(log_file, args, target_ratio, score_order):
    with open(log_file, "a+") as f:
        print(
            "\n"
            f"Prune run: method={args.prune_method}, "
            f"score_order={score_order}, "
            f"target_sparsity={target_ratio:.4f}, "
            f"score_seq_len={args.seqlen}, "
            f"calib_data={args.calib_data}, "
            f"l2_norm={args.l2_norm}, "
            f"l2_norm_mode={getattr(args, 'l2_norm_mode', 'per_example')}, "
            f"curvature_prune_scope={getattr(args, 'curvature_prune_scope', 'global')}, "
            f"prune_ops={','.join(getattr(args, 'prune_ops', None) or ['all'])}, "
            f"shared_top_k={args.shared_top_k}, "
            f"shared_seq_select={args.shared_seq_select}, "
            f"curvature_lpf_window={args.curvature_lpf_window}",
            file=f,
            flush=True,
        )
        if args.prune_method == "wanda":
            print(
                "WANDA eval mode: fixed_score_seq_len=precompute scores once at score_seq_len and sweep pp_seq_len",
                file=f,
                flush=True,
            )


def append_eval_result(
    log_file,
    args,
    score_order,
    target_ratio,
    actual_sparsity_ratio,
    eval_mode,
    score_seq_len,
    pp_seq_len,
    ppl_test,
):
    with open(log_file, "a+") as f:
        print(
            f"{args.prune_method:<15}{getattr(args, 'curvature_prune_scope', 'global'):<15}"
            f"{score_order:<15}{str(args.l2_norm):<10}{target_ratio:<18.4f}"
            f"{actual_sparsity_ratio:<18.4f}{args.calib_data:<20}{eval_mode:<28}"
            f"{score_seq_len:<16d}{pp_seq_len:<12d}{ppl_test:<12.4f}",
            file=f,
            flush=True,
        )
        print("", file=f, flush=True)


def append_per_layer_eval_result(
    log_file,
    args,
    layer_idx,
    score_order,
    target_ratio,
    layer_actual_sparsity,
    model_actual_sparsity,
    pp_seq_len,
    ppl_test,
    score_cutoff,
):
    with open(log_file, "a+") as f:
        print(
            f"per-layer method={args.prune_method}, layer={layer_idx}, score_order={score_order}, "
            f"target_sparsity={target_ratio:.4f}, layer_actual_sparsity={layer_actual_sparsity:.4f}, "
            f"model_actual_sparsity={model_actual_sparsity:.4f}, pp_seq_len={pp_seq_len}, "
            f"ppl_test={ppl_test:.4f}, score_cutoff={score_cutoff}",
            file=f,
            flush=True,
        )


def resolve_sparsity_ratios(args):
    ratios = [float(ratio) for ratio in args.sparsity_ratio]
    if not ratios:
        ratios = [0.0]

    deduped = []
    seen = set()
    for ratio in ratios:
        if ratio < 0 or ratio > 1:
            raise ValueError(f"sparsity_ratio must be in [0, 1], got {ratio}")
        if ratio not in seen:
            deduped.append(ratio)
            seen.add(ratio)

    if args.sparsity_schedule == "low_to_high":
        deduped.sort()
    elif args.sparsity_schedule == "high_to_low":
        deduped.sort(reverse=True)

    return deduped


def save_model_path(base_path, ratio, total_runs):
    if not base_path or total_runs <= 1:
        return base_path
    tag = f"{ratio:.4f}".rstrip("0").rstrip(".").replace(".", "p") or "0"
    return f"{base_path}_sparsity_{tag}"


def best_free_gpu_ids(count=1):
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return None

    gpu_free_memory = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            gpu_free_memory.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue
    if not gpu_free_memory:
        return None
    gpu_free_memory.sort(key=lambda item: item[1], reverse=True)
    return ",".join(str(index) for index, _ in gpu_free_memory[:count])


def downstream_model_path(args, compare_dir, compare_tag, ratio, total_runs, score_order, num_score_orders):
    base_path = getattr(args, "downstream_model_dir", "") or getattr(args, "save_model", "")
    if not base_path:
        base_path = os.path.join(compare_dir, "downstream_checkpoints", compare_tag)
    model_path = save_model_path(base_path, ratio, total_runs)
    if num_score_orders > 1:
        model_path = os.path.join(model_path, score_order)
    return model_path


def save_vllm_tokenizer_files(args, fallback_tokenizer, model_save_path):
    source_path = args.model if os.path.isdir(args.model) else None
    if source_path is None:
        from huggingface_hub import snapshot_download
        source_path = snapshot_download(args.model, local_files_only=True)

    fallback_tokenizer.save_pretrained(model_save_path)
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
    )
    for name in tokenizer_files:
        source_file = os.path.join(source_path, name)
        if os.path.exists(source_file):
            shutil.copy2(source_file, os.path.join(model_save_path, name))


def downstream_checkpoint_complete(model_path):
    if not model_path or not os.path.isdir(model_path):
        return False
    has_config = os.path.isfile(os.path.join(model_path, "config.json"))
    has_tokenizer = os.path.isfile(os.path.join(model_path, "tokenizer_config.json")) and (
        os.path.isfile(os.path.join(model_path, "tokenizer.json"))
        or os.path.isfile(os.path.join(model_path, "tokenizer.model"))
    )
    has_weights = (
        os.path.isfile(os.path.join(model_path, "model.safetensors"))
        or os.path.isfile(os.path.join(model_path, "model.safetensors.index.json"))
        or any(name.endswith(".safetensors") for name in os.listdir(model_path))
    )
    return has_config and has_tokenizer and has_weights


def save_downstream_checkpoint(args, model, tokenizer, model_path):
    if downstream_checkpoint_complete(model_path):
        print(f"reusing downstream checkpoint: {model_path}")
        return

    tmp_path = f"{model_path}.tmp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    shard_size = getattr(args, "downstream_save_shard_size", "2GB")
    model.save_pretrained(tmp_path, safe_serialization=True, max_shard_size=shard_size)
    save_vllm_tokenizer_files(args, tokenizer, tmp_path)
    os.replace(tmp_path, model_path)
    print(f"saved downstream checkpoint: {model_path}")


def parse_downstream_suite(args):
    if getattr(args, "downstream_suite", "") != "core":
        return None

    benchmarks = [item for item in getattr(args, "downstream_suite_benchmarks", "").split(";") if item]
    task_groups = [item for item in getattr(args, "downstream_suite_tasks", "").split(";") if item]
    backends = [item for item in getattr(args, "downstream_suite_backends", "").split(";") if item]
    fewshots = [item for item in getattr(args, "downstream_suite_fewshots", "").split(";") if item]
    limits = [item for item in getattr(args, "downstream_suite_limits", "").split(";")]
    if not benchmarks:
        return None
    if not (len(benchmarks) == len(task_groups) == len(backends) == len(fewshots) == len(limits)):
        raise ValueError("Downstream suite fields must have the same number of semicolon-separated entries.")

    suite = []
    for benchmark, tasks, backend, fewshot, limit in zip(benchmarks, task_groups, backends, fewshots, limits):
        suite.append(
            {
                "benchmark": benchmark,
                "tasks": [task for task in tasks.split(",") if task],
                "backend": backend,
                "num_fewshot": int(fewshot),
                "limit": limit or None,
            }
        )
    return suite


def hf_max_memory_per_gpu(memory_fraction, cuda_visible_devices=None):
    fraction = min(max(float(memory_fraction), 0.05), 0.95)
    if not torch.cuda.is_available():
        return None
    device_idx = 0
    if cuda_visible_devices:
        first_device = str(cuda_visible_devices).split(",", 1)[0].strip()
        if first_device.isdigit() and int(first_device) < torch.cuda.device_count():
            device_idx = int(first_device)
    total_bytes = torch.cuda.get_device_properties(device_idx).total_memory
    gib = max(1, int(total_bytes * fraction / (1024 ** 3)))
    return f"{gib}GiB"


def run_local_vllm_downstream_eval(
    args,
    model_path,
    benchmark_output_path,
    vllm_python,
    tensor_parallel_size,
    gpu_memory_utilization,
    dtype,
    env,
    max_examples=None,
    dataset_path=None,
    prompt_key=None,
):
    os.makedirs(benchmark_output_path, exist_ok=True)
    output_path = os.path.join(benchmark_output_path, "responses.jsonl")
    metrics_path = os.path.join(benchmark_output_path, "metrics.json")
    dataset_path = dataset_path or getattr(args, "downstream_task_data", "downstream_test/dataset/mathqa500/test.parquet")
    prompt_key = prompt_key or getattr(args, "downstream_prompt_key", "prompt")
    cmd = [
        vllm_python,
        "-m",
        "downstream_test.vllm_accuracy_runner",
        "--model_path",
        model_path,
        "--dataset_path",
        dataset_path,
        "--output_path",
        output_path,
        "--metrics_path",
        metrics_path,
        "--prompt_key",
        prompt_key,
        "--start_index",
        str(getattr(args, "downstream_start_index", 0)),
        "--max_examples",
        str(max_examples if max_examples is not None else getattr(args, "downstream_max_examples", 500)),
        "--batch_size",
        str(getattr(args, "downstream_local_batch_size", 1)),
        "--generation_max_batch_tokens",
        str(getattr(args, "downstream_generation_max_batch_tokens", 32768)),
        "--max_prompt_length",
        str(getattr(args, "downstream_max_prompt_length", 2048)),
        "--max_new_tokens",
        str(getattr(args, "downstream_max_new_tokens", 2048)),
        "--min_tokens",
        str(getattr(args, "downstream_min_tokens", 0)),
        "--temperature",
        str(getattr(args, "downstream_temperature", 0.0)),
        "--top_p",
        str(getattr(args, "downstream_top_p", 1.0)),
        "--top_k",
        str(getattr(args, "downstream_top_k", 0)),
        "--response_log_max",
        str(getattr(args, "downstream_response_log_max", 0)),
        "--tensor_parallel_size",
        str(tensor_parallel_size),
        "--gpu_memory_utilization",
        str(gpu_memory_utilization),
        "--dtype",
        dtype,
    ]
    response_key = getattr(args, "downstream_response_key", "")
    if response_key:
        cmd.extend(["--response_key", response_key])
    reward_score_dir = getattr(args, "downstream_reward_score_dir", "")
    if reward_score_dir:
        cmd.extend(["--reward_score_dir", reward_score_dir])
    if getattr(args, "downstream_apply_chat_template", False):
        cmd.append("--apply_chat_template")
    if getattr(args, "downstream_shuffle", False):
        cmd.append("--shuffle")

    print(f"running local downstream vLLM eval: dataset={dataset_path}")
    print(f"local downstream output={output_path}")
    subprocess.run(cmd, check=True, env=env)
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_downstream_lm_eval(args, model_path, compare_dir, compare_tag, target_ratio, score_order):
    output_dir = getattr(args, "downstream_output_dir", "") or os.path.join(compare_dir, "lm_eval_results")
    os.makedirs(output_dir, exist_ok=True)
    ratio_tag = f"{target_ratio:.4f}".rstrip("0").rstrip(".").replace(".", "p") or "0"
    result_prefix = f"{compare_tag}_{score_order}_sparsity_{ratio_tag}"
    output_path = os.path.join(output_dir, result_prefix)
    summary_csv = getattr(args, "downstream_summary_csv", "") or os.path.join("eval_results", "summary.csv")
    device = "cuda:0"
    batch_size = getattr(args, "downstream_batch_size", "auto")
    hf_batch_size = getattr(args, "downstream_hf_batch_size", "auto")
    hf_max_batch_size = int(getattr(args, "downstream_hf_max_batch_size", 8))
    hf_gpu_memory_utilization = float(getattr(args, "downstream_hf_gpu_memory_utilization", 0.6))
    final_tasks = getattr(args, "downstream_tasks", None) or DEFAULT_FINAL_TASKS
    benchmarks = parse_downstream_suite(args)
    eval_tasks = MMLU_CATEGORY_TASKS if final_tasks == ["mmlu"] else final_tasks
    num_fewshot = getattr(args, "downstream_num_fewshot", 5)
    apply_chat_template = bool(getattr(args, "downstream_apply_chat_template", False))
    fewshot_as_multiturn = bool(getattr(args, "downstream_fewshot_as_multiturn", False))
    chat_template_args = getattr(args, "downstream_chat_template_args", "") or None
    gen_kwargs = getattr(args, "downstream_gen_kwargs", "") or None
    limit = getattr(args, "downstream_limit", "") or None
    cache_requests = getattr(args, "downstream_cache_requests", "true") or None
    request_cache_path = getattr(args, "downstream_request_cache_path", "") or None
    include_path = getattr(args, "downstream_include_path", "") or None
    log_samples = bool(getattr(args, "downstream_log_samples", False))
    log_samples_limit = int(getattr(args, "downstream_log_samples_limit", 0))
    dtype = getattr(args, "downstream_dtype", "bfloat16")
    default_backend = getattr(args, "downstream_lm_eval_backend", "vllm")
    tensor_parallel_size = max(1, int(getattr(args, "downstream_tensor_parallel_size", 1)))
    data_parallel_size = max(1, int(getattr(args, "downstream_data_parallel_size", 1)))
    gpu_memory_utilization = min(float(getattr(args, "downstream_gpu_memory_utilization", 0.6)), 0.85)
    max_model_len = int(getattr(args, "downstream_max_model_len", 2048))
    max_num_batched_tokens = int(getattr(args, "downstream_max_num_batched_tokens", 8192))
    max_num_seqs = int(getattr(args, "downstream_max_num_seqs", 64))
    vllm_python = (
        getattr(args, "downstream_vllm_python", "")
        or os.environ.get("VLLM_PYTHON", "")
        or "/home/tans5/anaconda3/envs/vllm/bin/python"
    )
    if not os.path.exists(vllm_python):
        vllm_python = "python"
    env = os.environ.copy()
    if "DOWNSTREAM_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DOWNSTREAM_CUDA_VISIBLE_DEVICES"]
    else:
        gpu_ids = best_free_gpu_ids(tensor_parallel_size * data_parallel_size)
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    def score_metadata(benchmark, backend, shot_count, task_limit, benchmark_batch_size, cache_requests_value):
        return {
            "prune_method": args.prune_method,
            "method_tag": compare_tag,
            "prune_scope": getattr(args, "curvature_prune_scope", "global"),
            "score_order": score_order,
            "prune_ops": " ".join(getattr(args, "prune_ops", None) or []),
            "benchmark": benchmark,
            "lm_eval_backend": backend,
            "dtype": dtype,
            "num_fewshot": shot_count,
            "apply_chat_template": apply_chat_template,
            "fewshot_as_multiturn": fewshot_as_multiturn,
            "gen_kwargs": gen_kwargs or "",
            "limit": task_limit or "",
            "batch_size": benchmark_batch_size,
            "cache_requests": cache_requests_value or "",
        }

    def run_one_benchmark(benchmark, tasks, backend, shot_count, task_limit, include_tasks=None):
        benchmark_output_path = output_path if benchmark == "single" else os.path.join(output_path, benchmark)
        local_task = resolve_local_task(benchmark, tasks)
        if local_task is not None:
            backend = "local_vllm"
        lm_eval_python = vllm_python if backend == "vllm" else sys.executable
        model_args_extra = None
        benchmark_batch_size = batch_size
        if backend == "vllm":
            model_args = [
                f"tensor_parallel_size={tensor_parallel_size}",
                f"data_parallel_size={data_parallel_size}",
                f"gpu_memory_utilization={gpu_memory_utilization}",
                f"max_model_len={max_model_len}",
                f"max_num_batched_tokens={max_num_batched_tokens}",
                f"max_num_seqs={max_num_seqs}",
            ]
            if chat_template_args:
                model_args.append(chat_template_args)
            model_args_extra = ",".join(model_args)
        elif backend == "hf":
            benchmark_batch_size = hf_batch_size
            model_args = [f"max_batch_size={hf_max_batch_size}"]
            hf_memory_cap = hf_max_memory_per_gpu(
                hf_gpu_memory_utilization,
                cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES", ""),
            )
            if hf_memory_cap is not None:
                model_args.extend(["parallelize=True", f"max_memory_per_gpu={hf_memory_cap}"])
            if chat_template_args:
                model_args.append(chat_template_args)
            model_args_extra = ",".join(model_args)

        if backend == "local_vllm":
            metrics = run_local_vllm_downstream_eval(
                args,
                model_path,
                benchmark_output_path,
                vllm_python,
                tensor_parallel_size,
                gpu_memory_utilization,
                dtype,
                env,
                max_examples=int(float(task_limit)) if task_limit else None,
                dataset_path=local_task["dataset_path"] if local_task else None,
                prompt_key=local_task["prompt_key"] if local_task else None,
            )
            score = metrics.get("accuracy", metrics.get("pass@1", metrics.get("mean_score")))
            append_scores_to_csv(
                [{"task": tasks[0] if tasks else benchmark, "metric": "accuracy", "score": score}],
                args.model,
                target_ratio,
                summary_csv,
                metadata=score_metadata(
                    benchmark,
                    backend,
                    shot_count,
                    task_limit,
                    getattr(args, "downstream_local_batch_size", 1),
                    "",
                ),
            )
            return

        print(f"running downstream lm-eval {backend} final: benchmark={benchmark} tasks={','.join(tasks)}")
        print(f"downstream lm-eval python={lm_eval_python}")
        print(f"downstream lm-eval CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}")
        print(f"downstream lm-eval gpu_memory_utilization={gpu_memory_utilization}")
        print(
            "downstream lm-eval vLLM limits: "
            f"batch_size={batch_size}, max_model_len={max_model_len}, "
            f"max_num_batched_tokens={max_num_batched_tokens}, max_num_seqs={max_num_seqs}, "
            f"tensor_parallel_size={tensor_parallel_size}, data_parallel_size={data_parallel_size}"
        )
        print(
            "downstream lm-eval HF limits: "
            f"batch_size={hf_batch_size}, max_batch_size={hf_max_batch_size}, "
            f"gpu_memory_utilization={hf_gpu_memory_utilization}"
        )
        print(
            "downstream lm-eval shared args: "
            f"dtype={dtype}, num_fewshot={shot_count}, "
            f"apply_chat_template={apply_chat_template}, fewshot_as_multiturn={fewshot_as_multiturn}, "
            f"chat_template_args={chat_template_args}, gen_kwargs={gen_kwargs}, "
            f"limit={task_limit}, batch_size={benchmark_batch_size}, "
            f"cache_requests={cache_requests}, request_cache_path={request_cache_path}"
        )
        run_lm_eval(
            model_path,
            tasks,
            benchmark_output_path,
            device=device,
            batch_size=benchmark_batch_size,
            limit=task_limit,
            backend=backend,
            dtype=dtype,
            num_fewshot=shot_count,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            gen_kwargs=gen_kwargs,
            model_args_extra=model_args_extra,
            cache_requests=cache_requests,
            request_cache_path=request_cache_path,
            include_path=include_path,
            log_samples=log_samples,
            log_samples_limit=log_samples_limit,
            python_bin=lm_eval_python,
            env=env,
        )
        result_json = find_latest_result_json(benchmark_output_path)
        scores = extract_main_scores(
            result_json,
            include_tasks=include_tasks,
            include_sample_len=include_tasks is not None,
        )
        append_scores_to_csv(
            scores,
            args.model,
            target_ratio,
            summary_csv,
            metadata=score_metadata(
                benchmark,
                backend,
                shot_count,
                task_limit,
                benchmark_batch_size,
                cache_requests,
            ),
        )

    if benchmarks:
        for benchmark in benchmarks:
            include_tasks = MMLU_SUMMARY_TASKS if benchmark["benchmark"] == "mmlu" else None
            run_one_benchmark(
                benchmark["benchmark"],
                benchmark["tasks"],
                benchmark["backend"],
                benchmark["num_fewshot"],
                benchmark.get("limit"),
                include_tasks=include_tasks,
            )
    else:
        include_tasks = MMLU_SUMMARY_TASKS if final_tasks == ["mmlu"] else None
        run_one_benchmark(
            "single",
            eval_tasks,
            default_backend,
            num_fewshot,
            limit,
            include_tasks=include_tasks,
        )
    print(f"downstream lm-eval summary saved to {summary_csv}")
    return output_path


def cleanup_downstream_model(model_path):
    if not model_path or not os.path.exists(model_path):
        return
    if os.path.isdir(model_path):
        shutil.rmtree(model_path)
    else:
        os.remove(model_path)
    print(f"cleaned downstream model checkpoint: {model_path}")


def resolve_prune_score_orders(args):
    if args.prune_method in {"wanda", "magnitude"}:
        return ["low_to_high"]
    return list(dict.fromkeys(args.prune_score_order))


def per_layer_result_tag(args):
    scope = getattr(args, "curvature_prune_scope", "global")
    scope_tag = "local" if scope == "per_layer" else "per_op" if scope == "per_layer_op" else scope
    prune_ops = getattr(args, "prune_ops", None)
    op_tag = ""
    if prune_ops:
        op_tag = "_" + "-".join(str(op).removesuffix("_proj") for op in prune_ops)
    if args.prune_method != "curvature":
        return f"{args.prune_method}{op_tag}_{scope_tag}"

    setting_tag = curvature_seq_tag(
        args.shared_top_k,
        args.shared_seq_select,
        args.curvature_lpf_window,
    ).removesuffix("_pkl")
    return f"{args.prune_method}{op_tag}_{l2_path_tag(args)}_{setting_tag}_{scope_tag}"


def pp_result_tag(args):
    return per_layer_result_tag(args)


def reference_layer_indices(args, get_llm_fn, model_device, base_wanda_scores):
    def selected(layer_ids):
        requested = getattr(args, "per_layer_ids", None)
        if requested is None:
            return layer_ids
        requested_set = set(int(layer_idx) for layer_idx in requested)
        return [layer_idx for layer_idx in layer_ids if int(layer_idx) in requested_set]

    if args.prune_method == "curvature":
        layer_ids = list_curvature_pkl_layers(
            args.load_curvature_dir or args.save_curvature_dir,
            args.shared_top_k,
            args.shared_seq_select,
            args.curvature_lpf_window,
        )
        layer_ids = selected(layer_ids)
        if not layer_ids:
            raise ValueError("No curvature layer PKLs found for per-layer evaluation")
        return layer_ids

    if args.load_curvature_dir is not None:
        layer_ids = list_curvature_pkl_layers(
            args.load_curvature_dir,
            args.shared_top_k,
            args.shared_seq_select,
            args.curvature_lpf_window,
        )
        layer_ids = selected(layer_ids)
        if layer_ids:
            return layer_ids

    if args.prune_method == "wanda" and base_wanda_scores is not None:
        return selected(list(range(len(base_wanda_scores))))

    ref_model = get_llm_fn(args.model, args.cache_dir, model_device, args.seqlen)
    try:
        return selected(list(range(len(ref_model.model.layers))))
    finally:
        del ref_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_per_layer_eval(
    args,
    get_llm_fn,
    tokenizer,
    model_device,
    sparsity_ratios,
    prune_score_orders,
    eval_seq_lens,
    prune_n,
    prune_m,
    save_filepath,
    base_wanda_scores=None,
):
    if getattr(args, "curvature_prune_scope", "global") == "global":
        raise ValueError(
            "run_per_layer_eval supports only local/per-layer or per-op pruning scopes; "
            "global scope is only valid for all-layer pruning."
        )

    layer_ids = reference_layer_indices(args, get_llm_fn, model_device, base_wanda_scores)
    result_dir = os.path.dirname(save_filepath)
    result_tag = per_layer_result_tag(args)
    compare_dir = args.per_layer_compare_dir or result_dir
    plot_dir = os.path.join(compare_dir, "per_layer_plots", result_tag)
    compare_plot_dir = os.path.join(compare_dir, "method_compare_plots")
    per_layer_log_dir = os.path.join(compare_dir, "per_layer_logs", result_tag)
    os.makedirs(per_layer_log_dir, exist_ok=True)
    all_records = []
    if args.prune_method == "curvature":
        prune_score_orders = ["high_to_low"]

    for score_order in prune_score_orders:
        args.prune_score_order = score_order
        for layer_idx in layer_ids:
            layer_log_prefix = os.path.join(per_layer_log_dir, f"layer_{layer_idx:03d}")
            edge_log_path = f"{layer_log_prefix}_pruned_parameters.txt"
            pp_log_path = f"{layer_log_prefix}_pp_eval.txt"
            layer_records = []
            layer_curvature_scores = None
            if args.load_curvature_dir is not None:
                layer_curvature_scores = load_curvature_scores_for_layer(
                    args.load_curvature_dir,
                    layer_idx,
                    args.shared_top_k,
                    args.shared_seq_select,
                    args.curvature_lpf_window,
                    prune_ops=getattr(args, "prune_ops", None),
                )
            elif args.prune_method == "curvature":
                layer_curvature_scores = load_curvature_scores_for_layer(
                    args.save_curvature_dir,
                    layer_idx,
                    args.shared_top_k,
                    args.shared_seq_select,
                    args.curvature_lpf_window,
                    prune_ops=getattr(args, "prune_ops", None),
                )

            nonzero_sparsity_idx = 0
            for target_ratio in sparsity_ratios:
                print(
                    f"per-layer eval: layer={layer_idx} sparsity={target_ratio:.4f} "
                    f"score_order={score_order}"
                )
                current_model = get_llm_fn(args.model, args.cache_dir, model_device, args.seqlen)
                current_model.eval()
                current_model.seqlen = args.seqlen
                args.sparsity_ratio = target_ratio

                score_cutoff = None
                report_rank_offset = None
                if target_ratio != 0:
                    report_rank_offset = nonzero_sparsity_idx * 25
                    nonzero_sparsity_idx += 1
                    if args.prune_method == "curvature":
                        if layer_curvature_scores is None:
                            raise ValueError(
                                f"Missing curvature PKL for layer {layer_idx} in per-layer eval"
                            )
                        _, score_cutoff = prune_curvature_layer(
                            args,
                            current_model,
                            layer_idx,
                            layer_curvature_scores,
                            edge_log_path=edge_log_path,
                            report_rank_offset=report_rank_offset,
                        )
                    elif args.prune_method == "wanda":
                        _, score_cutoff = prune_wanda_layer(
                            args,
                            current_model,
                            layer_idx,
                            base_wanda_scores[layer_idx],
                            layer_curvature_scores=layer_curvature_scores,
                            prune_n=prune_n,
                            prune_m=prune_m,
                            edge_log_path=edge_log_path,
                            report_rank_offset=report_rank_offset,
                        )
                    elif args.prune_method == "magnitude":
                        _, score_cutoff = prune_magnitude_layer(
                            args,
                            current_model,
                            layer_idx,
                            layer_curvature_scores=layer_curvature_scores,
                            prune_n=prune_n,
                            prune_m=prune_m,
                            edge_log_path=edge_log_path,
                            report_rank_offset=report_rank_offset,
                        )

                model_actual_sparsity = check_sparsity(current_model)
                layer_actual = layer_sparsity(current_model, layer_idx)

                for seq in eval_seq_lens:
                    current_model.seqlen = seq
                    ppl_test = eval_ppl(args, current_model, tokenizer, model_device)
                    append_per_layer_eval_result(
                        pp_log_path,
                        args,
                        layer_idx,
                        score_order,
                        target_ratio,
                        layer_actual,
                        model_actual_sparsity,
                        seq,
                        ppl_test,
                        score_cutoff,
                    )
                    record = {
                        "method": args.prune_method,
                        "method_tag": result_tag,
                        "layer_idx": int(layer_idx),
                        "score_order": score_order,
                        "target_sparsity": float(target_ratio),
                        "layer_actual_sparsity": float(layer_actual),
                        "model_actual_sparsity": float(model_actual_sparsity),
                        "pp_seq_len": int(seq),
                        "ppl_test": float(ppl_test),
                        "score_cutoff": None if score_cutoff is None else float(score_cutoff),
                        "cutoff_nonpositive": (
                            score_cutoff is not None and float(score_cutoff) <= 0.0
                        ),
                    }
                    layer_records.append(record)
                    all_records.append(record)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                with open(pp_log_path, "a+", encoding="utf-8") as f:
                    print("", file=f, flush=True)

                del current_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            plot_paths = draw_per_layer_ppl_vs_sparsity(
                layer_records,
                plot_dir,
                annotate_cutoff=(args.prune_method == "curvature"),
            )
            if plot_paths:
                print(f"Saved layer {layer_idx} plot to {plot_paths[-1]}")

            with open(pp_log_path, "a+", encoding="utf-8") as f:
                print("", file=f, flush=True)

    if all_records:
        layer_tag = "all_layers"
        requested_layers = getattr(args, "per_layer_ids", None)
        if requested_layers is not None and len(requested_layers) == 1:
            layer_tag = f"layer_{int(requested_layers[0]):03d}"
        csv_path = os.path.join(compare_dir, f"per_layer_records_{result_tag}_{layer_tag}.csv")
        saved_csv = save_per_layer_records_csv(all_records, csv_path)
        if saved_csv is not None:
            print(f"Saved per-layer records: {saved_csv}")
        compare_paths = draw_per_layer_method_comparison(
            compare_dir,
            compare_plot_dir,
            eval_seq_lens=eval_seq_lens,
        )
        if compare_paths:
            print(f"Saved method comparison plot: {compare_paths[-1]}")


def run_pp_eval(
    args,
    get_llm_fn,
    tokenizer,
    model_device,
    sparsity_ratios,
    prune_score_orders,
    eval_seq_lens,
    prune_n,
    prune_m,
    save_filepath,
    base_curvature_scores=None,
    base_wanda_scores=None,
    base_wanda_input_scalers=None,
):
    result_dir = os.path.dirname(save_filepath)
    compare_dir = args.per_layer_compare_dir or result_dir
    compare_tag = pp_result_tag(args)
    pp_log_path = os.path.join(compare_dir, f"all_layer_pp_eval_{compare_tag}.txt")
    parameter_log_path = os.path.join(compare_dir, f"all_layer_pruned_parameters_{compare_tag}.txt")
    os.makedirs(compare_dir, exist_ok=True)

    with open(pp_log_path, "a+") as f:
        print(
            f"{'method':<15}{'prune_scope':<15}{'score_order':<15}{'l2_norm':<10}{'target_sparsity':<18}"
            f"{'actual_sparsity':<18}{'calib_data':<20}{'eval_mode':<28}"
            f"{'score_seq_len':<16}{'pp_seq_len':<12}{'ppl_test':<12}",
            file=f,
            flush=True,
        )

    eval_records = []
    for score_order in prune_score_orders:
        args.prune_score_order = score_order
        nonzero_sparsity_idx = 0
        for run_idx, target_ratio in enumerate(sparsity_ratios):
            print(
                f"starting sweep run {run_idx + 1}/{len(sparsity_ratios)} "
                f"with sparsity={target_ratio:.4f}, score_order={score_order}"
            )
            append_eval_run_header(pp_log_path, args, target_ratio, score_order)
            current_model = get_llm_fn(args.model, args.cache_dir, model_device, args.seqlen)
            current_model.eval()
            current_model.seqlen = args.seqlen
            args.sparsity_ratio = target_ratio

            if target_ratio != 0:
                print("pruning starts")
                args.all_layer_parameter_log_path = parameter_log_path
                args.all_layer_report_rank_offset = nonzero_sparsity_idx * 25
                nonzero_sparsity_idx += 1
                if args.prune_method == "curvature":
                    current_model.curvature_scores = base_curvature_scores
                    if args.curvature_prune_scope == "global":
                        prune_summary = prune_global_curvature(args, current_model)
                    else:
                        prune_summary = prune_scoped_curvature(args, current_model)
                    append_curvature_prune_summary(
                        pp_log_path,
                        prune_summary,
                        target_ratio=target_ratio,
                        score_order=score_order,
                    )
                elif args.prune_method == "wanda":
                    if base_wanda_input_scalers is not None:
                        current_model.wanda_input_scalers = base_wanda_input_scalers
                    else:
                        current_model.wanda_scores = base_wanda_scores
                    prune_wanda(args, current_model, tokenizer, model_device, prune_n, prune_m)
                elif args.prune_method == "magnitude":
                    prune_magnitude(args, current_model, tokenizer, model_device, prune_n, prune_m)

            print("*" * 30)
            actual_sparsity_ratio = check_sparsity(current_model)
            print(f"sparsity sanity check {actual_sparsity_ratio:.4f}")
            print("*" * 30)

            if not getattr(args, "downstream_only", False):
                for seq in eval_seq_lens:
                    current_model.seqlen = seq
                    ppl_test = eval_ppl(args, current_model, tokenizer, model_device)
                    eval_mode = "fixed_score_seq_len" if args.prune_method == "wanda" else "standard_eval"
                    print(f"wikitext perplexity {ppl_test} using pp_seqlen = {seq}")
                    append_eval_result(
                        pp_log_path,
                        args,
                        score_order,
                        target_ratio,
                        actual_sparsity_ratio,
                        eval_mode,
                        args.seqlen,
                        seq,
                        ppl_test,
                    )
                    eval_records.append(
                        {
                            "method": args.prune_method,
                            "method_tag": per_layer_result_tag(args),
                            "prune_scope": args.curvature_prune_scope,
                            "score_order": score_order,
                            "target_sparsity": float(target_ratio),
                            "actual_sparsity": float(actual_sparsity_ratio),
                            "pp_seq_len": int(seq),
                            "ppl_test": float(ppl_test),
                        }
                    )
                    release_cuda_memory()

            model_save_path = None
            if args.save_model or getattr(args, "run_downstream_eval", False):
                if getattr(args, "run_downstream_eval", False):
                    model_save_path = downstream_model_path(
                        args,
                        compare_dir,
                        compare_tag,
                        target_ratio,
                        len(sparsity_ratios),
                        score_order,
                        len(prune_score_orders),
                    )
                else:
                    model_save_path = save_model_path(args.save_model, target_ratio, len(sparsity_ratios))
                    if len(prune_score_orders) > 1:
                        model_save_path = os.path.join(model_save_path, score_order)
                if getattr(args, "run_downstream_eval", False):
                    save_downstream_checkpoint(args, current_model, tokenizer, model_save_path)
                else:
                    save_pruned_model(current_model, tokenizer, model_save_path)

            del current_model
            release_cuda_memory()

            if getattr(args, "run_downstream_eval", False):
                run_downstream_lm_eval(
                    args,
                    model_save_path,
                    compare_dir,
                    compare_tag,
                    target_ratio,
                    score_order,
                )
                if not getattr(args, "keep_downstream_model", False):
                    cleanup_downstream_model(model_save_path)
                release_cuda_memory()

            with open(pp_log_path, "a+", encoding="utf-8") as f:
                print("", file=f, flush=True)

    if eval_records:
        result_tag = pp_result_tag(args)
        csv_path = os.path.join(result_dir, f"ppl_vs_sparsity_{result_tag}.csv")
        plot_path = os.path.join(result_dir, f"ppl_vs_sparsity_{result_tag}.png")
        save_eval_records_csv(eval_records, csv_path)
        drawn_path = draw_ppl_vs_sparsity(eval_records, plot_path)
        if drawn_path is not None:
            print(f"Saved PPL vs sparsity plot: {drawn_path}")

        compare_plot_dir = os.path.join(compare_dir, "all_layer_method_compare_plots")
        compare_csv = os.path.join(compare_dir, f"pp_records_{compare_tag}.csv")
        saved_compare_csv = save_eval_records_csv(eval_records, compare_csv)
        if saved_compare_csv is not None:
            print(f"Saved all-layer comparison records: {saved_compare_csv}")
        compare_plot = draw_method_comparison(
            compare_dir,
            compare_plot_dir,
            eval_seq_lens=eval_seq_lens,
        )
        if compare_plot is not None:
            if isinstance(compare_plot, list):
                for plot_path in compare_plot:
                    print(f"Saved all-layer method comparison plot: {plot_path}")
            else:
                print(f"Saved all-layer method comparison plot: {compare_plot}")
