import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


DEFAULT_FINAL_TASKS = [
    "hellaswag",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "boolq",
]

MMLU_CATEGORY_TASKS = [
    "mmlu_stem",
    "mmlu_social_sciences",
]
MMLU_SUMMARY_TASKS = set(MMLU_CATEGORY_TASKS)

def save_pruned_model(model, tokenizer, model_path):
    model_path = Path(model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return str(model_path)


def check_saved_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code=True,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _lm_eval_command(python_bin=None):
    if python_bin:
        return [python_bin, "-m", "lm_eval"]
    lm_eval_bin = shutil.which("lm-eval")
    if lm_eval_bin:
        return [lm_eval_bin]
    return [sys.executable, "-m", "lm_eval"]


def _lm_eval_cache_root(request_cache_path=None):
    if request_cache_path:
        return Path(request_cache_path).parent / "hf_eval_cache"
    return Path("eval_results") / "hf_eval_cache"


def _hash_tokenizer_files(model_path):
    digest = hashlib.sha1()
    found = False
    for name in (
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
    ):
        path = Path(model_path) / name
        if not path.is_file():
            continue
        found = True
        digest.update(name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:12] if found else None


def _short_tokenizer_path(model_path, request_cache_path=None):
    tokenizer_hash = _hash_tokenizer_files(model_path)
    if tokenizer_hash is None:
        return None

    root = Path(request_cache_path).parent if request_cache_path else Path("eval_results")
    short_root = root / "lm_eval_tokenizers"
    short_root.mkdir(parents=True, exist_ok=True)
    short_path = short_root / f"tok_{tokenizer_hash}"
    target_path = Path(model_path).resolve()

    if short_path.is_symlink():
        if short_path.resolve() == target_path:
            return str(short_path)
        short_path.unlink()
    if short_path.exists():
        return str(short_path)

    try:
        short_path.symlink_to(target_path, target_is_directory=True)
        return str(short_path)
    except OSError:
        return str(model_path)


def run_lm_eval(
    model_path,
    tasks,
    output_path,
    device="cuda:0",
    batch_size="auto",
    limit=None,
    backend="hf",
    dtype="bfloat16",
    num_fewshot=None,
    apply_chat_template=False,
    fewshot_as_multiturn=False,
    gen_kwargs=None,
    model_args_extra=None,
    cache_requests=None,
    request_cache_path=None,
    include_path=None,
    log_samples=False,
    log_samples_limit=0,
    python_bin=None,
    env=None,
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    task_arg = ",".join(tasks)
    tokenizer_path = _short_tokenizer_path(model_path, request_cache_path)
    model_args_parts = [
        f"pretrained={model_path}",
        f"dtype={dtype}",
        "trust_remote_code=True",
    ]
    if tokenizer_path:
        model_args_parts.append(f"tokenizer={tokenizer_path}")
    model_args = ",".join(model_args_parts)
    if model_args_extra:
        model_args = f"{model_args},{model_args_extra}"
    cmd = _lm_eval_command(python_bin) + [
        "run",
        "--model",
        backend,
        "--model_args",
        model_args,
        "--tasks",
        task_arg,
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--output_path",
        str(output_path),
    ]
    if log_samples:
        cmd.append("--log_samples")
    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])
    if apply_chat_template:
        cmd.append("--apply_chat_template")
    cmd.extend(["--fewshot_as_multiturn", "true" if fewshot_as_multiturn else "false"])
    if gen_kwargs:
        cmd.extend(["--gen_kwargs", str(gen_kwargs)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if cache_requests:
        cmd.extend(["--cache_requests", str(cache_requests)])
    env = dict(os.environ if env is None else env)
    cache_root = _lm_eval_cache_root(request_cache_path)
    metrics_cache = cache_root / "metrics"
    evaluate_cache = cache_root / "evaluate"
    metrics_cache.mkdir(parents=True, exist_ok=True)
    evaluate_cache.mkdir(parents=True, exist_ok=True)
    env.setdefault("HF_METRICS_CACHE", str(metrics_cache))
    env.setdefault("HF_EVALUATE_CACHE", str(evaluate_cache))
    if request_cache_path:
        Path(request_cache_path).mkdir(parents=True, exist_ok=True)
        env["LM_HARNESS_CACHE_PATH"] = str(request_cache_path)
    if include_path:
        cmd.extend(["--include_path", str(include_path)])
    if any(task.startswith("humaneval") for task in tasks):
        cmd.append("--confirm_run_unsafe_code")
        env["HF_ALLOW_CODE_EVAL"] = "1"
    subprocess.run(cmd, check=True, env=env)
    trim_sample_logs(output_path, log_samples_limit)


def trim_sample_logs(output_path, max_samples):
    max_samples = int(max_samples or 0)
    if max_samples <= 0:
        return

    for sample_path in Path(output_path).rglob("samples*.jsonl"):
        tmp_path = sample_path.with_suffix(sample_path.suffix + ".tmp")
        kept = 0
        with sample_path.open("r", encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
            for line in src:
                if kept >= max_samples:
                    break
                dst.write(line)
                kept += 1
        os.replace(tmp_path, sample_path)
        print(f"kept {kept} logged samples in {sample_path}")


def find_latest_result_json(output_path):
    result_paths = list(Path(output_path).rglob("results*.json"))
    if not result_paths:
        raise FileNotFoundError(f"No results*.json found under {output_path}")
    return str(max(result_paths, key=lambda path: path.stat().st_mtime))


def _metric_value(metrics, metric_name):
    if metric_name in metrics:
        return metric_name, metrics[metric_name]
    for key, value in metrics.items():
        if key.split(",", 1)[0] == metric_name:
            return key, value
    return None, None


def extract_main_scores(result_json, include_tasks=None, include_sample_len=False):
    include_tasks = set(include_tasks) if include_tasks else None
    with open(result_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for task, metrics in data.get("results", {}).items():
        if include_tasks is not None and task not in include_tasks:
            continue
        if task == "ifeval":
            prompt_metric, prompt_score = _metric_value(metrics, "prompt_level_strict_acc")
            inst_metric, inst_score = _metric_value(metrics, "inst_level_strict_acc")
            if prompt_metric is not None and inst_metric is not None:
                rows.append(
                    {
                        "task": task,
                        "metric": "strict_acc_avg",
                        "score": (float(prompt_score) + float(inst_score)) / 2.0,
                    }
                )
                continue
        for metric_name in ("acc_norm", "acc", "exact_match", "exact", "f1", "contains", "pass@1"):
            metric, score = _metric_value(metrics, metric_name)
            if metric is not None:
                row = {"task": task, "metric": metric, "score": score}
                if include_sample_len:
                    row["sample_len"] = metrics.get("sample_len")
                rows.append(row)
                break
    return rows


def append_scores_to_csv(scores, model, sparsity, csv_path="eval_results/summary.csv", metadata=None):
    metadata = metadata or {}
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    metadata_fields = list(metadata.keys())
    fieldnames = ["model", "sparsity", *metadata_fields, "task", "metric", "score"]
    if not write_header and csv_path.stat().st_size > 0:
        with open(csv_path, newline="", encoding="utf-8") as existing_f:
            reader = csv.DictReader(row for row in existing_f if row.strip())
            existing_rows = list(reader)
            existing_header = reader.fieldnames or []
        if existing_header != fieldnames:
            merged_fields = list(existing_header)
            for name in fieldnames:
                if name not in merged_fields:
                    merged_fields.append(name)
            with open(csv_path, "w", newline="", encoding="utf-8") as upgraded_f:
                upgraded_writer = csv.DictWriter(upgraded_f, fieldnames=merged_fields)
                upgraded_writer.writeheader()
                for row in existing_rows:
                    upgraded_writer.writerow({name: row.get(name, "") for name in merged_fields})
            fieldnames = merged_fields
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for score in scores:
            row = {
                "model": model,
                "sparsity": sparsity,
                "task": score["task"],
                "metric": score["metric"],
                "score": score["score"],
            }
            row.update(metadata)
            writer.writerow(row)


def run_pruned_lm_eval_pipeline(
    model,
    tokenizer,
    model_path,
    output_path,
    model_name,
    sparsity,
    final_tasks=None,
    device="cuda:0",
    summary_csv="eval_results/summary.csv",
):
    final_tasks = final_tasks or DEFAULT_FINAL_TASKS
    save_pruned_model(model, tokenizer, model_path)
    check_saved_model(model_path)

    run_lm_eval(model_path, final_tasks, output_path, device=device)
    result_json = find_latest_result_json(output_path)
    scores = extract_main_scores(result_json)
    append_scores_to_csv(scores, model_name, sparsity, summary_csv)
    return scores
