from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tqdm.auto import tqdm


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    config: str | None
    split: str
    prompt_builder: Callable[[dict[str, Any]], str]
    answer_builder: Callable[[dict[str, Any]], Any]
    scorer: Callable[[str, Any], bool]


def _choice_prompt(question: Any, choices: Any) -> str:
    labels = ["A", "B", "C", "D", "E", "F"]
    choice_lines = []
    for idx, choice in enumerate(list(choices)):
        label = labels[idx] if idx < len(labels) else str(idx + 1)
        choice_lines.append(f"{label}. {choice}")
    return (
        f"Question: {question}\n"
        + "\n".join(choice_lines)
        + "\nReason step by step, then end with 'Final answer: <letter>'."
    )


def _build_gsm8k_prompt(row: dict[str, Any]) -> str:
    return f"Question: {row['question']}\nReason step by step, then end with 'Final answer: <answer>'."


def _build_mmlu_prompt(row: dict[str, Any]) -> str:
    return _choice_prompt(row["question"], row["choices"])


def _build_winogrande_prompt(row: dict[str, Any]) -> str:
    return (
        "Fill in the blank with the correct option.\n"
        f"Sentence: {row['sentence']}\n"
        f"A. {row['option1']}\n"
        f"B. {row['option2']}\n"
        "Reason step by step, then end with 'Final answer: <letter>'."
    )


def _build_truthfulqa_prompt(row: dict[str, Any]) -> str:
    targets = row.get("mc1_targets", {})
    choices = targets.get("choices", []) if isinstance(targets, dict) else []
    return _choice_prompt(row["question"], choices)


def _build_math_prompt(row: dict[str, Any]) -> str:
    problem = row.get("problem", row.get("question", row.get("prompt", "")))
    return f"Problem: {problem}\nReason step by step, then end with 'Final answer: <answer>'."


def _gsm8k_answer(row: dict[str, Any]) -> str:
    answer = str(row.get("answer", ""))
    if "####" in answer:
        return answer.split("####")[-1].strip()
    return answer.strip().splitlines()[-1] if answer.strip() else ""


def _mmlu_answer(row: dict[str, Any]) -> Any:
    labels = ["A", "B", "C", "D", "E", "F"]
    answer = row.get("answer")
    if isinstance(answer, int):
        choices = list(row.get("choices", []))
        return {"label": labels[answer], "text": choices[answer] if answer < len(choices) else ""}
    return str(answer).strip()


def _winogrande_answer(row: dict[str, Any]) -> dict[str, str]:
    if str(row.get("answer", "")).strip() == "1":
        return {"label": "A", "text": str(row.get("option1", ""))}
    return {"label": "B", "text": str(row.get("option2", ""))}


def _truthfulqa_answer(row: dict[str, Any]) -> Any:
    targets = row.get("mc1_targets", {})
    labels = targets.get("labels", []) if isinstance(targets, dict) else []
    choices = targets.get("choices", []) if isinstance(targets, dict) else []
    for idx, label in enumerate(labels):
        if int(label) == 1:
            label_text = ["A", "B", "C", "D", "E", "F"][idx] if idx < 6 else str(idx + 1)
            return {"label": label_text, "text": str(choices[idx]) if idx < len(choices) else ""}
    return str(choices[0]) if choices else ""


def _math_answer(row: dict[str, Any]) -> str:
    for key in ("answer", "final_answer", "solution"):
        value = row.get(key)
        if value:
            return _extract_math_answer(value)
    return ""


def _extract_choice(text: str) -> str:
    final_matches = re.findall(
        r"final\s+answer\s*[:\-]?\s*(?:\(?\s*)?([A-F])(?:\s*\)?)",
        text,
        flags=re.IGNORECASE,
    )
    if final_matches:
        return final_matches[-1].upper()

    matches = re.findall(r"(?:^|[^A-Za-z])([A-F])(?:[^A-Za-z]|$)", text.strip(), flags=re.IGNORECASE)
    return matches[-1].upper() if matches else ""


def _extract_final_answer_text(text: Any) -> str:
    text = str(text)
    matches = re.findall(r"final\s+answer\s*[:\-]?\s*(.+)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return text.strip().splitlines()[-1] if text.strip() else ""


def _normalize_number(text: Any) -> str:
    text = _extract_final_answer_text(text)
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else text.strip().lower()


def _extract_math_answer(text: Any) -> str:
    text = str(text)
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return matches[-1]
    return text.strip().splitlines()[-1] if text.strip() else ""


def _normalize_math(text: Any) -> str:
    text = _extract_final_answer_text(text)
    text = _extract_math_answer(text).strip().strip("$").strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", text)
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _score_number(response: str, target: Any) -> bool:
    return _normalize_number(response) == _normalize_number(target)


def _score_choice(response: str, target: Any) -> bool:
    final_answer = _extract_final_answer_text(response).lower()
    if isinstance(target, dict):
        label = str(target.get("label", "")).strip().upper()
        text = str(target.get("text", "")).strip().lower()
        return _extract_choice(response) == label or (bool(text) and final_answer.startswith(text))
    return _extract_choice(response) == str(target).strip().upper()


def _score_math(response: str, target: Any) -> bool:
    prediction = _normalize_math(response)
    answer = _normalize_math(target)
    return bool(prediction) and prediction == answer


TASK_SPECS = {
    "gsm8k": TaskSpec("openai/gsm8k", "main", "train", _build_gsm8k_prompt, _gsm8k_answer, _score_number),
    "mmlu": TaskSpec("cais/mmlu", "all", "auxiliary_train", _build_mmlu_prompt, _mmlu_answer, _score_choice),
    "mmlu_stem": TaskSpec("cais/mmlu", "all", "auxiliary_train", _build_mmlu_prompt, _mmlu_answer, _score_choice),
    "mmlu_social_sciences": TaskSpec("cais/mmlu", "all", "auxiliary_train", _build_mmlu_prompt, _mmlu_answer, _score_choice),
    "winogrande": TaskSpec("vikhyatk/winogrande", "debiased", "train", _build_winogrande_prompt, _winogrande_answer, _score_choice),
    "truthfulqa": TaskSpec("truthful_qa", "multiple_choice", "validation", _build_truthfulqa_prompt, _truthfulqa_answer, _score_choice),
    "truthfulqa_mc1": TaskSpec("truthful_qa", "multiple_choice", "validation", _build_truthfulqa_prompt, _truthfulqa_answer, _score_choice),
    "truthfulqa_mc2": TaskSpec("truthful_qa", "multiple_choice", "validation", _build_truthfulqa_prompt, _truthfulqa_answer, _score_choice),
    "math500": TaskSpec("xDAN2099/lighteval-MATH", None, "train", _build_math_prompt, _math_answer, _score_math),
}


def _parse_tasks(tasks: str) -> list[str]:
    parsed = []
    for chunk in tasks.replace(";", ",").split(","):
        task = chunk.strip()
        if task and task not in parsed:
            parsed.append(task)
    return parsed


def _load_dataset(spec: TaskSpec):
    from datasets import load_dataset

    if spec.config:
        return load_dataset(spec.dataset, spec.config, split=spec.split)
    return load_dataset(spec.dataset, split=spec.split)


def _sample_candidate_indices(length: int, count: int, seed: int, used: set[int]) -> list[int]:
    if length <= 0:
        return []
    count = min(count, length)
    rng = random.Random(seed)
    selected = []
    selected_set = set()
    max_fresh = max(length - len(used), 0)
    fresh_target = min(count, max_fresh)
    while len(selected) < fresh_target:
        idx = rng.randrange(length)
        if idx in used or idx in selected_set:
            continue
        selected.append(idx)
        selected_set.add(idx)

    while len(selected) < count:
        idx = rng.randrange(length)
        if idx in selected_set:
            continue
        selected.append(idx)
        selected_set.add(idx)

    return selected


def _prompt_token_count(tokenizer, prompt: str, max_prompt_length: int) -> int:
    return len(
        tokenizer(
            prompt,
            truncation=True,
            max_length=max_prompt_length,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
    )


def _microbatches(examples: list[dict[str, Any]], tokenizer, args: argparse.Namespace):
    batch = []
    batch_tokens = 0
    for example in examples:
        tokens = _prompt_token_count(tokenizer, example["prompt"], args.max_prompt_length)
        tokens += int(args.max_new_tokens)
        if batch and (len(batch) >= args.batch_size or batch_tokens + tokens > args.max_batch_tokens):
            yield batch
            batch = []
            batch_tokens = 0
        batch.append(example)
        batch_tokens += max(tokens, 1)
    if batch:
        yield batch


def _sampling_params(args: argparse.Namespace):
    from vllm import SamplingParams

    kwargs = {
        "temperature": float(args.temperature) if args.temperature > 0 else 0.0,
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_new_tokens),
        "min_tokens": int(args.min_tokens),
        "seed": int(args.seed),
    }
    if int(args.top_k) > 0:
        kwargs["top_k"] = int(args.top_k)
    return SamplingParams(**kwargs)


def _make_example(task: str, spec: TaskSpec, dataset, idx: int) -> dict[str, Any]:
    row = dict(dataset[int(idx)])
    return {
        "task": task,
        "source_dataset": spec.dataset,
        "source_config": spec.config,
        "source_split": spec.split,
        "example_id": int(idx),
        "prompt": spec.prompt_builder(row),
        "ground_truth": spec.answer_builder(row),
    }


def _task_output_path(output_dir: Path, task: str) -> Path:
    return output_dir / f"{task}.jsonl"


def _generate_correct_for_task(
    task: str,
    spec: TaskSpec,
    dataset,
    llm,
    tokenizer,
    sampling_params,
    output_dir: Path,
    args: argparse.Namespace,
    task_idx: int,
) -> None:
    output_path = _task_output_path(output_dir, task)
    correct_count = 0
    attempt_count = 0
    round_idx = 0
    used_indices = set()
    target_count = int(args.nsamples_per_task)
    candidate_count = max(int(args.candidates_per_round), int(args.batch_size), 1)
    max_attempts = int(args.max_attempts_per_task)

    with output_path.open("w", encoding="utf-8") as f:
        with tqdm(total=target_count, desc=f"{task} correct") as progress:
            while correct_count < target_count:
                if max_attempts > 0 and attempt_count >= max_attempts:
                    raise RuntimeError(
                        f"{task}: only collected {correct_count}/{target_count} correct answers "
                        f"after {attempt_count} attempts."
                    )

                seed = int(args.seed) + task_idx * 100000 + round_idx
                indices = _sample_candidate_indices(len(dataset), candidate_count, seed, used_indices)
                used_indices.update(indices)
                examples = [_make_example(task, spec, dataset, idx) for idx in indices]
                round_idx += 1

                for batch in _microbatches(examples, tokenizer, args):
                    if correct_count >= target_count:
                        break
                    prompts = [example["prompt"] for example in batch]
                    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                    for example, output in zip(batch, outputs):
                        attempt_count += 1
                        generation = output.outputs[0] if output.outputs else None
                        answer = generation.text if generation is not None else ""
                        is_correct = spec.scorer(answer, example["ground_truth"])
                        if not is_correct:
                            continue

                        row = {
                            **example,
                            "generated_answer": answer,
                            "prompt_answer": f"{example['prompt']}\n{answer}",
                            "is_correct": True,
                            "prompt_token_count": _prompt_token_count(
                                tokenizer, example["prompt"], args.max_prompt_length
                            ),
                            "generated_token_count": len(getattr(generation, "token_ids", []) or []),
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        correct_count += 1
                        progress.update(1)
                        if correct_count >= target_count:
                            break
                    f.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate downstream train-set calibration JSONL.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--nsamples_per_task", type=int, default=128)
    parser.add_argument("--candidates_per_round", type=int, default=128)
    parser.add_argument("--max_attempts_per_task", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batch_tokens", type=int, default=32768)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--min_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--dtype", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tasks = _parse_tasks(args.tasks)
    for task in tasks:
        if task not in TASK_SPECS:
            raise ValueError(f"Unsupported task {task!r}. Supported: {', '.join(sorted(TASK_SPECS))}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=max(1, int(args.tensor_parallel_size)),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        dtype=str(args.dtype),
        max_model_len=int(args.max_prompt_length) + int(args.max_new_tokens),
        trust_remote_code=True,
        enforce_eager=True,
    )
    sampling_params = _sampling_params(args)
    try:
        for task_idx, task in enumerate(tasks):
            spec = TASK_SPECS[task]
            dataset = _load_dataset(spec)
            _generate_correct_for_task(
                task,
                spec,
                dataset,
                llm,
                tokenizer,
                sampling_params,
                output_dir,
                args,
                task_idx,
            )
    finally:
        try:
            llm.shutdown()
        except AttributeError:
            pass
        del llm


if __name__ == "__main__":
    main()
