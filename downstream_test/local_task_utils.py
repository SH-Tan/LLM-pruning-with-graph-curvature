from __future__ import annotations

import importlib.util
import importlib
import os
import re
import sys
from pathlib import Path
from typing import Any

LOCAL_TASK_DATASETS = {
    "gsm8k": "downstream_test/dataset/gsm8k/test.parquet",
    "math500": "downstream_test/dataset/mathqa500/test.parquet",
}

LOCAL_TASK_PROMPT_KEYS = {
    "gsm8k": "question",
    "math500": "prompt",
}

LOCAL_TASK_ALIASES = {
    "gsm8k": "gsm8k",
    "openai/gsm8k": "gsm8k",
    "gsm8k_cot": "gsm8k",
    "mth500": "math500",
    "math500": "math500",
    "math_500": "math500",
    "mathqa500": "math500",
    "hendrycks_math500": "math500",
    "metamathqa_math_500": "math500",
}

MATH_DATA_SOURCES = {"lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500", "math_500"}
MATH_DAPO_DATA_SOURCES = {"math_dapo", "math", "math_dapo_reasoning"}
VERL_SCORER_BACKENDS = {"verl_default", "verl_math_reward", "verl_math_verify", "legacy_modules", "fallback"}
GSM8K_INSTRUCTION = ' Let\'s think step by step and output the final answer after "####".'


def local_task_name(benchmark: str, tasks: list[str] | tuple[str, ...] | None = None) -> str | None:
    candidates = [benchmark, *(tasks or [])]
    for candidate in candidates:
        normalized = str(candidate).strip().lower()
        if normalized in LOCAL_TASK_ALIASES:
            return LOCAL_TASK_ALIASES[normalized]
    return None


def local_task_dataset_path(task_name: str) -> str:
    return LOCAL_TASK_DATASETS[task_name]


def infer_data_source(dataset_path: str | Path, fallback: str = "") -> str:
    if fallback:
        return fallback
    path = str(dataset_path).lower()
    parts = {part.lower() for part in Path(path).parts}
    if "gsm8k" in parts or path in {"gsm8k", "openai/gsm8k"}:
        return "openai/gsm8k"
    if "mathqa500" in parts or "math500" in parts or path in {"math500", "math_500"}:
        return "math_500"
    return fallback


def resolve_local_task(benchmark: str, tasks: list[str] | tuple[str, ...] | None = None) -> dict[str, str] | None:
    task_name = local_task_name(benchmark, tasks)
    if task_name is None:
        return None
    return {
        "task_name": task_name,
        "dataset_path": local_task_dataset_path(task_name),
        "prompt_key": LOCAL_TASK_PROMPT_KEYS[task_name],
    }


def local_prompt_text(task_name: str, prompt_text: str) -> str:
    if task_name == "gsm8k" and "####" not in prompt_text:
        return prompt_text + GSM8K_INSTRUCTION
    return prompt_text


def normalize_local_ground_truth(data_source: str, ground_truth: Any) -> Any:
    if data_source == "openai/gsm8k":
        target = _extract_last_number(ground_truth)
        return target if target is not None else ground_truth
    return ground_truth


def _extract_last_number(text: Any) -> str | None:
    numbers = re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", str(text))
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def fallback_gsm8k_score(response_text: str, ground_truth: Any) -> float:
    response_answer = re.findall(r"####\s*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)", response_text)
    prediction = response_answer[-1].replace(",", "") if response_answer else _extract_last_number(response_text)
    target_answer = re.findall(r"####\s*(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)", str(ground_truth))
    target = target_answer[-1].replace(",", "") if target_answer else _extract_last_number(ground_truth)
    return float(prediction is not None and target is not None and prediction == target)


def _extract_math_answer(text: Any) -> str:
    text = str(text)
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return matches[-1]
    return text.strip().splitlines()[-1] if text.strip() else ""


def _normalize_math_answer(answer: Any) -> str:
    answer = _extract_math_answer(answer).strip().strip("$").strip()
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    answer = re.sub(r"\\text\{([^{}]*)\}", r"\1", answer)
    answer = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", answer)
    answer = answer.replace(",", "")
    answer = re.sub(r"\s+", "", answer)
    return answer.lower()


def fallback_math_score(response_text: str, ground_truth: Any) -> float:
    prediction = _normalize_math_answer(response_text)
    target = _normalize_math_answer(ground_truth)
    return float(bool(prediction) and prediction == target)


def scorer_backend() -> str:
    backend = os.environ.get("TASK_SCORER_BACKEND") or os.environ.get("MATH_SCORER") or "verl_math_reward"
    backend = backend.strip().lower()
    if backend == "verl_default":
        backend = "verl_math_reward"
    if backend not in VERL_SCORER_BACKENDS:
        choices = ", ".join(sorted(VERL_SCORER_BACKENDS))
        raise ValueError(f"Unsupported scorer backend {backend!r}. Choose one of: {choices}")
    return backend


def _ensure_verl_import_path() -> None:
    candidates = [
        Path(__file__).resolve().parents[1] / "verl",
        Path(__file__).resolve().parents[2] / "verl",
    ]
    for path in candidates:
        if (path / "verl" / "utils" / "reward_score").is_dir():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return


def _import_verl_default_compute_score():
    _ensure_verl_import_path()
    try:
        module = importlib.import_module("verl.utils.reward_score")
    except ImportError as exc:
        raise ImportError(
            "Unable to import verl.utils.reward_score. Activate the verl environment before scoring, "
            "or set TASK_SCORER_BACKEND=legacy_modules only if you intentionally do not want verl scoring."
        ) from exc
    return module.default_compute_score


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def compute_score_with_verl_default(data_source: str, response_text: str, ground_truth: Any) -> Any:
    return _import_verl_default_compute_score()(
        data_source,
        response_text,
        ground_truth,
        math_dapo_binary_reward=_env_flag("MATH_DAPO_BINARY_REWARD", default=True),
    )


def compute_score_with_verl_math_verify(response_text: str, ground_truth: Any) -> Any:
    _ensure_verl_import_path()
    try:
        module = importlib.import_module("verl.utils.reward_score.math_verify")
    except ImportError as exc:
        raise ImportError(
            "Unable to import verl.utils.reward_score.math_verify. Activate the verl environment and ensure "
            "math-verify is installed, or use TASK_SCORER_BACKEND=verl_math_reward."
        ) from exc
    return module.compute_score(model_output=response_text, ground_truth=str(ground_truth))


def reward_module_path(module_name: str, reward_score_dir: str | Path | None = None) -> Path:
    if reward_score_dir is not None:
        return Path(reward_score_dir).expanduser() / f"{module_name}.py"
    if os.environ.get("VERL_REWARD_SCORE_DIR"):
        return Path(os.environ["VERL_REWARD_SCORE_DIR"]).expanduser() / f"{module_name}.py"
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "verl" / "utils" / "reward_score" / f"{module_name}.py",
        here.parents[2] / "verl" / "utils" / "reward_score" / f"{module_name}.py",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[-1]


def load_reward_module(module_name: str, reward_score_dir: str | Path | None = None):
    module_path = reward_module_path(module_name, reward_score_dir)
    if not module_path.is_file():
        raise FileNotFoundError(f"Reward module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(f"_vllm_accuracy_reward_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load reward module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_score_with_legacy_reward_module(data_source: str, response_text: str, ground_truth: Any, reward_score_dir: str | Path | None = None) -> Any:
    if data_source == "openai/gsm8k":
        return load_reward_module("gsm8k", reward_score_dir).compute_score(response_text, ground_truth)
    if data_source in MATH_DATA_SOURCES:
        try:
            return load_reward_module("math_reward", reward_score_dir).compute_score(response_text, ground_truth)
        except FileNotFoundError:
            return fallback_math_score(response_text, ground_truth)
    if data_source in MATH_DAPO_DATA_SOURCES or data_source.startswith("aime"):
        try:
            return load_reward_module("math_dapo", reward_score_dir).compute_score(response_text, ground_truth, incorrect_reward=0.0)
        except FileNotFoundError:
            return fallback_math_score(response_text, ground_truth)
    return fallback_math_score(response_text, ground_truth)


def compute_score_with_reward_module(data_source: str, response_text: str, ground_truth: Any, reward_score_dir: str | Path | None = None) -> Any:
    if reward_score_dir is not None:
        return compute_score_with_legacy_reward_module(data_source, response_text, ground_truth, reward_score_dir=reward_score_dir)

    backend = scorer_backend()
    if backend == "verl_math_reward":
        return compute_score_with_verl_default(data_source, response_text, ground_truth)
    if backend == "verl_math_verify":
        if data_source in MATH_DATA_SOURCES or data_source in MATH_DAPO_DATA_SOURCES or data_source.startswith("aime"):
            return compute_score_with_verl_math_verify(response_text, ground_truth)
        return compute_score_with_verl_default(data_source, response_text, ground_truth)
    if backend == "legacy_modules":
        return compute_score_with_legacy_reward_module(data_source, response_text, ground_truth)
    if data_source == "openai/gsm8k":
        return fallback_gsm8k_score(response_text, ground_truth)
    return fallback_math_score(response_text, ground_truth)


def score_local_response(data_source: str, response_text: str, ground_truth: Any, reward_score_dir: str | Path | None = None) -> float:
    score = compute_score_with_reward_module(data_source, response_text, ground_truth, reward_score_dir=reward_score_dir)
    if isinstance(score, dict):
        for key in ("score", "reward", "accuracy", "acc"):
            if key in score:
                return float(score[key])
        raise ValueError(f"Cannot scalarize score dictionary: {score}")
    return float(score)
