#!/usr/bin/env python3
"""Evaluate USTBench question-answering data with an OpenAI-compatible vLLM server.

This script is intentionally independent from the original local-model USTBench
runner. It only targets the lightweight workflow used by this fork:

1. `uv sync`
2. `bash scripts/prepare_eval_data.sh`
3. `uv run python scripts/evaluate_qa_vllm.py --model <served-model-name>`

The default dataset split is `st_understanding,planning`. By default, tasks are
discovered from `data/`, so a customized Hugging Face
dataset can include only the task folders you want to evaluate.

Example for a local vLLM OpenAI-compatible server:

    export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
    export OPENAI_API_KEY=EMPTY
    uv run python scripts/evaluate_qa_vllm.py \\
      --model Qwen3-4B-Instruct-2507 \\
      --batch-size 32
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results"
DEFAULT_SYSTEM_PROMPT = REPO_ROOT / "prompts" / "system_prompt.json"
STRUCTURED_DATASETS = {"st_understanding", "planning"}


@dataclass(frozen=True)
class EvalTarget:
    task: str
    dataset: str
    path: Path


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "model"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_system_prompt(path: Path) -> str:
    if not path.exists():
        return "You are a helpful assistant."
    data = load_json(path)
    if isinstance(data, dict) and isinstance(data.get("template"), str):
        return data["template"]
    raise ValueError(f"System prompt file must contain a string field named 'template': {path}")


def discover_tasks(data_dir: Path, datasets: Iterable[str]) -> list[str]:
    datasets = list(datasets)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}. Run scripts/prepare_eval_data.sh first."
        )
    tasks = []
    for child in sorted(data_dir.iterdir()):
        if child.is_dir() and any((child / f"{dataset}_QA.json").exists() for dataset in datasets):
            tasks.append(child.name)
    if not tasks:
        wanted = ", ".join(f"{dataset}_QA.json" for dataset in datasets)
        raise FileNotFoundError(f"No task folders under {data_dir} contain any of: {wanted}")
    return tasks


def collect_targets(data_dir: Path, tasks_arg: str, datasets: list[str]) -> list[EvalTarget]:
    auto_tasks = tasks_arg.strip().lower() == "auto"
    tasks = discover_tasks(data_dir, datasets) if auto_tasks else comma_list(tasks_arg)
    targets: list[EvalTarget] = []
    missing: list[Path] = []

    for task in tasks:
        for dataset in datasets:
            path = data_dir / task / f"{dataset}_QA.json"
            if path.exists():
                targets.append(EvalTarget(task=task, dataset=dataset, path=path))
            elif not auto_tasks:
                missing.append(path)

    if missing:
        missing_text = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Explicitly requested data files are missing:\n{missing_text}")
    if not targets:
        raise FileNotFoundError("No evaluation targets were found.")
    return targets


def sample_prompt(sample: dict[str, Any]) -> str:
    if "prompt" in sample and "test_query" in sample:
        return f"{sample['prompt']}\n\n{sample['test_query']}"
    if "question" not in sample:
        raise KeyError("Sample must contain either question or prompt/test_query fields.")
    return str(sample["question"])


def final_response_segment(text: str) -> str:
    """Return the final answer segment after the last closing think tag."""
    match = re.search(r"</think>", text, flags=re.IGNORECASE)
    if not match:
        return text
    return re.split(r"</think>", text, flags=re.IGNORECASE)[-1]


def extract_json_object(text: str) -> Any | None:
    fenced = re.findall(r"```(?:json|JSON)?\s*(.*?)```", text, flags=re.DOTALL)
    candidates = list(reversed(fenced)) + [text]
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    if answer_match:
        return {"answer": answer_match.group(1)}
    return None


def extract_tagged_answers(text: str) -> list[str]:
    matches = re.findall(
        r"<Answer(?:_\d+)?>(.*?)</Answer(?:_\d+)?>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return [match.strip() for match in matches if match.strip()]


def parse_model_answer(text: str, dataset: str) -> Any | None:
    text = final_response_segment(text)
    tagged_answers = extract_tagged_answers(text)
    if dataset == "st_understanding" and tagged_answers:
        return tagged_answers[0] if len(tagged_answers) == 1 else tagged_answers

    parsed_json = extract_json_object(text)
    if isinstance(parsed_json, dict) and "answer" in parsed_json:
        return parsed_json["answer"]

    if tagged_answers:
        return tagged_answers[0] if len(tagged_answers) == 1 else tagged_answers

    compact = text.strip()
    if re.fullmatch(r"[A-Da-d]", compact):
        return compact

    final_answer = re.findall(
        r"(?:final\s+answer|answer)\s*[:：]\s*([A-Da-d])\b",
        compact,
        flags=re.IGNORECASE,
    )
    if final_answer:
        return final_answer[-1]
    return None


def normalize_answer(answer: Any) -> Any:
    if isinstance(answer, dict) and "answer" in answer:
        return normalize_answer(answer["answer"])
    if isinstance(answer, list):
        return [normalize_answer(item) for item in answer]
    text = str(answer).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            return normalize_answer(json.loads(text))
        except json.JSONDecodeError:
            pass
    text = re.sub(r"</?Answer(?:_\d+)?>", "", text, flags=re.IGNORECASE).strip()
    split_answers = [part.strip() for part in re.split(r"[,;/\s]+", text) if part.strip()]
    if len(split_answers) > 1 and all(re.fullmatch(r"[A-Da-d]", part) for part in split_answers):
        return [part.upper() for part in split_answers]
    if re.fullmatch(r"[A-Da-d]", text):
        return text.upper()
    return re.sub(r"\s+", " ", text).upper()


def answers_equal(predicted: Any, gold: Any) -> bool:
    return normalize_answer(predicted) == normalize_answer(gold)


def relation_metrics(samples: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for sample in samples:
        relation = str(sample.get("spatial_temporal_relation", "unknown"))
        entry = metrics.setdefault(relation, {"num": 0, "correct_num": 0, "accuracy": 0.0})
        entry["num"] = int(entry["num"]) + 1
        if sample.get("is_correct"):
            entry["correct_num"] = int(entry["correct_num"]) + 1
    for entry in metrics.values():
        total = int(entry["num"])
        correct = int(entry["correct_num"])
        entry["accuracy"] = correct / total if total else 0.0
    return metrics


class VllmChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout: float,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def complete_once(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        message = completion.choices[0].message
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning and content:
            return f"<think>\n{reasoning}\n</think>\n{content}"
        return content


def complete_with_retry(client: VllmChatClient, prompt: str, max_retries: int, retry_sleep: float) -> tuple[str | None, str | None]:
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            return client.complete_once(prompt), None
        except Exception as exc:  # noqa: BLE001 - CLI should report and retry service errors.
            last_error = repr(exc)
            if attempt < max_retries:
                time.sleep(retry_sleep)
    return None, last_error


def evaluate_samples(
    *,
    client: VllmChatClient,
    samples: list[dict[str, Any]],
    dataset: str,
    batch_size: int,
    max_retries: int,
    retry_sleep: float,
) -> list[dict[str, Any]]:
    answered = [dict(sample, reasoning=None, decision=None, is_correct=False, error=None) for sample in samples]
    pending = list(range(len(samples)))

    for attempt in range(max_retries + 1):
        if not pending:
            break
        next_pending: list[int] = []
        prompts = {idx: sample_prompt(samples[idx]) for idx in pending}
        desc = f"{dataset} attempt {attempt + 1}/{max_retries + 1}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            future_to_idx = {
                executor.submit(complete_with_retry, client, prompts[idx], 0, retry_sleep): idx
                for idx in pending
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx), desc=desc):
                idx = future_to_idx[future]
                try:
                    response, error = future.result()
                except Exception as exc:  # pragma: no cover - defensive, future should not raise.
                    response, error = None, repr(exc)

                if response is None:
                    answered[idx].update({"error": error or "empty response"})
                    if attempt < max_retries:
                        next_pending.append(idx)
                    continue

                decision = parse_model_answer(response, dataset)
                if decision is None:
                    answered[idx].update({"reasoning": response, "error": "failed to parse structured answer"})
                    if attempt < max_retries:
                        next_pending.append(idx)
                    continue

                is_correct = answers_equal(decision, samples[idx].get("answer"))
                answered[idx].update(
                    {
                        "reasoning": response,
                        "decision": decision,
                        "is_correct": is_correct,
                        "error": None,
                    }
                )
        pending = next_pending
        if pending and attempt < max_retries and retry_sleep > 0:
            time.sleep(retry_sleep)

    return answered


def evaluate_target(
    *,
    target: EvalTarget,
    client: VllmChatClient,
    batch_size: int,
    max_retries: int,
    retry_sleep: float,
    limit: int | None,
    output_dir: Path,
    model_name: str,
) -> dict[str, Any]:
    samples = load_json(target.path)
    if not isinstance(samples, list):
        raise ValueError(f"Expected a JSON list in {target.path}")
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"No samples to evaluate in {target.path}")

    print(f"========================== Task: {target.task} | Dataset: {target.dataset} ==========================")
    print(f"Loaded {len(samples)} questions from {target.path}")

    answered = evaluate_samples(
        client=client,
        samples=samples,
        dataset=target.dataset,
        batch_size=batch_size,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
    )

    total = len(answered)
    correct = sum(1 for sample in answered if sample.get("is_correct"))
    accuracy = correct / total if total else 0.0
    parse_failures = sum(1 for sample in answered if sample.get("decision") is None)
    metrics: dict[str, Any] = {
        "task": target.task,
        "dataset": target.dataset,
        "data_path": str(target.path),
        "num_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "parse_failures": parse_failures,
    }
    if target.dataset == "st_understanding":
        metrics["spatial_temporal_results"] = relation_metrics(answered)

    model_file = safe_filename(model_name)
    result_path = output_dir / target.task / f"{model_file}_{target.dataset}_QA.json"
    dump_json(answered, result_path)
    metrics["result_path"] = str(result_path)
    print(f"Accuracy: {correct}/{total} = {accuracy:.4f}; parse_failures={parse_failures}")
    print(f"Saved responses to: {result_path}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL"), help="vLLM served model name. Defaults to VLLM_MODEL.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"), help="OpenAI-compatible base URL. CLI value overrides OPENAI_BASE_URL.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"), help="API key for the OpenAI-compatible server. CLI value overrides OPENAI_API_KEY.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Installed QA data directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for per-sample outputs and summary metrics.")
    parser.add_argument("--tasks", default="auto", help='Comma-separated task list, or "auto" to discover available task folders.')
    parser.add_argument("--datasets", default="st_understanding,planning", help="Comma-separated QA subsets to evaluate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Maximum concurrent requests to the vLLM service.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries for API errors or unparsable structured answers.")
    parser.add_argument("--retry-sleep", type=float, default=2.0, help="Seconds to sleep between service retries.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-file sample limit for smoke tests.")
    parser.add_argument("--system-prompt-file", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.model:
        parser.error("--model is required unless VLLM_MODEL is set.")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1.")
    datasets = comma_list(args.datasets)
    unsupported = [dataset for dataset in datasets if dataset not in STRUCTURED_DATASETS]
    if unsupported:
        parser.error(f"This lightweight evaluator only supports: {', '.join(sorted(STRUCTURED_DATASETS))}; got {unsupported}")

    targets = collect_targets(args.data_dir, args.tasks, datasets)
    system_prompt = load_system_prompt(args.system_prompt_file)
    client = VllmChatClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        system_prompt=system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Targets: {', '.join(f'{target.task}/{target.dataset}' for target in targets)}")

    target_metrics = []
    for target in targets:
        target_metrics.append(
            evaluate_target(
                target=target,
                client=client,
                batch_size=args.batch_size,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                limit=args.limit,
                output_dir=args.output_dir,
                model_name=args.model,
            )
        )

    total_questions = sum(item["num_questions"] for item in target_metrics)
    total_correct = sum(item["correct"] for item in target_metrics)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "data_dir": str(args.data_dir),
        "datasets": datasets,
        "tasks_arg": args.tasks,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "overall": {
            "num_questions": total_questions,
            "correct": total_correct,
            "accuracy": total_correct / total_questions if total_questions else 0.0,
        },
        "targets": target_metrics,
    }
    summary_path = args.output_dir / f"{safe_filename(args.model)}_summary.json"
    dump_json(summary, summary_path)
    print("========================== Final Overall ==========================")
    print(f"Final accuracy: {total_correct}/{total_questions} = {summary['overall']['accuracy']:.4f}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
