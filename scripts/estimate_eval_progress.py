#!/usr/bin/env python3
"""Estimate resumable QA evaluation progress for all models under results/.

Progress is measured the same way as the resumable evaluator resumes work:
a sample is counted as completed only when its saved result record has a
non-null ``decision`` field. Missing result files count as 0 completed samples.

Examples:
    uv run python scripts/estimate_eval_progress.py
    uv run python scripts/estimate_eval_progress.py --models qwen3-4b-thinking-2507
    uv run python scripts/estimate_eval_progress.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_DATASETS = ("st_understanding", "planning")
COMPATIBILITY_ALIASES = {
    # The 10% HF sample may include this compatibility alias; counts metadata
    # treats the misspelled upstream task name as canonical and excludes alias
    # cases from totals.
    "socio_economic_prediction": "socio_ecomic_prediction",
}
RESULT_TASK_FALLBACKS = {
    canonical: [alias] for alias, canonical in COMPATIBILITY_ALIASES.items()
}


@dataclass(frozen=True)
class EvalTarget:
    task: str
    dataset: str
    path: Path
    num_questions: int


@dataclass(frozen=True)
class TargetProgress:
    task: str
    dataset: str
    completed: int
    total: int
    progress: float
    result_path: str | None
    result_records: int
    missing_result: bool
    length_mismatch: bool


@dataclass(frozen=True)
class ModelProgress:
    model: str
    completed: int
    total: int
    progress: float
    completed_targets: int
    started_targets: int
    total_targets: int
    targets: list[TargetProgress]


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_targets(data_dir: Path, datasets: Iterable[str], *, include_aliases: bool = False) -> list[EvalTarget]:
    datasets = list(datasets)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    task_dirs = sorted(child for child in data_dir.iterdir() if child.is_dir())
    task_names = {task_dir.name for task_dir in task_dirs}
    targets: list[EvalTarget] = []
    for task_dir in task_dirs:
        canonical = COMPATIBILITY_ALIASES.get(task_dir.name)
        if not include_aliases and canonical in task_names:
            continue
        for dataset in datasets:
            path = task_dir / f"{dataset}_QA.json"
            if not path.exists():
                continue
            data = load_json(path)
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list: {path}")
            targets.append(
                EvalTarget(
                    task=task_dir.name,
                    dataset=dataset,
                    path=path,
                    num_questions=len(data),
                )
            )
    if not targets:
        wanted = ", ".join(f"{dataset}_QA.json" for dataset in datasets)
        raise FileNotFoundError(f"No QA targets under {data_dir} contain any of: {wanted}")
    return targets


def model_from_result_name(name: str, datasets: Iterable[str]) -> str | None:
    for dataset in sorted(datasets, key=len, reverse=True):
        suffix = f"_{dataset}_QA.json"
        if name.endswith(suffix):
            model = name[: -len(suffix)]
            return model or None
    return None


def discover_models(results_dir: Path, datasets: Iterable[str]) -> list[str]:
    if not results_dir.exists():
        return []
    models = set()
    for path in results_dir.glob("*/*_QA.json"):
        model = model_from_result_name(path.name, datasets)
        if model:
            models.add(model)
    return sorted(models)


def completed_count(records: Any) -> int:
    if not isinstance(records, list):
        raise ValueError("Result file must contain a JSON list")
    return sum(1 for record in records if isinstance(record, dict) and record.get("decision") is not None)


def find_result_path(results_dir: Path, model: str, target: EvalTarget) -> Path:
    primary = results_dir / target.task / f"{model}_{target.dataset}_QA.json"
    if primary.exists():
        return primary
    for fallback_task in RESULT_TASK_FALLBACKS.get(target.task, []):
        fallback = results_dir / fallback_task / f"{model}_{target.dataset}_QA.json"
        if fallback.exists():
            return fallback
    return primary


def progress_for_model(model: str, targets: list[EvalTarget], results_dir: Path) -> ModelProgress:
    target_progress: list[TargetProgress] = []
    completed_total = 0
    question_total = 0
    completed_targets = 0
    started_targets = 0

    for target in targets:
        result_path = find_result_path(results_dir, model, target)
        missing_result = not result_path.exists()
        result_records = 0
        completed = 0
        length_mismatch = False

        if not missing_result:
            records = load_json(result_path)
            if not isinstance(records, list):
                raise ValueError(f"Expected a JSON list: {result_path}")
            result_records = len(records)
            completed = min(completed_count(records), target.num_questions)
            length_mismatch = result_records != target.num_questions
            started_targets += 1

        question_total += target.num_questions
        completed_total += completed
        if completed >= target.num_questions:
            completed_targets += 1

        target_progress.append(
            TargetProgress(
                task=target.task,
                dataset=target.dataset,
                completed=completed,
                total=target.num_questions,
                progress=completed / target.num_questions if target.num_questions else 0.0,
                result_path=str(result_path) if not missing_result else None,
                result_records=result_records,
                missing_result=missing_result,
                length_mismatch=length_mismatch,
            )
        )

    return ModelProgress(
        model=model,
        completed=completed_total,
        total=question_total,
        progress=completed_total / question_total if question_total else 0.0,
        completed_targets=completed_targets,
        started_targets=started_targets,
        total_targets=len(targets),
        targets=target_progress,
    )


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def ratio_text(completed: int, total: int) -> str:
    return f"{completed:,}/{total:,}"


def progress_style(progress: float) -> str:
    if progress >= 1.0:
        return "green"
    if progress > 0:
        return "yellow"
    return "red"


def status_text(target: TargetProgress) -> Text:
    status: list[tuple[str, str]] = []
    if target.missing_result:
        status.append(("missing", "red"))
    if target.length_mismatch:
        status.append((f"records={target.result_records}", "magenta"))
    if not status and target.completed >= target.total:
        status.append(("done", "green"))
    elif not status:
        status.append(("partial", "yellow"))

    text = Text()
    for idx, (label, style) in enumerate(status):
        if idx:
            text.append(", ", style="dim")
        text.append(label, style=style)
    return text


def render_summary(console: Console, model_progress: list[ModelProgress], *, data_dir: Path, results_dir: Path, num_targets: int, total_questions: int) -> None:
    panel_text = Text()
    panel_text.append("Data: ", style="bold")
    panel_text.append(str(data_dir))
    panel_text.append("\nResults: ", style="bold")
    panel_text.append(str(results_dir))
    panel_text.append("\nTargets: ", style="bold")
    panel_text.append(f"{num_targets}")
    panel_text.append("   Questions: ", style="bold")
    panel_text.append(f"{total_questions:,}")
    console.print(Panel(panel_text, title="USTBench QA Evaluation Progress", border_style="cyan", box=box.ROUNDED))

    if not model_progress:
        console.print("[yellow]No result files/models found.[/yellow]")
        return

    table = Table(title="Overall by model", box=box.SIMPLE_HEAVY, header_style="bold cyan")
    table.add_column("Model", overflow="fold")
    table.add_column("Progress", justify="right", no_wrap=True)
    table.add_column("Bar", min_width=20)
    table.add_column("Samples", justify="right", no_wrap=True)
    table.add_column("Targets", justify="right", no_wrap=True)

    for item in model_progress:
        style = progress_style(item.progress)
        targets_text = Text()
        targets_text.append(f"done {ratio_text(item.completed_targets, item.total_targets)}", style=style)
        targets_text.append(" · ", style="dim")
        targets_text.append(f"started {ratio_text(item.started_targets, item.total_targets)}", style="cyan")
        table.add_row(
            item.model,
            Text(format_pct(item.progress), style=style),
            ProgressBar(total=1.0, completed=item.progress, width=24, complete_style=style),
            ratio_text(item.completed, item.total),
            targets_text,
        )
    console.print(table)


def render_targets(console: Console, model_progress: list[ModelProgress], *, show_targets: bool) -> None:
    if not show_targets:
        return

    for item in model_progress:
        table = Table(title=f"{item.model} targets", box=box.SIMPLE, header_style="bold magenta")
        table.add_column("Task", overflow="fold")
        table.add_column("Dataset", no_wrap=True)
        table.add_column("Samples", justify="right", no_wrap=True)
        table.add_column("Progress", justify="right", no_wrap=True)
        table.add_column("Status")

        for target in item.targets:
            style = progress_style(target.progress)
            table.add_row(
                target.task,
                target.dataset,
                ratio_text(target.completed, target.total),
                Text(format_pct(target.progress), style=style),
                status_text(target),
            )
        console.print(table)


def print_report(
    model_progress: list[ModelProgress],
    *,
    data_dir: Path,
    results_dir: Path,
    num_targets: int,
    total_questions: int,
    show_targets: bool,
) -> None:
    console = Console()
    render_summary(
        console,
        model_progress,
        data_dir=data_dir,
        results_dir=results_dir,
        num_targets=num_targets,
        total_questions=total_questions,
    )
    render_targets(console, model_progress, show_targets=show_targets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="QA data directory. Default: ./data")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Evaluation results directory. Default: ./results")
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated QA subsets to count. Default: st_understanding,planning",
    )
    parser.add_argument("--models", default=None, help="Comma-separated model names. Default: discover all models in results/.")
    parser.add_argument("--no-targets", action="store_true", help="Only print the per-model summary table.")
    parser.add_argument(
        "--include-aliases",
        action="store_true",
        help="Include compatibility alias task directories, even when the canonical task directory is also present.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of Rich tables.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    datasets = comma_list(args.datasets)
    targets = discover_targets(args.data_dir, datasets, include_aliases=args.include_aliases)
    models = comma_list(args.models) if args.models else discover_models(args.results_dir, datasets)

    progress = [progress_for_model(model, targets, args.results_dir) for model in models]
    total_questions = sum(target.num_questions for target in targets)

    if args.json:
        payload = {
            "data_dir": str(args.data_dir),
            "results_dir": str(args.results_dir),
            "datasets": datasets,
            "include_aliases": args.include_aliases,
            "num_targets": len(targets),
            "total_questions": total_questions,
            "models": [asdict(item) for item in progress],
        }
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return

    print_report(
        progress,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        num_targets=len(targets),
        total_questions=total_questions,
        show_targets=not args.no_targets,
    )


if __name__ == "__main__":
    main()
