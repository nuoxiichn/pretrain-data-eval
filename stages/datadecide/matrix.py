"""Official-matrix and local OLMES result analysis."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq


OLMES_TASKS = (
    "arc_challenge",
    "arc_easy",
    "boolq",
    "csqa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
    "mmlu",
)


def _metric(metrics: str, name: str) -> float:
    value = json.loads(metrics).get(name)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"metric {name!r} is missing or non-finite")
    return float(value)


def load_final_scores(
    path: str | Path,
    *,
    tasks_and_metrics: Mapping[str, str],
    recipes: Iterable[str] | None = None,
    scales: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Load the last positive-token checkpoint for every scale/recipe/task/seed."""
    wanted_recipes = set(recipes or ())
    wanted_scales = set(scales or ())
    table = pq.read_table(
        path,
        columns=["params", "data", "task", "step", "seed", "tokens", "metrics"],
    )
    final: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in table.to_pylist():
        task = str(row["task"])
        recipe = str(row["data"])
        scale = str(row["params"])
        tokens = int(row["tokens"])
        if task not in tasks_and_metrics or tokens <= 0:
            continue
        if wanted_recipes and recipe not in wanted_recipes:
            continue
        if wanted_scales and scale not in wanted_scales:
            continue
        key = (scale, recipe, task, str(row["seed"]))
        if key in final and int(final[key]["tokens"]) >= tokens:
            continue
        final[key] = {
            "scale": scale,
            "recipe": recipe,
            "task": task,
            "seed": str(row["seed"]),
            "step": int(row["step"]),
            "tokens": tokens,
            "metric_name": tasks_and_metrics[task],
            "score": _metric(str(row["metrics"]), tasks_and_metrics[task]),
        }
    return sorted(
        final.values(),
        key=lambda row: (row["scale"], row["recipe"], row["task"], row["seed"]),
    )


def mean_scores(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in records:
        grouped[(str(row["scale"]), str(row["recipe"]), str(row["task"]))].append(
            float(row["score"])
        )
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def pairwise_agreement(
    proxy: Mapping[str, float],
    target: Mapping[str, float],
    *,
    min_target_gap: float = 0.0,
) -> dict[str, Any]:
    """Compare all shared recipe pairs without treating target ties as decisions."""
    names = sorted(set(proxy) & set(target))
    correct = 0
    incorrect = 0
    tied_proxy = 0
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            target_delta = float(target[left]) - float(target[right])
            if abs(target_delta) <= min_target_gap:
                continue
            proxy_delta = float(proxy[left]) - float(proxy[right])
            agrees = proxy_delta * target_delta > 0
            tied = proxy_delta == 0
            correct += int(agrees)
            incorrect += int(not agrees and not tied)
            tied_proxy += int(tied)
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "proxy_delta": proxy_delta,
                    "target_delta": target_delta,
                    "agrees": agrees,
                    "proxy_tie": tied,
                }
            )
    decided = correct + incorrect + tied_proxy
    return {
        "recipes": names,
        "pair_count": decided,
        "correct": correct,
        "incorrect": incorrect,
        "proxy_ties": tied_proxy,
        "accuracy": correct / decided if decided else None,
        "pairs": pairs,
    }


def seed_stability(
    proxy_by_seed: Mapping[str, Mapping[str, float]],
    target: Mapping[str, float],
) -> dict[str, Any]:
    """Require every available training seed to make the same non-tied decision."""
    proxy_names = (
        set.intersection(*(set(scores) for scores in proxy_by_seed.values()))
        if proxy_by_seed
        else set()
    )
    names = sorted(proxy_names & set(target))
    pairs: list[dict[str, Any]] = []
    correct = 0
    incorrect = 0
    abstained = 0
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            target_delta = float(target[left]) - float(target[right])
            if target_delta == 0:
                continue
            margins = {
                seed: float(scores[left]) - float(scores[right])
                for seed, scores in sorted(proxy_by_seed.items())
            }
            directions = {
                0 if margin == 0 else (1 if margin > 0 else -1) for margin in margins.values()
            }
            unanimous = len(directions) == 1 and 0 not in directions
            agrees = unanimous and next(iter(directions)) == (1 if target_delta > 0 else -1)
            correct += int(agrees)
            incorrect += int(unanimous and not agrees)
            abstained += int(not unanimous)
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "target_delta": target_delta,
                    "proxy_deltas_by_seed": margins,
                    "mean_proxy_delta": sum(margins.values()) / len(margins),
                    "unanimous_non_tied": unanimous,
                    "agrees": agrees if unanimous else None,
                }
            )
    pair_count = len(pairs)
    decided = correct + incorrect
    return {
        "seeds": sorted(proxy_by_seed),
        "recipes": names,
        "pair_count": pair_count,
        "correct": correct,
        "incorrect": incorrect,
        "abstained": abstained,
        "coverage": decided / pair_count if pair_count else None,
        "conditional_accuracy": correct / decided if decided else None,
        "pairs": pairs,
    }


def official_summary(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    target_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the preregistered score table and edge agreement report."""
    means = mean_scores(records)
    target_scale = str(config["official"]["target_scale"])
    target_task = str(config["official"]["target_task"])
    selected = config["recipes"]
    official_to_key = {str(values["official_name"]): key for key, values in selected.items()}
    score_table: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (scale, recipe, task), value in means.items():
        if recipe not in official_to_key:
            continue
        score_table[task].setdefault(scale, {})[official_to_key[recipe]] = value

    edge_results: list[dict[str, Any]] = []
    for edge in config["edges"]:
        winner = str(edge["winner"])
        loser = str(edge["loser"])
        target_scores = score_table[target_task][target_scale]
        target_delta = target_scores[winner] - target_scores[loser]
        by_scale: dict[str, Any] = {}
        for scale in config["official"]["proxy_scales"]:
            scale = str(scale)
            scores = score_table[target_task].get(scale, {})
            if winner not in scores or loser not in scores:
                continue
            delta = scores[winner] - scores[loser]
            by_scale[scale] = {"delta": delta, "agrees": delta * target_delta > 0}
        edge_results.append(
            {
                "name": str(edge["name"]),
                "winner": winner,
                "loser": loser,
                "target_delta": target_delta,
                "target_direction_valid": target_delta > 0,
                "proxy": by_scale,
            }
        )

    target_means = mean_scores(target_records or records)
    target_score_table: dict[str, dict[str, float]] = defaultdict(dict)
    for (scale, recipe, task), value in target_means.items():
        if scale == target_scale and recipe in official_to_key:
            target_score_table[task][official_to_key[recipe]] = value

    task_agreement: dict[str, dict[str, Any]] = {}
    for task, metric_name in config["official"]["tasks_and_metrics"].items():
        target = target_score_table.get(task, {})
        task_agreement[task] = {}
        for scale in config["official"]["proxy_scales"]:
            proxy = score_table.get(task, {}).get(str(scale), {})
            task_agreement[task][str(scale)] = {
                "metric": metric_name,
                **pairwise_agreement(proxy, target),
            }

    return {
        "target_scale": target_scale,
        "target_task": target_task,
        "scores": dict(score_table),
        "edges": edge_results,
        "task_pairwise_agreement": task_agreement,
    }


def all_recipe_pairwise_summary(
    records: Sequence[Mapping[str, Any]],
    target_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure proxy agreement across every recipe in the released matrix."""
    means = mean_scores(records)
    target_means = mean_scores(target_records)
    target_scale = str(config["official"]["target_scale"])
    tasks: dict[str, Any] = {}
    for task, metric_name in config["official"]["tasks_and_metrics"].items():
        target = {
            recipe: score
            for (scale, recipe, record_task), score in target_means.items()
            if scale == target_scale and record_task == task
        }
        by_scale: dict[str, Any] = {}
        for scale in config["official"]["proxy_scales"]:
            scale = str(scale)
            proxy = {
                recipe: score
                for (record_scale, recipe, record_task), score in means.items()
                if record_scale == scale and record_task == task
            }
            seeds = sorted(
                {
                    str(row["seed"])
                    for row in records
                    if row["scale"] == scale and row["task"] == task
                }
            )
            by_seed: dict[str, Any] = {}
            for seed in seeds:
                seed_proxy = {
                    str(row["recipe"]): float(row["score"])
                    for row in records
                    if row["scale"] == scale and row["task"] == task and str(row["seed"]) == seed
                }
                by_seed[seed] = pairwise_agreement(seed_proxy, target)
            seed_accuracies = [
                result["accuracy"] for result in by_seed.values() if result["accuracy"] is not None
            ]
            by_scale[scale] = {
                "mean_score_pairwise": pairwise_agreement(proxy, target),
                "by_seed": by_seed,
                "mean_seed_accuracy": (
                    sum(seed_accuracies) / len(seed_accuracies) if seed_accuracies else None
                ),
            }
        tasks[task] = {
            "metric": metric_name,
            "target_recipe_count": len(target),
            "proxy": by_scale,
        }
    return {"target_scale": target_scale, "tasks": tasks}


def _continuous_metrics(predictions_path: Path) -> dict[str, float] | None:
    if not predictions_path.is_file():
        return None
    correct_prob_per_char: list[float] = []
    total_prob_per_char: list[float] = []
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            outputs = row.get("model_output") or []
            label = int(row.get("label", -1))
            if not outputs or not 0 <= label < len(outputs):
                continue
            probabilities = [math.exp(float(item["logits_per_char"])) for item in outputs]
            correct_prob_per_char.append(probabilities[label])
            total_prob_per_char.append(sum(probabilities))
    if not correct_prob_per_char:
        return None
    return {
        "correct_prob_per_char": sum(correct_prob_per_char) / len(correct_prob_per_char),
        "total_prob_per_char": sum(total_prob_per_char) / len(total_prob_per_char),
    }


def load_olmes_run(path: str | Path) -> dict[str, Any]:
    """Read task metrics plus continuous answer likelihoods from one OLMES run."""
    path = Path(path)
    tasks: dict[str, dict[str, Any]] = {}
    for metrics_path in sorted(path.glob("task-*-metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        task = str(payload["task_name"])
        predictions_path = metrics_path.with_name(
            metrics_path.name.replace("-metrics.json", "-predictions.jsonl")
        )
        tasks[task] = {
            "num_instances": int(payload["num_instances"]),
            **payload["metrics"],
            **(_continuous_metrics(predictions_path) or {}),
        }
    if not tasks:
        raise ValueError(f"no task metrics found in {path}")
    return {"path": str(path.resolve()), "tasks": tasks, "aggregates": aggregate_tasks(tasks)}


def _task_family(task: str) -> str:
    return "mmlu" if task == "mmlu" or task.startswith("mmlu_") else task


def aggregate_tasks(tasks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Macro-average MMLU subjects first, matching DataDecide's ten-task structure."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for task, metrics in tasks.items():
        grouped[_task_family(task)].append(metrics)
    by_family: dict[str, dict[str, float]] = {}
    metric_names = ("primary_score", "correct_prob_per_char", "total_prob_per_char")
    for family, values in grouped.items():
        by_family[family] = {}
        for metric in metric_names:
            present = [float(value[metric]) for value in values if metric in value]
            if present:
                by_family[family][metric] = sum(present) / len(present)
    aggregates: dict[str, Any] = {"by_family": by_family}
    if all(task in by_family and "primary_score" in by_family[task] for task in OLMES_TASKS):
        aggregates["olmes_10_macro_avg"] = sum(
            by_family[task]["primary_score"] for task in OLMES_TASKS
        ) / len(OLMES_TASKS)
    proxy_tasks = ("arc_easy", "arc_challenge", "mmlu")
    if all(
        task in by_family and "correct_prob_per_char" in by_family[task] for task in proxy_tasks
    ):
        aggregates["decision_proxy_correct_prob_per_char"] = sum(
            by_family[task]["correct_prob_per_char"] for task in proxy_tasks
        ) / len(proxy_tasks)
    if all(task in by_family and "primary_score" in by_family[task] for task in proxy_tasks):
        aggregates["decision_target_primary_score"] = sum(
            by_family[task]["primary_score"] for task in proxy_tasks
        ) / len(proxy_tasks)
    return aggregates
