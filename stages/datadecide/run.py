"""CLI for the DataDecide benchmark-prediction reproduction."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import click
import yaml

from pretrain_data_eval.schema import use_output_dir, write_summary
from stages.datadecide.matrix import (
    all_recipe_pairwise_summary,
    load_final_scores,
    load_olmes_run,
    mean_scores,
    official_summary,
    pairwise_agreement,
    seed_stability,
)
from stages.datadecide.prepare import materialize_recipe
from stages.datadecide.train import export_trainer_state, train_20m


def _config(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section in ("official", "recipes", "edges", "olmes", "local_training"):
        if section not in values:
            raise click.ClickException(f"config is missing section {section!r}")
    return values


def _official_records(config: dict[str, Any], results_path: Path | None = None):
    official = config["official"]
    path = results_path or Path(official["results_path"])
    return load_final_scores(
        path,
        tasks_and_metrics=official["tasks_and_metrics"],
        recipes=[values["official_name"] for values in config["recipes"].values()],
        scales=[*official["proxy_scales"], official["target_scale"]],
    )


def _official_target_records(config: dict[str, Any], results_path: Path | None = None):
    official = config["official"]
    path = results_path or Path(official["results_path"])
    return load_final_scores(
        path,
        tasks_and_metrics={task: "primary_metric" for task in official["tasks_and_metrics"]},
        recipes=[values["official_name"] for values in config["recipes"].values()],
        scales=[official["target_scale"]],
    )


@click.group()
def cli() -> None:
    """Reproduce small-model benchmark prediction of 1B data rankings."""


@cli.command("official-matrix")
@click.option("--config", "config_path", type=click.Path(path_type=Path), required=True)
@click.option("--results", "results_path", type=click.Path(path_type=Path), default=None)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
def official_matrix(config_path: Path, results_path: Path | None, output_dir: Path) -> None:
    """Recompute the preregistered evidence from the released result parquet."""
    config = _config(config_path)
    official = config["official"]
    path = results_path or Path(official["results_path"])
    all_records = load_final_scores(
        path,
        tasks_and_metrics=official["tasks_and_metrics"],
        scales=[*official["proxy_scales"], official["target_scale"]],
    )
    all_target_records = load_final_scores(
        path,
        tasks_and_metrics={task: "primary_metric" for task in official["tasks_and_metrics"]},
        scales=[official["target_scale"]],
    )
    selected_names = {values["official_name"] for values in config["recipes"].values()}
    records = [row for row in all_records if row["recipe"] in selected_names]
    target_records = [row for row in all_target_records if row["recipe"] in selected_names]
    analysis = official_summary(records, config, target_records=target_records)
    output = use_output_dir(output_dir)
    summary_path = write_summary(
        {
            "stage": 12,
            "method": "datadecide_official_matrix_reanalysis",
            "config": str(config_path.resolve()),
            "record_count": len(records),
            "analysis": analysis,
            "all_recipe_record_count": len(all_records),
            "all_recipe_analysis": all_recipe_pairwise_summary(
                all_records, all_target_records, config
            ),
            "evidence_level": "official released evaluations; not a local training reproduction",
        },
        output,
    )
    click.echo(summary_path)


def _parse_run(value: str) -> tuple[str, str, str, Path]:
    try:
        identity, path = value.split("=", 1)
        recipe, scale, seed = identity.split(":", 2)
    except ValueError as exc:
        raise click.BadParameter("expected RECIPE:SCALE:SEED=PATH") from exc
    return recipe, scale, seed, Path(path)


@cli.command("aggregate-olmes")
@click.option("--config", "config_path", type=click.Path(path_type=Path), required=True)
@click.option("--results", "results_path", type=click.Path(path_type=Path), default=None)
@click.option("--run", "runs", multiple=True, required=True, help="RECIPE:SCALE:SEED=PATH")
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
def aggregate_olmes(
    config_path: Path,
    results_path: Path | None,
    runs: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Aggregate local OLMES outputs and compare them with official 1B rankings."""
    config = _config(config_path)
    proxy_records = _official_records(config, results_path)
    target_records = _official_target_records(config, results_path)
    decision_tasks = tuple(config["olmes"]["decision_tasks"])
    analysis_tasks = decision_tasks + tuple(config["olmes"]["supplemental_tasks"])
    local_runs: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    grouped_by_seed: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    def add_score(scale: str, metric: str, recipe: str, seed: str, score: float) -> None:
        seed_scores = grouped_by_seed[(scale, metric)][seed]
        if recipe in seed_scores:
            raise click.ClickException(
                f"duplicate run for recipe={recipe}, scale={scale}, seed={seed}, metric={metric}"
            )
        value = float(score)
        seed_scores[recipe] = value
        grouped[(scale, metric)][recipe].append(value)

    for value in runs:
        recipe, scale, seed, path = _parse_run(value)
        if recipe not in config["recipes"]:
            raise click.BadParameter(f"unknown recipe {recipe!r}", param_hint="--run")
        result = load_olmes_run(path)
        aggregates = result["aggregates"]
        decision_proxy = aggregates.get("decision_proxy_correct_prob_per_char")
        if decision_proxy is None:
            raise click.ClickException(
                f"{path} lacks ARC-Easy, ARC-Challenge, or MMLU continuous metrics"
            )
        add_score(scale, "decision_proxy_correct_prob_per_char", recipe, seed, decision_proxy)
        if "decision_target_primary_score" in aggregates:
            add_score(
                scale,
                "decision_target_primary_score",
                recipe,
                seed,
                aggregates["decision_target_primary_score"],
            )
        by_family = aggregates["by_family"]
        for task in analysis_tasks:
            task_metrics = by_family.get(task, {})
            if "correct_prob_per_char" in task_metrics:
                add_score(
                    scale,
                    f"{task}:correct_prob_per_char",
                    recipe,
                    seed,
                    task_metrics["correct_prob_per_char"],
                )
            if "primary_score" in task_metrics:
                add_score(
                    scale,
                    f"{task}:primary_score",
                    recipe,
                    seed,
                    task_metrics["primary_score"],
                )
        if "olmes_10_macro_avg" in aggregates:
            add_score(scale, "olmes_10_macro_avg", recipe, seed, aggregates["olmes_10_macro_avg"])
        local_runs.append({"recipe": recipe, "scale": scale, "seed": seed, **result})

    target_scale = str(config["official"]["target_scale"])
    official_to_key = {
        str(values["official_name"]): key for key, values in config["recipes"].items()
    }
    target_means = mean_scores(target_records)
    target_scores_by_task: dict[str, dict[str, float]] = defaultdict(dict)
    for (scale, recipe, task), score in target_means.items():
        if scale == target_scale and recipe in official_to_key:
            target_scores_by_task[task][official_to_key[recipe]] = score
    target_olmes = target_scores_by_task[config["official"]["target_task"]]
    target_decision: dict[str, float] = {}
    for recipe in config["recipes"]:
        values = [target_scores_by_task.get(task, {}).get(recipe) for task in decision_tasks]
        if all(value is not None for value in values):
            target_decision[recipe] = sum(float(value) for value in values) / len(values)
    targets = {
        "olmes_10_macro_avg": target_olmes,
        "decision_proxy_correct_prob_per_char": target_decision,
    }
    for task in analysis_tasks:
        targets[f"{task}:correct_prob_per_char"] = target_scores_by_task[task]
    proxy_default_by_scale_task: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in proxy_records:
        recipe = official_to_key.get(str(row["recipe"]))
        if recipe is not None and str(row["seed"]) == "default":
            proxy_default_by_scale_task[(str(row["scale"]), str(row["task"]))][recipe] = float(
                row["score"]
            )
    official_by_scale: dict[tuple[str, str], dict[str, float]] = {}
    for scale in (*config["official"]["proxy_scales"], target_scale):
        scale = str(scale)
        official_by_scale[(scale, "olmes_10_macro_avg")] = proxy_default_by_scale_task.get(
            (scale, config["official"]["target_task"]), {}
        )
        decision_scores: dict[str, float] = {}
        for recipe in config["recipes"]:
            values = [
                proxy_default_by_scale_task.get((scale, task), {}).get(recipe)
                for task in decision_tasks
            ]
            if all(value is not None for value in values):
                decision_scores[recipe] = sum(float(value) for value in values) / len(values)
        official_by_scale[(scale, "decision_proxy_correct_prob_per_char")] = decision_scores
        for task in analysis_tasks:
            official_by_scale[(scale, f"{task}:correct_prob_per_char")] = (
                proxy_default_by_scale_task.get((scale, task), {})
            )

    means_by_scale_metric: dict[tuple[str, str], dict[str, float]] = {}
    for key, recipes in grouped.items():
        means_by_scale_metric[key] = {
            name: sum(values) / len(values) for name, values in recipes.items()
        }
    comparisons: dict[str, Any] = {}
    comparison_metrics = (
        "olmes_10_macro_avg",
        "decision_proxy_correct_prob_per_char",
        *(f"{task}:correct_prob_per_char" for task in analysis_tasks),
    )
    local_target_metrics = {
        "olmes_10_macro_avg": "olmes_10_macro_avg",
        "decision_proxy_correct_prob_per_char": "decision_target_primary_score",
        **{f"{task}:correct_prob_per_char": f"{task}:primary_score" for task in analysis_tasks},
    }
    for (scale, metric), means in means_by_scale_metric.items():
        if metric not in comparison_metrics:
            continue
        target = targets[metric]
        comparison = {
            "metric": metric,
            "scores": means,
            "target_scale": target_scale,
            "target_scores": target,
            "pairwise": pairwise_agreement(means, target),
        }
        by_seed = grouped_by_seed[(scale, metric)]
        comparison["scores_by_seed"] = by_seed
        comparison["pairwise_by_seed"] = {
            seed: pairwise_agreement(scores, target) for seed, scores in sorted(by_seed.items())
        }
        if len(by_seed) > 1:
            comparison["unanimous_seed_decisions"] = seed_stability(by_seed, target)
        same_scale = official_by_scale.get((scale, metric), {})
        if same_scale:
            comparison["official_same_scale_scores"] = same_scale
            comparison["pairwise_vs_official_same_scale"] = pairwise_agreement(means, same_scale)
            comparison["score_deltas_vs_official_same_scale"] = {
                recipe: score - same_scale[recipe]
                for recipe, score in means.items()
                if recipe in same_scale
            }
        local_target = means_by_scale_metric.get((target_scale, local_target_metrics[metric]))
        if scale != target_scale and local_target:
            comparison["local_target_scores"] = local_target
            comparison["pairwise_vs_local_target"] = pairwise_agreement(means, local_target)
            if len(by_seed) > 1:
                comparison["unanimous_seed_decisions_vs_local_target"] = seed_stability(
                    by_seed, local_target
                )
        comparisons.setdefault(scale, {})[metric] = comparison
    output = use_output_dir(output_dir)
    summary_path = write_summary(
        {
            "stage": 12,
            "method": "datadecide_local_olmes_reproduction",
            "config": str(config_path.resolve()),
            "runs": local_runs,
            "comparison": comparisons,
            "warnings": [
                "Official 1B matrix remains the target unless a local 1B run is supplied.",
                "OLMES-10 accuracy predicts 1B OLMES-10 accuracy; the three-task "
                "continuous proxy predicts 1B accuracy on the same three tasks.",
                "When local 1B runs are present, pairwise_vs_local_target is the fully "
                "local evaluator comparison.",
                "The continuous proxy excludes BoolQ and uses ARC-Easy, ARC-Challenge, and MMLU.",
                "Per-task comparisons are primary; the three-task mean is an operational "
                "summary only for a calibrated general-knowledge use case.",
            ],
        },
        output,
    )
    click.echo(summary_path)


@cli.command("prepare-recipe")
@click.option("--config", "config_path", type=click.Path(path_type=Path), required=True)
@click.option("--recipe", required=True)
@click.option("--named-mixes-source", type=click.Path(path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--workers", default=8, show_default=True)
def prepare_recipe(
    config_path: Path,
    recipe: str,
    named_mixes_source: Path,
    output: Path,
    workers: int,
) -> None:
    """Materialize a deterministic sample spread across an official logical recipe."""
    config = _config(config_path)
    if recipe not in config["local_training"]["recipes"]:
        raise click.BadParameter("recipe is not preregistered for local training")
    settings = config["local_training"]
    manifest = materialize_recipe(
        named_mixes_source=named_mixes_source,
        recipe=config["recipes"][recipe]["official_recipe"],
        output=output,
        target_tokens=int(settings["target_tokens"]),
        endpoint=str(settings["endpoint"]),
        repo_id=str(settings["data_repo"]),
        revision=str(settings["data_revision"]),
        chunks=int(settings["chunks_per_recipe"]),
        workers=workers,
    )
    click.echo(json.dumps(manifest, indent=2, sort_keys=True))


@cli.command("train-20m")
@click.option("--config", "config_path", type=click.Path(path_type=Path), required=True)
@click.option("--recipe", required=True)
@click.option("--tokens", "tokens_path", type=click.Path(path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--seed", type=int, required=True)
@click.option("--device", default="cuda:1", show_default=True)
@click.option("--micro-batch-size", default=4, show_default=True)
@click.option("--resume/--no-resume", default=True, show_default=True)
def train_command(
    config_path: Path,
    recipe: str,
    tokens_path: Path,
    output_dir: Path,
    seed: int,
    device: str,
    micro_batch_size: int,
    resume: bool,
) -> None:
    """Train one exact-architecture 20M proxy and save an OLMES-compatible checkpoint."""
    config = _config(config_path)
    settings = config["local_training"]
    if recipe not in settings["recipes"]:
        raise click.BadParameter("recipe is not preregistered for local training")
    if seed not in settings["seeds"]:
        raise click.BadParameter("seed is not preregistered for local training")
    summary = train_20m(
        tokens_path=tokens_path,
        tokenizer_path=settings["tokenizer_path"],
        output_dir=output_dir,
        seed=seed,
        device=device,
        steps=int(settings["steps"]),
        micro_batch_size=micro_batch_size,
        resume=resume,
    )
    click.echo(json.dumps(summary, indent=2, sort_keys=True))


@cli.command("export-state")
@click.option("--config", "config_path", type=click.Path(path_type=Path), required=True)
@click.option("--trainer-state", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
def export_state(config_path: Path, trainer_state: Path, output_dir: Path) -> None:
    """Freeze an intermediate trainer state for benchmark evaluation."""
    config = _config(config_path)
    summary = export_trainer_state(
        trainer_state_path=trainer_state,
        tokenizer_path=config["local_training"]["tokenizer_path"],
        output_dir=output_dir,
    )
    click.echo(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    cli()
