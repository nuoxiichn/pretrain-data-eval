"""Statistical aggregation for training-based proxy signals."""

from __future__ import annotations

import math
import random
import statistics
from typing import Mapping, Sequence


def _weighted_mean(values: Sequence[float], weights: Sequence[int]) -> float:
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must be non-empty and have equal length")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weights must have positive sum")
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total_weight


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _benjamini_hochberg(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return monotone Benjamini-Hochberg q-values keyed like the input."""
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank, (key, p_value) in reversed(list(enumerate(ordered, start=1))):
        running = min(running, p_value * count / rank)
        adjusted[key] = min(1.0, running)
    return adjusted


def _document_distribution(seed_values: Sequence[Sequence[float]]) -> dict:
    means = [sum(values) / len(values) for values in zip(*seed_values)]
    return {
        "documents": len(means),
        "p10": _quantile(means, 0.10),
        "median": _quantile(means, 0.50),
        "p90": _quantile(means, 0.90),
    }


def _stratified_stats(
    seed_values: Sequence[Sequence[float]],
    weights: Sequence[int],
    strata: Sequence[Mapping[str, str]],
    bootstrap_samples: int,
    confidence: float,
    bootstrap_seed: int,
) -> dict:
    if len(strata) != len(weights):
        raise ValueError("strata must contain one row per document")
    output: dict[str, dict] = {}
    for field in ("language", "source", "length_bucket"):
        by_value: dict[str, list[int]] = {}
        for index, row in enumerate(strata):
            value = str(row.get(field) or "unknown")
            by_value.setdefault(value, []).append(index)
        output[field] = {}
        for value_index, (value, indices) in enumerate(sorted(by_value.items())):
            selected_seed_values = [
                [seed_row[index] for index in indices] for seed_row in seed_values
            ]
            selected_weights = [weights[index] for index in indices]
            stats = hierarchical_weighted_stats(
                selected_seed_values,
                selected_weights,
                bootstrap_samples,
                confidence,
                bootstrap_seed + value_index,
            )
            stats["distribution"] = _document_distribution(selected_seed_values)
            stats["tokens"] = sum(selected_weights)
            output[field][value] = stats
    return output


def hierarchical_weighted_stats(
    seed_values: Sequence[Sequence[float]],
    weights: Sequence[int],
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict:
    """Cluster-bootstrap seeds and resample documents within each seed."""
    if not seed_values:
        raise ValueError("at least one seed is required")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if any(len(values) != len(weights) for values in seed_values):
        raise ValueError("every seed must have one value per document")

    seed_means = [_weighted_mean(values, weights) for values in seed_values]
    point = sum(seed_means) / len(seed_means)
    rng = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(bootstrap_samples):
        cluster_means: list[float] = []
        for _cluster in seed_values:
            cluster_index = rng.randrange(len(seed_values))
            values = seed_values[cluster_index]
            doc_indices = [rng.randrange(len(weights)) for _ in weights]
            sampled_values = [values[index] for index in doc_indices]
            sampled_weights = [weights[index] for index in doc_indices]
            cluster_means.append(_weighted_mean(sampled_values, sampled_weights))
        bootstrap.append(sum(cluster_means) / len(cluster_means))

    tail = (1.0 - confidence) / 2.0
    return {
        "mean": point,
        "ci_low": _quantile(bootstrap, tail),
        "ci_high": _quantile(bootstrap, 1.0 - tail),
        "confidence": confidence,
        "seed_means": seed_means,
        "seed_std": statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0,
        "bootstrap_samples": bootstrap_samples,
    }


def hierarchical_difference_stats(
    left_seed_values: Sequence[Sequence[float]],
    left_weights: Sequence[int],
    right_seed_values: Sequence[Sequence[float]],
    right_weights: Sequence[int],
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    null_margin: float = 0.0,
) -> dict:
    """Hierarchical bootstrap for two corpora with different documents."""
    if len(left_seed_values) != len(right_seed_values) or not left_seed_values:
        raise ValueError("left and right must have the same non-zero seed count")
    left_seed_means = [_weighted_mean(values, left_weights) for values in left_seed_values]
    right_seed_means = [_weighted_mean(values, right_weights) for values in right_seed_values]
    seed_differences = [
        left - right for left, right in zip(left_seed_means, right_seed_means, strict=True)
    ]
    rng = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(bootstrap_samples):
        differences: list[float] = []
        for _cluster in left_seed_values:
            cluster_index = rng.randrange(len(left_seed_values))
            left_values = left_seed_values[cluster_index]
            right_values = right_seed_values[cluster_index]
            left_indices = [rng.randrange(len(left_weights)) for _ in left_weights]
            right_indices = [rng.randrange(len(right_weights)) for _ in right_weights]
            left_mean = _weighted_mean(
                [left_values[index] for index in left_indices],
                [left_weights[index] for index in left_indices],
            )
            right_mean = _weighted_mean(
                [right_values[index] for index in right_indices],
                [right_weights[index] for index in right_indices],
            )
            differences.append(left_mean - right_mean)
        bootstrap.append(sum(differences) / len(differences))

    tail = (1.0 - confidence) / 2.0
    return {
        "mean": sum(seed_differences) / len(seed_differences),
        "ci_low": _quantile(bootstrap, tail),
        "ci_high": _quantile(bootstrap, 1.0 - tail),
        "confidence": confidence,
        "seed_differences": seed_differences,
        "seed_std": statistics.stdev(seed_differences)
        if len(seed_differences) > 1
        else 0.0,
        "bootstrap_samples": bootstrap_samples,
        "p_value_left_higher": (
            sum(value <= null_margin for value in bootstrap) + 1
        )
        / (bootstrap_samples + 1),
        "p_value_right_higher": (
            sum(value >= -null_margin for value in bootstrap) + 1
        )
        / (bootstrap_samples + 1),
    }


def analyze_conditioning(
    primary_runs: Mapping[str, Mapping[str, dict]],
    corpus_names: Sequence[str],
    checkpoints: Sequence[int],
    bootstrap_samples: int,
    confidence: float,
    min_effect_bits: float,
    stable_horizon_fraction: float,
    bootstrap_seed: int,
) -> dict:
    """Build cross-loss matrices, paired margins and stable dominance edges."""
    seeds = list(primary_runs)
    by_checkpoint: dict[str, dict] = {}
    pair_directions: dict[tuple[str, str], list[int]] = {
        (source, target): []
        for source in corpus_names
        for target in corpus_names
        if source != target
    }

    for checkpoint_index, checkpoint in enumerate(checkpoints):
        key = str(checkpoint)
        loss_matrix: dict[str, dict[str, float]] = {}
        margins: dict[str, dict[str, dict]] = {}
        for source in corpus_names:
            loss_matrix[source] = {}
            margins[source] = {}
            for target in corpus_names:
                source_evals = [
                    primary_runs[seed][source]["evaluations"][key][target] for seed in seeds
                ]
                target_evals = [
                    primary_runs[seed][target]["evaluations"][key][target] for seed in seeds
                ]
                loss_matrix[source][target] = sum(
                    result["mean_bits_per_token"] for result in source_evals
                ) / len(source_evals)
                seed_deltas = [
                    [
                        (source_loss - target_loss) / math.log(2.0)
                        for source_loss, target_loss in zip(
                            source_result["doc_loss_nats"],
                            target_result["doc_loss_nats"],
                            strict=True,
                        )
                    ]
                    for source_result, target_result in zip(
                        source_evals, target_evals, strict=True
                    )
                ]
                stats = hierarchical_weighted_stats(
                    seed_deltas,
                    source_evals[0]["doc_token_counts"],
                    bootstrap_samples,
                    confidence,
                    bootstrap_seed + checkpoint_index * 10_000,
                )
                stats["dominates"] = (
                    source != target and stats["ci_high"] < -min_effect_bits
                )
                stats["significant_reverse"] = (
                    source != target and stats["ci_low"] > min_effect_bits
                )
                margins[source][target] = stats
                if source != target:
                    direction = -1 if stats["dominates"] else 1 if stats["significant_reverse"] else 0
                    pair_directions[(source, target)].append(direction)
        by_checkpoint[key] = {"loss_matrix_bits_per_token": loss_matrix, "margins": margins}

    stable_edges: list[dict] = []
    pair_stability: dict[str, dict] = {}
    for (source, target), directions in pair_directions.items():
        dominance_fraction = directions.count(-1) / len(directions)
        reverse_fraction = directions.count(1) / len(directions)
        nonzero = [direction for direction in directions if direction]
        flips = sum(left != right for left, right in zip(nonzero, nonzero[1:]))
        stable = dominance_fraction >= stable_horizon_fraction and reverse_fraction == 0.0
        pair_key = f"{source}->{target}"
        pair_stability[pair_key] = {
            "directions": directions,
            "dominance_fraction": dominance_fraction,
            "reverse_fraction": reverse_fraction,
            "direction_flips": flips,
            "stable_dominates": stable,
        }
        if stable:
            stable_edges.append(
                {
                    "source": source,
                    "target": target,
                    "dominance_fraction": dominance_fraction,
                }
            )
    return {
        "definition": "loss(train=source, eval=target) - loss(train=target, eval=target)",
        "unit": "bits_per_token",
        "min_effect_bits": min_effect_bits,
        "stable_horizon_fraction": stable_horizon_fraction,
        "by_checkpoint": by_checkpoint,
        "pair_stability": pair_stability,
        "stable_edges": stable_edges,
    }


def analyze_scaling_gain(
    scale_runs: Mapping[str, Mapping[str, dict]],
    corpus_names: Sequence[str],
    final_checkpoint: int,
    bootstrap_samples: int,
    confidence: float,
    min_effect_bits: float,
    bootstrap_seed: int,
    document_strata: Mapping[str, Sequence[Mapping[str, str]]] | None = None,
    compatibility_gate: Mapping[str, Mapping[str, object]] | None = None,
    checkpoints: Sequence[int] | None = None,
    stable_horizon_fraction: float = 0.8,
    require_unanimous_seed_direction: bool = False,
    fdr_max: float = 0.05,
) -> dict:
    """Aggregate small-vs-large loss reductions and relative corpus ordering."""
    seeds = list(scale_runs)
    checkpoint = str(final_checkpoint)
    corpus_stats: dict[str, dict] = {}
    seed_gaps: dict[str, list[list[float]]] = {}
    corpus_weights: dict[str, list[int]] = {}

    for corpus_index, corpus in enumerate(corpus_names):
        small_evals = [scale_runs[seed]["small"]["evaluations"][checkpoint][corpus] for seed in seeds]
        large_evals = [scale_runs[seed]["large"]["evaluations"][checkpoint][corpus] for seed in seeds]
        gaps = [
            [
                (small_loss - large_loss) / math.log(2.0)
                for small_loss, large_loss in zip(
                    small_result["doc_loss_nats"],
                    large_result["doc_loss_nats"],
                    strict=True,
                )
            ]
            for small_result, large_result in zip(small_evals, large_evals, strict=True)
        ]
        weights = small_evals[0]["doc_token_counts"]
        stats = hierarchical_weighted_stats(
            gaps,
            weights,
            bootstrap_samples,
            confidence,
            bootstrap_seed + corpus_index * 10_000,
        )
        stats["distribution"] = _document_distribution(gaps)
        if document_strata is not None:
            stats["strata"] = _stratified_stats(
                gaps,
                weights,
                document_strata[corpus],
                bootstrap_samples,
                confidence,
                bootstrap_seed + 1_000_000 + corpus_index * 100_000,
            )
        if compatibility_gate is not None:
            stats["compatibility"] = dict(compatibility_gate[corpus])
        stats["ppl_ratio_from_mean"] = 2.0 ** stats["mean"]
        corpus_stats[corpus] = stats
        seed_gaps[corpus] = gaps
        corpus_weights[corpus] = weights

    if not 0.0 < fdr_max <= 1.0:
        raise ValueError("fdr_max must be in (0, 1]")
    comparisons: dict[str, dict] = {}
    comparison_index = 0
    for left in corpus_names:
        for right in corpus_names:
            if left == right:
                continue
            gate_reason: list[str] = []
            shared_groups: list[str] = []
            if compatibility_gate is not None:
                left_gate = compatibility_gate[left]
                right_gate = compatibility_gate[right]
                if left_gate.get("decision") != "comparable":
                    gate_reason.append(f"{left}:compatibility_abstain")
                if right_gate.get("decision") != "comparable":
                    gate_reason.append(f"{right}:compatibility_abstain")
                left_groups = {str(value) for value in left_gate.get("comparison_groups", [])}
                right_groups = {str(value) for value in right_gate.get("comparison_groups", [])}
                shared_groups = sorted(left_groups & right_groups)
                if not shared_groups:
                    gate_reason.append("different_purpose")
            allowed = not gate_reason
            if not allowed:
                comparisons[f"{left}>{right}"] = {
                    "status": "abstain",
                    "gate_reasons": gate_reason,
                    "comparison_groups": shared_groups,
                    "left_higher": False,
                    "significant_reverse": False,
                }
                comparison_index += 1
                continue
            stats = hierarchical_difference_stats(
                seed_gaps[left],
                corpus_weights[left],
                seed_gaps[right],
                corpus_weights[right],
                bootstrap_samples,
                confidence,
                bootstrap_seed + 100_000 + comparison_index,
                min_effect_bits,
            )
            stats["status"] = "comparable"
            stats["gate_reasons"] = []
            stats["comparison_groups"] = shared_groups
            seed_differences = stats["seed_differences"]
            stats["seed_positive_fraction"] = sum(
                difference > min_effect_bits for difference in seed_differences
            ) / len(seed_differences)
            stats["seed_unanimous_positive"] = all(
                difference > min_effect_bits for difference in seed_differences
            )
            stats["seed_unanimous_negative"] = all(
                difference < -min_effect_bits for difference in seed_differences
            )
            seed_gate = (
                not require_unanimous_seed_direction
                or stats["seed_unanimous_positive"]
            )
            reverse_seed_gate = (
                not require_unanimous_seed_direction
                or stats["seed_unanimous_negative"]
            )
            stats["uncorrected_left_higher"] = (
                stats["ci_low"] > min_effect_bits and seed_gate
            )
            stats["uncorrected_significant_reverse"] = (
                stats["ci_high"] < -min_effect_bits and reverse_seed_gate
            )
            stats["left_higher"] = False
            stats["significant_reverse"] = False
            comparisons[f"{left}>{right}"] = stats
            comparison_index += 1

    q_values = _benjamini_hochberg(
        {
            key: stats["p_value_left_higher"]
            for key, stats in comparisons.items()
            if stats["status"] == "comparable"
        }
    )
    ordering_edges: list[dict] = []
    for key, stats in comparisons.items():
        if stats["status"] != "comparable":
            continue
        left, right = key.split(">", 1)
        stats["q_value_left_higher"] = q_values[key]
        reverse_key = f"{right}>{left}"
        stats["q_value_right_higher"] = q_values[reverse_key]
        stats["left_higher"] = (
            stats["uncorrected_left_higher"] and q_values[key] <= fdr_max
        )
        stats["significant_reverse"] = (
            stats["uncorrected_significant_reverse"]
            and q_values[reverse_key] <= fdr_max
        )
        if stats["left_higher"]:
            ordering_edges.append(
                {
                    "higher_scaling_gain": left,
                    "lower_scaling_gain": right,
                    "mean_difference_bits": stats["mean"],
                    "q_value": q_values[key],
                    "comparison_groups": stats["comparison_groups"],
                }
            )

    result = {
        "definition": "CE(scale_small) - CE(scale_large)",
        "unit": "bits_per_token",
        "min_effect_bits": min_effect_bits,
        "fdr_method": "benjamini_hochberg",
        "fdr_max": fdr_max,
        "checkpoint": final_checkpoint,
        "by_corpus": corpus_stats,
        "comparisons": comparisons,
        "ordering_edges": ordering_edges,
    }
    if checkpoints is None:
        return result

    selected_checkpoints = sorted(set(int(value) for value in checkpoints))
    if not selected_checkpoints or selected_checkpoints[-1] != final_checkpoint:
        raise ValueError("scaling checkpoints must be non-empty and end at final_checkpoint")
    by_checkpoint = {
        str(checkpoint_value): analyze_scaling_gain(
            scale_runs,
            corpus_names,
            checkpoint_value,
            bootstrap_samples,
            confidence,
            min_effect_bits,
            bootstrap_seed + checkpoint_index * 1_000_000,
            document_strata=None,
            compatibility_gate=compatibility_gate,
            checkpoints=None,
            require_unanimous_seed_direction=require_unanimous_seed_direction,
            fdr_max=fdr_max,
        )
        for checkpoint_index, checkpoint_value in enumerate(selected_checkpoints)
    }
    pair_stability: dict[str, dict] = {}
    stable_edges: list[dict] = []
    for left in corpus_names:
        for right in corpus_names:
            if left == right:
                continue
            key = f"{left}>{right}"
            points = [
                by_checkpoint[str(checkpoint_value)]["comparisons"][key]
                for checkpoint_value in selected_checkpoints
            ]
            comparable = [point for point in points if point["status"] == "comparable"]
            dominance_fraction = (
                sum(point["left_higher"] for point in comparable) / len(comparable)
                if comparable
                else 0.0
            )
            reverse_fraction = (
                sum(point["significant_reverse"] for point in comparable) / len(comparable)
                if comparable
                else 0.0
            )
            final_point = points[-1]
            stable = (
                len(comparable) == len(points)
                and dominance_fraction >= stable_horizon_fraction
                and reverse_fraction == 0.0
                and final_point["left_higher"]
            )
            pair_stability[key] = {
                "status": "stable_higher" if stable else "abstain",
                "dominance_fraction": dominance_fraction,
                "reverse_fraction": reverse_fraction,
                "comparable_horizons": len(comparable),
                "total_horizons": len(points),
                "require_unanimous_seed_direction": require_unanimous_seed_direction,
                "final_mean_difference_bits": final_point.get("mean"),
            }
            if stable:
                stable_edges.append(
                    {
                        "higher_scaling_gain": left,
                        "lower_scaling_gain": right,
                        "final_mean_difference_bits": final_point["mean"],
                        "dominance_fraction": dominance_fraction,
                        "comparison_groups": final_point["comparison_groups"],
                    }
                )
    result["checkpoints"] = selected_checkpoints
    result["stable_horizon_fraction"] = stable_horizon_fraction
    result["require_unanimous_seed_direction"] = require_unanimous_seed_direction
    result["by_checkpoint"] = by_checkpoint
    result["pair_stability"] = pair_stability
    result["stable_edges"] = stable_edges
    return result


def mean_document_scaling_scores(
    scale_runs: Mapping[str, Mapping[str, dict]],
    corpus: str,
    final_checkpoint: int,
) -> tuple[list[float], list[float], list[float], list[int]]:
    """Return seed-mean small loss, large loss and scaling gain per document."""
    checkpoint = str(final_checkpoint)
    small_by_seed: list[list[float]] = []
    large_by_seed: list[list[float]] = []
    token_counts: list[int] | None = None
    for run in scale_runs.values():
        small = run["small"]["evaluations"][checkpoint][corpus]
        large = run["large"]["evaluations"][checkpoint][corpus]
        small_by_seed.append(small["doc_loss_nats"])
        large_by_seed.append(large["doc_loss_nats"])
        token_counts = small["doc_token_counts"]
    if token_counts is None:
        raise ValueError("scale runs are empty")
    small_mean = [sum(values) / len(values) for values in zip(*small_by_seed)]
    large_mean = [sum(values) / len(values) for values in zip(*large_by_seed)]
    gaps = [
        (small - large) / math.log(2.0)
        for small, large in zip(small_mean, large_mean, strict=True)
    ]
    return small_mean, large_mean, gaps, token_counts
