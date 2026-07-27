"""Cross-scale analysis for the preregistered multiple-choice proxy matrix."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROXY_METRICS = (
    "correct_prob_per_char",
    "normalized_choice_probability",
    "correct_vs_best_incorrect_logprob_margin",
    "accuracy",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_matrix_runs(
    matrix_dir: str | Path,
    *,
    scales: Sequence[str],
    recipes: Sequence[str],
    benchmarks: Sequence[str],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    matrix_dir = Path(matrix_dir)
    runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for scale in scales:
        for recipe in recipes:
            for benchmark in benchmarks:
                path = matrix_dir / scale / recipe / benchmark / "summary.json"
                if not path.exists():
                    continue
                summary = _load_json(path)
                if summary.get("protocol") != "rc-zero-shot-v1":
                    raise ValueError(f"unexpected protocol in {path}")
                runs[(scale, recipe, benchmark)] = summary
    return runs


def _load_predictions(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path = Path(str(summary["predictions_path"]))
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[str(row["example_id"])] = row
    return rows


def _paired_units(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    *,
    metric: str,
    aggregation: str,
) -> np.ndarray:
    shared = sorted(set(left) & set(right))
    if set(shared) != set(left) or set(shared) != set(right):
        raise ValueError("paired benchmark runs do not contain identical example IDs")
    if aggregation == "micro":
        return np.asarray(
            [float(left[key][metric]) - float(right[key][metric]) for key in shared],
            dtype=np.float64,
        )
    if aggregation != "macro_by_group":
        raise ValueError(f"unknown aggregation {aggregation!r}")
    by_group: dict[str, list[float]] = {}
    for key in shared:
        group = str(left[key]["group"])
        if str(right[key]["group"]) != group:
            raise ValueError(f"group mismatch for {key}")
        by_group.setdefault(group, []).append(float(left[key][metric]) - float(right[key][metric]))
    return np.asarray(
        [float(np.mean(by_group[group])) for group in sorted(by_group)], dtype=np.float64
    )


def paired_bootstrap_interval(
    units: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if units.ndim != 1 or not len(units):
        raise ValueError("bootstrap units must be a non-empty vector")
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = 256
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = generator.integers(0, len(units), size=(count, len(units)))
        means[start : start + count] = units[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_sign_flip_pvalue(
    units: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> float:
    """Monte Carlo paired sign-flip p-value for a zero mean difference."""
    if units.ndim != 1 or not len(units):
        raise ValueError("sign-flip units must be a non-empty vector")
    observed = abs(float(units.mean()))
    if observed == 0.0:
        return 1.0
    generator = np.random.default_rng(seed)
    extreme = 0
    drawn = 0
    chunk = 256
    while drawn < samples:
        count = min(chunk, samples - drawn)
        signs = generator.integers(0, 2, size=(count, len(units)), dtype=np.int8) * 2 - 1
        null_means = (signs * units).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(null_means) >= observed - 1e-15))
        drawn += count
    return (extreme + 1.0) / (samples + 1.0)


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return monotone Benjamini-Hochberg adjusted p-values."""
    count = len(p_values)
    if not count:
        return []
    values = np.asarray(p_values, dtype=np.float64)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be between zero and one")
    order = np.argsort(values)
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, float(values[index]) * count / rank)
        adjusted[index] = running
    return [float(value) for value in adjusted]


def kendall_tau_b(left: Mapping[str, float], right: Mapping[str, float]) -> float | None:
    """Compute Kendall tau-b on shared recipes without an optional scipy dependency."""
    shared = sorted(set(left) & set(right))
    if len(shared) < 2:
        return None
    concordant = 0
    discordant = 0
    left_only_ties = 0
    right_only_ties = 0
    for first, second in combinations(shared, 2):
        left_direction = int(left[first] > left[second]) - int(left[first] < left[second])
        right_direction = int(right[first] > right[second]) - int(right[first] < right[second])
        if left_direction == 0 and right_direction == 0:
            continue
        if left_direction == 0:
            left_only_ties += 1
        elif right_direction == 0:
            right_only_ties += 1
        elif left_direction == right_direction:
            concordant += 1
        else:
            discordant += 1
    denominator = np.sqrt(
        (concordant + discordant + left_only_ties) * (concordant + discordant + right_only_ties)
    )
    if denominator == 0.0:
        return None
    return float((concordant - discordant) / denominator)


def target_resolution(
    target_runs: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
    minimum_decisive_pairs: int = 20,
    minimum_normalized_lift: float = 0.05,
) -> dict[str, Any]:
    recipes = sorted(target_runs)
    prediction_cache = {
        recipe: _load_predictions(summary) for recipe, summary in target_runs.items()
    }
    scores = {
        recipe: float(summary["aggregate"]["primary"]["accuracy"])
        for recipe, summary in target_runs.items()
    }
    pairs = []
    for pair_index, (left, right) in enumerate(combinations(recipes, 2)):
        aggregation = str(target_runs[left]["aggregate"]["aggregation"])
        if str(target_runs[right]["aggregate"]["aggregation"]) != aggregation:
            raise ValueError("target aggregation mismatch")
        units = _paired_units(
            prediction_cache[left],
            prediction_cache[right],
            metric="accuracy",
            aggregation=aggregation,
        )
        low, high = paired_bootstrap_interval(
            units,
            samples=bootstrap_samples,
            seed=seed + pair_index,
        )
        p_value = paired_sign_flip_pvalue(
            units,
            samples=bootstrap_samples,
            seed=seed + 100_000 + pair_index,
        )
        delta = scores[left] - scores[right]
        is_decisive = low > 0.0 or high < 0.0
        pairs.append(
            {
                "left": left,
                "right": right,
                "delta": delta,
                "bootstrap_95ci": [low, high],
                "bootstrap_unit_count": len(units),
                "sign_flip_p_value": p_value,
                "raw_decisive": is_decisive,
                "decisive": is_decisive,
                "direction": int(delta > 0.0) - int(delta < 0.0) if is_decisive else 0,
            }
        )
    q_values = benjamini_hochberg([float(pair["sign_flip_p_value"]) for pair in pairs])
    for pair, q_value in zip(pairs, q_values):
        pair["fdr_q_value"] = q_value
        pair["fdr_decisive"] = bool(pair["raw_decisive"] and q_value <= 0.05)
    decisive = sum(int(pair["raw_decisive"]) for pair in pairs)
    fdr_decisive = sum(int(pair["fdr_decisive"]) for pair in pairs)
    total = len(pairs)
    chance = []
    first_predictions = next(iter(prediction_cache.values()))
    for row in first_predictions.values():
        chance.append(
            {
                "group": str(row["group"]),
                "value": 1.0 / int(row["choice_count"]),
            }
        )
    aggregation = str(next(iter(target_runs.values()))["aggregate"]["aggregation"])
    if aggregation == "micro":
        random_choice_baseline = float(np.mean([row["value"] for row in chance]))
    else:
        by_group: dict[str, list[float]] = {}
        for row in chance:
            by_group.setdefault(str(row["group"]), []).append(float(row["value"]))
        random_choice_baseline = float(np.mean([np.mean(values) for values in by_group.values()]))
    score_mean = float(np.mean(list(scores.values())))
    normalized_lift = (score_mean - random_choice_baseline) / (1.0 - random_choice_baseline)
    clears_random_floor = normalized_lift >= minimum_normalized_lift
    return {
        "recipe_count": len(recipes),
        "pair_count": total,
        "decisive_pair_count": decisive,
        "decisive_pair_coverage": decisive / total if total else 0.0,
        "fdr_decisive_pair_count": fdr_decisive,
        "fdr_decisive_pair_coverage": fdr_decisive / total if total else 0.0,
        "minimum_decisive_pairs": minimum_decisive_pairs,
        "screening_outcome": (
            "screenable_single_seed"
            if decisive >= minimum_decisive_pairs and clears_random_floor
            else "needs_larger_target"
        ),
        "fdr_sensitivity_outcome": (
            "screenable_single_seed"
            if fdr_decisive >= minimum_decisive_pairs and clears_random_floor
            else "needs_larger_target"
        ),
        "scores": scores,
        "score_mean": score_mean,
        "score_std": float(np.std(list(scores.values()))),
        "score_range": max(scores.values()) - min(scores.values()),
        "random_choice_baseline": random_choice_baseline,
        "normalized_lift_over_random": normalized_lift,
        "minimum_normalized_lift_over_random": minimum_normalized_lift,
        "clears_random_floor": clears_random_floor,
        "pairs": pairs,
        "caveat": (
            "Bootstrap measures benchmark-sample uncertainty only; public checkpoints expose one "
            "training seed, so this cannot establish target training-seed stability."
        ),
    }


def _agreement_on_decisive_pairs(
    proxy_scores: Mapping[str, float],
    target: Mapping[str, Any],
    *,
    decisive_field: str = "decisive",
) -> dict[str, Any]:
    pairs = []
    correct = 0
    ties = 0
    for target_pair in target["pairs"]:
        if not target_pair[decisive_field]:
            continue
        left = str(target_pair["left"])
        right = str(target_pair["right"])
        if left not in proxy_scores or right not in proxy_scores:
            continue
        delta = float(proxy_scores[left]) - float(proxy_scores[right])
        direction = int(delta > 0.0) - int(delta < 0.0)
        agrees = direction == int(target_pair["direction"])
        correct += int(agrees)
        ties += int(direction == 0)
        pairs.append(
            {
                "left": left,
                "right": right,
                "proxy_delta": delta,
                "target_delta": target_pair["delta"],
                "agrees": agrees,
                "proxy_tie": direction == 0,
            }
        )
    count = len(pairs)
    return {
        "pair_count": count,
        "correct": correct,
        "proxy_ties": ties,
        "accuracy": correct / count if count else None,
        "pairs": pairs,
    }


def analyze_benchmark_matrix(
    runs: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    scales: Sequence[str],
    target_scale: str,
    recipes: Sequence[str],
    benchmarks: Sequence[str],
    bootstrap_samples: int = 2000,
    seed: int = 6198,
    minimum_decisive_pairs: int = 20,
    minimum_target_normalized_lift: float = 0.05,
) -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    for benchmark in benchmarks:
        target_runs = {
            recipe: runs[(target_scale, recipe, benchmark)]
            for recipe in recipes
            if (target_scale, recipe, benchmark) in runs
        }
        if len(target_runs) < 2:
            analysis[benchmark] = {
                "status": "incomplete_target",
                "target_recipe_count": len(target_runs),
            }
            continue
        target = target_resolution(
            target_runs,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
            minimum_decisive_pairs=minimum_decisive_pairs,
            minimum_normalized_lift=minimum_target_normalized_lift,
        )
        proxy: dict[str, Any] = {}
        for scale in scales:
            if scale == target_scale:
                continue
            scale_runs = {
                recipe: runs[(scale, recipe, benchmark)]
                for recipe in recipes
                if (scale, recipe, benchmark) in runs
            }
            if len(scale_runs) < 2:
                continue
            metric_results = {}
            for metric in PROXY_METRICS:
                scores = {
                    recipe: float(summary["aggregate"]["primary"][metric])
                    for recipe, summary in scale_runs.items()
                }
                metric_results[metric] = {
                    "scores": scores,
                    "kendall_tau_b_vs_target_accuracy": kendall_tau_b(scores, target["scores"]),
                    "agreement_on_decisive_target_pairs": _agreement_on_decisive_pairs(
                        scores, target
                    ),
                    "agreement_on_fdr_decisive_target_pairs": _agreement_on_decisive_pairs(
                        scores, target, decisive_field="fdr_decisive"
                    ),
                }
            proxy[scale] = {
                "recipe_count": len(scale_runs),
                "metrics": metric_results,
            }
        analysis[benchmark] = {
            "status": "analyzed",
            "target_scale": target_scale,
            "target_resolution": target,
            "proxy": proxy,
        }
    expected = len(scales) * len(recipes) * len(benchmarks)
    return {
        "scales": list(scales),
        "target_scale": target_scale,
        "recipes": list(recipes),
        "benchmarks": list(benchmarks),
        "expected_run_count": expected,
        "observed_run_count": len(runs),
        "complete": len(runs) == expected,
        "bootstrap_samples": bootstrap_samples,
        "analysis": analysis,
    }
