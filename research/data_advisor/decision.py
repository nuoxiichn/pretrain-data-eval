"""Pairwise recommendation metrics with an explicit abstention path."""

from __future__ import annotations

from typing import Any, Mapping


def pairwise_agreement(
    proxy: Mapping[str, float],
    target: Mapping[str, float],
    *,
    min_target_gap: float = 0.0,
) -> dict[str, Any]:
    """Compare shared pairs, excluding target ties from the decision set."""

    names = sorted(set(proxy) & set(target))
    correct = 0
    incorrect = 0
    proxy_ties = 0
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            target_delta = float(target[left]) - float(target[right])
            if abs(target_delta) <= min_target_gap:
                continue
            proxy_delta = float(proxy[left]) - float(proxy[right])
            agrees = proxy_delta * target_delta > 0.0
            tied = proxy_delta == 0.0
            correct += int(agrees)
            incorrect += int(not agrees and not tied)
            proxy_ties += int(tied)
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
    count = len(pairs)
    return {
        "recipes": names,
        "pair_count": count,
        "correct": correct,
        "incorrect": incorrect,
        "proxy_ties": proxy_ties,
        "accuracy": correct / count if count else None,
        "pairs": pairs,
    }


def seed_stability(
    proxy_by_seed: Mapping[str, Mapping[str, float]],
    target: Mapping[str, float],
) -> dict[str, Any]:
    """Decide only when every available seed gives the same non-tied direction."""

    shared_proxy = (
        set.intersection(*(set(scores) for scores in proxy_by_seed.values()))
        if proxy_by_seed
        else set()
    )
    names = sorted(shared_proxy & set(target))
    pairs: list[dict[str, Any]] = []
    correct = 0
    incorrect = 0
    abstained = 0
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            target_delta = float(target[left]) - float(target[right])
            if target_delta == 0.0:
                continue
            margins = {
                seed: float(scores[left]) - float(scores[right])
                for seed, scores in sorted(proxy_by_seed.items())
            }
            directions = {
                0 if value == 0.0 else (1 if value > 0.0 else -1)
                for value in margins.values()
            }
            unanimous = len(directions) == 1 and 0 not in directions
            agrees = unanimous and next(iter(directions)) == (1 if target_delta > 0.0 else -1)
            correct += int(agrees)
            incorrect += int(unanimous and not agrees)
            abstained += int(not unanimous)
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "target_delta": target_delta,
                    "proxy_deltas_by_seed": margins,
                    "unanimous_non_tied": unanimous,
                    "agrees": agrees if unanimous else None,
                }
            )
    count = len(pairs)
    decided = correct + incorrect
    return {
        "seeds": sorted(proxy_by_seed),
        "recipes": names,
        "pair_count": count,
        "correct": correct,
        "incorrect": incorrect,
        "abstained": abstained,
        "coverage": decided / count if count else None,
        "conditional_accuracy": correct / decided if decided else None,
        "pairs": pairs,
    }
