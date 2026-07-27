from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from research.data_advisor.benchmark_matrix import (
    _agreement_on_decisive_pairs,
    benjamini_hochberg,
    kendall_tau_b,
    paired_bootstrap_interval,
    paired_sign_flip_pvalue,
    target_resolution,
)
from research.data_advisor.decision import pairwise_agreement, seed_stability
from research.data_advisor.multiple_choice import (
    ModelBlobChecksumError,
    MultipleChoiceExample,
    _resolve_snapshot,
    _verify_model_blob_checksums,
    aggregate_predictions,
    build_prompt,
    choice_metrics,
    deterministic_sample,
)


def test_pairwise_agreement_counts_reverse_and_tie() -> None:
    result = pairwise_agreement(
        {"a": 3.0, "b": 1.0, "c": 1.0},
        {"a": 1.0, "b": 2.0, "c": 3.0},
    )
    assert result["pair_count"] == 3
    assert result["incorrect"] == 2
    assert result["proxy_ties"] == 1


def test_seed_stability_abstains_on_disagreement() -> None:
    result = seed_stability(
        {
            "1": {"a": 0.6, "b": 0.4, "c": 0.2},
            "2": {"a": 0.3, "b": 0.4, "c": 0.1},
            "3": {"a": 0.7, "b": 0.5, "c": 0.3},
        },
        {"a": 0.8, "b": 0.6, "c": 0.4},
    )
    assert result["pair_count"] == 3
    assert result["correct"] == 2
    assert result["abstained"] == 1
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["conditional_accuracy"] == 1.0


def test_multiple_choice_continuous_metrics_and_tie_credit() -> None:
    metrics = choice_metrics([-4.0, -3.0, -2.0], [2, 2, 2], [4, 4, 4], gold=2)
    assert metrics["accuracy"] == 1.0
    assert metrics["correct_prob_per_char"] == pytest.approx(np.exp(-0.5))
    assert metrics["correct_vs_best_incorrect_logprob_margin"] == pytest.approx(0.25)
    assert 0.0 < metrics["normalized_choice_probability"] < 1.0

    tied = choice_metrics([-2.0, -2.0], [1, 1], [2, 2], gold=0)
    assert tied["accuracy"] == 0.5
    assert tied["choice_tie"] == 1.0


def test_multiple_choice_sampling_and_macro_aggregation_are_deterministic() -> None:
    examples = [
        MultipleChoiceExample(str(index), "g", f"q{index}", ("a", "b"), index % 2)
        for index in range(20)
    ]
    assert deterministic_sample(examples, 5, 7) == deterministic_sample(examples, 5, 7)
    assert deterministic_sample(examples, 5, 7) != deterministic_sample(examples, 5, 8)
    assert build_prompt("  question? ") == "Question: question?\nAnswer:"

    predictions = [
        {
            "group": "large",
            "accuracy": 1.0,
            "correct_prob_per_char": 0.8,
            "normalized_choice_probability": 0.7,
            "correct_vs_best_incorrect_logprob_margin": 0.2,
            "correct_logprob_per_token": -1.0,
            "choice_tie": 0.0,
        },
        {
            "group": "large",
            "accuracy": 1.0,
            "correct_prob_per_char": 0.8,
            "normalized_choice_probability": 0.7,
            "correct_vs_best_incorrect_logprob_margin": 0.2,
            "correct_logprob_per_token": -1.0,
            "choice_tie": 0.0,
        },
        {
            "group": "small",
            "accuracy": 0.0,
            "correct_prob_per_char": 0.2,
            "normalized_choice_probability": 0.3,
            "correct_vs_best_incorrect_logprob_margin": -0.2,
            "correct_logprob_per_token": -2.0,
            "choice_tie": 0.0,
        },
    ]
    aggregate = aggregate_predictions(predictions, aggregation="macro_by_group")
    assert aggregate["micro"]["accuracy"] == pytest.approx(2 / 3)
    assert aggregate["primary"]["accuracy"] == pytest.approx(0.5)


def test_target_bootstrap_and_decisive_pair_agreement() -> None:
    low, high = paired_bootstrap_interval(np.ones(20), samples=500, seed=7)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)
    target = {
        "pairs": [
            {"left": "a", "right": "b", "delta": 0.1, "decisive": True, "direction": 1},
            {"left": "a", "right": "c", "delta": 0.0, "decisive": False, "direction": 0},
        ]
    }
    agreement = _agreement_on_decisive_pairs({"a": 0.4, "b": 0.3, "c": 0.5}, target)
    assert agreement["pair_count"] == 1
    assert agreement["accuracy"] == 1.0


def test_pairwise_sign_flip_fdr_and_kendall_sensitivity() -> None:
    assert paired_sign_flip_pvalue(np.zeros(8), samples=500, seed=7) == 1.0
    assert paired_sign_flip_pvalue(np.ones(20), samples=2000, seed=7) < 0.01
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    assert kendall_tau_b({"a": 3, "b": 2, "c": 1}, {"a": 4, "b": 2, "c": 0}) == 1.0
    assert kendall_tau_b({"a": 3, "b": 2, "c": 1}, {"a": 0, "b": 2, "c": 4}) == -1.0


def test_target_resolution_requires_lift_over_random_floor(tmp_path: Path) -> None:
    target_runs = {}
    for recipe, correct in (("a", [1.0, 0.0, 0.0, 0.0]), ("b", [0.0, 1.0, 0.0, 0.0])):
        predictions = tmp_path / f"{recipe}.jsonl"
        predictions.write_text(
            "".join(
                json.dumps(
                    {
                        "example_id": str(index),
                        "group": "all",
                        "choice_count": 4,
                        "accuracy": accuracy,
                    }
                )
                + "\n"
                for index, accuracy in enumerate(correct)
            ),
            encoding="utf-8",
        )
        target_runs[recipe] = {
            "predictions_path": str(predictions),
            "aggregate": {
                "aggregation": "micro",
                "primary": {"accuracy": float(np.mean(correct))},
            },
        }
    result = target_resolution(target_runs, bootstrap_samples=200, seed=7)
    assert result["random_choice_baseline"] == 0.25
    assert result["normalized_lift_over_random"] == 0.0
    assert result["clears_random_floor"] is False
    assert result["screening_outcome"] == "needs_larger_target"


def test_hub_model_blob_checksum_validation(tmp_path: Path) -> None:
    blob = tmp_path / "blob"
    blob.write_bytes(b"valid model bytes")
    digest = hashlib.sha256(blob.read_bytes()).hexdigest()
    hashed_blob = tmp_path / digest
    blob.rename(hashed_blob)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").symlink_to(hashed_blob)
    assert _verify_model_blob_checksums(snapshot) == {"model.safetensors": digest}

    hashed_blob.write_bytes(b"corrupted")
    with pytest.raises(ModelBlobChecksumError):
        _verify_model_blob_checksums(snapshot)


def test_remote_snapshot_disables_hf_transfer_and_serializes_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    blob = tmp_path / "blob"
    blob.write_bytes(b"valid model bytes")
    digest = hashlib.sha256(blob.read_bytes()).hexdigest()
    hashed_blob = tmp_path / digest
    blob.rename(hashed_blob)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").symlink_to(hashed_blob)
    kwargs = {}

    def fake_snapshot_download(**values):
        kwargs.update(values)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_ENABLE_HF_TRANSFER", True, raising=False)
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "1")

    resolved, checksums = _resolve_snapshot("org/remote-model", None, tmp_path / "cache")
    assert resolved == snapshot.resolve()
    assert checksums == {"model.safetensors": digest}
    assert kwargs["max_workers"] == 1
    assert "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ
    assert huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER is False
