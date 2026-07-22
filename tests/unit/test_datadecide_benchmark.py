from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from stages.datadecide.matrix import (
    aggregate_tasks,
    all_recipe_pairwise_summary,
    load_final_scores,
    load_olmes_run,
    official_summary,
    pairwise_agreement,
    seed_stability,
)
from stages.datadecide import prepare
from stages.datadecide.prepare import _download_range, _selected_paths, load_official_paths
from stages.datadecide.train import DEFAULT_TOKENS, ShuffledTokenBlocks


def _metrics(value: float) -> str:
    return json.dumps({"primary_metric": value, "correct_prob_per_char": value})


def test_official_matrix_uses_last_positive_token_checkpoint(tmp_path: Path) -> None:
    rows = []
    for scale, recipe, seed, score in (
        ("20M", "A", "default", 0.4),
        ("20M", "B", "default", 0.3),
        ("1B", "A", "default", 0.7),
        ("1B", "B", "default", 0.5),
    ):
        rows.extend(
            [
                {
                    "params": scale,
                    "data": recipe,
                    "task": "olmes_10_macro_avg",
                    "step": 1,
                    "seed": seed,
                    "tokens": 10,
                    "metrics": _metrics(score - 0.1),
                },
                {
                    "params": scale,
                    "data": recipe,
                    "task": "olmes_10_macro_avg",
                    "step": 2,
                    "seed": seed,
                    "tokens": 20,
                    "metrics": _metrics(score),
                },
            ]
        )
    path = tmp_path / "results.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    records = load_final_scores(
        path,
        tasks_and_metrics={"olmes_10_macro_avg": "primary_metric"},
    )
    assert len(records) == 4
    assert all(row["step"] == 2 for row in records)
    config = {
        "official": {
            "target_scale": "1B",
            "target_task": "olmes_10_macro_avg",
            "proxy_scales": ["20M"],
            "tasks_and_metrics": {"olmes_10_macro_avg": "primary_metric"},
        },
        "recipes": {
            "a": {"official_name": "A"},
            "b": {"official_name": "B"},
        },
        "edges": [{"name": "a_over_b", "winner": "a", "loser": "b"}],
    }
    summary = official_summary(records, config)
    assert summary["edges"][0]["target_delta"] == pytest.approx(0.2)
    assert summary["edges"][0]["proxy"]["20M"]["agrees"] is True


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
    assert result["incorrect"] == 0
    assert result["abstained"] == 1
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["conditional_accuracy"] == 1.0


def test_load_olmes_run_reconstructs_continuous_likelihood(tmp_path: Path) -> None:
    metrics = {
        "task_name": "arc_easy",
        "num_instances": 1,
        "metrics": {"primary_score": 0.5, "acc_per_char": 0.5},
    }
    (tmp_path / "task-000-arc_easy-metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    prediction = {
        "label": 1,
        "model_output": [
            {"logits_per_char": np.log(0.2)},
            {"logits_per_char": np.log(0.4)},
        ],
    }
    (tmp_path / "task-000-arc_easy-predictions.jsonl").write_text(
        json.dumps(prediction) + "\n", encoding="utf-8"
    )
    result = load_olmes_run(tmp_path)
    assert result["tasks"]["arc_easy"]["correct_prob_per_char"] == pytest.approx(0.4)
    assert result["tasks"]["arc_easy"]["total_prob_per_char"] == pytest.approx(0.6)


def test_mmlu_subjects_are_macro_averaged_before_olmes() -> None:
    tasks = {
        task: {"primary_score": 0.5, "correct_prob_per_char": 0.2}
        for task in (
            "arc_challenge",
            "arc_easy",
            "boolq",
            "csqa",
            "hellaswag",
            "openbookqa",
            "piqa",
            "socialiqa",
            "winogrande",
        )
    }
    tasks["mmlu_math"] = {"primary_score": 0.2, "correct_prob_per_char": 0.1}
    tasks["mmlu_history"] = {"primary_score": 0.8, "correct_prob_per_char": 0.3}
    result = aggregate_tasks(tasks)
    assert result["by_family"]["mmlu"]["primary_score"] == pytest.approx(0.5)
    assert result["olmes_10_macro_avg"] == pytest.approx(0.5)
    assert result["decision_proxy_correct_prob_per_char"] == pytest.approx(0.2)
    assert result["decision_target_primary_score"] == pytest.approx(0.5)


def test_official_targets_keep_olmes_and_continuous_metrics_separate() -> None:
    records = []
    for scale, recipe, olmes, continuous in (
        ("20M", "A", 0.4, 0.2),
        ("20M", "B", 0.3, 0.3),
        ("1B", "A", 0.7, 0.6),
        ("1B", "B", 0.5, 0.4),
    ):
        records.append(
            {
                "scale": scale,
                "recipe": recipe,
                "task": "olmes_10_macro_avg",
                "seed": "default",
                "score": olmes,
            }
        )
        for task in ("arc_easy", "arc_challenge", "mmlu"):
            records.append(
                {
                    "scale": scale,
                    "recipe": recipe,
                    "task": task,
                    "seed": "default",
                    "score": continuous,
                }
            )
    config = {
        "official": {
            "target_scale": "1B",
            "target_task": "olmes_10_macro_avg",
            "proxy_scales": ["20M"],
            "tasks_and_metrics": {
                "olmes_10_macro_avg": "primary_metric",
                "arc_easy": "correct_prob_per_char",
                "arc_challenge": "correct_prob_per_char",
                "mmlu": "correct_prob_per_char",
            },
        },
        "recipes": {
            "a": {"official_name": "A"},
            "b": {"official_name": "B"},
        },
        "edges": [{"name": "a_over_b", "winner": "a", "loser": "b"}],
    }
    summary = official_summary(records, config)
    assert summary["scores"]["olmes_10_macro_avg"]["1B"]["a"] == 0.7
    assert summary["scores"]["arc_easy"]["1B"]["a"] == 0.6

    full = all_recipe_pairwise_summary(records, records, config)
    agreement = full["tasks"]["olmes_10_macro_avg"]["proxy"]["20M"]["mean_score_pairwise"]
    assert agreement["pair_count"] == 1
    assert agreement["accuracy"] == 1.0


def test_continuous_proxy_predicts_target_accuracy_not_target_continuous() -> None:
    proxy_records = [
        {
            "scale": scale,
            "recipe": recipe,
            "task": task,
            "seed": "default",
            "score": score,
        }
        for scale, recipe, task, score in (
            ("20M", "A", "olmes_10_macro_avg", 0.4),
            ("20M", "B", "olmes_10_macro_avg", 0.3),
            ("1B", "A", "olmes_10_macro_avg", 0.7),
            ("1B", "B", "olmes_10_macro_avg", 0.5),
            ("20M", "A", "arc_easy", 0.2),
            ("20M", "B", "arc_easy", 0.3),
            ("1B", "A", "arc_easy", 0.1),
            ("1B", "B", "arc_easy", 0.9),
        )
    ]
    target_records = [
        {
            "scale": "1B",
            "recipe": recipe,
            "task": task,
            "seed": "default",
            "score": score,
        }
        for recipe, task, score in (
            ("A", "olmes_10_macro_avg", 0.7),
            ("B", "olmes_10_macro_avg", 0.5),
            ("A", "arc_easy", 0.8),
            ("B", "arc_easy", 0.7),
        )
    ]
    config = {
        "official": {
            "target_scale": "1B",
            "target_task": "olmes_10_macro_avg",
            "proxy_scales": ["20M"],
            "tasks_and_metrics": {
                "olmes_10_macro_avg": "primary_metric",
                "arc_easy": "correct_prob_per_char",
            },
        },
        "recipes": {
            "a": {"official_name": "A"},
            "b": {"official_name": "B"},
        },
        "edges": [{"name": "a_over_b", "winner": "a", "loser": "b"}],
    }
    summary = official_summary(proxy_records, config, target_records=target_records)
    arc = summary["task_pairwise_agreement"]["arc_easy"]["20M"]
    assert arc["accuracy"] == 0.0


def test_recipe_source_and_block_order_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "named_data_mixes.py"
    source.write_text(
        "DATA_PATHS = {'recipe': ['a.npy', 'b.npy', 'c.npy', 'd.npy']}\n",
        encoding="utf-8",
    )
    paths, digest = load_official_paths(source, "recipe")
    assert paths == ["a.npy", "b.npy", "c.npy", "d.npy"]
    assert len(digest) == 64
    assert _selected_paths(paths, 2) == ["b.npy", "d.npy"]

    tokens = tmp_path / "tokens.npy"
    np.arange(32, dtype="<u2").tofile(tokens)
    first = ShuffledTokenBlocks(tokens, block_size=4, seed=7)
    second = ShuffledTokenBlocks(tokens, block_size=4, seed=7)
    assert first.order.tolist() == second.order.tolist()
    assert first.batch(0, 2, torch.device("cpu")).shape == (2, 4)
    assert DEFAULT_TOKENS == 14_584 * 64 * 2_048


def test_download_range_reuses_a_complete_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "part.bin"
    output.write_bytes(b"abcd")
    monkeypatch.setattr(prepare, "_remote_size", lambda _url, _timeout: 16)

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("a complete part must not be downloaded again")

    monkeypatch.setattr(prepare.urllib.request, "urlopen", unexpected_request)
    result = _download_range(
        url="https://example.invalid/data.npy",
        output=output,
        length=4,
        seed_material="fixed",
        timeout=1,
        retries=0,
    )
    assert result["reused"] is True
    assert result["bytes"] == 4
    assert output.read_bytes() == b"abcd"
