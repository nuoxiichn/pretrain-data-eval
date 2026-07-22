from __future__ import annotations

import json

import numpy as np
import yaml
from click.testing import CliRunner
from tokenizers import Tokenizer, models

from stages.trainability.run import cli


def _write_corpus(path, prefix: str, suffix: str, domain: str = "general_web") -> None:
    rows = [
        {
            "doc_id": f"{prefix}-{index}",
            "text": (f"This is training document {index}. {suffix} " * 8).strip(),
            "language": "en",
            "source": prefix,
            "meta": {
                "anchor_id": prefix,
                "domain": domain,
                "comparison_groups": ["smoke_axis"],
                "dedup_cluster_id": f"{prefix}-{index}",
            },
        }
        for index in range(12)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_trainability_probe_cpu_smoke(tmp_path, schema_validator):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write_corpus(left, "left", "coherent prose")
    _write_corpus(right, "right", "different but coherent text", domain="code")
    output = tmp_path / "output"
    config = {
        "input": {"format": "jsonl", "field_map": {}},
        "sampling": {
            "max_docs_per_corpus": 12,
            "mode": "head",
            "seed": 1,
            "validation_fraction": 0.25,
            "split_seed": 2,
        },
        "tokenizer": {"vocab_size": 512, "min_frequency": 1, "max_chars_per_corpus": 5000},
        "model": {
            "context_length": 16,
            "primary": {"d_model": 32, "n_layers": 1, "n_heads": 4, "ffn_multiplier": 2},
            "scale_small": {
                "d_model": 16,
                "n_layers": 1,
                "n_heads": 4,
                "ffn_multiplier": 2,
            },
        },
        "training": {
            "steps": 2,
            "checkpoints": [1, 2],
            "batch_size": 2,
            "learning_rate": 0.001,
            "min_learning_rate_ratio": 0.1,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
            "max_train_tokens_per_corpus": 1000,
            "seeds": [3],
        },
        "evaluation": {
            "max_tokens_per_doc": 64,
            "batch_size": 4,
            "bootstrap_samples": 50,
            "confidence": 0.9,
            "conditioning_min_effect_bits": 0.01,
            "scaling_min_effect_bits": 0.01,
            "stable_horizon_fraction": 0.8,
            "scaling_checkpoints": "all",
        },
        "balanced_pool": {"policy": "recipe_multiset", "seed": 7, "chunk_tokens": 64},
        "runtime": {"device": "cpu", "torch_num_threads": 1},
        "output": {"base_dir": str(tmp_path / "unused")},
        "compatibility_gate": {
            "profiles": {"left": {"language": "en", "domain": "general_web"}}
        },
    }
    config_path = tmp_path / "stage11.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "probe",
            "--corpus",
            f"left={left}",
            "--corpus",
            f"right={right}",
            "--dataset",
            "smoke",
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    per_doc = [json.loads(line) for line in (output / "per_doc.jsonl").read_text().splitlines()]
    schema_validator("summary.schema.json").validate(summary)
    for row in per_doc:
        schema_validator("per_doc.schema.json").validate(row)
    assert summary["runtime"]["trained_model_count"] == 4
    assert set(summary["scaling_gain"]["by_corpus"]) == {"left", "right"}
    assert summary["scaling_gain"]["checkpoints"] == [1, 2]
    assert summary["protocol"]["balanced_pool"]["policy"] == "recipe_multiset"
    assert per_doc
    assert (output / "tokenizer.json").exists()
    assert (output / "balanced_pool.json").exists()

    scaling_output = tmp_path / "scaling-only"
    result = CliRunner().invoke(
        cli,
        [
            "probe",
            "--corpus",
            f"left={left}",
            "--corpus",
            f"right={right}",
            "--dataset",
            "scaling-smoke",
            "--config",
            str(config_path),
            "--output-dir",
            str(scaling_output),
            "--tokenizer-path",
            str(output / "tokenizer.json"),
            "--scaling-train-corpus",
            "left",
            "--scaling-only",
            "--anchor-profile",
            "left",
        ],
    )
    assert result.exit_code == 0, result.output
    scaling_summary = json.loads(
        (scaling_output / "summary.json").read_text(encoding="utf-8")
    )
    assert scaling_summary["runtime"]["trained_model_count"] == 2
    assert scaling_summary["conditioning"]["status"] == "not_run"
    assert scaling_summary["protocol"]["scale_training_source"] == "left"
    assert scaling_summary["tokenizer"]["source"] == str((output / "tokenizer.json").resolve())
    assert scaling_summary["compatibility_gate"]["by_corpus"]["left"]["decision"] == (
        "comparable"
    )
    assert scaling_summary["compatibility_gate"]["by_corpus"]["right"]["decision"] == (
        "abstain"
    )
    assert scaling_summary["scaling_gain"]["ordering_edges"] == []
    assert scaling_summary["scaling_gain"]["comparisons"]["left>right"]["status"] == (
        "abstain"
    )
    scaling_manifest = json.loads(
        (scaling_output / "manifest.json").read_text(encoding="utf-8")
    )
    assert scaling_manifest["artifact_type"] == "stage11_anchor_calibration_run_manifest"
    assert scaling_manifest["leakage_audit"] == scaling_summary["leakage_audit"]

    conditioning_output = tmp_path / "conditioning-only"
    result = CliRunner().invoke(
        cli,
        [
            "probe",
            "--corpus",
            f"left={left}",
            "--corpus",
            f"right={right}",
            "--dataset",
            "conditioning-smoke",
            "--config",
            str(config_path),
            "--output-dir",
            str(conditioning_output),
            "--tokenizer-path",
            str(output / "tokenizer.json"),
            "--conditioning-only",
        ],
    )
    assert result.exit_code == 0, result.output
    conditioning_summary = json.loads(
        (conditioning_output / "summary.json").read_text(encoding="utf-8")
    )
    conditioning_docs = [
        json.loads(line)
        for line in (conditioning_output / "per_doc.jsonl").read_text().splitlines()
    ]
    assert conditioning_summary["runtime"]["trained_model_count"] == 2
    assert conditioning_summary["scaling_gain"]["status"] == "not_run"
    assert conditioning_summary["protocol"]["conditioning_only"] is True
    assert "scaling_gain_bits" not in conditioning_docs[0]["scores"]
    assert conditioning_docs[0]["scores"]["primary_loss_bits_by_train_corpus"]


def test_trainability_pretokenized_fixed_ratio_smoke(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = Tokenizer(
        models.WordLevel(
            {"<unk>": 0, "<eod>": 1, "a": 2, "b": 3, "c": 4},
            unk_token="<unk>",
        )
    )
    tokenizer.save(str(tokenizer_path))
    left = tmp_path / "left.npy"
    right = tmp_path / "right.npy"
    np.array([2, 3, 1] * 12, dtype="<u2").tofile(left)
    np.array([2, 4, 1] * 12, dtype="<u2").tofile(right)
    config = {
        "input": {
            "format": "uint16_stream",
            "validation_reserve_tokens": 12,
        },
        "sampling": {
            "max_docs_per_corpus": 2,
            "mode": "head",
            "seed": 1,
            "validation_fraction": 0.2,
            "split_seed": 2,
        },
        "tokenizer": {
            "vocab_size": 5,
            "min_frequency": 1,
            "max_chars_per_corpus": 1,
            "eod_token": "<eod>",
        },
        "model": {
            "context_length": 8,
            "primary": {
                "d_model": 8,
                "n_layers": 1,
                "n_heads": 2,
                "ffn_multiplier": 2,
            },
            "scale_small": {
                "d_model": 8,
                "n_layers": 1,
                "n_heads": 2,
                "ffn_multiplier": 2,
            },
        },
        "training": {
            "steps": 2,
            "checkpoints": [1, 2],
            "batch_size": 2,
            "learning_rate": 0.001,
            "warmup_steps": 0,
            "max_train_tokens_per_corpus": 12,
            "seeds": [3],
        },
        "evaluation": {
            "max_tokens_per_doc": 4,
            "batch_size": 2,
            "bootstrap_samples": 10,
            "confidence": 0.9,
            "conditioning_min_effect_bits": 0.01,
            "scaling_min_effect_bits": 0.01,
            "stable_horizon_fraction": 0.5,
        },
        "runtime": {"device": "cpu", "torch_num_threads": 1},
        "output": {"base_dir": str(tmp_path / "unused")},
    }
    config_path = tmp_path / "stream.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "stream-output"
    result = CliRunner().invoke(
        cli,
        [
            "probe",
            "--corpus",
            f"left={left}",
            "--corpus",
            f"right={right}",
            "--dataset",
            "stream-smoke",
            "--config",
            str(config_path),
            "--output-dir",
            str(output),
            "--tokenizer-path",
            str(tokenizer_path),
            "--tokens-per-parameter",
            "0.01",
            "--conditioning-only",
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["protocol"]["requested_tokens_per_parameter"] == 0.01
    assert summary["protocol"]["tokens_per_parameter"] >= 0.01
    assert {corpus["source_format"] for corpus in summary["corpora"]} == {
        "uint16_le"
    }
