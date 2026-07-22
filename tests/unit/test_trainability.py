from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from stages.trainability.data import (
    CorpusData,
    EncodedTrainDocument,
    audit_document_overlaps,
    document_cluster_id,
    load_pretokenized_corpora,
    make_balanced_pool,
    split_documents,
)
from stages.trainability.metrics import analyze_scaling_gain, hierarchical_weighted_stats
from stages.trainability.model import ModelConfig, TinyCausalLM, model_parameter_counts
from stages.trainability.run import (
    _apply_tokens_per_parameter,
    _input_eligibility_gate,
    _rescale_checkpoints,
)
from stages.trainability.model import TrainingConfig


def _documents(count: int) -> list[dict]:
    return [
        {
            "doc_id": f"doc-{index}",
            "text": f"document {index}",
            "source": None,
            "url": None,
            "timestamp": None,
            "language": "en",
            "meta": {},
        }
        for index in range(count)
    ]


def test_document_split_is_deterministic_and_disjoint():
    first_train, first_validation = split_documents(_documents(50), 0.2, 42)
    second_train, second_validation = split_documents(_documents(50), 0.2, 42)
    assert [doc["doc_id"] for doc in first_train] == [doc["doc_id"] for doc in second_train]
    assert [doc["doc_id"] for doc in first_validation] == [
        doc["doc_id"] for doc in second_validation
    ]
    assert {doc["doc_id"] for doc in first_train}.isdisjoint(
        doc["doc_id"] for doc in first_validation
    )


def test_dedup_cluster_split_and_frozen_split_are_enforced():
    documents = _documents(20)
    documents[0]["meta"] = {"dedup_cluster_id": "shared"}
    documents[1]["meta"] = {"dedup_cluster_id": "shared"}
    train, validation = split_documents(documents, 0.2, 42)
    locations = {
        document_cluster_id(document): split
        for split, rows in (("train", train), ("validation", validation))
        for document in rows
    }
    assert locations["shared"] in {"train", "validation"}
    assert sum(document_cluster_id(row) == "shared" for row in train + validation) == 2

    frozen = _documents(4)
    for index, document in enumerate(frozen):
        document["meta"] = {
            "dedup_cluster_id": "conflict" if index < 2 else f"cluster-{index}",
            "split": "train" if index != 1 else "validation",
        }
    with pytest.raises(ValueError, match="both train and validation"):
        split_documents(frozen, 0.2, 42)

    exact_conflict = _documents(4)
    exact_conflict[1]["text"] = exact_conflict[0]["text"]
    for index, document in enumerate(exact_conflict):
        document["meta"] = {
            "dedup_cluster_id": f"declared-{index}",
            "split": "train" if index != 1 else "validation",
        }
    with pytest.raises(ValueError, match="dedup/exact component"):
        split_documents(exact_conflict, 0.2, 42)


def test_overlap_audit_reports_exact_and_near_cross_corpus_overlap():
    left_docs = _documents(4)
    right_docs = _documents(4)
    for index, document in enumerate(left_docs + right_docs):
        document["meta"] = {"dedup_cluster_id": f"cluster-{index % 4}"}
    right_docs[0]["text"] = left_docs[0]["text"]
    left = CorpusData("left", Path("left"), 4, left_docs[:2], left_docs[2:])
    right = CorpusData("right", Path("right"), 4, right_docs[:2], right_docs[2:])
    audit = audit_document_overlaps([left, right])
    assert audit["global_train_validation_leaking_clusters"] == 0
    assert audit["cross_corpus"]["left<->right"]["exact_hashes"] >= 1
    assert audit["cross_corpus"]["left<->right"]["near_clusters"] == 4


def test_pretokenized_stream_uses_disjoint_prefix_and_suffix(tmp_path):
    eod_id = 99
    values = np.array(
        [1, 2, eod_id, 3, 4, eod_id, 5, 6, eod_id, 7, 8, eod_id],
        dtype="<u2",
    )
    path = tmp_path / "tokens.npy"
    values.tofile(path)
    corpora = load_pretokenized_corpora(
        [("sample", path)],
        dtype="uint16_le",
        vocab_size=100,
        eod_id=eod_id,
        max_train_tokens_per_corpus=5,
        validation_reserve_tokens=7,
        max_validation_docs=2,
        max_eval_tokens_per_doc=8,
    )
    corpus = corpora[0]
    assert corpus.train_tokens is not None
    assert corpus.train_tokens[:].tolist() == [1, 2, eod_id, 3, 4]
    assert corpus.validation_region_start == 5
    assert [doc.token_ids for doc in corpus.encoded_validation or []] == [
        (eod_id, 5, 6),
        (eod_id, 7, 8),
    ]
    assert all(
        int(doc.doc_id.removeprefix("stream-")) >= corpus.validation_region_start
        for doc in corpus.encoded_validation or []
    )


def test_pretokenized_stream_rejects_token_outside_vocabulary(tmp_path):
    path = tmp_path / "bad.npy"
    np.array([2, 99, 1] * 5, dtype="<u2").tofile(path)
    with pytest.raises(ValueError, match="outside tokenizer vocabulary"):
        load_pretokenized_corpora(
            [("bad", path)],
            dtype="uint16_le",
            vocab_size=10,
            eod_id=1,
            max_train_tokens_per_corpus=6,
            validation_reserve_tokens=6,
            max_validation_docs=2,
            max_eval_tokens_per_doc=4,
        )


def test_balanced_pool_is_deterministic_equal_and_auditable():
    left = CorpusData("left", Path("left"), 2, [], [])
    right = CorpusData("right", Path("right"), 2, [], [])
    left.train_tokens = torch.tensor([1, 10, 11, 1, 12, 13])
    right.train_tokens = torch.tensor([1, 20, 21, 1, 22, 23, 24, 25])
    left.encoded_train_documents = [
        EncodedTrainDocument("left-1", "left-1", 0, 3),
        EncodedTrainDocument("left-2", "left-2", 3, 3),
    ]
    right.encoded_train_documents = [
        EncodedTrainDocument("right-1", "right-1", 0, 3),
        EncodedTrainDocument("right-2", "right-2", 3, 5),
    ]
    first, first_manifest = make_balanced_pool([left, right], seed=7)
    second, second_manifest = make_balanced_pool([left, right], seed=7)
    assert first[:].tolist() == second[:].tolist()
    assert first_manifest == second_manifest
    assert first_manifest["selected_tokens_by_corpus"] == {"left": 6, "right": 6}
    assert len(first) == 12
    assert first_manifest["membership_sha256"]


def test_unique_cluster_pool_removes_cross_recipe_membership():
    left = CorpusData("left", Path("left"), 2, [], [])
    right = CorpusData("right", Path("right"), 2, [], [])
    left.train_tokens = torch.tensor([1, 10, 1, 11])
    right.train_tokens = torch.tensor([1, 20, 1, 21])
    left.encoded_train_documents = [
        EncodedTrainDocument("shared-left", "shared", 0, 2),
        EncodedTrainDocument("left-only", "left-only", 2, 2),
    ]
    right.encoded_train_documents = [
        EncodedTrainDocument("shared-right", "shared", 0, 2),
        EncodedTrainDocument("right-only", "right-only", 2, 2),
    ]
    pool, manifest = make_balanced_pool(
        [left, right], seed=9, policy="unique_cluster_union"
    )
    assert len(pool) == 6
    assert manifest["overlap_clusters_removed"] == 1
    assert manifest["segment_count"] == 3


def test_scaling_gain_requires_seed_and_horizon_stability():
    def evaluation(gap: float) -> tuple[dict, dict]:
        large = {
            "doc_loss_nats": [1.0, 1.0],
            "doc_token_counts": [10, 10],
        }
        small = {
            "doc_loss_nats": [1.0 + gap, 1.0 + gap],
            "doc_token_counts": [10, 10],
        }
        return small, large

    runs: dict[str, dict] = {}
    for seed in ("17", "29", "41"):
        runs[seed] = {"small": {"evaluations": {}}, "large": {"evaluations": {}}}
        for checkpoint in (1, 2):
            a_small, a_large = evaluation(0.3)
            b_small, b_large = evaluation(0.1)
            runs[seed]["small"]["evaluations"][str(checkpoint)] = {
                "a": a_small,
                "b": b_small,
            }
            runs[seed]["large"]["evaluations"][str(checkpoint)] = {
                "a": a_large,
                "b": b_large,
            }
    summary = analyze_scaling_gain(
        runs,
        ["a", "b"],
        2,
        200,
        0.95,
        0.01,
        7,
        checkpoints=[1, 2],
        stable_horizon_fraction=0.8,
        require_unanimous_seed_direction=True,
    )
    assert summary["pair_stability"]["a>b"]["status"] == "stable_higher"
    assert summary["stable_edges"][0]["higher_scaling_gain"] == "a"
    assert summary["comparisons"]["a>b"]["seed_unanimous_positive"] is True


def test_fixed_tokens_per_parameter_derives_steps_and_checkpoints():
    model_config = ModelConfig(16, 128, 16, 1, 4, 2)
    training_config = TrainingConfig(100, 2, 1e-3, 0.1, 10, 0.0, 0.9, 0.95, 1.0)
    total_parameters, non_embedding_parameters = model_parameter_counts(model_config)
    adjusted, checkpoints, observed_total, observed_non_embedding = (
        _apply_tokens_per_parameter(
            training_config, [25, 50, 100], model_config, requested_ratio=2.0
        )
    )
    assert adjusted.steps == math.ceil(2.0 * total_parameters / (2 * 16))
    assert checkpoints[-1] == adjusted.steps
    assert adjusted.steps * 2 * 16 / total_parameters >= 2.0
    assert observed_total == total_parameters
    assert observed_non_embedding == non_embedding_parameters
    assert _rescale_checkpoints([25, 50, 100], 100, 200) == [50, 100, 200]


def test_input_eligibility_binds_corpus_hash_and_audit_decision(tmp_path):
    path = tmp_path / "left.jsonl"
    path.write_text('{"doc_id":"a","text":"text"}\n', encoding="utf-8")
    manifest = {
        "artifact_type": "balanced_input_eligibility",
        "protocol_version": "balanced_production_validation_v2_1",
        "corpora": {
            "left": {
                "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "decision": "eligible",
                "frozen_cluster_split": {"status": "pass"},
                "repetition": {"status": "pass", "method": "gopher_repetition_v1"},
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "input_eligibility": {
            "required": True,
            "allowed_repetition_methods": ["gopher_repetition_v1"],
        },
        "validation": {"protocol_version": "balanced_production_validation_v2_1"},
    }
    result = _input_eligibility_gate([("left", path)], config, manifest_path)
    assert result["decision"] == "eligible"

    manifest["corpora"]["left"]["decision"] = "ineligible"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(Exception, match="eligibility failed for: left"):
        _input_eligibility_gate([("left", path)], config, manifest_path)


def test_hierarchical_bootstrap_preserves_direction():
    stats = hierarchical_weighted_stats(
        [[-0.3, -0.2, -0.4], [-0.25, -0.35, -0.2]],
        [10, 20, 30],
        bootstrap_samples=500,
        confidence=0.95,
        seed=7,
    )
    assert stats["mean"] < 0
    assert stats["ci_high"] < 0
    assert math.isfinite(stats["seed_std"])


def test_tiny_causal_lm_forward_shape_and_parameter_scale():
    small_config = ModelConfig(16, 512, 16, 1, 4, 2)
    large_config = ModelConfig(16, 512, 32, 2, 4, 2)
    small = TinyCausalLM(small_config)
    large = TinyCausalLM(large_config)
    input_ids = torch.randint(0, 512, (2, 16))
    assert small(input_ids).shape == (2, 16, 512)
    assert sum(p.numel() for p in large.parameters()) > sum(
        p.numel() for p in small.parameters()
    )

