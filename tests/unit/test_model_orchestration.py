from __future__ import annotations

import sys
import types
from dataclasses import dataclass

from stages.dedup.utils import compute_minhash_dedup
from stages.domain.utils import compute_parsability
from stages.safety import utils as safety_utils
from stages.synthetic import utils as synthetic_utils
from stages.tokenization import utils as tokenization_utils


@dataclass
class _Encoding:
    ids: list[int]


class _FakeTokenizer:
    def encode_batch(self, texts):
        return [_Encoding([0 if char == "?" else 1 for char in text]) for text in texts]


def test_tokenization_aggregation_with_fake_tokenizer(monkeypatch):
    monkeypatch.setattr(tokenization_utils, "load_tokenizer", lambda _path: _FakeTokenizer())
    monkeypatch.setattr(tokenization_utils, "find_unk_id", lambda _tokenizer: 0)
    documents = [
        {"doc_id": "plain", "text": "ab?", "language": "en"},
        {"doc_id": "rich", "text": "```x``` and $y$", "language": "en"},
    ]
    results, summary = tokenization_utils.compute_tokenization(
        documents,
        tokenizer_path="fake",
        unk_threshold=0.1,
        fertility_threshold=2.0,
        batch_size=2,
    )
    by_id = {result.doc_id: result for result in results}
    assert by_id["plain"].flags["high_unk_rate"] is True
    assert by_id["rich"].scores["code_token_count"] > 0
    assert by_id["rich"].scores["latex_token_count"] > 0
    assert summary["total_docs"] == 2
    assert summary["unk_stats"]["total_unk_tokens"] == 1


def test_binoculars_aggregation_with_fake_scores(monkeypatch):
    monkeypatch.setattr(
        synthetic_utils,
        "_load_binoculars_models",
        lambda *_args, **_kwargs: (object(), object(), object(), "cpu"),
    )
    monkeypatch.setattr(
        synthetic_utils,
        "_compute_scores_batch",
        lambda _tok, _obs, _perf, texts, _length, _device: [
            0.7 if "generated" in text else 1.1 for text in texts
        ],
    )
    results, summary = synthetic_utils.compute_binoculars(
        [
            {"doc_id": "human", "text": "human prose"},
            {"doc_id": "ai", "text": "generated prose"},
        ],
        "observer",
        "performer",
        threshold=0.85,
    )
    assert [result.flags["is_ai_generated"] for result in results] == [False, True]
    assert summary["ai_generated_docs"] == 1
    assert summary["score_stats"]["min"] == 0.7


def test_toxicity_recall_and_judge_orchestration_without_models(monkeypatch):
    class FakeRecallTokenizer:
        def __call__(self, text, **_kwargs):
            return {"offset_mapping": [(idx, idx + 1) for idx in range(len(text))]}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_path):
            return FakeRecallTokenizer()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    def fake_predict(texts):
        return {"toxicity": [0.9 if "risk" in text else 0.01 for text in texts]}

    monkeypatch.setattr(safety_utils, "_make_hf_predictor", lambda *_a, **_k: (fake_predict, "binary"))

    def judge_factory(*_args, **_kwargs):
        def judge(texts):
            return [
                {"verdict": "promote", "confidence": 0.9, "reason": "test"}
                for _text in texts
            ]

        return judge

    results, summary = safety_utils.compute_toxicity(
        [
            {"doc_id": "safe", "text": "neutral"},
            {"doc_id": "risk", "text": "risk"},
        ],
        "recall",
        "judge",
        chunk_size=64,
        recall_threshold=0.5,
        judge_factory=judge_factory,
    )
    by_id = {result.doc_id: result for result in results}
    assert by_id["safe"].flags["high_risk"] is False
    assert by_id["risk"].flags["high_risk"] is True
    assert summary["recalled_docs"] == 1
    assert summary["high_risk_docs"] == 1


def test_safety_placeholder_and_judge_parsing_boundaries():
    assert safety_utils._is_placeholder("EMAIL_ADDRESS", "user@example.com") is True
    assert safety_utils._is_placeholder("IP_ADDRESS", "192.0.2.10") is True
    assert safety_utils._is_placeholder("EMAIL_ADDRESS", "person@company.cn") is False
    parsed = safety_utils._parse_judge_output(
        'prefix {"verdict":"promote","confidence":0.8,"reason":"x"} suffix'
    )
    assert parsed["verdict"] == "promote"
    assert safety_utils._parse_judge_output("not-json")["verdict"] == "benign"


def test_parsability_auto_mode_records_unsupported_language():
    results, summary = compute_parsability(
        [{"doc_id": "cobol", "text": "IDENTIFICATION DIVISION.", "meta": {"lang": "cobol"}}],
        language="auto",
    )
    assert results[0].flags["unsupported_lang"] is True
    assert summary["unsupported_lang_docs"] == 1
    assert summary["parsed_docs"] == 0


def test_minhash_detects_identical_documents(tmp_path):
    text = "one two three four five six seven eight nine ten eleven twelve"
    results, summary = compute_minhash_dedup(
        [
            {"doc_id": "a", "text": text},
            {"doc_id": "b", "text": text},
            {"doc_id": "c", "text": "completely different words appear in this record only"},
        ],
        num_hashes=16,
        ngram_size=2,
        num_bands=4,
        band_size=4,
        out_dir=tmp_path,
    )
    by_id = {result.doc_id: result for result in results}
    assert by_id["a"].flags["is_near_dup"] is True
    assert by_id["b"].flags["is_near_dup"] is True
    assert summary["near_dup_docs"] == 2
