from __future__ import annotations

import json

from stages.cleaning.utils import compute_extraction_audit
from stages.contamination.benchmarks import BenchItem
from stages.contamination.utils import compute_exact_contamination
from stages.dedup.utils import compute_exact_dedup, compute_repetition_audit
from stages.longctx.utils import compute_config_audit
from stages.source_audit.utils import compute_doc_stats, make_tokenizer


def test_exact_dedup_marks_all_members_of_duplicate_group(sample_documents):
    results, summary = compute_exact_dedup(sample_documents)
    by_id = {result.doc_id: result for result in results}
    assert summary["exact_dup_docs"] == 2
    assert summary["unique_doc_hashes"] == 2
    assert by_id["a"].flags["is_exact_dup"] is True
    assert by_id["b"].scores["dup_doc_count"] == 2
    assert by_id["c"].flags["is_exact_dup"] is False


def test_repetition_audit_separates_looped_from_varied_text():
    varied = " ".join(f"unique{index}" for index in range(200))
    looped = ("alpha beta gamma delta epsilon zeta eta theta " * 40).strip()
    results, summary = compute_repetition_audit(
        [
            {"doc_id": "varied", "text": varied},
            {"doc_id": "looped", "text": looped},
        ]
    )
    by_id = {result.doc_id: result for result in results}
    assert by_id["varied"].flags["is_high_repetition"] is False
    assert by_id["looped"].flags["is_high_repetition"] is True
    assert summary["high_repetition_docs"] == 1


def test_extraction_audit_separates_information_from_low_quality():
    docs = [
        {"doc_id": "url", "text": "Read https://example.com for a useful reference."},
        {"doc_id": "html", "text": "<div>x</div><p>y</p><footer>z</footer>"},
        {"doc_id": "clean", "text": "A clean and sufficiently long natural language document."},
    ]
    results, summary = compute_extraction_audit(docs, short_stub_chars=10)
    by_id = {result.doc_id: result for result in results}
    assert by_id["url"].scores["url_count"] == 1
    assert by_id["url"].flags["low_extraction_quality"] is False
    assert by_id["html"].flags["has_html_residue"] is True
    assert summary["html_residue_docs"] == 1


def test_exact_contamination_has_injected_truth():
    bench = BenchItem(bench_id="bench_0", text="What is two plus two?", code=None, benchmark="toy")
    docs = [
        {"doc_id": "hit", "text": bench["text"]},
        {"doc_id": "miss", "text": "An unrelated training document."},
    ]
    results, summary = compute_exact_contamination(docs, {"toy": [bench]})
    by_id = {result.doc_id: result for result in results}
    assert by_id["hit"].flags["is_exact_contaminated"] is True
    assert by_id["miss"].flags["is_exact_contaminated"] is False
    assert summary["per_benchmark_hits"] == {"toy": 1}


def test_source_stats_have_consistent_totals(sample_documents):
    results, summary = compute_doc_stats(sample_documents, make_tokenizer("words"))
    assert len(results) == summary["total_docs"] == 3
    assert summary["token_stats"]["total"] == sum(
        result.scores["token_count"] for result in results
    )
    assert sum(bucket["count"] for bucket in summary["length_buckets"].values()) == 3


def test_config_audit_accepts_valid_and_rejects_missing(tmp_path):
    valid = tmp_path / "valid.yaml"
    valid.write_text(
        "reset_position_ids: true\nreset_attention_mask: true\neod_mask_loss: true\n",
        encoding="utf-8",
    )
    _, valid_summary = compute_config_audit(str(valid))
    assert valid_summary["config_valid"] is True

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"reset_position_ids": False}), encoding="utf-8")
    _, invalid_summary = compute_config_audit(str(invalid))
    assert invalid_summary["config_valid"] is False
    assert "reset_attention_mask" in invalid_summary["missing_params"]
