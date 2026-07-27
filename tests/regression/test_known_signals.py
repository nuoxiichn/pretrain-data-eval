from __future__ import annotations

from stages.cleaning.utils import compute_extraction_audit
from stages.dedup.utils import compute_exact_dedup


def test_injected_duplicate_and_residue_signals_do_not_drift():
    repeated = "This is a repeated document with enough text for paragraph checks. " * 2
    documents = [
        {"doc_id": "dup-1", "text": repeated},
        {"doc_id": "dup-2", "text": repeated},
        {"doc_id": "html", "text": "<html><body>bad</body><footer>end</footer></html>"},
        {"doc_id": "short", "text": "404"},
        {"doc_id": "clean", "text": "A clean document that is long enough to pass the stub rule."},
    ]
    _, dedup = compute_exact_dedup(documents, min_para_chars=20)
    extraction_results, extraction = compute_extraction_audit(documents, short_stub_chars=10)
    flags = {result.doc_id: result.flags for result in extraction_results}

    assert dedup["total_docs"] == 5
    assert dedup["exact_dup_docs"] == 2
    assert dedup["exact_dup_pct"] == 0.4
    assert extraction["html_residue_docs"] == 1
    assert extraction["short_stub_docs"] == 1
    assert flags["clean"]["low_extraction_quality"] is False
