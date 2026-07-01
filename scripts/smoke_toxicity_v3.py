"""Smoke test for compute_toxicity_v3.

Runs the full chunk + recall + judge pipeline on a tiny in-memory doc set.
Recall uses the real XLM-R model (so we exercise the binary classifier path),
but the Qwen judge is stubbed out — no GPU LLM work, no 14 GB of weights to load.

Run:
  PYTHONPATH=. python scripts/smoke_toxicity_v3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stages.safety.utils import compute_toxicity_v3


DOCS = [
    {"doc_id": "clean_1", "text": "The educational value of well-structured pretraining corpora is high. " * 30},
    {"doc_id": "clean_2", "text": "Machine learning is a field of study."},
    {
        "doc_id": "tox_1",
        "text": (
            "I hate everyone, you are all worthless idiots and should die. "
            "These bastards deserve nothing but pain. "
        ) * 5,
    },
    {
        "doc_id": "mixed_1",
        "text": (
            "Charles Dickens, in Oliver Twist, depicted the cruel poverty of "
            "19th-century London. The villain Fagin called the children "
            'dirty thieves and said "you damned little wretches deserve nothing." '
            "This passage is widely studied in literature classes."
        ),
    },
]


def stub_judge_factory(model_path, *, max_tokens, temperature, gpu_memory_utilization):
    """Stand-in for vLLM/Qwen: classify by simple lexical heuristics."""
    promote_kw = ("hate", "die", "bastards", "worthless")
    discuss_kw = ("depicted", "literature", "Dickens", "passage")

    def judge(chunks: list[str]) -> list[dict]:
        out = []
        for c in chunks:
            low = c.lower()
            if any(k in low for k in promote_kw) and not any(k in c for k in discuss_kw):
                out.append({"verdict": "promote", "confidence": 0.95, "reason": "stub: hate kw"})
            elif any(k in c for k in discuss_kw):
                out.append({"verdict": "discuss", "confidence": 0.8, "reason": "stub: literature"})
            else:
                out.append({"verdict": "benign", "confidence": 0.6, "reason": "stub: neutral"})
        return out

    return judge


def main() -> int:
    per_doc, summary = compute_toxicity_v3(
        DOCS,
        recall_model_path="/mnt/public/data/models/xlmr-large-toxicity-classifier",
        judge_model_path="<stub>",
        chunk_size=128,
        chunk_overlap=16,
        recall_threshold=0.5,
        recall_batch_size=8,
        judge_max_chunks_per_doc=4,
        judge_factory=stub_judge_factory,
    )

    print(f"== summary ==")
    for k in (
        "total_docs_scanned", "total_chunks", "recalled_chunks",
        "recalled_docs", "high_risk_docs", "high_risk_pct",
        "verdict_distribution",
    ):
        print(f"  {k}: {summary[k]}")
    print(f"  xlmr_max_stats: {summary['xlmr_max_stats']}")

    print(f"\n== per_doc ==")
    for r in per_doc:
        flags = r.flags
        print(
            f"  {r.doc_id}: xlmr_max={r.scores['xlmr_max']:.4f} "
            f"n_chunks={r.scores['n_chunks']} recalled={r.scores['n_chunks_recalled']} "
            f"promote={flags['llm_promote']} verdicts="
            f"[{', '.join(j['verdict'] for j in r.scores['judgments'])}]"
        )

    # Sanity assertions.
    by_id = {r.doc_id: r for r in per_doc}
    assert by_id["clean_1"].flags["llm_promote"] is False, "clean_1 must not flag promote"
    assert by_id["clean_2"].flags["llm_promote"] is False, "clean_2 must not flag promote"
    assert by_id["tox_1"].scores["n_chunks_recalled"] > 0, "tox_1 must trigger recall"
    assert by_id["tox_1"].flags["llm_promote"] is True, "tox_1 must judge as promote"
    print("\n[OK] smoke assertions passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
