from __future__ import annotations
import json, sys
from src.reader import read_documents
from stages.safety.utils import compute_pii

def main():
    config = {"input": {"format": "jsonl"}}
    docs = list(read_documents("data/pii_sanity.jsonl", config))
    print("Input docs:", len(docs))
    per_doc, summary = compute_pii(docs, language="en", spacy_model="/tmp/spacy_blank_en")
    for d in per_doc:
        types = [h["entity_type"] for h in d.scores["pii_hits"]]
        print("  ", d.doc_id, "has_pii=", d.flags["has_pii"], "hits=", types)
    print("SUMMARY:", json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
