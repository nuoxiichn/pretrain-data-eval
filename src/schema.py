"""Output schema utilities shared across all pipeline stages.

Every stage produces two kinds of output:
  per_doc.jsonl  — one JSON line per document:
                   {"doc_id": ..., "scores": {...}, "flags": {...}}
  summary.json   — aggregate stats for the full dataset run
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class DocResult:
    doc_id: str
    scores: dict
    flags: dict


def make_output_dir(base: str | Path, stage: str, dataset: str) -> Path:
    """Return and create outputs/{base}/{dataset}_{ts}/{stage}/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / f"{dataset}_{ts}" / stage
    out.mkdir(parents=True, exist_ok=True)
    return out


def use_output_dir(path: str | Path) -> Path:
    """Use an explicit output directory (creates if needed)."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_per_doc(results: list[DocResult], out_dir: Path, name: str = "per_doc") -> Path:
    out = out_dir / f"{name}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    return out


def write_summary(summary: dict, out_dir: Path, name: str = "summary") -> Path:
    out = out_dir / f"{name}.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return out
