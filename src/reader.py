"""Input adapter for all pipeline stages.

Reads documents in the standard schema. Currently supports JSONL;
extend `read_documents` to add Parquet, HF Datasets, DataTrove format
without changing downstream code.

Standard fields
---------------
doc_id    : str        — unique identifier
text      : str        — document body
source    : str|None   — source/corpus name (e.g. "common_crawl", "github")
url       : str|None   — original URL
timestamp : str|None   — ISO-8601 date string
language  : str|None   — ISO-639-1 code (pre-labelled)
meta      : dict       — catch-all for remaining metadata
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

Document = dict[str, object]

_DEFAULTS: Document = {
    "doc_id": "",
    "text": "",
    "source": None,
    "url": None,
    "timestamp": None,
    "language": None,
    "meta": {},
}


def _normalize(raw: dict) -> Document:
    return {k: raw.get(k, v) for k, v in _DEFAULTS.items()}


def _read_jsonl(path: Path) -> Iterator[Document]:
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield _normalize(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc


def read_documents(path: str | Path, fmt: str = "auto") -> Iterator[Document]:
    """Yield normalized documents from *path*.

    fmt: "auto" infers from extension. Supported: "jsonl".
    Extend this function (not callers) when adding new formats.
    """
    path = Path(path)
    if fmt == "auto":
        name = path.name.lower()
        if name.endswith(".jsonl") or name.endswith(".json"):
            fmt = "jsonl"
        else:
            raise ValueError(
                f"Cannot infer format from '{path.name}'. Pass fmt= explicitly."
            )
    if fmt == "jsonl":
        yield from _read_jsonl(path)
    else:
        raise NotImplementedError(f"Format '{fmt}' not yet implemented in reader.py")
