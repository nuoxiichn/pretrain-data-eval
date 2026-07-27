"""Input adapter for all pipeline stages.

Reads documents in the standard schema. Supports JSONL and Parquet;
accepts a single file or a directory (all matching files are streamed
in sorted order). Extend `read_documents` to add new formats without
changing downstream code.

Standard fields
---------------
doc_id    : str       — unique identifier
text      : str       — document body
source    : str|None  — source/corpus name
url       : str|None  — original URL
timestamp : str|None  — ISO-8601 date string
language  : str|None  — ISO-639-1 code (pre-labelled or path-inferred)
meta      : dict      — catch-all for remaining metadata

Input config keys (passed from stage YAML)
-------------------------------------------
format      : "auto" | "jsonl" | "parquet"
field_map   : {src_field: dst_field}   — rename before normalizing
              e.g. {"uid": "doc_id", "content": "text"}
path_meta   : {std_field: {path_fragment: value}}
              — fill fields by matching path fragments
              e.g. {"language": {"ultrafineweb_en_l3": "en"}}
batch_size  : int  — rows per batch for Parquet (default 500)
glob        : str  — glob pattern when path is a directory (default "*.parquet")
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


# ── Field mapping & normalization ─────────────────────────────────────────────

def _apply_field_map(raw: dict, field_map: dict[str, str]) -> dict:
    """Rename raw fields according to field_map; unmentioned fields pass through."""
    if not field_map:
        return raw
    return {field_map.get(k, k): v for k, v in raw.items()}


def _normalize(raw: dict, meta_overrides: dict | None = None) -> Document:
    """Map raw dict to standard Document, applying overrides for None fields."""
    doc: Document = {k: raw.get(k, v) for k, v in _DEFAULTS.items()}
    # Collect unmapped fields into meta
    known = set(_DEFAULTS)
    extra = {k: v for k, v in raw.items() if k not in known}
    if extra:
        meta = dict(doc.get("meta") or {})
        meta.update(extra)
        doc["meta"] = meta
    # Apply path-inferred overrides only where field is still None
    if meta_overrides:
        for field, value in meta_overrides.items():
            if field in doc and doc[field] is None:
                doc[field] = value
    return doc


def _infer_meta_from_path(path: Path, path_meta: dict[str, dict[str, str]]) -> dict:
    """Infer document fields from file path using pattern matching.

    path_meta example:
      {"language": {"ultrafineweb_en_l3": "en", "ultrafineweb_zh_l3": "zh"},
       "source":   {"multi_style": "ultrafineweb_multi_style", "qa": "ultrafineweb_qa"}}
    """
    path_str = str(path)
    result: dict = {}
    for field, patterns in path_meta.items():
        for fragment, value in patterns.items():
            if fragment in path_str:
                result[field] = value
                break
    return result


# ── JSONL ─────────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path, field_map: dict, meta_overrides: dict) -> Iterator[Document]:
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = _apply_field_map(json.loads(line), field_map)
                yield _normalize(raw, meta_overrides)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON — {exc}") from exc


# ── Parquet ───────────────────────────────────────────────────────────────────

def _read_parquet_file(
    path: Path,
    field_map: dict,
    meta_overrides: dict,
    batch_size: int,
) -> Iterator[Document]:
    """Stream a single Parquet file in row-group batches. Requires pyarrow."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow not installed. Run: pip install pyarrow") from exc

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        columns = batch.to_pydict()
        n = len(next(iter(columns.values())))
        for i in range(n):
            raw = _apply_field_map({k: v[i] for k, v in columns.items()}, field_map)
            yield _normalize(raw, meta_overrides)


# ── Directory ─────────────────────────────────────────────────────────────────

def _read_directory(path: Path, config: dict) -> Iterator[Document]:
    """Stream all matching files in a directory (sorted, recursive-glob supported)."""
    pattern = config.get("glob", "*.parquet")
    field_map = config.get("field_map", {})
    path_meta = config.get("path_meta", {})
    batch_size = config.get("batch_size", 500)

    files = sorted(path.glob(f"**/{pattern}") if "/" not in pattern else path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found under {path}")

    for file_path in files:
        meta_overrides = _infer_meta_from_path(file_path, path_meta)
        yield from _read_parquet_file(file_path, field_map, meta_overrides, batch_size)


# ── Public entry point ────────────────────────────────────────────────────────

def read_documents(
    path: str | Path,
    config: dict | None = None,
) -> Iterator[Document]:
    """Yield normalized Documents from *path* (file or directory).

    config: input section from stage YAML. Keys: format, field_map,
            path_meta, batch_size, glob. All optional.
    """
    path = Path(path)
    cfg = config or {}
    fmt = cfg.get("format", "auto")
    field_map = cfg.get("field_map", {})
    path_meta = cfg.get("path_meta", {})
    batch_size = cfg.get("batch_size", 500)

    if path.is_dir():
        yield from _read_directory(path, cfg)
        return

    if fmt == "auto":
        name = path.name.lower()
        if name.endswith(".parquet"):
            fmt = "parquet"
        elif name.endswith(".jsonl") or name.endswith(".json"):
            fmt = "jsonl"
        else:
            raise ValueError(
                f"Cannot infer format from '{path.name}'. Set format: in config."
            )

    meta_overrides = _infer_meta_from_path(path, path_meta)

    if fmt == "jsonl":
        yield from _read_jsonl(path, field_map, meta_overrides)
    elif fmt == "parquet":
        yield from _read_parquet_file(path, field_map, meta_overrides, batch_size)
    else:
        raise NotImplementedError(f"Format '{fmt}' not yet implemented in reader.py")
