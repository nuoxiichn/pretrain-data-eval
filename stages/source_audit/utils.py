"""Stage 1 computation utilities.

Provides two independent computation functions:
  compute_doc_stats   — row 1: basic corpus statistics
  compute_license     — row 2: ScanCode license/copyright detection
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from typing import Callable, Iterable
from urllib.parse import urlparse

import numpy as np

from src.reader import Document
from src.schema import DocResult


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def make_tokenizer(backend: str = "words", hf_name: str | None = None) -> Callable[[str], int]:
    """Return a callable text -> token count.

    backend="words"  : whitespace split (fast; suitable for European languages)
    backend="hf"     : HF Tokenizers (accurate; requires hf_name)
    """
    if backend == "words":
        return lambda text: len(text.split())
    if backend == "hf":
        if not hf_name:
            raise ValueError("hf_name is required when backend='hf'")
        from tokenizers import Tokenizer
        tok = Tokenizer.from_pretrained(hf_name)
        return lambda text: len(tok.encode(text).ids)
    raise ValueError(f"Unknown tokenizer backend: {backend!r}")


# ── Length buckets ────────────────────────────────────────────────────────────

_BUCKET_ORDER = ["<4K", "4K-8K", "8K-32K", "32K-128K", "128K-256K", "256K+"]
_BUCKET_THRESHOLDS = [
    (256_000, "256K+"),
    (128_000, "128K-256K"),
    (32_000, "32K-128K"),
    (8_000, "8K-32K"),
    (4_000, "4K-8K"),
    (0, "<4K"),
]

def get_length_bucket(n_tokens: int) -> str:
    for threshold, label in _BUCKET_THRESHOLDS:
        if n_tokens >= threshold:
            return label
    return "<4K"


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_domain(url: str | None) -> str:
    if not url:
        return "(no_url)"
    try:
        netloc = urlparse(url).netloc
        return netloc or "(empty_netloc)"
    except Exception:
        return "(parse_error)"


_TS_FORMATS = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]

def parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def _dist_stats(values: list[int | float], pcts: tuple = (25, 50, 75, 90, 95, 99)) -> dict:
    if not values:
        return {}
    a = np.array(values, dtype=float)
    out: dict = {
        "count": len(values),
        "total": int(a.sum()),
        "mean": round(float(a.mean()), 2),
        "min": int(a.min()),
        "max": int(a.max()),
    }
    for p in pcts:
        out[f"p{p}"] = round(float(np.percentile(a, p)), 2)
    return out


def _group_table(doc_cnt: Counter, tok_cnt: Counter, total_toks: int) -> dict:
    """Build a sorted distribution table: {key: {docs, tokens, pct_tokens}}."""
    return {
        k: {
            "docs": doc_cnt[k],
            "tokens": tok_cnt[k],
            "pct_tokens": round(tok_cnt[k] / total_toks, 4) if total_toks else 0.0,
        }
        for k in sorted(doc_cnt, key=lambda x: -tok_cnt[x])
    }


# ── Row 1: document statistics ────────────────────────────────────────────────

def compute_doc_stats(
    docs: Iterable[Document],
    tokenize: Callable[[str], int],
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Compute corpus-level and per-document statistics.

    on_doc: if provided, called for each DocResult (streaming mode for large
            corpora). The returned per_doc list will be empty in this case.
    Returns (per_doc_results, summary_dict).
    """
    per_doc: list[DocResult] = []
    n_docs = 0
    char_counts: list[int] = []
    tok_counts: list[int] = []
    bucket_cnt: Counter = Counter()
    domain_docs: Counter = Counter()
    domain_toks: Counter = Counter()
    lang_docs: Counter = Counter()
    lang_toks: Counter = Counter()
    src_docs: Counter = Counter()
    src_toks: Counter = Counter()
    ts_present = 0
    ts_missing = 0
    ym_dist: Counter = Counter()

    for doc in docs:
        text: str = doc.get("text") or ""  # type: ignore[assignment]
        nc = len(text)
        nt = tokenize(text)
        bucket = get_length_bucket(nt)
        domain = extract_domain(doc.get("url"))  # type: ignore[arg-type]
        lang = doc.get("language") or "(unknown)"
        src = doc.get("source") or "(unknown)"
        ts = parse_timestamp(doc.get("timestamp"))  # type: ignore[arg-type]

        char_counts.append(nc)
        tok_counts.append(nt)
        bucket_cnt[bucket] += 1
        domain_docs[domain] += 1
        domain_toks[domain] += nt
        lang_docs[lang] += 1
        lang_toks[lang] += nt
        src_docs[src] += 1
        src_toks[src] += nt

        if ts is not None:
            ts_present += 1
            ym_dist[ts.strftime("%Y-%m")] += 1
        else:
            ts_missing += 1

        result = DocResult(
            doc_id=str(doc["doc_id"]),
            scores={"char_count": nc, "token_count": nt, "length_bucket": bucket},
            flags={
                "missing_timestamp": doc.get("timestamp") is None,
                "missing_url": doc.get("url") is None,
                "missing_language": doc.get("language") is None,
                "missing_source": doc.get("source") is None,
            },
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)
        n_docs += 1

    total = n_docs
    total_toks = sum(tok_counts)
    summary = {
        "total_docs": total,
        "char_stats": _dist_stats(char_counts),
        "token_stats": _dist_stats(tok_counts),
        "length_buckets": {
            b: {
                "count": bucket_cnt[b],
                "pct": round(bucket_cnt[b] / total, 4) if total else 0.0,
            }
            for b in _BUCKET_ORDER
        },
        "domain_distribution": _group_table(domain_docs, domain_toks, total_toks),
        "language_distribution": _group_table(lang_docs, lang_toks, total_toks),
        "source_distribution": _group_table(src_docs, src_toks, total_toks),
        "timestamp": {
            "present": ts_present,
            "missing": ts_missing,
            "present_pct": round(ts_present / total, 4) if total else 0.0,
            "year_month_distribution": dict(sorted(ym_dist.items())),
        },
    }
    return per_doc, summary


# ── Row 2: license detection ──────────────────────────────────────────────────

def compute_license(
    docs: Iterable[Document],
    timeout_per_doc: int = 30,
    max_docs: int | None = None,
) -> tuple[list[DocResult], dict]:
    """Detect licenses and copyrights using ScanCode Toolkit.

    Writes each document to a temp file, calls scancode.api, then cleans up.
    Requires: pip install scancode-toolkit
    """
    try:
        from scancode import api as sc
    except ImportError as exc:
        raise RuntimeError(
            "ScanCode not installed. Run: pip install scancode-toolkit"
        ) from exc

    doc_list = list(docs)
    if max_docs is not None:
        doc_list = doc_list[:max_docs]

    per_doc: list[DocResult] = []
    license_type_cnt: Counter = Counter()
    hit_docs = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for doc in doc_list:
            fpath = os.path.join(tmpdir, f"{doc['doc_id']}.txt")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(doc.get("text") or "")  # type: ignore[arg-type]

            lic_result = sc.get_licenses(fpath)
            cpy_result = sc.get_copyrights(fpath)

            detections = lic_result.get("license_detections", [])
            copyright_list = cpy_result.get("copyrights", [])

            hits = []
            for det in detections:
                spdx = det.get("license_expression_spdx") or "unknown"
                license_type_cnt[spdx] += 1
                for match in det.get("matches", []):
                    hits.append({
                        "spdx_expression": spdx,
                        "score": match.get("score"),
                        "start_line": match.get("start_line"),
                        "end_line": match.get("end_line"),
                        "rule_identifier": match.get("rule_identifier"),
                    })

            holders = [c.get("copyright") for c in copyright_list if c.get("copyright")]
            has_license = bool(hits)
            if has_license:
                hit_docs += 1

            per_doc.append(DocResult(
                doc_id=str(doc["doc_id"]),
                scores={
                    "license_hit_count": len(hits),
                    "license_hits": hits,
                    "copyright_holders": holders,
                },
                flags={
                    "has_license": has_license,
                    "has_copyright": bool(holders),
                },
            ))

    total = len(doc_list)
    summary = {
        "total_docs_scanned": total,
        "docs_with_license": hit_docs,
        "hit_pct": round(hit_docs / total, 4) if total else 0.0,
        "license_type_distribution": dict(license_type_cnt.most_common()),
    }
    return per_doc, summary
