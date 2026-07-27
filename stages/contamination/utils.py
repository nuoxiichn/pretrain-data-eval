"""Stage 5 computation utilities.

  compute_exact_contamination  — MD5-based exact match against benchmark index
  compute_code_near            — char 5-gram MinHash near-dup vs code benchmarks
  compute_code_ast             — tree-sitter AST fingerprint vs code benchmarks
"""

from __future__ import annotations

import hashlib
import struct
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np

from src.reader import Document
from src.schema import DocResult
from stages.contamination.benchmarks import BenchItem


# ── helpers ──────────────────────────────────────────────────────────────────

def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


# ── Exact contamination ─────────────────────────────────────────────────────

def build_bench_index(
    bench_items: dict[str, list[BenchItem]],
    paragraph_sep: str = "\n\n",
    min_para_chars: int = 50,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build MD5 lookup indexes from benchmark data.

    Returns (doc_index, para_index):
      doc_index  — MD5(full text) → set of benchmark labels
      para_index — MD5(paragraph) → set of benchmark labels
    """
    doc_index: dict[str, set[str]] = defaultdict(set)
    para_index: dict[str, set[str]] = defaultdict(set)

    for label, items in bench_items.items():
        for item in items:
            text = item["text"]
            code = item.get("code") or ""
            full = (text + "\n" + code).strip() if code else text

            doc_index[_md5(full)].add(label)
            doc_index[_md5(text)].add(label)
            if code:
                doc_index[_md5(code)].add(label)

            for part in (text, code):
                if not part:
                    continue
                paras = [p.strip() for p in part.split(paragraph_sep)
                         if len(p.strip()) >= min_para_chars]
                for para in paras:
                    para_index[_md5(para)].add(label)

    return dict(doc_index), dict(para_index)


def compute_exact_contamination(
    docs: Iterable[Document],
    bench_items: dict[str, list[BenchItem]],
    paragraph_sep: str = "\n\n",
    min_para_chars: int = 50,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Detect exact matches between training docs and benchmark data."""
    doc_index, para_index = build_bench_index(
        bench_items, paragraph_sep, min_para_chars,
    )

    doc_list = list(docs)
    per_doc: list[DocResult] = []
    contaminated_docs = 0
    para_contaminated_docs = 0
    bench_hit_counts: dict[str, int] = defaultdict(int)

    for doc in doc_list:
        doc_id = str(doc["doc_id"])
        text = str(doc.get("text") or "")
        dh = _md5(text)

        matched_benchmarks: list[str] = sorted(doc_index.get(dh, set()))
        is_exact = len(matched_benchmarks) > 0

        paras = [p.strip() for p in text.split(paragraph_sep)
                 if len(p.strip()) >= min_para_chars]
        para_matches = 0
        para_bench_hits: set[str] = set()
        for para in paras:
            ph = _md5(para)
            if ph in para_index:
                para_matches += 1
                para_bench_hits.update(para_index[ph])

        has_para = para_matches > 0
        para_ratio = round(para_matches / len(paras), 4) if paras else 0.0

        if is_exact:
            contaminated_docs += 1
        if has_para:
            para_contaminated_docs += 1

        all_hits = set(matched_benchmarks) | para_bench_hits
        for b in all_hits:
            bench_hit_counts[b] += 1

        result = DocResult(
            doc_id=doc_id,
            scores={
                "doc_hash": dh,
                "matched_benchmarks": matched_benchmarks,
                "para_match_count": para_matches,
                "para_match_ratio": para_ratio,
            },
            flags={
                "is_exact_contaminated": is_exact,
                "has_para_contamination": has_para,
            },
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    total = len(doc_list)
    summary = {
        "total_docs": total,
        "contaminated_docs": contaminated_docs,
        "contaminated_pct": round(contaminated_docs / total, 4) if total else 0.0,
        "para_contaminated_docs": para_contaminated_docs,
        "para_contaminated_pct": round(para_contaminated_docs / total, 4) if total else 0.0,
        "bench_index_doc_hashes": len(doc_index),
        "bench_index_para_hashes": len(para_index),
        "per_benchmark_hits": dict(bench_hit_counts),
    }
    return per_doc, summary


# ── Code near-dup (char n-gram MinHash) ──────────────────────────────────────

def _char_ngrams(text: str, n: int) -> set[str]:
    text = text.strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _minhash_signature(ngrams: set[str], num_hashes: int) -> np.ndarray:
    sig = np.full(num_hashes, 0xFFFFFFFF, dtype=np.uint32)
    for gram in ngrams:
        gram_bytes = gram.encode("utf-8", errors="replace")
        for i in range(num_hashes):
            seed = struct.pack("<I", i)
            h = int(hashlib.md5(seed + gram_bytes).hexdigest()[:8], 16) & 0xFFFFFFFF
            if h < sig[i]:
                sig[i] = h
    return sig


def _lsh_buckets(sig: np.ndarray, num_bands: int, band_size: int) -> list[tuple]:
    keys = []
    for b in range(num_bands):
        band = sig[b * band_size : (b + 1) * band_size]
        keys.append((b, band.tobytes()))
    return keys


def compute_code_near(
    docs: Iterable[Document],
    bench_items: dict[str, list[BenchItem]],
    ngram_size: int = 5,
    jaccard_threshold: float = 0.85,
    num_hashes: int = 128,
    num_bands: int = 16,
    band_size: int = 8,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Detect near-duplicate code between training docs and code benchmarks."""
    doc_list = list(docs)

    bench_sigs: list[tuple[str, str, np.ndarray]] = []
    for label, items in bench_items.items():
        for item in items:
            code = item.get("code")
            if not code:
                continue
            ngrams = _char_ngrams(code, ngram_size)
            if not ngrams:
                continue
            sig = _minhash_signature(ngrams, num_hashes)
            bench_sigs.append((item["bench_id"], label, sig))

    if not bench_sigs:
        empty_results: list[DocResult] = []
        for doc in doc_list:
            r = DocResult(
                doc_id=str(doc["doc_id"]),
                scores={"jaccard_max": 0.0, "matched_bench_id": "", "matched_benchmark": ""},
                flags={"is_code_near_dup": False},
            )
            if on_doc is not None:
                on_doc(r)
            else:
                empty_results.append(r)
        return empty_results, {
            "total_docs": len(doc_list), "code_near_dup_docs": 0,
            "code_near_dup_pct": 0.0, "bench_code_samples": 0,
        }

    bench_bucket_map: dict[tuple, list[int]] = defaultdict(list)
    for idx, (_, _, sig) in enumerate(bench_sigs):
        for key in _lsh_buckets(sig, num_bands, band_size):
            bench_bucket_map[key].append(idx)

    per_doc: list[DocResult] = []
    near_dup_docs = 0

    for doc in doc_list:
        doc_id = str(doc["doc_id"])
        text = str(doc.get("text") or "")

        ngrams = _char_ngrams(text, ngram_size)
        if not ngrams:
            result = DocResult(
                doc_id=doc_id,
                scores={"jaccard_max": 0.0, "matched_bench_id": "", "matched_benchmark": ""},
                flags={"is_code_near_dup": False},
            )
            if on_doc is not None:
                on_doc(result)
            else:
                per_doc.append(result)
            continue

        doc_sig = _minhash_signature(ngrams, num_hashes)

        candidates: set[int] = set()
        for key in _lsh_buckets(doc_sig, num_bands, band_size):
            if key in bench_bucket_map:
                candidates.update(bench_bucket_map[key])

        best_jaccard = 0.0
        best_bench_id = ""
        best_benchmark = ""

        for cidx in candidates:
            bench_id, label, bench_sig = bench_sigs[cidx]
            jaccard = float(np.mean(doc_sig == bench_sig))
            if jaccard > best_jaccard:
                best_jaccard = jaccard
                best_bench_id = bench_id
                best_benchmark = label

        is_near_dup = best_jaccard >= jaccard_threshold
        if is_near_dup:
            near_dup_docs += 1

        result = DocResult(
            doc_id=doc_id,
            scores={
                "jaccard_max": round(best_jaccard, 4),
                "matched_bench_id": best_bench_id,
                "matched_benchmark": best_benchmark,
            },
            flags={"is_code_near_dup": is_near_dup},
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    total = len(doc_list)
    summary = {
        "total_docs": total,
        "code_near_dup_docs": near_dup_docs,
        "code_near_dup_pct": round(near_dup_docs / total, 4) if total else 0.0,
        "bench_code_samples": len(bench_sigs),
        "jaccard_threshold": jaccard_threshold,
        "ngram_size": ngram_size,
        "num_hashes": num_hashes,
    }
    return per_doc, summary


# ── Code AST fingerprint ────────────────────────────────────────────────────

def _parse_ast_node_types(code: str, language: str = "python") -> list[str] | None:
    """Parse code with tree-sitter and return a list of AST node types.

    Strips identifier values, string literals, and comment bodies — keeps only
    the structural node type names so that variable-renamed / comment-rewritten
    copies produce the same fingerprint.

    Returns None if tree-sitter or the language grammar is not available, or if
    the code is empty / unparsable.
    """
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
    except ImportError:
        return None

    if not code.strip():
        return None

    lang = Language(tspython.language())
    parser = Parser(lang)
    tree = parser.parse(code.encode("utf-8", errors="replace"))

    node_types: list[str] = []
    _SKIP_VALUES = {"identifier", "string", "comment", "string_content",
                    "integer", "float", "true", "false", "none"}

    def _walk(node):
        node_types.append(node.type)
        if node.type not in _SKIP_VALUES:
            for child in node.children:
                _walk(child)

    _walk(tree.root_node)
    return node_types if len(node_types) > 1 else None


def _ast_fingerprint(node_types: list[str]) -> str:
    return _md5(" ".join(node_types))


def _node_type_set(node_types: list[str]) -> frozenset[str]:
    return frozenset(node_types)


def _jaccard_sets(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_code_ast(
    docs: Iterable[Document],
    bench_items: dict[str, list[BenchItem]],
    languages: list[str] | None = None,
    fingerprint_jaccard_threshold: float = 0.90,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Detect AST-structural contamination between training docs and code benchmarks."""
    if languages is None:
        languages = ["python"]

    doc_list = list(docs)

    bench_fps: dict[str, set[str]] = {}
    bench_node_sets: list[tuple[str, str, frozenset[str]]] = []

    for label, items in bench_items.items():
        for item in items:
            code = item.get("code")
            if not code:
                continue
            for lang in languages:
                ntypes = _parse_ast_node_types(code, lang)
                if ntypes is None:
                    continue
                fp = _ast_fingerprint(ntypes)
                if fp not in bench_fps:
                    bench_fps[fp] = set()
                bench_fps[fp].add(item["bench_id"])
                bench_node_sets.append((item["bench_id"], label, _node_type_set(ntypes)))

    per_doc: list[DocResult] = []
    ast_contaminated_docs = 0
    fingerprint_hits = 0

    for doc in doc_list:
        doc_id = str(doc["doc_id"])
        text = str(doc.get("text") or "")

        best_jaccard = 0.0
        best_bench_id = ""
        doc_fp = ""
        fp_match = False

        for lang in languages:
            ntypes = _parse_ast_node_types(text, lang)
            if ntypes is None:
                continue

            fp = _ast_fingerprint(ntypes)
            doc_fp = fp

            if fp in bench_fps:
                fp_match = True
                best_jaccard = 1.0
                best_bench_id = next(iter(bench_fps[fp]))
                break

            doc_nset = _node_type_set(ntypes)
            for bench_id, label, bench_nset in bench_node_sets:
                j = _jaccard_sets(doc_nset, bench_nset)
                if j > best_jaccard:
                    best_jaccard = j
                    best_bench_id = bench_id

        is_contaminated = fp_match or best_jaccard >= fingerprint_jaccard_threshold
        if is_contaminated:
            ast_contaminated_docs += 1
        if fp_match:
            fingerprint_hits += 1

        result = DocResult(
            doc_id=doc_id,
            scores={
                "ast_fingerprint": doc_fp,
                "ast_jaccard_max": round(best_jaccard, 4),
                "matched_bench_id": best_bench_id,
                "fingerprint_exact_match": fp_match,
            },
            flags={"is_ast_contaminated": is_contaminated},
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    total = len(doc_list)
    summary = {
        "total_docs": total,
        "ast_contaminated_docs": ast_contaminated_docs,
        "ast_contaminated_pct": round(ast_contaminated_docs / total, 4) if total else 0.0,
        "fingerprint_exact_hits": fingerprint_hits,
        "bench_fingerprints": len(bench_fps),
        "bench_node_set_samples": len(bench_node_sets),
        "fingerprint_jaccard_threshold": fingerprint_jaccard_threshold,
        "languages": languages,
    }
    return per_doc, summary
