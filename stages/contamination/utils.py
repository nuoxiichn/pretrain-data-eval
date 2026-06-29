"""Stage 5 computation utilities.

  compute_exact_contamination  — MD5-based exact match against benchmark index
  compute_near_contamination   — generic char-MinHash near-dup vs any benchmark text
  compute_code_near            — char 5-gram MinHash near-dup vs code benchmarks
  compute_code_ast             — tree-sitter AST fingerprint vs code benchmarks

MinHash 实现：xxhash + universal hashing（复用 stages/dedup/utils.py 的 _mh_permutations /
_minhash_signature_fast，比旧 MD5 实现快约 50×）。LSH 桶布局与 _lsh_buckets 保持。
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np

from src.reader import Document
from src.schema import DocResult
from stages.contamination.benchmarks import BenchItem
from stages.dedup.utils import _mh_permutations, _minhash_signature_fast


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

def _char_ngrams(text: str, n: int) -> list[str]:
    """Char-level n-grams as list (whitespace 不去除，与签名 reproducibility 一致)."""
    text = text.strip()
    if len(text) < n:
        return [text] if text else []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def _minhash_signature(ngrams: list[str], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Char-ngram → MinHash signature via xxhash + universal hashing (复用 dedup 实现)."""
    return _minhash_signature_fast(ngrams, a, b)


def _lsh_buckets(sig: np.ndarray, num_bands: int, band_size: int) -> list[tuple]:
    """切签名为 num_bands 个 band，返回 [(band_idx, band_bytes), ...] 作为桶 key.
    与 benchmark 端建索引的桶 key 必须保持一致。"""
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
    a, b = _mh_permutations(num_hashes, seed=1)

    bench_sigs: list[tuple[str, str, np.ndarray]] = []
    for label, items in bench_items.items():
        for item in items:
            code = item.get("code")
            if not code:
                continue
            ngrams = _char_ngrams(code, ngram_size)
            if not ngrams:
                continue
            sig = _minhash_signature(ngrams, a, b)
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

        doc_sig = _minhash_signature(ngrams, a, b)

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


# ── Generic text near-dup (L2 of cascade) ────────────────────────────────────
# 与 compute_code_near 区别：
#   1. benchmark 端不限 code_field，对每个 BenchItem 都建 signature（text + code 各一条）
#   2. doc 端 scores 多返 jaccard_max_per_benchmark（每个 benchmark 的最高 jaccard，供 cascade 决策）
#   3. 阈值判定走通用 jaccard_threshold；flag 名 is_near_contaminated
#   4. 支持 doc 端 sliding window：UFW doc 平均 745 chars，benchmark 仅 ~60 chars，
#      doc-level Jaccard 必然 ≈ 0（分母被 doc 撑大）。window 切片后每窗与 benchmark
#      长度量级匹配，再取 max(jaccard)，恢复 "benchmark 嵌入长文" 这种污染的召回。
#
# 用法：L2 cascade 单独跑 → 拿到 jaccard_max + per_benchmark 分布；cascade 路由用 jaccard_max


def _doc_windows(text: str, window_size: int, stride: int) -> list[str]:
    """把长文档切成若干 char 窗口，长度 ≤ window_size 不切（整体作为一个窗口返回）.

    步长 stride < window_size 时窗口之间有重叠（避免 benchmark 题目跨边界）.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= window_size or window_size <= 0 or stride <= 0:
        return [text]
    out: list[str] = []
    pos = 0
    while pos < len(text):
        out.append(text[pos : pos + window_size])
        if pos + window_size >= len(text):
            break
        pos += stride
    return out


def _query_l2_index(
    doc_sig: np.ndarray, bench_sigs: list, bucket_map: dict,
    num_bands: int, band_size: int,
) -> tuple[float, str, str, str, dict]:
    """给 doc signature 查 LSH 桶找候选 benchmark item，返回 (best_jaccard, bench_id, label, kind, per_bench_max).
    """
    candidates: set[int] = set()
    for key in _lsh_buckets(doc_sig, num_bands, band_size):
        if key in bucket_map:
            candidates.update(bucket_map[key])
    best_jaccard = 0.0
    best_bench_id = best_benchmark = best_kind = ""
    per_bench: dict[str, float] = {}
    for cidx in candidates:
        bench_id_c, label_c, kind_c, sig_c = bench_sigs[cidx]
        j = float(np.mean(doc_sig == sig_c))
        if j > per_bench.get(label_c, 0.0):
            per_bench[label_c] = j
        if j > best_jaccard:
            best_jaccard = j
            best_bench_id = bench_id_c
            best_benchmark = label_c
            best_kind = kind_c
    return best_jaccard, best_bench_id, best_benchmark, best_kind, per_bench


def _doc_l2_score(
    text: str, l2_index: dict,
    *, window_size: int, stride: int,
) -> tuple[float, str, str, str, dict]:
    """对 doc 文本算 L2 分数（支持 sliding window 模式）.

    window_size <= 0 → 整 doc 一次 signature（旧行为，兼容）.
    window_size > 0  → 切窗口，每个窗口独立 signature，取 max(jaccard).
    """
    a = l2_index["a"]; b = l2_index["b"]
    bench_sigs = l2_index["bench_sigs"]; bucket_map = l2_index["bucket_map"]
    num_bands = l2_index["num_bands"]; band_size = l2_index["band_size"]
    ngram_size = l2_index["ngram_size"]

    if not bench_sigs or not text:
        return 0.0, "", "", "", {}

    if window_size <= 0:
        windows = [text]
    else:
        windows = _doc_windows(text, window_size, stride)

    best_jaccard = 0.0
    best_bench_id = best_benchmark = best_kind = ""
    merged_per_bench: dict[str, float] = {}
    for w in windows:
        ngrams = _char_ngrams(w, ngram_size)
        if not ngrams:
            continue
        sig = _minhash_signature(ngrams, a, b)
        j, bid, lbl, kind, pb = _query_l2_index(
            sig, bench_sigs, bucket_map, num_bands, band_size,
        )
        for k, v in pb.items():
            if v > merged_per_bench.get(k, 0.0):
                merged_per_bench[k] = v
        if j > best_jaccard:
            best_jaccard = j
            best_bench_id = bid
            best_benchmark = lbl
            best_kind = kind
    return best_jaccard, best_bench_id, best_benchmark, best_kind, merged_per_bench


def build_bench_minhash(
    bench_items: dict[str, list[BenchItem]],
    *,
    ngram_size: int = 5,
    num_hashes: int = 128,
    num_bands: int = 16,
    band_size: int = 8,
    perm_seed: int = 1,
) -> dict:
    """Build MinHash signatures + LSH bucket map for ALL benchmark items.

    benchmark item 同时取 text 和 code 字段（任意非空字段建一个 signature 行），
    跨语言污染检测时 caller 应把 EN benchmark 的中文翻译版作为独立 dataset 加入。

    Returns a dict with:
      a, b               — universal hashing 参数（doc 端要复用）
      ngram_size, num_hashes, num_bands, band_size — 索引参数
      bench_sigs         — [(bench_id, label, content_kind, sig), ...]
      bucket_map         — dict[(band_idx, band_bytes)] → [idx_in_bench_sigs, ...]
    """
    a, b = _mh_permutations(num_hashes, seed=perm_seed)
    bench_sigs: list[tuple[str, str, str, np.ndarray]] = []
    for label, items in bench_items.items():
        for item in items:
            for kind in ("text", "code"):
                content = item.get(kind) or ""
                if not content:
                    continue
                ngrams = _char_ngrams(content, ngram_size)
                if not ngrams:
                    continue
                sig = _minhash_signature(ngrams, a, b)
                bench_sigs.append((item["bench_id"], label, kind, sig))

    bucket_map: dict[tuple, list[int]] = defaultdict(list)
    for idx, (_, _, _, sig) in enumerate(bench_sigs):
        for key in _lsh_buckets(sig, num_bands, band_size):
            bucket_map[key].append(idx)

    return {
        "a": a, "b": b,
        "ngram_size": ngram_size, "num_hashes": num_hashes,
        "num_bands": num_bands, "band_size": band_size,
        "bench_sigs": bench_sigs,
        "bucket_map": dict(bucket_map),
    }


def compute_near_contamination(
    docs: Iterable[Document],
    bench_items: dict[str, list[BenchItem]],
    *,
    ngram_size: int = 5,
    jaccard_threshold: float = 0.85,
    num_hashes: int = 128,
    num_bands: int = 16,
    band_size: int = 8,
    window_size: int = 0,                # 0 = doc 整体；>0 = char sliding window
    window_stride: int = 0,
    on_doc: Callable[[DocResult], None] | None = None,
    bench_index: dict | None = None,
) -> tuple[list[DocResult], dict]:
    """L2 通用文本 near-dup 污染检测.

    window_size > 0 时启用 sliding window（UFW doc 平均 745 chars vs benchmark ~60 chars，
    必须切窗才能恢复嵌入式污染的召回）.
    """
    if bench_index is None:
        bench_index = build_bench_minhash(
            bench_items,
            ngram_size=ngram_size, num_hashes=num_hashes,
            num_bands=num_bands, band_size=band_size,
        )

    doc_list = list(docs)
    per_doc: list[DocResult] = []
    near_dup_docs = 0
    per_benchmark_hits: dict[str, int] = defaultdict(int)

    for doc in doc_list:
        doc_id = str(doc["doc_id"])
        text = str(doc.get("text") or "")
        best_jaccard, best_bench_id, best_benchmark, best_kind, per_bench_max = (
            _doc_l2_score(text, bench_index,
                          window_size=window_size, stride=window_stride)
        )
        is_near = best_jaccard >= jaccard_threshold
        if is_near:
            near_dup_docs += 1
            per_benchmark_hits[best_benchmark] += 1

        result = DocResult(
            doc_id=doc_id,
            scores={
                "jaccard_max": round(best_jaccard, 4),
                "matched_bench_id": best_bench_id,
                "matched_benchmark": best_benchmark,
                "matched_kind": best_kind,
                "jaccard_max_per_benchmark": {k: round(v, 4) for k, v in per_bench_max.items()},
            },
            flags={"is_near_contaminated": is_near},
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    total = len(doc_list)
    summary = {
        "total_docs": total,
        "near_contaminated_docs": near_dup_docs,
        "near_contaminated_pct": round(near_dup_docs / total, 4) if total else 0.0,
        "per_benchmark_hits": dict(per_benchmark_hits),
        "bench_samples": len(bench_index["bench_sigs"]),
        "jaccard_threshold": jaccard_threshold,
        "ngram_size": ngram_size,
        "num_hashes": num_hashes,
        "num_bands": num_bands,
        "band_size": band_size,
        "window_size": window_size,
        "window_stride": window_stride,
    }
    return per_doc, summary


# ── Embedding contamination (L3 of cascade) ──────────────────────────────────

def _encode_texts_bge(
    texts: list[str], model_path: str,
    *, batch_size: int = 64, max_length: int = 512, device: str = "cuda",
) -> np.ndarray:
    """BGE-style encode (CLS pooling + L2 norm). 复用 stages/dedup/utils._encode_embeddings 的模式.

    返回 numpy float32 [N, dim]（CPU 上，供 FAISS 索引）.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    if device != "cpu":
        model = model.half()

    out_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, dim=1)
            out_chunks.append(cls.float().cpu().numpy())
    if not out_chunks:
        return np.empty((0, 1024), dtype=np.float32)
    return np.concatenate(out_chunks, axis=0)


def build_bench_embeddings(
    bench_items: dict[str, list[BenchItem]],
    model_path: str,
    *, batch_size: int = 64, max_length: int = 512, device: str = "cuda",
) -> dict:
    """对每个 benchmark item 的 text + code 各编码一次（任意非空字段编一行）.

    Returns {'embeddings': [N, dim], 'meta': [(bench_id, label, kind), ...], 'faiss_index', 'dim'}
    """
    import faiss

    texts: list[str] = []
    meta: list[tuple[str, str, str]] = []
    for label, items in bench_items.items():
        for item in items:
            for kind in ("text", "code"):
                content = (item.get(kind) or "").strip()
                if not content:
                    continue
                texts.append(content)
                meta.append((item["bench_id"], label, kind))

    emb = _encode_texts_bge(
        texts, model_path,
        batch_size=batch_size, max_length=max_length, device=device,
    )
    dim = int(emb.shape[1]) if emb.size else 1024
    index = faiss.IndexFlatIP(dim)
    if emb.size:
        index.add(emb)
    return {"embeddings": emb, "meta": meta, "faiss_index": index, "dim": dim}


def compute_embed_contamination(
    docs: Iterable[Document],
    bench_items: dict[str, list[BenchItem]],
    *,
    model_path: str = "/mnt/public/model/bge-m3",
    cos_threshold: float = 0.85,
    top_k: int = 5,
    batch_size: int = 64,
    max_length: int = 512,
    device: str = "cuda",
    on_doc: Callable[[DocResult], None] | None = None,
    bench_embed: dict | None = None,
    candidate_filter: Callable[[Document], bool] | None = None,
) -> tuple[list[DocResult], dict]:
    """L3 embedding-based contamination via BGE-m3 + FAISS IndexFlatIP.

    candidate_filter: cascade 编排时由 L2 jaccard 决定该 doc 是否进 L3；
      返回 False 的 doc 不算 embedding，直接出 placeholder DocResult.
    """
    if bench_embed is None:
        bench_embed = build_bench_embeddings(
            bench_items, model_path,
            batch_size=batch_size, max_length=max_length, device=device,
        )
    index = bench_embed["faiss_index"]
    meta = bench_embed["meta"]

    doc_list = list(docs)
    if candidate_filter is not None:
        wanted_idx = [i for i, d in enumerate(doc_list) if candidate_filter(d)]
    else:
        wanted_idx = list(range(len(doc_list)))

    wanted_texts = [str(doc_list[i].get("text") or "") for i in wanted_idx]
    if wanted_texts:
        doc_embs = _encode_texts_bge(
            wanted_texts, model_path,
            batch_size=batch_size, max_length=max_length, device=device,
        )
    else:
        doc_embs = np.empty((0, bench_embed.get("dim", 1024)), dtype=np.float32)

    if doc_embs.size and index.ntotal > 0:
        k = min(top_k, index.ntotal)
        sims, idxs = index.search(doc_embs, k)
    else:
        sims = np.zeros((len(wanted_texts), 0), dtype=np.float32)
        idxs = np.zeros((len(wanted_texts), 0), dtype=np.int64)

    results_per_idx: dict[int, tuple[float, str, str, str]] = {}
    for local_i, doc_i in enumerate(wanted_idx):
        if sims.shape[1] == 0:
            results_per_idx[doc_i] = (0.0, "", "", "")
            continue
        best_local = int(np.argmax(sims[local_i]))
        cos_max = float(sims[local_i, best_local])
        bench_meta_i = int(idxs[local_i, best_local])
        bench_id, label, kind = meta[bench_meta_i] if 0 <= bench_meta_i < len(meta) else ("", "", "")
        results_per_idx[doc_i] = (cos_max, bench_id, label, kind)

    per_doc: list[DocResult] = []
    semantic_dup_docs = 0
    per_benchmark_hits: dict[str, int] = defaultdict(int)

    for i, doc in enumerate(doc_list):
        doc_id = str(doc["doc_id"])
        if i in results_per_idx:
            cos_max, bench_id, label, kind = results_per_idx[i]
            l3_run = True
        else:
            cos_max, bench_id, label, kind = 0.0, "", "", ""
            l3_run = False

        is_semantic = cos_max >= cos_threshold and l3_run
        if is_semantic:
            semantic_dup_docs += 1
            per_benchmark_hits[label] += 1

        result = DocResult(
            doc_id=doc_id,
            scores={
                "cos_max": round(cos_max, 4),
                "matched_bench_id": bench_id,
                "matched_benchmark": label,
                "matched_kind": kind,
                "l3_run": l3_run,
            },
            flags={"is_semantic_contaminated": is_semantic},
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    total = len(doc_list)
    summary = {
        "total_docs": total,
        "l3_processed": len(wanted_idx),
        "semantic_contaminated_docs": semantic_dup_docs,
        "semantic_contaminated_pct": round(semantic_dup_docs / total, 4) if total else 0.0,
        "per_benchmark_hits": dict(per_benchmark_hits),
        "bench_samples": len(meta),
        "cos_threshold": cos_threshold,
        "top_k": top_k,
        "model_path": model_path,
    }
    return per_doc, summary


# ── Cascade orchestrator (L1 → L2 → L3) ──────────────────────────────────────

def compute_cascade_contamination(
    docs: Iterable[Document],
    bench_items: dict[str, list[BenchItem]] | None = None,
    *,
    # 预建索引（若传入则跳过对应层的 build_*）
    prebuilt_index: dict | None = None,
    # L1 (exact)
    paragraph_sep: str = "\n\n",
    min_para_chars: int = 50,
    # L2 (near MinHash)
    l2_ngram_size: int = 5,
    l2_num_hashes: int = 128,
    l2_num_bands: int = 16,
    l2_band_size: int = 8,
    l2_window_size: int = 150,
    l2_window_stride: int = 75,
    l2_enter_l3_low: float = 0.30,
    l2_enter_l3_high: float = 0.90,
    # L3 (embedding)
    l3_model_path: str = "/mnt/public/model/bge-m3",
    l3_top_k: int = 5,
    l3_batch_size: int = 64,
    l3_max_length: int = 512,
    l3_device: str = "cuda",
    l3_cos_red: float = 0.85,
    l3_cos_yellow: float = 0.70,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """三层 cascade 污染检测 (L1 exact → L2 MinHash → L3 embedding).

    路由规则（按顺序判定）:
        L1 exact_hit                      → verdict=red, 不进 L2/L3
        L2 jaccard >= enter_l3_high       → verdict=red, 不进 L3
        L2 jaccard <  enter_l3_low        → verdict=green, 不进 L3
        L2 jaccard in [low, high)         → 进 L3：
            L3 cos >= cos_red             → verdict=red
            L3 cos in [cos_yellow, cos_red) → verdict=yellow
            L3 cos <  cos_yellow          → verdict=green
        L2 hard hit (jaccard >= jaccard_threshold=high) 但未进 L3 时也算 red.

    DocResult 包含每层结果 + 综合 verdict + 跨语言标记（基于 L3 matched_kind / label）.
    Summary 含 cost_breakdown（每层处理 docs 数）+ verdict_distribution 三色分布.
    """
    # ── L1 端：构建 exact hash 索引 ──
    if prebuilt_index is not None and "hash_index" in prebuilt_index:
        doc_hash_index = prebuilt_index["hash_index"]["doc_index"]
        para_hash_index = prebuilt_index["hash_index"]["para_index"]
    else:
        doc_hash_index, para_hash_index = build_bench_index(
            bench_items, paragraph_sep=paragraph_sep, min_para_chars=min_para_chars,
        )

    # ── L2 端：构建 MinHash + LSH 桶 ──
    if prebuilt_index is not None and "l2_index" in prebuilt_index:
        l2_index = prebuilt_index["l2_index"]
    else:
        l2_index = build_bench_minhash(
            bench_items,
            ngram_size=l2_ngram_size, num_hashes=l2_num_hashes,
            num_bands=l2_num_bands, band_size=l2_band_size,
        )

    # ── L3 端：构建 BGE-m3 + FAISS 索引 ──
    if prebuilt_index is not None and "l3_embed" in prebuilt_index:
        l3_embed = prebuilt_index["l3_embed"]
    else:
        l3_embed = build_bench_embeddings(
            bench_items, l3_model_path,
            batch_size=l3_batch_size, max_length=l3_max_length, device=l3_device,
        )

    doc_list = list(docs)
    total = len(doc_list)

    # 先把 L1 + L2 在 CPU 上跑完，记录每篇 doc 是否需要进 L3
    layer_scores: list[dict] = []
    l3_candidate_local_idx: list[int] = []
    l1_red_count = 0
    l2_red_count = 0
    l2_green_count = 0

    bench_sigs_l2 = l2_index["bench_sigs"]

    for i, doc in enumerate(doc_list):
        text = str(doc.get("text") or "")
        doc_id = str(doc["doc_id"])

        # ── L1 ──
        dh = _md5(text)
        l1_matched = sorted(doc_hash_index.get(dh, set()))
        l1_hit = len(l1_matched) > 0
        paras = [p.strip() for p in text.split(paragraph_sep)
                 if len(p.strip()) >= min_para_chars]
        para_hits_count = 0
        para_bench_hits: set[str] = set()
        for para in paras:
            ph = _md5(para)
            if ph in para_hash_index:
                para_hits_count += 1
                para_bench_hits.update(para_hash_index[ph])
        l1_para_hit = para_hits_count > 0

        # ── L2 (sliding window) ──
        l2_best_jaccard, l2_best_bench_id, l2_best_benchmark, l2_best_kind, _ = (
            _doc_l2_score(text, l2_index,
                          window_size=l2_window_size, stride=l2_window_stride)
        )

        # ── 路由决策 ──
        # L1 命中 → red，不进 L3
        # L2 jaccard ≥ high → red，不进 L3
        # L2 jaccard < low → green，不进 L3
        # 否则 → 进 L3，verdict 待定

        if l1_hit or l1_para_hit:
            need_l3 = False
            preliminary_verdict = "red"
            l1_red_count += 1
        elif l2_best_jaccard >= l2_enter_l3_high:
            need_l3 = False
            preliminary_verdict = "red"
            l2_red_count += 1
        elif l2_best_jaccard < l2_enter_l3_low:
            need_l3 = False
            preliminary_verdict = "green"
            l2_green_count += 1
        else:
            need_l3 = True
            preliminary_verdict = "pending"

        if need_l3:
            l3_candidate_local_idx.append(i)

        layer_scores.append({
            "l1_doc_hit": l1_hit,
            "l1_para_hit": l1_para_hit,
            "l1_matched_benchmarks": l1_matched,
            "l1_para_matched_benchmarks": sorted(para_bench_hits),
            "l1_para_match_count": para_hits_count,
            "l2_jaccard_max": round(l2_best_jaccard, 4),
            "l2_matched_bench_id": l2_best_bench_id,
            "l2_matched_benchmark": l2_best_benchmark,
            "l2_matched_kind": l2_best_kind,
            "preliminary_verdict": preliminary_verdict,
            "need_l3": need_l3,
        })

    # ── L3 端：批量算 embedding ──
    l3_texts = [str(doc_list[i].get("text") or "") for i in l3_candidate_local_idx]
    if l3_texts:
        doc_embs = _encode_texts_bge(
            l3_texts, l3_model_path,
            batch_size=l3_batch_size, max_length=l3_max_length, device=l3_device,
        )
    else:
        doc_embs = np.empty((0, l3_embed.get("dim", 1024)), dtype=np.float32)

    index = l3_embed["faiss_index"]
    meta = l3_embed["meta"]
    if doc_embs.size and index.ntotal > 0:
        k = min(l3_top_k, index.ntotal)
        sims, idxs = index.search(doc_embs, k)
    else:
        sims = np.zeros((len(l3_texts), 0), dtype=np.float32)
        idxs = np.zeros((len(l3_texts), 0), dtype=np.int64)

    # 把 L3 结果回填到 layer_scores
    for local_j, doc_i in enumerate(l3_candidate_local_idx):
        if sims.shape[1] == 0:
            l3_cos, l3_id, l3_lbl, l3_kind = 0.0, "", "", ""
        else:
            best_k = int(np.argmax(sims[local_j]))
            l3_cos = float(sims[local_j, best_k])
            bench_meta_i = int(idxs[local_j, best_k])
            l3_id, l3_lbl, l3_kind = meta[bench_meta_i] if 0 <= bench_meta_i < len(meta) else ("", "", "")

        score = layer_scores[doc_i]
        score["l3_cos_max"] = round(l3_cos, 4)
        score["l3_matched_bench_id"] = l3_id
        score["l3_matched_benchmark"] = l3_lbl
        score["l3_matched_kind"] = l3_kind
        score["l3_run"] = True
        # 判定 L3 verdict
        if l3_cos >= l3_cos_red:
            score["preliminary_verdict"] = "red"
        elif l3_cos >= l3_cos_yellow:
            score["preliminary_verdict"] = "yellow"
        else:
            score["preliminary_verdict"] = "green"

    # ── 综合 DocResult ──
    per_doc: list[DocResult] = []
    verdict_dist: dict[str, int] = defaultdict(int)
    cross_lingual_docs = 0
    per_benchmark_red: dict[str, int] = defaultdict(int)
    cost_breakdown = {"l1_processed": total, "l2_processed": total,
                      "l3_processed": len(l3_candidate_local_idx)}

    for i, doc in enumerate(doc_list):
        doc_id = str(doc["doc_id"])
        score = layer_scores[i]
        verdict = score["preliminary_verdict"]
        verdict_dist[verdict] += 1

        # 跨语言污染标识: doc.language ≠ matched benchmark 推断语言
        # 简化策略：若 L1/L2/L3 命中的 benchmark label 是英文 benchmark（mmlu / hellaswag /
        # gsm8k / humaneval / arc_challenge）且 doc.language == "zh"，则记跨语言.
        doc_lang = doc.get("language") or ""
        EN_BENCHES = {"mmlu", "hellaswag", "gsm8k", "humaneval", "arc_challenge", "mbpp", "alpaca_eval"}
        matched_bench = (
            score["l1_matched_benchmarks"][0] if score.get("l1_matched_benchmarks")
            else score.get("l2_matched_benchmark") or score.get("l3_matched_benchmark", "")
        )
        is_cross_lingual = (
            verdict in ("red", "yellow")
            and doc_lang == "zh"
            and matched_bench in EN_BENCHES
        )
        if is_cross_lingual:
            cross_lingual_docs += 1
        if verdict == "red" and matched_bench:
            per_benchmark_red[matched_bench] += 1

        scores_out = {
            "l1_doc_hit": score["l1_doc_hit"],
            "l1_para_hit": score["l1_para_hit"],
            "l1_matched_benchmarks": score["l1_matched_benchmarks"],
            "l1_para_matched_benchmarks": score["l1_para_matched_benchmarks"],
            "l1_para_match_count": score["l1_para_match_count"],
            "l2_jaccard_max": score["l2_jaccard_max"],
            "l2_matched_bench_id": score["l2_matched_bench_id"],
            "l2_matched_benchmark": score["l2_matched_benchmark"],
            "l3_cos_max": score.get("l3_cos_max"),
            "l3_matched_bench_id": score.get("l3_matched_bench_id"),
            "l3_matched_benchmark": score.get("l3_matched_benchmark"),
            "l3_run": score.get("l3_run", False),
        }
        flags_out = {
            "is_exact_contaminated": score["l1_doc_hit"] or score["l1_para_hit"],
            "is_near_contaminated": (
                score["l2_jaccard_max"] >= l2_enter_l3_high
            ),
            "is_semantic_contaminated": (
                score.get("l3_run", False)
                and (score.get("l3_cos_max") or 0.0) >= l3_cos_red
            ),
            "is_cross_lingual": is_cross_lingual,
            "verdict": verdict,
        }

        result = DocResult(doc_id=doc_id, scores=scores_out, flags=flags_out)
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    summary = {
        "total_docs": total,
        "cost_breakdown": cost_breakdown,
        "verdict_distribution": dict(verdict_dist),
        "cross_lingual_docs": cross_lingual_docs,
        "per_benchmark_red_hits": dict(per_benchmark_red),
        "thresholds": {
            "l2_enter_l3_low": l2_enter_l3_low,
            "l2_enter_l3_high": l2_enter_l3_high,
            "l3_cos_red": l3_cos_red,
            "l3_cos_yellow": l3_cos_yellow,
        },
        "bench_l2_samples": len(bench_sigs_l2),
        "bench_l3_samples": len(meta),
        "l1_red_count": l1_red_count,
        "l2_red_count": l2_red_count,
        "l2_green_count": l2_green_count,
        "l3_model_path": l3_model_path,
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
