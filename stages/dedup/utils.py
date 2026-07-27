"""Stage 4 computation utilities.

  compute_exact_dedup   — MD5-based exact dedup (doc-level + paragraph-level)
  compute_minhash_dedup — MinHash + LSH near-dedup (word n-gram)
  compute_semdedup      — embedding semantic near-dedup (bge-m3 + cosine clustering)
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np

from src.reader import Document
from src.schema import DocResult


# ── Exact dedup ───────────────────────────────────────────────────────────────

def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


def compute_exact_dedup(
    docs: Iterable[Document],
    paragraph_sep: str = "\n\n",
    min_para_chars: int = 50,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Detect exact duplicates at document and paragraph level using MD5 hashes."""
    doc_list = list(docs)

    # ── pass 1: build hash → doc_id groups ────────────────────────────────────
    doc_hash_map: dict[str, list[str]] = defaultdict(list)
    para_hash_map: dict[str, list[str]] = defaultdict(list)  # hash → [doc_id, ...]

    doc_hashes: list[tuple[str, str]] = []  # (doc_id, hash)
    doc_para_hashes: dict[str, list[str]] = {}  # doc_id → [para_hash, ...]

    for doc in doc_list:
        doc_id = str(doc["doc_id"])
        text = str(doc.get("text") or "")
        dh = _md5(text)
        doc_hash_map[dh].append(doc_id)
        doc_hashes.append((doc_id, dh))

        paras = [p.strip() for p in text.split(paragraph_sep) if len(p.strip()) >= min_para_chars]
        ph_list = [_md5(p) for p in paras]
        doc_para_hashes[doc_id] = ph_list
        for ph in ph_list:
            para_hash_map[ph].append(doc_id)

    dup_doc_hashes = {h for h, ids in doc_hash_map.items() if len(ids) > 1}
    dup_para_hashes = {h for h, ids in para_hash_map.items() if len(ids) > 1}

    # ── pass 2: build per-doc results ─────────────────────────────────────────
    per_doc: list[DocResult] = []
    exact_dup_docs = 0
    para_dup_docs = 0

    for doc_id, dh in doc_hashes:
        is_exact_dup = dh in dup_doc_hashes
        ph_list = doc_para_hashes[doc_id]
        dup_para_count = sum(1 for ph in ph_list if ph in dup_para_hashes)
        has_para_dups = dup_para_count > 0
        para_dup_ratio = round(dup_para_count / len(ph_list), 4) if ph_list else 0.0

        if is_exact_dup:
            exact_dup_docs += 1
        if has_para_dups:
            para_dup_docs += 1

        result = DocResult(
            doc_id=doc_id,
            scores={
                "doc_hash": dh,
                "dup_doc_count": len(doc_hash_map[dh]),
                "para_dup_count": dup_para_count,
                "para_dup_ratio": para_dup_ratio,
            },
            flags={"is_exact_dup": is_exact_dup, "has_para_dups": has_para_dups},
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    total = len(doc_list)
    summary = {
        "total_docs": total,
        "exact_dup_docs": exact_dup_docs,
        "exact_dup_pct": round(exact_dup_docs / total, 4) if total else 0.0,
        "unique_doc_hashes": len(doc_hash_map),
        "para_dup_docs": para_dup_docs,
        "para_dup_pct": round(para_dup_docs / total, 4) if total else 0.0,
        "unique_para_hashes": len(para_hash_map),
    }
    return per_doc, summary


# ── MinHash near-dedup ────────────────────────────────────────────────────────
# Two-phase streaming impl, fits 100K–1M docs on a single box.
# Phase 1: stream docs → signatures.npy + doc_ids.jsonl (text dropped after sig)
# Phase 2: mmap signatures → banded LSH with per-bucket cap → Jaccard verify
#
# 旧实现 OOM 根因（在 100K 跨文件样本上 SIGKILL exit=137）：
#   1) 每 ngram 算 num_hashes 次 MD5 → CPU 长时间占内存；
#   2) bucket_map 在 boilerplate 上形成超热桶，O(B²) 候选对爆 set；
#   3) 全部 doc + signature 常驻 RAM。
# 新方案：xxhash + universal hashing 加速；签名落盘 mmap；hot_bucket_cap 截断。

_MH_PRIME = (1 << 61) - 1
_MH_MAX = (1 << 32) - 1


def _mh_permutations(num_hashes: int, seed: int = 1):
    rng = np.random.RandomState(seed)
    a = rng.randint(1, _MH_PRIME, size=num_hashes, dtype=np.uint64)
    b = rng.randint(0, _MH_PRIME, size=num_hashes, dtype=np.uint64)
    return a, b


_CJK_FRAC_THRESH = 0.3  # 文本中 CJK 字符占比超过此值视作中日韩文，走字符级 n-gram


def _is_cjk_dominant(text: str) -> bool:
    if not text:
        return False
    cjk = sum(1 for c in text if "　" <= c <= "鿿" or "가" <= c <= "힯")
    return cjk / max(len(text), 1) > _CJK_FRAC_THRESH


def _word_ngrams(text: str, n: int) -> list[str]:
    """Word n-grams for whitespace-tokenized langs; char n-grams for CJK.

    CJK 文本走 split() 几乎只剩 1-6 个"词"（无空格），所以按 5n 字符宽度
    切 char-level shingles（n=5 → 25 字符），与 word n-gram 的语义覆盖近似。
    """
    if _is_cjk_dominant(text):
        m = n * 5
        s = text.replace(" ", "").replace("\n", "").replace("\t", "")
        if len(s) < m:
            return [s] if s else []
        return [s[i : i + m] for i in range(len(s) - m + 1)]
    words = text.split()
    if len(words) < n:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _minhash_signature_fast(ngrams: list[str], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """xxhash + (a·h + b) mod prime 派生 K 个 MinHash 通道。比旧 MD5 实现快约 50×。"""
    import xxhash
    K = a.shape[0]
    if not ngrams:
        return np.full(K, _MH_MAX, dtype=np.uint32)
    base = {xxhash.xxh3_64_intdigest(g.encode("utf-8", "replace")) for g in ngrams}
    h = np.fromiter(base, dtype=np.uint64, count=len(base))
    perm = (np.outer(a, h) + b[:, None]) % _MH_PRIME
    return perm.min(axis=1).astype(np.uint32)


def compute_minhash_signatures(
    docs: Iterable[Document],
    out_dir,
    *,
    num_hashes: int = 64,
    ngram_size: int = 5,
    min_words: int = 5,
    perm_seed: int = 1,
    log_every: int = 10000,
    log_fn: Callable[[str], None] | None = None,
) -> dict:
    """Phase 1: 流式扫描文档生成 MinHash 签名，落盘 signatures.npy / doc_ids.jsonl。"""
    import json as _json
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    a, b = _mh_permutations(num_hashes, seed=perm_seed)
    sig_path = out_dir / "signatures.npy"
    ids_path = out_dir / "doc_ids.jsonl"
    chunks: list[np.ndarray] = []
    buf: list[np.ndarray] = []
    CHUNK = 4096
    n = 0
    skipped = 0
    with ids_path.open("w", encoding="utf-8") as id_f:
        for doc in docs:
            text = str(doc.get("text") or "")
            grams = _word_ngrams(text, ngram_size)
            if len(grams) < min_words:
                # 文本 ngram 太少（极短文档），给全 MAX 签名占位
                skipped += 1
                sig = np.full(num_hashes, _MH_MAX, dtype=np.uint32)
            else:
                sig = _minhash_signature_fast(grams, a, b)
            buf.append(sig)
            id_f.write(_json.dumps({"doc_id": str(doc["doc_id"])}) + "\n")
            n += 1
            if len(buf) >= CHUNK:
                chunks.append(np.stack(buf))
                buf.clear()
            if log_fn and n % log_every == 0:
                log_fn(f"[minhash] phase1: {n} docs")
    if buf:
        chunks.append(np.stack(buf))
    sigs = np.concatenate(chunks) if chunks else np.empty((0, num_hashes), dtype=np.uint32)
    np.save(sig_path, sigs)
    return {
        "n": n,
        "skipped_short": skipped,
        "num_hashes": num_hashes,
        "ngram_size": ngram_size,
        "signatures_path": str(sig_path),
        "doc_ids_path": str(ids_path),
    }


def compute_minhash_lsh(
    out_dir,
    *,
    num_bands: int = 8,
    band_size: int = 8,
    jaccard_threshold: float = 0.8,
    hot_bucket_cap: int = 1000,
    on_doc: Callable[[DocResult], None] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Phase 2: mmap 签名做 banded LSH，hot bucket 跳过候选对生成。"""
    import json as _json
    from pathlib import Path
    out_dir = Path(out_dir)
    sigs = np.load(out_dir / "signatures.npy", mmap_mode="r")
    doc_ids = [
        _json.loads(line)["doc_id"]
        for line in (out_dir / "doc_ids.jsonl").open(encoding="utf-8")
    ]
    n, K = int(sigs.shape[0]), int(sigs.shape[1])
    if K != num_bands * band_size:
        raise ValueError(f"K={K} but num_bands*band_size={num_bands*band_size}")

    bucket_map: dict[bytes, list[int]] = defaultdict(list)
    for band in range(num_bands):
        cs, ce = band * band_size, (band + 1) * band_size
        prefix = bytes((band,))
        for i in range(n):
            bucket_map[prefix + sigs[i, cs:ce].tobytes()].append(i)
        if log_fn:
            log_fn(f"[minhash] phase2 band {band+1}/{num_bands} done")

    cands: set[tuple[int, int]] = set()
    hot = 0
    nontriv = 0
    max_b = 0
    for idxs in bucket_map.values():
        s = len(idxs)
        if s < 2:
            continue
        nontriv += 1
        max_b = max(max_b, s)
        if s > hot_bucket_cap:
            hot += 1
            continue
        for ia in range(s):
            i = idxs[ia]
            for ib in range(ia + 1, s):
                j = idxs[ib]
                cands.add((i, j) if i < j else (j, i))
    del bucket_map
    if log_fn:
        log_fn(
            f"[minhash] candidates={len(cands)} nontriv_buckets={nontriv} "
            f"hot(>{hot_bucket_cap})={hot}"
        )

    pairs: list[tuple[int, int, float]] = []
    for i, j in cands:
        eq = int(np.count_nonzero(sigs[i] == sigs[j]))
        jac = eq / K
        if jac >= jaccard_threshold:
            pairs.append((i, j, jac))
    del cands

    near_cnt: dict[int, int] = defaultdict(int)
    jmax: dict[int, float] = defaultdict(float)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, jac in pairs:
        near_cnt[i] += 1
        near_cnt[j] += 1
        if jac > jmax[i]:
            jmax[i] = jac
        if jac > jmax[j]:
            jmax[j] = jac
        ri, rj = _find(i), _find(j)
        if ri != rj:
            if ri < rj:
                parent[rj] = ri
            else:
                parent[ri] = rj

    csize: dict[int, int] = defaultdict(int)
    for x in range(n):
        csize[_find(x)] += 1
    largest = max(csize.values(), default=0)
    multi = sum(1 for v in csize.values() if v > 1)

    per_doc: list[DocResult] = []
    near = 0
    for idx, did in enumerate(doc_ids):
        c = near_cnt.get(idx, 0)
        is_nd = c > 0
        if is_nd:
            near += 1
        r = DocResult(
            doc_id=did,
            scores={
                "jaccard_max": round(jmax.get(idx, 0.0), 4),
                "near_dup_count": c,
                "cluster_id": _find(idx),
                "cluster_size": csize[_find(idx)],
            },
            flags={"is_near_dup": is_nd},
        )
        if on_doc is not None:
            on_doc(r)
        else:
            per_doc.append(r)

    summary = {
        "total_docs": n,
        "near_dup_docs": near,
        "near_dup_pct": round(near / n, 4) if n else 0.0,
        "near_dup_pairs": len(pairs),
        "num_clusters_multi": multi,
        "largest_cluster_size": largest,
        "num_lsh_buckets_nontrivial": nontriv,
        "num_hot_buckets_skipped": hot,
        "hot_bucket_cap": hot_bucket_cap,
        "max_bucket_size": max_b,
        "jaccard_threshold": jaccard_threshold,
        "num_hashes": K,
        "num_bands": num_bands,
        "band_size": band_size,
    }
    return per_doc, summary


def compute_minhash_dedup(
    docs: Iterable[Document],
    num_hashes: int = 64,
    ngram_size: int = 5,
    jaccard_threshold: float = 0.8,
    num_bands: int = 8,
    band_size: int = 8,
    on_doc: Callable[[DocResult], None] | None = None,
    *,
    out_dir=None,
    hot_bucket_cap: int = 1000,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Two-phase MinHash + LSH near-dedup. Streams to disk; OK for 1M docs."""
    import tempfile
    from pathlib import Path
    cleanup = False
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="minhash_sigs_"))
        cleanup = True
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    p1 = compute_minhash_signatures(
        docs,
        out_dir,
        num_hashes=num_hashes,
        ngram_size=ngram_size,
        log_fn=log_fn,
    )
    per_doc, summary = compute_minhash_lsh(
        out_dir,
        num_bands=num_bands,
        band_size=band_size,
        jaccard_threshold=jaccard_threshold,
        hot_bucket_cap=hot_bucket_cap,
        on_doc=on_doc,
        log_fn=log_fn,
    )
    summary["ngram_size"] = ngram_size
    summary["skipped_short_docs"] = p1["skipped_short"]
    if cleanup:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
    return per_doc, summary


# ── N-gram paragraph dedup ────────────────────────────────────────────────────

def _para_shingles(text: str, n: int) -> frozenset[int]:
    """Return a frozenset of word n-gram hashes for one paragraph."""
    words = text.split()
    if len(words) < n:
        return frozenset({hash(text)})
    return frozenset(
        hash(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)
    )


def compute_ngram_dedup(
    docs: Iterable[Document],
    ngram_size: int = 13,
    overlap_threshold: float = 0.5,
    paragraph_sep: str = "\n\n",
    min_para_chars: int = 50,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Paragraph-level n-gram dedup (Dolma-style in-batch audit).

    Splits each doc into paragraphs, computes word n-gram shingle sets, and
    checks what fraction of a paragraph's shingles appeared in earlier docs.
    Reports per-doc contamination ratio.

    Note: order-dependent (first occurrence wins). For cross-corpus dedup use
    Dolma's bloom-filter pipeline directly.
    """
    doc_list = list(docs)

    seen_shingles: set[int] = set()
    per_doc: list[DocResult] = []
    contaminated_docs = 0
    total_paras = 0
    contaminated_paras = 0

    for doc in doc_list:
        doc_id = str(doc["doc_id"])
        text = str(doc.get("text") or "")
        paras = [p.strip() for p in text.split(paragraph_sep) if len(p.strip()) >= min_para_chars]

        para_overlap_ratios: list[float] = []

        for para in paras:
            shingles = _para_shingles(para, ngram_size)
            if not shingles:
                continue
            total_paras += 1

            dup_count = sum(1 for s in shingles if s in seen_shingles)
            overlap = dup_count / len(shingles)
            para_overlap_ratios.append(overlap)

            if overlap >= overlap_threshold:
                contaminated_paras += 1

            seen_shingles.update(shingles)

        doc_overlap = round(
            sum(r >= overlap_threshold for r in para_overlap_ratios) / len(para_overlap_ratios),
            4,
        ) if para_overlap_ratios else 0.0

        is_contaminated = doc_overlap > 0.0
        if is_contaminated:
            contaminated_docs += 1

        result = DocResult(
            doc_id=doc_id,
            scores={
                "para_count": len(paras),
                "para_contamination_ratio": doc_overlap,
                "mean_shingle_overlap": round(
                    sum(para_overlap_ratios) / len(para_overlap_ratios), 4
                ) if para_overlap_ratios else 0.0,
            },
            flags={"is_ngram_dup": is_contaminated},
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
        "total_paras": total_paras,
        "contaminated_paras": contaminated_paras,
        "contaminated_para_pct": round(contaminated_paras / total_paras, 4) if total_paras else 0.0,
        "ngram_size": ngram_size,
        "overlap_threshold": overlap_threshold,
    }
    return per_doc, summary


# ── Semantic dedup (embedding) ────────────────────────────────────────────────

def _encode_embeddings(
    texts: list[str],
    model_path: str,
    *,
    batch_size: int = 64,
    max_length: int = 512,
    device: str = "cuda",
):
    """Encode texts into L2-normalized embeddings with a bge-style model.

    Uses CLS pooling (last_hidden_state[:, 0]) per bge convention, fp16 on GPU.
    Returns a torch.Tensor [N, dim] on *device*. Imports torch/transformers
    lazily so the rest of the module works without them installed.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    if device != "cpu":
        model = model.half()

    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0]                  # CLS pooling
            cls = torch.nn.functional.normalize(cls, dim=1)    # L2 normalize
            embs.append(cls)
    return torch.cat(embs, dim=0)


def compute_semdedup(
    docs: Iterable[Document],
    *,
    model_path: str,
    eps: float = 0.07,
    batch_size: int = 64,
    max_length: int = 512,
    sim_block: int = 2048,
    device: str = "cuda",
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Detect semantic near-duplicates via embedding cosine similarity.

    Pipeline: embed each doc (bge-m3, CLS + L2 norm) → for each doc i, search
    earlier docs j < i for cosine ≥ (1 - eps); the earliest match becomes i's
    cluster representative (first-occurrence wins, like compute_ngram_dedup).
    A doc is a semantic dup iff it belongs to a size>1 cluster and is not the
    representative (i.e. removable redundancy).

    Audit tool for samples (use --max-docs). Similarity is exact (block matmul
    on GPU), no ANN index — fine up to ~100K docs. For TB-scale, use a
    distributed SemDeDup pipeline.
    """
    import torch

    doc_list = list(docs)
    n = len(doc_list)
    doc_ids = [str(d["doc_id"]) for d in doc_list]
    texts = [str(d.get("text") or "") for d in doc_list]
    thr = 1.0 - eps

    # Degenerate case: nothing to compare.
    if n == 0:
        return [], {
            "total_docs": 0, "semantic_dup_docs": 0, "semantic_dup_pct": 0.0,
            "num_clusters": 0, "largest_cluster_size": 0,
            "cos_threshold": round(thr, 4), "eps": eps, "model": model_path,
        }

    emb = _encode_embeddings(
        texts, model_path,
        batch_size=batch_size, max_length=max_length, device=device,
    )

    # rep[i] = representative (cluster anchor) for doc i; max_cos[i] = best
    # similarity to its representative. Default: each doc is its own rep.
    rep = list(range(n))
    max_cos = [0.0] * n

    for start in range(0, n, sim_block):
        end = min(start + sim_block, n)
        # cosine of block rows [start:end] against all earlier docs [0:end]
        block = emb[start:end] @ emb[:end].T          # [block, end]
        for local_i in range(end - start):
            i = start + local_i
            if i == 0:
                continue
            row = block[local_i, :i]                  # similarities to j < i
            best_j = int(torch.argmax(row).item())
            best_sim = float(row[best_j].item())
            if best_sim >= thr:
                # join the representative of the matched earlier doc
                rep[i] = rep[best_j]
                max_cos[i] = best_sim

    # ── build clusters from representative pointers ───────────────────────────
    members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        members[rep[i]].append(i)

    cluster_id_of = [0] * n
    cluster_size_of = [0] * n
    cid = 0
    largest = 0
    num_multi_clusters = 0
    for anchor, idxs in members.items():
        size = len(idxs)
        largest = max(largest, size)
        if size > 1:
            num_multi_clusters += 1
        for idx in idxs:
            cluster_id_of[idx] = cid
            cluster_size_of[idx] = size
        cid += 1

    # ── per-doc results ───────────────────────────────────────────────────────
    per_doc: list[DocResult] = []
    semantic_dup_docs = 0
    for i, doc_id in enumerate(doc_ids):
        size = cluster_size_of[i]
        is_representative = rep[i] == i
        is_dup = size > 1 and not is_representative
        if is_dup:
            semantic_dup_docs += 1

        result = DocResult(
            doc_id=doc_id,
            scores={
                "max_cos": round(max_cos[i], 4),
                "cluster_id": cluster_id_of[i],
                "cluster_size": size,
                "is_representative": is_representative,
            },
            flags={"is_semantic_dup": is_dup},
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    summary = {
        "total_docs": n,
        "semantic_dup_docs": semantic_dup_docs,
        "semantic_dup_pct": round(semantic_dup_docs / n, 4) if n else 0.0,
        "num_clusters": cid,
        "num_dup_clusters": num_multi_clusters,
        "largest_cluster_size": largest,
        "cos_threshold": round(thr, 4),
        "eps": eps,
        "model": model_path,
    }
    return per_doc, summary
