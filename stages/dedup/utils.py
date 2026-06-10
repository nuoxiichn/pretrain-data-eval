"""Stage 4 computation utilities.

  compute_exact_dedup   — MD5-based exact dedup (doc-level + paragraph-level)
  compute_minhash_dedup — MinHash + LSH near-dedup (word n-gram)
  compute_semdedup      — embedding semantic near-dedup (bge-m3 + cosine clustering)
"""

from __future__ import annotations

import hashlib
import struct
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

def _word_ngrams(text: str, n: int) -> set[str]:
    words = text.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _minhash_signature(ngrams: set[str], num_hashes: int) -> np.ndarray:
    """Compute MinHash signature using MD5-derived hash functions."""
    sig = np.full(num_hashes, 0xFFFFFFFF, dtype=np.uint32)
    for gram in ngrams:
        gram_bytes = gram.encode("utf-8", errors="replace")
        for i in range(num_hashes):
            # Use (seed || gram) as input to get independent hash functions
            seed = struct.pack("<I", i)
            h = int(hashlib.md5(seed + gram_bytes).hexdigest()[:8], 16) & 0xFFFFFFFF
            if h < sig[i]:
                sig[i] = h
    return sig


def _lsh_buckets(sig: np.ndarray, num_bands: int, band_size: int) -> list[tuple]:
    """Return LSH bucket keys for a signature."""
    keys = []
    for b in range(num_bands):
        band = sig[b * band_size : (b + 1) * band_size]
        keys.append((b, band.tobytes()))
    return keys


def compute_minhash_dedup(
    docs: Iterable[Document],
    num_hashes: int = 112,
    ngram_size: int = 5,
    jaccard_threshold: float = 0.72,
    num_bands: int = 14,
    band_size: int = 8,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Detect near-duplicates using MinHash + LSH.

    For audit purposes: works well up to ~100K docs in memory.
    For TB-scale production dedup, use DataTrove's distributed MinHash pipeline.
    """
    doc_list = list(docs)
    n = len(doc_list)

    # ── compute signatures ────────────────────────────────────────────────────
    signatures: list[np.ndarray] = []
    doc_ids: list[str] = []
    for doc in doc_list:
        text = str(doc.get("text") or "")
        ngrams = _word_ngrams(text, ngram_size)
        sig = _minhash_signature(ngrams, num_hashes)
        signatures.append(sig)
        doc_ids.append(str(doc["doc_id"]))

    # ── LSH: find candidate pairs ─────────────────────────────────────────────
    bucket_map: dict[tuple, list[int]] = defaultdict(list)
    for i, sig in enumerate(signatures):
        for key in _lsh_buckets(sig, num_bands, band_size):
            bucket_map[key].append(i)

    # ── verify candidates with exact Jaccard estimate ─────────────────────────
    near_dup_pairs: set[tuple[int, int]] = set()
    for bucket_indices in bucket_map.values():
        if len(bucket_indices) < 2:
            continue
        for a in range(len(bucket_indices)):
            for b in range(a + 1, len(bucket_indices)):
                i, j = bucket_indices[a], bucket_indices[b]
                if (min(i, j), max(i, j)) in near_dup_pairs:
                    continue
                jaccard = float(np.mean(signatures[i] == signatures[j]))
                if jaccard >= jaccard_threshold:
                    near_dup_pairs.add((min(i, j), max(i, j)))

    # ── aggregate per-doc near-dup counts ────────────────────────────────────
    near_dup_count: dict[int, int] = defaultdict(int)
    jaccard_max: dict[int, float] = defaultdict(float)
    for i, j in near_dup_pairs:
        jaccard = float(np.mean(signatures[i] == signatures[j]))
        near_dup_count[i] += 1
        near_dup_count[j] += 1
        jaccard_max[i] = max(jaccard_max[i], jaccard)
        jaccard_max[j] = max(jaccard_max[j], jaccard)

    per_doc: list[DocResult] = []
    near_dup_docs = 0

    for idx, doc_id in enumerate(doc_ids):
        cnt = near_dup_count[idx]
        jmax = round(jaccard_max[idx], 4)
        is_near_dup = cnt > 0
        if is_near_dup:
            near_dup_docs += 1

        result = DocResult(
            doc_id=doc_id,
            scores={"jaccard_max": jmax, "near_dup_count": cnt},
            flags={"is_near_dup": is_near_dup},
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    total = n
    summary = {
        "total_docs": total,
        "near_dup_docs": near_dup_docs,
        "near_dup_pct": round(near_dup_docs / total, 4) if total else 0.0,
        "near_dup_pairs": len(near_dup_pairs),
        "jaccard_threshold": jaccard_threshold,
        "num_hashes": num_hashes,
        "ngram_size": ngram_size,
    }
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
