"""一次性构建 + 加载 contamination cascade 三层 benchmark 索引（hash / MinHash / embedding）.

Cascade 批量跑时每个文件都会 reload benchmark + reencode embedding，68K × BGE-m3 ~1 min/次。
全量 596 文件 = 10h 浪费在重编。本模块负责离线一次构建，cascade 启动时 mmap 加载.

CLI:
  PYTHONPATH=. python stages/contamination/bench_index.py build --config configs/stage5.yaml
  PYTHONPATH=. python stages/contamination/bench_index.py info  --config configs/stage5.yaml

落盘结构（index_dir 由 yaml benchmarks.index_dir 控制）:
  hash_index.pkl        — {doc_md5: [labels], para_md5: [labels]}
  minhash_sigs.npy      — [N_bench, num_hashes] uint32
  minhash_meta.json     — {bench_id, label, kind} 列表，与 sigs 行对应
  minhash_perm.npy      — [2, num_hashes] uint64（universal hashing 参数 a, b）
  minhash_config.json   — {ngram_size, num_hashes, num_bands, band_size}
  embeddings.npy        — [N_bench, dim] float32（CLS + L2-normalized BGE-m3 输出）
  embed_meta.json       — {bench_id, label, kind} 列表，与 embeddings 行对应
  embed_config.json     — {model_path, batch_size, max_length, device, dim}
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import click
import numpy as np
import yaml

from stages.contamination.benchmarks import load_benchmarks
from stages.contamination.utils import (
    _char_ngrams,
    _minhash_signature,
    _mh_permutations,
    _lsh_buckets,
    _encode_texts_bge,
    build_bench_index,
)


def build_index(config_path: str) -> Path:
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    bench_cfg = cfg.get("benchmarks", {})
    index_dir = Path(bench_cfg.get("index_dir") or "/mnt/public/data/contamination_index_v3")
    index_dir.mkdir(parents=True, exist_ok=True)

    cas_cfg = cfg.get("cascade", {})
    exact_cfg = cfg.get("exact", {})
    l2cfg = cas_cfg.get("layer2", {})
    l3cfg = cas_cfg.get("layer3", {})

    click.echo(f"[bench_index] index_dir={index_dir}")
    click.echo("[bench_index] 加载 benchmark ...")
    bench_items = load_benchmarks(bench_cfg)
    n_items = sum(len(v) for v in bench_items.values())
    click.echo(f"[bench_index] {len(bench_items)} benchmark, {n_items} items")

    # ── L1 hash 索引 ──
    click.echo("[bench_index] 构建 L1 hash 索引 ...")
    doc_index, para_index = build_bench_index(
        bench_items,
        paragraph_sep=exact_cfg.get("paragraph_sep", "\n\n"),
        min_para_chars=exact_cfg.get("min_para_chars", 50),
    )
    with (index_dir / "hash_index.pkl").open("wb") as f:
        pickle.dump({"doc_index": doc_index, "para_index": para_index}, f)
    click.echo(f"  doc_hashes={len(doc_index)}, para_hashes={len(para_index)}")

    # ── L2 MinHash 索引 ──
    click.echo("[bench_index] 构建 L2 MinHash 签名 ...")
    ngram_size = l2cfg.get("ngram_size", 5)
    num_hashes = l2cfg.get("num_hashes", 128)
    num_bands = l2cfg.get("num_bands", 32)
    band_size = l2cfg.get("band_size", 4)
    a, b = _mh_permutations(num_hashes, seed=1)

    minhash_meta: list[dict] = []
    sigs_buf: list[np.ndarray] = []
    for label, items in bench_items.items():
        for item in items:
            for kind in ("text", "code"):
                content = (item.get(kind) or "").strip()
                if not content:
                    continue
                ngrams = _char_ngrams(content, ngram_size)
                if not ngrams:
                    continue
                sig = _minhash_signature(ngrams, a, b)
                sigs_buf.append(sig)
                minhash_meta.append({
                    "bench_id": item["bench_id"], "label": label, "kind": kind,
                })
    minhash_sigs = np.stack(sigs_buf) if sigs_buf else np.empty((0, num_hashes), dtype=np.uint32)
    np.save(index_dir / "minhash_sigs.npy", minhash_sigs)
    np.save(index_dir / "minhash_perm.npy", np.stack([a, b]))
    (index_dir / "minhash_meta.json").write_text(
        json.dumps(minhash_meta, ensure_ascii=False), encoding="utf-8")
    (index_dir / "minhash_config.json").write_text(json.dumps({
        "ngram_size": ngram_size, "num_hashes": num_hashes,
        "num_bands": num_bands, "band_size": band_size,
        "perm_seed": 1,
    }), encoding="utf-8")
    click.echo(f"  minhash_sigs shape={minhash_sigs.shape}")

    # ── L3 Embedding 索引 ──
    click.echo("[bench_index] 构建 L3 BGE-m3 embedding ...")
    model_path = l3cfg.get("model_path", "/mnt/public/model/bge-m3")
    batch_size = l3cfg.get("batch_size", 64)
    max_length = l3cfg.get("max_length", 512)
    device = l3cfg.get("device", "cuda")

    embed_texts: list[str] = []
    embed_meta: list[dict] = []
    for label, items in bench_items.items():
        for item in items:
            for kind in ("text", "code"):
                content = (item.get(kind) or "").strip()
                if not content:
                    continue
                embed_texts.append(content)
                embed_meta.append({
                    "bench_id": item["bench_id"], "label": label, "kind": kind,
                })

    embeddings = _encode_texts_bge(
        embed_texts, model_path,
        batch_size=batch_size, max_length=max_length, device=device,
    )
    np.save(index_dir / "embeddings.npy", embeddings)
    (index_dir / "embed_meta.json").write_text(
        json.dumps(embed_meta, ensure_ascii=False), encoding="utf-8")
    (index_dir / "embed_config.json").write_text(json.dumps({
        "model_path": model_path,
        "batch_size": batch_size,
        "max_length": max_length,
        "device": device,
        "dim": int(embeddings.shape[1]) if embeddings.size else 1024,
    }), encoding="utf-8")
    click.echo(f"  embeddings shape={embeddings.shape}, dtype={embeddings.dtype}")

    click.echo(f"[bench_index] done -> {index_dir}")
    return index_dir


def load_index(index_dir: Path) -> dict:
    """加载预构建索引，返回 cascade 可直接使用的 dict.

    返回结构（与 build_bench_minhash + build_bench_embeddings 输出兼容）:
      hash_index: {'doc_index', 'para_index'}
      l2_index:   {'a', 'b', 'bench_sigs', 'bucket_map', 'ngram_size',
                   'num_hashes', 'num_bands', 'band_size'}
      l3_embed:   {'embeddings', 'meta', 'faiss_index', 'dim', 'model_path',
                   'batch_size', 'max_length', 'device'}
    """
    import faiss
    index_dir = Path(index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(f"index_dir 不存在: {index_dir}")

    # L1
    with (index_dir / "hash_index.pkl").open("rb") as f:
        hash_data = pickle.load(f)

    # L2
    minhash_cfg = json.loads((index_dir / "minhash_config.json").read_text(encoding="utf-8"))
    minhash_meta = json.loads((index_dir / "minhash_meta.json").read_text(encoding="utf-8"))
    minhash_sigs = np.load(index_dir / "minhash_sigs.npy", mmap_mode="r")
    perm = np.load(index_dir / "minhash_perm.npy")
    a, b = perm[0], perm[1]

    # 重建 bench_sigs (list of tuples) + bucket_map
    bench_sigs = [
        (m["bench_id"], m["label"], m["kind"], minhash_sigs[i])
        for i, m in enumerate(minhash_meta)
    ]
    bucket_map: dict[tuple, list[int]] = defaultdict(list)
    num_bands = minhash_cfg["num_bands"]
    band_size = minhash_cfg["band_size"]
    for i, m in enumerate(minhash_meta):
        for key in _lsh_buckets(minhash_sigs[i], num_bands, band_size):
            bucket_map[key].append(i)

    l2_index = {
        "a": a, "b": b,
        "bench_sigs": bench_sigs,
        "bucket_map": dict(bucket_map),
        "ngram_size": minhash_cfg["ngram_size"],
        "num_hashes": minhash_cfg["num_hashes"],
        "num_bands": num_bands,
        "band_size": band_size,
    }

    # L3
    embed_cfg = json.loads((index_dir / "embed_config.json").read_text(encoding="utf-8"))
    embed_meta = json.loads((index_dir / "embed_meta.json").read_text(encoding="utf-8"))
    embeddings = np.load(index_dir / "embeddings.npy")
    dim = embed_cfg["dim"]
    faiss_index = faiss.IndexFlatIP(dim)
    if embeddings.size:
        faiss_index.add(embeddings.astype(np.float32, copy=False))
    l3_embed = {
        "embeddings": embeddings,
        "meta": [(m["bench_id"], m["label"], m["kind"]) for m in embed_meta],
        "faiss_index": faiss_index,
        "dim": dim,
        "model_path": embed_cfg["model_path"],
        "batch_size": embed_cfg["batch_size"],
        "max_length": embed_cfg["max_length"],
        "device": embed_cfg["device"],
    }

    return {
        "hash_index": hash_data,
        "l2_index": l2_index,
        "l3_embed": l3_embed,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Cascade benchmark index 离线构建工具"""


@cli.command()
@click.option("--config", "config_path", default="configs/stage5.yaml", show_default=True)
def build(config_path):
    """一次性构建 hash + MinHash + BGE-m3 embedding 索引并落盘"""
    build_index(config_path)


@cli.command()
@click.option("--config", "config_path", default="configs/stage5.yaml", show_default=True)
def info(config_path):
    """查看已构建索引的元信息"""
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    index_dir = Path(cfg["benchmarks"].get("index_dir") or "/mnt/public/data/contamination_index_v3")
    click.echo(f"index_dir={index_dir}")
    for name in ["hash_index.pkl", "minhash_sigs.npy", "minhash_meta.json",
                 "minhash_perm.npy", "minhash_config.json",
                 "embeddings.npy", "embed_meta.json", "embed_config.json"]:
        p = index_dir / name
        if p.exists():
            click.echo(f"  ✓ {name}  ({p.stat().st_size/1e6:.1f} MB)")
        else:
            click.echo(f"  ✗ {name}  (missing)")


if __name__ == "__main__":
    cli()
