"""DataTrove MinHash utils: adapter + config + aggregate.

跑在独立 conda env `pretrain-dedup` 下（详见 stages/dedup_datatrove/README.md）。
本文件不应被 main env 导入（datatrove 未装到 main env）。

顶部预导 `importlib.metadata` 是修 datatrove 0.9.0 的一个上游缺失
——它的 `utils/_import_utils.py` 用了 `importlib.metadata.distributions()`
但只 `import importlib.resources`，Python 3.11 下会报 AttributeError。
预导一次全局生效。
"""
from __future__ import annotations

import importlib.metadata  # noqa: F401 — see docstring

import json
import struct
import time
from pathlib import Path

from datatrove.io import get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.dedup.minhash import MinhashConfig


# ---------------------------------------------------------------------------
# Config：严格照 stages/dedup/utils.py 的自实现取值，保证 sanity 步骤可比对
# ---------------------------------------------------------------------------
# self-impl:  n_hashes=64 = num_bands(8) * band_size(8), jaccard=0.8, n_grams=5
MINHASH_CONFIG = MinhashConfig(
    n_grams=5,
    num_buckets=8,
    hashes_per_bucket=8,
    seed=1,
)

# datatrove language codes 用 ISO-639-3
LANG_CODE = {
    "ufw_en_l3": "eng",
    "ufw_zh_l3": "cmn",
}

# UFW-L3 parquet 字段映射
UFW_ADAPTER_TEXT_KEY = "content"
UFW_ADAPTER_ID_KEY = "uid"


def ufw_adapter(self, data: dict, path: str, id_in_file: int | str) -> dict:
    """Adapter for datatrove ParquetReader on UFW-L3 parquet.

    UFW-L3 fields: uid / content / style
    datatrove Document 需要: id / text / metadata / media
    """
    return {
        "text": data.get(UFW_ADAPTER_TEXT_KEY, "") or "",
        "id": str(data.get(UFW_ADAPTER_ID_KEY, id_in_file)),
        "media": [],
        "metadata": {
            "style": data.get("style"),
            "source_path": path,
        },
    }


# ---------------------------------------------------------------------------
# DocIdSink：pass-through step，把每 rank 的 doc_id 流写到 disk
# 放在 ParquetReader 和 MinhashDedupSignature 之间。
# stage 3 的 `.clusters/.sizes` 文件以 `{rank:06d}.` 命名，rank 对应 signature
# 阶段各 worker 的 rank；每 rank 内 doc_idx 按输入行序。回读时 (rank, doc_idx)
# 即可精确定位。
# ---------------------------------------------------------------------------
class DocIdSink(PipelineStep):
    type = "📑 - SINK"
    name = "📑 DocIdSink"

    def __init__(self, output_folder):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)

    def run(self, data, rank: int = 0, world_size: int = 1):
        with self.output_folder.open(f"{rank:06d}.docids.txt", "w") as f:
            for doc in data:
                f.write((doc.id or "") + "\n")
                yield doc


# ---------------------------------------------------------------------------
# Aggregate：从 stage 2 dups + stage 3 clusters/sizes + docids 侧写 → 项目契约
# ---------------------------------------------------------------------------
def _read_dups_pairs(buckets_folder: Path) -> int:
    """Stage 2 输出的 `{bucket:05d}_{worker:02d}.dups` 里每条记录 = 4×uint32
    表示 (file_id1, doc_id1, file_id2, doc_id2)。这里只关心总对数。
    """
    total = 0
    for f in sorted(buckets_folder.rglob("*.dups")):
        size = f.stat().st_size
        total += size // 16  # 4 × uint32
    return total


def _read_stage3_pairs(folder: Path, ext: str) -> dict[tuple[int, int], int]:
    """读 `{rank:06d}.<ext>` 文件；每条记录 = (doc_idx, value)×uint32 = 8 bytes.

    返回 {(rank, doc_idx): value}.
    `.remove` 是 uint32 单值（doc_idx），不用这里。
    """
    out: dict[tuple[int, int], int] = {}
    for f in sorted(folder.rglob(f"*.{ext}")):
        try:
            rank = int(f.stem)
        except ValueError:
            continue
        raw = f.read_bytes()
        n = len(raw) // 8
        for i in range(n):
            doc_idx, val = struct.unpack_from("<II", raw, i * 8)
            out[(rank, doc_idx)] = val
    return out


def _load_doc_id_streams(docids_folder: Path) -> dict[int, list[str]]:
    """{rank: [doc_id, ...]}."""
    out: dict[int, list[str]] = {}
    for f in sorted(docids_folder.rglob("*.docids.txt")):
        try:
            rank = int(f.stem.split(".")[0])
        except ValueError:
            continue
        with f.open() as fh:
            out[rank] = [ln.rstrip("\n") for ln in fh]
    return out


def aggregate(
    dataset: str,
    signatures_folder: Path,
    buckets_folder: Path,
    clusters_folder: Path,
    docids_folder: Path,
    output_dir: Path,
    n_workers: int,
    wall_time_sec: float,
) -> dict:
    """产出 per_doc.jsonl + summary.json，字段对齐 stages/dedup/utils.py。"""
    doc_id_streams = _load_doc_id_streams(docids_folder)
    if not doc_id_streams:
        raise FileNotFoundError(
            f"no docids in {docids_folder} — DocIdSink was not attached to signature stage"
        )

    cluster_ids = _read_stage3_pairs(clusters_folder, "clusters")
    cluster_sizes = _read_stage3_pairs(clusters_folder, "sizes")
    n_pairs = _read_dups_pairs(buckets_folder)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_doc_path = output_dir / "per_doc.jsonl"
    summary_path = output_dir / "summary.json"

    total_docs = 0
    near_dup_docs = 0
    cluster_id_to_members: dict[int, int] = {}

    with per_doc_path.open("w", encoding="utf-8") as f:
        for rank in sorted(doc_id_streams):
            for doc_idx, doc_id in enumerate(doc_id_streams[rank]):
                total_docs += 1
                cid = cluster_ids.get((rank, doc_idx))
                csize = cluster_sizes.get((rank, doc_idx), 1)
                in_multi = csize >= 2
                if in_multi:
                    near_dup_docs += 1
                if cid is not None:
                    cluster_id_to_members[cid] = cluster_id_to_members.get(cid, 0) + 1
                rec = {
                    "doc_id": doc_id,
                    "scores": {
                        "cluster_id": int(cid) if cid is not None else -1,
                        "cluster_size": int(csize),
                        "near_dup_count": max(int(csize) - 1, 0),
                    },
                    "flags": {
                        "is_near_dup": bool(in_multi),
                        "in_multi_cluster": bool(in_multi),
                    },
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    multi_clusters = [n for n in cluster_id_to_members.values() if n >= 2]

    summary = {
        "dataset": dataset,
        "total_docs": total_docs,
        "near_dup_docs": near_dup_docs,
        "near_dup_pct": near_dup_docs / total_docs if total_docs else 0.0,
        "near_dup_pairs": n_pairs,
        "num_clusters_multi": len(multi_clusters),
        "largest_cluster_size": max(multi_clusters) if multi_clusters else 0,
        "n_hashes": MINHASH_CONFIG.num_buckets * MINHASH_CONFIG.hashes_per_bucket,
        "num_bands": MINHASH_CONFIG.num_buckets,
        "band_size": MINHASH_CONFIG.hashes_per_bucket,
        "jaccard_threshold": _jaccard_threshold_estimate(
            MINHASH_CONFIG.num_buckets, MINHASH_CONFIG.hashes_per_bucket
        ),
        "n_workers": n_workers,
        "wall_time_sec": round(wall_time_sec, 2),
        "sig_folder": str(signatures_folder),
        "buckets_folder": str(buckets_folder),
        "clusters_folder": str(clusters_folder),
        "docids_folder": str(docids_folder),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _jaccard_threshold_estimate(num_bands: int, band_size: int) -> float:
    """LSH threshold ≈ (1/num_bands) ** (1/band_size)."""
    return round((1.0 / num_bands) ** (1.0 / band_size), 4)


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

