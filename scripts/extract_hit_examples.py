#!/usr/bin/env python3
"""从 stage2 PII/Secret 的 per_doc 命中记录回查原始 parquet，抽样命中案例写成 markdown。

只读：读 outputs/stage2 的 per_doc + 只读原始 parquet，输出到 stages/safety/hit_examples.md。
该 md 含命中原文片段，不进 git（见 .gitignore）。

用法:
  PYTHONPATH=. python scripts/extract_hit_examples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
OUT_STAGE2 = ROOT / "outputs" / "stage2"
DATA_ROOT = Path("/mnt/public/data/Ultra-FineWeb-L3/data")
REPORT = ROOT / "stages" / "safety" / "hit_examples.md"

# 每个 (语言, 类型) 组合最多抽多少条命中文档
MAX_PER_GROUP = 8
# 命中 span 前后保留的上下文字符数
CTX = 60

DATASETS = {
    "ufw_en_l3": DATA_ROOT / "ultrafineweb_en_l3",
    "ufw_zh_l3": DATA_ROOT / "ultrafineweb_zh_l3",
}


def out_name_to_parquet(dataset_dir: Path, out_dirname: str) -> Path | None:
    """outputs 目录名 multi_style__part-00000-...snappy → 原始 parquet 路径。"""
    rel = out_dirname.replace("__", "/") + ".parquet"
    p = dataset_dir / rel
    return p if p.exists() else None


def load_content_map(parquet_path: Path, wanted: set[str]) -> dict[str, str]:
    """从 parquet 读出 wanted 集合里 uid 的 content。"""
    out: dict[str, str] = {}
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=2000, columns=["uid", "content"]):
        cols = batch.to_pydict()
        for uid, content in zip(cols["uid"], cols["content"]):
            if uid in wanted:
                out[uid] = content
                if len(out) == len(wanted):
                    return out
    return out


def slice_span(text: str, start: int, end: int) -> tuple[str, str]:
    """返回 (命中片段, 带上下文的片段)。"""
    hit = text[start:end]
    a = max(0, start - CTX)
    b = min(len(text), end + CTX)
    ctx = text[a:start] + "【" + hit + "】" + text[end:b]
    ctx = ctx.replace("\n", " ⏎ ")
    return hit, ctx


def collect_pii(dataset: str, dataset_dir: Path, lines: list[str]) -> None:
    pii_root = OUT_STAGE2 / dataset / "pii"
    if not pii_root.is_dir():
        return
    # 按 entity_type 分组收集，每组 MAX_PER_GROUP 条
    by_type: dict[str, list[dict]] = {}
    for part_dir in sorted(pii_root.iterdir()):
        pd_file = part_dir / "per_doc.jsonl"
        if not pd_file.exists():
            continue
        parquet = out_name_to_parquet(dataset_dir, part_dir.name)
        if parquet is None:
            continue
        # 先扫一遍找命中 doc，确定还需要哪些 type
        hits_here: list[dict] = []
        wanted: set[str] = set()
        with pd_file.open() as f:
            for line in f:
                rec = json.loads(line)
                if not rec["flags"].get("has_pii"):
                    continue
                types = {h["entity_type"] for h in rec["scores"]["pii_hits"]}
                if all(len(by_type.get(t, [])) >= MAX_PER_GROUP for t in types):
                    continue
                hits_here.append(rec)
                wanted.add(rec["doc_id"])
        if not wanted:
            continue
        content = load_content_map(parquet, wanted)
        for rec in hits_here:
            text = content.get(rec["doc_id"])
            if text is None:
                continue
            for h in rec["scores"]["pii_hits"]:
                t = h["entity_type"]
                bucket = by_type.setdefault(t, [])
                if len(bucket) >= MAX_PER_GROUP:
                    continue
                hit, ctx = slice_span(text, h["start"], h["end"])
                bucket.append({
                    "doc_id": rec["doc_id"], "score": h["score"],
                    "hit": hit, "ctx": ctx, "part": part_dir.name,
                })

    lines.append(f"\n## {dataset} — PII 命中案例\n")
    for t in sorted(by_type):
        lines.append(f"\n### {t}（{len(by_type[t])} 例）\n")
        for e in by_type[t]:
            lines.append(f"- `score={e['score']}` doc=`{e['doc_id']}`")
            lines.append(f"  - 命中: `{e['hit']}`")
            lines.append(f"  - 上下文: {e['ctx']}")


def collect_secrets(dataset: str, dataset_dir: Path, lines: list[str]) -> None:
    sec_root = OUT_STAGE2 / dataset / "secrets"
    if not sec_root.is_dir():
        return
    by_rule: dict[str, list[dict]] = {}
    for part_dir in sorted(sec_root.iterdir()):
        pd_file = part_dir / "per_doc.jsonl"
        if not pd_file.exists():
            continue
        parquet = out_name_to_parquet(dataset_dir, part_dir.name)
        if parquet is None:
            continue
        hits_here: list[dict] = []
        wanted: set[str] = set()
        with pd_file.open() as f:
            for line in f:
                rec = json.loads(line)
                if not rec["flags"].get("has_secrets"):
                    continue
                rules = {s["rule_id"] for s in rec["scores"]["secrets"]}
                if all(len(by_rule.get(r, [])) >= MAX_PER_GROUP for r in rules):
                    continue
                hits_here.append(rec)
                wanted.add(rec["doc_id"])
        if not wanted:
            continue
        content = load_content_map(parquet, wanted)
        for rec in hits_here:
            text = content.get(rec["doc_id"])
            if text is None:
                continue
            doc_lines = text.split("\n")
            for s in rec["scores"]["secrets"]:
                r = s["rule_id"]
                bucket = by_rule.setdefault(r, [])
                if len(bucket) >= MAX_PER_GROUP:
                    continue
                ln = s.get("start_line", 1)
                snippet = doc_lines[ln - 1] if 1 <= ln <= len(doc_lines) else "(行号越界)"
                bucket.append({
                    "doc_id": rec["doc_id"], "entropy": s.get("entropy"),
                    "line": ln, "snippet": snippet.strip(), "part": part_dir.name,
                })

    lines.append(f"\n## {dataset} — Secret 命中案例\n")
    if not by_rule:
        lines.append("\n(无命中)\n")
        return
    for r in sorted(by_rule):
        lines.append(f"\n### {r}（{len(by_rule[r])} 例）\n")
        for e in by_rule[r]:
            lines.append(f"- `entropy={e['entropy']}` line={e['line']} doc=`{e['doc_id']}`")
            lines.append(f"  - 命中行: `{e['snippet']}`")


def main() -> None:
    lines = [
        "# Stage 2 命中案例抽样",
        "",
        "> 自动抽样自 `outputs/stage2/*/per_doc.jsonl` + 回查原始 parquet。",
        "> **含真实命中原文片段，不进 git。** 用于人工判断命中真伪（真 PII / 误报）。",
        f"> 每 (类型) 最多 {MAX_PER_GROUP} 例，命中片段用 【】 标出。",
    ]
    for dataset, dataset_dir in DATASETS.items():
        if not dataset_dir.is_dir():
            print(f"跳过 {dataset}: 数据目录不存在", file=sys.stderr)
            continue
        collect_pii(dataset, dataset_dir, lines)
        collect_secrets(dataset, dataset_dir, lines)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"写入 {REPORT}")


if __name__ == "__main__":
    main()
