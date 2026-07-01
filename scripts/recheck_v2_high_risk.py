#!/usr/bin/env python3
"""复审 v2 toxicity 的高位文档：用 toxicity-v3 (chunk + 召回 + LLM-judge) 重判，
统计假阳性下降率。

输入:
  outputs/stage2/ufw_{en,zh}_l3/toxicity_v2/per_doc.jsonl  (high_risk=true 的文档)
  /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_{en,zh}_l3/**.parquet (回查原文)

输出:
  outputs/stage2/ufw_{en,zh}_l3/toxicity_v3_recheck/
    ├── per_doc.jsonl        (v3 复审结果, schema 同 toxicity-v3)
    ├── summary.json         (复审汇总，含 fp_reduction)
    └── comparison.md        (人读对照表，v2 vs v3 verdict)

用法:
  PYTHONPATH=. python scripts/recheck_v2_high_risk.py
  # 默认两个数据集都跑；可以 --only en / --only zh
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from stages.safety.utils import compute_toxicity_v3

OUT_STAGE2 = ROOT / "outputs" / "stage2"
DATA_ROOT = Path("/mnt/public/data/Ultra-FineWeb-L3/data")

DATASETS = {
    "ufw_en_l3": DATA_ROOT / "ultrafineweb_en_l3",
    "ufw_zh_l3": DATA_ROOT / "ultrafineweb_zh_l3",
}
PATH_LANG = {"ufw_en_l3": "en", "ufw_zh_l3": "zh"}
WORKERS = 16


def load_v2_high_risk(dataset: str) -> dict[str, float]:
    pd_file = OUT_STAGE2 / dataset / "toxicity_v2" / "per_doc.jsonl"
    hits = {}
    with open(pd_file) as f:
        for ln in f:
            r = json.loads(ln)
            if r["flags"].get("high_risk"):
                hits[r["doc_id"]] = r["scores"]["toxicity"]
    return hits


def _scan_parquet(args):
    parquet_path, wanted = args
    found = {}
    try:
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=2000, columns=["uid", "content", "style"]):
            cols = batch.to_pydict()
            for uid, content, style in zip(cols["uid"], cols["content"], cols["style"]):
                if uid in wanted:
                    found[uid] = {
                        "content": content, "style": style,
                        "source": Path(parquet_path).parent.name,
                        "shard": Path(parquet_path).name,
                    }
                    if len(found) == len(wanted):
                        return found
    except Exception as e:
        print(f"  ! {parquet_path}: {e}", file=sys.stderr)
    return found


def fetch_contents(dataset_dir: Path, wanted_ids: set[str]) -> dict[str, dict]:
    parquets = sorted(dataset_dir.rglob("*.parquet"))
    print(f"  扫描 {len(parquets)} 个 parquet (目标 {len(wanted_ids)} 个 uid)...",
          file=sys.stderr)
    contents: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_scan_parquet, (str(p), set(wanted_ids))): p for p in parquets}
        for fut in as_completed(futures):
            for uid, payload in fut.result().items():
                contents[uid] = payload
            if len(contents) == len(wanted_ids):
                for f2 in futures:
                    f2.cancel()
                break
    return contents


def recheck_dataset(dataset: str, cfg: dict, output_root: Path) -> dict:
    print(f"\n=== {dataset} ===", file=sys.stderr)
    v2_hits = load_v2_high_risk(dataset)
    print(f"  v2 high_risk = {len(v2_hits)}", file=sys.stderr)
    if not v2_hits:
        return {"dataset": dataset, "v2_high_risk": 0, "skipped": True}

    contents = fetch_contents(DATASETS[dataset], set(v2_hits.keys()))
    missing = set(v2_hits) - set(contents)
    if missing:
        print(f"  ⚠ 未回查到 {len(missing)} 个: {sorted(missing)}", file=sys.stderr)

    docs = [
        {
            "doc_id": uid,
            "text": payload["content"],
            "source": payload["source"],
            "language": PATH_LANG[dataset],
            "meta": {"style": payload["style"], "shard": payload["shard"]},
        }
        for uid, payload in contents.items()
    ]

    out_dir = output_root / dataset / "toxicity_v3_recheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_doc, summary = compute_toxicity_v3(
        docs,
        recall_model_path=cfg["recall_model_path"],
        judge_model_path=cfg["judge_model_path"],
        chunk_size=cfg.get("chunk_size", 512),
        chunk_overlap=cfg.get("chunk_overlap", 64),
        recall_threshold=cfg.get("recall_threshold", 0.5),
        recall_batch_size=cfg.get("recall_batch_size", 16),
        judge_max_tokens=cfg.get("judge_max_tokens", 256),
        judge_temperature=cfg.get("judge_temperature", 0.0),
        judge_gpu_mem_util=cfg.get("judge_gpu_mem_util", 0.85),
        judge_max_chunks_per_doc=cfg.get("judge_max_chunks_per_doc", 8),
    )

    # 落 per_doc.jsonl + summary.json
    with (out_dir / "per_doc.jsonl").open("w", encoding="utf-8") as f:
        for r in per_doc:
            f.write(json.dumps({
                "doc_id": r.doc_id, "scores": r.scores, "flags": r.flags,
            }, ensure_ascii=False) + "\n")

    # FP reduction: v2 全部判为 high_risk；v3 只有 llm_promote=true 才算 high_risk。
    v3_promote = sum(1 for r in per_doc if r.flags["llm_promote"])
    n_v2 = len(v2_hits)
    fp_reduction = (n_v2 - v3_promote) / n_v2 if n_v2 else 0.0
    summary["v2_high_risk_count"] = n_v2
    summary["v3_promote_count"] = v3_promote
    summary["v3_recalled_count"] = summary["recalled_docs"]
    summary["fp_reduction_vs_v2"] = round(fp_reduction, 4)
    summary["missing_from_parquet"] = sorted(missing)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 人读对照表
    lines = [
        f"# {dataset} — v2 → v3 toxicity 复审对照",
        "",
        f"- v2 标记 high_risk: **{n_v2}**",
        f"- v3 召回（XLM-R chunk ≥ {summary['recall_threshold']}）: **{summary['recalled_docs']}**",
        f"- v3 判定 promote（LLM-judge）: **{v3_promote}**",
        f"- **假阳性下降率: {fp_reduction:.1%}**",
        "",
        "| doc_id | v2_tox | v3_xlmr_max | v3_chunks_recalled | v3_verdict | reason |",
        "|--------|--------|-------------|--------------------|-----------|--------|",
    ]
    by_id = {r.doc_id: r for r in per_doc}
    for uid, v2_tox in sorted(v2_hits.items(), key=lambda kv: -kv[1]):
        r = by_id.get(uid)
        if r is None:
            lines.append(f"| `{uid}` | {v2_tox:.4f} | — | — | (missing parquet) | — |")
            continue
        judgments = r.scores.get("judgments", [])
        if not judgments:
            verdict = "(no recall)"
            reason = ""
        else:
            # 优先选 promote, 其次 discuss
            order = {"promote": 0, "discuss": 1, "benign": 2}
            top = sorted(judgments, key=lambda j: order.get(j["verdict"], 9))[0]
            verdict = top["verdict"]
            reason = top.get("reason", "")
        lines.append(
            f"| `{uid}` | {v2_tox:.4f} | {r.scores['xlmr_max']:.4f} | "
            f"{r.scores['n_chunks_recalled']} | {verdict} | {reason} |"
        )
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"  v3 召回 {summary['recalled_docs']}/{n_v2}, "
        f"promote {v3_promote}/{n_v2}, FP↓ {fp_reduction:.1%}",
        file=sys.stderr,
    )
    print(f"  -> {out_dir}", file=sys.stderr)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "stage2.yaml"))
    parser.add_argument("--only", choices=["en", "zh"], default=None,
                        help="只跑一个数据集")
    args = parser.parse_args()

    cfg_all = yaml.safe_load(Path(args.config).read_text())
    t3_cfg = cfg_all.get("toxicity_v3") or {}
    if not t3_cfg:
        sys.exit("configs/stage2.yaml 缺少 toxicity_v3: 段")

    datasets = list(DATASETS)
    if args.only == "en":
        datasets = ["ufw_en_l3"]
    elif args.only == "zh":
        datasets = ["ufw_zh_l3"]

    summaries = {}
    for ds in datasets:
        summaries[ds] = recheck_dataset(ds, t3_cfg, OUT_STAGE2)

    print("\n=== 总结 ===", file=sys.stderr)
    for ds, s in summaries.items():
        if s.get("skipped"):
            print(f"  {ds}: 无 v2 高位，跳过", file=sys.stderr)
        else:
            print(
                f"  {ds}: v2={s['v2_high_risk_count']} → v3 promote={s['v3_promote_count']}  "
                f"FP↓ {s['fp_reduction_vs_v2']:.1%}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
