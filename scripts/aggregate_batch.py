#!/usr/bin/env python3
"""聚合 batch 产出的 per-file summary.json → 全局 aggregated_summary.json.

用法:
  python scripts/aggregate_batch.py outputs/stage2/ufw_en_l3/secrets
  python scripts/aggregate_batch.py outputs/stage4/ufw_zh_l3/exact
  python scripts/aggregate_batch.py outputs/stage2/ufw_en_l3/pii
  python scripts/aggregate_batch.py outputs/stage1/ufw_en_l3/stats  (含 percentile 近似)

自动检测子目录下的 summary.json 并按 schema 合并。
输出: <input_dir>/aggregated_summary.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def _find_summaries(base_dir: Path) -> list[dict]:
    """收集 base_dir 下所有子目录中的 summary.json。"""
    results = []
    for p in sorted(base_dir.rglob("summary.json")):
        if p.parent == base_dir:
            continue
        with p.open(encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def _merge_counter(summaries: list[dict], field: str) -> dict:
    c = Counter()
    for s in summaries:
        if field in s and isinstance(s[field], dict):
            c.update({k: v for k, v in s[field].items() if isinstance(v, (int, float))})
    return dict(c.most_common())


def _merge_distribution(summaries: list[dict], field: str) -> dict:
    """合并 {key: {docs: N, tokens: M, ...}} 类型的分布。"""
    merged: dict[str, Counter] = {}
    for s in summaries:
        dist = s.get(field, {})
        for key, val in dist.items():
            if key not in merged:
                merged[key] = Counter()
            if isinstance(val, dict):
                merged[key].update({k: v for k, v in val.items() if isinstance(v, (int, float))})
            elif isinstance(val, (int, float)):
                merged[key]["count"] += val
    result = {}
    for key in merged:
        entry = dict(merged[key])
        result[key] = entry
    return result


def _merge_length_buckets(summaries: list[dict], field: str = "length_buckets") -> dict:
    buckets: dict[str, int] = Counter()
    for s in summaries:
        for bname, bval in s.get(field, {}).items():
            if isinstance(bval, dict):
                buckets[bname] += bval.get("count", 0)
            elif isinstance(bval, (int, float)):
                buckets[bname] += int(bval)
    total = sum(buckets.values()) or 1
    return {k: {"count": v, "pct": round(v / total, 6)} for k, v in buckets.items()}


def _recompute_pct(total: int, count: int) -> float:
    return round(count / total, 6) if total > 0 else 0.0


def _collect_per_doc_values(base_dir: Path, score_fields: list[str] | str,
                            dtype=np.uint32) -> dict[str, np.ndarray]:
    """流式扫 per_doc.jsonl 抽 scores 中的数值字段，返回 dict[field -> np.ndarray].

    旧实现是 list[float]（每元素 ~28B），698M 行会吃 ~30GB；新实现 per-file bulk read
    + `re.findall` (C 实现) + numpy uint32 数组累积，内存降到 ~3GB、速度 ~5×。
    np.percentile 用 partition 算法 O(n) 给精确分位。
    """
    import re as _re
    if isinstance(score_fields, str):
        score_fields = [score_fields]
    pats = {
        f: _re.compile(rb'"' + f.encode() + rb'":\s*(-?\d+(?:\.\d+)?)')
        for f in score_fields
    }
    parts: dict[str, list[np.ndarray]] = {f: [] for f in score_fields}

    files = sorted(base_dir.rglob("per_doc.jsonl"))
    if not files:
        return {f: np.empty(0, dtype=dtype) for f in score_fields}

    t0 = time.monotonic()
    last_log = t0
    total_lines = 0

    for p in files:
        with p.open("rb") as fh:
            buf = fh.read()
        total_lines += buf.count(b"\n")
        for f, pat in pats.items():
            matches = pat.findall(buf)
            if matches:
                # bytes → uint32/float32：np.array 在 list[bytes] 上是 C 实现
                parts[f].append(np.array(matches, dtype=dtype))
        del buf
        now = time.monotonic()
        if now - last_log >= 30:
            rate = total_lines / max(now - t0, 1e-6) / 1e6
            print(f"[INFO] percentile collect: {total_lines/1e6:.1f}M lines, {rate:.2f} M/s")
            last_log = now

    out: dict[str, np.ndarray] = {}
    for f in score_fields:
        out[f] = np.concatenate(parts[f]) if parts[f] else np.empty(0, dtype=dtype)
        parts[f].clear()

    dt = time.monotonic() - t0
    sizes = {k: int(v.size) for k, v in out.items()}
    print(f"[INFO] percentile collect done: {total_lines/1e6:.1f}M lines in {dt:.1f}s "
          f"({total_lines/max(dt,1e-6)/1e6:.2f} M/s); fields={sizes}")
    return out


def _percentiles(arr: np.ndarray) -> dict:
    """精确 percentile + mean/min/max/total，O(n) partition；698M float32 约 30s。"""
    if arr.size == 0:
        return {}
    is_float = np.issubdtype(arr.dtype, np.floating)
    qs = [25, 50, 75, 90, 95, 99]
    pvals = np.percentile(arr, qs)
    def _r(x):
        return round(float(x), 2)
    if is_float:
        return {
            "count": int(arr.size),
            "total": round(float(arr.sum(dtype=np.float64)), 2),
            "mean": _r(arr.mean(dtype=np.float64)),
            "min": _r(arr.min()),
            "max": _r(arr.max()),
            "p25": _r(pvals[0]), "p50": _r(pvals[1]), "p75": _r(pvals[2]),
            "p90": _r(pvals[3]), "p95": _r(pvals[4]), "p99": _r(pvals[5]),
        }
    # 整数字段：sum/min/max 保持整数，percentile 保留 .0 但用整数显示
    return {
        "count": int(arr.size),
        "total": int(arr.sum(dtype=np.int64)),
        "mean": _r(arr.mean(dtype=np.float64)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "p25": _r(pvals[0]), "p50": _r(pvals[1]), "p75": _r(pvals[2]),
        "p90": _r(pvals[3]), "p95": _r(pvals[4]), "p99": _r(pvals[5]),
    }


# ── Stage-specific merge strategies ─────────────────────────────────────────

def merge_secrets(summaries: list[dict]) -> dict:
    total = sum(s.get("total_docs_scanned", 0) for s in summaries)
    hits = sum(s.get("docs_with_secrets", 0) for s in summaries)
    return {
        "total_docs_scanned": total,
        "docs_with_secrets": hits,
        "hit_pct": _recompute_pct(total, hits),
        "rule_distribution": _merge_counter(summaries, "rule_distribution"),
        "aggregated_from": len(summaries),
    }


def merge_pii(summaries: list[dict]) -> dict:
    total = sum(s.get("total_docs_scanned", 0) for s in summaries)
    hits = sum(s.get("docs_with_pii", 0) for s in summaries)
    mode = summaries[0].get("mode", "unknown") if summaries else "unknown"
    lang = summaries[0].get("language", "unknown") if summaries else "unknown"
    threshold = summaries[0].get("score_threshold", 0.5) if summaries else 0.5
    return {
        "total_docs_scanned": total,
        "docs_with_pii": hits,
        "hit_pct": _recompute_pct(total, hits),
        "entity_type_distribution": _merge_counter(summaries, "entity_type_distribution"),
        "mode": mode,
        "language": lang,
        "score_threshold": threshold,
        "aggregated_from": len(summaries),
    }


def merge_exact_dedup(summaries: list[dict]) -> dict:
    total = sum(s.get("total_docs", 0) for s in summaries)
    exact_dup = sum(s.get("exact_dup_docs", 0) for s in summaries)
    para_dup = sum(s.get("para_dup_docs", 0) for s in summaries)
    unique_doc = sum(s.get("unique_doc_hashes", 0) for s in summaries)
    unique_para = sum(s.get("unique_para_hashes", 0) for s in summaries)
    return {
        "total_docs": total,
        "exact_dup_docs": exact_dup,
        "exact_dup_pct": _recompute_pct(total, exact_dup),
        "unique_doc_hashes": unique_doc,
        "para_dup_docs": para_dup,
        "para_dup_pct": _recompute_pct(total, para_dup),
        "unique_para_hashes": unique_para,
        "note": "unique_*_hashes 为各文件内去重后求和，跨文件可能有重叠",
        "aggregated_from": len(summaries),
    }


def merge_ngram_dedup(summaries: list[dict]) -> dict:
    total = sum(s.get("total_docs", 0) for s in summaries)
    cont = sum(s.get("contaminated_docs", 0) for s in summaries)
    total_p = sum(s.get("total_paras", 0) for s in summaries)
    cont_p = sum(s.get("contaminated_paras", 0) for s in summaries)
    ngram = summaries[0].get("ngram_size", 13) if summaries else 13
    thresh = summaries[0].get("overlap_threshold", 0.5) if summaries else 0.5
    return {
        "total_docs": total,
        "contaminated_docs": cont,
        "contaminated_pct": _recompute_pct(total, cont),
        "total_paras": total_p,
        "contaminated_paras": cont_p,
        "contaminated_para_pct": _recompute_pct(total_p, cont_p),
        "ngram_size": ngram,
        "overlap_threshold": thresh,
        "aggregated_from": len(summaries),
    }


def merge_stats(summaries: list[dict], base_dir: Path) -> dict:
    total = sum(s.get("total_docs", 0) for s in summaries)
    buckets = _merge_length_buckets(summaries)
    domains = _merge_distribution(summaries, "domain_distribution")
    languages = _merge_distribution(summaries, "language_distribution")
    sources = _merge_distribution(summaries, "source_distribution")

    ts_present = sum(s.get("timestamp", {}).get("present", 0) for s in summaries)
    ts_missing = sum(s.get("timestamp", {}).get("missing", 0) for s in summaries)
    ts_total = ts_present + ts_missing
    ym_dist: Counter = Counter()
    for s in summaries:
        ym = s.get("timestamp", {}).get("year_month_distribution", {})
        ym_dist.update({k: v for k, v in ym.items() if isinstance(v, (int, float))})

    total_tokens_all = sum(
        s.get("token_stats", {}).get("total", 0) for s in summaries
    )
    for group in [domains, languages, sources]:
        g_total = sum(v.get("tokens", 0) for v in group.values()) or 1
        for v in group.values():
            if "tokens" in v:
                v["pct_tokens"] = round(v["tokens"] / g_total, 6)

    # 一次扫描 per_doc.jsonl 抽 char_count + token_count，numpy uint32 累积
    vals = _collect_per_doc_values(base_dir, ["char_count", "token_count"], dtype=np.uint32)
    char_arr = vals.get("char_count", np.empty(0, dtype=np.uint32))
    tok_arr = vals.get("token_count", np.empty(0, dtype=np.uint32))

    result = {
        "total_docs": total,
        "char_stats": _percentiles(char_arr) if char_arr.size else _sum_stats(summaries, "char_stats"),
        "token_stats": _percentiles(tok_arr) if tok_arr.size else _sum_stats(summaries, "token_stats"),
        "length_buckets": buckets,
        "domain_distribution": domains,
        "language_distribution": languages,
        "source_distribution": sources,
        "timestamp": {
            "present": ts_present,
            "missing": ts_missing,
            "present_pct": _recompute_pct(ts_total, ts_present),
            "year_month_distribution": dict(ym_dist.most_common()),
        },
        "aggregated_from": len(summaries),
    }
    if char_arr.size:
        result["note"] = "char_stats/token_stats 从 per_doc.jsonl 精确重算（numpy uint32）"
    else:
        result["note"] = "char_stats/token_stats 为各文件 total/count 的加权近似（无 per_doc 数据）"
    return result


def _sum_stats(summaries: list[dict], field: str) -> dict:
    total_count = sum(s.get(field, {}).get("count", 0) for s in summaries)
    total_sum = sum(s.get(field, {}).get("total", 0) for s in summaries)
    all_min = [s.get(field, {}).get("min", float("inf")) for s in summaries if field in s]
    all_max = [s.get(field, {}).get("max", 0) for s in summaries if field in s]
    return {
        "count": total_count,
        "total": round(total_sum, 2),
        "mean": round(total_sum / total_count, 2) if total_count else 0,
        "min": min(all_min) if all_min else 0,
        "max": max(all_max) if all_max else 0,
        "note": "percentiles 不可从子集 summary 精确合并，需从 per_doc.jsonl 重算",
    }


def merge_contamination_exact(summaries: list[dict]) -> dict:
    total = sum(s.get("total_docs", 0) for s in summaries)
    cont = sum(s.get("contaminated_docs", 0) for s in summaries)
    para_cont = sum(s.get("para_contaminated_docs", 0) for s in summaries)
    bench_hits: Counter = Counter()
    for s in summaries:
        bh = s.get("per_benchmark_hits", {})
        bench_hits.update({k: v for k, v in bh.items() if isinstance(v, (int, float))})
    return {
        "total_docs": total,
        "contaminated_docs": cont,
        "contaminated_pct": _recompute_pct(total, cont),
        "para_contaminated_docs": para_cont,
        "para_contaminated_pct": _recompute_pct(total, para_cont),
        "per_benchmark_hits": dict(bench_hits.most_common()),
        "aggregated_from": len(summaries),
    }


def merge_extraction(summaries: list[dict]) -> dict:
    total = sum(s.get("total_docs", s.get("total_docs_scanned", 0)) for s in summaries)
    result: dict = {"total_docs": total, "aggregated_from": len(summaries)}
    count_fields = [k for s in summaries for k in s if k.endswith("_count") or k.endswith("_docs")]
    for field in set(count_fields):
        if field == "total_docs":
            continue
        result[field] = sum(s.get(field, 0) for s in summaries)
    pct_fields = [k for k in result if k.endswith("_count") or k.endswith("_docs")]
    for field in pct_fields:
        pct_name = field.replace("_count", "_pct").replace("_docs", "_pct")
        if pct_name not in result:
            result[pct_name] = _recompute_pct(total, result[field])
    return result


def merge_quality(summaries: list[dict]) -> dict:
    total = sum(s.get("total_docs", 0) for s in summaries)
    fail_counts: Counter = Counter()
    fail_reasons: dict[str, Counter] = {}
    for s in summaries:
        fc = s.get("filter_fail_counts", {})
        fail_counts.update({k: v for k, v in fc.items() if isinstance(v, (int, float))})
        fr = s.get("filter_fail_reasons", {})
        for fname, reasons in fr.items():
            if fname not in fail_reasons:
                fail_reasons[fname] = Counter()
            if isinstance(reasons, dict):
                fail_reasons[fname].update(
                    {k: v for k, v in reasons.items() if isinstance(v, (int, float))}
                )
    fail_pcts = {k: _recompute_pct(total, v) for k, v in fail_counts.items()}
    return {
        "total_docs": total,
        "filter_fail_counts": dict(fail_counts.most_common()),
        "filter_fail_pcts": fail_pcts,
        "filter_fail_reasons": {k: dict(v.most_common()) for k, v in fail_reasons.items()},
        "aggregated_from": len(summaries),
    }


def merge_langid(summaries: list[dict]) -> dict:
    total = sum(s.get("total_docs", s.get("total_docs_scanned", 0)) for s in summaries)
    lang_dist = _merge_counter(summaries, "language_distribution")
    low_conf = sum(s.get("low_confidence_docs", 0) for s in summaries)
    mismatch = sum(s.get("lang_mismatch_docs", 0) for s in summaries)
    return {
        "total_docs": total,
        "language_distribution": lang_dist,
        "low_confidence_docs": low_conf,
        "low_confidence_pct": _recompute_pct(total, low_conf),
        "lang_mismatch_docs": mismatch,
        "lang_mismatch_pct": _recompute_pct(total, mismatch),
        "aggregated_from": len(summaries),
    }


def merge_generic(summaries: list[dict]) -> dict:
    """Fallback: sum all numeric fields, merge all dict fields as counters."""
    total_keys = set()
    for s in summaries:
        total_keys.update(s.keys())

    result: dict = {"aggregated_from": len(summaries)}
    for key in sorted(total_keys):
        samples = [s[key] for s in summaries if key in s]
        if not samples:
            continue
        first = samples[0]
        if isinstance(first, (int, float)):
            result[key] = sum(s.get(key, 0) for s in summaries)
        elif isinstance(first, dict):
            result[key] = _merge_counter(summaries, key)
        elif isinstance(first, str):
            result[key] = first

    total = result.get("total_docs", result.get("total_docs_scanned", 0))
    if total:
        for key in list(result.keys()):
            if key.endswith("_pct"):
                count_key = key.replace("_pct", "_docs")
                if count_key not in result:
                    count_key = key.replace("_pct", "_count")
                if count_key not in result:
                    count_key = key.replace("_pct", "")
                if count_key in result and isinstance(result[count_key], (int, float)):
                    result[key] = _recompute_pct(total, int(result[count_key]))

    result["note"] = "generic merge — verify fields manually"
    return result


# ── Dispatcher ───────────────────────────────────────────────────────────────

MERGE_MAP = {
    "secrets": merge_secrets,
    "pii": merge_pii,
    "pii_v2": merge_pii,
    "exact": merge_exact_dedup,
    "ngram": merge_ngram_dedup,
    "quality": merge_quality,
    "langid": merge_langid,
    "extraction": merge_extraction,
}

MERGE_MAP_WITH_DIR = {
    "stats": merge_stats,
}

MERGE_MAP_STAGE = {
    "stage5": {
        "exact": merge_contamination_exact,
    },
}


def detect_and_merge(base_dir: Path) -> dict:
    subcmd = base_dir.name
    stage = base_dir.parent.name if base_dir.parent.name.startswith("stage") else base_dir.parent.parent.name

    summaries = _find_summaries(base_dir)
    if not summaries:
        print(f"[WARN] 未找到 summary.json: {base_dir}")
        return {}

    print(f"[INFO] 找到 {len(summaries)} 个 summary.json in {base_dir}")

    stage_overrides = MERGE_MAP_STAGE.get(stage, {})
    if subcmd in stage_overrides:
        return stage_overrides[subcmd](summaries)
    if subcmd in MERGE_MAP_WITH_DIR:
        return MERGE_MAP_WITH_DIR[subcmd](summaries, base_dir)
    if subcmd in MERGE_MAP:
        return MERGE_MAP[subcmd](summaries)

    print(f"[WARN] 未知 subcmd '{subcmd}'，使用 generic merge")
    return merge_generic(summaries)


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/aggregate_batch.py <output_dir> [<output_dir2> ...]")
        print("示例: python scripts/aggregate_batch.py outputs/stage2/ufw_en_l3/secrets")
        sys.exit(1)

    for dir_arg in sys.argv[1:]:
        base_dir = Path(dir_arg).resolve()
        if not base_dir.is_dir():
            print(f"[ERROR] 不是目录: {base_dir}")
            continue

        merged = detect_and_merge(base_dir)
        if not merged:
            continue

        out_path = base_dir / "aggregated_summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"[OK] {out_path}")
        print(f"     total_docs = {merged.get('total_docs', merged.get('total_docs_scanned', '?'))}")
        print(f"     aggregated_from = {merged.get('aggregated_from', '?')} files")


if __name__ == "__main__":
    main()
