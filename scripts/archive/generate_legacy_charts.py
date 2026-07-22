#!/usr/bin/env python3
"""从聚合 / 采样 summary.json 生成报告图表。

用法:
  python scripts/archive/generate_legacy_charts.py [--output-dir outputs/report/charts]

自动查找 outputs/ 下的 aggregated_summary.json 和最新时间戳的 summary.json。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
})

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"


def _load(path: Path) -> dict | None:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return None


def _latest_sample(stage_dir: Path, subcmd: str, prefix: str) -> dict | None:
    """找 stage_dir 下 prefix_* 目录中最新的 summary.json。"""
    candidates = sorted(stage_dir.glob(f"{prefix}_*/{subcmd}/summary.json"), reverse=True)
    if not candidates:
        return None
    with candidates[0].open(encoding="utf-8") as f:
        return json.load(f)


def _save(fig, name: str, charts_dir: Path):
    p = charts_dir / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  [OK] {p.name}")


# ── Chart generators ─────────────────────────────────────────────────────────

def chart_secrets(charts_dir: Path):
    en = _load(OUT_DIR / "stage2/ufw_en_l3/secrets/aggregated_summary.json")
    zh = _load(OUT_DIR / "stage2/ufw_zh_l3/secrets/aggregated_summary.json")
    if not en or not zh:
        return
    en_dist = en.get("rule_distribution", {})
    zh_dist = zh.get("rule_distribution", {})
    all_rules = sorted(set(list(en_dist.keys())[:10] + list(zh_dist.keys())[:10]),
                       key=lambda r: en_dist.get(r, 0) + zh_dist.get(r, 0), reverse=True)[:12]
    en_vals = [en_dist.get(r, 0) for r in all_rules]
    zh_vals = [zh_dist.get(r, 0) for r in all_rules]

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(all_rules))
    ax.bar(x - 0.2, en_vals, 0.4, label="EN", color="#4C72B0")
    ax.bar(x + 0.2, zh_vals, 0.4, label="ZH", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(all_rules, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Stage 2: Secret Rule Distribution (EN vs ZH)")
    ax.legend()
    ax.set_yscale("log")
    fig.subplots_adjust(bottom=0.22)
    _save(fig, "stage2_secrets_rules", charts_dir)


def chart_exact_dedup(charts_dir: Path):
    en = _load(OUT_DIR / "stage4/ufw_en_l3/exact/aggregated_summary.json")
    zh = _load(OUT_DIR / "stage4/ufw_zh_l3/exact/aggregated_summary.json")
    if not en or not zh:
        return
    metrics = ["exact_dup_pct", "para_dup_pct"]
    labels = ["Doc-level Exact Dup", "Para-level Dup"]
    en_vals = [en.get(m, 0) * 100 for m in metrics]
    zh_vals = [zh.get(m, 0) * 100 for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, en_vals, w, label="EN", color="#4C72B0")
    ax.bar(x + w / 2, zh_vals, w, label="ZH", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Stage 4: Exact Deduplication (EN vs ZH)")
    ax.legend()
    ymax = max(max(en_vals), max(zh_vals))
    offset = ymax * 0.05 if ymax > 0 else 0.001
    for i, (e, z) in enumerate(zip(en_vals, zh_vals)):
        ax.text(i - w / 2, e + offset, f"{e:.4f}%", ha="center", fontsize=7, va="bottom")
        ax.text(i + w / 2, z + offset, f"{z:.4f}%", ha="center", fontsize=7, va="bottom")
    ax.set_ylim(top=ymax * 1.25 if ymax > 0 else 0.01)
    _save(fig, "stage4_exact_dedup", charts_dir)


def chart_ngram_dedup(charts_dir: Path):
    en = _load(OUT_DIR / "stage4/ufw_en_l3/ngram/aggregated_summary.json")
    zh = _load(OUT_DIR / "stage4/ufw_zh_l3/ngram/aggregated_summary.json")
    if not en or not zh:
        return
    labels = ["Doc Contaminated", "Para Contaminated"]
    en_vals = [en.get("contaminated_pct", 0) * 100, en.get("contaminated_para_pct", 0) * 100]
    zh_vals = [zh.get("contaminated_pct", 0) * 100, zh.get("contaminated_para_pct", 0) * 100]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, en_vals, w, label="EN", color="#4C72B0")
    ax.bar(x + w / 2, zh_vals, w, label="ZH", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Stage 4: N-gram Fuzzy Dedup (EN vs ZH)")
    ax.legend()
    ymax = max(max(en_vals), max(zh_vals))
    offset = ymax * 0.05 if ymax > 0 else 0.001
    for i, (e, z) in enumerate(zip(en_vals, zh_vals)):
        ax.text(i - w / 2, e + offset, f"{e:.4f}%", ha="center", fontsize=7, va="bottom")
        ax.text(i + w / 2, z + offset, f"{z:.4f}%", ha="center", fontsize=7, va="bottom")
    ax.set_ylim(top=ymax * 1.25 if ymax > 0 else 0.01)
    _save(fig, "stage4_ngram_dedup", charts_dir)


def chart_quality(charts_dir: Path):
    en = _latest_sample(OUT_DIR / "stage3", "quality", "ufw_en_l3")
    zh = _latest_sample(OUT_DIR / "stage3", "quality", "ufw_zh_l3")
    if not en or not zh:
        return
    en_pcts = en.get("filter_fail_pcts", {})
    zh_pcts = zh.get("filter_fail_pcts", {})
    filters = sorted(set(list(en_pcts.keys()) + list(zh_pcts.keys())))
    if not filters:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(filters))
    w = 0.35
    en_vals = [en_pcts.get(f, 0) * 100 for f in filters]
    zh_vals = [zh_pcts.get(f, 0) * 100 for f in filters]
    ax.bar(x - w / 2, en_vals, w, label="EN", color="#4C72B0")
    ax.bar(x + w / 2, zh_vals, w, label="ZH", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(filters, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Fail Rate (%)")
    ax.set_title("Stage 3: Quality Filter Fail Rates (EN vs ZH)")
    ax.legend()
    fig.subplots_adjust(bottom=0.20)
    _save(fig, "stage3_quality_filters", charts_dir)


def chart_pii(charts_dir: Path):
    en = _load(OUT_DIR / "stage2/ufw_en_l3/pii_v2/aggregated_summary.json")
    zh = _load(OUT_DIR / "stage2/ufw_zh_l3/pii_v2/aggregated_summary.json")
    if not en:
        en = _load(OUT_DIR / "stage2/ufw_en_l3/pii/aggregated_summary.json")
    if not zh:
        zh = _load(OUT_DIR / "stage2/ufw_zh_l3/pii/aggregated_summary.json")
    if not en and not zh:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, lang in zip(axes, [en, zh], ["EN", "ZH"]):
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"PII Entity Types ({lang})")
            continue
        dist = data.get("entity_type_distribution", {})
        if not dist:
            continue
        labels = list(dist.keys())[:10]
        values = [dist[k] for k in labels]
        y = np.arange(len(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        ax.barh(y, values, color=colors, height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        hit_pct = data.get("hit_pct", 0)
        ax.set_title(f"PII Entity Types ({lang})\nhit_pct={hit_pct:.2%}", fontsize=10)
        for i, v in enumerate(values):
            ax.text(v + max(values) * 0.02, i, f"{v:,}", va="center", fontsize=7)
    fig.subplots_adjust(wspace=0.45)
    _save(fig, "stage2_pii_entities", charts_dir)


def chart_stem(charts_dir: Path):
    en = _latest_sample(OUT_DIR / "stage8", "stem", "ufw_en_l3_sample")
    zh = _latest_sample(OUT_DIR / "stage8", "stem", "ufw_zh_l3_sample")
    if not en or not zh:
        return
    subjects = ["cs", "math", "physics", "chemistry", "biology", "engineering", "medicine"]
    en_dist = en.get("subject_distribution", {})
    zh_dist = zh.get("subject_distribution", {})
    en_vals = [en_dist.get(s, {}).get("pct", 0) * 100 for s in subjects]
    zh_vals = [zh_dist.get(s, {}).get("pct", 0) * 100 for s in subjects]

    angles = np.linspace(0, 2 * np.pi, len(subjects), endpoint=False).tolist()
    en_vals_r = en_vals + [en_vals[0]]
    zh_vals_r = zh_vals + [zh_vals[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, en_vals_r, "o-", label="EN", color="#4C72B0", markersize=5)
    ax.fill(angles, en_vals_r, alpha=0.15, color="#4C72B0")
    ax.plot(angles, zh_vals_r, "o-", label="ZH", color="#DD8452", markersize=5)
    ax.fill(angles, zh_vals_r, alpha=0.15, color="#DD8452")
    ax.set_thetagrids(np.degrees(angles[:-1]), subjects, fontsize=9)
    ax.set_title("Stage 8: STEM Subject Distribution (%)", pad=20, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))
    _save(fig, "stage8_stem_radar", charts_dir)


def chart_tokenize(charts_dir: Path):
    en = _latest_sample(OUT_DIR / "stage10", "tokenize", "ufw_en_l3_sample")
    zh = _latest_sample(OUT_DIR / "stage10", "tokenize", "ufw_zh_l3_sample")
    if not en or not zh:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    en_f = en.get("fertility_stats", {})
    zh_f = zh.get("fertility_stats", {})
    data_en = [en_f.get("min", 0), en_f.get("p25", 0), en_f.get("p50", 0),
               en_f.get("p75", 0), en_f.get("max", 0)]
    data_zh = [zh_f.get("min", 0), zh_f.get("p25", 0), zh_f.get("p50", 0),
               zh_f.get("p75", 0), zh_f.get("max", 0)]

    bp = ax.boxplot(
        [data_en, data_zh],
        tick_labels=["EN", "ZH"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][1].set_facecolor("#DD8452")
    ax.set_ylabel("Token / Char Fertility")
    ax.set_title("Stage 10: Token Fertility (Llama-2-7B)\nEN mean={:.3f}  ZH mean={:.3f}".format(
        en_f.get("mean", 0), zh_f.get("mean", 0)))
    _save(fig, "stage10_fertility", charts_dir)


def chart_binoculars(charts_dir: Path):
    en = _latest_sample(OUT_DIR / "stage7", "binoculars", "ufw_en_l3")
    if not en:
        return
    stats = en.get("score_stats", {})
    threshold = en.get("threshold", 0.8536)

    fig, ax = plt.subplots(figsize=(8, 4))
    mean = stats.get("mean", 1.16)
    std = stats.get("std", 0.08)
    x = np.linspace(mean - 4 * std, mean + 4 * std, 200)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    ax.plot(x, y, color="#4C72B0", linewidth=2)
    ax.fill_between(x, y, alpha=0.2, color="#4C72B0")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold={threshold}")
    ax.set_xlabel("Binoculars Score")
    ax.set_ylabel("Density")
    ax.set_title(f"Stage 7: Binoculars Score Distribution (EN, n={en.get('total_docs_scanned', 0)})")
    ax.legend()
    _save(fig, "stage7_binoculars", charts_dir)


def chart_stage6_experiments(charts_dir: Path):
    ctrl_pos_ppl = _load(OUT_DIR / "stage6/control_positive/ppl/summary.json")
    ctrl_neg_ppl = _load(OUT_DIR / "stage6/control_negative/ppl/summary.json")
    ufw_ppl = _load(OUT_DIR / "stage6/ufw_en_500/ppl/summary.json")
    ctrl_pos_edu = _load(OUT_DIR / "stage6/control_positive/edu/summary.json")
    ctrl_neg_edu = _load(OUT_DIR / "stage6/control_negative/edu/summary.json")
    ufw_edu = _load(OUT_DIR / "stage6/ufw_en_500/edu/summary.json")
    ctrl_pos_ms = _load(OUT_DIR / "stage6/control_positive/model_score/summary.json")
    ctrl_neg_ms = _load(OUT_DIR / "stage6/control_negative/model_score/summary.json")
    ufw_ms = _load(OUT_DIR / "stage6/ufw_en_500/model_score/summary.json")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    labels = ["OWT\n(pos)", "Raw CC\n(neg)", "UFW-L3"]
    colors = ["#55A868", "#C44E52", "#4C72B0"]

    # PPL
    ax = axes[0]
    medians = [
        ctrl_pos_ppl["ppl_stats"]["median"] if ctrl_pos_ppl else 0,
        ctrl_neg_ppl["ppl_stats"]["median"] if ctrl_neg_ppl else 0,
        ufw_ppl["ppl_stats"]["median"] if ufw_ppl else 0,
    ]
    bars = ax.bar(labels, medians, color=colors, width=0.6)
    ax.set_ylabel("Median PPL")
    ax.set_title("KenLM Perplexity", fontsize=10)
    for bar, v in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(medians) * 0.03,
                f"{v:.0f}", ha="center", fontsize=8, va="bottom")
    ax.set_ylim(top=max(medians) * 1.15)

    # EDU
    ax = axes[1]
    high_edu = [
        ctrl_pos_edu["high_edu_pct"] * 100 if ctrl_pos_edu else 0,
        ctrl_neg_edu["high_edu_pct"] * 100 if ctrl_neg_edu else 0,
        ufw_edu["high_edu_pct"] * 100 if ufw_edu else 0,
    ]
    bars = ax.bar(labels, high_edu, color=colors, width=0.6)
    ax.set_ylabel("% docs >= 3.0")
    ax.set_title("FineWeb-Edu Score", fontsize=10)
    for bar, v in zip(bars, high_edu):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(high_edu) * 0.03,
                f"{v:.1f}%", ha="center", fontsize=8, va="bottom")
    ax.set_ylim(top=max(high_edu) * 1.15)

    # Model Score
    ax = axes[2]
    hq_pcts = [
        ctrl_pos_ms["label_distribution"].get("hq", 0) / ctrl_pos_ms["total_docs"] * 100 if ctrl_pos_ms else 0,
        ctrl_neg_ms["label_distribution"].get("hq", 0) / ctrl_neg_ms["total_docs"] * 100 if ctrl_neg_ms else 0,
        ufw_ms["label_distribution"].get("hq", 0) / ufw_ms["total_docs"] * 100 if ufw_ms else 0,
    ]
    bars = ax.bar(labels, hq_pcts, color=colors, width=0.6)
    ax.set_ylabel("% labeled HQ")
    ax.set_title("DCLM fastText", fontsize=10)
    for bar, v in zip(bars, hq_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(hq_pcts) * 0.03,
                f"{v:.1f}%", ha="center", fontsize=8, va="bottom")
    ax.set_ylim(top=max(hq_pcts) * 1.15)

    fig.suptitle("Stage 6: Quality Scoring — Control Experiments (SKIP Decision)",
                 fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.85, wspace=0.3)
    _save(fig, "stage6_skip_experiments", charts_dir)


def chart_runtime_projection(charts_dir: Path):
    """基于已知耗时估算各 stage 10T 数据量的时间。"""
    ufw_size_tb = 1.06
    target_tb = 10.0
    scale = target_tb / ufw_size_tb

    stages = [
        ("S1 Stats", 50, "CPU"),
        ("S2 PII", 200, "CPU"),
        ("S2 Secrets", 80, "CPU"),
        ("S2 Toxicity", 120, "GPU"),
        ("S3 Cleaning", 180, "CPU"),
        ("S4 Exact Dedup", 100, "CPU"),
        ("S4 Ngram", 150, "CPU"),
        ("S4 MinHash", 60, "CPU"),
        ("S5 Contamination", 90, "CPU"),
        ("S7 Binoculars", 480, "GPU"),
        ("S8 Domain", 30, "CPU"),
        ("S10 Tokenize", 10, "CPU"),
    ]

    names = [s[0] for s in stages]
    ufw_hours = [s[1] / 60 for s in stages]
    proj_hours = [s[1] / 60 * scale for s in stages]
    colors = ["#DD8452" if s[2] == "GPU" else "#4C72B0" for s in stages]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(names))
    ax.barh(x, proj_hours, color=colors, alpha=0.7, height=0.6)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Estimated Hours (10T)")
    ax.set_title("Runtime Projection: UFW (~1T) → 10T Self-built Dataset")
    ax.invert_yaxis()

    xmax = max(proj_hours)
    for i, (uh, ph) in enumerate(zip(ufw_hours, proj_hours)):
        label = f"{ph:.0f}h (UFW: {uh:.0f}h)"
        if ph > xmax * 0.7:
            ax.text(ph - xmax * 0.02, i, label, va="center", ha="right",
                    fontsize=7, color="white", fontweight="bold")
        else:
            ax.text(ph + xmax * 0.01, i, label, va="center", fontsize=7)

    ax.set_xlim(right=xmax * 1.15)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#4C72B0", alpha=0.7, label="CPU"),
                       Patch(facecolor="#DD8452", alpha=0.7, label="GPU")]
    ax.legend(handles=legend_elements, loc="lower right")
    _save(fig, "runtime_projection", charts_dir)


def main():
    charts_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/report/charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating charts → {charts_dir}")

    chart_secrets(charts_dir)
    chart_exact_dedup(charts_dir)
    chart_ngram_dedup(charts_dir)
    chart_quality(charts_dir)
    chart_pii(charts_dir)
    chart_stem(charts_dir)
    chart_tokenize(charts_dir)
    chart_binoculars(charts_dir)
    chart_stage6_experiments(charts_dir)
    chart_runtime_projection(charts_dir)

    print(f"\nDone — {len(list(charts_dir.glob('*.png')))} charts generated.")


if __name__ == "__main__":
    main()
