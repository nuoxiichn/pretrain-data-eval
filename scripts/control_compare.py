"""控制组对照验证：把 control_positive vs control_negative 的各维度 summary
读出来，按预期方向判定每个指标是否"positive 优于 negative"，产出方向性表。

用法:
  PYTHONPATH=. python scripts/control_compare.py
  # 读 outputs/control/{positive,negative}/<dim>/summary.json
  # 写 outputs/control/comparison.md

判定（core 维度）：
  PASS 方向符合预期（有方向性区分证据）
  FAIL 方向相反（指标或混淆因素需复核）
  TIE  两侧相对差异 < 5%（无明显区分）
aux 维度只描述、不判定（~）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CTRL = ROOT / "outputs" / "control"
POS, NEG = "positive", "negative"

# 相对差异小于此值时视为无明显区分。
REL_EPS = 0.05


def _get(d: dict, path: str):
    """点路径取值；支持派生 'a/b' 表示 d[a]/d[b]（比例）。"""
    if "//" in path:  # 派生比例：num//den
        num, den = path.split("//")
        n, m = _get(d, num), _get(d, den)
        return (n / m) if (isinstance(n, (int, float)) and m) else None
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


# 每维度：dim 目录名 → 展示名 / tier(core|aux) / 指标列表
# 指标：(展示名, summary 内路径, better)  better ∈ {'lower','higher','none'}
SPEC = [
    ("quality", "S3 质量 (Gopher+C4)", "core", [
        ("Gopher 失败率", "filter_fail_pcts.gopher_quality", "lower"),
        ("C4 失败率", "filter_fail_pcts.c4_quality", "lower"),
    ]),
    ("extraction", "S3 抽取残留", "core", [
        ("低抽取质量率", "low_extraction_quality_pct", "lower"),
        ("HTML 残留率", "html_residue_pct", "lower"),
        ("boilerplate 率", "boilerplate_pct", "lower"),
        ("mojibake 率", "mojibake_pct", "lower"),
    ]),
    ("exact", "S4 精确重复", "core", [
        ("文档级重复率", "exact_dup_pct", "lower"),
        ("段落级重复率", "para_dup_pct", "lower"),
    ]),
    ("ngram", "S4 ngram 重复", "core", [
        ("污染段落率", "contaminated_para_pct", "lower"),
    ]),
    ("minhash", "S4 近重复", "core", [
        ("近重复率", "near_dup_pct", "lower"),
    ]),
    ("tokenize", "S10 Tokenization", "core", [
        ("fertility 均值", "fertility_stats.mean", "lower"),
        ("高 fertility 文档率", "high_fertility_pct", "lower"),
    ]),
    ("stats", "S1 文档统计", "aux", [
        ("平均字符数", "char_stats.mean", "none"),
        ("平均 token 数", "token_stats.mean", "none"),
    ]),
    ("pii", "S2 PII", "aux", [
        ("命中率", "hit_pct", "none"),
    ]),
    ("secrets", "S2 Secrets", "aux", [
        ("命中率", "hit_pct", "none"),
    ]),
]


def _load(cls: str, dim: str) -> dict | None:
    f = CTRL / cls / dim / "summary.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) < 1:            # 视作比例 → 百分比
            return f"{v*100:.2f}%"
        return f"{v:.2f}"
    return str(v)


def _verdict(pos, neg, better: str) -> str:
    if better == "none":
        return "~"
    if pos is None or neg is None:
        return "ERR"
    hi = max(abs(pos), abs(neg), 1e-12)
    if abs(pos - neg) / hi < REL_EPS:
        return "TIE"
    pos_is_lower = pos < neg
    want_lower = better == "lower"
    return "PASS" if (pos_is_lower == want_lower) else "FAIL"


def main() -> int:
    rows = []
    core_pass = core_total = 0
    missing = []

    for dim, label, tier, metrics in SPEC:
        sp, sn = _load(POS, dim), _load(NEG, dim)
        if sp is None or sn is None:
            missing.append(dim)
        for i, (mname, path, better) in enumerate(metrics):
            pv = _get(sp, path) if sp else None
            nv = _get(sn, path) if sn else None
            vd = _verdict(pv, nv, better)
            arrow = {"lower": "positive 更低", "higher": "positive 更高", "none": "—"}[better]
            rows.append((label if i == 0 else "", mname, _fmt(pv), _fmt(nv), arrow, vd, tier))
            # 只用每维度首个 core 指标计入总分
            if tier == "core" and i == 0:
                core_total += 1
                if vd == "PASS":
                    core_pass += 1

    # ── 渲染 markdown ──
    out = []
    out.append("# 控制组对照验证：区分度方向性表\n")
    out.append("> control_positive = OpenWebText（已知较优）  ·  control_negative = raw Common Crawl（已知较差），各 1000 条。")
    out.append("> 一个**有效**的审计指标应把好数据排在坏数据之上（方向符合预期）。\n")
    out.append(f"**核心区分维度结论：{core_pass}/{core_total} 个维度方向符合预期**"
               f"（按每维度头部指标计）。\n")
    out.append("| 维度 | 指标 | positive | negative | 预期 | 判定 |")
    out.append("|------|------|---------:|---------:|------|:---:|")
    for label, mname, pv, nv, arrow, vd, tier in rows:
        out.append(f"| {label} | {mname} | {pv} | {nv} | {arrow} | {vd} |")

    out.append("\n**判定图例**：PASS 方向符合 · FAIL 方向相反 · "
               "TIE 差异 <5% · ~ 描述性维度 · ERR 数据缺失\n")
    if missing:
        out.append(f"> 缺失产物（跑批未完成或失败）：{', '.join(missing)}\n")
    out.append("## 边界（诚实标注）\n")
    out.append("- 本实验只验证主指标的**方向性区分度**，没有逐条真值，不能估计 precision 或 recall。")
    out.append("- 对照是**自然网页**；对**合成数据特有失效模式**（模板坍缩 / 事实漂移 / 过度均匀）**零覆盖**，"
               "那需要 合成 vs 人写 对照集，属后续。")
    out.append("- raw CC 与 OpenWebText 有领域差异；方向符合支持"
               "「指标有区分力」，但个别偏离也可能来自领域而非质量，aux 维度不做强判定。")
    out.append("- FAIL / TIE 不代表方法一定错误，应先用 per_doc 区分实现问题、领域差异和数据结构混淆。")
    out.append("- **段落级重复率的反向结果属于结构混淆**：raw CC 多为无段落结构的短 blob"
               "（negative 仅 995 个 ≥50 字段落 vs positive 15796），段落级去重"
               "「没料可比」被人为压低；**文档级精确去重（0% vs 3%）才是干净的质量轴**。"
               "该反例反而说明对照实验能暴露被结构混淆的指标。")
    out.append("- S3 langid 本轮未纳入：本 env 缺 `lid.176.bin` 模型文件（fastText 176 语种）。")

    text = "\n".join(out) + "\n"
    dest = CTRL / "comparison.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    print(text)
    print(f"→ {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
