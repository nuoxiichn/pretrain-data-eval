"""一次性脚本：把 opencompass_data 缓存里的中文 benchmark 抽成统一 JSONL.

输出：/mnt/public/data/contamination_v3_benchmarks/{cmmlu,ceval,agieval_zh,cmb}.jsonl

每条记录字段：{id, question, options(list|None), answer(str|None), subject(str|None)}
contamination/benchmarks.py 的 text_field 用 "question"。

使用：python scripts/build_zh_benchmarks_jsonl.py
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

SRC_ROOT = Path("/mnt/public/data/opencompass_data/data")
OUT_ROOT = Path("/mnt/public/data/contamination_v3_benchmarks")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {path}  ({len(rows)} rows)")


# ── CMMLU ────────────────────────────────────────────────────────────────────

def build_cmmlu() -> None:
    test_dir = SRC_ROOT / "cmmlu" / "test"
    rows: list[dict] = []
    for csv_path in sorted(test_dir.glob("*.csv")):
        subject = csv_path.stem
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                q = (r.get("Question") or "").strip()
                if not q:
                    continue
                opts = [r.get(k, "") for k in ("A", "B", "C", "D")]
                rows.append({
                    "id": f"cmmlu_{subject}_{r.get('', '')}",
                    "question": q,
                    "options": opts,
                    "answer": r.get("Answer"),
                    "subject": subject,
                })
    write_jsonl(rows, OUT_ROOT / "cmmlu.jsonl")


# ── C-Eval ───────────────────────────────────────────────────────────────────

def build_ceval() -> None:
    # formal_ceval/test 题目无 answer 字段（官方留作私评），用 val 补全有答案版本
    rows: list[dict] = []
    for split in ("val", "test"):
        split_dir = SRC_ROOT / "ceval" / "formal_ceval" / split
        if not split_dir.exists():
            continue
        for csv_path in sorted(split_dir.glob("*.csv")):
            subject = csv_path.stem.replace(f"_{split}", "")
            with csv_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    q = (r.get("question") or "").strip()
                    if not q:
                        continue
                    opts = [r.get(k, "") for k in ("A", "B", "C", "D")]
                    rows.append({
                        "id": f"ceval_{split}_{subject}_{r.get('id', '')}",
                        "question": q,
                        "options": opts,
                        "answer": r.get("answer"),
                        "subject": subject,
                    })
    write_jsonl(rows, OUT_ROOT / "ceval.jsonl")


# ── AGIEval (中文子集) ───────────────────────────────────────────────────────

# Chinese tasks only — 排除 logiqa-en / lsat-* / sat-* / aqua-rat / math (math 是 EN)
AGIEVAL_ZH_FILES = {
    "gaokao-biology", "gaokao-chemistry", "gaokao-chinese", "gaokao-english",
    "gaokao-geography", "gaokao-history", "gaokao-mathcloze", "gaokao-mathqa",
    "gaokao-physics", "jec-qa-ca", "jec-qa-kd", "logiqa-zh",
}


def build_agieval_zh() -> None:
    src_dir = SRC_ROOT / "AGIEval" / "data" / "v1"
    rows: list[dict] = []
    for stem in sorted(AGIEVAL_ZH_FILES):
        jp = src_dir / f"{stem}.jsonl"
        if not jp.exists():
            print(f"  skip (missing): {jp}")
            continue
        with jp.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # passage + question 合并为完整文本（passage 是阅读材料，污染检测应包含）
                passage = (obj.get("passage") or "").strip()
                q = (obj.get("question") or "").strip()
                if not q:
                    continue
                full_q = (passage + "\n\n" + q) if passage else q
                rows.append({
                    "id": f"agieval_{stem}_{i}",
                    "question": full_q,
                    "options": obj.get("options"),
                    "answer": obj.get("label") or obj.get("answer"),
                    "subject": stem,
                })
    write_jsonl(rows, OUT_ROOT / "agieval_zh.jsonl")


# ── CMB (中文医学) ───────────────────────────────────────────────────────────

def build_cmb() -> None:
    rows: list[dict] = []
    for split in ("val", "test"):
        jp = SRC_ROOT / "CMB" / f"{split}.json"
        if not jp.exists():
            continue
        with jp.open(encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("data", [])
        for i, obj in enumerate(items):
            if not isinstance(obj, dict):
                continue
            q = (obj.get("question") or "").strip()
            if not q:
                continue
            opts_field = obj.get("option") or obj.get("options")
            if isinstance(opts_field, dict):
                opts = [opts_field.get(k, "") for k in ("A", "B", "C", "D", "E")]
                opts = [o for o in opts if o]
            elif isinstance(opts_field, list):
                opts = opts_field
            else:
                opts = None
            rows.append({
                "id": f"cmb_{split}_{i}",
                "question": q,
                "options": opts,
                "answer": obj.get("answer"),
                "subject": obj.get("exam_type") or obj.get("question_type") or "cmb",
            })
    if rows:
        write_jsonl(rows, OUT_ROOT / "cmb.jsonl")
    else:
        print("  CMB: nothing extracted (skip)")


# ── ARC-Challenge (扁平化 nested question.stem) ─────────────────────────────

def build_arc_challenge() -> None:
    src = SRC_ROOT / "ARC" / "ARC-c" / "ARC-Challenge-Test.jsonl"
    if not src.exists():
        print(f"  skip (missing): {src}")
        return
    rows: list[dict] = []
    with src.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q_obj = obj.get("question", {})
            stem = q_obj.get("stem") if isinstance(q_obj, dict) else None
            if not stem:
                continue
            choices = []
            if isinstance(q_obj, dict):
                for c in q_obj.get("choices", []) or []:
                    if isinstance(c, dict):
                        choices.append(c.get("text", ""))
            rows.append({
                "id": obj.get("id") or f"arc_{i}",
                "question": stem,
                "options": choices,
                "answer": obj.get("answerKey"),
                "subject": "arc_challenge",
            })
    write_jsonl(rows, OUT_ROOT / "arc_challenge.jsonl")


def main() -> None:
    print("[1/5] CMMLU ...")
    build_cmmlu()
    print("[2/5] C-Eval ...")
    build_ceval()
    print("[3/5] AGIEval-zh ...")
    build_agieval_zh()
    print("[4/5] CMB ...")
    build_cmb()
    print("[5/5] ARC-Challenge ...")
    build_arc_challenge()
    print("done.")


if __name__ == "__main__":
    main()
