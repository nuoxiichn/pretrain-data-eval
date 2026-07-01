"""把评测注册表里的 11 个 benchmark 抽成统一 JSONL（污染检测 v3 真正对齐评测后用）.

评测注册表（2026-06-29 用户确认，11 个去重后）：
  Pretrain: MMLU-Pro / GPQA-Diamond / MATH / EvalPlus(HEval+MBPP+) / LiveCodeBench / MGSM / MMMLU
  SFT:      IFEval / SimpleQA / LiveBench / MMLU-Pro / GPQA-Diamond / AIME 2025 / MATH-500 / LiveCodeBench

  *MATH 和 MATH-500 都跑（MATH-500 是 MATH 的子集，但评测同事两个都用；index 去重在 L1 hash 阶段自然完成）

输出 jsonl 字段统一为：{id, question, options(optional), answer(optional), subject(optional), code(optional)}
  - text_field = "question"（污染检测主字段）
  - code_field = "code"（仅 HumanEvalPlus / MBPPPlus，用于 code_near/ast 子命令）

输出目录：/mnt/public/data/contamination_v3_benchmarks/eval_aligned/

使用：python scripts/build_eval_aligned_benchmarks.py [--only mmlu_pro,gpqa_diamond,...]
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path
from typing import Iterable

import click

OUT_ROOT = Path("/mnt/public/data/contamination_v3_benchmarks/eval_aligned")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def write_jsonl(rows: list[dict], slug: str) -> Path:
    path = OUT_ROOT / f"{slug}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {path}  ({len(rows)} rows)")
    return path


def _glob_one(pattern: str) -> str | None:
    files = glob.glob(pattern)
    return files[0] if files else None


# ── 1. MMLU-Pro ──────────────────────────────────────────────────────────────

def build_mmlu_pro() -> None:
    import pyarrow.parquet as pq
    pat = "/root/.cache/huggingface/hub/datasets--TIGER-Lab--MMLU-Pro/snapshots/*/data/test-*.parquet"
    f = _glob_one(pat)
    if not f:
        print(f"  [SKIP] no parquet at {pat}"); return
    t = pq.read_table(f)
    rows = []
    for r in t.to_pylist():
        opts = list(r.get("options") or [])
        rows.append({
            "id": f"mmlu_pro_{r.get('question_id')}",
            "question": r["question"],
            "options": opts,
            "answer": r.get("answer"),
            "subject": r.get("category"),
        })
    write_jsonl(rows, "mmlu_pro")


# ── 2. GPQA-Diamond ──────────────────────────────────────────────────────────

def build_gpqa_diamond() -> None:
    from datasets import Dataset
    f = _glob_one("/mnt/public/code/chennuoxi/hf_cache/Idavidrein___gpqa/gpqa_diamond/0.0.0/*/gpqa-train.arrow")
    if not f:
        print("  [SKIP] gpqa diamond arrow missing"); return
    ds = Dataset.from_file(f)
    rows = []
    for i, r in enumerate(ds):
        q = (r.get("Question") or "").strip()
        if not q:
            continue
        correct = r.get("Correct Answer") or ""
        incs = [r.get(f"Incorrect Answer {k}") or "" for k in (1, 2, 3)]
        opts = [correct] + incs
        rows.append({
            "id": f"gpqa_diamond_{r.get('Record ID') or i}",
            "question": q,
            "options": opts,
            "answer": correct,
            "subject": r.get("High-level domain") or r.get("Subdomain"),
        })
    write_jsonl(rows, "gpqa_diamond")


# ── 3. MATH (Hendrycks, 7 子科目 test split) ─────────────────────────────────

def build_math() -> None:
    import pyarrow.parquet as pq
    root = "/root/.cache/huggingface/hub/datasets--EleutherAI--hendrycks_math/snapshots"
    snaps = sorted(glob.glob(f"{root}/*"))
    if not snaps:
        print("  [SKIP] hendrycks_math snapshots missing"); return
    snap = snaps[0]
    rows = []
    for subj_dir in sorted(p for p in glob.glob(f"{snap}/*") if os.path.isdir(p)):
        subject = os.path.basename(subj_dir)
        for f in sorted(glob.glob(f"{subj_dir}/test-*.parquet")):
            t = pq.read_table(f)
            for i, r in enumerate(t.to_pylist()):
                q = (r.get("problem") or "").strip()
                if not q:
                    continue
                rows.append({
                    "id": f"math_{subject}_{i}",
                    "question": q,
                    "answer": r.get("solution"),  # MATH 原集无独立 final answer 字段, 整个 solution 作为参考
                    "subject": subject,
                })
    write_jsonl(rows, "math")


# ── 4. MATH-500 (HuggingFaceH4) ──────────────────────────────────────────────

def build_math_500() -> None:
    from datasets import Dataset
    f = _glob_one("/mnt/public/code/chennuoxi/hf_cache/HuggingFaceH4___math-500/default/0.0.0/*/math-500-test.arrow")
    if not f:
        print("  [SKIP] math-500 arrow missing"); return
    ds = Dataset.from_file(f)
    rows = []
    for i, r in enumerate(ds):
        q = (r.get("problem") or "").strip()
        if not q:
            continue
        rows.append({
            "id": f"math_500_{r.get('unique_id') or i}",
            "question": q,
            "answer": r.get("answer"),
            "subject": r.get("subject"),
        })
    write_jsonl(rows, "math_500")


# ── 5. EvalPlus = HumanEvalPlus + MBPPPlus ───────────────────────────────────

def build_evalplus() -> None:
    import pyarrow.parquet as pq
    rows = []
    # HumanEvalPlus (parquet)
    f = _glob_one("/root/.cache/huggingface/hub/datasets--evalplus--humanevalplus/snapshots/*/data/test-*.parquet")
    if f:
        t = pq.read_table(f)
        for r in t.to_pylist():
            rows.append({
                "id": f"humaneval_plus_{r.get('task_id')}",
                "question": r.get("prompt") or "",
                "code": r.get("canonical_solution") or "",
                "subject": "humaneval_plus",
            })
    else:
        print("  [WARN] humanevalplus parquet missing")
    # MBPPPlus (jsonl @ data_mixture)
    p = "/mnt/public/code/data_mixture/evalchemy/eval/chat_benchmarks/MBPPPlus/data/mbppplus.jsonl"
    if os.path.exists(p):
        with open(p) as fh:
            for line in fh:
                d = json.loads(line)
                rows.append({
                    "id": f"mbpp_plus_{d.get('task_id')}",
                    "question": d.get("prompt") or "",
                    "code": d.get("code") or "",
                    "subject": "mbpp_plus",
                })
    else:
        print("  [WARN] mbppplus jsonl missing")
    write_jsonl(rows, "evalplus")


# ── 6. LiveCodeBench (multi-version test*.jsonl) ─────────────────────────────

def build_livecodebench() -> None:
    snap_glob = "/root/.cache/huggingface/hub/datasets--livecodebench--code_generation_lite/snapshots/*/test*.jsonl"
    files = sorted(glob.glob(snap_glob))
    if not files:
        print("  [SKIP] livecodebench jsonl missing"); return
    rows = []
    seen_qid = set()  # 不同 release 版本可能重叠，按 question_id 去重
    for f in files:
        rel = os.path.basename(f).replace(".jsonl", "")
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                qid = d.get("question_id") or f"{rel}_{len(seen_qid)}"
                if qid in seen_qid:
                    continue
                seen_qid.add(qid)
                q = d.get("question_content") or ""
                if not q.strip():
                    continue
                rows.append({
                    "id": f"livecodebench_{qid}",
                    "question": q,
                    "subject": d.get("platform"),
                    "answer": d.get("difficulty"),
                })
    write_jsonl(rows, "livecodebench")


# ── 7. MGSM (juletxara, 11 langs) ────────────────────────────────────────────

def build_mgsm() -> None:
    import pyarrow.parquet as pq
    snap = _glob_one("/root/.cache/huggingface/hub/datasets--juletxara--mgsm/snapshots/*")
    if not snap:
        print("  [SKIP] mgsm snapshot missing"); return
    rows = []
    for lang_dir in sorted(p for p in glob.glob(f"{snap}/*") if os.path.isdir(p)):
        lang = os.path.basename(lang_dir)
        f = _glob_one(f"{lang_dir}/test-*.parquet")
        if not f:
            continue
        t = pq.read_table(f)
        for i, r in enumerate(t.to_pylist()):
            q = (r.get("question") or "").strip()
            if not q:
                continue
            rows.append({
                "id": f"mgsm_{lang}_{i}",
                "question": q,
                "answer": str(r.get("answer_number") if r.get("answer_number") is not None else r.get("answer") or ""),
                "subject": f"mgsm_{lang}",
            })
    write_jsonl(rows, "mgsm")


# ── 8. MMMLU (OpenAI, 15 langs CSV) ──────────────────────────────────────────

def build_mmmlu() -> None:
    snap = _glob_one("/root/.cache/huggingface/hub/datasets--openai--MMMLU/snapshots/*")
    if not snap:
        print("  [SKIP] MMMLU snapshot missing"); return
    rows = []
    for csv_path in sorted(glob.glob(f"{snap}/test/mmlu_*.csv")):
        lang = os.path.basename(csv_path).replace("mmlu_", "").replace(".csv", "")
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, r in enumerate(reader):
                q = (r.get("Question") or "").strip()
                if not q:
                    continue
                opts = [r.get(k, "") for k in ("A", "B", "C", "D")]
                rows.append({
                    "id": f"mmmlu_{lang}_{i}",
                    "question": q,
                    "options": opts,
                    "answer": r.get("Answer"),
                    "subject": f"mmmlu_{lang}",
                })
    write_jsonl(rows, "mmmlu")


# ── 9. IFEval (google) ───────────────────────────────────────────────────────

def build_ifeval() -> None:
    f = _glob_one("/root/.cache/huggingface/hub/datasets--google--IFEval/snapshots/*/ifeval_input_data.jsonl")
    if not f:
        print("  [SKIP] ifeval jsonl missing"); return
    rows = []
    with open(f) as fh:
        for i, line in enumerate(fh):
            d = json.loads(line)
            q = (d.get("prompt") or "").strip()
            if not q:
                continue
            rows.append({
                "id": f"ifeval_{d.get('key', i)}",
                "question": q,
                "subject": ",".join(d.get("instruction_id_list") or [])[:120],
            })
    write_jsonl(rows, "ifeval")


# ── 10. SimpleQA (basicv8vc) ─────────────────────────────────────────────────

def build_simpleqa() -> None:
    from datasets import Dataset
    f = _glob_one("/mnt/public/code/chennuoxi/hf_cache/basicv8vc___simple_qa/default/0.0.0/*/simple_qa-test.arrow")
    if not f:
        print("  [SKIP] simpleqa arrow missing"); return
    ds = Dataset.from_file(f)
    rows = []
    for i, r in enumerate(ds):
        q = (r.get("problem") or "").strip()
        if not q:
            continue
        rows.append({
            "id": f"simpleqa_{i}",
            "question": q,
            "answer": str(r.get("answer") or ""),
        })
    write_jsonl(rows, "simpleqa")


# ── 11. LiveBench (6 子集, parquet, 题在 turns[0]) ───────────────────────────

LIVEBENCH_SUBSETS = ("coding", "math", "reasoning", "language", "data_analysis", "instruction_following")


def build_livebench() -> None:
    import pyarrow.parquet as pq
    rows = []
    for sub in LIVEBENCH_SUBSETS:
        f = _glob_one(f"/root/.cache/huggingface/hub/datasets--livebench--{sub}/snapshots/*/data/test-*.parquet")
        if not f:
            print(f"  [WARN] livebench/{sub} missing"); continue
        t = pq.read_table(f)
        for r in t.to_pylist():
            turns = r.get("turns") or []
            q = (turns[0] if turns else "").strip()
            if not q:
                continue
            rows.append({
                "id": f"livebench_{sub}_{r.get('question_id')}",
                "question": q,
                "subject": f"livebench_{sub}",
                "answer": str(r.get("ground_truth") or "")[:200],
            })
    write_jsonl(rows, "livebench")


# ── 12. AIME 2025 (test split = 实际题目) ────────────────────────────────────

def build_aime_2025() -> None:
    import pyarrow.parquet as pq
    # test 与 train 内容相同（30 题 AIME I/II 题库），用 test split
    f = _glob_one("/root/.cache/huggingface/hub/datasets--test-time-compute--aime_2025/snapshots/*/data/test-*.parquet")
    if not f:
        print("  [SKIP] aime_2025 parquet missing"); return
    t = pq.read_table(f)
    rows = []
    for i, r in enumerate(t.to_pylist()):
        q = (r.get("question") or "").strip()
        if not q:
            continue
        meta = r.get("metadata") or {}
        rows.append({
            "id": f"aime_2025_{meta.get('problem_idx', i)}",
            "question": q,
            "answer": str(r.get("answer") or ""),
            "subject": "aime_2025",
        })
    write_jsonl(rows, "aime_2025")


# ── 调度 ──────────────────────────────────────────────────────────────────────

BUILDERS = {
    "mmlu_pro": build_mmlu_pro,
    "gpqa_diamond": build_gpqa_diamond,
    "math": build_math,
    "math_500": build_math_500,
    "evalplus": build_evalplus,
    "livecodebench": build_livecodebench,
    "mgsm": build_mgsm,
    "mmmlu": build_mmmlu,
    "ifeval": build_ifeval,
    "simpleqa": build_simpleqa,
    "livebench": build_livebench,
    "aime_2025": build_aime_2025,
}


@click.command()
@click.option("--only", default=None, help="逗号分隔的 builder 名（默认全跑）")
def main(only: str | None) -> None:
    targets: Iterable[str] = BUILDERS.keys() if not only else [s.strip() for s in only.split(",") if s.strip()]
    for slug in targets:
        fn = BUILDERS.get(slug)
        if not fn:
            print(f"[WARN] unknown builder: {slug}"); continue
        print(f"[build] {slug}")
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {slug}: {type(e).__name__}: {e}")
    print("done.")


if __name__ == "__main__":
    main()
