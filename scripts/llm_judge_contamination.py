"""LLM-as-Judge for contamination cascade yellow verdicts.

对 cascade 产出的 yellow (verdict 未定) 文档做二阶段复审：
把 (训练文本, 匹配到的 benchmark 题目, benchmark 答案) 交给 Qwen2.5-7B-Instruct，
让它区分三类——

- contamination : 训练文本携带了 benchmark 特定实例的题干/答案信息
- abstraction   : 训练文本与 benchmark 同学科/知识点相关，但不含具体实例信息（能力抽象）
- unrelated     : 训练文本与 benchmark 主题基本无关（cascade 误召回）

Design notes
------------
- 复用 stage2 toxicity 的 vLLM judge 骨架（`_JUDGE_SYSTEM_PROMPT` 结构）
- 输入源：`outputs/stage5/ufw_*_l3_v3/cascade_sample500/*/per_doc.jsonl` 中 verdict=yellow 的 doc
- 需要重新从原 parquet 加载 doc.text（per_doc.jsonl 只有 doc_id）
- benchmark 内容从 `/mnt/public/data/contamination_v3_benchmarks/eval_aligned/*.jsonl` 拿
- 单次 doc.text 截断：取前 3000 char（yellow 匹配到的往往在文档开头/中间，实测足够）

Usage
-----
PYTHONPATH=. python scripts/llm_judge_contamination.py \\
    --datasets ufw_en_l3_v3 ufw_zh_l3_v3 \\
    --output outputs/stage5/llm_judge_v3/per_yellow_judgement.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import click

from src.reader import read_documents
from src.sampling import sample_documents


BENCH_DIR = Path("/mnt/public/data/contamination_v3_benchmarks/eval_aligned")
DATASET_DATA_ROOT = {
    "ufw_en_l3_v3": Path("/mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3"),
    "ufw_zh_l3_v3": Path("/mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_zh_l3"),
}
DATASET_CASCADE = {
    "ufw_en_l3_v3": Path("outputs/stage5/ufw_en_l3_v3/cascade_sample500"),
    "ufw_zh_l3_v3": Path("outputs/stage5/ufw_zh_l3_v3/cascade_sample500"),
}

_JUDGE_VERDICTS = ("contamination", "abstraction", "unrelated")

_JUDGE_SYSTEM_PROMPT = """你是预训练数据污染审核员。判断的唯一标准：**读者读完训练文本、不做任何推理，能否直接得到 benchmark 答案**。

【三分类定义】
- contamination：训练文本里**直接可以读到 benchmark 答案**（字面复述、翻译、近义改写、选项内容原文）；或训练文本包含题干独一无二的实例信息（相同数字组合/命名实体/选项列表/独特短语）。
- abstraction：训练文本与 benchmark 属于同一子知识点，但使用**不同的具体实例**（不同参数、不同数值、不同人名、不同上下文），只教方法/公式/背景，不给这道题的答案。
- unrelated：训练文本讨论的是与 benchmark **不同的子话题**（同一大学科但子话题不同也算），cascade 假阳召回。

【判定顺序，逐条问自己】

Q1: 训练文本里是否有一句话/一段话，直接说出了 benchmark 答案的内容？
  - 「说出答案」包括：字面复述、翻译成另一语言、近义改写、答案选项文字原文出现
  - **关键区别**：是「读到即得」，不是「给了公式让读者可以自己算」
  - 事实型答案（历史年代、物理常量、生物机制）就算是常识，只要文本里字面写出，也算污染
  - 是 → **contamination**（不用再看后面）

Q2: 训练文本是否包含 benchmark 题干的独一无二实例（相同的数字组合、命名实体、选项列表、专有短语）？
  - **同类题型 + 不同参数**不算（例如题目用 f=40cm，训练文本用 f=3.5m）
  - **同一学科的通用公式/术语/常量**不算（例如同讲万有引力但没算出题目要的量）
  - 是 → **contamination**

Q3: 训练文本讨论的**子话题**与 benchmark 相同吗？
  - 子话题相同（同一概念、同一现象、同一算法、同一定理） → **abstraction**
  - 子话题不同（同一大学科但不同子分支，例如同为经济学但一个讲定价一个讲雇佣决策） → **unrelated**
  - 完全跨学科 → **unrelated**

【判定示例（重点校准边界）】

示例 A（contamination — 答案句字面出现）
- 训练文本："The speed of sound increases approximately 0.61 m/s for each 1° C increase in temperature."
- benchmark："The speed of sound in air depends on: (A) air temperature (B) frequency (C) wavelength ..."
- benchmark 答案：A
- verdict: contamination
- reason: 训练文本明写「声速依赖温度」，正是答案 A 的内容。读到即得。

示例 B（contamination — 答案句跨语言翻译对应）
- 训练文本："Primers base pair to a specific portion of the DNA being replicated."
- benchmark（德语）："Was ist erforderlich, wenn PCR ..."
- benchmark 答案："(C) Primere, die bestimmte Teile der DNA komplementieren"
- verdict: contamination
- reason: 训练英文原句就是答案 (C) 的直译，读者选 (C) 无需推理。

示例 C（contamination — 独特参数 + 答案值都在文本里）
- 训练文本："The graph of y = 2/(x²+x-6) factors as (x-2)(x+3), giving vertical asymptotes at x=2 and x=-3, so 2 asymptotes."
- benchmark："How many vertical asymptotes does y=2/(x²+x-6) have?"
- benchmark 答案：2
- verdict: contamination
- reason: 独特分式复现 + 答案值 2 都写出来了。

示例 D（contamination — 常识型事实也算，只要字面出现）
- 训练文本："The Earth is approximately 4.6 billion years old."
- benchmark（德语）："Wie alt ist die Erde ungefähr? (C) 4,5 Milliarden Jahre"
- benchmark 答案：C
- verdict: contamination
- reason: 训练文本直接给出地球年龄数值，即使是通识常识，读者读完就能选对。

示例 E（abstraction — 同题型不同参数，公式给了但答案没给）
- 训练文本："Astronomical telescope with objective f0=3.5m and eyepiece fe=5cm; M = f0/fe = 70."
- benchmark："Objective lens f=40 cm, f-number 5, exit pupil 2 cm. Find angular magnification."
- benchmark 答案：D (4x, 10 cm)
- verdict: abstraction
- reason: 同类题型给了公式 M=f0/fe 但训练参数 3.5m/5cm 与题目 40cm 完全不同，答案要重新算。

示例 F（abstraction — 相关常量但没给答案）
- 训练文本："Isaac Newton's law W = GMm/R². Wmoon/Wearth ≈ 0.1668, using m_moon=7.35×10²² kg, m_earth=5.9743×10²⁴ kg, R_earth=6.378×10⁶ m."
- benchmark："Mass of Earth 5.97×10²⁴ kg, Moon 7.35×10²² kg, distance 3.84×10⁵ km. Gravitational force = X × 10²⁰ N. Find X."
- benchmark 答案：G (1.99)
- verdict: abstraction
- reason: 训练给了同样的地月质量常量，但没给题目要的引力 F 值；训练用的是地球半径而不是地月距离。

示例 G（abstraction — 相关背景但无答案句）
- 训练文本："ADHD treatment: psychostimulants (Adderall, Ritalin), behavioral therapy including routines and reward charts."
- benchmark："Which statement about ADHD stimulant treatment is NOT correct? (D) Low dose improves social behavior."
- benchmark 答案：D
- verdict: abstraction
- reason: 讲了 ADHD 治疗背景，但没有对应 (D) 选项的错误陈述文本，答案要靠专业推理。

示例 H（unrelated — 同大学科但完全不同子话题）
- 训练文本："Demand pricing, competitive pricing, markup calculation: markup% = (price - cost)/cost..."
- benchmark（法语）："完美竞争市场中技术创新对雇佣的影响 (A) 减少雇佣 (B) 加工资 (C) 增雇不加工资 (D) 增雇加工资"
- benchmark 答案：C
- verdict: unrelated
- reason: 同为经济学但子话题不同（定价策略 vs 边际生产力雇佣决策），训练完全不能回答题目。

示例 I（unrelated — 完全跨学科假阳）
- 训练文本："I love cooking pasta on Sunday evenings with my grandmother's recipe..."
- benchmark："Who received the IEEE Frank Rosenblatt Award in 2010?"
- verdict: unrelated
- reason: 主题完全无关，向量假阳召回。

【常见错判提醒】
- 有相关物理常量/学科术语 ≠ contamination（那是 abstraction）
- 有同类题型 + 不同参数 = abstraction
- 答案句字面/翻译/近义改写出现 = contamination（不管是不是常识）
- 同一大学科但子话题不同 = unrelated（不要滑向 abstraction）

【严格输出 JSON，仅一行】
{"verdict": "contamination|abstraction|unrelated", "confidence": 0.0-1.0, "reason": "简短说明（≤80 字）"}
"""


def _parse_judge_output(raw: str) -> dict:
    if not raw:
        return {"verdict": "abstraction", "confidence": 0.0, "reason": "empty"}
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    payload = m.group(0) if m else raw
    try:
        obj = json.loads(payload)
    except Exception:
        return {"verdict": "abstraction", "confidence": 0.0, "reason": "parse_error"}
    verdict = str(obj.get("verdict", "abstraction")).strip().lower()
    if verdict not in _JUDGE_VERDICTS:
        verdict = "abstraction"
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    reason = str(obj.get("reason", ""))[:200]
    return {"verdict": verdict, "confidence": round(conf, 4), "reason": reason}


def _load_benchmark_lookup() -> dict[str, dict]:
    """加载全部 benchmark jsonl，返回 {bench_id: {question, answer, benchmark}}.

    key 必须与 `stages/contamination/benchmarks.py::_load_local` 完全一致：
    f"{label}_{行号 idx}"，因为 cascade 生成的 matched_bench_id 用的就是这个。
    jsonl 里 row["id"] 字段是原生 id（各 benchmark 命名不一致），此处忽略。
    """
    lookup: dict[str, dict] = {}
    for jf in sorted(BENCH_DIR.glob("*.jsonl")):
        label = jf.stem
        idx = 0
        for line in jf.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            bench_id = f"{label}_{idx}"
            q = row.get("question") or row.get("prompt") or ""
            a = row.get("answer") or row.get("gold") or ""
            options = row.get("options")
            if isinstance(options, list) and options:
                q = q + "\n" + "\n".join(f"({chr(65+i)}) {o}" for i, o in enumerate(options))
            lookup[bench_id] = {"question": q, "answer": str(a), "benchmark": label}
            idx += 1
    return lookup


def _resolve_parquet(data_root: Path, part_name: str) -> Path:
    sub, fname = part_name.split("__", 1)
    return data_root / sub / f"{fname}.parquet"


def _collect_yellow_targets(datasets: list[str], input_cfg: dict, max_docs: int,
                             sample_mode: str, seed: int) -> list[dict]:
    """扫 cascade 输出目录，返回所有 yellow doc 及其 doc.text（重新读 parquet）."""
    all_targets: list[dict] = []
    for ds in datasets:
        cascade_dir = DATASET_CASCADE[ds]
        data_root = DATASET_DATA_ROOT[ds]
        if not cascade_dir.exists():
            click.echo(f"[warn] cascade dir 不存在: {cascade_dir}")
            continue

        # 先扫所有 part_dir 收集 yellow doc_id / matched_bench_id
        by_part: dict[str, list[dict]] = defaultdict(list)
        for pd in sorted(cascade_dir.iterdir()):
            if not pd.is_dir():
                continue
            per_doc = pd / "per_doc.jsonl"
            if not per_doc.exists():
                continue
            for line in per_doc.open(encoding="utf-8"):
                r = json.loads(line)
                if r["flags"].get("verdict") != "yellow":
                    continue
                sc = r["scores"]
                # 找出 matched bench id：优先 L3，其次 L2，最后 L1
                matched_id = (sc.get("l3_matched_bench_id")
                              or sc.get("l2_matched_bench_id")
                              or (sc.get("l1_matched_benchmarks") or [""])[0])
                matched_bench = (sc.get("l3_matched_benchmark")
                                 or sc.get("l2_matched_benchmark")
                                 or (sc.get("l1_matched_benchmarks") or [""])[0])
                layer = ("L3" if sc.get("l3_run") and sc.get("l3_matched_bench_id")
                         else "L2" if sc.get("l2_matched_bench_id")
                         else "L1")
                by_part[pd.name].append({
                    "dataset": ds,
                    "part": pd.name,
                    "doc_id": r["doc_id"],
                    "layer": layer,
                    "l3_cos_max": sc.get("l3_cos_max"),
                    "l2_jaccard_max": sc.get("l2_jaccard_max"),
                    "matched_bench_id": matched_id,
                    "matched_benchmark": matched_bench,
                })

        # 每 part 回读 parquet 拿 text（同 cascade batch 一致：head, max_docs=500）
        for part_name, items in by_part.items():
            parquet_path = _resolve_parquet(data_root, part_name)
            if not parquet_path.exists():
                click.echo(f"[warn] 找不到 parquet: {parquet_path}")
                continue
            docs = sample_documents(read_documents(str(parquet_path), config=input_cfg),
                                    max_docs, mode=sample_mode, seed=seed)
            text_by_id = {str(d["doc_id"]): str(d.get("text") or "") for d in docs}
            for it in items:
                it["text"] = text_by_id.get(str(it["doc_id"]), "")
                all_targets.append(it)
    return all_targets


def _make_qwen_judge(model_path: str, *, max_tokens: int, temperature: float,
                     gpu_memory_utilization: float):
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=False,
    )
    tok = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=temperature,
        top_p=1.0 if temperature == 0.0 else 0.9,
        max_tokens=max_tokens,
    )

    def _build_prompt(user_content: str) -> str:
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def judge(user_contents: list[str]) -> list[dict]:
        if not user_contents:
            return []
        prompts = [_build_prompt(c) for c in user_contents]
        outputs = llm.generate(prompts, sampling, use_tqdm=True)
        out: list[dict] = []
        for o in outputs:
            text = o.outputs[0].text if o.outputs else ""
            out.append(_parse_judge_output(text))
        return out

    return judge


def _build_user_content(target: dict, bench_info: dict, max_doc_chars: int) -> str:
    doc_text = target["text"] or ""
    if len(doc_text) > max_doc_chars:
        doc_text = doc_text[:max_doc_chars] + "……[truncated]"
    q = bench_info.get("question", "")
    a = bench_info.get("answer", "")
    return (
        f"<training_text>\n{doc_text}\n</training_text>\n\n"
        f"<benchmark_question>\n{q}\n</benchmark_question>\n\n"
        f"<benchmark_answer>\n{a}\n</benchmark_answer>"
    )


@click.command()
@click.option("--datasets", multiple=True, default=("ufw_en_l3_v3", "ufw_zh_l3_v3"),
              show_default=True, help="要复审的 cascade 数据集")
@click.option("--output", default="outputs/stage5/llm_judge_v3/per_yellow_judgement.jsonl",
              show_default=True)
@click.option("--summary", default="outputs/stage5/llm_judge_v3/summary.json",
              show_default=True)
@click.option("--model-path", default="/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
              show_default=True)
@click.option("--max-doc-chars", default=3000, type=int, show_default=True,
              help="doc.text 截断，避免 prompt 超长")
@click.option("--max-tokens", default=256, type=int, show_default=True)
@click.option("--temperature", default=0.0, type=float, show_default=True)
@click.option("--gpu-mem-util", default=0.85, type=float, show_default=True)
@click.option("--max-docs-per-part", default=500, type=int, show_default=True,
              help="必须与原 cascade batch 一致")
@click.option("--sample-mode", default="head", type=click.Choice(["head", "random"]),
              show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option("--config", "config_path", default="configs/stage5.yaml", show_default=True)
@click.option("--limit", default=0, type=int, help="调试用：只跑前 N 条 yellow")
@click.option("--dry-run", is_flag=True, help="只打印待复审数量，不加载 LLM")
def main(datasets, output, summary, model_path, max_doc_chars,
         max_tokens, temperature, gpu_mem_util,
         max_docs_per_part, sample_mode, seed, config_path, limit, dry_run):
    import yaml
    cfg = yaml.safe_load(open(config_path, encoding="utf-8")) or {}
    input_cfg = dict(cfg.get("input", {}))

    click.echo(f"[judge] 加载 benchmark ...")
    bench_lookup = _load_benchmark_lookup()
    click.echo(f"[judge] benchmark 索引: {len(bench_lookup)} 条")

    click.echo(f"[judge] 扫 cascade 输出收集 yellow doc ...")
    targets = _collect_yellow_targets(list(datasets), input_cfg,
                                       max_docs_per_part, sample_mode, seed)
    click.echo(f"[judge] 待复审 yellow doc: {len(targets)}")

    # 分布报告
    by_ds = Counter(t["dataset"] for t in targets)
    by_bench = Counter(t["matched_benchmark"] for t in targets)
    by_layer = Counter(t["layer"] for t in targets)
    click.echo(f"[judge]   dataset: {dict(by_ds)}")
    click.echo(f"[judge]   layer:   {dict(by_layer)}")
    click.echo(f"[judge]   top10 bench: {by_bench.most_common(10)}")

    # 缺 text / 缺 bench 的先剔除
    kept = []
    dropped_no_text = dropped_no_bench = 0
    for t in targets:
        if not t.get("text"):
            dropped_no_text += 1
            continue
        if t["matched_bench_id"] not in bench_lookup:
            dropped_no_bench += 1
            continue
        kept.append(t)
    if dropped_no_text or dropped_no_bench:
        click.echo(f"[judge] 跳过 no_text={dropped_no_text} no_bench={dropped_no_bench}")
    if limit > 0:
        kept = kept[:limit]
        click.echo(f"[judge] --limit={limit} 生效，实际复审 {len(kept)}")

    if dry_run:
        click.echo(f"[judge][dry-run] 预计复审 {len(kept)} 条，不加载模型")
        return

    if not kept:
        click.echo("[judge] 无待复审 doc，退出")
        return

    click.echo(f"[judge] 加载 Qwen judge: {model_path}")
    judge = _make_qwen_judge(model_path,
                              max_tokens=max_tokens,
                              temperature=temperature,
                              gpu_memory_utilization=gpu_mem_util)

    user_contents = [_build_user_content(t, bench_lookup[t["matched_bench_id"]], max_doc_chars)
                     for t in kept]
    click.echo(f"[judge] 送审 {len(user_contents)} 条 → LLM 生成 ...")
    verdicts = judge(user_contents)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for t, v in zip(kept, verdicts):
            rec = {
                "doc_id": t["doc_id"],
                "dataset": t["dataset"],
                "part": t["part"],
                "layer": t["layer"],
                "l2_jaccard_max": t["l2_jaccard_max"],
                "l3_cos_max": t["l3_cos_max"],
                "matched_bench_id": t["matched_bench_id"],
                "matched_benchmark": t["matched_benchmark"],
                "judge_verdict": v["verdict"],
                "judge_confidence": v["confidence"],
                "judge_reason": v["reason"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 聚合 summary
    verdict_dist = Counter(v["verdict"] for v in verdicts)
    contamination_by_bench: dict[str, int] = Counter()
    abstraction_by_bench: dict[str, int] = Counter()
    unrelated_by_bench: dict[str, int] = Counter()
    verdict_by_dataset: dict[str, dict[str, int]] = defaultdict(lambda: Counter())
    for t, v in zip(kept, verdicts):
        b = t["matched_benchmark"]
        if v["verdict"] == "contamination":
            contamination_by_bench[b] += 1
        elif v["verdict"] == "abstraction":
            abstraction_by_bench[b] += 1
        else:
            unrelated_by_bench[b] += 1
        verdict_by_dataset[t["dataset"]][v["verdict"]] += 1

    sm = {
        "total_yellow_reviewed": len(kept),
        "dropped_no_text": dropped_no_text,
        "dropped_no_bench": dropped_no_bench,
        "verdict_distribution": dict(verdict_dist),
        "verdict_pct": {k: v / len(kept) for k, v in verdict_dist.items()},
        "verdict_by_dataset": {k: dict(v) for k, v in verdict_by_dataset.items()},
        "contamination_by_benchmark": dict(contamination_by_bench),
        "abstraction_by_benchmark": dict(abstraction_by_bench),
        "unrelated_by_benchmark": dict(unrelated_by_bench),
        "model_path": model_path,
        "max_doc_chars": max_doc_chars,
        "temperature": temperature,
    }
    Path(summary).parent.mkdir(parents=True, exist_ok=True)
    Path(summary).write_text(json.dumps(sm, ensure_ascii=False, indent=2), encoding="utf-8")

    click.echo("\n=== LLM Judge 汇总 ===")
    click.echo(f"复审总数: {len(kept)}")
    click.echo(f"verdict:  {dict(verdict_dist)}")
    click.echo(f"contamination_by_benchmark (top): {contamination_by_bench.most_common(10)}")
    click.echo(f"  -> {out_path}")
    click.echo(f"  -> {summary}")


if __name__ == "__main__":
    main()
