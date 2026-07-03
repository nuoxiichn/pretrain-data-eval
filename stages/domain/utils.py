"""Stage 8: 专项能力 — compute functions.

Subcommands:
  parsability — code parsability via tree-sitter (ERROR node counting)
  stem        — subject + difficulty via EAI-Distill-0.5b inference
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Iterable

import numpy as np

from src.reader import Document
from src.schema import DocResult


# ── Distribution stats ───────────────────────────────────────────────────────

def _dist_stats(values: list[float], pcts: tuple = (5, 25, 50, 75, 95)) -> dict:
    if not values:
        return {}
    a = np.array(values, dtype=float)
    out: dict = {
        "count": len(values),
        "mean": round(float(a.mean()), 6),
        "std": round(float(a.std()), 6),
        "min": round(float(a.min()), 6),
        "max": round(float(a.max()), 6),
    }
    for p in pcts:
        out[f"p{p}"] = round(float(np.percentile(a, p)), 6)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Parsability (tree-sitter)
# ══════════════════════════════════════════════════════════════════════════════

_PARSER_CACHE: dict[str, tuple] = {}

# 语言 → (grammar 模块名, 取 Language 指针的函数名)。多数包为 language()，
# typescript / php 例外（一个包内含多个方言）。新增语言只需装 tree-sitter-<lang> 后加一行。
_GRAMMARS: dict[str, tuple[str, str]] = {
    "python":     ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "java":       ("tree_sitter_java", "language"),
    "go":         ("tree_sitter_go", "language"),
    "c":          ("tree_sitter_c", "language"),
    "cpp":        ("tree_sitter_cpp", "language"),
    "c-sharp":    ("tree_sitter_c_sharp", "language"),
    "rust":       ("tree_sitter_rust", "language"),
    "ruby":       ("tree_sitter_ruby", "language"),
    "php":        ("tree_sitter_php", "language_php"),
}

# 归一化：The Stack 的 `lang` 字段（如 "C++"）/常见别名 → _GRAMMARS 键。
_LANG_ALIASES: dict[str, str] = {
    "python": "python",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "java": "java",
    "go": "go", "golang": "go",
    "c": "c",
    "c++": "cpp", "cpp": "cpp",
    "c#": "c-sharp", "csharp": "c-sharp", "c-sharp": "c-sharp",
    "rust": "rust",
    "ruby": "ruby",
    "php": "php",
}


def normalize_language(language: str) -> str:
    """把 The Stack `lang` / 别名归一到 _GRAMMARS 键；未知则原样小写返回。"""
    return _LANG_ALIASES.get(language.strip().lower(), language.strip().lower())


def supported_languages() -> list[str]:
    """当前 env 实际可解析的语言（grammar 已安装）。"""
    out = []
    for name in _GRAMMARS:
        if _get_parser(name) is not None:
            out.append(name)
    return out


def _get_parser(language: str):
    """Get a (Parser, Language) tuple for the given language. Returns None if unsupported."""
    language = normalize_language(language)
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]

    try:
        from tree_sitter import Language, Parser
    except ImportError:
        return None

    spec = _GRAMMARS.get(language)
    if spec is None:
        return None
    module_name, attr = spec
    try:
        import importlib
        mod = importlib.import_module(module_name)
        lang = Language(getattr(mod, attr)())
    except (ImportError, AttributeError):
        return None

    parser = Parser(lang)
    _PARSER_CACHE[language] = (parser, lang)
    return parser, lang


def _count_errors(node) -> tuple[int, int]:
    """Count (error_nodes, total_nodes) in a tree-sitter AST.

    迭代遍历（显式栈），避免深层嵌套 AST（如大型 C/C++ 文件）触发
    Python 递归上限 RecursionError。
    """
    errors = 0
    total = 0
    stack = [node]
    while stack:
        n = stack.pop()
        total += 1
        if n.type == "ERROR" or n.is_missing:
            errors += 1
        stack.extend(n.children)
    return errors, total


def _doc_language(doc: Document, default: str) -> str:
    """从文档取代码语言：优先 meta.lang（The Stack）→ language 字段 → default。"""
    meta = doc.get("meta") or {}
    raw = meta.get("lang") or meta.get("language") or doc.get("language") or default
    return normalize_language(str(raw))


def compute_parsability(
    docs: Iterable[Document],
    language: str = "python",
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Check code parsability using tree-sitter.

    language="auto" 时按每条文档的 `lang`/`language` 字段选 parser（用于混合语料）；
    否则整批用同一语言。返回 (per_doc_results, summary_dict)。
    """
    doc_list = list(docs)
    if not doc_list:
        return [], {"total_docs": 0}

    auto = normalize_language(language) == "auto"
    if not auto:
        result_pair = _get_parser(language)
        if result_pair is None:
            raise RuntimeError(
                f"tree-sitter 不支持语言 '{language}'。"
                f"确保已安装 tree-sitter 和对应语法包（如 tree-sitter-python）。"
                f" 当前可用: {supported_languages()}"
            )

    per_doc: list[DocResult] = []
    error_ratios: list[float] = []
    error_counts: list[int] = []
    has_error_count = 0
    parsed_count = 0
    unparsable_count = 0
    unsupported_count = 0
    lang_counts: Counter = Counter()          # 实际解析的语言分布
    unsupported_langs: Counter = Counter()     # auto 模式下无 grammar 的语言

    for doc in doc_list:
        doc_id = str(doc["doc_id"])
        text = str(doc.get("text") or "")
        doc_lang = _doc_language(doc, language) if auto else normalize_language(language)

        if not text.strip():
            result = DocResult(
                doc_id=doc_id,
                scores={"error_node_count": 0, "total_node_count": 0, "error_ratio": 0.0},
                flags={"has_error": False, "unsupported_lang": False},
            )
            unparsable_count += 1
        else:
            parser_pair = _get_parser(doc_lang) if auto else _get_parser(language)
            if parser_pair is None:
                # auto 模式下遇到无 grammar 的语言：记录但不计入错误率
                unsupported_count += 1
                unsupported_langs[doc_lang] += 1
                result = DocResult(
                    doc_id=doc_id,
                    scores={"error_node_count": 0, "total_node_count": 0, "error_ratio": 0.0},
                    flags={"has_error": False, "unsupported_lang": True},
                )
                if on_doc is not None:
                    on_doc(result)
                else:
                    per_doc.append(result)
                continue
            parser, _ = parser_pair
            tree = parser.parse(text.encode("utf-8", errors="replace"))
            errors, total = _count_errors(tree.root_node)
            ratio = errors / total if total > 0 else 0.0
            has_err = errors > 0

            result = DocResult(
                doc_id=doc_id,
                scores={
                    "error_node_count": errors,
                    "total_node_count": total,
                    "error_ratio": round(ratio, 6),
                },
                flags={"has_error": has_err, "unsupported_lang": False},
            )
            parsed_count += 1
            lang_counts[doc_lang] += 1
            error_ratios.append(ratio)
            error_counts.append(errors)
            if has_err:
                has_error_count += 1

        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    total = len(doc_list)
    summary = {
        "total_docs": total,
        "parsed_docs": parsed_count,
        "unparsable_docs": unparsable_count,
        "unsupported_lang_docs": unsupported_count,
        "has_error_docs": has_error_count,
        "has_error_pct": round(has_error_count / parsed_count, 4) if parsed_count else 0.0,
        "language": language,
        "parsed_lang_distribution": dict(lang_counts.most_common()),
        "unsupported_lang_distribution": dict(unsupported_langs.most_common()),
        "error_ratio_stats": _dist_stats(error_ratios),
        "error_count_stats": _dist_stats([float(c) for c in error_counts]),
    }
    return per_doc, summary


# ══════════════════════════════════════════════════════════════════════════════
# STEM classification via EAI-Distill-0.5b (FDC + Bloom)
# ══════════════════════════════════════════════════════════════════════════════
# 模型：https://hf-mirror.com/EssentialAI/eai-distill-0.5b
# 输出 10 行 "{primary},{secondary or skip}" 格式（FDC / Bloom*2 / DocType-v1 /
# Extraction / Missing / DocType-v2 / ReasoningDepth / TechCorrect / EduLevel）。
# 本实现接入其中三维：FDC（学科）、reasoning_depth、educational_level。

FDC_TOP_LABELS: dict[int, str] = {
    0:   "General/Computer",
    100: "Philosophy",
    200: "Religion",
    300: "Social Sci",
    400: "Language",
    500: "Pure Science",
    600: "Technology",
    700: "Arts",
    800: "Literature",
    900: "History/Geography",
}

# CS / 自然科学 / 应用技术 视为 STEM。
STEM_TOP_CLASSES: frozenset[int] = frozenset({0, 500, 600})

_FDC_PRIMARY_RE = re.compile(r"^(\d{1,3})")


def _chunk_text(text: str, max_char_per_doc: int = 30000) -> str:
    """超长文本取首/中/尾各 1/3 拼接（确定性版本，便于复现）。

    模型 README 给的 chunk_text 用随机中点，这里改成几何中点。
    """
    if len(text) <= max_char_per_doc:
        return text
    chunk_size = max_char_per_doc // 3
    start = text[:chunk_size]
    mid_center = len(text) // 2
    middle = text[mid_center - chunk_size // 2 : mid_center + chunk_size // 2]
    end = text[-chunk_size:]
    return f"[beginning]\n{start}\n[middle]\n{middle}\n[end]\n{end}"


def _parse_primary(line: str) -> str | None:
    """取一行中逗号前的 primary 字段；'skip' 或空视为 None。"""
    if not line:
        return None
    head = line.split(",", 1)[0].strip()
    if not head or head.lower() == "skip":
        return None
    return head


def _parse_int_primary(line: str) -> int | None:
    p = _parse_primary(line)
    if p is None:
        return None
    try:
        return int(p)
    except ValueError:
        return None


def _parse_eai_output(text: str) -> dict:
    """解析 EAI-Distill 的 10 行 csv-like 输出。

    返回的 fdc_top_class 是杜威顶层（0/100/200/.../900）。
    任一关键字段缺失或 FDC 解析失败 → parse_failed=True。
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]

    fdc_raw = _parse_primary(lines[0]) if len(lines) > 0 else None
    reasoning_depth = _parse_int_primary(lines[7]) if len(lines) > 7 else None
    educational_level = _parse_int_primary(lines[9]) if len(lines) > 9 else None

    fdc_top_class: int | None = None
    if fdc_raw:
        m = _FDC_PRIMARY_RE.match(fdc_raw)
        if m:
            n = int(m.group(1))
            fdc_top_class = (n // 100) * 100

    return {
        "fdc_raw": fdc_raw,
        "fdc_top_class": fdc_top_class,
        "reasoning_depth": reasoning_depth,
        "educational_level": educational_level,
        "parse_failed": fdc_top_class is None,
    }


def compute_stem(
    docs: Iterable[Document],
    model_path: str,
    batch_size: int = 8,
    max_input_chars: int = 30000,
    max_new_tokens: int = 100,
    device: str | None = None,
    high_difficulty_threshold: int = 4,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """通过 EAI-Distill-0.5b 推理给文档打 FDC 学科 / reasoning_depth / educational_level。

    Returns (per_doc_results, summary_dict).
    """
    doc_list = list(docs)
    if not doc_list:
        return [], {"total_docs": 0, "model_path": model_path}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    per_doc: list[DocResult] = []
    fdc_counter: Counter[int] = Counter()
    reasoning_counter: Counter[int] = Counter()
    edu_counter: Counter[int] = Counter()
    stem_count = 0
    high_diff_count = 0
    parse_failed_count = 0

    # 按 source（如 ultrafineweb_multi_style / ultrafineweb_qa）拆分，
    # 供 ZH high_difficulty=0% 归因等子分布分析。
    src_total: Counter[str] = Counter()
    src_stem: Counter[str] = Counter()
    src_high_diff: Counter[str] = Counter()
    src_parse_failed: Counter[str] = Counter()
    src_fdc: dict[str, Counter[int]] = {}
    src_rd: dict[str, Counter[int]] = {}
    src_edu: dict[str, Counter[int]] = {}

    n = len(doc_list)
    for start_i in range(0, n, batch_size):
        batch_docs = doc_list[start_i : start_i + batch_size]

        prompts: list[str] = []
        for doc in batch_docs:
            text = str(doc.get("text") or "")
            chunked = _chunk_text(text, max_char_per_doc=max_input_chars)
            messages = [
                {"role": "system", "content": "taxonomy"},
                {"role": "user", "content": chunked},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16384,
        ).to(device)
        input_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_ids = out_ids[:, input_len:]
        decoded = tokenizer.batch_decode(new_ids, skip_special_tokens=True)

        for doc, gen_text in zip(batch_docs, decoded):
            doc_id = str(doc["doc_id"])
            parsed = _parse_eai_output(gen_text)

            top_cls = parsed["fdc_top_class"]
            top_label = FDC_TOP_LABELS.get(top_cls) if top_cls is not None else None
            is_stem = top_cls in STEM_TOP_CLASSES if top_cls is not None else False
            rd = parsed["reasoning_depth"]
            high_diff = rd is not None and rd >= high_difficulty_threshold

            result = DocResult(
                doc_id=doc_id,
                scores={
                    "fdc_raw": parsed["fdc_raw"],
                    "fdc_top_class": top_cls,
                    "fdc_top_label": top_label,
                    "reasoning_depth": rd,
                    "educational_level": parsed["educational_level"],
                },
                flags={
                    "is_stem": is_stem,
                    "high_difficulty": high_diff,
                    "parse_failed": parsed["parse_failed"],
                },
            )

            if on_doc is not None:
                on_doc(result)
            else:
                per_doc.append(result)

            if parsed["parse_failed"]:
                parse_failed_count += 1
            if top_cls is not None:
                fdc_counter[top_cls] += 1
            if is_stem:
                stem_count += 1
            if rd is not None:
                reasoning_counter[rd] += 1
            if parsed["educational_level"] is not None:
                edu_counter[parsed["educational_level"]] += 1
            if high_diff:
                high_diff_count += 1

            src = str(doc.get("source") or "unknown")
            src_total[src] += 1
            if parsed["parse_failed"]:
                src_parse_failed[src] += 1
            if is_stem:
                src_stem[src] += 1
            if high_diff:
                src_high_diff[src] += 1
            if top_cls is not None:
                src_fdc.setdefault(src, Counter())[top_cls] += 1
            if rd is not None:
                src_rd.setdefault(src, Counter())[rd] += 1
            if parsed["educational_level"] is not None:
                src_edu.setdefault(src, Counter())[parsed["educational_level"]] += 1

    fdc_top_distribution = {}
    for cls, label in FDC_TOP_LABELS.items():
        cnt = fdc_counter.get(cls, 0)
        fdc_top_distribution[f"{cls:03d} {label}"] = {
            "docs": cnt,
            "pct": round(cnt / n, 4) if n else 0.0,
        }

    source_breakdown: dict[str, dict] = {}
    for src, src_n in src_total.items():
        src_fdc_dist = {}
        for cls, label in FDC_TOP_LABELS.items():
            cnt = src_fdc.get(src, Counter()).get(cls, 0)
            src_fdc_dist[f"{cls:03d} {label}"] = {
                "docs": cnt,
                "pct": round(cnt / src_n, 4) if src_n else 0.0,
            }
        source_breakdown[src] = {
            "total_docs": src_n,
            "stem_docs": src_stem.get(src, 0),
            "stem_pct": round(src_stem.get(src, 0) / src_n, 4) if src_n else 0.0,
            "high_difficulty_docs": src_high_diff.get(src, 0),
            "high_difficulty_pct": round(src_high_diff.get(src, 0) / src_n, 4) if src_n else 0.0,
            "parse_failed_docs": src_parse_failed.get(src, 0),
            "parse_failed_pct": round(src_parse_failed.get(src, 0) / src_n, 4) if src_n else 0.0,
            "fdc_top_distribution": src_fdc_dist,
            "reasoning_depth_distribution": dict(sorted(src_rd.get(src, Counter()).items())),
            "educational_level_distribution": dict(sorted(src_edu.get(src, Counter()).items())),
        }

    summary = {
        "total_docs": n,
        "parse_failed_docs": parse_failed_count,
        "parse_failed_pct": round(parse_failed_count / n, 4) if n else 0.0,
        "stem_docs": stem_count,
        "stem_pct": round(stem_count / n, 4) if n else 0.0,
        "high_difficulty_docs": high_diff_count,
        "high_difficulty_pct": round(high_diff_count / n, 4) if n else 0.0,
        "fdc_top_distribution": fdc_top_distribution,
        "reasoning_depth_distribution": dict(sorted(reasoning_counter.items())),
        "educational_level_distribution": dict(sorted(edu_counter.items())),
        "source_breakdown": source_breakdown,
        "model_path": model_path,
        "device": device,
        "batch_size": batch_size,
        "high_difficulty_threshold": high_difficulty_threshold,
    }
    return per_doc, summary
