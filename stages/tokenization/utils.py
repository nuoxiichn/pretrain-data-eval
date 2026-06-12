"""Stage 10: Tokenization — compute functions.

Computes token/char ratio (fertility), UNK rate, per-language fertility,
and code/LaTeX token expansion rate.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from src.reader import Document
from src.schema import DocResult


# ── Tokenizer loading ────────────────────────────────────────────────────────

def _load_tokenizer(tokenizer_path: str):
    """Load a HF Tokenizer from a local path or hub name."""
    from tokenizers import Tokenizer

    path = Path(tokenizer_path)
    if path.is_file():
        return Tokenizer.from_file(str(path))
    if path.is_dir():
        tok_file = path / "tokenizer.json"
        if tok_file.exists():
            return Tokenizer.from_file(str(tok_file))
    return Tokenizer.from_pretrained(tokenizer_path)


def _find_unk_id(tokenizer) -> int | None:
    for unk_str in ("[UNK]", "<unk>", "<UNK>"):
        uid = tokenizer.token_to_id(unk_str)
        if uid is not None:
            return uid
    return None


# ── Text segment extraction ──────────────────────────────────────────────────

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_LATEX_DISPLAY_RE = re.compile(r"\$\$[\s\S]*?\$\$")
_LATEX_INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)")
_LATEX_ENV_RE = re.compile(r"\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}")


def _extract_code_blocks(text: str) -> list[str]:
    return _CODE_FENCE_RE.findall(text)


def _extract_latex(text: str) -> list[str]:
    spans = _LATEX_DISPLAY_RE.findall(text)
    text_no_display = _LATEX_DISPLAY_RE.sub("", text)
    spans.extend(_LATEX_INLINE_RE.findall(text_no_display))
    spans.extend(_LATEX_ENV_RE.findall(text))
    return spans


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


# ── Main compute ─────────────────────────────────────────────────────────────

def compute_tokenization(
    docs: Iterable[Document],
    tokenizer_path: str,
    unk_threshold: float = 0.01,
    fertility_threshold: float = 5.0,
    batch_size: int = 256,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Compute tokenization statistics for each document.

    Returns (per_doc_results, summary_dict).
    When on_doc is provided, results are streamed and the returned list is empty.
    """
    doc_list = list(docs)
    if not doc_list:
        return [], {"total_docs": 0}

    tokenizer = _load_tokenizer(tokenizer_path)
    unk_id = _find_unk_id(tokenizer)

    per_doc: list[DocResult] = []
    fertilities: list[float] = []
    unk_rates: list[float] = []
    lang_fertilities: dict[str, list[float]] = defaultdict(list)
    code_fertilities: list[float] = []
    latex_fertilities: list[float] = []
    high_unk = 0
    high_fert = 0
    total_tokens = 0
    total_unks = 0
    docs_with_code = 0
    docs_with_latex = 0
    docs_with_unk = 0

    for batch_start in range(0, len(doc_list), batch_size):
        batch = doc_list[batch_start : batch_start + batch_size]
        texts = [str(d.get("text") or "") for d in batch]
        encodings = tokenizer.encode_batch(texts)

        code_segments: list[str] = []
        code_doc_map: list[int] = []
        latex_segments: list[str] = []
        latex_doc_map: list[int] = []

        for i, text in enumerate(texts):
            for block in _extract_code_blocks(text):
                code_segments.append(block)
                code_doc_map.append(i)
            for span in _extract_latex(text):
                latex_segments.append(span)
                latex_doc_map.append(i)

        code_encodings = tokenizer.encode_batch(code_segments) if code_segments else []
        latex_encodings = tokenizer.encode_batch(latex_segments) if latex_segments else []

        # aggregate code/latex tokens per doc in this batch
        code_per_doc: dict[int, tuple[int, int]] = {}  # idx -> (chars, tokens)
        for seg_idx, enc in enumerate(code_encodings):
            doc_idx = code_doc_map[seg_idx]
            prev = code_per_doc.get(doc_idx, (0, 0))
            code_per_doc[doc_idx] = (
                prev[0] + len(code_segments[seg_idx]),
                prev[1] + len(enc.ids),
            )

        latex_per_doc: dict[int, tuple[int, int]] = {}
        for seg_idx, enc in enumerate(latex_encodings):
            doc_idx = latex_doc_map[seg_idx]
            prev = latex_per_doc.get(doc_idx, (0, 0))
            latex_per_doc[doc_idx] = (
                prev[0] + len(latex_segments[seg_idx]),
                prev[1] + len(enc.ids),
            )

        for i, (doc, enc) in enumerate(zip(batch, encodings)):
            doc_id = str(doc["doc_id"])
            text = texts[i]
            token_count = len(enc.ids)
            char_count = len(text)
            fertility = token_count / char_count if char_count > 0 else 0.0

            unk_count = 0
            if unk_id is not None:
                unk_count = sum(1 for tid in enc.ids if tid == unk_id)
            unk_rate = unk_count / token_count if token_count > 0 else 0.0

            code_chars, code_tokens = code_per_doc.get(i, (0, 0))
            code_fertility = code_tokens / code_chars if code_chars > 0 else 0.0
            latex_chars, latex_tokens = latex_per_doc.get(i, (0, 0))
            latex_fertility = latex_tokens / latex_chars if latex_chars > 0 else 0.0

            is_high_unk = unk_rate > unk_threshold
            is_high_fert = fertility > fertility_threshold

            result = DocResult(
                doc_id=doc_id,
                scores={
                    "token_count": token_count,
                    "char_count": char_count,
                    "fertility": round(fertility, 6),
                    "unk_count": unk_count,
                    "unk_rate": round(unk_rate, 6),
                    "code_char_count": code_chars,
                    "code_token_count": code_tokens,
                    "code_fertility": round(code_fertility, 6),
                    "latex_char_count": latex_chars,
                    "latex_token_count": latex_tokens,
                    "latex_fertility": round(latex_fertility, 6),
                },
                flags={
                    "high_unk_rate": is_high_unk,
                    "high_fertility": is_high_fert,
                },
            )

            if on_doc is not None:
                on_doc(result)
            else:
                per_doc.append(result)

            fertilities.append(fertility)
            unk_rates.append(unk_rate)
            total_tokens += token_count
            total_unks += unk_count
            if unk_count > 0:
                docs_with_unk += 1
            if is_high_unk:
                high_unk += 1
            if is_high_fert:
                high_fert += 1

            lang = doc.get("language")
            if lang:
                lang_fertilities[str(lang)].append(fertility)
            if code_chars > 0:
                docs_with_code += 1
                code_fertilities.append(code_fertility)
            if latex_chars > 0:
                docs_with_latex += 1
                latex_fertilities.append(latex_fertility)

    total = len(doc_list)
    per_lang = {}
    for lang, ferts in sorted(lang_fertilities.items()):
        per_lang[lang] = {
            "docs": len(ferts),
            "fertility_stats": _dist_stats(ferts),
        }

    summary = {
        "total_docs": total,
        "tokenizer_path": tokenizer_path,
        "fertility_stats": _dist_stats(fertilities),
        "unk_stats": {
            "total_unk_tokens": total_unks,
            "overall_unk_rate": round(total_unks / total_tokens, 6) if total_tokens else 0.0,
            "docs_with_unk": docs_with_unk,
            "docs_with_unk_pct": round(docs_with_unk / total, 4) if total else 0.0,
            "unk_rate_stats": _dist_stats(unk_rates),
        },
        "per_language_fertility": per_lang,
        "code_expansion": {
            "docs_with_code": docs_with_code,
            "code_fertility_stats": _dist_stats(code_fertilities),
        },
        "latex_expansion": {
            "docs_with_latex": docs_with_latex,
            "latex_fertility_stats": _dist_stats(latex_fertilities),
        },
        "high_unk_rate_docs": high_unk,
        "high_unk_rate_pct": round(high_unk / total, 4) if total else 0.0,
        "high_fertility_docs": high_fert,
        "high_fertility_pct": round(high_fert / total, 4) if total else 0.0,
        "thresholds": {
            "unk_rate": unk_threshold,
            "fertility": fertility_threshold,
        },
    }
    return per_doc, summary
