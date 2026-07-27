"""Stage 3 computation utilities.

  compute_langid          — fastText lid.176 language identification (row 2)
  compute_glotlid         — GlotLID v3 fine-grained language-script ID (row 3)
  compute_lang_crosscheck — coarse(lid.176) vs fine(GlotLID) disagreement audit
  compute_quality         — Gopher Quality + C4 signals, nltk-tokenized (row 4)
  compute_extraction_audit — markup/boilerplate/mojibake residue audit (row 1)
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable

import numpy as np

from pretrain_data_eval.reader import Document
from pretrain_data_eval.schema import DocResult


# ── Language identification ───────────────────────────────────────────────────

def compute_langid(
    docs: Iterable[Document],
    model_path: str,
    top_k: int = 3,
    confidence_threshold: float = 0.7,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Identify document language using fastText lid.176 model."""
    try:
        import fasttext
        fasttext.FastText.eprint = lambda x: None  # suppress warnings
    except ImportError as exc:
        raise RuntimeError("Run: pip install fasttext") from exc

    model = fasttext.load_model(model_path)

    per_doc: list[DocResult] = []
    lang_counter: Counter = Counter()
    mismatch_count = 0
    low_conf_count = 0
    confidences: list[float] = []
    total = 0

    for doc in docs:
        text = str(doc.get("text") or "").replace("\n", " ").strip()
        if not text:
            labels, scores = ["__label__unknown"], [0.0]
        else:
            labels, scores = model.predict(text, k=top_k)

        top_lang = labels[0].replace("__label__", "")
        top_conf = float(scores[0])
        top3 = [[lb.replace("__label__", ""), round(float(sc), 4)]
                for lb, sc in zip(labels, scores)]

        lang_counter[top_lang] += 1
        confidences.append(top_conf)
        is_low_conf = top_conf < confidence_threshold
        if is_low_conf:
            low_conf_count += 1

        declared = doc.get("language")
        is_mismatch = bool(declared and declared != top_lang)
        if is_mismatch:
            mismatch_count += 1

        result = DocResult(
            doc_id=str(doc["doc_id"]),
            scores={"lang": top_lang, "confidence": round(top_conf, 4), "top3": top3},
            flags={"lang_mismatch": is_mismatch, "low_confidence": is_low_conf},
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)
        total += 1

    a = np.array(confidences) if confidences else np.array([0.0])
    summary = {
        "total_docs": total,
        "language_distribution": dict(lang_counter.most_common()),
        "mismatch_count": mismatch_count,
        "mismatch_pct": round(mismatch_count / total, 4) if total else 0.0,
        "low_confidence_count": low_conf_count,
        "low_confidence_pct": round(low_conf_count / total, 4) if total else 0.0,
        "confidence_stats": {
            "mean": round(float(a.mean()), 4),
            "p50": round(float(np.percentile(a, 50)), 4),
            "p5": round(float(np.percentile(a, 5)), 4),
            "min": round(float(a.min()), 4),
        },
    }
    return per_doc, summary


# ── Quality filters (Gopher Quality + C4), nltk-tokenized ─────────────────────
#
# 忠实复刻 DataTrove 的 GopherQualityFilter 与 C4QualityFilter 规则。DataTrove 本身
# 在本环境跑不起来（spaCy 分词器未分配 + spaCy→thinc→torch 触发 numpy ABI 冲突），
# 但其过滤逻辑是纯 Python，只有「分词器」依赖 spaCy。这里用 nltk（punkt）作真分词器
# 替代：词用 word_tokenize、句用 sent_tokenize（CJK 用 jieba 分词 + 标点切分句子）。
#
# 重复检测（Gopher Repetition / 段落-行-ngram 重复）不在此处，归 stage 4 dedup。

import re as _re
import string as _string
from collections import Counter as _Counter

_BULLET_CHARS = frozenset("•-*·‐‑–—►▶◆●○")
# 标点集合（判定「纯符号词」）：ASCII + 常见 Unicode/CJK 标点
_PUNCTUATION_SET = set(_string.punctuation) | set("…—–“”‘’«»、，。；：！？「」『』（）【】《》〈〉·～")
# DataTrove GopherQualityFilter 的英文停用词集
_STOP_WORDS_EN = {"the", "be", "to", "of", "and", "that", "have", "with"}
_STOP_WORDS_ZH = {"的", "是", "在", "了", "不", "和", "有", "这"}

# C4 常量（对齐 DataTrove c4_filters.py）
_C4_CITATION_RE = _re.compile(r"\[\d*]|\[edit]|\[citation needed]")
_C4_END_PUNCT_LATIN = (".", "?", "!", '"', "'")
_C4_END_PUNCT_CJK = ("。", "！", "？", "”", "’", ".", "?", "!", '"', "'")
_C4_POLICY_SUBSTRINGS = (
    "terms of use", "privacy policy", "cookie policy",
    "uses cookies", "use of cookies", "use cookies",
)
_C4_POLICY_SUBSTRINGS_ZH = ("使用条款", "隐私政策", "cookie", "使用cookie")
_CJK_PREFIXES = ("zh", "cmn", "yue", "wuu", "nan", "ja", "jpn", "ko", "kor")
_CJK_CHAR_RE = _re.compile(r"[㐀-鿿぀-ヿ가-힯]")
_CJK_SENT_SPLIT_RE = _re.compile(r"[。！？…；;]+")


def _is_cjk_lang(lang: str | None) -> bool:
    l = (lang or "").lower()
    return any(l == p or l.startswith(p) for p in _CJK_PREFIXES)


def _word_tokenize(text: str, lang: str | None) -> list[str]:
    """真分词器：CJK 用 jieba 中文分词；其余用 nltk word_tokenize。"""
    if _is_cjk_lang(lang):
        import jieba
        return [w for w in jieba.cut(text) if w.strip()]
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)


def _sent_tokenize(text: str, lang: str | None) -> list[str]:
    """真分句：CJK 按句末标点切，其余用 nltk punkt。"""
    if _is_cjk_lang(lang):
        return [s for s in _CJK_SENT_SPLIT_RE.split(text) if s.strip()]
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)


def _gopher_quality(text: str, lang: str | None) -> tuple[bool, str | None]:
    """Gopher 质量启发式（复刻 DataTrove GopherQualityFilter）。"""
    words = _word_tokenize(text, lang)
    n_words = len(words)
    if n_words == 0:
        return False, "empty"
    non_symbol_words = [w for w in words if any(ch not in _PUNCTUATION_SET for ch in w)]
    n_non_symbol = len(non_symbol_words)

    # 文档词数上下限（按非符号词计）
    if n_non_symbol < 50:
        return False, "too_few_words"
    if n_non_symbol > 100_000:
        return False, "too_many_words"
    # 平均词长：英文 3–10，CJK（jieba 分词）1.3–10
    avg_len = sum(len(w) for w in non_symbol_words) / n_non_symbol
    min_avg_len = 1.3 if _is_cjk_lang(lang) else 3
    if avg_len < min_avg_len:
        return False, "avg_word_too_short"
    if avg_len > 10:
        return False, "avg_word_too_long"
    # 符号-词比：# 或 省略号 占词数 > 0.1
    if text.count("#") / n_words > 0.1:
        return False, "too_many_hashes"
    if (text.count("...") + text.count("…")) / n_words > 0.1:
        return False, "too_many_ellipsis"
    # 行级：>90% 以 bullet 开头 / >30% 以省略号结尾
    lines = text.splitlines()
    if lines:
        if sum(1 for l in lines if l.lstrip()[:1] and l.lstrip()[0] in _BULLET_CHARS) / len(lines) > 0.9:
            return False, "too_many_bullet_lines"
        if sum(1 for l in lines if l.rstrip().endswith(("...", "…"))) / len(lines) > 0.3:
            return False, "too_many_ellipsis_lines"
    # 词中含至少一个字母字符的比例下限。
    # DataTrove 默认 EN 0.8，但 0.8 对成品数据（学术文献、参考书目、教育材料）的
    # 高数字/引用密度文本误报严重——UFW-L3 EN 抽样 20 条命中里 19 条是真实学术
    # 内容（PMID、DOI、年份、化学单位），FP ~95%。这里把 EN 也下调到 0.6，与
    # CJK 保持一致；真正的网页噪声（导航/标签云）通常 ratio < 0.5，仍会被命中。
    min_alpha_ratio = 0.6
    if sum(1 for w in words if any(c.isalpha() for c in w)) / n_words < min_alpha_ratio:
        return False, "too_many_non_alpha_words"
    # 停用词：CJK 用中文集，英文用英文集
    if _is_cjk_lang(lang):
        if len(_STOP_WORDS_ZH.intersection(words)) < 2:
            return False, "not_enough_stop_words"
    elif lang is None or str(lang).lower().startswith("en"):
        if len(_STOP_WORDS_EN.intersection(w.lower() for w in words)) < 2:
            return False, "not_enough_stop_words"
    return True, None


def _c4_quality(text: str, lang: str | None) -> tuple[bool, str | None]:
    """C4 质量启发式只读判定（复刻 DataTrove C4QualityFilter）。

    逐行过滤：剔除超长词行、无终止标点行、词数不足行、javascript 行、policy 行；
    遇 lorem ipsum / 花括号直接判废；最后留存行的句子数 < 5 判废。不修改原文。
    """
    is_cjk = _is_cjk_lang(lang)
    end_punct = _C4_END_PUNCT_CJK if is_cjk else _C4_END_PUNCT_LATIN
    policy_subs = _C4_POLICY_SUBSTRINGS_ZH if is_cjk else _C4_POLICY_SUBSTRINGS
    min_words_per_line = 2 if is_cjk else 3
    lines = text.splitlines()
    num_sentences = 0
    kept = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # CJK 用 jieba 分词，其余用空格切分
        line_words = _word_tokenize(line, lang) if is_cjk else line.split()
        if not line_words:
            continue
        # 超长词（>1000 字符）行剔除
        if any(len(w) > 1000 for w in line_words):
            continue
        line = _C4_CITATION_RE.sub("", line)
        # 无终止标点（或以省略号结尾）行剔除
        if not line.endswith(end_punct) or line.endswith("..."):
            continue
        # 每行最少词数
        if len(line_words) < min_words_per_line:
            continue
        ll = line.lower()
        if "lorem ipsum" in ll:
            return False, "lorem_ipsum"
        if "javascript" in ll:
            continue
        if "{" in line:
            return False, "curly_bracket"
        if any(p in ll for p in policy_subs):
            continue
        num_sentences += len(_sent_tokenize(line, lang))
        kept += 1

    if num_sentences < 5:
        return False, "too_few_sentences"
    return True, None


_FILTER_FNS = {
    "gopher_quality": _gopher_quality,
    "c4_quality": _c4_quality,
}

_SUPPORTED_FILTERS = tuple(_FILTER_FNS)


def compute_quality(
    docs: Iterable[Document],
    filter_names: list[str] | None = None,
    default_language: str = "en",
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Apply Gopher Quality + C4 rules in read-only mode (nltk-tokenized).

    每文档按其 `language` 字段（缺省用 default_language）选择分词器。重复检测不在
    此处（归 stage 4 dedup）。
    """
    if filter_names is None:
        filter_names = list(_SUPPORTED_FILTERS)

    active = {name: _FILTER_FNS[name] for name in filter_names if name in _FILTER_FNS}
    per_doc: list[DocResult] = []
    fail_counters: dict[str, int] = {name: 0 for name in active}
    reason_counters: dict[str, _Counter] = {name: _Counter() for name in active}
    total = 0

    for doc in docs:
        text = str(doc.get("text") or "")
        lang = doc.get("language") or default_language
        scores: dict = {}
        any_failed = False
        for name, fn in active.items():
            passed, reason = fn(text, lang)
            scores[name] = passed
            if reason:
                scores[f"{name}_reason"] = reason
            if not passed:
                fail_counters[name] += 1
                reason_counters[name][reason] += 1
                any_failed = True

        result = DocResult(
            doc_id=str(doc["doc_id"]),
            scores=scores,
            flags={"any_filter_failed": any_failed},
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)
        total += 1

    summary = {
        "total_docs": total,
        "filter_fail_counts": fail_counters,
        "filter_fail_pcts": {
            name: round(cnt / total, 4) if total else 0.0
            for name, cnt in fail_counters.items()
        },
        "filter_fail_reasons": {
            name: dict(c.most_common()) for name, c in reason_counters.items()
        },
    }
    return per_doc, summary


# ── Extraction-quality audit (row 1, reframed) ────────────────────────────────
#
# pipeline_overview 行 1 原为「Trafilatura 正文抽取」，针对 raw HTML/PDF。
# 但本仓库审计的是「已清洗纯文本」（UFW-L3 content 无 HTML），抽取已是成品，
# 无对象可抽。故重定位为「抽取质量审计」：检测已清洗文本里残留的、本应被上游
# 抽取/清洗去掉的杂质，作为「抽取干净程度」的只读信号。不依赖 Trafilatura。

# HTML 标签白名单：只匹配真实 HTML 结构/语义标签，绕开技术文档里常见的代码占位
# （<code>, <source>, <var>, <kbd>, <samp>, <output> 等被故意排除——它们也是合法
# HTML 但在 rsync/eval/CLI 教程里常被当作 metavar，命中后 100% 误报）。
_HTML_TAG_WHITELIST = (
    # 结构 / 文档级
    "html", "head", "body", "title", "meta", "link", "script", "style", "noscript",
    "div", "span", "p", "br", "hr",
    # 标题 / 语义块
    "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "footer", "nav", "section", "article", "aside", "main",
    "figure", "figcaption", "blockquote", "pre", "address",
    # 表格
    "table", "tr", "td", "th", "thead", "tbody", "tfoot",
    "caption", "colgroup", "col",
    # 列表
    "ul", "ol", "li", "dl", "dt", "dd",
    # 表单
    "form", "input", "button", "label", "textarea", "select", "option",
    "optgroup", "fieldset", "legend",
    # 媒体
    "iframe", "img", "picture", "video", "audio", "canvas", "svg",
    # 链接 / 老式样式
    "a", "font", "center",
    # 详情 / 对话
    "details",
)
_HTML_TAG_RE = _re.compile(
    r"</?(?:" + "|".join(_HTML_TAG_WHITELIST) + r")(?:\s[^<>]*)?/?>",
    _re.IGNORECASE,
)
# HTML 实体：&amp; &nbsp; &#39; &#x27; 等
_HTML_ENTITY_RE = _re.compile(r"&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);")
# markdown 链接/图片残留：](url) 与 ![
_MD_ARTIFACT_RE = _re.compile(r"!?\[[^\]]*\]\([^)]*\)")
# 裸 URL
_URL_RE = _re.compile(r"https?://[^\s<>\"]+")
# mojibake / 编码损坏的替换符
_REPLACEMENT_CHAR = "�"

# Boilerplate / 导航 / 版权样板（en 小写匹配，zh 直接匹配）。启发式，宁缺毋滥。
_BOILERPLATE_EN = (
    "all rights reserved", "privacy policy", "terms of service",
    "terms of use", "we use cookies", "cookie policy", "accept cookies",
    "subscribe to our newsletter", "skip to (?:main )?content",
    "click here", "read more", "sign up", "log in", "sign in",
)
_BOILERPLATE_ZH = (
    "版权所有", "保留所有权利", "隐私政策", "服务条款", "未经许可", "未经授权",
    "转载请注明", "点击这里", "阅读更多", "下一页", "上一页", "关注我们",
    "扫描二维码", "免责声明",
)
_BOILERPLATE_EN_RE = _re.compile("|".join(_BOILERPLATE_EN), _re.IGNORECASE)
_BOILERPLATE_ZH_RE = _re.compile("|".join(_BOILERPLATE_ZH))


def compute_extraction_audit(
    docs: Iterable[Document],
    *,
    short_stub_chars: int = 50,
    html_tag_min_count: int = 3,
    html_entity_min_count: int = 1,
    boilerplate_edge_ratio: float = 0.05,
    boilerplate_edge_min_chars: int = 200,
    boilerplate_middle_weight: float = 0.2,
    boilerplate_weighted_threshold: float = 2.0,
    risk_score_threshold: float = 2.0,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """审计已清洗文本的抽取质量：检测 HTML/实体/markdown/boilerplate/mojibake 残留。

    判定改造（v2，2026-06）：
    - HTML：标签名走白名单（绕开 <code>/<source> 这类代码 metavar），且需累计
      `html_tag_min_count` 次才算残留；HTML 实体出现 `html_entity_min_count` 次即触发。
    - Boilerplate：按出现位置加权——文档头/尾区（`boilerplate_edge_ratio` * char 或
      至少 `boilerplate_edge_min_chars` 字符）权重 1.0，中段权重 `boilerplate_middle_weight`，
      加权和达 `boilerplate_weighted_threshold` 才算样板污染。
    - 汇总红灯 `low_extraction_quality` 不再是 OR-gate，而是加权风险分：
        score = 3·too_short + 2·has_mojibake + 2·has_html + min(boiler_weighted/2, 2)
      分数 >= `risk_score_threshold` 才算红灯。短残桩/mojibake 单独命中即可亮灯；
      HTML（已严苛阈值）单独命中也亮灯；boilerplate 需要权重 >= 4（约 4 处头/尾命中
      或更多中段命中）才能单独亮灯，避免「正文中 1 次版权声明」一刀切。

    short_stub_chars 用字符数（非词数）判断残桩，对 CJK 更公平（默认 50，只命中
    "404 Not Found" / "页面不存在" 这类近空抽取）。
    """
    per_doc: list[DocResult] = []
    total = 0
    n_html = n_entity = n_md = n_url = n_boiler = n_mojibake = n_short = n_lowq = 0
    risk_score_sum = 0.0
    boiler_phrase_counter: _Counter = _Counter()

    for doc in docs:
        text = str(doc.get("text") or "")
        n_chars = len(text)

        html_tags = len(_HTML_TAG_RE.findall(text))
        html_entities = len(_HTML_ENTITY_RE.findall(text))
        md_artifacts = len(_MD_ARTIFACT_RE.findall(text))
        urls = len(_URL_RE.findall(text))
        mojibake = text.count(_REPLACEMENT_CHAR)

        # Boilerplate 位置加权：头/尾权重 1.0，中段 0.2
        edge = max(boilerplate_edge_min_chars, int(n_chars * boilerplate_edge_ratio))
        head_end = edge
        tail_start = max(n_chars - edge, head_end)
        boiler_weighted = 0.0
        boiler_count = 0
        for rx in (_BOILERPLATE_EN_RE, _BOILERPLATE_ZH_RE):
            for m in rx.finditer(text):
                pos = m.start()
                if pos < head_end or pos >= tail_start:
                    boiler_weighted += 1.0
                else:
                    boiler_weighted += boilerplate_middle_weight
                boiler_phrase_counter[m.group(0).lower()] += 1
                boiler_count += 1

        has_html = (
            html_tags >= html_tag_min_count or html_entities >= html_entity_min_count
        )
        has_boiler = boiler_weighted >= boilerplate_weighted_threshold
        has_mojibake = mojibake > 0
        too_short = 0 < n_chars < short_stub_chars

        risk_score = 0.0
        if too_short:
            risk_score += 3.0
        if has_mojibake:
            risk_score += 2.0
        if has_html:
            risk_score += 2.0
        risk_score += min(boiler_weighted / 2.0, 2.0)
        low_quality = risk_score >= risk_score_threshold

        n_html += int(has_html)
        n_entity += int(html_entities > 0)
        n_md += int(md_artifacts > 0)
        n_url += int(urls > 0)
        n_boiler += int(has_boiler)
        n_mojibake += int(has_mojibake)
        n_short += int(too_short)
        n_lowq += int(low_quality)
        risk_score_sum += risk_score
        total += 1

        result = DocResult(
            doc_id=str(doc["doc_id"]),
            scores={
                "char_count": n_chars,
                "html_tag_count": html_tags,
                "html_entity_count": html_entities,
                "markdown_artifact_count": md_artifacts,
                "url_count": urls,
                "boilerplate_count": boiler_count,
                "boilerplate_weighted": round(boiler_weighted, 3),
                "mojibake_count": mojibake,
                "extraction_risk_score": round(risk_score, 3),
            },
            flags={
                "has_html_residue": has_html,
                "has_boilerplate": has_boiler,
                "has_mojibake": has_mojibake,
                "too_short_stub": too_short,
                "low_extraction_quality": low_quality,
            },
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    def _pct(n: int) -> float:
        return round(n / total, 4) if total else 0.0

    summary = {
        "total_docs": total,
        "html_residue_docs": n_html,
        "html_residue_pct": _pct(n_html),
        "html_entity_docs": n_entity,
        "markdown_artifact_docs": n_md,
        "url_docs": n_url,
        "boilerplate_docs": n_boiler,
        "boilerplate_pct": _pct(n_boiler),
        "mojibake_docs": n_mojibake,
        "mojibake_pct": _pct(n_mojibake),
        "short_stub_docs": n_short,
        "low_extraction_quality_docs": n_lowq,
        "low_extraction_quality_pct": _pct(n_lowq),
        "extraction_risk_score_mean": round(risk_score_sum / total, 4) if total else 0.0,
        "boilerplate_phrase_distribution": dict(boiler_phrase_counter.most_common(20)),
    }
    return per_doc, summary


# ── Row 3: GlotLID fine-grained language-script identification ────────────────

def compute_glotlid(
    docs: Iterable[Document],
    model_path: str,
    top_k: int = 3,
    confidence_threshold: float = 0.7,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Identify language-script using GlotLID v3 (2000+ categories).

    model_path: 本地 GlotLID 模型路径（如 /mnt/public/model/glotlid/model.bin）。
    输出标签格式：ISO3_Script，如 "eng_Latn"、"cmn_Hani"。
    """
    try:
        import fasttext
        fasttext.FastText.eprint = lambda x: None  # suppress warnings
    except ImportError as exc:
        raise RuntimeError("Run: pip install fasttext") from exc

    model = fasttext.load_model(model_path)

    per_doc: list[DocResult] = []
    lang_script_counter: Counter = Counter()
    low_conf_count = 0
    confidences: list[float] = []
    total = 0

    for doc in docs:
        text = str(doc.get("text") or "").replace("\n", " ").strip()
        if not text:
            label, conf, top3 = "und", 0.0, [["und", 0.0]]
        else:
            labels, scores = model.predict(text, k=top_k)
            label = labels[0].replace("__label__", "")
            conf = float(scores[0])
            top3 = [[lb.replace("__label__", ""), round(float(sc), 4)]
                    for lb, sc in zip(labels, scores)]

        lang_script_counter[label] += 1
        confidences.append(conf)
        is_low_conf = conf < confidence_threshold
        if is_low_conf:
            low_conf_count += 1

        result = DocResult(
            doc_id=str(doc["doc_id"]),
            scores={"lang_script": label, "confidence": round(conf, 4), "top3": top3},
            flags={"low_confidence": is_low_conf},
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)
        total += 1

    a = np.array(confidences) if confidences else np.array([0.0])
    summary = {
        "total_docs": total,
        "lang_script_distribution": dict(lang_script_counter.most_common(50)),
        "low_confidence_count": low_conf_count,
        "low_confidence_pct": round(low_conf_count / total, 4) if total else 0.0,
        "confidence_stats": {
            "mean": round(float(a.mean()), 4),
            "p50": round(float(np.percentile(a, 50)), 4),
            "p5": round(float(np.percentile(a, 5)), 4),
            "min": round(float(a.min()), 4),
        },
    }
    return per_doc, summary


# ── Coarse-vs-fine language cross-check ───────────────────────────────────────
#
# 审计动机：lid.176（粗，176 种）会把不认识的低资源语种/方言「自信地」吸收进
# 一个邻近大语言，且不报歧义——所以「langid 低置信再升级 glotlid」会系统性漏掉
# 这些。正确做法：两个模型都跑，把「分歧」当信号，尤其是 glotlid 识出一个
# lid.176 标签集里根本不存在的语种（必然被吸收）。
#
# 比对在「宏语言 alpha3」层面：langid 标签（ISO2 或 ISO3 混合）与 glotlid 的
# ISO3 都用 langcodes 归一到 alpha3 并折叠到宏语言（cmn→zho，使中文不误报；
# 而 arz 这类方言不折叠到 ara，正确暴露为分歧）。


def _norm_macro_alpha3(code: str, _cache: dict = {}) -> str | None:
    """语言码 → 宏语言 alpha3；失败返回 None。结果缓存（标签种类少）。"""
    if code in _cache:
        return _cache[code]
    try:
        import langcodes
        a3 = langcodes.Language.get(code).prefer_macrolanguage().to_alpha3()
    except Exception:
        a3 = None
    _cache[code] = a3
    return a3


def compute_lang_crosscheck(
    docs: Iterable[Document],
    langid_model_path: str,
    glotlid_model_path: str,
    confidence_threshold: float = 0.7,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """交叉核对粗(lid.176)与细(GlotLID)语种识别，暴露被粗模型吸收的低资源语种。

    每文档同时跑两个模型，在宏语言层面比对：
      - agree：两者宏语言一致。
      - disagreement：两者均高置信但宏语言不一致。
      - possibly_absorbed：disagreement 且 glotlid 的宏语言不在 lid.176 可表达集
        内 —— lid.176 结构性无法输出该语种，langid 的标签必然是吸收（强信号）。
    """
    try:
        import fasttext
        fasttext.FastText.eprint = lambda x: None
    except ImportError as exc:
        raise RuntimeError("Run: pip install fasttext") from exc

    lid = fasttext.load_model(langid_model_path)
    glot = fasttext.load_model(glotlid_model_path)

    # lid.176 可表达的宏语言 alpha3 集合（用于判定 glotlid 是否超出其能力）
    lid_macro_set = {
        m for lb in lid.get_labels()
        if (m := _norm_macro_alpha3(lb.replace("__label__", ""))) is not None
    }

    per_doc: list[DocResult] = []
    n_total = n_agree = n_disagree = n_absorbed = n_lowconf = 0
    disagree_pairs: Counter = Counter()
    absorbed_langs: Counter = Counter()

    for doc in docs:
        text = str(doc.get("text") or "").replace("\n", " ").strip()
        if not text:
            li_lang, li_conf, gl_label, gl_conf = "unknown", 0.0, "und", 0.0
        else:
            li_labels, li_scores = lid.predict(text, k=1)
            gl_labels, gl_scores = glot.predict(text, k=1)
            li_lang = li_labels[0].replace("__label__", "")
            li_conf = float(li_scores[0])
            gl_label = gl_labels[0].replace("__label__", "")
            gl_conf = float(gl_scores[0])

        gl_iso3 = gl_label.split("_")[0]
        li_macro = _norm_macro_alpha3(li_lang)
        gl_macro = _norm_macro_alpha3(gl_iso3)

        both_confident = (li_conf >= confidence_threshold
                          and gl_conf >= confidence_threshold)
        low_conf = not both_confident
        # 宏语言可比时才判 agree；任一无法归一则视为 unknown（不算 agree）
        agree = (li_macro is not None and gl_macro is not None
                 and li_macro == gl_macro)
        # disagreement：两者均高置信、但宏语言不一致（粗细模型在 lid.176 能表达的
        # 范围内冲突）。
        disagreement = both_confident and not agree and gl_macro is not None
        # possibly_absorbed：glotlid 高置信识出一个 lid.176 标签集里根本不存在的
        # 语种 → lid.176 结构性无法输出它，langid 的标签必然是「吸收」。
        # 注意：此处只看 glotlid 置信度，不要求 langid 高置信——langid 对它无法
        # 表达的语种，其「置信度」本身无意义（在错误的标签空间里自信）。
        absorbed = (gl_conf >= confidence_threshold
                    and gl_macro is not None
                    and gl_macro not in lid_macro_set)

        n_total += 1
        n_lowconf += int(low_conf)
        if agree:
            n_agree += 1
        if disagreement:
            n_disagree += 1
            disagree_pairs[f"{li_lang}->{gl_label}"] += 1
        if absorbed:
            n_absorbed += 1
            absorbed_langs[gl_label] += 1

        result = DocResult(
            doc_id=str(doc["doc_id"]),
            scores={
                "langid_lang": li_lang,
                "langid_conf": round(li_conf, 4),
                "glotlid_lang_script": gl_label,
                "glotlid_conf": round(gl_conf, 4),
            },
            flags={
                "agree": agree,
                "disagreement": disagreement,
                "possibly_absorbed": absorbed,
                "low_confidence": low_conf,
            },
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    def _pct(n: int) -> float:
        return round(n / n_total, 4) if n_total else 0.0

    summary = {
        "total_docs": n_total,
        "agree_count": n_agree,
        "agree_pct": _pct(n_agree),
        "disagreement_count": n_disagree,
        "disagreement_pct": _pct(n_disagree),
        "possibly_absorbed_count": n_absorbed,
        "possibly_absorbed_pct": _pct(n_absorbed),
        "low_confidence_count": n_lowconf,
        "top_disagreement_pairs": dict(disagree_pairs.most_common(20)),
        "absorbed_lang_distribution": dict(absorbed_langs.most_common(20)),
    }
    return per_doc, summary
