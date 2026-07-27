"""Stage 2 computation utilities.

  compute_pii         — row 1+2: Presidio PII detection (general & code modes)
  compute_secrets     — row 3:   Gitleaks secret scanning
  compute_toxicity    — row 4:   chunk + XLM-R recall + Qwen LLM-judge
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from typing import Callable, Iterable

import numpy as np

from pretrain_data_eval.reader import Document
from pretrain_data_eval.schema import DocResult


# ── PII entity lists ──────────────────────────────────────────────────────────

# 默认通用实体集。已移除三个在网页文本上几乎全是误报的实体（依据 UFW-L3 抽样）：
#   US_DRIVER_LICENSE — 命中多为 A2/A1/K4 等 2 字符代号（驾照等级/科学相位）
#   MEDICAL_LICENSE   — 命中多为 US5528166（专利号）/ rs1205081（DOI/dbSNP）
#   UK_NHS            — 命中多为美国电话号格式
# 如确需检测这些，可通过 --entities 或 yaml 显式启用。
_GENERAL_ENTITIES = [
    "EMAIL_ADDRESS", "PHONE_NUMBER",
    "CREDIT_CARD", "IBAN_CODE", "CRYPTO",
    "US_SSN", "US_BANK_NUMBER", "US_PASSPORT",
    "IN_PAN", "IN_AADHAAR",
    "IP_ADDRESS",
]

_CODE_ENTITIES = [
    "EMAIL_ADDRESS", "IP_ADDRESS", "URL", "CREDIT_CARD", "CRYPTO",
]


# ── Placeholder / example-value filtering ─────────────────────────────────────

# 占位符邮箱域名与本地部分（example.com 等示例邮箱不是真实隐私）
_PLACEHOLDER_EMAIL_DOMAINS = {
    "example.com", "example.org", "example.net", "example.edu",
    "test.com", "email.com", "domain.com", "yourdomain.com", "sample.com",
    "mydomain.com", "company.com", "acme.com", "foo.com", "foo.bar",
    "localhost", "localhost.localdomain",
}
# RFC 2606 §2 保留的顶级域（任何 *.test / *.example / *.invalid / *.localhost 都是占位）
_PLACEHOLDER_EMAIL_TLDS = (".test", ".example", ".invalid", ".localhost")
_PLACEHOLDER_EMAIL_LOCALS = {
    "firstname.lastname", "first.last", "lastname.firstname",
    "email", "user", "username", "name", "your", "yourname",
    "youremail", "your-email", "your_email", "test", "example", "someone",
    "john.doe", "jane.doe", "johndoe", "janedoe", "john", "jane",
    "noreply", "no-reply", "donotreply", "do-not-reply",
    "placeholder", "me", "you", "info", "admin", "contact",
    "anyone", "somebody", "nobody", "abc", "xyz", "foo", "bar",
}
# RFC5737 文档示例 IP 段（非真实主机）
_DOC_IP_NETWORKS = ("192.0.2.", "198.51.100.", "203.0.113.")


def _is_placeholder(entity_type: str, value: str) -> bool:
    """判断一条命中是否为示例/占位符/保留值（非真实 PII）。

    仅用于过滤判定，调用方不会把 value 写入任何输出（维持不落明文）。
    """
    v = value.strip()
    if entity_type == "EMAIL_ADDRESS":
        if "@" not in v:
            return False
        local, _, domain = v.rpartition("@")
        domain = domain.lower()
        if domain in _PLACEHOLDER_EMAIL_DOMAINS:
            return True
        if any(domain.endswith(tld) for tld in _PLACEHOLDER_EMAIL_TLDS):
            return True
        if local.lower() in _PLACEHOLDER_EMAIL_LOCALS:
            return True
        return False
    if entity_type == "IP_ADDRESS":
        if any(v.startswith(net) for net in _DOC_IP_NETWORKS):
            return True
        try:
            import ipaddress
            ip = ipaddress.ip_address(v)
        except ValueError:
            # 无法解析为合法 IP（如 :: 之外的残片）→ 当噪声丢弃
            return True
        # 私有 / 回环 / 保留 / 链路本地 / 未指定 均非个人隐私
        return (ip.is_private or ip.is_loopback or ip.is_reserved
                or ip.is_link_local or ip.is_unspecified or ip.is_multicast)
    return False


# ── Row 1+2: PII detection ────────────────────────────────────────────────────

def compute_pii(
    docs: Iterable[Document],
    language: str = "en",
    entities: list[str] | None = None,
    score_threshold: float = 0.5,
    mode: str = "general",
    spacy_model: str = "en_core_web_lg",
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Detect PII using Microsoft Presidio.

    mode="general" : broad entity set (names, phones, IDs, …)
    mode="code"    : code-relevant entities (email, IP, URL, crypto, …)
    mode="both"    : union of both entity sets
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError as exc:
        raise RuntimeError("Run: pip install presidio-analyzer") from exc

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": language, "model_name": spacy_model}],
    })
    analyzer = AnalyzerEngine(
        nlp_engine=provider.create_engine(),
        supported_languages=[language],
    )

    if entities:
        entity_list = entities
    elif mode == "code":
        entity_list = _CODE_ENTITIES
    elif mode == "both":
        entity_list = list(dict.fromkeys(_GENERAL_ENTITIES + _CODE_ENTITIES))
    else:
        entity_list = _GENERAL_ENTITIES

    per_doc: list[DocResult] = []
    entity_counter: Counter = Counter()
    hit_docs = 0
    total = 0
    filtered_count = 0

    for doc in docs:
        text = str(doc.get("text") or "")
        results = analyzer.analyze(
            text=text,
            language=language,
            entities=entity_list,
            score_threshold=score_threshold,
        )
        hits = []
        for r in results:
            # 用原文值判定是否为示例/占位符/保留值；value 仅用于判定，不写入输出。
            if _is_placeholder(r.entity_type, text[r.start : r.end]):
                filtered_count += 1
                continue
            hits.append({
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": round(r.score, 4),
            })
        for h in hits:
            entity_counter[h["entity_type"]] += 1
        has_pii = bool(hits)
        if has_pii:
            hit_docs += 1

        result = DocResult(
            doc_id=str(doc["doc_id"]),
            scores={"pii_count": len(hits), "pii_hits": hits},
            flags={"has_pii": has_pii},
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)
        total += 1

    summary = {
        "total_docs_scanned": total,
        "docs_with_pii": hit_docs,
        "hit_pct": round(hit_docs / total, 4) if total else 0.0,
        "entity_type_distribution": dict(entity_counter.most_common()),
        "filtered_placeholder_count": filtered_count,
        "mode": mode,
        "language": language,
        "score_threshold": score_threshold,
    }
    return per_doc, summary


# ── Row 3: Secret scanning ────────────────────────────────────────────────────

def compute_secrets(
    docs: Iterable[Document],
    gitleaks_bin: str = "gitleaks",
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Scan for secrets using Gitleaks in --no-git filesystem mode.

    Writes all documents to a temp directory, runs gitleaks once, then
    maps findings back to doc_ids via filename.
    """
    doc_list = list(docs)
    if not doc_list:
        empty = {"total_docs_scanned": 0, "docs_with_secrets": 0, "hit_pct": 0.0, "rule_distribution": {}}
        return [], empty

    try:
        subprocess.run([gitleaks_bin, "version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            f"Gitleaks not found at '{gitleaks_bin}'. "
            "Install from https://github.com/gitleaks/gitleaks/releases"
        )

    findings_by_doc: dict[str, list] = {str(doc["doc_id"]): [] for doc in doc_list}

    with tempfile.TemporaryDirectory() as tmpdir:
        scan_dir = os.path.join(tmpdir, "docs")
        report_path = os.path.join(tmpdir, "report.json")
        os.makedirs(scan_dir)

        for doc in doc_list:
            fname = f"{doc['doc_id']}.txt"
            with open(os.path.join(scan_dir, fname), "w", encoding="utf-8") as f:
                f.write(str(doc.get("text") or ""))

        proc = subprocess.run(
            [gitleaks_bin, "detect", "--source", scan_dir,
             "--no-git", "-f", "json", "-r", report_path],
            capture_output=True,
        )
        # exit 0 = clean, exit 1 = findings found, anything else = error
        if proc.returncode not in (0, 1):
            raise RuntimeError(
                f"Gitleaks error (exit {proc.returncode}): {proc.stderr.decode()}"
            )

        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as f:
                raw = json.load(f)
            for finding in (raw if isinstance(raw, list) else []):
                doc_id = os.path.basename(finding.get("File", "")).removesuffix(".txt")
                if doc_id in findings_by_doc:
                    findings_by_doc[doc_id].append({
                        "rule_id": finding.get("RuleID"),
                        "entropy": round(finding.get("Entropy", 0.0), 4),
                        "start_line": finding.get("StartLine"),
                        "end_line": finding.get("EndLine"),
                    })

    per_doc: list[DocResult] = []
    rule_counter: Counter = Counter()
    hit_docs = 0

    for doc in doc_list:
        doc_id = str(doc["doc_id"])
        secrets = findings_by_doc.get(doc_id, [])
        has_secrets = bool(secrets)
        if has_secrets:
            hit_docs += 1
        for s in secrets:
            if s["rule_id"]:
                rule_counter[s["rule_id"]] += 1

        result = DocResult(
            doc_id=doc_id,
            scores={"secret_count": len(secrets), "secrets": secrets},
            flags={"has_secrets": has_secrets},
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    total = len(doc_list)
    summary = {
        "total_docs_scanned": total,
        "docs_with_secrets": hit_docs,
        "hit_pct": round(hit_docs / total, 4) if total else 0.0,
        "rule_distribution": dict(rule_counter.most_common()),
    }
    return per_doc, summary


# ── Toxicity: shared HF predictor helpers ────────────────────────────────────

_TOXIC_LABEL_KEYWORDS = {"toxic", "offensive", "risk", "harmful", "hate"}


def _make_hf_predictor(model_path: str, *, device: str | None = None, max_length: int = 512):
    """加载 HF 文本分类模型，自动识别二分类（softmax）与多标签（sigmoid）。

    二分类（如 xlmr-large-toxicity-classifier、roberta-base-cold）→ softmax，
    输出 {"toxicity": [p_toxic, ...]}。
    多标签（如 detoxify unbiased）→ sigmoid，输出所有 id2label 维度。
    返回 (predict_fn, model_mode)，mode 为 "binary" 或 "multilabel"。
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    id2label = model.config.id2label
    is_binary = model.config.num_labels == 2

    toxic_idx = 1
    if is_binary:
        for idx_str, label in id2label.items():
            if any(kw in label.lower() for kw in _TOXIC_LABEL_KEYWORDS):
                toxic_idx = int(idx_str)
                break

    def predict(texts: list[str]) -> dict[str, list[float]]:
        enc = tok(
            list(texts), padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            if is_binary:
                probs = torch.softmax(logits, dim=-1)
                return {"toxicity": probs[:, toxic_idx].tolist()}
            probs = torch.sigmoid(logits)
        out: dict[str, list[float]] = {}
        for idx, label in id2label.items():
            out[label] = probs[:, int(idx)].tolist()
        return out

    return predict, "binary" if is_binary else "multilabel"


def _dim_summary(values: list[float]) -> dict:
    a = np.array(values)
    return {
        "mean": round(float(a.mean()), 6),
        "p50": round(float(np.percentile(a, 50)), 6),
        "p95": round(float(np.percentile(a, 95)), 6),
        "p99": round(float(np.percentile(a, 99)), 6),
        "max": round(float(a.max()), 6),
    }


# ── Row 4: chunk + XLM-R recall + Qwen LLM-judge ─────────────────────────────

_JUDGE_VERDICTS = ("benign", "discuss", "promote")

# 决策步骤 + 4 个 few-shot 锚定常见混淆边界。设计目标是降低
# discuss → promote 的假阳：批评、历史叙述、受害者陈述都不应触发 promote。
_JUDGE_SYSTEM_PROMPT = (
    "你是一位中英双语的内容安全审核员。下面给你一段网页文本，请判定它的「作者立场」属于哪一类：\n"
    "\n"
    "【三分类定义】\n"
    "- benign: 文本本身不涉及任何敏感/有害议题。\n"
    "- discuss: 文本提到、引用、批评、谴责、客观叙述、记录、转述他人观点，涉及敏感议题但作者立场是\n"
    "  中立 / 批判 / 学术 / 记录 / 受害者陈述 / 文学叙事。即使语气强烈、含负面评价或粗口，只要不鼓动具体行动\n"
    "  且不直接攻击具体人群人格，就归 discuss。\n"
    "- promote: 作者本人立场是煽动、教唆、号召、赞扬、传播仇恨/暴力/色情/违法行为，或以第一人称对\n"
    "  具体个人/群体发起人格侮辱与攻击；或美化施暴者、把暴力或违法行为正当化。\n"
    "\n"
    "【判定步骤，按顺序逐条问自己】\n"
    "1. 文本是否涉及敏感议题？若否 → benign。\n"
    "2. 作者是否在「叙述、引用、批评、记录、讨论」他人的言行 / 历史事件 / 文学情节 / 自己作为受害者的经历？\n"
    "   若是 → discuss（不论语气多激烈、负面评价多重）。\n"
    "3. 作者是否亲自号召读者做某种有害行为 / 美化施暴者 / 用第一人称对特定对象发起人格攻击或仇恨言论？\n"
    "   若是 → promote。否则 → discuss。\n"
    "4. 若仍不确定，**默认归 discuss**（高假阳的代价大于漏检）。\n"
    "\n"
    "【关键边界：以下情形都不是 promote】\n"
    "- 对某个人/事/现象的负面评价、强烈批评、道德质疑（无煽动行动）。\n"
    "- 历史事件、新闻报道、学术分析中对暴力/歧视/犯罪的描述。\n"
    "- 文学小说中虚构角色的暴力/恐吓/犯罪情节。\n"
    "- 受害者或旁观者陈述自己遭受的伤害、性侵、霸凌等经历。\n"
    "- 转述、引用、翻译他人的极端言论（作者本人未表态认同）。\n"
    "- 政治立场表达、政策呼吁、抗议口号（除非直接号召暴力或仇恨行动）。\n"
    "\n"
    "【few-shot 示例】\n"
    "示例 1（中文，历史叙述）\n"
    '<text>诸葛亮砍伐椰子树，是为了矫正这种轻狂之态，出于教化之目的。云南人在当时被视为道德上有缺陷。</text>\n'
    '{"verdict": "discuss", "confidence": 0.9, "reason": "客观叙述历史观点，无煽动"}\n'
    "\n"
    "示例 2（中文，强烈负面评价）\n"
    '<text>戴拨拉这个人道德败坏，言行不一，根本不配为人师表，应当受到舆论的强烈谴责。</text>\n'
    '{"verdict": "discuss", "confidence": 0.85, "reason": "负面评价批评，未煽动行动"}\n'
    "\n"
    "示例 3（英文，受害者陈述）\n"
    "<text>I was sexually assaulted when I was sixteen. For years I blamed myself. Writing this down is "
    "the only way I can move forward.</text>\n"
    '{"verdict": "discuss", "confidence": 0.95, "reason": "受害者亲历叙述"}\n'
    "\n"
    "示例 4（中文，真煽动 — 锚定 promote 上界）\n"
    '<text>那些异教徒就是垃圾，兄弟们抄家伙，今晚就去把他们的店砸了，让他们知道厉害。</text>\n'
    '{"verdict": "promote", "confidence": 0.95, "reason": "第一人称号召暴力行动"}\n'
    "\n"
    "【输出格式】\n"
    '只输出一行严格 JSON，不要任何解释：{"verdict": "benign|discuss|promote", "confidence": 0.0-1.0, "reason": "≤40字"}'
)


def _chunk_text_by_tokenizer(
    text: str,
    tokenizer,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int, str]]:
    """Split ``text`` into chunks of ≤ ``chunk_size`` subwords with ``overlap``.

    Returns ``[(char_start, char_end, chunk_text), ...]``. Always yields at least
    one chunk (the full text if short). Uses ``return_offsets_mapping`` so the
    chunk text can be sliced back from the original string — preserving original
    formatting for the LLM judge.
    """
    text = text or ""
    if not text:
        return [(0, 0, "")]
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]
    n_tokens = len(offsets)
    if n_tokens == 0:
        return [(0, len(text), text)]
    if n_tokens <= chunk_size:
        return [(0, len(text), text)]

    step = max(1, chunk_size - max(0, overlap))
    chunks: list[tuple[int, int, str]] = []
    i = 0
    while i < n_tokens:
        j = min(i + chunk_size, n_tokens)
        s_char = offsets[i][0]
        e_char = offsets[j - 1][1]
        if e_char <= s_char:
            i += step
            continue
        chunks.append((int(s_char), int(e_char), text[s_char:e_char]))
        if j >= n_tokens:
            break
        i += step
    return chunks


def _parse_judge_output(raw: str) -> dict:
    """Extract the JSON verdict from a Qwen completion. Falls back to ``benign``
    with low confidence on parse failure so a malformed reply never crashes a run.
    """
    if not raw:
        return {"verdict": "benign", "confidence": 0.0, "reason": "empty"}
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    payload = m.group(0) if m else raw
    try:
        obj = json.loads(payload)
    except Exception:
        return {"verdict": "benign", "confidence": 0.0, "reason": "parse_error"}
    verdict = str(obj.get("verdict", "benign")).strip().lower()
    if verdict not in _JUDGE_VERDICTS:
        verdict = "benign"
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    reason = str(obj.get("reason", ""))[:120]
    return {"verdict": verdict, "confidence": round(conf, 4), "reason": reason}


def _make_qwen_judge(
    model_path: str,
    *,
    max_tokens: int,
    temperature: float,
    gpu_memory_utilization: float,
):
    """Load Qwen-Instruct via vLLM. Returns ``judge(chunks) -> list[dict]``.

    Each judgment dict has keys ``verdict / confidence / reason``. Caller is
    responsible for not invoking this when there are zero chunks to judge.
    """
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
        stop=None,
    )

    def _build_prompt(chunk: str) -> str:
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"<text>\n{chunk}\n</text>"},
        ]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def judge(chunks: list[str]) -> list[dict]:
        if not chunks:
            return []
        prompts = [_build_prompt(c) for c in chunks]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        out: list[dict] = []
        for o in outputs:
            text = o.outputs[0].text if o.outputs else ""
            out.append(_parse_judge_output(text))
        return out

    return judge


def compute_toxicity(
    docs: Iterable[Document],
    recall_model_path: str,
    judge_model_path: str,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    recall_threshold: float = 0.5,
    recall_batch_size: int = 16,
    judge_max_tokens: int = 256,
    judge_temperature: float = 0.0,
    judge_gpu_mem_util: float = 0.85,
    judge_max_chunks_per_doc: int = 8,
    device: str | None = None,
    on_doc: Callable[[DocResult], None] | None = None,
    judge_factory: Callable[..., Callable[[list[str]], list[dict]]] | None = None,
) -> tuple[list[DocResult], dict]:
    """Two-stage toxicity classification.

    1. Chunk each doc with the XLM-R tokenizer (≤ ``chunk_size`` subwords,
       ``chunk_overlap`` token overlap).
    2. Run XLM-R binary classifier on every chunk — recall stage.
    3. Send chunks with score ≥ ``recall_threshold`` (top ``judge_max_chunks_per_doc``
       per doc by score) to Qwen-Instruct LLM-judge for the
       benign / discuss / promote verdict.
    4. ``high_risk`` (``flags.llm_promote``) requires at least one promote verdict.

    ``judge_factory`` is injected for tests; defaults to ``_make_qwen_judge``.
    """
    from transformers import AutoTokenizer

    doc_list = list(docs)
    recall_predict, recall_mode = _make_hf_predictor(
        recall_model_path, device=device, max_length=chunk_size
    )
    if recall_mode != "binary":
        raise RuntimeError(
            f"toxicity requires a binary recall model, got {recall_mode}"
        )
    recall_tok = AutoTokenizer.from_pretrained(recall_model_path)

    # ── Stage 1: chunk + recall ─────────────────────────────────────────────
    # Collect (doc_idx, chunk_idx, char_start, char_end, chunk_text, xlmr_score)
    all_chunks: list[tuple[int, int, int, int, str]] = []
    for di, doc in enumerate(doc_list):
        text = str(doc.get("text") or "")
        for ci, (s, e, ct) in enumerate(
            _chunk_text_by_tokenizer(text, recall_tok, chunk_size, chunk_overlap)
        ):
            all_chunks.append((di, ci, s, e, ct))

    chunk_scores: list[float] = [0.0] * len(all_chunks)
    for i in range(0, len(all_chunks), recall_batch_size):
        batch = all_chunks[i : i + recall_batch_size]
        preds = recall_predict([c[4] for c in batch])
        for k, s in enumerate(preds["toxicity"]):
            chunk_scores[i + k] = float(s)

    # ── Stage 2: pick recall hits, route to LLM-judge ───────────────────────
    # Per doc → list of chunk indices with score ≥ threshold, capped + sorted.
    doc_to_recalled: dict[int, list[int]] = {}
    for idx, (di, *_rest) in enumerate(all_chunks):
        if chunk_scores[idx] >= recall_threshold:
            doc_to_recalled.setdefault(di, []).append(idx)
    for di, lst in doc_to_recalled.items():
        lst.sort(key=lambda i: chunk_scores[i], reverse=True)
        if len(lst) > judge_max_chunks_per_doc:
            doc_to_recalled[di] = lst[:judge_max_chunks_per_doc]

    flat_recalled: list[int] = [i for lst in doc_to_recalled.values() for i in lst]
    judgments: dict[int, dict] = {}
    if flat_recalled:
        factory = judge_factory or _make_qwen_judge
        judge = factory(
            judge_model_path,
            max_tokens=judge_max_tokens,
            temperature=judge_temperature,
            gpu_memory_utilization=judge_gpu_mem_util,
        )
        verdicts = judge([all_chunks[i][4] for i in flat_recalled])
        for i, v in zip(flat_recalled, verdicts):
            judgments[i] = v

    # ── Stage 3: aggregate per doc ──────────────────────────────────────────
    per_doc: list[DocResult] = []
    # Group chunk indices by doc.
    chunks_by_doc: dict[int, list[int]] = {}
    for idx, (di, *_rest) in enumerate(all_chunks):
        chunks_by_doc.setdefault(di, []).append(idx)

    total_chunks = len(all_chunks)
    total_recalled_chunks = len(flat_recalled)
    verdict_counter: Counter = Counter()
    high_risk_count = 0
    recalled_doc_count = 0
    xlmr_max_dist: list[float] = []
    xlmr_mean_dist: list[float] = []

    for di, doc in enumerate(doc_list):
        idxs = chunks_by_doc.get(di, [])
        scores = [chunk_scores[i] for i in idxs]
        xlmr_max = max(scores) if scores else 0.0
        xlmr_mean = sum(scores) / len(scores) if scores else 0.0
        xlmr_max_dist.append(xlmr_max)
        xlmr_mean_dist.append(xlmr_mean)

        recalled = doc_to_recalled.get(di, [])
        chunk_judgments = []
        n_promote = 0
        n_discuss = 0
        for i in recalled:
            v = judgments.get(i)
            if v is None:
                continue
            verdict_counter[v["verdict"]] += 1
            if v["verdict"] == "promote":
                n_promote += 1
            elif v["verdict"] == "discuss":
                n_discuss += 1
            ci = all_chunks[i][1]
            ct = all_chunks[i][4]
            chunk_judgments.append({
                "chunk_idx": ci,
                "xlmr_score": round(chunk_scores[i], 6),
                "verdict": v["verdict"],
                "confidence": v["confidence"],
                "reason": v["reason"],
                "text_preview": ct[:200],
            })

        is_recalled = len(recalled) > 0
        is_promote = n_promote > 0
        if is_recalled:
            recalled_doc_count += 1
        if is_promote:
            high_risk_count += 1

        result = DocResult(
            doc_id=str(doc["doc_id"]),
            scores={
                "xlmr_max": round(xlmr_max, 6),
                "xlmr_mean": round(xlmr_mean, 6),
                "n_chunks": len(idxs),
                "n_chunks_recalled": len(recalled),
                "n_chunks_promote": n_promote,
                "n_chunks_discuss": n_discuss,
                "judgments": chunk_judgments,
            },
            flags={
                "recalled": is_recalled,
                "llm_promote": is_promote,
                "high_risk": is_promote,
            },
        )
        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    total = len(doc_list)
    summary = {
        "total_docs_scanned": total,
        "total_chunks": total_chunks,
        "recalled_chunks": total_recalled_chunks,
        "recalled_docs": recalled_doc_count,
        "high_risk_docs": high_risk_count,
        "high_risk_pct": round(high_risk_count / total, 4) if total else 0.0,
        "recall_threshold": recall_threshold,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "judge_max_chunks_per_doc": judge_max_chunks_per_doc,
        "recall_model_path": recall_model_path,
        "judge_model_path": judge_model_path,
        "verdict_distribution": dict(verdict_counter),
        "xlmr_max_stats": _dim_summary(xlmr_max_dist) if xlmr_max_dist else {},
        "xlmr_mean_stats": _dim_summary(xlmr_mean_dist) if xlmr_mean_dist else {},
    }
    return per_doc, summary
