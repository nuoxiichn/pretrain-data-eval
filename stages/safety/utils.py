"""Stage 2 computation utilities.

  compute_pii      — row 1+2: Presidio PII detection (general & code modes)
  compute_secrets  — row 3:   Gitleaks secret scanning
  compute_toxicity — row 4:   Detoxify toxicity classification
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections import Counter
from typing import Callable, Iterable

import numpy as np

from src.reader import Document
from src.schema import DocResult


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
}
_PLACEHOLDER_EMAIL_LOCALS = {
    "firstname.lastname", "first.last", "email", "user", "username",
    "name", "your", "yourname", "youremail", "test", "example", "someone",
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


# ── Row 4: Toxicity classification ────────────────────────────────────────────

_TOXICITY_DIMS = [
    "toxicity", "severe_toxicity", "obscene",
    "identity_attack", "insult", "threat", "sexual_explicit",
]


def compute_toxicity(
    docs: Iterable[Document],
    model_path: str,
    high_risk_threshold: float = 0.5,
    batch_size: int = 16,
    device: str | None = None,
    max_length: int = 512,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Classify toxicity with the Detoxify unbiased RoBERTa model.
    """
    if not model_path:
        raise RuntimeError(
            "toxicity 需要本地 HF 模型目录：在 configs/stage2.yaml 设置 "
            "toxicity.model_path（如 /mnt/public/model/detoxify/unbiased-toxic-roberta）"
        )
    doc_list = list(docs)
    predict = _make_hf_predictor(model_path, device=device, max_length=max_length)

    per_doc: list[DocResult] = []
    dim_scores: dict[str, list[float]] = {d: [] for d in _TOXICITY_DIMS}
    high_risk_count = 0

    for i in range(0, len(doc_list), batch_size):
        batch = doc_list[i : i + batch_size]
        texts = [str(doc.get("text") or "") for doc in batch]
        preds = predict(texts)

        for j, doc in enumerate(batch):
            scores = {
                dim: round(float(preds[dim][j]), 6)
                for dim in _TOXICITY_DIMS
                if dim in preds
            }
            is_high_risk = scores.get("toxicity", 0.0) >= high_risk_threshold
            if is_high_risk:
                high_risk_count += 1
            for dim, val in scores.items():
                dim_scores[dim].append(val)

            result = DocResult(
                doc_id=str(doc["doc_id"]),
                scores=scores,
                flags={"high_risk": is_high_risk},
            )
            if on_doc is not None:
                on_doc(result)
            else:
                per_doc.append(result)

    total = len(doc_list)
    summary = {
        "total_docs_scanned": total,
        "high_risk_docs": high_risk_count,
        "high_risk_pct": round(high_risk_count / total, 4) if total else 0.0,
        "high_risk_threshold": high_risk_threshold,
        "model_path": model_path,
        "dim_stats": {
            dim: _dim_summary(vals)
            for dim, vals in dim_scores.items()
            if vals
        },
    }
    return per_doc, summary


def _make_hf_predictor(model_path: str, *, device: str | None = None, max_length: int = 512):
    """transformers 后端：加载 HF 目录，返回与 detoxify.predict 同形状的闭包。

    detoxify unbiased = RobertaForSequenceClassification + 多标签 sigmoid。
    config.id2label 给出标签名；我们按 _TOXICITY_DIMS 取需要的 7 维。
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()
    id2label = model.config.id2label

    def predict(texts: list[str]) -> dict[str, list[float]]:
        enc = tok(
            list(texts), padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(**enc).logits)
        out: dict[str, list[float]] = {}
        for idx, label in id2label.items():
            out[label] = probs[:, int(idx)].tolist()
        return out

    return predict


def _dim_summary(values: list[float]) -> dict:
    a = np.array(values)
    return {
        "mean": round(float(a.mean()), 6),
        "p50": round(float(np.percentile(a, 50)), 6),
        "p95": round(float(np.percentile(a, 95)), 6),
        "p99": round(float(np.percentile(a, 99)), 6),
        "max": round(float(a.max()), 6),
    }
