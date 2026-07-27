"""Stage 7 computation utilities.

  compute_binoculars — AI-generated text detection via observer/performer
                       cross-perplexity ratio (Binoculars, ICML 2024).
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from pretrain_data_eval.reader import Document
from pretrain_data_eval.schema import DocResult


# ── Binoculars model loader ──────────────────────────────────────────────────

def _load_binoculars_models(
    observer_path: str,
    performer_path: str,
    device: str | None = None,
    dtype: str = "float16",
):
    """Load observer (base) and performer (instruct) models + shared tokenizer.

    Returns (tokenizer, observer_model, performer_model, device_str).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

    tokenizer = AutoTokenizer.from_pretrained(observer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    observer = AutoModelForCausalLM.from_pretrained(
        observer_path, dtype=torch_dtype,
    ).to(device).eval()
    performer = AutoModelForCausalLM.from_pretrained(
        performer_path, dtype=torch_dtype,
    ).to(device).eval()

    return tokenizer, observer, performer, device


# ── Score computation ────────────────────────────────────────────────────────

def _compute_scores_batch(
    tokenizer,
    observer,
    performer,
    texts: list[str],
    max_length: int,
    device: str,
) -> list[float]:
    """Compute Binoculars score for a batch of texts.

    Score = mean_NLL_observer / cross_entropy(performer_softmax, observer_log_softmax)

    Lower score → more likely AI-generated.
    """
    import torch
    import torch.nn.functional as F

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    with torch.no_grad():
        observer_logits = observer(**enc).logits
        performer_logits = performer(**enc).logits

    # Shift for next-token prediction: logits[:, :-1] predict tokens[:, 1:]
    shift_obs = observer_logits[:, :-1, :].contiguous()
    shift_perf = performer_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].float()

    batch_size, seq_len = shift_labels.shape
    vocab_size = shift_obs.size(-1)

    # Numerator: average NLL of observer predicting actual next tokens
    ce_loss = F.cross_entropy(
        shift_obs.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
    ).view(batch_size, seq_len)
    nll = (ce_loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)

    # Denominator: cross-entropy H(p_performer, p_observer)
    # = -sum softmax(performer) * log_softmax(observer)
    log_probs_obs = F.log_softmax(shift_obs, dim=-1)
    probs_perf = F.softmax(shift_perf, dim=-1)
    x_ent = -(probs_perf * log_probs_obs).sum(dim=-1)  # [batch, seq_len]
    x_ent_avg = (x_ent * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)

    scores = nll / x_ent_avg.clamp(min=1e-8)

    return scores.cpu().tolist()


# ── Public API ───────────────────────────────────────────────────────────────

def compute_binoculars(
    docs: Iterable[Document],
    observer_path: str,
    performer_path: str,
    threshold: float = 0.8536,
    batch_size: int = 8,
    max_length: int = 512,
    device: str | None = None,
    dtype: str = "float16",
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Run Binoculars AI-generated text detection.

    Returns (per_doc_results, summary_dict).
    When on_doc is provided, per_doc results are streamed and the returned list is empty.
    """
    doc_list = list(docs)
    if not doc_list:
        return [], {"total_docs_scanned": 0, "ai_generated_docs": 0, "ai_generated_pct": 0.0}

    tokenizer, observer, performer, dev = _load_binoculars_models(
        observer_path, performer_path, device=device, dtype=dtype,
    )

    per_doc: list[DocResult] = []
    all_scores: list[float] = []
    ai_count = 0

    for i in range(0, len(doc_list), batch_size):
        batch = doc_list[i : i + batch_size]
        texts = [str(doc.get("text") or "") for doc in batch]

        scores = _compute_scores_batch(
            tokenizer, observer, performer, texts, max_length, dev,
        )

        for j, doc in enumerate(batch):
            score = round(scores[j], 6)
            is_ai = score < threshold
            if is_ai:
                ai_count += 1
            all_scores.append(score)

            result = DocResult(
                doc_id=str(doc["doc_id"]),
                scores={"binoculars": score},
                flags={"is_ai_generated": is_ai},
            )
            if on_doc is not None:
                on_doc(result)
            else:
                per_doc.append(result)

    total = len(doc_list)
    arr = np.array(all_scores)
    summary = {
        "total_docs_scanned": total,
        "ai_generated_docs": ai_count,
        "ai_generated_pct": round(ai_count / total, 4) if total else 0.0,
        "threshold": threshold,
        "observer_model": observer_path,
        "performer_model": performer_path,
        "score_stats": {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "min": round(float(arr.min()), 6),
            "p5": round(float(np.percentile(arr, 5)), 6),
            "p25": round(float(np.percentile(arr, 25)), 6),
            "p50": round(float(np.percentile(arr, 50)), 6),
            "p75": round(float(np.percentile(arr, 75)), 6),
            "p95": round(float(np.percentile(arr, 95)), 6),
            "max": round(float(arr.max()), 6),
        },
    }
    return per_doc, summary
