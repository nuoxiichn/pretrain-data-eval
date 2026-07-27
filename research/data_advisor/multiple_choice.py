"""Base-model multiple-choice likelihood evaluation for benchmark proxy calibration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL = "rc-zero-shot-v1"
PROMPT_TEMPLATE = "Question: {question}\nAnswer:"


@dataclass(frozen=True)
class MultipleChoiceExample:
    example_id: str
    group: str
    question: str
    choices: tuple[str, ...]
    gold: int


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


def _stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _shuffle_choices(
    choices: Sequence[str], gold: int, *, example_id: str, seed: int = 111
) -> tuple[tuple[str, ...], int]:
    order = list(range(len(choices)))
    random.Random(_stable_int(f"{seed}:{example_id}")).shuffle(order)
    return tuple(str(choices[index]).strip() for index in order), order.index(gold)


def deterministic_sample(
    examples: Sequence[MultipleChoiceExample], limit: int | None, seed: int
) -> list[MultipleChoiceExample]:
    """Take a stable sample without depending on source row order."""
    if limit is None or limit >= len(examples):
        return list(examples)
    if limit <= 0:
        raise ValueError("sample limit must be positive")
    ranked = sorted(
        examples,
        key=lambda item: (_stable_int(f"{seed}:{item.example_id}"), item.example_id),
    )
    return sorted(ranked[:limit], key=lambda item: item.example_id)


def _answer_index(answer: str, labels: str = "ABCDEFGHIJKLMNO") -> int:
    value = str(answer).strip().upper()
    if value not in labels:
        raise ValueError(f"unknown answer label {answer!r}")
    return labels.index(value)


def _load_dataset(spec: Mapping[str, Any], cache_dir: str | Path):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional evaluator environment
        raise RuntimeError("multiple-choice evaluation requires Hugging Face datasets") from exc
    return load_dataset(
        str(spec["data_id"]),
        spec.get("data_subset"),
        revision=str(spec["revision"]),
        cache_dir=str(cache_dir),
        token=True,
    )


def load_multiple_choice_examples(
    benchmark: str,
    spec: Mapping[str, Any],
    *,
    cache_dir: str | Path,
    limit: int | None = None,
    sample_seed: int = 6198,
) -> tuple[list[MultipleChoiceExample], dict[str, Any]]:
    """Load one preregistered dataset and normalize it to RC examples."""
    dataset = _load_dataset(spec, cache_dir)
    examples: list[MultipleChoiceExample] = []

    if benchmark == "mmlu-pro":
        split = dataset["test"]
        for row in split:
            examples.append(
                MultipleChoiceExample(
                    example_id=f"mmlu-pro:{row['question_id']}",
                    group=str(row["category"]),
                    question=str(row["question"]),
                    choices=tuple(str(value).strip() for value in row["options"]),
                    gold=int(row["answer_index"]),
                )
            )
        source_splits = ["test"]
    elif benchmark == "gpqa-diamond":
        split = dataset["train"]
        for index, row in enumerate(split):
            example_id = f"gpqa-diamond:{row.get('Record ID') or index}"
            choices, gold = _shuffle_choices(
                [
                    row["Correct Answer"],
                    row["Incorrect Answer 1"],
                    row["Incorrect Answer 2"],
                    row["Incorrect Answer 3"],
                ],
                0,
                example_id=example_id,
            )
            examples.append(
                MultipleChoiceExample(
                    example_id=example_id,
                    group=str(row["High-level domain"]),
                    question=str(row["Question"]),
                    choices=choices,
                    gold=gold,
                )
            )
        source_splits = ["train"]
    elif benchmark == "mmmlu":
        split = dataset["test"]
        for index, row in enumerate(split):
            examples.append(
                MultipleChoiceExample(
                    example_id=f"mmmlu-zh-cn:{row.get('Unnamed: 0', index)}",
                    group=str(row["Subject"]),
                    question=str(row["Question"]),
                    choices=tuple(str(row[label]).strip() for label in "ABCD"),
                    gold=_answer_index(str(row["Answer"]), "ABCD"),
                )
            )
        source_splits = ["test"]
    elif benchmark == "mmlu-cf":
        source_splits = sorted(name for name in dataset if name.endswith("_val") and name != "val")
        for split_name in source_splits:
            group = split_name[: -len("_val")]
            for index, row in enumerate(dataset[split_name]):
                examples.append(
                    MultipleChoiceExample(
                        example_id=f"mmlu-cf:{group}:{index}",
                        group=group,
                        question=str(row["Question"]),
                        choices=tuple(str(row[label]).strip() for label in "ABCD"),
                        gold=_answer_index(str(row["Answer"]), "ABCD"),
                    )
                )
    else:
        raise ValueError(f"unsupported multiple-choice benchmark {benchmark!r}")

    raw_example_count = len(examples)
    excluded: list[dict[str, str]] = []
    valid: list[MultipleChoiceExample] = []
    for example in examples:
        reason = None
        if not example.question.strip() or len(example.choices) < 2:
            reason = "empty_question_or_too_few_choices"
        elif not 0 <= example.gold < len(example.choices):
            reason = "invalid_gold_index"
        elif any(not choice for choice in example.choices):
            reason = "empty_choice"
        if reason is None:
            valid.append(example)
        else:
            excluded.append({"example_id": example.example_id, "reason": reason})

    sampled = deterministic_sample(valid, limit, sample_seed)
    metadata = {
        "benchmark": benchmark,
        "data_id": str(spec["data_id"]),
        "data_subset": spec.get("data_subset"),
        "revision": str(spec["revision"]),
        "source_splits": source_splits,
        "full_example_count": raw_example_count,
        "eligible_example_count": len(valid),
        "excluded_examples": excluded,
        "evaluated_example_count": len(sampled),
        "sample_seed": sample_seed,
        "dataset_fingerprint": hashlib.sha256(
            "\n".join(item.example_id for item in valid).encode("utf-8")
        ).hexdigest(),
    }
    return sampled, metadata


def choice_metrics(
    log_likelihoods: Sequence[float],
    token_counts: Sequence[int],
    character_counts: Sequence[int],
    gold: int,
) -> dict[str, float]:
    """Build target accuracy and continuous candidates from one RC question."""
    if not (len(log_likelihoods) == len(token_counts) == len(character_counts) and log_likelihoods):
        raise ValueError("choice score arrays must have the same non-zero length")
    if not 0 <= gold < len(log_likelihoods):
        raise ValueError("gold index is out of range")
    if any(value <= 0 for value in (*token_counts, *character_counts)):
        raise ValueError("token and character counts must be positive")

    per_char = np.asarray(log_likelihoods, dtype=np.float64) / np.asarray(
        character_counts, dtype=np.float64
    )
    shifted = per_char - per_char.max()
    normalized = np.exp(shifted) / np.exp(shifted).sum()
    incorrect = np.delete(per_char, gold)
    best = float(per_char.max())
    tied = int(np.count_nonzero(np.isclose(per_char, best, rtol=0.0, atol=1e-12)))
    return {
        "accuracy": float(per_char[gold] >= best - 1e-12) / tied,
        "correct_prob_per_char": math.exp(float(per_char[gold])),
        "normalized_choice_probability": float(normalized[gold]),
        "correct_vs_best_incorrect_logprob_margin": float(per_char[gold] - incorrect.max()),
        "correct_logprob_per_token": float(log_likelihoods[gold] / token_counts[gold]),
        "choice_tie": float(tied > 1),
    }


def aggregate_predictions(
    predictions: Sequence[Mapping[str, Any]], *, aggregation: str
) -> dict[str, Any]:
    metric_names = (
        "accuracy",
        "correct_prob_per_char",
        "normalized_choice_probability",
        "correct_vs_best_incorrect_logprob_margin",
        "correct_logprob_per_token",
        "choice_tie",
    )
    if not predictions:
        raise ValueError("cannot aggregate empty predictions")
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_group[str(row["group"])].append(row)

    group_scores = {
        group: {
            metric: float(np.mean([float(row[metric]) for row in rows])) for metric in metric_names
        }
        for group, rows in sorted(by_group.items())
    }
    micro = {
        metric: float(np.mean([float(row[metric]) for row in predictions]))
        for metric in metric_names
    }
    macro = {
        metric: float(np.mean([scores[metric] for scores in group_scores.values()]))
        for metric in metric_names
    }
    if aggregation == "micro":
        primary = micro
    elif aggregation == "macro_by_group":
        primary = macro
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}")
    return {
        "aggregation": aggregation,
        "primary": primary,
        "micro": micro,
        "macro_by_group": macro,
        "by_group": group_scores,
        "group_sizes": {group: len(rows) for group, rows in sorted(by_group.items())},
    }


class ModelBlobChecksumError(RuntimeError):
    def __init__(self, mismatches: Sequence[Path]):
        self.mismatches = tuple(mismatches)
        super().__init__(
            "model blob checksum mismatch: " + ", ".join(str(path) for path in mismatches)
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model_blob_checksums(snapshot: Path) -> dict[str, str]:
    """Verify Hub LFS blobs whose cache filenames encode their SHA-256."""
    checksums: dict[str, str] = {}
    mismatches = []
    for model_path in sorted(snapshot.glob("model*.safetensors")):
        blob_path = model_path.resolve()
        expected = blob_path.name
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            continue
        actual = _sha256(blob_path)
        checksums[model_path.name] = actual
        if actual != expected:
            mismatches.append(blob_path)
    if mismatches:
        raise ModelBlobChecksumError(mismatches)
    return checksums


def _resolve_snapshot(
    model_name_or_path: str, revision: str | None, cache_dir: Path
) -> tuple[Path, dict[str, str]]:
    path = Path(model_name_or_path).expanduser()
    if path.exists():
        snapshot = path.resolve()
        return snapshot, _verify_model_blob_checksums(snapshot)
    # The mirror's hf_transfer path produced full-sized but non-contiguous blobs in this
    # experiment. Disable it here as well as in the matrix launcher, then verify LFS hashes.
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    try:
        from huggingface_hub import constants as hub_constants
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("model download requires huggingface_hub") from exc
    if hasattr(hub_constants, "HF_HUB_ENABLE_HF_TRANSFER"):
        hub_constants.HF_HUB_ENABLE_HF_TRANSFER = False
    for attempt in range(2):
        resolved = snapshot_download(
            repo_id=model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=True,
            max_workers=1,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model*.safetensors*",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )
        snapshot = Path(resolved).resolve()
        try:
            return snapshot, _verify_model_blob_checksums(snapshot)
        except ModelBlobChecksumError as exc:
            if attempt:
                raise
            for blob_path in exc.mismatches:
                blob_path.unlink()
    raise AssertionError("unreachable model download loop")


def _load_model(model_name_or_path: str, revision: str | None, cache_dir: Path, device: str):
    try:
        import torch
        import hf_olmo  # noqa: F401 - registers OLMo with Transformers Auto classes
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("multiple-choice evaluation requires the OLMES GPU environment") from exc

    snapshot, blob_checksums = _resolve_snapshot(model_name_or_path, revision, cache_dir)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(snapshot / "tokenizer.json"),
        pad_token="<|padding|>",
        eos_token="<|endoftext|>",
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()
    resolved_revision = snapshot.name if snapshot.parent.name == "snapshots" else None
    return model, tokenizer, snapshot, resolved_revision, blob_checksums


def _score_examples(
    model,
    tokenizer,
    examples: Sequence[MultipleChoiceExample],
    *,
    device: str,
    batch_size: int,
    max_sequence_length: int,
    progress_every: int = 100,
) -> tuple[list[dict[str, Any]], int]:
    import torch
    import torch.nn.functional as functional

    work: list[dict[str, Any]] = []
    truncated = 0
    for example_index, example in enumerate(examples):
        prompt_ids = tokenizer.encode(build_prompt(example.question), add_special_tokens=False)
        for choice_index, choice in enumerate(example.choices):
            continuation = " " + choice.strip()
            continuation_ids = tokenizer.encode(continuation, add_special_tokens=False)
            available_prompt = max_sequence_length - len(continuation_ids)
            if available_prompt < 1:
                raise ValueError(f"choice exceeds context window in {example.example_id}")
            selected_prompt = prompt_ids[-available_prompt:]
            truncated += int(len(selected_prompt) < len(prompt_ids))
            work.append(
                {
                    "example_index": example_index,
                    "choice_index": choice_index,
                    "prompt_length": len(selected_prompt),
                    "continuation_length": len(continuation_ids),
                    "character_count": max(1, len(choice.strip())),
                    "input_ids": selected_prompt + continuation_ids,
                }
            )

    work.sort(
        key=lambda item: (len(item["input_ids"]), item["example_index"], item["choice_index"])
    )
    scores: list[list[dict[str, Any] | None]] = [
        [None] * len(example.choices) for example in examples
    ]
    pad_token_id = int(tokenizer.pad_token_id)
    started = time.monotonic()
    for offset in range(0, len(work), batch_size):
        batch = work[offset : offset + batch_size]
        width = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        for row_index, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[row_index, :length] = torch.tensor(
                item["input_ids"], dtype=torch.long, device=device
            )
            attention_mask[row_index, :length] = 1

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
        for row_index, item in enumerate(batch):
            prompt_length = int(item["prompt_length"])
            continuation_length = int(item["continuation_length"])
            positions = logits[
                row_index,
                prompt_length - 1 : prompt_length + continuation_length - 1,
                :,
            ].float()
            targets = input_ids[
                row_index,
                prompt_length : prompt_length + continuation_length,
            ]
            log_probs = functional.log_softmax(positions, dim=-1)
            token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            scores[int(item["example_index"])][int(item["choice_index"])] = {
                "log_likelihood": float(token_log_probs.sum().cpu()),
                "token_count": continuation_length,
                "character_count": int(item["character_count"]),
            }
        del logits, input_ids, attention_mask
        batch_number = offset // batch_size + 1
        if progress_every and batch_number % progress_every == 0:
            elapsed = time.monotonic() - started
            print(
                f"scored {min(offset + batch_size, len(work))}/{len(work)} choices "
                f"in {elapsed:.1f}s",
                flush=True,
            )

    predictions: list[dict[str, Any]] = []
    for example, choice_rows in zip(examples, scores):
        if any(row is None for row in choice_rows):
            raise RuntimeError(f"missing choice score for {example.example_id}")
        rows = [row for row in choice_rows if row is not None]
        metrics = choice_metrics(
            [float(row["log_likelihood"]) for row in rows],
            [int(row["token_count"]) for row in rows],
            [int(row["character_count"]) for row in rows],
            example.gold,
        )
        predictions.append(
            {
                "example_id": example.example_id,
                "group": example.group,
                "choice_count": len(example.choices),
                "gold": example.gold,
                **metrics,
            }
        )
    return predictions, truncated


def _write_predictions(path: Path, predictions: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n")


def evaluate_multiple_choice_benchmarks(
    *,
    model_name_or_path: str,
    model_revision: str | None,
    benchmark_specs: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    model_cache_dir: str | Path,
    dataset_cache_dir: str | Path,
    device: str,
    batch_size: int,
    max_examples: int | None,
    sample_seed: int,
    max_sequence_length: int = 2048,
) -> list[Path]:
    """Load a model once, evaluate all requested benchmarks, and write compact artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, snapshot, resolved_revision, blob_checksums = _load_model(
        model_name_or_path, model_revision, Path(model_cache_dir), device
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    summaries: list[Path] = []
    for benchmark, spec in benchmark_specs.items():
        started = time.monotonic()
        examples, dataset_metadata = load_multiple_choice_examples(
            benchmark,
            spec,
            cache_dir=dataset_cache_dir,
            limit=max_examples,
            sample_seed=sample_seed,
        )
        predictions, truncated = _score_examples(
            model,
            tokenizer,
            examples,
            device=device,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
        )
        aggregate = aggregate_predictions(
            predictions, aggregation=str(spec.get("aggregation", "macro_by_group"))
        )
        benchmark_dir = output_dir / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = benchmark_dir / "predictions.jsonl"
        _write_predictions(predictions_path, predictions)
        summary = {
            "stage": 12,
            "method": "multiple_choice_proxy_evaluation_v1",
            "protocol": PROTOCOL,
            "prompt_template": PROMPT_TEMPLATE,
            "model": model_name_or_path,
            "requested_model_revision": model_revision,
            "resolved_model_revision": resolved_revision,
            "resolved_model_path": str(snapshot),
            "parameter_count": parameter_count,
            "model_blob_sha256": blob_checksums,
            "dataset": dataset_metadata,
            "aggregate": aggregate,
            "truncated_choice_count": truncated,
            "batch_size": batch_size,
            "max_sequence_length": max_sequence_length,
            "elapsed_seconds": time.monotonic() - started,
            "predictions_path": str(predictions_path.resolve()),
            "evidence_level": (
                "public checkpoint local reevaluation; default seed unless model says otherwise"
            ),
        }
        summary_path = benchmark_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries.append(summary_path)
        print(
            f"{benchmark}: accuracy={aggregate['primary']['accuracy']:.6f} "
            f"correct_prob_per_char={aggregate['primary']['correct_prob_per_char']:.6f} "
            f"elapsed={summary['elapsed_seconds']:.1f}s",
            flush=True,
        )
    return summaries
