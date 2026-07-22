"""Versioned output contracts shared by all pipeline stages."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "1.0.0"
PER_DOC_ARTIFACT = "per_doc"
SUMMARY_ARTIFACT = "summary"


class SchemaError(ValueError):
    """Raised when an input or output object violates the public contract."""


def _validate_json_object(value: object, name: str) -> None:
    if not isinstance(value, dict):
        raise SchemaError(f"{name} must be an object, got {type(value).__name__}")
    if not all(isinstance(key, str) for key in value):
        raise SchemaError(f"{name} keys must be strings")
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name} must contain only finite JSON values: {exc}") from exc


@dataclass
class DocResult:
    """One per-document result under the 1.0 output contract."""

    doc_id: str
    scores: dict[str, Any]
    flags: dict[str, bool]
    schema_version: str = field(default=SCHEMA_VERSION, init=False)
    artifact_type: str = field(default=PER_DOC_ARTIFACT, init=False)

    def __post_init__(self) -> None:
        validate_doc_result(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_doc_result(result: DocResult | Mapping[str, Any]) -> None:
    """Validate the common per-document fields and JSON compatibility."""
    payload = asdict(result) if isinstance(result, DocResult) else dict(result)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"per_doc schema_version must be {SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    if payload.get("artifact_type") != PER_DOC_ARTIFACT:
        raise SchemaError("per_doc artifact_type must be 'per_doc'")
    if not isinstance(payload.get("doc_id"), str) or not payload["doc_id"].strip():
        raise SchemaError("per_doc doc_id must be a non-empty string")
    _validate_json_object(payload.get("scores"), "per_doc scores")
    _validate_json_object(payload.get("flags"), "per_doc flags")
    invalid_flags = {
        key: value for key, value in payload["flags"].items() if not isinstance(value, bool)
    }
    if invalid_flags:
        raise SchemaError(f"per_doc flags must be boolean: {invalid_flags}")


def prepare_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated summary with the current public contract header."""
    if not isinstance(summary, Mapping):
        raise SchemaError(f"summary must be a mapping, got {type(summary).__name__}")
    version = summary.get("schema_version", SCHEMA_VERSION)
    artifact = summary.get("artifact_type", SUMMARY_ARTIFACT)
    if version != SCHEMA_VERSION:
        raise SchemaError(
            f"summary schema_version must be {SCHEMA_VERSION!r}, got {version!r}"
        )
    if artifact != SUMMARY_ARTIFACT:
        raise SchemaError("summary artifact_type must be 'summary'")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": SUMMARY_ARTIFACT,
        **summary,
    }
    _validate_json_object(payload, "summary")
    return payload


def validate_summary(summary: Mapping[str, Any]) -> None:
    """Validate a summary that already contains its contract header."""
    if "schema_version" not in summary or "artifact_type" not in summary:
        raise SchemaError("summary is missing schema_version or artifact_type")
    prepare_summary(summary)


def make_output_dir(base: str | Path, stage: str, dataset: str) -> Path:
    """Return and create ``<base>/<dataset>_<timestamp>/<stage>``."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base).resolve() / f"{dataset}_{ts}" / stage
    out.mkdir(parents=True, exist_ok=True)
    return out


def use_output_dir(path: str | Path) -> Path:
    """Use an explicit output directory, creating it when needed."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_per_doc(results: list[DocResult], out_dir: Path, name: str = "per_doc") -> Path:
    """Write validated per-document JSONL using strict JSON encoding."""
    out = out_dir / f"{name}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for result in results:
            validate_doc_result(result)
            handle.write(
                json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False) + "\n"
            )
    return out


def write_summary(summary: Mapping[str, Any], out_dir: Path, name: str = "summary") -> Path:
    """Write a summary with a validated 1.0 contract header."""
    payload = prepare_summary(summary)
    out = out_dir / f"{name}.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return out
