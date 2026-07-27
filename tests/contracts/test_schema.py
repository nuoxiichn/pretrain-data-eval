from __future__ import annotations

import json
import math

import pytest
from jsonschema.exceptions import ValidationError

from pretrain_data_eval.reader import read_documents
from pretrain_data_eval.schema import (
    SCHEMA_VERSION,
    DocResult,
    SchemaError,
    prepare_summary,
    validate_summary,
    write_per_doc,
    write_summary,
)


def test_normalized_input_matches_machine_schema(tmp_path, schema_validator):
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"doc_id": "x", "text": "hello"}) + "\n", encoding="utf-8")
    document = list(read_documents(path))[0]
    schema_validator("input_document.schema.json").validate(document)


def test_doc_result_matches_machine_schema(schema_validator):
    result = DocResult(doc_id="x", scores={"score": 0.2}, flags={"hit": False})
    assert result.schema_version == SCHEMA_VERSION
    schema_validator("per_doc.schema.json").validate(result.to_dict())


def test_doc_result_rejects_empty_id_non_boolean_flags_and_nan():
    with pytest.raises(SchemaError, match="non-empty"):
        DocResult(doc_id="", scores={}, flags={})
    with pytest.raises(SchemaError, match="must be boolean"):
        DocResult(doc_id="x", scores={}, flags={"hit": 1})  # type: ignore[arg-type]
    with pytest.raises(SchemaError, match="finite JSON"):
        DocResult(doc_id="x", scores={"score": math.nan}, flags={})


def test_summary_header_is_added_without_mutating_caller(schema_validator):
    source = {"total_docs": 3}
    payload = prepare_summary(source)
    assert source == {"total_docs": 3}
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["artifact_type"] == "summary"
    validate_summary(payload)
    schema_validator("summary.schema.json").validate(payload)


def test_summary_rejects_missing_or_wrong_contract_header():
    with pytest.raises(SchemaError, match="missing"):
        validate_summary({"total_docs": 1})
    with pytest.raises(SchemaError, match="schema_version"):
        prepare_summary({"schema_version": "2.0.0"})
    with pytest.raises(SchemaError, match="artifact_type"):
        prepare_summary({"artifact_type": "per_doc"})


def test_writers_emit_strict_schema_valid_json(tmp_path, schema_validator):
    write_per_doc([DocResult("x", {"count": 1}, {"hit": True})], tmp_path)
    write_summary({"total_docs": 1}, tmp_path)
    per_doc = json.loads((tmp_path / "per_doc.jsonl").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    schema_validator("per_doc.schema.json").validate(per_doc)
    schema_validator("summary.schema.json").validate(summary)


def test_machine_schema_rejects_extra_per_doc_fields(schema_validator):
    payload = DocResult("x", {}, {}).to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        schema_validator("per_doc.schema.json").validate(payload)
