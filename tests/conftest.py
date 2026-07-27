from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def schema_validator():
    def load(name: str) -> Draft202012Validator:
        payload = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(payload)
        return Draft202012Validator(payload)

    return load


@pytest.fixture
def sample_documents():
    return [
        {
            "doc_id": "a",
            "text": "A sufficiently long paragraph shared by two records for testing. " * 2,
            "source": "test",
            "url": "https://example.com/a",
            "timestamp": "2026-07-18T00:00:00Z",
            "language": "en",
            "meta": {},
        },
        {
            "doc_id": "b",
            "text": "A sufficiently long paragraph shared by two records for testing. " * 2,
            "source": "test",
            "url": None,
            "timestamp": None,
            "language": "en",
            "meta": {},
        },
        {
            "doc_id": "c",
            "text": "A distinct document with no duplicate body.",
            "source": "test",
            "url": None,
            "timestamp": None,
            "language": "en",
            "meta": {},
        },
    ]
