from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pretrain_data_eval.reader import read_documents


def test_jsonl_field_map_and_extra_metadata(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(
        json.dumps({"uid": 12, "content": "hello", "style": "qa"}) + "\n",
        encoding="utf-8",
    )
    docs = list(
        read_documents(path, {"field_map": {"uid": "doc_id", "content": "text"}})
    )
    assert docs == [
        {
            "doc_id": "12",
            "text": "hello",
            "source": None,
            "url": None,
            "timestamp": None,
            "language": None,
            "meta": {"style": "qa"},
        }
    ]


def test_path_metadata_fills_only_missing_values(tmp_path):
    folder = tmp_path / "corpus_zh"
    folder.mkdir()
    path = folder / "part.jsonl"
    path.write_text(
        json.dumps({"doc_id": "1", "text": "x", "language": "en"}) + "\n",
        encoding="utf-8",
    )
    doc = list(
        read_documents(
            path,
            {"path_meta": {"language": {"corpus_zh": "zh"}, "source": {"corpus": "web"}}},
        )
    )[0]
    assert doc["language"] == "en"
    assert doc["source"] == "web"


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"text": "missing id"}, "doc_id"),
        ({"doc_id": "x", "text": None}, "text must be a string"),
        ({"doc_id": "x", "text": "ok", "meta": []}, "meta must be an object"),
        (["not", "an", "object"], "JSON object"),
    ],
)
def test_invalid_documents_fail_with_location(tmp_path, payload, message):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message) as exc_info:
        list(read_documents(path))
    assert f"{path}:1" in str(exc_info.value)


def test_invalid_json_fails_with_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"doc_id":\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad.jsonl:1: invalid document"):
        list(read_documents(path))


def test_parquet_directory_is_sorted_and_streamed(tmp_path):
    for name, doc_id in (("b.parquet", "b"), ("a.parquet", "a")):
        pq.write_table(pa.table({"doc_id": [doc_id], "text": [name]}), tmp_path / name)
    docs = list(read_documents(tmp_path, {"glob": "*.parquet", "batch_size": 1}))
    assert [doc["doc_id"] for doc in docs] == ["a", "b"]


def test_missing_path_and_empty_directory_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(read_documents(tmp_path / "missing.jsonl"))
    with pytest.raises(FileNotFoundError, match="No files matching"):
        list(read_documents(tmp_path))
