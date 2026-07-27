from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from research.data_advisor.runtime import (
    RunRecorder,
    append_jsonl,
    atomic_write_json,
    directory_sha256,
    file_sha256,
)
from research.data_advisor.token_stream import TokenBlockStream


def test_atomic_json_and_append_jsonl(tmp_path):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"state": "running", "step": 1})
    atomic_write_json(target, {"state": "passed", "step": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "state": "passed",
        "step": 2,
    }

    events = tmp_path / "events.jsonl"
    append_jsonl(events, {"event": "start"})
    append_jsonl(events, {"event": "complete"})
    assert [json.loads(line)["event"] for line in events.read_text().splitlines()] == [
        "start",
        "complete",
    ]


def test_concurrent_recorders_attach_to_one_run_id(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "research.data_advisor.runtime.runtime_environment",
        lambda: {"test": True},
    )

    def create_recorder(_index):
        return RunRecorder.create(
            tmp_path,
            run_name="race-test",
            command="pytest",
            config={"world_size": 8},
        ).run_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        run_ids = list(executor.map(create_recorder, range(8)))

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert len(set(run_ids)) == 1
    assert manifest["run_id"] == run_ids[0]
    assert status["run_id"] == run_ids[0]
    assert not (tmp_path / ".manifest.lock").exists()


def test_checkpoint_index_records_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "research.data_advisor.runtime.runtime_environment",
        lambda: {"test": True},
    )
    recorder = RunRecorder.create(tmp_path / "run", run_name="checkpoint-test", command="pytest")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    first = checkpoint / "a.distcp"
    second = checkpoint / ".metadata"
    first.write_bytes(b"weights")
    second.write_bytes(b"metadata")

    index = recorder.checkpoint_index(checkpoint=checkpoint, step=3)
    assert index["step"] == 3
    assert {row["path"] for row in index["files"]} == {"a.distcp", ".metadata"}
    assert {row["sha256"] for row in index["files"]} == {
        file_sha256(first),
        file_sha256(second),
    }
    assert len(directory_sha256(checkpoint)) == 64


def test_token_stream_order_is_deterministic_and_cycles(tmp_path):
    path = tmp_path / "tokens.u32"
    np.arange(24, dtype="<u4").tofile(path)

    first = TokenBlockStream(path, sequence_length=4, seed=7)
    second = TokenBlockStream(path, sequence_length=4, seed=7)

    assert first.order.tolist() == second.order.tolist()
    assert torch.equal(
        first.block(0, device=torch.device("cpu")),
        first.block(len(first), device=torch.device("cpu")),
    )
    assert first.summary()["token_count"] == 24
