"""Atomic manifests, status files, event logs, and checkpoint indexes."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


UTC = timezone.utc
SCHEMA_VERSION = "data_advisor_runtime_v1"


def utc_now() -> str:
    """Return a stable, timezone-aware timestamp for manifests and events."""

    return datetime.now(UTC).isoformat(timespec="milliseconds")


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    """Atomically replace *path*, fsyncing both file and containing directory."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON with an atomic replace and deterministic formatting."""

    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_bytes(path, encoded.encode("utf-8"))


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Append one fsynced JSON record with a single ``O_APPEND`` write."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, default=_json_default, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"object is not JSON serializable: {type(value).__name__}")


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: str | Path) -> str:
    """Hash a directory's relative file names and contents deterministically."""

    root = Path(path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    digest = hashlib.sha256()
    for child in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = child.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(child)))
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def runtime_environment() -> dict[str, Any]:
    """Collect non-secret environment details useful for reproducing a smoke."""

    cuda = {
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        cuda["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "torch": torch.__version__,
        "cuda": cuda,
        "git_revision": _git_revision(),
    }


class RunRecorder:
    """Atomic manifest/status writer and append-only event recorder.

    Rank zero creates the initial files.  Other ranks may call ``event`` after
    the directory exists; each event is written with one ``O_APPEND`` write and
    fsynced so a pre-emption cannot leave a half JSON object in the log.
    """

    def __init__(self, output_dir: str | Path, *, run_name: str, manifest: Mapping[str, Any]):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "run_manifest.json"
        self.status_path = self.output_dir / "status.json"
        self.events_path = self.output_dir / "events.jsonl"
        self.run_id = str(manifest.get("run_id") or uuid.uuid4().hex)
        self._manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_name": run_name,
            "created_at": utc_now(),
            "environment": runtime_environment(),
            **dict(manifest),
        }
        self._manifest["run_id"] = self.run_id
        self._manifest.setdefault("run_name", run_name)

    @classmethod
    def create(
        cls,
        output_dir: str | Path,
        *,
        run_name: str,
        command: str,
        config: Mapping[str, Any] | None = None,
    ) -> "RunRecorder":
        recorder = cls(
            output_dir,
            run_name=run_name,
            manifest={"command": command, "config": dict(config or {})},
        )
        recorder.initialize()
        return recorder

    def initialize(self) -> None:
        """Create manifest/status exactly once, tolerating rank races.

        The lock is intentionally implemented with ``O_EXCL`` rather than a
        third-party dependency so two torchrun ranks cannot replace a manifest
        with different run IDs.  A stale lock older than one hour is safe to
        remove after a pre-empted process; the manifest itself remains the
        source of truth.
        """

        lock_path = self.output_dir / ".manifest.lock"
        acquired = False
        for _ in range(600):
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(fd)
                acquired = True
                break
            except FileExistsError:
                try:
                    if time.time() - lock_path.stat().st_mtime > 3600:
                        lock_path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                time.sleep(0.05)
        if not acquired:
            raise TimeoutError(f"timed out acquiring manifest lock: {lock_path}")
        try:
            if not self.manifest_path.exists():
                atomic_write_json(self.manifest_path, self._manifest)
            else:
                # The rank that lost the race must use the existing run ID.
                try:
                    existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = None
                if isinstance(existing, dict):
                    self.run_id = str(existing.get("run_id", self.run_id))
                    self._manifest = existing
            if not self.status_path.exists():
                atomic_write_json(
                    self.status_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": self.run_id,
                        "state": "running",
                        "updated_at": utc_now(),
                    },
                )
        finally:
            lock_path.unlink(missing_ok=True)

    def event(self, name: str, **fields: Any) -> dict[str, Any]:
        """Append one structured event and return the encoded record."""

        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "timestamp": utc_now(),
            "event": name,
            "rank": int(os.environ.get("RANK", "0")),
            **fields,
        }
        append_jsonl(self.events_path, record)
        return record

    def update_status(self, state: str, **fields: Any) -> dict[str, Any]:
        """Atomically merge a status update."""

        current: dict[str, Any] = {}
        if self.status_path.exists():
            try:
                loaded = json.loads(self.status_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    current = loaded
            except (OSError, json.JSONDecodeError):
                current = {}
        current.update(
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "state": state,
                "updated_at": utc_now(),
                **fields,
            }
        )
        atomic_write_json(self.status_path, current)
        return current

    def summary(self, payload: Mapping[str, Any]) -> None:
        atomic_write_json(self.output_dir / "summary.json", dict(payload))

    def checkpoint_index(
        self, *, checkpoint: str | Path, step: int, extra: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Record checkpoint files and hashes without exposing model contents."""

        root = Path(checkpoint).resolve()
        files = []
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(item for item in root.rglob("*") if item.is_file())
        else:
            raise FileNotFoundError(root)
        records = [
            {
                "path": str(item.relative_to(root)) if root.is_dir() else item.name,
                "size": item.stat().st_size,
                "sha256": file_sha256(item),
            }
            for item in sorted(files)
        ]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "step": int(step),
            "checkpoint": str(root),
            "files": records,
            **dict(extra or {}),
        }
        atomic_write_json(self.output_dir / "checkpoint_index.json", payload)
        return payload
