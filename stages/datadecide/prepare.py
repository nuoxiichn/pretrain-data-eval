"""Materialize a representative fixed-token sample from official recipe path lists."""

from __future__ import annotations

import hashlib
import json
import os
import re
import runpy
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence


_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_paths(source: str | Path, recipe: str) -> tuple[list[str], str]:
    """Load a recipe from a pinned local copy of OLMo's named_data_mixes.py."""
    source = Path(source).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    values = runpy.run_path(str(source))
    paths = values.get("DATA_PATHS", {}).get(recipe)
    if not paths:
        raise ValueError(f"recipe {recipe!r} is absent from {source}")
    if not all(isinstance(path, str) and path.endswith(".npy") for path in paths):
        raise ValueError(f"recipe {recipe!r} contains invalid paths")
    return list(paths), digest


def _selected_paths(paths: Sequence[str], chunks: int) -> list[str]:
    if chunks <= 0 or chunks > len(paths):
        raise ValueError("chunks must be in [1, number of recipe paths]")
    return [
        paths[min(len(paths) - 1, (2 * index + 1) * len(paths) // (2 * chunks))]
        for index in range(chunks)
    ]


def _remote_size(url: str, timeout: int) -> int:
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
        if response.status != 206 or match is None:
            raise RuntimeError(f"server did not honor range request for {url}")
        return int(match.group(3))


def _download_range(
    *,
    url: str,
    output: Path,
    length: int,
    seed_material: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            size = _remote_size(url, timeout)
            break
        except Exception as exc:  # pragma: no cover - exercised by real network retries
            error = exc
            if attempt < retries:
                time.sleep(2**attempt)
    else:
        raise RuntimeError(f"failed to inspect {url}: {error}")
    if length > size:
        raise ValueError(f"requested {length} bytes from {url}, which has only {size}")
    digest = hashlib.sha256(seed_material.encode("utf-8")).digest()
    max_start = size - length
    start = (int.from_bytes(digest[:8], "big") % (max_start + 1)) if max_start else 0
    start -= start % 2
    end = start + length - 1
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size == length:
        return {
            "url": url,
            "remote_bytes": size,
            "range_start": start,
            "range_end": end,
            "bytes": length,
            "sha256": _sha256(output),
            "reused": True,
        }
    temporary = output.with_suffix(output.suffix + ".partial")
    error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with (
                urllib.request.urlopen(request, timeout=timeout) as response,
                temporary.open("wb") as handle,
            ):
                if response.status != 206:
                    raise RuntimeError(f"server returned {response.status}, expected 206")
                expected_range = f"bytes {start}-{end}/{size}"
                if response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError("server returned an unexpected byte range")
                shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
            if temporary.stat().st_size != length:
                raise RuntimeError(f"short range response: {temporary.stat().st_size} != {length}")
            os.replace(temporary, output)
            return {
                "url": url,
                "remote_bytes": size,
                "range_start": start,
                "range_end": end,
                "bytes": length,
                "sha256": _sha256(output),
            }
        except Exception as exc:  # pragma: no cover - exercised by real network retries
            error = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}: {error}")


def materialize_recipe(
    *,
    named_mixes_source: str | Path,
    recipe: str,
    output: str | Path,
    target_tokens: int,
    endpoint: str,
    repo_id: str,
    revision: str,
    chunks: int = 24,
    workers: int = 8,
    timeout: int = 120,
    retries: int = 3,
) -> dict[str, Any]:
    """Download deterministic ranges spread across an official logical recipe."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    output = Path(output).resolve()
    paths, source_sha256 = load_official_paths(named_mixes_source, recipe)
    chosen = _selected_paths(paths, chunks)
    target_bytes = target_tokens * 2
    base, remainder = divmod(target_bytes, chunks)
    lengths = [base + int(index < remainder) for index in range(chunks)]
    # uint16 alignment is required for every range and concatenation boundary.
    for index in range(chunks - 1):
        if lengths[index] % 2:
            lengths[index] += 1
            lengths[-1] -= 1
    part_dir = output.with_suffix(output.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/{revision}"

    def fetch(index: int) -> dict[str, Any]:
        repo_path = chosen[index]
        result = _download_range(
            url=f"{base_url}/{repo_path}",
            output=part_dir / f"part-{index:03d}.bin",
            length=lengths[index],
            seed_material=f"{revision}:{recipe}:{index}:{repo_path}",
            timeout=timeout,
            retries=retries,
        )
        return {"repo_path": repo_path, **result}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        parts = list(executor.map(fetch, range(chunks)))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".partial")
    with temporary_output.open("wb") as destination:
        for index in range(chunks):
            with (part_dir / f"part-{index:03d}.bin").open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    if temporary_output.stat().st_size != target_bytes:
        raise RuntimeError("materialized stream has the wrong size")
    os.replace(temporary_output, output)
    manifest = {
        "recipe": recipe,
        "repo_id": repo_id,
        "revision": revision,
        "named_data_mixes_path": str(Path(named_mixes_source).resolve()),
        "named_data_mixes_sha256": source_sha256,
        "official_path_count": len(paths),
        "selection": "evenly spaced paths with deterministic within-file ranges",
        "format": "headerless little-endian uint16 token stream",
        "target_tokens": target_tokens,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": _sha256(output),
        "parts": parts,
        "warning": "Representative official-recipe sample, not the authors' exact shuffled order.",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
