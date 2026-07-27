"""Deterministic token-stream readers for bounded proxy experiments.

The DataDecide assets are headerless little-endian integer streams.  This module
keeps the reader deliberately small and auditable: a recipe is represented by a
memory map, examples are fixed-length contiguous blocks, and the block order is
an explicit function of ``seed``.  No framework data loader is involved, which
makes the token count and resume position easy to verify.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class TokenBlockStream:
    """Random-access fixed-length blocks over a uint16/uint32 token stream."""

    def __init__(self, path: str | Path, *, sequence_length: int, seed: int) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        size = self.path.stat().st_size
        suffix = self.path.suffix.lower()
        if suffix == ".u32" or suffix == ".uint32":
            dtype = np.dtype("<u4")
        elif suffix in {".npy", ".bin", ".u16", ".uint16"}:
            # DataDecide's .npy files are intentionally headerless uint16 files.
            if size % 2:
                raise ValueError(f"token stream is not uint16-aligned: {self.path}")
            dtype = np.dtype("<u2")
        else:
            # A manifest can disambiguate a nonstandard extension.
            dtype = np.dtype("<u4") if size % 4 == 0 and size % 2 != 0 else np.dtype("<u2")
        self.dtype = dtype
        self.token_count = size // dtype.itemsize
        self.sequence_length = int(sequence_length)
        if self.sequence_length <= 1:
            raise ValueError("sequence_length must be greater than one")
        self.block_count = self.token_count // self.sequence_length
        if self.block_count <= 0:
            raise ValueError(f"stream is shorter than one sequence: {self.path}")
        self.tokens = np.memmap(self.path, mode="r", dtype=dtype, shape=(self.token_count,))
        self.order = np.random.default_rng(int(seed)).permutation(self.block_count).astype(
            np.int64, copy=False
        )
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.block_count

    def block(self, logical_index: int, *, device: torch.device) -> torch.Tensor:
        """Return one block, cycling through a deterministic shuffled epoch."""

        block_index = int(self.order[int(logical_index) % self.block_count])
        start = block_index * self.sequence_length
        values = np.asarray(self.tokens[start : start + self.sequence_length], dtype=np.int64)
        return torch.from_numpy(values.copy()).to(device=device, dtype=torch.long, non_blocking=True)

    def batch(self, logical_start: int, batch_size: int, *, device: torch.device) -> torch.Tensor:
        rows = [self.block(logical_start + row, device=device) for row in range(int(batch_size))]
        return torch.stack(rows, dim=0)

    def summary(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return {
            "path": str(self.path),
            "bytes": self.path.stat().st_size,
            "sha256": digest.hexdigest(),
            "dtype": self.dtype.str,
            "token_count": self.token_count,
            "sequence_length": self.sequence_length,
            "block_count": self.block_count,
            "seed": self.seed,
        }


def load_recipe_manifest(path: str | Path) -> dict[str, Any] | None:
    """Load an adjacent token-stream manifest when one exists."""

    source = Path(path)
    candidates = [
        source.with_suffix(source.suffix + ".manifest.json"),
        source.with_name(source.name + ".manifest.json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {"path": str(candidate), "parse_error": True}
            return value if isinstance(value, dict) else {"value": value}
    return None
