"""Sampling helpers for stage CLIs.

`sample_documents` consumes a document iterator and returns up to `n` items
using one of two strategies:

- "head"     : the original behavior — take the first `n` items.
- "random"   : reservoir sampling — every input has equal probability of
               selection regardless of total size, single pass, O(n) memory.

Both strategies are single-pass. Random uses Algorithm R (Vitter, 1985) so it
works on iterators of unknown length (parquet files, stage outputs) without
materializing the full corpus.

The default is "random" because deterministic head sampling is biased toward
whatever order the underlying loader yields (file order, row-group order),
which collides with how Stage 4 dedup and similar correlate with file layout.

Reproducibility: pass `seed` to fix the sample. Without a seed the result is
non-deterministic.
"""

from __future__ import annotations

import random
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")

SAMPLE_MODES = ("head", "random")
DEFAULT_SAMPLE_MODE = "random"
DEFAULT_SEED = 42


def sample_documents(
    docs: Iterable[T],
    n: int | None,
    mode: str = DEFAULT_SAMPLE_MODE,
    seed: int | None = DEFAULT_SEED,
) -> list[T]:
    """Materialize up to `n` items from `docs`.

    If `n` is None or 0, materialize all items (preserves original full-scan
    behavior of `list(read_documents(...))`).

    mode="head" returns the first `n` in iteration order.
    mode="random" uses reservoir sampling; with `seed` set, the result is
    deterministic across runs.
    """
    if n is None or n <= 0:
        return list(docs)
    if mode == "head":
        return _take_head(docs, n)
    if mode == "random":
        return _reservoir(docs, n, seed)
    raise ValueError(f"unknown sample mode {mode!r}; expected one of {SAMPLE_MODES}")


def _take_head(docs: Iterable[T], n: int) -> list[T]:
    out: list[T] = []
    it: Iterator[T] = iter(docs)
    for _ in range(n):
        try:
            out.append(next(it))
        except StopIteration:
            break
    return out


def _reservoir(docs: Iterable[T], n: int, seed: int | None) -> list[T]:
    rng = random.Random(seed)
    reservoir: list[T] = []
    for i, doc in enumerate(docs):
        if i < n:
            reservoir.append(doc)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = doc
    return reservoir
