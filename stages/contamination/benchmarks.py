"""Generic benchmark loader for contamination detection.

Loads benchmarks from local files or HuggingFace datasets, normalises each
sample to a common schema. Supports two modes per benchmark entry:

1. **Local file** — set ``path`` to a local JSONL/JSON file
2. **HuggingFace** — set ``name`` to an HF dataset identifier (requires network)

Config example (in stage5.yaml):

    benchmarks:
      cache_dir: /mnt/public/data
      datasets:
        # HuggingFace mode
        - name: cais/mmlu
          subset: all
          split: test
          text_field: question
          label: mmlu
        # Local file mode
        - path: /mnt/public/data/cmmlu/test.jsonl
          text_field: question
          label: cmmlu
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict


class BenchItem(TypedDict):
    bench_id: str
    text: str
    code: str | None
    benchmark: str


def _load_local(spec: dict) -> list[BenchItem]:
    """Load benchmark items from a local JSON or JSONL file."""
    path = Path(spec["path"])
    text_field = spec["text_field"]
    label = spec["label"]
    code_field = spec.get("code_field")

    items: list[BenchItem] = []

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = str(row.get(text_field) or "")
                code = str(row[code_field]) if code_field and row.get(code_field) else None
                items.append(BenchItem(
                    bench_id=f"{label}_{idx}",
                    text=text,
                    code=code,
                    benchmark=label,
                ))
    elif path.suffix == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else [data]
        else:
            rows = [data]
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            text = str(row.get(text_field) or "")
            code = str(row[code_field]) if code_field and row.get(code_field) else None
            items.append(BenchItem(
                bench_id=f"{label}_{idx}",
                text=text,
                code=code,
                benchmark=label,
            ))
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow required for Parquet benchmarks") from exc
        table = pq.read_table(path)
        for idx, row in enumerate(table.to_pylist()):
            text = str(row.get(text_field) or "")
            code = str(row[code_field]) if code_field and row.get(code_field) else None
            items.append(BenchItem(
                bench_id=f"{label}_{idx}",
                text=text,
                code=code,
                benchmark=label,
            ))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return items


def _load_hf(spec: dict, cache_dir: str) -> list[BenchItem]:
    """Load benchmark items from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace `datasets` is required for remote benchmark loading. "
            "Run: pip install datasets"
        ) from exc

    name = spec["name"]
    split = spec["split"]
    text_field = spec["text_field"]
    label = spec["label"]
    subset = spec.get("subset")
    code_field = spec.get("code_field")

    ds = load_dataset(name, subset, split=split, cache_dir=cache_dir)

    items: list[BenchItem] = []
    for idx, row in enumerate(ds):
        text = str(row.get(text_field) or "")
        code = str(row[code_field]) if code_field and row.get(code_field) else None
        items.append(BenchItem(
            bench_id=f"{label}_{idx}",
            text=text,
            code=code,
            benchmark=label,
        ))

    return items


def load_benchmarks(config: dict) -> dict[str, list[BenchItem]]:
    """Load all benchmarks listed in *config* and return ``{label: [items]}``.

    Each dataset entry must have ``label`` and ``text_field``, plus either:
      - ``path`` — local JSONL/JSON/Parquet file
      - ``name`` + ``split`` — HuggingFace dataset identifier
    """
    cache_dir = config.get("cache_dir", "/mnt/public/data")
    dataset_specs = config.get("datasets", [])
    if not dataset_specs:
        raise ValueError("benchmarks.datasets is empty — nothing to load")

    result: dict[str, list[BenchItem]] = {}

    for spec in dataset_specs:
        label = spec["label"]
        if "path" in spec:
            items = _load_local(spec)
        elif "name" in spec:
            items = _load_hf(spec, cache_dir)
        else:
            raise ValueError(
                f"Benchmark '{label}': must have either 'path' (local) or 'name' (HF)"
            )
        result[label] = items

    return result
