"""Stage 9: 长上下文 — compute functions.

Audits Megatron-LM training config for correct packing boundary settings:
  - reset_position_ids
  - reset_attention_mask
  - eod_mask_loss
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from src.schema import DocResult


_REQUIRED_PARAMS = {
    "reset_position_ids": {
        "keys": [
            "reset_position_ids", "reset-position-ids",
            "--reset-position-ids", "--reset_position_ids",
        ],
        "expected": True,
        "description": "Reset position IDs at document boundaries to prevent cross-document position leakage",
    },
    "reset_attention_mask": {
        "keys": [
            "reset_attention_mask", "reset-attention-mask",
            "--reset-attention-mask", "--reset_attention_mask",
        ],
        "expected": True,
        "description": "Reset attention mask at document boundaries to prevent cross-document attention",
    },
    "eod_mask_loss": {
        "keys": [
            "eod_mask_loss", "eod-mask-loss",
            "--eod-mask-loss", "--eod_mask_loss",
        ],
        "expected": True,
        "description": "Mask loss at EOD tokens to prevent learning EOD prediction",
    },
}


def _parse_config_file(config_path: str) -> tuple[dict, str]:
    """Parse a training config file into a flat key-value dict.

    Tries YAML → JSON → text regex parsing.
    Returns (flat_dict, detected_format).
    """
    path = Path(config_path)
    content = path.read_text(encoding="utf-8")

    # Try YAML
    try:
        data = yaml.safe_load(content)
        if isinstance(data, dict):
            flat = _flatten_dict(data)
            return flat, "yaml"
    except Exception:
        pass

    # Try JSON
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            flat = _flatten_dict(data)
            return flat, "json"
    except Exception:
        pass

    # Text / shell script parsing
    flat: dict = {}

    # --flag (argparse boolean, treated as True)
    for m in re.finditer(r"(--[\w-]+)(?:\s|$)", content):
        key = m.group(1)
        flat[key] = True

    # --key=value or --key value (non-flag)
    for m in re.finditer(r"(--[\w-]+)[=\s]+(\S+)", content):
        key = m.group(1)
        val = _coerce_value(m.group(2))
        flat[key] = val

    # key = value or key: value (ini/yaml-like lines)
    for m in re.finditer(r"^([\w_][\w_.-]*)\s*[=:]\s*(.+)$", content, re.MULTILINE):
        key = m.group(1).strip()
        val = _coerce_value(m.group(2).strip())
        flat[key] = val

    return flat, "text"


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict into dotted keys, keeping leaf-level keys too."""
    flat: dict = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, full_key))
        else:
            flat[k] = v
            flat[full_key] = v
    return flat


def _coerce_value(s: str):
    """Coerce a string value to bool/int/float if possible."""
    low = s.lower().strip("'\"")
    if low in ("true", "yes", "1"):
        return True
    if low in ("false", "no", "0"):
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def compute_config_audit(
    config_path: str,
) -> tuple[list[DocResult], dict]:
    """Audit a Megatron-LM training config for packing boundary settings.

    Returns (per_doc_results, summary_dict).
    The per_doc list contains a single entry (the config file itself).
    """
    flat, fmt = _parse_config_file(config_path)

    params_result: dict[str, dict] = {}
    missing: list[str] = []
    invalid: list[str] = []

    for param_name, spec in _REQUIRED_PARAMS.items():
        found = False
        value = None
        for key in spec["keys"]:
            if key in flat:
                found = True
                value = flat[key]
                break

        valid = found and value == spec["expected"]
        params_result[param_name] = {
            "found": found,
            "value": value,
            "valid": valid,
            "description": spec["description"],
        }
        if not found:
            missing.append(param_name)
        elif not valid:
            invalid.append(param_name)

    config_valid = len(missing) == 0 and len(invalid) == 0

    doc_id = Path(config_path).name
    result = DocResult(
        doc_id=doc_id,
        scores=params_result,
        flags={
            "config_valid": config_valid,
            "has_missing_params": len(missing) > 0,
        },
    )

    summary = {
        "config_file": config_path,
        "config_format": fmt,
        "parameters": params_result,
        "config_valid": config_valid,
        "missing_params": missing,
        "invalid_params": invalid,
    }
    return [result], summary
