"""Exact-architecture 20M DataDecide proxy training on raw uint16 streams."""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file


CONTEXT_LENGTH = 2048
GLOBAL_BATCH_SIZE = 64
DEFAULT_STEPS = 14_584
DEFAULT_TOKENS = DEFAULT_STEPS * GLOBAL_BATCH_SIZE * CONTEXT_LENGTH
DEFAULT_LR = 8.4e-3


def official_20m_config(*, init_device: str = "cpu"):
    """Return the released 20M architecture, including affine RMSNorm weights."""
    try:
        from hf_olmo import OLMoConfig
    except ImportError as exc:  # pragma: no cover - depends on optional GPU environment
        raise RuntimeError("training requires ai2-olmo (the hf_olmo package)") from exc
    config = OLMoConfig(
        d_model=192,
        n_heads=8,
        n_layers=16,
        mlp_ratio=8,
        weight_tying=False,
        alibi=False,
        rope=True,
        flash_attention=False,
        attention_dropout=0.0,
        attention_layer_norm=False,
        include_bias=False,
        layer_norm_type="rms",
        layer_norm_with_affine=True,
        layer_norm_eps=1e-6,
        bias_for_layer_norm=False,
        attention_layer_norm_with_affine=False,
        activation_type="swiglu",
        residual_dropout=0.0,
        embedding_dropout=0.0,
        max_sequence_length=CONTEXT_LENGTH,
        vocab_size=50_280,
        embedding_size=50_304,
        eos_token_id=50_279,
        pad_token_id=1,
        init_device=init_device,
        init_fn="normal",
        init_std=0.02,
        init_cutoff_factor=3,
    )
    config.use_cache = False
    config._attn_implementation = "sdpa"
    return config


def parameter_counts(config=None) -> tuple[int, int]:
    """Return total and paper-style non-input-embedding parameter counts."""
    try:
        from hf_olmo import OLMoForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("training requires ai2-olmo (the hf_olmo package)") from exc
    model = OLMoForCausalLM(config or official_20m_config())
    total = sum(parameter.numel() for parameter in model.parameters())
    input_embedding = sum(
        parameter.numel() for parameter in model.model.transformer.wte.parameters()
    )
    return total, total - input_embedding


class ShuffledTokenBlocks:
    """Deterministic one-pass block order over a headerless uint16 token stream."""

    def __init__(self, path: str | Path, *, block_size: int, seed: int):
        self.path = Path(path).resolve()
        if self.path.stat().st_size % 2:
            raise ValueError("token stream is not uint16 aligned")
        self.tokens = np.memmap(self.path, mode="r", dtype="<u2")
        self.block_size = int(block_size)
        self.block_count = len(self.tokens) // self.block_size
        if self.block_count <= 0:
            raise ValueError("token stream is shorter than one context block")
        self.order = (
            np.random.default_rng(seed).permutation(self.block_count).astype(np.int32, copy=False)
        )

    def batch(self, start: int, count: int, device: torch.device) -> torch.Tensor:
        indices = self.order[start : start + count]
        if len(indices) != count:
            raise IndexError("token stream does not contain the requested batch")
        values = np.empty((count, self.block_size), dtype=np.int64)
        for row, block_index in enumerate(indices):
            offset = int(block_index) * self.block_size
            values[row] = self.tokens[offset : offset + self.block_size]
        return torch.from_numpy(values).to(device, non_blocking=True)


def _learning_rate(step: int, steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, steps - warmup_steps - 1)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * multiplier


def _save_hf_checkpoint(model, config, tokenizer_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config.architectures = ["OLMoForCausalLM"]
    config.save_pretrained(output_dir)
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.state_dict().items()
    }
    save_file(state, output_dir / "model.safetensors")
    shutil.copy2(tokenizer_path, output_dir / "tokenizer.json")
    tokenizer_config = {
        "eos_token": "<|endoftext|>",
        "pad_token": "<|padding|>",
        "model_max_length": CONTEXT_LENGTH,
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2) + "\n", encoding="utf-8"
    )


def export_trainer_state(
    *,
    trainer_state_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze a recoverable trainer state as an OLMES-compatible checkpoint."""
    try:
        from hf_olmo import OLMoForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("checkpoint export requires ai2-olmo") from exc
    trainer_state_path = Path(trainer_state_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    output_dir = Path(output_dir).resolve()
    state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
    step = int(state["step"])
    config = official_20m_config(init_device="cpu")
    model = OLMoForCausalLM(config)
    model.load_state_dict(state["model"])
    _save_hf_checkpoint(model, config, tokenizer_path, output_dir)
    summary = {
        "method": "datadecide_intermediate_checkpoint_export",
        "source": str(trainer_state_path),
        "step": step,
        "tokens": step * GLOBAL_BATCH_SIZE * CONTEXT_LENGTH,
        "checkpoint": str(output_dir),
    }
    (output_dir / "export-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def train_20m(
    *,
    tokens_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    seed: int,
    device: str,
    steps: int = DEFAULT_STEPS,
    global_batch_size: int = GLOBAL_BATCH_SIZE,
    micro_batch_size: int = 4,
    learning_rate: float = DEFAULT_LR,
    warmup_steps: int = 146,
    checkpoint_interval: int = 2_500,
    log_interval: int = 10,
    precision: str = "bf16",
    resume: bool = True,
) -> dict[str, Any]:
    """Train one proxy while preserving DataDecide's optimizer-step semantics."""
    try:
        from hf_olmo import OLMoForCausalLM
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("training requires ai2-olmo (the hf_olmo package)") from exc
    if global_batch_size % micro_batch_size:
        raise ValueError("global_batch_size must be divisible by micro_batch_size")
    if precision not in {"bf16", "float32"}:
        raise ValueError("precision must be bf16 or float32")
    if not 0 < warmup_steps < steps:
        raise ValueError("warmup_steps must be between zero and steps")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(tokenizer_path).resolve()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} is unavailable")

    config = official_20m_config(init_device="cpu")
    model = OLMoForCausalLM(config).to(resolved_device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    input_embedding_parameters = sum(
        parameter.numel() for parameter in model.model.transformer.wte.parameters()
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.05,
    )
    stream = ShuffledTokenBlocks(tokens_path, block_size=CONTEXT_LENGTH, seed=seed)
    required_blocks = steps * global_batch_size
    if stream.block_count < required_blocks:
        raise ValueError(
            f"stream has {stream.block_count} blocks, but training needs {required_blocks}"
        )

    trainer_state_path = output_dir / "trainer-state.pt"
    initial_step = 0
    if resume and trainer_state_path.is_file():
        state = torch.load(trainer_state_path, map_location=resolved_device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        initial_step = int(state["step"])

    log_path = output_dir / "train-log.jsonl"
    accumulation = global_batch_size // micro_batch_size
    started = time.perf_counter()
    model.train()
    with log_path.open("a", encoding="utf-8") as log_file:
        for step in range(initial_step, steps):
            lr = _learning_rate(step, steps, warmup_steps, learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for micro_step in range(accumulation):
                block_start = step * global_batch_size + micro_step * micro_batch_size
                input_ids = stream.batch(block_start, micro_batch_size, resolved_device)
                with torch.autocast(
                    device_type=resolved_device.type,
                    dtype=torch.bfloat16,
                    enabled=precision == "bf16" and resolved_device.type == "cuda",
                ):
                    loss = model(
                        input_ids=input_ids,
                        labels=input_ids,
                        use_cache=False,
                        return_dict=True,
                    ).loss
                    scaled_loss = loss / accumulation
                scaled_loss.backward()
                loss_sum += float(loss.detach().item())
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            completed = step + 1
            if completed % log_interval == 0 or completed == 1:
                row = {
                    "step": completed,
                    "tokens": completed * global_batch_size * CONTEXT_LENGTH,
                    "loss_nats": loss_sum / accumulation,
                    "learning_rate": lr,
                    "grad_norm": grad_norm,
                    "wall_seconds": time.perf_counter() - started,
                }
                log_file.write(json.dumps(row, allow_nan=False) + "\n")
                log_file.flush()
            if checkpoint_interval and completed % checkpoint_interval == 0:
                temporary = trainer_state_path.with_suffix(".tmp")
                torch.save(
                    {
                        "step": completed,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    temporary,
                )
                os.replace(temporary, trainer_state_path)

    final_dir = output_dir / "hf-final"
    _save_hf_checkpoint(model, config, tokenizer_path, final_dir)
    wall_seconds = time.perf_counter() - started
    summary = {
        "method": "datadecide_local_20m_benchmark_proxy",
        "seed": seed,
        "device": str(resolved_device),
        "precision": precision,
        "architecture": config.to_dict(),
        "total_parameters": total_parameters,
        "paper_style_parameters": total_parameters - input_embedding_parameters,
        "steps": steps,
        "context_length": CONTEXT_LENGTH,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation": accumulation,
        "tokens": steps * global_batch_size * CONTEXT_LENGTH,
        "tokens_per_total_parameter": steps * global_batch_size * CONTEXT_LENGTH / total_parameters,
        "tokens_per_paper_style_parameter": (
            steps
            * global_batch_size
            * CONTEXT_LENGTH
            / (total_parameters - input_embedding_parameters)
        ),
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.05,
        "wall_seconds": wall_seconds,
        "tokens_path": str(Path(tokens_path).resolve()),
        "tokenizer_path": str(tokenizer_path),
        "checkpoint": str(final_dir),
        "warning": "Local representative recipe sample; not the authors' exact shuffled token order.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
