"""Tiny causal language model training and evaluation primitives."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

from stages.trainability.data import (
    EncodedDocument,
    MemmapTokenStream,
    SegmentedTokenStream,
)


@dataclass(frozen=True)
class ModelConfig:
    context_length: int
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn_multiplier: int = 4

    def validate(self) -> None:
        if self.context_length < 8:
            raise ValueError("context_length must be at least 8")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.ffn_multiplier < 1:
            raise ValueError("ffn_multiplier must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    steps: int
    batch_size: int
    learning_rate: float
    min_learning_rate_ratio: float
    warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float

    def validate(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("training steps and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.min_learning_rate_ratio <= 1.0:
            raise ValueError("min_learning_rate_ratio must be in [0, 1]")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")


class TinyCausalLM(nn.Module):
    """Small decoder-style Transformer implemented with a causal encoder mask."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_multiplier * config.d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, config.n_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(config.context_length, config.context_length, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )
        self.apply(self._init_weights)
        self.output.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        length = input_ids.shape[1]
        if length > self.config.context_length:
            raise ValueError("input sequence exceeds configured context length")
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.transformer(
            hidden,
            mask=self.causal_mask[:length, :length],
            is_causal=True,
        )
        return self.output(self.final_norm(hidden))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def non_embedding_parameter_count(model: TinyCausalLM) -> int:
    """Count parameters excluding token and learned position embeddings."""
    embedding_parameters = {
        id(parameter)
        for module in (model.token_embedding, model.position_embedding)
        for parameter in module.parameters()
    }
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in embedding_parameters
    )


def model_parameter_counts(config: ModelConfig) -> tuple[int, int]:
    """Return total and non-embedding parameter counts without retaining a model."""
    model = TinyCausalLM(config)
    return parameter_count(model), non_embedding_parameter_count(model)


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` while keeping an explicit CPU fallback."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} requested but torch.cuda is unavailable")
    return device


def _make_eval_blocks(
    documents: Sequence[EncodedDocument], context_length: int
) -> list[tuple[int, list[int], list[int]]]:
    """Return ``(doc_index, inputs, labels)`` blocks without crossing documents."""
    blocks: list[tuple[int, list[int], list[int]]] = []
    for doc_index, document in enumerate(documents):
        eod_id = document.token_ids[0]
        content = document.token_ids[1:]
        for offset in range(0, len(content), context_length):
            labels = list(content[offset : offset + context_length])
            inputs = [eod_id, *labels[:-1]]
            blocks.append((doc_index, inputs, labels))
    return blocks


@torch.no_grad()
def evaluate_documents(
    model: TinyCausalLM,
    documents: Sequence[EncodedDocument],
    batch_size: int,
    device: torch.device,
    precision: str = "float32",
) -> dict:
    """Evaluate token and document weighted causal cross entropy."""
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    blocks = _make_eval_blocks(documents, model.config.context_length)
    loss_sums = [0.0] * len(documents)
    token_counts = [0] * len(documents)
    model.eval()

    for start in range(0, len(blocks), batch_size):
        group = blocks[start : start + batch_size]
        max_length = max(len(labels) for _, _, labels in group)
        input_ids = torch.zeros((len(group), max_length), dtype=torch.long)
        labels = torch.full((len(group), max_length), -100, dtype=torch.long)
        for row, (_, inputs, targets) in enumerate(group):
            input_ids[row, : len(inputs)] = torch.tensor(inputs)
            labels[row, : len(targets)] = torch.tensor(targets)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            logits = model(input_ids)
            token_losses = nn.functional.cross_entropy(
                logits.flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
                reduction="none",
            ).view(labels.shape)
        valid = labels.ne(-100)
        for row, (doc_index, _, _) in enumerate(group):
            count = int(valid[row].sum().item())
            loss_sums[doc_index] += float(token_losses[row].sum().item())
            token_counts[doc_index] += count

    document_losses = [
        loss_sum / count for loss_sum, count in zip(loss_sums, token_counts, strict=True)
    ]
    total_tokens = sum(token_counts)
    mean_nats = sum(loss_sums) / total_tokens
    return {
        "mean_nats_per_token": mean_nats,
        "mean_bits_per_token": mean_nats / math.log(2.0),
        "total_tokens": total_tokens,
        "doc_loss_nats": document_losses,
        "doc_token_counts": token_counts,
    }


def _learning_rate(step: int, config: TrainingConfig) -> float:
    if step <= config.warmup_steps and config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    ratio = config.min_learning_rate_ratio + (1.0 - config.min_learning_rate_ratio) * cosine
    return config.learning_rate * ratio


def train_proxy(
    train_tokens: torch.Tensor | MemmapTokenStream | SegmentedTokenStream,
    evaluation_sets: Mapping[str, Sequence[EncodedDocument]],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    checkpoints: Sequence[int],
    eval_batch_size: int,
    seed: int,
    device: torch.device,
    precision: str = "float32",
) -> dict:
    """Train one proxy and evaluate every target corpus at fixed horizons."""
    model_config.validate()
    training_config.validate()
    requested_checkpoints = sorted(set(int(value) for value in checkpoints))
    if not requested_checkpoints or requested_checkpoints[-1] != training_config.steps:
        raise ValueError("checkpoints must be non-empty and end at training.steps")
    if requested_checkpoints[0] <= 0 or requested_checkpoints[-1] > training_config.steps:
        raise ValueError("checkpoints must lie in [1, training.steps]")
    if len(train_tokens) <= model_config.context_length:
        raise ValueError("train stream is shorter than one context window")
    if precision not in {"float32", "bf16"}:
        raise ValueError("precision must be 'float32' or 'bf16'")
    if precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 proxy training currently requires a CUDA-compatible device")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyCausalLM(model_config).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device.index)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
        weight_decay=training_config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    positions = torch.arange(model_config.context_length + 1)
    start_time = time.perf_counter()
    interval_losses: list[float] = []
    interval_grad_norms: list[float] = []
    curve: list[dict] = []
    evaluations: dict[str, dict] = {}

    for step in range(1, training_config.steps + 1):
        offsets = torch.randint(
            0,
            len(train_tokens) - model_config.context_length,
            (training_config.batch_size,),
            generator=generator,
        )
        batch = train_tokens[offsets[:, None] + positions[None, :]].to(
            device=device, dtype=torch.long
        )
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            logits = model(input_ids)
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
        lr = _learning_rate(step, training_config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        interval_losses.append(float(loss.detach().item()))
        interval_grad_norms.append(float(grad_norm.detach().item()))

        if step in requested_checkpoints:
            checkpoint_eval = {
                name: evaluate_documents(
                    model, documents, eval_batch_size, device, precision=precision
                )
                for name, documents in evaluation_sets.items()
            }
            evaluations[str(step)] = checkpoint_eval
            curve.append(
                {
                    "step": step,
                    "tokens_seen": step
                    * training_config.batch_size
                    * model_config.context_length,
                    "train_loss_nats_mean": sum(interval_losses) / len(interval_losses),
                    "train_loss_nats_last": interval_losses[-1],
                    "grad_norm_mean": sum(interval_grad_norms) / len(interval_grad_norms),
                    "grad_norm_max": max(interval_grad_norms),
                    "grad_clip_fraction": sum(
                        value > training_config.grad_clip for value in interval_grad_norms
                    )
                    / len(interval_grad_norms),
                    "learning_rate": lr,
                }
            )
            interval_losses.clear()
            interval_grad_norms.clear()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - start_time
    return {
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "parameter_count": parameter_count(model),
        "non_embedding_parameter_count": non_embedding_parameter_count(model),
        "precision": precision,
        "seed": seed,
        "wall_seconds": wall_seconds,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device.index) if device.type == "cuda" else None
        ),
        "curve": curve,
        "evaluations": evaluations,
    }
