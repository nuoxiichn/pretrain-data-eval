"""Stage 11 CLI: tiny-proxy training diagnostics for pretraining corpora.

Usage:
  python stages/trainability/run.py probe \
    --corpus clean=data/clean.jsonl --corpus candidate=data/candidate.jsonl \
    --dataset comparison
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import click
import torch
import yaml

from pretrain_data_eval.sampling import DEFAULT_SAMPLE_MODE, SAMPLE_MODES
from pretrain_data_eval.schema import (
    DocResult,
    make_output_dir,
    use_output_dir,
    write_per_doc,
    write_summary,
)
from stages.trainability.data import (
    EOD_TOKEN,
    CorpusData,
    audit_document_overlaps,
    encode_corpora,
    load_corpora,
    load_pretokenized_corpora,
    load_shared_tokenizer,
    make_balanced_pool,
    parse_corpus_specs,
    train_shared_tokenizer,
)
from stages.trainability.metrics import (
    analyze_conditioning,
    analyze_scaling_gain,
    mean_document_scaling_scores,
)
from stages.trainability.model import (
    ModelConfig,
    TrainingConfig,
    model_parameter_counts,
    resolve_device,
    train_proxy,
)


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = {
        "sampling",
        "tokenizer",
        "model",
        "training",
        "evaluation",
        "runtime",
        "output",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"stage11 config is missing sections: {', '.join(missing)}")
    return config


def _config_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    for attempt in range(10):
        digest = hashlib.sha256()
        bytes_read = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                bytes_read += len(block)
                digest.update(block)
        if bytes_read or path.stat().st_size == 0:
            return digest.hexdigest()
        time.sleep(0.1 * (attempt + 1))
    raise OSError(f"non-empty file remained unreadable while hashing: {path}")


def _length_bucket(token_count: int) -> str:
    if token_count < 128:
        return "lt128"
    if token_count < 512:
        return "128_511"
    if token_count < 2048:
        return "512_2047"
    return "ge2048"


def _compatibility_gate(
    corpora: Sequence[CorpusData], config: dict, anchor_profile: str | None
) -> tuple[dict, dict[str, dict] | None]:
    """Gate ranking edges by declared language, domain, anchor and purpose."""
    if anchor_profile is None:
        return {"status": "not_run", "reason": "--anchor-profile not provided"}, None
    profiles = config.get("compatibility_gate", {}).get("profiles", {})
    if anchor_profile not in profiles:
        raise click.BadParameter(
            f"unknown compatibility profile {anchor_profile!r}",
            param_hint="--anchor-profile",
        )
    expected = profiles[anchor_profile]
    by_corpus: dict[str, dict] = {}
    for corpus in corpora:
        documents = corpus.encoded_validation or []
        languages = sorted({document.language or "unknown" for document in documents})
        domains = sorted({document.domain or "unknown" for document in documents})
        anchor_ids = sorted({document.anchor_id or "unknown" for document in documents})
        groups = sorted(
            {group for document in documents for group in document.comparison_groups}
        )
        reasons: list[str] = []
        if languages != [str(expected["language"])]:
            reasons.append("language_mismatch")
        if domains != [str(expected["domain"])]:
            reasons.append("domain_mismatch")
        if anchor_ids != [anchor_profile]:
            reasons.append("anchor_mismatch")
        if not groups:
            reasons.append("missing_comparison_purpose")
        by_corpus[corpus.name] = {
            "decision": "abstain" if reasons else "comparable",
            "reasons": reasons,
            "languages": languages,
            "domains": domains,
            "anchor_ids": anchor_ids,
            "comparison_groups": groups,
        }
    return {
        "status": "applied",
        "anchor_profile": anchor_profile,
        "expected_language": str(expected["language"]),
        "expected_domain": str(expected["domain"]),
        "by_corpus": by_corpus,
    }, by_corpus


def _input_eligibility_gate(
    parsed_specs: Sequence[tuple[str, Path]],
    config: dict,
    manifest_path: Path | None,
) -> dict:
    gate_config = config.get("input_eligibility", {})
    required = bool(gate_config.get("required", False))
    if manifest_path is None:
        if required:
            raise click.UsageError(
                "this protocol requires --eligibility-manifest before Balanced training"
            )
        return {"status": "not_required"}
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_protocol = config.get("validation", {}).get("protocol_version")
    if manifest.get("artifact_type") != "balanced_input_eligibility":
        raise click.ClickException("eligibility manifest has an invalid artifact_type")
    if expected_protocol and manifest.get("protocol_version") != expected_protocol:
        raise click.ClickException("eligibility manifest protocol version does not match config")
    records = manifest.get("corpora") or {}
    expected_names = {name for name, _ in parsed_specs}
    if set(records) != expected_names:
        raise click.ClickException(
            "eligibility manifest corpus names do not exactly match --corpus inputs"
        )
    allowed_methods = set(
        gate_config.get("allowed_repetition_methods", ["gopher_repetition_v1"])
    )
    checks: dict[str, dict] = {}
    failures: list[str] = []
    for name, path in parsed_specs:
        record = records[name]
        observed_sha256 = _file_sha256(path)
        repetition = record.get("repetition") or {}
        corpus_checks = {
            "input_sha256": record.get("input_sha256") == observed_sha256,
            "repetition_pass": repetition.get("status") == "pass",
            "repetition_method": repetition.get("method") in allowed_methods,
            "frozen_cluster_split": (
                (record.get("frozen_cluster_split") or {}).get("status") == "pass"
            ),
            "eligible_decision": record.get("decision") == "eligible",
        }
        checks[name] = corpus_checks
        if not all(corpus_checks.values()):
            failures.append(name)
    if failures:
        raise click.ClickException(
            "Balanced input eligibility failed for: " + ", ".join(failures)
        )
    return {
        "status": "applied",
        "decision": "eligible",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "allowed_repetition_methods": sorted(allowed_methods),
        "by_corpus": checks,
    }


def _parse_int_list(raw: str | None, default: Sequence[int]) -> list[int]:
    if raw is None:
        return [int(value) for value in default]
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise click.BadParameter("expected comma-separated integers") from exc
    if not values:
        raise click.BadParameter("list cannot be empty")
    return values


def _model_config(config: dict, section: str, vocab_size: int) -> ModelConfig:
    common = config["model"]
    values = common[section]
    return ModelConfig(
        context_length=int(common["context_length"]),
        vocab_size=vocab_size,
        d_model=int(values["d_model"]),
        n_layers=int(values["n_layers"]),
        n_heads=int(values["n_heads"]),
        ffn_multiplier=int(values.get("ffn_multiplier", 4)),
    )


def _training_config(config: dict) -> TrainingConfig:
    values = config["training"]
    return TrainingConfig(
        steps=int(values["steps"]),
        batch_size=int(values["batch_size"]),
        learning_rate=float(values["learning_rate"]),
        min_learning_rate_ratio=float(values.get("min_learning_rate_ratio", 0.1)),
        warmup_steps=int(values.get("warmup_steps", 0)),
        weight_decay=float(values.get("weight_decay", 0.0)),
        beta1=float(values.get("beta1", 0.9)),
        beta2=float(values.get("beta2", 0.95)),
        grad_clip=float(values.get("grad_clip", 1.0)),
    )


def _rescale_checkpoints(
    checkpoints: Sequence[int], old_steps: int, new_steps: int
) -> list[int]:
    """Preserve configured checkpoint fractions when the token budget changes."""
    scaled = {
        min(new_steps, max(1, round(checkpoint / old_steps * new_steps)))
        for checkpoint in checkpoints
    }
    scaled.add(new_steps)
    return sorted(scaled)


def _apply_tokens_per_parameter(
    training_config: TrainingConfig,
    checkpoints: Sequence[int],
    model_config: ModelConfig,
    requested_ratio: float | None,
) -> tuple[TrainingConfig, list[int], int, int]:
    """Optionally derive steps from the selected model's actual total parameter count."""
    total_parameters, non_embedding_parameters = model_parameter_counts(model_config)
    if requested_ratio is None:
        return training_config, list(checkpoints), total_parameters, non_embedding_parameters
    if not math.isfinite(requested_ratio) or requested_ratio <= 0:
        raise click.BadParameter(
            "must be a positive finite number", param_hint="--tokens-per-parameter"
        )
    tokens_per_step = training_config.batch_size * model_config.context_length
    new_steps = math.ceil(requested_ratio * total_parameters / tokens_per_step)
    new_checkpoints = _rescale_checkpoints(
        checkpoints, training_config.steps, new_steps
    )
    warmup_steps = min(
        new_steps - 1,
        round(training_config.warmup_steps / training_config.steps * new_steps),
    )
    return (
        replace(training_config, steps=new_steps, warmup_steps=warmup_steps),
        new_checkpoints,
        total_parameters,
        non_embedding_parameters,
    )


def _release_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _training_diagnostics(
    primary_runs: dict[str, dict[str, dict]], corpus_names: Sequence[str]
) -> dict:
    diagnostics: dict[str, dict] = {}
    for corpus in corpus_names:
        seed_runs = [runs[corpus] for runs in primary_runs.values()]
        checkpoints: list[dict] = []
        for curve_index, row in enumerate(seed_runs[0]["curve"]):
            step = row["step"]
            key = str(step)
            self_losses = [
                run["evaluations"][key][corpus]["mean_bits_per_token"] for run in seed_runs
            ]
            checkpoints.append(
                {
                    "step": step,
                    "tokens_seen": row["tokens_seen"],
                    "self_validation_bits_mean": sum(self_losses) / len(self_losses),
                    "self_validation_bits_seed_std": (
                        math.sqrt(
                            sum(
                                (value - sum(self_losses) / len(self_losses)) ** 2
                                for value in self_losses
                            )
                            / (len(self_losses) - 1)
                        )
                        if len(self_losses) > 1
                        else 0.0
                    ),
                    "train_loss_nats_mean": sum(
                        run["curve"][curve_index]["train_loss_nats_mean"] for run in seed_runs
                    )
                    / len(seed_runs),
                    "grad_norm_mean": sum(
                        run["curve"][curve_index]["grad_norm_mean"] for run in seed_runs
                    )
                    / len(seed_runs),
                    "grad_clip_fraction": sum(
                        run["curve"][curve_index]["grad_clip_fraction"] for run in seed_runs
                    )
                    / len(seed_runs),
                }
            )
        diagnostics[corpus] = {
            "checkpoints": checkpoints,
            "final_self_validation_bits": checkpoints[-1]["self_validation_bits_mean"],
            "observed_loss_drop_bits": checkpoints[0]["self_validation_bits_mean"]
            - checkpoints[-1]["self_validation_bits_mean"],
            "wall_seconds_mean": sum(run["wall_seconds"] for run in seed_runs) / len(seed_runs),
        }
    return diagnostics


def _make_per_doc_results(
    corpora: Sequence[CorpusData],
    primary_runs: dict[str, dict[str, dict]],
    scale_runs: dict[str, dict[str, dict]],
    final_checkpoint: int,
    context_length: int,
) -> list[DocResult]:
    results: list[DocResult] = []
    source_names = [corpus.name for corpus in corpora]
    checkpoint = str(final_checkpoint)
    for corpus in corpora:
        if corpus.encoded_validation is None:
            raise ValueError("corpus validation data is not encoded")
        scaling_scores: dict[str, list[float]] = {}
        if any(scale_runs.values()):
            small_nats, large_nats, gaps, token_counts = mean_document_scaling_scores(
                scale_runs, corpus.name, final_checkpoint
            )
            scaling_scores = {
                "scale_small_nats_per_token": small_nats,
                "scale_large_nats_per_token": large_nats,
                "scaling_gain_bits": gaps,
                "scaling_ppl_ratio": [2.0**gap for gap in gaps],
            }
        elif any(primary_runs.values()):
            first_seed_runs = next(iter(primary_runs.values()))
            first_source = source_names[0]
            token_counts = first_seed_runs[first_source]["evaluations"][checkpoint][
                corpus.name
            ]["doc_token_counts"]
        else:
            raise ValueError("at least one primary or scale run is required")
        primary_losses: dict[str, list[float]] = {}
        if any(primary_runs.values()):
            for source in source_names:
                losses_by_seed = [
                    runs[source]["evaluations"][checkpoint][corpus.name]["doc_loss_nats"]
                    for runs in primary_runs.values()
                ]
                primary_losses[source] = [
                    sum(values) / len(values) / math.log(2.0)
                    for values in zip(*losses_by_seed)
                ]

        for index, document in enumerate(corpus.encoded_validation):
            scores = {
                "corpus": corpus.name,
                "original_doc_id": document.doc_id,
                "evaluated_tokens": token_counts[index],
                "original_token_count": document.original_token_count,
                "language": document.language,
                "source": document.source,
                "length_bucket": _length_bucket(document.original_token_count),
                "dedup_cluster_id": document.dedup_cluster_id,
                "anchor_id": document.anchor_id,
                "domain": document.domain,
                "comparison_groups": list(document.comparison_groups),
                "primary_loss_bits_by_train_corpus": (
                    {source: primary_losses[source][index] for source in source_names}
                    if primary_losses
                    else {}
                ),
            }
            scores.update(
                {name: values[index] for name, values in scaling_scores.items()}
            )
            results.append(
                DocResult(
                    doc_id=f"{corpus.name}:{document.doc_id}",
                    scores=scores,
                    flags={"short_for_proxy": token_counts[index] < context_length},
                )
            )
    return results


@click.group()
def cli() -> None:
    """Stage 11: 基于微型训练代理的数据训练可用性评测。"""


@cli.command()
@click.option(
    "--corpus",
    "corpus_specs",
    multiple=True,
    required=True,
    help="重复传入 NAME=PATH；至少两批数据",
)
@click.option("--dataset", required=True, help="本次比较名称（用于输出目录）")
@click.option(
    "--config",
    "config_path",
    default="configs/stage11.yaml",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option("--output-base", default=None, help="覆盖 yaml 的 output.base_dir")
@click.option("--output-dir", default=None, type=click.Path(path_type=Path))
@click.option("--input-format", default=None, help="覆盖 input.format")
@click.option("--max-docs", default=None, type=int, help="覆盖每批最大抽样文档数")
@click.option("--sample-mode", type=click.Choice(SAMPLE_MODES), default=None)
@click.option("--seeds", default=None, help="覆盖训练 seeds，例如 17,29,41")
@click.option("--device", default=None, help="覆盖 runtime.device，例如 cuda:1 或 cpu")
@click.option(
    "--tokens-per-parameter",
    default=None,
    type=float,
    help="按所选模型的实际总参数量自动换算训练步数",
)
@click.option(
    "--tokenizer-path",
    default=None,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="复用包含 tokenizer.eod_token 的冻结 tokenizer；用于跨 run 可比",
)
@click.option("--scaling-only", is_flag=True, help="只训练 scale pair，跳过实验性的 conditioning")
@click.option("--conditioning-only", is_flag=True, help="只训练 conditioning 模型，跳过 scale pair")
@click.option(
    "--primary-model",
    default="primary",
    show_default=True,
    help="选择 model 下的模型配置，例如 conditioning_20m",
)
@click.option(
    "--scaling-train-corpus",
    default=None,
    help="让 scale pair 只在指定 corpus 上训练；默认使用各批等 token 混合",
)
@click.option(
    "--anchor-profile",
    default=None,
    help="启用 compatibility gate；值须在 compatibility_gate.profiles 中声明",
)
@click.option(
    "--eligibility-manifest",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="绑定输入哈希的重复/去重资格审计；v2 协议强制要求",
)
def probe(
    corpus_specs: tuple[str, ...],
    dataset: str,
    config_path: Path,
    output_base: str | None,
    output_dir: Path | None,
    input_format: str | None,
    max_docs: int | None,
    sample_mode: str | None,
    seeds: str | None,
    device: str | None,
    tokens_per_parameter: float | None,
    tokenizer_path: Path | None,
    scaling_only: bool,
    conditioning_only: bool,
    primary_model: str,
    scaling_train_corpus: str | None,
    anchor_profile: str | None,
    eligibility_manifest: Path | None,
) -> None:
    """Train tiny proxies and emit conditioning/scaling-gain evidence."""
    started = time.perf_counter()
    if scaling_only and conditioning_only:
        raise click.UsageError("--scaling-only and --conditioning-only are mutually exclusive")
    config_path = config_path.resolve()
    config = _load_config(config_path)
    sampling = config["sampling"]
    training_values = config["training"]
    evaluation = config["evaluation"]
    tokenizer_values = config["tokenizer"]
    selected_seeds = _parse_int_list(seeds, training_values["seeds"])
    checkpoints = _parse_int_list(None, training_values["checkpoints"])
    training_config = _training_config(config)
    if checkpoints[-1] != training_config.steps:
        raise click.ClickException("training.checkpoints must end at training.steps")

    requested_device = device or config["runtime"].get("device", "auto")
    torch_device = resolve_device(requested_device)
    runtime_precision = str(config["runtime"].get("precision", "float32"))
    if runtime_precision not in {"float32", "bf16"}:
        raise click.ClickException("runtime.precision must be 'float32' or 'bf16'")
    torch.set_num_threads(int(config["runtime"].get("torch_num_threads", 4)))
    deterministic = bool(config["runtime"].get("deterministic", False))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    parsed_specs = parse_corpus_specs(corpus_specs)
    input_eligibility = _input_eligibility_gate(
        parsed_specs, config, eligibility_manifest
    )

    input_config = dict(config.get("input", {}))
    if input_format:
        input_config["format"] = input_format
    source_format = input_config.get("format", "jsonl")
    selected_max_docs = (
        max_docs if max_docs is not None else int(sampling["max_docs_per_corpus"])
    )
    selected_sample_mode = sample_mode or sampling.get("mode", DEFAULT_SAMPLE_MODE)
    selected_output_base = output_base or config["output"]["base_dir"]
    out_dir = (
        use_output_dir(output_dir)
        if output_dir is not None
        else make_output_dir(selected_output_base, "probe", dataset)
    )
    command_path = out_dir / "command.json"
    command_path.write_text(
        json.dumps(
            {
                "argv": sys.argv,
                "cwd": str(Path.cwd()),
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_path": str(config_path),
                "config_sha256": _config_fingerprint(config_path),
                "corpora": [
                    {"name": name, "path": str(path)} for name, path in parsed_specs
                ],
                "device": str(torch_device),
                "eligibility_manifest": (
                    str(eligibility_manifest.resolve())
                    if eligibility_manifest is not None
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    eod_token = str(tokenizer_values.get("eod_token", EOD_TOKEN))
    if source_format in {"uint16_stream", "uint32_stream"} and tokenizer_path is None:
        raise click.UsageError("pretokenized input requires --tokenizer-path")

    corpora: list[CorpusData] | None = None
    if tokenizer_path is not None:
        tokenizer, tokenizer_summary = load_shared_tokenizer(
            tokenizer_path, out_dir / "tokenizer.json", eod_token=eod_token
        )
    elif source_format not in {"uint16_stream", "uint32_stream"}:
        corpora = load_corpora(
            parsed_specs,
            input_config,
            selected_max_docs,
            selected_sample_mode,
            int(sampling["seed"]),
            float(sampling["validation_fraction"]),
            int(sampling["split_seed"]),
        )
        tokenizer, tokenizer_summary = train_shared_tokenizer(
            corpora,
            int(tokenizer_values["vocab_size"]),
            int(tokenizer_values["min_frequency"]),
            int(tokenizer_values["max_chars_per_corpus"]),
            out_dir / "tokenizer.json",
        )
    else:
        raise AssertionError("unreachable tokenizer configuration")

    if source_format in {"uint16_stream", "uint32_stream"}:
        eod_id = tokenizer.token_to_id(eod_token)
        if eod_id is None:
            raise ValueError(f"tokenizer is missing {eod_token!r}")
        corpora = load_pretokenized_corpora(
            parsed_specs,
            dtype="uint16_le" if source_format == "uint16_stream" else "uint32_le",
            vocab_size=tokenizer.get_vocab_size(),
            eod_id=eod_id,
            max_train_tokens_per_corpus=int(
                training_values["max_train_tokens_per_corpus"]
            ),
            validation_reserve_tokens=int(
                input_config.get("validation_reserve_tokens", 2_000_000)
            ),
            max_validation_docs=selected_max_docs,
            max_eval_tokens_per_doc=int(evaluation["max_tokens_per_doc"]),
        )
    else:
        if corpora is None:
            corpora = load_corpora(
                parsed_specs,
                input_config,
                selected_max_docs,
                selected_sample_mode,
                int(sampling["seed"]),
                float(sampling["validation_fraction"]),
                int(sampling["split_seed"]),
            )
        encode_corpora(
            corpora,
            tokenizer,
            int(training_values["max_train_tokens_per_corpus"]),
            int(evaluation["max_tokens_per_doc"]),
        )
    if corpora is None:
        raise AssertionError("corpus loading did not initialize data")
    overlap_audit = audit_document_overlaps(corpora)
    if overlap_audit["global_train_validation_leaking_clusters"]:
        raise ValueError("dedup clusters cross train/validation; refusing to train")
    corpus_names = [corpus.name for corpus in corpora]
    evaluation_sets = {
        corpus.name: corpus.encoded_validation or [] for corpus in corpora
    }
    compatibility_summary, compatibility_by_corpus = _compatibility_gate(
        corpora, config, anchor_profile
    )
    document_strata = {
        corpus.name: [
            {
                "language": document.language or "unknown",
                "source": document.source or "unknown",
                "length_bucket": _length_bucket(document.original_token_count),
            }
            for document in (corpus.encoded_validation or [])
        ]
        for corpus in corpora
    }
    if primary_model not in config["model"] or not isinstance(
        config["model"][primary_model], dict
    ):
        raise click.BadParameter(
            f"unknown model profile {primary_model!r}", param_hint="--primary-model"
        )
    primary_config = _model_config(config, primary_model, tokenizer.get_vocab_size())
    (
        training_config,
        checkpoints,
        configured_parameter_count,
        configured_non_embedding_parameter_count,
    ) = _apply_tokens_per_parameter(
        training_config,
        checkpoints,
        primary_config,
        tokens_per_parameter,
    )
    small_config = (
        None
        if conditioning_only
        else _model_config(config, "scale_small", tokenizer.get_vocab_size())
    )
    scale_train_tokens = None
    scale_training_source: str | None = None
    balanced_pool_manifest: dict | None = None
    if not conditioning_only and scaling_train_corpus is None:
        pool_config = config.get("balanced_pool", {})
        scale_train_tokens, balanced_pool_manifest = make_balanced_pool(
            corpora,
            seed=int(pool_config.get("seed", sampling["seed"])),
            chunk_tokens=int(pool_config.get("chunk_tokens", 65536)),
            policy=str(pool_config.get("policy", "recipe_multiset")),
        )
        scale_training_source = "balanced_pool"
    elif not conditioning_only:
        matching = [corpus for corpus in corpora if corpus.name == scaling_train_corpus]
        if not matching:
            raise click.BadParameter(
                f"unknown scaling corpus {scaling_train_corpus!r}; expected one of {corpus_names}",
                param_hint="--scaling-train-corpus",
            )
        if matching[0].train_tokens is None:
            raise ValueError("scaling anchor train data is not encoded")
        scale_train_tokens = matching[0].train_tokens
        scale_training_source = matching[0].name

    balanced_pool_path: Path | None = None
    if balanced_pool_manifest is not None:
        balanced_pool_path = out_dir / "balanced_pool.json"
        balanced_pool_path.write_text(
            json.dumps(balanced_pool_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    scaling_checkpoint_mode = evaluation.get("scaling_checkpoints", "final")
    if scaling_checkpoint_mode == "all":
        scaling_checkpoints = checkpoints
    elif scaling_checkpoint_mode == "final":
        scaling_checkpoints = [training_config.steps]
    elif isinstance(scaling_checkpoint_mode, list):
        requested_scaling = {int(value) for value in scaling_checkpoint_mode}
        scaling_checkpoints = [value for value in checkpoints if value in requested_scaling]
        if not scaling_checkpoints or scaling_checkpoints[-1] != training_config.steps:
            raise click.ClickException(
                "evaluation.scaling_checkpoints must select valid checkpoints and include final step"
            )
    else:
        raise click.ClickException(
            "evaluation.scaling_checkpoints must be 'all', 'final', or a list"
        )

    primary_runs: dict[str, dict[str, dict]] = {}
    scale_runs: dict[str, dict[str, dict]] = {}
    for seed in selected_seeds:
        seed_key = str(seed)
        primary_runs[seed_key] = {}
        if not scaling_only:
            for corpus in corpora:
                if corpus.train_tokens is None:
                    raise ValueError("corpus train data is not encoded")
                click.echo(
                    f"[seed {seed}] primary train={corpus.name} "
                    f"tokens={len(corpus.train_tokens):,} device={torch_device}"
                )
                primary_runs[seed_key][corpus.name] = train_proxy(
                    corpus.train_tokens,
                    evaluation_sets,
                    primary_config,
                    training_config,
                    checkpoints,
                    int(evaluation["batch_size"]),
                    seed,
                    torch_device,
                    runtime_precision,
                )
                _release_device(torch_device)

        scale_runs[seed_key] = {}
        if not conditioning_only:
            if scale_train_tokens is None or small_config is None:
                raise ValueError("scale training configuration was not initialized")
            for label, model_config in (("small", small_config), ("large", primary_config)):
                click.echo(
                    f"[seed {seed}] scaling-{label} train={scale_training_source} "
                    f"tokens={len(scale_train_tokens):,} device={torch_device}"
                )
                scale_runs[seed_key][label] = train_proxy(
                    scale_train_tokens,
                    evaluation_sets,
                    model_config,
                    training_config,
                    scaling_checkpoints,
                    int(evaluation["batch_size"]),
                    seed + 1_000_000,
                    torch_device,
                    runtime_precision,
                )
                _release_device(torch_device)

    conditioning = (
        {"status": "not_run", "reason": "--scaling-only"}
        if scaling_only
        else analyze_conditioning(
            primary_runs,
            corpus_names,
            checkpoints,
            int(evaluation["bootstrap_samples"]),
            float(evaluation["confidence"]),
            float(evaluation["conditioning_min_effect_bits"]),
            float(evaluation["stable_horizon_fraction"]),
            int(sampling["split_seed"]) + 11,
        )
    )
    scaling_gain = (
        {"status": "not_run", "reason": "--conditioning-only"}
        if conditioning_only
        else analyze_scaling_gain(
            scale_runs,
            corpus_names,
            training_config.steps,
            int(evaluation["bootstrap_samples"]),
            float(evaluation["confidence"]),
            float(evaluation["scaling_min_effect_bits"]),
            int(sampling["split_seed"]) + 29,
            document_strata,
            compatibility_by_corpus,
            scaling_checkpoints,
            float(evaluation["stable_horizon_fraction"]),
            bool(evaluation.get("require_unanimous_seed_direction", False)),
            float(
                evaluation.get(
                    "fdr_max", config.get("validation", {}).get("fdr_max", 0.05)
                )
            ),
        )
    )
    per_doc = _make_per_doc_results(
        corpora,
        primary_runs,
        scale_runs,
        training_config.steps,
        primary_config.context_length,
    )
    per_doc_path = write_per_doc(per_doc, out_dir)

    total_wall_seconds = time.perf_counter() - started
    command_record = json.loads(command_path.read_text(encoding="utf-8"))
    command_record.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": total_wall_seconds,
            "exit_status": "completed",
        }
    )
    command_path.write_text(
        json.dumps(command_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    first_seed = str(selected_seeds[0])
    scale_reuse_ratio = (
        training_config.steps
        * training_config.batch_size
        * primary_config.context_length
        / len(scale_train_tokens)
        if scale_train_tokens is not None
        else None
    )
    warnings = []
    if len(selected_seeds) < 3:
        warnings.append(
            "fewer than three training seeds; confidence intervals do not establish seed stability"
        )
    if conditioning_only:
        warnings.append(
            "conditioning-only is a research calibration path and does not emit production rankings"
        )
    if scale_reuse_ratio is not None and scale_reuse_ratio > 2.0:
        warnings.append(
            "scale training samples more than two stream-equivalents; memorization risk is elevated"
        )
    if tokenizer_path is None:
        warnings.append("tokenizer was trained for this run; scores are not comparable across runs")
    if source_format in {"uint16_stream", "uint32_stream"}:
        warnings.append(
            "pretokenized streams lack source document IDs; train/validation are disjoint "
            "within each file, but overlap across nested corpus recipes cannot be audited"
        )
    if scale_training_source == "balanced_pool":
        warnings.append(
            "balanced-pool scaling gain is only calibrated for domain-matched corpus variants"
        )
    elif scale_training_source is not None:
        warnings.append(
            "anchor-trained scaling gain measures compatibility with the declared anchor distribution"
        )
    if compatibility_summary["status"] != "applied" and anchor_profile is not None:
        warnings.append("compatibility gate was requested but not applied")
    first_primary_run = (
        next(iter(primary_runs[first_seed].values())) if primary_runs[first_seed] else None
    )
    first_scale_runs = scale_runs[first_seed]
    primary_parameter_count = (
        first_primary_run["parameter_count"]
        if first_primary_run is not None
        else first_scale_runs["large"]["parameter_count"]
    )
    primary_non_embedding_parameter_count = (
        first_primary_run["non_embedding_parameter_count"]
        if first_primary_run is not None
        else first_scale_runs["large"]["non_embedding_parameter_count"]
    )
    if (
        primary_parameter_count != configured_parameter_count
        or primary_non_embedding_parameter_count
        != configured_non_embedding_parameter_count
    ):
        raise ValueError("runtime model parameter count differs from configured count")
    full_input_sha256 = bool(config["runtime"].get("full_input_sha256", False))
    model_runs = []
    for seed, runs in primary_runs.items():
        for train_corpus, run in runs.items():
            model_runs.append(
                {
                    "purpose": "conditioning",
                    "label": "primary",
                    "training_seed": int(seed),
                    "initialization_seed": run["seed"],
                    "train_corpus": train_corpus,
                    "parameter_count": run["parameter_count"],
                    "tokens_seen": training_config.steps
                    * training_config.batch_size
                    * primary_config.context_length,
                    "wall_seconds": run["wall_seconds"],
                    "peak_gpu_memory_bytes": run["peak_gpu_memory_bytes"],
                }
            )
    for seed, runs in scale_runs.items():
        for label, run in runs.items():
            model_runs.append(
                {
                    "purpose": "anchor_relative_scaling",
                    "label": label,
                    "training_seed": int(seed),
                    "initialization_seed": run["seed"],
                    "train_corpus": scale_training_source,
                    "parameter_count": run["parameter_count"],
                    "tokens_seen": training_config.steps
                    * training_config.batch_size
                    * primary_config.context_length,
                    "wall_seconds": run["wall_seconds"],
                    "peak_gpu_memory_bytes": run["peak_gpu_memory_bytes"],
                }
            )
    summary = {
        "stage": 11,
        "method": "tiny_proxy_trainability",
        "dataset": dataset,
        "config_path": str(config_path),
        "config_sha256": _config_fingerprint(config_path),
        "command_path": str(command_path),
        "command_sha256": _file_sha256(command_path),
        "device": str(torch_device),
        "corpora": [
            {
                "name": corpus.name,
                "path": str(corpus.path),
                "fingerprint_sha256": corpus.fingerprint,
                "file_sha256": (
                    _file_sha256(corpus.path)
                    if full_input_sha256 and corpus.path.is_file()
                    else None
                ),
                "file_bytes": corpus.path.stat().st_size if corpus.path.is_file() else None,
                "sampled_docs": corpus.sampled_docs,
                "train_docs": len(corpus.train_docs),
                "validation_docs": len(corpus.encoded_validation or []),
                "source_format": corpus.source_format,
                "validation_region_start": corpus.validation_region_start,
                "train_tokens": len(corpus.train_tokens) if corpus.train_tokens is not None else 0,
                "primary_token_reuse_ratio": (
                    training_config.steps
                    * training_config.batch_size
                    * primary_config.context_length
                    / len(corpus.train_tokens)
                    if corpus.train_tokens is not None
                    else None
                ),
                "validation_tokens": sum(
                    len(document.token_ids) - 1
                    for document in (corpus.encoded_validation or [])
                ),
            }
            for corpus in corpora
        ],
        "sampling": {
            "max_docs_per_corpus": selected_max_docs,
            "mode": selected_sample_mode,
            "seed": int(sampling["seed"]),
            "validation_fraction": float(sampling["validation_fraction"]),
            "split_seed": int(sampling["split_seed"]),
        },
        "leakage_audit": overlap_audit,
        "tokenizer": tokenizer_summary,
        "compatibility_gate": compatibility_summary,
        "input_eligibility": input_eligibility,
        "protocol": {
            "training_seeds": selected_seeds,
            "checkpoints": checkpoints,
            "primary_model_profile": primary_model,
            "primary_parameter_count": primary_parameter_count,
            "primary_non_embedding_parameter_count": (
                primary_non_embedding_parameter_count
            ),
            "scale_small_parameter_count": (
                first_scale_runs["small"]["parameter_count"] if first_scale_runs else None
            ),
            "scale_large_parameter_count": (
                first_scale_runs["large"]["parameter_count"] if first_scale_runs else None
            ),
            "scale_training_source": scale_training_source,
            "balanced_pool": balanced_pool_manifest,
            "scale_train_tokens": (
                len(scale_train_tokens) if scale_train_tokens is not None else None
            ),
            "scale_token_reuse_ratio": scale_reuse_ratio,
            "scaling_only": scaling_only,
            "conditioning_only": conditioning_only,
            "scaling_checkpoints": scaling_checkpoints,
            "tokens_seen_per_model": training_config.steps
            * training_config.batch_size
            * primary_config.context_length,
            "tokens_per_parameter": training_config.steps
            * training_config.batch_size
            * primary_config.context_length
            / primary_parameter_count,
            "tokens_per_non_embedding_parameter": training_config.steps
            * training_config.batch_size
            * primary_config.context_length
            / primary_non_embedding_parameter_count,
            "requested_tokens_per_parameter": tokens_per_parameter,
            "context_length": primary_config.context_length,
            "training": training_config.__dict__,
            "deterministic_algorithms": deterministic,
        },
        "conditioning": conditioning,
        "scaling_gain": scaling_gain,
        "training_diagnostics": (
            {} if scaling_only else _training_diagnostics(primary_runs, corpus_names)
        ),
        "runtime": {
            "total_wall_seconds": total_wall_seconds,
            "sum_model_wall_seconds": sum(
                run["wall_seconds"]
                for seed_runs in primary_runs.values()
                for run in seed_runs.values()
            )
            + sum(
                run["wall_seconds"]
                for seed_runs in scale_runs.values()
                for run in seed_runs.values()
            ),
            "trained_model_count": len(selected_seeds)
            * (2 if scaling_only else len(corpora) if conditioning_only else len(corpora) + 2),
            "torch_version": torch.__version__,
            "precision": runtime_precision,
            "device_name": (
                torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else "cpu"
            ),
            "model_runs": model_runs,
        },
        "interpretation": {
            "conditioning": "relative, optimizer- and horizon-dependent partial order",
            "scaling_gain": (
                "candidate-cohort-relative structural capacity gain; not a general quality score"
                if scale_training_source == "balanced_pool"
                else "relative to the specified Anchor: structural compatibility and capacity "
                "usability; not a general quality score"
            ),
            "not_claimed": [
                "downstream benchmark performance",
                "absolute or cross-run universal quality score",
                "factuality, safety, licensing or domain suitability",
            ],
            "warnings": warnings,
        },
    }
    summary_path = write_summary(summary, out_dir)
    artifact_paths = [
        summary_path,
        per_doc_path,
        out_dir / "tokenizer.json",
        command_path,
    ]
    if balanced_pool_path is not None:
        artifact_paths.append(balanced_pool_path)
    artifact_manifest = {
        "artifact_type": (
            "stage11_anchor_calibration_run_manifest"
            if anchor_profile is not None
            else "stage11_trainability_run_manifest"
        ),
        "dataset": dataset,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "leakage_audit": overlap_audit,
        "artifacts": {
            path.name: {
                "path": str(path),
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        },
        "inputs": {
            row["name"]: {
                "path": row["path"],
                "sha256": row["file_sha256"] or row["fingerprint_sha256"],
                "hash_scope": "full_file" if row["file_sha256"] else "sampled_content_fingerprint",
            }
            for row in summary["corpora"]
        },
        "config": {
            "path": str(config_path),
            "sha256": summary["config_sha256"],
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    click.echo(f"per-doc: {per_doc_path}")
    click.echo(f"summary: {summary_path}")
    click.echo(f"manifest: {manifest_path}")
    click.echo(json.dumps({"runtime_seconds": total_wall_seconds}, ensure_ascii=False))


if __name__ == "__main__":
    cli()
