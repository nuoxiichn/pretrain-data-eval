"""Stage 8 CLI entry point.

Usage:
  PYTHONPATH=. python stages/domain/run.py parsability --input <path> --dataset <name> [options]
  PYTHONPATH=. python stages/domain/run.py stem --input <path> --dataset <name> [options]
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import click
import yaml

from src.reader import read_documents
from src.sampling import DEFAULT_SAMPLE_MODE, DEFAULT_SEED, SAMPLE_MODES, sample_documents
from src.schema import DocResult, make_output_dir, use_output_dir, write_summary
from stages.domain.utils import compute_parsability, compute_stem


def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_output(output_dir: str | None, output_base: str, dataset: str, stage: str) -> Path:
    if output_dir:
        return use_output_dir(output_dir)
    return make_output_dir(output_base, stage, dataset)


@click.group()
def cli():
    """Stage 8: 专项能力"""


# ── parsability ──────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True, help="输入文件或目录路径")
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage8.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage8", show_default=True)
@click.option("--output-dir", default=None, help="覆盖自动生成的输出目录")
@click.option("--input-format", default=None, help="覆盖 yaml 中的 input.format")
@click.option("--max-docs", default=None, type=int, help="限制扫描文档数（调试用）")
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
@click.option("--language", default=None, help="tree-sitter 语言（覆盖 yaml，默认 python）")
def parsability(input_path, dataset, config_path, output_base, output_dir,
                input_format, max_docs, sample_mode, seed, language):
    """代码可解析率 / 语法错误严重度（tree-sitter）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    parse_cfg = cfg.get("parsability", {})

    lang = language or parse_cfg.get("language", "python")

    click.echo(f"[parsability] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[parsability] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")
    click.echo(f"[parsability] 共 {len(docs)} 文档，language={lang}")

    out_dir = _resolve_output(output_dir, output_base, dataset, "parsability")
    per_doc_path = out_dir / "per_doc.jsonl"

    t0 = time.time()
    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        _, summary = compute_parsability(docs, language=lang, on_doc=_write)
    elapsed = time.time() - t0
    summary["elapsed_seconds"] = round(elapsed, 1)

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[parsability] has_error {summary['has_error_docs']}/{summary['parsed_docs']}"
        f" ({summary['has_error_pct']:.1%}), unparsable={summary['unparsable_docs']}"
    )
    click.echo(f"[parsability] 耗时 {elapsed:.1f}s")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


# ── stem ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True, help="输入文件或目录路径")
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage8.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage8", show_default=True)
@click.option("--output-dir", default=None, help="覆盖自动生成的输出目录")
@click.option("--input-format", default=None, help="覆盖 yaml 中的 input.format")
@click.option("--max-docs", default=None, type=int, help="限制扫描文档数（调试用）")
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
@click.option("--model-path", default=None, help="EAI-Distill-0.5b 权重目录（覆盖 yaml）")
@click.option("--batch-size", default=None, type=int, help="推理 batch（覆盖 yaml）")
@click.option("--device", default=None, help="cuda / cpu（默认自动检测）")
def stem(input_path, dataset, config_path, output_base, output_dir,
         input_format, max_docs, sample_mode, seed,
         model_path, batch_size, device):
    """学科分类 / 难度分层（EAI-Distill-0.5b 推理）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    stem_cfg = cfg.get("stem", {})

    model_path = model_path or stem_cfg.get("model_path")
    if not model_path:
        raise click.UsageError("缺少 model_path（在 configs/stage8.yaml 或 --model-path 指定）")
    bsz = batch_size if batch_size is not None else stem_cfg.get("batch_size", 8)
    dev = device or stem_cfg.get("device")
    max_input_chars = stem_cfg.get("max_input_chars", 30000)
    max_new_tokens = stem_cfg.get("max_new_tokens", 100)
    hd_thr = stem_cfg.get("high_difficulty_threshold", 4)

    click.echo(f"[stem] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[stem] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")
    click.echo(f"[stem] 共 {len(docs)} 文档，model={model_path}，batch={bsz}")

    out_dir = _resolve_output(output_dir, output_base, dataset, "stem")
    per_doc_path = out_dir / "per_doc.jsonl"

    t0 = time.time()
    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        _, summary = compute_stem(
            docs,
            model_path=model_path,
            batch_size=bsz,
            max_input_chars=max_input_chars,
            max_new_tokens=max_new_tokens,
            device=dev,
            high_difficulty_threshold=hd_thr,
            on_doc=_write,
        )
    elapsed = time.time() - t0
    summary["elapsed_seconds"] = round(elapsed, 1)

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[stem] STEM {summary['stem_docs']}/{summary['total_docs']}"
        f" ({summary['stem_pct']:.1%}),"
        f" high_difficulty {summary['high_difficulty_docs']} ({summary['high_difficulty_pct']:.1%}),"
        f" parse_failed {summary['parse_failed_docs']} ({summary['parse_failed_pct']:.1%})"
    )
    top3 = sorted(
        summary["fdc_top_distribution"].items(),
        key=lambda kv: kv[1]["docs"], reverse=True,
    )[:3]
    click.echo(f"[stem] FDC top-3: {[(k, v['docs']) for k, v in top3]}")
    click.echo(f"[stem] 耗时 {elapsed:.1f}s（device={summary.get('device')}）")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


if __name__ == "__main__":
    cli()
