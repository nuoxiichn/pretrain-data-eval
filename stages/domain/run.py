"""Stage 8 CLI entry point.

Usage:
  PYTHONPATH=. python stages/domain/run.py parsability --input <path> --dataset <name> [options]
  PYTHONPATH=. python stages/domain/run.py stem --input <path> --dataset <name> [options]
"""

from __future__ import annotations

import itertools
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
@click.option("--language", default=None, help="tree-sitter 语言（覆盖 yaml，默认 python）")
def parsability(input_path, dataset, config_path, output_base, output_dir,
                input_format, max_docs, language):
    """代码可解析率 / 语法错误严重度（tree-sitter）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    parse_cfg = cfg.get("parsability", {})

    lang = language or parse_cfg.get("language", "python")

    click.echo(f"[parsability] 读取 {input_path} ...")
    doc_iter = read_documents(input_path, config=input_cfg)
    if max_docs:
        docs = list(itertools.islice(doc_iter, max_docs))
        click.echo(f"[parsability] 仅扫描前 {max_docs} 条（--max-docs）")
    else:
        docs = list(doc_iter)
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
@click.option("--min-density", default=None, type=float, help="最小关键词密度阈值")
def stem(input_path, dataset, config_path, output_base, output_dir,
         input_format, max_docs, min_density):
    """STEM 学科分布 / 难度分层（关键词密度分类）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    stem_cfg = cfg.get("stem", {})

    density = min_density if min_density is not None else stem_cfg.get("min_keyword_density", 0.001)

    click.echo(f"[stem] 读取 {input_path} ...")
    doc_iter = read_documents(input_path, config=input_cfg)
    if max_docs:
        docs = list(itertools.islice(doc_iter, max_docs))
        click.echo(f"[stem] 仅扫描前 {max_docs} 条（--max-docs）")
    else:
        docs = list(doc_iter)
    click.echo(f"[stem] 共 {len(docs)} 文档，min_density={density}")

    out_dir = _resolve_output(output_dir, output_base, dataset, "stem")
    per_doc_path = out_dir / "per_doc.jsonl"

    t0 = time.time()
    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        _, summary = compute_stem(docs, min_density=density, on_doc=_write)
    elapsed = time.time() - t0
    summary["elapsed_seconds"] = round(elapsed, 1)

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[stem] STEM {summary['stem_docs']}/{summary['total_docs']}"
        f" ({summary['stem_pct']:.1%})"
    )
    click.echo(f"[stem] 学科分布 top: {summary.get('primary_subject_top10', {})}")
    click.echo(f"[stem] 耗时 {elapsed:.1f}s")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


if __name__ == "__main__":
    cli()
