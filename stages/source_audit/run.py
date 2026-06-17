"""Stage 1 CLI entry point.

Usage:
  python stages/source_audit/run.py stats   --input <path> --dataset <name> [options]
  python stages/source_audit/run.py license --input <path> --dataset <name> [options]
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

# Make project root importable regardless of working directory.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import click
import yaml

from src.reader import read_documents
from src.sampling import DEFAULT_SAMPLE_MODE, DEFAULT_SEED, SAMPLE_MODES, sample_documents
from src.schema import DocResult, make_output_dir, use_output_dir, write_per_doc, write_summary
from stages.source_audit.utils import (
    compute_doc_stats,
    compute_license,
    get_length_bucket,
    make_tokenizer,
)


def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_output(output_dir: str | None, output_base: str, dataset: str, stage: str) -> Path:
    if output_dir:
        return use_output_dir(output_dir)
    return make_output_dir(output_base, stage, dataset)


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Stage 1: 来源审计 + 时间属性"""


# ── stats ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True, help="输入文件或目录路径")
@click.option("--dataset", required=True, help="数据集名称（用于输出目录命名）")
@click.option("--config", "config_path", default="configs/stage1.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage1", show_default=True)
@click.option("--output-dir", default=None, help="覆盖自动生成的输出目录")
@click.option("--input-format", default=None, help="覆盖 yaml 中的 input.format（auto/jsonl/parquet）")
def stats(input_path: str, dataset: str, config_path: str, output_base: str,
          output_dir: str | None, input_format: str | None):
    """行 1：文档统计 + 时间字段分析"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    tok_cfg = cfg.get("tokenizer", {})
    tokenize = make_tokenizer(
        backend=tok_cfg.get("backend", "words"),
        hf_name=tok_cfg.get("hf_name"),
    )

    click.echo(f"[stats] 读取 {input_path} ...")
    docs = read_documents(input_path, config=input_cfg)

    out_dir = _resolve_output(output_dir, output_base, dataset, "stats")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        _, summary = compute_doc_stats(docs, tokenize, on_doc=_write)

    sm_path = write_summary(summary, out_dir)

    click.echo(f"[stats] {summary['total_docs']} 条文档")
    click.echo(f"  token 总量: {summary['token_stats'].get('total', 'N/A'):,}")
    click.echo(f"  时间字段存在率: {summary['timestamp']['present_pct']:.1%}")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


# ── license ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True, help="输入 JSONL 路径")
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage1.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage1", show_default=True)
@click.option("--output-dir", default=None)
@click.option("--max-docs", default=None, type=int, help="限制扫描文档数（调试用）")
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
@click.option("--input-format", default=None, help="覆盖 yaml 中的 input.format（auto/jsonl/parquet）")
def license(
    input_path: str,
    dataset: str,
    config_path: str,
    output_base: str,
    output_dir: str | None,
    max_docs: int | None,
    sample_mode: str,
    seed: int,
    input_format: str | None,
):
    """行 2：ScanCode 许可证与版权检测"""
    cfg = _load_config(config_path)
    lic_cfg = cfg.get("license", {})
    timeout = lic_cfg.get("timeout_per_doc", 30)
    if max_docs is None:
        max_docs = lic_cfg.get("max_docs")

    click.echo(f"[license] 读取 {input_path} ...")
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[license] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    per_doc, summary = compute_license(docs, timeout_per_doc=timeout, max_docs=None)

    out_dir = _resolve_output(output_dir, output_base, dataset, "license")
    pd_path = write_per_doc(per_doc, out_dir)
    sm_path = write_summary(summary, out_dir)

    click.echo(f"[license] 扫描 {summary['total_docs_scanned']} 条，命中 {summary['docs_with_license']} 条 ({summary['hit_pct']:.1%})")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {pd_path}")


if __name__ == "__main__":
    cli()
