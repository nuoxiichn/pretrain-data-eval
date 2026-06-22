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
    DocStatsAggregator,
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
@click.option("--coalesce-stage10/--no-coalesce-stage10", default=False,
              help="一次 tokenizer 扫描同时产出 stage10 tokenize summary（需 hf backend）")
@click.option("--stage10-config", default="configs/stage10.yaml", show_default=True,
              help="--coalesce-stage10 模式下读取 tokenize.* 段（tokenizer_path/thresholds/batch_size）")
@click.option("--stage10-output-base", default="outputs/stage10", show_default=True)
@click.option("--stage10-output-dir", default=None,
              help="--coalesce-stage10 模式下 stage10 产物目录；不指定则按 stage10 自动生成")
def stats(input_path: str, dataset: str, config_path: str, output_base: str,
          output_dir: str | None, input_format: str | None,
          coalesce_stage10: bool, stage10_config: str,
          stage10_output_base: str, stage10_output_dir: str | None):
    """行 1：文档统计 + 时间字段分析"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    tok_cfg = cfg.get("tokenizer", {})
    stats_cfg = cfg.get("stats", {})
    percentiles = tuple(stats_cfg.get("percentiles", (25, 50, 75, 90, 95, 99)))

    click.echo(f"[stats] 读取 {input_path} ...")
    docs = read_documents(input_path, config=input_cfg)

    out_dir = _resolve_output(output_dir, output_base, dataset, "stats")
    per_doc_path = out_dir / "per_doc.jsonl"

    if coalesce_stage10:
        from stages.tokenization.utils import compute_tokenization

        s10_cfg = _load_config(stage10_config).get("tokenize", {})
        tok_path = s10_cfg.get("tokenizer_path") or tok_cfg.get("path")
        if not tok_path:
            raise click.UsageError(
                "--coalesce-stage10 需要 tokenizer 路径："
                "在 configs/stage10.yaml tokenize.tokenizer_path 或 configs/stage1.yaml tokenizer.path 中设置"
            )
        unk_th = s10_cfg.get("unk_threshold", 0.01)
        fert_th = s10_cfg.get("fertility_threshold", 5.0)
        bs = s10_cfg.get("batch_size", 256)

        s10_out = _resolve_output(stage10_output_dir, stage10_output_base, dataset, "tokenize")
        s10_per_doc_path = s10_out / "per_doc.jsonl"

        agg = DocStatsAggregator(percentiles=percentiles)
        click.echo(f"[stats] coalesce 模式：一次扫描产 stage1 + stage10，tokenizer={tok_path}")

        with per_doc_path.open("w", encoding="utf-8") as f1, \
             s10_per_doc_path.open("w", encoding="utf-8") as f10:
            def _s1_hook(doc, n_tokens, n_chars):
                r = agg.add(doc, n_tokens)
                f1.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

            def _s10_write(r: DocResult) -> None:
                f10.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

            _, s10_summary = compute_tokenization(
                docs,
                tokenizer_path=tok_path,
                unk_threshold=unk_th,
                fertility_threshold=fert_th,
                batch_size=bs,
                on_doc=_s10_write,
                extra_per_doc=_s1_hook,
            )

        s1_summary = agg.finalize()
        sm_path = write_summary(s1_summary, out_dir)
        s10_sm_path = write_summary(s10_summary, s10_out)

        click.echo(f"[stats] {s1_summary['total_docs']} 条文档")
        click.echo(f"  token 总量: {s1_summary['token_stats'].get('total', 'N/A'):,}")
        click.echo(f"  时间字段存在率: {s1_summary['timestamp']['present_pct']:.1%}")
        click.echo(f"  stage1 -> {sm_path}")
        click.echo(f"  stage1 -> {per_doc_path}")
        click.echo(f"  stage10 -> {s10_sm_path}")
        click.echo(f"  stage10 -> {s10_per_doc_path}")
        return

    # standalone 模式：原行为不变
    tokenize = make_tokenizer(
        backend=tok_cfg.get("backend", "words"),
        path=tok_cfg.get("path") or tok_cfg.get("hf_name"),  # 向后兼容旧字段名
    )

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        _, summary = compute_doc_stats(docs, tokenize, on_doc=_write, percentiles=percentiles)

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
