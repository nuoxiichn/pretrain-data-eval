"""Stage 5 CLI entry point — 污染检测.

Usage:
  python stages/contamination/run.py exact      --input <path> --dataset <name> [options]
  python stages/contamination/run.py code-near  --input <path> --dataset <name> [options]
  python stages/contamination/run.py code-ast   --input <path> --dataset <name> [options]
"""

from __future__ import annotations

import json
import sys
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
from stages.contamination.benchmarks import load_benchmarks
from stages.contamination.utils import (
    compute_code_ast,
    compute_code_near,
    compute_exact_contamination,
)


def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_output(output_dir: str | None, output_base: str, dataset: str, stage: str) -> Path:
    if output_dir:
        return use_output_dir(output_dir)
    return make_output_dir(output_base, stage, dataset)


# ── CLI group ────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Stage 5: 污染检测"""


# ── exact ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True, help="输入文件或目录路径")
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage5.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage5", show_default=True)
@click.option("--output-dir", default=None, help="覆盖自动生成的输出目录")
@click.option("--input-format", default=None, help="覆盖 yaml 中的 input.format")
@click.option("--max-docs", default=None, type=int, help="限制扫描文档数（调试用）")
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
def exact(input_path, dataset, config_path, output_base, output_dir,
          input_format, max_docs, sample_mode, seed):
    """精确污染检测（文档级 + 段落级 MD5 哈希对比 benchmark）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    exact_cfg = cfg.get("exact", {})
    bench_cfg = cfg.get("benchmarks", {})

    click.echo("[exact] 加载 benchmark 索引 ...")
    bench_items = load_benchmarks(bench_cfg)
    total_bench = sum(len(v) for v in bench_items.values())
    click.echo(f"[exact] 已加载 {len(bench_items)} 个 benchmark，共 {total_bench} 条样本")

    click.echo(f"[exact] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[exact] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    out_dir = _resolve_output(output_dir, output_base, dataset, "exact")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        _, summary = compute_exact_contamination(
            docs,
            bench_items,
            paragraph_sep=exact_cfg.get("paragraph_sep", "\n\n"),
            min_para_chars=exact_cfg.get("min_para_chars", 50),
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[exact] 精确污染 {summary['contaminated_docs']} / {summary['total_docs']} 条"
        f" ({summary['contaminated_pct']:.1%})"
        f"  段落污染 {summary['para_contaminated_docs']} ({summary['para_contaminated_pct']:.1%})"
    )
    if summary["per_benchmark_hits"]:
        click.echo(f"[exact] 各 benchmark 命中: {summary['per_benchmark_hits']}")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


# ── code-near ────────────────────────────────────────────────────────────────

@cli.command("code-near")
@click.option("--input", "input_path", required=True)
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage5.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage5", show_default=True)
@click.option("--output-dir", default=None)
@click.option("--input-format", default=None)
@click.option("--max-docs", default=None, type=int)
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
@click.option("--ngram-size", default=None, type=int)
@click.option("--jaccard-threshold", default=None, type=float)
def code_near(input_path, dataset, config_path, output_base, output_dir,
              input_format, max_docs, sample_mode, seed, ngram_size, jaccard_threshold):
    """代码 benchmark 近重复检测（字符级 MinHash + LSH）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    cn_cfg = cfg.get("code_near", {})
    bench_cfg = cfg.get("benchmarks", {})

    click.echo("[code-near] 加载 benchmark ...")
    bench_items = load_benchmarks(bench_cfg)

    click.echo(f"[code-near] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[code-near] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    _ngram = ngram_size or cn_cfg.get("ngram_size", 5)
    _threshold = jaccard_threshold if jaccard_threshold is not None else cn_cfg.get("jaccard_threshold", 0.85)
    _num_hashes = cn_cfg.get("num_hashes", 128)
    _num_bands = cn_cfg.get("num_bands", 16)
    _band_size = cn_cfg.get("band_size", 8)

    click.echo(
        f"[code-near] ngram={_ngram}, threshold={_threshold}, "
        f"hashes={_num_hashes}, bands={_num_bands}×{_band_size}"
    )

    out_dir = _resolve_output(output_dir, output_base, dataset, "code_near")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        _, summary = compute_code_near(
            docs,
            bench_items,
            ngram_size=_ngram,
            jaccard_threshold=_threshold,
            num_hashes=_num_hashes,
            num_bands=_num_bands,
            band_size=_band_size,
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[code-near] 近重复 {summary['code_near_dup_docs']} / {summary['total_docs']} 条"
        f" ({summary['code_near_dup_pct']:.1%})"
        f"  benchmark 代码样本={summary['bench_code_samples']}"
    )
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


# ── code-ast ─────────────────────────────────────────────────────────────────

@cli.command("code-ast")
@click.option("--input", "input_path", required=True)
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage5.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage5", show_default=True)
@click.option("--output-dir", default=None)
@click.option("--input-format", default=None)
@click.option("--max-docs", default=None, type=int)
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
def code_ast(input_path, dataset, config_path, output_base, output_dir,
             input_format, max_docs, sample_mode, seed):
    """代码 AST 结构污染检测（tree-sitter 指纹）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    ast_cfg = cfg.get("code_ast", {})
    bench_cfg = cfg.get("benchmarks", {})

    click.echo("[code-ast] 加载 benchmark ...")
    bench_items = load_benchmarks(bench_cfg)

    click.echo(f"[code-ast] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[code-ast] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    _languages = ast_cfg.get("languages", ["python"])
    _threshold = ast_cfg.get("fingerprint_jaccard_threshold", 0.90)

    click.echo(f"[code-ast] languages={_languages}, threshold={_threshold}")

    out_dir = _resolve_output(output_dir, output_base, dataset, "code_ast")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        _, summary = compute_code_ast(
            docs,
            bench_items,
            languages=_languages,
            fingerprint_jaccard_threshold=_threshold,
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[code-ast] AST 污染 {summary['ast_contaminated_docs']} / {summary['total_docs']} 条"
        f" ({summary['ast_contaminated_pct']:.1%})"
        f"  指纹精确命中={summary['fingerprint_exact_hits']}"
    )
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


if __name__ == "__main__":
    cli()
