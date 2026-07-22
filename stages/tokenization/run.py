"""Stage 10 CLI entry point.

Usage:
  PYTHONPATH=. python stages/tokenization/run.py tokenize --input <path> --dataset <name> [options]
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

from pretrain_data_eval.reader import read_documents
from pretrain_data_eval.sampling import DEFAULT_SAMPLE_MODE, DEFAULT_SEED, SAMPLE_MODES, sample_documents
from pretrain_data_eval.schema import DocResult, make_output_dir, use_output_dir, write_summary
from stages.tokenization.utils import compute_tokenization


def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_output(output_dir: str | None, output_base: str, dataset: str, stage: str) -> Path:
    if output_dir:
        return use_output_dir(output_dir)
    return make_output_dir(output_base, stage, dataset)


@click.group()
def cli():
    """Stage 10: Tokenization 分析"""


@cli.command()
@click.option("--input", "input_path", required=True, help="输入文件或目录路径")
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage10.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage10", show_default=True)
@click.option("--output-dir", default=None, help="覆盖自动生成的输出目录")
@click.option("--input-format", default=None, help="覆盖 yaml 中的 input.format")
@click.option("--max-docs", default=None, type=int, help="限制扫描文档数（调试用）")
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
@click.option("--tokenizer", "tokenizer_path", default=None, help="覆盖 yaml 中的 tokenizer 路径")
@click.option("--unk-threshold", default=None, type=float, help="UNK 率阈值")
@click.option("--fertility-threshold", default=None, type=float, help="fertility 阈值")
@click.option("--batch-size", default=None, type=int, help="encode_batch 批大小")
def tokenize(
    input_path, dataset, config_path, output_base, output_dir,
    input_format, max_docs, sample_mode, seed, tokenizer_path, unk_threshold,
    fertility_threshold, batch_size,
):
    """Token/char 比 / UNK 率 / 语种 fertility / 代码公式 token 膨胀率"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    tok_cfg = cfg.get("tokenize", {})

    tok_path = tokenizer_path or tok_cfg.get("tokenizer_path")
    if not tok_path:
        raise click.UsageError(
            "需要 tokenizer 路径。在 configs/stage10.yaml 的 tokenize 段设置 tokenizer_path，"
            "或通过 --tokenizer 指定。"
        )

    unk_th = unk_threshold if unk_threshold is not None else tok_cfg.get("unk_threshold", 0.01)
    fert_th = fertility_threshold if fertility_threshold is not None else tok_cfg.get("fertility_threshold", 5.0)
    bs = batch_size or tok_cfg.get("batch_size", 256)

    click.echo(f"[tokenize] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[tokenize] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")
    click.echo(f"[tokenize] 共 {len(docs)} 文档，tokenizer={tok_path}")

    out_dir = _resolve_output(output_dir, output_base, dataset, "tokenize")
    per_doc_path = out_dir / "per_doc.jsonl"

    t0 = time.time()
    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        _, summary = compute_tokenization(
            docs,
            tokenizer_path=tok_path,
            unk_threshold=unk_th,
            fertility_threshold=fert_th,
            batch_size=bs,
            on_doc=_write,
        )
    elapsed = time.time() - t0
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary["docs_per_second"] = round(len(docs) / elapsed, 1) if elapsed > 0 else 0

    sm_path = write_summary(summary, out_dir)
    fstats = summary.get("fertility_stats", {})
    click.echo(
        f"[tokenize] fertility mean={fstats.get('mean', 'N/A')}, "
        f"UNK 率={summary['unk_stats']['overall_unk_rate']:.4%}, "
        f"high_unk={summary['high_unk_rate_docs']}, high_fert={summary['high_fertility_docs']}"
    )
    click.echo(f"[tokenize] 耗时 {elapsed:.1f}s（{summary['docs_per_second']:.1f} docs/s）")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


if __name__ == "__main__":
    cli()
