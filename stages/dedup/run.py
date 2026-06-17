"""Stage 4 CLI entry point.

Usage:
  python dedup/run.py exact     --input <path> --dataset <name> [options]
  python dedup/run.py minhash   --input <path> --dataset <name> [options]
  python dedup/run.py ngram     --input <path> --dataset <name> [options]
  python dedup/run.py semdedup  --input <path> --dataset <name> [options]
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
from src.schema import DocResult, make_output_dir, use_output_dir, write_per_doc, write_summary
from stages.dedup.utils import (
    compute_exact_dedup,
    compute_minhash_dedup,
    compute_ngram_dedup,
    compute_semdedup,
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
    """Stage 4: 重复度分析"""


# ── exact ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True, help="输入文件或目录路径")
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage4.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage4", show_default=True)
@click.option("--output-dir", default=None, help="覆盖自动生成的输出目录")
@click.option("--input-format", default=None, help="覆盖 yaml 中的 input.format")
@click.option("--max-docs", default=None, type=int, help="限制扫描文档数（调试用）")
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True, help="抽样策略（注：dedup 用 random 会破坏单文件重复统计语义；调试可用）")
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
@click.option("--paragraph-sep", default=None, help="覆盖 yaml 中的 exact.paragraph_sep")
@click.option("--min-para-chars", default=None, type=int, help="覆盖 yaml 中的 exact.min_para_chars")
def exact(input_path, dataset, config_path, output_base, output_dir,
          input_format, max_docs, sample_mode, seed, paragraph_sep, min_para_chars):
    """MD5 精确去重（文档级 + 段落级）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    exact_cfg = cfg.get("exact", {})

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

        _, summary = compute_exact_dedup(
            docs,
            paragraph_sep=paragraph_sep or exact_cfg.get("paragraph_sep", "\n\n"),
            min_para_chars=min_para_chars or exact_cfg.get("min_para_chars", 50),
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[exact] 精确重复 {summary['exact_dup_docs']} / {summary['total_docs']} 条"
        f" ({summary['exact_dup_pct']:.1%})"
        f"  段落重复文档 {summary['para_dup_docs']} ({summary['para_dup_pct']:.1%})"
    )
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


# ── minhash ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True)
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage4.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage4", show_default=True)
@click.option("--output-dir", default=None)
@click.option("--input-format", default=None)
@click.option("--max-docs", default=None, type=int)
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
@click.option("--num-hashes", default=None, type=int)
@click.option("--ngram-size", default=None, type=int)
@click.option("--jaccard-threshold", default=None, type=float)
@click.option("--num-bands", default=None, type=int)
@click.option("--band-size", default=None, type=int)
def minhash(input_path, dataset, config_path, output_base, output_dir,
            input_format, max_docs, sample_mode, seed, num_hashes, ngram_size,
            jaccard_threshold, num_bands, band_size):
    """MinHash + LSH 近重复检测（word n-gram）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    mh_cfg = cfg.get("minhash", {})

    click.echo(f"[minhash] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[minhash] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    _num_hashes = num_hashes or mh_cfg.get("num_hashes", 112)
    _ngram_size = ngram_size or mh_cfg.get("ngram_size", 5)
    _jaccard_threshold = jaccard_threshold or mh_cfg.get("jaccard_threshold", 0.72)
    _num_bands = num_bands or mh_cfg.get("num_bands", 14)
    _band_size = band_size or mh_cfg.get("band_size", 8)

    if _num_hashes != _num_bands * _band_size:
        click.echo(
            f"[minhash] 警告: num_hashes={_num_hashes} != num_bands*band_size="
            f"{_num_bands}*{_band_size}={_num_bands*_band_size}",
            err=True,
        )

    click.echo(
        f"[minhash] num_hashes={_num_hashes}, ngram={_ngram_size}, "
        f"threshold={_jaccard_threshold}, bands={_num_bands}×{_band_size}"
    )

    out_dir = _resolve_output(output_dir, output_base, dataset, "minhash")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        _, summary = compute_minhash_dedup(
            docs,
            num_hashes=_num_hashes,
            ngram_size=_ngram_size,
            jaccard_threshold=_jaccard_threshold,
            num_bands=_num_bands,
            band_size=_band_size,
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[minhash] 近重复 {summary['near_dup_docs']} / {summary['total_docs']} 条"
        f" ({summary['near_dup_pct']:.1%})"
        f"  near_dup_pairs={summary['near_dup_pairs']}"
    )
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


# ── ngram ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True)
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage4.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage4", show_default=True)
@click.option("--output-dir", default=None)
@click.option("--input-format", default=None)
@click.option("--max-docs", default=None, type=int)
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
@click.option("--ngram-size", default=None, type=int, help="词级 n-gram 大小，默认 13")
@click.option("--overlap-threshold", default=None, type=float,
              help="段落 shingle 重叠率阈值，默认 0.5")
def ngram(input_path, dataset, config_path, output_base, output_dir,
          input_format, max_docs, sample_mode, seed, ngram_size, overlap_threshold):
    """段落级 N-gram 重复检测（Dolma-style 审计，无需 dolma 二进制）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    ng_cfg = cfg.get("ngram", {})
    exact_cfg = cfg.get("exact", {})

    _ngram_size = ngram_size or ng_cfg.get("ngram_size", 13)
    _overlap_threshold = overlap_threshold or ng_cfg.get("overlap_threshold", 0.5)
    _para_sep = exact_cfg.get("paragraph_sep", "\n\n")
    _min_para_chars = exact_cfg.get("min_para_chars", 50)

    click.echo(f"[ngram] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[ngram] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    click.echo(f"[ngram] ngram_size={_ngram_size}, overlap_threshold={_overlap_threshold}")

    out_dir = _resolve_output(output_dir, output_base, dataset, "ngram")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        _, summary = compute_ngram_dedup(
            docs,
            ngram_size=_ngram_size,
            overlap_threshold=_overlap_threshold,
            paragraph_sep=_para_sep,
            min_para_chars=_min_para_chars,
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[ngram] 段落污染 {summary['contaminated_paras']} / {summary['total_paras']} 段"
        f" ({summary['contaminated_para_pct']:.1%})"
        f"  受影响文档 {summary['contaminated_docs']} ({summary['contaminated_pct']:.1%})"
    )
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


# ── semdedup ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True)
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage4.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage4", show_default=True)
@click.option("--output-dir", default=None)
@click.option("--input-format", default=None)
@click.option("--max-docs", default=None, type=int)
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
@click.option("--model", "model_path", default=None, help="embedding 模型路径，默认 bge-m3")
@click.option("--eps", default=None, type=float, help="cos 阈值 = 1 - eps，默认 0.07")
@click.option("--batch-size", default=None, type=int, help="编码批大小")
@click.option("--max-length", default=None, type=int, help="token 截断长度")
@click.option("--sim-block", default=None, type=int, help="相似度分块行数（控制显存）")
@click.option("--device", default=None, help="cuda | cpu")
def semdedup(input_path, dataset, config_path, output_base, output_dir,
             input_format, max_docs, sample_mode, seed, model_path, eps, batch_size,
             max_length, sim_block, device):
    """Embedding 语义重复检测（bge-m3 + 余弦相似度聚簇）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    sd_cfg = cfg.get("semdedup", {})

    _model = model_path or sd_cfg.get("model_path", "/mnt/public/model/bge-m3")
    _eps = eps if eps is not None else sd_cfg.get("eps", 0.07)
    _batch_size = batch_size or sd_cfg.get("batch_size", 64)
    _max_length = max_length or sd_cfg.get("max_length", 512)
    _sim_block = sim_block or sd_cfg.get("sim_block", 2048)
    _device = device or sd_cfg.get("device", "cuda")

    click.echo(f"[semdedup] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[semdedup] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    click.echo(
        f"[semdedup] model={_model}, eps={_eps} (cos≥{1 - _eps:.3f}), "
        f"batch={_batch_size}, device={_device}"
    )

    out_dir = _resolve_output(output_dir, output_base, dataset, "semdedup")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        _, summary = compute_semdedup(
            docs,
            model_path=_model,
            eps=_eps,
            batch_size=_batch_size,
            max_length=_max_length,
            sim_block=_sim_block,
            device=_device,
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[semdedup] 语义重复 {summary['semantic_dup_docs']} / {summary['total_docs']} 条"
        f" ({summary['semantic_dup_pct']:.1%})"
        f"  簇数={summary['num_clusters']} 最大簇={summary['largest_cluster_size']}"
    )
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


if __name__ == "__main__":
    cli()
