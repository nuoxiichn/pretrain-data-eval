"""Stage 3 CLI entry point.

Usage:
  python stages/cleaning/run.py extraction --input <path> --dataset <name> [options]
  python stages/cleaning/run.py langid     --input <path> --dataset <name> --model <path> [options]
  python stages/cleaning/run.py glotlid    --input <path> --dataset <name> [options]
  python stages/cleaning/run.py langcross  --input <path> --dataset <name> [options]
  python stages/cleaning/run.py quality    --input <path> --dataset <name> [options]
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
from stages.cleaning.utils import (
    compute_extraction_audit,
    compute_glotlid,
    compute_lang_crosscheck,
    compute_langid,
    compute_quality,
)


def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_output(output_dir: str | None, output_base: str, dataset: str, stage: str) -> Path:
    if output_dir:
        return use_output_dir(output_dir)
    return make_output_dir(output_base, stage, dataset)


@click.group()
def cli():
    """Stage 3: 抽取质量 + 语言识别 + 多语言覆盖 + 文本质量"""


# ── langid ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True)
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage3.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage3", show_default=True)
@click.option("--output-dir", default=None)
@click.option("--input-format", default=None)
@click.option("--model", "model_path", default=None, help="lid.176.bin 路径（覆盖 yaml）")
@click.option("--max-docs", default=None, type=int)
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True, help="抽样策略：random=蓄水池, head=取前 N")
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True,
              help="抽样随机种子（仅 random 模式生效）")
def langid(input_path, dataset, config_path, output_base, output_dir,
           input_format, model_path, max_docs, sample_mode, seed):
    """行 2: fastText 语言识别"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    lid_cfg = cfg.get("langid", {})
    resolved_model = model_path or lid_cfg.get("model_path")
    if not resolved_model:
        raise click.UsageError("需要指定 --model 或在 configs/stage3.yaml 中设置 langid.model_path")

    click.echo(f"[langid] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[langid] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    out_dir = _resolve_output(output_dir, output_base, dataset, "langid")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        _, summary = compute_langid(
            docs,
            model_path=resolved_model,
            top_k=lid_cfg.get("top_k", 3),
            confidence_threshold=lid_cfg.get("confidence_threshold", 0.7),
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    click.echo(f"[langid] {summary['total_docs']} 条  "
               f"mismatch={summary['mismatch_pct']:.1%}  "
               f"low_conf={summary['low_confidence_pct']:.1%}")
    click.echo(f"  语言分布: { {k: v for k, v in list(summary['language_distribution'].items())[:5]} }")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


# ── glotlid (row 3) ───────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True)
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage3.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage3", show_default=True)
@click.option("--output-dir", default=None)
@click.option("--input-format", default=None)
@click.option("--model", "model_path", default=None, help="GlotLID 模型路径（覆盖 yaml）")
@click.option("--max-docs", default=None, type=int)
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
def glotlid(input_path, dataset, config_path, output_base, output_dir,
            input_format, model_path, max_docs, sample_mode, seed):
    """行 3: GlotLID v3 细粒度语种-脚本识别"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    gl_cfg = cfg.get("glotlid", {})
    resolved_model = model_path or gl_cfg.get("model_path")
    if not resolved_model:
        raise click.UsageError(
            "GlotLID 需要本地模型路径：在 configs/stage3.yaml 设置 glotlid.model_path "
            "（如 /mnt/public/model/glotlid/model.bin），或用 --model 传入"
        )

    click.echo(f"[glotlid] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[glotlid] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    out_dir = _resolve_output(output_dir, output_base, dataset, "glotlid")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        _, summary = compute_glotlid(
            docs,
            model_path=resolved_model,
            top_k=gl_cfg.get("top_k", 3),
            confidence_threshold=gl_cfg.get("confidence_threshold", 0.7),
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    top5 = list(summary["lang_script_distribution"].items())[:5]
    click.echo(f"[glotlid] {summary['total_docs']} 条  low_conf={summary['low_confidence_pct']:.1%}")
    click.echo(f"  top5: {top5}")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


# ── langcross (粗细语种交叉核对) ──────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True)
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage3.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage3", show_default=True)
@click.option("--output-dir", default=None)
@click.option("--input-format", default=None)
@click.option("--max-docs", default=None, type=int)
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
def langcross(input_path, dataset, config_path, output_base, output_dir,
              input_format, max_docs, sample_mode, seed):
    """交叉核对：lid.176(粗) vs GlotLID(细)，暴露被吸收的低资源语种"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    lid_model = cfg.get("langid", {}).get("model_path")
    glot_model = cfg.get("glotlid", {}).get("model_path")
    if not lid_model or not glot_model:
        raise click.UsageError(
            "langcross 需要 langid.model_path 与 glotlid.model_path 都在 "
            "configs/stage3.yaml 中配置"
        )

    click.echo(f"[langcross] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[langcross] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    out_dir = _resolve_output(output_dir, output_base, dataset, "langcross")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        _, summary = compute_lang_crosscheck(
            docs,
            langid_model_path=lid_model,
            glotlid_model_path=glot_model,
            confidence_threshold=cfg.get("glotlid", {}).get("confidence_threshold", 0.7),
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    click.echo(f"[langcross] {summary['total_docs']} 条  "
               f"disagree={summary['disagreement_pct']:.1%}  "
               f"absorbed={summary['possibly_absorbed_count']} 条")
    if summary["absorbed_lang_distribution"]:
        click.echo(f"  被吸收语种: {summary['absorbed_lang_distribution']}")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


# ── quality ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True)
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage3.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage3", show_default=True)
@click.option("--output-dir", default=None)
@click.option("--input-format", default=None)
@click.option("--max-docs", default=None, type=int)
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
@click.option("--filters", default=None, help="逗号分隔的过滤器列表，如 gopher_quality,c4_quality")
def quality(input_path, dataset, config_path, output_base, output_dir,
            input_format, max_docs, sample_mode, seed, filters):
    """行 4: 质量过滤器（自实现 Gopher/C4 信号，只读模式）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    q_cfg = cfg.get("quality", {})
    filter_names = filters.split(",") if filters else q_cfg.get("filters")

    click.echo(f"[quality] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[quality] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    out_dir = _resolve_output(output_dir, output_base, dataset, "quality")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        _, summary = compute_quality(docs, filter_names=filter_names, on_doc=_write)

    sm_path = write_summary(summary, out_dir)
    click.echo(f"[quality] {summary['total_docs']} 条")
    for name, pct in summary["filter_fail_pcts"].items():
        click.echo(f"  {name}: 失败 {summary['filter_fail_counts'][name]} 条 ({pct:.1%})")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


# ── extraction (row 1: 抽取质量审计) ──────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True)
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage3.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage3", show_default=True)
@click.option("--output-dir", default=None)
@click.option("--input-format", default=None)
@click.option("--max-docs", default=None, type=int)
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
def extraction(input_path, dataset, config_path, output_base, output_dir,
               input_format, max_docs, sample_mode, seed):
    """行 1: 抽取质量审计（已清洗文本的 HTML/markup/boilerplate/mojibake 残留）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    ex_cfg = cfg.get("extraction", {})

    click.echo(f"[extraction] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[extraction] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    out_dir = _resolve_output(output_dir, output_base, dataset, "extraction")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        _, summary = compute_extraction_audit(
            docs,
            short_stub_chars=ex_cfg.get("short_stub_chars", 100),
            html_tag_min_count=ex_cfg.get("html_tag_min_count", 3),
            html_entity_min_count=ex_cfg.get("html_entity_min_count", 1),
            boilerplate_edge_ratio=ex_cfg.get("boilerplate_edge_ratio", 0.05),
            boilerplate_edge_min_chars=ex_cfg.get("boilerplate_edge_min_chars", 200),
            boilerplate_middle_weight=ex_cfg.get("boilerplate_middle_weight", 0.2),
            boilerplate_weighted_threshold=ex_cfg.get("boilerplate_weighted_threshold", 2.0),
            risk_score_threshold=ex_cfg.get("risk_score_threshold", 2.0),
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    click.echo(f"[extraction] {summary['total_docs']} 条  "
               f"low_quality={summary['low_extraction_quality_pct']:.1%}  "
               f"(html={summary['html_residue_pct']:.1%} "
               f"boiler={summary['boilerplate_pct']:.1%} "
               f"mojibake={summary['mojibake_pct']:.1%})")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


if __name__ == "__main__":
    cli()
