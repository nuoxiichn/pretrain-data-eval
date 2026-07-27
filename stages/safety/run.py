"""Stage 2 CLI entry point.

Usage:
  python stages/safety/run.py pii      --input <path> --dataset <name> [options]
  python stages/safety/run.py secrets  --input <path> --dataset <name> [options]
  python stages/safety/run.py toxicity --input <path> --dataset <name> [options]
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
from stages.safety.utils import compute_pii, compute_secrets, compute_toxicity


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
    """Stage 2: 安全隐私检测"""


# ── pii ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True, help="输入文件或目录路径")
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage2.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage2", show_default=True)
@click.option("--output-dir", default=None, help="覆盖自动生成的输出目录")
@click.option("--input-format", default=None, help="覆盖 yaml 中的 input.format")
@click.option("--mode", default="general",
              type=click.Choice(["general", "code", "both"]), show_default=True,
              help="general=通用文本  code=代码语料  both=合并")
@click.option("--language", default=None, help="覆盖 yaml 中的 pii.language（en/zh/…）")
@click.option("--spacy-model", "spacy_model", default=None, help="覆盖 yaml 中的 pii.spacy_model")
@click.option("--max-docs", default=None, type=int, help="限制扫描文档数（调试用）")
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True, help="抽样策略：random=蓄水池, head=取前 N")
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True,
              help="抽样随机种子（仅 random 模式生效）")
def pii(input_path, dataset, config_path, output_base, output_dir,
        input_format, mode, language, spacy_model, max_docs, sample_mode, seed):
    """行 1+2: Presidio PII 检测"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    pii_cfg = cfg.get("pii", {})

    click.echo(f"[pii] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[pii] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    out_dir = _resolve_output(output_dir, output_base, dataset, "pii")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        _, summary = compute_pii(
            docs,
            language=language or pii_cfg.get("language", "en"),
            entities=pii_cfg.get("entities") or None,
            score_threshold=pii_cfg.get("score_threshold", 0.5),
            mode=mode,
            spacy_model=spacy_model or pii_cfg.get("spacy_model", "en_core_web_lg"),
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[pii] 命中 {summary['docs_with_pii']} / {summary['total_docs_scanned']} 条"
        f" ({summary['hit_pct']:.1%})  mode={mode}"
    )
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


# ── secrets ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True)
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage2.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage2", show_default=True)
@click.option("--output-dir", default=None)
@click.option("--input-format", default=None)
@click.option("--max-docs", default=None, type=int)
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
def secrets(input_path, dataset, config_path, output_base, output_dir,
            input_format, max_docs, sample_mode, seed):
    """行 3: Gitleaks Secret 扫描"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    sec_cfg = cfg.get("secrets", {})

    click.echo(f"[secrets] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[secrets] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    out_dir = _resolve_output(output_dir, output_base, dataset, "secrets")
    per_doc, summary = compute_secrets(
        docs, gitleaks_bin=sec_cfg.get("gitleaks_bin", "gitleaks")
    )

    pd_path = write_per_doc(per_doc, out_dir)
    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[secrets] 命中 {summary['docs_with_secrets']} / {summary['total_docs_scanned']} 条"
        f" ({summary['hit_pct']:.1%})"
    )
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {pd_path}")


# ── toxicity ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True)
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage2.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage2", show_default=True)
@click.option("--output-dir", default=None)
@click.option("--input-format", default=None)
@click.option("--max-docs", default=None, type=int)
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True)
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
@click.option("--device", default=None, help="cuda/cpu，默认自动检测（仅 HF model_path 后端用）")
def toxicity(input_path, dataset, config_path, output_base, output_dir,
             input_format, max_docs, sample_mode, seed, device):
    """行 4: Detoxify 毒性分类"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    tox_cfg = cfg.get("toxicity", {})

    click.echo(f"[toxicity] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[toxicity] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    out_dir = _resolve_output(output_dir, output_base, dataset, "toxicity")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        _, summary = compute_toxicity(
            docs,
            model_path=tox_cfg.get("model_path"),
            high_risk_threshold=tox_cfg.get("high_risk_threshold", 0.5),
            batch_size=tox_cfg.get("batch_size", 16),
            device=device,
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[toxicity] 高风险 {summary['high_risk_docs']} / {summary['total_docs_scanned']} 条"
        f" ({summary['high_risk_pct']:.1%})"
    )
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


if __name__ == "__main__":
    cli()
