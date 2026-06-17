"""Stage 9 CLI entry point.

Usage:
  PYTHONPATH=. python stages/longctx/run.py config-audit --config-file <path> --dataset <name> [options]
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

from src.schema import make_output_dir, use_output_dir, write_per_doc, write_summary
from stages.longctx.utils import compute_config_audit


def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_output(output_dir: str | None, output_base: str, dataset: str, stage: str) -> Path:
    if output_dir:
        return use_output_dir(output_dir)
    return make_output_dir(output_base, stage, dataset)


@click.group()
def cli():
    """Stage 9: 长上下文（训练配置审计）"""


@cli.command("config-audit")
@click.option("--config-file", required=True, help="训练配置文件路径（YAML/JSON/shell script）")
@click.option("--dataset", required=True, help="数据集名称（仅用于输出目录命名）")
@click.option("--config", "config_path", default="configs/stage9.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage9", show_default=True)
@click.option("--output-dir", default=None, help="覆盖自动生成的输出目录")
def config_audit(config_file, dataset, config_path, output_base, output_dir):
    """审计 Megatron-LM packing 训练配置（reset_position_ids / reset_attention_mask / eod_mask_loss）"""
    click.echo(f"[config-audit] 审计训练配置 {config_file} ...")

    results, summary = compute_config_audit(config_file)

    out_dir = _resolve_output(output_dir, output_base, dataset, "config-audit")
    pd_path = write_per_doc(results, out_dir)
    sm_path = write_summary(summary, out_dir)

    valid = summary["config_valid"]
    status = "PASS" if valid else "FAIL"
    click.echo(f"[config-audit] 配置验证: {status}")
    if summary["missing_params"]:
        click.echo(f"  缺失参数: {', '.join(summary['missing_params'])}")
    if summary["invalid_params"]:
        click.echo(f"  无效参数: {', '.join(summary['invalid_params'])}")
    for pname, pinfo in summary["parameters"].items():
        mark = "✓" if pinfo["valid"] else "✗"
        click.echo(f"  {mark} {pname}: found={pinfo['found']}, value={pinfo['value']}")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {pd_path}")


if __name__ == "__main__":
    cli()
