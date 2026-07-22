"""Stage 7 CLI entry point.

Usage:
  PYTHONPATH=. python stages/synthetic/run.py binoculars --input <path> --dataset <name> [options]
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
from stages.synthetic.utils import compute_binoculars


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
    """Stage 7: 合成数据检测"""


# ── binoculars ────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--input", "input_path", required=True, help="输入文件或目录路径")
@click.option("--dataset", required=True)
@click.option("--config", "config_path", default="configs/stage7.yaml", show_default=True)
@click.option("--output-base", default="outputs/stage7", show_default=True)
@click.option("--output-dir", default=None, help="覆盖自动生成的输出目录")
@click.option("--input-format", default=None, help="覆盖 yaml 中的 input.format")
@click.option("--observer-model", default=None, help="覆盖 yaml 中的 observer 模型路径")
@click.option("--performer-model", default=None, help="覆盖 yaml 中的 performer 模型路径")
@click.option("--threshold", default=None, type=float, help="覆盖 yaml 中的判定阈值")
@click.option("--batch-size", default=None, type=int, help="推理批大小")
@click.option("--max-length", default=None, type=int, help="最大 token 数（截断）")
@click.option("--device", default=None, help="cuda/cpu，默认自动检测")
@click.option("--dtype", default=None, type=click.Choice(["float16", "bfloat16", "float32"]))
@click.option("--max-docs", default=None, type=int, help="限制扫描文档数（调试用）")
@click.option("--sample-mode", default=DEFAULT_SAMPLE_MODE, type=click.Choice(SAMPLE_MODES),
              show_default=True, help="抽样策略：random=蓄水池, head=取前 N")
@click.option("--seed", default=DEFAULT_SEED, type=int, show_default=True)
def binoculars(
    input_path, dataset, config_path, output_base, output_dir,
    input_format, observer_model, performer_model, threshold,
    batch_size, max_length, device, dtype, max_docs, sample_mode, seed,
):
    """AI 生成文本检测（Binoculars observer/performer 双模型对比）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    bino_cfg = cfg.get("binoculars", {})

    obs_path = observer_model or bino_cfg.get("observer_model")
    perf_path = performer_model or bino_cfg.get("performer_model")
    if not obs_path or not perf_path:
        raise click.UsageError(
            "需要 observer 和 performer 模型路径。"
            "在 configs/stage7.yaml 的 binoculars 段设置，或通过 --observer-model / --performer-model 指定。"
        )

    thresh = threshold if threshold is not None else bino_cfg.get("threshold", 0.8536)
    bs = batch_size or bino_cfg.get("batch_size", 8)
    ml = max_length or bino_cfg.get("max_length", 512)
    dt = dtype or bino_cfg.get("dtype", "float16")

    click.echo(f"[binoculars] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[binoculars] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")
    click.echo(f"[binoculars] 共 {len(docs)} 文档，observer={obs_path}, performer={perf_path}")

    out_dir = _resolve_output(output_dir, output_base, dataset, "binoculars")
    per_doc_path = out_dir / "per_doc.jsonl"

    t0 = time.time()
    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        _, summary = compute_binoculars(
            docs,
            observer_path=obs_path,
            performer_path=perf_path,
            threshold=thresh,
            batch_size=bs,
            max_length=ml,
            device=device,
            dtype=dt,
            on_doc=_write,
        )
    elapsed = time.time() - t0
    summary["elapsed_seconds"] = round(elapsed, 1)
    summary["docs_per_second"] = round(len(docs) / elapsed, 1) if elapsed > 0 else 0

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[binoculars] AI 生成 {summary['ai_generated_docs']} / {summary['total_docs_scanned']} 条"
        f" ({summary['ai_generated_pct']:.1%})  threshold={thresh}"
    )
    click.echo(f"[binoculars] 耗时 {elapsed:.1f}s（{summary['docs_per_second']:.1f} docs/s）")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


if __name__ == "__main__":
    cli()
