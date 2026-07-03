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
    build_bench_embeddings,
    build_bench_minhash,
    compute_cascade_contamination,
    compute_code_ast,
    compute_code_near,
    compute_embed_contamination,
    compute_exact_contamination,
    compute_near_contamination,
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


# ── near (L2 通用文本) ───────────────────────────────────────────────────────

@cli.command("near")
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
@click.option("--ngram-size", default=None, type=int)
@click.option("--jaccard-threshold", default=None, type=float)
def near(input_path, dataset, config_path, output_base, output_dir,
         input_format, max_docs, sample_mode, seed, ngram_size, jaccard_threshold):
    """L2 通用文本 MinHash 近重复污染（char n-gram + LSH，对 benchmark text + code 都建签名）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    near_cfg = cfg.get("near", {})
    bench_cfg = cfg.get("benchmarks", {})

    click.echo("[near] 加载 benchmark ...")
    bench_items = load_benchmarks(bench_cfg)
    total_bench = sum(len(v) for v in bench_items.values())
    click.echo(f"[near] 已加载 {len(bench_items)} 个 benchmark，共 {total_bench} 条样本")

    click.echo(f"[near] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[near] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    _ngram = ngram_size or near_cfg.get("ngram_size", 5)
    _threshold = jaccard_threshold if jaccard_threshold is not None else near_cfg.get("jaccard_threshold", 0.85)
    _num_hashes = near_cfg.get("num_hashes", 128)
    _num_bands = near_cfg.get("num_bands", 16)
    _band_size = near_cfg.get("band_size", 8)

    click.echo(
        f"[near] ngram={_ngram}, threshold={_threshold}, "
        f"hashes={_num_hashes}, bands={_num_bands}×{_band_size}"
    )

    out_dir = _resolve_output(output_dir, output_base, dataset, "near")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        _, summary = compute_near_contamination(
            docs, bench_items,
            ngram_size=_ngram, jaccard_threshold=_threshold,
            num_hashes=_num_hashes, num_bands=_num_bands, band_size=_band_size,
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[near] 近污染 {summary['near_contaminated_docs']} / {summary['total_docs']} 条"
        f" ({summary['near_contaminated_pct']:.1%})"
        f"  benchmark 签名={summary['bench_samples']}"
    )
    if summary.get("per_benchmark_hits"):
        click.echo(f"[near] 各 benchmark 命中: {summary['per_benchmark_hits']}")
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


# ── embed (L3 BGE-m3 + FAISS) ────────────────────────────────────────────────

@cli.command("embed")
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
@click.option("--model-path", default=None, help="覆盖 yaml 中的 embed.model_path")
@click.option("--cos-threshold", default=None, type=float)
@click.option("--top-k", default=None, type=int)
def embed(input_path, dataset, config_path, output_base, output_dir,
          input_format, max_docs, sample_mode, seed, model_path, cos_threshold, top_k):
    """L3 embedding 语义污染（BGE-m3 + FAISS IndexFlatIP）"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    embed_cfg = cfg.get("embed", {})
    bench_cfg = cfg.get("benchmarks", {})

    _model = model_path or embed_cfg.get("model_path", "/mnt/public/model/bge-m3")
    _cos = cos_threshold if cos_threshold is not None else embed_cfg.get("cos_threshold", 0.85)
    _topk = top_k or embed_cfg.get("top_k", 5)
    _bs = embed_cfg.get("batch_size", 64)
    _maxlen = embed_cfg.get("max_length", 512)
    _device = embed_cfg.get("device", "cuda")

    click.echo("[embed] 加载 benchmark ...")
    bench_items = load_benchmarks(bench_cfg)
    total_bench = sum(len(v) for v in bench_items.values())
    click.echo(f"[embed] 已加载 {len(bench_items)} 个 benchmark，共 {total_bench} 条样本")
    click.echo(f"[embed] model={_model}, cos_threshold={_cos}, top_k={_topk}, device={_device}")

    click.echo(f"[embed] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[embed] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    out_dir = _resolve_output(output_dir, output_base, dataset, "embed")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        _, summary = compute_embed_contamination(
            docs, bench_items,
            model_path=_model, cos_threshold=_cos, top_k=_topk,
            batch_size=_bs, max_length=_maxlen, device=_device,
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    click.echo(
        f"[embed] 语义污染 {summary['semantic_contaminated_docs']} / {summary['total_docs']} 条"
        f" ({summary['semantic_contaminated_pct']:.1%})"
        f"  benchmark embeddings={summary['bench_samples']}"
    )
    if summary.get("per_benchmark_hits"):
        click.echo(f"[embed] 各 benchmark 命中: {summary['per_benchmark_hits']}")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


# ── cascade (L1 → L2 → L3 编排) ──────────────────────────────────────────────

@cli.command("cascade")
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
@click.option("--index-dir", default=None,
              help="预构建索引目录（由 bench_index.py build 产出）；不传则每次重建（耗时）")
def cascade(input_path, dataset, config_path, output_base, output_dir,
            input_format, max_docs, sample_mode, seed, index_dir):
    """三层 cascade 污染检测 (L1 exact → L2 MinHash → L3 BGE-m3 embedding)"""
    cfg = _load_config(config_path)
    input_cfg = dict(cfg.get("input", {}))
    if input_format:
        input_cfg["format"] = input_format
    cas_cfg = cfg.get("cascade", {})
    exact_cfg = cfg.get("exact", {})
    bench_cfg = cfg.get("benchmarks", {})

    # L1
    l1_para_sep = exact_cfg.get("paragraph_sep", "\n\n")
    l1_min_para = exact_cfg.get("min_para_chars", 50)

    # L2
    l2cfg = cas_cfg.get("layer2", {})
    # L3
    l3cfg = cas_cfg.get("layer3", {})

    # 预建索引（如果 --index-dir 或 yaml benchmarks.index_dir 指向已构建目录）
    from pathlib import Path as _Path
    prebuilt = None
    idx_dir = index_dir or bench_cfg.get("index_dir")
    if idx_dir and _Path(idx_dir).exists() and (_Path(idx_dir) / "hash_index.pkl").exists():
        click.echo(f"[cascade] 加载预建索引 {idx_dir} ...")
        from stages.contamination.bench_index import load_index
        prebuilt = load_index(_Path(idx_dir))
        bench_items = {}   # 预建模式不需 raw items；cascade 函数会跳过 build_*
        n_l2 = len(prebuilt["l2_index"]["bench_sigs"])
        n_l3 = len(prebuilt["l3_embed"]["meta"])
        click.echo(f"[cascade] 预建：L2 sigs={n_l2}, L3 embeddings={n_l3}")
    else:
        click.echo("[cascade] 加载 benchmark（无预建索引，将动态构建 L2/L3）...")
        bench_items = load_benchmarks(bench_cfg)
        total_bench = sum(len(v) for v in bench_items.values())
        click.echo(f"[cascade] 已加载 {len(bench_items)} 个 benchmark，共 {total_bench} 条样本")

    click.echo(
        f"[cascade] L2 阈值: enter_l3=[{l2cfg.get('enter_l3_low', 0.30)}, "
        f"{l2cfg.get('enter_l3_high', 0.90)})  "
        f"L3 阈值: red≥{l3cfg.get('cos_red', 0.85)}, yellow≥{l3cfg.get('cos_yellow', 0.70)}"
    )

    click.echo(f"[cascade] 读取 {input_path} ...")
    docs = sample_documents(read_documents(input_path, config=input_cfg),
                            max_docs, mode=sample_mode, seed=seed)
    if max_docs:
        click.echo(f"[cascade] 抽样 {len(docs)} 条 (mode={sample_mode}, seed={seed})")

    out_dir = _resolve_output(output_dir, output_base, dataset, "cascade")
    per_doc_path = out_dir / "per_doc.jsonl"

    with per_doc_path.open("w", encoding="utf-8") as f:
        def _write(r: DocResult) -> None:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        _, summary = compute_cascade_contamination(
            docs, bench_items,
            prebuilt_index=prebuilt,
            paragraph_sep=l1_para_sep, min_para_chars=l1_min_para,
            l2_ngram_size=l2cfg.get("ngram_size", 5),
            l2_num_hashes=l2cfg.get("num_hashes", 128),
            l2_num_bands=l2cfg.get("num_bands", 16),
            l2_band_size=l2cfg.get("band_size", 8),
            l2_window_size=l2cfg.get("window_size", 150),
            l2_window_stride=l2cfg.get("window_stride", 75),
            l2_enter_l3_low=l2cfg.get("enter_l3_low", 0.30),
            l2_enter_l3_high=l2cfg.get("enter_l3_high", 0.90),
            l3_model_path=l3cfg.get("model_path", "/mnt/public/model/bge-m3"),
            l3_top_k=l3cfg.get("top_k", 5),
            l3_batch_size=l3cfg.get("batch_size", 64),
            l3_max_length=l3cfg.get("max_length", 512),
            l3_device=l3cfg.get("device", "cuda"),
            l3_cos_red=l3cfg.get("cos_red", 0.85),
            l3_cos_yellow=l3cfg.get("cos_yellow", 0.70),
            on_doc=_write,
        )

    sm_path = write_summary(summary, out_dir)
    vd = summary["verdict_distribution"]
    click.echo(
        f"[cascade] verdict: red={vd.get('red', 0)} yellow={vd.get('yellow', 0)} green={vd.get('green', 0)} "
        f"/ total={summary['total_docs']}"
    )
    cb = summary["cost_breakdown"]
    click.echo(f"[cascade] cost: L1={cb['l1_processed']} L2={cb['l2_processed']} L3={cb['l3_processed']}")
    if summary.get("per_benchmark_red_hits"):
        click.echo(f"[cascade] per-benchmark red: {summary['per_benchmark_red_hits']}")
    if summary.get("cross_lingual_docs"):
        click.echo(f"[cascade] 跨语言污染嫌疑 docs: {summary['cross_lingual_docs']}")
    click.echo(f"  -> {sm_path}")
    click.echo(f"  -> {per_doc_path}")


if __name__ == "__main__":
    cli()
