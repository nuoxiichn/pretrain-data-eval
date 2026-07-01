"""DataTrove MinHash pipeline runner — 4 subcommands.

**必须在独立 conda env `pretrain-dedup` 下运行**，main env 不装 datatrove。

```
signature → buckets → cluster → aggregate
```

前三个是 datatrove 官方管线，最后一个是回读 `.clusters/.sizes` + 生成项目契约
`per_doc.jsonl + summary.json` 的适配层。

Usage
-----
```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate pretrain-dedup
export PYTHONPATH=/mnt/public/code/chennuoxi/pretrain-data-eval:$PYTHONPATH

# 单个 run 走完 4 阶段（推荐）
python stages/dedup_datatrove/run.py all \
    --input /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3/multi_style \
    --glob 'part-*.snappy.parquet' \
    --dataset ufw_en_l3 \
    --output-root outputs/stage4/datatrove_minhash/ufw_en_l3_sanity_100K \
    --limit 100000 \
    --tasks 8
```
"""
from __future__ import annotations

import importlib.metadata  # noqa: F401 — datatrove 0.9.0 compat
import shutil
import time
from pathlib import Path

import click

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup.minhash import (
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupSignature,
)
from datatrove.pipeline.readers import ParquetReader

from stages.dedup_datatrove.utils import (
    DocIdSink,
    LANG_CODE,
    MINHASH_CONFIG,
    UFW_ADAPTER_ID_KEY,
    UFW_ADAPTER_TEXT_KEY,
    aggregate,
    now_ts,
    ufw_adapter,
)


# ---------------------------------------------------------------------------
# 通用参数
# ---------------------------------------------------------------------------
def _default_language(dataset: str) -> str:
    return LANG_CODE.get(dataset, "eng")


def _paths(output_root: Path) -> dict[str, Path]:
    return {
        "signatures": output_root / "signatures",
        "buckets": output_root / "buckets",
        "clusters": output_root / "clusters",
        "docids": output_root / "docids",
        "logs": output_root / "logs",
        "final": output_root / "final",
    }


@click.group()
def cli():
    """DataTrove MinHash 4-stage runner."""


# ---------------------------------------------------------------------------
# Stage 1: signature
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--input", "input_dir", required=True, help="parquet 文件目录")
@click.option("--glob", "glob_pattern", default="part-*.snappy.parquet", show_default=True)
@click.option("--dataset", required=True, help="ufw_en_l3 / ufw_zh_l3")
@click.option("--output-root", "output_root", required=True)
@click.option("--limit", default=-1, type=int, help="每次跑最多读多少 doc；-1 全量")
@click.option("--tasks", default=8, type=int, help="signature 阶段 world_size")
@click.option("--workers", default=-1, type=int, help="并行 worker 数；-1 = tasks")
@click.option("--language", default=None, help="ISO-639-3；缺省按 dataset 自动")
def signature(input_dir, glob_pattern, dataset, output_root, limit, tasks, workers, language):
    output_root = Path(output_root)
    paths = _paths(output_root)
    paths["signatures"].mkdir(parents=True, exist_ok=True)
    paths["docids"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)

    lang = language or _default_language(dataset)
    click.echo(f"[sig] dataset={dataset} language={lang} tasks={tasks} limit={limit}")

    reader = ParquetReader(
        data_folder=input_dir,
        glob_pattern=glob_pattern,
        text_key=UFW_ADAPTER_TEXT_KEY,
        id_key=UFW_ADAPTER_ID_KEY,
        adapter=ufw_adapter,
        limit=limit,
        recursive=True,
    )
    sink = DocIdSink(output_folder=str(paths["docids"]))
    sig_step = MinhashDedupSignature(
        output_folder=str(paths["signatures"]),
        config=MINHASH_CONFIG,
        language=lang,
    )
    executor = LocalPipelineExecutor(
        pipeline=[reader, sink, sig_step],
        tasks=tasks,
        workers=workers,
        logging_dir=str(paths["logs"] / "signature"),
    )
    t0 = time.time()
    executor.run()
    click.echo(f"[sig] wall={time.time()-t0:.1f}s → {paths['signatures']}")


# ---------------------------------------------------------------------------
# Stage 2: buckets
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--output-root", "output_root", required=True)
@click.option("--tasks", default=None, type=int,
              help="必须能整除 num_buckets；缺省 = num_buckets = 8")
@click.option("--workers", default=-1, type=int)
def buckets(output_root, tasks, workers):
    output_root = Path(output_root)
    paths = _paths(output_root)
    paths["buckets"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)

    t = tasks or MINHASH_CONFIG.num_buckets
    if t % MINHASH_CONFIG.num_buckets != 0:
        raise click.BadParameter(
            f"buckets: tasks({t}) must be a multiple of num_buckets({MINHASH_CONFIG.num_buckets})"
        )
    click.echo(f"[buckets] tasks={t}")
    step = MinhashDedupBuckets(
        input_folder=str(paths["signatures"]),
        output_folder=str(paths["buckets"]),
        config=MINHASH_CONFIG,
        only_dedup_in_index=False,
    )
    executor = LocalPipelineExecutor(
        pipeline=[step],
        tasks=t,
        workers=workers,
        logging_dir=str(paths["logs"] / "buckets"),
    )
    t0 = time.time()
    executor.run()
    click.echo(f"[buckets] wall={time.time()-t0:.1f}s → {paths['buckets']}")


# ---------------------------------------------------------------------------
# Stage 3: cluster（world_size 必须 == 1）
# ---------------------------------------------------------------------------
@cli.command()
@click.option("--output-root", "output_root", required=True)
def cluster(output_root):
    output_root = Path(output_root)
    paths = _paths(output_root)
    paths["clusters"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)

    click.echo(f"[cluster] tasks=1 (datatrove requires world_size==1)")
    step = MinhashDedupCluster(
        input_folder=str(paths["buckets"]),
        output_folder=str(paths["clusters"]),
        config=MINHASH_CONFIG,
        save_cluster_id=True,
        save_cluster_size=True,
    )
    executor = LocalPipelineExecutor(
        pipeline=[step],
        tasks=1,
        workers=1,
        logging_dir=str(paths["logs"] / "cluster"),
    )
    t0 = time.time()
    executor.run()
    click.echo(f"[cluster] wall={time.time()-t0:.1f}s → {paths['clusters']}")


# ---------------------------------------------------------------------------
# Stage 4: aggregate → per_doc.jsonl + summary.json
# ---------------------------------------------------------------------------
@cli.command("aggregate")
@click.option("--dataset", required=True)
@click.option("--output-root", "output_root", required=True)
@click.option("--tasks-signature", default=8, type=int,
              help="回填 summary.n_workers 用；填 signature 阶段用的 tasks")
def aggregate_cmd(dataset, output_root, tasks_signature):
    output_root = Path(output_root)
    paths = _paths(output_root)

    sm = aggregate(
        dataset=dataset,
        signatures_folder=paths["signatures"],
        buckets_folder=paths["buckets"],
        clusters_folder=paths["clusters"],
        docids_folder=paths["docids"],
        output_dir=paths["final"],
        n_workers=tasks_signature,
        wall_time_sec=0.0,
    )
    click.echo(f"[agg] total={sm['total_docs']} near_dup={sm['near_dup_docs']} "
               f"({sm['near_dup_pct']*100:.4f}%) pairs={sm['near_dup_pairs']}")
    click.echo(f"[agg] → {paths['final']}/")


# ---------------------------------------------------------------------------
# 一键 all：signature → buckets → cluster → aggregate
# ---------------------------------------------------------------------------
@cli.command("all")
@click.option("--input", "input_dir", required=True)
@click.option("--glob", "glob_pattern", default="part-*.snappy.parquet", show_default=True)
@click.option("--dataset", required=True)
@click.option("--output-root", "output_root", required=True)
@click.option("--limit", default=-1, type=int)
@click.option("--tasks", default=8, type=int,
              help="signature/buckets 阶段的 world_size；必须是 num_buckets(8) 的倍数")
@click.option("--workers", default=-1, type=int)
@click.option("--language", default=None)
@click.option("--clean", is_flag=True, help="删已有中间产物后重跑")
@click.pass_context
def all_cmd(ctx, input_dir, glob_pattern, dataset, output_root, limit,
            tasks, workers, language, clean):
    output_root = Path(output_root)
    paths = _paths(output_root)

    if clean:
        for k in ["signatures", "buckets", "clusters", "docids", "final"]:
            if paths[k].exists():
                shutil.rmtree(paths[k])
        click.echo(f"[all] cleaned {output_root}")

    t_start = time.time()
    ctx.invoke(signature, input_dir=input_dir, glob_pattern=glob_pattern,
               dataset=dataset, output_root=str(output_root), limit=limit,
               tasks=tasks, workers=workers, language=language)
    ctx.invoke(buckets, output_root=str(output_root), tasks=tasks, workers=workers)
    ctx.invoke(cluster, output_root=str(output_root))
    wall = time.time() - t_start

    click.echo(f"[all] 3 stages done in {wall:.1f}s；进入 aggregate")
    sm = aggregate(
        dataset=dataset,
        signatures_folder=paths["signatures"],
        buckets_folder=paths["buckets"],
        clusters_folder=paths["clusters"],
        docids_folder=paths["docids"],
        output_dir=paths["final"],
        n_workers=tasks,
        wall_time_sec=wall,
    )
    click.echo(f"\n=== summary ({dataset}) ===")
    click.echo(f"total_docs           = {sm['total_docs']:,}")
    click.echo(f"near_dup_docs        = {sm['near_dup_docs']:,} ({sm['near_dup_pct']*100:.4f}%)")
    click.echo(f"near_dup_pairs       = {sm['near_dup_pairs']:,}")
    click.echo(f"num_clusters_multi   = {sm['num_clusters_multi']:,}")
    click.echo(f"largest_cluster_size = {sm['largest_cluster_size']}")
    click.echo(f"wall_time_sec        = {sm['wall_time_sec']}")
    click.echo(f"\n→ {paths['final']}/per_doc.jsonl + summary.json")


if __name__ == "__main__":
    cli()
