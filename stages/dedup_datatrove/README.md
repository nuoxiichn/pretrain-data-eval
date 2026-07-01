# stages/dedup_datatrove — DataTrove MinHash 三段管线

> **本 stage 跑在独立 conda env `pretrain-dedup`**，不与项目 main env 共依赖。
> 目的：接入 DataTrove 的分布式 MinHash 管线，把 `stages/dedup/utils.py` 自实现无法扩到 TB 级的问题解决。

## 定位

- 只跑 stages 1-3（signature / buckets / cluster），**跳过 stage 4 (filter)**——项目是只读审计，只出重复率与簇统计，不输出去重后的数据集。
- 参数严格照 `stages/dedup/utils.py` 现值：`n_grams=5, num_buckets=8, hashes_per_bucket=8, jaccard≈0.8`。
- 输出契约字段对齐 v3 报告 §2.4.3 的表头。

## 环境准备（一次性）

```bash
conda create -n pretrain-dedup python=3.11 -y
conda activate pretrain-dedup
pip install "datatrove[processing,io]>=0.3" pyarrow numpy tqdm click pyyaml spacy jieba
```

env freeze 存于 `envs/datatrove.txt`。**不要**改 project 主 `pyproject.toml` 的 datatrove 声明——独立 env 是唯一入口。

## 运行

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate pretrain-dedup
export PYTHONPATH=/mnt/public/code/chennuoxi/pretrain-data-eval:$PYTHONPATH

# Smoke（EN 1 parquet × 500 docs）
python stages/dedup_datatrove/run.py all \
    --input /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3/multi_style \
    --glob 'part-00000-*.snappy.parquet' \
    --dataset ufw_en_l3 \
    --output-root outputs/stage4/datatrove_minhash/smoke \
    --limit 500 --tasks 8

# Sanity 100K EN（对齐 v3 §2.4.3 自实现 0%）
python stages/dedup_datatrove/run.py all \
    --input /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3/multi_style \
    --dataset ufw_en_l3 \
    --output-root outputs/stage4/datatrove_minhash/ufw_en_l3_sanity_100K \
    --limit 100000 --tasks 8

# EN 全量 698M
python stages/dedup_datatrove/run.py all \
    --input /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3 \
    --dataset ufw_en_l3 \
    --output-root outputs/stage4/datatrove_minhash/ufw_en_l3_full \
    --tasks 32

# ZH 全量 360M
python stages/dedup_datatrove/run.py all \
    --input /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_zh_l3 \
    --dataset ufw_zh_l3 \
    --output-root outputs/stage4/datatrove_minhash/ufw_zh_l3_full \
    --tasks 32
```

## 子命令拆开跑（断点续跑用）

```bash
python stages/dedup_datatrove/run.py signature --input ... --dataset ... --output-root ... --limit ... --tasks 8
python stages/dedup_datatrove/run.py buckets   --output-root ...
python stages/dedup_datatrove/run.py cluster   --output-root ...
python stages/dedup_datatrove/run.py aggregate-cmd --input ... --dataset ... --output-root ... --limit ...
```

`buckets` 阶段 tasks 必须是 `num_buckets=8` 的倍数；`cluster` 阶段 datatrove 强制 world_size==1。

## 输入 / 输出契约

**输入**：UFW-L3 parquet（字段 `uid / content / style`）。datatrove `ParquetReader` 直接读，`utils.ufw_adapter` 把 `content → text, uid → id`。

**中间产物**（在 `<output-root>/{signatures,buckets,clusters,logs}/`）：
- `signatures/bucket_XXX/YYYYY.minhash.sig` — 二进制签名文件
- `buckets/XXXXX_YY.dups` — 每记录 4×uint32 = `(file_id1, doc_id1, file_id2, doc_id2)`
- `clusters/000000.clusters` + `.sizes` + `.remove` — 每 doc 一条 uint32

**最终产物**（`<output-root>/final/`，对齐项目 `src/schema.py` 契约）：
- `per_doc.jsonl` — 每行：
  ```json
  {"doc_id": "...", "scores": {"cluster_id": 42, "cluster_size": 3, "near_dup_count": 2},
   "flags": {"is_near_dup": true, "in_multi_cluster": true}}
  ```
- `summary.json`：
  ```json
  {"total_docs": ..., "near_dup_docs": ..., "near_dup_pct": ...,
   "near_dup_pairs": ..., "num_clusters_multi": ..., "largest_cluster_size": ...,
   "n_hashes": 64, "num_bands": 8, "band_size": 8, "jaccard_threshold": 0.8,
   "n_workers": ..., "wall_time_sec": ...}
  ```

## 已知差异 / gap（与 `stages/dedup/` 自实现相比）

1. **ZH 分词**：自实现按 CJK-dominant 切换到 25-char 级 shingle；本 stage 用 datatrove `SpaCyTokenizer("zh")`（内部走 jieba）。Sanity 步骤会暴露差异——如 ZH 100K 上跑出的 `near_dup_pct` 显著偏离自实现，只在 README 记 known-gap，不 hack。
2. **`jaccard_est` 字段不给**：datatrove 只给簇成员关系，不给每对具体 Jaccard 估计值。用 `cluster_size ≥ 2` 当 `is_near_dup` flag，与自实现"Jaccard≥0.8 才成对"效果等价。
3. **datatrove 0.9.0 上游 bug**：`utils/_import_utils.py` 未 `import importlib.metadata`，Python 3.11 会 AttributeError。`utils.py` 顶部 `import importlib.metadata` 一次全局解决，不改依赖库。
4. **stage3 world_size 强制 == 1**：datatrove 官方限制。大数据集时 stage3 是单进程瓶颈；stage1/2 靠 `--tasks 32` 或更高并行。

## 目录结构

```
stages/dedup_datatrove/
  __init__.py
  run.py       # click CLI 4 子命令 + `all`
  utils.py     # MinhashConfig + Adapter + aggregate
  README.md    # 本文件
```
