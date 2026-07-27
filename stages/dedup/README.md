# Stage 4：重复度分析

## 评估目标

对语料样本进行重复度审计，输出文档级/段落级精确重复率和近重复率，供人工判断是否需要执行去重 pipeline。

> **注意**：本脚本为只读审计工具，对样本（默认 `--max-docs`）计算重复率。
> TB 级全量去重请使用独立的 DataTrove / Dolma pipeline。

## 子命令

| 子命令 | 对应行 | 说明 |
|--------|--------|------|
| `exact` | 行 1+2 | MD5 hash 精确重复检测（文档级 + 段落级） |
| `minhash` | 行 3 | MinHash + LSH 近重复检测（word 5-gram，112 hash，Jaccard ≈ 0.72） |
| `ngram` | 行 4 | Dolma N-gram 模糊重复（需要 dolma 二进制） |
| `semdedup` | 行 5 | 语义重复：bge-m3 embedding + 余弦相似度聚簇（eps=0.07 → cos≥0.93） |
| `stats` | 行 6 | 汇总重复率 / 来源分组 / epoch 等效次数 |

> 行 5（语义去重）原计划 NeMo SemDeDup + FAISS GPU；本机为 **MetaX C500**（非 NVIDIA），faiss-gpu/nemo 装不上。
> 改用本地 **bge-m3** + `transformers`/`torch` 在 MetaX GPU 上编码，分块算精确余弦相似度做聚簇，零额外依赖。

## 输出格式

### exact（per_doc）
```json
{
  "scores": {"doc_hash": "a3f2...", "dup_doc_count": 3,
             "para_dup_ratio": 0.25, "para_dup_count": 2},
  "flags": {"is_exact_dup": true, "has_para_dups": true}
}
```

### minhash（per_doc）
```json
{
  "scores": {"jaccard_max": 0.84, "near_dup_count": 2},
  "flags": {"is_near_dup": true}
}
```

### semdedup（per_doc）
```json
{
  "scores": {"max_cos": 0.97, "cluster_id": 12, "cluster_size": 4, "is_representative": false},
  "flags": {"is_semantic_dup": true}
}
```

- `is_semantic_dup`：属于 size>1 的语义簇且非代表（即可删的冗余）；簇代表 `is_representative=true`、不计为重复。
- `max_cos`：与本文档簇代表的余弦相似度。

> **阈值标定**：eps=0.07（cos≥0.93）是 SemDeDup 原论文默认，抓"近乎逐字改写"。该阈值随 embedding 模型变化需重标。
> UFW-L3 en multi_style 2000 条样本上 top-cos 最高仅 0.876（p99=0.77），cos≥0.93 命中 **0%** —— 这说明该批已去重良好，是有效结论而非 bug。
> 要抓"同义改写级"语义重复，放宽到 eps=0.15（cos≥0.85，约 0.3% 命中），代价是同主题不同内容可能被误判，需人工核查高 `max_cos` 簇。

## 依赖

```
# 标准库 + numpy（exact + minhash + ngram 无额外依赖）
pip install dolma            # ngram 子命令（可选，当前为自实现 in-batch 审计，不强制）
# semdedup：transformers + torch（已在核心依赖中）；embedding 模型 bge-m3 本地已有
```

## 运行示例

```bash
# mock 数据
PYTHONPATH=. python stages/dedup/run.py exact \
  --input data/mock.jsonl --dataset mock --input-format jsonl

PYTHONPATH=. python stages/dedup/run.py minhash \
  --input data/mock.jsonl --dataset mock --input-format jsonl

# UFW-L3
PYTHONPATH=. python stages/dedup/run.py exact \
  --input /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3/multi_style/part-00000-f36f5a53-4a77-434b-b9bc-67ed69b93fe2-c000.snappy.parquet \
  --dataset ufw_en_l3 --max-docs 5000

# 语义重复（GPU；中英都用 bge-m3）
PYTHONPATH=. python stages/dedup/run.py semdedup \
  --input data/mock.jsonl --dataset mock --input-format jsonl \
  --model /mnt/public/model/bge-m3 --device cuda

PYTHONPATH=. python stages/dedup/run.py semdedup \
  --input /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3/multi_style/part-00000-f36f5a53-4a77-434b-b9bc-67ed69b93fe2-c000.snappy.parquet \
  --dataset ufw_en_l3 --config configs/stage4.yaml --max-docs 2000
```
