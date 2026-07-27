# Stage 4：重复度分析

Stage 4 对当前命令输入范围内的文档执行只读重复审计，不输出清洗后的语料。

## 子命令

| 子命令 | 方法 | 适用范围 |
|---|---|---|
| `exact` | 规范化后 MD5，文档级和段落级 | 精确重复；状态集合随规模增长 |
| `minhash` | 5-gram MinHash + LSH | 单进程小中规模近重复 |
| `ngram` | 段落 shingle 重叠 | 模板化或局部段落重复 |
| `repetition` | 重复行/段落与 2–10 gram 集中度 | 单文档内部循环、复制和模板重复 |
| `semdedup` | BGE-M3 embedding + 分块余弦聚簇 | 有界样本的语义近重复 |

跨大量 shard 的全局 MinHash 使用 [`stages/dedup_datatrove`](../dedup_datatrove/README.md)。
逐 shard 分别运行只能得到 shard 内重复率，不能发现 shard 间重复。

## 输入输出

输入由 `src.reader` 统一读取。输出为带公共 schema 头的 `per_doc.jsonl` 和 `summary.json`。
各子命令的关键 summary 字段：

- `exact`：`exact_dup_docs/_pct`、`para_dup_docs/_pct`、唯一 hash 数；
- `minhash`：`near_dup_docs/_pct`、候选 pair、簇和 `jaccard_max`；
- `ngram`：文档/段落 contaminated 数和比例；
- `repetition`：高重复文档比例、触发阈值分布和逐文档重复指标；
- `semdedup`：语义重复文档数、簇数和最大簇。

## 运行

```bash
PYTHONPATH=. python stages/dedup/run.py exact \
  --input data/mock.jsonl --dataset mock --input-format jsonl

PYTHONPATH=. python stages/dedup/run.py minhash \
  --input data/mock.jsonl --dataset mock --input-format jsonl \
  --num-hashes 64 --num-bands 8 --band-size 8

PYTHONPATH=. python stages/dedup/run.py ngram \
  --input data/mock.jsonl --dataset mock --input-format jsonl

PYTHONPATH=. python stages/dedup/run.py repetition \
  --input data/mock.jsonl --dataset mock --input-format jsonl

PYTHONPATH=. python stages/dedup/run.py semdedup \
  --input data/mock.jsonl --dataset mock --input-format jsonl \
  --max-docs 1000 --device cpu
```

所有抽样命令默认 `--sample-mode random --seed 42`。`exact` 和 `ngram` 的正式全量语义不应
用抽样结果替代，因为未采样到的重复对不可见。

## 参数与限制

- MinHash/LSH 是概率召回，`num_hashes`、band 划分、shingle 和阈值共同决定误报与漏检。
- `hot_bucket_cap` 会跳过超大桶以控制二次复杂度，可能漏掉高频模板候选。
- 中文 dominant 文本使用字符 shingle；不同分词协议的结果不可直接比较。
- SemDedup 的阈值绑定 BGE-M3、截断长度和数据领域，同主题不等于重复，命中必须抽检。
- 当前实现会物化输入或维护增长状态；大数据不能仅因 reader 流式就假定常量内存。

配置见 [`configs/stage4.yaml`](../../configs/stage4.yaml)，统计解释边界见
[`docs/guide/statistical_limitations.md`](../../docs/guide/statistical_limitations.md)。
