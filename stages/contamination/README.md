# Stage 5: 污染检测（Contamination Detection）

检测预训练语料与公开 benchmark 之间的重叠，防止评测分数虚高。

## 子命令

| 子命令 | 说明 | 关键参数 |
|--------|------|----------|
| `exact` | L1 精确污染（文档/段落级 MD5 哈希对比） | `--max-docs` |
| `near` | L2 通用文本 MinHash 近重复（char n-gram + LSH） | `--ngram-size`, `--jaccard-threshold` |
| `embed` | L3 BGE-m3 + FAISS 语义污染 | `--cos-threshold`, `--top-k` |
| `cascade` | **三层 cascade 编排（推荐）**：L1 → L2 → L3 + 三色 verdict | `--index-dir` |
| `code-near` | 代码近重复（字符 5-gram MinHash，仅对 benchmark 的 code_field） | `--jaccard-threshold` |
| `code-ast` | 代码 AST 结构污染（tree-sitter 指纹） | yaml `code_ast` |

## Cascade 三层架构

```
doc → L1 exact (MD5 hash)
        ├─ hit                          → red
        └─ miss → L2 MinHash near-dup (char 5-gram + LSH, sliding window)
                    ├─ jaccard ≥ 0.90   → red
                    ├─ jaccard < 0.20   → green
                    └─ ∈ [0.20, 0.90)  → L3 BGE-m3 embedding (FAISS top-k)
                                            ├─ cos ≥ 0.85    → red
                                            ├─ cos ∈ [0.70, 0.85) → yellow
                                            └─ cos < 0.70    → green
```

### 关键设计点
- **L2 sliding window**：UFW doc 平均 745 chars，benchmark 仅 ~60 chars，整 doc-level Jaccard 必然 ≈ 0（分母被 doc n-gram 撑大）。默认 window=150 / stride=75（与 benchmark p90 长度匹配）。
- **LSH s-curve**：默认 32 bands × 4 rows，使中低 Jaccard 候选仍有机会进入 L3 复核。
- **预建索引**：`bench_index.py build` 一次性把 hash + MinHash 签名 + BGE-m3 embedding 落盘到 `index_dir`，cascade 子命令通过 `--index-dir` 或 yaml `benchmarks.index_dir` 加载，避免每文件重复编码（~1 min 节省）。
- **三色 verdict**：`red` 高度疑似污染、`yellow` 需人工复核、`green` 正常。聚合层 `aggregate_batch.py merge_contamination_cascade` 输出红绿灯分布。

## Benchmark 配置

`configs/stage5.yaml` 注册 12 个与实际评测清单对齐的 benchmark。默认路径位于本项目验证
环境的 `/mnt/public/data/contamination_v3_benchmarks/eval_aligned/`，迁移环境时必须改为本地
可读路径并重新构建索引。

新 benchmark 接入两种方式：
- 本地文件：`{path: /path/to/file.jsonl, text_field: question, label: my_bench}`
- HuggingFace：`{name: cais/mmlu, subset: all, split: test, text_field: question, label: mmlu}`

中文 benchmark 通过 `scripts/build_zh_benchmarks_jsonl.py` 从 opencompass_data 缓存抽取为统一 JSONL。

## 输入输出

- 输入：Parquet/JSONL，通过 `pretrain_data_eval/reader.py` 标准化
- 输出：`outputs/stage5/{dataset}_{timestamp}/{subcommand}/`
  - `per_doc.jsonl`
  - `summary.json` —— cascade 含 `verdict_distribution` / `cost_breakdown` / `cross_lingual_docs` / `per_benchmark_red_hits`

## 运行示例

```bash
# 一次性构建 cascade 索引（hash + MinHash + BGE-m3 embedding）
PYTHONPATH=. python stages/contamination/bench_index.py build --config configs/stage5.yaml

# Cascade 三层（推荐）
PYTHONPATH=. python stages/contamination/run.py cascade \
  --input /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_zh_l3/multi_style/part-00000-*.parquet \
  --dataset ufw_zh_l3 --config configs/stage5.yaml --max-docs 1000

# 仅 L2 单独跑
PYTHONPATH=. python stages/contamination/run.py near \
  --input data/mock.jsonl --dataset mock --input-format jsonl --max-docs 50

# 仅 L3 单独跑
PYTHONPATH=. python stages/contamination/run.py embed \
  --input data/mock.jsonl --dataset mock --input-format jsonl --max-docs 50

# 全量 batch（每文件抽 500 docs head-sample）
bash scripts/contamination_cascade_batch.sh zh 500

# 状态检查
bash scripts/contamination_status.sh

# 聚合
python scripts/aggregate_batch.py outputs/stage5/ufw_zh_l3/cascade_sample500
```

## 依赖

```bash
python -m pip install -e .                 # exact、near、本地 JSONL/Parquet benchmark
python -m pip install -e '.[benchmarks]'   # 远程 HuggingFace benchmark
python -m pip install -e '.[code]'         # code-ast
python -m pip install -e '.[gpu]'          # embed/cascade 的 BGE-M3 + FAISS
```

方法边界见 [`docs/guide/metrics.md`](../../docs/guide/metrics.md)，已被替代的设计和调研记录
位于 [`docs/archive/`](../../docs/archive/README.md)。
