# Stage 5: 污染检测（Contamination Detection）

检测预训练语料与公开 benchmark 之间的重叠，防止评测分数虚高。

## 子命令

| 子命令 | 说明 | 关键参数 |
|--------|------|----------|
| `exact` | 精确污染检测（文档/段落级 MD5 哈希对比） | `--max-docs` |
| `code-near` | 代码近重复检测（字符 5-gram MinHash） | `--ngram-size`, `--jaccard-threshold` |
| `code-ast` | 代码 AST 结构污染（tree-sitter 指纹） | languages / threshold 在 yaml |

## Benchmark 配置

在 `configs/stage5.yaml` 的 `benchmarks.datasets` 段定义。任何 HuggingFace dataset 都可以接入：

```yaml
benchmarks:
  cache_dir: /mnt/public/data
  datasets:
    - name: cais/mmlu        # HF dataset 名
      subset: all            # 可选 subset
      split: test
      text_field: question   # 用作 text 的字段
      label: mmlu            # 自定义标签
    - name: openai/openai_humaneval
      split: test
      text_field: prompt
      code_field: canonical_solution  # 代码字段（code-near/code-ast 用）
      label: humaneval
```

## 输入输出

- 输入：Parquet/JSONL，通过 `src/reader.py` 标准化
- 输出：`outputs/stage5/{dataset}_{timestamp}/{subcommand}/`
  - `per_doc.jsonl` — 每行 `{"doc_id", "scores": {...}, "flags": {...}}`
  - `summary.json` — 聚合统计

## 运行示例

```bash
# 精确污染（MMLU benchmark）
PYTHONPATH=. python stages/contamination/run.py exact \
  --input /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3/multi_style/ \
  --dataset ufw_en_l3 --config configs/stage5.yaml --max-docs 500

# 代码近重复
PYTHONPATH=. python stages/contamination/run.py code-near \
  --input data/mock.jsonl --dataset mock --input-format jsonl --max-docs 50

# AST 结构污染
PYTHONPATH=. python stages/contamination/run.py code-ast \
  --input data/mock.jsonl --dataset mock --input-format jsonl --max-docs 50
```

## 依赖

- `datasets` — HuggingFace datasets 库（benchmark 加载）
- `tree-sitter` + `tree-sitter-python` — AST 解析（code-ast 子命令）
- 其余为项目已有依赖
