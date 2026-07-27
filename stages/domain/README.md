# Stage 8: 专项能力

## 评估目标

评估预训练数据在专项能力维度的特性：

- **代码可解析率**：用 tree-sitter 解析代码文档，统计 AST ERROR 节点，评估代码质量
- **学科分布 + 难度**：用 EAI-Distill-0.5b 推理打 FDC（杜威十进顶层）学科 + reasoning_depth + educational_level

## 子命令

| 子命令 | 功能 | 工具 |
|--------|------|------|
| `parsability` | 代码 AST 解析错误检测 | tree-sitter |
| `stem` | FDC 学科 + Bloom 难度 | EAI-Distill-0.5b (Qwen2.5-0.5B 微调) |

## parsability

**输出 scores**：`error_node_count, total_node_count, error_ratio`
**输出 flags**：`has_error`（有 ERROR/MISSING 节点）

注册语言包括 Python、JavaScript、TypeScript、Java、Go、C、C++、C#、Rust、Ruby 和 PHP。
只有安装了对应 `tree-sitter-<lang>` grammar 的语言才可用；`--language auto` 会按文档
`meta.lang` / `language` 选择 grammar，并把缺失 grammar 的记录标为 `unsupported_lang`。

## stem

**输出 scores**：
- `fdc_raw` — 模型原始 FDC 码（如 `"510"`），可能为 `null`
- `fdc_top_class` — 杜威顶层（`0/100/200/.../900`），下游聚合用此字段
- `fdc_top_label` — 顶层标签文字（如 `"Pure Science"`）
- `reasoning_depth` — `1–6`，可能为 `null`
- `educational_level` — `1–5`，可能为 `null`

**输出 flags**：
- `is_stem` — `fdc_top_class ∈ {0, 500, 600}`（General/Computer + Pure Science + Technology）
- `high_difficulty` — `reasoning_depth ≥ high_difficulty_threshold`（默认 4）
- `parse_failed` — 模型输出无法解析（FDC 无法识别）

**杜威顶层 10 类**（FDC top）：

| top | label |
|---|---|
| 000 | General/Computer |
| 100 | Philosophy |
| 200 | Religion |
| 300 | Social Sci |
| 400 | Language |
| 500 | Pure Science |
| 600 | Technology |
| 700 | Arts |
| 800 | Literature |
| 900 | History/Geography |

### 模型路径与下载

`stem` 子命令依赖本地权重 `/mnt/public/model/EssentialAI/eai-distill-0.5b/`。**该路径是项目共享只读区，需要手动下载一次**：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 本机直连 huggingface.co 走不通
huggingface-cli download EssentialAI/eai-distill-0.5b \
  --local-dir /mnt/public/model/EssentialAI/eai-distill-0.5b
```

模型 License: Apache-2.0。500M 参数，bfloat16 显存约 1GB；CPU 也能跑但每条 1-3s 量级。

### GPU 依赖

`torch` 和 `transformers` 位于 `gpu` 附加依赖。首次跑：

```bash
pip install -e .[gpu]
```

无 GPU 时 `device` 自动 fallback 到 cpu（仍能跑，慢一个量级）。

## 依赖

```bash
python -m pip install -e '.[code]'  # parsability 及全部已注册 grammar
python -m pip install -e '.[gpu]'   # STEM 模型推理
```

## 运行示例

```bash
# parsability
PYTHONPATH=. python stages/domain/run.py parsability \
  --input data/mock.jsonl --dataset mock --input-format jsonl

# stem（默认 GPU，无 GPU 自动用 CPU）
PYTHONPATH=. python stages/domain/run.py stem \
  --input data/mock.jsonl --dataset mock --input-format jsonl --max-docs 5

# stem on 真实样本
PYTHONPATH=. python stages/domain/run.py stem \
  --input /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3/multi_style \
  --dataset ufw_en_l3 --max-docs 50

# 覆盖参数
PYTHONPATH=. python stages/domain/run.py stem \
  --input data/mock.jsonl --dataset mock --input-format jsonl \
  --model-path /path/to/eai-distill-0.5b --batch-size 4 --device cuda
```
