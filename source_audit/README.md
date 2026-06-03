# Stage 1：来源审计 + 时间属性

**类型**：📋 审计阶段  
**目录**：`source_audit/`  
**配置**：`configs/stage1.yaml`

## 评估目标

对输入语料做基础画像：规模、长度分布、域名/来源/语种分布、时间字段完整性，以及许可证与版权信息审计。不修改数据，输出审计报告供人工决策。

## 子命令

| 子命令 | pipeline 行 | 描述 |
|--------|------------|------|
| `stats` | 行 1 | 文档数/字符数/token 数、长度分桶、域名/来源/语种分布、时间字段完整性 |
| `license` | 行 2 | ScanCode 许可证与版权检测（SPDX 表达式 + 置信度 + 行号） |
| `snapshot` | 行 3 | DataTrove `executor.json` 快照记录（pipeline 可复现性） |

## 输入格式

JSONL，每行一个文档（字段约定见 `src/reader.py`）：

```json
{"doc_id": "...", "text": "...", "source": "...", "url": "...", "timestamp": "...", "language": "...", "meta": {}}
```

`url`、`timestamp`、`language` 可为 `null`；`meta` 可为空 `{}`。

## 输出格式

`outputs/stage1/{dataset}_{timestamp}/`

```
stats/
  summary.json       # 聚合统计（规模、分布、时间完整性）
  per_doc.jsonl      # 每文档 {"doc_id":..., "scores":{...}, "flags":{...}}
license/
  summary.json       # 许可证类型分布、命中率
  per_doc.jsonl      # 每文档许可证命中列表 + 版权持有人
snapshot/
  executor.json      # DataTrove pipeline 参数快照原样存档
```

## 依赖

| 子命令 | 依赖 |
|--------|------|
| `stats` | `numpy`（core deps） |
| `license` | `scancode-toolkit`（需单独安装） |
| `snapshot` | 无额外依赖 |

## 运行

```bash
# 开发模式安装（让 src/ 包可导入）
pip install -e .

# 基础统计
python source_audit/run.py stats --input data/mock.jsonl --dataset mock

# 使用 HF tokenizer（更精确）
python source_audit/run.py stats --input data/mock.jsonl --dataset mock \
    --config configs/stage1.yaml  # 在 yaml 中设置 tokenizer.backend=hf

# 许可证扫描（需已安装 scancode-toolkit）
python source_audit/run.py license --input data/mock.jsonl --dataset mock

# executor.json 快照
python source_audit/run.py snapshot --executor-json path/to/executor.json --dataset mock

# 共享输出目录（方便三步一起运行）
OUT=outputs/stage1/mock_$(date +%Y%m%d_%H%M%S)
python source_audit/run.py stats   --input data/mock.jsonl --dataset mock --output-dir $OUT/stats
python source_audit/run.py license --input data/mock.jsonl --dataset mock --output-dir $OUT/license
python source_audit/run.py snapshot --executor-json path/to/executor.json --dataset mock --output-dir $OUT/snapshot
```

## 注意

- `stats` 默认使用空格分词（速度快、适合欧语）；CJK 语料建议在配置中设置 `tokenizer.backend: hf`
- `license` 逐文档写临时文件后调用 ScanCode Python API，大语料耗时较长，可用 `--max-docs` 限制扫描数量
- `snapshot` 只做存档，不验证 executor.json 内容有效性
