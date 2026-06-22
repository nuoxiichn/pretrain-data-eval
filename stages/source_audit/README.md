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
```

## 依赖

| 子命令 | 依赖 |
|--------|------|
| `stats` | `numpy` |
| `license` | `scancode-toolkit`（需单独安装） |

## 运行

```bash
# 开发模式安装（让 src/ 包可导入）
pip install -e .

# 基础统计（默认 hf backend，Qwen3-4B-Base tokenizer）
python stages/source_audit/run.py stats --input data/mock.jsonl --dataset mock

# 一次扫描同时产出 stage1 stats + stage10 tokenize（生产推荐）
python stages/source_audit/run.py stats --input data/mock.jsonl --dataset mock \
    --coalesce-stage10 \
    --output-dir outputs/stage1/mock/stats \
    --stage10-output-dir outputs/stage10/mock/tokenize
# 或直接用批处理脚本（自带断点续跑）：
#   bash scripts/stats_tokenize_batch.sh zh

# 许可证扫描（需已安装 scancode-toolkit）
python stages/source_audit/run.py license --input data/mock.jsonl --dataset mock

# 共享输出目录（方便两步一起运行）
OUT=outputs/stage1/mock_$(date +%Y%m%d_%H%M%S)
python stages/source_audit/run.py stats   --input data/mock.jsonl --dataset mock --output-dir $OUT/stats
python stages/source_audit/run.py license --input data/mock.jsonl --dataset mock --output-dir $OUT/license
```

## 注意

- `stats` 默认 `tokenizer.backend: hf`，路径 `/mnt/public/model/huggingface/Qwen3-4B-Base`（vocab 151936）。`backend: words` 仅用于 mock smoke，对 CJK 严重失真，**不要在真实数据上用**。
- `length_buckets` 4K/8K/.../256K+ 是 **token 数桶**（不是 word 数），与训练侧 context length 口径一致。
- `--coalesce-stage10` 复用同一 tokenizer pass，stage10 配置（tokenizer 路径/阈值/batch_size）从 `--stage10-config`（默认 `configs/stage10.yaml`）读。
- `license` 逐文档写临时文件后调用 ScanCode Python API，大语料耗时较长，可用 `--max-docs` 限制扫描数量。
