# Stage 9: 长上下文（训练配置审计）

## 评估目标

审计 Megatron-LM 训练配置，验证 packing 边界设置是否正确，防止跨文档注意力干扰。

## 子命令

| 子命令 | 功能 | 说明 |
|--------|------|------|
| `config-audit` | 三参数配置验证 | 检查 reset_position_ids / reset_attention_mask / eod_mask_loss |

## 与其他 stage 的差异

本 stage **不处理文档数据**，而是审计训练配置文件。使用 `--config-file` 代替 `--input`，不支持 `--max-docs` / `--input-format`。

## 检查参数

| 参数 | 期望值 | 说明 |
|------|--------|------|
| `reset_position_ids` | True | 在文档边界重置 position ID |
| `reset_attention_mask` | True | 在文档边界重置 attention mask |
| `eod_mask_loss` | True | 在 EOD token 处 mask loss |

## 支持的配置格式

- YAML（优先尝试）
- JSON
- 文本/Shell 脚本（正则提取 `--flag`、`key=value`、`key: value`）

## 输出

- `per_doc.jsonl` — 单条记录（doc_id = 配置文件名）
- `summary.json` — 各参数 found/value/valid 状态

## 运行示例

```bash
PYTHONPATH=. python stages/longctx/run.py config-audit \
  --config-file data/mock_megatron_config.yaml --dataset mock
```
