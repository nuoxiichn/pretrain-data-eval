# 输入输出契约

## 输入

`pretrain_data_eval.reader.read_documents` 接受单个 JSONL、单个 Parquet 或 Parquet 分片目录，并把每条
记录归一为统一文档对象。

| 字段 | 类型 | 必需 | 规则 |
|---|---|---|---|
| `doc_id` | string | 是 | 数据集内唯一且非空；整数 ID 会转为字符串 |
| `text` | string | 是 | 允许空字符串，但不得为 null 或其他类型 |
| `source` | string/null | 否 | 数据来源或生成管线标签 |
| `url` | string/null | 否 | 原始 URL |
| `timestamp` | string/null | 否 | 推荐 ISO-8601；当前 reader 不解析日期语义 |
| `language` | string/null | 否 | 推荐 ISO 639 语言码 |
| `meta` | object | 否 | 任意附加元数据，默认 `{}` |

源字段通过 `input.field_map` 重命名后再校验。未映射且不属于标准字段的内容并入 `meta`；
`input.path_meta` 只填充当前值为 null 的可选字段，不覆盖记录中已有值。

```yaml
input:
  format: parquet
  batch_size: 500
  glob: "*.parquet"
  field_map:
    uid: doc_id
    content: text
  path_meta:
    language:
      corpus_zh: zh
```

目录模式按路径排序、递归读取匹配的 Parquet 文件。JSONL 目录尚不支持。无匹配文件、
无效 JSON、缺少必需字段或字段类型错误都会立即报错，不会静默生成空 ID。

机器契约见 [`input_document.schema.json`](../../schemas/input_document.schema.json)。

## Per-document 输出

`per_doc.jsonl` 每行一个对象：

```json
{
  "schema_version": "1.0.0",
  "artifact_type": "per_doc",
  "doc_id": "doc-001",
  "scores": {"pii_count": 1},
  "flags": {"has_pii": true}
}
```

`scores` 保存可分析的数值、分类或结构化明细；`flags` 只保存布尔判断。命中片段可能包含
敏感信息，正式发布前必须脱敏。契约见
[`per_doc.schema.json`](../../schemas/per_doc.schema.json)。

## Summary 输出

`summary.json` 的公共头为：

```json
{
  "schema_version": "1.0.0",
  "artifact_type": "summary",
  "total_docs": 1000,
  "exact_dup_docs": 12,
  "exact_dup_pct": 0.012
}
```

`schema_version` 采用语义版本：破坏字段含义或移除字段时提升 major；只增加可选字段提升
minor；不改变契约的修复提升 patch。其余业务字段由指标目录和 Stage README 定义。契约见
[`summary.schema.json`](../../schemas/summary.schema.json)。

## 兼容与可复现性

- 消费者必须先检查 `schema_version` major，不应只按文件名猜测格式。
- 新代码写出前校验公共字段；历史无版本产物不自动伪装为当前 schema。
- 正式运行应保留实际配置、Git commit、模型版本、抽样参数和资源记录。当前 Stage CLI
  尚未统一生成 run manifest，因此这些信息必须随正式报告记录，不能从 summary 猜测。
- `summary.json` 表示一个子命令的一次运行，不能把不同配置或不同样本的 summary 直接相加。
