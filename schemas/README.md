# Machine-readable schemas

本目录定义 `pretrain-data-eval` 的公共 JSON 契约：

- [`input_document.schema.json`](input_document.schema.json)：reader 归一后的标准文档；
- [`per_doc.schema.json`](per_doc.schema.json)：每行逐文档结果；
- [`summary.schema.json`](summary.schema.json)：所有子命令的公共 summary 头。

当前版本为 `1.0.0`。Python 写出路径由 `src.schema` 执行同等的基础校验；测试再使用
JSON Schema 验证样例和 CLI 产物。业务指标字段保留为开放对象，因为不同子命令的字段不同，
其含义由 [`docs/guide/metrics.md`](../docs/guide/metrics.md) 和 Stage README 约束。

兼容策略：消费者必须拒绝未知 major；同一 major 的新增业务字段应被忽略而不是报错。
