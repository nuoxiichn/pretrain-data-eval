# 文档导航

公开文档只描述当前实现和可复用结论。旧方案、否决实验和一次性调研记录位于
[`archive/`](archive/README.md)，不作为用户契约或当前能力证明。

| 需要回答的问题 | 文档 |
|---|---|
| 支持哪些指标，各自能说明什么 | [指标目录](guide/metrics.md) |
| 输入字段和输出 JSON 如何定义 | [输入输出契约](guide/input_output_contract.md) |
| 抽样结果的置信度和局限如何解释 | [置信度、效力与局限](guide/statistical_limitations.md) |
| 每项实验需要多少资源、可扩展到多大 | [性能与规模](guide/performance.md) |
| 自动化测试保证什么、外部依赖如何验收 | [测试与验收](guide/testing.md) |
| 公开数据在本链路上的结果 | [参考报告](reports/README.md) |
| Stage 11 三种训练代理为何退役 | [Stage 11 最终实验与决策](reports/stage11_final_report.md) |

各子命令的安装依赖和完整参数仍由 `stages/<name>/README.md` 维护。机器可校验契约位于
[`schemas/`](../schemas/README.md)。
