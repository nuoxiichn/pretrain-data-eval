# 参考报告

仓库只提交经过整理、可公开引用的 Markdown 报告和必要的小型汇总，不提交逐文档命中、
原始语料、大模型输出或可能含 PII/secret 的片段。

| 报告 | 用途 | 证据范围 |
|---|---|---|
| [Ultra-FineWeb-L3 画像](ufw_l3.md) | 中英文合成预训练语料的主参考 | 全量轻量指标 + 有界重指标抽样 |
| [OpenWebText 与 Raw CC 对照](control_openwebtext_rawcc.md) | 检查部分指标方向性 | 6 个维度的方向性对照，不验证 recall |
| [The Stack 代码能力验证](thestack_code.md) | 检查代码 PII、secret、污染和 parsability | 小样本能力验证，不是数据集体检 |
| [Stage 11 最终实验与决策](stage11_final_report.md) | 汇总 Anchor、Balanced 和 data-conditioning 全部验证链 | 三种方法均为 Production No-Go，Stage 11 已退役 |
| [DataDecide 小模型排序复现](datadecide_reproduction_report.md) | 检查小模型 benchmark 排序能否预测 1B 数据 recipe 排序 | 官方 25-recipe 矩阵回算 + 公开权重复评 + 本地 20M 训练 |

报告中的命中率不是通用阈值。数据分布、模型、语言、样本设计或配置变化后必须重新校准。
