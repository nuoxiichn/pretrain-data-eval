# Ultra-FineWeb-L3 数据画像

评估对象为 Ultra-FineWeb-L3（UFW-L3）英文与中文的 `multi_style`、`qa` 子集。轻量指标
覆盖约 1.06B 文档；模型指标采用有界样本。本文只保留当前采用的方法和结果口径。

## 数据规模

| 子集 | Parquet 文件 | 文档数 |
|---|---:|---:|
| EN multi_style | 552 | 378M |
| EN qa | 616 | 320M |
| ZH multi_style | 286 | 204M |
| ZH qa | 310 | 156M |
| 合计 | 1,764 | 1.06B |

Qwen3-4B-Base tokenizer 下，EN 约 451B token、ZH 约 241B token，合计约 692B token。
平均文档 token 数为 EN 647、ZH 669。数据没有 URL 和时间字段，因此不能从本数据集审计
原始域名与时效性。

## 结果概览

| 维度 | 覆盖 | 结果 | 解释 |
|---|---|---|---|
| PII | EN/ZH 各 48K | EN 0.227%，ZH 1.144% | Presidio 候选率，仍需人工确认 |
| Secret | EN 698M / ZH 360M | EN 0.0018%，ZH 0.0012% | 多数为 generic-api-key，不能视为有效凭据率 |
| Toxicity | EN/ZH 各 5K | EN high-risk 0.14%，ZH 0.06% | XLM-R 召回后由 Qwen Judge 判 promote |
| 抽取残留 | EN 698M / ZH 360M | EN 2.19%，ZH 1.48% | 主要是登录、版权和社交 boilerplate |
| 语种 | EN/ZH 各 500 | EN 100%，ZH 99.6% 主语种一致 | 样本较小，只说明明显混入较少 |
| Gopher/C4 | EN/ZH 各 5K | Gopher EN 0.14% / ZH 1.18%；C4 EN 2.32% / ZH 2.66% | 规则失败率，不等于训练价值差 |
| Exact 重复 | EN 698M / ZH 360M | EN 0.0053%，ZH 0.0016% | 按当前规范化和运行口径 |
| N-gram 重复 | EN 698M / ZH 360M | EN 1.069%，ZH 0.037% | 对段落结构和参数敏感 |
| MinHash | 每语种 100K | EN 0%，ZH 0.006% | 4 个文件 head 样本，不代表总体无近重复 |
| 污染 cascade | EN 584K / ZH 298K | L1/L2 red 为 0；L3 候选需 Judge/人工复核 | 约覆盖总体 0.08%，benchmark 注册表有边界 |
| Binoculars | EN 5K | 阈值命中 0.02% | 模型对未在 UFW 分布校准，不作通过结论 |
| STEM | 每语种 25K 分层样本 | STEM EN 58.5% / ZH 46.6%；高难 EN 3.02% / ZH 0.74% | EAI-Distill 分类器画像 |

## 关键观察

- UFW-L3 在当前规则下的精确重复、secret 和明显语种混入较低。
- 抽取残留集中在网页模板内容，EN 高于 ZH；该结果可用于定位清洗改进，但不应把所有命中
  文档自动删除。
- 中文在 Qwen3 tokenizer 下平均 token 数与英文接近；token 成本结论绑定该 tokenizer。
- STEM 分类显示中英文主题和难度分布不同，但跨语言比较同时包含数据组成和分类器校准差异。
- Binoculars 使用的阈值来自其他模型设置。该结果只保留为分数分布参考，不能证明数据是
  人写、不能证明合成方式，也不能作为质量绿灯。

## 污染结果边界

Cascade 跨全部 shard 每文件取固定数量记录，共检查 EN 584K、ZH 298K 文档，与 12 个对齐
benchmark 比较。该方案提高了 shard 覆盖，但不是总体等概率样本；每 shard 固定配额会改变
不同 shard 的权重。Exact/MinHash 没有发现 red，embedding 层召回的候选依赖 Qwen Judge
和人工校准。深度改写、翻译、答案泄漏及未注册 benchmark 仍可能漏检。

因此报告支持的表述是“在当前样本、benchmark 注册表和检测能力内，未观察到高比例明显
污染”，而不是“数据无污染”。

## 统计与工程限制

- PII、toxicity、语言、质量、MinHash、污染、Binoculars 和 STEM 均有不同抽样设计，比例
  不能假定具有相同统计精度。
- 500、5K 样本不足以稳定估计万分之一级低频风险；零命中尤其不能解释为总体为零。
- 检测器 precision/recall 尚未在覆盖全部目标语言和领域的统一标注集上估计。
- 全量 exact/ngram 的全局或逐 shard 状态必须随运行记录确认，不能只从聚合比例推断。
- 本报告不建立通用红绿灯阈值，也不预测使用该数据训练后的模型收益。

基于现有证据，UFW-L3 可作为后续数据审计的参考分布，但任何新数据集都必须保留原始指标、
重新检查阈值并结合训练后模型评测作最终决策。
