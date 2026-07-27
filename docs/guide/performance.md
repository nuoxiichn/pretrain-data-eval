# 性能与规模

下表总结当前验证环境中的量级。不同文本长度、模型、存储和硬件会显著改变吞吐；外推不是
服务等级承诺。正式运行应记录实际资源，而不是复用本页数字。

| 任务 | 已验证规模/资源 | 推荐策略 | 主要瓶颈 |
|---|---|---|---|
| S1 stats + S10 | UFW-L3 约 1.06B docs 全量 | 合并一次流式扫描 | tokenizer 与存储吞吐 |
| S2 PII | EN/ZH 各 48K 抽样 | 50K-100K 分层/随机样本 | spaCy/Presidio CPU |
| S2 secrets | UFW-L3 约 1.06B docs 全量 | 可按 shard 并行全量 | 临时文件与 Gitleaks 进程 |
| S2 toxicity | EN/ZH 各 5K，XLM-R + 7B Judge | 先 10K-50K 抽样 | GPU 推理和召回 chunk 数 |
| S3 extraction | UFW-L3 约 1.06B docs 全量 | 按 shard 流式并行 | Python regex 与 I/O |
| S4 exact/ngram | UFW-L3 全量逐 shard | 全量需明确“单 shard”或“全局”语义 | hash 状态内存、段落数 |
| S4 MinHash | 自实现 100K；DataTrove 面向 TB 级 | 小样本校准后用分布式实现 | signature 产物与 shuffle |
| S5 cascade | UFW-L3 882K 跨 shard 样本 | L1/L2 可扩大，L3/LLM 有界运行 | embedding 索引和 GPU |
| S7 Binoculars | 5K docs 约 8 min，双 7B fp16 约 28GB | 5K-50K 校准样本 | 双模型显存与前向计算 |
| S8 stem | 每语种 25K 约 79-81 min | 分层抽样 | 自回归分类模型推理 |
| S9 config-audit | 单配置秒级 | 全量配置文件 | 可忽略 |

## 规模口径

- **可读取**：reader 能流式遍历，不代表指标实现是常量内存。
- **可全量审计**：已在目标规模完成整个子命令，并包含跨 shard 语义。
- **可抽样审计**：总体很大，但只对有界样本执行重模型；结论受抽样效力限制。
- **可分布式运行**：任务能拆分，但聚合是否保持全局语义仍需单独验证。

对于 10T 级数据，推荐全量运行 stats、tokenization、轻量残留和按 shard secret；PII、毒性、
Binoculars、STEM 与 embedding 指标使用有统计设计的抽样。精确/近似重复如果需要全局结论，
必须使用跨 shard 全局状态或 DataTrove 等分布式实现，不能把各 shard 比例简单相加冒充全局
重复率。

## 运行记录模板

正式报告应为每个子命令记录：输入 docs/bytes/tokens、shard 数、抽样协议、CPU/内存、GPU/
显存、wall time、吞吐、峰值资源、产物大小、失败 shard 和重试次数。当前 CLI 只在部分 summary
中记录耗时，统一资源采集仍是工程限制，报告必须人工补齐。
