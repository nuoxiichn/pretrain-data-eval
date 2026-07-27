# 测试与验收

## 默认门禁

`make check` 在无模型、无 GPU、无 Gitleaks/DataTrove 的普通 Python 环境执行：

- Ruff 未定义名称和未使用导入检查；
- Python 源码编译和 shell 语法检查；
- 输入、per-doc、summary JSON Schema 契约；
- JSONL/Parquet reader、字段映射、错误定位和随机抽样；
- stats、extraction、exact dedup、exact contamination、config audit 的 CLI 集成；
- batch 聚合后的 schema 与计数一致性；
- safety toxicity、Binoculars、tokenization 的 fake 模型编排；
- MinHash、污染注入、抽取残留和重复真值回归；
- 公开 Markdown 的本地链接和配置完整性。

这些测试用于证明“代码按当前定义工作、输出满足契约、关键路由没有静默漂移”。它们不证明
外部模型的 precision/recall，也不覆盖真实 GPU kernel、模型权重损坏、Gitleaks 二进制行为、
DataTrove 分布式 shuffle 或 TB 级资源上限。

## 目标环境验收

发布或更换模型/机器后，至少执行：

1. `make check`；
2. 所有计划启用的 CLI `--help` 和 10-100 条真实格式 smoke；
3. PII 的 Presidio + 对应 spaCy 模型 smoke；
4. Gitleaks 版本检查和注入 fake secret 的命中测试；
5. toxicity、embedding、STEM、Binoculars 各运行一批含阳性/阴性的校准集；
6. 安装的 tree-sitter grammar 逐语言解析最小合法样例；
7. DataTrove 按 `signature -> buckets -> cluster -> aggregate` 运行 500 条 smoke，验证最终
   `per_doc.jsonl` / `summary.json` schema；
8. 100K 以上规模记录 wall time、峰值内存/显存和产物大小，再决定生产并发。

任何真实模型、prompt、tokenizer、grammar、Gitleaks 或 DataTrove 版本变化都应重跑对应验收，
不能只依赖 Python 单元测试。

## 当前自动化边界

覆盖率是定位未测代码的工程信号，不是检测器置信度。当前共享 reader/schema/sampling 已有较高
离线覆盖；大规模 cascade、远程 benchmark、真实 PII/LLM 和 DataTrove 路径依赖外部资产，
留在目标环境验收。其统计效力另见[置信度、效力与局限](statistical_limitations.md)。
