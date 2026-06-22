# pretrain-data-eval

基模预训练数据评估链路。从第三方、多维度视角评估清洗/合成好的数据质量，输出审计报告，辅助判断一份数据**是否值得用于预训练**。

- **定位**：只读审计。不清洗、不去重、不修改数据，只产出信号与报告。TB 级清洗/去重 pipeline 是独立工程。
- **当前**：在开源数据集 **Ultra-FineWeb-L3** 上跑通全链路，为后续评估自研合成数据（~10T）提供参考基线。

## 维度全景

10 个 stage 的指标 / 工具 / 状态见 **[`pipeline_overview.md`](pipeline_overview.md)**（含执行编排：依赖、并行分组、启动顺序、抽样策略）。

## 仓库结构

```
stages/      所有评估 stage（每个 = run.py + utils.py + README.md）
src/         跨 stage 共享：reader.py（输入适配）/ schema.py（输出 schema）
configs/     每 stage 一份 YAML 配置（stage1~10.yaml）
scripts/     批量/断点续跑脚本 + 巡检 + 一次性工具
outputs/     产物（git 忽略）
data/        本地 mock/样本（git 忽略）
```

## 快速开始

```bash
pip install -e .          # 让 src/ 与 stages/ 可导入

# mock 数据冒烟（任意 stage）
PYTHONPATH=. python stages/dedup/run.py exact \
  --input data/mock.jsonl --dataset mock --input-format jsonl

# 真实数据务必带 --max-docs 限量
PYTHONPATH=. python stages/safety/run.py pii \
  --input <parquet> --dataset ufw_en_l3 --config configs/stage2.yaml --max-docs 500
```

每个 stage 的子命令、输入输出格式、依赖见各自 README。

## 输入 / 输出契约

- **输入**：统一经 `src/reader.py::read_documents`，标准 `Document`（`doc_id/text/source/url/timestamp/language/meta`），支持 JSONL / Parquet。
- **输出**：统一经 `src/schema.py`。每个子命令产出 `per_doc.jsonl`（每行 `{doc_id, scores, flags}`）+ `summary.json`（聚合统计），落在 `outputs/stage<N>/{dataset}_{ts}/{subcommand}/`。
