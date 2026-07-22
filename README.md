# pretrain-data-eval

`pretrain-data-eval` 是一个面向基模预训练数据的第三方、只读质量审计工具。它读取
JSONL 或 Parquet 语料，从来源、安全、文本完整性、重复、评测集污染、领域属性和
tokenization 等维度输出可追踪的信号，帮助团队判断一批数据是否值得进入预训练。

本项目不清洗、不删除或改写输入数据，也不输出一个掩盖维度差异的“综合质量分”。
数据指标只能描述固定协议下观察到的风险，不能替代训练后的模型效果评测。

## 当前能力

| Stage | 维度 | 主要方法 | 推荐范围 |
|---|---|---|---|
| 1 | 规模、长度、来源、时间、许可证 | 流式统计、HF tokenizer、ScanCode | 统计可全量；许可证建议抽样 |
| 2 | PII、secret、毒性 | Presidio、Gitleaks、XLM-R 召回 + Qwen Judge | PII/毒性抽样；secret 可全量 |
| 3 | 抽取残留、语种、文本规则 | 自定义残留规则、lid.176、GlotLID、Gopher/C4 | 轻量项可全量 |
| 4 | 精确、近似、语义重复 | hash、MinHash/LSH、N-gram、embedding | 大规模 MinHash 用 DataTrove |
| 5 | 评测集污染 | exact、char-MinHash、BGE-M3、LLM 复审、代码 AST | benchmark 对齐后运行 |
| 7 | AI 生成文本信号 | Binoculars | 仅作校准后的抽样信号 |
| 8 | 代码可解析性、STEM 学科与难度 | tree-sitter、EAI-Distill | 按数据类型选用并抽样 |
| 9 | 长上下文训练配置 | Megatron 参数审计 | 不读取语料，秒级 |
| 10 | fertility、UNK、代码/LaTeX 膨胀 | HF tokenizer | 可与 Stage 1 合并扫描 |
| 11 | 微型训练代理（已退役） | Anchor、Balanced、data-conditioning | 三种方法均为 Production No-Go；不得调度或消费输出 |

Stage 编号沿用项目早期实验编号，因此没有 Stage 6。被否决的通用质量评分实验保存在
[归档](docs/archive/quality_metric_rejection.md)，不属于当前能力。

完整指标、输出字段、适用边界和依赖见[指标目录](docs/guide/metrics.md)。Stage 11 三种方法的实验、
失败控制和最终退役决定见[统一终局报告](docs/reports/stage11_final_report.md)。

## 选择评估项

不要机械地执行所有 Stage。先按数据类型选择有解释力的指标：

| 数据类型 | 必测 | 选测或限制 |
|---|---|---|
| 通用文本 | S1 stats、S2 safety、S3、S4、S5、S10 | S7 必须先校准；S8 parsability 不适用；S11 已退役 |
| 代码 | S1 stats、S2 code-PII/secrets、S4、S5 code、S8 parsability、S10 | 自然语言质量规则只用于注释类内容 |
| STEM/数学 | 通用文本项、S5、S8 stem、S10 | Gopher/C4 可能把公式密集文本误判为低质量 |
| 多语言 | S1、S3 GlotLID/langcross、S4、S10 | 模型覆盖外不可外推；S11 已退役 |
| 合成文本 | 通用文本项、生成元数据画像 | Binoculars 命中率不等于合成数据质量 |

## 快速开始

要求 Python 3.10 及以上。

```bash
python -m pip install -e .
python scripts/gen_mock_data.py

PYTHONPATH=. python stages/dedup/run.py exact \
  --input data/mock.jsonl \
  --dataset mock \
  --input-format jsonl \
  --output-dir outputs/smoke/exact
```

真实数据首次运行必须先限制样本量，并固定抽样种子：

```bash
PYTHONPATH=. python stages/safety/run.py pii \
  --input /path/to/data \
  --dataset example \
  --config configs/stage2.yaml \
  --max-docs 1000 \
  --sample-mode random \
  --seed 42
```

确认字段映射、依赖和输出后再扩大样本或执行全量。按实际 Stage 安装附加依赖：

| 能力 | 安装项 |
|---|---|
| 语言识别与 Gopher/C4 | `python -m pip install -e '.[text]'` |
| PII | `python -m pip install -e '.[safety]'` |
| 多语言 tree-sitter | `python -m pip install -e '.[code]'` |
| 远程 HuggingFace benchmark | `python -m pip install -e '.[benchmarks]'` |
| ScanCode 许可证 | `python -m pip install -e '.[license]'` |
| embedding、STEM、Binoculars、微型训练代理 | `python -m pip install -e '.[gpu]'` |
| Qwen/vLLM Judge | `python -m pip install -e '.[gpu,judge]'` |

DataTrove 分布式去重使用独立环境，见
[dedup_datatrove README](stages/dedup_datatrove/README.md)。模型文件和 Gitleaks 等外部工具
需要按各 Stage README 单独准备，配置中的 `/mnt/public/...` 是本项目验证环境的示例路径，
不是可移植默认值。

## 输入契约

推荐 Parquet，也支持 JSONL。标准记录至少包含非空 `doc_id` 和字符串 `text`：

```json
{
  "doc_id": "doc-001",
  "text": "document body",
  "source": "corpus-name",
  "url": null,
  "timestamp": "2026-07-18T00:00:00Z",
  "language": "zh",
  "meta": {"style": "qa"}
}
```

数据集字段名不同，在 YAML 的 `input.field_map` 中映射，例如：

```yaml
input:
  format: parquet
  field_map:
    uid: doc_id
    content: text
```

未声明的字段会并入 `meta`。输入类型、目录读取行为和失败规则见
[输入输出契约](docs/guide/input_output_contract.md)，机器定义见 [`schemas/`](schemas/README.md)。

## 输出契约

每个子命令输出两个文件：

```text
outputs/stage<N>/<dataset>_<timestamp>/<subcommand>/
├── per_doc.jsonl
└── summary.json
```

`per_doc.jsonl` 保存逐文档原始分数和布尔 flag；`summary.json` 保存数据集级聚合。
两种产物均携带 `schema_version` 和 `artifact_type`，写出前会执行基础契约校验。
业务字段随子命令不同，不应仅凭 flag 自动删除训练数据。

## 抽样与结论

`--max-docs N` 默认使用 seed 固定的单遍蓄水池随机抽样；`--sample-mode head` 只建议用于
调试。未传 `--max-docs` 表示全量处理，但具体方法可能把数据物化到内存，不能据此推断
所有子命令都支持 TB 级全量。

抽样命中率不是总体真值。正式报告至少应同时记录总体规模、样本量、抽样方式、seed、
阈值、模型版本和硬件。零命中只表示“在本次样本和检测能力内未发现”，不证明总体无风险。
完整的统计推断边界见[置信度、效力与局限](docs/guide/statistical_limitations.md)。

## 可靠性证据

仓库当前提供四层验证：

- `tests/unit`：reader、抽样和指标纯函数；
- `tests/contracts`：输入、per-doc 和 summary schema；
- `tests/integration`：不依赖外部模型的 Stage CLI 冒烟；
- `tests/regression`：固定合成真值的关键指标回归。

CI 中的模型指标使用可控 fake 验证切块、召回路由、聚合和 schema，不下载真实权重，也不以
fake 结果证明模型效力。DataTrove、Presidio、真实 tokenizer/embedding/Judge 和多语言 grammar
需要在目标环境按[测试与验收](docs/guide/testing.md)执行外部验收。

公开参考包括 UFW-L3 的完整画像、OpenWebText 对 Raw Common Crawl 的方向性对照、微型训练
代理控制实验，以及 The Stack 代码样本的能力验证。它们证明实现可以发现预设类型的信号，不证明所有指标都有
足够 recall，也不能直接成为其他数据集的通用合格阈值。详见[公开报告](docs/reports/README.md)。

开发检查：

```bash
python -m pip install -e '.[dev]'
make check
```

## 文档与目录

```text
configs/       Stage 配置和示例字段映射
docs/guide/    当前用户契约、指标、统计与性能说明
docs/reports/  可公开、经过整理的参考报告
docs/archive/  否决实验和研究过程记录，不是当前用户契约
schemas/       JSON Schema 机器契约
pretrain_data_eval/  reader、sampling、schema 等可安装共享实现
stages/        各评估 Stage 的 CLI、实现和运行说明
tests/         unit / contracts / integration / regression
outputs/       本地运行产物，默认不提交 Git
```

文档入口见 [docs/README.md](docs/README.md)，资源和规模边界见
[性能与规模](docs/guide/performance.md)。
