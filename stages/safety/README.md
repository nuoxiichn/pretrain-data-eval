# Stage 2：安全隐私

## 评估目标

检测预训练语料中的 PII、Secret 与有害内容，输出命中文档列表供人工决策是否过滤。

## 子命令

| 子命令 | 对应行 | 工具 | 说明 |
|--------|--------|------|------|
| `pii` | 行 1（通用文本）+ 行 2（代码语料） | Microsoft Presidio | 60+ 实体类型；通过 `--mode` 切换通用/代码模式 |
| `secrets` | 行 3 | Gitleaks（二进制） | 高置信度 Secret 扫描，`--no-git` 模式逐文档扫描 |
| `toxicity` | 行 4 | Detoxify `unbiased` | RoBERTa 7 维打分（toxicity / obscene / threat / …） |

> **BigCode PII scripts（行 2）**：原始实现需 git clone bigcode-dataset 仓库。
> 本阶段用 Presidio + 代码专用实体列表（`EMAIL_ADDRESS`, `IP_ADDRESS`, `URL`, `CRYPTO`, `CREDIT_CARD`）近似替代，已覆盖主要模式。

### PII 降误报（基于 UFW-L3 抽样）

抽样回查发现 Presidio 在网页文本上对若干实体几乎全是误报，已做两层处理：

1. **默认实体集剔除三类高误报实体**（仍可经 `--entities`/yaml 显式启用）：
   - `US_DRIVER_LICENSE` —— 命中多为 `A2/A1/K4` 等 2 字符代号（驾照等级、Korotkoff 相位）
   - `MEDICAL_LICENSE` —— 命中多为 `US5528166`（专利号）、`rs1205081`（DOI/dbSNP）
   - `UK_NHS` —— 命中多为美国电话号格式
2. **命中后过滤示例/占位符/保留值**（不计入命中，仅作判定不写明文）：
   - 占位符邮箱：`example.com/.org` 域名、`firstname.lastname`/`email`/`user` 等本地部分
   - 保留/私有/文档 IP：`127.0.0.1`、内网段（10/172.16/192.168）、RFC5737 文档段、链路本地等

被过滤的命中数记入 `summary.filtered_placeholder_count`，过滤过程可审计、不静默丢弃。

## 输入

标准 JSONL / Parquet（通过 `src/reader.py`），配置见 `configs/stage2.yaml`。

## 输出格式

`per_doc.jsonl` — 每条一行：
```json
{"doc_id": "...", "scores": {...}, "flags": {...}}
```

### pii
```json
{
  "scores": {"pii_count": 3, "pii_hits": [{"entity_type": "EMAIL_ADDRESS", "start": 10, "end": 28, "score": 0.85}]},
  "flags": {"has_pii": true}
}
```

### secrets
```json
{
  "scores": {"secret_count": 1, "secrets": [{"rule_id": "generic-api-key", "entropy": 4.72, "start_line": 5, "end_line": 5}]},
  "flags": {"has_secrets": true}
}
```

### toxicity
```json
{
  "scores": {"toxicity": 0.03, "severe_toxicity": 0.01, "obscene": 0.02, "identity_attack": 0.01, "insult": 0.02, "threat": 0.01, "sexual_explicit": 0.01},
  "flags": {"high_risk": false}
}
```

## 依赖

```
presidio-analyzer>=2.2
presidio-anonymizer>=2.2
spacy>=3.7
transformers>=4.40    # toxicity：加载本地 detoxify unbiased HF 模型
```

毒性模型走本地 HF 目录（`/mnt/public/model/detoxify/unbiased-toxic-roberta`，
见 `configs/stage2.yaml` 的 `toxicity.model_path`），用 transformers 直接加载，
**不依赖 detoxify pip 包**（其默认从 GitHub 拉 `.ckpt`，已弃用该链路）。

Presidio 需要 spaCy 语言模型（首次运行自动下载）：
```bash
python -m spacy download en_core_web_lg
```

Gitleaks 二进制（非 pip，需单独安装）：
```bash
# macOS
brew install gitleaks
# Linux
# https://github.com/gitleaks/gitleaks/releases
```

## 运行示例

```bash
# 通用 PII（英文）
PYTHONPATH=. python stages/safety/run.py pii \
  --input data/mock.jsonl --dataset mock --input-format jsonl

# 代码 PII 模式
PYTHONPATH=. python stages/safety/run.py pii \
  --input data/mock.jsonl --dataset mock --input-format jsonl --mode code

# Secret 扫描
PYTHONPATH=. python stages/safety/run.py secrets \
  --input data/mock.jsonl --dataset mock --input-format jsonl

# 毒性检测（必须带 --config，从中读 toxicity.model_path）
PYTHONPATH=. python stages/safety/run.py toxicity \
  --input data/mock.jsonl --dataset mock --input-format jsonl \
  --config configs/stage2.yaml

```
