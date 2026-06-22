# Stage 2：安全隐私

## 评估目标

检测预训练语料中的 PII、Secret 与有害内容，输出命中文档列表供人工决策是否过滤。

## 子命令

| 子命令 | 对应行 | 工具 | 说明 |
|--------|--------|------|------|
| `pii` | 行 1（通用文本）+ 行 2（代码语料） | Microsoft Presidio | 60+ 实体类型；通过 `--mode` 切换通用/代码模式 |
| `secrets` | 行 3 | Gitleaks（二进制） | 高置信度 Secret 扫描，`--no-git` 模式逐文档扫描 |
| `toxicity` | 行 4 (v2) | HF 文本分类模型 | 整篇打分；自动识别多标签 / 二分类 |
| `toxicity-v3` | 行 4 (v3) | XLM-R 召回 + Qwen2.5-7B-Instruct LLM-judge | chunk 切分（≤512 token）+ 召回 + 三类（benign/discuss/promote）复审；解决 v2 长文档「文学引用 / 历史叙述」误报 |

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

**多标签模型**（detoxify unbiased，英文）：
```json
{
  "scores": {"toxicity": 0.03, "severe_toxicity": 0.01, "obscene": 0.02, "identity_attack": 0.01, "insult": 0.02, "threat": 0.01, "sexual_explicit": 0.01},
  "flags": {"high_risk": false}
}
```

**二分类模型**（xlmr-large-toxicity-classifier，中英多语言）：
```json
{
  "scores": {"toxicity": 0.85},
  "flags": {"high_risk": true}
}
```

> 模型类型（`model_mode: "binary"` / `"multilabel"`）自动识别，无需手动切换。
> 切换模型只需改 `configs/stage2.yaml` 中的 `toxicity.model_path`。
>
> **推荐**：中英语料统一用 `textdetox/xlmr-large-toxicity-classifier`（XLM-RoBERTa-Large，
> PAN/CLEF 2024 共享任务官方分类器，覆盖 9+ 语言）。纯英文语料如需细粒度维度可用
> detoxify unbiased。

### toxicity-v3（v3：召回 + LLM-judge 复审）

**动机**：v2 报告抽查显示 XLM-R 在 UFW-L3（教育/百科长文档）上把「文学引用、历史叙述、新闻
报道」误判为 promote，假阳性 ≈ 100%。v3 引入二阶段流水线：

1. **chunk** — 用 XLM-R tokenizer 把每文档切成 ≤512 子词的 chunk（含 overlap），避免长文档
   被全文一次性强行截断。
2. **召回** — XLM-R 二分类器对每个 chunk 打分，超过 `recall_threshold`（默认 0.5）的 chunk 进入复审。
3. **LLM-judge** — Qwen2.5-7B-Instruct 复审每个召回 chunk，输出三类：
   - `benign`  — 内容无关有害议题
   - `discuss` — 提到/引用/批判/客观叙述（文学、历史、新闻、学术），作者立场中立 → **不算 high_risk**
   - `promote` — 作者立场为煽动/教唆/赞扬/传播 → **算 high_risk**
4. **聚合** — `flags.high_risk == flags.llm_promote`：至少一个 chunk 被判 `promote` 才算高位。

```json
{
  "scores": {
    "xlmr_max": 0.99, "xlmr_mean": 0.12,
    "n_chunks": 6, "n_chunks_recalled": 2,
    "n_chunks_promote": 0, "n_chunks_discuss": 2,
    "judgments": [
      {"chunk_idx": 3, "xlmr_score": 0.987, "verdict": "discuss",
       "confidence": 0.92, "reason": "客观叙述 19 世纪伦敦贫民窟", "text_preview": "Charles Dickens..."}
    ]
  },
  "flags": {"recalled": true, "llm_promote": false, "high_risk": false}
}
```

配置见 `configs/stage2.yaml -> toxicity_v3:`。Judge 走本地 vLLM，需 GPU；与召回模型共卡时
调低 `judge_gpu_mem_util`。

**v2 高位文档复审**：`scripts/recheck_v2_high_risk.py` 自动从 `outputs/stage2/ufw_{en,zh}_l3/toxicity_v2/`
取 high_risk doc_id，回查 parquet 原文，跑 toxicity-v3 复审，落
`outputs/stage2/ufw_{en,zh}_l3/toxicity_v3_recheck/`，并产 `comparison.md` 对照表。

## 依赖

```
presidio-analyzer>=2.2
presidio-anonymizer>=2.2
spacy>=3.7
transformers>=4.40    # toxicity：加载本地 HF 分类模型（detoxify / xlmr / COLD 等）
vllm>=0.6             # toxicity-v3：LLM-judge 推理（Qwen2.5-7B-Instruct）
```

毒性模型走本地 HF 目录（见 `configs/stage2.yaml` 的 `toxicity.model_path`），
用 transformers 直接加载。支持任意 `*ForSequenceClassification` 模型：
- **多标签**（如 detoxify unbiased）→ sigmoid，输出多维度分数
- **二分类**（如 xlmr-large-toxicity-classifier、roberta-base-cold）→ softmax，输出单维 toxicity

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
