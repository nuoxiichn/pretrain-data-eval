# pretrain-data-eval

预训练数据质量评测工具库，覆盖数据生命周期的 13 个评估维度，从数据来源审计到 tokenization 完整性校验。

## 快速开始

```bash
# 安装 Python 依赖 + 克隆 vendor 仓库 + 系统工具
bash setup.sh

# 仅安装核心依赖（无 GPU）
pip install -e .

# 含 GPU 支持（Binoculars、FineWeb-Edu、propella-1）
pip install -e ".[gpu]"
```

## 目录结构

```
pretrain-data-eval/
├── configs/           # 各阶段参数配置（yaml）
├── outputs/           # 评估输出（.gitignore 排除）
├── vendor/            # 无 pip 包的克隆仓库（.gitignore 排除）
├── source_audit/      # 2.1  数据来源、版本与处理链路
├── safety/            # 2.2  安全、隐私与敏感内容
├── cleaning/          # 2.3  基础清洗、语言识别与格式
├── multilingual/      # 2.4  多语言与低资源覆盖
├── dedup/             # 2.5  去重、近重复与语义重复
├── contamination/     # 2.6  评测污染与数据可追溯
├── quality/           # 2.7  内容质量评分与过滤策略
├── synthetic/         # 2.8  合成、改写与生成数据
├── temporal/          # 2.9  时间性与知识新鲜度
├── domain/            # 2.10 专项能力数据（数学/代码/STEM）
├── longctx/           # 2.11 长上下文数据构造
├── mixture/           # 2.12 数据分布、mixture 与采样策略
├── tokenization/      # 2.13 Tokenization 与入训可用性
├── setup.sh           # 环境初始化脚本
├── pyproject.toml     # Python 依赖
└── TOOLS.md           # 工具安装与用法文档
```

## 评测维度

| 维度 | 核心工具 |
|------|----------|
| 2.1 来源与版本 | DataTrove, ScanCode Toolkit |
| 2.2 安全隐私 | Presidio, Detoxify, Gitleaks, TruffleHog |
| 2.3 基础清洗 | DataTrove, Trafilatura, jusText, fasttext-langdetect |
| 2.4 多语言 | fasttext-langdetect, Lingua |
| 2.5 去重 | DataTrove MinHash, datasketch, FAISS |
| 2.6 污染检测 | LLMSanitize, llm-decontaminator, lm-eval |
| 2.7 质量评分 | KenLM, FineWeb-Edu, propella-1, cleanlab |
| 2.8 合成数据 | Binoculars, Detoxify |
| 2.9 时间性 | DataTrove metadata pipeline |
| 2.10 专项能力 | tree-sitter, ScanCode, lm-eval |
| 2.11 长上下文 | DataTrove, LongBench/RULER/HELMET |
| 2.12 Mixture | DSIR, DataTrove |
| 2.13 Tokenization | HF Tokenizers, DataTrove |

工具详细用法见 [TOOLS.md](TOOLS.md)。

## 输出格式

各阶段脚本统一输出结构化 JSON：

```json
{
  "doc_id": "...",
  "scores": { "toxicity": 0.02, "edu_score": 3 },
  "flags":  { "is_duplicate": false, "has_pii": true }
}
```

输出目录：`outputs/{阶段名}/{数据集名}_{时间戳}/`

## 参考文档

评测方案设计来源：`../pretrain-data-quality-survey/report/draft_v2.md` 三、预训练数据质量评测方案（2.1–2.13）
