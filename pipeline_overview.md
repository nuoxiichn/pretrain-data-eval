# 预训练数据评测流水线

各阶段对输入数据执行**只读分析**，输出审计报告与问题文档列表。

---

| 阶段 | 维度/指标 | 输出字段 | 实现 | 状态 |
|------|-----------|--------------|------|------|
| **1 来源审计 + 时间属性** | 文档统计：字符 / token / 长度分桶 / 域名 / 声明语种 / 来源 / 时间字段完整性与年月分布 | `total_docs` / `char_stats` / `token_stats` / `length_buckets`（4K/8K/32K/128K/256K+，以 **token** 数为桶）/ `domain_distribution` / `language_distribution` / `source_distribution` / `timestamp.{present, missing, present_pct, year_month_distribution}` | 自实现（`source_audit/utils.py::DocStatsAggregator`，标准库 + numpy）；token 后端默认 `hf`（Qwen3-4B-Base，vocab 151936，与 stage10 共享，可 `--coalesce-stage10` 一次扫描产两份 summary）；`words` 仅 mock smoke 用 | done |
| | 许可证检测 / 版权声明 | `total_docs_scanned` / `docs_with_license` / `hit_pct` / `license_type_distribution` | 调用 ScanCode Toolkit (`scancode.api`) | done（功能可用，未在 UFW-L3 全量跑——合成数据无许可证字段） |
| | 处理参数快照（pipeline 可复现性） | — | — | SKIP（上游清洗/合成工具链不固定，无法依赖 DataTrove `executor.json` 等特定产物） |
| | 样本级处理 diff | — | — | SKIP（参考 Data-Juicer `Tracer`，与"成品审计"定位冲突） |
| **2 安全隐私** | PII 检测（通用文本：邮箱 / 电话 / 信用卡 / IP / 人名 / 地址等 60+ 类型） | `total_docs_scanned` / `docs_with_pii` / `hit_pct` / `entity_type_distribution` / `mode` / `language` / `score_threshold` | 调用 Microsoft Presidio (`presidio_analyzer.AnalyzerEngine`)；spaCy 模型可换（`en_core_web_lg` / `zh_core_web_sm` / blank）；自实现占位符 + 保留值过滤（example.com / RFC5737 IP / 私有/保留段） | done |
| | PII 检测（代码语料） | 同上，`mode=code` 走代码专用 recognizer 子集 | 自实现，参照 BigCode PII scripts 的规则集（regex + 实体类型映射），未独立部署 BigCode 工具 | done |
| | Secret 扫描（API Key / Token / 私钥） | `total_docs_scanned` / `docs_with_secrets` / `hit_pct` / `rule_distribution` | 调用 Gitleaks 二进制（`subprocess` 包装），文档导出到临时目录批量扫描 | done |
| | 毒性 / 仇恨 / 色情 / 暴力分类（多维分类，英文） | `total_docs_scanned` / `high_risk_docs` / `high_risk_pct` / 多维分数分布（toxicity / severe_toxicity / obscene / threat / insult / identity_attack） | 调用 HuggingFace `transformers` 加载 `unitary/toxic-bert` 或 `unbiased-toxic-roberta`（在 `/mnt/public/model/detoxify/`），不依赖 detoxify pip 包 | done |
| | 毒性分类（中英多语二分类） | `high_risk_docs` / `high_risk_pct` / `toxicity` 分数分布 | 调用 transformers 加载 `textdetox/xlmr-large-toxicity-classifier`（XLM-R-Large 二分类，覆盖 en/zh + 9 语）；备选 `thu-coai/roberta-base-cold` | done |
| **3 语言识别 + 文本质量审计 + 多语言覆盖** | 抽取质量审计（成品文本的杂质残留） | `low_extraction_quality_pct` / `html_residue_pct` / `boilerplate_pct` / `mojibake_pct` / `url_docs` / `short_stub_docs` / `boilerplate_phrase_distribution` | 自实现（regex + 词表），无外部库 | done |
| | 主流语种识别（176 种） | `language_distribution` / `mismatch_count/_pct` / `low_confidence_count/_pct` | 调用 fastText (`lid.176.bin`，在 `/mnt/public/model/`) | done |
| | 低资源细粒度语种-脚本识别（2000+ 类别） | `lang_script_distribution`（`ISO3_Script` 格式） / `low_confidence_count/_pct` | 调用 fastText 加载 GlotLID v3 (`/mnt/public/model/glotlid/model.bin`)，不依赖 glotlid pip 包 | done |
| | 粗细语种交叉核对 | `disagreement_count/_pct` / `possibly_absorbed_count/_pct` / `absorbed_lang_distribution` | 自实现：同时调 lid.176 与 GlotLID，宏语言层归一后比对（用 `langcodes`），暴露被粗模型吸收的低资源语种 | done |
| | 文本质量审计：Gopher Quality + C4 规则 | `total_docs` / `filter_fail_counts` / `filter_fail_pcts` / `filter_fail_reasons` | 自实现，参照 DataTrove `GopherQualityFilter` + `C4QualityFilter` 规则集；CJK 用 `jieba` 分词，英文用 `nltk`（punkt） | done |
| **4 重复度分析** | 精确重复（文档级 + 段落级 MD5） | `total_docs` / `exact_dup_docs` / `exact_dup_pct` / `unique_doc_hashes` / `para_dup_docs` / `para_dup_pct` / `unique_para_hashes` | 自实现（`hashlib.md5`），参照 DataTrove `ExactDedup` / `SentenceDedup` 思路 | done（单文件内；跨文件需全局 hash 表） |
| | MinHash + LSH 近重复检测 | `total_docs` / `near_dup_docs` / `near_dup_pct` / `near_dup_pairs` / `jaccard_max`（per-doc） | 自实现 MinHash 签名（MD5 派生 hash 函数）+ LSH 桶式聚类，参照 DataTrove `MinhashDedupSignature/Buckets/Cluster` 接口 | done（单文件内；跨文件 100K 样本即 OOM，TB 级需分布式） |
| | 段落级 N-gram 重复（Bloom-style 审计） | `total_docs` / `contaminated_docs` / `contaminated_pct` / `total_paras` / `contaminated_paras` / `contaminated_para_pct` | 自实现（shingle Jaccard），参照 Dolma `dedupe --by_ngram` 阈值约定（13-gram, 0.5 overlap），未调用 dolma Rust 二进制 | done |
| | 语义重复（embedding 聚类） | `total_docs` / `semantic_dup_docs` / `semantic_dup_pct` / `num_clusters` / `largest_cluster_size` | 自实现：调用 transformers 加载 bge-m3 出 embedding + torch 分块余弦相似度聚簇（参照 NeMo SemDeDup 思路，因本机非 NVIDIA 不能用 faiss-gpu） | done |
| | 跨来源重复率 / epoch 等效次数 | — | 未实现（依赖 token 计数 + 来源分组聚合，可在聚合层补） | TODO |
| **5 污染检测** | 精确污染（文档级 + 段落级 hash 对比 benchmark） | `total_docs` / `contaminated_docs` / `contaminated_pct` / `para_contaminated_docs` / `para_contaminated_pct` / `per_benchmark_hits` | 自实现（`hashlib.md5` + benchmark 文本归一化），benchmark 加载经 `stages/contamination/benchmarks.py`，支持 HuggingFace `datasets` / 本地 JSONL / 本地 Parquet；参照 Dolma `dedupe --read_only` 的成员查询思路 | done |
| | N-gram 污染（词级 13-gram） | — | — | SKIP（lm-eval-harness 的污染检测模块对成品数据冗余，与 S4 ngram 重叠） |
| | 代码 benchmark 近重复（字符 5-gram MinHash） | `total_docs` / `code_near_dup_docs` / `code_near_dup_pct` / `bench_code_samples` | 自实现字符级 MinHash + LSH，参照 BigCode `minhash_deduplication.py` 阈值（Jaccard 0.85） | done |
| | 代码 AST 结构污染（变量重命名 / 注释改写） | `total_docs` / `ast_contaminated_docs` / `ast_contaminated_pct` / `fingerprint_exact_hits` | 自实现 AST 节点类型指纹（参照 tree-sitter，Python 用 `tree_sitter_python`）+ 指纹 Jaccard | done |
| | 多路综合污染（N-gram / embedding / Min-K% Prob） | — | — | TODO（参照 LLMSanitize，未集成） |
| **6 质量评分** | PPL 质量代理（KenLM） | — | — | SKIP — 对照实验有效但成品数据上冗余：PPL tail 85% 是文学/口语/列表，非真实低质量 |
| | 教育价值评分（FineWeb-Edu，0–5 分） | — | — | SKIP — 对照实验证伪：测的是"教育性"非"质量"（OpenWebText 仅 3.7% ≥ 3） |
| | DCLM fastText 二元打分 | — | — | SKIP — 对照实验证伪：Raw CC 的 hq 率(8%) > OpenWebText(2%)，无区分力 |
| **7 合成数据检测** | 合成元数据统计（`source` 字段 / generation_depth） | — | — | TODO（依赖上游 metadata 字段，UFW-L3 无此字段） |
| | AI 生成文本检测（observer/performer 双模型对比） | `total_docs_scanned` / `ai_generated_docs` / `ai_generated_pct` / score 分布 | 自实现 Binoculars（参照 Binoculars 原论文），调用 transformers 加载 Llama-2-7B base+chat（或 Qwen2.5-7B base+instruct），GPU ~28GB | done |
| | Fast-DetectGPT（Sampling Discrepancy） | — | — | TODO |
| **8 专项能力** | 代码可解析率 / 语法错误严重度（默认 Python） | `total_docs` / `parsed_docs` / `has_error_docs` / `has_error_pct` / `unparsable_docs` / `error_ratio` 分布 | 调用 tree-sitter (`tree_sitter` + `tree_sitter_python`) | done（方法论上对自然语言数据无意义，UFW-L3 命中率 ~100%） |
| | STEM 学科分布 | `total_docs` / `stem_docs` / `stem_pct` / `subject_distribution` / `primary_subject_top10` | 自实现（关键词词表 + 密度阈值），仅含英文关键词 | done（英文有效，中文需多语词表） |
| **9 长上下文** | 训练配置审计（packing 边界三参数） | `parameters.{reset_position_ids, reset_attention_mask, eod_mask_loss}.{found, value, valid}` / `config_valid` / `missing_params` / `invalid_params` | 自实现 yaml / json / 正则三路解析（针对 Megatron-LM `--reset-position-ids` / `--reset-attention-mask` / `--eod-mask-loss`），不依赖 Megatron 包 | done（不消费数据集，秒级） |
| **10 Tokenization** | Token/char fertility / UNK 率 / 代码 & LaTeX token 膨胀率 | `total_docs` / `fertility_stats`（mean/p50/p95 等） / `unk_stats.{total_unk_tokens, overall_unk_rate, docs_with_unk, ...}` / `code_stats.docs_with_code` / `latex_stats.docs_with_latex` / `high_unk_rate_docs` / `high_fertility_docs` | 调用 HuggingFace `tokenizers.Tokenizer`（默认 `/mnt/public/model/huggingface/Qwen3-4B-Base`，与 stage1 共享）；可与 stage1 stats 通过 `--coalesce-stage10` 一次扫描共出两份 summary | done |

---

## 版本与变更

| 版本 | 分支 | 状态 | 报告产物 |
|------|------|------|----------|
| v2   | `main`（2026-06-22 归档） | 已交付 | `outputs/report/ufw_l3_profile_v2.md` — Stage 1–5 + 7 + 8 + 10 完成 |
| v3   | `dev-v3`（进行中） | 见下方 §"v3 计划" | `outputs/report/ufw_l3_profile_v3.md`（新建，含 v2→v3 增量与回归对照） |

> **v2 报告 / v2 子命令产物不动。** v3 出新报告与（必要时）新子命令；同一 stage 的旧子命令保留可调用，便于回归对比。

---

## v3 计划（dev-v3 分支）

v3 针对 v2 已暴露的方法论缺口与「为 10T 自研数据准备工程能力」两条主线，**只动以下四个 stage**，其他维度（S1 / S3 / S6 / S7 / S9 / S10）结论与 v2 保持一致。

### v2 → v3 改动 diff

| 优先级 | Stage | v2 现状 | 缺口 | v3 改动 | 类型 |
|--------|-------|---------|------|---------|------|
| P0 | S8 stem | EN+ZH 各 50 head 抽样，EAI-Distill 2.5s/条；ZH high_difficulty=0% 异常 | 全量推理吞吐不可行（10× vLLM 加速后仍 ~73K GPU·h）；head 抽样系统性偏向单一 row-group | 改 head→**分层抽样**（lang × style × char-length 桶），总 50K，复用 `src/sampling.py` 的 Vitter R；归因 ZH 难度异常 | 决策 + 抽样脚本 |
| P1 | S2 toxicity | XLM-R 整篇打分 5K 采样，EN 11 / ZH 5 高位文档 | 人工核查显示高位 ≈ 100% 假阳性（文学引用、历史叙述、新闻报道被误判为"传播毒性"）。XLM-R 训练于社交媒体短文本，对教育百科长文档领域 mismatch | 二阶段方法：**chunk 切分（≤512 token）+ XLM-R 召回 + Qwen2.5-7B-Instruct 作为 LLM-judge 复审**（"discussing vs promoting"）。新增子命令 `toxicity-v3`，v2 子命令保留 | 方法 |
| P2 | S5 contamination | MMLU 全量，文档级 0% / 段落级 2/698M；ZH 360M 无对应中文 benchmark | ZH 完全无 benchmark 覆盖；MMLU 是英文选择题，与 UFW 长文本形态不匹配 | 补 **CMMLU / C-Eval / AGIEval / GSM8K**（中文）+ HumanEval（保持 code-near/code-ast 一致性）；`benchmarks.py` 加 loader，对 ZH 全量重跑 | 覆盖 |
| P3 | S4 minhash | 100K 跨 4 文件 head 采样，单机 OOM；自实现 MinHash 无法扩展到 TB 级 | 跨文件全量在自实现下不可达；为 10T 自研数据准备的分布式 dedup pipeline 未跑通 | 新建独立 conda env（隔离 `pyproject.toml` 主依赖），接入 **DataTrove `MinhashDedupSignature / Buckets / Cluster`**；`src/reader.py` 在 DataTrove 入口处搭桥，保 `Document` dict 契约不变；对 UFW-L3 全量 1.06B 文档跑通 | 工程 |

### 交付顺序与理由

**P0 → P1 → P2 → P3**（与优先级一致）。

- **P0 先做**：工程量最小（抽样脚本改造），直接消除 v2 报告 §2.7 里"ZH 难度评估存疑"的悬而未决；且不依赖任何环境变更。
- **P1 第二做**：v2 报告里 S2 toxicity 是仅有的 🟡 之一，方法论错（领域 mismatch），不修则 v3 报告该项仍不可信。LLM-judge 用本地 Qwen2.5-7B-Instruct，无外部依赖。
- **P2 第三做**：中等工程量（多 benchmark loader + 全量 ZH 重跑），但路径清晰、无方法论风险，可与 P3 的环境准备并行启动。
- **P3 放最后**：新 conda env + DataTrove 依赖较重，会触动 base 环境；放最后避免与其他改动相互污染。**P3 路径若卡住可独立延后到 v4 而不阻塞 v3 报告**。

### 范围边界（v3 **不做**的事，明确写下来防止漂移）

- v2 报告 `ufw_l3_profile_v2.md` 与 v2 各 stage 子命令的旧产物**不动**；v3 出新报告，对应字段标注「v2 → v3」回归对照。
- S5 软匹配方法（SoftMatcha 2 / LLMSanitize 多路综合）**推迟到 v4**——v3 只动 benchmark 覆盖，不改方法，避免「补覆盖 + 改方法」两路混合无法归因。
- S6 KenLM / FineWeb-Edu / DCLM 维持 SKIP（v2 已三组否决）；S7 Fast-DetectGPT、合成元数据维持 TODO；S8 parsability 维持「对自然语言不适用」结论不重跑；S1 / S3 / S10 全量结论沿用 v2。
- S2 toxicity v3 走**新子命令**`toxicity-v3`，**不在原子命令上原地改**，保证回归路径可走。
- 不为 v3 强行做 S2 toxicity / S5 contamination 的全量 EN 重跑——只在数据覆盖真的扩展（如 ZH benchmark）时才扩量。

### v3 验收清单

| 项 | 通过标准 |
|----|---------|
| P0 S8 | 分层抽样脚本落地，50K 样本跑出 FDC 分布，**ZH high_difficulty=0% 给出归因**（"语料偏通俗" vs "EAI 模型对中文校准偏低"，可与 ufw_zh_l3/qa 子集对照） |
| P1 S2 | `toxicity-v3` 子命令实现；v2 高位文档（EN 11 / ZH 5）经 LLM-judge 复审后**假阳性率下降 ≥ 80%**；5K 采样重跑且新报告给出真实高位文档列表 |
| P2 S5 | `benchmarks.py` 至少加入 CMMLU / C-Eval / GSM8K-zh 三个中文 loader；ZH 360M 全量重跑产出新 `aggregated_summary.json`；v3 报告 §2.5 含中文 benchmark 命中数 |
| P3 S4 | DataTrove env 独立、`reader.py` 桥接通过；UFW-L3 全量 1.06B 跑出跨文件 near_dup 数值；新 stage 文档说明「为 10T 自研流水线服务」的定位 |

---

## 执行编排（依赖与并行）

> 编号 1–10 是**调研阶段**的维度分组，**不是执行顺序**。本节给出实际跑流水线时的依赖图、并行分组和启动顺序。

### 启动顺序（按 10T 耗时从大到小）

**核心思路**：耗时长的 stage 必须最早启动，否则会卡住总进度。`scripts/*_batch.sh` 全部支持断点续跑（`summary.json` 存在则跳过），所以"先全启动后调度"是安全的。

| 优先级 | Stage | 类型 | 10T 外推 | 策略 |
|--------|-------|------|---------|------|
| **P0 立即启动** | S7 Binoculars | GPU | ~4800h | **采样 5K-50K**；全量不可行 |
| | S2 PII | CPU | ~2000h | **采样 50K-100K**；常驻 spaCy 进程 |
| | S2 Toxicity | GPU | ~1200h | **采样 10K-50K** |
| **P1 早启动** | S3 Cleaning（5 子命令） | CPU | ~30h | 全量逐文件，并行 5 个子命令 |
| | S4 ngram | CPU | ~24h | 全量 |
| | S4 exact | CPU | ~16h | 全量 |
| | S5 Contamination | CPU | ~15h | 全量；多 benchmark 时一次加载多次扫描 |
| | S2 Secrets | CPU | ~13h | 全量（Gitleaks 二进制速度快） |
| | S4 MinHash | CPU | ~10h | 全量（单文件内）；跨文件需分布式 |
| **P2 随时穿插** | S1 Stats | CPU | ~8h | 全量；轻量 |
| | S8 STEM / Parsability | CPU | ~5h | 采样即可（方法对 NL/ZH 局限大） |
| | S10 Tokenize | CPU | ~2h | 全量；最轻 |
| **独立** | S9 Config Audit | — | 秒级 | 单独跑训练 config |

**并行分组**（一台 ≥16 核 + 1×A100 的机器，3 个池子互不抢资源）：

- **GPU 池**：S7 Binoculars、S2 Toxicity（按优先级串行；GPU 显存 80GB 也只能放一个 7B 模型）
- **CPU 重池**（IO 与算力都重，控制总进程 ≤ 核数）：S3 Cleaning × 5、S4 ngram / exact / minhash / semdedup、S5 Contamination
- **CPU 轻池**（可挤进重池空隙）：S1 Stats、S2 Secrets、S2 PII、S8、S10

**实际跑法**：所有 `scripts/*_batch.sh` 同时 `nohup` 启动 + 分别 tee 到 log，单文件粒度的断点续跑天然就是机器调度——某个 stage 卡住或失败不影响其他 stage 进度。

### 抽样策略（统一）

所有 stage CLI 的 `--max-docs` 默认走 **`random` 蓄水池抽样**（`src/sampling.py`，Vitter Algorithm R，单遍 O(n) 内存）。可用参数：

- `--max-docs N`：抽 N 条；不传则全量
- `--sample-mode {random,head}`：默认 `random`；`head` 复刻旧行为（取前 N，调试/精确重复检测时有用）
- `--seed INT`：默认 `42`，固定后样本可复现

**为什么默认随机**：旧的 `docs[:N]` / `islice` 取的是 parquet 文件前 N 行，受 row-group 顺序影响——同一文件内连续行往往来自同一时间窗口或同一来源切片，会系统性高估/低估某些指标（特别是 Stage 4 dedup、Stage 2 toxicity 这类与上游切片强相关的维度）。蓄水池抽样让每条文档被采概率相等，与文件物理布局无关。

**例外**：Stage 4 的 `exact` / `ngram` 在**单文件内**计重复率时，`head` 与 `random` 都会破坏跨"被采样部分外"的重复对——但因为这两个子命令的 batch 跑法是全量逐文件，`--max-docs` 只在调试时用，不影响生产语义。
