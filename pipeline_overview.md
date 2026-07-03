# 预训练数据评测流水线

各阶段对输入数据执行**只读分析**，输出审计报告与问题文档列表。

---

## 维度全景

| 阶段 | 维度/指标 | 输出字段 | 实现 | 状态 |
|------|-----------|--------------|------|------|
| **1 来源审计 + 时间属性** | 文档统计：字符 / token / 长度分桶 / 域名 / 声明语种 / 来源 / 时间字段完整性与年月分布 | `total_docs` / `char_stats` / `token_stats` / `length_buckets`（4K/8K/32K/128K/256K+，以 **token** 数为桶）/ `domain_distribution` / `language_distribution` / `source_distribution` / `timestamp.{present, missing, present_pct, year_month_distribution}` | 自实现（`source_audit/utils.py::DocStatsAggregator`，标准库 + numpy）；token 后端默认 `hf`（Qwen3-4B-Base，vocab 151936，与 stage10 共享，可 `--coalesce-stage10` 一次扫描产两份 summary）；`words` 仅 mock smoke 用 | done |
| | 许可证检测 / 版权声明 | `total_docs_scanned` / `docs_with_license` / `hit_pct` / `license_type_distribution` | 调用 ScanCode Toolkit (`scancode.api`) | done（功能可用；合成数据无许可证字段，未全量跑） |
| **2 安全隐私** | PII 检测（通用文本：邮箱 / 电话 / 信用卡 / IP / 人名 / 地址等 60+ 类型） | `total_docs_scanned` / `docs_with_pii` / `hit_pct` / `entity_type_distribution` / `mode` / `language` / `score_threshold` | 调用 Microsoft Presidio (`presidio_analyzer.AnalyzerEngine`)；spaCy 模型可换（`en_core_web_lg` / `zh_core_web_sm` / blank）；自实现占位符 + 保留值过滤（example.com / RFC5737 IP / 私有/保留段） | done |
| | PII 检测（代码语料） | 同上，`mode=code` 走代码专用 recognizer 子集 | 自实现，参照 BigCode PII scripts 的规则集（regex + 实体类型映射），未独立部署 BigCode 工具 | done · **已在真实代码验证**（The Stack Python 500 抽样命中率 89.8%，绝大多数为注释/文档 URL，敏感 EMAIL 39/IP 162——命中率非隐私风险率） |
| | Secret 扫描（API Key / Token / 私钥） | `total_docs_scanned` / `docs_with_secrets` / `hit_pct` / `rule_distribution` | 调用 Gitleaks 二进制（`subprocess` 包装），文档导出到临时目录批量扫描 | done · **已在真实代码验证**（The Stack Python 500 抽样命中 0.4%，generic/gcp-api-key；未验真伪，gitleaks 有假阳） |
| | 毒性检测：chunk + XLM-R 召回 + Qwen LLM-Judge 复审 | `total_docs_scanned` / `high_risk_docs` / `high_risk_pct` / `judge_verdict_distribution`（真阳/假阳/待定）/ chunk 级分数分布 | 二阶段自实现：文档切 ≤512 token chunk → `textdetox/xlmr-large-toxicity-classifier` 召回可疑段（覆盖 en/zh + 9 语）→ Qwen2.5-7B-Instruct 判「discussing vs promoting」；子命令 `toxicity-v3` | done |
| **3 语言识别 + 文本质量审计 + 多语言覆盖** | 抽取质量审计（成品文本的杂质残留） | `low_extraction_quality_pct` / `html_residue_pct` / `boilerplate_pct` / `mojibake_pct` / `url_docs` / `short_stub_docs` / `boilerplate_phrase_distribution` | 自实现（regex + 词表），无外部库 | done |
| | 主流语种识别（176 种） | `language_distribution` / `mismatch_count/_pct` / `low_confidence_count/_pct` | 调用 fastText (`lid.176.bin`，在 `/mnt/public/model/`) | done |
| | 低资源细粒度语种-脚本识别（2000+ 类别） | `lang_script_distribution`（`ISO3_Script` 格式） / `low_confidence_count/_pct` | 调用 fastText 加载 GlotLID v3 (`/mnt/public/model/glotlid/model.bin`) | done |
| | 粗细语种交叉核对 | `disagreement_count/_pct` / `possibly_absorbed_count/_pct` / `absorbed_lang_distribution` | 自实现：同时调 lid.176 与 GlotLID，宏语言层归一后比对（用 `langcodes`），暴露被粗模型吸收的低资源语种 | done |
| | 文本质量审计：Gopher Quality + C4 规则 | `total_docs` / `filter_fail_counts` / `filter_fail_pcts` / `filter_fail_reasons` | 自实现，参照 DataTrove `GopherQualityFilter` + `C4QualityFilter` 规则集；CJK 用 `jieba` 分词，英文用 `nltk`（punkt） | done |
| **4 重复度分析** | 精确重复（文档级 + 段落级 MD5） | `total_docs` / `exact_dup_docs` / `exact_dup_pct` / `unique_doc_hashes` / `para_dup_docs` / `para_dup_pct` / `unique_para_hashes` | 自实现（`hashlib.md5`），参照 DataTrove `ExactDedup` / `SentenceDedup` 思路 | done（单文件内；跨文件需全局 hash 表） |
| | MinHash + LSH 近重复检测（自实现，单文件） | `total_docs` / `near_dup_docs` / `near_dup_pct` / `near_dup_pairs` / `jaccard_max`（per-doc） | 自实现 MinHash 签名（MD5 派生 hash 函数）+ LSH 桶式聚类 | done（单文件内；跨文件 100K 样本即 OOM，TB 级需分布式） |
| | MinHash 分布式（DataTrove 三段管线） | `total_docs` / `duplicate_docs` / `duplicate_pct` / `num_clusters` / `largest_cluster_size` / stage1-3 各段耗时与产物路径 | 独立 conda env `pretrain-dedup`；接入 DataTrove `MinhashDedupSignature/Buckets/Cluster`（跳过 stage 4 filter，只出统计）；参数对齐自实现版：`n_grams=5, num_buckets=8, hashes_per_bucket=8, jaccard≈0.8`；env freeze 存 `envs/datatrove.txt` | running（EN 全量 stage1 pid 57228，~13h ETA；ZH 待启动） |
| | 段落级 N-gram 重复（Bloom-style 审计） | `total_docs` / `contaminated_docs` / `contaminated_pct` / `total_paras` / `contaminated_paras` / `contaminated_para_pct` | 自实现（shingle Jaccard），参照 Dolma `dedupe --by_ngram` 阈值约定（13-gram, 0.5 overlap），未调用 dolma Rust 二进制 | done |
| | 语义重复（embedding 聚类） | `total_docs` / `semantic_dup_docs` / `semantic_dup_pct` / `num_clusters` / `largest_cluster_size` | 自实现：调用 transformers 加载 bge-m3 出 embedding + torch 分块余弦相似度聚簇（参照 NeMo SemDeDup 思路，本机非 NVIDIA 不能用 faiss-gpu） | done |
| **5 污染检测** | 精确污染（文档级 + 段落级 hash 对比 benchmark） | `total_docs` / `contaminated_docs` / `contaminated_pct` / `para_contaminated_docs` / `para_contaminated_pct` / `per_benchmark_hits` | 自实现（`hashlib.md5` + benchmark 文本归一化），benchmark 加载经 `benchmarks.py`；**11 个评测对齐 benchmark**：EN 7（MMLU-Pro / GPQA-Diamond / MATH / EvalPlus / LiveCodeBench / MGSM / MMMLU）+ ZH 4（CMMLU / C-Eval / AGIEval-zh / CMB），benchmark JSONL 输出到 `/mnt/public/data/contamination_v3_benchmarks/eval_aligned/` | done |
| | 级联污染（cascade：L1 exact + L2 MinHash + L3 BGE-m3 embedding） | `total_docs` / `verdict_distribution`（red/yellow/green）/ `l1_hits` / `l2_jaccard_max`（per-doc）/ `l3_cos_max`（per-doc）/ 每层耗时 | 自实现：L1 doc/para hash → L2 char 5-gram MinHash → L3 BGE-m3 embedding + FAISS，阈值 `enter_l3_low=0.20`；索引由 `bench_index.py build` 预构，配置 `configs/stage5_v3.yaml` | done |
| | LLM-as-Judge（cascade yellow 复审） | `per_yellow_judgement.jsonl`（含 verdict ∈ {contamination, abstraction, unrelated} + reason）；收官 46 条真污染，EN 0.008% / ZH 0% | 自实现：Qwen2.5-7B-Instruct 通过 vLLM 部署；prompt 定案含 rubric + few-shot | done |
| | 代码 benchmark 近重复（字符 5-gram MinHash） | `total_docs` / `code_near_dup_docs` / `code_near_dup_pct` / `bench_code_samples` | 自实现字符级 MinHash + LSH，参照 BigCode `minhash_deduplication.py` 阈值（Jaccard 0.85） | done · **已在真实代码验证**（The Stack Python 2000 抽样 vs EvalPlus 542 条：近重复 0%） |
| | 代码 AST 结构污染（变量重命名 / 注释改写） | `total_docs` / `ast_contaminated_docs` / `ast_contaminated_pct` / `fingerprint_exact_hits` | 自实现 AST 节点类型指纹（tree-sitter，Python 用 `tree_sitter_python`）+ 指纹 Jaccard | done · **已在真实代码验证**（The Stack Python 2000 抽样 vs EvalPlus：AST 命中 0.2%，指纹精确 0，多为短函数假阳） |
| **7 合成数据检测** | AI 生成文本检测（observer/performer 双模型对比） | `total_docs_scanned` / `ai_generated_docs` / `ai_generated_pct` / score 分布 | 自实现 Binoculars（参照 Binoculars 原论文），调用 transformers 加载 Llama-2-7B base+chat（或 Qwen2.5-7B base+instruct），GPU ~28GB | done |
| **8 专项能力** | 代码可解析率 / 语法错误严重度（多语言） | `total_docs` / `parsed_docs` / `has_error_docs` / `has_error_pct` / `unparsable_docs` / `unsupported_lang_docs` / `parsed_lang_distribution` / `unsupported_lang_distribution` / `error_ratio` 分布 | 调用 tree-sitter（`>=0.22`）+ 11 种 grammar（python/js/ts/java/go/c/cpp/c-sharp/rust/ruby/php）；`--language auto` 按每条文档 `lang` 字段选 parser、无 grammar 的记 `unsupported`；AST 遍历改迭代避免深嵌套 RecursionError | done · **已在真实代码验证**（The Stack 多语言 500 抽样：go 0% / ruby 0.4% / py 1.2% / rust 2.2% / js 3.0% / c# 3.4% / **c 62%**——error 率随 grammar 质量而异，C 预处理器/宏使 tree-sitter-c 大量报 ERROR；对自然语言无意义） |
| | STEM 学科分布（25K 分层抽样） | `total_docs` / `stem_docs` / `stem_pct` / `subject_distribution` / `primary_subject_top10` + FDC 难度分布 | 自实现（关键词词表 + 密度阈值，仅英文关键词）；`lang × style × char-length` 三维分层抽样，复用 `src/sampling.py` 蓄水池 R，`scripts/stem_p0_run.sh` 串行 EN+ZH | done |
| **9 长上下文** | 训练配置审计（packing 边界三参数） | `parameters.{reset_position_ids, reset_attention_mask, eod_mask_loss}.{found, value, valid}` / `config_valid` / `missing_params` / `invalid_params` | 自实现 yaml / json / 正则三路解析（针对 Megatron-LM 三参数），不依赖 Megatron 包 | done（不消费数据集，秒级） |
| **10 Tokenization** | Token/char fertility / UNK 率 / 代码 & LaTeX token 膨胀率 | `total_docs` / `fertility_stats`（mean/p50/p95 等） / `unk_stats.{total_unk_tokens, overall_unk_rate, docs_with_unk, ...}` / `code_stats.docs_with_code` / `latex_stats.docs_with_latex` / `high_unk_rate_docs` / `high_fertility_docs` | 调用 HuggingFace `tokenizers.Tokenizer`（默认 `/mnt/public/model/huggingface/Qwen3-4B-Base`，与 stage1 共享）；可与 stage1 stats 通过 `--coalesce-stage10` 一次扫描共出两份 summary | done |

---

## 执行编排（依赖与并行）

> 编号 1–10 是**调研阶段**的维度分组，**不是执行顺序**。本节给出实际跑流水线时的依赖图、并行分组和启动顺序。

### 启动顺序（按 10T 耗时从大到小）

**核心思路**：耗时长的 stage 必须最早启动，否则会卡住总进度。`scripts/*_batch.sh` 全部支持断点续跑（`summary.json` 存在则跳过），所以"先全启动后调度"是安全的。

| 优先级 | Stage | 类型 | 10T 外推 | 策略 |
|--------|-------|------|---------|------|
| **P0 立即启动** | S7 Binoculars | GPU | ~4800h | **采样 5K-50K**；全量不可行 |
| | S2 PII | CPU | ~2000h | **采样 50K-100K**；常驻 spaCy 进程 |
| | S2 Toxicity | GPU | ~1200h | **采样 10K-50K**；Qwen judge 走 vLLM |
| **P1 早启动** | S3 Cleaning（5 子命令） | CPU | ~30h | 全量逐文件，并行 5 个子命令 |
| | S4 ngram | CPU | ~24h | 全量 |
| | S4 exact | CPU | ~16h | 全量 |
| | S5 Contamination cascade | CPU + GPU | ~15h | 全量 L1+L2；L3 embedding GPU 批推；LLM-Judge 仅 yellow |
| | S2 Secrets | CPU | ~13h | 全量（Gitleaks 二进制速度快） |
| | S4 MinHash (DataTrove) | CPU 分布式 | ~10h | 全量；signature/buckets/cluster 三段独立 env |
| **P2 随时穿插** | S1 Stats | CPU | ~8h | 全量；轻量 |
| | S8 STEM / Parsability | CPU | ~5h | 采样即可（STEM 走 25K 分层抽样） |
| | S10 Tokenize | CPU | ~2h | 全量；最轻 |
| **独立** | S9 Config Audit | — | 秒级 | 单独跑训练 config |

**并行分组**（一台 ≥16 核 + 1×A100 的机器，3 个池子互不抢资源）：

- **GPU 池**：S7 Binoculars、S2 Toxicity judge、S5 L3 embedding（按优先级串行；GPU 显存 80GB 也只能放一个 7B 模型）
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

**S8 STEM 分层抽样**：STEM 走 `lang × style × char-length` 三维分层，`scripts/stem_p0_run.sh` 固定 25K/EN + 25K/ZH，seed=42。

---

## 待办（TODO / 未来）

尚未实现或推迟的维度，按 stage 归档：

| Stage | 维度 | 说明 |
|-------|------|------|
| S4 重复度 | 跨来源重复率 / epoch 等效次数 | 依赖 token 计数 + 来源分组聚合，可在聚合层补 |
| S5 污染检测 | 多路综合软匹配（Min-K% Prob / SoftMatcha / LLMSanitize） | 推迟到 v4——不与当前 benchmark 覆盖 + cascade 混合 |
| S7 合成检测 | 合成元数据统计（`source` 字段 / generation_depth） | 依赖上游 metadata 字段，UFW-L3 无此字段 |
| S7 合成检测 | Fast-DetectGPT（Sampling Discrepancy） | 未实现 |
