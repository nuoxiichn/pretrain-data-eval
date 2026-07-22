# 已归档：预训练阶段污染检测调研及方案

## 总结

污染检测分**数据级**（训练前）和**模型级**（训练后）两大类。数据级以 n-gram 匹配为主流但阈值不统一、假阳性高；模型级以 Min-K% Prob 为代表但在真实大规模预训练场景下 MIA 表现接近随机（AUC < 0.6）。所有现有检测方法均存在已知规避手段。学术界正转向动态基准和抗污染基准设计等新范式。

关键发现：

*   **近似污染不可忽视**：仅做 n-gram 精确匹配会漏掉大量"软污染"，MMLU 上保守 web 检测发现 13.8% 污染，行为探测发现高达 72.5%。
    
*   **单一方法不可靠**：所有方法都有已知规避手段（EAL 释义攻击可完全逃避所有检测）。
    
*   **跨语言污染已被验证**：英文 benchmark 翻译为中文后训练，仍可显著提升英文分数，传统检测完全无法检出。
    
*   建议结合 n-gram 精确匹配（快速筛查）+ SoftMatcha 2 软匹配（变体污染）+ 扰动测试（诊断记忆依赖），构建多层检测架构。
    

---

## 一、方法分类全景

Xu et al. (ACL 2024 Findings) 将检测方法系统分为七大类：

| 类别 | 需要训练数据 | 需要 logit | 适用场景 |
| --- | --- | --- | --- |
| 1. 检索式（n-gram 匹配、后缀数组） | 是 | 否 | 白盒/数据级 |
| 2. 时间截断 | 否 | 否 | 通用 |
| 3. 基于掩码（cloze 补全） | 否 | 是 | 灰盒 |
| 4. 基于扰动（释义后对比） | 否 | 否/是 | 黑盒/灰盒 |
| 5. 规范顺序（排列偏好） | 否 | 是 | 灰盒 |
| 6. 行为操纵（引导式提示） | 否 | 否 | 黑盒 |
| 7. 成员推断攻击（MIA） | 否 | 是 | 灰盒 |

白盒场景可用检索式 + MIA，黑盒场景主要依赖扰动式和行为操纵方法。

> 来源: [Xu et al., ACL 2024 Findings](https://arxiv.org/abs/2406.14644)

---

## 二、训练前——数据级检测

### 2.1 精确匹配

N-gram 匹配是当前工业界污染检测的主流方法：在训练语料中搜索与 benchmark 样本重叠的 n-gram 子串，超过阈值则标记为污染。后缀数组（suffix array）和 FM-index 是让这类搜索在 TB 级语料上可行的索引结构。

*   **GPT-2** — 8-gram，word 为单位，百分比重叠阈值。([Brown et al., 2020](https://arxiv.org/abs/2005.14165))
    
*   **GPT-3** — 13-gram，character 为单位，≥8 词命中即标记。([Brown et al., 2020](https://arxiv.org/abs/2005.14165))
    
*   **GPT-4** — 50 字符子串碰撞，每个样本随机抽 3 个子串，移除空格和符号后匹配。被广泛认为不够充分：1500 字符的题目只查 3 个 50 字符片段，大部分内容被忽略。后续 DCQ 研究发现实际污染率远高于 OpenAI 报告（如 HumanEval：OpenAI 报 ~25%，DCQ 检出 55.6%–56.7%）。([OpenAI, GPT-4 Technical Report, 2023](https://arxiv.org/abs/2303.08774))
    
*   **PaLM** — 8-gram，70% 重叠阈值。([Chowdhery et al., 2022](https://arxiv.org/abs/2204.02311))
    
*   **LLaMA-2** — 10-token n-gram + 后缀数组，将样本分为 clean(<20%)、not clean(20–80%)、dirty(>80%) 三类；引入 skip-gram budget=4（允许匹配 span 中最多 4 个位置不同），比 GPT-3 的精确 13-gram 更灵活。([Touvron et al., 2023](https://arxiv.org/abs/2307.09288))
    
*   **LLaMA-3** — 8-token n-gram，经验阈值调优。([Meta, 2024](https://arxiv.org/abs/2407.21783))
    

PIQA、HellaSwag、HumanEval、MMLU、GSM8K 等主流 benchmark 均被至少一个主要模型的去污染分析标记为受污染。

**工程实现——后缀数组与 FM-index**:

后缀数组将整个训练语料拼接为一个字符串，所有后缀按字典序排序建索引，任意子串查询通过二分搜索在 O(m log n) 完成。内存开销 8 bytes/token + 原文。FM-index 是其压缩变体，存储仅 0.44x 原文大小，Infini-gram Mini 用它索引了 83TB 文本。([Lee et al., 2022](https://arxiv.org/abs/2107.06499); [Liu et al., Infini-gram, 2024](https://arxiv.org/abs/2401.17377))

**局限性**:

*   **假阳性和假阴性率高**: Jiang et al. (2024) 复现实验表明，按上述 n-gram 定义过滤掉 3%–30% 被标记 token 后，模型性能并未一致下降。将测试集分为 dirty/clean 子集对比表现相似，也不能证明模型对污染免疫。([Jiang et al., 2024](https://arxiv.org/html/2401.06059v1))
    
*   **各家阈值不统一**: n-gram 大小从 8 到 13 不等，匹配单位有 word/character/token 三种，重叠比例阈值从 20% 到 70% 不等，缺乏共识。([Xu et al., ACL 2024](https://arxiv.org/abs/2406.14644))
    
*   **无法检测改写/翻译后的污染**: 对释义、翻译、变量重命名等表面变换均不鲁棒。([arXiv:2404.00699](https://arxiv.org/abs/2404.00699))
    

### 2.2 软匹配检测

精确匹配无法捕获拼写变体、同义改写、格式微调等"软污染"。软匹配方法通过 word embedding 或编辑距离容忍一定程度的变换，在精确匹配之上提供补充检测层。

*   **SoftMatcha v1** (ICLR 2025) — 用 word embedding 扩展倒排索引，在 3.4B 词英文语料上单次查询 < 0.1s，不需 GPU。GitHub: [softmatcha/softmatcha](https://github.com/softmatcha/softmatcha)
    
*   **SoftMatcha v2** (ICML 2026) — 基于后缀数组 + 磁盘感知设计 + 动态语料感知剪枝，扩展到万亿级。在 FineWeb-Edu（1.4T tokens）上单次查询 <0.3s，支持替换、插入、删除。GitHub: [softmatcha/softmatcha2](https://github.com/softmatcha/softmatcha2)。([arXiv:2602.10908](https://arxiv.org/abs/2602.10908))
    

**实测效果**:

精确匹配标记 338/2564 样本（13.2%）为污染，SoftMatcha v2 额外发现 36 个遗漏样本（1.4%），人工验证精度 81%（29/36 真阳性）——包括 18 例语义污染（如逗号插入）和 11 例模板泄露（如数值替换）。

**局限性**:

*   **索引构建成本高**: 需要对整个语料构建 embedding 倒排索引，构建时间和磁盘开销显著
    
*   **仍依赖词级相似度**: 对深度改写（如跨语言翻译、结构重组）覆盖有限
    

### 2.3 Embedding / 语义相似度方法

用 sentence-transformer 编码训练样本和 benchmark 样本，计算 cosine similarity，超过阈值标记为近重复。相比软匹配进一步从词级扩展到句级语义。

*   **LLM Decontaminator** (UC Berkeley + SJTU) — Embedding 检索 + GPT-4 二阶段判断。在 CodeAlpaca 中检出 21 个 HumanEval 的改写样本（12.8%），13-gram 方法完全漏掉。GitHub: [lm-sys/llm-decontaminator](https://github.com/lm-sys/llm-decontaminator)
    
*   **Llama-3.1 的分层方法** — Stage 1 精确匹配 + 8-gram（>50% token 重叠即标记）；Stage 2 Sentence Transformer embedding，cosine similarity 0.75–0.95 区间捕获改写。([Meta, 2024](https://arxiv.org/abs/2407.21783))
    
*   **Hierarchical Contamination Detection** (2025.11) — 纯 embedding F1 仅 0.49，引入 embedding clustering + 分布分析后提升至 F1 0.755（+20.5%）。
    

**局限性**:

*   **阈值选择敏感**: cosine similarity 阈值从 0.75 到 0.95 不等，缺乏统一标准
    
*   **高假阳性率**: embedding 模型质量依赖强，主题相似但非污染的文本易被误判
    
*   **不可单独使用**: 需与 n-gram / 软匹配方法组合，作为候选精筛层而非独立检测
    

---

## 三、训练后——模型级检测

### 3.1 成员推断攻击（MIA）

MIA 的核心问题：模型给某段文本的 loss/概率低，到底是因为"训练时见过"还是因为"文本本身就简单"？各方法的区别在于如何校准这个信号。

**原理**: 通过模型输出的概率分布推断某段文本是否在训练集中，不需要访问训练数据本身，需要 logprobs。

**代表性方法**:

*   **LOSS Attack** (Yeom et al., 2018) — 最朴素基线：loss 低于阈值即判定为训练成员。问题是"简单"文本天然 loss 低，假阳性率高。
    
*   **Zlib 熵归一化** (Carlini et al., 2021) — 用 zlib 压缩率衡量文本本身的"简单程度"，拿模型 perplexity 除以它来校准。略优于 LOSS。([Carlini et al., 2021](https://arxiv.org/abs/2012.07805))
    
*   **参考模型校准** — 用一个小模型的 loss 作为"文本本身该多难"的基线，目标模型 loss 显著低于基线才算见过。需额外训练/访问参考模型。
    
*   **Neighbor Attack** (Mattern et al., 2023) — 对文本做微扰生成几个"邻居"版本，若模型对原文 loss 明显低于邻居们，说明记住了原文而非泛化。计算开销大。
    
*   **Min-K% Prob** (Shi et al., ICLR 2024) — 当前最流行的方法。不校准整体 loss，而是只看概率**最低的 K% token**（默认 K=20）的平均 log-probability。假设：训练集中的样本即使是"意外词"概率也不会极低，而未见过的文本会有几个概率极低的 outlier。无需参考模型（reference-free），WikiMIA 上 AUC 0.72，比此前最佳提升 7.4 点。([Shi et al., 2024](https://arxiv.org/abs/2310.16789))
    
*   **Min-K%++** (Zhang et al., 2024) — 对 token log-probability 做 z-score 归一化（均值-标准差），在 Min-K% 基础上再提升最多 10 AUC 点。([Zhang et al., 2024](https://arxiv.org/abs/2404.02936))
    
*   **LiRA** (Carlini et al., 2022) — 训练数百个影子模型（一半包含目标文本、一半不包含），用似然比检验判定。理论最优，但需训练 4000+ 参考模型，对现代 LLM 完全不实际。([Carlini et al., 2022](https://arxiv.org/abs/2112.03570))
    

**根本局限**:

Duan et al. ([COLM 2024](https://arxiv.org/abs/2402.07841)) 在 Pythia（160M–12B）上系统评估 5 种 SOTA MIA，**所有方法 ROC-AUC < 0.6**，几乎与随机猜测无异。[MIMIR](https://github.com/iamgroot42/mimir) 基准控制时间偏移后的实测数据：

| 方法 | Wikipedia 160M AUC | Wikipedia 1.4B AUC |
| --- | --- | --- |
| LOSS | 0.505 | 0.511 |
| Reference | 0.517 | 0.545 |
| Min-K% | 0.490 | 0.508 |
| Zlib | 0.517 | 0.520 |

根本原因：

1.  LLM 通常只训练 **1 个 epoch**，每条数据的记忆信号极弱
    
2.  训练数据量巨大，成员与非成员在分布上难以区分
    
3.  此前报告的高 AUC（如 WikiMIA 上的 0.72）很大程度源于**时间分布偏移**——成员/非成员按发布日期划分，风格和主题随时间变化，一个简单词袋分类器就能达到高 AUC
    

[Hayes et al. (2025)](https://arxiv.org/abs/2505.18773) 将 LiRA 扩展到 GPT-2 架构，训练超过 4000 个参考模型（10M–1B），最佳情况 AUC 也仅 0.753（中等数据集 ~100 万样本），大数据集降到 0.7 以下。即使整体 AUC 高于随机，许多单样本的推断与抛硬币无统计差异。

### 3.2 行为探测类方法

通过各种方式"试探"模型对 benchmark 的记忆程度——给前缀让它续写、打乱顺序看它是否偏好原始排列、反复采样看输出是否过于集中。不需要访问训练数据，部分方法需要 logprobs（灰盒），部分纯黑盒。

**代表性方法**:

*   **Guided Instruction** ([Golchin & Surdeanu, ICLR 2024](https://arxiv.org/abs/2308.14352)) — 给模型提供数据集名称 + 分区类型 + 原始实例前缀，要求续写；对照组只给前缀不告知数据集名称。若 guided 续写与真实后缀的 ROUGE-L/BLEURT 显著高于对照 → 污染。测试 7 个数据集，准确率 92%–100%。黑盒，但可能被安全过滤器阻断。
    
*   **DCQ (Data Contamination Quiz)** ([Golchin & Surdeanu, TACL 2025](https://github.com/shahriargolchin/DCQ)) — 将检测转化为多选题：用 GPT-4 对原始实例生成同义词替换的扰动版本作为干扰项，构造 5 选 1，让模型选"哪个是原始版本"。黑盒，可绕过版权过滤器。检测发现 GPT-4 实际污染率远超 OpenAI 报告（HumanEval: OpenAI ~25% vs DCQ 55.6%；GSM8K: DCQ 78.8%）。
    
*   **TS-Guessing** ([Deng et al., NAACL 2024](https://arxiv.org/abs/2310.16789)) — 将 QA 基准转化为 cloze 测试，遮蔽一个**错误选项**让模型猜。猜中错误答案只能靠记忆不能靠推理。MMLU 上 ChatGPT 精确匹配率 52%、GPT-4 57%，强烈暗示污染。黑盒，仅适用 QA 格式。
    
*   **分片排列检验** ([Oren et al., ICLR 2024](https://arxiv.org/abs/2310.17623)) — 若模型未见过某数据集，它对任何排列应赋予相同似然度；若见过，会偏好原始顺序。将测试集分片，比较原始顺序 vs 随机打乱的 log-likelihood，做单侧 t 检验。**唯一有数学 FPR 保证的方法**。灰盒（需 logprobs），仅需 1000 样本即可检出，但只能做数据集级判定，无法定位单个样本。
    
*   **选项排列检测** ([Yang et al., AAAI 2024](https://arxiv.org/abs/2305.10403)) — 打乱多选题的选项顺序，看模型是否偏好原始选项位置。灰盒（需 logprobs），专用于多选题格式。在 C-Eval 上发现 Qwen 系列泄漏值是其他模型的近 10 倍；Qwen2-72B 在 CMB 上 42% 数据泄漏。
    
*   **CDD (输出分布检测)** ([Dong et al., ACL Findings 2024](https://arxiv.org/abs/2310.16789)) — 反复采样看输出是否过于集中（尖锐）。训练过的数据会使输出分布变得尖锐——每次生成几乎一样。纯黑盒，但后续研究发现在小模型上基本等于随机猜测，Perplexity 和 Min-K% 在所有条件下都优于 CDD。
    

**局限性**:

*   **补全式和 DCQ 依赖生成质量**: 安全过滤器可能阻断续写，DCQ 需要 GPT-4 生成扰动项（成本高）
    
*   **排列检验无法定位单样本**: 只能做数据集级判定；且假设可通过将 benchmark 与大量预训练数据混合来破坏
    
*   **CDD 不可靠**: 输出分布的"尖锐度"不只受记忆影响——简单题、格式化题天然输出就很集中，与 LOSS Attack 同样的校准问题
    
*   **选项排列仅适用多选题格式**: 对生成式、代码类 benchmark 无法使用
    

---

## 四、中文数据的污染检测注意事项

### 4.1 中文污染现状与跨语言污染

[Yang et al. (AAAI 2024)](https://arxiv.org/abs/2305.10403) 对 35 个模型做选项排列检测，发现中文基准污染严重：

*   C-Eval 泄漏最高的 5 个模型全部来自 Qwen 系列，泄漏值是其他模型的近 10 倍
    
*   Qwen2-72B 在 CMB 上 42% 数据泄漏
    
*   风险最低：GLM4-9B、MiniCPM3-4B
    

[Yao et al. (EMNLP 2024)](https://arxiv.org/abs/2406.13236) 首次验证跨语言污染：英文 benchmark 翻译为中文/法语/德语后用于训练，可**显著提升英文分数**，传统基于文本重叠的检测**完全无法检出**。这意味着检测中文数据时，不仅要匹配中文 benchmark，还需匹配英文 benchmark 的中文翻译版本。

### 4.2 检测中文数据时的技术难点

1.  **分词问题**: n-gram 匹配对中文效果差（无自然空格分词），需先分词或用字符级/token 级匹配
    
2.  **改写逃逸**: 中文同义改写极为丰富（"计算"↔"求解"↔"算出"），简单文本重叠检测更易被绕过，软匹配（SoftMatcha）的必要性比英文更高
    

### 4.3 抗污染 Benchmark 选型

评估模型是否受污染时，可优先选用以下不易被污染的 benchmark 作为检测参照：

*   **MMLU-CF** — 专门设计的无污染版 MMLU 替代方案，改写了原始题目以避免与训练数据重叠
    
*   **LiveBench** — 滚动更新，持续从最新来源出题，数据天然晚于训练截止日期
    
*   **LiveCodeBench** — 代码评测的滚动更新版本，同理
    
*   **LatestEval** — 用最新语料自动生成评测题目
    

---

## 五、开源工具全景

### 5.1 数据级检测工具

| 工具 | 来源 | 方法 | 规模 | GitHub |
| --- | --- | --- | --- | --- |
| **Infini-gram Mini** | — | FM-index | **83TB 已验证** | [infini-gram-mini.io](http://infini-gram-mini.io/) |
| **SoftMatcha v2** | ICML 2026 | 后缀数组 + 软匹配 | 1.4T token | [softmatcha/softmatcha2](https://github.com/softmatcha/softmatcha2) |
| **SoftMatcha v1** | ICLR 2025 | embedding 倒排索引 | 3.4B 词 | [softmatcha/softmatcha](https://github.com/softmatcha/softmatcha) |
| **Data Portraits** | NeurIPS 2023 | Bloom filter | ~3% overhead | [ruyimarone/data-portraits](https://github.com/ruyimarone/data-portraits) |
| **LLM Decontaminator** | UC Berkeley | Embedding + GPT-4 | 候选级 | [lm-sys/llm-decontaminator](https://github.com/lm-sys/llm-decontaminator) |
| **Google deduplicate-text-datasets** | Lee 2022 | Suffix array | TB 级 | google-research/deduplicate-text-datasets |
| **ROOTS Search Tool** | BigScience | 全文搜索 | 1.6TB, 59 语言 | [huggingface/roots-search-tool](https://github.com/huggingface/roots-search-tool) |
| **EleutherAI lm-eval-harness** | — | 13-gram decontamination | The Pile/C4 | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) |
| **OLMoTrace** | Ai2, ACL 2025 | infini-gram 扩展 | 4.6T token | [arXiv:2504.07096](https://arxiv.org/abs/2504.07096) |

### 5.2 模型级检测工具

| 工具 | 方法 | 访问要求 | GitHub |
| --- | --- | --- | --- |
| **detect-pretrain-code** | Min-K% Prob + WikiMIA | 灰盒（logprobs） | [swj0419/detect-pretrain-code](https://github.com/swj0419/detect-pretrain-code) |
| **MIMIR** | 统一 MIA 评估框架 | 灰盒 | [iamgroot42/mimir](https://github.com/iamgroot42/mimir) |
| **DCQ** | 多选题式检测 | 黑盒 | [shahriargolchin/DCQ](https://github.com/shahriargolchin/DCQ) |
| **Contamination Detector** | Bing Search + CC Index | 黑盒 | [liyucheng09/Contamination\_Detector](https://github.com/liyucheng09/Contamination_Detector) |

### 5.3 参考资源

| 资源 | 类型 | 链接 |
| --- | --- | --- |
| **LM Contamination Index** | 人工标注污染证据库 | [hitz-zentroa/lm-contamination](https://github.com/hitz-zentroa/lm-contamination) |
| **Infini-gram Mini Bulletin** | 自动化污染率报告 | [infini-gram-mini.io](http://infini-gram-mini.io/) |
| **awesome-data-contamination** | 论文索引（50+ 方法，100+ 论文） | [lyy1994/awesome-data-contamination](https://github.com/lyy1994/awesome-data-contamination) |
| **DCLM 53 任务清单** | Benchmark 注册表 | DataComp-LM |

### 5.4 2025–2026 新工具

| 工具 | 时间 | 特点 |
| --- | --- | --- |
| **CoDeC** | 2025.10 | 轻量级 model-agnostic，通过 in-context examples 影响检测，每个 benchmark 仅需几分钟 |
| **SimMIA** | 2026.01 | 纯黑盒 MIA，仅需生成文本 |
| **BenchMarker** | 2026.02 | 用互联网搜索作为训练数据代理，model-agnostic |
| **Tracer** | 2025 | 代码 LLM 的语义感知细粒度污染检测 |

---

## 六、方法综合对比

| 方法类别 | 代表工具 / 论文 | 类型 | 访问要求 | 粒度 | 主要局限 | 规模化 |
| --- | --- | --- | --- | --- | --- | --- |
| N-gram 精确匹配 | Suffix Array, FM-index (Infini-gram), lm-eval-harness | 数据级 | 白盒 | 实例级 | 假阳性高，无法检测改写 | **83TB 已验证** |
| 软模式匹配 | SoftMatcha v1/v2 | 数据级 | 白盒 | 实例级 | 索引构建重 | 1.4T 已验证 |
| Embedding 语义相似度 | LLM Decontaminator, sentence-transformers + FAISS | 数据级 | 白盒 | 实例级 | 阈值敏感，高假阳性 | 候选级 |
| 成员推断攻击（MIA） | Min-K% Prob, Min-K%++, LiRA, MIMIR | 模型级 | 灰盒（logprobs） | 实例级 | 实际 AUC ~0.5 | 高 |
| 补全式探测 | Guided Instruction, DCQ, TS-Guessing | 模型级 | 黑盒 | 实例级 | 安全过滤器阻断；DCQ 依赖 GPT-4；TS-Guessing 仅 QA 格式 | 中 |
| 排列检验 | Oren 分片检验（**唯一有 FPR 数学保证**）, 选项排列检测 | 模型级 | 灰盒（logprobs） | 数据集级 / 实例级 | Oren 无法定位单样本；选项排列仅多选题 | 高 |
| 输出分布检测 | CDD | 模型级 | 黑盒 | 实例级 | 小模型上≈随机猜测 | 中 |
| 扰动测试 | entity swap / 否定 / 改写 | 诊断 | 黑盒/灰盒 | 实例级 | 需针对性设计扰动 | 中 |

**数据级工具的 TB 级规模可行性**:

| 工具 | 1TB | 10TB | 内存需求 | 索引时间 | 查询吞吐 |
| --- | --- | --- | --- | --- | --- |
| **Suffix Array** | 可行 | 困难（需分片） | 8 bytes/token + 原文 | 线性，~数天 | ~20ms/次 |
| **FM-index** | 可行 | **已验证 83TB** | 0.44x 原文 | 较长但已工程化 | 毫秒级 |
| **Data Portraits** | 可行 | 可行 | ~3% 数据集大小 | 线性扫描 | 快速 |
| **SoftMatcha v2** | 可行 | 可行（1.4T 已验证） | 倒排索引 + embedding | 较重 | 亚秒级 |
| **MinHash/LSH** | 可行 | 可行 | 中等（hash 签名） | 线性 | 高吞吐 |
| **Embedding 检索** | 有条件 | 困难 | 每样本 1 embedding + FAISS | 推理瓶颈（需 GPU） | 中等 |
| **LLM 判断** | 仅候选 | 不可行 | N/A | N/A | 极低 |

---

## 七、对 Pipeline 设计的建议

### 7.1 数据级评测链路

| 层次 | 方法 | 工具选型 | 作用 |
| --- | --- | --- | --- |
| **第一层** | n-gram 精确匹配 | 已有 `exact`；或复用 EleutherAI decontamination | 大规模快速筛查 |
| **第二层** | 软匹配 | SoftMatcha v2 | 捕获释义/模板变体 |
| **第三层** | Embedding 近似检测 | sentence-transformers + FAISS，阈值 0.75–0.95 | 语义级近重复 |
| **第四层（可选）** | LLM 精判 | 对第三层候选用 LLM 做语义判断 | 小量候选精确确认 |

### 7.2 模型级评测链路——方法选型与落地

调研显示模型级检测方法众多但均有局限、无统一标准。选型应分两步：先用文献结论 + 场景约束做预筛，再在自己数据上做轻量 sanity check。

#### 第一步：基于场景约束预筛

基于真实约束 + 文献中已有的效果数据，做如下预筛：

| 方法 | 文献结论 | 适用性判定 | 决策 |
| --- | --- | --- | --- |
| **Min-K% Prob / ++** | MIMIR 控制时间偏移后 AUC ~0.5（第三章表格）；WikiMIA 上的 0.72 受时间偏移虚高 | 作为独立判据不可靠，但计算成本极低（单次前向传播），可作为**辅助信号** | **保留为辅助信号** |
| **Oren 分片排列检验** | 唯一有数学 FPR 保证；1000 样本即可做数据集级判定；不受时间偏移影响 | 我们有 logprobs，可直接用；缺点是只能判定数据集级，无法定位单样本 | **保留为主信号** |
| **选项排列检测** | Yang et al. 在 C-Eval/CMB 上已验证中文有效性；Qwen 系列检出率高 | 有 logprobs 可用；但仅适用多选题格式 | **保留，限多选题** |
| **Guided Instruction** | 准确率 92%–100%；纯黑盒；但可能被安全过滤器阻断 | 自研模型无安全过滤器问题；实现简单 | **保留** |
| **TS-Guessing** | MMLU 上 ChatGPT 52%、GPT-4 57%；仅适用 QA 格式 | 仅多选题；与选项排列信号部分重叠 | **备选**（若选项排列已覆盖则不需要） |
| **DCQ** | 检出率高（HumanEval 55.6%）；但依赖 GPT-4 生成扰动项，成本高 | 不依赖外部 API 的约束下优先级低 | **暂不采用** |
| **CDD** | 后续研究发现小模型上≈随机猜测；Perplexity 和 Min-K% 在所有条件下都优于 CDD | 不可靠 | **淘汰** |

#### 第二步： Sanity Check

具体做法：

1.  **选 2–3 个已知被广泛污染的 benchmark**（如 MMLU、C-Eval、GSM8K）和 **1 个训练截止后发布的新 benchmark**（如 LiveBench 最新一期）作为正/负参照
    
2.  对自研模型跑保留的 4 个方法，检查：
    
    *   Oren 排列检验：已知污染的 benchmark 上 p-value 是否显著低于新 benchmark？
        
    *   选项排列：已知污染的多选题上泄漏值是否明显高于新 benchmark？
        
    *   Guided Instruction：对已知污染数据集的续写 ROUGE-L 是否显著高于对照？
        
    *   Min-K%：分数分布是否在已知污染/未污染之间有可见差异？（即使差异小也记录）
        
3.  **一致性检查**：多个方法对同一 benchmark 的判定是否一致？不一致的需要人工分析原因
    

> 这一步的目的不是评估方法本身，而是确认方法在我们的模型上没有异常（如 tokenizer 不兼容、中文分词导致排列检验失效等）。

#### 第三步：定型评测链路

```plaintext
自研模型 + benchmark 参考集（8.3）
    │
    ├─ 信号 1: Oren 分片排列检验 → p-value（数据集级，主信号）
    ├─ 信号 2: 选项排列检测 → 泄漏值（多选题 benchmark）
    ├─ 信号 3: Guided Instruction → ROUGE-L 差值（补全式，通用格式）
    ├─ 信号 4: Min-K% Prob → 分数分布（辅助参考，不单独做判据）
    │
    ├─ 扰动诊断: entity swap / 否定 / 改写 → 性能下降幅度
    │
    └─ 综合判定:
         - Oren p < 0.01 且 ≥2 个其他信号异常 → 红灯（高度疑似污染）
         - 仅 1 个信号异常 → 黄灯（需人工复核）
         - 全部正常 → 绿灯
         - 扰动下降 >15% → 额外标记"记忆依赖风险"

```

### 7.3 Benchmark 参考集

数据级和模型级检测都需要一个 benchmark 参考集作为匹配/探测的对象。对齐预训练模型效果评测方案中的链路注册表。

### 7.4 开放问题

1.  **规模化**: 10T 级自研数据上 SoftMatcha 2 的内存/构建时间？是否需分片索引？FM-index 是目前唯一 83TB 验证的方案
    
2.  **中文适配**: CJK 分词差异影响 n-gram 匹配；中文同义改写丰富使软匹配更重要
    
3.  **跨语言污染**: 翻译后训练可提升原语言分数，需跨语言检测能力
    
4.  **近似污染**: 仅精确匹配远不够（MMLU 13.8% vs 72.5%），软匹配和扰动测试必不可少
    

---

## 八、核心论文/工具索引

| 论文 | 方法 | 年份/会议 | 类型 |
| --- | --- | --- | --- |
| Carlini et al. "Extracting Training Data from LLMs" | 训练数据提取 | USENIX 2021 | 记忆 |
| Carlini et al. "Quantifying Memorization Across Neural LMs" | 记忆量化 | ICLR 2023 | 记忆 |
| Carlini et al. "Membership Inference From First Principles" | LiRA | 2022 | MIA |
| Zhang et al. "Counterfactual Memorization" | 反事实记忆 | NeurIPS 2023 | 记忆 |
| Shi et al. "Detecting Pretraining Data from LLMs" | Min-K% Prob + WikiMIA | ICLR 2024 | 灰盒 MIA |
| Zhang et al. "Min-K%++" | Min-K% 改进 | 2024 | MIA |
| Oren et al. "Proving Test Set Contamination in Black Box LMs" | 分片似然比检验 | ICLR 2024 | 统计检验 |
| Golchin & Surdeanu "Time Travel Benchmark" | 引导提示法 | ICLR 2024 | 黑盒 |
| Golchin & Surdeanu "DCQ" | 多选题式检测 | TACL 2025 | 黑盒 |
| Deng et al. "TS-Guessing" | Cloze 检测 | NAACL 2024 | 黑盒 |
| Dong et al. "CDD" | 输出分布检测 | ACL 2024 | 黑盒 |
| Dekoninck et al. "Evading Data Contamination Detection" | EAL 规避方法 | 2024 | 对抗 |
| Jiang et al. "Investigating Data Contamination" | 从零受控实验 | 2024 | 实证 |
| Duan et al. "Do MIAs Work on LLMs?" | MIA 系统评估 | COLM 2024 | 实证 |
| Hayes et al. "Strong MIA on Large LMs" | LiRA 扩展 | 2025 | 实证 |
| Yang et al. "Training on the Benchmark Is Not All You Need" | 选项排列检测 | AAAI 2024 | 中文 |
| Yao et al. "Data Contamination Can Cross Language Barriers" | 跨语言污染 | EMNLP 2024 | 跨语言 |
| Nasr & Carlini et al. "Scalable Extraction from Production LMs" | 发散攻击 | 2023 | 提取 |
| SoftMatcha v1 | embedding 软匹配 | ICLR 2025 | 工具 |
| SoftMatcha v2 | 万亿级软匹配 | ICML 2026 | 工具 |
| Xu et al. "Benchmark Data Contamination of LLMs" | 七类分类综述 | ACL 2024 | 综述 |
| Chen et al. "Dynamic Benchmarks" | 动态基准综述 | EMNLP 2025 | 综述 |
| Al-Lawati et al. "Contamination-Resistant Benchmarks" | 抗污染基准设计 | 2026 | 立场论文 |
| Yang et al. "Rethinking Benchmark and Contamination" | 释义规避 | 2023 | 实证 |
| Lee et al. "Deduplicating Training Data" | Suffix array 去重 | 2022 | 工具 |
| Mosaic Memory | 模糊重复记忆 | 2024 | 记忆 |

## TODO：

预训练的污染和SFT的污染是否有方法、影响上的不同
