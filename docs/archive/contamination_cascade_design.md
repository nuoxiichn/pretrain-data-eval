# 已归档：Stage 5 级联污染检测实施方案

> 配套调研：`预训练阶段污染检测调研及方案.md`。本文档只给具体实施路径，不重复调研结论。
> 范围调整：原 v3 计划将「软匹配/多路综合」推迟到 v4，本方案将其前移并入 P2。**P2 内部分两个独立交付物**（P2a 中文 benchmark 覆盖、P2b 三层 cascade 方法），保证回归对照不混合。

---

## 1. 背景与决策

### 1.1 v2 现状的两个缺口

| 缺口 | 证据 | 影响 |
|------|------|------|
| 中文 benchmark 完全无覆盖 | UFW-ZH 360M 全量 0% 命中（无对应 benchmark） | ZH 污染率不可信 |
| 仅精确匹配 → 漏检软污染 | MMLU 全量 EN 698M 段落级 2 条命中，文档级 0 | EAL/释义/翻译/模板变体全漏，调研 §1 给出 13.8% vs 72.5% 的差异 |

### 1.2 为什么用 cascade 而不是单层加固

- 单纯换软匹配（如 SoftMatcha v2）：内存/构建成本数十 TB 级，且仍漏 cross-lingual 与深度释义。
- 单纯换 embedding：单层假阳性高（调研 §2.3 纯 embedding F1 仅 0.49），且 10T 全量推理不可行（BGE-m3 GPU ~50-100 docs/s → 3000+ GPU·h）。
- **Cascade 的本质收益**：用各层互补的召回/精度特性 + 路由策略，让 expensive layer 只跑在前层划出的候选子集上。

### 1.3 不上 SoftMatcha v2 的理由

SoftMatcha v2 是独立科研仓（万亿级后缀数组 + 磁盘感知），集成需要：(1) 重建 benchmark + corpus 的整体倒排索引；(2) 与本仓 `pretrain_data_eval/reader.py` 的 streaming Document 契约不兼容。性价比低于 **char-MinHash near-dup（已有原型）+ BGE-m3 embedding（已有 semdedup 复用模式）** 的组合。v4 再评估。

---

## 2. 三层 Cascade 架构

```
                    ┌─────────────────────────────────────────────────┐
                    │  Benchmark 索引（启动时一次性构建，常驻内存）   │
                    │  ├─ L1 hash 索引（已有 build_bench_index）      │
                    │  ├─ L2 MinHash 签名 + LSH 桶（新）              │
                    │  └─ L3 BGE-m3 embedding + FAISS IndexFlatIP（新）│
                    └─────────────────────────────────────────────────┘
                                          │
            每篇 doc 进入 cascade ─────────┘
                    │
        ┌───────────▼──────────┐
        │ L1 exact hash 匹配   │   成本：µs/doc
        └───────────┬──────────┘
                    │ hit?
              ┌─────┴─────┐
              │           │
            yes           no
              │           │
              ▼           ▼
        🔴 红灯  ┌─────────────────────┐
        终止    │ L2 MinHash 近似匹配 │   成本：~ms/doc（xxhash + LSH 桶查询）
                └──────────┬──────────┘
                           │ jaccard ∈ ?
                ┌──────────┼──────────┬────────────┐
                ▼          ▼          ▼            ▼
              ≥0.9       0.5–0.9    0.3–0.5      <0.3
                │          │          │             │
                ▼          ▼          ▼             ▼
              🔴红      → L3      → L3        ✅ 绿灯
                       (验证)    (深度释义)     终止
                          │
                          ▼
              ┌─────────────────────────┐
              │ L3 Embedding cos 相似度 │   成本：~10ms/doc（GPU 批处理）
              │ (BGE-m3 + FAISS top-k)  │
              └────────────┬────────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
            cos ≥ 0.85            cos < 0.85
                │                     │
                ▼                     ▼
              🔴红               🟡黄/✅绿
```

### 2.1 各层职责

| 层 | 检测对象 | 复用 / 新建 | 成本估算（1B docs） |
|----|----------|-------------|---------------------|
| **L1 exact** | 完全 byte 级一致 | 复用 `compute_exact_contamination` | ~µs/doc，无 GPU |
| **L2 MinHash near-dup** | 释义/模板/数值替换/标点变体（char 5-gram Jaccard ≥ τ） | 把 `compute_code_near` 从代码扩展到通用文本，迁移 `stages/dedup/utils.py:153 _minhash_signature_fast` 的 xxhash 实现替换现有 MD5 版本 | ~2-5ms/doc，CPU；64 核机约 35-90h |
| **L3 Embedding** | 深度释义/翻译/跨语言 | 新增；复用 `stages/dedup/utils.py:504 _encode_embeddings` 模式（BGE-m3 CLS pooling + L2 norm），但用 FAISS `IndexFlatIP` 做 benchmark 端检索 | 仅跑候选子集（L2 jaccard ≥ 0.3 的 docs），预计 0.1-1% → 1M-10M 个 doc → 3-30 GPU·h |

### 2.2 路由阈值（初版，sanity check 后调优）

| L2 jaccard | 进入下一层 | 终态 |
|------------|-----------|------|
| ≥ 0.9 | 不进 L3 | 🔴 red（high-confidence 污染） |
| [0.5, 0.9) | → L3 验证 | L3 cos ≥ 0.85 → red；cos ∈ [0.7, 0.85) → 🟡 yellow；< 0.7 → ✅ green |
| [0.3, 0.5) | → L3（深度释义/跨语言） | 同上 |
| < 0.3 | 不进 L3 | ✅ green |

> 阈值来源：`code-near` 已用 0.85 verified，0.3 为 MinHash 在 char 5-gram 下「主题相关但非污染」的经验下界（参见 LLM Decontaminator 0.75 句子级 cosine 对应 jaccard ~0.3-0.4）。需在 sanity check 阶段对照已知污染/已知干净 benchmark 校准。

### 2.3 跨语言污染处理

调研 §4.1 指出英文 benchmark 翻译为中文后训练仍可显著提升英文分数。L3 用 **BGE-m3（多语对齐 embedding）** 自然覆盖：
- benchmark 端同时索引 **原文 + 自动翻译版**（仅对 MMLU/GSM8K/HumanEval 这类高风险 EN benchmark 生成 zh 翻译，benchmark 量小，翻译一次永久缓存）。
- doc 端不变。
- 文档若与英文 benchmark 的中文翻译版 cos ≥ 0.85，触发"跨语言污染"flag。

---

## 3. 文件级改动

### 3.1 新增 / 修改文件

| 路径 | 动作 | 关键内容 |
|------|------|---------|
| `stages/contamination/run.py` | 新增子命令 `near` / `embed` / `cascade`，保留 `exact` / `code-near` / `code-ast` | CLI 骨架照 `exact` 模式 |
| `stages/contamination/utils.py` | 新增 `compute_near_contamination` / `compute_embed_contamination` / `compute_cascade_contamination`；把 `_minhash_signature` 切换到 `dedup/utils.py` 的 xxhash 实现 | 见 §3.2 函数签名 |
| `stages/contamination/benchmarks.py` | 加 `translation_path` 字段支持「benchmark + 中文翻译」双索引；config 里可指定预生成的翻译 JSONL | 翻译生成离线一次性脚本，不在本仓常驻 |
| `stages/contamination/bench_index.py` **（新）** | 抽离 benchmark 端索引构建：hash / MinHash / embedding 三套索引的统一持久化（pickle / faiss .index），让 cascade 启动无需重算 | 详见 §3.3 |
| `configs/stage5_v3.yaml` **（新）** | v3 专用配置，含中文 benchmark + cascade 阈值；v2 `stage5.yaml` 不动 | 详见 §3.4 |
| `scripts/contamination_v3_batch.sh` **（新）** | 全量分文件批跑 cascade，断点续跑 | 照 `scripts/toxicity_v3_batch.sh` 模式 |
| `stages/contamination/README.md` | 加 cascade 子命令表 + 路由说明 | — |
| `pipeline_overview.md` | 更新 v3 P2 章节：标注 cascade 提前，给出新输出字段 | 等本方案确认后再改 |

### 3.2 核心函数签名

```python
# utils.py 新增

def compute_near_contamination(
    docs: Iterable[Document],
    bench_items: dict[str, list[BenchItem]],
    ngram_size: int = 5,              # CJK 默认 char 5-gram；西文同样 char-level
    jaccard_threshold: float = 0.85,
    num_hashes: int = 128,
    num_bands: int = 16,
    band_size: int = 8,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """L2 通用文本 MinHash 近似匹配。与 compute_code_near 区别：
       - 不要求 code_field，对 benchmark 所有 text + code 字段都建签名
       - 在 DocResult.scores 增加 jaccard_max_per_benchmark dict（每个 benchmark 的最高 Jaccard）"""


def compute_embed_contamination(
    docs: Iterable[Document],
    bench_items: dict[str, list[BenchItem]],
    *,
    model_path: str = "/mnt/public/model/bge-m3",
    cos_threshold: float = 0.85,
    top_k: int = 5,                   # FAISS top-k 检索
    batch_size: int = 64,
    max_length: int = 512,
    device: str = "cuda",
    candidate_filter: Callable[[Document], bool] | None = None,  # cascade 路由用
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """L3。FAISS IndexFlatIP（benchmark 量小，flat 足够）。
       cascade 模式下 candidate_filter 由 L2 jaccard 决定是否进 L3。"""


def compute_cascade_contamination(
    docs: Iterable[Document],
    bench_items: dict[str, list[BenchItem]],
    *,
    cascade_config: dict,             # 路由阈值表
    embed_model_path: str,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """编排三层。每个 doc 的 DocResult 形如：
       scores = {
         "l1_exact_hit": bool,
         "l2_jaccard_max": float,
         "l2_matched_benchmarks": [str, ...],
         "l3_cos_max": float | None,         # None 表示未进 L3
         "l3_matched_bench_id": str | None,
         "l3_matched_language": "zh" | "en" | "cross",  # 跨语言污染标识
       }
       flags = {
         "is_exact_contaminated": bool,
         "is_near_contaminated": bool,
         "is_semantic_contaminated": bool,
         "is_cross_lingual": bool,
         "verdict": "red" | "yellow" | "green",
       }
       summary 增加 cost_breakdown（每层处理 docs 数）+ verdict 三色分布。"""
```

### 3.3 Benchmark 索引持久化（`bench_index.py`）

启动 cascade 子命令前先跑：

```bash
PYTHONPATH=. python stages/contamination/bench_index.py build \
  --config configs/stage5_v3.yaml \
  --out-dir /mnt/public/data/contamination_index_v3/
```

产物：
```
/mnt/public/data/contamination_index_v3/
  ├─ hash_index.pkl          # doc_index + para_index（dict[md5] → set[label]）
  ├─ minhash_sigs.npy        # [N_bench, num_hashes] uint32
  ├─ minhash_meta.json       # bench_id / benchmark 列表，与 sigs 行对应
  ├─ minhash_lsh.pkl         # bucket → [row_idx, ...]
  ├─ embeddings.npy          # [N_bench, 1024] float16，含 zh 翻译版
  ├─ embed_meta.json         # bench_id / benchmark / language 列表
  └─ faiss.index             # IndexFlatIP
```

子命令运行时 mmap 加载，避免每次重建。

### 3.4 配置示例（`configs/stage5_v3.yaml` 关键段）

> **2026-06-29 注册表对齐评测更新**：v3 P2 cascade 代码完成后，与评测同事确认实际评测 benchmark 清单，**完全重写本节注册表**。下面 yaml 是当前生效版本（去重后 12 个 jsonl）。原 v3 初版注册表（MMLU/HellaSwag/ARC/HumanEval/GSM8K/CMMLU/C-Eval/AGIEval-zh/CMB）已全部移除——污染检测的目标必须等于评测目标，否则检测的是无关物。
>
> 评测清单（用户口径）：
> - **Pretrain**: MMLU-Pro / GPQA-Diamond / MATH / EvalPlus(HEval+ + MBPP+) / LiveCodeBench / MGSM / MMMLU
> - **SFT**: IFEval / SimpleQA / LiveBench / MMLU-Pro / GPQA-Diamond / AIME 2025 / MATH-500 / LiveCodeBench
> - 中文 benchmark（CMMLU/C-Eval/AGIEval/CMB）**全部移除**，严格对齐评测
> - MATH 与 MATH-500 都保留：评测同事两个都用，索引 L1 hash 阶段自然去重重叠题
>
> 12 个 jsonl 由 `scripts/build_eval_aligned_benchmarks.py` 一次性抽取，输出 `/mnt/public/data/contamination_v3_benchmarks/eval_aligned/`，schema 统一为 `{id, question, options?, answer?, subject?, code?}`，text_field=question，code_field=code（仅 EvalPlus）。
>
> 旧索引/旧 outputs 重命名为 `*_legacy_pre_eval_alignment_20260629`，作为方法学验证证据保留，不进 v3 报告。

```yaml
benchmarks:
  cache_dir: /mnt/public/data
  index_dir: /mnt/public/data/contamination_index_v3   # 预构建索引位置
  datasets:
    # ── Pretrain 评测清单 ──
    - {path: /mnt/public/data/contamination_v3_benchmarks/eval_aligned/mmlu_pro.jsonl,      text_field: question, label: mmlu_pro}
    - {path: /mnt/public/data/contamination_v3_benchmarks/eval_aligned/gpqa_diamond.jsonl,  text_field: question, label: gpqa_diamond}
    - {path: /mnt/public/data/contamination_v3_benchmarks/eval_aligned/math.jsonl,          text_field: question, label: math}
    - {path: /mnt/public/data/contamination_v3_benchmarks/eval_aligned/evalplus.jsonl,      text_field: question, code_field: code, label: evalplus}
    - {path: /mnt/public/data/contamination_v3_benchmarks/eval_aligned/livecodebench.jsonl, text_field: question, label: livecodebench}
    - {path: /mnt/public/data/contamination_v3_benchmarks/eval_aligned/mgsm.jsonl,          text_field: question, label: mgsm}
    - {path: /mnt/public/data/contamination_v3_benchmarks/eval_aligned/mmmlu.jsonl,         text_field: question, label: mmmlu}
    # ── SFT 评测清单（与 Pretrain 重叠的不重复列） ──
    - {path: /mnt/public/data/contamination_v3_benchmarks/eval_aligned/ifeval.jsonl,        text_field: question, label: ifeval}
    - {path: /mnt/public/data/contamination_v3_benchmarks/eval_aligned/simpleqa.jsonl,      text_field: question, label: simpleqa}
    - {path: /mnt/public/data/contamination_v3_benchmarks/eval_aligned/livebench.jsonl,     text_field: question, label: livebench}
    - {path: /mnt/public/data/contamination_v3_benchmarks/eval_aligned/aime_2025.jsonl,     text_field: question, label: aime_2025}
    - {path: /mnt/public/data/contamination_v3_benchmarks/eval_aligned/math_500.jsonl,      text_field: question, label: math_500}
```

每个 benchmark 的本地数据来源（含 row 数）：

| label | rows | 本地数据源 |
|------|------|---|
| mmlu_pro | 12,032 | `datasets--TIGER-Lab--MMLU-Pro` HF cache parquet |
| gpqa_diamond | 198 | `chennuoxi/hf_cache/Idavidrein___gpqa/gpqa_diamond` arrow |
| math | 5,000 | `datasets--EleutherAI--hendrycks_math` 7 子科目 test parquet |
| math_500 | 500 | `chennuoxi/hf_cache/HuggingFaceH4___math-500` arrow |
| evalplus | 542 | HumanEvalPlus(164, HF parquet) + MBPPPlus(378, data_mixture jsonl) |
| livecodebench | 1,055 | `datasets--livecodebench--code_generation_lite` test*.jsonl 全合并去重 |
| mgsm | 2,750 | `datasets--juletxara--mgsm` 11 语言 test parquet |
| mmmlu | 196,588 | `datasets--openai--MMMLU` 15 语言 CSV |
| ifeval | 541 | `datasets--google--IFEval` jsonl |
| simpleqa | 4,326 | `chennuoxi/hf_cache/basicv8vc___simple_qa` arrow |
| livebench | 1,436 | `datasets--livebench--{coding,math,reasoning,language,data_analysis,instruction_following}` |
| aime_2025 | 30 | `datasets--test-time-compute--aime_2025` test parquet |

**总计 224,998 条**。L1 hash 索引 doc=222,731 / para=192,126；L2 MinHash sigs (225,540, 128)；L3 BGE-m3 embeddings (224,998, 1024)。

> 跨语言污染镜像（v3 初版 mmlu_zh_mirror / gsm8k_zh_mirror）也一并移除——评测清单本身已含 MGSM（GSM8K 多语版）和 MMMLU（MMLU 多语版），原生覆盖跨语言场景，无需额外翻译镜像。

---

### 3.5 Known-Gap：L3 路由漏检（2026-06-29 sanity 暴露）

**问题**：cascade 的 L2→L3 路由用 `l2_best_jaccard >= enter_l3_low` 决定是否进 L3，导致 **L2 jaccard 极低但语义高度重合的样本被直接判 green，L3 BGE-m3 形同虚设**。

**Sanity 25 条结果**（`data/sanity_contamination.jsonl`）：

| 模式 | 期望 | 实际 (0.30) | 阈值降到 0.20 | 仍漏 |
|---|---|---|---|---|
| L1_doc (全文复制) | 5 red | 5 red ✓ | 5 red | 0 |
| L1_para (段落嵌入) | 5 red | 5 red ✓ | 5 red | 0 |
| L2_near (轻改写) | 5 red/yellow | 4 red (simpleqa 漏) | 4 red | 1 |
| **L3_cos (语义重写)** | **5 red/yellow** | **0 ❌** | **预期救回 2** | **仍漏 3** |
| clean | 5 green | 5 green ✓ | 5 green | 0 |

5 条 L3_cos 的实测 `l2_best_jaccard`：`0.000, 0.156, 0.273, 0.000, 0.273`。

**决策（2026-06-29）**：`enter_l3_low: 0.30 → 0.20`（救回 0.273×2 条）。

**剩余 gap（已接受）**：
1. **L2 jaccard ≈ 0** 的语义重写型污染（5 条中 2 条 jaccard=0.000）任何 L2 阈值都救不回，必须 L3 全量扫描才能彻底覆盖。当前 cascade 实质上**仍是 L1+L2 主导**，L3 只能捕获"L2 已经有一定相似度 + 语义更高"这类边缘 case。
2. **短 benchmark（simpleqa 题目 ~60 字符）** 即使 L2_near 也漏（jaccard=0.055 < 0.20）。短题目天然 char n-gram 重合低，需要单独通道（token-level MinHash 或 simpleqa 子集全量 L3）。

**对报告的影响**：UFW-L3 v3 污染数据反映的是 **L1 + L2 + 部分 L3** 的下界，而非真实污染上限。汇总报告须显著标注此 gap。如未来要彻底关闭：选项 4（L3 全量扫描，GPU 时间 ×100）或对 simpleqa 等短 benchmark 走独立全量通道。

**复用策略（避免重跑全量 cascade）**：阈值变更不影响 L1/L2 原始分数（`per_doc.jsonl` 保留 `l2_jaccard_max`），只影响路由决策。降阈值后只需对 `l2_jaccard_max ∈ [0.20, 0.30)` 的 doc 增量补跑 L3，重写 verdict + summary —— 见 `scripts/patch_l3_rerouting.py`（待实施）。

---

## 4. 实施步骤（按 P2a / P2b 拆分）

### P2a — 中文 benchmark 覆盖（不依赖 cascade，可立刻并行 P1）

| 步骤 | 输入 | 验证 |
|------|------|------|
| 1. 在 `configs/stage5_v3.yaml` 加 CMMLU/C-Eval/GSM8K-zh/AGIEval 条目 | HF 网络（首次拉取后缓存到 `/mnt/public/data`） | `python stages/contamination/run.py exact --config configs/stage5_v3.yaml --input <mock.jsonl> --dataset mock --input-format jsonl --max-docs 50` 加载成功 |
| 2. UFW-ZH 全量重跑 `exact` 子命令 | 360M ZH docs | summary 含中文 benchmark 命中数；产物落 `outputs/stage5/ufw_zh_l3_*/exact/` |
| 3. UFW-EN 不重跑（v2 全量已跑过，新中文 benchmark 对 EN 命中率必然极低）。 | — | — |

**P2a 验收**：v3 报告 §2.5 中文部分有非零 benchmark 命中（哪怕仍然很低，至少摆脱"无覆盖"的状态）。

### P2b — Cascade 方法（与 P2a 解耦，可在 P2a 跑批的同时开发代码）

| 步骤 | 内容 | 验证 |
|------|------|------|
| 1. 重构 `utils.py`：把 MinHash 切到 `dedup/utils.py:_minhash_signature_fast`；新增 `compute_near_contamination`（提取自 `compute_code_near` 的通用版） | 代码 | smoke：mock 数据上 `near` 子命令在已知改写 doc 上 jaccard ≥ 0.5 |
| 2. 抽离 `bench_index.py`：把 hash + MinHash 索引构建落盘 | 代码 + 离线脚本 | `bench_index.py build` 跑通，目录产出齐全 |
| 3. 新增 `compute_embed_contamination`：BGE-m3 + FAISS IndexFlatIP；复用 `dedup/utils.py:_encode_embeddings` 的 CLS pooling 模式 | 代码 | smoke：3 条已知释义 doc vs MMLU benchmark cos ≥ 0.85 |
| 4. 实现 `compute_cascade_contamination`：路由 + verdict 三色 | 代码 | smoke：cascade 子命令在 mock 上对每篇 doc 给出 verdict + per-layer scores |
| 5. **Sanity check**：选 1K UFW-EN + 1K UFW-ZH 随机抽样跑 cascade。手工核查 10 条红 + 10 条黄 + 10 条绿是否合理 | 抽样 | 红色 ≥ 80% 真阳；黄色 50% 左右真阳；绿色 ≥ 95% 真阴 |
| 6. 阈值调优（如有必要） | 调 `cascade.layer3.cos_red / cos_yellow` + L2 阈值 | sanity 通过 |
| 7. UFW-ZH + UFW-EN 全量重跑 cascade | 全量 | summary 含 cost_breakdown：L1→L2→L3 进入数；三色分布 |

**P2b 验收**：v3 报告 §2.5 给出三色 verdict 分布 + cost_breakdown（每层处理 doc 数 + 实际机时）+ 与 v2 纯 exact 的差值（"新发现的软污染" 条数）。

### 计算资源估算

| 任务 | 资源 | 时长（估算） |
|------|------|--------------|
| Benchmark 索引构建（hash + MinHash + embed） | 1×GPU | ~30min（benchmark 10K 条 × BGE-m3 fp16） |
| ZH 翻译镜像 benchmark 生成（可选） | 1×GPU | ~2h（Qwen2.5-72B 离线翻译 10K 条） |
| UFW-ZH 全量 cascade | 64 核 CPU + 1×GPU（GPU 仅给 L3 候选） | L2 ~15h；L3 候选 ~3-10 GPU·h |
| UFW-EN 全量 cascade | 同上 | L2 ~30h；L3 候选 ~5-20 GPU·h |

> 若 L3 候选量超出预算，可在 cascade_config 加 `layer3.max_per_batch_file` 控制每文件 L3 上限并随机降采，summary 标注。

---

## 5. 输出与聚合层对接

### 5.1 cascade per_doc.jsonl 字段（新）

```json
{
  "doc_id": "ufw_zh_l3_qa_0001234",
  "scores": {
    "l1_exact_hit": false,
    "l2_jaccard_max": 0.62,
    "l2_matched_benchmarks": ["cmmlu"],
    "l3_cos_max": 0.89,
    "l3_matched_bench_id": "cmmlu_0421",
    "l3_matched_language": "zh"
  },
  "flags": {
    "is_exact_contaminated": false,
    "is_near_contaminated": false,
    "is_semantic_contaminated": true,
    "is_cross_lingual": false,
    "verdict": "red"
  }
}
```

### 5.2 summary.json 新增字段

```json
{
  "total_docs": 360123456,
  "cost_breakdown": {
    "l1_processed": 360123456,
    "l2_processed": 360123456,
    "l3_processed": 524389
  },
  "verdict_distribution": {"red": 1234, "yellow": 5678, "green": 360116544},
  "cross_lingual_docs": 89,
  "per_benchmark_red_hits": {"mmlu": 0, "cmmlu": 412, "gsm8k_zh": 23, ...},
  "v2_comparison": {"exact_only_red": 2, "new_red_via_cascade": 1232}
}
```

聚合层（红绿灯）：直接消费 `verdict_distribution` + `cross_lingual_docs` + `v2_comparison`。

---

## 6. 风险与回退

| 风险 | 应对 |
|------|------|
| L2 中文 char 5-gram 噪音高（短停用词组合 jaccard 假高） | sanity check 阶段对中文额外用 `_is_cjk_dominant` 判定 + 提高 n=6 或加 IDF 加权（参考 `dedup/utils.py` 已有 CJK 分支） |
| L3 BGE-m3 显存峰值（max_length=512 + batch=64） | 候选量大时降 batch_size 至 32 / 改用 FAISS HNSW + fp16 |
| 翻译镜像 benchmark 不准 | 可降级：cross-lingual 信号弱，先不出，明确写入 v3 报告 known-gap |
| L2 + L3 全量超预算 | 增加 `--max-docs` 改抽样跑；cost_breakdown 显式标注是否全量 |
| `cascade` 与 `exact` 输出不一致引发回归质疑 | cascade 内部仍调 `compute_exact_contamination`，L1 部分输出与原 `exact` 子命令完全一致；v3 报告显式做"cascade 的 L1 结果 == v2 exact 结果"的回归检查 |

回退路径：若 P2b cascade 在 sanity check 阶段未通过（红色真阳 < 80%），v3 退回仅 P2a（中文 benchmark 覆盖），cascade 推回 v4，与原计划一致。

---

## 7. 与 v3 其他 stage 的依赖

- **不依赖 P1 toxicity-v3**：完全独立的 stage，可与 P1 并行启动开发与跑批。
- **不依赖 P3 dedup**：MinHash 实现虽复用 `dedup/utils.py`，但是 read-only 复用，不触动 dedup stage。
- **GPU 资源与 P1 共享**：P1 toxicity-v3 用 Qwen2.5-7B-Instruct，P2b cascade L3 用 BGE-m3。两者显存占用都 < 20GB，A100-80G 可同时常驻；如同时跑批可按文件分批，互不抢占。

---

## 8. TODO 清单（按交付顺序）

- [ ] P2a-1：写 `configs/stage5_v3.yaml`，加中文 benchmark 条目（CMMLU/C-Eval/GSM8K-zh/AGIEval）
- [ ] P2a-2：mock smoke：`stage5.exact --config stage5_v3.yaml`
- [ ] P2a-3：UFW-ZH 全量 `exact` 重跑（脚本：`scripts/contamination_v3_zh_exact.sh`）
- [ ] P2b-1：重构 `utils.py` MinHash → 复用 dedup xxhash 实现
- [ ] P2b-2：新增 `compute_near_contamination` + `near` 子命令
- [ ] P2b-3：新增 `bench_index.py` 索引构建脚本
- [ ] P2b-4：新增 `compute_embed_contamination` + `embed` 子命令（BGE-m3 + FAISS）
- [ ] P2b-5：新增 `compute_cascade_contamination` + `cascade` 子命令
- [ ] P2b-6：（可选）翻译镜像 benchmark 生成脚本
- [ ] P2b-7：Sanity check 2K 抽样 + 阈值调优
- [ ] P2b-8：UFW-ZH + UFW-EN 全量 cascade 重跑
- [ ] P2b-9：更新 `stages/contamination/README.md` + `pipeline_overview.md` v3 P2 章节
- [ ] P2b-10：v3 报告 §2.5 落地（含 verdict 分布 + v2 对比）
