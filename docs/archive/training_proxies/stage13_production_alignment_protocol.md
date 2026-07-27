# 已归档：Production-aligned training proxy validation plan v2

原协议版本：`v2.0-active`

归档状态：**Closed after Stage 1 screen / no promotion**

日期：2026-07-24

> 归档说明：本文保留实验启动时的冻结计划和未来时措辞。实际执行完成了 Stage 1
> 的 8/8 个 0.6B、100M-token run 及冻结知识评测，但未通过预注册迁移门；Stage 2
> 因 gate 取消，Stage 3-5 未启动。最终状态和统计以
> [Stage 13 生产对齐训练 proxy 报告](stage13_qwen_screen_report.md)为准。

本计划是在 DataDecide/OLMo/Pythia 公开模型实验基础上，验证训练型数据质量 proxy 是否能迁移到生产中对齐的 Qwen、DeepSeek、Kimi、GPT、Claude 等模型家族。它不把公开老模型上的成功直接解释为生产有效。

本轮执行授权已于 2026-07-24 冻结：允许 Qwen3-like 0.6B/1.7B/4B、Qwen3 Base continued pretraining、本地数据落盘和重分词、训练日志与中间 checkpoint；72 小时是本轮训练墙钟上限；当前没有内部 production checkpoint；能力范围暂限通用知识、代码和数学；GPT/Claude/Kimi API 暂不启动（没有 endpoint、凭据和费用上限）。

## 1. 协议冻结时的证据和目标

协议冻结时的实验已经证明：在透明的 DataDecide、OLMo、Pythia 类 base-model 家族中，部分 benchmark 可以对数据配方产生排序信号；MMLU-CF + 20M `correct_prob_per_char` 目前只能标记为 `early_screen_only`。MMLU-Pro、GPQA-Diamond、MMMLU-ZH 的 1B target 仍是 `needs_larger_target`。

这不等于已经证明 proxy 能预测生产模型。生产模型可能有不同 tokenizer、dense/MoE 架构、训练规模、数据混合和 SFT/RL 后训练。因此本计划的目标是依次验证：

1. 小模型排序能否迁移到更接近 Qwen 的架构和 tokenizer；
2. continued pretraining 对候选数据的增益排序，能否复现从零训练 proxy 的排序；
3. 排序能否在内部生产 checkpoint 或 production-like target 上复现；
4. dense proxy 和 MoE proxy 的结论是否一致；
5. 知识、代码、数学和长程任务是否需要各自的 proxy。

本轮已确认的能力范围仅为通用知识、代码和数学；长程任务延后，不纳入 72 小时 pilot 的晋级裁决。

最终状态分为：

- `transparent_family_validated`：只在公开透明模型家族中成立；
- `architecture_general`：至少两个架构/模型家族同向；
- `production_aligned`：在生产模型 checkpoint 或 production-style continued pretraining 上通过留出验证；
- `family_specific`：只对某一模型家族有效；
- `early_screen_only`：只能淘汰明显失败数据；
- `needs_larger_target`：target 没有越过随机地板或缺少稳定 recipe pair；
- `blocked`：人工依赖、数据、模型或 evaluator 尚未就绪。

## 2. 不可混淆的两类实验

### 2.1 Benchmark 分辨率审计

使用 OLMo-2 7B/13B、Pythia 6.9B/12B 等公开模型，只回答 benchmark 是否在更大模型上越过随机地板、是否存在足够的能力信号。这些模型没有使用 DataDecide 的 25 个 recipe，不能替代 DataDecide 大 target 矩阵。

### 2.2 数据质量排序迁移

必须在相同架构、相同训练协议下改变候选数据，比较 recipe 间方向和排序。可用的 target 包括 Qwen-like 从零训练、Qwen base continued pretraining、内部生产 checkpoint。GPT/Claude/Kimi API 只能作为 benchmark 外部锚点，不能直接提供候选数据的 ground truth，因为它们没有分别用候选 recipe 训练。

## 3. 阶段和执行门槛

### 阶段 0：协议冻结和基线归档

目标：把现有结果固定为透明模型基线，避免新实验事后改门槛。

工作：

- 固定 benchmark registry revision、数据 recipe revision、tokenizer、模型 config、训练代码 commit；
- 为每个 run 生成 `run_manifest.json`，记录输入 SHA-256、模型 revision、容器/依赖、GPU、seed、token budget、checkpoint 和日志路径；
- 保留当前 `needs_larger_target`、`early_screen_only` 结论；
- 预注册 8 个 recipe、至少 3 个 family 的 pilot 集合，不根据新结果更换；
- 预注册 primary 指标和 held-out 指标，禁止只报告事后冠军指标。

阶段 0 通过条件：所有后续 run 都能从 manifest 恢复，且不依赖人工记忆的命令行参数。

执行记录（2026-07-24）：契约测试 24 项通过；两卡 all-reduce、tiny Qwen3 FSDP forward/backward、DCP checkpoint save/resume 均通过，恢复后参数 checksum 差异为 0。Qwen3 1.7B/4B/8B Base 的 config、tokenizer、safetensors header 和离线 meta-loadability 检查通过。一个 0.6B Qwen3-like 两卡 trainer smoke 已完成 8 步/2,048 token，loss 从 12.11 降到 10.62，checkpoint、RNG 和训练 manifest 均落盘。

阶段 0 预检时仍有一个未关闭的数据表示 gate：现有 DataDecide 主资产是 OLMo tokenizer 的 headerless uint16 stream，不是 raw text。后续已完成到 Qwen tokenizer uint32 stream 的可审计转换并用于 Stage 1，但 OLMo decode 再重编码仍不等价于 raw text，因此最终结果仍不能标为 `production_aligned`。

此外必须先做三项极小运行时 smoke：两卡 all-reduce、两卡 Qwen3 config FSDP forward/backward、checkpoint save/resume。任何一项失败都先修运行时或缩小并行配置，不启动 8 个 recipe 的正式训练。

### 阶段 1：Qwen3-like 从零训练 proxy

目标：验证更新、更加接近当前生产对齐目标的 tokenizer/架构是否保持 recipe 排序。

首批模型（优先使用本机已缓存且可审计的版本）：

- Qwen3-like 0.6B：使用公开 Qwen3 配置缩小得到，适合低成本 pilot；
- Qwen3 1.7B Base 配置：作为首个正式 dense proxy；
- 结果稳定后再加 Qwen3 4B。

参考：[Qwen3 技术报告](https://arxiv.org/abs/2505.09388)、[Qwen3 1.7B Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)、[Qwen3 4B Base](https://huggingface.co/Qwen/Qwen3-4B-Base)。模型 revision、tokenizer revision 和本地权重 SHA-256 必须在阶段 0 后冻结。

这里使用公开配置和 tokenizer，不宣称复现 Qwen 的完整预训练，也不把 Qwen 的未公开数据配方当 ground truth。Qwen2.5 1.5B/3B 可作为代际敏感性对照，但不是首批主线。首批采用以下已经从现有 DataDecide 报告中预注册的 8 个 recipe，每个 recipe 一个 seed；只有通过 pilot gate 的 recipe 才补齐三 seed：

```text
falcon_raw, falcon_qc20, falcon_orig10,
dclm_raw, dclm_fw2, dclm_fw3,
dolma_raw, dolma_no_flan
```

它们覆盖 Falcon、DCLM、Dolma 三个 family，同时包含稳定正例、临界边和已知 crossover 对照。不能因为 Qwen 结果不理想而在事后替换这 8 个 recipe。

数据表示是一个单独的 gate：当前本地 DataDecide 资产主要是 OLMo tokenizer 的 headerless `uint16` token stream，不是原始文本。要声称“Qwen tokenizer 对齐”，必须获得原始文本，或用冻结的 OLMo tokenizer 解码后再用冻结的 Qwen tokenizer 重编码，并记录 round-trip 损失和新旧 token 数。若只能使用现有 OLMo token stream，则可以做 Qwen-like 架构对照，但结论必须标为 `architecture_only`，不能标为 `production_aligned`。

训练设置同时跑两个臂：

- `fixed_tokens`：所有 recipe 看到相同 token 数，作为主数据质量比较；
- `fixed_tokens_per_parameter`：固定约 20–30 tokens/parameter，作为 scaling 敏感性分析。

正式长 horizon 前设置逐级停止点，避免一次性支付完整 20–30 tokens/parameter：

- smoke：约 0.3B fixed tokens，只验证数据、loss、checkpoint 和 evaluator；
- early：1B fixed tokens，同时观察约 1 token/parameter 的早期方向；
- mid：3B fixed tokens或 5 tokens/parameter；
- confirmatory：只有 early/mid 方向稳定才扩展到 20 tokens/parameter；
- 30 tokens/parameter 仅用于少数已通过 recipe 和额外资源充足时的敏感性分析。

执行时新增一个资源保护层：在本机实测 0.6B/2048-context/4-card 的稳态吞吐后，首轮 pilot 先以 100M token screen 覆盖全部 frozen recipe；它是上述 smoke/early gate 之间的可审计中间点，不改变 fixed-token 与 fixed-token/parameter 的正式比较定义。只有方向和 evaluator 通过 screen gate 的 recipe 才扩到约 0.3B 或 1 token/parameter。

每个 horizon 都保存独立结论；如果 1 和 5 tokens/parameter 已出现系统性反向或完全无分辨率，不继续支付 20–30 tokens/parameter。

建议主指标：注册表中的 MMLU-CF/MMLU-Pro、EvalPlus/LiveCodeBench、MATH/MATH-500/GSM8K/MGSM；动态或新鲜 benchmark 作为 held-out，不参与模型选择。

pilot gate：

- Qwen-like 与现有透明模型排序的 Kendall tau 为正且非偶然；
- 至少 20 个方向明确的 recipe pair；
- pairwise direction accuracy 目标 ≥ 80%；
- 不出现明显系统性 crossover。

未通过时，不升级 4B 或补齐所有 seed，直接记录 `family_specific` 或 `rejected_proxy`。

### 阶段 2：Qwen base continued pretraining

目标：用更新的 Qwen3 production-like base checkpoint 测量候选数据的边际价值。

设置：

- 从同一个 Qwen3 base checkpoint 开始；
- 第一轮 4–8 个 recipe；
- 固定学习率、batch、seed、训练步数和 token 数；
- 先做 Qwen3 1.7B/4B，条件允许再做 8B；
- 第一轮建议每个 recipe 1–5B continued-pretraining tokens；
- 主要比较 `delta_score = after - before`，而不是绝对分数。

可选的现代模型审计：本机若已有并且 forward/loss smoke 通过，可加入
[Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) 的文本-only continued-pretraining。它是 hybrid/multimodal 架构，不能与 Qwen3 dense 结果合并；先冻结 vision tower，并单独记录为 `modern_hybrid_target`。Qwen3.5-35B-A3B 只做推理分辨率审计，不安排全参数训练。

通过条件：continued-pretraining 的 recipe 排序与 Qwen-like 从零训练排序在 held-out benchmark 上同向；否则单独标记为 `family_specific`，不能合并。

### 阶段 3：内部生产 checkpoint 对照

目标：直接验证生产相关模型。

优先请求同事提供：

- 同一模型架构和参数量；
- 至少两种数据配方或数据阶段；
- base checkpoint 优先，SFT/RL checkpoint 作为单独阶段；
- tokenizer、训练 token 数、训练时间和数据 mix 说明；
- benchmark 评测所需的权重或可审计 API。

如果只有 GPT/Claude/Kimi API，只用于 benchmark 相关性和模型排名外部校准，不能用于候选数据质量 ground truth。

通过条件：至少一个 held-out production-like family 上，recipe pairwise direction accuracy ≥ 80%，cluster bootstrap 下界 ≥ 65%，且覆盖率 ≥ 50%。

### 阶段 4：小规模 MoE proxy

目标：检查 dense proxy 排序是否迁移到 DeepSeek/Kimi 类 MoE。

设置：

- 小规模 DeepSeekMoE-like 配置，总参数约 1–3B、active 参数更低；
- 仅在阶段 1 通过的 4–8 个 recipe 上运行；
- 首轮一个 seed，稳定后补三 seed；
- 与 dense proxy 使用同一 benchmark 和统计协议。

若 dense/MoE 排序明显冲突，分别维护 `dense_proxy` 和 `moe_proxy`，不强行平均。

### 阶段 5：留一模型家族验证和最终裁决

禁止在看过所有 target 结果后调阈值。推荐使用 leave-one-family-out：用公开家族确定协议，留出一个 Qwen-like、MoE-like 或内部生产 family 作为最终验证。

默认晋级门槛：

- pairwise direction accuracy ≥ 80%；
- cluster bootstrap 下界 ≥ 65%；
- held-out family ≥ 65%；
- coverage ≥ 50%；
- Kendall tau 为稳定正相关；
- 无已知系统性 crossover；
- 不能只在一个 benchmark 上成立。

每个能力方向单独裁决，不允许用知识 benchmark 的成功替代码、数学或长程任务的结论。

## 4. 并行和资源策略

可并行的任务：

1. 阶段 1 的不同模型规模和不同 recipe；
2. 阶段 1 的知识、代码、数学 benchmark evaluator；
3. 阶段 2 的不同 continued-pretraining recipe；
4. 外部模型 benchmark resolution audit；
5. contamination audit、数据 hash 和训练日志整理。

必须串行的任务：

1. 阶段 0 协议冻结后才能启动正式 pilot；
2. 阶段 1 pilot gate 通过后才能扩展 3B/三 seed；
3. 阶段 2 不能在未审计 Qwen base 权重和 tokenizer 前启动；
4. 阶段 5 只能使用预先冻结的 held-out family。

8 张 GPU 的默认分配：

- 4 张用于主训练；
- 2 张用于并行 benchmark resolution/evaluator；
- 2 张保留给 checkpoint 转换、失败重试和小规模 smoke test。

本机 GPU 拓扑只有两组四卡互联（`[0,1,2,3]` 和 `[4,5,6,7]`）。分布式训练优先限制在单组四卡内；若使用 8 卡，必须先完成 MetaX/MXCCl 通信 smoke 并记录跨组通信代价。默认建议 `train=[0,1,2,3]`、`eval=[4,5]`、`reserve=[6,7]`，而不是把两组卡混入同一个 DDP 训练作业。

如果增加机器，优先扩展“不同 recipe 的独立 run”，不要把同一 run 随意拆成不可恢复的跨机器任务。

## 5. 日志、断点和错误处理

每个 run 必须写入：

- `run_manifest.json`；
- `config.resolved.yaml`；
- `stdout.log`、`stderr.log`；
- `train_metrics.jsonl`；
- checkpoint 索引和 SHA-256；
- evaluator summary 和原始预测；
- failure/resume 事件记录；
- 阶段结论 `conclusion.json`。

训练应使用可恢复 checkpoint；评测矩阵使用 recipe/model/benchmark 级 manifest，单个失败只重跑失败单元。若错误可能污染权重、tokenizer 或数据顺序，不能从中间 checkpoint 继续，必须删除该 run 并重新开始；若只是 evaluator、网络下载或节点故障，可从最近可信 checkpoint 恢复。

每个阶段结束时生成阶段报告，至少包含：完成率、GPU 小时、墙钟时间、失败原因、重试成本、统计结果、是否通过 gate 和下一阶段建议。

## 6. 需要人工配合的事项

以下事项如果没有确认，相关阶段应标为 `blocked`，不能默认为已授权：

1. **内部生产 checkpoint**：是否可以提供 Qwen/DeepSeek/Kimi-like 的 base 或阶段 checkpoint；权重存储位置、访问权限、许可证和可否用于本项目评测。
2. **模型版本选择**：首批是否允许使用 Qwen3-like 0.6B、`Qwen/Qwen3-1.7B-Base` 和 `Qwen/Qwen3-4B-Base`；如团队已有指定 Qwen3/Qwen2.5 版本，提供准确 model ID/revision。Qwen2.5 只作为代际对照；Qwen3.5-4B 仅作为可选 hybrid target audit。
3. **生产 API**：是否允许使用 GPT/Claude/Kimi API；提供 endpoint、固定 model revision、API key 注入方式、费用上限和数据出境限制。API key 不应写入仓库或日志。
4. **外部机器**：是否可以使用其他 GPU 机器；提供 SSH/Slurm/Kubernetes 入口、镜像要求、共享存储和网络策略。
5. **数据权限**：阶段 1/2 使用哪些候选数据目录；是否允许复制、tokenize、continued pretraining；确认数据不会被上传到第三方 API。
6. **预算上限**：为 3B/7B continued pretraining、API 评测和模型下载分别给出 GPU 小时、存储和 API 费用上限。
7. **原始数据或重编码授权**：确认是否能获取 8 个 recipe 的 raw text；若不能，是否允许 OLMo-token decode → Qwen-token re-encode，并接受该转换作为独立敏感性实验。
8. **生产 benchmark**：确认哪些注册表 benchmark 是真实验收目标，哪些只能作为 proxy 或污染高风险对照。
9. **数学和代码评测许可**：确认 MATH、GSM8K、EvalPlus、LiveCodeBench 等数据的本地缓存和生成式评测方式。

## 7. 实验启动时建议的人工回复格式

在当时的新对话开始时，建议直接提供：

```text
Qwen3-like 首批模型：0.6B / 1.7B / 4B（允许或不允许）
Qwen3.5-4B modern hybrid audit：允许或不允许
Qwen base continued pretraining：允许或不允许，最大 GPU 小时：...
内部 checkpoint：有/无；模型、规模、数据配方、路径：...
生产 API：允许哪些；费用上限：...
外部机器：有/无；调度方式：...
可使用的数据目录：...
可接受的存储上限：...
```

以上为实验启动时的授权模板，不是当前待办事项。实际执行和停止点见本文顶部的归档说明和最终报告。
