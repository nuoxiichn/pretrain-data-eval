# Stage 13 生产对齐训练 proxy 报告

状态：阶段 0 runtime gate 已通过；Stage 1 的 8/8 个 0.6B/100M-token run 及全量冻结知识评测已完成；screen 证据不足，不允许晋级。

本报告按[冻结实验协议](stage13_production_alignment_protocol.md)记录各阶段 gate。当前不对 Qwen、DeepSeek、Kimi、GPT 或 Claude 的生产迁移有效性作结论。

## 当前基线

- DataDecide/OLMo/Pythia：公开透明模型家族内的初步验证；
- MMLU-CF + 20M `correct_prob_per_char`：`early_screen_only`；
- MMLU-Pro、GPQA-Diamond、MMMLU-ZH：当前 1B target 为 `needs_larger_target`；
- 生产模型迁移：`not_validated`。

## 阶段 0 预检记录

- 硬件：8× MetaX C500 64 GiB；GPU 0–3 和 4–7 各自为一组四卡互联，组间无直连；
- 当前 GPU：空闲；`torch.cuda` 可见 8 卡；
- 主机：约 160 CPU 核、约 1.8 TiB 内存；
- `/mnt/public`：约 20 TiB 可用，使用率约 91%，需要限制 checkpoint 保留策略；
- 网络：GitHub/arXiv 可达，Hugging Face 直连不稳定；下载必须单 worker、固定镜像并做 SHA-256 校验；
- 调度：无 Slurm/Kubernetes；长任务使用 `torchrun` + `tmux` + 独立 manifest；
- 现有 GPU 训练进程：无；
- 已完成：计划 YAML 解析、Stage 13 入口配置；相关 contract/unit 测试 24 项通过；
- 已完成：两卡 all-reduce、tiny Qwen3 FSDP forward/backward、DCP checkpoint save/resume；恢复后参数 checksum 最大差异为 0；
- 已完成：Qwen3-1.7B/4B/8B Base 离线 config/tokenizer/safetensors header/meta-loadability 审计；
- 已完成：0.6B Qwen3-like 两卡 trainer smoke，8 step/2,048 token，loss 12.1091 → 10.6203；
- 已实现：独立 run manifest、JSONL 训练日志、FSDP sharded checkpoint、RNG/data-order 恢复、checkpoint SHA-256 index、HF 导出入口；
- 已实现：OLMo/DataDecide uint16 stream → Qwen tokenizer uint32 stream 的可审计转换，记录输入/输出 hash、token ratio 和边界策略；
- 已完成：Falcon raw/QC20 两条 0.6B、100M-token Qwen-like 正式 screen run；continued pretraining、MoE 和生产 API 尚未启动。
- 用户确认：阶段 1/2 本地训练、数据落盘、tokenize、continued pretraining、日志和中间 checkpoint 均允许；本轮硬墙钟预算为 72 小时；能力范围暂限通用知识、代码和数学。
- 长程任务本轮明确延后，不进入 72 小时 pilot 的晋级裁决。
- 用户暂未提供生产 API endpoint/预算，因此 GPT/Claude/Kimi 外部锚点评测暂缓，不影响阶段 1/2。

## 阶段记录

| 阶段 | 状态 | 结果 | 结论 |
|---|---|---|---|
| 0 协议冻结和基线归档 | passed_with_open_data_gate | runtime、checkpoint 和模型 loadability 通过；DataDecide raw text 不存在 | 允许进入重分词/数据准备，尚不可作 production claim |
| 1 Qwen3-like 从零训练 | completed_screen_only | 8/8 个 0.6B、100M-token run 及冻结知识评测完成 | 证据不足，不允许晋级 |
| 2 Qwen3 base continued pretraining | canceled_by_gate | Stage 1 未通过晋级门槛 | 未启动 |
| 3 内部生产 checkpoint | not_started | Stage 2 未启动，且无内部 checkpoint | 未启动 |
| 4 小规模 MoE proxy | not_started | Stage 1 未通过晋级门槛 | 未启动 |
| 5 留一模型家族裁决 | not_started | 前置阶段未启动 | 未启动 |

逐阶段的 manifest、日志、原始预测和统计汇总保存在 `outputs/production_alignment/`，不提交大模型权重和原始语料。

## 当前有价值的工程观察

1. 当前 0.6B 配置实际参数量约 508M；其 Qwen 词表占比较大。它适合低成本方向筛选，但不能被解释为对官方 Qwen3-0.6B 配方的复现。
2. 两卡 0.6B DCP checkpoint 约 3.3GB。逐步保存会造成严重 I/O 和磁盘放大；正式 run 只按固定 horizon/milestone 保存，并保留最近可信 checkpoint。
3. OLMo stream 没有可恢复的原始文档边界。当前转换按 8,192 个源 token 解码，每块追加一个 Qwen EOS；这是可重复的近似转换，不是 raw-text 等价物。报告必须同时给出边界策略敏感性，不能隐藏这一误差。
4. Falcon 8M-token 转换的 Qwen/OLMo token ratio 为 0.98334，说明转换没有出现明显 token 数爆炸；完整 recipe 的 ratio 仍需分别记录。
5. 本机 FSDP 可以运行，但 PyTorch 对当前 ShardedTensor checkpoint API 给出弃用提示；不影响本轮正确性，后续应迁移到 DTensor checkpoint API。
6. Qwen3-8B 本地目录为 root-only；如果后续换非 root 运行身份，需要先做权限 gate。Qwen3-1.7B cache 中另有一个未参与加载的 768MB `.incomplete` 文件，应标记为废弃下载残留。
7. 0.6B、2048 context、4 卡、每卡 micro-batch 2 的稳态吞吐约为每步 0.57 秒，即约 28k token/s；100M token 单 recipe 预计约一小时。为了保证 8 recipe、评测和重试都落在 72 小时内，首轮从原计划的约 0.3B smoke 拆成 100M screen，只有方向稳定者再扩到 0.3B/1 token-per-parameter。
8. Pilot 的最初 6M token 中，Falcon raw/QC20 loss 都从约 12 降至约 6.2；这只证明训练链路健康，不是 recipe 质量结论。必须等待同 horizon benchmark/held-out 评测。
9. Falcon raw/QC20 均在 6,104 step 完成 100M token，耗时分别为 3,546/3,551 秒；最终单 batch loss 为 4.5355/4.6869。两者的同一冻结 benchmark smoke 和全量评测均已完成；不能根据训练 loss 排 recipe，且 accuracy 与连续 proxy 指标存在方向冲突，必须按预注册指标分别报告。
10. 编排中断只影响第二波是否被启动，不影响第一波权重：两个 run 的状态均为 `passed`，各有 step 2,000/4,000/6,000/6,104 checkpoint 和约 1.3GB HF 导出。

## 冻结 benchmark smoke（2026-07-24）

对两个已完成 HF 导出使用同一 `rc-zero-shot-v1` evaluator、sample seed `6198`，每个 benchmark 固定抽取 100 题；模型分别绑定 GPU 0/4 并行评测。四项均成功落盘，原始预测和逐项 summary 位于：

- `outputs/production_alignment/eval/falcon_raw-0.6B/smoke_100_20260724/`
- `outputs/production_alignment/eval/falcon_qc20-0.6B/smoke_100_20260724/`

下表为 smoke 的 primary accuracy（不是 full benchmark，也不用于 screen gate）：

| benchmark | falcon_raw | falcon_qc20 |
|---|---:|---:|
| mmlu-pro | 0.050794 | 0.072619 |
| gpqa-diamond | 0.220000 | 0.170000 |
| mmmlu (ZH_CN) | 0.248635 | 0.281287 |
| mmlu-cf | 0.203725 | 0.189439 |

这一步确认模型导出、MetaX/CUDA 推理、冻结题面/选项协议、固定 revision 数据缓存和 summary 写入链路均可运行；100 题分数仅作 evaluator smoke，不作 raw 与 qc20 的能力排序结论。smoke 通过后已按相同协议补跑全量评测。

### Falcon raw/QC20 全量结果

全量输出位于 `outputs/production_alignment/eval/{falcon_raw,falcon_qc20}-0.6B/full_20260724/`；四项 evaluator summary 均为 `passed`。下表为 primary accuracy：

| benchmark (eligible n) | falcon_raw | falcon_qc20 |
|---|---:|---:|
| mmlu-pro (12,032) | 0.072533 | 0.071868 |
| gpqa-diamond (198) | 0.207071 | 0.196970 |
| mmmlu ZH_CN (14,040) | 0.263910 | 0.259237 |
| mmlu-cf (10,000) | 0.244809 | 0.243631 |

accuracy 在四项上都给出 `falcon_raw > falcon_qc20` 的弱方向；但 `correct_prob_per_char` 在 MMLU-Pro、GPQA-Diamond 和 MMLU 上给出相反方向，仅 MMLU-CF 同向。该冲突必须保留，不能在看到结果后挑选指标；8 recipe 到齐后按冻结的 accuracy 对照和三个连续 proxy 指标分别计算 pairwise direction、coverage、Kendall tau 和 family-level bootstrap。

## Stage 1 wave2：Falcon orig10 + DCLM raw（2026-07-24）

两个 recipe 分别使用 GPU 岛 `[0,1,2,3]` 和 `[4,5,6,7]`，均在 step 6,104 完成 100M token；step 2,000/4,000/6,000/6,104 checkpoint、SHA-256 index、HF 导出、summary 和 conclusion 完整，launcher 正常退出。

| recipe | elapsed seconds | final batch loss | run status |
|---|---:|---:|---|
| falcon_orig10 | 3,539.93 | 4.396754 | passed |
| dclm_raw | 3,538.93 | 4.495671 | passed |

完成后立即使用同一 `rc-zero-shot-v1` evaluator 和固定 revision 本地缓存做四项全量评测；输出位于 `outputs/production_alignment/eval/{falcon_orig10,dclm_raw}-0.6B/full_20260724/`。下表为 primary accuracy：

| benchmark (eligible n) | falcon_orig10 | dclm_raw |
|---|---:|---:|
| mmlu-pro (12,032) | 0.069772 | 0.072575 |
| gpqa-diamond (198) | 0.232323 | 0.196970 |
| mmmlu ZH_CN (14,040) | 0.255208 | 0.255373 |
| mmlu-cf (10,000) | 0.241664 | 0.245013 |

四项 evaluator summary 均为 `passed`。这些结果是单训练 seed、单模型规模的中间 screen 证据；8 个 recipe 到齐并完成 pairwise direction、coverage、Kendall tau 和 family-level bootstrap 前，不作晋级或总体质量结论。

## Stage 1 wave3：DCLM FW2 + FW3（2026-07-24）

两个 recipe 分别使用 GPU 岛 `[0,1,2,3]` 和 `[4,5,6,7]`，均在 step 6,104 完成 100M token；checkpoint、SHA-256 index、HF 导出、summary 和 conclusion 完整，launcher 正常退出。

| recipe | elapsed seconds | final batch loss | run status |
|---|---:|---:|---|
| dclm_fw2 | 3,540.75 | 4.566796 | passed |
| dclm_fw3 | 3,540.31 | 4.044649 | passed |

本 wave 有一个必须保留的覆盖局限：Qwen 重分词后的 `dclm_fw2` stream 有 203,929,337 token，而 `dclm_fw3` 只有 60,104,405 token。`TokenBlockStream` 对固定 seed permutation 使用 `logical_index % block_count` 确定性循环，因此 FW3 的 100M horizon 约为 1.664 个 stream pass；训练在约 step 3,669 后复用同一批独立 block。该行为可恢复且未替换冻结 recipe，但 FW2/FW3 pair 的独立数据覆盖不对称，证据等级必须降级，不能把 FW3 的更低训练 loss解释为数据更优。

全量评测输出位于 `outputs/production_alignment/eval/{dclm_fw2,dclm_fw3}-0.6B/full_20260724/`；四项 evaluator summary 均为 `passed`。下表为 primary accuracy：

| benchmark (eligible n) | dclm_fw2 | dclm_fw3 |
|---|---:|---:|
| mmlu-pro (12,032) | 0.069862 | 0.073929 |
| gpqa-diamond (198) | 0.227273 | 0.217172 |
| mmmlu ZH_CN (14,040) | 0.260002 | 0.257158 |
| mmlu-cf (10,000) | 0.246436 | 0.246740 |

方向按 benchmark 混合：accuracy 在 MMLU-Pro/MMLU-CF 略偏 FW3，在 GPQA/MMMLU 略偏 FW2；连续指标也存在不完全一致。结合 FW3 的 stream 复用，此 pair 当前只能作为带覆盖警告的 screen 证据。

<!-- stage13-screen-final:start -->
## Stage 1 八 recipe 100M-token screen 自动汇总

自动收尾时间：`2026-07-24T19:08:32+08:00`。8/8 训练和四项全量冻结知识评测均通过；原始机器可读汇总位于 `outputs/production_alignment/screen_100m_20260724/summary.json`。

| recipe | MMLU-Pro | GPQA-Diamond | MMMLU ZH_CN | MMLU-CF |
|---|---:|---:|---:|---:|
| falcon_raw | 0.072533 | 0.207071 | 0.263910 | 0.244809 |
| falcon_qc20 | 0.071868 | 0.196970 | 0.259237 | 0.243631 |
| falcon_orig10 | 0.069772 | 0.232323 | 0.255208 | 0.241664 |
| dclm_raw | 0.072575 | 0.196970 | 0.255373 | 0.245013 |
| dclm_fw2 | 0.069862 | 0.227273 | 0.260002 | 0.246436 |
| dclm_fw3 | 0.073929 | 0.217172 | 0.257158 | 0.246740 |
| dolma_raw | 0.077354 | 0.196970 | 0.261799 | 0.247950 |
| dolma_no_flan | 0.071439 | 0.207071 | 0.256666 | 0.243771 |

| metric | raw edge direction accuracy | decisive coverage | decisive accuracy | mean Kendall tau-b | family bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| `accuracy` | 0.250 | 0.100 | 0.000 | -0.188 | [0.125, 0.375] |
| `correct_prob_per_char` | 0.600 | 0.750 | 0.600 | 0.304 | [0.000, 0.875] |
| `normalized_choice_probability` | 0.450 | 0.250 | 0.800 | -0.196 | [0.375, 0.500] |
| `correct_vs_best_incorrect_logprob_margin` | 0.550 | 0.650 | 0.538 | 0.018 | [0.500, 0.750] |

裁决为 `insufficient_evidence_screen_only`，不允许晋级：当前只有 5 个唯一预注册 recipe edge，低于冻结门槛 20；每个 recipe 也只有一个训练 seed。四项 benchmark 上重复观察同一 edge 不能被计作 20 个独立 pair，不能虚报通过 gate。accuracy 和三个连续 proxy 分开报告，不作事后指标选择。

`dclm_fw3` 只有 60,104,405 个独立 Qwen token，100M horizon 使用约 1.664 个确定性 stream pass；相关证据继续降级。代码和数学没有专项 recipe，本轮只保留 evaluator 可运行性结论，不作能力质量结论。OLMo decode → Qwen re-tokenize 仍不是 raw-text 等价物，并保留 8,192-token 人工边界及追加 EOS 偏差。
<!-- stage13-screen-final:end -->
