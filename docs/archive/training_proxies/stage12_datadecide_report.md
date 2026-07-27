# DataDecide 小模型排序复现实验报告

状态：实验完成（2026-07-23）。

## 1. 结论边界

本实验验证一个窄命题：在模型架构、训练 token/parameter 比例、优化器和评测协议固定时，候选
预训练数据 recipe 在小模型上的目标能力排序，能否预测同 recipe 在较大模型上的排序。它不把
benchmark 分数解释为数据的“总体质量”，也不替代污染、隐私、许可、事实性和语言覆盖审计。

方法来自 [DataDecide](https://arxiv.org/abs/2504.11393)，评测协议来自
[OLMES](https://arxiv.org/abs/2406.08446)。官方还公开了
[模型集合](https://huggingface.co/collections/allenai/datadecide-67edb1d2bacba40b5d3ed633)、
[数据 recipe](https://huggingface.co/datasets/allenai/DataDecide-data-recipes)和
[逐任务结果矩阵](https://huggingface.co/datasets/allenai/DataDecide-eval-results)。本报告把证据分为五层：

1. 重算官方结果矩阵，只验证统计口径，不算本地复现；
2. 用固定 OLMES commit 本地复评官方 20M 权重，验证 evaluator 和指标重建；
3. 本地物化 recipe、重新训练 20M 模型并评测，验证代理端不依赖官方训练结果；
4. 本地复评官方 1B 权重，验证目标端排序不只来自官方矩阵；
5. 对全 25-recipe 扩展 MMLU-Pro、GPQA-Diamond、MMMLU-ZH 和 MMLU-CF，验证新增 benchmark
   自己是否可由小模型代理。

## 2. 论文方法与本次口径

DataDecide 的 single-scale 方法很直接：对所有候选数据训练同规模小模型，其他变量不变；小模型
上表现更好的 recipe 被预测为大模型上的赢家。论文用 `6ND` 近似训练 FLOPs，并以所有 recipe
pair 的方向准确率衡量 decision accuracy，而不是要求精确预测 benchmark 分数。目标是 1B 模型
三个 seed 的平均 accuracy。

低参数模型的离散 accuracy 容易落在随机水平。论文因此把字符归一化的连续答案似然作为小模型
代理：`CORRECT PROB per char` 是每题正确 continuation 的字符归一化概率，再在题目间平均。
需要特别强调：它预测的目标仍是 **1B 上同一任务的 accuracy**，不是 1B 的 continuous
probability。论文发现 ARC、MMLU 较早有信号，HellaSwag 需要更多 compute，BoolQ 在所测规模内
接近无效；benchmark 和指标必须按目标能力选择。

论文公开表 2 和固定的
[OLMo DataDecide 训练代码](https://github.com/allenai/OLMo/blob/55d4871d7777f5cc4561f8e508f635f9c6308bbc/scripts/ladder.py)
给出的 20M 配置为 19,101,888 个非 input-embedding 参数、global batch 64、hidden
size 192、8 heads、16 layers、LR `8.4e-3`、14,584 step、约 1.9B token。这里按相同口径实现。
论文固定约 100 token/非 embedding 参数；这与“所有模型固定相同 token 数”的实验不是同一控制。

## 3. 预注册实验组

本次先选 8 个 recipe、3 个数据家族和 5 条边。选择同时包含稳定正例、临界边和明确反例，避免只
挑容易成功的 pair。

| 边 | 选择理由 | 官方 1B OLMES-10 delta | 官方 20M / 60M / 150M delta |
|---|---|---:|---:|
| Falcon QC20 > raw | 同源质量过滤，稳定正例 | +2.443pp | +1.319 / +0.516 / +1.025pp |
| Falcon Orig10 > QC20 | 20M 临界 crossover，用于测最小可靠规模 | +1.101pp | -0.145 / +2.317 / +0.926pp |
| DCLM FW2 > raw | 另一数据家族的稳定正例 | +1.067pp | +0.968 / +0.409 / +2.566pp |
| DCLM raw > FW3 | 强制失败对照 | +0.756pp | -1.483 / -1.166 / -2.783pp |
| Dolma no-Flan > raw | source ablation，检查任务依赖 | +1.467pp | +0.228 / +0.462 / +1.750pp |

主 decision tasks 为 ARC-Easy、ARC-Challenge、MMLU；HellaSwag 单列为较晚出现信号的补充任务；
BoolQ 不进入决策分数。这个选择依据论文和官方矩阵，不是 held-out benchmark 选择，结果必须结合
下面的全 25-recipe 回算解读。

## 4. 官方矩阵回算

机器可读结果在
`outputs/datadecide_reproduction/official_matrix_paper_protocol/summary.json`。对 25 个 recipe 的
300 个 pair，目标是三个 1B seed 的平均 accuracy；下表的 proxy 使用公开 default seed，正是论文
图中单次 run 的口径。

| 小模型指标 -> 1B 同任务 accuracy | 20M | 60M | 150M |
|---|---:|---:|---:|
| OLMES-10 accuracy -> OLMES-10 accuracy | 71.3% | 71.3% | 78.0% |
| ARC-Easy correct prob/char -> ARC-Easy accuracy | 90.0% | 94.3% | 92.3% |
| ARC-Challenge correct prob/char -> ARC-Challenge accuracy | 86.7% | 91.0% | 92.7% |
| MMLU correct prob/char -> MMLU accuracy | 86.3% | 91.7% | 89.3% |
| HellaSwag correct prob/char -> HellaSwag accuracy | 73.0% | 81.7% | 85.7% |
| BoolQ correct prob/char -> BoolQ accuracy | 50.3% | 46.7% | 46.3% |

论文每个点实际报告三个 proxy seed 的均值和标准差。本地回算的 OLMES-10 三 seed 平均 decision
accuracy 为 20M `64.6%`、60M `71.4%`、150M `80.3%`，复现了“150M 约 80%”的主结论。
若先把三个 proxy seed 分数求均值再排序，则三档为 `74.0%/76.7%/83.0%`；这是更稳定但更贵的
另一口径，不能与单 seed 数字混用。

这些结果说明：

- 20M 并非普遍可靠，OLMES-10 的单 seed 历史命中率只有 71.3%；
- ARC/MMLU 的连续指标在 20M 已有较高方向性，HellaSwag 应升级规模，BoolQ 应剔除；
- 多 seed 能降低单次训练噪声，但不能修复真实 scaling crossover；
- 150M 是论文中更稳的通用上限，不代表所有生产任务都必须训练到 150M。

## 5. 官方 20M 权重的本地复评

评测使用 OLMES commit
[`5a51f502`](https://github.com/allenai/olmes/tree/5a51f502d463b8cdc4a2dcad7d7096c41ff1197e)，
十个 OLMES task 展开为 66 个实际任务。当前公开 alias 对 ARC-Easy、BoolQ、HellaSwag、PIQA、
SocialIQA 等任务带 1,000 题限制，而论文明确使用完整 split；因此本地数值与论文旧 evaluator
存在版本差异，报告比较方向并保留原始 predictions，不假装逐数值相同。
隔离环境的实测核心版本为 Transformers `4.57.6`、Datasets `3.6.0`、Hugging Face Hub
`0.36.2`，底层复用机器的 MetaX PyTorch。

完整聚合在 `outputs/datadecide_reproduction/published_20m_aggregate_per_task/summary.json`。
本地重建的 `CORRECT PROB per char` 与官方同一 default checkpoint 的最大绝对差只有 `0.00168`，
8 个 recipe 的同规模排序 28/28 一致，验证了连续指标实现。OLMES-10 同规模排序为 27/28，最大
绝对分数差 `0.00325`，差异主要来自 evaluator 数据限制和版本。

| 本地 20M proxy -> 官方 1B target | 正确 pair | Decision accuracy |
|---|---:|---:|
| 三任务 correct prob/char 均值 -> 三任务 accuracy 均值 | 28/28 | 100.0% |
| ARC-Easy correct prob/char -> ARC-Easy accuracy | 27/28 | 96.4% |
| ARC-Challenge correct prob/char -> ARC-Challenge accuracy | 28/28 | 100.0% |
| MMLU correct prob/char -> MMLU accuracy | 27/28 | 96.4% |
| HellaSwag correct prob/char -> HellaSwag accuracy | 24/28 | 85.7% |
| OLMES-10 accuracy -> OLMES-10 accuracy | 26/28 | 92.9% |

这是经过官方矩阵辅助选组后的高命中子集，不能用 28/28 代替全 25-recipe 的 20M 历史命中率。
三任务平均还会让任务误差互相抵消：ARC-Easy 单独把 Dolma no-Flan/raw 判反，MMLU 单独把
DCLM FW2/FW3 判反，而组合均值仍全部同向。因此生产输出必须保留分任务结果。

五条 OLMES-10 预注册边命中 4/5。唯一失败是强制反例 `DCLM raw > FW3`：20M 本地结果反而是
FW3 高 `0.476pp`，1B target 则 raw 高 `0.756pp`。这不是 evaluator bug；官方 20M/60M/150M
也都预测反方向，是 DataDecide single-scale 方法无法识别的真实 crossover。

连续代理在自己的三任务 accuracy target 上 28/28，但不能拿它“修复”上述 OLMES crossover：
1B 的 ARC/MMLU 三任务均值本身就是 FW3 高于 raw，而 OLMES-10 综合目标是 raw 高于 FW3。两者
回答不同能力目标，恰好说明 benchmark 选择会改变“哪批数据更好”的结论。

这个 crossover 有清晰的能力结构，不只是总分噪声。以官方 1B 三 seed 均值计，`raw - FW3` 在
ARC-Challenge、ARC-Easy、MMLU 上分别为 `-7.57/-8.21/-4.34pp`，但在 BoolQ、CSQA、
HellaSwag、PIQA、SocialIQA、WinoGrande 上分别为 `+3.36/+5.21/+8.97/+4.01/+3.33/+5.00pp`；
OpenBookQA 为 `-2.20pp`。团队若生产“通用知识”“代码”“长程”等不同类型数据，必须先定义目标
能力和对应 benchmark 族，不能拿一个宏平均给所有数据排统一质量。

## 6. 本地数据与 20M 训练实现

Stage 12 实验时新增了 `configs/stage12_datadecide.yaml` 和 `stages/datadecide/`。原 CLI
现已退役，冻结配置移至
[`protocols/stage12/stage12_datadecide.yaml`](protocols/stage12/stage12_datadecide.yaml)；通用的成对统计、
选择题评测、token stream 和运行记录原语保留在
[`research/data_advisor/`](../../../research/data_advisor/)。以下为已退役实现的历史记录：

- `prepare.py` 从固定 OLMo `named_data_mixes.py` 解析官方 recipe，跨完整 shard 列表做确定性
  range 抽样；支持 TLS 重试、完整 part 复用、Content-Range 校验和原子落盘；
- `train.py` 实现官方 20M OLMo 架构、AdamW、cosine schedule、BF16、global batch 64 和可恢复
  trainer state，导出 OLMES 可直接读取的 Hugging Face checkpoint；
- `matrix.py` 重建 continuous likelihood、先对 MMLU 学科做宏平均，并严格区分小模型 proxy
  metric 与 1B target accuracy；
- `run.py` 提供官方矩阵回算、recipe 物化、训练和 OLMES 聚合 CLI。

三份本地流各含 1,911,554,048 个 `uint16` token、约 3.56GiB。抽样检查没有词表越界。
这些文件是无 NPY header 的 little-endian raw stream；`.npy` 后缀来自官方路径习惯，读取接口是
NumPy memmap/raw stream，不是 `np.load`。

| Recipe | 官方 path 数 | 本地 SHA-256 |
|---|---:|---|
| Falcon raw | 411 | `91246916a6f54ea33f75b37379f33a182415e7670d7f678e88a1ec60aba4addd` |
| Falcon QC20 | 334 | `23949b9cb60fe9dffd7afccffcc2e8e2f8b22840dd84a00cfd72491d33565e0f` |
| Falcon Orig10 | 240 | `b1ab8c60203adceebbf025e8d7b953a96c47edde5468e3f41b32e572eb4e4c39` |

每份数据从 24 个均匀分布的官方 path 内取确定性 byte range，以保持来源覆盖并控制下载量。它不是
作者训练时的精确 global shuffle，也没有 doc boundary 或 doc ID；所以本地训练属于“官方逻辑
recipe 的代表性复现”，不是逐 bit 复现。这个差异可能改变小 margin pair，必须作为主要漏洞保留。
tokenizer SHA-256 为
`30d71356c5ba154006df5bbb4a0583fc434525ceeb2f27a7d8a237ce5db26dc6`；固定的
`named_data_mixes.py` 副本 SHA-256 为
`3c25125174f26298e9cdce8b904cf63b00de53465521be8c731ba9b62d948d79`。
数据仓库 revision 固定为
[`3baf34b`](https://huggingface.co/datasets/allenai/DataDecide-data-recipes/tree/3baf34baf5b636f0943401b5c6a2ccb7e5cf3bb9)。

20-step 吞吐探测显示 micro-batch 64 在 64GB 卡上 OOM，micro-batch 32 稳定，global batch 64
通过两次梯度累积保持不变；20 step 用时 17.4 秒。导出的 smoke checkpoint 已被同一 OLMES
环境成功加载并完成评测，验证了训练到正式 benchmark 的链路。

为验证更短周期，三个 default-seed run 都在 step 2,500 冻结了独立 HF checkpoint。该点每个
recipe 只训练 327,680,000 token，约 17.2 token/论文口径参数，实测训练约 34 分钟。中间点和
最终点来自同一训练轨迹，只能比较“多训练是否改变决策”，不能把它们当成独立 seed。

step-2,500 的完整聚合在
`outputs/datadecide_reproduction/local_step2500_crossscale/summary.json`。下表左侧为本地重新训练的
20M seed 6198，右侧 target 为相同 evaluator 下的公开 1B default checkpoint；命中数覆盖 Falcon
三个 pair。

| 指标 | step-2,500 raw / QC20 / Orig10 | 本地 1B target raw / QC20 / Orig10 | Pair 命中 |
|---|---|---|---:|
| 三任务 correct prob -> 三任务 accuracy | .3083 / .3112 / .3346 | .4635 / .4882 / .5171 | 3/3 |
| ARC-Easy correct prob -> accuracy | .3060 / .3130 / .3479 | .683 / .704 / .746 | 3/3 |
| ARC-Challenge correct prob -> accuracy | .3216 / .3220 / .3458 | .3805 / .4147 / .4454 | 3/3 |
| MMLU correct prob -> accuracy | .2971 / .2988 / .3101 | .3271 / .3459 / .3598 | 3/3 |
| HellaSwag correct prob -> accuracy | .4143 / .4192 / .4162 | .616 / .658 / .635 | 3/3 |
| OLMES-10 accuracy | .3261 / .3513 / .3322 | .5587 / .5768 / .5976 | 2/3 |

随后补齐了本地 3×3 早停矩阵，聚合在
`outputs/datadecide_reproduction/local_step2500_three_seed_crossscale/summary.json`。三任务 continuous
的三个 seed 都给出 `Orig10 > QC20 > raw`，逐 seed、seed 均值和“全 seed 同向”规则均命中
3/3；均值分数为 raw/QC20/Orig10 `.3001/.3179/.3375`。三个 seed 的两条相邻边 margin 为：

| 边 | seed 6198 / 14 / 15 margin | 全 seed 同向 |
|---|---|---|
| QC20 - raw | +0.297 / +1.523 / +3.529pp | 是 |
| Orig10 - QC20 | +2.334 / +3.092 / +0.440pp | 是 |

离散 OLMES-10 则在三个 seed 上只命中 2/3、1/3、2/3；seed 均值排序仅命中 1/3，全 seed 同向
规则覆盖 2/3 且其中只有 1/2 正确。因此在这组三个本地 Falcon recipe 上，327M-token continuous
早停是**有效的快速筛选器**，离散 accuracy 明确无效。这比只看 seed 6198 的 3/3 结论更可靠。

外推限制仍然存在：官方相同 step、作者 exact data order 的三个 seed 中，`QC20 - raw` margin 为
`+3.805/-0.146/+1.204pp`，出现一次反向；`Orig10 - QC20` 为
`+0.001/+2.167/+3.466pp`，其中一次几乎为 tie。本地全 seed 同向不能覆盖官方训练次序和完整
recipe 的不确定性，所以早停可在这批数据上晋级，但上线门槛仍必须由团队自己的历史 recipe 矩阵
校准，不能直接宣称普适。

完整 20M 的 3×3 训练与评测聚合在
`outputs/datadecide_reproduction/local_final_three_seed_crossscale/summary.json`。九个 run 都严格训练
14,584 step、1,911,554,048 token，单 run 实测 3.29-3.33 小时。结果与早停点一致，但 margin
更大、更稳定：

| 小模型 proxy -> 本地 1B 同任务 accuracy | 20M seed 均值 raw / QC20 / Orig10 | seed 6198 / 14 / 15 命中 | 均值命中 | 全 seed 同向覆盖/正确率 |
|---|---|---|---:|---:|
| 三任务 correct prob | .3527 / .3728 / .3891 | 3/3 / 3/3 / 3/3 | 3/3 | 100% / 100% |
| ARC-Easy correct prob | .3649 / .3902 / .4136 | 3/3 / 3/3 / 3/3 | 3/3 | 100% / 100% |
| ARC-Challenge correct prob | .3575 / .3801 / .3960 | 3/3 / 3/3 / 3/3 | 3/3 | 100% / 100% |
| MMLU correct prob | .3356 / .3480 / .3577 | 3/3 / 3/3 / 3/3 | 3/3 | 100% / 100% |
| HellaSwag correct prob | .4559 / .4627 / .4599 | 3/3 / 3/3 / 3/3 | 3/3 | 100% / 100% |
| OLMES-10 accuracy | .3571 / .3535 / .3555 | 2/3 / 1/3 / 1/3 | 1/3 | 0% / 不可判 |

三任务 continuous 的相邻边在 seed 6198/14/15 上分别为 `QC20 - raw`
`+1.977/+1.630/+2.418pp`、`Orig10 - QC20` `+2.103/+1.721/+1.082pp`，没有靠 seed
均值掩盖反向 run。它对本地 1B target 和官方 20M 同规模排序都为 3/3。

相反，离散 OLMES-10 三个 seed 的 recipe 排序互相冲突，seed 均值为
`raw > Orig10 > QC20`，只命中 1/3；保守规则对三个 pair 全部弃权。它不仅比 continuous 差，
而且增加训练到 1.9B token 也没有修复 step-2,500 的不稳定性。因此本次验证后认定：**20M
continuous 有效，20M 离散 OLMES-10 无效；在这组三 recipe 上，step-2,500 continuous 已足够
给出同样的稳定排序，完整 20M 主要增加 margin。**

## 7. 本地 1B 目标端复评

三个 Falcon 公开 1B default checkpoint 已用与 20M 完全相同的本地 evaluator 复评。聚合结果在
`outputs/datadecide_reproduction/published_falcon_crossscale/summary.json`。这消除了“当前正确性
验证只有官方结果矩阵”的缺口，但目标模型权重仍由 DataDecide 作者训练，本项目没有重训 1B/100B
token 模型。

| 指标 | 本地 20M raw / QC20 / Orig10 | 本地 1B raw / QC20 / Orig10 | Pair 命中 |
|---|---|---|---:|
| 三任务 correct prob -> 三任务 accuracy | .3550 / .3643 / .3959 | .4635 / .4882 / .5171 | 3/3 |
| ARC-Easy correct prob -> accuracy | .3645 / .3804 / .4199 | .683 / .704 / .746 | 3/3 |
| ARC-Challenge correct prob -> accuracy | .3613 / .3652 / .4045 | .3805 / .4147 / .4454 | 3/3 |
| MMLU correct prob -> accuracy | .3393 / .3472 / .3632 | .3271 / .3459 / .3598 | 3/3 |
| HellaSwag correct prob -> accuracy | .4577 / .4673 / .4634 | .616 / .658 / .635 | 3/3 |
| OLMES-10 accuracy | .3484 / .3621 / .3775 | .5587 / .5768 / .5976 | 3/3 |

HellaSwag 的局部顺序是 QC20 > Orig10 > raw，与 OLMES-10 的 Orig10 > QC20 > raw 不同；小模型
仍正确预测了 HellaSwag 自己的 1B 顺序。这再次说明“预测能力排序”成立不等于存在唯一数据质量序。

DCLM raw 与 FW3 的公开 1B checkpoint 也已在本地完整复评，聚合在
`outputs/datadecide_reproduction/published_dclm_crossscale/summary.json`。结果复现了预注册的
crossover：本地 OLMES-10 在 20M 是 FW3 高 `0.476pp`，到 1B 却变成 raw 高 `0.702pp`。
因此这个反例不是只存在于官方 parquet，也不是当前 evaluator 版本差异造成的。

| 目标 | 本地 20M proxy raw / FW3 | 本地 1B target raw / FW3 | 方向 |
|---|---|---|---|
| OLMES-10 accuracy -> OLMES-10 accuracy | .3551 / .3599 | .5777 / .5707 | crossover，预测失败 |
| 三任务 correct prob -> 三任务 accuracy | .3816 / .4397 | .4932 / .5674 | FW3 > raw，预测成功 |
| HellaSwag correct prob -> HellaSwag accuracy | .4570 / .4451 | .646 / .525 | raw > FW3，预测成功 |

这个结果也解释了为什么不能拿连续三任务代理“修复”OLMES-10：ARC-Easy、ARC-Challenge、MMLU
的 1B accuracy 分别是 raw/FW3 `.715/.799`、`.412/.503`、`.353/.400`，三项都支持 FW3；
HellaSwag 则是 `.646/.525`，强烈支持 raw。两种汇总在回答不同能力目标，均与各自的小模型代理
一致，但 OLMES-10 的跨任务权衡随规模翻转。

## 8. 能力 benchmark 扩展复评

### 8.1 实验范围与 target 先行门

为验证注册名单中的其他能力任务是否也能由小模型代理，本轮固定
`rc-zero-shot-v1` base-model 协议，复评 [DataDecide](https://arxiv.org/abs/2504.11393) 全部
25 个公开 recipe。模型规模为 20M、60M、150M 和 1B，训练预算沿用固定约
100 token/parameter；所有 scale 使用同一题面、选项顺序、答案前缀和字符归一化评分。实验设计和
停止规则见[冻结实验协议](stage12_benchmark_proxy_protocol.md)，机器可读配置见
[`stage12_benchmark_validation.yaml`](protocols/stage12/stage12_benchmark_validation.yaml)。

第一步只运行 1B target。公开 checkpoint 只有 default 训练 seed，因此 bootstrap 和 FDR 只估计
benchmark 题目抽样不确定性，不估计训练 seed 方差。结果位于
`outputs/datadecide_benchmark_proxy/analysis_1b/summary.json`。

| 1B target | 平均 accuracy | 随机线 | 归一化提升 | 原始/FDR 明确 pair | 裁决 |
|---|---:|---:|---:|---:|---|
| MMLU-Pro | 10.55% | 11.25% | -0.79% | 235/218 | `needs_larger_target` |
| GPQA-Diamond | 21.15% | 25.00% | -5.14% | 14/0 | `needs_larger_target` |
| MMMLU-ZH | 26.76% | 25.00% | 2.34% | 27/0 | `needs_larger_target` |
| MMLU-CF | 32.07% | 25.00% | 9.43% | 204/176 | `screenable_single_seed` |

MMLU-Pro 是本轮最重要的反例：1B 虽在 FDR 后稳定区分 218 个 recipe pair，平均表现却低于按实际
选项数计算的随机线。它更可能在测稳定的答案文本或选项偏好，而不是“模型会做 MMLU-Pro”；若只看
显著 pair 数，会错误批准这个 benchmark。GPQA 同样低于随机；MMMLU 当前只测 ZH_CN，归一化提升
只有 2.34%，且 FDR 后没有明确 pair。三者均应升级 target，而不是评价 proxy 成败，也不能用本轮
结果代表 GPQA 更大子集或 MMMLU 其他语言。

### 8.2 MMLU-CF 跨尺度结果

只有 MMLU-CF 进入 20M/60M/150M。完整联合结果位于
`outputs/datadecide_benchmark_proxy/analysis_proxy/summary.json`。下表每格为
`Kendall tau-b；原始明确 pair accuracy / FDR 明确 pair accuracy`，分母分别为 204 和 176；四个
预注册指标全部展示。

| Proxy 指标 | 20M | 60M | 150M |
|---|---:|---:|---:|
| accuracy | .580；86.27% / 89.77% | .447；80.88% / 82.39% | .747；96.57% / 98.30% |
| correct prob/char | .720；95.59% / 97.16% | .747；95.10% / 97.16% | .767；96.57% / 97.73% |
| correct-vs-best-incorrect margin | .720；92.16% / 92.61% | .393；78.43% / 81.82% | .600；90.20% / 93.75% |
| normalized choice probability | .700；94.61% / 96.59% | .600；90.20% / 92.61% | .780；98.04% / 98.86% |

20M `correct_prob_per_char` 已达到 171/176，且是 DataDecide 既有的预指定 continuous 指标，因此
本轮把它保留为**最小早期筛选档**。60M 没有提高其 FDR 命中率；150M 的 normalized choice
probability 最好，但这是在同一 target 上比较四项后看到的结果，不能据此事后替换生产指标。它应
作为 held-out family 实验的候选，而不是本轮“冠军”。离散 accuracy 在 60M 反而低于 20M，margin
也明显非单调，说明“扩大 proxy 就必然更可靠”不成立。

20M correct-prob 的 5 个 FDR 错误 pair，proxy 绝对 margin 都不超过 0.00647。事后用 0.0065
阈值弃权可保留 159/176 pair 并达到 159/159；60M 同一阈值为 161/176、161/161。这是有价值的
阈值候选，但阈值来自当前矩阵，不能作为有效性证据，必须在新的 recipe family 上冻结验证。
`dclm-baseline` 对 `falcon-and-cc-qc-20p` 的 correct-prob 方向在 20M、60M、150M 都与 1B 相反，
证明即使总体命中率很高，仍存在扩大到 150M 也不能修复的系统性 pair。

因此正式状态为：MMLU-CF + 20M correct-prob 是 `early_screen_only`，不是 `validated_proxy`。
升级仍缺三个训练 seed、按 recipe family 的 held-out 确认、预先冻结的 margin/弃权规则，以及至少
8 个与团队“通用知识”生产变化匹配的专项 recipe。代码、数学和长程数据仍需各自 benchmark；本轮
不能替它们背书。

### 8.3 周期、完整性事故与修复

仅看已经缓存权重后的 MMLU-CF evaluator，单模型平均用时为 20M `21.5s`、60M `21.7s`、150M
`22.4s`、1B `43.2s`。评测只快约两倍，不应冒充训练周期收益；真正的节省来自固定
token/parameter 时训练计算量约按参数量平方增长。按名义 scale，20M/60M/150M 约为 1B 训练
FLOPs 的 `0.04%/0.36%/2.25%`。公开权重首次下载的 123GiB 缓存是本次复评成本，不是生产代理
每批数据都要重复支付的成本。

下载阶段曾启用 `HF_HUB_ENABLE_HF_TRANSFER=1`。镜像返回 `no permits available` 后，并行 partial
不能作为连续前缀被普通 HTTP 正确续传，产生了**尺寸正确但内容损坏**的 safetensors。以
`dclm-baseline-50p-dolma1.7-50p-1B` 为例，Hub blob ID 应为
`12e19d94388b6e2095a78efc2b48593b87923cbb86cec538062f4c67ca5e3809`，损坏文件实际 SHA-256 前缀为
`891f4d...`；模型仍能加载，却把 MMLU-Pro correct-prob/char 从健康值约 `.428` 降到 `.056`，并把
GPQA accuracy 异常抬到 28.79%。这类错误若只检查文件尺寸或“能否加载”会静默污染结论。

实现现已在直接 evaluator 和矩阵调度器中都禁用 `hf_transfer`，单模型下载固定
`max_workers=1`，并在加载前把每个 safetensors 实际 SHA-256 与 64 位 Hub LFS blob ID 比较；不符
时删除 blob、普通 HTTP 重下并再次校验。所有 175 份最终 summary 均记录
`model_blob_sha256`，四个 manifest 的最终失败列表为空。首次下载仍可能遇到 503 或断流，但只影响
周期，不再允许损坏权重进入结果。

## 9. 成本与周期

单个本地 proxy 使用 19,101,888 个论文口径参数和 1,911,554,048 token；官方 1B target 使用
1,176,832,000 个论文口径参数和 100,015,669,248 token。按论文共同的 `6ND` 近似，单 recipe
proxy/target compute 比为：

`(19,101,888 × 1,911,554,048) / (1,176,832,000 × 100,015,669,248) = 0.0310%`

参数比为 1.62%，token 比为 1.91%。比较多个 recipe 时分子分母同乘 recipe 数，比例不变。这个
比例证明训练资源量级显著更低，但不是相同硬件上的目标 1B full run wall-time 实测；报告不把 FLOPs
比冒充日历时间加速比。

step-2,500 早停点的同口径 compute 比为 `0.00532%`，比完整 20M proxy 再低约 5.8 倍；它是否
可用必须由本地 benchmark 排序决定，不能只因便宜就晋级。

MXC500-64G 上，九条训练轨迹到 step 2,500 的平均 wall time 为约 `2,039s`（34.0 分钟）；完整
20M 已完成 run 的 wall time 集中在 `11,854-11,980s`（3.29-3.33 小时）。同一 20M checkpoint
的 66-task OLMES 评测约 14 分钟。因此单 recipe/seed 的实测周期约为：

| 代理点 | 训练 token | 训练 | evaluator | 合计 | 对 1B target 训练 FLOPs 比 |
|---|---:|---:|---:|---:|---:|
| step 2,500 | 327.68M | 34.0 分钟 | 约 14 分钟 | 约 48 分钟 | 0.00532% |
| 完整 20M | 1.912B | 约 3.30 小时 | 约 14 分钟 | 约 3.54 小时 | 0.0310% |

三 recipe × 三 seed 若全串行，早停约 `7.2 GPU-hour`、完整点约 `31.9 GPU-hour`（均含 evaluator）；
本机可用七张实验卡并行，实际日历时间更短。本次早停点来自完整轨迹，所以补测早停没有重复支付
训练 FLOPs，只增加 checkpoint 存储和 evaluator。

本项目没有在本机重训 1B/100B-token target，因而不能报告同硬件 target wall time；上表最后一列
是透明的 `6ND` 计算量比，不是实测日历加速比。不过参数与 token 均缩小约 50-60 倍，计算量缩小
约 3,200 倍，
加上本地每个候选 48 分钟或 3.54 小时的实测，已经满足“反馈周期显著短于 full target training”
的工程目标。公开 1B target 的本地 evaluator 只验证结果端，不应与昂贵的 target 训练混为一谈。

## 10. 最终判断、漏洞与生产建议

### 10.1 方法去留

- **保留 20M continuous ARC/MMLU 类代理**：官方全 25-recipe 历史回测在 20M 已有
  `86.3%-90.0%` decision accuracy，本地 step-2,500 和公开 20M checkpoint 复评也显示它比
  离散 accuracy 更早形成方向。它只能标注为对应知识/推理能力代理。
- **不把 20M OLMES-10 离散 accuracy 作为默认决策器**：全矩阵单 seed 只有 `71.3%`，三 seed
  decision-accuracy 均值只有 `64.6%`，本地早停也把边界 pair 判反。需要综合能力宏平均时，应先在
  历史 recipe 上验证 60M/150M 是否达到门槛；论文和本地回算支持 150M 更稳，但仍非必然正确。
- **step-2,500 只保留为第一级筛选**：它的单 run 成本很低且本地有信号，但官方 seed 已出现方向
  不一致。它可以淘汰大 margin 失败方案，不能单独批准小 margin winner。
- **排除 BoolQ continuous**：官方 20M/60M/150M 都接近随机方向，不值得继续消耗本地算力。
- **HellaSwag 单独校准**：它在 20M 的全矩阵方向性弱于 ARC/MMLU，且 Falcon、DCLM 都展示了与
  知识任务不同的数据偏好；需要更大 proxy 或更多 token，不能混入三任务均值后掩盖。
- **MMLU-CF 保留为 20M early screen**：全 25-recipe 本地复评中，20M correct-prob 对 1B 的
  FDR 明确 pair 命中 97.16%，但公开权重只有单训练 seed，且没有 held-out 生产 family，因此不能
  标为 `validated_proxy`。MMLU-Pro、GPQA-Diamond 和 MMMLU-ZH 在当前 1B target 上停止。

本地复现对 DataDecide 核心命题给出**有限但明确的支持**：从代表性 Falcon recipe 重新物化数据、
训练九个 20M 模型后，小模型 continuous 排序在两个训练预算、三个 seed 和四个分任务上均预测了
本地复评的 1B 排序；这不再只是官方结果矩阵的二次统计。约 48 分钟的早停周期已经能给这三个
recipe 产生有参考价值的方向，方法值得保留。

支持不能扩大为“任意小模型都能预测任意大模型”。证据同时否定了两个更强说法：20M 离散宏平均
在本地三 seed 上无可用性；DCLM raw/FW3 的本地 target 复评证明综合排序可以真实 crossover。
所以最终上线形式应是**按能力校准的、小模型 continuous 排序器 + 弃权/升级机制**，而不是统一的
数据质量分，更不能替代一次性的 target-scale 历史校准。

### 10.2 建议的生产协议

1. **先写能力契约，再选 benchmark。** 通用知识数据可用 ARC/MMLU 类 held-out 任务；代码数据用
   代码生成、补全和测试通过率任务；长程数据用长上下文检索、跨段推理和持续一致性任务。三类数据
   分开出报告，不计算跨类型的“总体质量分”。本次实现只校准了第一类。
2. **建立一次性的历史校准矩阵。** 对已有 recipe 在 20M/60M/150M、至少三个训练 seed 上运行
   proxy，并用少量已经存在的 1B 或目标规模 run 作 target。按 benchmark 分别选出满足内部准确率
   和覆盖率门槛的最小规模；target 校准成本可以跨后续数据批次摊销。
3. **候选数据之间固定所有非数据变量。** 同一次决策使用相同架构、参数量、token 数、tokenizer、
   batch、优化器和训练 seed。跨模型规模做校准时保持论文的固定 token/parameter 比；固定 token
   与固定比例回答不同问题，结果不可混用。
4. **连续指标优先，保留分任务结果。** 小模型先读取字符归一化 correct probability，不等离散
   accuracy 脱离随机线；MMLU 先做学科宏平均。任何组合均值旁边必须同时展示每个任务，避免相反
   能力偏好互相抵消。
5. **三 seed 均值用于排序，全 seed 同向用于自动决策。** 当前汇总器同时输出逐 seed、seed 均值和
   `unanimous_seed_decisions`。只要任一 seed 反向或 tie，就对该 pair 弃权并升级 scale/token；
   三 seed 同向是运行规则，不宣称统计显著。
6. **把 proxy benchmark 与最终模型验收分开管理。** DataDecide 的正确性校准必须比较同一能力在
   小/大模型上的方向，但日常数据生产不应反复调参追逐最终 leaderboard。建议从目标能力族中划分
   冻结的 proxy holdout 和独立的最终验收集，并定期更换、做污染审计；若换任务，必须重新校准跨
   尺度方向性。
7. **保留升级与否决路径。** 小 margin、seed 分歧、能力任务互相冲突或历史 crossover 区域一律
   输出 `abstain`；升级到 60M/150M，或在高价值决策上补一个 target-scale anchor。许可、隐私、
   污染、安全和覆盖率仍由现有数据本体链路一票否决，proxy 高分不能覆盖这些问题。

### 10.3 主要漏洞

- **选组偏差**：8 个 recipe 是在阅读官方矩阵后预注册的诊断子集；子集 28/28 不能替代全矩阵
  300 pair 的历史准确率。最终是否上线应以团队自己的历史 recipe 矩阵重新校准。
- **本地数据不逐 bit 等价**：range 抽样覆盖官方逻辑 path，但缺少作者 exact shuffle、文档边界和
  doc ID；它可能改变小 margin pair，也无法进行逐文档归因。
- **目标权重未本地重训**：1B target 已在本地 evaluator 上复评，但权重仍来自作者。完整独立复现
  还需要约 100B token/recipe 的 1B 训练，本项目刻意不支付该成本。
- **evaluator 版本差异**：当前 OLMES alias 对多个任务限制 1,000 题，而论文使用完整 split；方向
  基本一致不等于逐数值复现。OLMES commit、依赖版本、原始 predictions 和 task hash 必须固化。
- **benchmark 泄漏**：若候选 recipe 对 proxy 题库或近重复内容覆盖不同，方法会把污染误认为数据
  能力。运行前必须接入现有 contamination stage，且不能公开后长期复用同一小题集做数据调参。
- **真实 crossover**：DCLM raw/FW3 已证明 20M、60M、150M 全部可能预测错 1B 综合排序。更多
  seed 只能估计训练噪声，不能消除模型规模与数据相互作用；生产输出必须承认这一不可识别区间。
- **能力覆盖有限**：ARC/MMLU/HellaSwag 不覆盖代码、数学证明、多语言、长上下文、事实时效和
  生成质量。本报告不能为这些数据类型背书，需要分别建立 benchmark 与跨尺度校准。
- **新增矩阵仍是单训练 seed**：MMLU-CF 的题目 bootstrap 和 FDR 不能替代训练 seed 方差；当前
  97.16% 只能用于候选筛选。MMLU-Pro 还证明“显著区分 recipe”与“模型具备该能力”并不等价。

## 11. 复现入口与测试

原实验 CLI 已随方法退役，不再提供可执行入口。保留的关键输出：

- 官方论文口径回算：`outputs/datadecide_reproduction/official_matrix_paper_protocol/summary.json`；
- 官方 20M 本地复评：`outputs/datadecide_reproduction/published_20m/`；
- 20M 分任务聚合：`outputs/datadecide_reproduction/published_20m_aggregate_per_task/summary.json`；
- 本地物化 manifest：`data/trainability/datadecide_local/*.manifest.json`；
- 本地训练：`outputs/datadecide_reproduction/local_train/`。
- 本地三 seed 早停：`outputs/datadecide_reproduction/local_step2500_three_seed_crossscale/summary.json`；
- 本地三 seed 完整 20M：`outputs/datadecide_reproduction/local_final_three_seed_crossscale/summary.json`；
- DCLM 本地 crossover：`outputs/datadecide_reproduction/published_dclm_crossscale/summary.json`。
- 新增 benchmark 的 1B target 审计：`outputs/datadecide_benchmark_proxy/analysis_1b/summary.json`；
- MMLU-CF 三档 proxy 联合结果：`outputs/datadecide_benchmark_proxy/analysis_proxy/summary.json`。

最终验证：`pytest -q` 为 `73 passed`；本次新增范围的 `ruff check` 与 `ruff format --check`
通过；`compileall` 通过。全仓 ruff 仍报告整理工作区已有的 8 个 lint 和 43 个格式问题，均不在
Stage 12；为避免覆盖另一位 agent 的整理变更，本任务没有顺手改写这些文件。

原始 benchmark predictions 可能包含评测题面，只保存在 git-ignored `outputs/`，报告和提交中不嵌入。
