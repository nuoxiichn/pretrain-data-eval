# 小模型 benchmark 代理扩展实验设计

状态：首轮公开权重筛选已完成（2026-07-23）。MMLU-CF 可进入单 seed 早期筛选，MMLU-Pro、
GPQA-Diamond 和当前只含 ZH_CN 的 MMMLU 均因 1B target 地板效应停止；这不是三 seed
`validated_proxy` 结论。机器可读计划见
[`stage12_benchmark_validation.yaml`](protocols/stage12/stage12_benchmark_validation.yaml)，
完整注册名单见[`benchmark_registry.yaml`](protocols/stage12/benchmark_registry.yaml)。

## 1. 要验证的命题

本实验不问“小模型 benchmark 分数高不高”，而问一个成对排序问题：在训练设置和计算预算固定时，
数据 recipe A 在小模型的某个代理指标上优于 B，能否预测 A 训练的 1B 模型在**同一能力**上也优于
B。方法沿用 [DataDecide](https://arxiv.org/abs/2504.11393) 的跨尺度排序思想和
[OLMES](https://arxiv.org/abs/2406.08446) 的可复现评测原则。

每个 benchmark 都是独立待验证对象。ARC/MMLU 的成功不能证明 MMLU-Pro、GPQA、MATH 或
EvalPlus 也能由 20M 模型代理；同一 benchmark 的 accuracy、答案似然和生成通过率也不能混作一个
指标。最终允许的结论形如“150M 的指标 X 可以筛选 1B 上 benchmark Y 的数据排序”，而不是
“benchmark Y 可以衡量数据总体质量”。本轮仍只把 1B 作为 target，不回答 3B/7B/35B 外推。

## 2. 为什么按三类能力拆开

| 能力族 | 注册表中的主要任务 | 小模型的主要困难 | 候选代理 |
|---|---|---|---|
| 知识/多选 | MMLU-Pro、GPQA-Diamond、MMMLU | accuracy 容易接近随机 | 正确答案连续概率、答案相对错误项的 margin |
| 数学生成 | MATH、MGSM、GSM-Plus | 完整答案准确率可能全为 0 | 最终答案似然、参考解答似然、生成准确率 |
| 代码生成 | EvalPlus、LBPP、MHPP | pass@1 和编译率可能全为 0；正确程序不唯一 | 参考实现似然、正确实现对行为错误 mutant 的 margin、执行通过率 |

只在 SFT 阶段注册的 IFEval、SimpleQA、LiveBench、MT-Bench、Arena-Hard、AIME 和 MATH-500 暂不
进入预训练代理实验。它们的分数同时依赖 post-training、提示模板或 Judge；直接比较 20M base 与
SFT 模型会把多种变量混在一起。以后若要纳入，必须给每个预训练 recipe 接上完全相同的
post-training bridge，或者先定义可由 base 模型计算的目标指标，再单独校准。

## 3. 两阶段证据链

### 3.1 公开权重筛选

第一阶段复用 DataDecide 的 25 个 recipe，以及公开的 20M、60M、150M、1B checkpoint。先只评测
1B target：如果 1B accuracy 本身接近随机，或者三个 seed 无法给出至少 20 个方向明确的 recipe
pair，就没有可靠的“标准答案”供小模型预测。此时结论是 `needs_larger_target`，不是 proxy 失败。

target 有分辨率后，对同一个 checkpoint 一次加载并运行全部可用候选任务。proxy 先跑 default
seed；只有同时满足“不是地板/天花板、recipe 间差异超过评测噪声、对 1B 排序明显优于随机”的
指标才补齐三个 seed。default seed 只用于筛选，不能单独晋级。

这一阶段几乎不支付训练成本，适合快速否决 benchmark 或确定最小可学习规模。但 25 个 recipe 主要
是通用网页数据处理方案。如果某个代码 benchmark 在这批 recipe 上有信号，只能说明它值得进入下
一阶段，不能说明它对团队的代码数据版本也可靠；没有信号也可能只是这些 recipe 的代码差异太小。

### 3.2 能力专项确认

知识、数学、代码分别准备至少 8 个 recipe、3 个来源或生产 family。每组至少包含两个真实生产版本、
明显差异边、临界边和可能发生 crossover 的边。人工打乱答案、破坏代码等只能作为负对照，不能充当
主要 ground truth。所有 recipe 使用等量 token 替换，不能通过给某组额外增加领域 token 制造优势。

每个 recipe 在 20M、60M、150M 上按约 17、50、100 token/parameter 保存 checkpoint，最终用 1B、
100 token/parameter 作为 target。所有同规模候选固定模型、tokenizer、优化器、batch、上下文长度和
训练 seed。先训练一个 seed 检查任务是否有分辨率，再补齐三个 seed；不能把同一轨迹的三个
checkpoint 当作三个独立 seed。

候选代理指标只能在开发 recipe family 上选择，最后必须在未参与选择的 family 上确认。这样可以避免
看过 target 后，从多个指标里挑一个碰巧最好的结果。

专项 recipe 的 1B ground truth 优先复用训练协议匹配的团队历史 checkpoint，其次使用数据 recipe
可解释的公开权重。若都不足，先用 20M/60M 预测 150M 做 pilot，只有通过的 benchmark 才为少数
关键 pair 新训练 1B anchor。这样能控制一次性校准成本，但没有 1B anchor 时只能记为初步证据。

### 3.3 统一评测协议

本轮评价的是 base model，不使用 chat template。一个 benchmark 在所有 scale 上必须使用相同题干、
选项顺序、shot 数和答案前缀；小模型连续指标也从这同一批输出重建，不能另换更简单的提示。默认禁用
chain-of-thought；如果团队的 base evaluator 明确使用 CoT，应把它注册成另一条实验，不能和直接作答
结果混合。执行前还必须固定 evaluator commit、答案解析器、task hash 和聚合方式。

这意味着本实验得到的可能是“MMLU-Pro base-direct-answer”代理，而不是官方聊天模型 leaderboard
口径。名称和报告必须把 prompt variant 写清楚，否则看似在验证同一个 benchmark，实际比较的是两种
不同任务。

## 4. 分能力实验

### 4.1 知识与多选题：第一优先级

首轮运行 MMLU-Pro、GPQA-Diamond、MMMLU 和 MMLU-CF。MMLU、C-Eval、CMMLU、CMB 只作污染或
中文方向 sanity check，不直接晋级生产；GPQA main 与 Diamond 有重叠，也不能当成两份独立证据。
MMLU-Pro 和 GPQA 分别参考其[公开数据](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)与
[公开数据](https://huggingface.co/datasets/Idavidrein/gpqa)，GPQA 需要授权 token。

每题同时记录四项：accuracy、字符归一化正确答案概率、选项内归一化正确答案概率，以及正确项相对
最高错误项的 log-prob margin。1B target 使用该 benchmark 的 accuracy。MMLU 类任务先对学科宏
平均；MMMLU 必须逐语言报告，不能只用当前 `en` 或 `ZH_CN` 子集代表“多语言”。

### 4.2 数学：第二优先级

主要任务为 MATH 和 MGSM，GSM-Plus 是扰动对照，GSM8K 只作已知高曝光 sanity check；GSM1K 在
本地数据取得前保持 blocked。MATH 来自
[Hendrycks 等人的工作](https://arxiv.org/abs/2103.03874)，MGSM 来自
[多语言数学推理工作](https://arxiv.org/abs/2210.03057)。

1B target 使用最终答案 exact match。小模型分别计算最终答案似然、完整参考解答的每 token 似然和
真实生成 exact match。参考解答似然容易偏爱常见措辞或记忆过的模板，因此不能单独晋级。如果 20M、
60M 的生成准确率全为零，但 150M 连续指标能够稳定预测 1B，最终结论应是“最小代理为 150M”，而
不是 benchmark 失败。

### 4.3 代码：第二优先级

主要任务为 [EvalPlus](https://arxiv.org/abs/2305.01210)，LBPP、MHPP 作相对干净的同族对照，
HumanEval、MBPP 只作高曝光 sanity check。LiveCodeBench 参考
[其公开工作](https://arxiv.org/abs/2403.07974)，但注册表的可用 `data_id` 确认前保持 blocked。

1B target 使用真实执行的 pass@1。小模型候选指标包括参考实现每 token 似然、正确实现相对“语法
正确但行为错误”的 mutant 的似然 margin、生成代码编译率和真实 pass@1。因为同一题可以有很多种
正确程序，单一参考实现似然只反映一种写法，必须同时通过 mutant margin 与 target pass@1 的排序
验证。执行环境需固定依赖、超时和沙箱，原始生成结果保留但不写进公开报告。

## 5. 统计单位与晋级规则

统计单位是 recipe pair，不是 benchmark 题目。若 1B 的三个 seed 对 A/B 方向不一致，该 pair 的
target 本身不确定，应标为 ambiguous；不能把它算成 proxy 预测错误，也不能偷偷删除而不报告。

主要输出为 pairwise decision accuracy、弃权覆盖率、弃权后的条件正确率，以及按 recipe family
聚类 bootstrap 的 95% 区间。Kendall tau、逐 seed 方向、margin 和地板/天花板比例作为辅助信息。
题目 bootstrap 会忽略 recipe pair 的相关性，不能作为唯一置信区间。

公开单 seed 筛选还同时报告配对 sign-flip 检验经 Benjamini-Hochberg FDR 校正后的敏感性结果。
这是因为 25 个 recipe 会产生 300 个 pair，只看 300 个未经校正的 95% 区间可能把少量偶然差异
误记为明确方向。原始预注册口径与 FDR 口径都保留；二者结论不一致时不得晋级，只能增加 target
seed 或更换更有分辨率的 target。FDR 仍不能替代训练 seed，因为它只处理题目抽样与多重比较。

预注册的首版工程门槛是：至少 8 个 recipe、3 个 family 和 20 个 target 明确 pair；方向准确率至少
80%；聚类 bootstrap 下界至少 65%；采用“三 seed 同向且超过 margin，否则弃权”后，条件正确率至少
90%、覆盖率至少 50%；最差 held-out family 不低于 65%，并且没有已知系统性 crossover。门槛不是
论文常数，只能在新的 held-out family 上修改，不能看过同一结果后降低。

target 还必须越过随机地板：定义归一化提升为
`(平均 accuracy - 随机选择 accuracy) / (1 - 随机选择 accuracy)`，至少达到 `0.05`。例如四选一
任务的门槛是 28.75%，十选一任务是 14.5%。这不是能力合格线，而是避免把几乎不会做题的模型中
由选项偏好造成的小幅、统计显著波动当成可代理的能力排序。随机线按任务实际选项数和相同聚合方式
计算；未越过时统一记为 `needs_larger_target`。

通过后还要区分几种结论：`validated_proxy` 可进入相应能力的常规排序；`early_screen_only` 只能淘汰
明显失败方案；`needs_larger_proxy` 必须升级小模型；`needs_larger_target` 表示 1B 本身无法提供稳定
ground truth。未通过记为 `rejected_proxy`，数据源或 evaluator 未就绪记为 `blocked`。所有失败指标
也要进入报告，禁止只展示最好的一个。

## 6. 推荐执行顺序与停止条件

1. 固化注册表 revision、base 评测 prompt、答案解析器、依赖和污染审计结果。
2. 先评测公开 1B checkpoint；target 不足 20 个明确 pair 时标为 `needs_larger_target` 并停止。
3. 实现通用多选连续指标，先用现有 ARC/MMLU predictions 做回归测试。
4. 对公开 DataDecide default-seed 小 checkpoint 运行四个首轮多选任务。
5. 若某任务在 150M 仍无分辨率，标为当前规模 `rejected_proxy`，不补 proxy seed。
6. 对有信号的“benchmark + metric + 最小规模”补齐公开三 seed，并做 family-held-out 分析。
7. 只有公开权重筛选通过，才准备该能力的 8 个专项 recipe 和本地训练矩阵。
8. 多选确认完成后再实现数学解析器；数学有候选后再实现代码执行与 mutant 管线。
9. 任一环节发现污染差异、target seed 不稳定或稳定 crossover，停止自动晋级并记录失败原因。

第一批应只启动公开权重的多选评测，不应立即训练新模型。这会先回答 MMLU-Pro、GPQA-Diamond、
MMMLU、MMLU-CF 中哪些在 20M/60M/150M 上有可用连续信号，再决定专项训练预算。

## 7. 首轮执行结果

公开矩阵覆盖 DataDecide 全部 25 个 recipe。1B target 共评测四项任务；只有 MMLU-CF 同时越过
随机地板并在 FDR 后保留至少 20 个明确 pair。随后只对 MMLU-CF 运行 20M、60M、150M，遵守第
6 节的 target-first 停止规则。正式结果与漏洞见
[DataDecide 小模型排序复现报告](stage12_datadecide_report.md#8-能力-benchmark-扩展复评)。

| 1B target | 平均 accuracy | 随机线 | 归一化提升 | 原始/FDR 明确 pair | 结论 |
|---|---:|---:|---:|---:|---|
| MMLU-Pro | 10.55% | 11.25% | -0.79% | 235/218 | `needs_larger_target` |
| GPQA-Diamond | 21.15% | 25.00% | -5.14% | 14/0 | `needs_larger_target` |
| MMMLU-ZH | 26.76% | 25.00% | 2.34% | 27/0 | `needs_larger_target` |
| MMLU-CF | 32.07% | 25.00% | 9.43% | 204/176 | `screenable_single_seed` |

MMLU-CF 的 20M `correct_prob_per_char` 已在 176 个 FDR 明确 pair 上命中 171 个（97.16%），
所以最小筛选档暂定为 20M；150M 的选项内归一化概率达到 174/176（98.86%），但这是看过同一
target 后的候选比较，必须在 held-out recipe family 上确认后才能替换默认指标。公开 checkpoint
只有 default 训练 seed，且 recipe 主要是通用网页处理方案，因此当前状态只能是
`early_screen_only`。
