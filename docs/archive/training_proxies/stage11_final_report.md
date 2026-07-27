# Stage 11 微型训练代理最终实验与决策

日期：2026-07-22

状态：**Retired / Production No-Go**

本文是 Stage 11 的唯一权威结论入口，覆盖 Anchor-relative scaling gain、Balanced-pool scaling
gain 和 data-conditioning。三种方法均未通过生产或辅助决策校准，不得进入质量排名、过滤、采样
权重、推荐字段或综合分。历史 `data/`、`outputs/` 和冻结配置继续作为只读实验资产保留。

Stage 12 DataDecide 不属于本文范围，也不因 Stage 11 退役而改变状态。

## 1. 统一决策

| 方法 | 最强正面证据 | 否决证据 | 最终状态 |
|---|---|---|---|
| Anchor-relative | 四类 Anchor 的词序控制均单调；英文独立复跑排序一致 | 24 个 Anchor/轴组合中 12 个失败；重复、模板、截断存在反向；最稳定的严重词序控制缺少真实生产场景 | **No-Go** |
| Balanced | Null 20/20 abstain；词序和模板控制通过 | Repeat 与 Truncate 前瞻硬门失败；P5-P10 按协议停止 | **No-Go** |
| Data-conditioning | 训练和推理成本显著低于目标大模型 | 固定比例实验 0 条正确边；官方 60M/150M checkpoint 在近分布 pair 上显著反向 | **No-Go** |

统一原因不是单纯“样本量不足”，而是目标量与生产问题不一致：

- scaling gain 会奖励可被更大模型利用的结构，也可能奖励低熵重复、模板和领域复杂度；
- cross-corpus conditioning loss 主要反映模型对自身训练分布的专化，不能稳定预测最终能力排序；
- 合成控制上的统计响应不自动具有真实数据场景中的检测价值；
- 三种方法都强依赖 Anchor、tokenizer、模型族、训练 horizon、语言和领域，不能产生跨 run 总分。

## 2. Anchor-relative scaling gain

### 2.1 冻结协议

- Anchor：英文网页、中文网页、代码、数学；每类 3,000 篇。
- 模型：953,856 / 4,273,664 参数；context 256。
- Seeds：17、29、41、53、67。
- 每模型训练 2,457,600 token；每个 Anchor 训练 5 组 Small/Large，共 10 个模型。
- 每个 run 同时评估 clean、六类合成退化轴和另外三个高质量 OOD Anchor。
- 数据集均值采用 token 加权；CI 使用 seed cluster 加文档层次 bootstrap。
- split 同时约束 declared dedup/near cluster 和 normalized exact-text hash，无 cluster 跨
  train/validation。

四类 Anchor 的 train/validation 文档数分别为：

| Anchor | Train / validation docs | Train / validation clean tokens | License |
|---|---:|---:|---|
| English web | 2,400 / 600 | 1,961,178 / 503,577 | Ultra-FineWeb-L3, Apache-2.0 |
| Chinese web | 2,426 / 574 | 1,645,403 / 411,834 | Ultra-FineWeb-L3, Apache-2.0 |
| Code | 2,417 / 583 | 3,379,440 / 831,377 | The Stack dedup, per-document permissive metadata |
| Math | 2,391 / 609 | 4,412,653 / 1,192,531 | OpenWebMath, ODC-By-1.0 |

每类保留 200 条 `not_reviewed` 人工抽检入口；本轮没有虚构人工结论。完整 SHA-256、来源 shard、
tokenizer、抽检和 leakage audit 位于
`data/trainability/anchor_calibration_v1/manifest.json` 及各 run 的 `manifest.json`。

### 2.2 复现性、泄漏和兼容性

- 英文 Anchor 做了一次独立完整复跑。
- 19 个 gate-comparable corpus 的平均绝对差为 0.001361 bit/token，最大 corpus 均值差为
  0.002002 bit/token。
- 六个同用途组的原始排序完全一致，阈值化排序边集合一致。
- 四类 Anchor 的 train-validation exact、near、doc-id overlap 均为 0。
- 12 个 OOD corpus 全部由 compatibility gate 输出 `abstain`，没有形成跨域排序边。

这些结果证明实验管线在冻结条件下可复现，并证明 gate 能阻止已声明的不兼容比较；它们不证明
gain 是质量信号。

### 2.3 合成控制结果

下表为 clean 与最差档的 gain 差。正值表示 clean gain 更高。

| Anchor | Word order | Template | Repeat | Truncate | Extraction residue | HTML residue |
|---|---:|---:|---:|---:|---:|---:|
| English web | +0.2168 pass | -0.0592 fail | +0.0145 pass | +0.0025 fail | +0.0745 pass | +0.0227 pass |
| Chinese web | +0.2457 pass | -0.1183 fail | -0.0058 fail | -0.0030 fail | -0.0000 fail | +0.0154 fail |
| Code | +0.3131 pass | +0.0400 pass | -0.0283 fail | -0.0161 fail | +0.0280 pass | +0.0069 fail |
| Math | +0.2001 pass | -0.0322 fail | +0.0253 pass | +0.0100 pass | +0.0135 pass | +0.0067 fail |

共 24 个 Anchor/轴验收项，12 个失败。所有失败和反向轴均保留，没有事后删除。

Word order 是唯一在四类 Anchor 上均达到 `rho=1.0` 且端点 CI 分离的轴，但当前控制模拟的是严重
词序扰动。网页生产故障通常是 DOM block、表格、OCR 或抽取顺序错误；合成数据也极少自然产生
这种词级破坏。它因此只能证明指标对极端顺序熵变化敏感，不能支持有实际需求的生产检测主张。

重复和模板的反向结果揭示了更根本的限制：低熵或高度可预测文本可能更容易被模型利用，从而获得
更高 gain。降低 rho 或 CI 门槛不能修复这个含义冲突。

### 2.4 Anchor 决策

Anchor-relative gain 只能描述“相对于指定 Anchor 的结构兼容性和容量可利用性”，不能命名为
通用质量分。虽然协议、复现性和 gate 验收完成，A3 的效度验收失败，最终不保留生产或辅助决策
用途。词序结果仅作为历史正控制证据，不值得单独建设线上检测器。

原始产物：

- `data/trainability/anchor_calibration_v1/`
- `outputs/stage11_anchor_calibration_v1/`
- 冻结配置：[`protocols/stage11/stage11_anchor_calibration_v1.yaml`](protocols/stage11/stage11_anchor_calibration_v1.yaml)

首次模板控制曾造成 normalized exact text 跨 split；失败数据和运行保留在两个
`failed_exact_split_overlap/` 目录，修复后所有正式实验已重跑。

## 3. Balanced-pool scaling gain

### 3.1 方法和验证链

Balanced 在同一个 A/B 等 token 混合池上训练 Small/Large 模型，再比较 A、B validation 的
`CE_small - CE_large`。它控制了训练数据差异，但没有控制 validation 难度、低熵模式和容量效应。

| 阶段 | 结果 | 决策 |
|---|---|---|
| 早期探索 | 词序、模板有响应；跨域 OpenWebText/Raw CC 方向错误 | 限制为 matched recipes |
| v1 P4 | Null 20/20 abstain；词序和模板通过；Repeat 失败 | v1 No-Go，停止剩余 P4 与 P5-P10 |
| v2 资格预检 | 独立 repetition gate 拒绝 Repeat；Extraction residue 也超过 5% 重复门 | v2 在训练前被 v2.1 取代 |
| v2.1 P4 | 资格门与 Null 通过；Truncate 硬门失败 | 最终 No-Go；HTML、OCR、P5-P10 不再运行 |

两轮 Null 复用了冻结控制，不能解释为 40 组独立证据。

### 3.2 关键控制

| 控制轴 | 四档 gain | rho | 端点差 [95% CI] | 结论 |
|---|---|---:|---|---|
| Word order | 0.2647 / 0.2477 / 0.2320 / 0.2160 | +1.0 | +0.0487 [0.0279, 0.0656] | 通过，但场景价值不足 |
| Template | 0.5776 / 0.3279 / 0.1449 / 0.0019 | +1.0 | +0.5758 [0.5449, 0.5958] | 通过，但 horizon 有风险 |
| Repeat | 0.2928 / 0.2944 / 0.2967 / 0.2995 | -1.0 | -0.00669 [-0.01491, 0.00171] | 失败 |
| Truncate | 0.29487 / 0.29569 / 0.29463 / 0.29697 | -0.4 | -0.00210 [-0.00762, 0.00359] | 失败 |

独立 repetition detector 能在训练前识别 Repeat 和本轮 Extraction residue，只说明已有 Stage 4
能力应承担重复判断，不能使 Balanced 变成新的质量维度。Truncate 在通过重复资格门后仍失败，
是直接的前瞻反例。

### 3.3 Balanced 决策

Balanced 不得进入生产质量维度、总分、过滤或推荐。已有词序/模板响应不再保留为候选辅助信号：
词序缺少实际生产场景，模板坍缩也有更便宜、可解释的重复和模板检测方法。P5-P10 不补跑。

原始产物：

- `outputs/stage11_balanced_validation_v1/`
- `outputs/stage11_balanced_validation_v2_1/`
- `data/trainability/balanced_p4/`
- `data/trainability/balanced_v2_1_eligibility/`

冻结协议版本和输入 hash 已保存在上述 run/eligibility manifest。已失去执行入口的 v1-v2.1
registry 与 validation YAML 不再保留在活动配置目录。

## 4. Data-conditioning

### 4.1 判据

每个 corpus 各训练一个模型并交叉评估 validation。对候选 A、B，目标 margin 为
`loss(train=A, eval=B) - loss(train=B, eval=B)`；只有外部已知更优的 A 在 B validation 上形成
稳定负 margin，才支持 `A -> B`。该判据没有运行下游 benchmark。

### 4.2 实验链

| 实验 | 规模与预算 | 结果 |
|---|---|---|
| Tiny proxy | 约 4.27M 参数、2.46M token | 方向随 horizon 翻转，未形成稳定边 |
| FineWeb-Edu 阈值与规模 | 4M-124M，固定 20M token | 没有恢复已知质量方向；放大模型同时造成 token/parameter 下降 |
| Falcon+CC 固定比例 | 3.90M / 21.55M / 66.70M，5 token/parameter | QC20 优于 raw 的外部真值下，三个规模均为 0 条正确边；目标 margin 从 +0.156 增至约 +0.26 |
| 官方近分布复核 | DataDecide FW2/FW3，官方 20M / 60M / 150M、约 100 token/parameter、3 seeds | 20M seed 不稳定；60M/150M 三 seed 均显著反向；正确显著边为 0 |

固定比例主实验训练 6 个模型、约 9.215 亿 token，模型 wall 累计 6,769 秒。FP32/BF16、五个
horizon 和三个规模方向一致，排除了精度变化与固定 token 导致比例下降这两个主要混淆。目标方向
远离 0 且随规模没有改善，因此没有追加 seed 或继续 125M。

官方近分布复核使用同一 DCLM top-7% 池上的 FW2/FW3。1B OLMES 外部真值为 FW2 优于 FW3
1.823pp。conditioning 结果为：

| 官方模型 | FW2 -> FW3 margin [95% CI] | 结论 |
|---:|---:|---|
| 20M final | +0.004 [-0.050, +0.087] | 三 seed 方向不稳定 |
| 60M final | +0.069 [+0.048, +0.086] | 显著反向 |
| 150M final | +0.070 [+0.059, +0.079] | 显著反向 |

60M/150M 在两个交叉方向都显示各模型的“主场优势”。这说明 cross loss 识别的是分布专化，而非
最终能力排序。该失败发生在官方架构、约 100 token/parameter 和三个 seed 下，不能用继续放宽
本项目 tiny proxy 的条件解释。

### 4.3 Conditioning 决策

Data-conditioning 不进入生产质量排名，也不再保留为辅助覆盖分数。若业务需要预测数据 recipe
的最终能力，应使用经过历史回测的目标 benchmark 小模型，并允许在 seed 不稳定或规模翻转时输出
“不确定”；不能用 cross loss 补出排序。

原始产物：

- `outputs/stage11_conditioning_scale/`
- `outputs/stage11_conditioning_threshold/`
- `outputs/stage11_conditioning_fixed_ratio/`
- `outputs/datadecide_official_conditioning/combined_summary.json`
- 冻结配置目录：[`protocols/stage11/`](protocols/stage11/)，其中保留
  `stage11_conditioning_*.yaml` 和 `stage11_datadecide_official.yaml`

## 5. 生产边界与保留策略

从本报告起：

1. 生产编排不得调用 Stage 11 `probe`。
2. summary 聚合不得消费 `scaling_gain`、Balanced edge 或 conditioning edge。
3. 不得按逐文档 gain/loss 删除、排序或加权数据。
4. 不得把任何 Stage 11 数值命名为“数据质量分”或最终 benchmark 能力预测。
5. `data/`、`outputs/` 和冻结配置只用于审计、复核历史结论及防止重复投入。
6. 新的训练型代理必须以新协议重新提出真实故障场景、外部真值、廉价 baseline 和前瞻停止门，
   不能通过降低本轮 rho、CI、兼容性或硬门标准恢复上线资格。

## 6. 未验证范围

- Anchor 与 Balanced 未验证 7B/70B 外推，也未验证最终 benchmark 能力。
- Data-conditioning 的失败不构成对所有可能数据 pair 的数学不可能性证明；它证明的是当前短周期
  判据在多个自然和近分布对照上未通过上线校准。
- 合成控制不等于真实生产故障分布。
- 公开预分词流缺少完整源 doc ID，部分跨 recipe 训练重叠无法严格排除。
- Stage 12 DataDecide 的 benchmark-based 方法需要独立评估，不能由本文结论代替。

最终结论：Stage 11 的三种方法均无可实施的生产或辅助决策用途，停止继续放宽条件、扩大模型或
补跑未启动阶段。历史产物保留，活动方法退役。
