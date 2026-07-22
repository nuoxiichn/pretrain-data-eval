# 指标目录

本页是当前维度和结论边界的权威入口。具体 CLI 参数和外部依赖见各 Stage README。

## Stage 1：来源与规模

| 子命令 | 输出重点 | 能支持的结论 | 主要限制 |
|---|---|---|---|
| `stats` | 文档/字符/token 数、长度桶、来源/语言/域名/时间分布 | 数据规模与元数据完整性 | 声明语言和来源不等于真实内容；token 结果绑定 tokenizer |
| `license` | 许可证命中、SPDX 类型 | 文本内可识别的许可证候选 | 无法推断缺失声明的版权状态；大规模运行昂贵 |

## Stage 2：安全与隐私

| 子命令 | 方法 | 能支持的结论 | 主要限制 |
|---|---|---|---|
| `pii` | Presidio + 占位符/保留值过滤；代码模式使用精简 recognizer | 召回待复核的 PII 候选 | 语言、实体类型和上下文影响 precision/recall；命中率不是泄漏率 |
| `secrets` | Gitleaks | 召回 token、key、私钥候选 | 示例 key 和高熵字符串会假阳；不验证凭据是否有效 |
| `toxicity` | 文档切块，XLM-R 召回，Qwen 判 benign/discuss/promote | 区分客观讨论与推动有害内容的候选 | recall 受首阶段模型限制，Judge 受 prompt/模型影响，必须抽检 |

## Stage 3：文本完整性与语言

| 子命令 | 输出重点 | 主要限制 |
|---|---|---|
| `extraction` | HTML、乱码、模板尾部、短残桩和加权风险 | 词表和位置权重是启发式，领域模板可能假阳 |
| `langid` | lid.176 语言、置信度、声明一致性 | 低资源语种、短文本和混合语种更难 |
| `glotlid` | 语种与 script 细分类 | 模型标签粒度和训练覆盖决定解释范围 |
| `langcross` | 两模型归一后的分歧 | 两模型一致不等于一定正确，可能共享偏差 |
| `quality` | Gopher/C4 规则失败原因 | 测规则符合度，不等于总体训练价值；公式、代码、短 QA 需单独解释 |

## Stage 4：重复度

| 子命令 | 检测对象 | 规模边界和限制 |
|---|---|---|
| `exact` | 规范化后的文档/段落 hash | 单次进程内全局集合占内存；规范化之外的变化不会命中 |
| `minhash` | 词或字符 shingle 的近重复 | LSH 是概率方法；阈值、语言分词和 hot bucket 会影响 recall |
| `ngram` | 段落 N-gram 重叠 | 短文本和无段落结构数据不可直接横向比较 |
| `semdedup` | embedding 余弦相似聚类 | 模型、截断和阈值会混淆同主题与重复；建议抽样人工核验 |
| DataTrove MinHash | 跨 shard 分布式近重复 | 使用独立环境；参数必须与报告一同保存 |

## Stage 5：评测集污染

`exact`、`near`、`embed` 和 `cascade` 将待评数据与显式 benchmark 注册表比较；代码数据另有
`code-near` 和 `code-ast`。Cascade 依次使用 exact、char-MinHash 和 BGE-M3，召回结果可交给
LLM Judge 复审。

这些方法只能检测注册 benchmark 中可见的字面、局部改写或语义近似。翻译、深度改写、
只泄露答案或 benchmark 未纳入注册表都会造成漏检；embedding 和 AST 对短题/短函数容易
假阳。零命中不等于无污染。

## Stage 7：合成文本信号

`binoculars` 比较共享 tokenizer 的 base/instruct 模型行为。论文阈值不能迁移到任意模型对、
语言和生成模型。未经人写/机写真值集校准时，只报告分数分布，不输出合格结论。

## Stage 8：专项能力

- `parsability` 使用 tree-sitter 报告代码 AST 错误率。预处理宏、grammar 差异和代码片段
  不完整都会影响结果；对自然语言无意义。
- `stem` 使用 EAI-Distill 输出学科和 reasoning depth。它适合做相对分布画像，不能把
  `high_difficulty` 直接解释为训练效用，跨语言结果要先检查模型校准。

## Stage 9：训练配置

`config-audit` 检查 Megatron 风格的 `reset_position_ids`、`reset_attention_mask` 和
`eod_mask_loss`。它审计配置文本，不读取训练数据，也不证明训练代码实际采用了该配置。

## Stage 10：Tokenization

输出 token/char fertility、UNK、代码块和 LaTeX 块的 token 膨胀。所有结果只对指定 tokenizer
成立；不同 script 的字符单位不可简单解释为语言质量。生产建议与 Stage 1 合并扫描。

## Stage 11：微型训练代理

Stage 11 已退役。Anchor-relative、Balanced-pool scaling gain 和 data-conditioning 均未通过生产或
辅助决策校准，生产编排不得调用 `probe`，汇总不得消费其边或逐文档分数。实现、冻结配置、数据
和输出仅为历史实验复核保留。

失败不只是统计效力不足：重复、模板和截断存在反向或无响应，cross-corpus loss 主要识别分布
专化，而最稳定的严重词序控制缺少真实生产应用场景。完整协议、数值和停止决定见
[Stage 11 最终实验与决策](../reports/stage11_final_report.md)。

## 不提供的结论

当前链路不提供跨维度综合分、生产规模模型或下游 benchmark 收益预测、法律合规意见、凭据
有效性验证，也不声称
一次抽样能够证明整库“安全”或“无污染”。
