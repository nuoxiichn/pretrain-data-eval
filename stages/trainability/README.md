# Stage 11：微型训练代理（已退役）

Stage 11 的 Anchor-relative scaling gain、Balanced-pool scaling gain 和 data-conditioning 均未通过
生产或辅助决策校准。生产编排不得调用 `probe`，其他 summary 不得消费 Stage 11 排序边或逐文档
分数。

统一实验记录、关键数值、失败原因和停止决定见
[Stage 11 最终实验与决策](../../docs/reports/stage11_final_report.md)。该报告取代此前分散的 Anchor、
Balanced 和 conditioning 报告。

## 当前边界

- `stages/trainability/` 中保留的通用训练代码只用于复核历史实验，不是当前生产能力。
- `configs/stage11*.yaml` 是冻结协议资产；接口存在不代表方法获得上线资格。
- `data/trainability/` 和 `outputs/` 下的历史产物保持只读，不应重新聚合成质量分。
- 不得把 gain 命名为通用质量分，不得跨 Anchor、tokenizer 或 run 比较绝对值。
- 不得通过降低 rho、CI、兼容性 gate 或前瞻硬门恢复任何方法的上线状态。

## 保留产物

| 方法 | 数据/输出入口 |
|---|---|
| Anchor-relative | `data/trainability/anchor_calibration_v1/`、`outputs/stage11_anchor_calibration_v1/` |
| Balanced | `data/trainability/balanced_*`、`outputs/stage11_balanced_validation_*` |
| Data-conditioning | `outputs/stage11_conditioning_*`、`outputs/datadecide_official_conditioning/` |

Stage 12 DataDecide 是独立的 benchmark-based 方法，不属于本 Stage，也不受这里的代码退役影响。
