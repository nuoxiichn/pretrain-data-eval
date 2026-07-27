# 归档脚本

本目录脚本只用于回查旧的本地产物路径或重建历史图表，不是当前运行入口，也不纳入 CI。

- `generate_legacy_charts.py`：读取旧报告和已否决 Stage 6 产物生成图表；
- `pii_legacy_rerun.sh`：旧 PII 输出目录的定点重跑；
- `extract_legacy_hit_examples.py`：从旧 PII/secret 产物回查敏感片段。

当前正式报告位于 `docs/reports/`，当前 Stage 命令以根 README 和各 Stage README 为准。
