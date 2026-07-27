#!/usr/bin/env bash
# 控制组对照验证：control_positive vs control_negative 全 CPU 维度跑批
# 用法: bash scripts/control_batch.sh
# 输出: outputs/control/{positive,negative}/<dim>/summary.json  (git 忽略)
# 断点续跑：summary.json 存在则跳过。并发上限 MAX_PAR。

set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT"

CONFIG=configs/control.yaml
OUT=outputs/control
LOGDIR="$OUT/logs"; mkdir -p "$LOGDIR"

declare -A INPUT=(
  [positive]=data/control_positive/openwebtext_1000.jsonl
  [negative]=data/control_negative/raw_cc_1000.jsonl
)

# 每项："<stage 目录> <子命令>"；子命令名即输出维度目录名（跨 stage 唯一）
JOBS=(
  "source_audit stats"
  "safety pii"
  "safety secrets"
  "cleaning extraction"
  "cleaning quality"
  "dedup exact"
  "dedup ngram"
  "dedup minhash"
  "tokenization tokenize"
)

MAX_PAR=6

run_one() {
  local cls="$1" stage="$2" cmd="$3"
  local outdir="$OUT/$cls/$cmd"
  local log="$LOGDIR/${cls}_${cmd}.log"
  if [[ -s "$outdir/summary.json" ]]; then echo "[skip] $cls/$cmd"; return; fi
  echo "[run]  $cls/$cmd  (log: $log)"
  HF_HUB_OFFLINE=1 python "stages/$stage/run.py" "$cmd" \
    --input "${INPUT[$cls]}" --dataset "control_$cls" \
    --config "$CONFIG" --input-format jsonl \
    --output-dir "$outdir" > "$log" 2>&1 \
    && echo "[ok]   $cls/$cmd" || echo "[FAIL] $cls/$cmd (见 $log)"
}

for cls in positive negative; do
  for j in "${JOBS[@]}"; do
    read -r stage cmd <<< "$j"
    run_one "$cls" "$stage" "$cmd" &
    while (( $(jobs -r | wc -l) >= MAX_PAR )); do wait -n; done
  done
done
wait
echo "[done] 全部对照跑批结束 → $OUT/{positive,negative}/"
