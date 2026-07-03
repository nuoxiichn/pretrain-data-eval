#!/usr/bin/env bash
# toxicity 真实数据批跑：reservoir 抽样 (seed=42, n=5000)
#
# 前置：GPU 空闲。
# 用法：bash scripts/toxicity_batch.sh
#
# 输出：
#   outputs/stage2/ufw_en_l3/toxicity/{per_doc.jsonl,summary.json}
#   outputs/stage2/ufw_zh_l3/toxicity/{per_doc.jsonl,summary.json}

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT"

EN_OUT="outputs/stage2/ufw_en_l3/toxicity"
ZH_OUT="outputs/stage2/ufw_zh_l3/toxicity"
LOG_DIR="outputs/stage2"
TS="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$EN_OUT" "$ZH_OUT" "$LOG_DIR"

# 断点续跑：summary.json 存在则跳过
run_one() {
    local dataset="$1"
    local input_dir="$2"
    local out_dir="$3"
    local log="$LOG_DIR/toxicity_${dataset}_${TS}.log"

    if [[ -s "$out_dir/summary.json" ]]; then
        echo "[skip] $dataset: $out_dir/summary.json 已存在"
        return
    fi
    echo "[run]  $dataset → $out_dir   (log: $log)"
    python stages/safety/run.py toxicity \
        --input "$input_dir" \
        --dataset "$dataset" \
        --config configs/stage2.yaml \
        --output-dir "$out_dir" \
        --max-docs 5000 --seed 42 --sample-mode random \
        2>&1 | tee "$log"
}

run_one ufw_en_l3 /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3 "$EN_OUT"
run_one ufw_zh_l3 /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_zh_l3 "$ZH_OUT"

echo
echo "[done] toxicity 5K 完成。关键产物："
echo "  $EN_OUT/summary.json"
echo "  $ZH_OUT/summary.json"
