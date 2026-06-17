#!/usr/bin/env bash
# Stage 1 stats 全量逐文件扫描，支持断点续跑
# 用法: bash scripts/stats_batch.sh en|zh
set -uo pipefail

LANG=${1:?用法: stats_batch.sh en|zh}

CODE=/mnt/public/code/chennuoxi/pretrain-data-eval
cd "$CODE"

case "$LANG" in
  en)
    DATA=/mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3
    DATASET=ufw_en_l3 ;;
  zh)
    DATA=/mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_zh_l3
    DATASET=ufw_zh_l3 ;;
  *)
    echo "未知语言: $LANG"; exit 1 ;;
esac

OUT_BASE="$CODE/outputs/stage1/$DATASET/stats"
LOG="$CODE/outputs/stage1/stats_${DATASET}.log"
mkdir -p "$OUT_BASE"

echo "=== Stats $DATASET 启动 $(date) ===" | tee -a "$LOG"

pass=0; fail=0; skip=0

while IFS= read -r f; do
    rel="${f#$DATA/}"; rel="${rel%.parquet}"; fname="${rel//\//__}"
    outdir="$OUT_BASE/$fname"

    if [[ -f "$outdir/summary.json" ]]; then
        ((skip+=1)) || true; continue
    fi

    mkdir -p "$outdir"
    echo "[RUN ] $fname  $(date +%H:%M:%S)" | tee -a "$LOG"

    if PYTHONPATH=. python stages/source_audit/run.py stats \
        --input "$f" \
        --dataset "$DATASET" \
        --config configs/stage1.yaml \
        --output-dir "$outdir" \
        >> "$LOG" 2>&1; then
        echo "[OK  ] $fname" | tee -a "$LOG"
        ((pass+=1)) || true
    else
        echo "[FAIL] $fname  exit=$?" | tee -a "$LOG"
        ((fail+=1)) || true
    fi
done < <(find "$DATA" -name "*.parquet" | sort)

echo "=== Stats $DATASET 结束: pass=$pass fail=$fail skip=$skip  $(date) ===" | tee -a "$LOG"
