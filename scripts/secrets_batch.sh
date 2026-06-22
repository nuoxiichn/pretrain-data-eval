#!/usr/bin/env bash
# Gitleaks 全量逐文件扫描，支持断点续跑
# 用法: bash scripts/secrets_batch.sh en|zh
set -uo pipefail

LANG=${1:?用法: secrets_batch.sh en|zh}

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

OUT_BASE="$CODE/outputs/stage2/$DATASET/secrets"
LOG="$CODE/outputs/stage2/secrets_${DATASET}.log"
mkdir -p "$OUT_BASE"

echo "=== Secrets $DATASET 启动 $(date) ===" | tee -a "$LOG"

pass=0; fail=0; skip=0

while IFS= read -r f; do
    rel="${f#$DATA/}"; rel="${rel%.parquet}"; fname="${rel//\//__}"
    outdir="$OUT_BASE/$fname"

    if [[ -f "$outdir/summary.json" ]]; then
        echo "[SKIP] $fname" | tee -a "$LOG"
        ((skip+=1)) || true
        continue
    fi

    mkdir -p "$outdir"
    echo "[RUN ] $fname  $(date +%H:%M:%S)" | tee -a "$LOG"

    if PYTHONPATH=. python stages/safety/run.py secrets \
        --input "$f" \
        --dataset "$DATASET" \
        --output-dir "$outdir" \
        >> "$LOG" 2>&1; then
        echo "[OK  ] $fname" | tee -a "$LOG"
        ((pass+=1)) || true
    else
        echo "[FAIL] $fname  exit=$?" | tee -a "$LOG"
        ((fail+=1)) || true
    fi
done < <(find "$DATA" -name "*.parquet" | sort)

echo "=== Secrets $DATASET 结束: pass=$pass fail=$fail skip=$skip  $(date) ===" | tee -a "$LOG"
