#!/usr/bin/env bash
# PII 全量逐文件扫描，支持断点续跑
# 用法: bash scripts/pii_batch.sh en|zh
set -uo pipefail

LANG=${1:?用法: pii_batch.sh en|zh}

CODE=/mnt/public/code/chennuoxi/pretrain-data-eval
cd "$CODE"

case "$LANG" in
  en)
    DATA=/mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3
    DATASET=ufw_en_l3
    LANGUAGE=en
    SPACY=en_core_web_lg
    ;;
  zh)
    DATA=/mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_zh_l3
    DATASET=ufw_zh_l3
    LANGUAGE=zh
    SPACY=zh_core_web_lg
    ;;
  *)
    echo "未知语言: $LANG（只支持 en/zh）"; exit 1 ;;
esac

OUT_BASE="$CODE/outputs/stage2/$DATASET/pii"
LOG="$CODE/outputs/stage2/pii_${DATASET}.log"
mkdir -p "$OUT_BASE"

echo "=== PII $DATASET ($LANGUAGE) 启动 $(date) ===" | tee -a "$LOG"

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

    if PYTHONPATH=. python stages/safety/run.py pii \
        --input "$f" \
        --dataset "$DATASET" \
        --output-dir "$outdir" \
        --language "$LANGUAGE" \
        --spacy-model "$SPACY" \
        >> "$LOG" 2>&1; then
        echo "[OK  ] $fname" | tee -a "$LOG"
        ((pass+=1)) || true
    else
        echo "[FAIL] $fname  exit=$?" | tee -a "$LOG"
        ((fail+=1)) || true
    fi
done < <(find "$DATA" -name "*.parquet" | sort)

echo "=== PII $DATASET 结束: pass=$pass fail=$fail skip=$skip  $(date) ===" | tee -a "$LOG"
