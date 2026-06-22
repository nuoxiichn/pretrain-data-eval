#!/usr/bin/env bash
# Dedup 全量跑：exact + ngram 逐文件；minhash 跨文件样本
# 用法: bash scripts/dedup_batch.sh en|zh [minhash_sample_size]
#   minhash_sample_size 默认 100000
set -uo pipefail

LANG=${1:?用法: dedup_batch.sh en|zh [sample_size]}
MINHASH_SAMPLE=${2:-100000}

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

OUT_BASE="$CODE/outputs/stage4/$DATASET"
LOG="$CODE/outputs/stage4/dedup_${DATASET}.log"
mkdir -p "$OUT_BASE"

echo "=== Dedup $DATASET 启动 $(date) ===" | tee -a "$LOG"
echo "  exact + ngram: 逐文件 | minhash: 跨文件样本 $MINHASH_SAMPLE 条" | tee -a "$LOG"

# ── 1. exact 逐文件 ───────────────────────────────────────────────────────────
echo "--- [exact] 开始 ---" | tee -a "$LOG"
pass=0; fail=0; skip=0

while IFS= read -r f; do
    rel="${f#$DATA/}"; rel="${rel%.parquet}"; fname="${rel//\//__}"
    outdir="$OUT_BASE/exact/$fname"

    if [[ -f "$outdir/summary.json" ]]; then
        ((skip+=1)) || true; continue
    fi

    mkdir -p "$outdir"
    echo "[exact][RUN ] $fname  $(date +%H:%M:%S)" | tee -a "$LOG"

    if PYTHONPATH=. python stages/dedup/run.py exact \
        --input "$f" \
        --dataset "$DATASET" \
        --output-dir "$outdir" \
        >> "$LOG" 2>&1; then
        echo "[exact][OK  ] $fname" | tee -a "$LOG"
        ((pass+=1)) || true
    else
        echo "[exact][FAIL] $fname  exit=$?" | tee -a "$LOG"
        ((fail+=1)) || true
    fi
done < <(find "$DATA" -name "*.parquet" | sort)

echo "--- [exact] 结束: pass=$pass fail=$fail skip=$skip ---" | tee -a "$LOG"

# ── 2. ngram 逐文件 ───────────────────────────────────────────────────────────
echo "--- [ngram] 开始 ---" | tee -a "$LOG"
pass=0; fail=0; skip=0

while IFS= read -r f; do
    rel="${f#$DATA/}"; rel="${rel%.parquet}"; fname="${rel//\//__}"
    outdir="$OUT_BASE/ngram/$fname"

    if [[ -f "$outdir/summary.json" ]]; then
        ((skip+=1)) || true; continue
    fi

    mkdir -p "$outdir"
    echo "[ngram][RUN ] $fname  $(date +%H:%M:%S)" | tee -a "$LOG"

    if PYTHONPATH=. python stages/dedup/run.py ngram \
        --input "$f" \
        --dataset "$DATASET" \
        --output-dir "$outdir" \
        >> "$LOG" 2>&1; then
        echo "[ngram][OK  ] $fname" | tee -a "$LOG"
        ((pass+=1)) || true
    else
        echo "[ngram][FAIL] $fname  exit=$?" | tee -a "$LOG"
        ((fail+=1)) || true
    fi
done < <(find "$DATA" -name "*.parquet" | sort)

echo "--- [ngram] 结束: pass=$pass fail=$fail skip=$skip ---" | tee -a "$LOG"

# ── 3. minhash 跨文件样本（跑一次整目录，取前 N 条）────────────────────────────
echo "--- [minhash] 开始 跨文件样本=$MINHASH_SAMPLE ---" | tee -a "$LOG"
outdir="$OUT_BASE/minhash/crossfile_sample${MINHASH_SAMPLE}"

if [[ -f "$outdir/summary.json" ]]; then
    echo "[minhash][SKIP] 样本已存在" | tee -a "$LOG"
else
    mkdir -p "$outdir"
    if PYTHONPATH=. python stages/dedup/run.py minhash \
        --input "$DATA" \
        --dataset "$DATASET" \
        --output-dir "$outdir" \
        --max-docs "$MINHASH_SAMPLE" \
        >> "$LOG" 2>&1; then
        echo "[minhash][OK  ] 跨文件样本 $MINHASH_SAMPLE 条" | tee -a "$LOG"
    else
        echo "[minhash][FAIL] exit=$?" | tee -a "$LOG"
    fi
fi

echo "=== Dedup $DATASET 结束 $(date) ===" | tee -a "$LOG"
