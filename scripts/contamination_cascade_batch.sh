#!/usr/bin/env bash
# Stage 5 — cascade 跑批（L1 exact + L2 MinHash + L3 BGE-m3 embedding）
# 必须先跑过：python stages/contamination/bench_index.py build --config configs/stage5.yaml
#
# **不全量**：每文件随机抽样 N docs，跨全部文件汇总污染率
# （全量 cascade 单文件 ~15min × 596 文件 = 150h 不切实际；采样 500/file × 596 ≈ 5-7h）
#
# 用法:
#   bash scripts/contamination_cascade_batch.sh en|zh [docs_per_file] [max_files]
#     docs_per_file: 默认 500（每文件抽样 N docs）
#     max_files:     默认 0（全量文件）
set -uo pipefail

LANG=${1:?用法: contamination_cascade_batch.sh en|zh [docs_per_file] [max_files]}
DOCS_PER_FILE=${2:-500}
MAX_FILES=${3:-0}

CODE=/mnt/public/code/chennuoxi/pretrain-data-eval
cd "$CODE"

CONFIG=configs/stage5.yaml

case "$LANG" in
  en)
    DATA=/mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3
    DATASET=ufw_en_l3_v3 ;;
  zh)
    DATA=/mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_zh_l3
    DATASET=ufw_zh_l3_v3 ;;
  *)
    echo "未知语言: $LANG"; exit 1 ;;
esac

OUT_BASE="$CODE/outputs/stage5/$DATASET/cascade_sample${DOCS_PER_FILE}"
LOG="$CODE/outputs/stage5/contamination_${DATASET}_cascade_sample${DOCS_PER_FILE}.log"
mkdir -p "$OUT_BASE"

echo "=== Cascade $DATASET 启动 $(date) docs/file=$DOCS_PER_FILE max_files=$MAX_FILES ===" | tee -a "$LOG"

pass=0; fail=0; skip=0; count=0

while IFS= read -r f; do
    if [[ $MAX_FILES -gt 0 && $count -ge $MAX_FILES ]]; then
        echo "[STOP] 达到 max_files=$MAX_FILES" | tee -a "$LOG"
        break
    fi
    count=$((count + 1))

    rel="${f#$DATA/}"; rel="${rel%.parquet}"; fname="${rel//\//__}"
    outdir="$OUT_BASE/$fname"

    if [[ -f "$outdir/summary.json" ]]; then
        ((skip+=1)) || true; continue
    fi

    mkdir -p "$outdir"
    echo "[RUN ] $fname  $(date +%H:%M:%S)" | tee -a "$LOG"

    # head 模式抽样：避免 reservoir sampling 在大 parquet 上扫全文件
    if PYTHONPATH=. python stages/contamination/run.py cascade \
        --input "$f" \
        --dataset "$DATASET" \
        --config "$CONFIG" \
        --output-dir "$outdir" \
        --max-docs "$DOCS_PER_FILE" \
        --sample-mode head \
        >> "$LOG" 2>&1; then
        echo "[OK  ] $fname" | tee -a "$LOG"
        ((pass+=1)) || true
    else
        echo "[FAIL] $fname  exit=$?" | tee -a "$LOG"
        ((fail+=1)) || true
    fi
done < <(find "$DATA" -name "*.parquet" | sort)

echo "=== Cascade $DATASET 结束: pass=$pass fail=$fail skip=$skip  $(date) ===" | tee -a "$LOG"

