#!/usr/bin/env bash
# PII 采样重跑（去假阳性版）：从多个文件各采样 --max-docs，2h 内跑完
# 用法: bash scripts/archive/pii_legacy_rerun.sh
set -uo pipefail

CODE=/mnt/public/code/chennuoxi/pretrain-data-eval
cd "$CODE"

MAX_DOCS=12000

EN_DATA=/mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3
ZH_DATA=/mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_zh_l3

EN_OUT="$CODE/outputs/stage2/ufw_en_l3/pii_v2"
ZH_OUT="$CODE/outputs/stage2/ufw_zh_l3/pii_v2"
LOG="$CODE/outputs/stage2/pii_rerun.log"
BLANK_EN=/tmp/spacy_blank_en
mkdir -p "$EN_OUT" "$ZH_OUT"

# en_core_web_lg 不可用（网络隔离），用 blank EN 模型（regex 识别器正常工作）
if [ ! -d "$BLANK_EN" ]; then
    python -c "import spacy; spacy.blank('en').to_disk('$BLANK_EN')"
fi

echo "=== PII rerun (no false positives) 启动 $(date) ===" | tee "$LOG"
echo "  max_docs=$MAX_DOCS per file" | tee -a "$LOG"

run_pii() {
    local input="$1" dataset="$2" language="$3" spacy="$4" outdir="$5"
    local rel="${input##*/}"; rel="${rel%.parquet}"
    local odir="$outdir/$rel"

    if [[ -f "$odir/summary.json" ]]; then
        echo "[SKIP] $rel" | tee -a "$LOG"
        return 0
    fi

    mkdir -p "$odir"
    echo "[RUN ] $dataset/$rel  $(date +%H:%M:%S)" | tee -a "$LOG"

    if PYTHONPATH=. python stages/safety/run.py pii \
        --input "$input" \
        --dataset "$dataset" \
        --output-dir "$odir" \
        --language "$language" \
        --spacy-model "$spacy" \
        --max-docs "$MAX_DOCS" \
        >> "$LOG" 2>&1; then
        echo "[OK  ] $rel" | tee -a "$LOG"
    else
        echo "[FAIL] $rel  exit=$?" | tee -a "$LOG"
    fi
}

# EN: 2 multi_style + 2 qa
EN_FILES=(
    "$EN_DATA/multi_style/part-00000-f36f5a53-4a77-434b-b9bc-67ed69b93fe2-c000.snappy.parquet"
    "$EN_DATA/multi_style/part-00100-f36f5a53-4a77-434b-b9bc-67ed69b93fe2-c000.snappy.parquet"
    "$EN_DATA/qa/part-00000-37dc9f21-f87f-4f43-8dd2-134424f1537a-c000.snappy.parquet"
    "$EN_DATA/qa/part-00100-37dc9f21-f87f-4f43-8dd2-134424f1537a-c000.snappy.parquet"
)

# ZH: 2 multi_style + 2 qa
ZH_FILES=(
    "$ZH_DATA/multi_style/part-00000-c13afd3b-b5fb-4acd-97dc-e045a844c126-c000.snappy.parquet"
    "$ZH_DATA/multi_style/part-00100-c13afd3b-b5fb-4acd-97dc-e045a844c126-c000.snappy.parquet"
    "$ZH_DATA/qa/part-00000-100cee91-ea2e-4268-9b98-1403cd5d9d11-c000.snappy.parquet"
    "$ZH_DATA/qa/part-00100-100cee91-ea2e-4268-9b98-1403cd5d9d11-c000.snappy.parquet"
)

echo "--- EN (4 files × $MAX_DOCS docs) ---" | tee -a "$LOG"
for f in "${EN_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        run_pii "$f" ufw_en_l3 en "$BLANK_EN" "$EN_OUT"
    else
        echo "[WARN] $f not found, skip" | tee -a "$LOG"
    fi
done

echo "--- ZH (4 files × $MAX_DOCS docs) ---" | tee -a "$LOG"
for f in "${ZH_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        run_pii "$f" ufw_zh_l3 zh zh_core_web_sm "$ZH_OUT"
    else
        echo "[WARN] $f not found, skip" | tee -a "$LOG"
    fi
done

echo "=== PII rerun 结束 $(date) ===" | tee -a "$LOG"
