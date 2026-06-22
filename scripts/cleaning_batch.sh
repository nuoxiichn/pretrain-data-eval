#!/usr/bin/env bash
# Stage 3 cleaning 全量逐文件扫描（5 个子命令），支持断点续跑 + 文件级并行
# 用法: bash scripts/cleaning_batch.sh en|zh [subcmd] [parallel]
#   subcmd   可选: extraction|langid|glotlid|langcross|quality；不指定则跑全部 5 个
#   parallel 可选: 并行文件数，默认 1（串行）。CPU 重的子命令（extraction/quality）建议 8-16
set -uo pipefail

LANG=${1:?用法: cleaning_batch.sh en|zh [subcmd] [parallel]}
SUBCMD=${2:-all}
PARALLEL=${3:-1}

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

if [[ "$SUBCMD" == "all" ]]; then
    SUBCMDS=(extraction langid glotlid langcross quality)
else
    SUBCMDS=("$SUBCMD")
fi

OUT_BASE="$CODE/outputs/stage3/$DATASET"
LOG="$CODE/outputs/stage3/cleaning_${DATASET}.log"
mkdir -p "$OUT_BASE"

echo "=== Cleaning $DATASET 启动 $(date) subcmds=${SUBCMDS[*]} parallel=$PARALLEL ===" | tee -a "$LOG"

# 单文件 worker（被 xargs 调用）：参数 = cmd parquet_path data_root out_base log
run_one() {
    local cmd="$1" f="$2" data="$3" outbase="$4" dataset="$5" log="$6"
    local rel="${f#$data/}"
    rel="${rel%.parquet}"
    local fname="${rel//\//__}"
    local outdir="$outbase/$cmd/$fname"

    if [[ -f "$outdir/summary.json" ]]; then
        return 0
    fi

    mkdir -p "$outdir"
    echo "[$cmd][RUN ] $fname  $(date +%H:%M:%S)" >> "$log"

    if PYTHONPATH=. python stages/cleaning/run.py "$cmd" \
        --input "$f" \
        --dataset "$dataset" \
        --config configs/stage3.yaml \
        --output-dir "$outdir" \
        >> "$log" 2>&1; then
        echo "[$cmd][OK  ] $fname" >> "$log"
    else
        echo "[$cmd][FAIL] $fname  exit=$?" >> "$log"
    fi
}
export -f run_one

for cmd in "${SUBCMDS[@]}"; do
    echo "--- [$cmd] 开始 (parallel=$PARALLEL) ---" | tee -a "$LOG"
    t0=$(date +%s)
    find "$DATA" -name "*.parquet" | sort | \
        xargs -P "$PARALLEL" -I{} bash -c \
            'run_one "$@"' _ "$cmd" "{}" "$DATA" "$OUT_BASE" "$DATASET" "$LOG"
    t1=$(date +%s)
    pass=$(find "$OUT_BASE/$cmd" -mindepth 2 -name summary.json | wc -l)
    total=$(find "$DATA" -name "*.parquet" | wc -l)
    echo "--- [$cmd] 结束: 已完成 $pass/$total  耗时 $((t1-t0))s ---" | tee -a "$LOG"
done

echo "=== Cleaning $DATASET 结束 $(date) ===" | tee -a "$LOG"
