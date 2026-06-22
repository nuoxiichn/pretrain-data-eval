#!/usr/bin/env bash
# Stage 1 stats + Stage 10 tokenize 全量逐文件扫描（共用一次 tokenizer pass）
# 支持断点续跑 + mkdir 原子锁（多进程并行安全）
# 用法:
#   bash scripts/stats_tokenize_batch.sh en|zh                # 单进程
#   for i in 1 2 3 4; do
#     nohup bash scripts/stats_tokenize_batch.sh zh w$i \
#       > outputs/stage1/stats_tokenize_ufw_zh_l3.w$i.log 2>&1 &
#   done                                                      # 4 并行
#
# 锁机制: 处理每个文件前 `mkdir <stage1_outdir>/.lock` 抢占；成功后清掉锁。
#   - 锁存在 -> 其他 worker 在做，本进程跳过该文件
#   - 两边 summary.json 都存在 -> 已完成，本进程跳过
#   - Python 进程崩溃会留下孤立锁，需手动清:
#       find outputs/stage1/<dataset>/stats -type d -name .lock -mmin +30 -exec rmdir {} +
set -uo pipefail

LANG=${1:?用法: stats_tokenize_batch.sh en|zh [worker-tag]}
WORKER=${2:-w0}

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

S1_BASE="$CODE/outputs/stage1/$DATASET/stats"
S10_BASE="$CODE/outputs/stage10/$DATASET/tokenize"
LOG="$CODE/outputs/stage1/stats_tokenize_${DATASET}.${WORKER}.log"
mkdir -p "$S1_BASE" "$S10_BASE"

echo "=== [$WORKER] Stats+Tokenize $DATASET 启动 $(date) PID=$$ ===" | tee -a "$LOG"

pass=0; fail=0; skip_done=0; skip_lock=0

while IFS= read -r f; do
    rel="${f#$DATA/}"; rel="${rel%.parquet}"; fname="${rel//\//__}"
    s1_dir="$S1_BASE/$fname"
    s10_dir="$S10_BASE/$fname"
    lock_dir="$s1_dir/.lock"

    # 已完成 -> 跳过
    if [[ -f "$s1_dir/summary.json" && -f "$s10_dir/summary.json" ]]; then
        ((skip_done+=1)) || true; continue
    fi

    mkdir -p "$s1_dir" "$s10_dir"

    # 原子锁：mkdir 失败 = 另一 worker 已占
    if ! mkdir "$lock_dir" 2>/dev/null; then
        ((skip_lock+=1)) || true; continue
    fi
    # 锁拿到后，再确认一遍是否别人刚跑完了（双 check 避免锁释放-完成的竞态：
    # 极端少见，但成本低）
    if [[ -f "$s1_dir/summary.json" && -f "$s10_dir/summary.json" ]]; then
        rmdir "$lock_dir"
        ((skip_done+=1)) || true; continue
    fi

    echo "[$WORKER RUN ] $fname  $(date +%H:%M:%S)" | tee -a "$LOG"

    if PYTHONPATH=. python stages/source_audit/run.py stats \
        --input "$f" \
        --dataset "$DATASET" \
        --config configs/stage1.yaml \
        --output-dir "$s1_dir" \
        --coalesce-stage10 \
        --stage10-config configs/stage10.yaml \
        --stage10-output-dir "$s10_dir" \
        >> "$LOG" 2>&1; then
        echo "[$WORKER OK  ] $fname" | tee -a "$LOG"
        ((pass+=1)) || true
    else
        echo "[$WORKER FAIL] $fname  exit=$?" | tee -a "$LOG"
        ((fail+=1)) || true
    fi
    rmdir "$lock_dir" 2>/dev/null || true
done < <(find "$DATA" -name "*.parquet" | sort)

echo "=== [$WORKER] $DATASET 结束: pass=$pass fail=$fail skip_done=$skip_done skip_lock=$skip_lock  $(date) ===" | tee -a "$LOG"
