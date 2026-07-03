#!/usr/bin/env bash
# Stage 5 后台批量任务状态总览
# 用法: bash scripts/contamination_status.sh

CODE=/mnt/public/code/chennuoxi/pretrain-data-eval

echo "=== Contamination 后台任务状态 ($(date +%H:%M:%S)) ==="
echo

# 后台进程
echo "[进程]"
ps aux 2>/dev/null | grep -E "contamination_batch|contamination_cascade_batch|run\.py exact|run\.py cascade" | grep -v grep | awk '{printf "  %s %s %s %s %s\n", $1, $2, $9, $10, $11}' | head -8
echo

# 各 batch 进度
echo "[ZH exact (596 files)]"
done=$(ls "$CODE/outputs/stage5/ufw_zh_l3_v3/exact/" 2>/dev/null | wc -l)
echo "  done: $done / 596 ($((done * 100 / 596))%)"

echo "[ZH cascade sample500 (596 files)]"
done=$(ls "$CODE/outputs/stage5/ufw_zh_l3_v3/cascade_sample500/" 2>/dev/null | grep -v aggregated | wc -l)
echo "  done: $done / 596 ($((done * 100 / 596))%)"

echo "[EN cascade sample500 (1168 files)]"
done=$(ls "$CODE/outputs/stage5/ufw_en_l3_v3/cascade_sample500/" 2>/dev/null | grep -v aggregated | wc -l)
echo "  done: $done / 1168 ($((done * 100 / 1168))%)"
echo

# GPU 用量
echo "[GPU]"
python -c "import torch; m=torch.cuda.mem_get_info(); print(f'  used={(m[1]-m[0])/1e9:.1f}GB / total={m[1]/1e9:.0f}GB')" 2>&1
echo

# 最近 cascade 输出
echo "[最近 ZH cascade]"
tail -3 "$CODE/outputs/stage5/contamination_ufw_zh_l3_v3_cascade_sample500.log" 2>/dev/null | sed 's/^/  /'
echo "[最近 EN cascade]"
tail -3 "$CODE/outputs/stage5/contamination_ufw_en_l3_v3_cascade_sample500.log" 2>/dev/null | sed 's/^/  /'
