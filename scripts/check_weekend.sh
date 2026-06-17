#!/usr/bin/env bash
# 周一快速检查：汇总所有输出目录的 pass/fail/关键指标
# 用法: bash scripts/check_weekend.sh
set -uo pipefail

CODE=/mnt/public/code/chennuoxi/pretrain-data-eval
cd "$CODE"

ok=0; fail=0

check_stage() {
    local label=$1
    local pattern=$2

    echo ""
    echo "══════════════════════════════════════"
    echo "  $label"
    echo "══════════════════════════════════════"

    local found=0
    while IFS= read -r summary; do
        found=1
        relpath="${summary#$CODE/}"
        # 提取关键指标（任意 key 含 _pct 或 _docs 或 hit_ 的字段）
        stats=$(python3 -c "
import json, sys
d = json.load(open('$summary'))
keys = [k for k in d if any(x in k for x in ['_pct','_docs','hit_','near_dup','exact_dup','contaminated'])]
print('  ' + '  '.join(f'{k}={d[k]}' for k in keys[:6]))
" 2>/dev/null || echo "  (无法解析)")
        echo "  [OK] $relpath"
        echo "$stats"
        ((ok+=1)) || true
    done < <(find "outputs" -path "*/$pattern/*/summary.json" 2>/dev/null | sort)

    if [[ $found -eq 0 ]]; then
        echo "  (无输出)"
    fi
}

check_fail() {
    local log=$1
    if [[ -f "$log" ]]; then
        local n
        n=$(grep -c "\[FAIL\]" "$log" 2>/dev/null || true)
        if [[ $n -gt 0 ]]; then
            echo ""
            echo "  !! FAIL 条目 ($log):"
            grep "\[FAIL\]" "$log" | head -20
            ((fail+=n)) || true
        fi
    fi
}

# ── Stage 2 ───────────────────────────────────────────────────────────────────
check_stage "Stage 2 PII (EN)" "pii"
check_fail "outputs/stage2/pii_ufw_en_l3.log"

check_stage "Stage 2 PII (ZH)" "pii"
check_fail "outputs/stage2/pii_ufw_zh_l3.log"

check_stage "Stage 2 Secrets" "secrets"
check_fail "outputs/stage2/secrets_ufw_en_l3.log"
check_fail "outputs/stage2/secrets_ufw_zh_l3.log"

# ── Stage 4 ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════"
echo "  Stage 4 Dedup"
echo "══════════════════════════════════════"

for subtype in exact ngram minhash; do
    n_ok=$(find "outputs/stage4" -path "*/$subtype/*/summary.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $subtype: $n_ok 个 summary.json"
done
check_fail "outputs/stage4/dedup_ufw_en_l3.log"
check_fail "outputs/stage4/dedup_ufw_zh_l3.log"

echo ""
echo "══════════════════════════════════════"
echo "  汇总: OK=$ok  FAIL=$fail"
echo "══════════════════════════════════════"
