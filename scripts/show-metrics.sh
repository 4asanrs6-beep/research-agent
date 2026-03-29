#!/bin/bash
# scripts/show-metrics.sh
# 反復メトリクスの要約を表示
#
# Usage:
#   bash scripts/show-metrics.sh
#   bash scripts/show-metrics.sh logs/iterations/metrics.jsonl

FILE="${1:-logs/iterations/metrics.jsonl}"

if [ ! -f "$FILE" ]; then
  echo "メトリクスログがありません: $FILE"
  exit 0
fi

python -c "
import json

lines = open('$FILE', encoding='utf-8').read().strip().split('\n')
entries = []
for line in lines:
    if line.strip():
        try:
            entries.append(json.loads(line))
        except:
            pass

if not entries:
    print('記録なし')
    exit()

print(f'=== 反復メトリクス ({len(entries)}件) ===')
print()

# revise連続回数を追跡
consecutive_revise = 0
max_consecutive_revise = 0
current_type = None

for e in entries:
    ts = e.get('timestamp', '?')
    rt = e.get('review_type', '?')
    dec = e.get('decision', '?')
    score = e.get('overall_score', '?')
    dur = e.get('duration_s', '?')
    n_iss = e.get('n_issues', '?')
    n_crit = e.get('n_critical', 0)
    status = e.get('status', '')

    if status == 'error':
        mark = 'ERROR'
    elif dec == 'approve':
        mark = 'APPROVE'
        consecutive_revise = 0
    elif dec == 'revise':
        consecutive_revise += 1
        max_consecutive_revise = max(max_consecutive_revise, consecutive_revise)
        mark = f'REVISE (x{consecutive_revise})'
    elif dec == 'reject':
        mark = 'REJECT'
        consecutive_revise = 0
    else:
        mark = dec

    crit_warn = f' !!{n_crit}crit' if n_crit and int(n_crit) > 0 else ''
    print(f'  {ts} | {rt:5s} | {mark:15s} | score={score} | {n_iss}issues{crit_warn} | {dur}s')

print()
print(f'合計: {len(entries)}件')
print(f'approve: {sum(1 for e in entries if e.get(\"decision\")==\"approve\")}')
print(f'revise:  {sum(1 for e in entries if e.get(\"decision\")==\"revise\")}')
print(f'reject:  {sum(1 for e in entries if e.get(\"decision\")==\"reject\")}')
print(f'最大連続revise: {max_consecutive_revise}回')
print(f'平均所要時間: {sum(e.get(\"duration_s\",0) for e in entries)/len(entries):.0f}秒')
" 2>/dev/null
