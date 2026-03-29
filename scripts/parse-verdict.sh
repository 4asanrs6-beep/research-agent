#!/bin/bash
# scripts/parse-verdict.sh
# Codex判定結果からdecision/score/issuesを抽出して表示
#
# Usage:
#   bash scripts/parse-verdict.sh
#   bash scripts/parse-verdict.sh logs/iterations/20260327_123456_idea_verdict.json

FILE="${1:-logs/iterations/latest_verdict.json}"

if [ ! -f "$FILE" ]; then
  echo "エラー: ファイルが見つかりません: $FILE" >&2
  exit 1
fi

python -c "
import json, re, sys

text = open('$FILE', encoding='utf-8').read()

# JSON部分を抽出（テキスト混在に対応）
m = re.search(r'\{.*\}', text, re.DOTALL)
if not m:
    print('parse_error: JSONが見つかりません')
    sys.exit(1)

try:
    d = json.loads(m.group())
except json.JSONDecodeError as e:
    print(f'parse_error: {e}')
    sys.exit(1)

review_type = d.get('review_type', '?')
decision = d.get('decision', 'unknown')
score = d.get('overall_score', '?')
issues = d.get('issues', [])
n_issues = len(issues)
critical = sum(1 for i in issues if i.get('severity') == 'critical')
major = sum(1 for i in issues if i.get('severity') == 'major')
minor = sum(1 for i in issues if i.get('severity') == 'minor')

print(f'Review Type: {review_type}')
print(f'Decision:    {decision}')
print(f'Score:       {score}')
print(f'Issues:      {n_issues} (critical={critical}, major={major}, minor={minor})')
print()

if issues:
    print('--- Issues ---')
    for i, issue in enumerate(issues, 1):
        sev = issue.get('severity', '?')
        cat = issue.get('category', '?')
        desc = issue.get('description', '')
        sugg = issue.get('suggestion', '')
        print(f'  [{sev}] {cat}: {desc}')
        if sugg:
            print(f'         → {sugg}')
    print()

if decision == 'revise':
    guidance = d.get('revision_guidance', '')
    if guidance:
        print('--- Revision Guidance ---')
        print(f'  {guidance}')
        print()

strengths = d.get('strengths', [])
if strengths:
    print('--- Strengths ---')
    for s in strengths:
        print(f'  + {s}')

# スコア詳細
scores = d.get('scores', {})
if scores:
    print()
    print('--- Scores ---')
    for k, v in scores.items():
        bar = '#' * int(v * 10) if isinstance(v, (int, float)) else ''
        print(f'  {k}: {v} {bar}')
" 2>/dev/null
