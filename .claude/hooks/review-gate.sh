#!/bin/bash
# .claude/hooks/review-gate.sh
# git commit を含むコマンド実行時、Codexレビューがapproveか確認する
# approve でない場合はコミットをブロックする
# ただし実験ログやプロセス改善のコミットはブロックしない

COMMAND=$(cat - | python -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null)

# git commit コマンドかどうかチェック
if echo "$COMMAND" | grep -q "git commit"; then
  # コミットメッセージに "feat:" や "fix:" が含まれる実装コミットのみチェック
  # "chore:", "docs:", "logs:" などはスキップ
  if echo "$COMMAND" | grep -qE "(feat|fix):.*\.(py|sh)"; then
    VERDICT_FILE="logs/iterations/latest_verdict.json"

    if [ ! -f "$VERDICT_FILE" ]; then
      echo "Codexレビュー未実施です。" >&2
      exit 2
    fi

    DECISION=$(python -c "
import json, re
text = open('$VERDICT_FILE', encoding='utf-8').read()
m = re.search(r'\{.*\}', text, re.DOTALL)
if m:
    print(json.loads(m.group()).get('decision','unknown'))
else:
    print('unknown')
" 2>/dev/null)

    if [ "$DECISION" != "approve" ]; then
      echo "Codexレビューが approve ではありません (現在: $DECISION)" >&2
      exit 2
    fi
  fi
fi

exit 0
