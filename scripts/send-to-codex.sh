#!/bin/bash
# scripts/send-to-codex.sh
# Claude側の成果物をCodexに送ってレビューを受ける
#
# Usage:
#   bash scripts/send-to-codex.sh idea "アイデア3案" logs/iterations/ideas.json
#   bash scripts/send-to-codex.sh plan "実装計画" logs/iterations/plan.json
#   bash scripts/send-to-codex.sh code "イベントスタディ" core/stock_allocation/event_study.py
#   bash scripts/send-to-codex.sh code "直近の変更" "--diff HEAD~3"

REVIEW_TYPE="${1:-code}"  # idea | plan | code
DESCRIPTION="${2:-レビュー対象}"
TARGET="${3:---diff HEAD}"  # ファイルパス or "--diff REF"

LOGDIR="logs/iterations"
METRICS_LOG="$LOGDIR/metrics.jsonl"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="$LOGDIR/${TIMESTAMP}_${REVIEW_TYPE}_verdict.json"
START_EPOCH=$(date +%s)

# 対象コンテンツを取得
if [[ "$TARGET" == --diff* ]]; then
  REF="${TARGET#--diff }"
  REF="${REF:- HEAD}"
  CONTENT=$(git diff $REF 2>/dev/null | head -3000)
  CONTENT_LABEL="変更差分 (git diff $REF)"
else
  CONTENT=$(cat "$TARGET" 2>/dev/null | head -3000)
  CONTENT_LABEL="ファイル: $TARGET"
fi

if [ -z "$CONTENT" ]; then
  echo "エラー: 対象コンテンツが空です。ファイルパスまたは差分を確認してください。" >&2
  exit 1
fi

# プロンプト構築
PROMPT="以下をレビューしてください。AGENTS.mdの評価基準に従い、review_type=${REVIEW_TYPE} の観点でJSON判定を出してください。

## review_type
${REVIEW_TYPE}

## レビュー対象
${DESCRIPTION}

## ${CONTENT_LABEL}
${CONTENT}"

echo "Codexにレビュー送信中 (review_type=${REVIEW_TYPE})..."
echo "$PROMPT" | codex exec - -m gpt-5.4 --ephemeral -o "$OUTPUT" 2>/dev/null
CODEX_EXIT=$?
END_EPOCH=$(date +%s)
DURATION=$((END_EPOCH - START_EPOCH))

if [ $CODEX_EXIT -ne 0 ] || [ ! -f "$OUTPUT" ]; then
  echo "{\"timestamp\": \"$TIMESTAMP\", \"review_type\": \"$REVIEW_TYPE\", \"status\": \"error\", \"duration_s\": $DURATION}" \
    >> "$METRICS_LOG"
  echo "エラー: Codex実行に失敗しました。(${DURATION}秒)" >&2
  exit 1
fi

cp "$OUTPUT" "$LOGDIR/latest_verdict.json" 2>/dev/null

# 判定結果を抽出してメトリクスログに記録
METRICS=$(python -c "
import json, re
text = open('$OUTPUT', encoding='utf-8').read()
m = re.search(r'\{.*\}', text, re.DOTALL)
if m:
    d = json.loads(m.group())
    decision = d.get('decision', 'unknown')
    score = d.get('overall_score', 0)
    n_issues = len(d.get('issues', []))
    critical = sum(1 for i in d.get('issues',[]) if i.get('severity')=='critical')
    print(json.dumps({
        'timestamp': '$TIMESTAMP',
        'review_type': '$REVIEW_TYPE',
        'decision': decision,
        'overall_score': score,
        'n_issues': n_issues,
        'n_critical': critical,
        'duration_s': $DURATION,
        'target': '$TARGET',
        'output': '$OUTPUT',
    }))
else:
    print(json.dumps({
        'timestamp': '$TIMESTAMP',
        'review_type': '$REVIEW_TYPE',
        'status': 'parse_error',
        'duration_s': $DURATION,
    }))
" 2>/dev/null)

echo "$METRICS" >> "$METRICS_LOG"

# 会話ログに記録
ROUND=$(cat "$LOGDIR/round_counter.txt" 2>/dev/null || echo "0")
bash scripts/log-conversation.sh "$ROUND" codex verdict "$OUTPUT" 2>/dev/null

echo ""
echo "=== Codex判定結果 (${REVIEW_TYPE}) | ${DURATION}秒 ==="
echo "ファイル: $OUTPUT"
echo ""
cat "$OUTPUT"
