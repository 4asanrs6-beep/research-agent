#!/bin/bash
# scripts/log-conversation.sh
# 反復ワークフローの会話ログを記録する
#
# Usage:
#   bash scripts/log-conversation.sh <round> <speaker> <message_type> <content_file_or_text>
#
# Examples:
#   bash scripts/log-conversation.sh 1 claude idea_generation logs/iterations/ideas.json
#   bash scripts/log-conversation.sh 1 codex verdict logs/iterations/latest_verdict.json
#   bash scripts/log-conversation.sh 1 human decision "Bを選択: 実装に進む"
#   bash scripts/log-conversation.sh 2 claude code_revision "n_eventsをpeak_lagベースに修正"

ROUND="${1:-0}"
SPEAKER="${2:-unknown}"    # claude | codex | human
MSG_TYPE="${3:-note}"      # idea_generation | verdict | revision | decision | code_review | note
CONTENT_SOURCE="${4:-}"

LOGDIR="logs/iterations"
CONV_JSONL="$LOGDIR/conversation.jsonl"
CONV_MD="$LOGDIR/conversation.md"
mkdir -p "$LOGDIR"

TIMESTAMP=$(date +%Y-%m-%d\ %H:%M:%S)

# コンテンツ取得
if [ -f "$CONTENT_SOURCE" ]; then
  CONTENT=$(cat "$CONTENT_SOURCE" 2>/dev/null | head -200)
  SOURCE_FILE="$CONTENT_SOURCE"
else
  CONTENT="$CONTENT_SOURCE"
  SOURCE_FILE=""
fi

# JSONL記録
python -c "
import json
entry = {
    'timestamp': '$TIMESTAMP',
    'round': int('$ROUND'),
    'speaker': '$SPEAKER',
    'message_type': '$MSG_TYPE',
    'source_file': '$SOURCE_FILE',
    'content_preview': '''$CONTENT'''[:500],
}
print(json.dumps(entry, ensure_ascii=False))
" >> "$CONV_JSONL" 2>/dev/null

# Markdown追記
ICON=""
case "$SPEAKER" in
  claude) ICON="🔵 Claude" ;;
  codex)  ICON="🟠 Codex" ;;
  human)  ICON="🟢 Human" ;;
  *)      ICON="⚪ $SPEAKER" ;;
esac

TYPE_LABEL=""
case "$MSG_TYPE" in
  idea_generation) TYPE_LABEL="アイデア生成" ;;
  verdict)         TYPE_LABEL="評価判定" ;;
  revision)        TYPE_LABEL="修正" ;;
  decision)        TYPE_LABEL="判断" ;;
  code_review)     TYPE_LABEL="コードレビュー" ;;
  note)            TYPE_LABEL="メモ" ;;
  *)               TYPE_LABEL="$MSG_TYPE" ;;
esac

# conversation.md が存在しなければヘッダー作成
if [ ! -f "$CONV_MD" ]; then
  echo "# 反復ワークフロー 会話ログ" > "$CONV_MD"
  echo "" >> "$CONV_MD"
  echo "---" >> "$CONV_MD"
  echo "" >> "$CONV_MD"
fi

cat >> "$CONV_MD" << MDEOF

### Round $ROUND | $ICON | $TYPE_LABEL
**$TIMESTAMP**

MDEOF

# コンテンツをMarkdownに追記（長すぎる場合は切り詰め）
if [ -n "$SOURCE_FILE" ]; then
  echo "ファイル: \`$SOURCE_FILE\`" >> "$CONV_MD"
  echo "" >> "$CONV_MD"
  echo '```' >> "$CONV_MD"
  head -30 "$SOURCE_FILE" >> "$CONV_MD" 2>/dev/null
  TOTAL_LINES=$(wc -l < "$SOURCE_FILE" 2>/dev/null || echo 0)
  if [ "$TOTAL_LINES" -gt 30 ]; then
    echo "... (${TOTAL_LINES}行中、先頭30行を表示)" >> "$CONV_MD"
  fi
  echo '```' >> "$CONV_MD"
else
  echo "$CONTENT" >> "$CONV_MD"
fi

echo "" >> "$CONV_MD"
echo "---" >> "$CONV_MD"
