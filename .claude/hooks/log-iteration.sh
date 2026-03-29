#!/bin/bash
# .claude/hooks/log-iteration.sh
# 各ターン終了時に簡易ログを記録する（非同期実行）

LOGDIR="logs/iterations"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "{\"timestamp\": \"$TIMESTAMP\", \"session\": \"${CLAUDE_SESSION_ID:-unknown}\", \"event\": \"stop\"}" \
  >> "$LOGDIR/session_log.jsonl"
exit 0
