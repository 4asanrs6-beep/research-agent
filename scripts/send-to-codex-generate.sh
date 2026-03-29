#!/bin/bash
# scripts/send-to-codex-generate.sh
# Codexに「評価者」ではなく「生成者」として素朴な問い・仮説を出させる
#
# Usage:
#   bash scripts/send-to-codex-generate.sh "研究テーマ"
#   bash scripts/send-to-codex-generate.sh "米国株と日本株の個別株の関係"

THEME="${1:-研究テーマ}"

LOGDIR="logs/iterations"
METRICS_LOG="$LOGDIR/metrics.jsonl"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="$LOGDIR/${TIMESTAMP}_codex_brainstorm.json"
START_EPOCH=$(date +%s)

FINDINGS_FILE="logs/experiments/knowledge.md"
FINDINGS=""
if [ -f "$FINDINGS_FILE" ]; then
  FINDINGS=$(cat "$FINDINGS_FILE")
fi

PROMPT="あなたは独立した研究者です。確定した知見のパターン全体から**統合的な問いを少数だけ**生成してください。

## テーマ
${THEME}

## 確定した知見
${FINDINGS}

## あなたの仕事
知見を個別にではなく**パターンとして**読み、**3-5問だけ**に厳選して問いを立てる。

## 問いの条件（厳守）

**良い問い = 知見の複数を同時に使わないと立てられない問い**

- 1つの知見だけから導ける問いは浅い。棄却すること
- 2つ以上の知見の組み合わせからしか導けない問いだけを残す
- 各問いに**二段構えの判定基準**を含めること

## 深い問いの基準（few-shot例。このレベルの深さを必ず出すこと）

### 例1（使う知見: F1+F2）
ボラと回転率は同じ潜在因子を測っているのか、独立した2つのチャネルか。両方有意ということは、(a)共通の潜在因子の2つの表出か、(b)異なるメカニズムが同時に効いているか。ボラを統制した残差回転率、回転率を統制した残差ボラ、それぞれ単独で超過下落を説明するか検証する。片方が消えるなら同一因子、両方生き残るなら独立チャネル。
→ これが分かると: メカニズムが1つか2つか確定。仮説空間が半分に。

### 例2（使う知見: F1+F2+F3）
超過下落は情報の反映か、流動性の一時的枯渇か。β非有意は市場リスクの再評価ではないことを示す。情報なら下落は永続的。流動性なら数日以内に反転する。高ボラ・高回転群の後5日累積超過リターンを見る。**さらに**、反転幅がS&P500の下落幅と無相関なら、日本市場内部の需給現象という証拠がさらに強まる。
→ これが分かると: なぜβが効かないかの答え。活用法が逆張りか回避かも決まる。

### 例3（使う知見: F1+F2+F3+F4）
同じパターンは米国急騰日でも対称的に成立するか。対称なら増幅器、非対称ならパニック固有。**同時に**βと信用残が上方ショックで復活するかも確認する。βが上方では有意になるなら、下方だけ効かない理由はパニック時の意思決定ルールの変質。
→ これが分かると: 増幅かパニック固有か確定。仮説の大半を一発棄却。

### ❌ 浅い問い（これは出すな）
- 「ボラの時間窓を変えたら？」— 知見1つだけ
- 「交互作用は有意か？」— 統計テクニックの確認
- 「mean reversionはあるか？」— 判定基準が一段で二段構えがない

## 重要な制約
- 実装手法（PCA、機械学習等）に踏み込まない
- 必要データ: 日本株日足、信用取引、財務、空売り、米国株日足、VIX、為替が使える
- 別の研究者（Claude）も同じ知見で問いを立てている。Claudeが思いつかない角度を優先
- 仮説は内部の思考に留め、出力には含めない。出力は**問いだけ**

## 出力形式（Claude/Codex共通スキーマ。厳守）
{
  \"source\": \"codex\",
  \"findings_used\": [\"F1: ...\", \"F2: ...\"],
  \"questions\": [
    {
      \"id\": \"kebab-case英語短縮名（例: liquidity-fragility, panic-asymmetry）。q1/cq1のような連番は使わない\",
      \"question\": \"問い（3-5文。背景・予測・二段構えの判別基準を含む）\",
      \"findings_leveraged\": [\"F1\", \"F2\"],
      \"what_this_decides\": \"この結果で何が決まるか（1-2文）\",
      \"first_step\": \"検証の第一歩（1-2文）\",
      \"difficulty\": \"low|medium|high\"
    }
  ]
}"

echo "Codexにアイデア生成を依頼中..."
REVIEWER="codex"

# Codex実行を試行
if command -v codex &>/dev/null; then
  echo "$PROMPT" | codex exec - -m gpt-5.4 --ephemeral -o "$OUTPUT" 2>/dev/null
  CODEX_EXIT=$?
else
  echo "WARNING: codexコマンドが見つかりません。Claude Codeサブエージェントで代替します"
  CODEX_EXIT=1
fi

# Codex失敗時のフォールバック: Claude Codeで代替
if [ $CODEX_EXIT -ne 0 ] || [ ! -f "$OUTPUT" ] || [ ! -s "$OUTPUT" ]; then
  echo "Codex実行失敗(exit=$CODEX_EXIT)。Claude Codeサブエージェントで代替..."
  REVIEWER="claude-fallback"

  # Claude CLIで代替実行
  if command -v claude &>/dev/null; then
    echo "$PROMPT" | claude --print -m claude-sonnet-4-6 > "$OUTPUT" 2>/dev/null
    CODEX_EXIT=$?
  else
    echo "エラー: claudeコマンドも見つかりません。" >&2
    CODEX_EXIT=1
  fi
fi

END_EPOCH=$(date +%s)
DURATION=$((END_EPOCH - START_EPOCH))

if [ $CODEX_EXIT -ne 0 ] || [ ! -f "$OUTPUT" ]; then
  echo "{\"timestamp\": \"$TIMESTAMP\", \"review_type\": \"brainstorm\", \"status\": \"error\", \"duration_s\": $DURATION}" \
    >> "$METRICS_LOG"
  echo "エラー: Codex実行に失敗しました。(${DURATION}秒)" >&2
  exit 1
fi

# メトリクス記録
python -c "
import json, re
text = open('$OUTPUT', encoding='utf-8').read()
m = re.search(r'\{.*\}', text, re.DOTALL)
n_items = 0
if m:
    try:
        d = json.loads(m.group())
        n_items = len(d.get('hypotheses', d.get('questions', [])))
    except: pass
print(json.dumps({
    'timestamp': '$TIMESTAMP',
    'review_type': 'brainstorm',
    'decision': 'generated',
    'reviewer': '$REVIEWER',
    'overall_score': n_items,
    'n_issues': 0,
    'n_critical': 0,
    'duration_s': $DURATION,
    'target': 'codex_brainstorm',
    'output': '$OUTPUT',
}))
" >> "$METRICS_LOG" 2>/dev/null

# 会話ログに記録
ROUND=$(cat "$LOGDIR/round_counter.txt" 2>/dev/null || echo "0")
bash scripts/log-conversation.sh "$ROUND" codex brainstorm "$OUTPUT" 2>/dev/null

# brainstorm_discussion.md に追記
DISC_MD="$LOGDIR/brainstorm_discussion.md"
python -c "
import json, re

text = open('$OUTPUT', encoding='utf-8').read()
m = re.search(r'\{.*\}', text, re.DOTALL)
if not m:
    exit()
d = json.loads(m.group())
questions = d.get('questions', [])

with open('$DISC_MD', 'a', encoding='utf-8') as f:
    f.write('\n---\n\n')
    reviewer = '$REVIEWER'
    if reviewer == 'claude-fallback':
        f.write('### 🟡 Claude-fallback [Generator] (Codex代替)\n\n')
        f.write('> Codexが利用不可のためClaude Codeサブエージェントで代替生成。\n\n')
    else:
        f.write('### 🟠 Codex [Generator]\n\n')
        f.write('> Claudeの問いを見ずに独立生成。\n\n')
    for q in questions:
        qid = q.get('id', '?')
        qt = q.get('question', '')
        fl = '+'.join(q.get('findings_leveraged', []))
        wd = q.get('what_this_decides', '')
        diff = q.get('difficulty', '?')
        f.write(f'#### {qid}（使う知見: {fl} | 難易度: {diff}）\n')
        f.write(f'{qt}\n')
        f.write(f'**これが分かると:** {wd}\n\n')
    f.write(f'({len(questions)}問生成)\n')
    f.write('\n---\n')
" 2>/dev/null

echo ""
echo "=== Codexアイデア生成完了 | ${DURATION}秒 ==="
echo "ファイル: $OUTPUT"
echo ""
cat "$OUTPUT"
