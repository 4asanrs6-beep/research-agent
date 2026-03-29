---
name: rewrite-after-review
description: Codexレビューの差し戻しを受けてコードを修正する
allowed-tools: Read, Grep, Glob, Edit, Write, Bash
argument-hint: "[verdict.jsonのパス]"
effort: high
---

## Codex評価結果
!`cat $ARGUMENTS 2>/dev/null || echo "ファイルが見つかりません: $ARGUMENTS"`

## 指示

1. 上記の issues を全て確認する
2. severity=critical の項目を最優先で修正する
3. severity=major の項目を次に修正する
4. severity=minor は可能なら対応、難しければ理由を記載
5. suggestion に従って具体的にコードを修正する
6. 修正しない項目がある場合、理由を明記する
7. 修正完了後、変更サマリーを表示する

## 修正後の記録

修正内容を logs/iterations/ に追記してください:
- 何を修正したか
- 何を修正しなかったか（理由つき）
- 次のCodexレビューで確認すべきポイント
