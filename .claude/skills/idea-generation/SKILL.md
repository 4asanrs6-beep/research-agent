---
name: idea-generation
description: 研究目標に対して複数のアプローチ案を生成し、推奨案を明示する
allowed-tools: Read, Grep, Glob
argument-hint: "[研究目標]"
effort: high
---

## 研究目標
$ARGUMENTS

## あなたの仕事

1. `logs/experiments/knowledge.md` を読み、確定知見を把握する
2. `logs/iterations/decision_summary.md` を読み、現在の着手対象を把握する
3. 研究目標に対して**2-3個のアプローチ案**を生成する
4. **推奨案を1つ選び、理由を明示する**

## 各案に含める情報

- **何をするか**: 専門用語を避けて1-2文で説明
- **前回との違い**: 前回実験（knowledge.md）と何が変わるか
- **既存コードの流用**: 何を再利用し、何を新規に書く必要があるか
- **リスク**: この案が失敗する最も可能性の高い理由
- **実装コスト**: low（1日以内）/ medium（2-3日）/ high（1週間+）

## 推奨案の選定基準

以下の優先順で推奨を決める：
1. **前回実験との比較可能性** — 前回と同じ枠組みなら結果を直接比較でき、知見の蓄積が効率的
2. **実装コストの低さ** — 同じ精度なら軽い方を選ぶ
3. **統計的な頑健性** — 仮定が少ない方を優先

## 出力形式（Claude/Codex共通）

```json
{
  "source": "claude",
  "goal": "研究目標（1文）",
  "ideas": [
    {
      "id": "1",
      "title": "アプローチ名",
      "what": "何をするか（専門用語なし、1-2文）",
      "diff_from_previous": "前回実験と何が違うか（1文）",
      "reuse": "既存コードの何を再利用するか（1文）",
      "new_work": "新規に書く必要があるもの（1文）",
      "risk": "失敗する最も可能性の高い理由（1文）",
      "cost": "low|medium|high"
    }
  ],
  "recommendation": {
    "id": "推奨案のID",
    "reason": "なぜこの案を推奨するか（2-3文）",
    "robustness_check": "頑健性確認として他の案をどう併用するか（1文、不要ならnull）"
  }
}
```

## 禁止事項
- 推奨なしで選択を丸投げすること
- 専門用語だけのpros/cons羅列（判断できない形式）
- 4案以上の大量提示

出力はJSONのみ。結果を `logs/iterations/cq3_new_ideas.json`（問い名に合わせて変更）に保存してください。

## 討論ログの記録（必須）

JSON保存後、`logs/iterations/conversation.md` に以下の形式で追記してください。

```markdown
### Round N | 🔵 Claude | idea-generation (問い名)

**目標:** ...

**案:**

| ID | アプローチ | 何をするか | 前回との違い | コスト |
|---|---|---|---|---|
| 1 | ... | ... | ... | low |

**推奨:** 案X — (理由 1-2文)
**頑健性確認:** (併用する案)

→ 人間の承認待ち → /implementation-planning へ
```
