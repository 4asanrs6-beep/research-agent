# 研究エージェント

## 反復開発ルール
- 実装は1回で完成を目指さない。差し戻し前提で進める
- 自己評価だけで完了にしない。Codexレビューを通す
- 実験結果は必ず記録し、再議論してから次に進む

## 研究プロセス

```
テーマ → 問い → 知見 → 新しい問い → 知見 → ...
```

### テーマ
人間が設定する研究の大目標。すべての問いはここから派生する。

### 問い
テーマ（または過去の知見）から生まれる検証可能な仮説。
各問いは以下の5 Phaseを順に回す:

```
Phase 1 生成 → Phase 2 選定 → Phase 3 設計 ←──┐
                                  ↓            │
                              Phase 4 実験      │ 修正して
                                  ↓            │ 再実験
                              Phase 5 議論 ─────┘
                                  ↓
                              知見を記録
                                  ↓
                          次の問いへ / 新しい問いを生成
```

| Phase | 何をするか | 誰が判断 |
|---|---|---|
| **Phase 1 生成** | /brainstorm + Codexで問いを複数生成。/multi-perspectiveで深掘り | 自動 |
| **Phase 2 選定** | 人間が3-5問を選ぶ | **人間** |
| **Phase 3 設計** | /idea-generation → /implementation-planning → Codexレビュー → approve | 自動 |
| **Phase 4 実験** | 実装 → Codexコードレビュー → approve → 実行 → 記録 | 自動 |
| **Phase 5 議論** | /experiment-review → 知見を記録 → Codex裁定 | 自動 |

Phase 5の議論で再帰が発生する:
- **不備修正** → Phase 3（設計）に戻る
- **修正して再実験** → Phase 3（設計）に戻る
- **完了 or 棄却** → 知見を記録 → 次の問いへ
- **全問い完了** → 知見から新しい問いを生成 → Phase 1へ **[人間が選定]**

### 知見
実験から得られた確定的な発見。棄却された実験からも知見は残る。
知見は次の問い生成の入力になる。`logs/experiments/knowledge.md` に蓄積。

### 人間が介入するポイント（2箇所のみ）
1. **選定**: 問い生成後、人間が3-5問を選ぶ
2. **新テーマ/再生成**: 全問い完了後、新しいテーマまたは知見からの再生成を判断

## Codex代替ルール

Codexが利用できない場合（未インストール、API障害等）、Claude Codeのサブエージェントで代替する。

### 代替の仕組み
- **brainstorm生成:** send-to-codex-generate.shが自動フォールバック。codexコマンド失敗時にclaude CLIで代替実行
- **設計/コードレビュー:** Bash経由のcodex execが失敗したら、Agent toolでサブエージェントを起動し同じプロンプトを投入
- **裁定:** 同上

### ログへの明記（厳守）
代替実行時は以下を必ず明記する:
- メトリクスJSON: `"reviewer": "claude-fallback"`
- discussion.md: `🟡 Claude-fallback [Generator] (Codex代替)` （通常のCodexは `🟠 Codex`）
- conversation.md: `[codex-fallback]` タグを付ける

### 注意
Claude-fallbackはCodexと異なるモデルであり、独立した視点ではない。
ログにその旨を明記し、結果の解釈時に「独立レビューではない」ことを考慮すること。

## 実装ゲート（厳守）

「設計」ステップの成果物がない限り、コードを書いてはならない:
1. `logs/iterations/*_ideas.json` — /idea-generationの出力
2. `logs/iterations/*_plan.json` — /implementation-planningの出力。Codex approve済み
3. `logs/iterations/conversation.md` に上記2つのログが記録済み

この3条件を満たさずにWrite/Editでスクリプトを書いた場合、プロセス違反。

## 各ステップの詳細

### 生成
/brainstorm "テーマ" → 知見パターンから統合的な問いを3-5個生成
bash scripts/send-to-codex-generate.sh "テーマ" → Codexも独立に生成
→ 統合して人間に提示

**知見駆動:** knowledge.mdを必ず読み、確定知見から問いを導出する。初回のみテーマからの自由生成。

/multi-perspective "選ばれた問い" → 5ロール x 2ラウンドで問いが進化（v0->v1->v2）
→ Codex裁定で着手順を決定

### 選定
**人間が3-5問を選ぶ。**

### 設計
/idea-generation → 2-3案を生成し推奨案を自動採用
/implementation-planning → 実装計画を作成 → Codex設計レビュー → approveまで自動修正

### 実験
コード記述 → Codexコードレビュー → approveまで自動修正 → 実行
結果を logs/experiments/ に記録。知見を knowledge.md に追記。

### 議論
/experiment-review → 5ロール+Codexで結果を議論
→ Codex裁定: 不備修正 / 修正して再実験 / 棄却確定 / 次の問いに移る

## ログ構成

### 人間が見るファイル
1. **`decision_summary.md`** — 進捗と系譜。今どこにいて次は何か
2. **`research_diary.md`** — 研究の解説。各問いの「なぜ→何を→結果→解釈」を物語として読める

### decision_summary.md の内容:
- テーマと蓄積された知見
- 問いの系譜（どの知見からどの問いが生まれたか）
- 各問いの進捗（生成→選定→設計→実験→議論のどこか）
- 直近の動き

### decision_summary.md の書き方ルール
1. ステップ完了ごとに「直近の動き」に1エントリ追加する
2. 各エントリに: 何の議論か / 何が分かったか / 次は何か
3. 技術的詳細は最小限。必要なら experiments/ へリンク

### 各議論ログの冒頭ルール
brainstorm_discussion.md、multi_perspective.md、experiment_review.mdの各エントリは冒頭に:

```
## [問いのID] — [何の議論か（1文）]
**結論:** [何が決まったか（1文）]
```

### conversation.md の見出しルール
Roundは使わない。代わりにPhaseで区切る:

```
## T1-Q03 Phase 3 設計
### idea-generation
（内容）
### implementation-planning
（内容）
### Codex設計レビュー 1回目
（内容）
### Codex設計レビュー 2回目（修正後）
（内容）

## T1-Q03 Phase 4 実験
### 実装
（内容）
### Codexコードレビュー 1回目
（内容）
```

### research_diary.md の書き方ルール
問いのPhase 5（議論）完了時にエントリを追加する。各エントリに:
- **なぜこの問いを立てたか** — 前の問いの知見からの流れ
- **何を検証したか** — 設計の概要（専門用語は最小限）
- **何が分かったか** — 結果と解釈（表があると分かりやすい）
- **次にどうつながるか** — この知見が次の問いにどう活きるか

### 状態の視覚表示
問いの状態はdecision_summary.mdで以下の記号を使う:
- ❌ 棄却
- ✅ 完了
- 🔄 実験中
- ⬜ 未着手

### ファイル構成
```
logs/
  research_diary.md           <- 研究の解説（人間向け）
  iterations/
    decision_summary.md       <- 進捗と系譜（人間向け）
    brainstorm_discussion.md  <- 問いの生成と選定（監査用）
    multi_perspective.md      <- 問いの深掘り議論（監査用）
    conversation.md           <- 設計/実装のCodex査読（監査用）
  experiments/
    knowledge.md              <- 知見の蓄積
    [問いID].md               <- 実験記録（監査用）
    [問いID]_review.md        <- 結果議論ログ（監査用）
```

### 命名規則
- テーマ: `T1`, `T2`, ...（通し番号）
- 問い: `T1-Q01 margin-shock`, `T1-Q02 holder-instability`, ...（テーマ番号+通し番号+英語短縮名）
- Phase 1の問い生成時にIDを付与する
- IDはすべてのログ・実験記録・knowledge.mdで一貫して使う

### ファイル命名
- 実験記録: `logs/experiments/T1-Q02_stage1.md`
- 結果議論: `logs/experiments/T1-Q02_stage1_review.md`
- 結果JSON: `logs/iterations/T1-Q03_result.json`
- アイデア: `logs/iterations/T1-Q03_ideas.json`
- 計画: `logs/iterations/T1-Q03_plan.json`
