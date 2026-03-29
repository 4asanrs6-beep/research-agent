---
name: never-skip-implementation-process
description: 実装フェーズを飛ばしてコードを書いてはならない。idea-generation→implementation-planning→実装の順序は厳守
type: feedback
---

実装を直接書いてプロセスを飛ばし、ユーザーに強く叱責された（2026-03-29）。

**Why:** multi-perspectiveの直後に「効率的だから」とスクリプトを書いて実行した。idea-generationとimplementation-planningを完全にスキップした。ユーザーは「二度とするな」「痕跡を全部消して」と言った。

**How to apply:**
- コードを書く前に必ず確認: ideas.jsonとplan.jsonは存在するか？conversation.mdに記録されているか？
- 「簡単だから直接書ける」「前回のコードを流用するだけ」は飛ばす理由にならない
- プロセスの各フェーズには意味がある。idea-generationで選択肢を出し、implementation-planningで設計レビューを通し、その上で初めて実装する
- CLAUDE.mdに実験サイクル全体を明記済み。必ず参照すること
- 人間が介入するのは2箇所だけ: (1)問い選定 (2)棄却後の新テーマ決定。実装部分は自動で回す
