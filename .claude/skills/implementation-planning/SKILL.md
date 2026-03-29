---
name: implementation-planning
description: 採用アイデアの詳細実装計画を作成する
allowed-tools: Read, Grep, Glob
argument-hint: "[アイデアJSON or 説明]"
---

以下のアイデアの実装計画を作成してください。

## アイデア
$ARGUMENTS

## 既存コードベース構造
!`find core/ -name "*.py" -type f 2>/dev/null | head -20`

## 既存パターン
対象に近いモジュールがあれば読んで、同じパターンに合わせること。

## 出力形式
以下のJSON形式で出力してください。

```json
{
  "plan": {
    "files": [
      {"path": "core/xxx.py", "action": "create|modify", "description": "何を書くか"}
    ],
    "functions": [
      {"name": "func_name", "signature": "def func_name(arg1, arg2) -> ReturnType", "purpose": "役割"}
    ],
    "data_flow": "入力 → 処理 → 出力の流れ",
    "test_method": "どう検証するか",
    "implementation_order": ["step1", "step2", "step3"],
    "estimated_complexity": "low|medium|high"
  }
}
```

出力後、結果を logs/iterations/ にも保存してください。
