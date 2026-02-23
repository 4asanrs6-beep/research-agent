"""AI による分析計画生成モジュール

投資アイデア（テキスト）を受け取り、検証用の分析計画を生成する。
AIモデルはAPI経由で呼び出し、モデル名は設定で変更可能。
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

PLAN_GENERATION_PROMPT = """\
あなたは日本株市場の定量分析研究者です。
以下の投資アイデアを検証するための分析計画を作成してください。

## 投資アイデア
{idea_text}
{universe_filter_section}
## 利用可能なデータ
- 株価日足（OHLCV、調整後価格） - J-Quants API経由
- 上場銘柄一覧（セクター情報含む）
- 財務サマリー（PER, PBR, ROE等）
- TOPIX等の指数四本値
- 信用取引残高
- 業種別空売り比率
- 分析期間: 最大過去10年

## 出力形式
以下のJSON形式で回答してください。JSONのみを出力し、他のテキストは含めないでください。

```json
{{
    "plan_name": "計画の名前（簡潔に）",
    "hypothesis": "検証する仮説（1-2文）",
    "data_requirements": {{
        "price_data": true,
        "financial_data": false,
        "index_data": false,
        "margin_data": false,
        "short_selling_data": false,
        "description": "必要なデータの説明"
    }},
    "universe": {{
        "type": "all | sector | individual",
        "detail": "具体的な対象（例: 全銘柄、情報通信セクター、7203）",
        "reason": "ユニバース選択理由"
    }},
    "analysis_period": {{
        "start_date": "YYYY-MM-DD",
        "end_date": "YYYY-MM-DD",
        "reason": "期間選択理由"
    }},
    "methodology": {{
        "approach": "分析アプローチの概要",
        "steps": [
            "ステップ1: ...",
            "ステップ2: ...",
            "ステップ3: ..."
        ],
        "statistical_tests": ["使用する統計検定"],
        "metrics": ["評価指標"]
    }},
    "backtest": {{
        "strategy_description": "バックテスト戦略の概要",
        "entry_rule": "エントリールール",
        "exit_rule": "エグジットルール",
        "rebalance_frequency": "daily | weekly | monthly",
        "benchmark": "TOPIX"
    }},
    "expected_outcome": "期待される結果と判断基準"
}}
```
"""


class AiPlanner:
    """AIを使用して投資アイデアから分析計画を生成"""

    def __init__(self, ai_client: Any):
        """
        Args:
            ai_client: AIモデルクライアント。send_message(prompt) -> str を持つオブジェクト。
        """
        self.ai_client = ai_client

    def generate_plan(
        self,
        idea_text: str,
        universe_filter_text: str = "",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """投資アイデアから分析計画を生成

        Args:
            idea_text: ユーザーが入力した投資アイデアのテキスト
            universe_filter_text: ユニバースフィルタ条件のテキスト
            start_date: 分析開始日（例: "2021-01-01"）
            end_date: 分析終了日（例: "2026-02-23"）

        Returns:
            分析計画の辞書
        """
        universe_filter_section = ""
        if universe_filter_text:
            universe_filter_section += (
                "\n## ユニバースの制約条件\n"
                "以下の条件で分析対象銘柄を絞り込んでください:\n"
                f"{universe_filter_text}\n\n"
            )
        if start_date and end_date:
            universe_filter_section += (
                "\n## 分析期間の制約（必須）\n"
                "分析期間は以下の通り固定です。変更しないでください:\n"
                f"- 開始日: {start_date}\n"
                f"- 終了日: {end_date}\n\n"
            )

        prompt = PLAN_GENERATION_PROMPT.format(
            idea_text=idea_text,
            universe_filter_section=universe_filter_section,
        )

        try:
            response = self.ai_client.send_message(prompt)
            plan = self._parse_response(response)
            logger.info("分析計画生成完了: %s", plan.get("plan_name", "unnamed"))
            return plan
        except Exception as e:
            logger.error("AI計画生成エラー: %s", e)
            return {"error": str(e)}

    def _parse_response(self, response: str | None) -> dict:
        """AIレスポンスからJSONを抽出・パース"""
        if not response:
            raise ValueError("AIから空の応答を受信しました")
        text = response.strip()

        # ```json ... ``` ブロックを抽出
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        # { } の範囲を抽出
        brace_start = text.find("{")
        brace_end = text.rfind("}") + 1
        if brace_start >= 0 and brace_end > brace_start:
            text = text[brace_start:brace_end]

        return json.loads(text)

    def refine_plan(self, plan: dict, feedback: str) -> dict:
        """ユーザーフィードバックに基づき計画を修正

        Args:
            plan: 現在の分析計画
            feedback: ユーザーからの修正指示

        Returns:
            修正された分析計画
        """
        prompt = f"""\
以下の分析計画に対するフィードバックを反映して、修正版を作成してください。
出力は元と同じJSON形式のみで、他のテキストは含めないでください。

## 現在の計画
```json
{json.dumps(plan, ensure_ascii=False, indent=2)}
```

## フィードバック
{feedback}
"""
        try:
            response = self.ai_client.send_message(prompt)
            return self._parse_response(response)
        except Exception as e:
            logger.error("AI計画修正エラー: %s", e)
            return {"error": str(e)}
