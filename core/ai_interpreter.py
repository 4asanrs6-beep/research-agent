"""AI による結果解釈モジュール

分析結果を入力として、仮説の評価（valid / invalid / needs_review）と
判断理由を生成する。
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

INTERPRETATION_PROMPT = """\
あなたは日本株市場の定量分析研究者です。
以下の投資アイデアと分析結果を評価し、仮説の妥当性を判定してください。

## 投資アイデア
{idea_text}

## 分析計画
{plan_summary}

## 統計分析結果
```json
{statistics_json}
```

## バックテスト結果
```json
{backtest_json}
```

## 評価基準
- **統計的有意性**: p値 < 0.05 か？
- **効果量**: Cohen's d >= 0.2 か？
- **サンプルサイズ**: 十分な観測数か？
- **バックテスト**: 実運用で利益が出るか？(シャープ比、ドローダウン、ベンチマーク超過)
- **頑健性**: 結果は安定しているか？

## 出力形式
以下のJSON形式で回答してください。JSONのみを出力し、他のテキストは含めないでください。

```json
{{
    "evaluation_label": "valid | invalid | needs_review",
    "confidence": 0.0 ~ 1.0,
    "summary": "評価サマリー（2-3文）",
    "reasons": [
        "判断理由1",
        "判断理由2",
        "判断理由3"
    ],
    "strengths": ["この分析の強み"],
    "weaknesses": ["この分析の弱み・注意点"],
    "suggestions": ["改善提案や追加検証のアイデア"],
    "knowledge_entry": {{
        "hypothesis": "検証された仮説（1文）",
        "valid_conditions": "有効と判断される条件（もしあれば）",
        "invalid_conditions": "無効と判断される条件（もしあれば）",
        "tags": ["タグ1", "タグ2"]
    }}
}}
```
"""


class AiInterpreter:
    """AIを使用して分析結果を解釈・評価"""

    def __init__(self, ai_client: Any):
        """
        Args:
            ai_client: AIモデルクライアント。send_message(prompt) -> str を持つオブジェクト。
        """
        self.ai_client = ai_client

    def interpret(
        self,
        idea_text: str,
        plan: dict,
        statistics_result: dict | None,
        backtest_result: dict | None,
    ) -> dict:
        """分析結果を解釈し評価を返す

        Args:
            idea_text: 投資アイデアのテキスト
            plan: 分析計画
            statistics_result: 統計分析結果
            backtest_result: バックテスト結果

        Returns:
            評価結果の辞書
        """
        plan_summary = self._format_plan_summary(plan)

        # equity_curve/benchmark_curveは長大なのでサマリー化
        stats_json = json.dumps(
            self._compact(statistics_result) if statistics_result else {},
            ensure_ascii=False,
            indent=2,
        )
        bt_json = json.dumps(
            self._compact(backtest_result) if backtest_result else {},
            ensure_ascii=False,
            indent=2,
        )

        prompt = INTERPRETATION_PROMPT.format(
            idea_text=idea_text,
            plan_summary=plan_summary,
            statistics_json=stats_json,
            backtest_json=bt_json,
        )

        try:
            response = self.ai_client.send_message(prompt)
            result = self._parse_response(response)
            logger.info(
                "AI評価完了: %s (confidence=%.2f)",
                result.get("evaluation_label", "unknown"),
                result.get("confidence", 0),
            )
            return result
        except Exception as e:
            logger.error("AI結果解釈エラー: %s", e)
            return {
                "evaluation_label": "needs_review",
                "confidence": 0.0,
                "summary": f"AI解釈エラー: {e}",
                "reasons": [str(e)],
                "strengths": [],
                "weaknesses": [],
                "suggestions": [],
                "knowledge_entry": {
                    "hypothesis": idea_text,
                    "valid_conditions": None,
                    "invalid_conditions": None,
                    "tags": [],
                },
            }

    def _format_plan_summary(self, plan: dict) -> str:
        """計画の要約テキストを生成"""
        parts = []
        if plan.get("plan_name"):
            parts.append(f"計画名: {plan['plan_name']}")
        if plan.get("hypothesis"):
            parts.append(f"仮説: {plan['hypothesis']}")
        methodology = plan.get("methodology", {})
        if methodology.get("approach"):
            parts.append(f"アプローチ: {methodology['approach']}")
        if methodology.get("steps"):
            parts.append("ステップ:")
            for s in methodology["steps"]:
                parts.append(f"  - {s}")
        universe = plan.get("universe", {})
        if universe.get("detail"):
            parts.append(f"対象: {universe['detail']}")
        period = plan.get("analysis_period", {})
        if period.get("start_date"):
            parts.append(f"期間: {period['start_date']} ~ {period.get('end_date', '')}")
        return "\n".join(parts) if parts else json.dumps(plan, ensure_ascii=False)

    def _compact(self, data: dict) -> dict:
        """大きなリストフィールドを要約"""
        result = {}
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 10:
                result[k] = f"[{len(v)} items, first={v[0]}, last={v[-1]}]"
            else:
                result[k] = v
        return result

    def _parse_response(self, response: str) -> dict:
        """AIレスポンスからJSONを抽出"""
        text = response.strip()

        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        brace_start = text.find("{")
        brace_end = text.rfind("}") + 1
        if brace_start >= 0 and brace_end > brace_start:
            text = text[brace_start:brace_end]

        return json.loads(text)
