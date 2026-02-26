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

## 重要な評価ルール — エッジの方向（ロング/ショート）
- エッジにはロング（買い）とショート（空売り）の2方向がある。どちらも有効なエッジである。
- **超過リターンが正 + 有意** → ロングエッジ。シグナル銘柄を買えば利益が出る。"valid" と判定。
- **超過リターンが負 + 有意** → ショートエッジ。シグナル銘柄を空売りすれば利益が出る。"valid" と判定。
  この場合「ショートエッジとして有効（超過リターン -N% → ショートで +N% の利益）」と明記すること。
- **非有意** → どちらの方向にもエッジなし。"invalid" または "needs_review" と判定。
- **矛盾する表現は避けてください**: 「有意だがエッジがない」のような表現はNG。
  有意なら必ずどちらかの方向のエッジがある。方向を明記した上で一貫した評価をすること。

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


BEST_ANALYSIS_PROMPT = """\
あなたは日本株市場の定量分析研究者です。
以下の研究結果を分析し、ベスト結果のパラメータがなぜ有効だったのかを詳しく言語化してください。

## 投資仮説
{hypothesis}

## 研究の全体像
Phase 1（探索）で{n_phase1}通り、Phase 3（グリッドサーチ）で{n_phase3}通り、合計{n_total}通りのパラメータ設定を試行しました。

## 全イテレーション結果サマリー
{iterations_summary}

## グリッド設計（Phase 2）
{grid_spec_text}

## ベスト結果の詳細
- アプローチ名: {best_approach}
- パラメータ設定:
```json
{best_config}
```
- 統計結果:
```json
{best_stats}
```

## 回答してほしいこと（Markdown で記述）
1. **パラメータ設計のロジック**: なぜこのパラメータの組み合わせが選ばれたのか。仮説との対応を説明
2. **効いたパラメータの分析**: どのパラメータが結果に特に寄与したか。Phase 1の探索結果やグリッドサーチの傾向から根拠を示す
3. **効かなかったアプローチとの比較**: 他のアプローチが劣った理由の考察
4. **市場メカニズムの解釈**: なぜこの条件で超過リターンが発生するのか、市場の構造・行動ファイナンスの観点から説明
5. **実運用への示唆**: この結果を実際のトレードに活かすならどうするか。注意点やリスク

Markdownのみで回答してください。JSONは不要です。
"""


NEXT_PARAM_SUGGESTION_PROMPT = """\
あなたは日本株市場の定量分析システムの開発者兼研究者です。
以下の研究結果を踏まえ、「現在のパラメータ体系では測れていないが、この仮説の検証にはあると良さそうな条件・指標」を提案してください。

## 投資仮説
{hypothesis}

## 現在利用可能なパラメータ体系
{available_params}

## ベスト結果のパラメータ設定
```json
{best_config}
```

## ベスト結果の統計
- シグナル数: {n_signals}
- 超過リターン: {mean_excess}
- p値: {p_value}
- 効果量: {cohens_d}

## 全イテレーション結果サマリー
{iterations_summary}

## AI分析（なぜこのパラメータが効いたか）
{best_analysis_excerpt}

## 回答してほしいこと

以下の形式でMarkdownを出力してください。

### 1. 現在のパラメータ体系の限界
今回の仮説を検証するにあたり、現在のSignalConfigでは捉えきれていない側面を指摘してください。

### 2. 追加すると有効そうな新パラメータ提案
各提案について以下を記述:
- **パラメータ名**（英語のフィールド名案）
- **概要**: 何を測る指標か
- **計算方法**: どのデータを使ってどう算出するか
- **仮説との関連**: なぜこの仮説の検証に有効か
- **実装難易度**: 低/中/高（J-Quants APIで取得可能なデータで実現できるか）

3〜5個程度提案してください。実現可能性が高く、かつインパクトが大きい順に並べてください。

### 3. 既存パラメータの組み合わせで代替できないか
新パラメータを追加せずとも、既存パラメータの工夫（組み合わせ・閾値設定）で部分的に代替できるアイデアがあれば記述してください。

Markdownのみで回答してください。JSONは不要です。
"""


class AiInterpreter:
    """AIを使用して分析結果を解釈・評価"""

    def __init__(self, ai_client: Any):
        """
        Args:
            ai_client: AIモデルクライアント。send_message(prompt) -> str を持つオブジェクト。
        """
        self.ai_client = ai_client

    def analyze_best_result(
        self,
        hypothesis: str,
        iterations: list,
        best_iteration,
        grid_spec: dict | None = None,
    ) -> str:
        """ベスト結果に対して、全イテレーション文脈を踏まえた深い分析を行う

        Returns:
            Markdown形式の分析テキスト
        """
        # イテレーションサマリー構築
        phase1_its = [it for it in iterations if it.phase == 1]
        phase3_its = [it for it in iterations if it.phase == 3]

        lines = []
        for it in iterations:
            bt = it.backtest_result
            if "error" in bt:
                lines.append(
                    f"- #{it.iteration} [Phase {it.phase}] {it.approach_name}: エラー"
                )
                continue
            stats = bt.get("statistics", {})
            bd = bt.get("backtest", bt)
            n_valid = bd.get("n_valid_signals", 0)
            me = stats.get("excess_mean") or bd.get("mean_excess_return") or 0
            pv = stats.get("p_value") or 1.0
            cd = stats.get("cohens_d") or 0
            lines.append(
                f"- #{it.iteration} [Phase {it.phase}] {it.approach_name}: "
                f"シグナル{n_valid}件, 超過リターン{me:+.2%}, p={pv:.4f}, d={cd:+.3f}"
            )
        iterations_summary = "\n".join(lines)

        # グリッド設計テキスト
        if grid_spec:
            grid_parts = []
            if grid_spec.get("selected_approach"):
                grid_parts.append(f"ベースアプローチ: {grid_spec['selected_approach']}")
            if grid_spec.get("analysis"):
                grid_parts.append(f"AI分析: {grid_spec['analysis']}")
            if grid_spec.get("sweep_parameters"):
                grid_parts.append(f"スイープパラメータ: {json.dumps(grid_spec['sweep_parameters'], ensure_ascii=False)}")
            if grid_spec.get("reasoning"):
                grid_parts.append(f"設計理由: {grid_spec['reasoning']}")
            grid_spec_text = "\n".join(grid_parts)
        else:
            grid_spec_text = "（グリッド設計なし）"

        # ベスト結果
        best_bt = best_iteration.backtest_result
        best_stats = best_bt.get("statistics", {})
        best_config_json = json.dumps(best_iteration.signal_config_dict, ensure_ascii=False, indent=2)
        best_stats_json = json.dumps(self._compact(best_stats), ensure_ascii=False, indent=2)

        prompt = BEST_ANALYSIS_PROMPT.format(
            hypothesis=hypothesis,
            n_phase1=len(phase1_its),
            n_phase3=len(phase3_its),
            n_total=len(iterations),
            iterations_summary=iterations_summary,
            grid_spec_text=grid_spec_text,
            best_approach=best_iteration.approach_name or "N/A",
            best_config=best_config_json,
            best_stats=best_stats_json,
        )

        try:
            response = self.ai_client.send_message(prompt)
            if response:
                return response.strip()
            return "（AI分析を取得できませんでした）"
        except Exception as e:
            logger.error("ベスト結果分析エラー: %s", e)
            return f"（AI分析エラー: {e}）"

    def suggest_next_parameters(
        self,
        hypothesis: str,
        iterations: list,
        best_iteration,
        best_analysis: str = "",
        available_params_schema: str = "",
    ) -> str:
        """研究結果を踏まえ、現在のパラメータ体系にない新しい条件・指標を提案する

        Returns:
            Markdown形式の提案テキスト
        """
        # イテレーションサマリー
        lines = []
        for it in iterations:
            bt = it.backtest_result
            if "error" in bt:
                lines.append(f"- #{it.iteration} {it.approach_name}: エラー")
                continue
            stats = bt.get("statistics", {})
            bd = bt.get("backtest", bt)
            n_valid = bd.get("n_valid_signals", 0)
            me = stats.get("excess_mean") or bd.get("mean_excess_return") or 0
            pv = stats.get("p_value") or 1.0
            cd = stats.get("cohens_d") or 0
            lines.append(
                f"- #{it.iteration} {it.approach_name}: "
                f"シグナル{n_valid}件, 超過リターン{me:+.2%}, p={pv:.4f}, d={cd:+.3f}"
            )
        iterations_summary = "\n".join(lines)

        # ベスト結果
        best_bt = best_iteration.backtest_result
        best_stats = best_bt.get("statistics", {})
        best_bd = best_bt.get("backtest", best_bt)
        n_signals = best_bd.get("n_valid_signals", 0)
        mean_excess = best_stats.get("excess_mean") or best_bd.get("mean_excess_return") or 0
        p_value = best_stats.get("p_value") or 1.0
        cohens_d = best_stats.get("cohens_d") or 0

        best_config_json = json.dumps(best_iteration.signal_config_dict, ensure_ascii=False, indent=2)

        prompt = NEXT_PARAM_SUGGESTION_PROMPT.format(
            hypothesis=hypothesis,
            available_params=available_params_schema,
            best_config=best_config_json,
            n_signals=n_signals,
            mean_excess=f"{mean_excess:+.2%}",
            p_value=f"{p_value:.4f}",
            cohens_d=f"{cohens_d:+.3f}",
            iterations_summary=iterations_summary,
            best_analysis_excerpt=best_analysis[:2000] if best_analysis else "（なし）",
        )

        try:
            response = self.ai_client.send_message(prompt)
            if response:
                return response.strip()
            return "（AI提案を取得できませんでした）"
        except Exception as e:
            logger.error("次パラメータ提案エラー: %s", e)
            return f"（AI提案エラー: {e}）"

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
        """計画の要約テキストを生成（コード生成プラン・パラメータ選択プラン両対応）"""
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
        # パラメータ選択プランの場合、signal_configを表示
        params = plan.get("parameters", {})
        if params.get("signal_config"):
            parts.append(f"シグナル設定: {json.dumps(params['signal_config'], ensure_ascii=False)}")
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

    def _parse_response(self, response: str | None) -> dict:
        """AIレスポンスからJSONを抽出"""
        if not response:
            raise ValueError("AIから空の応答を受信しました")
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
