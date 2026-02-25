"""AIパラメータ選択モジュール

仮説テキストからSignalConfigのパラメータをJSON出力させる。
コード生成は行わず、パラメータ空間の選択のみをAIに委任する。

イテレーション調整: 前回結果を踏まえてパラメータを微調整し再実行を繰り返す。
"""

import json
import logging
from dataclasses import dataclass, fields
from typing import Any

from core.signal_generator import SignalConfig

logger = logging.getLogger(__name__)


@dataclass
class ParameterSelectionResult:
    """AIが選択したパラメータとその理由"""
    signal_config_dict: dict            # SignalConfigに渡すフィールド
    universe_adjustments: dict | None   # ユニバース調整（将来拡張用）
    reasoning: str                      # パラメータ選択理由
    hypothesis_mapping: str             # 仮説とパラメータの対応説明
    unmappable_aspects: list[str]       # テスト不可能な側面


# ======================================================================
# SignalConfig ↔ dict 変換ヘルパー
# ======================================================================

def dict_to_signal_config(d: dict) -> SignalConfig:
    """辞書からSignalConfigを生成する（未知キーは無視）"""
    valid_fields = {f.name for f in fields(SignalConfig)}
    filtered = {k: v for k, v in d.items() if k in valid_fields}
    return SignalConfig(**filtered)


def signal_config_to_dict(cfg: SignalConfig) -> dict:
    """SignalConfigをデフォルト値と異なるフィールドのみの辞書に変換"""
    default = SignalConfig()
    result = {}
    for f in fields(cfg):
        val = getattr(cfg, f.name)
        default_val = getattr(default, f.name)
        if val != default_val:
            result[f.name] = val
    return result


# ======================================================================
# プロンプトテンプレート
# ======================================================================

_SIGNAL_CONFIG_SCHEMA = """\
SignalConfigの全パラメータ一覧（None/False/デフォルト値のものは無効）:

## テクニカル
- consecutive_bullish_days: int|None  連続陽線日数（例: 3 → 3日連続陽線）
- consecutive_bearish_days: int|None  連続陰線日数
- volume_surge_ratio: float|None  出来高倍率閾値（例: 2.0 → 過去20日平均の2倍）
- volume_surge_window: int  出来高MA期間（デフォルト20）
- price_vs_ma25: "above"|"below"|null  終値と25日移動平均の関係
- price_vs_ma75: "above"|"below"|null  終値と75日移動平均の関係
- price_vs_ma200: "above"|"below"|null  終値と200日移動平均の関係
- ma_deviation_pct: float|None  移動平均乖離率(%)
- ma_deviation_window: int  乖離率の基準MA期間（デフォルト25）
- rsi_window: int  RSI期間（デフォルト14）
- rsi_lower: float|None  RSI下限（例: 30 → RSI<30で売られすぎ）
- rsi_upper: float|None  RSI上限（例: 70 → RSI>70で買われすぎ）
- bb_window: int  ボリンジャーバンド期間（デフォルト20）
- bb_std: float  BB標準偏差倍数（デフォルト2.0）
- bb_buy_below_lower: bool  BB下限タッチで買い（デフォルトFalse）
- ma_cross_short: int|None  ゴールデンクロス/デッドクロスの短期MA
- ma_cross_long: int|None  ゴールデンクロス/デッドクロスの長期MA
- ma_cross_type: "golden_cross"|"dead_cross"  クロス方向（デフォルト"golden_cross"）
- macd_fast: int|None  MACD短期EMA
- macd_slow: int|None  MACD長期EMA
- macd_signal: int  MACDシグナル（デフォルト9）
- atr_window: int  ATR期間（デフォルト14）
- atr_max: float|None  ATRフィルター（%）

## 一目均衡表
- ichimoku_cloud: "above"|"below"|null  雲の上/下
- ichimoku_tenkan_above_kijun: bool  転換線>基準線（デフォルトFalse）

## セクター相対強度
- sector_relative_strength_min: float|None  パーセンタイル下限(0-100)
- sector_relative_lookback: int  ルックバック期間（デフォルト20）

## 信用取引（データは週次発表。水準フィルターと前週比変化率フィルターの両方が使用可能）
- margin_type: "combined"|"standard"|"negotiable"  信用取引データ種別（デフォルト"combined"）
- margin_ratio_min: float|None  貸借倍率下限（水準）
- margin_ratio_max: float|None  貸借倍率上限（水準）
- margin_buy_change_pct_min: float|None  信用買い残の前週比変化率 下限（%。例: 10.0 → 前週比+10%以上増加）
- margin_buy_change_pct_max: float|None  信用買い残の前週比変化率 上限（%。例: -5.0 → 前週比5%以上減少）
- margin_sell_change_pct_min: float|None  信用売り残の前週比変化率 下限（%）
- margin_sell_change_pct_max: float|None  信用売り残の前週比変化率 上限（%）
- margin_ratio_change_pct_min: float|None  貸借倍率の前週比変化率 下限（%）
- margin_ratio_change_pct_max: float|None  貸借倍率の前週比変化率 上限（%）
- short_selling_ratio_max: float|None  空売り比率上限
- margin_buy_turnover_days_min: float|None  買い残回転日数 下限（買い残÷20日平均出来高）
- margin_buy_turnover_days_max: float|None  買い残回転日数 上限
- margin_sell_turnover_days_min: float|None  売り残回転日数 下限
- margin_sell_turnover_days_max: float|None  売り残回転日数 上限
- margin_buy_vol_ratio_min: float|None  買い残対出来高比率 下限（買い残÷当日出来高）
- margin_buy_vol_ratio_max: float|None  買い残対出来高比率 上限
- margin_sell_vol_ratio_min: float|None  売り残対出来高比率 下限
- margin_sell_vol_ratio_max: float|None  売り残対出来高比率 上限
- margin_buy_vol_ratio_change_pct_min: float|None  買い残対出来高比率 前週比変化率 下限(%)
- margin_buy_vol_ratio_change_pct_max: float|None  買い残対出来高比率 前週比変化率 上限(%)
- margin_sell_vol_ratio_change_pct_min: float|None  売り残対出来高比率 前週比変化率 下限(%)
- margin_sell_vol_ratio_change_pct_max: float|None  売り残対出来高比率 前週比変化率 上限(%)

## ポジション管理
- holding_period_days: int  測定期間（デフォルト20営業日）
- signal_logic: "AND"|"OR"  シグナル結合ロジック（デフォルト"AND"）
"""

SELECT_PROMPT = """\
あなたは日本株市場の定量分析研究者です。
以下の投資仮説を検証するために、テクニカルシグナルのパラメータを選択してください。

## 投資仮説
{hypothesis}

## ユニバース
{universe_desc}

## 分析期間
{start_date} 〜 {end_date}

## 利用可能なパラメータ
{schema}

## 重要な注意事項
- 仮説に関連するパラメータのみをセットしてください。関係ないパラメータはデフォルト値のまま省略してください。
- 条件を厳しくしすぎないでください。シグナルが0件になるとテストできません。
  - 例: consecutive_bullish_days=3 と volume_surge_ratio=2.0 を AND で組み合わせると厳しすぎる可能性があります。
  - 条件が3つ以上の場合は signal_logic="OR" も検討してください。
- holding_period_days は仮説の時間軸に合わせてください（短期仮説なら5-10日、中期なら20日、長期なら60日）。
- テスト不可能な側面（このパラメータ体系では表現できない条件）を明記してください。

## 出力形式
以下のJSON形式のみを出力してください。

```json
{{
    "signal_config": {{
        "パラメータ名": 値,
        ...
    }},
    "reasoning": "パラメータ選択の理由（2-3文）",
    "hypothesis_mapping": "仮説のどの部分がどのパラメータに対応するかの説明",
    "unmappable_aspects": ["テスト不可能な側面1", "テスト不可能な側面2"]
}}
```
"""

ADJUST_PROMPT = """\
あなたは日本株市場の定量分析研究者です。
前回のバックテスト結果を踏まえて、パラメータを調整してください。

## 投資仮説
{hypothesis}

## 現在のパラメータ
```json
{current_config}
```

## 前回の結果サマリー
- シグナル数: {n_signals}
- 有効シグナル数: {n_valid}
- 平均超過リターン: {mean_excess:.4f} ({mean_excess_pct:.2f}%)
- p値: {p_value:.4f}
- 効果量 (Cohen's d): {cohens_d:.4f}
- 勝率: {win_rate:.1%}
- 統計的有意: {is_significant}

## 過去のイテレーション履歴
{iteration_history}

## 現在のイテレーション
{iteration}/{max_iterations} 回目

## 利用可能なパラメータ
{schema}

## 調整ガイドライン
- シグナル数が20未満: 条件を緩和（閾値を緩める、条件を減らす、signal_logic="OR"に変更）
- シグナル数が500超: 条件を追加・強化して絞り込む
- 有意でない（p > 0.05）: パラメータ微調整（holding_period_days変更、閾値微調整）
- 既に十分良い結果 or これ以上の改善見込みなし: action="stop" を返す

## 重要: 多様な条件を探索すること
- 閾値の微調整だけではなく、**条件の追加・削除・入れ替え**を積極的に行ってください
- 前回と同じ指標の閾値を少し変えるだけの調整は避け、**別のテクニカル指標の組み合わせ**を試してください
- 例: RSI条件がうまくいかない → RSIを削除してMACD・ボリンジャーバンド・一目均衡表などに切り替える
- 例: holding_period_daysを5/10/20/60と大きく変えて時間軸の効果を検証する
- 例: signal_logicをAND→ORに変更して条件の緩和度を大幅に変える
- 各イテレーションで「前回とは明確に異なるアプローチ」を試すことが重要です
- changes_descriptionには、前回から何を変えたか・なぜ変えたかを具体的に書いてください

## 出力形式
改善する場合:
```json
{{
    "action": "adjust",
    "signal_config": {{
        "パラメータ名": 値,
        ...
    }},
    "reasoning": "調整理由",
    "changes_description": "前回からの変更点"
}}
```

これ以上の改善が見込めない場合:
```json
{{
    "action": "stop",
    "reasoning": "停止理由"
}}
```
"""


class AiParameterSelector:
    """AIを使用して仮説からSignalConfigパラメータを選択する"""

    def __init__(self, ai_client: Any):
        self.ai_client = ai_client

    def select_parameters(
        self,
        hypothesis: str,
        universe_desc: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> ParameterSelectionResult:
        """仮説からSignalConfigパラメータを選択する

        Returns:
            ParameterSelectionResult
        """
        prompt = SELECT_PROMPT.format(
            hypothesis=hypothesis,
            universe_desc=universe_desc or "全銘柄",
            start_date=start_date or "指定なし",
            end_date=end_date or "指定なし",
            schema=_SIGNAL_CONFIG_SCHEMA,
        )

        response = self.ai_client.send_message(prompt)
        parsed = self._parse_json_response(response)

        signal_config_dict = parsed.get("signal_config", {})

        # バリデーション: 少なくとも1つのシグナル条件が必要
        test_cfg = dict_to_signal_config(signal_config_dict)
        if not test_cfg.has_any_signal():
            logger.warning("AIが有効なシグナル条件を選択しませんでした。デフォルトを追加します。")
            signal_config_dict["consecutive_bullish_days"] = 3

        return ParameterSelectionResult(
            signal_config_dict=signal_config_dict,
            universe_adjustments=parsed.get("universe_adjustments"),
            reasoning=parsed.get("reasoning", ""),
            hypothesis_mapping=parsed.get("hypothesis_mapping", ""),
            unmappable_aspects=parsed.get("unmappable_aspects", []),
        )

    def adjust_parameters(
        self,
        hypothesis: str,
        current_config_dict: dict,
        result_summary: dict,
        iteration: int,
        max_iterations: int,
        previous_iterations: list[dict],
    ) -> ParameterSelectionResult | None:
        """前回結果を踏まえてパラメータを調整する

        Returns:
            ParameterSelectionResult or None (これ以上改善不要の場合)
        """
        # イテレーション履歴を整形
        history_lines = []
        for prev in previous_iterations:
            it = prev.get("iteration", "?")
            ns = prev.get("n_signals", 0)
            me = prev.get("mean_excess", 0)
            pv = prev.get("p_value", 1.0)
            cd = prev.get("cohens_d", 0)
            changes = prev.get("changes_description", "初回")
            history_lines.append(
                f"  - イテレーション{it}: シグナル{ns}件, "
                f"超過リターン{me:.4f}, p={pv:.4f}, d={cd:.4f} [{changes}]"
            )
        iteration_history = "\n".join(history_lines) if history_lines else "（なし）"

        n_signals = result_summary.get("n_signals", 0)
        n_valid = result_summary.get("n_valid", 0)
        mean_excess = result_summary.get("mean_excess", 0.0)
        p_value = result_summary.get("p_value", 1.0)
        cohens_d = result_summary.get("cohens_d", 0.0)
        win_rate = result_summary.get("win_rate", 0.0)
        is_significant = result_summary.get("is_significant", False)

        prompt = ADJUST_PROMPT.format(
            hypothesis=hypothesis,
            current_config=json.dumps(current_config_dict, ensure_ascii=False, indent=2),
            n_signals=n_signals,
            n_valid=n_valid,
            mean_excess=mean_excess,
            mean_excess_pct=mean_excess * 100,
            p_value=p_value,
            cohens_d=cohens_d,
            win_rate=win_rate,
            is_significant="はい" if is_significant else "いいえ",
            iteration_history=iteration_history,
            iteration=iteration,
            max_iterations=max_iterations,
            schema=_SIGNAL_CONFIG_SCHEMA,
        )

        response = self.ai_client.send_message(prompt)
        parsed = self._parse_json_response(response)

        action = parsed.get("action", "adjust")

        if action == "stop":
            logger.info("AI判断: これ以上の改善なし — %s", parsed.get("reasoning", ""))
            return None

        signal_config_dict = parsed.get("signal_config", {})
        test_cfg = dict_to_signal_config(signal_config_dict)
        if not test_cfg.has_any_signal():
            logger.warning("調整後のパラメータに有効なシグナル条件がありません。停止します。")
            return None

        return ParameterSelectionResult(
            signal_config_dict=signal_config_dict,
            universe_adjustments=None,
            reasoning=parsed.get("reasoning", ""),
            hypothesis_mapping="",
            unmappable_aspects=[],
        )

    def _parse_json_response(self, response: str | None) -> dict:
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
