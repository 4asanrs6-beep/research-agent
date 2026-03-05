"""Phase 1: スター株共通点発見 + 追加スター株発見 + 初動特定"""

import logging
import random
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from .config import OnsetDetectorConfig

logger = logging.getLogger(__name__)

# 82特徴量キー（A〜I カテゴリ、マルチウィンドウ展開）
WIDE_FEATURE_KEYS = [
    # A: 出来高ダイナミクス (7 既存 + 7 追加 = 14)
    "vol_ratio_5d_20d", "vol_ratio_5d_60d", "vol_surge_count_10d",
    "up_volume_ratio_10d", "quiet_accum_rate_20d", "vol_acceleration", "vpin_5d",
    "vol_ratio_5d_40d", "vol_ratio_10d_20d", "vol_ratio_10d_40d",
    "vol_surge_count_5d", "vol_surge_count_20d",
    "up_volume_ratio_5d", "up_volume_ratio_20d",
    "vpin_10d",
    # B: 価格/リターン (6 既存 + 8 追加 = 14)
    "ret_5d", "ret_20d", "up_days_ratio_10d", "max_gap_up_5d",
    "higher_lows_slope_10d", "range_position_20d",
    "ret_3d", "ret_10d", "ret_40d",
    "up_days_ratio_5d", "up_days_ratio_20d",
    "max_gap_up_10d",
    "higher_lows_slope_5d", "higher_lows_slope_20d",
    "range_position_10d", "range_position_40d",
    # C: ボラティリティ・レジーム (4 既存 + 4 追加 = 8)
    "atr_ratio_5d_20d", "bb_width_pctile_60d", "intraday_range_ratio_5d",
    "realized_vol_5d_vs_20d",
    "atr_ratio_5d_40d", "atr_ratio_10d_20d",
    "bb_width_pctile_120d",
    "intraday_range_ratio_10d",
    "realized_vol_5d_vs_40d", "realized_vol_10d_vs_20d",
    # D: トレンド/OBV (5 既存 + 7 追加 = 12)
    "obv_slope_10d", "obv_divergence", "ma5_ma25_gap", "price_vs_ma25_pct",
    "consecutive_up_days",
    "obv_slope_5d", "obv_slope_20d",
    "obv_divergence_40d",
    "ma25_ma75_gap", "ma5_ma75_gap",
    "price_vs_ma5_pct", "price_vs_ma75_pct", "price_vs_ma200_pct",
    # E: クロスセクショナル (4 既存 + 3 追加 = 7)
    "sector_rel_ret_10d", "topix_beta_20d", "residual_vol_ratio",
    "vol_vs_market_vol",
    "sector_rel_ret_5d", "sector_rel_ret_20d",
    "topix_beta_40d",
    # F: 信用取引 (8) — 変更なし
    "margin_ratio", "margin_buy_change_pct", "margin_ratio_change_pct",
    "margin_buy_turnover_days", "margin_buy_vol_ratio", "margin_net_position",
    "margin_divergence", "has_margin_data",
    # G: テクニカル追加 (6 既存 + 2 追加 = 8)
    "macd_histogram", "stochastic_k", "williams_r",
    "cci_20d", "ma_deviation_25d", "ma_deviation_75d",
    "cci_10d",
    "ma_deviation_5d", "ma_deviation_200d",
    # H: 流動性 (3 既存 + 3 追加 = 6)
    "amihud_illiquidity_20d", "turnover_change_10d_20d", "spread_proxy_5d",
    "amihud_illiquidity_10d",
    "turnover_change_5d_20d", "turnover_change_5d_10d",
    "spread_proxy_10d",
    # I: 価格パターン (3 既存 + 2 追加 = 5)
    "gap_frequency_20d", "higher_highs_ratio_10d", "proximity_52w_high",
    "gap_frequency_10d",
    "higher_highs_ratio_5d", "higher_highs_ratio_20d",
]

WIDE_FEATURE_LABELS_JP = {
    # A: 出来高ダイナミクス
    "vol_ratio_5d_20d": "出来高比(5/20日)",
    "vol_ratio_5d_60d": "出来高比(5/60日)",
    "vol_surge_count_10d": "出来高急増回数(10日)",
    "up_volume_ratio_10d": "上昇日出来高比(10日)",
    "quiet_accum_rate_20d": "静的蓄積率(20日)",
    "vol_acceleration": "出来高加速度",
    "vpin_5d": "注文偏り度(5日)",
    "vol_ratio_5d_40d": "出来高比(5/40日)",
    "vol_ratio_10d_20d": "出来高比(10/20日)",
    "vol_ratio_10d_40d": "出来高比(10/40日)",
    "vol_surge_count_5d": "出来高急増回数(5日)",
    "vol_surge_count_20d": "出来高急増回数(20日)",
    "up_volume_ratio_5d": "上昇日出来高比(5日)",
    "up_volume_ratio_20d": "上昇日出来高比(20日)",
    "vpin_10d": "注文偏り度(10日)",
    # B: 価格/リターン
    "ret_5d": "リターン(5日)",
    "ret_20d": "リターン(20日)",
    "up_days_ratio_10d": "上昇日比率(10日)",
    "max_gap_up_5d": "最大ギャップアップ(5日)",
    "higher_lows_slope_10d": "安値切上り傾き(10日)",
    "range_position_20d": "レンジ位置(20日)",
    "ret_3d": "リターン(3日)",
    "ret_10d": "リターン(10日)",
    "ret_40d": "リターン(40日)",
    "up_days_ratio_5d": "上昇日比率(5日)",
    "up_days_ratio_20d": "上昇日比率(20日)",
    "max_gap_up_10d": "最大ギャップアップ(10日)",
    "higher_lows_slope_5d": "安値切上り傾き(5日)",
    "higher_lows_slope_20d": "安値切上り傾き(20日)",
    "range_position_10d": "レンジ位置(10日)",
    "range_position_40d": "レンジ位置(40日)",
    # C: ボラティリティ・レジーム
    "atr_ratio_5d_20d": "ATR比(5/20日)",
    "bb_width_pctile_60d": "価格帯収縮度(60日)",
    "intraday_range_ratio_5d": "日中値幅比(5/20日)",
    "realized_vol_5d_vs_20d": "短期ボラ比(5/20日)",
    "atr_ratio_5d_40d": "ATR比(5/40日)",
    "atr_ratio_10d_20d": "ATR比(10/20日)",
    "bb_width_pctile_120d": "価格帯収縮度(120日)",
    "intraday_range_ratio_10d": "日中値幅比(10/20日)",
    "realized_vol_5d_vs_40d": "短期ボラ比(5/40日)",
    "realized_vol_10d_vs_20d": "短期ボラ比(10/20日)",
    # D: トレンド/OBV
    "obv_slope_10d": "出来高累計傾き(10日)",
    "obv_divergence": "出来高価格連動度(20日)",
    "ma5_ma25_gap": "MA(5-25)乖離率",
    "price_vs_ma25_pct": "対MA25乖離率",
    "consecutive_up_days": "連続上昇日数",
    "obv_slope_5d": "出来高累計傾き(5日)",
    "obv_slope_20d": "出来高累計傾き(20日)",
    "obv_divergence_40d": "出来高価格連動度(40日)",
    "ma25_ma75_gap": "MA(25-75)乖離率",
    "ma5_ma75_gap": "MA(5-75)乖離率",
    "price_vs_ma5_pct": "対MA5乖離率",
    "price_vs_ma75_pct": "対MA75乖離率",
    "price_vs_ma200_pct": "対MA200乖離率",
    # E: クロスセクショナル
    "sector_rel_ret_10d": "セクター相対リターン(10日)",
    "topix_beta_20d": "市場感応度(20日)",
    "residual_vol_ratio": "固有ボラ比(10/60日)",
    "vol_vs_market_vol": "対市場出来高比",
    "sector_rel_ret_5d": "セクター相対リターン(5日)",
    "sector_rel_ret_20d": "セクター相対リターン(20日)",
    "topix_beta_40d": "市場感応度(40日)",
    # F: 信用取引
    "margin_ratio": "貸借倍率",
    "margin_buy_change_pct": "信用買い残変化率",
    "margin_ratio_change_pct": "貸借倍率変化率",
    "margin_buy_turnover_days": "信用買い残回転日数",
    "margin_buy_vol_ratio": "信用買い残/出来高比",
    "margin_net_position": "信用ネットポジション",
    "margin_divergence": "信用買い残乖離",
    "has_margin_data": "信用データ有無",
    # G: テクニカル追加
    "macd_histogram": "MACDヒストグラム",
    "stochastic_k": "過熱度指数K",
    "williams_r": "高値近接指数",
    "cci_20d": "価格乖離指数(20日)",
    "ma_deviation_25d": "MA25乖離率",
    "ma_deviation_75d": "MA75乖離率",
    "cci_10d": "価格乖離指数(10日)",
    "ma_deviation_5d": "MA5乖離率",
    "ma_deviation_200d": "MA200乖離率",
    # H: 流動性
    "amihud_illiquidity_20d": "流動性の低さ(20日)",
    "turnover_change_10d_20d": "売買代金変化(10/20日)",
    "spread_proxy_5d": "スプレッド代理(5日)",
    "amihud_illiquidity_10d": "流動性の低さ(10日)",
    "turnover_change_5d_20d": "売買代金変化(5/20日)",
    "turnover_change_5d_10d": "売買代金変化(5/10日)",
    "spread_proxy_10d": "スプレッド代理(10日)",
    # I: 価格パターン
    "gap_frequency_20d": "ギャップ頻度(20日)",
    "higher_highs_ratio_10d": "高値更新比率(10日)",
    "proximity_52w_high": "52週高値近接度",
    "gap_frequency_10d": "ギャップ頻度(10日)",
    "higher_highs_ratio_5d": "高値更新比率(5日)",
    "higher_highs_ratio_20d": "高値更新比率(20日)",
}

# 各特徴量の平易な日本語説明
WIDE_FEATURE_DESCRIPTIONS_JP = {
    # A: 出来高ダイナミクス
    "vol_ratio_5d_20d": "直近5日の出来高が1ヶ月平均の何倍か（大きいほど出来高急増）",
    "vol_ratio_5d_60d": "直近5日の出来高が3ヶ月平均の何倍か（大きいほど出来高急増）",
    "vol_surge_count_10d": "直近10日で出来高が平均の2倍を超えた日数",
    "up_volume_ratio_10d": "直近10日の取引のうち株価が上がった日の出来高割合（買い優勢度）",
    "quiet_accum_rate_20d": "株価が横ばいなのに出来高が多い日の比率（機関投資家の静かな買い集め）",
    "vol_acceleration": "直近5日の出来高が前の5日より増えた倍率",
    "vpin_5d": "直近5日の買い注文と売り注文の偏り度合い（高いほど一方向に集中）",
    "vol_ratio_5d_40d": "直近5日の出来高が2ヶ月平均の何倍か（大きいほど出来高急増）",
    "vol_ratio_10d_20d": "直近10日の出来高が1ヶ月平均の何倍か（大きいほど出来高急増）",
    "vol_ratio_10d_40d": "直近10日の出来高が2ヶ月平均の何倍か（大きいほど出来高急増）",
    "vol_surge_count_5d": "直近5日で出来高が平均の2倍を超えた日数",
    "vol_surge_count_20d": "直近20日で出来高が平均の2倍を超えた日数",
    "up_volume_ratio_5d": "直近5日の取引のうち株価が上がった日の出来高割合（買い優勢度）",
    "up_volume_ratio_20d": "直近20日の取引のうち株価が上がった日の出来高割合（買い優勢度）",
    "vpin_10d": "直近10日の買い注文と売り注文の偏り度合い（高いほど一方向に集中）",
    # B: 価格/リターン
    "ret_5d": "直近5営業日（1週間）の株価騰落率",
    "ret_20d": "直近20営業日（約1ヶ月）の株価騰落率",
    "up_days_ratio_10d": "直近10日のうち株価が上昇した日の割合",
    "max_gap_up_5d": "直近5日で最も大きかった寄り付き窓開け上昇幅",
    "higher_lows_slope_10d": "直近10日の安値の切り上がり速度（正=下値が堅固に上昇）",
    "range_position_20d": "直近20日の高値-安値の範囲内での現在値の位置（1=高値圏、0=安値圏）",
    "ret_3d": "直近3営業日の株価騰落率",
    "ret_10d": "直近10営業日（2週間）の株価騰落率",
    "ret_40d": "直近40営業日（約2ヶ月）の株価騰落率",
    "up_days_ratio_5d": "直近5日のうち株価が上昇した日の割合",
    "up_days_ratio_20d": "直近20日のうち株価が上昇した日の割合",
    "max_gap_up_10d": "直近10日で最も大きかった寄り付き窓開け上昇幅",
    "higher_lows_slope_5d": "直近5日の安値の切り上がり速度（正=下値が堅固に上昇）",
    "higher_lows_slope_20d": "直近20日の安値の切り上がり速度（正=下値が堅固に上昇）",
    "range_position_10d": "直近10日の高値-安値の範囲内での現在値の位置（1=高値圏、0=安値圏）",
    "range_position_40d": "直近40日の高値-安値の範囲内での現在値の位置（1=高値圏、0=安値圏）",
    # C: ボラティリティ・レジーム
    "atr_ratio_5d_20d": "直近5日の1日の値動き幅が20日平均より大きいか（ボラ拡大度）",
    "bb_width_pctile_60d": "価格帯（ボリンジャーバンド幅）の過去60日中での細さのランク（低=収縮中）",
    "intraday_range_ratio_5d": "直近5日の日中の値幅が20日平均より大きいか",
    "realized_vol_5d_vs_20d": "直近5日の日々の値動きの激しさが20日平均より大きいか",
    "atr_ratio_5d_40d": "直近5日の1日の値動き幅が40日平均より大きいか（ボラ拡大度）",
    "atr_ratio_10d_20d": "直近10日の1日の値動き幅が20日平均より大きいか（ボラ拡大度）",
    "bb_width_pctile_120d": "価格帯（ボリンジャーバンド幅）の過去120日中での細さのランク（低=収縮中）",
    "intraday_range_ratio_10d": "直近10日の日中の値幅が20日平均より大きいか",
    "realized_vol_5d_vs_40d": "直近5日の日々の値動きの激しさが40日平均より大きいか",
    "realized_vol_10d_vs_20d": "直近10日の日々の値動きの激しさが20日平均より大きいか",
    # D: トレンド/OBV
    "obv_slope_10d": "出来高の累積値（上昇日プラス・下落日マイナス）の10日トレンド方向",
    "obv_divergence": "株価と出来高累計値の20日連動度（高い=出来高が株価上昇を裏付け）",
    "ma5_ma25_gap": "5日移動平均が25日移動平均より何%上にあるか",
    "price_vs_ma25_pct": "現在株価が25日移動平均より何%上にあるか",
    "consecutive_up_days": "直近の連続上昇日数",
    "obv_slope_5d": "出来高の累積値（上昇日プラス・下落日マイナス）の5日トレンド方向",
    "obv_slope_20d": "出来高の累積値（上昇日プラス・下落日マイナス）の20日トレンド方向",
    "obv_divergence_40d": "株価と出来高累計値の40日連動度（高い=出来高が株価上昇を裏付け）",
    "ma25_ma75_gap": "25日移動平均が75日移動平均より何%上にあるか",
    "ma5_ma75_gap": "5日移動平均が75日移動平均より何%上にあるか",
    "price_vs_ma5_pct": "現在株価が5日移動平均より何%上にあるか",
    "price_vs_ma75_pct": "現在株価が75日移動平均より何%上にあるか（中期トレンドとの乖離）",
    "price_vs_ma200_pct": "現在株価が200日移動平均より何%上にあるか（長期トレンドとの乖離）",
    # E: クロスセクショナル
    "sector_rel_ret_10d": "同業他社の平均に対してその銘柄だけ何%余分に上昇したか（10日）",
    "topix_beta_20d": "市場全体（TOPIX）が動いたときのその銘柄の感応度（高い=市場と同方向に大きく動く）",
    "residual_vol_ratio": "市場の動きを除いた銘柄固有の値動きの激しさ（直近vs長期）",
    "vol_vs_market_vol": "市場全体と比べた出来高の多さ",
    "sector_rel_ret_5d": "同業他社の平均に対してその銘柄だけ何%余分に上昇したか（5日）",
    "sector_rel_ret_20d": "同業他社の平均に対してその銘柄だけ何%余分に上昇したか（20日）",
    "topix_beta_40d": "市場全体（TOPIX）が動いたときのその銘柄の40日感応度（高い=市場と同方向に大きく動く）",
    # F: 信用取引
    "margin_ratio": "信用買い残÷信用売り残の倍率（高い=買い方が優勢）",
    "margin_buy_change_pct": "信用買い残の直近変化率（正=新規信用買いが増加）",
    "margin_ratio_change_pct": "貸借倍率の変化率（正=買い方優勢が強まる）",
    "margin_buy_turnover_days": "信用買い残を一日の出来高で割った返済所要日数（低い=返済が速い）",
    "margin_buy_vol_ratio": "信用買い残を出来高で割った比率",
    "margin_net_position": "信用買い残から信用売り残を引いた差（大きい=買い方優勢）",
    "margin_divergence": "信用買い残と株価の乖離度合い",
    "has_margin_data": "信用取引データが存在するか",
    # G: テクニカル追加
    "macd_histogram": "短期と長期の移動平均の差の変化（正=上昇モメンタムが加速）",
    "stochastic_k": "過去一定期間の高安値の範囲内での現在値の位置（高い=高値圏）",
    "williams_r": "高値に対する現在値の近さ（0に近い=高値圏）",
    "cci_20d": "価格の移動平均からの乖離を標準化した指数（20日、高い=上昇トレンドが強い）",
    "ma_deviation_25d": "現在株価が25日移動平均より何%上にあるか",
    "ma_deviation_75d": "現在株価が75日移動平均より何%上にあるか（長期トレンドとの乖離）",
    "cci_10d": "価格の移動平均からの乖離を標準化した指数（10日、高い=上昇トレンドが強い）",
    "ma_deviation_5d": "現在株価が5日移動平均より何%上にあるか（短期トレンドとの乖離）",
    "ma_deviation_200d": "現在株価が200日移動平均より何%上にあるか（超長期トレンドとの乖離）",
    # H: 流動性
    "amihud_illiquidity_20d": "値動き÷出来高の比率（20日、高い=少ない出来高で大きく動く＝流動性が低い）",
    "turnover_change_10d_20d": "直近10日の売買代金が20日平均の何倍か",
    "spread_proxy_5d": "直近5日の高値と安値の差（高い=価格が跳びやすい＝流動性が低い）",
    "amihud_illiquidity_10d": "値動き÷出来高の比率（10日、高い=少ない出来高で大きく動く＝流動性が低い）",
    "turnover_change_5d_20d": "直近5日の売買代金が20日平均の何倍か",
    "turnover_change_5d_10d": "直近5日の売買代金が10日平均の何倍か",
    "spread_proxy_10d": "直近10日の高値と安値の差（高い=価格が跳びやすい＝流動性が低い）",
    # I: 価格パターン
    "gap_frequency_20d": "直近20日のうち寄り付きで窓開けした日の比率",
    "higher_highs_ratio_10d": "直近10日のうち前日高値を超えた日の割合",
    "proximity_52w_high": "52週高値に対する現在値の近さ（1=52週高値圏）",
    "gap_frequency_10d": "直近10日のうち寄り付きで窓開けした日の比率",
    "higher_highs_ratio_5d": "直近5日のうち前日高値を超えた日の割合",
    "higher_highs_ratio_20d": "直近20日のうち前日高値を超えた日の割合",
}

# 初動シグナルの日本語名
ONSET_SIGNAL_NAMES_JP = {
    "volume_surge": "出来高急増（平均の2倍超）",
    "quiet_accumulation": "静かな買い集め（横ばい中に出来高増）",
    "consecutive_accumulation": "継続的な蓄積",
    "obv_breakout": "出来高累計の急上昇",
    "bb_squeeze": "価格帯の収縮（エネルギー蓄積）",
    "volatility_compression": "値動きの収縮（爆発前の静寂）",
    "higher_lows": "安値の切り上げ（下値堅固）",
    "range_breakout": "高値更新（上抜けブレイク）",
    "ma_crossover": "短期平均線が長期平均線を上抜け",
    "up_volume_dominance": "上昇日の出来高が売り日を圧倒",
}

MULTI_ONSET_SIGNAL_NAMES = [
    "volume_surge", "quiet_accumulation", "consecutive_accumulation",
    "obv_breakout", "bb_squeeze", "volatility_compression",
    "higher_lows", "range_breakout", "ma_crossover", "up_volume_dominance",
]


class OnsetDiscoverer:
    """Phase 1: スター株共通点発見 + 追加発見 + 初動特定"""

    def __init__(self, config: OnsetDetectorConfig):
        self.config = config

    def run(
        self,
        user_star_codes: list[str],
        star_stocks: list[dict],
        all_prices: pd.DataFrame,
        topix: pd.DataFrame,
        listed_stocks: pd.DataFrame,
        progress_callback=None,
        market_caps: dict[str, float] | None = None,
    ) -> dict:
        """Phase 1パイプライン実行

        Parameters
        ----------
        user_star_codes : 元のユーザー指定コード（参照用）
        star_stocks : 解決済みスター株リスト（Step 2の出力）
        all_prices : 時価総額フィルタ済みの全銘柄株価
        topix : TOPIXデータ
        listed_stocks : 上場銘柄情報
        progress_callback : (step, total, msg) コールバック

        Returns
        -------
        dict with keys:
            common_features: 共通特徴量発見結果
            additional_stars: 追加発見されたスター株
            onset_dates: 各スター株の初動情報
            all_stars: 全スター株リスト（ユーザー指定+追加）
            ai_interpretation: AI解釈テキスト
        """
        cfg = self.config
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"

        def _progress(msg):
            if progress_callback:
                progress_callback(msg)

        # --- TOPIX日次リターン ---
        topix_close_col = "close" if "close" in topix.columns else topix.columns[-1]
        topix_sorted = topix.sort_values("date").copy()
        topix_sorted["date"] = pd.to_datetime(topix_sorted["date"])
        topix_ret = topix_sorted[topix_close_col].pct_change()
        topix_ret_series = pd.Series(topix_ret.values, index=topix_sorted["date"].values)

        star_codes = [str(s["code"]) for s in star_stocks]
        logger.info(f"Phase 1開始: スター株 {len(star_codes)}件")

        # --- Step 2.5: 入力スター株の初動特定（先行実行） ---
        # Step 3 で「初動時点の特徴量」を使うために、初動日を先に特定する。
        # 上昇後の最新データで特徴量を計算すると、ret_40d 等が結果を反映して
        # 閾値が非現実的になり母集団で条件合致ゼロとなるため。
        _progress("入力スター株の初動日特定中...")
        star_onset_dates = self._detect_onset_dates(
            star_stocks, all_prices, close_col, progress_callback=_progress,
            topix=topix_sorted,
        )

        # --- Step 3: 共通特徴量発見 ---
        _progress("共通特徴量発見中...")
        common_features = self._find_discriminative_features(
            star_stocks, all_prices, topix_ret_series, listed_stocks, close_col,
            progress_callback=_progress,
            star_onset_dates=star_onset_dates,
        )

        # --- Step 4: 追加スター株発見 ---
        _progress("追加スター株探索中...")
        additional_stars = self._discover_additional_stars(
            common_features, star_codes, all_prices, topix, topix_ret_series,
            listed_stocks, close_col,
            market_caps=market_caps,
        )

        all_stars = list(star_stocks) + additional_stars
        all_star_codes = [str(s["code"]) for s in all_stars]

        # --- Step 4.5: 母集団精度計算（全宇宙×複数時点） ---
        _progress("母集団確率計算中（全銘柄×複数時点）...")
        all_star_codes_set = set(all_star_codes)
        universe_probs = self._compute_universe_precision(
            all_prices=all_prices,
            star_codes=all_star_codes_set,
            best_combos=common_features.get("best_combos", []),
            close_col=close_col,
            topix_ret_series=topix_ret_series,
            progress_callback=_progress,
        )
        # コンボに母集団精度を追記
        for i, c in enumerate(common_features.get("best_combos", [])[:5]):
            if i < len(universe_probs):
                c.update(universe_probs[i])
        # base_rate も母集団ベースに更新
        if universe_probs:
            up = universe_probs[0]
            n_total = up.get("universe_n_total", 0)
            n_stars = len(all_star_codes_set)
            if n_total > 0:
                common_features["base_rate_universe"] = round(n_stars / n_total, 4)
                common_features["n_universe"] = n_total

        # --- Step 4.6: 代替手法の母集団精度計算 ---
        alt_methods = common_features.get("alt_methods", {})
        n_universe_total = common_features.get("n_universe", 0)
        if alt_methods and n_universe_total > 0:
            _progress("代替手法の母集団精度計算中...")
            alt_universe = self._compute_universe_precision_alt(
                all_prices=all_prices,
                star_codes=all_star_codes_set,
                alt_methods=alt_methods,
                signals=common_features.get("signals", []),
                close_col=close_col,
                topix_ret_series=topix_ret_series,
                progress_callback=_progress,
            )
            # 結果を alt_methods に追記
            for method_key, u_result in alt_universe.items():
                if method_key in alt_methods and "best" in alt_methods[method_key]:
                    alt_methods[method_key]["universe"] = u_result
            common_features["alt_methods"] = alt_methods

        # --- Step 5: 初動特定 ---
        _progress("初動日特定中...")
        onset_dates = self._detect_onset_dates(
            all_stars, all_prices, close_col, progress_callback=_progress,
            topix=topix_sorted,
        )

        # --- Step 6: AI解釈 ---
        ai_interpretation = ""
        if cfg.use_ai_interpretation:
            _progress("AI解釈生成中...")
            ai_interpretation = self._generate_ai_interpretation(
                star_stocks, additional_stars, common_features, onset_dates,
            )

        return {
            "common_features": common_features,
            "additional_stars": additional_stars,
            "onset_dates": onset_dates,
            "all_stars": all_stars,
            "ai_interpretation": ai_interpretation,
        }

    # ------------------------------------------------------------------
    # Step 3: 共通特徴量発見
    # ------------------------------------------------------------------
    def _find_discriminative_features(
        self,
        star_stocks: list[dict],
        all_prices: pd.DataFrame,
        topix_ret_series: pd.Series,
        listed_stocks: pd.DataFrame,
        close_col: str,
        progress_callback=None,
        star_onset_dates: dict | None = None,
    ) -> dict:
        """Youden's J最適化 + コンボ探索で判別特徴量を発見"""
        cfg = self.config
        star_codes = set(str(s["code"]) for s in star_stocks)
        all_codes = [str(c) for c in all_prices["code"].unique()]
        non_star_codes = [c for c in all_codes if c not in star_codes]

        # --- サンプル構築 ---
        pos_features = []  # 正例: スター株
        neg_features = []  # 負例: 非スター株

        # 正例: 各スター株の**初動時点**の特徴量
        # 最新データ（上昇後）で計算すると ret_40d 等が結果を反映し
        # 閾値が非現実的→母集団で条件合致ゼロとなるため、
        # 10シグナル方式で特定した初動日までのデータで特徴量を計算する。
        for star in star_stocks:
            code = str(star["code"])
            grp = all_prices[all_prices["code"] == code].sort_values("date")
            if len(grp) < 20:
                continue

            # 初動日が特定されていれば、その日までのデータで特徴量計算
            onset_info = (star_onset_dates or {}).get(code, {})
            onset_date_str = onset_info.get("onset_date", "")

            if onset_date_str and "date" in grp.columns:
                onset_dt = pd.Timestamp(onset_date_str)
                grp_dates = pd.to_datetime(grp["date"])
                mask = grp_dates <= onset_dt
                if mask.sum() >= 20:
                    grp_pre = grp.loc[mask]
                else:
                    grp_pre = grp
            else:
                grp_pre = grp

            feat = self._compute_wide_features(grp_pre, close_col, topix_ret_series)
            if feat is not None:
                pos_features.append(feat)

        if len(pos_features) < 3:
            logger.warning("正例サンプル不足: %d件", len(pos_features))
            return {"signals": [], "combo_signals": [], "best_combos": [],
                    "n_star": len(pos_features), "error": "正例サンプル不足"}

        # 負例: ランダム非スター株
        n_neg = min(cfg.discovery_neg_sample_size, len(non_star_codes))
        neg_sample_codes = random.sample(non_star_codes, n_neg)
        for code in neg_sample_codes:
            grp = all_prices[all_prices["code"] == code].sort_values("date")
            if len(grp) < 20:
                continue
            feat = self._compute_wide_features(grp, close_col, topix_ret_series)
            if feat is not None:
                neg_features.append(feat)

        if len(neg_features) < 10:
            logger.warning("負例サンプル不足: %d件", len(neg_features))
            return {"signals": [], "combo_signals": [], "best_combos": [],
                    "n_star": len(pos_features), "error": "負例サンプル不足"}

        logger.info(f"サンプル: 正例={len(pos_features)}, 負例={len(neg_features)}")

        # --- 特徴量行列 ---
        all_samples = pos_features + neg_features
        labels = np.array([1] * len(pos_features) + [0] * len(neg_features))
        X = np.array([[s.get(k, 0.0) for k in WIDE_FEATURE_KEYS] for s in all_samples])
        X = np.where(np.isfinite(X), X, 0.0)

        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        base_rate = n_pos / len(labels) if len(labels) > 0 else 0.0

        # --- Youden's J最適化 ---
        if progress_callback:
            progress_callback("Youden's J最適化中...")

        signals = []
        for fi, feat_name in enumerate(WIDE_FEATURE_KEYS):
            col = X[:, fi]
            pos_vals = col[labels == 1]
            neg_vals = col[labels == 0]

            best_j = -1
            best_result = None

            for pct in range(10, 91, 10):
                threshold = float(np.percentile(pos_vals, pct))
                pred = col >= threshold
                tp = int((pred & (labels == 1)).sum())
                fp = int((pred & (labels == 0)).sum())
                fn = int((~pred & (labels == 1)).sum())
                tn = int((~pred & (labels == 0)).sum())

                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tpr
                j_stat = tpr - fpr
                f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
                lift = prec / base_rate if base_rate > 0 else 0

                if j_stat > best_j:
                    best_j = j_stat
                    best_result = {
                        "feature": feat_name,
                        "feature_jp": WIDE_FEATURE_LABELS_JP.get(feat_name, feat_name),
                        "threshold": threshold,
                        "direction": ">=",
                        "j_stat": round(j_stat, 4),
                        "tpr": round(tpr, 4),
                        "fpr": round(fpr, 4),
                        "precision": round(prec, 4),
                        "recall": round(recall, 4),
                        "f1": round(f1, 4),
                        "lift": round(lift, 2),
                        "pos_mean": round(float(np.mean(pos_vals)), 4),
                        "neg_mean": round(float(np.mean(neg_vals)), 4),
                        "pos_std": round(float(np.std(pos_vals)), 4),
                        "neg_std": round(float(np.std(neg_vals)), 4),
                        "pos_median": round(float(np.median(pos_vals)), 4),
                        "neg_median": round(float(np.median(neg_vals)), 4),
                    }

            if best_result is not None:
                # verdict
                lift = best_result["lift"]
                j = best_result["j_stat"]
                if lift >= 3.0 and j >= 0.3:
                    best_result["verdict"] = "strong"
                elif lift >= 2.0:
                    best_result["verdict"] = "weak_useful"
                elif lift >= 1.2:
                    best_result["verdict"] = "weak"
                else:
                    best_result["verdict"] = "meaningless"
                signals.append(best_result)

        # Youden's Jでソート
        signals.sort(key=lambda s: s["j_stat"], reverse=True)

        # --- コンボ探索（相関ベース冗長排除） ---
        if progress_callback:
            progress_callback("コンボ探索中（相関分析 + 独立特徴量選別）...")

        top_n = min(15, len(signals))
        top_signals = signals[:top_n]
        thresholds = {s["feature"]: s["threshold"] for s in top_signals}

        # ============================================================
        # 相関行列による冗長特徴量の排除
        # ============================================================
        CORR_THRESHOLD = 0.7  # |corr| >= 0.7 は「同じ情報」とみなす

        feat_indices_all = []
        for s in top_signals:
            feat_indices_all.append(WIDE_FEATURE_KEYS.index(s["feature"]))
        X_top = X[:, feat_indices_all]

        # ペアワイズ相関（Spearman: 非線形関係も捕捉）
        n_feat = len(top_signals)
        corr_matrix = np.zeros((n_feat, n_feat))
        for i in range(n_feat):
            for j in range(i, n_feat):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    continue
                a, b = X_top[:, i], X_top[:, j]
                if np.ptp(a) == 0 or np.ptp(b) == 0:
                    corr_matrix[i, j] = 0.0
                    corr_matrix[j, i] = 0.0
                    continue
                try:
                    c, _ = stats.spearmanr(a, b)
                    c = float(c) if np.isfinite(c) else 0.0
                except Exception:
                    c = 0.0
                corr_matrix[i, j] = c
                corr_matrix[j, i] = c

        # グリーディ選別: J stat順に取り、既選択と |corr| >= 0.7 の特徴量はスキップ
        selected_indices = []  # top_signals内のインデックス
        for i in range(n_feat):
            is_redundant = False
            for j in selected_indices:
                if abs(corr_matrix[i, j]) >= CORR_THRESHOLD:
                    is_redundant = True
                    break
            if not is_redundant:
                selected_indices.append(i)

        independent_signals = [top_signals[i] for i in selected_indices]
        dropped = [top_signals[i]["feature"] for i in range(n_feat) if i not in selected_indices]

        logger.info(
            f"相関フィルタ: {n_feat}特徴量 → {len(independent_signals)}独立特徴量 "
            f"(排除: {dropped})"
        )

        # 相関情報を保存（UI表示用）
        corr_info = {
            "matrix": corr_matrix.tolist(),
            "feature_names": [s["feature"] for s in top_signals],
            "feature_names_jp": [s.get("feature_jp", s["feature"]) for s in top_signals],
            "selected_indices": selected_indices,
            "dropped_features": dropped,
            "threshold": CORR_THRESHOLD,
        }

        # ============================================================
        # ペアの独立性スコア付きコンボ探索
        # ============================================================
        combo_results = []

        # 最低合致銘柄数: 正例の30%以上、最低3件
        min_total_hits = max(3, int(n_pos * 0.3))

        def _make_combo_entry(features_names, thresholds_list, tp, fp, total_pred,
                              diversity_score=0.0):
            prec = tp / total_pred if total_pred > 0 else 0
            recall = tp / n_pos if n_pos > 0 else 0
            lift = prec / base_rate if base_rate > 0 else 0
            reliability = "高" if tp >= 5 else ("中" if tp >= 3 else "低")
            return {
                "features": list(features_names),
                "features_jp": [WIDE_FEATURE_LABELS_JP.get(f, f) for f in features_names],
                "thresholds": list(thresholds_list),
                "directions": [">="] * len(features_names),
                "n_features": len(features_names),
                "n_combo": total_pred,
                "total_hits": total_pred,
                "precision": round(prec, 4),
                "recall": round(recall, 4),
                "lift": round(lift, 2),
                "tp": tp,
                "fp": fp,
                "reliability": reliability,
                "diversity_score": round(diversity_score, 3),
            }

        def _combo_diversity(feat_names):
            """コンボ内の特徴量ペア間の平均独立性（1 - |corr|）"""
            idxs = []
            for fn in feat_names:
                for si, s in enumerate(top_signals):
                    if s["feature"] == fn:
                        idxs.append(si)
                        break
            if len(idxs) < 2:
                return 0.0
            pairs = list(combinations(idxs, 2))
            return sum(1.0 - abs(corr_matrix[i, j]) for i, j in pairs) / len(pairs)

        # --- 独立特徴量のみでコンボ生成（メイン: 高品質） ---
        for (s1, s2) in combinations(independent_signals, 2):
            f1_name, f2_name = s1["feature"], s2["feature"]
            i1 = WIDE_FEATURE_KEYS.index(f1_name)
            i2 = WIDE_FEATURE_KEYS.index(f2_name)
            pred = (X[:, i1] >= s1["threshold"]) & (X[:, i2] >= s2["threshold"])
            tp = int((pred & (labels == 1)).sum())
            fp = int((pred & (labels == 0)).sum())
            total_pred = int(pred.sum())
            if total_pred < min_total_hits:
                continue
            div = _combo_diversity([f1_name, f2_name])
            combo_results.append(
                _make_combo_entry([f1_name, f2_name], [s1["threshold"], s2["threshold"]],
                                  tp, fp, total_pred, div)
            )

        # 3特徴量コンボ（独立特徴量から）
        min_hits_multi = max(3, int(n_pos * 0.15))
        ind_top = independent_signals[:min(10, len(independent_signals))]
        for combo in combinations(ind_top, 3):
            f_names = [s["feature"] for s in combo]
            indices = [WIDE_FEATURE_KEYS.index(f) for f in f_names]
            ths = [s["threshold"] for s in combo]
            pred = np.ones(len(X), dtype=bool)
            for idx, th in zip(indices, ths):
                pred &= (X[:, idx] >= th)
            tp = int((pred & (labels == 1)).sum())
            fp = int((pred & (labels == 0)).sum())
            total_pred = int(pred.sum())
            if total_pred < min_hits_multi:
                continue
            div = _combo_diversity(f_names)
            combo_results.append(
                _make_combo_entry(f_names, ths, tp, fp, total_pred, div)
            )

        # 4特徴量コンボ（独立特徴量から）
        ind_top8 = independent_signals[:min(8, len(independent_signals))]
        for combo in combinations(ind_top8, 4):
            f_names = [s["feature"] for s in combo]
            indices = [WIDE_FEATURE_KEYS.index(f) for f in f_names]
            ths = [s["threshold"] for s in combo]
            pred = np.ones(len(X), dtype=bool)
            for idx, th in zip(indices, ths):
                pred &= (X[:, idx] >= th)
            tp = int((pred & (labels == 1)).sum())
            fp = int((pred & (labels == 0)).sum())
            total_pred = int(pred.sum())
            if total_pred < min_hits_multi:
                continue
            div = _combo_diversity(f_names)
            combo_results.append(
                _make_combo_entry(f_names, ths, tp, fp, total_pred, div)
            )

        # --- フォールバック: 独立特徴量だけでは不足の場合、全特徴量でも探索 ---
        if len(combo_results) < 5:
            logger.info("独立特徴量のみでコンボ不足 → 全特徴量でフォールバック探索")
            for (s1, s2) in combinations(top_signals, 2):
                f1_name, f2_name = s1["feature"], s2["feature"]
                # 既に結果にあるペアはスキップ
                existing = {tuple(sorted(c["features"])) for c in combo_results}
                if tuple(sorted([f1_name, f2_name])) in existing:
                    continue
                i1 = WIDE_FEATURE_KEYS.index(f1_name)
                i2 = WIDE_FEATURE_KEYS.index(f2_name)
                pred = (X[:, i1] >= s1["threshold"]) & (X[:, i2] >= s2["threshold"])
                tp = int((pred & (labels == 1)).sum())
                fp = int((pred & (labels == 0)).sum())
                total_pred = int(pred.sum())
                if total_pred < min_total_hits:
                    continue
                div = _combo_diversity([f1_name, f2_name])
                combo_results.append(
                    _make_combo_entry([f1_name, f2_name],
                                      [s1["threshold"], s2["threshold"]],
                                      tp, fp, total_pred, div)
                )

        # ランキング: precision × diversity_score でソート（独立性が高いほど上位）
        for c in combo_results:
            div = c.get("diversity_score", 0)
            # 総合スコア = precision × (1 + 0.5 × diversity)
            # → 精度が同じなら多様性が高いコンボが上位に
            c["combined_score"] = round(
                c["precision"] * (1.0 + 0.5 * div), 4
            )
        combo_results.sort(
            key=lambda c: (c["combined_score"], c["recall"]),
            reverse=True,
        )

        # ベストコンボ選定（精度・再現率フィルタ）
        best_combos = [
            c for c in combo_results
            if c["precision"] >= cfg.discovery_min_precision
            and c["recall"] >= cfg.discovery_min_recall
        ]

        logger.info(
            f"特徴量発見完了: シグナル={len(signals)}, "
            f"コンボ={len(combo_results)}, ベスト={len(best_combos)}"
        )

        # --- 代替手法の比較探索 ---
        if progress_callback:
            progress_callback("代替手法の比較分析中...")
        alt_methods = self._explore_alternative_methods(
            X, labels, top_signals, n_pos, n_neg, base_rate,
        )

        return {
            "signals": signals,
            "combo_signals": combo_results[:30],
            "best_combos": best_combos[:10],
            "alt_methods": alt_methods,
            "corr_info": corr_info,
            "n_star": n_pos,
            "n_non_star": n_neg,
            "base_rate": round(base_rate, 4),
            "feature_keys": WIDE_FEATURE_KEYS,
        }

    # ------------------------------------------------------------------
    # 代替手法の比較探索
    # ------------------------------------------------------------------
    def _explore_alternative_methods(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        top_signals: list,
        n_pos: int,
        n_neg: int,
        base_rate: float,
    ) -> dict:
        """3つの代替手法でスター株確率を算出し、AND条件と比較。

        Returns dict with keys:
            weighted_scoring: 加重スコアリング結果
            decision_tree: 浅い決定木結果
            percentile_rank: パーセンタイルランク結果
        """
        results = {}

        # ================================================================
        # 手法1: 加重スコアリング
        # ================================================================
        try:
            # 各特徴量に J stat ベースの重みを付与し、合計スコアで判定
            ws_features = top_signals[:min(10, len(top_signals))]
            if ws_features:
                weights = []
                feat_indices = []
                feat_thresholds = []
                for s in ws_features:
                    idx = WIDE_FEATURE_KEYS.index(s["feature"])
                    feat_indices.append(idx)
                    feat_thresholds.append(s["threshold"])
                    weights.append(s["j_stat"])

                weights = np.array(weights)
                weights = weights / weights.sum()  # 正規化

                # 各サンプルのスコア計算: 閾値超過 × 重み の合計
                scores = np.zeros(len(X))
                for fi, idx in enumerate(feat_indices):
                    exceeds = (X[:, idx] >= feat_thresholds[fi]).astype(float)
                    scores += exceeds * weights[fi]

                # 複数の閾値でprecision/recallを計算
                ws_results = []
                for score_th in np.arange(0.15, 0.85, 0.05):
                    pred = scores >= score_th
                    n_pred = int(pred.sum())
                    if n_pred < 3:
                        continue
                    tp = int((pred & (labels == 1)).sum())
                    fp = n_pred - tp
                    prec = tp / n_pred if n_pred > 0 else 0
                    recall = tp / n_pos if n_pos > 0 else 0
                    lift = prec / base_rate if base_rate > 0 else 0
                    ws_results.append({
                        "score_threshold": round(float(score_th), 2),
                        "n_hits": n_pred,
                        "tp": tp, "fp": fp,
                        "precision": round(prec, 4),
                        "recall": round(recall, 4),
                        "lift": round(lift, 2),
                    })

                # F1スコアでベストを選定
                for r in ws_results:
                    p, rc = r["precision"], r["recall"]
                    r["f1"] = round(2 * p * rc / (p + rc), 4) if (p + rc) > 0 else 0

                ws_results.sort(key=lambda x: x["f1"], reverse=True)

                results["weighted_scoring"] = {
                    "method": "加重スコアリング",
                    "description": (
                        "各特徴量のJ統計量に比例した重みを付与。"
                        "閾値超過×重みの合計スコアで判定。"
                        "AND条件より柔軟で、一部の条件未達でもスコアが高ければ候補になる。"
                    ),
                    "features": [
                        {
                            "name": s["feature"],
                            "name_jp": s.get("feature_jp", s["feature"]),
                            "weight": round(float(w), 3),
                            "threshold": s["threshold"],
                        }
                        for s, w in zip(ws_features, weights)
                    ],
                    "best": ws_results[0] if ws_results else None,
                    "all_results": ws_results[:10],
                }
                logger.info(
                    f"加重スコアリング: ベスト precision={ws_results[0]['precision']:.1%}"
                    f" (lift={ws_results[0]['lift']:.1f}x, recall={ws_results[0]['recall']:.0%})"
                    if ws_results else "加重スコアリング: 結果なし"
                )
        except Exception as e:
            logger.warning(f"加重スコアリング失敗: {e}")
            results["weighted_scoring"] = {"method": "加重スコアリング", "error": str(e)}

        # ================================================================
        # 手法2: 浅い決定木
        # ================================================================
        try:
            from sklearn.tree import DecisionTreeClassifier

            # 上位特徴量のみ使用（過学習防止）
            dt_signals = top_signals[:min(8, len(top_signals))]
            if dt_signals and n_pos >= 5:
                dt_indices = [WIDE_FEATURE_KEYS.index(s["feature"]) for s in dt_signals]
                X_sub = X[:, dt_indices]
                feat_names_sub = [s.get("feature_jp", s["feature"]) for s in dt_signals]

                # 複数のmax_depthで試行
                dt_results = []
                for depth in (2, 3):
                    clf = DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_leaf=max(3, int(n_pos * 0.1)),
                        min_samples_split=max(6, int(n_pos * 0.2)),
                        class_weight="balanced",
                        random_state=42,
                    )
                    clf.fit(X_sub, labels)
                    pred = clf.predict(X_sub).astype(bool)
                    n_pred = int(pred.sum())
                    if n_pred < 3:
                        continue
                    tp = int((pred & (labels == 1)).sum())
                    fp = n_pred - tp
                    prec = tp / n_pred if n_pred > 0 else 0
                    recall = tp / n_pos if n_pos > 0 else 0
                    lift = prec / base_rate if base_rate > 0 else 0
                    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0

                    # 決定木のルールを抽出
                    tree = clf.tree_
                    rules = []
                    def _extract_rules(node, path):
                        if tree.feature[node] == -2:  # leaf
                            if tree.value[node][0][1] > tree.value[node][0][0]:
                                rules.append(list(path))
                            return
                        f_name = feat_names_sub[tree.feature[node]]
                        th = round(float(tree.threshold[node]), 4)
                        _extract_rules(
                            tree.children_left[node],
                            path + [(f_name, "<=", th)],
                        )
                        _extract_rules(
                            tree.children_right[node],
                            path + [(f_name, ">", th)],
                        )
                    _extract_rules(0, [])

                    dt_results.append({
                        "max_depth": depth,
                        "n_hits": n_pred,
                        "tp": tp, "fp": fp,
                        "precision": round(prec, 4),
                        "recall": round(recall, 4),
                        "lift": round(lift, 2),
                        "f1": round(f1, 4),
                        "rules": rules,
                        "n_rules": len(rules),
                    })

                dt_results.sort(key=lambda x: x["f1"], reverse=True)
                results["decision_tree"] = {
                    "method": "浅い決定木",
                    "description": (
                        "2-3段の決定木で分岐条件を自動学習。"
                        "閾値の共同最適化が可能で、OR条件も自然に表現できる。"
                        "ただしサンプル少数時は過学習リスクあり。"
                    ),
                    "features_used": feat_names_sub,
                    "best": dt_results[0] if dt_results else None,
                    "all_results": dt_results,
                }
                logger.info(
                    f"決定木: ベスト precision={dt_results[0]['precision']:.1%}"
                    f" (depth={dt_results[0]['max_depth']}, lift={dt_results[0]['lift']:.1f}x)"
                    if dt_results else "決定木: 結果なし"
                )
        except ImportError:
            logger.warning("sklearn未インストール — 決定木スキップ")
            results["decision_tree"] = {"method": "浅い決定木", "error": "sklearn未インストール"}
        except Exception as e:
            logger.warning(f"決定木失敗: {e}")
            results["decision_tree"] = {"method": "浅い決定木", "error": str(e)}

        # ================================================================
        # 手法3: パーセンタイルランク
        # ================================================================
        try:
            pr_signals = top_signals[:min(10, len(top_signals))]
            if pr_signals:
                pr_indices = [WIDE_FEATURE_KEYS.index(s["feature"]) for s in pr_signals]

                # 各特徴量のパーセンタイルランクを計算（0-100）
                X_percentile = np.zeros((len(X), len(pr_indices)))
                for pi, idx in enumerate(pr_indices):
                    col = X[:, idx]
                    # ランク化: 0-100のパーセンタイル
                    ranked = stats.rankdata(col, method="average")
                    X_percentile[:, pi] = ranked / len(ranked) * 100

                # 「上位 K パーセンタイルに入る特徴量が M 個以上」で判定
                pr_results = []
                for pct_th in (3, 5, 10, 15, 20):
                    # 各サンプルについて上位pct_th%に入る特徴量の数を計算
                    in_top = (X_percentile >= (100 - pct_th)).astype(int)
                    n_in_top = in_top.sum(axis=1)

                    for min_count in range(2, min(6, len(pr_signals) + 1)):
                        pred = n_in_top >= min_count
                        n_pred = int(pred.sum())
                        if n_pred < 3:
                            continue
                        tp = int((pred & (labels == 1)).sum())
                        fp = n_pred - tp
                        prec = tp / n_pred if n_pred > 0 else 0
                        recall = tp / n_pos if n_pos > 0 else 0
                        lift = prec / base_rate if base_rate > 0 else 0
                        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
                        pr_results.append({
                            "percentile_threshold": pct_th,
                            "min_features_in_top": min_count,
                            "label": f"上位{pct_th}%が{min_count}個以上",
                            "n_hits": n_pred,
                            "tp": tp, "fp": fp,
                            "precision": round(prec, 4),
                            "recall": round(recall, 4),
                            "lift": round(lift, 2),
                            "f1": round(f1, 4),
                        })

                pr_results.sort(key=lambda x: x["f1"], reverse=True)
                results["percentile_rank"] = {
                    "method": "パーセンタイルランク",
                    "description": (
                        "特徴量の絶対値ではなく全銘柄中の順位（上位何%）で判定。"
                        "市場環境の変化に強い。"
                        "例: 上位5%に入る特徴量が3個以上 → 候補。"
                    ),
                    "features": [
                        {
                            "name": s["feature"],
                            "name_jp": s.get("feature_jp", s["feature"]),
                        }
                        for s in pr_signals
                    ],
                    "best": pr_results[0] if pr_results else None,
                    "all_results": pr_results[:15],
                }
                logger.info(
                    f"パーセンタイル: ベスト precision={pr_results[0]['precision']:.1%}"
                    f" ({pr_results[0]['label']}, lift={pr_results[0]['lift']:.1f}x)"
                    if pr_results else "パーセンタイル: 結果なし"
                )
        except Exception as e:
            logger.warning(f"パーセンタイルランク失敗: {e}")
            results["percentile_rank"] = {"method": "パーセンタイルランク", "error": str(e)}

        return results

    # ------------------------------------------------------------------
    # Step 4: 追加スター株発見
    # ------------------------------------------------------------------
    def _discover_additional_stars(
        self,
        common_features: dict,
        star_codes: list[str],
        all_prices: pd.DataFrame,
        topix: pd.DataFrame,
        topix_ret_series: pd.Series,
        listed_stocks: pd.DataFrame,
        close_col: str,
        market_caps: dict[str, float] | None = None,
    ) -> list[dict]:
        """ベストコンボで全銘柄をスキャンし追加スター株を発見"""
        cfg = self.config
        min_cap_yen = cfg.scan_min_market_cap * 1e8 if cfg.scan_min_market_cap > 0 else 0
        best_combos = common_features.get("best_combos", [])
        if not best_combos:
            logger.info("ベストコンボなし → 追加スター株発見スキップ")
            return []

        best_combo = best_combos[0]
        star_set = set(star_codes)
        all_codes = [str(c) for c in all_prices["code"].unique()]
        non_star_codes = [c for c in all_codes if c not in star_set]

        # メタデータ
        meta_map = {}
        cols = ["code", "name", "sector_17_name"]
        available = [c for c in cols if c in listed_stocks.columns]
        if available:
            for _, row in listed_stocks[available].iterrows():
                meta_map[str(row["code"])] = {c: row.get(c, "") for c in available}

        # TOPIX期間リターン
        topix_close_col = "close" if "close" in topix.columns else topix.columns[-1]
        topix_sorted = topix.sort_values("date")
        topix_vals = topix_sorted[topix_close_col].astype(float).values
        if len(topix_vals) >= 2 and topix_vals[0] > 0:
            topix_period_ret = topix_vals[-1] / topix_vals[0] - 1
        else:
            topix_period_ret = 0.0

        additional = []
        feature_names = best_combo["features"]
        thresholds = best_combo["thresholds"]

        for code in non_star_codes:
            # 時価総額フィルター（cap=0は取得失敗→追加スター株では除外）
            if market_caps and min_cap_yen > 0:
                cap = market_caps.get(code, 0)
                if cap < min_cap_yen:
                    continue
            grp = all_prices[all_prices["code"] == code].sort_values("date")
            if len(grp) < 20:
                continue
            feat = self._compute_wide_features(grp, close_col, topix_ret_series)
            if feat is None:
                continue

            # コンボ条件チェック
            passes = all(
                feat.get(fn, 0.0) >= th
                for fn, th in zip(feature_names, thresholds)
            )
            if not passes:
                continue

            # リターン計算
            close_vals = grp[close_col].astype(float).values
            if close_vals[0] <= 0:
                continue
            total_ret = close_vals[-1] / close_vals[0] - 1
            excess_ret = total_ret - topix_period_ret

            # 実際にリターンが高い銘柄のみ
            if excess_ret < 0.10:
                continue

            meta = meta_map.get(code, {})
            additional.append({
                "code": code,
                "name": meta.get("name", ""),
                "sector": meta.get("sector_17_name", ""),
                "total_return": round(total_ret, 4),
                "excess_return": round(excess_ret, 4),
                "source": "discovered",
                "matched_combo": {
                    "features": feature_names,
                    "values": [round(feat.get(fn, 0.0), 4) for fn in feature_names],
                    "thresholds": thresholds,
                },
            })

        # excess_return降順でソート、上限適用
        additional.sort(key=lambda x: x["excess_return"], reverse=True)
        additional = additional[:cfg.discovery_max_additional_stars]

        logger.info(f"追加スター株: {len(additional)}件発見")
        return additional

    # ------------------------------------------------------------------
    # Step 5: 初動特定（10シグナル方式）
    # ------------------------------------------------------------------
    def _detect_onset_dates(
        self,
        all_stars: list[dict],
        all_prices: pd.DataFrame,
        close_col: str,
        progress_callback=None,
        topix: pd.DataFrame | None = None,
    ) -> dict:
        """全スター株の初動日を10シグナル方式で特定

        Returns
        -------
        dict: code -> {"onset_date", "signals", "score", "fwd_return_60d", ...}
        """
        results = {}
        for i, star in enumerate(all_stars):
            code = str(star["code"])
            grp = all_prices[all_prices["code"] == code].sort_values("date")
            if len(grp) < 60:
                results[code] = {
                    "onset_date": "", "signals": [], "score": 0,
                    "fwd_return_60d": 0.0,
                }
                continue

            result = self._detect_single_onset(grp, close_col, topix=topix)
            results[code] = result

            if progress_callback and (i + 1) % 5 == 0:
                progress_callback(f"初動特定中... ({i + 1}/{len(all_stars)})")

        logger.info(
            f"初動特定完了: {sum(1 for r in results.values() if r['onset_date'])}/"
            f"{len(results)}件で初動日特定"
        )
        return results

    def _detect_single_onset(
        self, grp: pd.DataFrame, close_col: str,
        topix: pd.DataFrame | None = None,
    ) -> dict:
        """1銘柄の初動日を検出"""
        empty = {
            "onset_date": "", "signals": [], "score": 0,
            "fwd_return_60d": None, "max_return": None, "max_drawdown": None,
            "excess_return": None, "sharpe_ratio": None,
        }
        if len(grp) < 60:
            return empty

        signals_df = self._compute_daily_onset_signals(grp, close_col)
        if signals_df.empty:
            return empty

        daily_score = signals_df.sum(axis=1).astype(int)
        dates = pd.to_datetime(grp["date"].values)
        close_vals = grp[close_col].astype(float).values
        date_to_idx = {d: i for i, d in enumerate(dates)}

        # 閾値 4 → 3 の順で候補日を探す
        for threshold in (4, 3):
            candidate_mask = daily_score >= threshold
            candidate_dates = signals_df.index[candidate_mask]

            for cand_date in candidate_dates:
                cand_ts = pd.Timestamp(cand_date)
                if cand_ts not in date_to_idx:
                    continue
                cand_idx = date_to_idx[cand_ts]

                # 60営業日先リターン >= 15%
                future_idx = min(cand_idx + 60, len(close_vals) - 1)
                if future_idx <= cand_idx:
                    continue

                price_at_onset = close_vals[cand_idx]
                if price_at_onset <= 0:
                    continue
                fwd_return = close_vals[future_idx] / price_at_onset - 1

                if fwd_return >= 0.15:
                    row = signals_df.loc[cand_date]
                    fired = [name for name in MULTI_ONSET_SIGNAL_NAMES if row[name]]

                    # 初動後〜分析期間終端までの最大リターン・最大ドローダウン計算
                    # （60日ではなく、データが続く限り全期間を対象にする）
                    window = close_vals[cand_idx:]
                    max_price = float(np.nanmax(window))
                    max_ret = max_price / price_at_onset - 1
                    if not np.isfinite(max_ret):
                        max_ret = 0.0
                    peak = price_at_onset
                    max_dd = 0.0
                    for p in window:
                        if p > peak:
                            peak = p
                        dd = (p - peak) / peak
                        if dd < max_dd:
                            max_dd = dd

                    # --- ベンチマーク超過リターン・シャープレシオ ---
                    excess_ret = None
                    sharpe = None
                    if topix is not None and len(topix) > 0:
                        try:
                            topix_close_col = (
                                "close" if "close" in topix.columns
                                else topix.columns[-1]
                            )
                            topix_s = topix.copy()
                            topix_s["date"] = pd.to_datetime(topix_s["date"])
                            topix_s = topix_s.sort_values("date").set_index("date")
                            topix_c = topix_s[topix_close_col].astype(float)

                            onset_ts = pd.Timestamp(cand_date)
                            stock_dates = pd.to_datetime(grp["date"].values)
                            end_ts = stock_dates.max()

                            # TOPIX の onset→end 区間
                            topix_window = topix_c.loc[
                                (topix_c.index >= onset_ts)
                                & (topix_c.index <= end_ts)
                            ]
                            if len(topix_window) >= 2 and topix_window.iloc[0] > 0:
                                topix_total_ret = (
                                    topix_window.max() / topix_window.iloc[0] - 1
                                )
                                excess_ret = round(max_ret - topix_total_ret, 4)

                            # シャープレシオ: 日次超過リターンから年率換算
                            stock_window = pd.Series(
                                close_vals[cand_idx:],
                                index=stock_dates[cand_idx:],
                            )
                            stock_daily = stock_window.pct_change().dropna()
                            topix_daily = topix_c.pct_change().dropna()
                            common_idx = stock_daily.index.intersection(topix_daily.index)
                            if len(common_idx) >= 5:
                                excess_daily = (
                                    stock_daily.loc[common_idx]
                                    - topix_daily.loc[common_idx]
                                )
                                mean_ex = excess_daily.mean()
                                std_ex = excess_daily.std()
                                if std_ex > 0:
                                    sharpe = round(
                                        mean_ex / std_ex * np.sqrt(252), 2
                                    )
                        except Exception:
                            pass  # TOPIX計算失敗時はNoneのまま

                    # 各シグナルの定量値を計算
                    sig_qty = {}
                    close_s = grp[close_col].astype(float)
                    vol_s = grp["volume"].astype(float)
                    high_s = grp["high"].astype(float) if "high" in grp.columns else close_s
                    low_s = grp["low"].astype(float) if "low" in grp.columns else close_s
                    vol_ma20 = vol_s.rolling(20, min_periods=10).mean()

                    if "volume_surge" in fired:
                        v, vm = vol_s.iloc[cand_idx], vol_ma20.iloc[cand_idx]
                        if vm > 0:
                            sig_qty["volume_surge"] = round(v / vm, 1)

                    if "higher_lows" in fired and cand_idx >= 10:
                        lows = low_s.iloc[cand_idx - 10: cand_idx + 1].values
                        if lows[0] > 0:
                            sig_qty["higher_lows"] = round((lows[-1] / lows[0] - 1) * 100, 1)

                    if "range_breakout" in fired and cand_idx >= 20:
                        recent_high = high_s.iloc[cand_idx - 20: cand_idx].max()
                        if recent_high > 0:
                            sig_qty["range_breakout"] = round(
                                (close_s.iloc[cand_idx] / recent_high - 1) * 100, 1)

                    if "up_volume_dominance" in fired and cand_idx >= 10:
                        ret_w = close_s.pct_change().iloc[cand_idx - 10: cand_idx + 1]
                        vol_w = vol_s.iloc[cand_idx - 10: cand_idx + 1]
                        total = vol_w.sum()
                        if total > 0:
                            sig_qty["up_volume_dominance"] = round(
                                vol_w[ret_w > 0].sum() / total * 100, 0)

                    if "obv_breakout" in fired and cand_idx >= 10:
                        ret_10 = close_s.pct_change(10).iloc[cand_idx]
                        sig_qty["obv_breakout"] = round(ret_10 * 100, 1)

                    if "quiet_accumulation" in fired and cand_idx >= 5:
                        v5 = vol_s.iloc[cand_idx - 5: cand_idx + 1].mean()
                        vm = vol_ma20.iloc[cand_idx]
                        if vm > 0:
                            sig_qty["quiet_accumulation"] = round(v5 / vm, 1)

                    if "ma_crossover" in fired and cand_idx >= 25:
                        ma5 = close_s.rolling(5).mean().iloc[cand_idx]
                        ma25 = close_s.rolling(25).mean().iloc[cand_idx]
                        if ma25 > 0:
                            sig_qty["ma_crossover"] = round((ma5 / ma25 - 1) * 100, 1)

                    return {
                        "onset_date": str(cand_date)[:10],
                        "signals": fired,
                        "signal_quantities": sig_qty,
                        "score": int(row.sum()),
                        "fwd_return_60d": round(fwd_return, 4),
                        "max_return": round(max_ret, 4),     # 初動後〜期間末までのピーク
                        "max_drawdown": round(max_dd, 4),    # 同期間の最大DD
                        "excess_return": excess_ret,          # TOPIX対比超過リターン
                        "sharpe_ratio": sharpe,               # 年率シャープレシオ
                        # 旧フィールド名との後方互換（既存JSONロード時用）
                        "max_return_60d": round(max_ret, 4),
                        "max_drawdown_60d": round(max_dd, 4),
                    }

        return empty

    @staticmethod
    def _compute_daily_onset_signals(df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """1銘柄の日次10シグナルを計算"""
        if len(df) < 60:
            return pd.DataFrame()

        out = pd.DataFrame(index=df.index)
        close = df[close_col].astype(float)
        high = df["high"].astype(float) if "high" in df.columns else close
        low = df["low"].astype(float) if "low" in df.columns else close
        volume = df["volume"].astype(float)
        ret = close.pct_change()

        vol_ma20 = volume.rolling(20, min_periods=10).mean()

        # 1. volume_surge
        out["volume_surge"] = volume > (vol_ma20 * 2.5)

        # 2. quiet_accumulation
        out["quiet_accumulation"] = (ret.abs() < 0.005) & (volume > vol_ma20 * 1.3)

        # 3. consecutive_accumulation
        qa_roll = out["quiet_accumulation"].astype(int).rolling(5, min_periods=5).sum()
        out["consecutive_accumulation"] = qa_roll >= 4

        # 4. obv_breakout
        signed_vol = volume * ret.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv = signed_vol.cumsum()
        obv_ma20 = obv.rolling(20, min_periods=10).mean()
        obv_ma60 = obv.rolling(60, min_periods=30).mean()
        obv_change10 = obv.pct_change(10)
        obv_change_std = obv_change10.rolling(60, min_periods=20).std()
        obv_change_mean = obv_change10.rolling(60, min_periods=20).mean()
        obv_z = (obv_change10 - obv_change_mean) / obv_change_std.replace(0, np.nan)
        out["obv_breakout"] = (obv_ma20 > obv_ma60) & (obv_z > 2.0)

        # 5. bb_squeeze
        ma20 = close.rolling(20, min_periods=10).mean()
        std20 = close.rolling(20, min_periods=10).std()
        bb_width = (2 * std20) / ma20.replace(0, np.nan)
        bb_width_ma = bb_width.rolling(20, min_periods=10).mean()
        out["bb_squeeze"] = bb_width < (bb_width_ma * 0.6)

        # 6. volatility_compression
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr10 = tr.rolling(10, min_periods=5).mean()
        atr60 = tr.rolling(60, min_periods=30).mean()
        out["volatility_compression"] = atr10 < (atr60 * 0.6)

        # 7. higher_lows
        def _higher_lows_check(s):
            vals = s.values
            if np.isnan(vals).any():
                return False
            x = np.arange(len(vals), dtype=float)
            slope, intercept, r, p, se = stats.linregress(x, vals)
            return (slope > 0) and (r ** 2 > 0.6)

        hl_series = low.rolling(5, min_periods=5).apply(_higher_lows_check, raw=False)
        out["higher_lows"] = hl_series.fillna(0).astype(bool)

        # 8. range_breakout
        high40 = high.rolling(40, min_periods=20).max().shift(1)
        out["range_breakout"] = close > high40

        # 9. ma_crossover
        ma5 = close.rolling(5, min_periods=3).mean()
        cross_today = ma5 > ma20
        cross_yesterday = ma5.shift(1) <= ma20.shift(1)
        out["ma_crossover"] = cross_today & cross_yesterday

        # 10. up_volume_dominance
        up_vol = volume.where(ret > 0, 0.0)
        down_vol = volume.where(ret < 0, 0.0)
        up_vol_sum10 = up_vol.rolling(10, min_periods=5).sum()
        down_vol_sum10 = down_vol.rolling(10, min_periods=5).sum()
        out["up_volume_dominance"] = up_vol_sum10 > (down_vol_sum10.replace(0, np.nan) * 2.0)

        out = out.fillna(False).astype(bool)
        if "date" in df.columns:
            out.index = df["date"].values
        return out

    # ------------------------------------------------------------------
    # Step 6: AI解釈
    # ------------------------------------------------------------------
    def _generate_ai_interpretation(
        self,
        star_stocks: list[dict],
        additional_stars: list[dict],
        common_features: dict,
        onset_dates: dict,
    ) -> str:
        """Claude CLIで結果を自然言語解釈"""
        try:
            from core.ai_client import create_ai_client
            client = create_ai_client()
        except Exception as e:
            logger.warning(f"AIクライアント初期化失敗: {e}")
            return ""

        # プロンプト構築
        signals = common_features.get("signals", [])
        best_combos = common_features.get("best_combos", [])
        top_signals = [s for s in signals if s.get("verdict") in ("strong", "weak_useful")][:10]

        # 初動リターン統計を計算（max_return（新）> max_return_60d（旧）を優先）
        def _pick_return(od):
            for key in ("max_return", "max_return_60d", "fwd_return_60d"):
                v = od.get(key)
                if v is not None:
                    return v
            return None

        max_returns = []
        for od in onset_dates.values():
            if not od.get("onset_date"):
                continue
            r = _pick_return(od)
            if r is not None and np.isfinite(r):
                max_returns.append(r)
        fwd_ret_stats = {}
        if max_returns:
            import statistics
            fwd_ret_stats = {
                "count": len(max_returns),
                "mean": sum(max_returns) / len(max_returns),
                "median": statistics.median(max_returns),
                "min": min(max_returns),
                "max": max(max_returns),
            }

        base_rate = common_features.get("base_rate", 0)
        base_rate_universe = common_features.get("base_rate_universe", base_rate)
        n_universe = common_features.get("n_universe", 0)
        n_all_stars = len(star_stocks) + len(additional_stars)

        prompt_parts = [
            "以下の株式分析結果を投資家向けに日本語で解釈してください。",
            "【重要ルール】",
            "- 英語の指標名（VPIN、ATR、OBV、MACD、Liftなど）は一切使わず、誰でも理解できる平易な日本語で記述してください。",
            "- 技術用語は必ず括弧内に意味を補足するか、日本語で言い換えてください。",
            "- 投資未経験者でも理解できる言葉を使ってください。",
            "",
            "Markdown形式で、以下のセクションで構成してください：",
            "",
            "1. **スター株に共通していた「株の状態」（最重要）**",
            "   - 上位5つの特徴量が何を意味するか、普通の言葉で説明",
            "   - 「スター株はXXの状態にあった（市場平均のY倍）」の形式で書く",
            "   - 特徴量名の英単語は使わず、意味だけを日本語で説明すること",
            "",
            "2. **スター株になる確率（数字で明示）**",
        ]
        # 母集団ベースレートが利用可能か確認
        if n_universe > 0 and base_rate_universe > 0:
            prompt_parts.append(
                f"   - ベースレート（何もしない場合）: {base_rate_universe:.2%}"
                f"（分析対象{n_universe:,}銘柄中、スター株は{n_all_stars}件）"
            )
        else:
            prompt_parts.append(f"   - ベースレート（何もしない場合）: {base_rate:.1%}")
        if best_combos:
            best = best_combos[0]
            best_names_jp = " かつ ".join(best.get("features_jp", best["features"]))
            u_prec = best.get("universe_precision")
            u_hits = best.get("universe_n_hits", 0)
            u_stars = best.get("universe_n_stars", 0)
            if u_prec is not None and n_universe > 0:
                u_lift = u_prec / base_rate_universe if base_rate_universe > 0 else 0
                prompt_parts.append(
                    f"   - 「{best_names_jp}」を対象期間中に一度でも満たした銘柄: {u_hits}件"
                    f"（全{n_universe:,}銘柄中）"
                )
                prompt_parts.append(
                    f"   - そのうちスター株になった: {u_stars}件 → 確率={u_prec:.1%}（ベースレートの{u_lift:.1f}倍）"
                )
                prompt_parts.append(
                    f"   - ※この確率は訓練データではなく全銘柄×複数時点のスキャンで計算した「真の母集団確率」です"
                )
            else:
                prompt_parts.append(
                    f"   - 「{best_names_jp}」を同時に満たした場合: 精度={best['precision']:.0%}（{best['lift']:.1f}倍）"
                    f"　※同一データ内評価のため過大評価の可能性あり"
                )
        prompt_parts.extend([
            "   - 「条件なし→条件あり」の確率変化を、わかりやすく解説してください",
            "",
            "3. **初動で買った場合の期待リターン（数字で明示）**",
            "   ※ 注意: 初動日の選定は「その後60日以内に高い上昇を達成した日」を基準にしています。",
            "     そのため「初動後に上昇した」という事実は、定義上当然のことです（循環論法）。",
            "     ここでは「どのくらいの上昇規模が期待できるか」の参考値として解釈してください。",
        ])
        if fwd_ret_stats:
            prompt_parts.extend([
                f"   - 初動後〜分析期間末までのピーク到達リターン: 平均={fwd_ret_stats['mean']:.1%}, 中央値={fwd_ret_stats['median']:.1%}",
                f"   - 最大={fwd_ret_stats['max']:.1%}, 最小={fwd_ret_stats['min']:.1%}（{fwd_ret_stats['count']}銘柄）",
                f"   - 「勝率」は言及しないでください（定義上ほぼ100%のため無意味です）",
                f"   - 「60日後に売却」などの特定の売却タイミングには言及しないでください",
            ])
        prompt_parts.extend([
            "   - このリターンの現実的な解釈と注意点を書いてください",
            "",
            "4. **追加スター株の評価**",
            "",
            "5. **初動のパターン（必須条件と加速条件）**",
            "",
            "6. **実践的な投資への示唆**",
            "",
            f"## 入力スター株: {len(star_stocks)}件",
        ])
        for s in star_stocks[:20]:
            prompt_parts.append(f"- {s.get('code', '')} {s.get('name', '')} "
                                f"(超過リターン: {s.get('excess_return', 0):.1%})")

        prompt_parts.append(f"\n## 判別力のある特徴量 (上位{len(top_signals)}件): ※以下を日本語で言い換えてください")
        for sig in top_signals:
            desc = WIDE_FEATURE_DESCRIPTIONS_JP.get(sig["feature"], "")
            prompt_parts.append(
                f"- 【{sig['feature_jp']}】意味:「{desc}」"
                f" → スター株平均={sig['pos_mean']:.4f} vs 市場平均={sig['neg_mean']:.4f}"
                f"（{sig['lift']:.1f}倍高い）"
            )

        if best_combos:
            prompt_parts.append(f"\n## ベストコンボ (上位{min(5, len(best_combos))}件):")
            for combo in best_combos[:5]:
                names = " かつ ".join(combo.get("features_jp", combo["features"]))
                u_prec = combo.get("universe_precision")
                u_hits = combo.get("universe_n_hits", 0)
                u_stars = combo.get("universe_n_stars", 0)
                if u_prec is not None and n_universe > 0:
                    u_lift = u_prec / base_rate_universe if base_rate_universe > 0 else 0
                    prompt_parts.append(
                        f"- 条件「{names}」: 対象期間中に条件を満たした銘柄={u_hits}件、"
                        f"そのうちスター株={u_stars}件 → 確率={u_prec:.1%}（ベースレートの{u_lift:.1f}倍）、"
                        f"全スター株の{combo['recall']:.0%}をカバー"
                    )
                else:
                    prompt_parts.append(
                        f"- 条件「{names}」: この条件を満たした銘柄の{combo['precision']:.0%}がスター株化"
                        f"（市場平均{base_rate:.1%}の{combo['lift']:.1f}倍、全スター株の{combo['recall']:.0%}をカバー）"
                    )

        prompt_parts.append(f"\n## 追加発見スター株: {len(additional_stars)}件")
        for a in additional_stars[:10]:
            prompt_parts.append(
                f"- {a.get('code', '')} {a.get('name', '')} "
                f"(超過リターン: {a.get('excess_return', 0):.1%})"
            )

        # 初動パターン集計
        signal_counts = {}
        onset_count = 0
        for code, od in onset_dates.items():
            if od.get("onset_date"):
                onset_count += 1
                for sig in od.get("signals", []):
                    signal_counts[sig] = signal_counts.get(sig, 0) + 1

        prompt_parts.append(f"\n## 初動検出: {onset_count}/{len(onset_dates)}件で特定")
        prompt_parts.append("発火シグナル頻度（日本語で意味を説明してください）:")
        for sig, cnt in sorted(signal_counts.items(), key=lambda x: -x[1]):
            sig_jp = ONSET_SIGNAL_NAMES_JP.get(sig, sig)
            prompt_parts.append(f"- {sig_jp}: {cnt}件")

        prompt = "\n".join(prompt_parts)

        try:
            response = client.send_message(prompt)
            logger.info(f"AI解釈取得完了: {len(response)}文字")
            return response
        except Exception as e:
            logger.warning(f"AI解釈生成失敗: {e}")
            return ""

    # ------------------------------------------------------------------
    # 母集団精度計算（全宇宙×複数時点）
    # ------------------------------------------------------------------
    def _compute_universe_precision(
        self,
        all_prices: pd.DataFrame,
        star_codes: set,
        best_combos: list,
        close_col: str,
        topix_ret_series,
        progress_callback=None,
    ) -> list:
        """
        全宇宙銘柄を複数時点でサンプリングし、
        「対象期間中にいつかでもコンボ条件を満たした銘柄のうちスター株の割合」を計算。

        これが真の「スター株確率」:
          P(スター株 | 対象期間中にコンボ条件を一度でも満たした)

        現在の train precision（同一データで発見・評価）とは異なり、
        全銘柄×複数時点でのクロスチェックを行う。
        """
        if not best_combos:
            return []

        all_codes = list(str(c) for c in all_prices["code"].unique())
        n_total = len(all_codes)
        if n_total == 0:
            return []

        # コンボの (特徴量名, 閾値) ペアを事前構築
        combo_defs = []
        for c in best_combos[:5]:
            combo_defs.append(list(zip(c["features"], c["thresholds"])))

        # 各コンボで条件を満たしたことのある銘柄セット + 合致詳細
        combo_hit_sets = [set() for _ in combo_defs]
        combo_hit_details: list[list[dict]] = [[] for _ in combo_defs]

        STRIDE = 20   # 約20営業日おきにサンプリング
        LOOKBACK = 60  # 特徴量計算用ルックバック日数

        for ci_count, code in enumerate(all_codes):
            if progress_callback and ci_count % 300 == 0:
                progress_callback(
                    f"母集団確率計算中... ({ci_count}/{n_total}銘柄, "
                    f"条件合致数: {[len(s) for s in combo_hit_sets]})"
                )

            grp = all_prices[all_prices["code"] == code].sort_values("date").reset_index(drop=True)
            n = len(grp)
            if n < 25:
                continue

            # 全コンボ既にhit済みならスキップ
            if all(code in s for s in combo_hit_sets):
                continue

            # 20日ストライドで複数時点サンプリング
            for t in range(20, n, STRIDE):
                window = grp.iloc[max(0, t - LOOKBACK): t + 1].reset_index(drop=True)
                feat = self._compute_wide_features(window, close_col, topix_ret_series)
                if feat is None:
                    continue

                for ci, ck in enumerate(combo_defs):
                    if code in combo_hit_sets[ci]:
                        continue
                    if all(feat.get(fn, 0.0) >= th for fn, th in ck):
                        combo_hit_sets[ci].add(code)
                        # 合致詳細を保存（銘柄あたり最初の合致のみ）
                        match_date = str(grp.iloc[t]["date"])[:10]
                        feat_snapshot = {fn: round(feat.get(fn, 0.0), 4) for fn, _ in ck}
                        combo_hit_details[ci].append({
                            "code": code,
                            "match_date": match_date,
                            "is_star": code in star_codes,
                            "feature_values": feat_snapshot,
                        })

        # 合致銘柄の60日後リターン計算
        if progress_callback:
            progress_callback("合致銘柄の60日後リターン計算中...")
        match_results = []
        for ci, details in enumerate(combo_hit_details):
            m_stats, m_examples = self._compute_match_forward_returns(
                details, all_prices, close_col, topix_ret_series, star_codes,
                max_stocks=50,
            )
            match_results.append((m_stats, m_examples))

        # 精度計算
        results = []
        for ci, hits in enumerate(combo_hit_sets):
            n_hits = len(hits)
            n_star_hits = len(hits & star_codes)
            universe_prec = n_star_hits / n_hits if n_hits > 0 else 0
            r = {
                "universe_n_total": n_total,
                "universe_n_hits": n_hits,
                "universe_n_stars": n_star_hits,
                "universe_precision": round(universe_prec, 4),
                "universe_hit_rate": round(n_hits / n_total, 4) if n_total > 0 else 0,
            }
            if ci < len(match_results):
                r["match_stats"] = match_results[ci][0]
                r["match_examples"] = match_results[ci][1]
            results.append(r)

        logger.info(
            f"母集団精度計算完了: {n_total}銘柄スキャン, "
            f"Top1コンボ: {results[0].get('universe_precision', 'N/A') if results else 'N/A'}"
        )
        return results

    # ------------------------------------------------------------------
    # 合致銘柄の60日後フォワードリターン計算
    # ------------------------------------------------------------------
    def _compute_match_forward_returns(
        self,
        match_details: list[dict],
        all_prices: pd.DataFrame,
        close_col: str,
        topix_ret_series,
        star_codes: set,
        max_stocks: int = 50,
    ) -> tuple[dict, list[dict]]:
        """合致銘柄の60日後リターンと統計を計算する。

        Parameters
        ----------
        match_details : list[dict]
            各要素: {code, match_date, is_star, feature_values}
        all_prices : DataFrame
            全銘柄の株価データ
        close_col : str
            終値カラム名
        topix_ret_series : Series or similar
            TOPIX日次リターン系列（index=date）
        star_codes : set
            スター株コードの集合
        max_stocks : int
            最大サンプル数（超過時はスター株を保護しつつサンプリング）

        Returns
        -------
        (stats_dict, examples_list)
            stats_dict: 集計統計（平均リターン、中央値、勝率、シャープ等）
            examples_list: リターン上位5+下位5の実例リスト
        """
        import random

        empty_stats = {
            "n_samples": 0,
            "mean_return": 0.0,
            "median_return": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "mean_excess": 0.0,
            "star_mean_return": 0.0,
            "star_win_rate": 0.0,
            "star_count": 0,
            "nonstar_mean_return": 0.0,
            "nonstar_win_rate": 0.0,
            "nonstar_count": 0,
        }

        if not match_details:
            return empty_stats, []

        # --- サンプリング: スター株は全保持、非スター株をランダム抽出 ---
        stars = [d for d in match_details if d["is_star"]]
        non_stars = [d for d in match_details if not d["is_star"]]

        if len(match_details) > max_stocks:
            remaining = max(max_stocks - len(stars), 5)
            if len(non_stars) > remaining:
                non_stars = random.sample(non_stars, remaining)
            sampled = stars + non_stars
        else:
            sampled = match_details

        # --- TOPIX系列をdate indexで引けるように準備 ---
        topix_cum = None
        if topix_ret_series is not None:
            try:
                topix_idx = pd.Series(topix_ret_series.values, index=pd.to_datetime(topix_ret_series.index))
                topix_cum = (1 + topix_idx).cumprod()
            except Exception:
                topix_cum = None

        # --- 各銘柄の60日後リターンを計算 ---
        FORWARD_DAYS = 60
        results_list = []

        for item in sampled:
            code = item["code"]
            match_date_str = item["match_date"]

            grp = all_prices[all_prices["code"] == code].sort_values("date").reset_index(drop=True)
            if grp.empty:
                continue

            # match_dateの位置を特定
            grp_dates = pd.to_datetime(grp["date"])
            match_dt = pd.to_datetime(match_date_str)

            # match_date以降の最も近い行を探す
            mask = grp_dates >= match_dt
            if mask.sum() == 0:
                continue
            start_idx = mask.idxmax()

            if start_idx + FORWARD_DAYS >= len(grp):
                # 60日後のデータが不足 → 利用可能な最終日まで
                end_idx = len(grp) - 1
                if end_idx <= start_idx:
                    continue
            else:
                end_idx = start_idx + FORWARD_DAYS

            price_start = float(grp.iloc[start_idx][close_col])
            price_end = float(grp.iloc[end_idx][close_col])

            if price_start <= 0:
                continue

            fwd_return = price_end / price_start - 1
            actual_days = end_idx - start_idx

            # TOPIX同期間リターン（超過リターン計算用）
            excess_return = fwd_return  # デフォルト = 生リターン
            topix_return = 0.0
            if topix_cum is not None:
                try:
                    start_date = grp_dates.iloc[start_idx]
                    end_date = grp_dates.iloc[end_idx]
                    # 最も近いTOPIX日付を探す
                    t_start_mask = topix_cum.index >= start_date
                    t_end_mask = topix_cum.index >= end_date
                    if t_start_mask.any() and t_end_mask.any():
                        t_start_val = topix_cum[t_start_mask].iloc[0]
                        t_end_val = topix_cum[t_end_mask].iloc[0]
                        if t_start_val > 0:
                            topix_return = t_end_val / t_start_val - 1
                            excess_return = fwd_return - topix_return
                except Exception:
                    pass

            results_list.append({
                "code": code,
                "match_date": match_date_str,
                "is_star": item["is_star"],
                "forward_return": round(fwd_return, 4),
                "excess_return": round(excess_return, 4),
                "topix_return": round(topix_return, 4),
                "actual_days": actual_days,
                "feature_values": item.get("feature_values", {}),
            })

        if not results_list:
            return empty_stats, []

        # --- 集計統計 ---
        returns = np.array([r["forward_return"] for r in results_list])
        excess_rets = np.array([r["excess_return"] for r in results_list])
        n = len(returns)

        mean_ret = float(np.mean(returns))
        median_ret = float(np.median(returns))
        win_rate = float(np.sum(returns > 0) / n) if n > 0 else 0.0
        mean_excess = float(np.mean(excess_rets))

        # シャープレシオ（60日リターンを年率換算: ×sqrt(252/60)）
        if n >= 2 and np.std(returns) > 0:
            sharpe = float(np.mean(excess_rets) / np.std(returns) * np.sqrt(252 / 60))
        else:
            sharpe = 0.0

        # スター株 vs 非スター株の内訳
        star_rets = [r["forward_return"] for r in results_list if r["is_star"]]
        nonstar_rets = [r["forward_return"] for r in results_list if not r["is_star"]]

        stats = {
            "n_samples": n,
            "mean_return": round(mean_ret, 4),
            "median_return": round(median_ret, 4),
            "win_rate": round(win_rate, 4),
            "sharpe": round(sharpe, 2),
            "mean_excess": round(mean_excess, 4),
            "star_mean_return": round(float(np.mean(star_rets)), 4) if star_rets else 0.0,
            "star_win_rate": round(float(np.sum(np.array(star_rets) > 0) / len(star_rets)), 4) if star_rets else 0.0,
            "star_count": len(star_rets),
            "nonstar_mean_return": round(float(np.mean(nonstar_rets)), 4) if nonstar_rets else 0.0,
            "nonstar_win_rate": round(float(np.sum(np.array(nonstar_rets) > 0) / len(nonstar_rets)), 4) if nonstar_rets else 0.0,
            "nonstar_count": len(nonstar_rets),
        }

        # --- 実例: リターン上位5 + 下位5 ---
        sorted_by_ret = sorted(results_list, key=lambda x: x["forward_return"], reverse=True)
        top5 = sorted_by_ret[:5]
        bottom5 = sorted_by_ret[-5:] if len(sorted_by_ret) > 5 else []
        # 重複除去（5件以下の場合）
        top_codes_dates = {(r["code"], r["match_date"]) for r in top5}
        bottom5 = [r for r in bottom5 if (r["code"], r["match_date"]) not in top_codes_dates]
        examples = top5 + bottom5

        return stats, examples

    def _compute_universe_precision_alt(
        self,
        all_prices: pd.DataFrame,
        star_codes: set,
        alt_methods: dict,
        signals: list,
        close_col: str,
        topix_ret_series,
        progress_callback=None,
    ) -> dict:
        """代替手法の母集団精度を計算。

        全銘柄×複数時点で特徴量を計算し、各手法の条件を適用して精度を測定。
        """
        all_codes = list(str(c) for c in all_prices["code"].unique())
        n_total = len(all_codes)
        if n_total == 0:
            return {}

        top_signals = [s for s in signals if s.get("verdict") in ("strong", "weak_useful")][:10]
        if not top_signals:
            top_signals = signals[:10]

        # 手法1準備: 加重スコアリング
        ws_info = alt_methods.get("weighted_scoring", {})
        ws_best = ws_info.get("best")
        ws_features = ws_info.get("features", [])
        ws_weights = np.array([f["weight"] for f in ws_features]) if ws_features else np.array([])
        ws_thresholds = [f["threshold"] for f in ws_features]
        ws_feat_names = [f["name"] for f in ws_features]
        ws_score_th = ws_best["score_threshold"] if ws_best else 0
        ws_hit_codes = set()

        # 手法3準備: パーセンタイルランク — コード→特徴量値を収集
        pr_info = alt_methods.get("percentile_rank", {})
        pr_best = pr_info.get("best")
        pr_features = pr_info.get("features", [])
        pr_feat_names = [f["name"] for f in pr_features]
        pr_pct_th = pr_best["percentile_threshold"] if pr_best else 10
        pr_min_count = pr_best["min_features_in_top"] if pr_best else 3
        # パーセンタイルは全データ収集後に計算するため、コード→特徴量値を蓄積
        code_pr_feats = {}  # code -> [feat_values...]

        STRIDE = 20
        LOOKBACK = 60

        for ci_count, code in enumerate(all_codes):
            if progress_callback and ci_count % 500 == 0:
                progress_callback(
                    f"代替手法 母集団計算中... ({ci_count}/{n_total}銘柄)"
                )

            grp = all_prices[all_prices["code"] == code].sort_values("date").reset_index(drop=True)
            n = len(grp)
            if n < 25:
                continue

            best_ws_score = 0.0
            best_pr_vals = None  # 最大の特徴量値セットを保持

            for t in range(20, n, STRIDE):
                window = grp.iloc[max(0, t - LOOKBACK): t + 1].reset_index(drop=True)
                feat = self._compute_wide_features(window, close_col, topix_ret_series)
                if feat is None:
                    continue

                # 加重スコアリング判定
                if ws_features and code not in ws_hit_codes:
                    score = 0.0
                    for fi, fn in enumerate(ws_feat_names):
                        if feat.get(fn, 0.0) >= ws_thresholds[fi]:
                            score += ws_weights[fi]
                    if score > best_ws_score:
                        best_ws_score = score

                # パーセンタイル: 各特徴量の最大値を保持
                if pr_features:
                    vals = [feat.get(fn, 0.0) for fn in pr_feat_names]
                    if best_pr_vals is None:
                        best_pr_vals = vals
                    else:
                        best_pr_vals = [max(old, new) for old, new in zip(best_pr_vals, vals)]

            # 加重スコアリング: 最大スコアが閾値以上か
            if ws_features and best_ws_score >= ws_score_th:
                ws_hit_codes.add(code)

            # パーセンタイル: 最大値を保存
            if pr_features and best_pr_vals is not None:
                code_pr_feats[code] = best_pr_vals

        results = {}

        # 加重スコアリング母集団精度
        if ws_features and ws_best:
            n_hits = len(ws_hit_codes)
            n_star_hits = len(ws_hit_codes & star_codes)
            u_prec = n_star_hits / n_hits if n_hits > 0 else 0
            results["weighted_scoring"] = {
                "universe_n_total": n_total,
                "universe_n_hits": n_hits,
                "universe_n_stars": n_star_hits,
                "universe_precision": round(u_prec, 4),
            }
            logger.info(
                f"加重スコアリング母集団: {n_hits}件合致, "
                f"{n_star_hits}件スター株, 精度={u_prec:.1%}"
            )

        # パーセンタイルランク母集団精度
        if pr_features and pr_best and code_pr_feats:
            # 全銘柄の最大特徴量値からパーセンタイルを計算
            all_vals = np.array(list(code_pr_feats.values()))  # (n_codes, n_features)
            all_code_list = list(code_pr_feats.keys())
            # 各特徴量のランクを計算
            n_samples = len(all_vals)
            in_top = np.zeros(n_samples, dtype=int)
            for fi in range(all_vals.shape[1]):
                ranked = stats.rankdata(all_vals[:, fi], method="average")
                percentile = ranked / n_samples * 100
                in_top += (percentile >= (100 - pr_pct_th)).astype(int)

            pr_hit_codes = set()
            for i, code in enumerate(all_code_list):
                if in_top[i] >= pr_min_count:
                    pr_hit_codes.add(code)

            n_hits = len(pr_hit_codes)
            n_star_hits = len(pr_hit_codes & star_codes)
            u_prec = n_star_hits / n_hits if n_hits > 0 else 0
            results["percentile_rank"] = {
                "universe_n_total": n_total,
                "universe_n_hits": n_hits,
                "universe_n_stars": n_star_hits,
                "universe_precision": round(u_prec, 4),
            }
            logger.info(
                f"パーセンタイル母集団: {n_hits}件合致, "
                f"{n_star_hits}件スター株, 精度={u_prec:.1%}"
            )

        # 決定木は訓練データ内のみの精度（母集団精度は計算困難なので訓練結果を表示）
        dt_info = alt_methods.get("decision_tree", {})
        if dt_info.get("best"):
            results["decision_tree"] = {
                "note": "決定木は訓練データ内精度のみ（母集団検証はAND条件に変換後に実施可能）",
                "train_precision": dt_info["best"]["precision"],
            }

        return results

    # ------------------------------------------------------------------
    # 26特徴量計算（star_stock_analyzer.py準拠）
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_wide_features(
        df: pd.DataFrame,
        close_col: str,
        topix_ret_series: pd.Series | None = None,
        sector_ret_10d: float | None = None,
        market_vol_ratio: float | None = None,
    ) -> dict | None:
        """1銘柄の直近データから26個のワイド特徴量を計算"""
        if len(df) < 20:
            return None

        close = df[close_col].astype(float).values
        volume = df["volume"].astype(float).values
        high = df["high"].astype(float).values if "high" in df.columns else close
        low = df["low"].astype(float).values if "low" in df.columns else close
        open_ = df["open"].astype(float).values if "open" in df.columns else close

        n = len(close)
        ret = np.diff(close) / np.where(close[:-1] != 0, close[:-1], 1.0)
        ret = np.where(np.isfinite(ret), ret, 0.0)

        def _ma(arr, w):
            if len(arr) < w:
                return np.full(len(arr), np.nanmean(arr))
            cs = np.cumsum(arr)
            cs = np.insert(cs, 0, 0.0)
            out = np.full(len(arr), np.nan)
            out[w - 1:] = (cs[w:] - cs[:-w]) / w
            for i in range(w - 1):
                out[i] = np.mean(arr[:i + 1])
            return out

        vol_ma5 = _ma(volume, 5)
        vol_ma10 = _ma(volume, 10)
        vol_ma20 = _ma(volume, 20)
        vol_ma40 = _ma(volume, 40) if n >= 40 else _ma(volume, max(n, 1))
        vol_ma60 = _ma(volume, 60) if n >= 60 else _ma(volume, max(n, 1))
        close_ma5 = _ma(close, 5)
        close_ma20 = _ma(close, 20)
        close_ma25 = _ma(close, 25) if n >= 25 else _ma(close, max(n, 1))
        close_ma75 = _ma(close, 75) if n >= 75 else _ma(close, max(n, 1))
        close_ma200 = _ma(close, 200) if n >= 200 else _ma(close, max(n, 1))

        feat = {}

        # A: 出来高ダイナミクス
        feat["vol_ratio_5d_20d"] = float(vol_ma5[-1] / vol_ma20[-1]) if vol_ma20[-1] > 0 else 1.0
        feat["vol_ratio_5d_60d"] = float(vol_ma5[-1] / vol_ma60[-1]) if vol_ma60[-1] > 0 else 1.0
        window_10 = min(10, n)
        feat["vol_surge_count_10d"] = int(np.sum(volume[-window_10:] > vol_ma20[-window_10:] * 2.0))

        if len(ret) >= 10:
            vol_10 = volume[-10:]
            ret_10 = ret[-(min(10, len(ret))):]
            r_len = min(len(vol_10) - 1, len(ret_10))
            if r_len > 0:
                up_m = ret_10[-r_len:] > 0
                up_v = vol_10[-r_len:][up_m].sum()
                total_v = vol_10[-r_len:].sum()
                feat["up_volume_ratio_10d"] = float(up_v / total_v) if total_v > 0 else 0.5
            else:
                feat["up_volume_ratio_10d"] = 0.5
        else:
            feat["up_volume_ratio_10d"] = 0.5

        w20 = min(20, len(ret))
        if w20 > 0:
            r20 = ret[-w20:]
            v20 = volume[-w20:]
            vm20 = vol_ma20[-w20:]
            quiet_mask = (np.abs(r20) < 0.003) & (v20[:len(r20)] > vm20[:len(r20)] * 1.3)
            feat["quiet_accum_rate_20d"] = float(quiet_mask.sum() / w20)
        else:
            feat["quiet_accum_rate_20d"] = 0.0

        if n >= 10:
            first_half = vol_ma5[-10:-5].mean() if len(vol_ma5) >= 10 else vol_ma5.mean()
            second_half = vol_ma5[-5:].mean()
            feat["vol_acceleration"] = float(second_half / first_half) if first_half > 0 else 1.0
        else:
            feat["vol_acceleration"] = 1.0

        if len(ret) >= 20:
            ret_s = pd.Series(ret)
            rolling_std = ret_s.rolling(20, min_periods=5).std().values
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(rolling_std > 0, ret / rolling_std, 0)
            buy_pct = norm.cdf(z)
            bv = volume[1:len(ret) + 1] * buy_pct
            sv_ = volume[1:len(ret) + 1] * (1 - buy_pct)
            tv = volume[1:len(ret) + 1]
            w5 = min(5, len(bv))
            bv_sum = bv[-w5:].sum()
            sv_sum = sv_[-w5:].sum()
            tv_sum = tv[-w5:].sum()
            feat["vpin_5d"] = float(abs(bv_sum - sv_sum) / tv_sum) if tv_sum > 0 else 0.0
        else:
            feat["vpin_5d"] = 0.0

        # A追加: マルチウィンドウバリアント
        feat["vol_ratio_5d_40d"] = float(vol_ma5[-1] / vol_ma40[-1]) if vol_ma40[-1] > 0 else 1.0
        feat["vol_ratio_10d_20d"] = float(vol_ma10[-1] / vol_ma20[-1]) if vol_ma20[-1] > 0 else 1.0
        feat["vol_ratio_10d_40d"] = float(vol_ma10[-1] / vol_ma40[-1]) if vol_ma40[-1] > 0 else 1.0

        window_5 = min(5, n)
        feat["vol_surge_count_5d"] = int(np.sum(volume[-window_5:] > vol_ma20[-window_5:] * 2.0))
        window_20 = min(20, n)
        feat["vol_surge_count_20d"] = int(np.sum(volume[-window_20:] > vol_ma20[-window_20:] * 2.0))

        # up_volume_ratio 5d / 20d
        for _w, _key in [(5, "up_volume_ratio_5d"), (20, "up_volume_ratio_20d")]:
            if len(ret) >= _w:
                _vol_w = volume[-_w:]
                _ret_w = ret[-(min(_w, len(ret))):]
                _rl = min(len(_vol_w) - 1, len(_ret_w))
                if _rl > 0:
                    _up_m = _ret_w[-_rl:] > 0
                    _up_v = _vol_w[-_rl:][_up_m].sum()
                    _total_v = _vol_w[-_rl:].sum()
                    feat[_key] = float(_up_v / _total_v) if _total_v > 0 else 0.5
                else:
                    feat[_key] = 0.5
            else:
                feat[_key] = 0.5

        # vpin_10d
        if len(ret) >= 20:
            w10v = min(10, len(bv))
            bv_sum10 = bv[-w10v:].sum()
            sv_sum10 = sv_[-w10v:].sum()
            tv_sum10 = tv[-w10v:].sum()
            feat["vpin_10d"] = float(abs(bv_sum10 - sv_sum10) / tv_sum10) if tv_sum10 > 0 else 0.0
        else:
            feat["vpin_10d"] = 0.0

        # B: 価格/リターン
        feat["ret_5d"] = float(close[-1] / close[-6] - 1) if n >= 6 and close[-6] > 0 else 0.0
        feat["ret_20d"] = float(close[-1] / close[-21] - 1) if n >= 21 and close[-21] > 0 else (
            float(close[-1] / close[0] - 1) if close[0] > 0 else 0.0
        )
        w10r = min(10, len(ret))
        feat["up_days_ratio_10d"] = float((ret[-w10r:] > 0).mean()) if w10r > 0 else 0.5

        if n >= 2:
            w5g = min(5, n - 1)
            gaps = open_[-w5g:] / close[-w5g - 1:-1] - 1
            gaps = np.where(np.isfinite(gaps), gaps, 0.0)
            feat["max_gap_up_5d"] = float(np.max(gaps)) if len(gaps) > 0 else 0.0
        else:
            feat["max_gap_up_5d"] = 0.0

        w10l = min(10, n)
        if w10l >= 5:
            lows_w = low[-w10l:]
            x = np.arange(len(lows_w), dtype=float)
            try:
                slope = stats.linregress(x, lows_w).slope
                mean_c = np.mean(close[-w10l:])
                feat["higher_lows_slope_10d"] = float(slope / mean_c) if mean_c > 0 else 0.0
            except Exception:
                feat["higher_lows_slope_10d"] = 0.0
        else:
            feat["higher_lows_slope_10d"] = 0.0

        w20p = min(20, n)
        low_min = np.min(low[-w20p:])
        high_max = np.max(high[-w20p:])
        rng = high_max - low_min
        feat["range_position_20d"] = float((close[-1] - low_min) / rng) if rng > 0 else 0.5

        # B追加: マルチウィンドウバリアント
        feat["ret_3d"] = float(close[-1] / close[-4] - 1) if n >= 4 and close[-4] > 0 else 0.0
        feat["ret_10d"] = float(close[-1] / close[-11] - 1) if n >= 11 and close[-11] > 0 else 0.0
        feat["ret_40d"] = float(close[-1] / close[-41] - 1) if n >= 41 and close[-41] > 0 else (
            float(close[-1] / close[0] - 1) if close[0] > 0 else 0.0
        )

        w5r = min(5, len(ret))
        feat["up_days_ratio_5d"] = float((ret[-w5r:] > 0).mean()) if w5r > 0 else 0.5
        w20r = min(20, len(ret))
        feat["up_days_ratio_20d"] = float((ret[-w20r:] > 0).mean()) if w20r > 0 else 0.5

        if n >= 2:
            w10g = min(10, n - 1)
            gaps10 = open_[-w10g:] / close[-w10g - 1:-1] - 1
            gaps10 = np.where(np.isfinite(gaps10), gaps10, 0.0)
            feat["max_gap_up_10d"] = float(np.max(gaps10)) if len(gaps10) > 0 else 0.0
        else:
            feat["max_gap_up_10d"] = 0.0

        for _wl, _key_l in [(5, "higher_lows_slope_5d"), (20, "higher_lows_slope_20d")]:
            _wl_act = min(_wl, n)
            if _wl_act >= 5:
                _lows_w = low[-_wl_act:]
                _x = np.arange(len(_lows_w), dtype=float)
                try:
                    _slope = stats.linregress(_x, _lows_w).slope
                    _mean_c = np.mean(close[-_wl_act:])
                    feat[_key_l] = float(_slope / _mean_c) if _mean_c > 0 else 0.0
                except Exception:
                    feat[_key_l] = 0.0
            else:
                feat[_key_l] = 0.0

        for _wr, _key_r in [(10, "range_position_10d"), (40, "range_position_40d")]:
            _wr_act = min(_wr, n)
            _lo = np.min(low[-_wr_act:])
            _hi = np.max(high[-_wr_act:])
            _rng = _hi - _lo
            feat[_key_r] = float((close[-1] - _lo) / _rng) if _rng > 0 else 0.5

        # C: ボラティリティ・レジーム
        tr_vals = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        if len(tr_vals) >= 20:
            atr5 = np.mean(tr_vals[-5:])
            atr20 = np.mean(tr_vals[-20:])
            feat["atr_ratio_5d_20d"] = float(atr5 / atr20) if atr20 > 0 else 1.0
        else:
            feat["atr_ratio_5d_20d"] = 1.0

        if n >= 20:
            bb_std = pd.Series(close).rolling(20, min_periods=10).std().values
            bb_ma = _ma(close, 20)
            with np.errstate(divide="ignore", invalid="ignore"):
                bb_width = np.where(bb_ma > 0, 2 * bb_std / bb_ma, 0.0)
            bb_width = np.where(np.isfinite(bb_width), bb_width, 0.0)
            valid_bw = bb_width[~np.isnan(bb_width)]
            if len(valid_bw) >= 10:
                current_bw = valid_bw[-1]
                feat["bb_width_pctile_60d"] = float(
                    np.searchsorted(np.sort(valid_bw[-60:]), current_bw) / len(valid_bw[-60:])
                )
            else:
                feat["bb_width_pctile_60d"] = 0.5
        else:
            feat["bb_width_pctile_60d"] = 0.5

        if n >= 20:
            intra = (high - low) / np.where(close > 0, close, 1.0)
            intra = np.where(np.isfinite(intra), intra, 0.0)
            mean_intra_20 = np.mean(intra[-20:])
            feat["intraday_range_ratio_5d"] = float(
                np.mean(intra[-5:]) / mean_intra_20
            ) if mean_intra_20 > 0 else 1.0
        else:
            feat["intraday_range_ratio_5d"] = 1.0

        if len(ret) >= 20:
            rv5 = np.std(ret[-5:])
            rv20 = np.std(ret[-20:])
            feat["realized_vol_5d_vs_20d"] = float(rv5 / rv20) if rv20 > 0 else 1.0
        else:
            feat["realized_vol_5d_vs_20d"] = 1.0

        # C追加: マルチウィンドウバリアント
        if len(tr_vals) >= 40:
            atr5 = np.mean(tr_vals[-5:])
            atr40 = np.mean(tr_vals[-40:])
            feat["atr_ratio_5d_40d"] = float(atr5 / atr40) if atr40 > 0 else 1.0
        else:
            feat["atr_ratio_5d_40d"] = 1.0

        if len(tr_vals) >= 20:
            atr10 = np.mean(tr_vals[-10:])
            atr20_ = np.mean(tr_vals[-20:])
            feat["atr_ratio_10d_20d"] = float(atr10 / atr20_) if atr20_ > 0 else 1.0
        else:
            feat["atr_ratio_10d_20d"] = 1.0

        if n >= 20:
            bb_std_ = pd.Series(close).rolling(20, min_periods=10).std().values
            bb_ma_ = _ma(close, 20)
            with np.errstate(divide="ignore", invalid="ignore"):
                bb_width_ = np.where(bb_ma_ > 0, 2 * bb_std_ / bb_ma_, 0.0)
            bb_width_ = np.where(np.isfinite(bb_width_), bb_width_, 0.0)
            valid_bw_ = bb_width_[~np.isnan(bb_width_)]
            if len(valid_bw_) >= 10:
                current_bw_ = valid_bw_[-1]
                lookback_120 = min(120, len(valid_bw_))
                feat["bb_width_pctile_120d"] = float(
                    np.searchsorted(np.sort(valid_bw_[-lookback_120:]), current_bw_) / lookback_120
                )
            else:
                feat["bb_width_pctile_120d"] = 0.5
        else:
            feat["bb_width_pctile_120d"] = 0.5

        if n >= 20:
            intra_ = (high - low) / np.where(close > 0, close, 1.0)
            intra_ = np.where(np.isfinite(intra_), intra_, 0.0)
            mean_intra_20_ = np.mean(intra_[-20:])
            feat["intraday_range_ratio_10d"] = float(
                np.mean(intra_[-10:]) / mean_intra_20_
            ) if mean_intra_20_ > 0 else 1.0
        else:
            feat["intraday_range_ratio_10d"] = 1.0

        if len(ret) >= 40:
            rv5_ = np.std(ret[-5:])
            rv40 = np.std(ret[-40:])
            feat["realized_vol_5d_vs_40d"] = float(rv5_ / rv40) if rv40 > 0 else 1.0
        else:
            feat["realized_vol_5d_vs_40d"] = 1.0

        if len(ret) >= 20:
            rv10_ = np.std(ret[-10:])
            rv20_ = np.std(ret[-20:])
            feat["realized_vol_10d_vs_20d"] = float(rv10_ / rv20_) if rv20_ > 0 else 1.0
        else:
            feat["realized_vol_10d_vs_20d"] = 1.0

        # D: トレンド/OBV
        signed_vol = np.sign(ret) * volume[1:len(ret) + 1]
        obv = np.cumsum(signed_vol)
        w10o = min(10, len(obv))
        if w10o >= 5:
            x = np.arange(w10o, dtype=float)
            try:
                slope = stats.linregress(x, obv[-w10o:]).slope
                avg_vol = np.mean(volume[-w10o:])
                feat["obv_slope_10d"] = float(slope / avg_vol) if avg_vol > 0 else 0.0
            except Exception:
                feat["obv_slope_10d"] = 0.0
        else:
            feat["obv_slope_10d"] = 0.0

        if len(obv) >= 20:
            try:
                corr, _ = stats.spearmanr(close[-20:], obv[-20:])
                feat["obv_divergence"] = float(corr) if np.isfinite(corr) else 0.0
            except Exception:
                feat["obv_divergence"] = 0.0
        else:
            feat["obv_divergence"] = 0.0

        feat["ma5_ma25_gap"] = float(
            (close_ma5[-1] - close_ma25[-1]) / close_ma25[-1]
        ) if close_ma25[-1] > 0 else 0.0
        feat["price_vs_ma25_pct"] = float(
            close[-1] / close_ma25[-1] - 1
        ) if close_ma25[-1] > 0 else 0.0

        consec = 0
        for i in range(len(ret) - 1, -1, -1):
            if ret[i] > 0:
                consec += 1
            else:
                break
        feat["consecutive_up_days"] = min(consec, 20)

        # D追加: マルチウィンドウバリアント
        # obv_slope_5d / obv_slope_20d
        for _wo, _key_o in [(5, "obv_slope_5d"), (20, "obv_slope_20d")]:
            _wo_act = min(_wo, len(obv))
            if _wo_act >= 5:
                _x = np.arange(_wo_act, dtype=float)
                try:
                    _slope = stats.linregress(_x, obv[-_wo_act:]).slope
                    _avg_vol = np.mean(volume[-_wo_act:])
                    feat[_key_o] = float(_slope / _avg_vol) if _avg_vol > 0 else 0.0
                except Exception:
                    feat[_key_o] = 0.0
            else:
                feat[_key_o] = 0.0

        # obv_divergence_40d
        if len(obv) >= 40:
            try:
                corr40, _ = stats.spearmanr(close[-40:], obv[-40:])
                feat["obv_divergence_40d"] = float(corr40) if np.isfinite(corr40) else 0.0
            except Exception:
                feat["obv_divergence_40d"] = 0.0
        else:
            feat["obv_divergence_40d"] = 0.0

        # MA gap variants (MA=5,25,75,200)
        feat["ma25_ma75_gap"] = float(
            (close_ma25[-1] - close_ma75[-1]) / close_ma75[-1]
        ) if n >= 75 and close_ma75[-1] > 0 else 0.0
        feat["ma5_ma75_gap"] = float(
            (close_ma5[-1] - close_ma75[-1]) / close_ma75[-1]
        ) if n >= 75 and close_ma75[-1] > 0 else 0.0

        # price_vs_ma variants
        feat["price_vs_ma5_pct"] = float(
            close[-1] / close_ma5[-1] - 1
        ) if close_ma5[-1] > 0 else 0.0
        feat["price_vs_ma75_pct"] = float(
            close[-1] / close_ma75[-1] - 1
        ) if n >= 75 and close_ma75[-1] > 0 else 0.0
        feat["price_vs_ma200_pct"] = float(
            close[-1] / close_ma200[-1] - 1
        ) if n >= 200 and close_ma200[-1] > 0 else 0.0

        # E: クロスセクショナル
        stock_ret_10d = float(close[-1] / close[-min(11, n)] - 1) if close[-min(11, n)] > 0 else 0.0
        if sector_ret_10d is not None:
            feat["sector_rel_ret_10d"] = stock_ret_10d - sector_ret_10d
        else:
            feat["sector_rel_ret_10d"] = stock_ret_10d

        if topix_ret_series is not None and len(ret) >= 20:
            try:
                if "date" in df.columns:
                    dates_ts = pd.to_datetime(df["date"].values)
                    stock_s = pd.Series(ret, index=dates_ts[1:])
                    topix_idx = pd.to_datetime(topix_ret_series.index)
                    topix_aligned = pd.Series(topix_ret_series.values, index=topix_idx)
                    common = stock_s.index.intersection(topix_aligned.index)
                    if len(common) >= 15:
                        sr = stock_s.loc[common].values[-20:]
                        tr = topix_aligned.loc[common].values[-20:]
                        if len(sr) >= 10 and len(tr) >= 10:
                            cov = np.cov(sr, tr)
                            feat["topix_beta_20d"] = float(
                                cov[0, 1] / cov[1, 1]
                            ) if cov[1, 1] > 0 else 1.0
                        else:
                            feat["topix_beta_20d"] = 1.0
                    else:
                        feat["topix_beta_20d"] = 1.0
                else:
                    feat["topix_beta_20d"] = 1.0
            except Exception:
                feat["topix_beta_20d"] = 1.0
        else:
            feat["topix_beta_20d"] = 1.0

        if topix_ret_series is not None and len(ret) >= 60 and "date" in df.columns:
            try:
                dates_ts = pd.to_datetime(df["date"].values)
                stock_s = pd.Series(ret, index=dates_ts[1:])
                topix_idx = pd.to_datetime(topix_ret_series.index)
                topix_aligned = pd.Series(topix_ret_series.values, index=topix_idx)
                common = stock_s.index.intersection(topix_aligned.index)
                if len(common) >= 60:
                    resid = (stock_s.loc[common].values
                             - topix_aligned.loc[common].values * feat.get("topix_beta_20d", 1.0))
                    rv10 = np.std(resid[-10:])
                    rv60 = np.std(resid[-60:])
                    feat["residual_vol_ratio"] = float(rv10 / rv60) if rv60 > 0 else 1.0
                else:
                    rv10 = np.std(ret[-10:])
                    rv60 = np.std(ret[-60:])
                    feat["residual_vol_ratio"] = float(rv10 / rv60) if rv60 > 0 else 1.0
            except Exception:
                rv10 = np.std(ret[-10:])
                rv60 = np.std(ret[-60:])
                feat["residual_vol_ratio"] = float(rv10 / rv60) if rv60 > 0 else 1.0
        elif len(ret) >= 60:
            rv10 = np.std(ret[-10:])
            rv60 = np.std(ret[-60:])
            feat["residual_vol_ratio"] = float(rv10 / rv60) if rv60 > 0 else 1.0
        else:
            feat["residual_vol_ratio"] = 1.0

        stock_vol_ratio = feat["vol_ratio_5d_60d"]
        if market_vol_ratio is not None and market_vol_ratio > 0:
            feat["vol_vs_market_vol"] = float(stock_vol_ratio / market_vol_ratio)
        else:
            feat["vol_vs_market_vol"] = stock_vol_ratio

        # E追加: マルチウィンドウバリアント
        stock_ret_5d = float(close[-1] / close[-min(6, n)] - 1) if close[-min(6, n)] > 0 else 0.0
        stock_ret_20d = float(close[-1] / close[-min(21, n)] - 1) if close[-min(21, n)] > 0 else 0.0
        if sector_ret_10d is not None:
            feat["sector_rel_ret_5d"] = stock_ret_5d - sector_ret_10d
            feat["sector_rel_ret_20d"] = stock_ret_20d - sector_ret_10d
        else:
            feat["sector_rel_ret_5d"] = stock_ret_5d
            feat["sector_rel_ret_20d"] = stock_ret_20d

        # topix_beta_40d
        if topix_ret_series is not None and len(ret) >= 40 and "date" in df.columns:
            try:
                dates_ts = pd.to_datetime(df["date"].values)
                stock_s_ = pd.Series(ret, index=dates_ts[1:])
                topix_idx_ = pd.to_datetime(topix_ret_series.index)
                topix_aligned_ = pd.Series(topix_ret_series.values, index=topix_idx_)
                common_ = stock_s_.index.intersection(topix_aligned_.index)
                if len(common_) >= 30:
                    sr_ = stock_s_.loc[common_].values[-40:]
                    tr_ = topix_aligned_.loc[common_].values[-40:]
                    if len(sr_) >= 20 and len(tr_) >= 20:
                        cov_ = np.cov(sr_, tr_)
                        feat["topix_beta_40d"] = float(
                            cov_[0, 1] / cov_[1, 1]
                        ) if cov_[1, 1] > 0 else 1.0
                    else:
                        feat["topix_beta_40d"] = 1.0
                else:
                    feat["topix_beta_40d"] = 1.0
            except Exception:
                feat["topix_beta_40d"] = 1.0
        else:
            feat["topix_beta_40d"] = 1.0

        # F: 信用取引 (8特徴量)
        has_margin = "margin_ratio" in df.columns and df["margin_ratio"].notna().any()
        feat["has_margin_data"] = 1.0 if has_margin else 0.0

        if has_margin:
            feat["margin_ratio"] = float(
                df["margin_ratio"].dropna().iloc[-1]
            ) if df["margin_ratio"].notna().any() else 0.0

            feat["margin_buy_change_pct"] = float(
                df["margin_buy_change_pct"].dropna().iloc[-1]
            ) if "margin_buy_change_pct" in df.columns and df["margin_buy_change_pct"].notna().any() else 0.0

            feat["margin_ratio_change_pct"] = float(
                df["margin_ratio_change_pct"].dropna().iloc[-1]
            ) if "margin_ratio_change_pct" in df.columns and df["margin_ratio_change_pct"].notna().any() else 0.0

            # 回転日数: margin_buy_balance / vol_ma20
            if "margin_buy_balance" in df.columns and df["margin_buy_balance"].notna().any():
                mb = float(df["margin_buy_balance"].dropna().iloc[-1])
                feat["margin_buy_turnover_days"] = float(mb / vol_ma20[-1]) if vol_ma20[-1] > 0 else 0.0
            else:
                feat["margin_buy_turnover_days"] = 0.0

            # 買い残/当日出来高
            if "margin_buy_balance" in df.columns and df["margin_buy_balance"].notna().any():
                mb = float(df["margin_buy_balance"].dropna().iloc[-1])
                feat["margin_buy_vol_ratio"] = float(mb / volume[-1]) if volume[-1] > 0 else 0.0
            else:
                feat["margin_buy_vol_ratio"] = 0.0

            # ネットポジション: (買残 - 売残) / 買残
            if ("margin_buy_balance" in df.columns and "margin_sell_balance" in df.columns
                    and df["margin_buy_balance"].notna().any()):
                mb = float(df["margin_buy_balance"].dropna().iloc[-1])
                ms = float(df["margin_sell_balance"].dropna().iloc[-1]) if df["margin_sell_balance"].notna().any() else 0.0
                feat["margin_net_position"] = float((mb - ms) / mb) if mb > 0 else 0.0
            else:
                feat["margin_net_position"] = 0.0

            # ダイバージェンス: margin_ratio と close の Spearman相関（20日）
            if "margin_ratio" in df.columns and n >= 20:
                mr_vals = df["margin_ratio"].values[-20:]
                close_20 = close[-20:]
                valid_mask = np.isfinite(mr_vals) & np.isfinite(close_20)
                if valid_mask.sum() >= 10:
                    c_arr = close_20[valid_mask]
                    m_arr = mr_vals[valid_mask]
                    # 定数配列はspearmanrが未定義 → スキップ
                    if np.ptp(c_arr) == 0 or np.ptp(m_arr) == 0:
                        feat["margin_divergence"] = 0.0
                    else:
                        try:
                            corr, _ = stats.spearmanr(c_arr, m_arr)
                            feat["margin_divergence"] = float(corr) if np.isfinite(corr) else 0.0
                        except Exception:
                            feat["margin_divergence"] = 0.0
                else:
                    feat["margin_divergence"] = 0.0
            else:
                feat["margin_divergence"] = 0.0
        else:
            feat["margin_ratio"] = 0.0
            feat["margin_buy_change_pct"] = 0.0
            feat["margin_ratio_change_pct"] = 0.0
            feat["margin_buy_turnover_days"] = 0.0
            feat["margin_buy_vol_ratio"] = 0.0
            feat["margin_net_position"] = 0.0
            feat["margin_divergence"] = 0.0

        # G: テクニカル追加 (6特徴量)
        # MACD histogram: (EMA12 - EMA26) - Signal(9) / close
        if n >= 26:
            close_s = pd.Series(close)
            ema12 = close_s.ewm(span=12, adjust=False).mean().values
            ema26 = close_s.ewm(span=26, adjust=False).mean().values
            macd_line = ema12 - ema26
            signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
            histogram = macd_line - signal_line
            feat["macd_histogram"] = float(histogram[-1] / close[-1]) if close[-1] > 0 else 0.0
        else:
            feat["macd_histogram"] = 0.0

        # Stochastic %K (14日)
        if n >= 14:
            low_14 = np.min(low[-14:])
            high_14 = np.max(high[-14:])
            rng_14 = high_14 - low_14
            feat["stochastic_k"] = float((close[-1] - low_14) / rng_14) if rng_14 > 0 else 0.5
        else:
            feat["stochastic_k"] = 0.5

        # Williams %R (14日)
        if n >= 14:
            high_14 = np.max(high[-14:])
            low_14 = np.min(low[-14:])
            rng_14 = high_14 - low_14
            feat["williams_r"] = float((high_14 - close[-1]) / rng_14) if rng_14 > 0 else 0.5
        else:
            feat["williams_r"] = 0.5

        # CCI (20日): (TP - TP_mean) / (0.015 * TP_mean_deviation)
        if n >= 20:
            tp = (high[-20:] + low[-20:] + close[-20:]) / 3.0
            tp_mean = np.mean(tp)
            tp_mean_dev = np.mean(np.abs(tp - tp_mean))
            feat["cci_20d"] = float(
                (tp[-1] - tp_mean) / (0.015 * tp_mean_dev)
            ) if tp_mean_dev > 0 else 0.0
        else:
            feat["cci_20d"] = 0.0

        # MA乖離率 (25日, 75日)
        if n >= 25:
            ma25 = np.mean(close[-25:])
            feat["ma_deviation_25d"] = float(close[-1] / ma25 - 1) if ma25 > 0 else 0.0
        else:
            feat["ma_deviation_25d"] = 0.0

        if n >= 75:
            ma75 = np.mean(close[-75:])
            feat["ma_deviation_75d"] = float(close[-1] / ma75 - 1) if ma75 > 0 else 0.0
        else:
            feat["ma_deviation_75d"] = 0.0

        # G追加: マルチウィンドウバリアント
        # CCI (10日)
        if n >= 10:
            tp10 = (high[-10:] + low[-10:] + close[-10:]) / 3.0
            tp10_mean = np.mean(tp10)
            tp10_mean_dev = np.mean(np.abs(tp10 - tp10_mean))
            feat["cci_10d"] = float(
                (tp10[-1] - tp10_mean) / (0.015 * tp10_mean_dev)
            ) if tp10_mean_dev > 0 else 0.0
        else:
            feat["cci_10d"] = 0.0

        # MA乖離率 (5日)
        if n >= 5:
            ma5_val = np.mean(close[-5:])
            feat["ma_deviation_5d"] = float(close[-1] / ma5_val - 1) if ma5_val > 0 else 0.0
        else:
            feat["ma_deviation_5d"] = 0.0

        # MA乖離率 (200日)
        if n >= 200:
            ma200_val = np.mean(close[-200:])
            feat["ma_deviation_200d"] = float(close[-1] / ma200_val - 1) if ma200_val > 0 else 0.0
        else:
            feat["ma_deviation_200d"] = 0.0

        # H: 流動性 (3特徴量)
        # Amihud非流動性 (20日): mean(|ret| / 売買代金) * 10^6
        if len(ret) >= 20:
            turnover_20 = close[-20:] * volume[-20:]  # 売買代金の近似
            abs_ret_20 = np.abs(ret[-20:])
            turnover_valid = turnover_20[:len(abs_ret_20)]
            with np.errstate(divide="ignore", invalid="ignore"):
                illiq = np.where(turnover_valid > 0, abs_ret_20 / turnover_valid, 0.0)
            illiq = np.where(np.isfinite(illiq), illiq, 0.0)
            feat["amihud_illiquidity_20d"] = float(np.mean(illiq) * 1e6)
        else:
            feat["amihud_illiquidity_20d"] = 0.0

        # 売買代金変化 (10日/20日)
        if n >= 20:
            turnover = close * volume
            to_ma10 = np.mean(turnover[-10:])
            to_ma20 = np.mean(turnover[-20:])
            feat["turnover_change_10d_20d"] = float(to_ma10 / to_ma20) if to_ma20 > 0 else 1.0
        else:
            feat["turnover_change_10d_20d"] = 1.0

        # スプレッド代理 (5日): mean((high - low) / close)
        if n >= 5:
            hl_spread = (high[-5:] - low[-5:]) / np.where(close[-5:] > 0, close[-5:], 1.0)
            hl_spread = np.where(np.isfinite(hl_spread), hl_spread, 0.0)
            feat["spread_proxy_5d"] = float(np.mean(hl_spread))
        else:
            feat["spread_proxy_5d"] = 0.0

        # H追加: マルチウィンドウバリアント
        # Amihud非流動性 (10日)
        if len(ret) >= 10:
            turnover_10_ = close[-10:] * volume[-10:]
            abs_ret_10_ = np.abs(ret[-10:])
            turnover_valid_10 = turnover_10_[:len(abs_ret_10_)]
            with np.errstate(divide="ignore", invalid="ignore"):
                illiq_10 = np.where(turnover_valid_10 > 0, abs_ret_10_ / turnover_valid_10, 0.0)
            illiq_10 = np.where(np.isfinite(illiq_10), illiq_10, 0.0)
            feat["amihud_illiquidity_10d"] = float(np.mean(illiq_10) * 1e6)
        else:
            feat["amihud_illiquidity_10d"] = 0.0

        # 売買代金変化 (5日/20日, 5日/10日)
        if n >= 20:
            turnover_ = close * volume
            to_ma5_ = np.mean(turnover_[-5:])
            to_ma10_ = np.mean(turnover_[-10:])
            to_ma20_ = np.mean(turnover_[-20:])
            feat["turnover_change_5d_20d"] = float(to_ma5_ / to_ma20_) if to_ma20_ > 0 else 1.0
            feat["turnover_change_5d_10d"] = float(to_ma5_ / to_ma10_) if to_ma10_ > 0 else 1.0
        else:
            feat["turnover_change_5d_20d"] = 1.0
            feat["turnover_change_5d_10d"] = 1.0

        # スプレッド代理 (10日)
        if n >= 10:
            hl_spread_10 = (high[-10:] - low[-10:]) / np.where(close[-10:] > 0, close[-10:], 1.0)
            hl_spread_10 = np.where(np.isfinite(hl_spread_10), hl_spread_10, 0.0)
            feat["spread_proxy_10d"] = float(np.mean(hl_spread_10))
        else:
            feat["spread_proxy_10d"] = 0.0

        # I: 価格パターン (3特徴量)
        # ギャップ頻度 (20日): |gap| > 0.5% の日数比率
        if n >= 21:
            gaps_20 = open_[-20:] / np.where(close[-21:-1] > 0, close[-21:-1], 1.0) - 1
            gaps_20 = np.where(np.isfinite(gaps_20), gaps_20, 0.0)
            feat["gap_frequency_20d"] = float(np.mean(np.abs(gaps_20) > 0.005))
        else:
            feat["gap_frequency_20d"] = 0.0

        # 高値更新比率 (10日): high > 前日high の日数比率
        if n >= 11:
            hh = high[-10:] > high[-11:-1]
            feat["higher_highs_ratio_10d"] = float(np.mean(hh))
        else:
            feat["higher_highs_ratio_10d"] = 0.5

        # 52週高値近接度: close / 52週高値
        lookback_52w = min(252, n)
        high_52w = np.max(high[-lookback_52w:])
        feat["proximity_52w_high"] = float(close[-1] / high_52w) if high_52w > 0 else 0.0

        # I追加: マルチウィンドウバリアント
        # ギャップ頻度 (10日)
        if n >= 11:
            gaps_10_ = open_[-10:] / np.where(close[-11:-1] > 0, close[-11:-1], 1.0) - 1
            gaps_10_ = np.where(np.isfinite(gaps_10_), gaps_10_, 0.0)
            feat["gap_frequency_10d"] = float(np.mean(np.abs(gaps_10_) > 0.005))
        else:
            feat["gap_frequency_10d"] = 0.0

        # 高値更新比率 (5日, 20日)
        if n >= 6:
            hh5 = high[-5:] > high[-6:-1]
            feat["higher_highs_ratio_5d"] = float(np.mean(hh5))
        else:
            feat["higher_highs_ratio_5d"] = 0.5

        if n >= 21:
            hh20 = high[-20:] > high[-21:-1]
            feat["higher_highs_ratio_20d"] = float(np.mean(hh20))
        else:
            feat["higher_highs_ratio_20d"] = 0.5

        # NaN/Inf安全化
        for k in feat:
            v = feat[k]
            if not np.isfinite(v):
                feat[k] = 0.0

        return feat
