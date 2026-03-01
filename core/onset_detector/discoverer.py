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

# 46特徴量キー（A〜I カテゴリ）
WIDE_FEATURE_KEYS = [
    # A: 出来高ダイナミクス (7)
    "vol_ratio_5d_20d", "vol_ratio_5d_60d", "vol_surge_count_10d",
    "up_volume_ratio_10d", "quiet_accum_rate_20d", "vol_acceleration", "vpin_5d",
    # B: 価格/リターン (6)
    "ret_5d", "ret_20d", "up_days_ratio_10d", "max_gap_up_5d",
    "higher_lows_slope_10d", "range_position_20d",
    # C: ボラティリティ・レジーム (4)
    "atr_ratio_5d_20d", "bb_width_pctile_60d", "intraday_range_ratio_5d",
    "realized_vol_5d_vs_20d",
    # D: トレンド/OBV (4)
    "obv_slope_10d", "obv_divergence", "ma5_ma20_gap", "price_vs_ma20_pct",
    "consecutive_up_days",
    # E: クロスセクショナル (4)
    "sector_rel_ret_10d", "topix_beta_20d", "residual_vol_ratio",
    "vol_vs_market_vol",
    # F: 信用取引 (8)
    "margin_ratio", "margin_buy_change_pct", "margin_ratio_change_pct",
    "margin_buy_turnover_days", "margin_buy_vol_ratio", "margin_net_position",
    "margin_divergence", "has_margin_data",
    # G: テクニカル追加 (6)
    "macd_histogram", "stochastic_k", "williams_r",
    "cci_20d", "ma_deviation_25d", "ma_deviation_75d",
    # H: 流動性 (3)
    "amihud_illiquidity_20d", "turnover_change_10d_20d", "spread_proxy_5d",
    # I: 価格パターン (3)
    "gap_frequency_20d", "higher_highs_ratio_10d", "proximity_52w_high",
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
    # B: 価格/リターン
    "ret_5d": "リターン(5日)",
    "ret_20d": "リターン(20日)",
    "up_days_ratio_10d": "上昇日比率(10日)",
    "max_gap_up_5d": "最大ギャップアップ(5日)",
    "higher_lows_slope_10d": "安値切上り傾き(10日)",
    "range_position_20d": "レンジ位置(20日)",
    # C: ボラティリティ・レジーム
    "atr_ratio_5d_20d": "日中値幅比(5/20日)",
    "bb_width_pctile_60d": "価格帯収縮度(60日)",
    "intraday_range_ratio_5d": "日中値幅比(5/20日)",
    "realized_vol_5d_vs_20d": "短期ボラ比(5/20日)",
    # D: トレンド/OBV
    "obv_slope_10d": "出来高累計傾き(10日)",
    "obv_divergence": "出来高価格連動度",
    "ma5_ma20_gap": "MA(5-20)乖離率",
    "price_vs_ma20_pct": "対MA20乖離率",
    "consecutive_up_days": "連続上昇日数",
    # E: クロスセクショナル
    "sector_rel_ret_10d": "セクター相対リターン(10日)",
    "topix_beta_20d": "市場感応度(20日)",
    "residual_vol_ratio": "固有ボラ比(10/60日)",
    "vol_vs_market_vol": "対市場出来高比",
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
    # H: 流動性
    "amihud_illiquidity_20d": "流動性の低さ(20日)",
    "turnover_change_10d_20d": "売買代金変化(10/20日)",
    "spread_proxy_5d": "スプレッド代理(5日)",
    # I: 価格パターン
    "gap_frequency_20d": "ギャップ頻度(20日)",
    "higher_highs_ratio_10d": "高値更新比率(10日)",
    "proximity_52w_high": "52週高値近接度",
}

# 各特徴量の平易な日本語説明
WIDE_FEATURE_DESCRIPTIONS_JP = {
    "vol_ratio_5d_20d": "直近5日の出来高が1ヶ月平均の何倍か（大きいほど出来高急増）",
    "vol_ratio_5d_60d": "直近5日の出来高が3ヶ月平均の何倍か（大きいほど出来高急増）",
    "vol_surge_count_10d": "直近10日で出来高が平均の2倍を超えた日数",
    "up_volume_ratio_10d": "直近10日の取引のうち株価が上がった日の出来高割合（買い優勢度）",
    "quiet_accum_rate_20d": "株価が横ばいなのに出来高が多い日の比率（機関投資家の静かな買い集め）",
    "vol_acceleration": "直近5日の出来高が前の5日より増えた倍率",
    "vpin_5d": "直近5日の買い注文と売り注文の偏り度合い（高いほど一方向に集中）",
    "ret_5d": "直近5営業日（1週間）の株価騰落率",
    "ret_20d": "直近20営業日（約1ヶ月）の株価騰落率",
    "up_days_ratio_10d": "直近10日のうち株価が上昇した日の割合",
    "max_gap_up_5d": "直近5日で最も大きかった寄り付き窓開け上昇幅",
    "higher_lows_slope_10d": "直近10日の安値の切り上がり速度（正=下値が堅固に上昇）",
    "range_position_20d": "直近20日の高値-安値の範囲内での現在値の位置（1=高値圏、0=安値圏）",
    "atr_ratio_5d_20d": "直近5日の1日の値動き幅が20日平均より大きいか（ボラ拡大度）",
    "bb_width_pctile_60d": "価格帯（ボリンジャーバンド幅）の過去60日中での細さのランク（低=収縮中）",
    "intraday_range_ratio_5d": "直近5日の日中の値幅が20日平均より大きいか",
    "realized_vol_5d_vs_20d": "直近5日の日々の値動きの激しさが20日平均より大きいか",
    "obv_slope_10d": "出来高の累積値（上昇日プラス・下落日マイナス）の10日トレンド方向",
    "obv_divergence": "株価と出来高累計値の連動度（高い=出来高が株価上昇を裏付け）",
    "ma5_ma20_gap": "5日移動平均が20日移動平均より何%上にあるか",
    "price_vs_ma20_pct": "現在株価が20日移動平均より何%上にあるか",
    "consecutive_up_days": "直近の連続上昇日数",
    "sector_rel_ret_10d": "同業他社の平均に対してその銘柄だけ何%余分に上昇したか（10日）",
    "topix_beta_20d": "市場全体（TOPIX）が動いたときのその銘柄の感応度（高い=市場と同方向に大きく動く）",
    "residual_vol_ratio": "市場の動きを除いた銘柄固有の値動きの激しさ（直近vs長期）",
    "vol_vs_market_vol": "市場全体と比べた出来高の多さ",
    "margin_ratio": "信用買い残÷信用売り残の倍率（高い=買い方が優勢）",
    "margin_buy_change_pct": "信用買い残の直近変化率（正=新規信用買いが増加）",
    "margin_ratio_change_pct": "貸借倍率の変化率（正=買い方優勢が強まる）",
    "margin_buy_turnover_days": "信用買い残を一日の出来高で割った返済所要日数（低い=返済が速い）",
    "margin_buy_vol_ratio": "信用買い残を出来高で割った比率",
    "margin_net_position": "信用買い残から信用売り残を引いた差（大きい=買い方優勢）",
    "margin_divergence": "信用買い残と株価の乖離度合い",
    "has_margin_data": "信用取引データが存在するか",
    "macd_histogram": "短期と長期の移動平均の差の変化（正=上昇モメンタムが加速）",
    "stochastic_k": "過去一定期間の高安値の範囲内での現在値の位置（高い=高値圏）",
    "williams_r": "高値に対する現在値の近さ（0に近い=高値圏）",
    "cci_20d": "価格の移動平均からの乖離を標準化した指数（高い=上昇トレンドが強い）",
    "ma_deviation_25d": "現在株価が25日移動平均より何%上にあるか",
    "ma_deviation_75d": "現在株価が75日移動平均より何%上にあるか（長期トレンドとの乖離）",
    "amihud_illiquidity_20d": "値動き÷出来高の比率（高い=少ない出来高で大きく動く＝流動性が低い）",
    "turnover_change_10d_20d": "直近10日の売買代金が20日平均の何倍か",
    "spread_proxy_5d": "1日の高値と安値の差（高い=価格が跳びやすい＝流動性が低い）",
    "gap_frequency_20d": "直近20日のうち寄り付きで窓開けした日の比率",
    "higher_highs_ratio_10d": "直近10日のうち前日高値を超えた日の割合",
    "proximity_52w_high": "52週高値に対する現在値の近さ（1=52週高値圏）",
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

        # --- Step 3: 共通特徴量発見 ---
        _progress("共通特徴量発見中...")
        common_features = self._find_discriminative_features(
            star_stocks, all_prices, topix_ret_series, listed_stocks, close_col,
            progress_callback=_progress,
        )

        # --- Step 4: 追加スター株発見 ---
        _progress("追加スター株探索中...")
        additional_stars = self._discover_additional_stars(
            common_features, star_codes, all_prices, topix, topix_ret_series,
            listed_stocks, close_col,
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

        # --- Step 5: 初動特定 ---
        _progress("初動日特定中...")
        onset_dates = self._detect_onset_dates(
            all_stars, all_prices, close_col, progress_callback=_progress,
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
    ) -> dict:
        """Youden's J最適化 + コンボ探索で判別特徴量を発見"""
        cfg = self.config
        star_codes = set(str(s["code"]) for s in star_stocks)
        all_codes = [str(c) for c in all_prices["code"].unique()]
        non_star_codes = [c for c in all_codes if c not in star_codes]

        # --- サンプル構築 ---
        pos_features = []  # 正例: スター株
        neg_features = []  # 負例: 非スター株

        # 正例: 各スター株の直近データから特徴量
        for star in star_stocks:
            code = str(star["code"])
            grp = all_prices[all_prices["code"] == code].sort_values("date")
            if len(grp) < 20:
                continue
            feat = self._compute_wide_features(grp, close_col, topix_ret_series)
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

        # --- コンボ探索 ---
        if progress_callback:
            progress_callback("コンボ探索中...")

        top_n = min(15, len(signals))
        top_signals = signals[:top_n]
        thresholds = {s["feature"]: s["threshold"] for s in top_signals}

        combo_results = []

        # 最低合致銘柄数: 正例の30%以上、最低3件（小サンプル問題を軽減）
        min_total_hits = max(3, int(n_pos * 0.3))

        def _make_combo_entry(features_names, thresholds_list, tp, fp, total_pred):
            prec = tp / total_pred if total_pred > 0 else 0
            recall = tp / n_pos if n_pos > 0 else 0
            lift = prec / base_rate if base_rate > 0 else 0
            # 信頼度フラグ: tp >= 5 かつ fp が一定以下なら高信頼
            reliability = "高" if tp >= 5 else ("中" if tp >= 3 else "低")
            return {
                "features": list(features_names),
                "features_jp": [WIDE_FEATURE_LABELS_JP.get(f, f) for f in features_names],
                "thresholds": list(thresholds_list),
                "directions": [">="] * len(features_names),
                "n_features": len(features_names),   # 特徴量の数
                "n_combo": total_pred,                # 合致した銘柄の総数
                "total_hits": total_pred,             # 同上（後方互換）
                "precision": round(prec, 4),
                "recall": round(recall, 4),
                "lift": round(lift, 2),
                "tp": tp,
                "fp": fp,
                "reliability": reliability,
            }

        # 2特徴量コンボ
        for (s1, s2) in combinations(top_signals, 2):
            f1_name, f2_name = s1["feature"], s2["feature"]
            i1 = WIDE_FEATURE_KEYS.index(f1_name)
            i2 = WIDE_FEATURE_KEYS.index(f2_name)
            pred = (X[:, i1] >= s1["threshold"]) & (X[:, i2] >= s2["threshold"])
            tp = int((pred & (labels == 1)).sum())
            fp = int((pred & (labels == 0)).sum())
            total_pred = int(pred.sum())
            if total_pred < min_total_hits:
                continue
            combo_results.append(
                _make_combo_entry([f1_name, f2_name], [s1["threshold"], s2["threshold"]],
                                  tp, fp, total_pred)
            )

        # 3・4特徴量コンボ（上位10から）— 多特徴量は合致数が少なくなるため閾値を緩める
        min_hits_multi = max(3, int(n_pos * 0.15))  # 3+特徴量用：最低15%カバー
        top_10 = top_signals[:min(10, len(top_signals))]

        for (s1, s2, s3) in combinations(top_10, 3):
            f1_name, f2_name, f3_name = s1["feature"], s2["feature"], s3["feature"]
            i1 = WIDE_FEATURE_KEYS.index(f1_name)
            i2 = WIDE_FEATURE_KEYS.index(f2_name)
            i3 = WIDE_FEATURE_KEYS.index(f3_name)
            pred = (
                (X[:, i1] >= s1["threshold"]) &
                (X[:, i2] >= s2["threshold"]) &
                (X[:, i3] >= s3["threshold"])
            )
            tp = int((pred & (labels == 1)).sum())
            fp = int((pred & (labels == 0)).sum())
            total_pred = int(pred.sum())
            if total_pred < min_hits_multi:
                continue
            combo_results.append(
                _make_combo_entry([f1_name, f2_name, f3_name],
                                  [s1["threshold"], s2["threshold"], s3["threshold"]],
                                  tp, fp, total_pred)
            )

        # 4特徴量コンボ（上位8から）
        top_8 = top_signals[:min(8, len(top_signals))]
        for (s1, s2, s3, s4) in combinations(top_8, 4):
            f_names = [s["feature"] for s in (s1, s2, s3, s4)]
            indices = [WIDE_FEATURE_KEYS.index(f) for f in f_names]
            ths = [s["threshold"] for s in (s1, s2, s3, s4)]
            pred = np.ones(len(X), dtype=bool)
            for idx, th in zip(indices, ths):
                pred &= (X[:, idx] >= th)
            tp = int((pred & (labels == 1)).sum())
            fp = int((pred & (labels == 0)).sum())
            total_pred = int(pred.sum())
            if total_pred < min_hits_multi:
                continue
            combo_results.append(
                _make_combo_entry(f_names, ths, tp, fp, total_pred)
            )

        # Precision降順 → recall降順でソート
        combo_results.sort(key=lambda c: (c["precision"], c["recall"]), reverse=True)

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

        return {
            "signals": signals,
            "combo_signals": combo_results[:30],
            "best_combos": best_combos[:10],
            "n_star": n_pos,
            "n_non_star": n_neg,
            "base_rate": round(base_rate, 4),
            "feature_keys": WIDE_FEATURE_KEYS,
        }

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
    ) -> list[dict]:
        """ベストコンボで全銘柄をスキャンし追加スター株を発見"""
        cfg = self.config
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
    ) -> dict:
        """全スター株の初動日を10シグナル方式で特定

        Returns
        -------
        dict: code -> {"onset_date", "signals", "score", "fwd_return_60d"}
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

            result = self._detect_single_onset(grp, close_col)
            results[code] = result

            if progress_callback and (i + 1) % 5 == 0:
                progress_callback(f"初動特定中... ({i + 1}/{len(all_stars)})")

        logger.info(
            f"初動特定完了: {sum(1 for r in results.values() if r['onset_date'])}/"
            f"{len(results)}件で初動日特定"
        )
        return results

    def _detect_single_onset(self, grp: pd.DataFrame, close_col: str) -> dict:
        """1銘柄の初動日を検出"""
        empty = {
            "onset_date": "", "signals": [], "score": 0,
            "fwd_return_60d": 0.0, "max_return": 0.0, "max_drawdown": 0.0,
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
                    max_price = float(np.max(window))
                    max_ret = max_price / price_at_onset - 1
                    peak = price_at_onset
                    max_dd = 0.0
                    for p in window:
                        if p > peak:
                            peak = p
                        dd = (p - peak) / peak
                        if dd < max_dd:
                            max_dd = dd

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
        max_returns = [
            od.get("max_return") or od.get("max_return_60d") or od.get("fwd_return_60d")
            for od in onset_dates.values()
            if od.get("onset_date") and (
                od.get("max_return") is not None or
                od.get("max_return_60d") is not None or
                od.get("fwd_return_60d") is not None
            )
        ]
        max_returns = [r for r in max_returns if r is not None]
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

        # 各コンボで条件を満たしたことのある銘柄セット
        combo_hit_sets = [set() for _ in combo_defs]

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

        # 精度計算
        results = []
        for ci, hits in enumerate(combo_hit_sets):
            n_hits = len(hits)
            n_star_hits = len(hits & star_codes)
            universe_prec = n_star_hits / n_hits if n_hits > 0 else 0
            results.append({
                "universe_n_total": n_total,
                "universe_n_hits": n_hits,
                "universe_n_stars": n_star_hits,
                "universe_precision": round(universe_prec, 4),
                "universe_hit_rate": round(n_hits / n_total, 4) if n_total > 0 else 0,
            })

        logger.info(
            f"母集団精度計算完了: {n_total}銘柄スキャン, "
            f"Top1コンボ: {results[0] if results else 'N/A'}"
        )
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
        vol_ma20 = _ma(volume, 20)
        vol_ma60 = _ma(volume, 60) if n >= 60 else _ma(volume, max(n, 1))
        close_ma5 = _ma(close, 5)
        close_ma20 = _ma(close, 20)

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

        feat["ma5_ma20_gap"] = float(
            (close_ma5[-1] - close_ma20[-1]) / close_ma20[-1]
        ) if close_ma20[-1] > 0 else 0.0
        feat["price_vs_ma20_pct"] = float(
            close[-1] / close_ma20[-1] - 1
        ) if close_ma20[-1] > 0 else 0.0

        consec = 0
        for i in range(len(ret) - 1, -1, -1):
            if ret[i] > 0:
                consec += 1
            else:
                break
        feat["consecutive_up_days"] = min(consec, 20)

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
                    try:
                        corr, _ = stats.spearmanr(close_20[valid_mask], mr_vals[valid_mask])
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

        # NaN/Inf安全化
        for k in feat:
            v = feat[k]
            if not np.isfinite(v):
                feat[k] = 0.0

        return feat
