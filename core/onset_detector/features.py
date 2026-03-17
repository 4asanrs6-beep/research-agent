"""時間的特徴量エンジン (~200特徴量)

Layer 0: 基本特徴量 (26個) — 既存_compute_wide_features()ベース
Layer 1: 変化速度 (26 × 3窓 = 78個)
Layer 2: 加速度 (26 × 2 = 52個)
Layer 3: 自己z-score (26個)
Layer 4: クロスセクショナル順位 (8個)
Layer 5: 合成特徴量 (8個)
Layer 6: 類型特徴量 (3-5個)
"""

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from .config import OnsetDetectorConfig

logger = logging.getLogger(__name__)

# 26個のベース特徴量キー
BASE_FEATURE_KEYS = [
    "vol_ratio_5d_20d", "vol_ratio_5d_60d", "vol_surge_count_10d",
    "up_volume_ratio_10d", "quiet_accum_rate_20d", "vol_acceleration", "vpin_5d",
    "ret_5d", "ret_20d", "up_days_ratio_10d", "max_gap_up_5d",
    "higher_lows_slope_10d", "range_position_20d",
    "atr_ratio_5d_20d", "bb_width_pctile_60d", "intraday_range_ratio_5d",
    "realized_vol_5d_vs_20d",
    "obv_slope_10d", "obv_divergence", "ma5_ma20_gap", "price_vs_ma20_pct",
    "consecutive_up_days",
    "sector_rel_ret_10d", "topix_beta_20d", "residual_vol_ratio",
    "vol_vs_market_vol",
]

VELOCITY_WINDOWS = [5, 10, 20]


# ==================================================================
# Top-level worker functions for ProcessPoolExecutor (must be picklable)
# ==================================================================

def _worker_compute_universe_batch(
    args: tuple,
) -> list[tuple[str, list[dict]]]:
    """ワーカー: 銘柄バッチのbase_features計算（ユニバース用）"""
    codes, prices_dict, close_col, topix_ret_index, topix_ret_values, sample_offsets, history_days = args
    topix_ret_series = pd.Series(topix_ret_values, index=pd.DatetimeIndex(topix_ret_index))

    results = []
    for code in codes:
        grp_data = prices_dict.get(code)
        if grp_data is None:
            continue
        grp = pd.DataFrame(grp_data)
        if len(grp) < 40:
            continue

        n = len(grp)
        base_history = []
        for offset in sample_offsets:
            t = n - 1 - offset
            if t < 20:
                continue
            df_window = grp.iloc[max(0, t - history_days):t + 1].reset_index(drop=True)
            bf = TemporalFeatureEngine.compute_base_features(
                df_window, close_col,
                topix_ret_series=topix_ret_series,
            )
            if bf:
                base_history.append(bf)

        base_history.reverse()
        results.append((str(code), base_history))

    return results


def _worker_compute_batch_features(
    args: tuple,
) -> list[tuple[str, dict[int, dict]]]:
    """ワーカー: 銘柄バッチのbase_features計算（学習用）"""
    codes_data, close_col, topix_ret_index, topix_ret_values, history_days = args
    topix_ret_series = pd.Series(topix_ret_values, index=pd.DatetimeIndex(topix_ret_index))

    results = []
    for code, grp_data in codes_data:
        grp = pd.DataFrame(grp_data)
        if len(grp) < 20:
            results.append((code, {}))
            continue

        all_dates_features = {}
        for t in range(20, len(grp)):
            window_start = max(0, t - history_days)
            df_window = grp.iloc[window_start:t + 1].reset_index(drop=True)
            bf = TemporalFeatureEngine.compute_base_features(
                df_window, close_col,
                topix_ret_series=topix_ret_series,
            )
            if bf:
                all_dates_features[t] = bf

        results.append((code, all_dates_features))

    return results


def _safe_ma(arr: np.ndarray, w: int) -> np.ndarray:
    """安全なrolling mean"""
    n = len(arr)
    if n < w:
        return np.full(n, np.nanmean(arr))
    cs = np.cumsum(arr)
    cs = np.insert(cs, 0, 0.0)
    out = np.full(n, np.nan)
    out[w - 1:] = (cs[w:] - cs[:-w]) / w
    for i in range(w - 1):
        out[i] = np.mean(arr[:i + 1])
    return out


class TemporalFeatureEngine:
    """時間的特徴量エンジン"""

    def __init__(self, config: OnsetDetectorConfig):
        self.config = config
        self._feature_names: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    # ==================================================================
    # Layer 0: 基本26特徴量（既存ロジック移植）
    # ==================================================================
    @staticmethod
    def compute_base_features(
        df: pd.DataFrame,
        close_col: str = "adj_close",
        sector_ret_10d: float | None = None,
        market_vol_ratio: float | None = None,
        topix_ret_series: pd.Series | None = None,
    ) -> dict | None:
        """1銘柄の指定時点までのデータから26個の基本特徴量を計算する"""
        if len(df) < 20:
            return None

        close = df[close_col].astype(float).values
        volume = df["volume"].astype(float).values
        high = df["high"].astype(float).values if "high" in df.columns else close.copy()
        low = df["low"].astype(float).values if "low" in df.columns else close.copy()
        open_ = df["open"].astype(float).values if "open" in df.columns else close.copy()

        n = len(close)
        ret = np.diff(close) / np.where(close[:-1] != 0, close[:-1], 1.0)
        ret = np.where(np.isfinite(ret), ret, 0.0)

        vol_ma5 = _safe_ma(volume, 5)
        vol_ma20 = _safe_ma(volume, 20)
        vol_ma60 = _safe_ma(volume, 60) if n >= 60 else _safe_ma(volume, max(n, 1))
        close_ma5 = _safe_ma(close, 5)
        close_ma20 = _safe_ma(close, 20)

        feat = {}

        # A: 出来高ダイナミクス (7)
        feat["vol_ratio_5d_20d"] = float(vol_ma5[-1] / vol_ma20[-1]) if vol_ma20[-1] > 0 else 1.0
        feat["vol_ratio_5d_60d"] = float(vol_ma5[-1] / vol_ma60[-1]) if vol_ma60[-1] > 0 else 1.0

        w10 = min(10, n)
        feat["vol_surge_count_10d"] = int(np.sum(volume[-w10:] > vol_ma20[-w10:] * 2.0))

        # up_volume_ratio_10d
        if len(ret) >= 10:
            vol_10 = volume[-10:]
            ret_10 = ret[-min(10, len(ret)):]
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

        # quiet_accum_rate_20d
        w20 = min(20, len(ret))
        if w20 > 0:
            r20 = ret[-w20:]
            v20 = volume[-w20:]
            vm20 = vol_ma20[-w20:]
            quiet_mask = (np.abs(r20) < 0.003) & (v20[:len(r20)] > vm20[:len(r20)] * 1.3)
            feat["quiet_accum_rate_20d"] = float(quiet_mask.sum() / w20)
        else:
            feat["quiet_accum_rate_20d"] = 0.0

        # vol_acceleration
        if n >= 10:
            fh = vol_ma5[-10:-5].mean() if len(vol_ma5) >= 10 else vol_ma5.mean()
            sh = vol_ma5[-5:].mean()
            feat["vol_acceleration"] = float(sh / fh) if fh > 0 else 1.0
        else:
            feat["vol_acceleration"] = 1.0

        # vpin_5d
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

        # B: 価格/リターン (6)
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

        # C: ボラティリティ (4)
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
            bb_ma = _safe_ma(close, 20)
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
            mean_20 = np.mean(intra[-20:])
            feat["intraday_range_ratio_5d"] = float(np.mean(intra[-5:]) / mean_20) if mean_20 > 0 else 1.0
        else:
            feat["intraday_range_ratio_5d"] = 1.0

        if len(ret) >= 20:
            rv5 = np.std(ret[-5:])
            rv20 = np.std(ret[-20:])
            feat["realized_vol_5d_vs_20d"] = float(rv5 / rv20) if rv20 > 0 else 1.0
        else:
            feat["realized_vol_5d_vs_20d"] = 1.0

        # D: トレンド/OBV (5)
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
                c20 = close[-20:]
                o20 = obv[-20:]
                if np.std(c20) > 1e-10 and np.std(o20) > 1e-10:
                    corr, _ = stats.spearmanr(c20, o20)
                    feat["obv_divergence"] = float(corr) if np.isfinite(corr) else 0.0
                else:
                    feat["obv_divergence"] = 0.0
            except Exception:
                feat["obv_divergence"] = 0.0
        else:
            feat["obv_divergence"] = 0.0

        feat["ma5_ma20_gap"] = float((close_ma5[-1] - close_ma20[-1]) / close_ma20[-1]) if close_ma20[-1] > 0 else 0.0
        feat["price_vs_ma20_pct"] = float(close[-1] / close_ma20[-1] - 1) if close_ma20[-1] > 0 else 0.0

        consec = 0
        for i in range(len(ret) - 1, -1, -1):
            if ret[i] > 0:
                consec += 1
            else:
                break
        feat["consecutive_up_days"] = min(consec, 20)

        # E: クロスセクショナル (4)
        stock_ret_10d = float(close[-1] / close[-min(11, n)] - 1) if close[-min(11, n)] > 0 else 0.0
        feat["sector_rel_ret_10d"] = (stock_ret_10d - sector_ret_10d) if sector_ret_10d is not None else stock_ret_10d

        if topix_ret_series is not None and len(ret) >= 20 and "date" in df.columns:
            try:
                dates_ts = pd.to_datetime(df["date"].values)
                stock_s = pd.Series(ret, index=dates_ts[1:])
                topix_idx = pd.to_datetime(topix_ret_series.index)
                topix_al = pd.Series(topix_ret_series.values, index=topix_idx)
                common = stock_s.index.intersection(topix_al.index)
                if len(common) >= 15:
                    sr = stock_s.loc[common].values[-20:]
                    tr = topix_al.loc[common].values[-20:]
                    if len(sr) >= 10 and len(tr) >= 10:
                        cov = np.cov(sr, tr)
                        feat["topix_beta_20d"] = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 1.0
                    else:
                        feat["topix_beta_20d"] = 1.0
                else:
                    feat["topix_beta_20d"] = 1.0
            except Exception:
                feat["topix_beta_20d"] = 1.0
        else:
            feat["topix_beta_20d"] = 1.0

        if len(ret) >= 60:
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

        # NaN/Inf安全化
        for k in feat:
            v = feat[k]
            if not np.isfinite(v):
                feat[k] = 0.0

        return feat

    # ==================================================================
    # Layer 1-6: 時間的拡張特徴量
    # ==================================================================
    def compute_all_layers(
        self,
        base_history: list[dict],
        sample_meta: dict,
    ) -> dict:
        """base_historyは直近N日分のbase_features辞書リスト（古い順）。
        全Layerの特徴量を一括計算して返す。

        Parameters
        ----------
        base_history : list[dict]
            直近120日分の日次base_features (len <= 120)
        sample_meta : dict
            サンプルメタ情報 (star_type, is_overheated等)

        Returns
        -------
        dict  全特徴量
        """
        if not base_history:
            return {}

        current = base_history[-1]
        feat = {}

        # Layer 0: 基本特徴量
        for k in BASE_FEATURE_KEYS:
            feat[f"L0_{k}"] = current.get(k, 0.0)

        # Layer 1: 変化速度 (Δ)
        for w in VELOCITY_WINDOWS:
            if len(base_history) > w:
                past = base_history[-w - 1]
                for k in BASE_FEATURE_KEYS:
                    delta = current.get(k, 0.0) - past.get(k, 0.0)
                    feat[f"L1_delta{w}d_{k}"] = delta
            else:
                for k in BASE_FEATURE_KEYS:
                    feat[f"L1_delta{w}d_{k}"] = 0.0

        # Layer 2: 加速度 (ΔΔ) + 符号反転フラグ
        for k in BASE_FEATURE_KEYS:
            d5 = feat.get(f"L1_delta5d_{k}", 0.0)
            d20 = feat.get(f"L1_delta20d_{k}", 0.0)
            feat[f"L2_accel_{k}"] = d5 - d20  # 短期変化 - 長期変化
            feat[f"L2_sign_flip_{k}"] = float(
                (d5 > 0 and d20 < 0) or (d5 < 0 and d20 > 0)
            )

        # Layer 3: 自己z-score (120日ヒストリカル)
        if len(base_history) >= 20:
            for k in BASE_FEATURE_KEYS:
                vals = [h.get(k, 0.0) for h in base_history]
                arr = np.array(vals, dtype=float)
                arr = np.where(np.isfinite(arr), arr, 0.0)
                mean = np.mean(arr)
                std = np.std(arr)
                if std > 1e-10:
                    feat[f"L3_zscore_{k}"] = float((current.get(k, 0.0) - mean) / std)
                else:
                    feat[f"L3_zscore_{k}"] = 0.0
        else:
            for k in BASE_FEATURE_KEYS:
                feat[f"L3_zscore_{k}"] = 0.0

        # Layer 4: クロスセクショナル順位（スキャン時に注入、学習時は0）
        for rank_key in [
            "L4_rank_vol_ratio", "L4_rank_ret_5d", "L4_rank_ret_20d",
            "L4_rank_vpin", "L4_rank_obv_slope", "L4_rank_bb_width",
            "L4_rank_atr_ratio", "L4_rank_quiet_accum",
        ]:
            feat[rank_key] = sample_meta.get(rank_key, 0.0)

        # Layer 5: 合成特徴量
        feat["L5_vol_price_momentum"] = (
            feat.get("L0_vol_ratio_5d_20d", 1.0) * feat.get("L0_ret_5d", 0.0)
        )
        feat["L5_stealth_score"] = (
            feat.get("L0_quiet_accum_rate_20d", 0.0)
            * feat.get("L0_vol_acceleration", 1.0)
        )
        feat["L5_breakout_readiness"] = (
            (1 - feat.get("L0_bb_width_pctile_60d", 0.5))
            * feat.get("L0_range_position_20d", 0.5)
        )
        feat["L5_trend_strength"] = (
            feat.get("L0_obv_slope_10d", 0.0)
            + feat.get("L0_higher_lows_slope_10d", 0.0)
        )
        feat["L5_vol_regime_shift"] = (
            feat.get("L0_vol_ratio_5d_60d", 1.0)
            - feat.get("L0_realized_vol_5d_vs_20d", 1.0)
        )
        feat["L5_accumulation_intensity"] = (
            feat.get("L0_up_volume_ratio_10d", 0.5)
            * feat.get("L0_vol_surge_count_10d", 0)
        )
        feat["L5_price_vol_divergence"] = (
            feat.get("L0_ret_20d", 0.0) - feat.get("L0_obv_divergence", 0.0)
        )
        feat["L5_market_relative_activity"] = feat.get("L0_vol_vs_market_vol", 1.0)

        # Layer 6: 類型特徴量
        star_type = sample_meta.get("star_type", -1)
        n_types = self.config.n_star_types
        for t in range(n_types):
            feat[f"L6_type_{t}"] = float(star_type == t)
        feat["L6_is_overheated"] = float(sample_meta.get("is_overheated", False))

        # NaN/Inf安全化
        for k in feat:
            if not np.isfinite(feat[k]):
                feat[k] = 0.0

        return feat

    # ==================================================================
    # バッチ計算（学習データ用）
    # ==================================================================
    def compute_features_batch(
        self,
        dataset: dict,
        all_prices: pd.DataFrame,
        topix: pd.DataFrame,
        listed_stocks: pd.DataFrame,
        progress_callback=None,
    ) -> tuple[np.ndarray, list[str]]:
        """学習データセット全体の特徴量を計算

        Parameters
        ----------
        progress_callback : callable | None
            (processed, total, elapsed_sec) を受け取るコールバック

        Returns
        -------
        X : np.ndarray  (n_samples, n_features)
        feature_names : list[str]
        """
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"
        samples = dataset["samples"]
        history_days = self.config.feature_history_days
        max_workers = self.config.scan_max_workers

        # TOPIX日次リターン
        topix_sorted = topix.sort_values("date").reset_index(drop=True)
        topix_close_col = "close" if "close" in topix_sorted.columns else topix_sorted.columns[-1]
        topix_close = topix_sorted[topix_close_col].astype(float)
        topix_ret_series = topix_close.pct_change(fill_method=None).dropna()
        if "date" in topix_sorted.columns:
            topix_ret_series.index = pd.to_datetime(topix_sorted["date"].iloc[1:].values)

        # 銘柄ごとにグループ化して効率化
        code_to_samples = {}
        for i, s in enumerate(samples):
            code = s["code"]
            if code not in code_to_samples:
                code_to_samples[code] = []
            code_to_samples[code].append((i, s))

        all_features = [None] * len(samples)
        feature_names = None

        n_codes = len(code_to_samples)
        codes_list = list(code_to_samples.keys())

        # 銘柄ごとの価格データ辞書を構築
        code_grps = {}
        for code in codes_list:
            grp = all_prices[all_prices["code"] == code].sort_values("date").reset_index(drop=True)
            code_grps[code] = grp

        # 並列base_features計算
        batch_size = max(1, n_codes // max_workers)
        code_batches = [
            [(code, code_grps[code].to_dict("list")) for code in codes_list[i:i + batch_size]]
            for i in range(0, n_codes, batch_size)
        ]

        topix_ret_index = topix_ret_series.index.tolist()
        topix_ret_values = topix_ret_series.values.tolist()

        t_start = time.time()
        processed_codes = 0
        all_dates_features_map = {}

        if max_workers > 1 and n_codes > 10:
            worker_args = [
                (batch, close_col, topix_ret_index, topix_ret_values, history_days)
                for batch in code_batches
            ]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_worker_compute_batch_features, arg): i for i, arg in enumerate(worker_args)}
                for future in as_completed(futures):
                    results = future.result()
                    for code, dates_features in results:
                        all_dates_features_map[code] = dates_features
                    processed_codes += len(results)
                    if progress_callback:
                        progress_callback(processed_codes, n_codes, time.time() - t_start)
        else:
            for ci, code in enumerate(codes_list):
                grp = code_grps[code]
                if len(grp) < 20:
                    all_dates_features_map[code] = {}
                    processed_codes += 1
                    continue

                all_dates_features = {}
                for t in range(20, len(grp)):
                    window_start = max(0, t - history_days)
                    df_window = grp.iloc[window_start:t + 1].reset_index(drop=True)
                    bf = self.compute_base_features(
                        df_window, close_col,
                        topix_ret_series=topix_ret_series,
                    )
                    if bf:
                        all_dates_features[t] = bf
                all_dates_features_map[code] = all_dates_features
                processed_codes += 1
                if progress_callback and ci % 20 == 0:
                    progress_callback(processed_codes, n_codes, time.time() - t_start)

        # 各サンプルの特徴量計算
        for code, code_samples in code_to_samples.items():
            dates_features = all_dates_features_map.get(code, {})
            if not dates_features:
                for idx, s in code_samples:
                    all_features[idx] = {}
                continue

            for idx, s in code_samples:
                t = s["date_idx"]
                history_start = max(20, t - history_days)
                base_history = []
                for h_t in range(history_start, t + 1):
                    if h_t in dates_features:
                        base_history.append(dates_features[h_t])

                if not base_history:
                    all_features[idx] = {}
                    continue

                feat = self.compute_all_layers(base_history, s)
                all_features[idx] = feat

                if feature_names is None and feat:
                    feature_names = sorted(feat.keys())

        if feature_names is None:
            feature_names = []

        self._feature_names = feature_names

        # numpy配列に変換
        X = np.zeros((len(samples), len(feature_names)), dtype=np.float32)
        for i, feat in enumerate(all_features):
            if feat:
                for j, fn in enumerate(feature_names):
                    X[i, j] = feat.get(fn, 0.0)

        # NaN/Inf安全化
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"特徴量計算完了: {X.shape[0]}サンプル × {X.shape[1]}特徴量")
        return X, feature_names

    # ==================================================================
    # ユニバース全体計算（推論時）
    # ==================================================================
    def compute_features_universe(
        self,
        all_prices: pd.DataFrame,
        topix: pd.DataFrame,
        listed_stocks: pd.DataFrame,
        scan_codes: list[str] | None = None,
        progress_callback=None,
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """全銘柄の最新時点の特徴量を計算

        Parameters
        ----------
        scan_codes : list[str] | None
            スキャン対象銘柄コード。Noneなら全銘柄。
        progress_callback : callable | None
            (processed, total, elapsed_sec) を受け取るコールバック

        Returns
        -------
        code_features : dict[str, np.ndarray]  code -> feature vector
        feature_names : list[str]
        """
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"
        history_days = self.config.feature_history_days
        max_workers = self.config.scan_max_workers

        topix_sorted = topix.sort_values("date").reset_index(drop=True)
        topix_close_col = "close" if "close" in topix_sorted.columns else topix_sorted.columns[-1]
        topix_close = topix_sorted[topix_close_col].astype(float)
        topix_ret_series = topix_close.pct_change(fill_method=None).dropna()
        if "date" in topix_sorted.columns:
            topix_ret_series.index = pd.to_datetime(topix_sorted["date"].iloc[1:].values)

        # 市場全体の出来高比率（クロスセクショナル用）
        market_vol_ratio = None

        code_features = {}

        # スキャン対象銘柄の絞り込み
        if scan_codes is not None:
            scan_codes_set = set(str(c) for c in scan_codes)
            codes = [c for c in all_prices["code"].unique() if str(c) in scan_codes_set]
        else:
            codes = list(all_prices["code"].unique())
        n_codes = len(codes)
        logger.info(f"ユニバース特徴量計算: {n_codes}銘柄")

        # スキャン時はLayer 1-3に必要な最小限の日数のみ計算
        sample_offsets = sorted(set(
            [0]  # 現在
            + [5, 10, 20]  # Layer 1 velocity
            + list(range(0, min(history_days, 120), 5))  # Layer 3 z-score用（5日間隔）
        ))

        # 並列化用データ準備
        topix_ret_index = topix_ret_series.index.tolist()
        topix_ret_values = topix_ret_series.values.tolist()

        t_start = time.time()
        processed_codes = 0
        code_base_features = {}

        if max_workers > 1 and n_codes > 20:
            # 銘柄ごとの価格データをバッチに分割
            batch_size = max(1, n_codes // (max_workers * 4))
            code_batches = [codes[i:i + batch_size] for i in range(0, n_codes, batch_size)]

            # 各バッチの価格データを辞書化
            prices_dict_all = {}
            for code in codes:
                grp = all_prices[all_prices["code"] == code].sort_values("date").reset_index(drop=True)
                prices_dict_all[code] = grp.to_dict("list")

            worker_args = [
                (batch, {c: prices_dict_all[c] for c in batch}, close_col,
                 topix_ret_index, topix_ret_values, sample_offsets, history_days)
                for batch in code_batches
            ]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_worker_compute_universe_batch, arg): i for i, arg in enumerate(worker_args)}
                for future in as_completed(futures):
                    results = future.result()
                    for code_str, base_history in results:
                        code_base_features[code_str] = base_history
                    processed_codes += len(results)
                    if progress_callback:
                        progress_callback(processed_codes, n_codes, time.time() - t_start)
        else:
            for ci, code in enumerate(codes):
                grp = all_prices[all_prices["code"] == code].sort_values("date").reset_index(drop=True)
                if len(grp) < 40:
                    processed_codes += 1
                    continue

                n = len(grp)
                base_history = []
                for offset in sample_offsets:
                    t = n - 1 - offset
                    if t < 20:
                        continue
                    df_window = grp.iloc[max(0, t - history_days):t + 1].reset_index(drop=True)
                    bf = self.compute_base_features(
                        df_window, close_col,
                        topix_ret_series=topix_ret_series,
                        market_vol_ratio=market_vol_ratio,
                    )
                    if bf:
                        base_history.append(bf)

                base_history.reverse()
                code_base_features[str(code)] = base_history
                processed_codes += 1
                if progress_callback and ci % 100 == 0:
                    progress_callback(processed_codes, n_codes, time.time() - t_start)

        if progress_callback:
            progress_callback(n_codes, n_codes, time.time() - t_start)

        # クロスセクショナル順位計算
        rank_data = self._compute_cross_sectional_ranks(code_base_features)

        # 全Layer計算
        feature_names = self._feature_names if self._feature_names else None

        for code_str, base_history in code_base_features.items():
            if not base_history:
                continue

            meta = {
                "star_type": -1,
                "is_overheated": False,
            }
            if code_str in rank_data:
                meta.update(rank_data[code_str])

            feat = self.compute_all_layers(base_history, meta)
            if feat:
                if feature_names is None:
                    feature_names = sorted(feat.keys())

                vec = np.array([feat.get(fn, 0.0) for fn in feature_names], dtype=np.float32)
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                code_features[code_str] = vec

        if feature_names is None:
            feature_names = []
        self._feature_names = feature_names

        logger.info(f"ユニバース特徴量完了: {len(code_features)}銘柄 × {len(feature_names)}特徴量")
        return code_features, feature_names

    # ==================================================================
    # クロスセクショナル順位
    # ==================================================================
    @staticmethod
    def _compute_cross_sectional_ranks(
        code_base_features: dict[str, list[dict]],
    ) -> dict[str, dict]:
        """全銘柄の最新base_featuresからクロスセクショナル順位を計算"""
        rank_keys = {
            "L4_rank_vol_ratio": "vol_ratio_5d_20d",
            "L4_rank_ret_5d": "ret_5d",
            "L4_rank_ret_20d": "ret_20d",
            "L4_rank_vpin": "vpin_5d",
            "L4_rank_obv_slope": "obv_slope_10d",
            "L4_rank_bb_width": "bb_width_pctile_60d",
            "L4_rank_atr_ratio": "atr_ratio_5d_20d",
            "L4_rank_quiet_accum": "quiet_accum_rate_20d",
        }

        # 各特徴量の全銘柄値を収集
        code_vals = {}
        for code, history in code_base_features.items():
            if history:
                code_vals[code] = history[-1]

        result = {}
        if not code_vals:
            return result

        n = len(code_vals)
        codes = list(code_vals.keys())

        for rank_name, base_key in rank_keys.items():
            vals = np.array([code_vals[c].get(base_key, 0.0) for c in codes])
            ranks = np.argsort(np.argsort(vals)).astype(float) / max(n - 1, 1)
            for i, c in enumerate(codes):
                if c not in result:
                    result[c] = {}
                result[c][rank_name] = float(ranks[i])

        return result
