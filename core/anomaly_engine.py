"""異常スキャンエンジン — 特徴量計算 + ルール評価 + ユニバーススキャン"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# UI公開特徴量（13個）
UI_FEATURE_KEYS = [
    "daily_return",
    "vol_ratio_5d_20d",
    "trading_value_vs_20d_ma",
    "new_high_52w",
    "new_low_52w",
    "sector_rel_ret_10d",
    "ret_5d",
    "range_position_20d",
    "bb_width_pctile_60d",
    "atr_ratio_5d_20d",
    "turnover_change_5d_20d",
    "up_days_ratio_10d",
    "vol_surge_count_10d",
]

UI_FEATURE_LABELS_JP = {
    "daily_return": "当日騰落率",
    "vol_ratio_5d_20d": "出来高比(5/20日)",
    "trading_value_vs_20d_ma": "売買代金20日比",
    "new_high_52w": "52週新高値",
    "new_low_52w": "52週新安値",
    "sector_rel_ret_10d": "業種相対強度(10日)",
    "ret_5d": "リターン(5日)",
    "range_position_20d": "20日レンジ位置",
    "bb_width_pctile_60d": "ボラ圧縮度(60日)",
    "atr_ratio_5d_20d": "ATR比(5/20日)",
    "turnover_change_5d_20d": "売買回転変化(5/20日)",
    "up_days_ratio_10d": "上昇日比率(10日)",
    "vol_surge_count_10d": "出来高急増回数(10日)",
}

UI_FEATURE_CATEGORIES = {
    "daily_return": "価格",
    "vol_ratio_5d_20d": "出来高",
    "trading_value_vs_20d_ma": "流動性",
    "new_high_52w": "価格",
    "new_low_52w": "価格",
    "sector_rel_ret_10d": "クロスセクショナル",
    "ret_5d": "価格",
    "range_position_20d": "価格",
    "bb_width_pctile_60d": "ボラティリティ",
    "atr_ratio_5d_20d": "ボラティリティ",
    "turnover_change_5d_20d": "流動性",
    "up_days_ratio_10d": "価格",
    "vol_surge_count_10d": "出来高",
}

OPERATOR_LABELS_JP = {
    "gt": "より大きい (>)",
    "lt": "より小さい (<)",
    "gte": "以上 (>=)",
    "lte": "以下 (<=)",
    "between": "範囲内",
}


def _safe_ma(arr: np.ndarray, w: int) -> np.ndarray:
    """安全なrolling mean"""
    n = len(arr)
    if n < w:
        return np.full(n, np.nanmean(arr) if n > 0 else 0.0)
    cs = np.cumsum(arr)
    cs = np.insert(cs, 0, 0.0)
    out = np.full(n, np.nan)
    out[w - 1:] = (cs[w:] - cs[:-w]) / w
    for i in range(w - 1):
        out[i] = np.mean(arr[:i + 1])
    return out


def compute_anomaly_features(
    prices_df: pd.DataFrame,
    topix_ret: float | None = None,
    sector_ret_10d: float | None = None,
    hist_52w_high: float | None = None,
    hist_52w_low: float | None = None,
) -> dict:
    """1銘柄のUI公開特徴量を計算。

    Parameters
    ----------
    prices_df : 直近60営業日+の価格データ（1銘柄、adj_close, volume, high, low列必須）
    topix_ret : 当日のTOPIXリターン（セクター相対の参考）
    sector_ret_10d : 過去10日のセクター平均リターン
    hist_52w_high : 過去52週の最高値
    hist_52w_low : 過去52週の最安値

    Returns
    -------
    dict : feature_key -> value
    """
    df = prices_df.sort_values("date").reset_index(drop=True) if "date" in prices_df.columns else prices_df.reset_index(drop=True)
    close_col = "adj_close" if "adj_close" in df.columns else "close"
    close = df[close_col].astype(float).values
    volume = df["volume"].astype(float).values
    high = df["high"].astype(float).values if "high" in df.columns else close.copy()
    low = df["low"].astype(float).values if "low" in df.columns else close.copy()

    n = len(close)
    if n < 20:
        return {}

    ret = np.diff(close) / np.where(close[:-1] != 0, close[:-1], 1.0)
    ret = np.where(np.isfinite(ret), ret, 0.0)

    feat = {}

    # daily_return: 当日騰落率
    feat["daily_return"] = float(ret[-1]) if len(ret) > 0 else 0.0

    # vol_ratio_5d_20d
    vol_ma5 = _safe_ma(volume, 5)
    vol_ma20 = _safe_ma(volume, 20)
    feat["vol_ratio_5d_20d"] = float(vol_ma5[-1] / vol_ma20[-1]) if vol_ma20[-1] > 0 else 1.0

    # trading_value_vs_20d_ma
    tv = close * volume
    tv_ma20 = _safe_ma(tv, 20)
    feat["trading_value_vs_20d_ma"] = float(tv[-1] / tv_ma20[-1]) if tv_ma20[-1] > 0 else 1.0

    # new_high_52w / new_low_52w
    if hist_52w_high is not None:
        feat["new_high_52w"] = 1.0 if high[-1] >= hist_52w_high else 0.0
    else:
        feat["new_high_52w"] = 0.0
    if hist_52w_low is not None:
        feat["new_low_52w"] = 1.0 if low[-1] <= hist_52w_low else 0.0
    else:
        feat["new_low_52w"] = 0.0

    # sector_rel_ret_10d
    stock_ret_10d = float(close[-1] / close[-min(11, n)] - 1) if close[-min(11, n)] > 0 else 0.0
    feat["sector_rel_ret_10d"] = (stock_ret_10d - sector_ret_10d) if sector_ret_10d is not None else stock_ret_10d

    # ret_5d
    feat["ret_5d"] = float(close[-1] / close[-6] - 1) if n >= 6 and close[-6] > 0 else 0.0

    # range_position_20d
    w20p = min(20, n)
    low_min = np.min(low[-w20p:])
    high_max = np.max(high[-w20p:])
    rng = high_max - low_min
    feat["range_position_20d"] = float((close[-1] - low_min) / rng) if rng > 0 else 0.5

    # bb_width_pctile_60d
    if n >= 20:
        bb_std = pd.Series(close).rolling(20, min_periods=10).std().values
        bb_ma = _safe_ma(close, 20)
        with np.errstate(divide="ignore", invalid="ignore"):
            bb_width = np.where(bb_ma > 0, 2 * bb_std / bb_ma, 0.0)
        bb_width = np.where(np.isfinite(bb_width), bb_width, 0.0)
        valid_bw = bb_width[~np.isnan(bb_width)]
        if len(valid_bw) >= 10:
            current_bw = valid_bw[-1]
            window = valid_bw[-60:] if len(valid_bw) >= 60 else valid_bw
            feat["bb_width_pctile_60d"] = float(
                np.searchsorted(np.sort(window), current_bw) / len(window)
            ) * 100  # パーセンタイル（0-100）
        else:
            feat["bb_width_pctile_60d"] = 50.0
    else:
        feat["bb_width_pctile_60d"] = 50.0

    # atr_ratio_5d_20d
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

    # turnover_change_5d_20d
    tv_ma5 = _safe_ma(tv, 5)
    feat["turnover_change_5d_20d"] = float(tv_ma5[-1] / tv_ma20[-1]) if tv_ma20[-1] > 0 else 1.0

    # up_days_ratio_10d
    w10r = min(10, len(ret))
    feat["up_days_ratio_10d"] = float((ret[-w10r:] > 0).mean()) if w10r > 0 else 0.5

    # vol_surge_count_10d
    w10 = min(10, n)
    feat["vol_surge_count_10d"] = int(np.sum(volume[-w10:] > vol_ma20[-w10:] * 2.0))

    # NaN/Inf安全化
    for k in feat:
        v = feat[k]
        if isinstance(v, float) and not np.isfinite(v):
            feat[k] = 0.0

    return feat


def evaluate_rule(
    rule_config: dict,
    feature_values: dict,
) -> tuple[bool, list[str]]:
    """ルール合致判定。

    Parameters
    ----------
    rule_config : {"conditions": [...], "logic": "AND"/"OR"}
    feature_values : {feature_key: value}

    Returns
    -------
    (matched: bool, reasons: list[str])
    """
    conditions = rule_config.get("conditions", [])
    logic = rule_config.get("logic", "AND")
    reasons = []
    match_flags = []

    for cond in conditions:
        fkey = cond.get("feature_key", "")
        op = cond.get("operator", "gt")
        threshold = cond.get("value")
        threshold_upper = cond.get("value_upper")

        if fkey not in feature_values or threshold is None:
            match_flags.append(False)
            continue

        actual = feature_values[fkey]
        label = UI_FEATURE_LABELS_JP.get(fkey, fkey)
        matched = False

        if op == "gt":
            matched = actual > threshold
            reason = f"{label} = {actual:.4f} > {threshold}"
        elif op == "lt":
            matched = actual < threshold
            reason = f"{label} = {actual:.4f} < {threshold}"
        elif op == "gte":
            matched = actual >= threshold
            reason = f"{label} = {actual:.4f} >= {threshold}"
        elif op == "lte":
            matched = actual <= threshold
            reason = f"{label} = {actual:.4f} <= {threshold}"
        elif op == "between":
            upper = threshold_upper if threshold_upper is not None else threshold
            matched = threshold <= actual <= upper
            reason = f"{label} = {actual:.4f} ({threshold}〜{upper})"
        else:
            reason = f"{label}: 不明な演算子 {op}"

        match_flags.append(matched)
        if matched:
            reasons.append(reason)

    if not match_flags:
        return False, []

    if logic == "AND":
        return all(match_flags), reasons
    else:  # OR
        return any(match_flags), reasons


def scan_universe(
    rule_config: dict,
    universe_df: pd.DataFrame,
    all_prices: pd.DataFrame,
    scan_date: str,
    topix_ret: float | None = None,
    sector_returns_10d: dict | None = None,
    hist_52w: dict | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """全銘柄スキャン。

    Parameters
    ----------
    rule_config : ルール設定
    universe_df : listed_stocks相当のDataFrame
    all_prices : 過去60日+のデータ（全銘柄）
    scan_date : スキャン日付 (YYYY-MM-DD)
    topix_ret : 当日TOPIX騰落率
    sector_returns_10d : {sector_name: 10日リターン}
    hist_52w : {code: {"high": float, "low": float}}
    progress_callback : (processed, total) を受け取るコールバック

    Returns
    -------
    pd.DataFrame with columns: code, name, sector, reasons, features, matched
    """
    if sector_returns_10d is None:
        sector_returns_10d = {}
    if hist_52w is None:
        hist_52w = {}

    codes = universe_df["code"].unique().tolist()
    stock_info = universe_df.set_index("code")[["name", "sector_17_name"]].to_dict("index")

    results = []
    total = len(codes)

    for i, code in enumerate(codes):
        code_prices = all_prices[all_prices["code"] == code].copy()
        if len(code_prices) < 20:
            if progress_callback and i % 200 == 0:
                progress_callback(i, total)
            continue

        info = stock_info.get(code, {})
        sector = info.get("sector_17_name", "")
        sect_ret = sector_returns_10d.get(sector)

        h52 = hist_52w.get(code, {})
        features = compute_anomaly_features(
            code_prices,
            topix_ret=topix_ret,
            sector_ret_10d=sect_ret,
            hist_52w_high=h52.get("high"),
            hist_52w_low=h52.get("low"),
        )

        if not features:
            if progress_callback and i % 200 == 0:
                progress_callback(i, total)
            continue

        matched, reasons = evaluate_rule(rule_config, features)

        if matched:
            results.append({
                "code": code,
                "name": info.get("name", ""),
                "sector": sector,
                "reasons": reasons,
                "features": {k: features.get(k) for k in UI_FEATURE_KEYS if k in features},
                "matched": True,
            })

        if progress_callback and i % 200 == 0:
            progress_callback(i, total)

    if progress_callback:
        progress_callback(total, total)

    if not results:
        return pd.DataFrame(columns=["code", "name", "sector", "reasons", "features", "matched"])

    return pd.DataFrame(results)
