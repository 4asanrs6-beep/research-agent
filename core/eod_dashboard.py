"""EODマーケットダッシュボード — ランキング・騰落分布・セクター集計"""

import numpy as np
import pandas as pd

from core.universe_filter import SECTOR_17_LIST


def compute_eod_rankings(
    prices_today: pd.DataFrame,
    prices_prev: pd.DataFrame,
    listed_stocks: pd.DataFrame,
) -> pd.DataFrame:
    """騰落率・出来高・売買代金をまとめたランキング用DataFrameを返す。

    Returns
    -------
    pd.DataFrame with columns:
        code, name, sector_17_name, market_name, scale_category,
        adj_close, daily_return, volume, volume_prev, volume_change,
        trading_value, trading_value_prev, trading_value_change
    """
    today = prices_today.copy()
    prev = prices_prev.copy()

    # 株情報のマッピング
    stock_info = listed_stocks.set_index("code")[
        ["name", "sector_17_name", "market_name", "scale_category"]
    ].to_dict("index")

    # 前日終値を結合
    prev_close = prev.set_index("code")["adj_close"].to_dict()
    prev_volume = prev.set_index("code")["volume"].to_dict()

    today["prev_close"] = today["code"].map(prev_close)
    today["daily_return"] = np.where(
        today["prev_close"] > 0,
        (today["adj_close"] / today["prev_close"] - 1),
        np.nan,
    )

    # 売買代金
    today["trading_value"] = today["adj_close"] * today["volume"]
    today["prev_volume"] = today["code"].map(prev_volume)
    today["volume_change"] = np.where(
        today["prev_volume"] > 0,
        today["volume"] / today["prev_volume"],
        np.nan,
    )

    # 前日売買代金
    prev["trading_value_prev"] = prev["adj_close"] * prev["volume"]
    prev_tv = prev.set_index("code")["trading_value_prev"].to_dict()
    today["trading_value_prev"] = today["code"].map(prev_tv)
    today["trading_value_change"] = np.where(
        today["trading_value_prev"] > 0,
        today["trading_value"] / today["trading_value_prev"],
        np.nan,
    )

    # 株情報付与
    today["name"] = today["code"].map(lambda c: stock_info.get(c, {}).get("name", ""))
    today["sector_17_name"] = today["code"].map(lambda c: stock_info.get(c, {}).get("sector_17_name", ""))
    today["market_name"] = today["code"].map(lambda c: stock_info.get(c, {}).get("market_name", ""))
    today["scale_category"] = today["code"].map(lambda c: stock_info.get(c, {}).get("scale_category", ""))

    return today


def compute_advance_decline(
    prices_today: pd.DataFrame,
    prices_prev: pd.DataFrame,
) -> dict:
    """騰落銘柄数を返す。

    Returns
    -------
    dict: {advance, decline, unchanged, total, advance_ratio}
    """
    prev_close = prices_prev.set_index("code")["adj_close"]
    today_close = prices_today.set_index("code")["adj_close"]
    common = prev_close.index.intersection(today_close.index)
    if common.empty:
        return {"advance": 0, "decline": 0, "unchanged": 0, "total": 0, "advance_ratio": 0.5}

    ret = today_close[common] / prev_close[common] - 1
    advance = int((ret > 0.0001).sum())
    decline = int((ret < -0.0001).sum())
    unchanged = int(len(ret) - advance - decline)
    total = len(ret)
    return {
        "advance": advance,
        "decline": decline,
        "unchanged": unchanged,
        "total": total,
        "advance_ratio": advance / total if total > 0 else 0.5,
    }


def compute_sector_summary(
    ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    """業種別の平均騰落率・銘柄数を集計。

    Returns
    -------
    pd.DataFrame with index=sector, columns=[mean_return, median_return, advance, decline, count]
    """
    df = ranking_df.dropna(subset=["daily_return"])
    df = df[df["sector_17_name"].isin(SECTOR_17_LIST)]

    def _agg(g):
        return pd.Series({
            "mean_return": g["daily_return"].mean(),
            "median_return": g["daily_return"].median(),
            "advance": int((g["daily_return"] > 0.0001).sum()),
            "decline": int((g["daily_return"] < -0.0001).sum()),
            "count": len(g),
        })

    summary = df.groupby("sector_17_name").apply(_agg, include_groups=False)
    return summary.reindex(SECTOR_17_LIST).fillna(0)


def compute_market_segment_strength(
    ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    """プライム/スタンダード/グロース別の平均騰落率を返す。"""
    df = ranking_df.dropna(subset=["daily_return", "market_name"])
    segments = ["プライム", "スタンダード", "グロース"]
    df = df[df["market_name"].isin(segments)]

    def _agg(g):
        return pd.Series({
            "mean_return": g["daily_return"].mean(),
            "advance": int((g["daily_return"] > 0.0001).sum()),
            "decline": int((g["daily_return"] < -0.0001).sum()),
            "count": len(g),
        })

    return df.groupby("market_name").apply(_agg, include_groups=False).reindex(segments).fillna(0)


def detect_new_highs_lows(
    prices_today: pd.DataFrame,
    prices_hist: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """52週新高値/安値を検出。

    Parameters
    ----------
    prices_today : 当日の価格データ
    prices_hist : 過去252営業日程度の価格データ（全銘柄・全日付）

    Returns
    -------
    (new_highs_df, new_lows_df) : 各々code列を持つDataFrame
    """
    if prices_hist.empty or prices_today.empty:
        return pd.DataFrame(columns=["code"]), pd.DataFrame(columns=["code"])

    # 過去の高値・安値
    hist_high = prices_hist.groupby("code")["high"].max()
    hist_low = prices_hist.groupby("code")["low"].min()

    today = prices_today.set_index("code")
    common_high = hist_high.index.intersection(today.index)
    common_low = hist_low.index.intersection(today.index)

    new_highs = today.loc[common_high][today.loc[common_high, "high"] >= hist_high[common_high]]
    new_lows = today.loc[common_low][today.loc[common_low, "low"] <= hist_low[common_low]]

    return new_highs.reset_index()[["code"]], new_lows.reset_index()[["code"]]


def apply_compound_filter(
    df: pd.DataFrame,
    conditions: list[dict],
) -> pd.DataFrame:
    """プリセット複合条件を適用。

    Parameters
    ----------
    df : ranking_df
    conditions : list of dicts with keys: column, operator, value
        operator: "gt", "lt", "gte", "lte", "between", "top_n", "bottom_n"

    Returns
    -------
    Filtered DataFrame
    """
    result = df.copy()
    for cond in conditions:
        col = cond["column"]
        op = cond["operator"]
        val = cond["value"]

        if col not in result.columns:
            continue

        if op == "gt":
            result = result[result[col] > val]
        elif op == "lt":
            result = result[result[col] < val]
        elif op == "gte":
            result = result[result[col] >= val]
        elif op == "lte":
            result = result[result[col] <= val]
        elif op == "between":
            result = result[result[col].between(val, cond.get("value_upper", val))]
        elif op == "top_n":
            result = result.nlargest(int(val), col)
        elif op == "bottom_n":
            result = result.nsmallest(int(val), col)

    return result
