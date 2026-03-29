"""ポートフォリオ構築 — フィルター型 & ランキング型 (Phase 1)"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_filter_portfolio(
    stock_signals: pd.DataFrame,
    n_sectors_long: int = 5,
    n_sectors_short: int = 5,
) -> pd.DataFrame:
    """フィルター型: シグナル上位セクターの全銘柄をロング、下位をショート。

    Args:
        stock_signals: assign_sector_signals() の結果
            columns: [date, code, sector_signal, etf_code]
        n_sectors_long: ロング対象セクター数
        n_sectors_short: ショート対象セクター数

    Returns:
        DataFrame[date, code, weight]
        weight > 0: ロング, weight < 0: ショート
    """
    rows = []

    for dt, group in stock_signals.groupby("date"):
        # セクター別の平均シグナル（同一セクター内は同じ値のはず）
        sector_signals = group.groupby("etf_code")["sector_signal"].first()
        sector_signals = sector_signals.dropna().sort_values(ascending=False)

        if len(sector_signals) < n_sectors_long + n_sectors_short:
            continue

        long_sectors = set(sector_signals.head(n_sectors_long).index)
        short_sectors = set(sector_signals.tail(n_sectors_short).index)

        long_stocks = group[group["etf_code"].isin(long_sectors)]["code"].tolist()
        short_stocks = group[group["etf_code"].isin(short_sectors)]["code"].tolist()

        if not long_stocks and not short_stocks:
            continue

        # 等金額ウェイト
        for code in long_stocks:
            rows.append({
                "date": dt,
                "code": code,
                "weight": 1.0 / len(long_stocks) if long_stocks else 0,
            })
        for code in short_stocks:
            rows.append({
                "date": dt,
                "code": code,
                "weight": -1.0 / len(short_stocks) if short_stocks else 0,
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        n_dates = result["date"].nunique()
        avg_long = (result["weight"] > 0).sum() / max(1, n_dates)
        avg_short = (result["weight"] < 0).sum() / max(1, n_dates)
        logger.info(
            "フィルター型ポートフォリオ: %d日, 平均ロング%.0f/ショート%.0f銘柄",
            n_dates, avg_long, avg_short,
        )
    return result


def build_ranking_portfolio(
    stock_signals: pd.DataFrame,
    n_stocks_long: int = 30,
    n_stocks_short: int = 30,
    stock_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """ランキング型: セクターシグナル（＋銘柄スコア）で銘柄を順位づけ。

    Args:
        stock_signals: assign_sector_signals() の結果
        n_stocks_long: ロング銘柄数
        n_stocks_short: ショート銘柄数
        stock_scores: 銘柄別スコア (columns: [date, code, score])
            セクターシグナルと組み合わせてランキング。Noneならセクターシグナルのみ。

    Returns:
        DataFrame[date, code, weight]
    """
    rows = []

    for dt, group in stock_signals.groupby("date"):
        df_day = group[["code", "sector_signal"]].copy()

        # 銘柄別スコアがあれば結合
        if stock_scores is not None:
            day_scores = stock_scores[stock_scores["date"] == dt][["code", "score"]]
            if len(day_scores) > 0:
                df_day = df_day.merge(day_scores, on="code", how="left")
                # 総合スコア = セクターシグナル（正規化）+ 銘柄スコア（正規化）
                sig_std = df_day["sector_signal"].std()
                score_std = df_day["score"].std()
                if sig_std > 1e-10 and score_std > 1e-10:
                    df_day["total_score"] = (
                        df_day["sector_signal"] / sig_std
                        + df_day["score"] / score_std
                    )
                else:
                    df_day["total_score"] = df_day["sector_signal"]
            else:
                df_day["total_score"] = df_day["sector_signal"]
        else:
            df_day["total_score"] = df_day["sector_signal"]

        df_day = df_day.dropna(subset=["total_score"])
        if len(df_day) < n_stocks_long + n_stocks_short:
            continue

        df_day = df_day.sort_values("total_score", ascending=False)

        long_codes = df_day.head(n_stocks_long)["code"].tolist()
        short_codes = df_day.tail(n_stocks_short)["code"].tolist()

        for code in long_codes:
            rows.append({
                "date": dt,
                "code": code,
                "weight": 1.0 / n_stocks_long,
            })
        for code in short_codes:
            rows.append({
                "date": dt,
                "code": code,
                "weight": -1.0 / n_stocks_short,
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        logger.info(
            "ランキング型ポートフォリオ: %d日, L%d/S%d",
            result["date"].nunique(), n_stocks_long, n_stocks_short,
        )
    return result
