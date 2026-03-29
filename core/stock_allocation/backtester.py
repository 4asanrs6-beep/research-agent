"""個別株 L/S バックテスター (Phase 1)

ETF版と同じ方式 (寄引リターンベース) で個別株のL/Sリターンを計算する。
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StockBacktestResult:
    """個別株配賦バックテスト結果"""
    daily_returns: pd.Series | None = None  # 日次L/Sリターン
    metrics: dict = field(default_factory=dict)
    # 比較用: ETF版のメトリクス
    etf_metrics: dict = field(default_factory=dict)
    # ポジション情報
    daily_long_count: pd.Series | None = None
    daily_short_count: pd.Series | None = None
    daily_turnover: pd.Series | None = None
    # 配賦モード
    mode: str = ""
    # セクター別成績
    sector_returns: dict | None = None


def run_stock_backtest(
    portfolio: pd.DataFrame,
    stock_prices: pd.DataFrame,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.001,
    benchmark_returns: pd.Series | None = None,
) -> StockBacktestResult:
    """個別株L/Sポートフォリオのバックテストを実行する。

    ETF版と整合性を取るため、寄引リターンベースで計算する。
    各日: Σ weight_i × oc_return_i - round_trip_cost

    Args:
        portfolio: DataFrame[date, code, weight]
            weight > 0 ロング, weight < 0 ショート
        stock_prices: DataFrame[date, code, adj_open, adj_close]
            個別株の日次価格データ
        commission_rate: 片道手数料率
        slippage_rate: 片道スリッページ率
        benchmark_returns: TOPIX日次リターン (比較用)

    Returns:
        StockBacktestResult
    """
    if portfolio.empty:
        return StockBacktestResult(metrics={"error": "空のポートフォリオ"})

    # 寄引リターンを計算
    price_col_close = "adj_close" if "adj_close" in stock_prices.columns else "close"
    price_col_open = "adj_open" if "adj_open" in stock_prices.columns else "open"

    prices = stock_prices[["date", "code", price_col_open, price_col_close]].copy()
    prices["oc_return"] = prices[price_col_close] / prices[price_col_open] - 1

    # ラウンドトリップコスト（片道×2: エントリー+エグジット）
    round_trip_cost = 2 * (commission_rate + slippage_rate)

    all_dates = sorted(portfolio["date"].unique())
    daily_returns = []
    daily_long_counts = []
    daily_short_counts = []
    prev_codes = set()

    for dt in all_dates:
        day_portfolio = portfolio[portfolio["date"] == dt]
        day_prices = prices[prices["date"] == dt].set_index("code")

        # ポートフォリオ内の銘柄で価格がある銘柄のみ
        ret = 0.0
        n_long = 0
        n_short = 0
        n_traded = 0

        for _, row in day_portfolio.iterrows():
            code = row["code"]
            w = row["weight"]
            if code in day_prices.index:
                oc_ret = day_prices.loc[code, "oc_return"]
                if isinstance(oc_ret, pd.Series):
                    oc_ret = oc_ret.iloc[0]
                if not np.isnan(oc_ret):
                    ret += w * oc_ret
                    n_traded += 1
                    if w > 0:
                        n_long += 1
                    elif w < 0:
                        n_short += 1

        # コスト: 売買した銘柄数に応じて
        current_codes = set(day_portfolio["code"])
        new_entries = current_codes - prev_codes
        turnover_ratio = len(new_entries) / max(1, len(current_codes))
        cost = round_trip_cost * turnover_ratio
        prev_codes = current_codes

        if n_traded > 0:
            daily_returns.append({"date": dt, "return": ret - cost})
        else:
            daily_returns.append({"date": dt, "return": np.nan})

        daily_long_counts.append({"date": dt, "count": n_long})
        daily_short_counts.append({"date": dt, "count": n_short})

    df_returns = pd.DataFrame(daily_returns)
    if df_returns.empty:
        return StockBacktestResult(metrics={"error": "リターンが計算できませんでした"})

    ret_series = pd.Series(
        df_returns["return"].values,
        index=pd.DatetimeIndex(df_returns["date"]),
        name="stock_ls",
    )

    long_series = pd.Series(
        [r["count"] for r in daily_long_counts],
        index=pd.DatetimeIndex([r["date"] for r in daily_long_counts]),
    )
    short_series = pd.Series(
        [r["count"] for r in daily_short_counts],
        index=pd.DatetimeIndex([r["date"] for r in daily_short_counts]),
    )

    # メトリクス計算 (lead_lag の compute_metrics と同じ形式)
    from core.lead_lag.strategy import compute_metrics
    metrics = compute_metrics(ret_series.values, ret_series.index)

    return StockBacktestResult(
        daily_returns=ret_series,
        metrics=metrics,
        daily_long_count=long_series,
        daily_short_count=short_series,
    )
