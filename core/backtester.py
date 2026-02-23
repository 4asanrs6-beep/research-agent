"""バックテストエンジン"""

import logging
from dataclasses import asdict

import numpy as np
import pandas as pd

from core.models import BacktestResult

logger = logging.getLogger(__name__)


class Backtester:
    """シンプルなイベントドリブン型バックテスター"""

    def __init__(
        self,
        initial_capital: float = 10_000_000,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.001,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        benchmark_prices: pd.DataFrame | None = None,
        rebalance_frequency: str = "monthly",
    ) -> dict:
        """バックテストを実行

        Args:
            prices: 株価データ (columns: date, code, close or adj_close)
            signals: シグナルデータ (columns: date, code, weight)
                     weight > 0: ロング, weight = 0: ノーポジション
            benchmark_prices: ベンチマーク価格 (columns: date, close)
            rebalance_frequency: リバランス頻度 (daily, weekly, monthly)
        """
        try:
            result = self._execute(prices, signals, benchmark_prices, rebalance_frequency)
            return asdict(result)
        except Exception as e:
            logger.error("バックテストエラー: %s", e)
            return {"error": str(e)}

    def _execute(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame,
        benchmark_prices: pd.DataFrame | None,
        rebalance_frequency: str,
    ) -> BacktestResult:
        close_col = "adj_close" if "adj_close" in prices.columns else "close"

        # 日付でソート
        prices = prices.sort_values("date")
        signals = signals.sort_values("date")

        # リバランス日を決定
        all_dates = sorted(prices["date"].unique())
        rebalance_dates = self._get_rebalance_dates(all_dates, rebalance_frequency)

        # ポートフォリオ計算
        capital = self.initial_capital
        equity_curve = []
        trade_log = []
        holdings = {}  # code -> shares
        wins = 0
        total_trades = 0

        for i, date in enumerate(all_dates):
            day_prices = prices[prices["date"] == date].set_index("code")[close_col].to_dict()

            # ポートフォリオ時価評価
            portfolio_value = capital
            for code, shares in holdings.items():
                if code in day_prices:
                    portfolio_value += shares * day_prices[code]

            equity_curve.append({"date": str(date)[:10], "value": portfolio_value})

            # リバランス日の場合
            if date in rebalance_dates:
                day_signals = signals[signals["date"] == date]
                if len(day_signals) > 0:
                    target_weights = dict(zip(day_signals["code"], day_signals["weight"]))

                    # 既存ポジションを清算
                    for code, shares in list(holdings.items()):
                        if code in day_prices and shares != 0:
                            sell_price = day_prices[code] * (1 - self.slippage_rate)
                            proceeds = shares * sell_price
                            commission = abs(proceeds) * self.commission_rate
                            pnl = proceeds - commission
                            capital += pnl
                            if pnl > 0:
                                wins += 1
                            total_trades += 1
                            trade_log.append({
                                "date": str(date)[:10],
                                "code": code,
                                "action": "sell",
                                "shares": shares,
                                "price": sell_price,
                                "pnl": pnl,
                            })
                    holdings = {}

                    # 新規ポジション構築
                    total_weight = sum(w for w in target_weights.values() if w > 0)
                    if total_weight > 0:
                        for code, weight in target_weights.items():
                            if weight > 0 and code in day_prices:
                                alloc = capital * (weight / total_weight)
                                buy_price = day_prices[code] * (1 + self.slippage_rate)
                                shares = int(alloc / buy_price / 100) * 100  # 100株単位
                                if shares > 0:
                                    cost = shares * buy_price
                                    commission = cost * self.commission_rate
                                    capital -= cost + commission
                                    holdings[code] = shares
                                    trade_log.append({
                                        "date": str(date)[:10],
                                        "code": code,
                                        "action": "buy",
                                        "shares": shares,
                                        "price": buy_price,
                                    })

        # 最終評価
        if not equity_curve:
            return BacktestResult()

        initial = equity_curve[0]["value"]
        final = equity_curve[-1]["value"]
        cumulative_return = (final / initial) - 1

        # 日次リターン系列
        values = [e["value"] for e in equity_curve]
        daily_returns = np.diff(values) / values[:-1] if len(values) > 1 else np.array([])

        # 年率リターン
        n_days = len(equity_curve)
        annual_return = (1 + cumulative_return) ** (252 / max(n_days, 1)) - 1 if n_days > 0 else 0

        # シャープ比
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # 最大ドローダウン
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd

        result = BacktestResult(
            cumulative_return=float(cumulative_return),
            annual_return=float(annual_return),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            win_rate=float(wins / total_trades) if total_trades > 0 else 0.0,
            total_trades=total_trades,
            equity_curve=equity_curve,
            trade_log=trade_log,
        )

        # ベンチマーク計算
        if benchmark_prices is not None and len(benchmark_prices) > 0:
            bench = benchmark_prices.sort_values("date")
            bench_close = "close" if "close" in bench.columns else bench.columns[-1]
            bench_values = bench[bench_close].values
            if len(bench_values) > 1:
                bench_returns = np.diff(bench_values) / bench_values[:-1]
                result.benchmark_cumulative_return = float(bench_values[-1] / bench_values[0] - 1)
                n_bench = len(bench_values)
                result.benchmark_annual_return = float(
                    (1 + result.benchmark_cumulative_return) ** (252 / max(n_bench, 1)) - 1
                )
                if np.std(bench_returns) > 0:
                    result.benchmark_sharpe_ratio = float(
                        np.mean(bench_returns) / np.std(bench_returns) * np.sqrt(252)
                    )
                result.benchmark_curve = [
                    {"date": str(d)[:10], "value": float(v * self.initial_capital / bench_values[0])}
                    for d, v in zip(bench["date"].values, bench_values)
                ]

        return result

    @staticmethod
    def _get_rebalance_dates(dates: list, frequency: str) -> set:
        """リバランス日のセットを返す"""
        if frequency == "daily":
            return set(dates)

        rebalance = set()
        prev_key = None
        for d in dates:
            d_ts = pd.Timestamp(d)
            if frequency == "weekly":
                key = (d_ts.isocalendar()[0], d_ts.isocalendar()[1])
            elif frequency == "monthly":
                key = (d_ts.year, d_ts.month)
            else:
                key = (d_ts.year, d_ts.month)

            if key != prev_key:
                rebalance.add(d)
                prev_key = key
        return rebalance

    def generate_signals_from_analysis(
        self,
        prices: pd.DataFrame,
        method: str,
        parameters: dict,
    ) -> pd.DataFrame:
        """分析結果からバックテスト用シグナルを生成"""
        close_col = "adj_close" if "adj_close" in prices.columns else "close"
        df = prices.sort_values(["code", "date"]).copy()

        if method == "calendar_effect":
            effect_type = parameters.get("effect_type", "month_of_year")
            target = parameters.get("target_period")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                if effect_type == "month_of_year":
                    df["period"] = df["date"].dt.month
                elif effect_type == "day_of_week":
                    df["period"] = df["date"].dt.dayofweek
                elif effect_type == "turn_of_month":
                    df["period"] = df["date"].dt.day.apply(lambda d: 1 if d <= 5 or d >= 25 else 0)
                if target is not None:
                    df["weight"] = (df["period"] == target).astype(float)
                else:
                    df["weight"] = 1.0

        elif method in ("momentum", "reversal"):
            lookback = parameters.get("lookback_days", 20)
            n_q = parameters.get("n_quantiles", 5)
            long_q = parameters.get("long_quantile", n_q if method == "momentum" else 1)
            df["past_return"] = df.groupby("code")[close_col].pct_change(lookback)
            df["quantile"] = df.groupby("date")["past_return"].transform(
                lambda x: pd.qcut(x, n_q, labels=False, duplicates="drop") + 1
                if len(x) >= n_q else np.nan
            )
            df["weight"] = (df["quantile"] == long_q).astype(float)

        elif method == "technical":
            indicator = parameters.get("indicator", "sma_cross")
            if indicator == "sma_cross":
                short_w = parameters.get("short_window", 5)
                long_w = parameters.get("long_window", 25)
                df["sma_short"] = df.groupby("code")[close_col].transform(
                    lambda x: x.rolling(short_w).mean()
                )
                df["sma_long"] = df.groupby("code")[close_col].transform(
                    lambda x: x.rolling(long_w).mean()
                )
                df["weight"] = (df["sma_short"] > df["sma_long"]).astype(float)
            else:
                df["weight"] = 1.0

        elif method == "volatility":
            vol_lb = parameters.get("vol_lookback_days", 20)
            n_q = parameters.get("n_quantiles", 5)
            long_q = parameters.get("long_quantile", 1)
            df["daily_return"] = df.groupby("code")[close_col].pct_change()
            df["volatility"] = df.groupby("code")["daily_return"].transform(
                lambda x: x.rolling(vol_lb).std()
            )
            df["quantile"] = df.groupby("date")["volatility"].transform(
                lambda x: pd.qcut(x, n_q, labels=False, duplicates="drop") + 1
                if len(x) >= n_q else np.nan
            )
            df["weight"] = (df["quantile"] == long_q).astype(float)

        else:
            df["weight"] = 1.0

        df = df.dropna(subset=["weight"])
        return df[["date", "code", "weight"]]
