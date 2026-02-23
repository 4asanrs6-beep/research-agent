"""統計分析エンジン"""

import logging
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy import stats

from core.models import StatisticsResult

logger = logging.getLogger(__name__)


class Analyzer:
    """統計分析エンジン: 各種分析手法を実行し結果を返す"""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def analyze(
        self,
        prices: pd.DataFrame,
        method: str,
        parameters: dict,
        index_prices: pd.DataFrame | None = None,
    ) -> dict:
        """分析を実行して結果辞書を返す"""
        dispatch = {
            "calendar_effect": self._analyze_calendar_effect,
            "momentum": self._analyze_momentum,
            "value": self._analyze_quantile_sort,
            "reversal": self._analyze_momentum,  # 同じロジック、パラメータで制御
            "volatility": self._analyze_volatility,
            "technical": self._analyze_technical,
            "sector_rotation": self._analyze_sector_rotation,
            "custom": self._analyze_custom,
        }
        func = dispatch.get(method, self._analyze_custom)
        try:
            result = func(prices, parameters, index_prices)
            return asdict(result) if isinstance(result, StatisticsResult) else result
        except Exception as e:
            logger.error("分析エラー (%s): %s", method, e)
            return {"error": str(e)}

    def _compute_stats(
        self,
        condition_returns: np.ndarray,
        baseline_returns: np.ndarray,
        test_name: str = "",
    ) -> StatisticsResult:
        """2群のリターン比較の統計量を計算"""
        result = StatisticsResult(test_name=test_name)

        cond = condition_returns[~np.isnan(condition_returns)]
        base = baseline_returns[~np.isnan(baseline_returns)]

        result.n_condition = len(cond)
        result.n_baseline = len(base)

        if len(cond) < 2 or len(base) < 2:
            return result

        result.condition_mean = float(np.mean(cond))
        result.baseline_mean = float(np.mean(base))
        result.condition_std = float(np.std(cond, ddof=1))
        result.baseline_std = float(np.std(base, ddof=1))
        result.win_rate_condition = float(np.mean(cond > 0))
        result.win_rate_baseline = float(np.mean(base > 0))

        # t検定
        t_stat, p_val = stats.ttest_ind(cond, base, equal_var=False)
        result.t_statistic = float(t_stat)
        result.p_value = float(p_val)

        # Mann-Whitney U検定（補助）
        try:
            _, p_mw = stats.mannwhitneyu(cond, base, alternative="two-sided")
        except ValueError:
            p_mw = 1.0

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(cond) - 1) * result.condition_std ** 2 + (len(base) - 1) * result.baseline_std ** 2)
            / (len(cond) + len(base) - 2)
        )
        if pooled_std > 0:
            result.cohens_d = float((result.condition_mean - result.baseline_mean) / pooled_std)

        result.is_significant = result.p_value < self.significance_level

        return result

    def _analyze_calendar_effect(
        self,
        prices: pd.DataFrame,
        params: dict,
        index_prices: pd.DataFrame | None = None,
    ) -> StatisticsResult:
        """カレンダー効果の分析"""
        df = prices.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        else:
            return StatisticsResult(test_name="calendar_effect")

        # 調整後終値があればそちらを使用
        close_col = "adj_close" if "adj_close" in df.columns else "close"

        # 銘柄ごとのリターンを計算
        df = df.sort_values(["code", "date"])
        horizon = params.get("return_horizon", 1)
        df["return"] = df.groupby("code")[close_col].pct_change(horizon)

        effect_type = params.get("effect_type", "month_of_year")
        target = params.get("target_period")

        if effect_type == "month_of_year":
            df["period"] = df["date"].dt.month
        elif effect_type == "day_of_week":
            df["period"] = df["date"].dt.dayofweek  # 0=Mon
        elif effect_type == "turn_of_month":
            df["period"] = df["date"].dt.day.apply(lambda d: 1 if d <= 5 or d >= 25 else 0)
        else:
            df["period"] = df["date"].dt.month

        df = df.dropna(subset=["return"])

        if target is not None:
            condition = df[df["period"] == target]["return"].values
            baseline = df[df["period"] != target]["return"].values
        else:
            # targetが未指定: 全期間グループごとに最もリターンが高い期間を条件群とする
            group_means = df.groupby("period")["return"].mean()
            best_period = group_means.idxmax()
            condition = df[df["period"] == best_period]["return"].values
            baseline = df[df["period"] != best_period]["return"].values

        return self._compute_stats(condition, baseline, "calendar_effect")

    def _analyze_momentum(
        self,
        prices: pd.DataFrame,
        params: dict,
        index_prices: pd.DataFrame | None = None,
    ) -> StatisticsResult:
        """モメンタム / リバーサルの分析"""
        df = prices.copy()
        close_col = "adj_close" if "adj_close" in df.columns else "close"

        lookback = params.get("lookback_days", 20)
        holding = params.get("holding_days", 20)
        n_q = params.get("n_quantiles", 5)
        long_q = params.get("long_quantile", n_q)

        df = df.sort_values(["code", "date"])
        df["past_return"] = df.groupby("code")[close_col].pct_change(lookback)
        df["future_return"] = df.groupby("code")[close_col].shift(-holding) / df[close_col] - 1
        df = df.dropna(subset=["past_return", "future_return"])

        # 日付ごとにクロスセクショナルにランク付け
        df["quantile"] = df.groupby("date")["past_return"].transform(
            lambda x: pd.qcut(x, n_q, labels=False, duplicates="drop") + 1
            if len(x) >= n_q else np.nan
        )
        df = df.dropna(subset=["quantile"])

        condition = df[df["quantile"] == long_q]["future_return"].values
        baseline = df[df["quantile"] != long_q]["future_return"].values

        return self._compute_stats(condition, baseline, "momentum")

    def _analyze_quantile_sort(
        self,
        prices: pd.DataFrame,
        params: dict,
        index_prices: pd.DataFrame | None = None,
    ) -> StatisticsResult:
        """バリュー等のクォンタイルソート分析"""
        # バリュー指標がpricesに含まれている前提 (per, pbr等)
        df = prices.copy()
        close_col = "adj_close" if "adj_close" in df.columns else "close"
        metric = params.get("metric", "per")
        holding = params.get("holding_days", 60)
        n_q = params.get("n_quantiles", 5)
        long_q = params.get("long_quantile", 1)

        df = df.sort_values(["code", "date"])
        df["future_return"] = df.groupby("code")[close_col].shift(-holding) / df[close_col] - 1

        if metric not in df.columns:
            # 指標がない場合はモメンタムにフォールバック
            return self._analyze_momentum(prices, params, index_prices)

        df = df.dropna(subset=[metric, "future_return"])

        df["quantile"] = df.groupby("date")[metric].transform(
            lambda x: pd.qcut(x, n_q, labels=False, duplicates="drop") + 1
            if len(x) >= n_q else np.nan
        )
        df = df.dropna(subset=["quantile"])

        condition = df[df["quantile"] == long_q]["future_return"].values
        baseline = df[df["quantile"] != long_q]["future_return"].values

        return self._compute_stats(condition, baseline, "quantile_sort")

    def _analyze_volatility(
        self,
        prices: pd.DataFrame,
        params: dict,
        index_prices: pd.DataFrame | None = None,
    ) -> StatisticsResult:
        """ボラティリティベースの分析"""
        df = prices.copy()
        close_col = "adj_close" if "adj_close" in df.columns else "close"

        vol_lookback = params.get("vol_lookback_days", 20)
        holding = params.get("holding_days", 20)
        n_q = params.get("n_quantiles", 5)
        long_q = params.get("long_quantile", 1)

        df = df.sort_values(["code", "date"])
        df["daily_return"] = df.groupby("code")[close_col].pct_change()
        df["volatility"] = df.groupby("code")["daily_return"].transform(
            lambda x: x.rolling(vol_lookback).std()
        )
        df["future_return"] = df.groupby("code")[close_col].shift(-holding) / df[close_col] - 1
        df = df.dropna(subset=["volatility", "future_return"])

        df["quantile"] = df.groupby("date")["volatility"].transform(
            lambda x: pd.qcut(x, n_q, labels=False, duplicates="drop") + 1
            if len(x) >= n_q else np.nan
        )
        df = df.dropna(subset=["quantile"])

        condition = df[df["quantile"] == long_q]["future_return"].values
        baseline = df[df["quantile"] != long_q]["future_return"].values

        return self._compute_stats(condition, baseline, "volatility")

    def _analyze_technical(
        self,
        prices: pd.DataFrame,
        params: dict,
        index_prices: pd.DataFrame | None = None,
    ) -> StatisticsResult:
        """テクニカル指標分析"""
        df = prices.copy()
        close_col = "adj_close" if "adj_close" in df.columns else "close"
        indicator = params.get("indicator", "sma_cross")
        holding = params.get("holding_days", 10)

        df = df.sort_values(["code", "date"])
        df["future_return"] = df.groupby("code")[close_col].shift(-holding) / df[close_col] - 1

        if indicator == "sma_cross":
            short_w = params.get("short_window", 5)
            long_w = params.get("long_window", 25)
            df["sma_short"] = df.groupby("code")[close_col].transform(
                lambda x: x.rolling(short_w).mean()
            )
            df["sma_long"] = df.groupby("code")[close_col].transform(
                lambda x: x.rolling(long_w).mean()
            )
            df = df.dropna(subset=["sma_short", "sma_long", "future_return"])
            df["signal"] = (df["sma_short"] > df["sma_long"]).astype(int)

        elif indicator == "rsi":
            window = params.get("short_window", 14)
            df["daily_return"] = df.groupby("code")[close_col].pct_change()
            df["gain"] = df["daily_return"].clip(lower=0)
            df["loss"] = (-df["daily_return"]).clip(lower=0)
            df["avg_gain"] = df.groupby("code")["gain"].transform(
                lambda x: x.rolling(window).mean()
            )
            df["avg_loss"] = df.groupby("code")["loss"].transform(
                lambda x: x.rolling(window).mean()
            )
            df["rsi"] = 100 - 100 / (1 + df["avg_gain"] / df["avg_loss"].replace(0, np.nan))
            df = df.dropna(subset=["rsi", "future_return"])
            df["signal"] = (df["rsi"] < 30).astype(int)  # 売られすぎ → 買い

        else:
            df = df.dropna(subset=["future_return"])
            df["signal"] = 0

        condition = df[df["signal"] == 1]["future_return"].values
        baseline = df[df["signal"] == 0]["future_return"].values

        return self._compute_stats(condition, baseline, f"technical_{indicator}")

    def _analyze_sector_rotation(
        self,
        prices: pd.DataFrame,
        params: dict,
        index_prices: pd.DataFrame | None = None,
    ) -> StatisticsResult:
        """セクターローテーション分析"""
        df = prices.copy()
        close_col = "adj_close" if "adj_close" in df.columns else "close"
        sector_col = params.get("sector_type", "sector_33")
        lookback = params.get("lookback_days", 20)
        holding = params.get("holding_days", 20)

        if sector_col not in df.columns:
            return StatisticsResult(test_name="sector_rotation")

        # セクター別リターン計算
        df = df.sort_values(["code", "date"])
        df["return"] = df.groupby("code")[close_col].pct_change(lookback)
        df["future_return"] = df.groupby("code")[close_col].shift(-holding) / df[close_col] - 1
        df = df.dropna(subset=["return", "future_return"])

        # セクター平均リターンでランク → 上位セクターが条件群
        sector_mean = df.groupby(["date", sector_col])["return"].mean().reset_index()
        sector_median = sector_mean.groupby("date")["return"].median()
        sector_mean = sector_mean.merge(
            sector_median.rename("median_return"), on="date"
        )
        top_sectors = sector_mean[sector_mean["return"] > sector_mean["median_return"]]
        top_keys = set(zip(top_sectors["date"], top_sectors[sector_col]))

        df["is_top"] = df.apply(
            lambda r: (r["date"], r[sector_col]) in top_keys, axis=1
        )

        condition = df[df["is_top"]]["future_return"].values
        baseline = df[~df["is_top"]]["future_return"].values

        return self._compute_stats(condition, baseline, "sector_rotation")

    def _analyze_custom(
        self,
        prices: pd.DataFrame,
        params: dict,
        index_prices: pd.DataFrame | None = None,
    ) -> dict:
        """カスタム分析: 基本的なリターン統計を返す"""
        df = prices.copy()
        close_col = "adj_close" if "adj_close" in df.columns else "close"
        df = df.sort_values(["code", "date"])
        df["return"] = df.groupby("code")[close_col].pct_change()
        df = df.dropna(subset=["return"])

        returns = df["return"].values
        return {
            "test_name": "custom",
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns, ddof=1)),
            "median_return": float(np.median(returns)),
            "skewness": float(stats.skew(returns)),
            "kurtosis": float(stats.kurtosis(returns)),
            "n_observations": len(returns),
            "win_rate": float(np.mean(returns > 0)),
        }
