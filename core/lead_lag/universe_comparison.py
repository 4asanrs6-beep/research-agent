"""US11 vs US33 ユニバース比較分析 (Phase 2)

US 11セクター版と US 33種版の寄与分解を比較し、
細分化による追加情報の有無を検証する。
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import LeadLagConfig
from .strategy import BacktestResult, compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class UniverseComparisonResult:
    """US11 vs US33 の比較結果"""
    # 各ユニバースのメトリクス
    us11_metrics: dict = field(default_factory=dict)
    us33_metrics: dict = field(default_factory=dict)

    # 差分
    delta_ar: float = 0.0        # 年率差 (US33 - US11)
    delta_sharpe: float = 0.0    # シャープ比差
    delta_mdd: float = 0.0       # MDD差

    # 相関
    return_correlation: float = 0.0  # 日次リターンの相関
    signal_correlation: dict = field(default_factory=dict)  # JP ETFごとのシグナル相関

    # US33追加分の貢献度
    marginal_ic: float = float("nan")  # US33のみの追加シグナルのIC

    # 寄与分解の差分
    us33_unique_pairs: list = field(default_factory=list)  # US33でのみ有意なペア


def compare_universes(
    result_us11: BacktestResult,
    result_us33: BacktestResult,
) -> UniverseComparisonResult:
    """US11とUS33の結果を比較する。

    Args:
        result_us11: us_universe="US_11" で実行したBacktestResult
        result_us33: us_universe="US_33" で実行したBacktestResult

    Returns:
        UniverseComparisonResult
    """
    comp = UniverseComparisonResult()

    # --- メトリクス比較 ---
    pca11 = result_us11.strategies.get("PCA_SUB")
    pca33 = result_us33.strategies.get("PCA_SUB")

    if pca11 is None or pca33 is None:
        logger.warning("PCA_SUBが両方に必要です")
        return comp

    comp.us11_metrics = pca11.metrics
    comp.us33_metrics = pca33.metrics

    comp.delta_ar = pca33.metrics.get("AR", 0) - pca11.metrics.get("AR", 0)
    comp.delta_sharpe = pca33.metrics.get("R/R", 0) - pca11.metrics.get("R/R", 0)
    comp.delta_mdd = pca33.metrics.get("MDD", 0) - pca11.metrics.get("MDD", 0)

    # --- 日次リターン相関 ---
    ret11 = pca11.daily_returns.dropna()
    ret33 = pca33.daily_returns.dropna()
    common_idx = ret11.index.intersection(ret33.index)
    if len(common_idx) > 10:
        comp.return_correlation = float(ret11.loc[common_idx].corr(ret33.loc[common_idx]))

    # --- JP ETFごとのシグナル相関 ---
    if pca11.signals is not None and pca33.signals is not None:
        jp_tickers_11 = pca11.signals.columns
        jp_tickers_33 = pca33.signals.columns
        common_jp = [t for t in jp_tickers_11 if t in jp_tickers_33]
        common_dates = pca11.signals.index.intersection(pca33.signals.index)

        for jp_t in common_jp:
            s11 = pca11.signals.loc[common_dates, jp_t].dropna()
            s33 = pca33.signals.loc[common_dates, jp_t].dropna()
            ci = s11.index.intersection(s33.index)
            if len(ci) > 20:
                corr = float(s11.loc[ci].corr(s33.loc[ci]))
                comp.signal_correlation[jp_t] = corr

    return comp


def build_comparison_table(comp: UniverseComparisonResult) -> pd.DataFrame:
    """比較結果を表形式にまとめる。"""
    rows = []
    metrics_keys = ["AR", "RISK", "R/R", "MDD", "monthly_win_rate"]
    labels = {
        "AR": "年率リターン (%)",
        "RISK": "リスク (%)",
        "R/R": "シャープ比",
        "MDD": "最大下落率 (%)",
        "monthly_win_rate": "月次勝率 (%)",
    }

    for key in metrics_keys:
        v11 = comp.us11_metrics.get(key, float("nan"))
        v33 = comp.us33_metrics.get(key, float("nan"))
        rows.append({
            "指標": labels.get(key, key),
            "US 11セクター": v11,
            "US 33種": v33,
            "差分": v33 - v11 if not (np.isnan(v11) or np.isnan(v33)) else float("nan"),
        })

    rows.append({
        "指標": "日次リターン相関",
        "US 11セクター": "-",
        "US 33種": "-",
        "差分": comp.return_correlation,
    })

    return pd.DataFrame(rows)
