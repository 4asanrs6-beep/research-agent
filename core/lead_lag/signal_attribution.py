"""PCAシグナルの寄与分解 (Signal Attribution)

ETF-to-ETFのPCAシグナルを、各USセクター → 各JPセクターへの
因子空間上の寄与に分解する。Phase 0: 伝播ルート候補の分析。

注意:
    ここで得られるのは **因子空間での結合強度の分解** であり、
    経済的因果を意味する「ルート」ではない。
    あくまで寄与候補として扱い、安定性・有意性を検証した上で
    Phase 1以降の入力として使う。
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .strategy import (
    BacktestResult,
    build_prior_subspace,
    estimate_prior_correlation,
    _get_sector_indices,
)
from .config import LeadLagConfig, get_ticker_name

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """寄与分解の結果"""

    # 日次寄与行列: contribution_cube[t, i, j] = US_i → JP_j への寄与
    # shape: (T, n_us, n_jp)
    contribution_cube: np.ndarray | None = None

    # 結合強度行列: coupling[t, i, j] = Σ_k V_J[j,k] * V_U[i,k]
    # (USリターンを掛ける前の構造的結合強度)
    # shape: (T, n_us, n_jp)
    coupling_cube: np.ndarray | None = None

    # 期間別安定性: period_stability[period_label] = {
    #   "mean_coupling": (n_us, n_jp),
    #   "std_coupling": (n_us, n_jp),
    #   "ic_by_pair": (n_us, n_jp),
    #   "hit_rate_by_pair": (n_us, n_jp),
    # }
    period_stability: dict = field(default_factory=dict)

    # 全期間の平均結合強度 (n_us, n_jp)
    mean_coupling: np.ndarray | None = None

    # 安定ペアの一覧: [(us_ticker, jp_ticker, mean_coupling, stability_score), ...]
    stable_pairs: list = field(default_factory=list)

    # メタデータ
    dates: pd.DatetimeIndex | None = None
    us_tickers: list[str] = field(default_factory=list)
    jp_tickers: list[str] = field(default_factory=list)


def compute_signal_attribution(
    backtest_result: BacktestResult,
    config: LeadLagConfig,
    sub_periods: dict[str, tuple[str, str]] | None = None,
    progress_callback=None,
) -> AttributionResult:
    """PCA_SUBシグナルの寄与分解を実行する。

    既存のBacktestResultから、rolling_pca_signalと同じ計算を再実行しつつ
    各時点での US→JP 寄与を分解して記録する。

    Args:
        backtest_result: run_backtest() の結果
        config: 使用したLeadLagConfig
        sub_periods: 期間別分析の定義。例:
            {"2015-2019": ("2015-01-01", "2019-12-31"),
             "2020-2024": ("2020-01-01", "2024-12-31")}
            None の場合、5年ごとに自動分割
        progress_callback: (msg, pct) コールバック

    Returns:
        AttributionResult
    """
    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    us_cc = backtest_result.us_cc_returns
    if us_cc is None:
        raise ValueError("BacktestResult に us_cc_returns がありません")

    dates = us_cc.index
    us_tickers = list(us_cc.columns)
    jp_tickers = backtest_result.jp_tickers

    us_cc_vals = us_cc.values
    # JP close-to-close を復元: PCA_SUBのシグナル生成に使われたもの
    pca_sub = backtest_result.strategies.get("PCA_SUB")
    if pca_sub is None or pca_sub.signals is None:
        raise ValueError("PCA_SUB の結果がありません。run_pca_sub=True で実行してください")

    # JP cc リターンを再構築
    # run_all_strategiesでは jp_cc (aligned) を直接使っている
    # BacktestResultにはjp_ccが保存されていないので、
    # jp_close_pricesからcc リターンを計算する
    jp_close = backtest_result.jp_close_prices
    if jp_close is None:
        raise ValueError("BacktestResult に jp_close_prices がありません")

    # 共通日付に整列
    common_dates = dates.intersection(jp_close.index)
    jp_close_aligned = jp_close.loc[common_dates, jp_tickers]
    jp_cc_vals = jp_close_aligned.pct_change().values
    # 最初の行はNaN（pct_changeの結果）
    jp_cc_vals[0] = 0.0

    us_cc_aligned = us_cc.loc[common_dates].values

    n_us = len(us_tickers)
    n_jp = len(jp_tickers)
    T = len(common_dates)
    L = config.rolling_window
    K = config.n_components

    # --- 事前部分空間・相関行列の再構築 ---
    _progress("事前部分空間を再構築中...", 0.05)
    us_cyc, us_def, jp_cyc, jp_def = _get_sector_indices(us_tickers, jp_tickers, config)
    V0 = build_prior_subspace(n_us, n_jp, us_cyc, us_def, jp_cyc, jp_def)

    prior_end = pd.Timestamp(config.prior_end_date)
    prior_mask = common_dates <= prior_end
    if prior_mask.sum() < L:
        prior_mask = np.ones(T, dtype=bool)

    combined_prior = np.hstack([us_cc_aligned[prior_mask], jp_cc_vals[prior_mask]])
    C0 = estimate_prior_correlation(V0, combined_prior)

    # --- ローリングPCA + 寄与分解 ---
    _progress("ローリングPCA寄与分解を実行中...", 0.10)
    combined = np.hstack([us_cc_aligned, jp_cc_vals])

    contribution_cube = np.full((T, n_us, n_jp), np.nan)
    coupling_cube = np.full((T, n_us, n_jp), np.nan)

    for t in range(L, T):
        if t % 100 == 0:
            _progress(f"寄与分解中... ({t}/{T})", 0.10 + 0.70 * (t - L) / max(1, T - L))

        window = combined[t - L: t]
        mu = window.mean(axis=0)
        sigma = window.std(axis=0)
        sigma[sigma < 1e-12] = 1.0

        # ウィンドウ内相関行列
        Z = (window - mu) / sigma
        Ct = np.corrcoef(Z, rowvar=False)

        # 正則化
        C_reg = (1 - config.lambda_reg) * Ct + config.lambda_reg * C0

        # 固有分解
        eigvals, eigvecs = np.linalg.eigh(C_reg)
        idx_sort = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx_sort]

        V_K = eigvecs[:, :K]
        V_U = V_K[:n_us, :]  # (n_us, K)
        V_J = V_K[n_us:, :]  # (n_jp, K)

        # 結合強度行列: A[i, j] = Σ_k V_U[i,k] * V_J[j,k]
        # = V_U @ V_J^T → (n_us, n_jp)
        A = V_U @ V_J.T
        coupling_cube[t] = A

        # 当日USリターン (標準化)
        z_U_t = (combined[t, :n_us] - mu[:n_us]) / sigma[:n_us]

        # 寄与分解: contribution[i, j] = A[i, j] * z_U[i]
        # z_hat_J[j] = Σ_i A[i,j] * z_U[i] が成立する
        contribution_cube[t] = A * z_U_t[:, np.newaxis]  # (n_us, n_jp)

    _progress("期間別安定性を分析中...", 0.85)

    # --- 期間別分析 ---
    if sub_periods is None:
        sub_periods = _auto_split_periods(common_dates, years=5)

    period_stability = {}
    for label, (p_start, p_end) in sub_periods.items():
        p_mask = (common_dates >= pd.Timestamp(p_start)) & (common_dates <= pd.Timestamp(p_end))
        p_indices = np.where(p_mask)[0]
        # ウィンドウ不足の時点を除外
        p_indices = p_indices[p_indices >= L]

        if len(p_indices) < 20:
            continue

        p_coupling = coupling_cube[p_indices]
        p_contribution = contribution_cube[p_indices]

        # JP oc リターン
        jp_oc_vals = None
        if backtest_result.jp_open_prices is not None:
            jp_open = backtest_result.jp_open_prices
            jp_open_aligned = jp_open.loc[common_dates, jp_tickers]
            jp_oc_vals = (jp_close_aligned.values / jp_open_aligned.values) - 1.0

        # ペア別IC: 各(i,j)について、contribution[t,i,j]とjp_oc[t,j]のSpearman相関
        ic_by_pair = np.full((n_us, n_jp), np.nan)
        hit_rate_by_pair = np.full((n_us, n_jp), np.nan)

        if jp_oc_vals is not None:
            for j in range(n_jp):
                jp_ret_j = jp_oc_vals[p_indices, j]
                for i in range(n_us):
                    contrib_ij = p_contribution[:, i, j]
                    valid = ~(np.isnan(contrib_ij) | np.isnan(jp_ret_j))
                    if valid.sum() < 20:
                        continue
                    c = contrib_ij[valid]
                    r = jp_ret_j[valid]
                    # Spearman rank correlation
                    corr, _ = spearmanr(c, r)
                    ic_by_pair[i, j] = corr
                    # ヒット率: 符号一致率
                    hit_rate_by_pair[i, j] = np.mean(np.sign(c) == np.sign(r))

        period_stability[label] = {
            "mean_coupling": np.nanmean(p_coupling, axis=0),
            "std_coupling": np.nanstd(p_coupling, axis=0),
            "ic_by_pair": ic_by_pair,
            "hit_rate_by_pair": hit_rate_by_pair,
            "n_days": len(p_indices),
        }

    _progress("安定ペアを抽出中...", 0.95)

    # --- 全期間の平均結合強度 ---
    valid_coupling = coupling_cube[L:]
    mean_coupling = np.nanmean(valid_coupling, axis=0)
    std_coupling = np.nanstd(valid_coupling, axis=0)

    # --- 安定ペアの抽出 ---
    stable_pairs = _extract_stable_pairs(
        mean_coupling, std_coupling, period_stability,
        us_tickers, jp_tickers,
    )

    _progress("完了", 1.0)

    return AttributionResult(
        contribution_cube=contribution_cube,
        coupling_cube=coupling_cube,
        period_stability=period_stability,
        mean_coupling=mean_coupling,
        stable_pairs=stable_pairs,
        dates=common_dates,
        us_tickers=us_tickers,
        jp_tickers=jp_tickers,
    )


def _auto_split_periods(
    dates: pd.DatetimeIndex,
    years: int = 5,
) -> dict[str, tuple[str, str]]:
    """日付範囲を自動的にN年ごとの期間に分割する。"""
    start_year = dates[0].year
    end_year = dates[-1].year
    periods = {}
    y = start_year
    while y <= end_year:
        p_end = min(y + years - 1, end_year)
        label = f"{y}-{p_end}"
        periods[label] = (f"{y}-01-01", f"{p_end}-12-31")
        y += years
    return periods


def _extract_stable_pairs(
    mean_coupling: np.ndarray,
    std_coupling: np.ndarray,
    period_stability: dict,
    us_tickers: list[str],
    jp_tickers: list[str],
    min_periods_positive: int = 2,
) -> list[dict]:
    """複数期間で安定的に正（or 負）の結合強度を持つペアを抽出する。

    安定性スコア = |mean_coupling| / (std_coupling + eps)  × 期間一致率
    """
    n_us, n_jp = mean_coupling.shape
    pairs = []

    for i in range(n_us):
        for j in range(n_jp):
            mc = mean_coupling[i, j]
            sc = std_coupling[i, j]
            if np.isnan(mc) or np.isnan(sc):
                continue

            # 期間別の符号一致チェック
            n_periods = len(period_stability)
            if n_periods == 0:
                continue

            same_sign_count = 0
            avg_ic = []
            avg_hit = []
            for label, pdata in period_stability.items():
                p_mc = pdata["mean_coupling"][i, j]
                if not np.isnan(p_mc) and np.sign(p_mc) == np.sign(mc):
                    same_sign_count += 1
                p_ic = pdata["ic_by_pair"][i, j]
                if not np.isnan(p_ic):
                    avg_ic.append(p_ic)
                p_hit = pdata["hit_rate_by_pair"][i, j]
                if not np.isnan(p_hit):
                    avg_hit.append(p_hit)

            if same_sign_count < min(min_periods_positive, n_periods):
                continue

            consistency = same_sign_count / n_periods
            stability_score = abs(mc) / (sc + 1e-8) * consistency

            pairs.append({
                "us_ticker": us_tickers[i],
                "jp_ticker": jp_tickers[j],
                "us_name": get_ticker_name(us_tickers[i]),
                "jp_name": get_ticker_name(jp_tickers[j]),
                "mean_coupling": float(mc),
                "std_coupling": float(sc),
                "stability_score": float(stability_score),
                "period_consistency": float(consistency),
                "avg_ic": float(np.mean(avg_ic)) if avg_ic else float("nan"),
                "avg_hit_rate": float(np.mean(avg_hit)) if avg_hit else float("nan"),
            })

    pairs.sort(key=lambda x: x["stability_score"], reverse=True)
    return pairs


def summarize_attribution(
    attr: AttributionResult,
    top_n: int = 20,
) -> pd.DataFrame:
    """寄与分解結果を読みやすいDataFrameに要約する。

    Args:
        attr: AttributionResult
        top_n: 上位N件を返す

    Returns:
        DataFrame with columns:
            US, JP, mean_coupling, stability_score,
            period_consistency, avg_IC, avg_hit_rate
    """
    if not attr.stable_pairs:
        return pd.DataFrame()

    rows = attr.stable_pairs[:top_n]
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "us_name": "US",
        "jp_name": "JP",
        "avg_ic": "avg_IC",
        "avg_hit_rate": "avg_hit_rate",
    })
    cols = ["US", "JP", "mean_coupling", "stability_score",
            "period_consistency", "avg_IC", "avg_hit_rate"]
    return df[[c for c in cols if c in df.columns]]


def get_coupling_heatmap_data(
    attr: AttributionResult,
    period: str | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """ヒートマップ表示用のデータを返す。

    Args:
        attr: AttributionResult
        period: 期間ラベル。Noneの場合は全期間平均

    Returns:
        (matrix, us_labels, jp_labels)
    """
    if period and period in attr.period_stability:
        matrix = attr.period_stability[period]["mean_coupling"]
    else:
        matrix = attr.mean_coupling

    us_labels = [get_ticker_name(t) for t in attr.us_tickers]
    jp_labels = [get_ticker_name(t) for t in attr.jp_tickers]

    return matrix, us_labels, jp_labels
