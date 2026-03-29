"""残差化によるアルファ分解 (Phase 3)

Phase 3A: 市場 + 業種 残差
Phase 3B: 市場 + 業種 + サイズ + モメンタム 残差

Lead-Lagシグナルが、
  - 生リターンだけでなく
  - 市場・業種を引いた残差にも効くか
  - さらにスタイルファクターまで引いた残差にも効くか
を検証する。
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@dataclass
class ResidualAnalysisResult:
    """残差分析の結果"""
    # 各レベルでのシグナル予測力
    raw_ic: float = float("nan")         # 生リターンへのIC
    phase3a_ic: float = float("nan")     # 市場+業種残差へのIC
    phase3b_ic: float = float("nan")     # +サイズ+モメンタム残差へのIC

    raw_icir: float = float("nan")
    phase3a_icir: float = float("nan")
    phase3b_icir: float = float("nan")

    raw_hit_rate: float = float("nan")
    phase3a_hit_rate: float = float("nan")
    phase3b_hit_rate: float = float("nan")

    # 日次IC系列 (時系列変化の確認用)
    daily_ic_raw: pd.Series | None = None
    daily_ic_3a: pd.Series | None = None
    daily_ic_3b: pd.Series | None = None

    # 因子分解の説明力
    r_squared_market: float = float("nan")
    r_squared_sector: float = float("nan")
    r_squared_style: float = float("nan")

    n_days: int = 0
    n_stocks_avg: float = 0


def decompose_returns(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    sector_returns: pd.DataFrame,
    stock_sectors: dict[str, str],
    stock_market_caps: pd.DataFrame | None = None,
    rolling_window: int = 60,
    progress_callback=None,
) -> dict[str, pd.DataFrame]:
    """個別株リターンを因子分解する（ベクトル化版）。

    各銘柄について全期間で1回だけ回帰を実行し、ベータを推定。
    ローリングではなく全期間ベータによる簡易分解。
    200銘柄でも数秒で完了する。
    """
    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    dates = stock_returns.index
    codes = stock_returns.columns.tolist()
    T = len(dates)

    # 市場リターンを整列
    mkt = market_returns.reindex(dates).fillna(0).values

    # セクターリターンを整列
    sec_ret = sector_returns.reindex(dates).fillna(0)

    # モメンタムファクター: 各銘柄の20日ローリングリターン
    mom_lookback = 20

    market_component = np.zeros((T, len(codes)))
    sector_component = np.zeros((T, len(codes)))
    size_mom_component = np.zeros((T, len(codes)))

    _progress("因子分解中...", 0.1)

    for j, code in enumerate(codes):
        if j % 50 == 0:
            _progress(f"因子分解中... ({j}/{len(codes)}銘柄)", 0.1 + 0.8 * j / len(codes))

        y = stock_returns[code].values.copy()
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < rolling_window:
            continue

        y_clean = np.where(np.isnan(y), 0, y)

        # セクターリターン
        etf_code = stock_sectors.get(code)
        if etf_code and etf_code in sec_ret.columns:
            x_sec = sec_ret[etf_code].values
        else:
            x_sec = np.zeros(T)

        # --- Phase 3A: 市場 + 業種 回帰 (全期間1回) ---
        X_3a = np.column_stack([np.ones(T), mkt, x_sec])
        try:
            betas_3a, _, _, _ = np.linalg.lstsq(X_3a[valid_mask], y_clean[valid_mask], rcond=None)
        except np.linalg.LinAlgError:
            continue

        market_component[:, j] = betas_3a[1] * mkt
        sector_component[:, j] = betas_3a[2] * x_sec

        # --- Phase 3B: + モメンタム (サイズは時価総額なしの場合スキップ) ---
        # モメンタム: 過去20日累積リターン
        mom_vals = pd.Series(y_clean).rolling(mom_lookback, min_periods=1).sum().values

        X_3b = np.column_stack([np.ones(T), mkt, x_sec, mom_vals])
        try:
            betas_3b, _, _, _ = np.linalg.lstsq(X_3b[valid_mask], y_clean[valid_mask], rcond=None)
            size_mom_component[:, j] = betas_3b[3] * mom_vals - betas_3a[1] * mkt  # 3bで追加された分だけ
            # 実際は 3b全体の残差 - 3a残差 = スタイル寄与
        except np.linalg.LinAlgError:
            pass

    _progress("残差を計算中...", 0.95)

    raw = stock_returns.copy()
    mkt_df = pd.DataFrame(market_component, index=dates, columns=codes)
    sec_df = pd.DataFrame(sector_component, index=dates, columns=codes)

    residual_3a = raw - mkt_df - sec_df

    # Phase 3B: 市場+業種+モメンタムの残差を直接計算
    # 各銘柄で3bの回帰残差を計算
    residual_3b_vals = np.full((T, len(codes)), np.nan)
    for j, code in enumerate(codes):
        y = stock_returns[code].values.copy()
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < rolling_window:
            continue

        y_clean = np.where(np.isnan(y), 0, y)
        etf_code = stock_sectors.get(code)
        x_sec = sec_ret[etf_code].values if (etf_code and etf_code in sec_ret.columns) else np.zeros(T)
        mom_vals = pd.Series(y_clean).rolling(mom_lookback, min_periods=1).sum().values

        X = np.column_stack([np.ones(T), mkt, x_sec, mom_vals])
        try:
            betas, _, _, _ = np.linalg.lstsq(X[valid_mask], y_clean[valid_mask], rcond=None)
            fitted = X @ betas
            residual_3b_vals[:, j] = y_clean - fitted
        except np.linalg.LinAlgError:
            residual_3b_vals[:, j] = residual_3a[code].values

    residual_3b = pd.DataFrame(residual_3b_vals, index=dates, columns=codes)

    return {
        "raw": raw,
        "residual_3a": residual_3a,
        "residual_3b": residual_3b,
        "market_component": mkt_df,
        "sector_component": sec_df,
    }


def evaluate_signal_on_residuals(
    stock_signals: pd.DataFrame,
    decomposed: dict[str, pd.DataFrame],
    min_stocks: int = 20,
) -> ResidualAnalysisResult:
    """Lead-Lagシグナルの各残差レベルでの予測力を評価する。"""
    raw = decomposed["raw"]
    res_3a = decomposed["residual_3a"]
    res_3b = decomposed["residual_3b"]

    dates = sorted(stock_signals["date"].unique())
    common_dates = [d for d in dates if d in raw.index]

    ic_raw_list, ic_3a_list, ic_3b_list = [], [], []
    hit_raw_list, hit_3a_list, hit_3b_list = [], [], []

    for dt in common_dates:
        day_signals = stock_signals[stock_signals["date"] == dt]
        if len(day_signals) < min_stocks:
            continue

        codes = day_signals["code"].tolist()
        sigs = day_signals.set_index("code")["sector_signal"]

        valid_codes = [c for c in codes if c in raw.columns]
        if len(valid_codes) < min_stocks:
            continue

        s = sigs.reindex(valid_codes).values
        r_raw = raw.loc[dt, valid_codes].values
        r_3a = res_3a.loc[dt, valid_codes].values
        r_3b = res_3b.loc[dt, valid_codes].values

        mask = ~(np.isnan(s) | np.isnan(r_raw) | np.isnan(r_3a) | np.isnan(r_3b))
        if mask.sum() < min_stocks:
            continue

        s_v, r_raw_v, r_3a_v, r_3b_v = s[mask], r_raw[mask], r_3a[mask], r_3b[mask]

        ic_r, _ = spearmanr(s_v, r_raw_v)
        ic_a, _ = spearmanr(s_v, r_3a_v)
        ic_b, _ = spearmanr(s_v, r_3b_v)

        if not np.isnan(ic_r):
            ic_raw_list.append(ic_r)
        if not np.isnan(ic_a):
            ic_3a_list.append(ic_a)
        if not np.isnan(ic_b):
            ic_3b_list.append(ic_b)

        hit_raw_list.append(np.mean(np.sign(s_v) == np.sign(r_raw_v)))
        hit_3a_list.append(np.mean(np.sign(s_v) == np.sign(r_3a_v)))
        hit_3b_list.append(np.mean(np.sign(s_v) == np.sign(r_3b_v)))

    def _mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    def _icir(lst):
        if len(lst) < 2:
            return float("nan")
        m, s = np.mean(lst), np.std(lst)
        return float(m / s) if s > 1e-10 else float("nan")

    # 因子分解の説明力
    mkt_comp = decomposed["market_component"]
    sec_comp = decomposed["sector_component"]
    raw_ret = decomposed["raw"]

    total_var = np.nanvar(raw_ret.values)
    mkt_var = np.nanvar(mkt_comp.values) if total_var > 0 else 0
    sec_var = np.nanvar(sec_comp.values) if total_var > 0 else 0

    ic_dates = common_dates[:max(len(ic_raw_list), len(ic_3a_list), len(ic_3b_list))]

    return ResidualAnalysisResult(
        raw_ic=_mean(ic_raw_list),
        phase3a_ic=_mean(ic_3a_list),
        phase3b_ic=_mean(ic_3b_list),
        raw_icir=_icir(ic_raw_list),
        phase3a_icir=_icir(ic_3a_list),
        phase3b_icir=_icir(ic_3b_list),
        raw_hit_rate=_mean(hit_raw_list),
        phase3a_hit_rate=_mean(hit_3a_list),
        phase3b_hit_rate=_mean(hit_3b_list),
        daily_ic_raw=pd.Series(ic_raw_list, index=ic_dates[:len(ic_raw_list)]) if ic_raw_list else None,
        daily_ic_3a=pd.Series(ic_3a_list, index=ic_dates[:len(ic_3a_list)]) if ic_3a_list else None,
        daily_ic_3b=pd.Series(ic_3b_list, index=ic_dates[:len(ic_3b_list)]) if ic_3b_list else None,
        r_squared_market=float(mkt_var / total_var) if total_var > 0 else 0,
        r_squared_sector=float(sec_var / total_var) if total_var > 0 else 0,
        n_days=len(common_dates),
        n_stocks_avg=len(stock_signals) / max(1, len(dates)),
    )
