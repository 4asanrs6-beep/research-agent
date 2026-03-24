"""部分空間正則化PCAによる日米業種リードラグ戦略のコアアルゴリズム

Reference:
    中川ら (SIG-FIN-036, 2026)
    "部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略"
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import LeadLagConfig

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """単一戦略のバックテスト結果"""
    name: str
    daily_returns: pd.Series
    metrics: dict
    signals: pd.DataFrame | None = None  # シグナル行列 (PCA系のみ)
    daily_long_count: pd.Series | None = None  # 日次ロング数
    daily_short_count: pd.Series | None = None  # 日次ショート数


@dataclass
class BacktestResult:
    """全戦略のバックテスト結果"""
    strategies: dict[str, StrategyResult] = field(default_factory=dict)
    config: LeadLagConfig | None = None
    n_common_days: int = 0
    period_start: str = ""
    period_end: str = ""
    us_tickers: list[str] = field(default_factory=list)
    jp_tickers: list[str] = field(default_factory=list)
    eigenvalue_history: np.ndarray | None = None  # (T, K) 固有値の推移
    us_cc_returns: pd.DataFrame | None = None  # 米国 close-to-close リターン (日付×US)
    jp_overnight_gaps: pd.DataFrame | None = None  # JP overnight gap (日付×JP)
    jp_close_prices: pd.DataFrame | None = None   # JP adj_close 価格 (日付×JP)
    jp_open_prices: pd.DataFrame | None = None    # JP adj_open 価格 (日付×JP)
    benchmark_returns: pd.Series | None = None  # TOPIX 日次リターン (共通日付)
    jp_to_us_date_map: dict | None = None  # JP日付 → 実際のUS日付
    us_trading_dates: list | None = None  # 全US営業日リスト


# ---------------------------------------------------------------------------
# 事前部分空間の構築 (論文 Section 3.1)
# ---------------------------------------------------------------------------
def build_prior_subspace(
    n_us: int,
    n_jp: int,
    us_cyclical_idx: list[int],
    us_defensive_idx: list[int],
    jp_cyclical_idx: list[int],
    jp_defensive_idx: list[int],
) -> np.ndarray:
    """事前部分空間 V0 (N x K0) を修正Gram-Schmidt で構築する。

    v1: グローバルファクター — 全銘柄に等しい重み
    v2: 国スプレッド — US正, JP負 → v1 に直交化
    v3: シクリカル/ディフェンシブ — 景気敏感+, 防衛- → v1, v2 に直交化
    """
    N = n_us + n_jp

    # v1: グローバル
    v1 = np.ones(N)
    v1 /= np.linalg.norm(v1)

    # v2: 国スプレッド (US=+1, JP=-1)
    v2 = np.zeros(N)
    v2[:n_us] = 1.0
    v2[n_us:] = -1.0
    # 直交化: v2 = v2 - (v2 . v1) v1
    v2 -= np.dot(v2, v1) * v1
    v2 /= np.linalg.norm(v2)

    # v3: シクリカル/ディフェンシブ
    v3 = np.zeros(N)
    for i in us_cyclical_idx:
        v3[i] = 1.0
    for i in us_defensive_idx:
        v3[i] = -1.0
    for i in jp_cyclical_idx:
        v3[n_us + i] = 1.0
    for i in jp_defensive_idx:
        v3[n_us + i] = -1.0
    # 直交化: v3 から v1, v2 成分を除去
    v3 -= np.dot(v3, v1) * v1
    v3 -= np.dot(v3, v2) * v2
    norm_v3 = np.linalg.norm(v3)
    if norm_v3 > 1e-10:
        v3 /= norm_v3
    else:
        logger.warning("v3 のノルムがほぼ0。シクリカル/ディフェンシブラベルを確認してください。")
        v3 = np.zeros(N)

    V0 = np.column_stack([v1, v2, v3])
    return V0


def _get_sector_indices(
    us_tickers: list[str],
    jp_tickers: list[str],
    config: LeadLagConfig,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """ティッカーリストからシクリカル/ディフェンシブのインデックスを取得"""
    us_cyc_idx = [us_tickers.index(t) for t in config.us_cyclical if t in us_tickers]
    us_def_idx = [us_tickers.index(t) for t in config.us_defensive if t in us_tickers]
    jp_cyc_idx = [jp_tickers.index(t) for t in config.jp_cyclical if t in jp_tickers]
    jp_def_idx = [jp_tickers.index(t) for t in config.jp_defensive if t in jp_tickers]
    return us_cyc_idx, us_def_idx, jp_cyc_idx, jp_def_idx


# ---------------------------------------------------------------------------
# 事前相関行列 C0 の推定 (論文 Section 3.1, 式 10-12)
# ---------------------------------------------------------------------------
def estimate_prior_correlation(
    V0: np.ndarray,
    returns_prior: np.ndarray,
) -> np.ndarray:
    """長期データから事前エクスポージャー C0 を推定する。

    Args:
        V0: 事前固有ベクトル (N x K0)
        returns_prior: 長期リターン行列 (T_prior x N)

    Returns:
        C0: 事前相関行列 (N x N)
    """
    # 標準化
    mu = returns_prior.mean(axis=0)
    sigma = returns_prior.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    Z = (returns_prior - mu) / sigma

    # C_full: 長期相関行列
    C_full = np.corrcoef(Z, rowvar=False)

    # D0 = diag(V0^T C_full V0)  (式10)
    D0 = np.diag(np.diag(V0.T @ C_full @ V0))

    # C0_raw = V0 D0 V0^T  (式11)
    C0_raw = V0 @ D0 @ V0.T

    # 対角正規化して相関行列に変換 (式12)
    diag_vals = np.diag(C0_raw)
    diag_vals = np.maximum(diag_vals, 1e-12)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(diag_vals))
    C0 = D_inv_sqrt @ C0_raw @ D_inv_sqrt

    # 対角を厳密に1にする
    np.fill_diagonal(C0, 1.0)

    return C0


# ---------------------------------------------------------------------------
# ローリング PCA シグナル生成 (論文 Section 3.2-3.3, 式 13-19)
# ---------------------------------------------------------------------------
def rolling_pca_signal(
    us_returns: np.ndarray,
    jp_returns: np.ndarray,
    C0: np.ndarray,
    lambda_reg: float,
    rolling_window: int,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """ローリングウィンドウで部分空間正則化PCAシグナルを生成する。

    Args:
        us_returns: US close-to-close リターン (T x N_US)
        jp_returns: JP close-to-close リターン (T x N_JP)
        C0: 事前相関行列 (N x N)
        lambda_reg: 正則化パラメータ λ
        rolling_window: ウィンドウ長 L
        n_components: 使用する固有ベクトル数 K

    Returns:
        signals: JP側予測シグナル (T x N_JP), NaN = ウィンドウ不足
        eigenvalues: 上位K固有値の推移 (T x K)
    """
    T = us_returns.shape[0]
    n_us = us_returns.shape[1]
    n_jp = jp_returns.shape[1]
    N = n_us + n_jp
    L = rolling_window
    K = n_components

    # 結合リターン
    combined = np.hstack([us_returns, jp_returns])  # (T x N)

    signals = np.full((T, n_jp), np.nan)
    eigenvalues = np.full((T, K), np.nan)

    for t in range(L, T):
        window = combined[t - L: t]  # (L x N)

        # ウィンドウ内の標準化 (式8-9)
        mu = window.mean(axis=0)
        sigma = window.std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        Z = (window - mu) / sigma

        # ウィンドウ内相関行列 Ct
        Ct = np.corrcoef(Z, rowvar=False)

        # 正則化相関行列 (式13)
        C_reg = (1 - lambda_reg) * Ct + lambda_reg * C0

        # 固有分解 (式14-15)
        eigvals, eigvecs = np.linalg.eigh(C_reg)
        # eigh は昇順なので反転
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 上位K固有ベクトル
        V_K = eigvecs[:, :K]  # (N x K)
        eigenvalues[t] = eigvals[:K]

        # US/JP ブロックに分割 (式16)
        V_U = V_K[:n_us, :]  # (N_US x K)
        V_J = V_K[n_us:, :]  # (N_JP x K)

        # 当日のUS標準化リターン (式17)
        z_U_t = (combined[t, :n_us] - mu[:n_us]) / sigma[:n_us]

        # ファクタースコア (式18)
        f_t = V_U.T @ z_U_t  # (K,)

        # 日本側シグナル (式19)
        z_hat_J = V_J @ f_t  # (N_JP,)

        signals[t] = z_hat_J

    return signals, eigenvalues


# ---------------------------------------------------------------------------
# ポートフォリオ構築 (論文 Section 2.2, 式 3-7)
# ---------------------------------------------------------------------------
def build_portfolio(
    signals: np.ndarray,
    jp_oc_returns: np.ndarray,
    q: float,
) -> np.ndarray:
    """シグナルに基づくロング・ショートポートフォリオの日次リターンを計算する。

    Args:
        signals: 予測シグナル (T x N_JP)
        jp_oc_returns: JP open-to-close リターン (T x N_JP)
        q: 分位点割合 (0, 0.5)

    Returns:
        strategy_returns: 日次戦略リターン (T,)
    """
    T, N = signals.shape
    strategy_returns = np.full(T, np.nan)

    for t in range(T):
        sig = signals[t]
        if np.isnan(sig).all():
            continue

        # NaN でないシグナルのみ使用
        valid = ~np.isnan(sig)
        n_valid = valid.sum()
        if n_valid < 4:
            continue

        # 上位/下位 q (式3-4)
        n_long = max(1, int(np.ceil(n_valid * q)))
        n_short = max(1, int(np.ceil(n_valid * q)))

        ranked = np.argsort(sig[valid])[::-1]  # 降順
        valid_indices = np.where(valid)[0]

        long_idx = valid_indices[ranked[:n_long]]
        short_idx = valid_indices[ranked[-n_short:]]

        # 等ウェイト (式5-6)
        weights = np.zeros(N)
        weights[long_idx] = 1.0 / len(long_idx)
        weights[short_idx] = -1.0 / len(short_idx)

        # 戦略リターン: signals[t] は前日US終値で生成 → 当日JP寄引で評価
        ret = jp_oc_returns[t]
        if not np.isnan(ret).all():
            strategy_returns[t] = np.nansum(weights * ret)

    return strategy_returns


# ---------------------------------------------------------------------------
# ギャップフィルター: シグナル強度に対する寄付き織り込み率でスキップ
# ---------------------------------------------------------------------------
def build_portfolio_gap_filter(
    signals: np.ndarray,
    jp_oc_returns: np.ndarray,
    jp_overnight_gaps: np.ndarray,
    q: float,
    absorption_rate: float = 0.5,
    net_exposure_limit: float = 1.0,
    net_exposure_mode: str = "trim",
    net_exposure_skip_long: int = 0,
    net_exposure_skip_short: int = 0,
) -> np.ndarray:
    """シグナル強度に対してギャップが大きすぎる銘柄をスキップするポートフォリオ。

    スキップ条件:
      ロング候補: gap > signal * absorption_rate → 消化済み
      ショート候補: gap < signal * absorption_rate → 同上

    Returns:
        strategy_returns, entry_stats の2要素タプル
        entry_stats: {"total_candidates": int, "entered": int, "skipped": int}
    """
    T, N = signals.shape
    strategy_returns = np.full(T, np.nan)
    daily_long_count = np.full(T, np.nan)
    daily_short_count = np.full(T, np.nan)
    total_candidates = 0
    total_entered = 0

    for t in range(T):
        sig = signals[t]
        if np.isnan(sig).all():
            continue

        valid = ~np.isnan(sig)
        n_valid = valid.sum()
        if n_valid < 4:
            continue

        # シグナルの絶対値のスケール (クロスセクション標準偏差)
        sig_scale = np.nanstd(sig[valid])
        if sig_scale < 1e-10:
            sig_scale = 1.0

        n_long = max(1, int(np.ceil(n_valid * q)))
        n_short = max(1, int(np.ceil(n_valid * q)))

        ranked = np.argsort(sig[valid])[::-1]
        valid_indices = np.where(valid)[0]

        long_candidates = valid_indices[ranked[:n_long]]
        short_candidates = valid_indices[ranked[-n_short:]]

        # signals[t] は前日US終値ベース → 当日JPの寄引で評価
        gap = jp_overnight_gaps[t]
        n_candidates_day = len(long_candidates) + len(short_candidates)

        long_idx = []
        for i in long_candidates:
            if np.isnan(gap[i]) or np.isnan(sig[i]):
                long_idx.append(i)
                continue
            if abs(sig[i]) < 1e-10:
                long_idx.append(i)
                continue
            if gap[i] <= abs(sig[i]) * absorption_rate:
                long_idx.append(i)

        short_idx = []
        for i in short_candidates:
            if np.isnan(gap[i]) or np.isnan(sig[i]):
                short_idx.append(i)
                continue
            if abs(sig[i]) < 1e-10:
                short_idx.append(i)
                continue
            if gap[i] >= -abs(sig[i]) * absorption_rate:
                short_idx.append(i)

        # ネットエクスポージャー制限: L/Sの差をフィルター前ロング数×比率以内に抑える
        if net_exposure_limit < 1.0:
            max_diff = max(1, int(np.ceil(n_long * net_exposure_limit)))
            n_l, n_s = len(long_idx), len(short_idx)

            if net_exposure_mode == "fill":
                # fillモード: 少ない側にGAPフィルター除外銘柄を復活
                long_excluded = [i for i in long_candidates if i not in long_idx]
                short_excluded = [i for i in short_candidates if i not in short_idx]

                if n_l - n_s > max_diff and short_excluded:
                    short_excluded.sort(key=lambda i: abs(sig[i]), reverse=True)
                    n_to_fill = min(len(short_excluded), n_l - n_s - max_diff)
                    short_idx.extend(short_excluded[:n_to_fill])
                elif n_s - n_l > max_diff and long_excluded:
                    long_excluded.sort(key=lambda i: abs(sig[i]), reverse=True)
                    n_to_fill = min(len(long_excluded), n_s - n_l - max_diff)
                    long_idx.extend(long_excluded[:n_to_fill])
            else:
                # trimモード: 多い側を削減
                if n_l - n_s > max_diff:
                    long_sigs = [(i, abs(sig[i])) for i in long_idx]
                    long_sigs.sort(key=lambda x: x[1], reverse=True)
                    long_idx = [i for i, _ in long_sigs[:n_s + max_diff]]
                elif n_s - n_l > max_diff:
                    short_sigs = [(i, abs(sig[i])) for i in short_idx]
                    short_sigs.sort(key=lambda x: x[1], reverse=True)
                    short_idx = [i for i, _ in short_sigs[:n_l + max_diff]]

        total_candidates += n_candidates_day

        # ネットエクスポージャー絶対値スキップ
        net_val = len(long_idx) - len(short_idx)
        skip_by_ne = False
        if net_exposure_skip_long > 0 and net_val >= net_exposure_skip_long:
            skip_by_ne = True
        if net_exposure_skip_short > 0 and net_val <= -net_exposure_skip_short:
            skip_by_ne = True
        if skip_by_ne:
            daily_long_count[t] = len(long_idx)
            daily_short_count[t] = len(short_idx)
            continue

        total_entered += len(long_idx) + len(short_idx)

        daily_long_count[t] = len(long_idx)
        daily_short_count[t] = len(short_idx)

        if not long_idx and not short_idx:
            continue

        weights = np.zeros(N)
        if long_idx:
            weights[long_idx] = 1.0 / len(long_idx)
        if short_idx:
            weights[short_idx] = -1.0 / len(short_idx)

        ret = jp_oc_returns[t]
        if not np.isnan(ret).all():
            strategy_returns[t] = np.nansum(weights * ret)

    entry_rate = total_entered / total_candidates * 100 if total_candidates > 0 else 100.0
    return strategy_returns, entry_rate, daily_long_count, daily_short_count



# ---------------------------------------------------------------------------
# K=3/K=4 動的アンサンブル: 2つのK値のシグナルを加重平均
# ---------------------------------------------------------------------------
def k3k4_ensemble_signal(
    us_returns: np.ndarray,
    jp_returns: np.ndarray,
    C0: np.ndarray,
    lambda_reg: float,
    rolling_window: int,
    vix_series: np.ndarray | None = None,
    vix_low: float = 18.0,
    vix_high: float = 25.0,
    k3_weight_low_vix: float = 0.7,  # 低VIX: K=3主体 (安定)
    k3_weight_high_vix: float = 0.3,  # 高VIX: K=4主体 (クライシスアルファ)
) -> tuple[np.ndarray, np.ndarray]:
    """K=3とK=4のPCAシグナルをVIXに応じて動的に加重平均する。

    低VIX (安定相場) → K=3を重視 (月次勝率78%の安定性)
    高VIX (危機相場) → K=4を重視 (2026年型の構造変化への耐性)
    """
    sig_k3, eig_k3 = rolling_pca_signal(us_returns, jp_returns, C0, lambda_reg, rolling_window, 3)
    sig_k4, eig_k4 = rolling_pca_signal(us_returns, jp_returns, C0, lambda_reg, rolling_window, 4)

    T, n_jp = sig_k3.shape
    signals = np.full((T, n_jp), np.nan)

    for t in range(T):
        s3 = sig_k3[t]
        s4 = sig_k4[t]
        if np.isnan(s3).all() and np.isnan(s4).all():
            continue

        # VIXに基づくK=3の重み (線形補間)
        if vix_series is not None and t < len(vix_series) and not np.isnan(vix_series[t]):
            vix = vix_series[t]
            if vix <= vix_low:
                w3 = k3_weight_low_vix
            elif vix >= vix_high:
                w3 = k3_weight_high_vix
            else:
                w3 = k3_weight_low_vix + (k3_weight_high_vix - k3_weight_low_vix) * (vix - vix_low) / (vix_high - vix_low)
        else:
            w3 = 0.5  # VIXなし: 均等

        w4 = 1.0 - w3
        s3_safe = np.where(np.isnan(s3), 0.0, s3)
        s4_safe = np.where(np.isnan(s4), 0.0, s4)
        signals[t] = w3 * s3_safe + w4 * s4_safe

    return signals, eig_k3


# ---------------------------------------------------------------------------
# ローリング学習PCA: 半年ごとにC0を再推定してシグナル生成
# ---------------------------------------------------------------------------
def rolling_learning_pca_signal(
    us_returns: np.ndarray,
    jp_returns: np.ndarray,
    V0: np.ndarray,
    lambda_reg: float,
    rolling_window: int,
    n_components: int,
    dates: pd.DatetimeIndex,
    update_months: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """半年ごとにC0を再推定し、最新の市場構造を反映したPCAシグナルを生成する。

    update_months: C0を再推定する間隔 (月)。6 = 半年ごと
    各期間の先頭でその時点までの全データからC0を再推定し、
    次の更新まではそのC0を使い続ける。
    """
    T = us_returns.shape[0]
    n_us = us_returns.shape[1]
    n_jp = jp_returns.shape[1]
    N = n_us + n_jp
    L = rolling_window
    K = n_components

    combined = np.hstack([us_returns, jp_returns])
    signals = np.full((T, n_jp), np.nan)
    eigenvalues = np.full((T, K), np.nan)

    # C0更新タイミングを計算
    update_points = set()
    update_points.add(L)  # 最初
    if len(dates) > 0:
        current = dates[L]
        while current < dates[-1]:
            current = current + pd.DateOffset(months=update_months)
            # currentに最も近い日付のインデックス
            idx = dates.searchsorted(current)
            if idx < T:
                update_points.add(idx)

    current_C0 = None

    for t in range(L, T):
        # C0更新タイミング
        if t in update_points or current_C0 is None:
            # この時点までの全データでC0を再推定
            prior_data = combined[:t]
            current_C0 = estimate_prior_correlation(V0, prior_data)

        window = combined[t - L: t]
        mu = window.mean(axis=0)
        sigma = window.std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        Z = (window - mu) / sigma

        Ct = np.corrcoef(Z, rowvar=False)
        C_reg = (1 - lambda_reg) * Ct + lambda_reg * current_C0

        eigvals, eigvecs = np.linalg.eigh(C_reg)
        idx_sort = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]

        V_K = eigvecs[:, :K]
        eigenvalues[t] = eigvals[:K]

        V_U = V_K[:n_us, :]
        V_J = V_K[n_us:, :]
        z_U_t = (combined[t, :n_us] - mu[:n_us]) / sigma[:n_us]
        f_t = V_U.T @ z_U_t
        signals[t] = V_J @ f_t

    return signals, eigenvalues


# ---------------------------------------------------------------------------
# REGIME + FX統合フィルター: VIX高騰 OR 円高急伸でポジション縮小
# ---------------------------------------------------------------------------
def apply_hybrid_regime_filter(
    strategy_returns: np.ndarray,
    vix_series: np.ndarray,
    usdjpy_returns: np.ndarray,
    vix_halt: float = 35.0,
    vix_reduce: float = 28.0,
    yen_spike_threshold: float = -0.01,
    vix_reduce_factor: float = 0.5,
    fx_reduce_factor: float = 0.3,
) -> np.ndarray:
    """VIX高騰 OR 円高急伸のいずれかでポジション縮小するハイブリッドフィルター。"""
    T = len(strategy_returns)
    filtered = np.copy(strategy_returns)

    for t in range(T):
        if np.isnan(filtered[t]):
            continue

        scale = 1.0

        # VIXフィルター
        vix = vix_series[t] if t < len(vix_series) and not np.isnan(vix_series[t]) else 20.0
        if vix >= vix_halt:
            scale = 0.0
        elif vix >= vix_reduce:
            scale = min(scale, vix_reduce_factor)

        # FXフィルター
        fx_ret = usdjpy_returns[t] if t < len(usdjpy_returns) and not np.isnan(usdjpy_returns[t]) else 0.0
        if fx_ret <= yen_spike_threshold:
            scale = min(scale, fx_reduce_factor)

        filtered[t] *= scale

    return filtered


# ---------------------------------------------------------------------------
# レジーム検出: VIXベースのポジションスケーリング
# ---------------------------------------------------------------------------
def apply_regime_scaling(
    strategy_returns: np.ndarray,
    vix_series: np.ndarray,
    vix_halt_threshold: float = 35.0,
    vix_reduce_threshold: float = 28.0,
    reduce_factor: float = 0.5,
) -> np.ndarray:
    """VIXが高い (リードラグ崩壊期) にポジションを縮小/停止する。

    VIX >= halt_threshold → ポジション0 (戦略停止)
    VIX >= reduce_threshold → ポジション50%に縮小
    VIX < reduce_threshold → フルポジション
    """
    T = len(strategy_returns)
    scaled = np.copy(strategy_returns)
    for t in range(T):
        if np.isnan(scaled[t]):
            continue
        vix = vix_series[t] if t < len(vix_series) and not np.isnan(vix_series[t]) else 20.0
        if vix >= vix_halt_threshold:
            scaled[t] = 0.0
        elif vix >= vix_reduce_threshold:
            scaled[t] *= reduce_factor
    return scaled


# ---------------------------------------------------------------------------
# モメンタム戦略 (論文 Section 4.3, 式 31)
# ---------------------------------------------------------------------------
def momentum_strategy(
    jp_cc_returns: np.ndarray,
    jp_oc_returns: np.ndarray,
    rolling_window: int,
    q: float,
) -> np.ndarray:
    """JP側ウィンドウ内平均リターン (単純モメンタム) によるL/S戦略。"""
    T, N = jp_cc_returns.shape
    strategy_returns = np.full(T, np.nan)
    L = rolling_window

    for t in range(L, T - 1):
        # ウィンドウ内平均 (式31)
        mom = np.nanmean(jp_cc_returns[t - L: t], axis=0)

        valid = ~np.isnan(mom)
        n_valid = valid.sum()
        if n_valid < 4:
            continue

        n_long = max(1, int(np.ceil(n_valid * q)))
        n_short = max(1, int(np.ceil(n_valid * q)))

        ranked = np.argsort(mom[valid])[::-1]
        valid_indices = np.where(valid)[0]

        long_idx = valid_indices[ranked[:n_long]]
        short_idx = valid_indices[ranked[-n_short:]]

        weights = np.zeros(N)
        weights[long_idx] = 1.0 / len(long_idx)
        weights[short_idx] = -1.0 / len(short_idx)

        ret = jp_oc_returns[t + 1]
        if not np.isnan(ret).all():
            strategy_returns[t + 1] = np.nansum(weights * ret)

    return strategy_returns


# ---------------------------------------------------------------------------
# ダブルソート戦略 (論文 Section 4.3)
# ---------------------------------------------------------------------------
def double_sort_strategy(
    mom_scores: np.ndarray,
    pca_scores: np.ndarray,
    jp_oc_returns: np.ndarray,
    jp_cc_returns: np.ndarray,
    rolling_window: int,
) -> np.ndarray:
    """MOM × PCA_SUB の 2×2 ダブルソート戦略。

    各シグナルをメディアンで High/Low に分割し、
    High×High をロング、Low×Low をショート。
    """
    T, N = jp_oc_returns.shape
    strategy_returns = np.full(T, np.nan)
    L = rolling_window

    for t in range(L, T - 1):
        sig_pca = pca_scores[t]
        if np.isnan(sig_pca).all():
            continue

        mom = np.nanmean(jp_cc_returns[t - L: t], axis=0)

        valid = ~np.isnan(sig_pca) & ~np.isnan(mom)
        n_valid = valid.sum()
        if n_valid < 4:
            continue

        # メディアンで分割
        med_pca = np.nanmedian(sig_pca[valid])
        med_mom = np.nanmedian(mom[valid])

        high_pca = sig_pca >= med_pca
        high_mom = mom >= med_mom
        low_pca = sig_pca < med_pca
        low_mom = mom < med_mom

        long_mask = valid & high_pca & high_mom  # High × High
        short_mask = valid & low_pca & low_mom  # Low × Low

        n_long = long_mask.sum()
        n_short = short_mask.sum()
        if n_long == 0 or n_short == 0:
            continue

        weights = np.zeros(N)
        weights[long_mask] = 1.0 / n_long
        weights[short_mask] = -1.0 / n_short

        ret = jp_oc_returns[t + 1]
        if not np.isnan(ret).all():
            strategy_returns[t + 1] = np.nansum(weights * ret)

    return strategy_returns


# ---------------------------------------------------------------------------
# パフォーマンス指標 (論文 Section 4.2, 式 27-30)
# ---------------------------------------------------------------------------
def compute_metrics(daily_returns: np.ndarray, dates: pd.DatetimeIndex | None = None, entry_rate: float | None = None) -> dict:
    """AR, RISK, R/R, MDD + 月次安定性指標を計算する。"""
    total_days = len(daily_returns)
    valid = daily_returns[~np.isnan(daily_returns)]
    trade_ratio = len(valid) / total_days * 100 if total_days > 0 else 0
    if len(valid) == 0:
        return {
            "AR": 0, "RISK": 0, "R/R": 0, "MDD": 0, "n_days": 0,
            "trade_ratio": 0, "entry_rate": 0,
            "monthly_win_rate": 0, "monthly_mean": 0, "monthly_std": 0,
            "monthly_worst": 0, "monthly_best": 0, "negative_months": 0,
            "total_months": 0,
        }

    # 年率換算 (252営業日)
    ar = valid.mean() * 252
    risk = valid.std() * np.sqrt(252)
    rr = ar / risk if risk > 1e-12 else 0.0

    # MDD (式30)
    cumulative = np.cumprod(1 + valid)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    mdd = drawdowns.min()

    result = {
        "AR": ar * 100,
        "RISK": risk * 100,
        "R/R": rr,
        "MDD": mdd * 100,
        "n_days": len(valid),
        "trade_ratio": trade_ratio,
        "entry_rate": entry_rate if entry_rate is not None else 100.0,
    }

    # --- 月次安定性指標 ---
    if dates is not None:
        valid_dates = dates[~np.isnan(daily_returns)]
        if len(valid_dates) > 0:
            s = pd.Series(valid, index=valid_dates)
            monthly = s.groupby([s.index.year, s.index.month]).apply(
                lambda x: ((1 + x).prod() - 1) * 100
            )
            if len(monthly) > 0:
                result["monthly_win_rate"] = (monthly > 0).mean() * 100
                result["monthly_mean"] = monthly.mean()
                result["monthly_std"] = monthly.std()
                result["monthly_worst"] = monthly.min()
                result["monthly_best"] = monthly.max()
                result["negative_months"] = int((monthly <= 0).sum())
                result["total_months"] = len(monthly)
            else:
                result.update({
                    "monthly_win_rate": 0, "monthly_mean": 0, "monthly_std": 0,
                    "monthly_worst": 0, "monthly_best": 0, "negative_months": 0,
                    "total_months": 0,
                })
    else:
        result.update({
            "monthly_win_rate": 0, "monthly_mean": 0, "monthly_std": 0,
            "monthly_worst": 0, "monthly_best": 0, "negative_months": 0,
            "total_months": 0,
        })

    return result


# ---------------------------------------------------------------------------
# 全戦略の一括実行
# ---------------------------------------------------------------------------
def run_all_strategies(
    us_cc: np.ndarray,
    jp_cc: np.ndarray,
    jp_oc: np.ndarray,
    us_tickers: list[str],
    jp_tickers: list[str],
    dates: pd.DatetimeIndex,
    config: LeadLagConfig,
    progress_callback=None,
    jp_overnight_gaps: np.ndarray | None = None,
) -> BacktestResult:
    """全戦略を実行して比較結果を返す。"""
    result = BacktestResult(
        config=config,
        n_common_days=len(dates),
        period_start=dates[0].strftime("%Y-%m-%d") if len(dates) > 0 else "",
        period_end=dates[-1].strftime("%Y-%m-%d") if len(dates) > 0 else "",
        us_tickers=us_tickers,
        jp_tickers=jp_tickers,
        us_cc_returns=pd.DataFrame(us_cc, index=dates, columns=us_tickers),
        jp_overnight_gaps=pd.DataFrame(jp_overnight_gaps, index=dates, columns=jp_tickers) if jp_overnight_gaps is not None else None,
    )

    n_us = us_cc.shape[1]
    n_jp = jp_cc.shape[1]

    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    # --- 事前部分空間の構築 ---
    _progress("事前部分空間を構築中...", 0.1)
    us_cyc, us_def, jp_cyc, jp_def = _get_sector_indices(us_tickers, jp_tickers, config)
    V0 = build_prior_subspace(n_us, n_jp, us_cyc, us_def, jp_cyc, jp_def)

    # --- 事前相関行列 C0 の推定 ---
    _progress("事前相関行列を推定中...", 0.15)
    prior_end = pd.Timestamp(config.prior_end_date)
    prior_mask = dates <= prior_end
    if prior_mask.sum() < config.rolling_window:
        logger.warning(
            "事前期間のデータが不足 (%d日 < L=%d)。全データで C_full を推定します。",
            prior_mask.sum(), config.rolling_window,
        )
        prior_mask = np.ones(len(dates), dtype=bool)

    combined_prior = np.hstack([us_cc[prior_mask], jp_cc[prior_mask]])
    C0 = estimate_prior_correlation(V0, combined_prior)

    # --- PCA_SUB (提案手法) ---
    if config.run_pca_sub:
        _progress("PCA_SUB (提案手法) シグナル生成中...", 0.2)
        signals_sub, eigvals = rolling_pca_signal(
            us_cc, jp_cc, C0,
            lambda_reg=config.lambda_reg,
            rolling_window=config.rolling_window,
            n_components=config.n_components,
        )
        result.eigenvalue_history = eigvals

        _progress("PCA_SUB ポートフォリオ構築中...", 0.35)
        ret_sub = build_portfolio(signals_sub, jp_oc, config.quantile_q)
        result.strategies["PCA_SUB"] = StrategyResult(
            name="PCA_SUB",
            daily_returns=pd.Series(ret_sub, index=dates, name="PCA_SUB"),
            metrics=compute_metrics(ret_sub, dates),
            signals=pd.DataFrame(signals_sub, index=dates, columns=jp_tickers),
        )

        # カスタムギャップフィルター (手動設定時)
        if config.gap_threshold < 1.0 and jp_overnight_gaps is not None:
            _progress(f"ギャップフィルター ({int(config.gap_threshold*100)}%) 適用中...", 0.38)
            ret_gap_custom, entry_custom, dl_custom, ds_custom = build_portfolio_gap_filter(
                signals_sub, jp_oc, jp_overnight_gaps, config.quantile_q,
                absorption_rate=config.gap_threshold,
                net_exposure_limit=config.net_exposure_limit,
                net_exposure_mode=config.net_exposure_mode,
                net_exposure_skip_long=config.net_exposure_skip_long,
                net_exposure_skip_short=config.net_exposure_skip_short,
            )
            result.strategies["GAP_CUSTOM"] = StrategyResult(
                name="GAP_CUSTOM",
                daily_returns=pd.Series(ret_gap_custom, index=dates, name="GAP_CUSTOM"),
                metrics=compute_metrics(ret_gap_custom, dates, entry_rate=entry_custom),
                signals=pd.DataFrame(signals_sub, index=dates, columns=jp_tickers),
                daily_long_count=pd.Series(dl_custom, index=dates, name="long_count"),
                daily_short_count=pd.Series(ds_custom, index=dates, name="short_count"),
            )

    # --- PCA_PLAIN (正則化なし) ---
    if config.run_pca_plain:
        _progress("PCA_PLAIN (正則化なし) シグナル生成中...", 0.4)
        signals_plain, _ = rolling_pca_signal(
            us_cc, jp_cc, C0,
            lambda_reg=0.0,  # 正則化なし
            rolling_window=config.rolling_window,
            n_components=config.n_components,
        )
        ret_plain = build_portfolio(signals_plain, jp_oc, config.quantile_q)
        result.strategies["PCA_PLAIN"] = StrategyResult(
            name="PCA_PLAIN",
            daily_returns=pd.Series(ret_plain, index=dates, name="PCA_PLAIN"),
            metrics=compute_metrics(ret_plain, dates),
            signals=pd.DataFrame(signals_plain, index=dates, columns=jp_tickers),
        )

    # --- MOM (モメンタム) ---
    if config.run_mom:
        _progress("MOM (モメンタム) 生成中...", 0.6)
        ret_mom = momentum_strategy(jp_cc, jp_oc, config.rolling_window, config.quantile_q)
        result.strategies["MOM"] = StrategyResult(
            name="MOM",
            daily_returns=pd.Series(ret_mom, index=dates, name="MOM"),
            metrics=compute_metrics(ret_mom, dates),
        )

    # --- DOUBLE (ダブルソート) ---
    if config.run_double and config.run_pca_sub:
        _progress("DOUBLE (ダブルソート) 生成中...", 0.75)
        ret_double = double_sort_strategy(
            mom_scores=None,  # 内部で計算
            pca_scores=signals_sub,
            jp_oc_returns=jp_oc,
            jp_cc_returns=jp_cc,
            rolling_window=config.rolling_window,
        )
        result.strategies["DOUBLE"] = StrategyResult(
            name="DOUBLE",
            daily_returns=pd.Series(ret_double, index=dates, name="DOUBLE"),
            metrics=compute_metrics(ret_double, dates),
        )

    # === 拡張戦略 (効果が確認されたもののみ) ===

    # --- VIX + ドル円データの一括取得 ---
    vix_data = None
    usdjpy_rets = None
    if config.run_pca_sub:
        try:
            import yfinance as yf
            _progress("VIX/ドル円データ取得中...", 0.80)
            for ticker, key in [("^VIX", "vix"), ("JPY=X", "fx")]:
                raw = yf.download(ticker, start=dates[0].strftime("%Y-%m-%d"),
                                 end=(dates[-1] + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                                 auto_adjust=True, progress=False)
                if not raw.empty:
                    close = raw["Close"].squeeze()
                    if hasattr(close.index, 'tz') and close.index.tz is not None:
                        close.index = close.index.tz_localize(None)
                    if key == "vix":
                        vals = close.values
                    else:
                        vals = close.pct_change().values
                    df_tmp = pd.DataFrame({key: vals}, index=close.index)
                    dates_df = pd.DataFrame({"idx": range(len(dates))}, index=dates)
                    merged = pd.merge_asof(dates_df.sort_index(), df_tmp.sort_index(),
                                           left_index=True, right_index=True, direction="backward")
                    if key == "vix":
                        vix_data = merged["vix"].values
                    else:
                        usdjpy_rets = merged["fx"].values
        except Exception as e:
            logger.warning("VIX/ドル円データ取得失敗: %s", e)

    # --- HYBRID (VIX + 円高 統合フィルター) ---
    if config.run_pca_sub and vix_data is not None and usdjpy_rets is not None:
        _progress("HYBRID (VIX+FX統合) フィルター適用中...", 0.85)
        ret_hybrid = apply_hybrid_regime_filter(ret_sub, vix_data, usdjpy_rets)
        result.strategies["HYBRID"] = StrategyResult(
            name="HYBRID",
            daily_returns=pd.Series(ret_hybrid, index=dates, name="HYBRID"),
            metrics=compute_metrics(ret_hybrid, dates),
        )

    # --- K3K4_ENS (K=3/K=4 動的アンサンブル) ---
    if config.run_pca_sub:
        _progress("K3/K4 アンサンブル シグナル生成中...", 0.88)
        k3k4_sig, k3k4_eig = k3k4_ensemble_signal(
            us_cc, jp_cc, C0,
            lambda_reg=config.lambda_reg,
            rolling_window=config.rolling_window,
            vix_series=vix_data,
        )
        ret_k3k4 = build_portfolio(k3k4_sig, jp_oc, config.quantile_q)
        result.strategies["K3K4_ENS"] = StrategyResult(
            name="K3K4_ENS",
            daily_returns=pd.Series(ret_k3k4, index=dates, name="K3K4_ENS"),
            metrics=compute_metrics(ret_k3k4, dates),
            signals=pd.DataFrame(k3k4_sig, index=dates, columns=jp_tickers),
        )

    # --- ギャップフィルター感応度テスト (NE制限なし = 純粋なGAPフィルターのみ) ---
    if config.run_pca_sub and jp_overnight_gaps is not None:
        for rate, label in [(-0.1, "GAP_-10"), (0.0, "GAP_0"), (0.05, "GAP_5"), (0.1, "GAP_10"), (0.2, "GAP_20"), (0.3, "GAP_30")]:
            ret_gap, gap_entry_rate, dl_gap, ds_gap = build_portfolio_gap_filter(
                signals_sub, jp_oc, jp_overnight_gaps, config.quantile_q,
                absorption_rate=rate,
            )
            result.strategies[label] = StrategyResult(
                name=label,
                daily_returns=pd.Series(ret_gap, index=dates, name=label),
                metrics=compute_metrics(ret_gap, dates, entry_rate=gap_entry_rate),
                daily_long_count=pd.Series(dl_gap, index=dates, name="long_count"),
                daily_short_count=pd.Series(ds_gap, index=dates, name="short_count"),
            )

        # K3K4_ENSにも同じGAPフィルター＋NE制限を適用
        if "K3K4_ENS" in result.strategies and config.gap_threshold < 1.0:
            _progress("K3K4+ギャップフィルター適用中...", 0.98)
            ret_k3k4_gap, k3k4_gap_entry, dl_k3k4, ds_k3k4 = build_portfolio_gap_filter(
                k3k4_sig, jp_oc, jp_overnight_gaps, config.quantile_q,
                absorption_rate=config.gap_threshold,
                net_exposure_limit=config.net_exposure_limit,
                net_exposure_mode=config.net_exposure_mode,
                net_exposure_skip_long=config.net_exposure_skip_long,
                net_exposure_skip_short=config.net_exposure_skip_short,
            )
            result.strategies["K3K4_GAP"] = StrategyResult(
                name="K3K4_GAP",
                daily_returns=pd.Series(ret_k3k4_gap, index=dates, name="K3K4_GAP"),
                metrics=compute_metrics(ret_k3k4_gap, dates, entry_rate=k3k4_gap_entry),
                daily_long_count=pd.Series(dl_k3k4, index=dates, name="long_count"),
                daily_short_count=pd.Series(ds_k3k4, index=dates, name="short_count"),
            )

    _progress("完了", 1.0)
    return result
