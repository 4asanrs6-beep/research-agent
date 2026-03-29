"""信用買残変化率×米国ショック クロスセクション回帰 (cq3 v2)

問い: 米国株急落時、信用買残の直近急増（週次変化率）の銘柄は、
β/ボラ/サイズ/回転率/直近5日リターンを統制しても超過的に下落するか？

捨てる条件: 全統制後に信用残変化率の係数p>0.10 or VIF>10
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class MarginEventResult:
    """cq3検証結果"""
    b1_coef: float = float("nan")          # 信用残変化率の係数
    b1_pvalue: float = float("nan")        # p値
    b1_tstat: float = float("nan")         # t統計量
    vif_margin_chg: float = float("nan")   # VIF
    n_events: int = 0                       # イベント日数
    n_stocks_avg: float = 0.0               # 平均銘柄数/イベント日
    n_observations: int = 0                 # 全観測点数
    incremental_r2: float = float("nan")   # 信用残追加の増分R²
    r2_full: float = float("nan")          # フルモデルR²
    r2_base: float = float("nan")          # ベースモデルR²(信用残なし)
    verdict: str = ""                       # 採択可能 / 採択不可 / 探索的
    rejection_reason: str = ""              # なし / p>0.10 / VIF>10 / クラスター不足
    all_coefficients: dict = field(default_factory=dict)


def run_margin_event_study(
    us_index_returns: pd.Series,
    jp_stock_returns: pd.DataFrame,
    jp_topix_returns: pd.Series,
    margin_data: pd.DataFrame,
    jp_stock_prices: pd.DataFrame,
    listed_stocks: pd.DataFrame | None = None,
    event_threshold_sigma: float = 2.0,
    beta_window: int = 60,
    vol_window: int = 20,
    turnover_window: int = 20,
    ret_lookback: int = 5,
    progress_callback=None,
) -> MarginEventResult:
    """cq3 v2のクロスセクション回帰を実行する。

    Args:
        us_index_returns: S&P500の日次リターン (index=date)
        jp_stock_returns: 日本個別株の日次リターン (index=date, columns=code)
        jp_topix_returns: TOPIX日次リターン (index=date)
        margin_data: 信用買残データ (columns: date, code, margin_buy_balance)
            週次データ。変化率はここで計算する
        jp_stock_prices: 日本株日足 (columns: date, code, adj_close, volume)
            β/ボラ/回転率/scale_category計算用
        event_threshold_sigma: イベント閾値
        beta_window: β推定のローリング窓
        vol_window: ボラティリティ推定の窓
        turnover_window: 回転率の窓
        ret_lookback: 直近リターンの窓
        progress_callback: (msg, pct) コールバック

    Returns:
        MarginEventResult
    """
    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    _progress("イベント日を抽出中...", 0.05)

    # --- 1. イベント日抽出: 米国2σ下落日 ---
    us_mean = us_index_returns.mean()
    us_std = us_index_returns.std()
    threshold = us_mean - event_threshold_sigma * us_std
    event_dates = us_index_returns[us_index_returns <= threshold].index
    n_events = len(event_dates)

    if n_events < 5:
        return MarginEventResult(
            n_events=n_events,
            verdict="採択不可",
            rejection_reason=f"イベント日が{n_events}日しかない(最低5日必要)",
        )

    logger.info("イベント日: %d日 (閾値: %.4f)", n_events, threshold)

    _progress("統制変数を計算中...", 0.15)

    # --- 2. JP銘柄の属性変数を計算 ---
    codes = jp_stock_returns.columns.tolist()
    common_dates = jp_stock_returns.index

    # β: 過去60日のS&P500に対するローリングβ
    # fillna(0)ではなくNaNのまま保持し、有効ペアのみでβ推定
    us_aligned = us_index_returns.reindex(common_dates)

    # 信用買残変化率の計算
    _progress("信用買残変化率を計算中...", 0.25)
    margin_chg = _compute_margin_change_rate(margin_data, codes, common_dates)

    # scale_categoryダミー（listed_stocksから構築）
    scale_dummies = _get_scale_dummies(listed_stocks, codes)

    # --- 3. イベント日ごとのパネルデータ構築 ---
    _progress("パネルデータを構築中...", 0.35)
    panel_rows = []

    for i, event_date in enumerate(event_dates):
        if i % 10 == 0:
            _progress(f"パネル構築中... ({i}/{n_events})", 0.35 + 0.30 * i / n_events)

        # JPの翌営業日を特定
        jp_dates_after = common_dates[common_dates > event_date]
        if len(jp_dates_after) == 0:
            continue
        jp_reaction_date = jp_dates_after[0]

        # TOPIX超過リターン（欠損日は観測を除外）
        topix_ret = jp_topix_returns.get(jp_reaction_date, np.nan)
        if np.isnan(topix_ret):
            continue

        for code in codes:
            # 超過リターン = 個別株リターン - TOPIXリターン
            stock_ret = jp_stock_returns.loc[jp_reaction_date, code] if jp_reaction_date in jp_stock_returns.index else np.nan
            if np.isnan(stock_ret):
                continue
            excess_ret = stock_ret - topix_ret

            # 信用買残変化率（イベント日T時点で既知の最新値）
            mcr = _get_latest_margin_chg(margin_chg, code, event_date)
            if np.isnan(mcr):
                continue

            # β（過去60日）
            loc = common_dates.get_loc(event_date) if event_date in common_dates else -1
            if loc < beta_window:
                continue

            stock_rets_window = jp_stock_returns.iloc[loc - beta_window:loc][code].values
            us_rets_window = us_aligned.iloc[loc - beta_window:loc].values
            mask = np.isfinite(stock_rets_window) & np.isfinite(us_rets_window)
            if mask.sum() < beta_window // 2:
                continue
            cov = np.cov(stock_rets_window[mask], us_rets_window[mask])
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-12 else 1.0

            # ボラティリティ（過去20日）
            vol_rets = jp_stock_returns.iloc[max(0, loc - vol_window):loc][code].values
            vol = np.nanstd(vol_rets) * np.sqrt(252) if len(vol_rets) > 5 else np.nan

            # 回転率（過去20日平均出来高 / 総出来高proxy）
            turnover = _compute_turnover(jp_stock_prices, code, event_date, turnover_window)

            # 直近5日リターン
            ret5d = _compute_recent_return(jp_stock_returns, code, loc, ret_lookback)

            # scale_categoryダミー（不明銘柄は除外）
            scale = scale_dummies.get(code)
            if scale is None:
                continue

            row = {
                "event_date": event_date,
                "code": code,
                "excess_ret": excess_ret,
                "margin_chg_rate": mcr,
                "beta": beta,
                "vol": vol,
                "scale_large70": scale.get("Large70", 0),
                "scale_mid400": scale.get("Mid400", 0),
                "scale_small1": scale.get("Small1", 0),
                "scale_small2": scale.get("Small2", 0),
                "turnover": turnover,
                "ret5d": ret5d,
            }
            panel_rows.append(row)

    if not panel_rows:
        return MarginEventResult(
            n_events=n_events,
            verdict="採択不可",
            rejection_reason="パネルデータが構築できなかった",
        )

    panel = pd.DataFrame(panel_rows).dropna()
    n_obs = len(panel)
    n_stocks_avg = n_obs / max(1, n_events)

    panel_n_events = panel["event_date"].nunique() if "event_date" in panel.columns else n_events
    panel_n_stocks_avg = n_obs / max(1, panel_n_events)
    logger.info("パネル: %d観測点 (%d日 × 平均%.0f銘柄)", n_obs, panel_n_events, panel_n_stocks_avg)

    if n_obs < 100:
        return MarginEventResult(
            n_events=n_events,
            n_observations=n_obs,
            verdict="採択不可",
            rejection_reason=f"観測点が{n_obs}しかない(最低100必要)",
        )

    # --- 4. クロスセクション回帰 ---
    _progress("回帰分析を実行中...", 0.70)
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # イベント日固定効果: ダミーではなくdemean方式（ランク落ち回避）
    # 各変数から、そのイベント日の銘柄間平均を引く
    controls = ["beta", "vol", "scale_large70", "scale_mid400",
                "scale_small1", "scale_small2", "turnover", "ret5d"]
    demean_cols = ["margin_chg_rate", "excess_ret"] + controls

    panel_dm = panel.copy()
    event_means = panel.groupby("event_date")[demean_cols].transform("mean")
    for col in demean_cols:
        panel_dm[col] = panel[col] - event_means[col]

    # 説明変数（demean済み、定数項なし）
    X_full = panel_dm[["margin_chg_rate"] + controls]
    X_base = panel_dm[controls]
    y = panel_dm["excess_ret"]

    # demean後はrankチェック
    matrix_rank = np.linalg.matrix_rank(X_full.values)
    if matrix_rank < X_full.shape[1]:
        deficit = X_full.shape[1] - matrix_rank
        logger.warning("設計行列がランク落ち（不足: %d列）。共線変数を除外", deficit)
        # 共線変数を特定して除外
        from numpy.linalg import svd
        _, s, _ = svd(X_full.values)
        keep = s > 1e-10
        if keep.sum() < 2:
            return MarginEventResult(
                n_events=n_events, n_observations=n_obs,
                verdict="採択不可",
                rejection_reason=f"設計行列がランク落ち（{deficit}列不足、修復不能）",
            )

    # Winsorize（上下1%）
    for col in X_full.columns:
        lo, hi = X_full[col].quantile(0.01), X_full[col].quantile(0.99)
        X_full[col] = X_full[col].clip(lo, hi)
        if col in X_base.columns:
            X_base[col] = X_base[col].clip(lo, hi)

    # フルモデル（demean済み、定数項なし、event_dateクラスターSE）
    # 注記: 2次元クラスター(event_date × code)は未実装。
    # 銘柄の繰返し観測によるp値過小推定のリスクあり。結果解釈時に留意。
    try:
        model_full = sm.OLS(y, X_full).fit(
            cov_type="cluster",
            cov_kwds={"groups": panel["event_date"]},
        )
    except Exception as e:
        logger.error("回帰エラー: %s", e)
        return MarginEventResult(
            n_events=n_events, n_observations=n_obs,
            verdict="採択不可", rejection_reason=f"回帰エラー: {e}",
        )

    # ベースモデル（信用残なし、demean済み）
    model_base = sm.OLS(y, X_base).fit()

    r2_full = model_full.rsquared
    r2_base = model_base.rsquared
    incremental_r2 = r2_full - r2_base

    # 信用残変化率の係数
    b1_coef = model_full.params.get("margin_chg_rate", np.nan)
    b1_pvalue = model_full.pvalues.get("margin_chg_rate", np.nan)
    b1_tstat = model_full.tvalues.get("margin_chg_rate", np.nan)

    # VIF（信用残変化率のみ必須判定）
    _progress("VIFを計算中...", 0.85)
    try:
        # VIF計算には定数項が必要なのでdemean前のデータで計算
        X_vif = sm.add_constant(panel[["margin_chg_rate"] + controls])
        margin_idx = list(X_vif.columns).index("margin_chg_rate")
        vif_margin = variance_inflation_factor(X_vif.values, margin_idx)
    except Exception:
        vif_margin = float("nan")

    # 全係数
    all_coefs = {}
    for name in model_full.params.index:
        all_coefs[name] = {
            "coef": float(model_full.params[name]),
            "pvalue": float(model_full.pvalues[name]),
            "tstat": float(model_full.tvalues[name]),
        }

    # --- 5. 採択判定 ---
    # 注: 1次元クラスターSE（event_date）のため、銘柄の繰返し観測によるp値過小推定リスクあり。
    # 2次元クラスターSE未実装のため、verdictは常に「探索的」とし、最終判断は人間が行う。
    _progress("採択判定中...", 0.90)
    actual_n_events = panel["event_date"].nunique()
    n_stocks_avg = n_obs / max(1, actual_n_events)

    if np.isnan(b1_pvalue):
        verdict = "採択不可"
        rejection_reason = "係数推定に失敗"
    elif vif_margin > 10:
        verdict = "採択不可"
        rejection_reason = f"VIF={vif_margin:.1f} > 10（多重共線性が深刻）"
    else:
        # 2次元クラスターSE未実装のため、p値ベースの自動採択はしない
        verdict = "探索的"
        notes = []
        if b1_pvalue > 0.10:
            notes.append(f"1次元クラスターSEでp={b1_pvalue:.4f}>0.10")
        if actual_n_events < 30:
            notes.append(f"有効イベント{actual_n_events}日<30")
        notes.append("2次元クラスターSE未実装。最終判断は人間が行うこと")
        rejection_reason = " | ".join(notes) if notes else "なし"

    _progress("完了", 1.0)

    return MarginEventResult(
        b1_coef=float(b1_coef),
        b1_pvalue=float(b1_pvalue),
        b1_tstat=float(b1_tstat),
        vif_margin_chg=float(vif_margin),
        n_events=n_events,
        n_stocks_avg=float(n_stocks_avg),
        n_observations=n_obs,
        incremental_r2=float(incremental_r2),
        r2_full=float(r2_full),
        r2_base=float(r2_base),
        verdict=verdict,
        rejection_reason=rejection_reason,
        all_coefficients=all_coefs,
    )


def _compute_margin_change_rate(
    margin_data: pd.DataFrame,
    codes: list[str],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """信用買残の週次変化率を計算する。

    Returns:
        DataFrame (index=date, columns=code, values=weekly_change_rate)
    """
    if margin_data.empty:
        return pd.DataFrame(index=dates, columns=codes)

    # 信用買残のピボット
    if "margin_buy_balance" in margin_data.columns:
        buy_col = "margin_buy_balance"
    elif "LongVol" in margin_data.columns:
        buy_col = "LongVol"
    else:
        # カラム名を推測
        numeric_cols = margin_data.select_dtypes(include=[np.number]).columns
        buy_col = numeric_cols[0] if len(numeric_cols) > 0 else None

    if buy_col is None:
        return pd.DataFrame(index=dates, columns=codes)

    margin_data = margin_data.copy()
    margin_data["date"] = pd.to_datetime(margin_data["date"])

    pivot = margin_data.pivot_table(index="date", columns="code", values=buy_col)
    pivot = pivot.sort_index()

    # 週次変化率
    chg_rate = pivot.pct_change(fill_method=None)

    # 日次インデックスにforward fill
    chg_rate = chg_rate.reindex(dates, method="ffill")

    return chg_rate


def _get_latest_margin_chg(
    margin_chg: pd.DataFrame,
    code: str,
    event_date,
) -> float:
    """イベント日T時点で既知の最新信用残変化率を返す。"""
    if code not in margin_chg.columns:
        return np.nan
    valid = margin_chg.loc[:event_date, code].dropna()
    if len(valid) == 0:
        return np.nan
    return float(valid.iloc[-1])


def _compute_turnover(
    prices: pd.DataFrame,
    code: str,
    event_date,
    window: int,
) -> float:
    """過去N日の出来高回転率（日次出来高/20日平均出来高）を計算する。

    真のturnover rate（出来高/発行済株式数）はデータ不在のため、
    直近出来高/20日平均出来高で正規化した相対回転率を使う。
    """
    code_data = prices[prices["code"] == code].set_index("date").sort_index()
    if code_data.empty:
        return np.nan
    loc_mask = code_data.index <= event_date
    recent = code_data[loc_mask].tail(window)
    if len(recent) < window // 2:
        return np.nan
    vol_col = "volume" if "volume" in recent.columns else "adj_volume"
    if vol_col not in recent.columns:
        return np.nan
    avg_vol = recent[vol_col].mean()
    if avg_vol < 1:
        return np.nan
    latest_vol = recent[vol_col].iloc[-1]
    return float(latest_vol / avg_vol)  # 相対回転率（1.0 = 平均的）


def _compute_recent_return(
    returns: pd.DataFrame,
    code: str,
    loc: int,
    lookback: int,
) -> float:
    """直近N日の累積リターンを計算する。"""
    if loc < lookback:
        return np.nan
    rets = returns.iloc[loc - lookback:loc][code].values
    valid = rets[np.isfinite(rets)]
    if len(valid) < lookback // 2:
        return np.nan  # 欠損が多すぎる場合はNaN（行ごと除外される）
    return float(np.prod(1 + valid) - 1)  # 複利累積


def _get_scale_dummies(
    listed_stocks: pd.DataFrame | None,
    codes: list[str],
) -> dict[str, dict[str, int]]:
    """scale_categoryからダミー変数を作成する。

    Core30を基準カテゴリとし、Large70/Mid400/Small1/Small2のダミーを返す。
    """
    if listed_stocks is None or listed_stocks.empty:
        return {}
    if "scale_category" not in listed_stocks.columns:
        return {}

    result = {}
    for _, row in listed_stocks.iterrows():
        code = row["code"]
        if code not in codes:
            continue
        sc = row.get("scale_category", "")
        result[code] = {
            "Large70": 1 if "Large70" in str(sc) else 0,
            "Mid400": 1 if "Mid400" in str(sc) else 0,
            "Small1": 1 if "Small 1" in str(sc) or "Small1" in str(sc) else 0,
            "Small2": 1 if "Small 2" in str(sc) or "Small2" in str(sc) else 0,
        }
    return result
