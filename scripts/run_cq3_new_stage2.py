"""cq3新 Stage 2: 信用残ストックproxyでturnover効果をどこまで説明できるか

Stage 1で確認済み: turnover p=0.026（ボラ独立の効果あり）
Stage 2の問い: 信用残比率を追加するとturnover係数はどう変わるか

設計:
  step1: Stage 1と同一モデルを全サンプルで再現確認
  step2: margin非欠損matched sampleでベースライン
  step3: matched sampleでmargin_ratioを追加
  step5_aux: サイズ層別（大型/中小型）でmodel_s1+model_s2a

注: 信用売買代金は取得不能。信用残比率（信用買残/20日平均出来高）で代理。
    解釈は限定的（ストック != フロー）。

Usage:
    python scripts/run_cq3_new_stage2.py
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import JQUANTS_API_KEY, MARKET_DATA_DIR
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --- Stage 1から流用: fetch_data, build_panel, run_regression ---
# （Stage 1のコードをそのまま使うため、同一関数をここにも定義）

def fetch_data(provider, start, end, n_stocks):
    """Stage 1と同一のデータ取得。"""
    import yfinance as yf
    from core.stock_allocation.event_study import get_jp_top_stocks
    from core.stock_allocation import _fetch_stock_prices

    print("S&P500リターンを取得中...")
    sp500 = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    sp500_close = sp500["Close"].squeeze()
    if hasattr(sp500_close.index, "tz") and sp500_close.index.tz is not None:
        sp500_close.index = sp500_close.index.tz_localize(None)
    us_returns = sp500_close.pct_change().dropna()
    print(f"  S&P500: {len(us_returns)}")

    print("JP銘柄リストを取得中...")
    listed = provider.get_listed_stocks()
    jp_codes, _ = get_jp_top_stocks(provider, n=n_stocks)
    print(f"  JP銘柄: {len(jp_codes)}")

    print("JP株価を取得中...")
    jp_prices_raw = _fetch_stock_prices(
        provider, jp_codes, start, end,
        progress_callback=lambda m, p: print(f"  {m}") if p % 0.2 < 0.01 else None,
    )
    close_col = "adj_close" if "adj_close" in jp_prices_raw.columns else "close"
    jp_close = jp_prices_raw.pivot(index="date", columns="code", values=close_col)
    jp_returns = jp_close.pct_change(fill_method=None).dropna(how="all")
    print(f"  JP株価: {len(jp_returns)} x {len(jp_returns.columns)}")

    print("TOPIXリターンを取得中...")
    topix_returns = None
    try:
        topix = provider.get_index_prices("0000", start_date=start, end_date=end)
        topix["date"] = pd.to_datetime(topix["date"])
        topix = topix.set_index("date")["close"]
        topix_returns = topix.pct_change().dropna()
    except Exception:
        pass
    if topix_returns is None or len(topix_returns) == 0:
        topix_raw = yf.download("1306.T", start=start, end=end, auto_adjust=True, progress=False)
        topix_close = topix_raw["Close"].squeeze()
        if hasattr(topix_close.index, "tz") and topix_close.index.tz is not None:
            topix_close.index = topix_close.index.tz_localize(None)
        topix_returns = topix_close.pct_change().dropna()
    print(f"  TOPIX: {len(topix_returns)}")

    return us_returns, jp_returns, topix_returns, jp_prices_raw, listed, jp_codes


def fetch_margin_data(provider, jp_codes, start, end):
    """信用残データ取得（run_cq3.pyから流用）。"""
    print("信用買残を取得中...")
    margin_frames = []
    for code in jp_codes:
        try:
            df = provider.get_margin_trading(code=code, start_date=start, end_date=end)
            if df is not None and len(df) > 0:
                if "Date" in df.columns:
                    df = df.rename(columns={"Date": "date"})
                if "Code" in df.columns:
                    df = df.rename(columns={"Code": "code"})
                for col in ["LongVol", "LongBalance", "margin_buy_balance"]:
                    if col in df.columns:
                        df["margin_buy_balance"] = df[col]
                        break
                if "margin_buy_balance" in df.columns and "date" in df.columns and "code" in df.columns:
                    margin_frames.append(df[["date", "code", "margin_buy_balance"]].copy())
        except Exception:
            continue
    if not margin_frames:
        return pd.DataFrame()
    margin_data = pd.concat(margin_frames, ignore_index=True)
    margin_data["date"] = pd.to_datetime(margin_data["date"])
    print(f"  信用買残: {margin_data['code'].nunique()}銘柄, {len(margin_data)}行")
    return margin_data


def build_panel(us_returns, jp_returns, topix_returns, jp_prices_raw, listed, event_dates):
    """Stage 1と同一のパネル構築。"""
    from core.stock_allocation.margin_event_study import (
        _compute_turnover, _compute_recent_return, _get_scale_dummies,
    )
    codes = jp_returns.columns.tolist()
    common_dates = jp_returns.index
    us_aligned = us_returns.reindex(common_dates)
    scale_dummies = _get_scale_dummies(listed, codes)
    beta_window, vol_window, turnover_window, ret_lookback = 60, 20, 20, 5

    panel_rows = []
    for event_date in event_dates:
        jp_dates_after = common_dates[common_dates > event_date]
        if len(jp_dates_after) == 0:
            continue
        jp_reaction_date = jp_dates_after[0]
        topix_ret = topix_returns.get(jp_reaction_date, np.nan)
        if np.isnan(topix_ret):
            continue
        for code in codes:
            stock_ret = jp_returns.loc[jp_reaction_date, code] if jp_reaction_date in jp_returns.index else np.nan
            if np.isnan(stock_ret):
                continue
            excess_ret = stock_ret - topix_ret
            loc = common_dates.get_loc(event_date) if event_date in common_dates else -1
            if loc < beta_window:
                continue
            stock_rets_w = jp_returns.iloc[loc - beta_window:loc][code].values
            us_rets_w = us_aligned.iloc[loc - beta_window:loc].values
            mask = np.isfinite(stock_rets_w) & np.isfinite(us_rets_w)
            if mask.sum() < beta_window // 2:
                continue
            cov = np.cov(stock_rets_w[mask], us_rets_w[mask])
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-12 else 1.0
            vol_rets = jp_returns.iloc[max(0, loc - vol_window):loc][code].values
            vol = np.nanstd(vol_rets) * np.sqrt(252) if len(vol_rets) > 5 else np.nan
            turnover = _compute_turnover(jp_prices_raw, code, event_date, turnover_window)
            ret5d = _compute_recent_return(jp_returns, code, loc, ret_lookback)
            scale = scale_dummies.get(code)
            if scale is None:
                continue
            panel_rows.append({
                "event_date": event_date, "code": code,
                "excess_ret": excess_ret, "beta": beta, "vol": vol,
                "turnover": turnover, "ret5d": ret5d,
                "scale_large70": scale.get("Large70", 0),
                "scale_mid400": scale.get("Mid400", 0),
                "scale_small1": scale.get("Small1", 0),
                "scale_small2": scale.get("Small2", 0),
            })
    return pd.DataFrame(panel_rows).dropna()


def add_margin_ratio(panel, margin_data, jp_prices_raw, turnover_window=20):
    """パネルに信用残比率を追加。margin_ratio = margin_buy_balance / 20日平均出来高。"""
    if margin_data.empty:
        panel["margin_ratio"] = np.nan
        return panel

    margin_pivot = margin_data.pivot_table(index="date", columns="code", values="margin_buy_balance")
    margin_pivot = margin_pivot.sort_index()

    ratios = []
    for _, row in panel.iterrows():
        code = row["code"]
        event_date = row["event_date"]
        # 信用残: イベント日以前の最新値
        if code not in margin_pivot.columns:
            ratios.append(np.nan)
            continue
        valid = margin_pivot.loc[:event_date, code].dropna()
        if len(valid) == 0:
            ratios.append(np.nan)
            continue
        margin_bal = valid.iloc[-1]
        # 20日平均出来高
        code_data = jp_prices_raw[jp_prices_raw["code"] == code].set_index("date").sort_index()
        vol_col = "volume" if "volume" in code_data.columns else "adj_volume"
        if vol_col not in code_data.columns:
            ratios.append(np.nan)
            continue
        recent = code_data[code_data.index <= event_date].tail(turnover_window)
        avg_vol = recent[vol_col].mean() if len(recent) > 0 else np.nan
        if avg_vol is None or np.isnan(avg_vol) or avg_vol < 1:
            ratios.append(np.nan)
            continue
        ratios.append(margin_bal / avg_vol)

    panel = panel.copy()
    panel["margin_ratio"] = ratios
    return panel


def run_regression_with_vars(panel, xvars, cluster_2way=False):
    """汎用回帰: demean -> Winsorize -> OLS(cluster SE)。
    cluster_2way=True なら 2-way cluster SE (event_date x code) を近似実装。
    """
    all_vars = ["excess_ret"] + xvars
    panel_dm = panel.copy()
    event_means = panel.groupby("event_date")[all_vars].transform("mean")
    for col in all_vars:
        panel_dm[col] = panel[col] - event_means[col]
    for col in all_vars:
        lo, hi = panel_dm[col].quantile(0.01), panel_dm[col].quantile(0.99)
        panel_dm[col] = panel_dm[col].clip(lo, hi)
    y = panel_dm["excess_ret"]
    X = panel_dm[xvars]

    ols = sm.OLS(y, X)
    model_1way = ols.fit(cov_type="cluster", cov_kwds={"groups": panel["event_date"]})

    if not cluster_2way:
        return model_1way

    # 2-way cluster SE (Cameron-Gelbach-Miller 2011 近似)
    try:
        from scipy import stats as sp_stats
        cov_event = ols.fit(cov_type="cluster", cov_kwds={"groups": panel["event_date"]}).cov_params()
        cov_code = ols.fit(cov_type="cluster", cov_kwds={"groups": panel["code"]}).cov_params()
        cov_hc0 = ols.fit(cov_type="HC0").cov_params()
        cov_2way = cov_event + cov_code - cov_hc0
        diag = np.diag(cov_2way)
        diag = np.maximum(diag, np.diag(cov_event) * 0.5)
        np.fill_diagonal(cov_2way.values if hasattr(cov_2way, 'values') else cov_2way, diag)
        se = np.sqrt(np.diag(cov_2way))
        result = ols.fit()
        result.bse = pd.Series(se, index=result.params.index)
        result.tvalues = result.params / result.bse
        df_resid = max(result.df_resid, 1)
        result.pvalues = pd.Series(
            2 * sp_stats.t.sf(np.abs(result.tvalues.values), df_resid),
            index=result.params.index
        )
        return result
    except Exception as e:
        logger.warning("2-way cluster SE failed (%s), returning 1-way", e)
        return model_1way


def compute_vif(panel, vars_to_check):
    """VIFを計算。"""
    X = sm.add_constant(panel[vars_to_check])
    vifs = {}
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vifs[col] = float(variance_inflation_factor(X.values, i))
    return vifs


def print_model(name, model, highlight=None):
    """回帰結果を表示。"""
    print(f"\n{name}: R2={model.rsquared:.4f}")
    for var in model.params.index:
        if var.startswith("scale_"):
            continue
        marker = " <--" if var == highlight else ""
        print(f"  {var:15s}: coef={model.params[var]:+.6f}  t={model.tvalues[var]:+.3f}  p={model.pvalues[var]:.4f}{marker}")


def main():
    parser = argparse.ArgumentParser(description="cq3新 Stage 2")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--n-stocks", type=int, default=200)
    args = parser.parse_args()

    cache = DataCache(MARKET_DATA_DIR)
    provider = JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)

    print("=== cq3新 Stage 2: 信用残proxyでturnover効果の説明を試みる ===")
    print(f"期間: {args.start} - {args.end}")
    print()

    # データ取得
    us_returns, jp_returns, topix_returns, jp_prices_raw, listed, jp_codes = \
        fetch_data(provider, args.start, args.end, args.n_stocks)
    margin_data = fetch_margin_data(provider, jp_codes, args.start, args.end)

    # イベント日
    us_mean = us_returns.mean()
    us_std = us_returns.std()
    threshold = us_mean - args.threshold * us_std
    event_dates = us_returns[us_returns <= threshold].index.sort_values()
    print(f"\nイベント日: {len(event_dates)}日")

    # パネル構築
    print("パネル構築中...")
    panel_full = build_panel(us_returns, jp_returns, topix_returns, jp_prices_raw, listed, event_dates)
    print(f"パネル(全): {len(panel_full)}obs ({panel_full['event_date'].nunique()}日)")

    # 信用残比率を追加
    panel_full = add_margin_ratio(panel_full, margin_data, jp_prices_raw)
    n_with_margin = panel_full["margin_ratio"].notna().sum()
    print(f"信用残比率あり: {n_with_margin}/{len(panel_full)} ({n_with_margin/len(panel_full)*100:.1f}%)")

    # matched sample（margin非欠損）
    panel_matched = panel_full.dropna(subset=["margin_ratio"]).copy()
    print(f"パネル(matched): {len(panel_matched)}obs ({panel_matched['event_date'].nunique()}日)")

    controls = ["beta", "vol", "scale_large70", "scale_mid400",
                "scale_small1", "scale_small2", "ret5d"]
    xvars_s1 = ["turnover"] + controls
    xvars_s2a = ["turnover", "margin_ratio"] + controls

    # ===== step1: Stage 1再現（全サンプル）=====
    print("\n" + "=" * 60)
    print("=== step1: Stage 1再現確認（全サンプル）===")
    print("=" * 60)
    model_s1_full = run_regression_with_vars(panel_full, xvars_s1)
    print_model("model_s1_full", model_s1_full, highlight="turnover")

    # 再現ゲート
    t_coef = float(model_s1_full.params.get("turnover", np.nan))
    v_p = float(model_s1_full.pvalues.get("vol", np.nan))
    b_p = float(model_s1_full.pvalues.get("beta", np.nan))
    if t_coef >= 0 or v_p > 0.10 or b_p < 0.10:
        print("NG: Stage 1再現失敗。中止")
        return
    print("OK: Stage 1再現")

    # ===== step2: matched sampleベースライン =====
    print("\n" + "=" * 60)
    print("=== step2: matched sampleベースライン ===")
    print("=" * 60)
    model_s1_matched = run_regression_with_vars(panel_matched, xvars_s1)
    print_model("model_s1_matched", model_s1_matched, highlight="turnover")

    t_coef_matched = float(model_s1_matched.params.get("turnover", np.nan))
    print(f"\n診断: turnover係数変化 full={t_coef:.6f} -> matched={t_coef_matched:.6f} ({t_coef_matched/t_coef*100:.0f}%)")

    # ===== step3: margin_ratio追加 =====
    print("\n" + "=" * 60)
    print("=== step3: margin_ratio追加 ===")
    print("=" * 60)

    # VIF + 判定ゲート
    vifs = compute_vif(panel_matched, ["turnover", "margin_ratio", "vol", "beta", "ret5d"])
    vif_warning = False
    vif_critical = False
    print("VIF:")
    for k, v in vifs.items():
        flag = ""
        if v > 10:
            flag = " [CRITICAL: VIF>10]"
            vif_critical = True
        elif v > 5:
            flag = " [WARNING: VIF>5]"
            vif_warning = True
        print(f"  {k:15s}: {v:.2f}{flag}")

    if vif_critical:
        print("\nNG: VIF>10の変数あり。主解釈を停止。多重共線性が深刻")
        interpretation = "VIF>10で主解釈停止。多重共線性が深刻"
        t_coef_s2a = t_p_s2a = m_coef = m_p = incr_r2 = float("nan")
    else:
        model_s2a = run_regression_with_vars(panel_matched, xvars_s2a)
        print_model("model_s2a (+ margin_ratio)", model_s2a, highlight="turnover")

        # 2-way cluster SE 補助
        model_s2a_2way = run_regression_with_vars(panel_matched, xvars_s2a, cluster_2way=True)
        print_model("model_s2a (2-way cluster SE, 補助)", model_s2a_2way, highlight="turnover")

        t_coef_s2a = float(model_s2a.params.get("turnover", np.nan))
        t_p_s2a = float(model_s2a.pvalues.get("turnover", np.nan))
        t_p_s2a_2way = float(model_s2a_2way.pvalues.get("turnover", np.nan))
        m_coef = float(model_s2a.params.get("margin_ratio", np.nan))
        m_p = float(model_s2a.pvalues.get("margin_ratio", np.nan))
        incr_r2 = model_s2a.rsquared - model_s1_matched.rsquared

        print(f"\nturnover係数変化: matched={t_coef_matched:.6f} -> s2a={t_coef_s2a:.6f} ({t_coef_s2a/t_coef_matched*100:.0f}%)")
        print(f"margin_ratio: coef={m_coef:+.6f}  p={m_p:.4f}")
        print(f"増分R2: {incr_r2:.4f}")

        # 2-way cluster SEとの矛盾チェック
        two_way_conflict = False
        if t_coef_s2a < 0 and t_p_s2a < 0.10:
            # 1-wayで有意だが2-wayで符号反転or消失
            t_coef_2way = float(model_s2a_2way.params.get("turnover", np.nan))
            if t_coef_2way >= 0 or t_p_s2a_2way >= 0.10:
                two_way_conflict = True
                print("WARNING: 2-way cluster SEでturnoverの有意性が消失/符号反転。結論を弱める")

        # 解釈
        t_survives = t_p_s2a < 0.10 and t_coef_s2a < 0
        m_significant = m_p < 0.10
        if t_survives and not m_significant:
            interpretation = "turnover残る+margin非有意: turnoverは信用残ストックでは説明できない"
        elif t_survives and m_significant:
            interpretation = "turnover残る+margin有意: 両方独立に効く。別チャネル"
        elif not t_survives and m_significant:
            interpretation = "turnover消える+margin有意: turnoverは信用残の代理変数だった可能性（解釈限定的）"
        else:
            interpretation = "両方消える: 多重共線性の問題の可能性"

        if vif_warning:
            interpretation += " [VIF>5あり: 解釈は弱い]"
        if two_way_conflict:
            interpretation += " [2-way cluster SEと矛盾: 要注意]"

    print(f"\n解釈: {interpretation}")

    # ===== step5_aux: サイズ層別 =====
    print("\n" + "=" * 60)
    print("=== 補助: サイズ層別 ===")
    print("=" * 60)

    # 大型 = scale_large70==1 or (全scaleが0 = Core30)
    # 中小型 = scale_mid400==1 or scale_small1==1 or scale_small2==1
    large_mask = (panel_matched["scale_large70"] == 1) | (
        (panel_matched["scale_large70"] == 0) & (panel_matched["scale_mid400"] == 0) &
        (panel_matched["scale_small1"] == 0) & (panel_matched["scale_small2"] == 0)
    )
    small_mask = ~large_mask

    # サイズ層別ではscaleダミーを除外（層内で定数化しランク落ちするため）
    controls_no_scale = ["beta", "vol", "ret5d"]
    xvars_s1_no_scale = ["turnover"] + controls_no_scale
    xvars_s2a_no_scale = ["turnover", "margin_ratio"] + controls_no_scale

    size_results = {}
    for label, mask in [("large", large_mask), ("small", small_mask)]:
        sub = panel_matched[mask]
        if len(sub) < 100:
            print(f"  {label}: 観測不足({len(sub)})")
            size_results[label] = {"error": f"obs={len(sub)}"}
            continue

        print(f"\n--- {label} ({len(sub)}obs, {sub['event_date'].nunique()}日) ---")
        m_s1 = run_regression_with_vars(sub, xvars_s1_no_scale)
        print_model(f"  {label} model_s1", m_s1, highlight="turnover")

        m_s2 = run_regression_with_vars(sub, xvars_s2a_no_scale)
        print_model(f"  {label} model_s2a", m_s2, highlight="turnover")

        size_results[label] = {
            "n_obs": len(sub), "n_events": int(sub["event_date"].nunique()),
            "s1_turnover_coef": float(m_s1.params.get("turnover", np.nan)),
            "s1_turnover_p": float(m_s1.pvalues.get("turnover", np.nan)),
            "s2a_turnover_coef": float(m_s2.params.get("turnover", np.nan)),
            "s2a_turnover_p": float(m_s2.pvalues.get("turnover", np.nan)),
            "s2a_margin_coef": float(m_s2.params.get("margin_ratio", np.nan)),
            "s2a_margin_p": float(m_s2.pvalues.get("margin_ratio", np.nan)),
        }

    # ===== 総合 =====
    print("\n" + "=" * 60)
    print("=== Stage 2 総合 ===")
    print("=" * 60)
    print(f"主分析: {interpretation}")
    for label, sr in size_results.items():
        if "error" in sr:
            print(f"サイズ層別 {label}: {sr['error']}")
        else:
            print(f"サイズ層別 {label}: s1 turnover p={sr['s1_turnover_p']:.4f}, s2a turnover p={sr['s2a_turnover_p']:.4f}, margin p={sr['s2a_margin_p']:.4f}")

    # JSON保存
    output = {
        "experiment": "cq3_new_stage2",
        "question": "信用残ストックproxyでturnover効果をどこまで説明できるか",
        "step1_reproduction": {
            "turnover_coef": t_coef, "vol_p": v_p, "beta_p": b_p, "pass": True,
        },
        "step2_matched_baseline": {
            "n_obs": len(panel_matched), "n_events": int(panel_matched["event_date"].nunique()),
            "turnover_coef": t_coef_matched,
            "turnover_p": float(model_s1_matched.pvalues.get("turnover", np.nan)),
        },
        "step3_margin_added": {
            "turnover_coef": t_coef_s2a, "turnover_p": t_p_s2a,
            "margin_ratio_coef": m_coef, "margin_ratio_p": m_p,
            "incremental_r2": incr_r2,
            "vifs": vifs,
        },
        "interpretation": interpretation,
        "size_stratified": size_results,
    }
    output_path = Path("logs/iterations/cq3_new_stage2_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果を保存: {output_path}")


if __name__ == "__main__":
    main()
