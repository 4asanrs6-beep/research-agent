"""T3-Q04: 2月の勝者=3月の敗者? モメンタム反転 vs 原油波及の分解

設計v2準拠:
- Step 1: セクターレベル散布図(記述統計)
- Step 2: 各銘柄の原油β/TOPIXβ/USDJPYβ推定(2024-01〜2026-01)
- Step 3: 銘柄レベル回帰(3月超過ret = 2月ret + 原油β + TOPIXβ + USDJPYβ + セクターFE)
- Step 4: centered fitted value分解
- Step 5: 週次分解
"""

import json
import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from datetime import date

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import JQUANTS_API_KEY, MARKET_DATA_DIR
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider

import yfinance as yf


def main():
    cache = DataCache(MARKET_DATA_DIR)
    provider = JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)

    print("=== T3-Q04: モメンタム反転 vs 原油波及の分解 ===\n")

    # --- データ取得 ---
    print("データ取得中...")
    listed = provider.get_listed_stocks()
    jp_codes = listed["code"].tolist()[:200]  # 上位200銘柄（簡易的に先頭200）
    n_stocks = len(jp_codes)
    print(f"  銘柄数: {n_stocks}")

    # セクター分類
    sector_map = {}
    if "sector_17" in listed.columns:
        for _, row in listed.iterrows():
            if row["code"] in jp_codes:
                sector_map[row["code"]] = row.get("sector_17", "unknown")

    # 日本株日次リターン(2024-01〜2026-03)
    from core.stock_allocation import _fetch_stock_prices
    print("  JP株価取得中...")
    jp_prices = _fetch_stock_prices(provider, jp_codes, "2024-01-01", date.today().strftime("%Y-%m-%d"), cache=cache)
    close_col = "adj_close" if "adj_close" in jp_prices.columns else "close"
    jp_close = jp_prices.pivot(index="date", columns="code", values=close_col)
    jp_returns = jp_close.pct_change(fill_method=None).dropna(how="all")
    print(f"  JP日次リターン: {jp_returns.shape}")

    # WTI原油
    print("  WTI原油取得中...")
    oil = yf.download("CL=F", start="2024-01-01", end=date.today().strftime("%Y-%m-%d"), progress=False)
    oil_ret = oil["Close"].squeeze().pct_change().dropna()
    oil_ret.index = oil_ret.index.tz_localize(None) if hasattr(oil_ret.index, 'tz') and oil_ret.index.tz else oil_ret.index
    print(f"  WTI日次リターン: {len(oil_ret)}")

    # TOPIX(1306.Tまたはyfinance)
    print("  TOPIX取得中...")
    try:
        topix_data = provider.get_topix_daily("2024-01-01", date.today().strftime("%Y-%m-%d"))
        topix_ret = topix_data.set_index("date")["close"].pct_change().dropna()
    except Exception:
        topix_raw = yf.download("^TPX", start="2024-01-01", end=date.today().strftime("%Y-%m-%d"), progress=False)
        if len(topix_raw) == 0:
            topix_raw = yf.download("1306.T", start="2024-01-01", end=date.today().strftime("%Y-%m-%d"), progress=False)
        topix_ret = topix_raw["Close"].squeeze().pct_change().dropna()
        topix_ret.index = topix_ret.index.tz_localize(None) if hasattr(topix_ret.index, 'tz') and topix_ret.index.tz else topix_ret.index
    print(f"  TOPIX日次リターン: {len(topix_ret)}")

    # USDJPY
    print("  USDJPY取得中...")
    usdjpy = yf.download("JPY=X", start="2024-01-01", end=date.today().strftime("%Y-%m-%d"), progress=False)
    usdjpy_ret = usdjpy["Close"].squeeze().pct_change().dropna()
    usdjpy_ret.index = usdjpy_ret.index.tz_localize(None) if hasattr(usdjpy_ret.index, 'tz') and usdjpy_ret.index.tz else usdjpy_ret.index
    print(f"  USDJPY日次リターン: {len(usdjpy_ret)}")

    # --- 月次リターン計算 ---
    print("\n月次リターン計算中...")
    feb_start, feb_end = pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-28")
    mar_start, mar_end = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-31")

    feb_rets = jp_returns[(jp_returns.index >= feb_start) & (jp_returns.index <= feb_end)]
    mar_rets = jp_returns[(jp_returns.index >= mar_start) & (jp_returns.index <= mar_end)]
    topix_mar = topix_ret[(topix_ret.index >= mar_start) & (topix_ret.index <= mar_end)]

    feb_monthly = (1 + feb_rets).prod() - 1
    mar_monthly = (1 + mar_rets).prod() - 1
    topix_mar_monthly = float((1 + topix_mar).prod() - 1)

    # 3月超過リターン
    mar_excess = mar_monthly - topix_mar_monthly

    # 欠損フィルタ(80%基準)
    feb_valid = feb_rets.count() >= len(feb_rets) * 0.8
    mar_valid = mar_rets.count() >= len(mar_rets) * 0.8
    valid_codes = [c for c in jp_codes if feb_valid.get(c, False) and mar_valid.get(c, False)
                   and not np.isnan(feb_monthly.get(c, np.nan)) and not np.isnan(mar_excess.get(c, np.nan))]
    print(f"  有効銘柄数: {len(valid_codes)}/{n_stocks}")

    # --- Step 1: セクターレベル記述統計 ---
    print("\n--- Step 1: セクターレベル散布図(記述統計) ---")
    sector_rets = {}
    for code in valid_codes:
        sec = sector_map.get(code, "unknown")
        if sec not in sector_rets:
            sector_rets[sec] = {"feb": [], "mar": []}
        sector_rets[sec]["feb"].append(float(feb_monthly[code]))
        sector_rets[sec]["mar"].append(float(mar_excess[code]))

    sector_summary = []
    for sec, vals in sector_rets.items():
        sector_summary.append({
            "sector": sec,
            "feb_mean": float(np.mean(vals["feb"])),
            "mar_mean": float(np.mean(vals["mar"])),
            "n": len(vals["feb"]),
        })
    sector_df = pd.DataFrame(sector_summary)
    if len(sector_df) >= 5:
        from scipy import stats as scipy_stats
        r, _ = scipy_stats.pearsonr(sector_df["feb_mean"], sector_df["mar_mean"])
        print(f"  セクター数: {len(sector_df)}")
        print(f"  2月vs3月超過リターンの相関(記述): r={r:.3f} (n={len(sector_df)}, 検定省略)")
        for _, row in sector_df.sort_values("feb_mean", ascending=False).iterrows():
            print(f"    {row['sector']:>10s}: 2月{row['feb_mean']*100:+.1f}% → 3月超過{row['mar_mean']*100:+.1f}% (n={row['n']})")
    else:
        r = np.nan
        print("  セクターデータ不足")

    # --- Step 2: β推定 ---
    print("\n--- Step 2: 各銘柄のβ推定(2024-01〜2026-01) ---")
    beta_end = pd.Timestamp("2026-01-31")
    est_rets = jp_returns[jp_returns.index <= beta_end]

    betas = {}
    for code in valid_codes:
        stock_r = est_rets[code].dropna() if code in est_rets.columns else pd.Series(dtype=float)
        common = stock_r.index.intersection(oil_ret.index).intersection(topix_ret.index).intersection(usdjpy_ret.index)
        if len(common) < 100:
            continue
        y = stock_r.reindex(common)
        X = pd.DataFrame({
            "oil": oil_ret.reindex(common),
            "topix": topix_ret.reindex(common),
            "usdjpy": usdjpy_ret.reindex(common),
        }).dropna()
        common2 = y.index.intersection(X.index)
        if len(common2) < 100:
            continue
        y = y.reindex(common2)
        X = X.reindex(common2)
        try:
            model = sm.OLS(y, sm.add_constant(X)).fit()
            betas[code] = {
                "oil_beta": float(model.params.get("oil", np.nan)),
                "topix_beta": float(model.params.get("topix", np.nan)),
                "usdjpy_beta": float(model.params.get("usdjpy", np.nan)),
            }
        except Exception:
            continue

    print(f"  β推定成功: {len(betas)}/{len(valid_codes)}")
    final_codes = [c for c in valid_codes if c in betas]

    # --- Step 3: 銘柄レベル回帰 ---
    print("\n--- Step 3: 銘柄レベル回帰 ---")
    reg_data = []
    for code in final_codes:
        sec = sector_map.get(code, "unknown")
        reg_data.append({
            "code": code,
            "mar_excess": float(mar_excess[code]),
            "feb_ret": float(feb_monthly[code]),
            "oil_beta": betas[code]["oil_beta"],
            "topix_beta": betas[code]["topix_beta"],
            "usdjpy_beta": betas[code]["usdjpy_beta"],
            "sector": sec,
        })
    reg_df = pd.DataFrame(reg_data)
    print(f"  回帰サンプル: {len(reg_df)}銘柄")

    # セクターダミー
    sector_dummies = pd.get_dummies(reg_df["sector"], prefix="sec", drop_first=True).astype(float)
    xvars = ["feb_ret", "oil_beta", "topix_beta", "usdjpy_beta"]
    X = pd.concat([reg_df[xvars].astype(float).reset_index(drop=True), sector_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = reg_df["mar_excess"].astype(float).reset_index(drop=True)

    # VIF
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = reg_df[xvars].dropna()
    vif_X = sm.add_constant(vif_data)
    vifs = {}
    for i, col in enumerate(vif_X.columns):
        if col == "const":
            continue
        vifs[col] = float(variance_inflation_factor(vif_X.values, i))
    print(f"  VIF: {vifs}")

    # OLS with sector-cluster SE
    try:
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": reg_df["sector"].reset_index(drop=True)})
        print(f"\n  R2 = {model.rsquared:.4f}")
        for var in xvars:
            coef = float(model.params.get(var, np.nan))
            p = float(model.pvalues.get(var, np.nan))
            print(f"  {var:15s}: coef={coef:+.6f}  p={p:.4f}")

        # HC1版(補助)
        model_hc1 = sm.OLS(y, X).fit(cov_type="HC1")
        print(f"\n  (HC1補助)")
        for var in xvars:
            print(f"  {var:15s}: p_HC1={float(model_hc1.pvalues.get(var, np.nan)):.4f}")
    except Exception as e:
        print(f"  回帰エラー: {e}")
        model = sm.OLS(y, X).fit(cov_type="HC1")
        print(f"  (HC1にフォールバック) R2={model.rsquared:.4f}")
        for var in xvars:
            coef = float(model.params.get(var, np.nan))
            p = float(model.pvalues.get(var, np.nan))
            print(f"  {var:15s}: coef={coef:+.6f}  p={p:.4f}")

    # 単変量部分R2
    print(f"\n  単変量R2:")
    partial_r2 = {}
    for var in xvars:
        m_single = sm.OLS(y, sm.add_constant(reg_df[[var]])).fit()
        partial_r2[var] = float(m_single.rsquared)
        print(f"    {var}: R2={m_single.rsquared:.4f}")

    # --- Step 4: 寄与度分解 ---
    print("\n--- Step 4: centered fitted value分解 ---")
    centered = {}
    contributions = {}
    for var in xvars:
        centered[var] = reg_df[var] - reg_df[var].mean()
        contributions[var] = float(model.params.get(var, 0)) * centered[var]

    avg_contributions = {var: float(contributions[var].mean()) for var in xvars}
    total_excess = float(y.mean())
    print(f"  平均3月超過リターン: {total_excess*100:+.3f}%")
    for var in xvars:
        pct = avg_contributions[var] / total_excess * 100 if abs(total_excess) > 1e-8 else 0
        print(f"  {var:15s}: 寄与={avg_contributions[var]*100:+.4f}% ({pct:+.1f}%)")

    # --- Step 5: 週次分解 ---
    print("\n--- Step 5: 週次分解 ---")
    weeks = [
        ("W10", "2026-03-02", "2026-03-06"),
        ("W11", "2026-03-09", "2026-03-13"),
        ("W12", "2026-03-16", "2026-03-20"),
        ("W13", "2026-03-23", "2026-03-27"),
    ]
    weekly_results = []
    for wlabel, ws, we in weeks:
        w_rets = jp_returns[(jp_returns.index >= pd.Timestamp(ws)) & (jp_returns.index <= pd.Timestamp(we))]
        topix_w = topix_ret[(topix_ret.index >= pd.Timestamp(ws)) & (topix_ret.index <= pd.Timestamp(we))]
        if len(w_rets) < 3:
            print(f"  {wlabel}: 営業日不足({len(w_rets)}日)。スキップ")
            weekly_results.append({"week": wlabel, "status": "hold"})
            continue
        w_monthly = (1 + w_rets).prod() - 1
        topix_w_ret = float((1 + topix_w).prod() - 1)
        w_excess = w_monthly - topix_w_ret

        w_reg = reg_df.copy()
        w_reg["w_excess"] = [float(w_excess.get(c, np.nan)) for c in w_reg["code"]]
        w_reg = w_reg.dropna(subset=["w_excess"])

        if len(w_reg) < 50:
            weekly_results.append({"week": wlabel, "status": "insufficient"})
            continue

        w_X = pd.concat([w_reg[xvars], pd.get_dummies(w_reg["sector"], prefix="sec", drop_first=True)], axis=1)
        w_X = sm.add_constant(w_X)
        try:
            w_model = sm.OLS(w_reg["w_excess"], w_X).fit(cov_type="HC1")
            b1 = float(w_model.params.get("feb_ret", np.nan))
            p1 = float(w_model.pvalues.get("feb_ret", np.nan))
            b2 = float(w_model.params.get("oil_beta", np.nan))
            p2 = float(w_model.pvalues.get("oil_beta", np.nan))
            print(f"  {wlabel}: feb_ret coef={b1:+.4f} p={p1:.4f} | oil_beta coef={b2:+.4f} p={p2:.4f} | R2={w_model.rsquared:.4f}")
            weekly_results.append({
                "week": wlabel, "status": "computed",
                "feb_ret_coef": b1, "feb_ret_p": p1,
                "oil_beta_coef": b2, "oil_beta_p": p2,
                "r2": float(w_model.rsquared),
            })
        except Exception as e:
            print(f"  {wlabel}: エラー - {e}")
            weekly_results.append({"week": wlabel, "status": "error", "msg": str(e)})

    # --- 総合判定 ---
    print("\n" + "=" * 60)
    print("=== T3-Q04 総合判定 ===")
    print("=" * 60)

    feb_coef = float(model.params.get("feb_ret", np.nan))
    feb_p = float(model.pvalues.get("feb_ret", np.nan))
    oil_coef = float(model.params.get("oil_beta", np.nan))
    oil_p = float(model.pvalues.get("oil_beta", np.nan))

    momentum_reversal = feb_coef < 0 and feb_p < 0.10
    oil_transmission = oil_coef < 0 and oil_p < 0.10
    r2 = float(model.rsquared)

    if momentum_reversal and oil_transmission:
        verdict = "BOTH"
        print(f"判定: BOTH -- モメンタム反転(β1={feb_coef:+.4f}, p={feb_p:.4f})と原油波及(β2={oil_coef:+.4f}, p={oil_p:.4f})の両方が寄与")
    elif momentum_reversal:
        verdict = "MOMENTUM"
        print(f"判定: MOMENTUM -- モメンタム反転が主因(β1={feb_coef:+.4f}, p={feb_p:.4f})。原油波及は非有意(p={oil_p:.4f})")
    elif oil_transmission:
        verdict = "OIL"
        print(f"判定: OIL -- 原油波及が主因(β2={oil_coef:+.4f}, p={oil_p:.4f})。モメンタム反転は非有意(p={feb_p:.4f})")
    elif feb_p > 0.20 and oil_p > 0.20 and r2 < 0.10:
        verdict = "INDETERMINATE"
        print(f"判定: INDETERMINATE -- この設計では分離不能。R2={r2:.4f}")
    else:
        verdict = "AMBIGUOUS"
        print(f"判定: AMBIGUOUS -- β1 p={feb_p:.4f}, β2 p={oil_p:.4f}, R2={r2:.4f}")

    # --- JSON保存 ---
    output = {
        "experiment": "T3-Q04",
        "n_stocks": len(final_codes),
        "topix_mar_return": topix_mar_monthly,
        "sector_correlation": float(r) if not np.isnan(r) else None,
        "sector_summary": sector_summary,
        "regression": {
            "r2": r2,
            "feb_ret": {"coef": feb_coef, "p": feb_p},
            "oil_beta": {"coef": oil_coef, "p": oil_p},
            "topix_beta": {"coef": float(model.params.get("topix_beta", np.nan)), "p": float(model.pvalues.get("topix_beta", np.nan))},
            "usdjpy_beta": {"coef": float(model.params.get("usdjpy_beta", np.nan)), "p": float(model.pvalues.get("usdjpy_beta", np.nan))},
        },
        "vif": vifs,
        "partial_r2": partial_r2,
        "contributions": avg_contributions,
        "avg_mar_excess": total_excess,
        "weekly": weekly_results,
        "verdict": verdict,
    }

    output_path = Path("logs/iterations/T3-Q04_oil-shock-sector-contagion-path_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果を保存: {output_path}")


if __name__ == "__main__":
    main()
