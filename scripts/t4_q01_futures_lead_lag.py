"""T4-Q01: 日経ETF(1321.T)→個別株β分位ポートフォリオのリード・ラグ分析

Step 1: β分位ポートフォリオ構築
Step 2: 1h足リード・ラグ回帰
Step 3: VIX区間分割
Step 4: thin trading確認
"""

import json
import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from datetime import date, timedelta

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yfinance as yf


def main():
    print("=== T4-Q01: 先物(ETF)→個別株のリード・ラグ分析 ===\n")

    # --- Step 0: データ取得 ---
    print("データ取得中...")

    # 1321.T(日経ETF) 1h足
    etf = yf.download("1321.T", period="3mo", interval="1h", progress=False)
    if hasattr(etf.index, 'tz') and etf.index.tz is not None:
        etf.index = etf.index.tz_localize(None)
    etf_ret = etf["Close"].squeeze().pct_change().dropna()
    print(f"  1321.T 1h足: {len(etf_ret)}行")

    # 個別株 1h足（大型50銘柄）
    top_stocks = [
        "7203.T", "6758.T", "8306.T", "9984.T", "6501.T",
        "7267.T", "8035.T", "6902.T", "4502.T", "9432.T",
        "6861.T", "7741.T", "6098.T", "4063.T", "8058.T",
        "6954.T", "7974.T", "3382.T", "2914.T", "4568.T",
        "8766.T", "6367.T", "4519.T", "9433.T", "6762.T",
        "7751.T", "8801.T", "5401.T", "4901.T", "6503.T",
        "8031.T", "6988.T", "3407.T", "7832.T", "4543.T",
        "6857.T", "9983.T", "2802.T", "1925.T", "4307.T",
    ]

    stock_data = {}
    for ticker in top_stocks:
        try:
            df = yf.download(ticker, period="3mo", interval="1h", progress=False)
            if len(df) > 10:
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                ret = df["Close"].squeeze().pct_change().dropna()
                stock_data[ticker] = ret
        except Exception:
            pass
    print(f"  個別株取得: {len(stock_data)}/{len(top_stocks)}銘柄")

    if len(stock_data) < 10:
        print("個別株データ不足。中止")
        return

    # VIX日次
    vix_raw = yf.download("^VIX", period="3mo", interval="1d", progress=False)
    if hasattr(vix_raw.index, 'tz') and vix_raw.index.tz is not None:
        vix_raw.index = vix_raw.index.tz_localize(None)
    vix_daily = vix_raw["Close"].squeeze()
    print(f"  VIX日次: {len(vix_daily)}日")

    # β推定（日足ベース、1321.T対比）
    print("\nβ推定中...")
    etf_daily = yf.download("1321.T", period="6mo", interval="1d", progress=False)
    if hasattr(etf_daily.index, 'tz') and etf_daily.index.tz is not None:
        etf_daily.index = etf_daily.index.tz_localize(None)
    etf_daily_ret = etf_daily["Close"].squeeze().pct_change().dropna()

    betas = {}
    for ticker, ret_1h in stock_data.items():
        # 日足リターンを計算（1h足からresample）
        daily_ret = ret_1h.resample("D").apply(lambda x: (1 + x).prod() - 1 if len(x) > 0 else np.nan).dropna()
        common = daily_ret.index.intersection(etf_daily_ret.index)
        if len(common) < 30:
            continue
        y = daily_ret.reindex(common)
        x = etf_daily_ret.reindex(common)
        valid = ~(y.isna() | x.isna())
        if valid.sum() < 30:
            continue
        cov = np.cov(y[valid], x[valid])
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-12 else 1.0
        betas[ticker] = float(beta)

    print(f"  β推定成功: {len(betas)}銘柄")

    if len(betas) < 15:
        print("β推定銘柄不足。中止")
        return

    # --- Step 1: β分位ポートフォリオ構築 ---
    print("\n--- Step 1: β分位ポートフォリオ構築 ---")
    sorted_by_beta = sorted(betas.items(), key=lambda x: x[1], reverse=True)
    n_per_group = max(len(sorted_by_beta) // 3, 5)

    high_beta = [t for t, b in sorted_by_beta[:n_per_group]]
    mid_beta = [t for t, b in sorted_by_beta[n_per_group:2*n_per_group]]
    low_beta = [t for t, b in sorted_by_beta[2*n_per_group:3*n_per_group]]

    print(f"  高β({n_per_group}銘柄): β平均={np.mean([betas[t] for t in high_beta]):.3f}")
    print(f"  中β({n_per_group}銘柄): β平均={np.mean([betas[t] for t in mid_beta]):.3f}")
    print(f"  低β({min(n_per_group, len(low_beta))}銘柄): β平均={np.mean([betas[t] for t in low_beta]):.3f}")

    # ポートフォリオ1hリターン（等ウェイト）
    def portfolio_return(tickers, timestamp):
        rets = []
        for t in tickers:
            if t in stock_data and timestamp in stock_data[t].index:
                r = stock_data[t][timestamp]
                if not np.isnan(r):
                    rets.append(r)
        return float(np.mean(rets)) if len(rets) >= 3 else np.nan

    # 共通タイムスタンプ
    common_times = etf_ret.index
    port_data = []
    for ts in common_times:
        etf_r = float(etf_ret.get(ts, np.nan))
        if np.isnan(etf_r):
            continue
        high_r = portfolio_return(high_beta, ts)
        mid_r = portfolio_return(mid_beta, ts)
        low_r = portfolio_return(low_beta, ts)
        # VIX（当日の値を使用）
        vix_date = ts.date() if hasattr(ts, 'date') else ts
        vix_val = vix_daily.get(pd.Timestamp(vix_date), np.nan) if pd.Timestamp(vix_date) in vix_daily.index else np.nan

        port_data.append({
            "timestamp": ts,
            "etf_ret": etf_r,
            "high_beta_ret": high_r,
            "mid_beta_ret": mid_r,
            "low_beta_ret": low_r,
            "vix": float(vix_val) if not np.isnan(vix_val) else np.nan,
        })

    df = pd.DataFrame(port_data).dropna(subset=["etf_ret", "high_beta_ret"])
    print(f"  有効タイムスタンプ: {len(df)}")

    # 前1hのETFリターンを追加
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["etf_ret_lag1"] = df["etf_ret"].shift(1)
    df["etf_ret_lag2"] = df["etf_ret"].shift(2)
    df = df.dropna(subset=["etf_ret_lag1"])

    results = {}

    # --- Step 2: リード・ラグ回帰 ---
    print("\n--- Step 2: リード・ラグ回帰 ---")
    for label, col in [("high_beta", "high_beta_ret"), ("mid_beta", "mid_beta_ret"), ("low_beta", "low_beta_ret")]:
        valid = df.dropna(subset=[col])
        if len(valid) < 30:
            print(f"  {label}: データ不足({len(valid)})")
            continue

        y = valid[col].astype(float)
        X = sm.add_constant(valid[["etf_ret", "etf_ret_lag1"]].astype(float))

        try:
            model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
            contemp = float(model.params.get("etf_ret", np.nan))
            contemp_p = float(model.pvalues.get("etf_ret", np.nan))
            lead = float(model.params.get("etf_ret_lag1", np.nan))
            lead_p = float(model.pvalues.get("etf_ret_lag1", np.nan))
            print(f"  {label}: 同時刻coef={contemp:+.4f}(p={contemp_p:.4f}), 前1h(リード)coef={lead:+.4f}(p={lead_p:.4f}), R2={model.rsquared:.4f}")

            results[label] = {
                "n": len(valid),
                "contemp_coef": contemp, "contemp_p": contemp_p,
                "lead_coef": lead, "lead_p": lead_p,
                "r2": float(model.rsquared),
            }
        except Exception as e:
            print(f"  {label}: エラー - {e}")

    # --- Step 3: VIX区間分割 ---
    print("\n--- Step 3: VIX区間分割 ---")
    df_vix = df.dropna(subset=["vix"])
    n_high_vix = (df_vix["vix"] > 25).sum()
    if n_high_vix >= 20:
        vix_bins = [("VIX<=20", df_vix["vix"] <= 20), ("VIX 20-25", (df_vix["vix"] > 20) & (df_vix["vix"] <= 25)), ("VIX>25", df_vix["vix"] > 25)]
    else:
        vix_bins = [("VIX<=20", df_vix["vix"] <= 20), ("VIX>20", df_vix["vix"] > 20)]
        print(f"  VIX>25が{n_high_vix}日。2区間にフォールバック")

    vix_results = {}
    for vix_label, vix_mask in vix_bins:
        sub = df_vix[vix_mask].dropna(subset=["high_beta_ret", "etf_ret_lag1"])
        if len(sub) < 15:
            print(f"  {vix_label}: データ不足({len(sub)})")
            vix_results[vix_label] = {"status": "insufficient", "n": len(sub)}
            continue
        y = sub["high_beta_ret"].astype(float)
        X = sm.add_constant(sub[["etf_ret", "etf_ret_lag1"]].astype(float))
        try:
            model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
            lead = float(model.params.get("etf_ret_lag1", np.nan))
            lead_p = float(model.pvalues.get("etf_ret_lag1", np.nan))
            print(f"  {vix_label}(n={len(sub)}): 高βリードcoef={lead:+.4f}(p={lead_p:.4f})")
            vix_results[vix_label] = {"n": len(sub), "lead_coef": lead, "lead_p": lead_p}
        except Exception as e:
            print(f"  {vix_label}: エラー - {e}")

    results["vix_split"] = vix_results

    # --- Step 4: thin trading確認 ---
    print("\n--- Step 4: thin trading確認 ---")
    # 出来高上位20%の時間帯のみで再推定
    etf_vol = etf["Volume"].squeeze()
    if hasattr(etf_vol.index, 'tz') and etf_vol.index.tz is not None:
        etf_vol.index = etf_vol.index.tz_localize(None)
    vol_threshold = etf_vol.quantile(0.80)
    high_vol_times = set(etf_vol[etf_vol >= vol_threshold].index)

    df_highvol = df[df["timestamp"].isin(high_vol_times)].dropna(subset=["high_beta_ret", "etf_ret_lag1"])
    if len(df_highvol) >= 20:
        y = df_highvol["high_beta_ret"].astype(float)
        X = sm.add_constant(df_highvol[["etf_ret", "etf_ret_lag1"]].astype(float))
        try:
            model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
            lead = float(model.params.get("etf_ret_lag1", np.nan))
            lead_p = float(model.pvalues.get("etf_ret_lag1", np.nan))
            print(f"  出来高上位20%(n={len(df_highvol)}): 高βリードcoef={lead:+.4f}(p={lead_p:.4f})")
            results["thin_trading_check"] = {"n": len(df_highvol), "lead_coef": lead, "lead_p": lead_p}
        except Exception as e:
            print(f"  エラー: {e}")
    else:
        print(f"  出来高上位20%のサンプル不足({len(df_highvol)})")

    # --- 総合判定 ---
    print(f"\n{'='*60}")
    print("=== T4-Q01 総合判定 ===")
    print(f"{'='*60}")

    high_lead = results.get("high_beta", {}).get("lead_coef", 0)
    high_lead_p = results.get("high_beta", {}).get("lead_p", 1)
    thin_lead_p = results.get("thin_trading_check", {}).get("lead_p", 1)

    if high_lead_p < 0.10 and high_lead > 0:
        if thin_lead_p < 0.10:
            verdict = "LEAD_CONFIRMED"
            print(f"判定: LEAD_CONFIRMED -- 高β前1hリードcoef={high_lead:+.4f}(p={high_lead_p:.4f})。出来高上位でも維持")
        else:
            verdict = "THIN_TRADING_ONLY"
            print(f"判定: THIN_TRADING_ONLY -- リードあるが出来高上位で消失。thin tradingアーティファクト")
    else:
        verdict = "NO_LEAD"
        print(f"判定: NO_LEAD -- 高β前1hリードcoef={high_lead:+.4f}(p={high_lead_p:.4f})。リードなし")

    results["verdict"] = verdict

    # JSON保存
    output_path = Path("logs/iterations/T4-Q01_futures-stock-lead-lag_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果を保存: {output_path}")


if __name__ == "__main__":
    main()
