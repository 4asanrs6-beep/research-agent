"""T4-Q04 (旧Q03): PCR(プットコール比率)の方向性シグナル

Step 0: データ取得フィージビリティ確認
Step 1: ΔPCR回帰
Step 2: 五分位単調性
"""

import json
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import yfinance as yf


def main():
    print("=== T4-Q04: PCR方向性シグナル ===\n")

    results = {}

    # --- Step 0: データフィージビリティ ---
    print("--- Step 0: PCRデータ取得確認 ---")

    # yfinanceでオプションデータを取得してPCRを計算
    # 日経225のオプションは直接取れないので、代替アプローチ
    # (1) CBOEのPCR(^PCALL)は取れるか
    # (2) VIXの変化率で代替
    # (3) 日経VIで代替

    pcr_available = False

    # 試行1: CBOE Put/Call Ratio
    try:
        pcr_raw = yf.download("^PCALL", period="6mo", interval="1d", progress=False)
        if len(pcr_raw) > 20:
            pcr = pcr_raw["Close"].squeeze()
            if hasattr(pcr.index, 'tz') and pcr.index.tz is not None:
                pcr.index = pcr.index.tz_localize(None)
            print(f"  CBOE PCR(^PCALL): {len(pcr)}日 取得成功")
            pcr_available = True
            pcr_source = "CBOE_PCALL"
        else:
            print("  CBOE PCR: データ不足")
    except Exception as e:
        print(f"  CBOE PCR: 取得失敗 ({e})")

    if not pcr_available:
        # 試行2: VIXの変化率で代替
        print("  PCR取得不可。VIX変化率で代替")
        vix_raw = yf.download("^VIX", period="6mo", interval="1d", progress=False)
        if hasattr(vix_raw.index, 'tz') and vix_raw.index.tz is not None:
            vix_raw.index = vix_raw.index.tz_localize(None)
        vix = vix_raw["Close"].squeeze()
        pcr = vix  # VIXをPCRの代理として使用
        pcr_source = "VIX_proxy"
        pcr_available = len(pcr) > 20
        print(f"  VIX代理: {len(pcr)}日")

    results["step0_feasibility"] = {"pcr_available": pcr_available, "source": pcr_source if pcr_available else "none"}

    if not pcr_available:
        results["verdict"] = "data_unavailable"
        print("PCRデータ取得不可。終了")
    else:
        # 日経ETF日次リターン
        etf = yf.download("1321.T", period="6mo", interval="1d", progress=False)
        if hasattr(etf.index, 'tz') and etf.index.tz is not None:
            etf.index = etf.index.tz_localize(None)
        etf_ret = etf["Close"].squeeze().pct_change().dropna()

        # ΔPCR（前日比変化率）
        delta_pcr = pcr.pct_change().dropna()

        # 共通日付
        common = delta_pcr.index.intersection(etf_ret.index)
        data = pd.DataFrame({
            "delta_pcr": delta_pcr.reindex(common),
            "etf_ret": etf_ret.reindex(common),
            "etf_ret_next": etf_ret.shift(-1).reindex(common),  # 翌日リターン
            "etf_ret_next5": etf_ret.rolling(5).sum().shift(-5).reindex(common),  # 翌5日累積
        }).dropna()

        print(f"\n  有効日数: {len(data)}")
        print(f"  PCRソース: {pcr_source}")

        # --- Step 1: ΔPCR→翌日リターン回帰 ---
        print("\n--- Step 1: ΔPCR→翌日リターン ---")

        # 当日リターン統制
        import statsmodels.api as sm
        X = sm.add_constant(data[["delta_pcr", "etf_ret"]].astype(float))
        y = data["etf_ret_next"].astype(float)
        model = sm.OLS(y, X).fit(cov_type="HC1")
        pcr_coef = float(model.params["delta_pcr"])
        pcr_p = float(model.pvalues["delta_pcr"])
        print(f"  ΔPCR→翌日: coef={pcr_coef:+.6f} (p={pcr_p:.4f})")
        print(f"  当日リターン統制: coef={float(model.params['etf_ret']):+.4f} (p={float(model.pvalues['etf_ret']):.4f})")
        print(f"  R2={model.rsquared:.4f}")

        # 翌5日
        data5 = data.dropna(subset=["etf_ret_next5"])
        if len(data5) >= 20:
            X5 = sm.add_constant(data5[["delta_pcr", "etf_ret"]].astype(float))
            y5 = data5["etf_ret_next5"].astype(float)
            model5 = sm.OLS(y5, X5).fit(cov_type="HC1")
            pcr_coef5 = float(model5.params["delta_pcr"])
            pcr_p5 = float(model5.pvalues["delta_pcr"])
            print(f"  ΔPCR→翌5日: coef={pcr_coef5:+.6f} (p={pcr_p5:.4f})")
        else:
            pcr_coef5, pcr_p5 = np.nan, np.nan

        results["step1_regression"] = {
            "next1d_coef": pcr_coef, "next1d_p": pcr_p, "r2": float(model.rsquared),
            "next5d_coef": float(pcr_coef5) if not np.isnan(pcr_coef5) else None,
            "next5d_p": float(pcr_p5) if not np.isnan(pcr_p5) else None,
        }

        # --- Step 2: 五分位単調性 ---
        print("\n--- Step 2: ΔPCR五分位と翌日リターン ---")
        data["quintile"] = pd.qcut(data["delta_pcr"], 5, labels=False, duplicates="drop")
        quintile_means = data.groupby("quintile")["etf_ret_next"].mean()

        for q, m in quintile_means.items():
            n_q = (data["quintile"] == q).sum()
            print(f"  Q{q}: ΔPCR五分位{q}, 翌日平均ret={m*100:+.4f}% (n={n_q})")

        # Spearman相関
        rho, rho_p = scipy_stats.spearmanr(quintile_means.index, quintile_means.values)
        print(f"  五分位単調性: Spearman rho={rho:.3f} (p={rho_p:.4f})")

        results["step2_quintile"] = {
            "quintile_means": {int(k): float(v) for k, v in quintile_means.items()},
            "spearman_rho": float(rho), "spearman_p": float(rho_p),
        }

        # --- 総合判定 ---
        print(f"\n{'='*60}")
        print("=== T4-Q04 総合判定 ===")
        print(f"{'='*60}")

        if pcr_source == "VIX_proxy":
            print(f"注: PCR取得不可のためVIX代理で分析。結果の解釈は限定的")

        if pcr_p < 0.10:
            verdict = "PCR_SIGNAL"
            print(f"判定: PCR_SIGNAL -- ΔPCR→翌日リターンが有意(p={pcr_p:.4f})")
        elif abs(rho) > 0.8 and rho_p < 0.20:
            verdict = "MONOTONIC_ONLY"
            print(f"判定: MONOTONIC_ONLY -- 五分位に単調性あり(rho={rho:.3f})だが回帰は非有意")
        else:
            verdict = "NO_SIGNAL"
            print(f"判定: NO_SIGNAL -- ΔPCR→翌日p={pcr_p:.4f}, 五分位rho={rho:.3f}")

        results["verdict"] = verdict
        results["pcr_source"] = pcr_source

    # JSON保存
    output_path = Path("logs/iterations/T4-Q04_put-call-ratio-directional-signal_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果を保存: {output_path}")


if __name__ == "__main__":
    main()
