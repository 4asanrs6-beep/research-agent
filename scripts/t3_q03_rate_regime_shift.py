"""T3-Q03: 金利レジーム転換がLS構造をいつ壊したか

Step 0: Fisher z検出力分析
Step 1: IWF/IWD相関(リスクオフ一体化判定)
Step 2: 金利×(IWF-IWD)スプレッド相関差(Fisher z検定)
Step 3: 日本株代理(JP_Banks/JP_Machine)で同一分析
"""

import json
import sys
import os
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import yfinance as yf


def fisher_z_power(n1, n2, rho_diff, alpha=0.05):
    """Fisher z変換の検出力を解析的に計算"""
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z_crit = scipy_stats.norm.ppf(1 - alpha/2)
    z_effect = np.arctanh(rho_diff) / se  # 近似
    power = 1 - scipy_stats.norm.cdf(z_crit - abs(z_effect)) + scipy_stats.norm.cdf(-z_crit - abs(z_effect))
    return float(power)


def fisher_z_test(r1, n1, r2, n2):
    """2つの相関係数の差のFisher z検定"""
    z1 = np.arctanh(min(max(r1, -0.999), 0.999))
    z2 = np.arctanh(min(max(r2, -0.999), 0.999))
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z_stat = (z1 - z2) / se
    p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
    return float(z_stat), float(p_val)


def main():
    print("=== T3-Q03: 金利レジーム転換がLS構造をいつ壊したか ===\n")

    # --- データ取得 ---
    print("データ取得中...")
    tickers = {
        "IWF": "IWF", "IWD": "IWD", "TNX": "^TNX",
        "JP_Banks": "1615.T", "JP_Machine": "1620.T",
    }
    data = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, start="2026-01-01", end="2026-04-01", progress=False)
        if len(df) > 0:
            ret = df["Close"].squeeze().pct_change().dropna()
            if hasattr(ret.index, 'tz') and ret.index.tz is not None:
                ret.index = ret.index.tz_localize(None)
            data[name] = ret
            print(f"  {name}: {len(ret)}日")
        else:
            print(f"  {name}: 取得失敗")

    results = {}

    # --- 期間分割 ---
    # ブレーク=2/28(イラン攻撃日)
    break_date = pd.Timestamp("2026-02-28")
    jan_start = pd.Timestamp("2026-01-01")
    feb_end = break_date
    mar_start = pd.Timestamp("2026-03-01")
    mar_end = pd.Timestamp("2026-03-31")

    def split_period(series):
        jan_feb = series[(series.index >= jan_start) & (series.index <= feb_end)]
        mar = series[(series.index >= mar_start) & (series.index <= mar_end)]
        jan = series[(series.index >= jan_start) & (series.index < pd.Timestamp("2026-02-01"))]
        feb = series[(series.index >= pd.Timestamp("2026-02-01")) & (series.index <= feb_end)]
        return {"jan": jan, "feb": feb, "jan_feb": jan_feb, "mar": mar}

    # --- Step 0: Fisher z検出力 ---
    print("\n--- Step 0: Fisher z検出力分析 ---")
    n_feb = 19  # 2月の営業日(概算)
    n_mar = 22  # 3月の営業日(概算)
    power_05 = fisher_z_power(n_feb, n_mar, 0.5)
    power_03 = fisher_z_power(n_feb, n_mar, 0.3)
    print(f"  n_feb~{n_feb}, n_mar~{n_mar}")
    print(f"  rho_diff=0.5検出力: {power_05:.2%}")
    print(f"  rho_diff=0.3検出力: {power_03:.2%}")
    skip_step2 = power_05 < 0.50
    if skip_step2:
        print(f"  WARNING: rho_diff=0.5でも検出力{power_05:.0%}<50%。Step 2は参考値")
    results["step0_power"] = {"n_feb": n_feb, "n_mar": n_mar, "power_05": power_05, "power_03": power_03, "skip_step2": skip_step2}

    # --- Step 1: IWF/IWD相関(リスクオフ一体化) ---
    print("\n--- Step 1: IWF/IWD日次リターン相関 ---")
    if "IWF" in data and "IWD" in data:
        iwf_split = split_period(data["IWF"])
        iwd_split = split_period(data["IWD"])

        step1 = {}
        for period_name in ["jan", "feb", "mar"]:
            iwf_p = iwf_split[period_name]
            iwd_p = iwd_split[period_name]
            common = iwf_p.index.intersection(iwd_p.index)
            if len(common) >= 5:
                r, p = scipy_stats.pearsonr(iwf_p.reindex(common), iwd_p.reindex(common))
                print(f"  {period_name}: IWF-IWD相関 r={r:.3f} (p={p:.4f}, n={len(common)})")
                step1[period_name] = {"r": float(r), "p": float(p), "n": len(common)}
            else:
                step1[period_name] = {"status": "insufficient"}

        mar_r = step1.get("mar", {}).get("r", 0)
        if mar_r > 0.9:
            print(f"\n  *** 3月のIWF-IWD相関={mar_r:.3f} > 0.9 → リスクオフ一体化 ***")
            print(f"  グロースもバリューも同じ方向に動く=LS戦略のヘッジが崩壊")
            results["step1_unification"] = step1
            results["early_termination"] = True
            results["verdict"] = "risk_off_unification"
        else:
            results["step1_unification"] = step1
            results["early_termination"] = False
    else:
        print("  IWF/IWDデータなし")
        results["early_termination"] = False

    # --- Step 2: 金利×スプレッド相関差 ---
    if not results.get("early_termination"):
        print("\n--- Step 2: 金利×(IWF-IWD)スプレッド相関差 ---")
    else:
        print("\n--- Step 2: (早期終了だが参考値として計算) ---")

    if "IWF" in data and "IWD" in data and "TNX" in data:
        # IWF-IWDスプレッド
        common_all = data["IWF"].index.intersection(data["IWD"].index).intersection(data["TNX"].index)
        spread = data["IWF"].reindex(common_all) - data["IWD"].reindex(common_all)
        tnx = data["TNX"].reindex(common_all)

        step2 = {}
        for period_name in ["feb", "mar"]:
            if period_name == "feb":
                mask = (spread.index >= pd.Timestamp("2026-02-01")) & (spread.index <= break_date)
            else:
                mask = (spread.index >= mar_start) & (spread.index <= mar_end)
            sp_p = spread[mask].dropna()
            tnx_p = tnx[mask].dropna()
            common = sp_p.index.intersection(tnx_p.index)
            if len(common) >= 5:
                r, p = scipy_stats.pearsonr(sp_p.reindex(common), tnx_p.reindex(common))
                print(f"  {period_name}: 金利×スプレッド相関 r={r:.3f} (p={p:.4f}, n={len(common)})")
                step2[period_name] = {"r": float(r), "p": float(p), "n": len(common)}

        # Fisher z検定
        if "feb" in step2 and "mar" in step2:
            z, pz = fisher_z_test(step2["feb"]["r"], step2["feb"]["n"], step2["mar"]["r"], step2["mar"]["n"])
            print(f"  Fisher z検定: z={z:.3f}, p={pz:.4f}")
            step2["fisher_z"] = {"z": float(z), "p": float(pz)}
            if skip_step2:
                print(f"  (検出力不足のため参考値)")
        results["step2_rate_spread"] = step2

    # --- Step 3: 日本株代理 ---
    print("\n--- Step 3: 日本株代理分析 ---")
    if "JP_Banks" in data and "JP_Machine" in data:
        jb_split = split_period(data["JP_Banks"])
        jm_split = split_period(data["JP_Machine"])

        step3 = {}
        for period_name in ["jan", "feb", "mar"]:
            jb_p = jb_split[period_name]
            jm_p = jm_split[period_name]
            common = jb_p.index.intersection(jm_p.index)
            if len(common) >= 5:
                r, p = scipy_stats.pearsonr(jb_p.reindex(common), jm_p.reindex(common))
                print(f"  {period_name}: JP_Banks-JP_Machine相関 r={r:.3f} (p={p:.4f}, n={len(common)})")
                step3[period_name] = {"r": float(r), "p": float(p), "n": len(common)}

        # 3月で金利上昇日にJP_Banksが下落した日を特定
        if "TNX" in data:
            tnx_mar = data["TNX"][(data["TNX"].index >= mar_start) & (data["TNX"].index <= mar_end)]
            jb_mar = data["JP_Banks"][(data["JP_Banks"].index >= mar_start) & (data["JP_Banks"].index <= mar_end)]
            common_mar = tnx_mar.index.intersection(jb_mar.index)
            rate_up_bank_down = []
            for d in common_mar:
                if tnx_mar[d] > 0 and jb_mar[d] < 0:
                    rate_up_bank_down.append({"date": str(d.date()), "tnx_ret": float(tnx_mar[d]), "bank_ret": float(jb_mar[d])})
            print(f"  3月で金利上昇+銀行下落の日: {len(rate_up_bank_down)}/{len(common_mar)}日")
            step3["rate_up_bank_down_days"] = rate_up_bank_down

        results["step3_japan"] = step3

    # --- 総合判定 ---
    print(f"\n{'='*60}")
    print("=== T3-Q03 総合判定 ===")
    print(f"{'='*60}")

    if results.get("verdict") == "risk_off_unification":
        print(f"判定: risk_off_unification -- 3月はIWF-IWD相関>{mar_r:.2f}でリスクオフ一体化")
        print(f"  金利レジーム転換は副次的。主因はリスクオフによる全資産同時下落")
    else:
        fisher_p = results.get("step2_rate_spread", {}).get("fisher_z", {}).get("p", 1.0)
        if fisher_p < 0.10 and not skip_step2:
            results["verdict"] = "rate_regime_confirmed"
            print(f"判定: rate_regime_confirmed -- 金利×スプレッド相関が2月→3月で有意に変化(p={fisher_p:.4f})")
        elif skip_step2:
            results["verdict"] = "insufficient_power"
            print(f"判定: insufficient_power -- 検出力不足で金利レジーム転換を検出できない")
        else:
            results["verdict"] = "rate_regime_not_detected"
            print(f"判定: rate_regime_not_detected -- Fisher z p={fisher_p:.4f}>0.10")

    # JSON保存
    output_path = Path("logs/iterations/T3-Q03_rate-regime-shift-impact_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果を保存: {output_path}")


if __name__ == "__main__":
    main()
