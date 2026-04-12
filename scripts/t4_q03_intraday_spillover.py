"""T4-Q03: 前場先物→後場個別株の波及。出来高急増日の増分効果

Step 1: ベースライン自己相関
Step 2: 先物リード
Step 3: 出来高交互作用
"""

import json
import sys
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import yfinance as yf


def main():
    print("=== T4-Q03: 前場先物→後場個別株の波及 ===\n")

    # データ取得
    print("データ取得中...")
    etf = yf.download("1321.T", period="3mo", interval="1h", progress=False)
    if hasattr(etf.index, 'tz') and etf.index.tz is not None:
        etf.index = etf.index.tz_localize(None)
    print(f"  1321.T 1h足: {len(etf)}行")

    # 個別株（トヨタで代表）
    toyota = yf.download("7203.T", period="3mo", interval="1h", progress=False)
    if hasattr(toyota.index, 'tz') and toyota.index.tz is not None:
        toyota.index = toyota.index.tz_localize(None)
    print(f"  7203.T 1h足: {len(toyota)}行")

    # 前場/後場の分割
    # 東証: 前場 9:00-11:30 (UTC 0:00-2:30), 後場 12:30-15:00 (UTC 3:30-6:00)
    # yfinanceのタイムスタンプはUTCまたはJST
    etf_close = etf["Close"].squeeze()
    etf_volume = etf["Volume"].squeeze()
    toyota_close = toyota["Close"].squeeze()

    # 日ごとに前場/後場リターンを計算
    results_list = []
    dates = sorted(set(etf_close.index.date))

    for d in dates:
        day_mask = etf_close.index.date == d
        day_etf = etf_close[day_mask]
        day_vol = etf_volume[day_mask]
        day_toyota = toyota_close[toyota_close.index.date == d]

        if len(day_etf) < 4 or len(day_toyota) < 4:
            continue

        # 前場リターン: 最初の3時間足の累積
        # 後場リターン: 残りの時間足の累積
        n_bars = len(day_etf)
        mid = min(3, n_bars // 2)  # 前場=最初3本, 後場=残り

        etf_am_ret = float(day_etf.iloc[mid] / day_etf.iloc[0] - 1)
        etf_pm_ret = float(day_etf.iloc[-1] / day_etf.iloc[mid] - 1)
        etf_am_vol = float(day_vol.iloc[:mid].sum())

        if len(day_toyota) >= mid + 1:
            toyota_am_ret = float(day_toyota.iloc[mid] / day_toyota.iloc[0] - 1)
            toyota_pm_ret = float(day_toyota.iloc[-1] / day_toyota.iloc[mid] - 1)
        else:
            continue

        results_list.append({
            "date": pd.Timestamp(d),
            "etf_am_ret": etf_am_ret,
            "etf_pm_ret": etf_pm_ret,
            "toyota_am_ret": toyota_am_ret,
            "toyota_pm_ret": toyota_pm_ret,
            "etf_am_vol": etf_am_vol,
        })

    df = pd.DataFrame(results_list).dropna()
    print(f"  有効日数: {len(df)}")

    # 出来高中央値
    vol_median = df["etf_am_vol"].median()
    df["high_vol"] = (df["etf_am_vol"] > vol_median * 1.5).astype(float)
    n_high_vol = int(df["high_vol"].sum())
    print(f"  前場出来高1.5倍超の日: {n_high_vol}/{len(df)}")

    results = {"n_days": len(df), "n_high_vol": n_high_vol}

    # --- Step 1: ベースライン自己相関 ---
    print("\n--- Step 1: トヨタの前場→後場の自己相関 ---")
    from scipy import stats as scipy_stats
    r_auto, p_auto = scipy_stats.pearsonr(df["toyota_am_ret"], df["toyota_pm_ret"])
    print(f"  トヨタ前場→後場: r={r_auto:.3f} (p={p_auto:.4f})")
    results["step1_autocorr"] = {"r": float(r_auto), "p": float(p_auto)}

    # --- Step 2: 先物(ETF)前場→トヨタ後場のリード ---
    print("\n--- Step 2: ETF前場→トヨタ後場のリード ---")
    X = sm.add_constant(df[["etf_am_ret", "toyota_am_ret"]].astype(float))
    y = df["toyota_pm_ret"].astype(float)
    model = sm.OLS(y, X).fit(cov_type="HC1")
    etf_lead = float(model.params["etf_am_ret"])
    etf_lead_p = float(model.pvalues["etf_am_ret"])
    auto_ctrl = float(model.params["toyota_am_ret"])
    auto_ctrl_p = float(model.pvalues["toyota_am_ret"])
    print(f"  ETF前場→トヨタ後場: coef={etf_lead:+.4f} (p={etf_lead_p:.4f})")
    print(f"  トヨタ前場(自己相関統制): coef={auto_ctrl:+.4f} (p={auto_ctrl_p:.4f})")
    print(f"  R2={model.rsquared:.4f}")
    results["step2_etf_lead"] = {
        "etf_am_coef": etf_lead, "etf_am_p": etf_lead_p,
        "toyota_am_coef": auto_ctrl, "toyota_am_p": auto_ctrl_p,
        "r2": float(model.rsquared),
    }

    # --- Step 3: 出来高交互作用 ---
    print("\n--- Step 3: 出来高急増日の増分効果 ---")
    df["etf_am_x_highvol"] = df["etf_am_ret"] * df["high_vol"]
    X3 = sm.add_constant(df[["etf_am_ret", "toyota_am_ret", "high_vol", "etf_am_x_highvol"]].astype(float))
    model3 = sm.OLS(y, X3).fit(cov_type="HC1")
    interaction_coef = float(model3.params.get("etf_am_x_highvol", np.nan))
    interaction_p = float(model3.pvalues.get("etf_am_x_highvol", np.nan))
    print(f"  ETF前場×出来高1.5倍超: coef={interaction_coef:+.4f} (p={interaction_p:.4f})")
    print(f"  R2={model3.rsquared:.4f}")

    # VIF確認
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X_check = df[["etf_am_ret", "toyota_am_ret", "high_vol", "etf_am_x_highvol"]].astype(float)
    X_check = sm.add_constant(X_check)
    vifs = {}
    for i, col in enumerate(X_check.columns):
        if col == "const":
            continue
        vifs[col] = float(variance_inflation_factor(X_check.values, i))
    print(f"  VIF: {vifs}")

    results["step3_volume_interaction"] = {
        "interaction_coef": interaction_coef, "interaction_p": interaction_p,
        "r2": float(model3.rsquared), "vif": vifs,
    }

    # --- 総合判定 ---
    print(f"\n{'='*60}")
    print("=== T4-Q03 総合判定 ===")
    print(f"{'='*60}")

    if etf_lead_p < 0.10:
        if interaction_p < 0.10:
            verdict = "LEAD_WITH_VOLUME"
            print(f"判定: LEAD_WITH_VOLUME -- ETF前場→後場リードあり(p={etf_lead_p:.4f})、出来高で増幅(p={interaction_p:.4f})")
        else:
            verdict = "LEAD_NO_VOLUME_EFFECT"
            print(f"判定: LEAD_NO_VOLUME_EFFECT -- リードあり(p={etf_lead_p:.4f})だが出来高増幅なし(p={interaction_p:.4f})")
    else:
        verdict = "NO_LEAD"
        print(f"判定: NO_LEAD -- ETF前場→トヨタ後場リードなし(p={etf_lead_p:.4f})")

    results["verdict"] = verdict

    output_path = Path("logs/iterations/T4-Q03_intraday-futures-momentum-spillover-timing_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果を保存: {output_path}")


if __name__ == "__main__":
    main()
