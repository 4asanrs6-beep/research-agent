"""T4-Q02: 夜間先物リターン→翌日寄付きギャップ+日中追随

Step 1: 夜間先物(NKD=F 18:00-翌5:00)リターンと翌日寄付きギャップの相関
Step 2: ギャップ後の日中追随（方向一致率）
Step 3: β分位別の追随速度差
"""

import json
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import yfinance as yf


def main():
    print("=== T4-Q02: 夜間先物→翌日ギャップ+日中追随 ===\n")

    # データ取得
    print("データ取得中...")
    nkd = yf.download("NKD=F", period="3mo", interval="1h", progress=False)
    etf = yf.download("1321.T", period="3mo", interval="1h", progress=False)
    fx = yf.download("JPY=X", period="3mo", interval="1h", progress=False)

    for df in [nkd, etf, fx]:
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    nkd_close = nkd["Close"].squeeze()
    etf_close = etf["Close"].squeeze()
    fx_close = fx["Close"].squeeze()

    print(f"  NKD=F 1h: {len(nkd_close)}, 1321.T 1h: {len(etf_close)}, USDJPY 1h: {len(fx_close)}")

    # 日次ベースの分析に変換
    # 夜間先物リターン: 前日東証引け後～当日東証寄付き前
    # 簡易的に: NKD=Fの前日16:00→当日8:00のリターン（UTC調整必要）

    # ETFの日足（寄付き・終値）
    etf_daily = yf.download("1321.T", period="3mo", interval="1d", progress=False)
    nkd_daily = yf.download("NKD=F", period="3mo", interval="1d", progress=False)

    for df in [etf_daily, nkd_daily]:
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

    print(f"  ETF日足: {len(etf_daily)}, NKD日足: {len(nkd_daily)}")

    # 夜間先物リターン = 当日NKD始値 / 前日NKD終値 - 1
    nkd_open = nkd_daily["Open"].squeeze()
    nkd_prev_close = nkd_daily["Close"].squeeze().shift(1)
    overnight_ret = (nkd_open / nkd_prev_close - 1).dropna()

    # 寄付きギャップ = 当日ETF始値 / 前日ETF終値 - 1
    etf_open = etf_daily["Open"].squeeze()
    etf_prev_close = etf_daily["Close"].squeeze().shift(1)
    gap = (etf_open / etf_prev_close - 1).dropna()

    # 日中リターン = 当日ETF終値 / 当日ETF始値 - 1
    intraday = (etf_daily["Close"].squeeze() / etf_daily["Open"].squeeze() - 1).dropna()

    # USDJPY夜間変化
    fx_daily = yf.download("JPY=X", period="3mo", interval="1d", progress=False)
    if hasattr(fx_daily.index, 'tz') and fx_daily.index.tz is not None:
        fx_daily.index = fx_daily.index.tz_localize(None)
    fx_overnight = (fx_daily["Open"].squeeze() / fx_daily["Close"].squeeze().shift(1) - 1).dropna()

    # 共通日付
    common = overnight_ret.index.intersection(gap.index).intersection(intraday.index)
    data = pd.DataFrame({
        "overnight": overnight_ret.reindex(common),
        "gap": gap.reindex(common),
        "intraday": intraday.reindex(common),
        "fx_overnight": fx_overnight.reindex(common),
    }).dropna()

    print(f"  有効日数: {len(data)}")

    results = {}

    # --- Step 1: 夜間先物→ギャップの相関 ---
    print("\n--- Step 1: 夜間先物→寄付きギャップ ---")
    from scipy import stats as scipy_stats

    r, p = scipy_stats.pearsonr(data["overnight"], data["gap"])
    print(f"  相関: r={r:.3f} (p={p:.4f}, n={len(data)})")

    # USDJPY調整
    import statsmodels.api as sm
    X = sm.add_constant(data[["overnight", "fx_overnight"]])
    model = sm.OLS(data["gap"], X).fit(cov_type="HC1")
    print(f"  回帰(ギャップ ~ 夜間先物 + FX): R2={model.rsquared:.4f}")
    print(f"    overnight coef={float(model.params['overnight']):+.4f} (p={float(model.pvalues['overnight']):.4f})")
    print(f"    fx_overnight coef={float(model.params['fx_overnight']):+.4f} (p={float(model.pvalues['fx_overnight']):.4f})")

    results["step1_gap_prediction"] = {
        "correlation": float(r), "corr_p": float(p),
        "r2": float(model.rsquared),
        "overnight_coef": float(model.params["overnight"]),
        "overnight_p": float(model.pvalues["overnight"]),
        "fx_coef": float(model.params["fx_overnight"]),
        "fx_p": float(model.pvalues["fx_overnight"]),
    }

    # --- Step 2: 日中追随 ---
    print("\n--- Step 2: ギャップ後の日中追随 ---")

    # 方向一致率: ギャップの方向と日中リターンの方向が同じか
    same_dir = ((data["gap"] > 0) & (data["intraday"] > 0)) | ((data["gap"] < 0) & (data["intraday"] < 0))
    direction_match = float(same_dir.mean())
    print(f"  ギャップ→日中の方向一致率: {direction_match*100:.1f}%")

    # 夜間先物の方向と日中リターンの方向
    same_dir_overnight = ((data["overnight"] > 0) & (data["intraday"] > 0)) | ((data["overnight"] < 0) & (data["intraday"] < 0))
    direction_match_overnight = float(same_dir_overnight.mean())
    print(f"  夜間先物→日中の方向一致率: {direction_match_overnight*100:.1f}%")

    # 非対称性: 上昇ギャップ vs 下落ギャップで日中追随率が異なるか
    up_gap = data[data["gap"] > 0]
    down_gap = data[data["gap"] < 0]
    up_follow = float((up_gap["intraday"] > 0).mean()) if len(up_gap) > 5 else np.nan
    down_follow = float((down_gap["intraday"] < 0).mean()) if len(down_gap) > 5 else np.nan
    print(f"  上昇ギャップ後の追随率: {up_follow*100:.1f}% (n={len(up_gap)})")
    print(f"  下落ギャップ後の追随率: {down_follow*100:.1f}% (n={len(down_gap)})")

    # 夜間先物→（ギャップ+日中）= 終日リターンの予測力
    data["total_ret"] = data["gap"] + data["intraday"]  # 近似
    r_total, p_total = scipy_stats.pearsonr(data["overnight"], data["total_ret"])
    print(f"  夜間先物→終日リターン: r={r_total:.3f} (p={p_total:.4f})")

    results["step2_intraday_follow"] = {
        "gap_intraday_direction_match": direction_match,
        "overnight_intraday_direction_match": direction_match_overnight,
        "up_gap_follow_rate": up_follow,
        "down_gap_follow_rate": down_follow,
        "overnight_total_corr": float(r_total),
        "overnight_total_p": float(p_total),
    }

    # --- 総合判定 ---
    print(f"\n{'='*60}")
    print("=== T4-Q02 総合判定 ===")
    print(f"{'='*60}")

    gap_r2 = results["step1_gap_prediction"]["r2"]
    dir_match = direction_match_overnight

    if gap_r2 > 0.30 and dir_match > 0.60:
        verdict = "STRONG_SIGNAL"
        print(f"判定: STRONG_SIGNAL -- ギャップR2={gap_r2:.3f}>0.30、方向一致率{dir_match*100:.1f}%>60%")
    elif gap_r2 > 0.10:
        verdict = "MODERATE_SIGNAL"
        print(f"判定: MODERATE_SIGNAL -- ギャップR2={gap_r2:.3f}>0.10")
    else:
        verdict = "WEAK_SIGNAL"
        print(f"判定: WEAK_SIGNAL -- ギャップR2={gap_r2:.3f}<0.10")

    results["verdict"] = verdict

    # JSON保存
    output_path = Path("logs/iterations/T4-Q02_overnight-gap-futures-signal_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果を保存: {output_path}")


if __name__ == "__main__":
    main()
