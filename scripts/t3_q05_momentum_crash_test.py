"""T3-Q05: K37リアルワールドテスト（N=1アウトオブサンプル確認）

Step 1: KS検定 — 3月第1週の下落銘柄に2月勝者が過剰集中しているか
Step 2: モメンタム上位20% vs 下位20%のday+1～3月末リターン差
Step 3: K37仮想適用 — ショック日のturnover-high銘柄をday+1解消した場合の損失軽減幅
"""

import json
import sys
import os
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
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

    print("=== T3-Q05: K37リアルワールドテスト ===\n")

    # --- データ取得 ---
    print("データ取得中...")
    listed = provider.get_listed_stocks()
    jp_codes = listed["code"].tolist()[:200]

    from core.stock_allocation import _fetch_stock_prices
    jp_prices = _fetch_stock_prices(provider, jp_codes, "2026-01-01", "2026-03-31", cache=cache)
    close_col = "adj_close" if "adj_close" in jp_prices.columns else "close"
    jp_close = jp_prices.pivot(index="date", columns="code", values=close_col)
    jp_returns = jp_close.pct_change(fill_method=None).dropna(how="all")
    print(f"  JP日次: {jp_returns.shape}")

    # S&P500
    sp = yf.download("^GSPC", start="2026-01-01", end="2026-04-01", progress=False)
    sp_ret = sp["Close"].squeeze().pct_change().dropna()
    sp_ret.index = sp_ret.index.tz_localize(None) if hasattr(sp_ret.index, 'tz') and sp_ret.index.tz else sp_ret.index

    # 日経225
    nk = yf.download("^N225", start="2026-01-01", end="2026-04-01", progress=False)
    nk_ret = nk["Close"].squeeze().pct_change().dropna()
    nk_ret.index = nk_ret.index.tz_localize(None) if hasattr(nk_ret.index, 'tz') and nk_ret.index.tz else nk_ret.index

    # --- 月次リターン計算 ---
    feb_start, feb_end = pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-28")
    feb_rets = jp_returns[(jp_returns.index >= feb_start) & (jp_returns.index <= feb_end)]
    feb_monthly = (1 + feb_rets).prod() - 1

    # 有効銘柄
    valid_codes = [c for c in jp_codes if not np.isnan(feb_monthly.get(c, np.nan)) and feb_rets[c].count() >= len(feb_rets) * 0.8]
    print(f"  有効銘柄: {len(valid_codes)}")

    # --- Step 1: KS検定 ---
    print("\n--- Step 1: KS検定（3月第1週の下落銘柄に2月勝者が過剰集中？） ---")

    w1_start, w1_end = pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-06")
    w1_rets = jp_returns[(jp_returns.index >= w1_start) & (jp_returns.index <= w1_end)]
    w1_weekly = (1 + w1_rets).prod() - 1

    all_feb = np.array([float(feb_monthly[c]) for c in valid_codes if not np.isnan(feb_monthly[c])])
    decline_codes = [c for c in valid_codes if not np.isnan(w1_weekly.get(c, np.nan)) and w1_weekly[c] < 0]
    decline_feb = np.array([float(feb_monthly[c]) for c in decline_codes])

    n_decline = len(decline_codes)
    n_all = len(valid_codes)
    print(f"  全銘柄: {n_all}, 3月第1週下落銘柄: {n_decline}")

    if n_decline >= 10 and n_all >= 20:
        ks_stat, ks_p = scipy_stats.ks_2samp(decline_feb, all_feb)
        print(f"  KS検定: stat={ks_stat:.4f}, p={ks_p:.4f}")

        # 2月上位20%の比率
        top20_threshold = np.percentile(all_feb, 80)
        top20_in_decline = sum(1 for f in decline_feb if f >= top20_threshold)
        top20_ratio = top20_in_decline / n_decline
        print(f"  2月上位20%閾値: {top20_threshold*100:+.2f}%")
        print(f"  下落銘柄中の2月上位20%比率: {top20_ratio*100:.1f}% ({top20_in_decline}/{n_decline})")
        print(f"  期待値(一律下落なら): 20.0%")

        step1 = {
            "n_all": n_all, "n_decline": n_decline,
            "ks_stat": float(ks_stat), "ks_p": float(ks_p),
            "top20_threshold": float(top20_threshold),
            "top20_in_decline": top20_in_decline,
            "top20_ratio": float(top20_ratio),
        }
    else:
        step1 = {"status": "insufficient_data"}
        print("  データ不足")

    # --- Step 2: モメンタム上位20% vs 下位20% ---
    print("\n--- Step 2: 2月モメンタム上位20% vs 下位20%のday+1～3月末リターン ---")

    feb_sorted = sorted(valid_codes, key=lambda c: float(feb_monthly.get(c, 0)), reverse=True)
    n20 = max(len(feb_sorted) // 5, 1)
    top20_codes = feb_sorted[:n20]
    bot20_codes = feb_sorted[-n20:]

    # day+1(3/3)～3月末のリターン
    post_start = pd.Timestamp("2026-03-03")
    post_end = pd.Timestamp("2026-03-31")
    post_rets = jp_returns[(jp_returns.index >= post_start) & (jp_returns.index <= post_end)]
    post_monthly = (1 + post_rets).prod() - 1

    top20_post = np.array([float(post_monthly.get(c, np.nan)) for c in top20_codes if not np.isnan(post_monthly.get(c, np.nan))])
    bot20_post = np.array([float(post_monthly.get(c, np.nan)) for c in bot20_codes if not np.isnan(post_monthly.get(c, np.nan))])

    if len(top20_post) >= 5 and len(bot20_post) >= 5:
        top_mean = float(np.mean(top20_post))
        bot_mean = float(np.mean(bot20_post))
        diff = top_mean - bot_mean
        t_stat, t_p = scipy_stats.ttest_ind(top20_post, bot20_post, equal_var=False)
        print(f"  上位20% (n={len(top20_post)}): day+1～3月末 平均{top_mean*100:+.3f}%")
        print(f"  下位20% (n={len(bot20_post)}): day+1～3月末 平均{bot_mean*100:+.3f}%")
        print(f"  差(上位-下位): {diff*100:+.3f}% (t={t_stat:.3f}, p={t_p:.4f})")

        step2 = {
            "n_top": len(top20_post), "n_bot": len(bot20_post),
            "top_mean": top_mean, "bot_mean": bot_mean,
            "diff": diff, "t_stat": float(t_stat), "p": float(t_p),
        }
    else:
        step2 = {"status": "insufficient_data"}
        print("  データ不足")

    # --- Step 3: K37仮想適用 ---
    print("\n--- Step 3: K37仮想適用（ショック日のturnover-high銘柄をday+1解消） ---")

    # ショック判定
    # S&P500基準: 2026年のσを使って-2sigma
    sp_2026 = sp_ret[(sp_ret.index >= pd.Timestamp("2026-01-01"))]
    sp_mean = float(sp_2026.mean())
    sp_std = float(sp_2026.std())
    sp_threshold = sp_mean - 2 * sp_std

    # 日経基準
    nk_2026 = nk_ret[(nk_ret.index >= pd.Timestamp("2026-01-01"))]
    nk_mean = float(nk_2026.mean())
    nk_std = float(nk_2026.std())
    nk_threshold = nk_mean - 2 * nk_std

    print(f"  S&P500 -2sigma閾値: {sp_threshold*100:+.3f}%")
    print(f"  日経225 -2sigma閾値: {nk_threshold*100:+.3f}%")

    # 3月のショック日
    mar_sp = sp_ret[(sp_ret.index >= pd.Timestamp("2026-03-01")) & (sp_ret.index <= pd.Timestamp("2026-03-31"))]
    mar_nk = nk_ret[(nk_ret.index >= pd.Timestamp("2026-03-01")) & (nk_ret.index <= pd.Timestamp("2026-03-31"))]

    sp_shock_days = mar_sp[mar_sp < sp_threshold].index.tolist()
    nk_shock_days = mar_nk[mar_nk < nk_threshold].index.tolist()
    all_shock_days = sorted(set(sp_shock_days + nk_shock_days))

    print(f"  S&P500基準ショック日: {len(sp_shock_days)}日 {[d.strftime('%m/%d') for d in sp_shock_days]}")
    print(f"  日経基準ショック日: {len(nk_shock_days)}日 {[d.strftime('%m/%d') for d in nk_shock_days]}")

    # turnover計算（簡易: 出来高の20日平均からの乖離）
    vol_col = "volume" if "volume" in jp_prices.columns else "adj_volume"
    if vol_col in jp_prices.columns:
        jp_vol = jp_prices.pivot(index="date", columns="code", values=vol_col)
    else:
        jp_vol = None

    step3 = {"sp_shock_days": [str(d) for d in sp_shock_days], "nk_shock_days": [str(d) for d in nk_shock_days]}

    if jp_vol is not None and len(all_shock_days) > 0:
        # K37仮想適用: ショック日のturnover-high銘柄のday+1以降リターン
        k37_savings = []
        for shock_day in all_shock_days:
            # turnover: ショック日の出来高 / 前20日平均
            shock_vol = jp_vol.loc[shock_day] if shock_day in jp_vol.index else None
            if shock_vol is None:
                continue
            pre_start = shock_day - pd.Timedelta(days=30)
            pre_vol = jp_vol[(jp_vol.index >= pre_start) & (jp_vol.index < shock_day)].tail(20)
            if len(pre_vol) < 10:
                continue
            avg_vol = pre_vol.mean()
            ratio = shock_vol / avg_vol.replace(0, np.nan)

            # turnover-high: 中央値以上
            ratio_valid = ratio.dropna()
            if len(ratio_valid) < 20:
                continue
            median_ratio = float(ratio_valid.median())
            high_codes = [c for c in ratio_valid.index if ratio_valid[c] > median_ratio and c in valid_codes]

            # day+1以降5日間のリターン
            jp_dates_after = jp_returns.index[jp_returns.index > shock_day]
            if len(jp_dates_after) < 5:
                continue
            post_5d = jp_returns.loc[jp_dates_after[:5]]
            post_5d_ret = (1 + post_5d).prod() - 1

            high_post = np.array([float(post_5d_ret.get(c, np.nan)) for c in high_codes if not np.isnan(post_5d_ret.get(c, np.nan))])
            if len(high_post) < 5:
                continue

            # K37適用: turnover-high銘柄をday+1で解消 → day+1以降のリターンを回避
            avoided_loss = float(np.mean(high_post))
            print(f"  {shock_day.strftime('%Y-%m-%d')}: turnover-high {len(high_post)}銘柄, day+1〜5d平均ret={avoided_loss*100:+.3f}%")
            k37_savings.append({
                "shock_day": str(shock_day),
                "n_high": len(high_post),
                "avg_post_5d_ret": avoided_loss,
            })

        if k37_savings:
            total_avoided = float(np.mean([s["avg_post_5d_ret"] for s in k37_savings]))
            print(f"\n  K37仮想適用の平均損失軽減: {total_avoided*100:+.3f}%/イベント")
            step3["k37_savings"] = k37_savings
            step3["avg_avoided"] = total_avoided
        else:
            print("  K37適用可能なショック日なし")
            step3["k37_savings"] = []
    else:
        print("  出来高データなしまたはショック日なし")

    # --- 総合判定 ---
    print(f"\n{'='*60}")
    print("=== T3-Q05 総合判定 ===")
    print(f"{'='*60}")

    ks_p = step1.get("ks_p", 1.0)
    mom_diff = abs(step2.get("diff", 0))
    mom_p = step2.get("p", 1.0)

    if ks_p < 0.10 and mom_diff > 0.01 and mom_p < 0.20:
        verdict = "momentum_crash_confirmed"
        print(f"判定: momentum_crash_confirmed -- KS p={ks_p:.4f}, 上位-下位差={step2.get('diff',0)*100:+.2f}%")
    elif ks_p > 0.20 and mom_diff < 0.01:
        verdict = "not_applicable"
        print(f"判定: not_applicable -- 一律下落でありK37適用外。KS p={ks_p:.4f}, 差={mom_diff*100:.2f}%")
    else:
        verdict = "ambiguous"
        print(f"判定: ambiguous -- KS p={ks_p:.4f}, 差={step2.get('diff',0)*100:+.2f}%(p={mom_p:.4f})")

    print(f"\n注: N=1アウトオブサンプル確認。K37の有効性の最終判断ではない")

    # --- JSON保存 ---
    output = {
        "experiment": "T3-Q05",
        "step1_ks_test": step1,
        "step2_momentum_diff": step2,
        "step3_k37_simulation": step3,
        "verdict": verdict,
        "note": "N=1 out-of-sample confirmation. Not a definitive test of K37.",
    }

    output_path = Path("logs/iterations/T3-Q05_momentum-crash-realworld-test_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果を保存: {output_path}")


if __name__ == "__main__":
    main()
