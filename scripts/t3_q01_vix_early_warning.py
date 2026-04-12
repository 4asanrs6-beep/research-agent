"""T3-Q01: VIXの段階的上昇は早期警戒指標として機能したか（N=1ケーススタディ）

Step 1: モメンタムLS(前月上位20%ロング/下位20%ショート)の日次リターン構築
Step 2: VIX推移とLS日次リターンの並列分析。VIX20超持続開始日 vs LS負転日
Step 3: VIX閾値(15/20/25/30)グリッドサーチでリスク縮小シミュレーション
Step 4: 原油ショック(2/28)前後の記述
"""

import json
import sys
import os
import numpy as np
import pandas as pd
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

    print("=== T3-Q01: VIX早期警戒シミュレーション ===\n")

    # --- データ取得 ---
    print("データ取得中...")
    listed = provider.get_listed_stocks()
    jp_codes = listed["code"].tolist()[:200]

    from core.stock_allocation import _fetch_stock_prices
    jp_prices = _fetch_stock_prices(provider, jp_codes, "2025-12-01", "2026-03-31", cache=cache)
    close_col = "adj_close" if "adj_close" in jp_prices.columns else "close"
    jp_close = jp_prices.pivot(index="date", columns="code", values=close_col)
    jp_returns = jp_close.pct_change(fill_method=None).dropna(how="all")
    print(f"  JP日次: {jp_returns.shape}")

    # VIX
    vix_raw = yf.download("^VIX", start="2026-01-01", end="2026-04-01", progress=False)
    vix = vix_raw["Close"].squeeze()
    if hasattr(vix.index, 'tz') and vix.index.tz is not None:
        vix.index = vix.index.tz_localize(None)
    print(f"  VIX: {len(vix)}日")

    results = {}

    # --- Step 1: モメンタムLS日次リターン構築 ---
    print("\n--- Step 1: モメンタムLS日次リターン構築 ---")

    # 各月のポートフォリオ: 前月リターンで上位/下位20%を決定
    months = [
        ("2026-01", "2025-12-01", "2025-12-31", "2026-01-01", "2026-01-31"),
        ("2026-02", "2026-01-01", "2026-01-31", "2026-02-01", "2026-02-28"),
        ("2026-03", "2026-02-01", "2026-02-28", "2026-03-01", "2026-03-31"),
    ]

    ls_daily = {}
    for label, prev_start, prev_end, cur_start, cur_end in months:
        prev_rets = jp_returns[(jp_returns.index >= pd.Timestamp(prev_start)) & (jp_returns.index <= pd.Timestamp(prev_end))]
        cur_rets = jp_returns[(jp_returns.index >= pd.Timestamp(cur_start)) & (jp_returns.index <= pd.Timestamp(cur_end))]

        prev_monthly = (1 + prev_rets).prod() - 1
        valid = [c for c in jp_codes if not np.isnan(prev_monthly.get(c, np.nan))]
        if len(valid) < 20:
            continue

        sorted_codes = sorted(valid, key=lambda c: float(prev_monthly.get(c, 0)), reverse=True)
        n20 = max(len(sorted_codes) // 5, 1)
        long_codes = sorted_codes[:n20]
        short_codes = sorted_codes[-n20:]

        for d in cur_rets.index:
            long_ret = cur_rets.loc[d, long_codes].dropna().mean() if len(long_codes) > 0 else 0
            short_ret = cur_rets.loc[d, short_codes].dropna().mean() if len(short_codes) > 0 else 0
            ls_daily[d] = float(long_ret - short_ret)  # ロング-ショート

    ls_series = pd.Series(ls_daily).sort_index()
    ls_cum = (1 + ls_series).cumprod() - 1
    ls_ma5 = ls_series.rolling(5).mean()
    print(f"  モメンタムLS日次リターン: {len(ls_series)}日")
    print(f"  1月累積: {ls_cum[ls_cum.index < pd.Timestamp('2026-02-01')].iloc[-1]*100:+.2f}%" if len(ls_cum[ls_cum.index < pd.Timestamp('2026-02-01')]) > 0 else "  1月: データなし")
    print(f"  2月累積: {((1+ls_series[(ls_series.index >= pd.Timestamp('2026-02-01')) & (ls_series.index <= pd.Timestamp('2026-02-28'))]).prod()-1)*100:+.2f}%")
    print(f"  3月累積: {((1+ls_series[(ls_series.index >= pd.Timestamp('2026-03-01'))]).prod()-1)*100:+.2f}%")

    results["ls_monthly"] = {
        "jan": float((1 + ls_series[(ls_series.index >= pd.Timestamp('2026-01-01')) & (ls_series.index < pd.Timestamp('2026-02-01'))]).prod() - 1),
        "feb": float((1 + ls_series[(ls_series.index >= pd.Timestamp('2026-02-01')) & (ls_series.index <= pd.Timestamp('2026-02-28'))]).prod() - 1),
        "mar": float((1 + ls_series[(ls_series.index >= pd.Timestamp('2026-03-01'))]).prod() - 1),
    }

    # --- Step 2: VIX推移とLS並列分析 ---
    print("\n--- Step 2: VIX推移 vs モメンタムLS ---")

    # VIX20超が持続した最初の日
    vix_above_20 = vix[vix > 20]
    if len(vix_above_20) > 0:
        # 3日連続20超の最初の日を探す
        consecutive = 0
        first_sustained_20 = None
        for d in vix.index:
            if vix[d] > 20:
                consecutive += 1
                if consecutive >= 3 and first_sustained_20 is None:
                    first_sustained_20 = vix.index[vix.index.get_loc(d) - 2]  # 3日連続の最初の日
            else:
                consecutive = 0
        print(f"  VIX>20が3日連続の最初の日: {first_sustained_20}")
    else:
        first_sustained_20 = None
        print("  VIX>20の日なし")

    # LS 5日MA負転日
    ls_ma5_neg = ls_ma5[ls_ma5 < 0]
    first_ma5_neg = ls_ma5_neg.index[0] if len(ls_ma5_neg) > 0 else None
    print(f"  LS 5日MA負転日: {first_ma5_neg}")

    if first_sustained_20 and first_ma5_neg:
        if first_sustained_20 < first_ma5_neg:
            print(f"  ** VIXが先行 ({first_sustained_20} < {first_ma5_neg}) **")
            results["vix_leads"] = True
        else:
            print(f"  ** LS負転が先行 ({first_ma5_neg} < {first_sustained_20}) **")
            results["vix_leads"] = False
        results["vix_20_sustained_date"] = str(first_sustained_20)
        results["ls_ma5_neg_date"] = str(first_ma5_neg)

    # --- Step 3: VIX閾値グリッドサーチ ---
    print("\n--- Step 3: VIX閾値グリッドサーチ ---")

    # 基準: VIXなし(フルリスク)の3月累積リターン
    mar_ls = ls_series[(ls_series.index >= pd.Timestamp('2026-03-01'))]
    full_loss = float(mar_ls.sum())  # 近似的に日次リターンの和

    # 1-3月全体の累積損失
    total_loss = float(ls_series.sum())

    print(f"  3月LS累積損失(近似): {full_loss*100:+.3f}%")
    print(f"  1-3月LS全期間損失(近似): {total_loss*100:+.3f}%")

    grid_results = {}
    for threshold in [15, 20, 25, 30]:
        for reduction in [0.5, 1.0]:
            # VIXが閾値超の日はリスクを(1-reduction)に縮小
            adjusted = []
            for d in ls_series.index:
                vix_d = vix.get(d, np.nan)
                if np.isnan(vix_d):
                    # VIXがない日(日米市場ずれ)はそのまま
                    adjusted.append(float(ls_series[d]))
                elif vix_d > threshold:
                    adjusted.append(float(ls_series[d]) * (1 - reduction))
                else:
                    adjusted.append(float(ls_series[d]))

            adjusted_series = pd.Series(adjusted, index=ls_series.index)
            adjusted_loss = float(adjusted_series.sum())
            saved = total_loss - adjusted_loss
            saved_pct = saved / abs(total_loss) * 100 if abs(total_loss) > 1e-6 else 0

            # VIX超過日数
            n_above = int((vix > threshold).sum())

            key = f"VIX>{threshold}_reduce{int(reduction*100)}%"
            print(f"  {key}: 損失{adjusted_loss*100:+.3f}%, 軽減{saved*100:+.3f}% ({saved_pct:+.1f}%), 超過{n_above}日")
            grid_results[key] = {
                "threshold": threshold,
                "reduction_pct": int(reduction * 100),
                "adjusted_loss": adjusted_loss,
                "saved": saved,
                "saved_pct_of_total": saved_pct,
                "n_days_above": n_above,
            }

    results["grid_search"] = grid_results

    # --- Step 4: 原油ショック前後の記述 ---
    print("\n--- Step 4: 原油ショック(2/28)前後のVIX-LS関係 ---")
    pre_shock = ls_series[ls_series.index < pd.Timestamp("2026-02-28")]
    post_shock = ls_series[ls_series.index >= pd.Timestamp("2026-02-28")]
    vix_pre = vix[vix.index < pd.Timestamp("2026-02-28")]
    vix_post = vix[vix.index >= pd.Timestamp("2026-02-28")]

    print(f"  2/28前: LS平均日次{pre_shock.mean()*100:+.4f}%, VIX平均{vix_pre.mean():.1f}")
    print(f"  2/28後: LS平均日次{post_shock.mean()*100:+.4f}%, VIX平均{vix_post.mean():.1f}")

    # 2月中のVIX上昇期にLSはまだ正だったか
    feb_ls = ls_series[(ls_series.index >= pd.Timestamp("2026-02-01")) & (ls_series.index <= pd.Timestamp("2026-02-28"))]
    feb_vix = vix[(vix.index >= pd.Timestamp("2026-02-01")) & (vix.index <= pd.Timestamp("2026-02-28"))]
    print(f"  2月: LS累積{feb_ls.sum()*100:+.3f}%, VIX平均{feb_vix.mean():.1f}(月初{feb_vix.iloc[0]:.1f}→月末{feb_vix.iloc[-1]:.1f})")

    results["pre_post_shock"] = {
        "pre_ls_mean": float(pre_shock.mean()),
        "post_ls_mean": float(post_shock.mean()),
        "pre_vix_mean": float(vix_pre.mean()),
        "post_vix_mean": float(vix_post.mean()),
        "feb_ls_cum": float(feb_ls.sum()),
        "feb_vix_start": float(feb_vix.iloc[0]),
        "feb_vix_end": float(feb_vix.iloc[-1]),
    }

    # --- 総合判定 ---
    print(f"\n{'='*60}")
    print("=== T3-Q01 総合判定 ===")
    print(f"{'='*60}")

    # 最も軽減率が高い閾値を特定
    best_key = max(grid_results, key=lambda k: grid_results[k]["saved_pct_of_total"])
    best = grid_results[best_key]

    if best["saved_pct_of_total"] > 10:
        verdict = "vix_useful"
        print(f"判定: vix_useful -- {best_key}で損失の{best['saved_pct_of_total']:.1f}%を軽減可能")
    else:
        verdict = "vix_not_useful"
        print(f"判定: vix_not_useful -- 最良の閾値({best_key})でも軽減{best['saved_pct_of_total']:.1f}%<10%")

    print(f"\n** N=1ケーススタディ。一般化は禁止。過去データでのバックテストが必要 **")

    results["verdict"] = verdict
    results["best_threshold"] = best_key
    results["note"] = "N=1 case study. Do not generalize. Requires backtest on historical data."

    # JSON保存
    output_path = Path("logs/iterations/T3-Q01_vix-creep-early-signal_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果を保存: {output_path}")


if __name__ == "__main__":
    main()
