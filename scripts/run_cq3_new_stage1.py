"""cq3新 Stage 1: 保有者の不安定さ仮説 --回転率の独自効果確認

前回cq3 v2と同一パネル構築ロジックを使うが、回帰モデルから信用残を除外。
Model A: excess_ret ~ beta + vol + scale_dummies + ret5d（turnoverなし）
Model B: excess_ret ~ beta + vol + turnover + scale_dummies + ret5d

判定基準:
  主判定: (1)turnover係数が負 (2)1-way cluster SEでp<0.10 (3)有効イベント数>=20
  anti-snooping: (4)期間2分割で両方符号負 (5)各期間で係数絶対値>=全期間の25% (6)増分R2>=0.1%
  再現確認: vol_p~0.026+-0.02, beta_p>0.80 をハードゲート

Usage:
    python scripts/run_cq3_new_stage1.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import JQUANTS_API_KEY, MARKET_DATA_DIR
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_data(provider, start, end, n_stocks):
    """前回cq3と同一のデータ取得。"""
    import yfinance as yf
    from core.stock_allocation.event_study import get_jp_top_stocks
    from core.stock_allocation import _fetch_stock_prices

    print("S&P500リターンを取得中...")
    sp500 = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    sp500_close = sp500["Close"].squeeze()
    if hasattr(sp500_close.index, "tz") and sp500_close.index.tz is not None:
        sp500_close.index = sp500_close.index.tz_localize(None)
    us_returns = sp500_close.pct_change().dropna()
    print(f"  S&P500: {len(us_returns)}日")

    print("JP銘柄リストを取得中...")
    listed = provider.get_listed_stocks()
    jp_codes, _ = get_jp_top_stocks(provider, n=n_stocks)
    print(f"  JP銘柄: {len(jp_codes)}銘柄")

    print("JP株価を取得中...")
    jp_prices_raw = _fetch_stock_prices(
        provider, jp_codes, start, end,
        progress_callback=lambda m, p: print(f"  {m}") if p % 0.2 < 0.01 else None,
    )
    close_col = "adj_close" if "adj_close" in jp_prices_raw.columns else "close"
    jp_close = jp_prices_raw.pivot(index="date", columns="code", values=close_col)
    jp_returns = jp_close.pct_change(fill_method=None).dropna(how="all")
    print(f"  JP株価: {len(jp_returns)}日 x {len(jp_returns.columns)}銘柄")

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
    print(f"  TOPIX: {len(topix_returns)}日")

    return us_returns, jp_returns, topix_returns, jp_prices_raw, listed, jp_codes


def build_panel(us_returns, jp_returns, topix_returns, jp_prices_raw, listed, event_dates):
    """パネル構築。margin_event_studyと同一ロジックだが信用残を含めない。"""
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


def run_regression(panel, include_turnover):
    """demean->Winsorize->OLS(1-way cluster SE)。"""
    controls = ["beta", "vol", "scale_large70", "scale_mid400",
                "scale_small1", "scale_small2", "ret5d"]
    if include_turnover:
        xvars = ["turnover"] + controls
    else:
        xvars = controls

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
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": panel["event_date"]})


def main():
    parser = argparse.ArgumentParser(description="cq3新 Stage 1")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--n-stocks", type=int, default=200)
    args = parser.parse_args()

    cache = DataCache(MARKET_DATA_DIR)
    provider = JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)

    print("=== cq3新 Stage 1: 回転率の独自効果確認 ===")
    print(f"期間: {args.start} - {args.end}")
    print(f"設計: 前回cq3 v2と同一パネル。信用残を除外。turnover主判定")
    print()

    # データ取得
    us_returns, jp_returns, topix_returns, jp_prices_raw, listed, jp_codes = \
        fetch_data(provider, args.start, args.end, args.n_stocks)

    # イベント日を全期間で固定
    us_mean = us_returns.mean()
    us_std = us_returns.std()
    threshold = us_mean - args.threshold * us_std
    event_dates = us_returns[us_returns <= threshold].index.sort_values()
    print(f"\nイベント日: {len(event_dates)}日 (閾値: {threshold:.4f})")

    # パネル構築（信用残なし）
    print("パネル構築中...")
    panel = build_panel(us_returns, jp_returns, topix_returns, jp_prices_raw, listed, event_dates)
    n_effective_events = panel["event_date"].nunique()
    n_obs = len(panel)
    print(f"パネル: {n_obs}観測点 ({n_effective_events}有効イベント日)")

    if n_obs < 100:
        print(f"エラー: 観測点不足({n_obs})")
        return

    # --- Model A（turnoverなし）& Model B（turnoverあり）---
    print("\n回帰を実行中...")
    model_a = run_regression(panel, include_turnover=False)
    model_b = run_regression(panel, include_turnover=True)

    vol_p_a = float(model_a.pvalues.get("vol", np.nan))
    beta_p_a = float(model_a.pvalues.get("beta", np.nan))
    turnover_coef = float(model_b.params.get("turnover", np.nan))
    turnover_p = float(model_b.pvalues.get("turnover", np.nan))
    turnover_t = float(model_b.tvalues.get("turnover", np.nan))
    incr_r2 = model_b.rsquared - model_a.rsquared

    # --- 再現確認（ハードゲート）---
    print("\n" + "=" * 60)
    print("=== 再現確認（ハードゲート）===")
    print("=" * 60)
    print(f"vol p値:   {vol_p_a:.4f}  (前回: 0.026, 許容: ±0.02)")
    print(f"beta p値:  {beta_p_a:.4f}  (前回: 0.90, 許容: >0.80)")

    repro_fail = False
    if beta_p_a < 0.10:
        print("NG: betaが有意化。パネル構築にバグの可能性。中止")
        repro_fail = True
    if vol_p_a > 0.10:
        print("NG: volが非有意化。パネル構築にバグの可能性。中止")
        repro_fail = True
    if repro_fail:
        return

    print("OK: 再現確認パス")

    # --- 主判定 ---
    print("\n" + "=" * 60)
    print("=== Stage 1 主判定 ===")
    print("=" * 60)
    print(f"\nModel A (turnoverなし): R2={model_a.rsquared:.4f}")
    print(f"Model B (turnoverあり): R2={model_b.rsquared:.4f}  増分R2={incr_r2:.4f}")
    print()
    print("Model B 係数:")
    for name in model_b.params.index:
        if name.startswith("scale_"):
            continue
        marker = " <-- 主判定" if name == "turnover" else ""
        print(f"  {name:12s}: coef={model_b.params[name]:+.6f}  t={model_b.tvalues[name]:+.3f}  p={model_b.pvalues[name]:.4f}{marker}")

    # --- 期間2分割（固定イベント日を分割）---
    print("\n--- 期間2分割チェック ---")
    event_list = sorted(panel["event_date"].unique())
    mid = len(event_list) // 2
    first_dates = set(event_list[:mid])
    second_dates = set(event_list[mid:])

    split_results = {}
    for label, date_set in [("first_half", first_dates), ("second_half", second_dates)]:
        sub_panel = panel[panel["event_date"].isin(date_set)]
        if len(sub_panel) < 50:
            split_results[label] = {"error": f"観測不足({len(sub_panel)})"}
            continue
        try:
            sub_model = run_regression(sub_panel, include_turnover=True)
            split_results[label] = {
                "turnover_coef": float(sub_model.params.get("turnover", np.nan)),
                "turnover_p": float(sub_model.pvalues.get("turnover", np.nan)),
                "n_events": int(sub_panel["event_date"].nunique()),
                "n_obs": len(sub_panel),
            }
        except Exception as e:
            split_results[label] = {"error": str(e)}

    for label, sr in split_results.items():
        if "error" in sr:
            print(f"  {label}: エラー --{sr['error']}")
        else:
            print(f"  {label}: coef={sr['turnover_coef']:+.6f}  p={sr['turnover_p']:.4f}  ({sr['n_events']}日, {sr['n_obs']}obs)")

    # --- 総合判定 ---
    print("\n" + "=" * 60)
    print("=== 総合判定 ===")
    print("=" * 60)

    sign_ok = turnover_coef < 0
    p_ok = turnover_p < 0.10
    events_ok = n_effective_events >= 20

    fh = split_results.get("first_half", {})
    sh = split_results.get("second_half", {})
    fh_coef = fh.get("turnover_coef", float("nan"))
    sh_coef = sh.get("turnover_coef", float("nan"))
    both_neg = (not np.isnan(fh_coef) and fh_coef < 0) and (not np.isnan(sh_coef) and sh_coef < 0)
    coef_abs = abs(turnover_coef) if not np.isnan(turnover_coef) else 0
    fh_suf = abs(fh_coef) >= coef_abs * 0.25 if (not np.isnan(fh_coef) and coef_abs > 0) else False
    sh_suf = abs(sh_coef) >= coef_abs * 0.25 if (not np.isnan(sh_coef) and coef_abs > 0) else False
    incr_ok = incr_r2 >= 0.001

    print(f"主判定:")
    print(f"  (1) turnover係数が負:  {'PASS' if sign_ok else 'FAIL'} ({turnover_coef:+.6f})")
    print(f"  (2) p<0.10:            {'PASS' if p_ok else 'FAIL'} (p={turnover_p:.4f})")
    print(f"  (3) 有効イベント>=20:   {'PASS' if events_ok else 'FAIL'} ({n_effective_events}日)")
    print(f"anti-snooping:")
    print(f"  (4) 両期間で符号負:    {'PASS' if both_neg else 'FAIL'}")
    if coef_abs > 0:
        print(f"  (5) 各期間>=25%:        {'PASS' if fh_suf and sh_suf else 'FAIL'} (前半{abs(fh_coef)/coef_abs*100:.0f}% 後半{abs(sh_coef)/coef_abs*100:.0f}%)")
    else:
        print(f"  (5) 各期間>=25%:        FAIL (全期間係数がゼロ)")
    print(f"  (6) 増分R2>=0.1%:       {'PASS' if incr_ok else 'FAIL'} ({incr_r2:.4f})")

    primary = sign_ok and p_ok and events_ok
    anti_snoop = both_neg and fh_suf and sh_suf and incr_ok

    if primary and anti_snoop:
        verdict = "Stage 1通過。Stage 2に進む"
    elif primary:
        verdict = "主判定通過だがanti-snooping不十分。Stage 2は要検討"
    else:
        verdict = "Stage 1不通過。Stage 2に進まない"

    print(f"\n結論: {verdict}")
    print("=" * 60)

    # JSON保存
    output = {
        "experiment": "cq3_new_stage1",
        "design": "前回cq3 v2と同一パネル。信用残除外。turnover主判定",
        "reproduction": {"vol_p": vol_p_a, "beta_p": beta_p_a, "pass": not repro_fail},
        "model_a": {"r2": float(model_a.rsquared)},
        "model_b": {
            "r2": float(model_b.rsquared), "incremental_r2": float(incr_r2),
            "turnover_coef": turnover_coef, "turnover_p": turnover_p, "turnover_t": turnover_t,
        },
        "split_period": split_results,
        "judgment": {
            "primary": {"sign_ok": sign_ok, "p_ok": p_ok, "events_ok": events_ok},
            "anti_snooping": {"both_neg": both_neg, "fh_suf": fh_suf, "sh_suf": sh_suf, "incr_ok": incr_ok},
        },
        "verdict": verdict,
        "n_effective_events": n_effective_events, "n_observations": n_obs,
        "all_coefficients": {
            name: {"coef": float(model_b.params[name]), "p": float(model_b.pvalues[name]), "t": float(model_b.tvalues[name])}
            for name in model_b.params.index
        },
    }
    output_path = Path("logs/iterations/cq3_new_stage1_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果を保存: {output_path}")


if __name__ == "__main__":
    main()
