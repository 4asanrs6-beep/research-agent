"""symmetry-test: 上方/下方ショックの対称性検証

問い: 知見パターン（vol有意、turnover有意、beta非有意、margin非有意）は
      米国急騰日でも対称的に成立するか。対称なら増幅器、非対称ならパニック固有。

設計:
  - 4係数（vol, turnover, beta, margin_ratio）を上方/下方で比較
  - 主判定: pooled panelのdirection x variable交互作用のp値
  - 層別: VIX変化（前日比）+ サイズ（大型/中小型）
  - 閾値: +-2%, +-1.5%, +-1sigma の3つでロバストネス

Usage:
    python scripts/run_symmetry_test.py
"""

import argparse
import json
import sys
import os
import logging
from pathlib import Path
from datetime import date

# バッファリングを無効化（バックグラウンド実行時にログが見えるように）
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import JQUANTS_API_KEY, MARKET_DATA_DIR
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --- holder-instability Stage 1/2 から流用 ---

def fetch_data(provider, start, end, n_stocks):
    """データ取得。VIXも追加。"""
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

    print("VIXを取得中...")
    vix_raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    vix_close = vix_raw["Close"].squeeze()
    if hasattr(vix_close.index, "tz") and vix_close.index.tz is not None:
        vix_close.index = vix_close.index.tz_localize(None)
    vix_change = vix_close.pct_change().dropna()
    vix_change.name = "vix_change"
    print(f"  VIX: {len(vix_change)}")

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

    return us_returns, jp_returns, topix_returns, jp_prices_raw, listed, jp_codes, vix_change


def fetch_margin_data(provider, jp_codes, start, end):
    """信用残データ取得。"""
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
    print(f"  信用買残: {margin_data['code'].nunique()}銘柄")
    return margin_data


def build_panel(us_returns, jp_returns, topix_returns, jp_prices_raw, listed, event_dates):
    """パネル構築（Stage 1と同一）。"""
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
    """Stage 2から流用。信用残比率を追加。"""
    if margin_data.empty:
        panel = panel.copy()
        panel["margin_ratio"] = np.nan
        return panel
    margin_pivot = margin_data.pivot_table(index="date", columns="code", values="margin_buy_balance")
    margin_pivot = margin_pivot.sort_index()
    ratios = []
    for _, row in panel.iterrows():
        code = row["code"]
        event_date = row["event_date"]
        if code not in margin_pivot.columns:
            ratios.append(np.nan)
            continue
        valid = margin_pivot.loc[:event_date, code].dropna()
        if len(valid) == 0:
            ratios.append(np.nan)
            continue
        margin_bal = valid.iloc[-1]
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


def run_regression_with_vars(panel, xvars):
    """demean -> Winsorize -> OLS(1-way cluster SE)。"""
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


def extract_events_sigma(us_returns, threshold_sigma, direction):
    """sigma閾値でイベント日を抽出。"""
    us_mean = us_returns.mean()
    us_std = us_returns.std()
    if direction == "down":
        threshold = us_mean - threshold_sigma * us_std
        return us_returns[us_returns <= threshold].index.sort_values()
    else:
        threshold = us_mean + threshold_sigma * us_std
        return us_returns[us_returns >= threshold].index.sort_values()


def extract_events_pct(us_returns, threshold_pct, direction):
    """固定%閾値でイベント日を抽出。"""
    if direction == "down":
        return us_returns[us_returns <= -threshold_pct / 100].index.sort_values()
    else:
        return us_returns[us_returns >= threshold_pct / 100].index.sort_values()


def extract_coefs(model, vars_of_interest):
    """モデルから対象変数の係数/p値/t値を抽出。"""
    result = {}
    for var in vars_of_interest:
        result[var] = {
            "coef": float(model.params.get(var, np.nan)),
            "p": float(model.pvalues.get(var, np.nan)),
            "t": float(model.tvalues.get(var, np.nan)),
        }
    return result


def add_abnormal_volume_share(panel, jp_prices_raw, baseline_window=60, event_window=5):
    """注目集中proxy: 直近5日出来高シェア / 過去60日平均出来高シェアの異常度。"""
    vol_col = "volume" if "volume" in jp_prices_raw.columns else "adj_volume"
    if vol_col not in jp_prices_raw.columns:
        panel = panel.copy()
        panel["abnormal_vol_share"] = np.nan
        return panel

    # 全銘柄の日次出来高テーブル
    vol_pivot = jp_prices_raw.pivot_table(index="date", columns="code", values=vol_col)
    vol_pivot = vol_pivot.sort_index()
    daily_total = vol_pivot.sum(axis=1)
    daily_share = vol_pivot.div(daily_total, axis=0)

    ratios = []
    for _, row in panel.iterrows():
        code = row["code"]
        event_date = row["event_date"]
        if code not in daily_share.columns:
            ratios.append(np.nan)
            continue
        col = daily_share[code]
        before = col[col.index <= event_date]
        if len(before) < baseline_window:
            ratios.append(np.nan)
            continue
        recent = before.tail(event_window).mean()
        baseline = before.tail(baseline_window).mean()
        if baseline is None or np.isnan(baseline) or baseline < 1e-10:
            ratios.append(np.nan)
            continue
        ratios.append(recent / baseline - 1.0)

    panel = panel.copy()
    panel["abnormal_vol_share"] = ratios
    return panel


def compute_post_event_car(jp_returns, topix_returns, panel, windows=[5, 10, 20]):
    """ショック後の累積超過リターン(CAR)をパネルに追加。"""
    common_dates = jp_returns.index
    panel = panel.copy()

    for w in windows:
        cars = []
        for _, row in panel.iterrows():
            code = row["code"]
            event_date = row["event_date"]
            # JPの翌営業日から開始
            jp_dates_after = common_dates[common_dates > event_date]
            if len(jp_dates_after) < w + 1:
                cars.append(np.nan)
                continue
            # 翌日(reaction day)の翌日からw日間
            start_idx = 1  # reaction dayの翌日
            end_idx = start_idx + w
            if end_idx > len(jp_dates_after):
                cars.append(np.nan)
                continue
            post_dates = jp_dates_after[start_idx:end_idx]
            stock_rets = jp_returns.loc[post_dates, code] if code in jp_returns.columns else pd.Series(dtype=float)
            topix_rets = topix_returns.reindex(post_dates)
            excess = stock_rets - topix_rets
            valid = excess.dropna()
            if len(valid) < w // 2:
                cars.append(np.nan)
                continue
            cars.append(float(valid.sum()))
        panel[f"car_{w}d"] = cars

    return panel


def run_reversal_test(panel, windows=[5, 10, 20]):
    """事前説明変数の高低群でpost-event CARを比較。当日excess_retを統制。"""
    results = {}
    group_vars = ["vol", "turnover"]
    if "abnormal_vol_share" in panel.columns and panel["abnormal_vol_share"].notna().sum() > 100:
        group_vars.append("abnormal_vol_share")

    for gvar in group_vars:
        gvar_results = {}
        median_val = panel[gvar].median()
        high_mask = panel[gvar] > median_val
        low_mask = ~high_mask

        for w in windows:
            car_col = f"car_{w}d"
            if car_col not in panel.columns:
                continue
            valid = panel[[car_col, "excess_ret", gvar]].dropna()
            if len(valid) < 100:
                gvar_results[f"{w}d"] = {"error": "insufficient data"}
                continue

            high = valid[valid[gvar] > valid[gvar].median()]
            low = valid[valid[gvar] <= valid[gvar].median()]

            high_car = high[car_col].mean()
            low_car = low[car_col].mean()

            # 当日excess_retを統制した回帰: car ~ excess_ret + gvar_high_dummy
            valid_reg = valid.copy()
            valid_reg["high"] = (valid_reg[gvar] > valid_reg[gvar].median()).astype(float)
            try:
                model = sm.OLS(valid_reg[car_col], sm.add_constant(valid_reg[["excess_ret", "high"]])).fit()
                gvar_results[f"{w}d"] = {
                    "high_car_mean": float(high_car),
                    "low_car_mean": float(low_car),
                    "diff": float(high_car - low_car),
                    "controlled_coef": float(model.params.get("high", np.nan)),
                    "controlled_p": float(model.pvalues.get("high", np.nan)),
                    "n_high": len(high),
                    "n_low": len(low),
                }
            except Exception as e:
                gvar_results[f"{w}d"] = {"error": str(e)}

        results[gvar] = gvar_results
    return results


def main():
    parser = argparse.ArgumentParser(description="symmetry-test + attention-penalty")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--n-stocks", type=int, default=200)
    parser.add_argument("--q04", action="store_true", help="T1-Q04 attention-penalty分析を実行")
    args = parser.parse_args()

    cache = DataCache(MARKET_DATA_DIR)
    provider = JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)

    print("=== symmetry-test: 上方/下方ショックの対称性検証 ===")
    print(f"期間: {args.start} - {args.end}")
    print()

    # データ取得
    us_returns, jp_returns, topix_returns, jp_prices_raw, listed, jp_codes, vix_change = \
        fetch_data(provider, args.start, args.end, args.n_stocks)
    margin_data = fetch_margin_data(provider, jp_codes, args.start, args.end)

    # 4係数の対象変数
    controls = ["beta", "vol", "turnover", "margin_ratio",
                "scale_large70", "scale_mid400", "scale_small1", "scale_small2", "ret5d"]
    vars_of_interest = ["vol", "turnover", "beta", "margin_ratio"]

    # 複数閾値でループ。最初の閾値でStage 1との再現確認を行う
    # Stage 1は2.0sigmaで実施済み。再現ゲートはこれと照合
    thresholds = [
        ("2.0sigma", "sigma", 2.0),   # 基本閾値（Stage 1と同一）
        ("1.5pct", "pct", 1.5),       # ロバストネス
        ("1.0sigma", "sigma", 1.0),   # ロバストネス（サンプル増）
    ]

    all_results = {}
    repro_passed = False

    for thresh_label, thresh_type, thresh_val in thresholds:
        print(f"\n{'='*60}")
        print(f"=== 閾値: {thresh_label} ===")
        print(f"{'='*60}")

        # イベント日抽出（上方/下方）
        panels = {}
        for direction in ["down", "up"]:
            if thresh_type == "sigma":
                events = extract_events_sigma(us_returns, thresh_val, direction)
            else:
                events = extract_events_pct(us_returns, thresh_val, direction)

            print(f"\n--- {direction} ({len(events)}日) ---")

            # パネル構築
            panel = build_panel(us_returns, jp_returns, topix_returns, jp_prices_raw, listed, events)
            panel = add_margin_ratio(panel, margin_data, jp_prices_raw)
            panel_matched = panel.dropna(subset=["margin_ratio"])
            n_eff = panel_matched["event_date"].nunique()

            print(f"  パネル: {len(panel_matched)}obs ({n_eff}有効日)")

            # サンプルゲート: 上下とも>=20（統一）
            if n_eff < 20:
                print(f"  イベント不足({n_eff} < 20)。判定保留")
                all_results[f"{thresh_label}_{direction}"] = {
                    "n_events": n_eff, "sample_gate_pass": False, "decision_status": "hold",
                }
                continue

            # 回帰
            model = run_regression_with_vars(panel_matched, controls)
            coefs = extract_coefs(model, vars_of_interest)

            print(f"  R2={model.rsquared:.4f}")
            for var in vars_of_interest:
                c = coefs[var]
                print(f"  {var:15s}: coef={c['coef']:+.6f}  t={c['t']:+.3f}  p={c['p']:.4f}")

            all_results[f"{thresh_label}_{direction}"] = {
                "n_events": n_eff, "n_obs": len(panel_matched),
                "r2": float(model.rsquared),
                "coefficients": coefs,
                "sample_gate_pass": True, "decision_status": "pass",
            }
            panels[direction] = panel_matched

            # 再現ゲート: 最初の下方閾値でStage 1と照合
            if direction == "down" and not repro_passed and thresh_label == thresholds[0][0]:
                vol_c = coefs["vol"]["coef"]
                turn_c = coefs["turnover"]["coef"]
                beta_p = coefs["beta"]["p"]
                margin_p = coefs["margin_ratio"]["p"]
                checks = [
                    ("vol符号負", vol_c < 0),
                    ("turnover符号負", turn_c < 0),
                    ("beta非有意(p>0.10)", beta_p > 0.10),
                    ("margin非有意(p>0.10)", margin_p > 0.10),
                    ("vol係数オーダー", abs(vol_c - (-0.0199)) / 0.0199 < 0.50),
                    ("turnover係数オーダー", abs(turn_c - (-0.00256)) / 0.00256 < 0.50),
                    ("イベント日数", abs(n_eff - 45) <= 5),
                ]
                print("\n  --- 再現ゲート ---")
                all_pass = True
                for name, ok in checks:
                    print(f"  {name}: {'PASS' if ok else 'FAIL'}")
                    if not ok:
                        all_pass = False
                if not all_pass:
                    print("  NG: 再現ゲート失敗。中止")
                    return
                print("  OK: 再現ゲートパス")
                repro_passed = True

        # 対称性比較: 50%ルール（表示用）
        down_key = f"{thresh_label}_down"
        up_key = f"{thresh_label}_up"
        down_r = all_results.get(down_key, {})
        up_r = all_results.get(up_key, {})

        if down_r.get("sample_gate_pass") and up_r.get("sample_gate_pass"):
            print(f"\n--- 対称性比較 ({thresh_label}, 50%ルール表示用) ---")
            sym = {}
            for var in vars_of_interest:
                d_coef = down_r["coefficients"][var]["coef"]
                u_coef = up_r["coefficients"][var]["coef"]
                if abs(d_coef) > 1e-10:
                    ratio = abs(u_coef) / abs(d_coef)
                    same_sign = (d_coef * u_coef) > 0
                else:
                    ratio = float("nan")
                    same_sign = False
                symmetric = ratio >= 0.5 and same_sign if not np.isnan(ratio) else False
                sym[var] = {
                    "down_coef": d_coef, "up_coef": u_coef,
                    "ratio": ratio, "same_sign": same_sign, "symmetric_heuristic": symmetric,
                }
                label = "symmetric" if symmetric else "ASYMMETRIC"
                print(f"  {var:15s}: down={d_coef:+.6f}  up={u_coef:+.6f}  ratio={ratio:.2f}  {label}")

            # Pooled交互作用検定（主判定）
            if "down" in panels and "up" in panels:
                print(f"\n--- Pooled交互作用検定 ({thresh_label}, 主判定) ---")
                p_down = panels["down"].copy()
                p_up = panels["up"].copy()
                p_down["direction"] = 0
                p_up["direction"] = 1
                pooled = pd.concat([p_down, p_up], ignore_index=True)

                # 交互作用項を作成
                for var in vars_of_interest:
                    pooled[f"dir_x_{var}"] = pooled["direction"] * pooled[var]

                interaction_vars = [f"dir_x_{var}" for var in vars_of_interest]
                pooled_controls = ["direction"] + controls + interaction_vars
                pooled_model = run_regression_with_vars(pooled, pooled_controls)

                for var in vars_of_interest:
                    ix_var = f"dir_x_{var}"
                    ix_coef = float(pooled_model.params.get(ix_var, np.nan))
                    ix_p = float(pooled_model.pvalues.get(ix_var, np.nan))
                    symmetric_stat = ix_p > 0.10  # 差が非有意なら対称
                    sym[var]["interaction_coef"] = ix_coef
                    sym[var]["interaction_p"] = ix_p
                    sym[var]["symmetric_statistical"] = symmetric_stat
                    label = "symmetric(p>0.10)" if symmetric_stat else f"ASYMMETRIC(p={ix_p:.3f})"
                    print(f"  dir_x_{var:12s}: coef={ix_coef:+.6f}  p={ix_p:.4f}  {label}")

            all_results[f"{thresh_label}_comparison"] = sym

    # === サイズ層別（主分析の最初の閾値を使用）===
    base_thresh = thresholds[0]  # 1.0sigma
    print(f"\n{'='*60}")
    print(f"=== サイズ層別 ({base_thresh[0]}) ===")
    print(f"{'='*60}")

    controls_no_scale = ["beta", "vol", "turnover", "margin_ratio", "ret5d"]
    size_results = {}

    for direction in ["down", "up"]:
        if base_thresh[1] == "sigma":
            events = extract_events_sigma(us_returns, base_thresh[2], direction)
        else:
            events = extract_events_pct(us_returns, base_thresh[2], direction)
        panel = build_panel(us_returns, jp_returns, topix_returns, jp_prices_raw, listed, events)
        panel = add_margin_ratio(panel, margin_data, jp_prices_raw)
        panel_matched = panel.dropna(subset=["margin_ratio"])

        large_mask = (panel_matched["scale_large70"] == 1) | (
            (panel_matched["scale_large70"] == 0) & (panel_matched["scale_mid400"] == 0) &
            (panel_matched["scale_small1"] == 0) & (panel_matched["scale_small2"] == 0)
        )

        for size_label, mask in [("large", large_mask), ("small", ~large_mask)]:
            sub = panel_matched[mask]
            n_eff = sub["event_date"].nunique()
            key = f"{direction}_{size_label}"

            if n_eff < 10:
                print(f"  {key}: イベント不足({n_eff}日)。判定保留")
                size_results[key] = {"n_events": n_eff, "decision_status": "hold"}
                continue

            model = run_regression_with_vars(sub, controls_no_scale)
            coefs = extract_coefs(model, vars_of_interest)

            print(f"\n  {key} ({len(sub)}obs, {n_eff}日):")
            for var in vars_of_interest:
                c = coefs[var]
                print(f"    {var:15s}: coef={c['coef']:+.6f}  p={c['p']:.4f}")

            size_results[key] = {
                "n_obs": len(sub), "n_events": n_eff,
                "coefficients": coefs, "decision_status": "pass",
            }

    # === VIX変化層別（主分析の最初の閾値を使用）===
    print(f"\n{'='*60}")
    print(f"=== VIX変化層別 ({base_thresh[0]}) ===")
    print(f"{'='*60}")

    vix_results = {}
    for direction in ["down", "up"]:
        if base_thresh[1] == "sigma":
            events = extract_events_sigma(us_returns, base_thresh[2], direction)
        else:
            events = extract_events_pct(us_returns, base_thresh[2], direction)
        panel = build_panel(us_returns, jp_returns, topix_returns, jp_prices_raw, listed, events)
        panel = add_margin_ratio(panel, margin_data, jp_prices_raw)
        panel_matched = panel.dropna(subset=["margin_ratio"])

        # 各イベント日のVIX変化を取得
        event_vix = {}
        for ed in panel_matched["event_date"].unique():
            event_vix[ed] = vix_change.get(ed, np.nan)

        panel_matched = panel_matched.copy()
        panel_matched["vix_chg"] = panel_matched["event_date"].map(event_vix)
        vix_median = panel_matched.groupby("event_date")["vix_chg"].first().median()

        for vix_label, vix_mask_fn in [
            ("vix_low", lambda p: p["vix_chg"] <= vix_median),
            ("vix_high", lambda p: p["vix_chg"] > vix_median),
        ]:
            sub = panel_matched[vix_mask_fn(panel_matched)]
            n_eff = sub["event_date"].nunique()
            key = f"{direction}_{vix_label}"

            if n_eff < 10:
                print(f"  {key}: イベント不足({n_eff}日)。判定保留")
                vix_results[key] = {"n_events": n_eff, "decision_status": "hold"}
                continue

            model = run_regression_with_vars(sub, controls)
            coefs = extract_coefs(model, vars_of_interest)

            print(f"\n  {key} ({len(sub)}obs, {n_eff}日):")
            for var in vars_of_interest:
                c = coefs[var]
                print(f"    {var:15s}: coef={c['coef']:+.6f}  p={c['p']:.4f}")

            vix_results[key] = {
                "n_obs": len(sub), "n_events": n_eff,
                "coefficients": coefs, "decision_status": "pass",
            }

    # === T1-Q04 attention-penalty（--q04フラグで有効化）===
    q04_results = {}
    if args.q04:
        print(f"\n{'='*60}")
        print("=== T1-Q04 attention-penalty ===")
        print(f"{'='*60}")

        # 下方2.0sigmaのパネルを使用（Q03と同一）
        down_events = extract_events_sigma(us_returns, 2.0, "down")
        panel_q04 = build_panel(us_returns, jp_returns, topix_returns, jp_prices_raw, listed, down_events)
        panel_q04 = add_margin_ratio(panel_q04, margin_data, jp_prices_raw)
        panel_q04 = panel_q04.dropna(subset=["margin_ratio"])

        # 異常出来高シェアを追加
        print("異常出来高シェアを計算中...")
        panel_q04 = add_abnormal_volume_share(panel_q04, jp_prices_raw)
        n_avs = panel_q04["abnormal_vol_share"].notna().sum()
        print(f"  異常出来高シェアあり: {n_avs}/{len(panel_q04)}")

        panel_q04_matched = panel_q04.dropna(subset=["abnormal_vol_share"])
        print(f"  matched panel: {len(panel_q04_matched)}obs ({panel_q04_matched['event_date'].nunique()}日)")

        # --- 再現ゲート（model_base = Q03下方と同一）---
        print("\n--- 再現ゲート ---")
        model_base = run_regression_with_vars(panel_q04_matched, controls)
        base_coefs = extract_coefs(model_base, vars_of_interest)
        for var in vars_of_interest:
            c = base_coefs[var]
            print(f"  {var:15s}: coef={c['coef']:+.6f}  p={c['p']:.4f}")

        vol_ok = base_coefs["vol"]["coef"] < 0 and base_coefs["vol"]["p"] < 0.05
        turn_ok = base_coefs["turnover"]["coef"] < 0
        if not vol_ok:
            print("NG: vol再現失敗。中止")
        else:
            print("OK: 再現ゲートパス")

            # --- model_vs（abnormal_vol_share追加）---
            print("\n--- model_vs: 異常出来高シェア追加 ---")
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            controls_vs = ["beta", "vol", "turnover", "abnormal_vol_share", "margin_ratio",
                          "scale_large70", "scale_mid400", "scale_small1", "scale_small2", "ret5d"]

            # VIF
            vif_vars = ["turnover", "abnormal_vol_share", "vol", "beta"]
            X_vif = sm.add_constant(panel_q04_matched[vif_vars])
            vifs = {}
            for i, col in enumerate(X_vif.columns):
                if col == "const":
                    continue
                vifs[col] = float(variance_inflation_factor(X_vif.values, i))
            print("VIF:")
            vif_critical = False
            for k, v in vifs.items():
                flag = ""
                if v > 10:
                    flag = " [CRITICAL]"
                    vif_critical = True
                elif v > 5:
                    flag = " [WARNING]"
                print(f"  {k:20s}: {v:.2f}{flag}")

            if vif_critical:
                print("NG: VIF>10。主解釈停止")
                q04_results["vif_critical"] = True
            else:
                model_vs = run_regression_with_vars(panel_q04_matched, controls_vs)
                vs_coefs = extract_coefs(model_vs, vars_of_interest + ["abnormal_vol_share"])
                incr_r2 = model_vs.rsquared - model_base.rsquared

                print(f"\nmodel_vs R2={model_vs.rsquared:.4f} (増分R2={incr_r2:.4f})")
                for var in vars_of_interest + ["abnormal_vol_share"]:
                    c = vs_coefs[var]
                    marker = " <--" if var == "abnormal_vol_share" else ""
                    print(f"  {var:20s}: coef={c['coef']:+.6f}  p={c['p']:.4f}{marker}")

                avs_p = vs_coefs["abnormal_vol_share"]["p"]
                avs_coef = vs_coefs["abnormal_vol_share"]["coef"]
                turn_coef_vs = vs_coefs["turnover"]["coef"]
                turn_p_vs = vs_coefs["turnover"]["p"]

                # 効果量ゲート
                effect_size_pass = incr_r2 >= 0.001
                p_pass = avs_p < 0.10

                print(f"\n効果量: 増分R2={incr_r2:.4f} ({'PASS' if effect_size_pass else 'FAIL'} >=0.1%)")
                print(f"有意性: p={avs_p:.4f} ({'PASS' if p_pass else 'FAIL'} <0.10)")

                # --- placebo（上方ショックで同じ効果が出ないか確認）---
                print("\n--- placebo: 上方ショックで確認 ---")
                up_events = extract_events_sigma(us_returns, 2.0, "up")
                panel_up = build_panel(us_returns, jp_returns, topix_returns, jp_prices_raw, listed, up_events)
                panel_up = add_margin_ratio(panel_up, margin_data, jp_prices_raw)
                panel_up = add_abnormal_volume_share(panel_up, jp_prices_raw)
                panel_up_m = panel_up.dropna(subset=["margin_ratio", "abnormal_vol_share"])

                if len(panel_up_m) > 100:
                    model_placebo = run_regression_with_vars(panel_up_m, controls_vs)
                    placebo_coefs = extract_coefs(model_placebo, ["abnormal_vol_share"])
                    placebo_p = placebo_coefs["abnormal_vol_share"]["p"]
                    placebo_coef = placebo_coefs["abnormal_vol_share"]["coef"]
                    placebo_pass = placebo_p > 0.10  # 上方で非有意ならplaceboパス
                    print(f"  上方 abnormal_vol_share: coef={placebo_coef:+.6f}  p={placebo_p:.4f}  {'PASS(非有意)' if placebo_pass else 'FAIL(有意)'}")
                else:
                    placebo_pass = None
                    placebo_p = float("nan")
                    print(f"  上方サンプル不足({len(panel_up_m)}obs)。placebo判定保留")

                # --- 反転テスト ---
                print("\n--- 反転テスト ---")
                panel_q04_car = compute_post_event_car(jp_returns, topix_returns, panel_q04_matched)
                reversal = run_reversal_test(panel_q04_car)

                for gvar, gres in reversal.items():
                    print(f"\n  {gvar}:")
                    for window, wres in gres.items():
                        if "error" in wres:
                            print(f"    {window}: {wres['error']}")
                        else:
                            print(f"    {window}: 高群CAR={wres['high_car_mean']:+.4f}  低群CAR={wres['low_car_mean']:+.4f}  統制後coef={wres['controlled_coef']:+.4f}  p={wres['controlled_p']:.4f}")

                # --- Q04 総合判定 ---
                print(f"\n{'='*60}")
                print("=== T1-Q04 総合判定 ===")
                print(f"{'='*60}")

                if p_pass and effect_size_pass:
                    if placebo_pass:
                        q04_verdict = "abnormal_vol_shareに追加説明力あり（下方固有）"
                    elif placebo_pass is None:
                        q04_verdict = "abnormal_vol_shareに追加説明力あり（placebo判定保留）"
                    else:
                        q04_verdict = "abnormal_vol_shareに追加説明力あるが上方でも有意（下方固有ではない）"
                else:
                    q04_verdict = "abnormal_vol_shareに追加説明力なし。T1-Q04棄却"

                print(f"判定: {q04_verdict}")

                q04_results = {
                    "model_base": {"r2": float(model_base.rsquared), "coefficients": base_coefs},
                    "model_vs": {
                        "r2": float(model_vs.rsquared), "incremental_r2": float(incr_r2),
                        "avs_coef": float(avs_coef), "avs_p": float(avs_p),
                        "turnover_coef_change": f"{base_coefs['turnover']['coef']:.6f} -> {turn_coef_vs:.6f}",
                    },
                    "placebo": {"p": float(placebo_p), "pass": placebo_pass},
                    "reversal": reversal,
                    "vifs": vifs,
                    "verdict": q04_verdict,
                }

    # === JSON保存 ===
    output = {
        "experiment": "symmetry-test",
        "thresholds": {t[0]: all_results.get(f"{t[0]}_comparison", {}) for t in thresholds},
        "direction_results": {k: v for k, v in all_results.items() if not k.endswith("_comparison")},
        "size_stratified": size_results,
        "vix_stratified": vix_results,
    }
    if args.q04:
        output["q04_attention_penalty"] = q04_results

    output_path = Path("logs/iterations/T1-Q03_symmetry-test_result.json")
    if args.q04:
        output_path = Path("logs/iterations/T1-Q04_attention-penalty_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果を保存: {output_path}")


if __name__ == "__main__":
    main()
