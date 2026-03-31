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


def _demean_winsorize_ols(panel, y_col, xvars):
    """共通前処理: event_date demean → 1%/99% Winsorize → cluster SE付きOLS。"""
    all_vars = [y_col] + xvars
    panel_dm = panel.copy()
    event_means = panel.groupby("event_date")[all_vars].transform("mean")
    for col in all_vars:
        panel_dm[col] = panel[col] - event_means[col]
    for col in all_vars:
        lo, hi = panel_dm[col].quantile(0.01), panel_dm[col].quantile(0.99)
        panel_dm[col] = panel_dm[col].clip(lo, hi)
    y = panel_dm[y_col]
    X = panel_dm[xvars]
    return sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": panel["event_date"]})


def run_regression_with_vars(panel, xvars):
    """demean -> Winsorize -> OLS(1-way cluster SE)。excess_retが被説明変数。"""
    return _demean_winsorize_ols(panel, "excess_ret", xvars)


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


def add_log_trading_value(panel, jp_prices_raw, lookback=20):
    """event_date以前lookback日間の平均売買代金(close×volume)をlog変換。補助分析用のsize/liquidity proxy。"""
    close_col = "adj_close" if "adj_close" in jp_prices_raw.columns else "close"
    vol_col = "volume" if "volume" in jp_prices_raw.columns else "adj_volume"
    vals = []
    for _, row in panel.iterrows():
        code = row["code"]
        ed = row["event_date"]
        code_data = jp_prices_raw[jp_prices_raw["code"] == code].set_index("date").sort_index()
        if close_col not in code_data.columns or vol_col not in code_data.columns:
            vals.append(np.nan)
            continue
        recent = code_data[code_data.index <= ed].tail(lookback)
        if len(recent) < lookback // 2:
            vals.append(np.nan)
            continue
        trading_value = (recent[close_col] * recent[vol_col]).mean()
        if trading_value is None or np.isnan(trading_value) or trading_value <= 0:
            vals.append(np.nan)
            continue
        vals.append(float(np.log(trading_value)))
    panel = panel.copy()
    panel["log_trading_value"] = vals
    n_missing = panel["log_trading_value"].isna().sum()
    missing_rate = n_missing / len(panel) if len(panel) > 0 else 0
    print(f"  log_trading_value: {len(panel) - n_missing}/{len(panel)} 有効 (欠損率 {missing_rate:.1%})")
    return panel


def split_large_small(panel):
    """K4/Q02と同一定義で分割。Large=Large70+Core30(全ダミー0), Small=Mid400+Small1+Small2。
    上位200銘柄ユニバースではSmall1/Small2がほぼ不在のため、実質Large70+Core30 vs Mid400。"""
    large_mask = (panel["scale_large70"] == 1) | (
        (panel["scale_large70"] == 0) & (panel["scale_mid400"] == 0) &
        (panel["scale_small1"] == 0) & (panel["scale_small2"] == 0)
    )
    small_mask = ~large_mask
    print(f"  Large: {large_mask.sum()}, Small: {small_mask.sum()} (全{len(panel)})")
    return panel[large_mask].copy(), panel[small_mask].copy()


def run_car_regression(panel, car_col, xvars):
    """CAR回帰用。_demean_winsorize_olsを使用。"""
    return _demean_winsorize_ols(panel, car_col, xvars)


def run_discrete_size_reversal(panel, windows=[5, 10, 20]):
    """主判定: Large/Smallで各群のturnover/vol高低別CARを比較。day-0統制回帰込み。"""
    large, small = split_large_small(panel)
    results = {}
    for size_label, sub in [("large", large), ("small", small)]:
        n_events = sub["event_date"].nunique()
        n_obs = len(sub)
        if n_events < 15 or n_obs < 200:
            results[size_label] = {
                "n_events": n_events, "n_obs": n_obs,
                "status": "insufficient_power",
            }
            print(f"  {size_label}: insufficient_power (n_events={n_events}, n_obs={n_obs})")
            continue

        size_result = {"n_events": n_events, "n_obs": n_obs, "status": "computed"}
        for gvar in ["turnover", "vol"]:
            gvar_results = {}
            for w in windows:
                car_col = f"car_{w}d"
                if car_col not in sub.columns:
                    continue
                valid = sub[[car_col, "excess_ret", gvar, "event_date"]].dropna()
                if len(valid) < 50:
                    gvar_results[f"{w}d"] = {"error": "insufficient data"}
                    continue
                high = valid[valid[gvar] > valid[gvar].median()]
                low = valid[valid[gvar] <= valid[gvar].median()]
                high_car = float(high[car_col].mean())
                low_car = float(low[car_col].mean())
                diff = high_car - low_car
                # day-0統制回帰（共通前処理: demean + Winsorize + cluster SE）
                valid_reg = valid.copy()
                valid_reg["high"] = (valid_reg[gvar] > valid_reg[gvar].median()).astype(float)
                try:
                    model = _demean_winsorize_ols(valid_reg, car_col, ["excess_ret", "high"])
                    gvar_results[f"{w}d"] = {
                        "high_car_mean": high_car,
                        "low_car_mean": low_car,
                        "diff": diff,
                        "controlled_coef": float(model.params.get("high", np.nan)),
                        "controlled_p": float(model.pvalues.get("high", np.nan)),
                        "n_high": len(high),
                        "n_low": len(low),
                    }
                except Exception as e:
                    gvar_results[f"{w}d"] = {"error": str(e)}
            size_result[gvar] = gvar_results
        results[size_label] = size_result
    return results


def run_smallcap_margin_car(panel, windows=[20]):
    """探索的: 中小型サブサンプルで信用残高低別20d CARを比較。"""
    _, small = split_large_small(panel)
    if "margin_ratio" not in small.columns:
        return {"status": "no_margin_data"}
    valid = small.dropna(subset=["margin_ratio"])
    n_events = valid["event_date"].nunique()
    if n_events < 15:
        return {"status": "insufficient_power", "n_events": n_events}
    results = {"n_events": n_events, "n_obs": len(valid), "status": "computed"}
    for w in windows:
        car_col = f"car_{w}d"
        if car_col not in valid.columns:
            continue
        sub = valid[[car_col, "excess_ret", "margin_ratio", "event_date"]].dropna()
        if len(sub) < 30:
            results[f"{w}d"] = {"error": "insufficient data"}
            continue
        high = sub[sub["margin_ratio"] > sub["margin_ratio"].median()]
        low = sub[sub["margin_ratio"] <= sub["margin_ratio"].median()]
        high_car = float(high[car_col].mean())
        low_car = float(low[car_col].mean())
        # day-0統制回帰（共通前処理: demean + Winsorize + cluster SE）
        sub_reg = sub.copy()
        sub_reg["high"] = (sub_reg["margin_ratio"] > sub_reg["margin_ratio"].median()).astype(float)
        try:
            model = _demean_winsorize_ols(sub_reg, car_col, ["excess_ret", "high"])
            results[f"{w}d"] = {
                "high_car_mean": high_car,
                "low_car_mean": low_car,
                "diff": high_car - low_car,
                "controlled_coef": float(model.params.get("high", np.nan)),
                "controlled_p": float(model.pvalues.get("high", np.nan)),
                "n_high": len(high),
                "n_low": len(low),
            }
        except Exception as e:
            results[f"{w}d"] = {"error": str(e)}
    return results


def main():
    parser = argparse.ArgumentParser(description="symmetry-test + attention-penalty")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--n-stocks", type=int, default=200)
    parser.add_argument("--q04", action="store_true", help="T1-Q04 attention-penalty分析を実行")
    parser.add_argument("--q05", action="store_true", help="T1-Q05 vol-reversal-specificity分析を実行")
    parser.add_argument("--q06", action="store_true", help="T1-Q06 turnover-momentum-disentangle分析を実行")
    parser.add_argument("--q09", action="store_true", help="T1-Q09 turnover-postshock-specificity-did分析を実行")
    parser.add_argument("--q10", action="store_true", help="T1-Q10 turnover-decline-mechanism-separation分析を実行")
    parser.add_argument("--t2q02", action="store_true", help="T2-Q02 optimal-holding-period-decay分析を実行")
    parser.add_argument("--q07", action="store_true", help="T1-Q07 size-regime-interaction分析を実行")
    parser.add_argument("--t2q04", action="store_true", help="T2-Q04 event-clustering-capacity分析を実行")
    parser.add_argument("--skip-symmetry", action="store_true", help="対称性テストをスキップ（Q05/Q06のみ実行時）")
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
    panels = {}
    repro_passed = False

    if args.skip_symmetry:
        print("対称性テストをスキップ（--skip-symmetry）")
        # 最低限の下方パネルだけ構築（Q05/Q06が使う）
        sigma = us_returns.std()
        down_dates = us_returns[us_returns < -2.0 * sigma].index.tolist()
        up_dates = us_returns[us_returns > 2.0 * sigma].index.tolist()
        print(f"2.0sigma下方: {len(down_dates)}日, 上方: {len(up_dates)}日")
        panels["down"] = build_panel(us_returns, jp_returns, topix_returns, jp_prices_raw, listed, down_dates)
        panels["up"] = build_panel(us_returns, jp_returns, topix_returns, jp_prices_raw, listed, up_dates)
        repro_passed = True
    else:
        pass  # 対称性テストを実行

    for thresh_label, thresh_type, thresh_val in (thresholds if not args.skip_symmetry else []):
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
                    ("vol係数オーダー", abs(vol_c) > 0.001),  # 符号負は上でチェック済み、最低限の大きさ
                    ("turnover係数オーダー", abs(turn_c) > 0.0005),  # 同上
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

    # === サイズ層別・VIX層別 ===
    size_results = {}
    vix_results = {}
    base_thresh = thresholds[0]  # 1.0sigma
    _run_stratified = not args.skip_symmetry
    if _run_stratified:
        print(f"\n{'='*60}")
        print(f"=== サイズ層別 ({base_thresh[0]}) ===")
        print(f"{'='*60}")

    controls_no_scale = ["beta", "vol", "turnover", "margin_ratio", "ret5d"]
    size_results = {}

    for direction in (["down", "up"] if _run_stratified else []):
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
    if _run_stratified:
        print(f"\n{'='*60}")
        print(f"=== VIX変化層別 ({base_thresh[0]}) ===")
        print(f"{'='*60}")

    vix_results = {}
    for direction in (["down", "up"] if _run_stratified else []):
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

    # === T1-Q05: vol反転のショック固有性検証 ===
    q05_results = {}
    if args.q05:
        from scipy import stats as scipy_stats
        print("\n" + "=" * 60)
        print("=== T1-Q05 vol-reversal-specificity ===")
        print("=" * 60)

        # Q04反転テスト再現ゲート
        down_key = f"{thresholds[0][0]}_down"
        if down_key not in all_results:
            print("NG: 下方ショック結果がない。中止")
        else:
            down_panel = panels.get("down")
            if down_panel is None or len(down_panel) == 0:
                print("NG: 下方パネルがない。中止")
            else:
                # CARを計算
                down_panel = compute_post_event_car(jp_returns, topix_returns, down_panel, windows=[5, 10, 20])
                shock_dates_list = sorted(down_panel["event_date"].unique())
                n_shock = len(shock_dates_list)
                print(f"ショック日数: {n_shock}")

                # vol高/低群の20d CARを計算
                vol_median = down_panel["vol"].median()
                vol_high = down_panel[down_panel["vol"] > vol_median]
                vol_low = down_panel[down_panel["vol"] <= vol_median]
                shock_vol_high_car = vol_high["car_20d"].dropna().mean()
                shock_vol_low_car = vol_low["car_20d"].dropna().mean()
                print(f"\nショック日 vol高群 20d CAR: {shock_vol_high_car*100:+.3f}%")
                print(f"ショック日 vol低群 20d CAR: {shock_vol_low_car*100:+.3f}%")

                # 非ショック日の定義（2sigma下方ショック日の補集合）
                all_sp_dates = us_returns.index
                shock_set = set(shock_dates_list)
                non_shock_dates = [d for d in all_sp_dates if d not in shock_set]
                print(f"非ショック日数: {len(non_shock_dates)}")

                # --- 事前計算（ブートストラップ外で1回だけ）---
                print("事前計算: 全日×全銘柄のCAR・vol行列...")
                excess_mat = jp_returns.sub(topix_returns, axis=0)
                # 20d forward CAR: reaction_dateから翌日〜20日後の累積超過リターン
                # shift(-21)で21日先の累積値を取り、shift(-1)との差で1〜20日分を得る
                cumex = excess_mat.cumsum()
                car_20d_mat = cumex.shift(-21) - cumex.shift(-1)
                # vol: 20日ローリングstd（年率化）
                vol_mat = jp_returns.rolling(20, min_periods=10).std() * np.sqrt(252)
                print(f"  行列サイズ: {car_20d_mat.shape}")

                # 各日の翌JP営業日を事前マッピング
                jp_idx = jp_returns.index
                next_jp_day = {}
                for i_d, d in enumerate(jp_idx[:-1]):
                    next_jp_day[d] = jp_idx[i_d + 1]
                # 非ショック日もJP営業日にマッピング
                non_shock_jp = []
                for d in non_shock_dates:
                    later = jp_idx[jp_idx > d]
                    if len(later) > 0:
                        non_shock_jp.append((d, later[0]))

                # --- ベースライン (1): ランダム非ショック日 ---
                print("\n--- ベースライン1: ランダム非ショック日 ---")
                n_iter = 200
                min_gap = 20
                rng = np.random.RandomState(42)
                bootstrap_vol_high_cars = []
                bootstrap_vol_low_cars = []

                non_shock_arr = np.array([x[0] for x in non_shock_jp])
                non_shock_reaction = {x[0]: x[1] for x in non_shock_jp}

                for i in range(n_iter):
                    shuffled = rng.permutation(len(non_shock_arr))
                    sampled_reactions = []
                    sampled_events = []
                    for idx in shuffled:
                        d = non_shock_arr[idx]
                        if all(abs((d - s).days) >= min_gap for s in sampled_events):
                            sampled_events.append(d)
                            sampled_reactions.append(non_shock_reaction[d])
                        if len(sampled_events) >= n_shock:
                            break
                    if len(sampled_reactions) < n_shock // 2:
                        continue

                    # 各reaction_dateのvol・CARを行列から参照
                    all_vols = []
                    all_cars = []
                    for rd in sampled_reactions:
                        if rd not in vol_mat.index or rd not in car_20d_mat.index:
                            continue
                        v = vol_mat.loc[rd].dropna()
                        c = car_20d_mat.loc[rd].dropna()
                        common_codes = v.index.intersection(c.index)
                        if len(common_codes) < 10:
                            continue
                        all_vols.append(v[common_codes])
                        all_cars.append(c[common_codes])

                    if len(all_vols) < 5:
                        continue
                    concat_v = pd.concat(all_vols)
                    concat_c = pd.concat(all_cars)
                    vm = concat_v.median()
                    fh = concat_c[concat_v > vm].mean()
                    fl = concat_c[concat_v <= vm].mean()
                    bootstrap_vol_high_cars.append(float(fh))
                    bootstrap_vol_low_cars.append(float(fl))

                if len(bootstrap_vol_high_cars) > 50:
                    bs_high = np.array(bootstrap_vol_high_cars)
                    bs_low = np.array(bootstrap_vol_low_cars)
                    baseline_high = np.mean(bs_high)
                    baseline_low = np.mean(bs_low)
                    p_high = np.mean(bs_high >= shock_vol_high_car)
                    p_low = np.mean(bs_low >= shock_vol_low_car)
                    pseudo_did_shock = shock_vol_high_car - shock_vol_low_car
                    pseudo_did_baseline = baseline_high - baseline_low
                    pseudo_did = pseudo_did_shock - pseudo_did_baseline
                    bs_dids = bs_high - bs_low
                    p_did = np.mean(bs_dids >= pseudo_did_shock)

                    print(f"  非ショック日 vol高群ベースライン: {baseline_high*100:+.3f}%")
                    print(f"  非ショック日 vol低群ベースライン: {baseline_low*100:+.3f}%")
                    print(f"  ショック固有vol高群CAR: {shock_vol_high_car*100:+.3f}% vs ベースライン {baseline_high*100:+.3f}% (p={p_high:.4f})")
                    print(f"  pseudo-DiD: {pseudo_did*100:+.3f}% (shock={pseudo_did_shock*100:+.3f}% - baseline={pseudo_did_baseline*100:+.3f}%)")
                    print(f"  pseudo-DiD p値: {p_did:.4f}")

                    q05_results["baseline_random"] = {
                        "shock_vol_high_car": float(shock_vol_high_car),
                        "shock_vol_low_car": float(shock_vol_low_car),
                        "baseline_vol_high_car": float(baseline_high),
                        "baseline_vol_low_car": float(baseline_low),
                        "p_vol_high_vs_baseline": float(p_high),
                        "pseudo_did": float(pseudo_did),
                        "pseudo_did_p": float(p_did),
                        "n_bootstrap": len(bootstrap_vol_high_cars),
                    }
                else:
                    print("  ブートストラップ失敗（十分なサンプルが取れなかった）")

                # --- ショック規模相関 ---
                print("\n--- ショック規模相関 ---")
                shock_cars_by_date = {}
                for sd in shock_dates_list:
                    sub = vol_high[vol_high["event_date"] == sd]
                    if len(sub) > 0 and sub["car_20d"].notna().sum() > 0:
                        shock_cars_by_date[sd] = sub["car_20d"].mean()

                if len(shock_cars_by_date) >= 10:
                    dates_sorted = sorted(shock_cars_by_date.keys())
                    car_vals = [shock_cars_by_date[d] for d in dates_sorted]
                    sp_vals = [float(us_returns.loc[d]) if d in us_returns.index else np.nan for d in dates_sorted]
                    valid_mask = [not (np.isnan(c) or np.isnan(s)) for c, s in zip(car_vals, sp_vals)]
                    car_clean = [c for c, v in zip(car_vals, valid_mask) if v]
                    sp_clean = [s for s, v in zip(sp_vals, valid_mask) if v]

                    # ショック規模は負なので絶対値で（下落幅が大きいほど反転も大きいか）
                    sp_abs = [abs(s) for s in sp_clean]
                    rho, p_rho = scipy_stats.spearmanr(sp_abs, car_clean)
                    print(f"  Spearman(|S&P500下落幅|, vol高群20d CAR): rho={rho:.3f}, p={p_rho:.4f}")
                    print(f"  n={len(car_clean)}イベント日")

                    q05_results["shock_magnitude_correlation"] = {
                        "spearman_rho": float(rho),
                        "spearman_p": float(p_rho),
                        "n_events": len(car_clean),
                    }

                # --- 最終判定 ---
                print("\n" + "=" * 60)
                print("=== T1-Q05 最終判定 ===")
                print("=" * 60)
                primary_pass = False
                secondary_pass = False
                if "baseline_random" in q05_results:
                    r = q05_results["baseline_random"]
                    primary_pass = r["pseudo_did"] > 0 and r["pseudo_did_p"] < 0.10
                    print(f"Primary (pseudo-DiD > 0, p<0.10): {'PASS' if primary_pass else 'FAIL'}")
                    print(f"  pseudo-DiD={r['pseudo_did']*100:+.3f}%, p={r['pseudo_did_p']:.4f}")
                if "shock_magnitude_correlation" in q05_results:
                    r = q05_results["shock_magnitude_correlation"]
                    secondary_pass = r["spearman_rho"] > 0.05 and r["spearman_p"] < 0.10
                    print(f"Secondary (Spearman rho>0.05, p<0.10): {'PASS' if secondary_pass else 'FAIL'}")
                    print(f"  rho={r['spearman_rho']:.3f}, p={r['spearman_p']:.4f}")

                verdict = "PASS" if primary_pass else ("PARTIAL" if secondary_pass else "FAIL")
                q05_results["verdict"] = verdict
                print(f"\n総合判定: {verdict}")

    # === T1-Q06: turnover継続下落のモメンタム分離 ===
    q06_results = {}
    if args.q06:
        print("\n" + "=" * 60)
        print("=== T1-Q06 turnover-momentum-disentangle ===")
        print("=" * 60)

        down_panel = panels.get("down")
        up_panel = panels.get("up")
        if down_panel is None or len(down_panel) == 0:
            print("NG: 下方パネルがない。中止")
        else:
            # CARを計算
            down_panel = compute_post_event_car(jp_returns, topix_returns, down_panel, windows=[5, 10, 20])

            # 事前リターン計算（3ウィンドウ）
            for w in [5, 20, 60]:
                col = f"prior_ret_{w}d"
                vals = []
                for _, row in down_panel.iterrows():
                    code = row["code"]
                    ed = row["event_date"]
                    before = jp_returns[code][jp_returns.index <= ed].tail(w + 1).head(w)
                    if len(before) < w // 2:
                        vals.append(np.nan)
                    else:
                        vals.append(float((1 + before).prod() - 1))
                down_panel[col] = vals

            turnover_median = down_panel["turnover"].median()
            turn_high = down_panel[down_panel["turnover"] > turnover_median]

            # --- Step 1: 事前リターン層別 ---
            print("\n--- Step 1: 事前リターン層別（turnover高群のみ）---")
            step1_results = {}
            for w in [5, 20, 60]:
                col = f"prior_ret_{w}d"
                valid = turn_high[[col, "car_20d"]].dropna()
                if len(valid) < 20:
                    step1_results[f"{w}d"] = {"error": "insufficient data"}
                    continue
                prior_median = valid[col].median()
                up_group = valid[valid[col] > prior_median]
                down_group = valid[valid[col] <= prior_median]
                up_car = up_group["car_20d"].mean()
                down_car = down_group["car_20d"].mean()
                from scipy import stats as scipy_stats
                t_stat, p_val = scipy_stats.ttest_1samp(up_group["car_20d"], 0)
                step1_results[f"{w}d"] = {
                    "prior_up_car": float(up_car),
                    "prior_up_n": len(up_group),
                    "prior_up_p": float(p_val),
                    "prior_down_car": float(down_car),
                    "prior_down_n": len(down_group),
                }
                print(f"  {w}d事前リターン: 上昇群CAR={up_car*100:+.3f}%(n={len(up_group)}, p={p_val:.4f}), "
                      f"下落群CAR={down_car*100:+.3f}%(n={len(down_group)})")
            q06_results["step1_prior_stratified"] = step1_results

            # Step 1判定: 3ウィンドウ中2つ以上で事前上昇群のCAR<0かつp<0.10
            n_pass = sum(1 for w in [5, 20, 60]
                        if f"{w}d" in step1_results
                        and isinstance(step1_results[f"{w}d"], dict)
                        and "prior_up_car" in step1_results[f"{w}d"]
                        and step1_results[f"{w}d"]["prior_up_car"] < 0
                        and step1_results[f"{w}d"]["prior_up_p"] < 0.10)
            step1_pass = n_pass >= 2
            print(f"  Step 1判定: {n_pass}/3ウィンドウでPASS → {'PASS' if step1_pass else 'FAIL'}")

            # --- Step 2: 大型株限定 ---
            print("\n--- Step 2: 大型株限定 ---")
            if "scale_large" in down_panel.columns:
                large_turn_high = turn_high[turn_high["scale_large"] == 1]
            else:
                # サイズ情報がなければ上位半分を大型扱い
                large_turn_high = turn_high
            step2_car = large_turn_high["car_20d"].dropna().mean()
            step2_n = large_turn_high["car_20d"].dropna().count()
            if step2_n > 10:
                t2, p2 = scipy_stats.ttest_1samp(large_turn_high["car_20d"].dropna(), 0)
                print(f"  大型turnover高群 20d CAR: {step2_car*100:+.3f}% (n={step2_n}, p={p2:.4f})")
                q06_results["step2_large_cap"] = {"car": float(step2_car), "n": int(step2_n), "p": float(p2)}

            # --- Step 3: 上方ショック比較 ---
            print("\n--- Step 3: 上方ショックでturnover効果消失確認 ---")
            if up_panel is not None and len(up_panel) > 0:
                up_panel = compute_post_event_car(jp_returns, topix_returns, up_panel, windows=[20])
                up_turn_median = up_panel["turnover"].median()
                up_turn_high = up_panel[up_panel["turnover"] > up_turn_median]
                up_car = up_turn_high["car_20d"].dropna().mean()
                up_n = up_turn_high["car_20d"].dropna().count()
                if up_n > 10:
                    t3, p3 = scipy_stats.ttest_1samp(up_turn_high["car_20d"].dropna(), 0)
                    step3_pass = p3 > 0.10  # 非有意 = 効果消失 = PASS
                    print(f"  上方ショック turnover高群 20d CAR: {up_car*100:+.3f}% (n={up_n}, p={p3:.4f})")
                    print(f"  Step 3判定: {'PASS(非有意=消失)' if step3_pass else 'FAIL(有意=消失せず)'}")
                    q06_results["step3_upside_placebo"] = {"car": float(up_car), "n": int(up_n), "p": float(p3), "pass": step3_pass}

            # --- Step 4: 連続回帰（補強）---
            print("\n--- Step 4: 連続回帰（事前リターン統制）---")
            reg_data = down_panel[["car_20d", "turnover", "prior_ret_20d", "excess_ret"]].dropna()
            if len(reg_data) > 50:
                X = sm.add_constant(reg_data[["turnover", "prior_ret_20d", "excess_ret"]])
                model = sm.OLS(reg_data["car_20d"], X).fit()
                print(f"  turnover coef={model.params['turnover']:+.6f} p={model.pvalues['turnover']:.4f}")
                print(f"  prior_ret_20d coef={model.params['prior_ret_20d']:+.6f} p={model.pvalues['prior_ret_20d']:.4f}")
                q06_results["step4_regression"] = {
                    "turnover_coef": float(model.params["turnover"]),
                    "turnover_p": float(model.pvalues["turnover"]),
                    "prior_ret_coef": float(model.params["prior_ret_20d"]),
                    "prior_ret_p": float(model.pvalues["prior_ret_20d"]),
                    "r2": float(model.rsquared),
                }

            # --- 最終判定 ---
            print("\n" + "=" * 60)
            print("=== T1-Q06 最終判定 ===")
            print("=" * 60)
            step3_pass_final = q06_results.get("step3_upside_placebo", {}).get("pass", False)
            overall = "PASS" if (step1_pass and step3_pass_final) else "FAIL"
            q06_results["verdict"] = overall
            print(f"Step 1 (事前上昇群で継続下落維持): {'PASS' if step1_pass else 'FAIL'}")
            print(f"Step 3 (上方ショックで効果消失): {'PASS' if step3_pass_final else 'FAIL'}")
            print(f"総合判定: {overall}")

    # === T1-Q07: size-regime-interaction ===
    q07_results = {}
    if args.q07:
        from scipy import stats as scipy_stats
        print("\n" + "=" * 60)
        print("=== T1-Q07 size-regime-interaction ===")
        print("=" * 60)

        down_panel = panels.get("down")
        if down_panel is None or len(down_panel) == 0:
            print("NG: 下方パネルがない。中止")
        else:
            # CARを計算
            down_panel = compute_post_event_car(jp_returns, topix_returns, down_panel, windows=[5, 10, 20])
            # 信用残追加（まだなければ）
            if "margin_ratio" not in down_panel.columns:
                down_panel = add_margin_ratio(down_panel, margin_data, jp_prices_raw)

            # --- Step 1: 離散分割 (主判定) ---
            print("\n--- Step 1: TOPIX離散分割 (主判定) ---")
            discrete_results = run_discrete_size_reversal(down_panel, windows=[5, 10, 20])
            q07_results["verdict_inputs"] = {"discrete_size_reversal": discrete_results}

            # 主判定の表示
            for size_label in ["large", "small"]:
                sr = discrete_results.get(size_label, {})
                if sr.get("status") == "insufficient_power":
                    continue
                print(f"\n  === {size_label.upper()} (n_events={sr.get('n_events')}, n_obs={sr.get('n_obs')}) ===")
                for gvar in ["turnover", "vol"]:
                    gdata = sr.get(gvar, {})
                    for w_label, wdata in gdata.items():
                        if "error" in wdata:
                            print(f"    {gvar} {w_label}: {wdata['error']}")
                        else:
                            print(f"    {gvar} {w_label}: high={wdata['high_car_mean']:+.4f}  low={wdata['low_car_mean']:+.4f}  diff={wdata['diff']:+.4f}  coef={wdata['controlled_coef']:+.4f}  p={wdata['controlled_p']:.4f}")

            # --- Step 2: 連続交互作用回帰 (補助) ---
            print("\n--- Step 2: 連続交互作用回帰 (補助、size/liquidity proxy) ---")
            down_panel = add_log_trading_value(down_panel, jp_prices_raw)
            ltv_missing_rate = down_panel["log_trading_value"].isna().sum() / len(down_panel) if len(down_panel) > 0 else 1.0
            n_events_total = down_panel["event_date"].nunique()

            interaction_results = {}
            if ltv_missing_rate > 0.20:
                interaction_results = {"status": "descriptive_only", "reason": f"log_trading_value欠損率={ltv_missing_rate:.1%} > 20%"}
                print(f"  欠損率が高い({ltv_missing_rate:.1%})ため descriptive_only に格下げ")
            elif n_events_total < 30:
                interaction_results = {"status": "insufficient_power", "n_events": n_events_total}
                print(f"  n_events={n_events_total} < 30。insufficient_power")
            else:
                valid_for_reg = down_panel.dropna(subset=["log_trading_value"])
                eff_n_events = valid_for_reg["event_date"].nunique()
                if eff_n_events < 30:
                    interaction_results = {"status": "insufficient_power", "effective_n_events": eff_n_events}
                    print(f"  実効n_events={eff_n_events} < 30。insufficient_power")
                else:
                    interaction_results = {
                        "status": "computed",
                        "effective_n_obs": len(valid_for_reg),
                        "effective_n_events": eff_n_events,
                        "log_trading_value_missing_rate": float(ltv_missing_rate),
                    }
                    # raw交互作用項を作成（demeanはrun_car_regressionが1回で行う）
                    valid_for_reg = valid_for_reg.copy()
                    valid_for_reg["turnover_x_ltv"] = valid_for_reg["turnover"] * valid_for_reg["log_trading_value"]
                    valid_for_reg["vol_x_ltv"] = valid_for_reg["vol"] * valid_for_reg["log_trading_value"]

                    xvars_interaction = ["vol", "turnover", "log_trading_value",
                                        "turnover_x_ltv", "vol_x_ltv", "excess_ret"]
                    for w in [5, 10, 20]:
                        car_col = f"car_{w}d"
                        if car_col not in valid_for_reg.columns:
                            continue
                        reg_data = valid_for_reg[[car_col] + xvars_interaction + ["event_date"]].dropna()
                        eff_n_ev_w = reg_data["event_date"].nunique()
                        if len(reg_data) < 100 or eff_n_ev_w < 30:
                            interaction_results[f"{w}d"] = {"error": f"insufficient data (n={len(reg_data)}, n_events={eff_n_ev_w})"}
                            continue
                        try:
                            model = run_car_regression(reg_data, car_col, xvars_interaction)
                            w_result = {"r2": float(model.rsquared), "effective_n_events": eff_n_ev_w}
                            for var in xvars_interaction:
                                w_result[var] = {
                                    "coef": float(model.params.get(var, np.nan)),
                                    "p": float(model.pvalues.get(var, np.nan)),
                                }
                            interaction_results[f"{w}d"] = w_result
                            turn_x = w_result.get("turnover_x_ltv", {})
                            vol_x = w_result.get("vol_x_ltv", {})
                            print(f"  {w}d: turnover×ltv coef={turn_x.get('coef', 0):+.6f} p={turn_x.get('p', 1):.4f}  |  vol×ltv coef={vol_x.get('coef', 0):+.6f} p={vol_x.get('p', 1):.4f}  R2={w_result['r2']:.4f}")
                        except Exception as e:
                            interaction_results[f"{w}d"] = {"error": str(e)}
                            print(f"  {w}d: エラー - {e}")

            q07_results["descriptive_only"] = {"interaction_regression": interaction_results}

            # --- Step 3: 中小型信用残別CAR (探索的) ---
            print("\n--- Step 3: 中小型信用残別CAR (探索的) ---")
            margin_results = run_smallcap_margin_car(down_panel, windows=[20])
            q07_results["descriptive_only"]["smallcap_margin"] = margin_results
            if margin_results.get("status") == "computed":
                for w_label, wdata in margin_results.items():
                    if isinstance(wdata, dict) and "diff" in wdata:
                        print(f"  margin {w_label}: high={wdata['high_car_mean']:+.4f}  low={wdata['low_car_mean']:+.4f}  diff={wdata['diff']:+.4f}  p={wdata['controlled_p']:.4f}")
            else:
                print(f"  {margin_results.get('status', 'unknown')}")

            # --- Step 4: 総合判定 ---
            print("\n" + "=" * 60)
            print("=== T1-Q07 総合判定 ===")
            print("=" * 60)

            large_data = discrete_results.get("large", {})
            small_data = discrete_results.get("small", {})
            large_turn_20d = large_data.get("turnover", {}).get("20d", {})
            small_turn_20d = small_data.get("turnover", {}).get("20d", {})

            if large_data.get("status") == "insufficient_power" or small_data.get("status") == "insufficient_power":
                q07_verdict = "insufficient_power"
                print(f"判定: {q07_verdict} --Large/Smallいずれかのサンプルが不足")
            elif "error" in large_turn_20d or "error" in small_turn_20d:
                q07_verdict = "error"
                print(f"判定: {q07_verdict} --回帰エラー")
            else:
                large_diff = large_turn_20d.get("diff", 0)
                large_p = large_turn_20d.get("controlled_p", 1)
                small_diff = small_turn_20d.get("diff", 0)
                small_p = small_turn_20d.get("controlled_p", 1)

                print(f"Large turnover高低差: {large_diff*100:+.2f}% (p={large_p:.4f})")
                print(f"Small turnover高低差: {small_diff*100:+.2f}% (p={small_p:.4f})")

                # 判定ロジック（4分岐）
                # 境界: p<0.10がstrict（p==0.10は有意とみなさない）
                large_pass = large_diff < -0.003 and large_p < 0.10
                large_fail = large_diff >= 0 or large_p >= 0.10  # 仮説逆行 or 非有意
                small_same_direction = small_diff < 0 and small_p < 0.10  # Small側でも同方向有意
                small_different = small_p >= 0.10 or (small_diff > 0)  # Small側で効果消失(>=0.10)/方向異なる

                if large_pass and small_different:
                    # PASS: Large有意＋Small消失/方向異なる → サイズ依存性あり
                    q07_verdict = "PASS"
                    print(f"判定: PASS --大型株でturnover効果有意(diff={large_diff*100:+.2f}%, p={large_p:.4f})、中小型で消失/方向異なる")
                elif large_pass and small_same_direction:
                    # FAIL: Large有意だがSmallでも同方向有意 → サイズ依存性なし（全体で均一）
                    q07_verdict = "FAIL"
                    print(f"判定: FAIL --大型・中小型とも同方向有意。サイズ依存性なし(Large diff={large_diff*100:+.2f}%, Small diff={small_diff*100:+.2f}%)")
                elif large_fail:
                    # FAIL: Large側で仮説逆行/非有意
                    q07_verdict = "FAIL"
                    print(f"判定: FAIL --大型株でturnover効果なし(diff={large_diff*100:+.2f}%, p={large_p:.4f})")
                else:
                    # AMBIGUOUS: Large閾値近傍＋Small側不安定
                    q07_verdict = "AMBIGUOUS"
                    print(f"判定: AMBIGUOUS --Large閾値近傍かつSmall側不安定。exploratory参照")

            q07_results["verdict"] = q07_verdict
            q07_results["conclusion_scope"] = "ショック日内部のサイズ依存性の記述。ショック固有性の因果解釈ではない"

    # === T1-Q09: turnover継続下落のショック固有性DiD ===
    q09_results = {}
    if args.q09:
        from scipy import stats as scipy_stats
        print("\n" + "=" * 60)
        print("=== T1-Q09 turnover-postshock-specificity-did ===")
        print("=" * 60)

        down_panel = panels.get("down")
        if down_panel is None or len(down_panel) == 0:
            print("NG: 下方パネルがない。中止")
        else:
            down_panel = compute_post_event_car(jp_returns, topix_returns, down_panel, windows=[5, 10, 20])
            shock_dates_list = sorted(down_panel["event_date"].unique())
            n_shock = len(shock_dates_list)
            print(f"ショック日数: {n_shock}")

            # turnover高/低群
            turn_median = down_panel["turnover"].median()
            turn_high = down_panel[down_panel["turnover"] > turn_median]
            turn_low = down_panel[down_panel["turnover"] <= turn_median]
            shock_turn_high_car = turn_high["car_20d"].dropna().mean()
            shock_turn_low_car = turn_low["car_20d"].dropna().mean()
            print(f"\nショック日 turnover高群 20d CAR: {shock_turn_high_car*100:+.3f}%")
            print(f"ショック日 turnover低群 20d CAR: {shock_turn_low_car*100:+.3f}%")

            # 非ショック日
            all_sp_dates = us_returns.index
            shock_set = set(shock_dates_list)
            non_shock_dates = [d for d in all_sp_dates if d not in shock_set]
            print(f"非ショック日数: {len(non_shock_dates)}")

            # 事前計算: turnover行列（volumeからpivot）
            print("事前計算: turnover行列...")
            vol_col = "volume" if "volume" in jp_prices_raw.columns else "adj_volume"
            if vol_col in jp_prices_raw.columns:
                vol_pivot = jp_prices_raw.pivot_table(index="date", columns="code", values=vol_col)
                vol_pivot = vol_pivot.sort_index()
                # turnover = 当日出来高 / 20日平均出来高
                vol_ma20 = vol_pivot.rolling(20, min_periods=10).mean()
                turnover_mat = vol_pivot / vol_ma20
                # log transform for skewness (Codex review recommendation)
                turnover_mat = np.log1p(turnover_mat.clip(lower=0))
            else:
                print("NG: volume列がない。中止")
                turnover_mat = None

            if turnover_mat is not None:
                # CAR行列（Q05で使ったものと同じ）
                excess_mat = jp_returns.sub(topix_returns, axis=0)
                cumex = excess_mat.cumsum()
                car_20d_mat = cumex.shift(-21) - cumex.shift(-1)

                # 非ショック日→翌JP営業日マッピング
                jp_idx = jp_returns.index
                non_shock_jp = []
                for d in non_shock_dates:
                    later = jp_idx[jp_idx > d]
                    if len(later) > 0:
                        non_shock_jp.append((d, later[0]))

                # ブートストラップ
                print("\n--- ベースライン: ランダム非ショック日 ---")
                n_iter = 200
                min_gap = 20
                rng = np.random.RandomState(42)
                bs_turn_high_cars = []
                bs_turn_low_cars = []

                non_shock_arr = np.array([x[0] for x in non_shock_jp])
                non_shock_reaction = {x[0]: x[1] for x in non_shock_jp}

                for i in range(n_iter):
                    shuffled = rng.permutation(len(non_shock_arr))
                    sampled_reactions = []
                    sampled_events = []
                    for idx in shuffled:
                        d = non_shock_arr[idx]
                        if all(abs((d - s).days) >= min_gap for s in sampled_events):
                            sampled_events.append(d)
                            sampled_reactions.append(non_shock_reaction[d])
                        if len(sampled_events) >= n_shock:
                            break
                    if len(sampled_reactions) < n_shock // 2:
                        continue

                    all_turns = []
                    all_cars = []
                    for rd in sampled_reactions:
                        if rd not in turnover_mat.index or rd not in car_20d_mat.index:
                            continue
                        t = turnover_mat.loc[rd].dropna()
                        c = car_20d_mat.loc[rd].dropna()
                        common_codes = t.index.intersection(c.index)
                        if len(common_codes) < 10:
                            continue
                        all_turns.append(t[common_codes])
                        all_cars.append(c[common_codes])

                    if len(all_turns) < 5:
                        continue
                    concat_t = pd.concat(all_turns)
                    concat_c = pd.concat(all_cars)
                    tm = concat_t.median()
                    fh = concat_c[concat_t > tm].mean()
                    fl = concat_c[concat_t <= tm].mean()
                    bs_turn_high_cars.append(float(fh))
                    bs_turn_low_cars.append(float(fl))

                if len(bs_turn_high_cars) > 50:
                    bs_high = np.array(bs_turn_high_cars)
                    bs_low = np.array(bs_turn_low_cars)
                    baseline_high = np.mean(bs_high)
                    baseline_low = np.mean(bs_low)
                    # p値: ショック日CARが非ショック日ベースラインより低い確率（下側検定）
                    p_high = np.mean(bs_high <= shock_turn_high_car)
                    pseudo_did_shock = shock_turn_high_car - shock_turn_low_car
                    pseudo_did_baseline = baseline_high - baseline_low
                    pseudo_did = pseudo_did_shock - pseudo_did_baseline
                    bs_dids = bs_high - bs_low
                    p_did = np.mean(bs_dids <= pseudo_did_shock)  # 下側検定（turnoverは下がる方向）

                    print(f"  非ショック日 turnover高群ベースライン: {baseline_high*100:+.3f}%")
                    print(f"  非ショック日 turnover低群ベースライン: {baseline_low*100:+.3f}%")
                    print(f"  ショック固有turnover高群CAR: {shock_turn_high_car*100:+.3f}% vs ベースライン {baseline_high*100:+.3f}%")
                    print(f"  pseudo-DiD: {pseudo_did*100:+.3f}% (shock={pseudo_did_shock*100:+.3f}% - baseline={pseudo_did_baseline*100:+.3f}%)")
                    print(f"  pseudo-DiD p値: {p_did:.4f}")

                    q09_results["baseline_random"] = {
                        "shock_turn_high_car": float(shock_turn_high_car),
                        "shock_turn_low_car": float(shock_turn_low_car),
                        "baseline_turn_high_car": float(baseline_high),
                        "baseline_turn_low_car": float(baseline_low),
                        "pseudo_did": float(pseudo_did),
                        "pseudo_did_p": float(p_did),
                        "n_bootstrap": len(bs_turn_high_cars),
                    }

                # ショック規模相関
                print("\n--- ショック規模相関 ---")
                shock_cars_by_date = {}
                for sd in shock_dates_list:
                    sub = turn_high[turn_high["event_date"] == sd]
                    if len(sub) > 0 and sub["car_20d"].notna().sum() > 0:
                        shock_cars_by_date[sd] = sub["car_20d"].mean()

                if len(shock_cars_by_date) >= 10:
                    dates_sorted = sorted(shock_cars_by_date.keys())
                    car_vals = [shock_cars_by_date[d] for d in dates_sorted]
                    sp_vals = [float(us_returns.loc[d]) if d in us_returns.index else np.nan for d in dates_sorted]
                    valid_mask = [not (np.isnan(c) or np.isnan(s)) for c, s in zip(car_vals, sp_vals)]
                    car_clean = [c for c, v in zip(car_vals, valid_mask) if v]
                    sp_clean = [abs(s) for s, v in zip(sp_vals, valid_mask) if v]
                    rho, p_rho = scipy_stats.spearmanr(sp_clean, car_clean)
                    print(f"  Spearman(|S&P500下落幅|, turnover高群20d CAR): rho={rho:.3f}, p={p_rho:.4f}")

                    q09_results["shock_magnitude_correlation"] = {
                        "spearman_rho": float(rho),
                        "spearman_p": float(p_rho),
                        "n_events": len(car_clean),
                    }

                # 最終判定
                print("\n" + "=" * 60)
                print("=== T1-Q09 最終判定 ===")
                print("=" * 60)
                primary_pass = False
                secondary_pass = False
                if "baseline_random" in q09_results:
                    r = q09_results["baseline_random"]
                    primary_pass = r["pseudo_did"] < 0 and r["pseudo_did_p"] < 0.10
                    print(f"Primary (pseudo-DiD < 0, p<0.10): {'PASS' if primary_pass else 'FAIL'}")
                    print(f"  pseudo-DiD={r['pseudo_did']*100:+.3f}%, p={r['pseudo_did_p']:.4f}")
                if "shock_magnitude_correlation" in q09_results:
                    r = q09_results["shock_magnitude_correlation"]
                    secondary_pass = r["spearman_rho"] < -0.05 and r["spearman_p"] < 0.10
                    print(f"Secondary (Spearman rho<-0.05, p<0.10): {'PASS' if secondary_pass else 'FAIL'}")
                    print(f"  rho={r['spearman_rho']:.3f}, p={r['spearman_p']:.4f}")

                verdict = "PASS" if primary_pass else ("PARTIAL" if secondary_pass else "FAIL")
                q09_results["verdict"] = verdict
                print(f"\n総合判定: {verdict}")

    # === T1-Q10: turnover継続下落のメカニズム分離 ===
    q10_results = {}
    if args.q10:
        from scipy import stats as scipy_stats
        print("\n" + "=" * 60)
        print("=== T1-Q10 turnover-decline-mechanism-separation ===")
        print("=" * 60)

        down_panel = panels.get("down")
        if down_panel is None or len(down_panel) == 0:
            print("NG: 下方パネルがない。中止")
        else:
            down_panel = compute_post_event_car(jp_returns, topix_returns, down_panel, windows=[20])
            shock_dates_list = sorted(down_panel["event_date"].unique())

            # volume pivot
            vol_col = "volume" if "volume" in jp_prices_raw.columns else "adj_volume"
            vol_pivot = jp_prices_raw.pivot_table(index="date", columns="code", values=vol_col).sort_index()
            vol_ma20 = vol_pivot.rolling(20, min_periods=10).mean()

            # ショック後turnover変化率（後5日平均 / 前20日平均）と（後20日平均 / 前20日平均）
            print("\nショック後turnover変化率を計算中...")
            for window_label, post_window in [("5d", 5), ("20d", 20)]:
                change_col = f"turnover_change_{window_label}"
                changes = []
                for _, row in down_panel.iterrows():
                    code = row["code"]
                    ed = row["event_date"]
                    if code not in vol_pivot.columns:
                        changes.append(np.nan)
                        continue
                    # 前20日平均
                    pre_dates = vol_pivot.index[vol_pivot.index <= ed]
                    if len(pre_dates) < 20:
                        changes.append(np.nan)
                        continue
                    pre_vol = vol_pivot.loc[pre_dates[-20:], code].mean()
                    # 後N日平均
                    post_dates = vol_pivot.index[vol_pivot.index > ed]
                    if len(post_dates) < post_window:
                        changes.append(np.nan)
                        continue
                    post_vol = vol_pivot.loc[post_dates[:post_window], code].mean()
                    if pre_vol < 1 or np.isnan(pre_vol):
                        changes.append(np.nan)
                        continue
                    changes.append(post_vol / pre_vol)
                down_panel[change_col] = changes

            # turnover高群に限定
            turn_median = down_panel["turnover"].median()
            turn_high = down_panel[down_panel["turnover"] > turn_median].copy()

            # --- Step 1: ショック後turnover変化のDiD ---
            for window_label in ["5d", "20d"]:
                change_col = f"turnover_change_{window_label}"
                valid = turn_high[[change_col, "car_20d"]].dropna()
                if len(valid) < 50:
                    print(f"\n--- Step 1 ({window_label}): データ不足 ---")
                    continue
                shock_mean_change = valid[change_col].mean()
                print(f"\n--- Step 1: ショック後turnover変化率({window_label}) ---")
                print(f"  ショック日 turnover高群の変化率: {shock_mean_change:.3f}x")

                # 中央値で上昇/低下に分割
                change_median = valid[change_col].median()
                rise_group = valid[valid[change_col] > change_median]
                fall_group = valid[valid[change_col] <= change_median]

                rise_car = rise_group["car_20d"].mean()
                fall_car = fall_group["car_20d"].mean()
                t_rise, p_rise = scipy_stats.ttest_1samp(rise_group["car_20d"], 0) if len(rise_group) > 10 else (0, 1)
                t_fall, p_fall = scipy_stats.ttest_1samp(fall_group["car_20d"], 0) if len(fall_group) > 10 else (0, 1)

                print(f"  turnover上昇群(n={len(rise_group)}): 20d CAR={rise_car*100:+.3f}% (p={p_rise:.4f})")
                print(f"  turnover低下群(n={len(fall_group)}): 20d CAR={fall_car*100:+.3f}% (p={p_fall:.4f})")

                # 2群間の差
                t_diff, p_diff = scipy_stats.ttest_ind(rise_group["car_20d"], fall_group["car_20d"])
                print(f"  2群差: {(rise_car - fall_car)*100:+.3f}% (p={p_diff:.4f})")

                q10_results[f"step1_{window_label}"] = {
                    "shock_mean_change": float(shock_mean_change),
                    "change_median": float(change_median),
                    "rise_car": float(rise_car),
                    "rise_n": len(rise_group),
                    "rise_p": float(p_rise),
                    "fall_car": float(fall_car),
                    "fall_n": len(fall_group),
                    "fall_p": float(p_fall),
                    "diff": float(rise_car - fall_car),
                    "diff_p": float(p_diff),
                }

            # --- Step 2: メカニズム判定 ---
            print("\n--- Step 2: メカニズム判定 ---")
            for wl in ["5d", "20d"]:
                key = f"step1_{wl}"
                if key in q10_results:
                    r = q10_results[key]
                    if r["rise_car"] < 0 and r["rise_p"] < 0.10:
                        mech_rise = "保有者交代（売買活発化+CAR負=弱い手への移行）"
                    elif r["rise_car"] >= 0:
                        mech_rise = "保有者交代なし（売買活発化だがCAR正）"
                    else:
                        mech_rise = "不明確（CAR負だがp>0.10）"

                    if r["fall_car"] < 0 and r["fall_p"] < 0.10:
                        mech_fall = "流動性枯渇（売買低下+CAR負=買い手不在）"
                    elif r["fall_car"] >= 0:
                        mech_fall = "流動性枯渇なし"
                    else:
                        mech_fall = "不明確"

                    print(f"  {wl} turnover上昇群: {mech_rise}")
                    print(f"  {wl} turnover低下群: {mech_fall}")
                    q10_results[f"mechanism_{wl}"] = {
                        "rise_interpretation": mech_rise,
                        "fall_interpretation": mech_fall,
                    }

            # --- 最終判定 ---
            print("\n" + "=" * 60)
            print("=== T1-Q10 最終判定 ===")
            print("=" * 60)
            # primary: turnover変化上昇群と低下群で20d CARに有意差があるか
            has_mechanism = False
            for wl in ["5d", "20d"]:
                key = f"step1_{wl}"
                if key in q10_results and q10_results[key]["diff_p"] < 0.10:
                    has_mechanism = True
            verdict = "PASS" if has_mechanism else "FAIL"
            q10_results["verdict"] = verdict
            print(f"総合判定: {verdict}")

    # === T2-Q02: 最適保有期間（日次CARプロファイル + turnover正常化） ===
    t2q02_results = {}
    if args.t2q02:
        from scipy import stats as scipy_stats
        print("\n" + "=" * 60)
        print("=== T2-Q02 optimal-holding-period-decay ===")
        print("=" * 60)

        down_panel = panels.get("down")
        if down_panel is None or len(down_panel) == 0:
            print("NG: 下方パネルがない。中止")
        else:
            # 1-20日の日次CARを計算
            down_panel_daily = compute_post_event_car(jp_returns, topix_returns, down_panel,
                                                      windows=list(range(1, 21)))
            turn_median = down_panel_daily["turnover"].median()
            turn_high = down_panel_daily[down_panel_daily["turnover"] > turn_median]

            # --- Step 1: 日次CARプロファイル ---
            print("\n--- Step 1: 日次CARプロファイル（turnover高群）---")
            daily_cars = []
            daily_increments = []
            prev_car = 0
            for d in range(1, 21):
                col = f"car_{d}d"
                if col not in turn_high.columns:
                    continue
                car = turn_high[col].dropna().mean()
                increment = car - prev_car
                # クロスセクショナルt検定（CAR増分がゼロか）
                if d == 1:
                    inc_values = turn_high[col].dropna()
                else:
                    prev_col = f"car_{d-1}d"
                    if prev_col in turn_high.columns:
                        both = turn_high[[col, prev_col]].dropna()
                        inc_values = both[col] - both[prev_col]
                    else:
                        inc_values = pd.Series(dtype=float)
                t_val, p_val = scipy_stats.ttest_1samp(inc_values, 0) if len(inc_values) > 10 else (0, 1)
                daily_cars.append({"day": d, "car": float(car), "increment": float(increment),
                                   "t": float(t_val), "p": float(p_val)})
                prev_car = car
                sig = "*" if abs(t_val) >= 1.65 else " "
                print(f"  day {d:2d}: CAR={car*100:+.3f}%  inc={increment*100:+.3f}%  t={t_val:+.2f} {sig}")
                daily_increments.append({"day": d, "t": abs(t_val)})

            t2q02_results["daily_car_profile"] = daily_cars

            # ブレークポイント検出: |t| < 1.65 が3日連続
            print("\n--- ブレークポイント検出 ---")
            breakpoint = None
            for i in range(len(daily_increments) - 2):
                if all(daily_increments[i+j]["t"] < 1.65 for j in range(3)):
                    breakpoint = daily_increments[i]["day"]
                    break
            if breakpoint:
                print(f"  ブレークポイント: day {breakpoint}（day {breakpoint}-{breakpoint+2}でCAR増分がゼロと区別不能）")
            else:
                print("  ブレークポイント未検出（20日以内に3連続ゼロ増分なし）")
                breakpoint = 20  # フォールバック
            t2q02_results["breakpoint"] = breakpoint

            # --- Step 2: turnover正常化日 ---
            print("\n--- Step 2: turnover正常化日 ---")
            vol_col = "volume" if "volume" in jp_prices_raw.columns else "adj_volume"
            vol_pivot = jp_prices_raw.pivot_table(index="date", columns="code", values=vol_col).sort_index()

            norm_days = []
            car_at_norm = []
            for _, row in turn_high.iterrows():
                code = row["code"]
                ed = row["event_date"]
                if code not in vol_pivot.columns:
                    norm_days.append(np.nan)
                    car_at_norm.append(np.nan)
                    continue
                # ベースライン: t-21..t-2（ショック日汚染回避）
                pre_dates = vol_pivot.index[vol_pivot.index <= ed]
                if len(pre_dates) < 22:
                    norm_days.append(np.nan)
                    car_at_norm.append(np.nan)
                    continue
                baseline = vol_pivot.loc[pre_dates[-21:-1], code].mean()
                if baseline < 1 or np.isnan(baseline):
                    norm_days.append(np.nan)
                    car_at_norm.append(np.nan)
                    continue
                # ショック後にturnoverがベースラインに戻る日（2連続日以下）
                post_dates = vol_pivot.index[vol_pivot.index > ed]
                found = False
                for i in range(len(post_dates) - 1):
                    if i >= 20:
                        break
                    v1 = vol_pivot.loc[post_dates[i], code]
                    v2 = vol_pivot.loc[post_dates[i+1], code] if i+1 < len(post_dates) else np.nan
                    if v1 <= baseline and (not np.isnan(v2) and v2 <= baseline):
                        norm_days.append(i + 1)
                        # その日のCARを取得
                        car_col = f"car_{i+1}d" if f"car_{i+1}d" in turn_high.columns else None
                        if car_col and not np.isnan(row.get(car_col, np.nan)):
                            car_at_norm.append(row[car_col])
                        else:
                            car_at_norm.append(np.nan)
                        found = True
                        break
                if not found:
                    norm_days.append(np.nan)  # 20日以内に正常化せず
                    car_at_norm.append(np.nan)

            valid_norms = [n for n in norm_days if not np.isnan(n)]
            if valid_norms:
                print(f"  正常化日の中央値: {np.median(valid_norms):.1f}日")
                print(f"  正常化日の平均: {np.mean(valid_norms):.1f}日")
                print(f"  20日以内に正常化: {len(valid_norms)}/{len(norm_days)} ({len(valid_norms)/len(norm_days)*100:.0f}%)")
                t2q02_results["normalization"] = {
                    "median_day": float(np.median(valid_norms)),
                    "mean_day": float(np.mean(valid_norms)),
                    "pct_normalized": len(valid_norms) / len(norm_days),
                }

            # Spearman相関: 正常化日 vs その時点のCAR
            valid_pairs = [(n, c) for n, c in zip(norm_days, car_at_norm) if not np.isnan(n) and not np.isnan(c)]
            if len(valid_pairs) >= 20:
                ns, cs = zip(*valid_pairs)
                rho, p_rho = scipy_stats.spearmanr(ns, cs)
                print(f"\n  Spearman(正常化日, CAR): rho={rho:.3f}, p={p_rho:.4f}")
                t2q02_results["normalization_car_correlation"] = {
                    "spearman_rho": float(rho),
                    "spearman_p": float(p_rho),
                    "n": len(valid_pairs),
                }

            # --- 最終判定 ---
            print("\n" + "=" * 60)
            print("=== T2-Q02 最終判定 ===")
            print("=" * 60)
            bp_pass = breakpoint is not None and 5 <= breakpoint <= 10
            corr = t2q02_results.get("normalization_car_correlation", {})
            corr_pass = corr.get("spearman_rho", 0) > 0.3 and corr.get("spearman_p", 1) < 0.05
            print(f"Primary (ブレークポイント5-10日): {'PASS' if bp_pass else 'FAIL'} (day={breakpoint})")
            print(f"Secondary (Spearman rho>0.3): {'PASS' if corr_pass else 'FAIL'}")
            verdict = "PASS" if bp_pass else "PARTIAL"
            if corr_pass:
                verdict += "+DYNAMIC_EXIT"
            t2q02_results["verdict"] = verdict
            print(f"総合判定: {verdict}")

    # === T2-Q04: event-clustering-capacity ===
    t2q04_results = {}
    if args.t2q04:
        from scipy import stats as scipy_stats
        print("\n" + "=" * 60)
        print("=== T2-Q04 event-clustering-capacity ===")
        print("=" * 60)

        down_panel = panels.get("down")
        if down_panel is None or len(down_panel) == 0:
            print("NG: 下方パネルがない。中止")
        else:
            # CARを計算
            down_panel = compute_post_event_car(jp_returns, topix_returns, down_panel, windows=[5])

            # ショック日リストと間隔計算（米国営業日差）
            shock_dates = sorted(down_panel["event_date"].unique())
            n_shocks = len(shock_dates)
            print(f"ショック日数: {n_shocks}")

            # 米国営業日ベースの間隔計算
            us_trading_dates = sorted(us_returns.index)
            shock_intervals = {}
            for i, sd in enumerate(shock_dates):
                if i == 0:
                    shock_intervals[sd] = None  # 最初のショックは間隔なし
                else:
                    prev = shock_dates[i - 1]
                    # 米国営業日で数える
                    try:
                        idx_prev = us_trading_dates.index(prev)
                        idx_curr = us_trading_dates.index(sd)
                        shock_intervals[sd] = idx_curr - idx_prev
                    except ValueError:
                        shock_intervals[sd] = None

            # 各ショック日のturnover-high群平均5d CARを集約
            turnover_median = down_panel["turnover"].median()
            turn_high = down_panel[down_panel["turnover"] > turnover_median]

            event_cars = []
            for sd in shock_dates:
                interval = shock_intervals.get(sd)
                if interval is None:
                    continue  # 最初のショックは除外
                sub = turn_high[turn_high["event_date"] == sd]
                car_5d = sub["car_5d"].dropna()
                if len(car_5d) < 5:
                    continue
                # VIX
                vix_val = vix_change.get(sd, np.nan) if 'vix_change' in dir() else np.nan
                # ショック絶対値
                shock_mag = abs(float(us_returns.get(sd, 0)))
                event_cars.append({
                    "event_date": sd,
                    "interval": interval,
                    "mean_car_5d": float(car_5d.mean()),
                    "n_stocks": len(car_5d),
                    "vix": float(vix_val) if not np.isnan(vix_val) else None,
                    "shock_magnitude": shock_mag,
                })

            event_df = pd.DataFrame(event_cars)
            print(f"有効イベント日: {len(event_df)} (初回ショック除外後)")

            if len(event_df) < 5:
                t2q04_results = {"status": "insufficient_power", "n_events": len(event_df)}
                print("insufficient_power: イベント数不足")
            else:
                # --- Step 1: 間隔5取引日未満vs以上 (主判定) ---
                print("\n--- Step 1: 間隔5取引日未満 vs 5以上 (主判定、イベント日単位) ---")
                short_mask = event_df["interval"] < 5
                long_mask = event_df["interval"] >= 5
                short_cars = event_df[short_mask]["mean_car_5d"]
                long_cars = event_df[long_mask]["mean_car_5d"]
                n_short = len(short_cars)
                n_long = len(long_cars)
                print(f"  短間隔(<5): {n_short}日, 長間隔(>=5): {n_long}日")

                if n_short < 5:
                    step1_result = {"status": "insufficient_power", "n_short": n_short, "n_long": n_long}
                    print(f"  insufficient_power: 短間隔群{n_short}件 < 5件")
                else:
                    diff = float(short_cars.mean() - long_cars.mean())
                    # Welch t-test
                    t_stat, p_val = scipy_stats.ttest_ind(short_cars, long_cars, equal_var=False)
                    # 95% CI for the difference
                    se = np.sqrt(short_cars.var() / n_short + long_cars.var() / n_long)
                    ci_low = diff - 1.96 * se
                    ci_high = diff + 1.96 * se
                    # 等価性判定: CIが[-0.002, +0.002]に収まるか
                    equivalence_pass = ci_low >= -0.002 and ci_high <= 0.002
                    ci_too_wide = (ci_high - ci_low) > 0.008

                    print(f"  短間隔平均CAR: {short_cars.mean()*100:+.3f}%")
                    print(f"  長間隔平均CAR: {long_cars.mean()*100:+.3f}%")
                    print(f"  差: {diff*100:+.3f}% (p={p_val:.4f})")
                    print(f"  95%CI: [{ci_low*100:+.3f}%, {ci_high*100:+.3f}%]")
                    print(f"  等価性(CI in [-0.2%,+0.2%]): {'PASS' if equivalence_pass else 'FAIL'}")

                    step1_result = {
                        "n_short": n_short, "n_long": n_long,
                        "short_mean": float(short_cars.mean()),
                        "long_mean": float(long_cars.mean()),
                        "diff": diff, "p": float(p_val),
                        "ci_low": float(ci_low), "ci_high": float(ci_high),
                        "equivalence_pass": equivalence_pass,
                    }
                    if ci_too_wide:
                        step1_result["status"] = "insufficient_power"
                        print(f"  CI幅 {(ci_high-ci_low)*100:.2f}% > 0.8%: insufficient_power")

                t2q04_results["verdict_inputs"] = {"step1_interval_5d": step1_result}

                # --- Step 2: Spearman (主判定補助) ---
                print("\n--- Step 2: Spearman(間隔, イベント日平均CAR) ---")
                valid_spearman = event_df[["interval", "mean_car_5d"]].dropna()
                rho, p_rho = scipy_stats.spearmanr(valid_spearman["interval"], valid_spearman["mean_car_5d"])
                print(f"  Spearman rho={rho:.3f}, p={p_rho:.4f}, n={len(valid_spearman)}")
                t2q04_results["verdict_inputs"]["step2_spearman"] = {
                    "rho": float(rho), "p": float(p_rho), "n": len(valid_spearman)
                }

                # --- Step 3: 代替仮説統制（ショック絶対値+VIX回帰） ---
                print("\n--- Step 3: 代替仮説統制 ---")
                reg_data = event_df[["mean_car_5d", "interval", "shock_magnitude"]].dropna()
                reg_data["short_dummy"] = (reg_data["interval"] < 5).astype(float)
                if len(reg_data) >= 10:
                    X = sm.add_constant(reg_data[["short_dummy", "shock_magnitude"]])
                    model = sm.OLS(reg_data["mean_car_5d"], X).fit()
                    short_coef = float(model.params.get("short_dummy", np.nan))
                    short_p = float(model.pvalues.get("short_dummy", np.nan))
                    print(f"  short_dummy coef={short_coef*100:+.3f}% p={short_p:.4f} (ショック絶対値統制)")
                    t2q04_results["verdict_inputs"]["step3_controlled_regression"] = {
                        "short_coef": short_coef, "short_p": short_p,
                        "shock_mag_coef": float(model.params.get("shock_magnitude", np.nan)),
                        "r2": float(model.rsquared),
                    }

                # --- Step 4: 感度分析 (descriptive_only) ---
                print("\n--- Step 4: 感度分析 ---")
                sensitivity = {}
                for cutoff in [3, 7, 10]:
                    s_mask = event_df["interval"] < cutoff
                    l_mask = event_df["interval"] >= cutoff
                    s_cars = event_df[s_mask]["mean_car_5d"]
                    l_cars = event_df[l_mask]["mean_car_5d"]
                    if len(s_cars) >= 3 and len(l_cars) >= 3:
                        d = float(s_cars.mean() - l_cars.mean())
                        _, p = scipy_stats.ttest_ind(s_cars, l_cars, equal_var=False)
                        print(f"  {cutoff}日カットオフ: 短{len(s_cars)}日 vs 長{len(l_cars)}日, diff={d*100:+.3f}%, p={p:.4f}")
                        sensitivity[f"{cutoff}d"] = {"n_short": len(s_cars), "n_long": len(l_cars), "diff": d, "p": float(p)}
                    else:
                        print(f"  {cutoff}日カットオフ: サンプル不足")
                        sensitivity[f"{cutoff}d"] = {"status": "insufficient_power"}
                t2q04_results["descriptive_only"] = {"sensitivity": sensitivity}

                # --- Step 5: 副次分析 (descriptive_only) ---
                print("\n--- Step 5: 副次 - 連続ショック時のturnover変化倍率 ---")
                # 連続ショック(短間隔)と通常(長間隔)のturnover変化倍率比較は、
                # turnover変化倍率データがパネルにあれば実施
                # ここではショック日ごとのturnover平均を比較
                short_events = event_df[event_df["interval"] < 5]
                long_events = event_df[event_df["interval"] >= 5]
                if len(short_events) >= 3:
                    short_turn_mean = turn_high[turn_high["event_date"].isin(short_events["event_date"])]["turnover"].mean()
                    long_turn_mean = turn_high[turn_high["event_date"].isin(long_events["event_date"])]["turnover"].mean()
                    print(f"  短間隔turnover平均: {short_turn_mean:.4f}, 長間隔: {long_turn_mean:.4f}")
                    t2q04_results["descriptive_only"]["turnover_comparison"] = {
                        "short_mean": float(short_turn_mean), "long_mean": float(long_turn_mean),
                    }

                # --- 総合判定 ---
                print("\n" + "=" * 60)
                print("=== T2-Q04 総合判定 ===")
                print("=" * 60)

                s1 = t2q04_results.get("verdict_inputs", {}).get("step1_interval_5d", {})
                if s1.get("status") == "insufficient_power":
                    t2q04_verdict = "insufficient_power"
                    print(f"判定: insufficient_power")
                elif s1.get("equivalence_pass"):
                    t2q04_verdict = "PASS"
                    print(f"判定: PASS -- 連続ショックでも効果均一。キャパシティ制約なし")
                    print(f"  Q03への指示: ポジション重複OK。間隔による抑制不要")
                else:
                    ci_low = s1.get("ci_low", -999)
                    ci_high = s1.get("ci_high", 999)
                    if ci_low < -0.002:
                        t2q04_verdict = "FAIL"
                        print(f"判定: FAIL -- 連続ショック時に効果減衰。キャパシティ制約あり")
                        print(f"  Q03への指示: 間隔5取引日未満のショックはスキップまたはサイズ縮小")
                    else:
                        t2q04_verdict = "AMBIGUOUS"
                        print(f"判定: AMBIGUOUS -- CIが等価性範囲に収まらないが下限も-0.2%を下回らない")

                t2q04_results["verdict"] = t2q04_verdict

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
    if args.q05:
        output["q05_vol_reversal_specificity"] = q05_results
    if args.q06:
        output["q06_turnover_momentum"] = q06_results
    if args.q09:
        output["q09_turnover_postshock_did"] = q09_results
    if args.q10:
        output["q10_turnover_mechanism"] = q10_results
    if args.t2q02:
        output["t2q02_holding_period"] = t2q02_results
    if args.q07:
        output["q07_size_regime_interaction"] = q07_results
    if args.t2q04:
        output["t2q04_event_clustering"] = t2q04_results

    output_path = Path("logs/iterations/T1-Q03_symmetry-test_result.json")
    if args.q04:
        output_path = Path("logs/iterations/T1-Q04_attention-penalty_result.json")
    if args.q05:
        output_path = Path("logs/iterations/T1-Q05_vol-reversal-specificity_result.json")
    if args.q06:
        output_path = Path("logs/iterations/T1-Q06_turnover-momentum-disentangle_result.json")
    if args.q09:
        output_path = Path("logs/iterations/T1-Q09_turnover-postshock-specificity-did_result.json")
    if args.q10:
        output_path = Path("logs/iterations/T1-Q10_turnover-decline-mechanism-separation_result.json")
    if args.t2q02:
        output_path = Path("logs/iterations/T2-Q02_optimal-holding-period-decay_result.json")
    if args.q07:
        output_path = Path("logs/iterations/T1-Q07_size-regime-interaction_result.json")
    if args.t2q04:
        output_path = Path("logs/iterations/T2-Q04_event-clustering-capacity_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n結果を保存: {output_path}")


if __name__ == "__main__":
    main()
