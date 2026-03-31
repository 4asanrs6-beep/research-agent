"""
2026年Q1（1月〜3月）の市場データ取得・分析スクリプト
yfinanceでデータを取得し、月次/週次のリターン・ボラティリティ等を算出する
"""

import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ── 設定 ──────────────────────────────────────────────
OUTPUT_DIR = r"C:\Users\PC\Downloads\作業用\20260307\research-agent\logs\iterations"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "T3_market_analysis_2026q1.json")

START_DATE = "2025-12-30"   # 12月末の終値を基準にするため少し前から取得
END_DATE   = "2026-03-31"   # 3月末まで（yfinanceは end exclusive なので翌日不要だが安全マージン）

TICKERS = {
    # 米国株指数
    "SP500":     "^GSPC",
    "NASDAQ":    "^IXIC",
    "Russell2000": "^RUT",
    "VIX":       "^VIX",
    # 日本株指数
    "Nikkei225": "^N225",
    "TOPIX_ETF": "1306.T",
    # 為替
    "USDJPY":    "JPY=X",
    # 米国金利
    "US10Y":     "^TNX",
    "US2Y":      "2YY=F",
    # セクターETF (米国)
    "XLK":  "XLK",   # テック
    "XLF":  "XLF",   # 金融
    "XLE":  "XLE",   # エネルギー
    "XLV":  "XLV",   # ヘルスケア
    "XLI":  "XLI",   # 資本財
    "XLP":  "XLP",   # 生活必需品
    "XLU":  "XLU",   # 公益
    # スタイルETF (米国)
    "IWF":  "IWF",   # グロース
    "IWD":  "IWD",   # バリュー
    "MTUM": "MTUM",  # モメンタム
    "QUAL": "QUAL",  # クオリティ
    # 日本セクターETF
    "JP_Banks":   "1615.T",
    "JP_Foods":   "1617.T",
    "JP_Machine": "1620.T",
}

# ── データ取得 ─────────────────────────────────────────
print("=" * 70)
print("データ取得中...")
print("=" * 70)

all_close = {}
failed_tickers = []

for name, ticker in TICKERS.items():
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
        if data.empty:
            print(f"  [WARN] {name} ({ticker}): データなし")
            failed_tickers.append(name)
            continue
        # yfinance 1.x returns MultiIndex columns when single ticker
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"].iloc[:, 0]
        else:
            close = data["Close"]
        all_close[name] = close
        print(f"  [OK]   {name} ({ticker}): {len(close)} rows, {close.index[0].date()} ~ {close.index[-1].date()}")
    except Exception as e:
        print(f"  [FAIL] {name} ({ticker}): {e}")
        failed_tickers.append(name)

print(f"\n取得成功: {len(all_close)}/{len(TICKERS)}  失敗: {failed_tickers}")

# DataFrameにまとめる
df = pd.DataFrame(all_close)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# ── ユーティリティ関数 ─────────────────────────────────
def calc_return(series):
    """期間リターン (%)"""
    s = series.dropna()
    if len(s) < 2:
        return None
    return float(round((s.iloc[-1] / s.iloc[0] - 1) * 100, 4))

def calc_volatility(series):
    """期間内の日次リターンの標準偏差 (年率換算%, sqrt(252))"""
    s = series.dropna()
    if len(s) < 3:
        return None
    daily_ret = s.pct_change().dropna()
    return float(round(daily_ret.std() * np.sqrt(252) * 100, 4))

def calc_max_drawdown(series):
    """最大ドローダウン (%)"""
    s = series.dropna()
    if len(s) < 2:
        return None
    cummax = s.cummax()
    dd = (s - cummax) / cummax
    return float(round(dd.min() * 100, 4))

def safe_float(v):
    """JSON-safe float"""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return float(round(v, 4))

# ── 月次分析 ───────────────────────────────────────────
print("\n" + "=" * 70)
print("月次分析")
print("=" * 70)

MONTHS = {
    "2026-01": ("2026-01-01", "2026-01-31"),
    "2026-02": ("2026-02-01", "2026-02-28"),
    "2026-03": ("2026-03-01", "2026-03-31"),
}

monthly_results = {}

for month_label, (m_start, m_end) in MONTHS.items():
    monthly_results[month_label] = {}
    # 月の最初の営業日の終値を基準にするため、前月末の終値を使う
    # → リターンは月初終値 → 月末終値
    mask = (df.index >= m_start) & (df.index <= m_end)
    month_df = df.loc[mask]

    # 前月末の終値を取得（リターン計算の基準）
    prev_mask = df.index < m_start
    prev_df = df.loc[prev_mask]

    for col in df.columns:
        if col not in month_df.columns:
            continue
        series = month_df[col].dropna()
        if len(series) < 2:
            monthly_results[month_label][col] = {
                "return_pct": None,
                "volatility_ann_pct": None,
                "max_drawdown_pct": None,
            }
            continue

        # 前月末終値があれば、そこからのリターンを計算
        prev_series = prev_df[col].dropna()
        if len(prev_series) > 0:
            full_series = pd.concat([prev_series.iloc[[-1]], series])
            ret = calc_return(full_series)
        else:
            ret = calc_return(series)

        monthly_results[month_label][col] = {
            "return_pct": safe_float(ret),
            "volatility_ann_pct": safe_float(calc_volatility(series)),
            "max_drawdown_pct": safe_float(calc_max_drawdown(series)),
        }

    # VIXの月平均・月末値
    if "VIX" in month_df.columns:
        vix_series = month_df["VIX"].dropna()
        if len(vix_series) > 0:
            monthly_results[month_label]["_VIX_stats"] = {
                "mean": safe_float(vix_series.mean()),
                "month_end": safe_float(vix_series.iloc[-1]),
                "month_high": safe_float(vix_series.max()),
                "month_low": safe_float(vix_series.min()),
            }

# ── 3月 週次分析 ──────────────────────────────────────
print("\n" + "=" * 70)
print("3月 週次分析")
print("=" * 70)

march_mask = (df.index >= "2026-03-01") & (df.index <= "2026-03-31")
march_df = df.loc[march_mask]

# 週ごとに分割（ISO week）
march_df_copy = march_df.copy()
march_df_copy["week"] = march_df_copy.index.isocalendar().week.values

weekly_results = {}
weeks_sorted = sorted(march_df_copy["week"].unique())

# 2月末の終値（第1週のリターン基準）
feb_end_mask = (df.index >= "2026-02-01") & (df.index < "2026-03-01")
feb_end_df = df.loc[feb_end_mask]

prev_week_end = {}  # 前週末の終値を格納
if len(feb_end_df) > 0:
    for col in df.columns:
        s = feb_end_df[col].dropna()
        if len(s) > 0:
            prev_week_end[col] = s.iloc[-1]

for week_num in weeks_sorted:
    week_mask = march_df_copy["week"] == week_num
    week_df = march_df_copy.loc[week_mask]

    week_start = week_df.index[0].strftime("%Y-%m-%d")
    week_end = week_df.index[-1].strftime("%Y-%m-%d")
    week_label = f"W{week_num} ({week_start} ~ {week_end})"

    weekly_results[week_label] = {}

    for col in df.columns:
        series = week_df[col].dropna()
        if len(series) < 1:
            weekly_results[week_label][col] = {"return_pct": None}
            continue

        # 前週末終値からのリターン
        if col in prev_week_end and prev_week_end[col] is not None:
            ret = float(round((series.iloc[-1] / prev_week_end[col] - 1) * 100, 4))
        elif len(series) >= 2:
            ret = calc_return(series)
        else:
            ret = None

        weekly_results[week_label][col] = {
            "return_pct": safe_float(ret),
        }

    # 次の週の基準を更新
    for col in df.columns:
        s = week_df[col].dropna()
        if len(s) > 0:
            prev_week_end[col] = s.iloc[-1]

# ── 3月の転換点検出 ────────────────────────────────────
print("\n" + "=" * 70)
print("3月 転換点検出（日次リターン上位/下位5）")
print("=" * 70)

turning_points = {}
key_indices = ["SP500", "NASDAQ", "Russell2000", "Nikkei225", "USDJPY"]

for col in key_indices:
    if col not in march_df.columns:
        continue
    series = march_df[col].dropna()
    if len(series) < 2:
        continue
    daily_ret = series.pct_change().dropna() * 100
    # 上位5 (急騰)
    top5 = daily_ret.nlargest(5)
    # 下位5 (急落)
    bottom5 = daily_ret.nsmallest(5)

    turning_points[col] = {
        "largest_gains": [
            {"date": d.strftime("%Y-%m-%d"), "return_pct": safe_float(v)}
            for d, v in top5.items()
        ],
        "largest_losses": [
            {"date": d.strftime("%Y-%m-%d"), "return_pct": safe_float(v)}
            for d, v in bottom5.items()
        ],
    }

# ── Q1全体のサマリ ────────────────────────────────────
q1_mask = (df.index >= "2026-01-01") & (df.index <= "2026-03-31")
q1_df = df.loc[q1_mask]

# 2025年末の終値を基準
dec_end_mask = df.index < "2026-01-01"
dec_end_df = df.loc[dec_end_mask]

q1_summary = {}
for col in df.columns:
    q1_series = q1_df[col].dropna()
    dec_series = dec_end_df[col].dropna()
    if len(q1_series) < 2:
        continue
    if len(dec_series) > 0:
        full = pd.concat([dec_series.iloc[[-1]], q1_series])
        ret = calc_return(full)
    else:
        ret = calc_return(q1_series)

    q1_summary[col] = {
        "q1_return_pct": safe_float(ret),
        "q1_volatility_ann_pct": safe_float(calc_volatility(q1_series)),
        "q1_max_drawdown_pct": safe_float(calc_max_drawdown(q1_series)),
    }

# ── 結果構築 ──────────────────────────────────────────
result = {
    "metadata": {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "period": "2026-01-01 to 2026-03-30",
        "tickers": {k: v for k, v in TICKERS.items()},
        "failed_tickers": failed_tickers,
        "data_rows": len(df),
    },
    "q1_summary": q1_summary,
    "monthly": monthly_results,
    "march_weekly": weekly_results,
    "march_turning_points": turning_points,
}

# ── JSON保存 ──────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"\n結果を保存しました: {OUTPUT_FILE}")

# ── サマリ出力 ─────────────────────────────────────────
print("\n" + "=" * 70)
print("2026 Q1 市場分析サマリ")
print("=" * 70)

# Q1全体
print("\n■ Q1全体リターン")
print(f"  {'銘柄':<16} {'Q1リターン':>10} {'年率Vol':>10} {'最大DD':>10}")
print("  " + "-" * 50)
for col in ["SP500", "NASDAQ", "Russell2000", "Nikkei225", "TOPIX_ETF", "USDJPY", "US10Y"]:
    if col in q1_summary:
        s = q1_summary[col]
        r = f"{s['q1_return_pct']:+.2f}%" if s['q1_return_pct'] is not None else "N/A"
        v = f"{s['q1_volatility_ann_pct']:.1f}%" if s['q1_volatility_ann_pct'] is not None else "N/A"
        d = f"{s['q1_max_drawdown_pct']:.2f}%" if s['q1_max_drawdown_pct'] is not None else "N/A"
        print(f"  {col:<16} {r:>10} {v:>10} {d:>10}")

# セクターETF Q1
print("\n■ セクターETF Q1リターン")
sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU"]
for col in sector_etfs:
    if col in q1_summary:
        r = q1_summary[col]['q1_return_pct']
        print(f"  {col:<8} {r:+.2f}%" if r is not None else f"  {col:<8} N/A")

# スタイルETF Q1
print("\n■ スタイルETF Q1リターン")
style_etfs = ["IWF", "IWD", "MTUM", "QUAL"]
for col in style_etfs:
    if col in q1_summary:
        r = q1_summary[col]['q1_return_pct']
        print(f"  {col:<8} {r:+.2f}%" if r is not None else f"  {col:<8} N/A")

# 月次
for month_label in ["2026-01", "2026-02", "2026-03"]:
    print(f"\n■ {month_label} 月次リターン")
    m = monthly_results.get(month_label, {})
    for col in ["SP500", "NASDAQ", "Russell2000", "Nikkei225", "USDJPY"]:
        if col in m:
            r = m[col].get('return_pct')
            print(f"  {col:<16} {r:+.2f}%" if r is not None else f"  {col:<16} N/A")
    if "_VIX_stats" in m:
        vs = m["_VIX_stats"]
        print(f"  VIX  平均={vs['mean']:.1f}  月末={vs['month_end']:.1f}  高={vs['month_high']:.1f}  低={vs['month_low']:.1f}")

# 週次
print("\n■ 3月 週次リターン (主要指数)")
for week_label, wd in weekly_results.items():
    print(f"\n  {week_label}")
    for col in ["SP500", "NASDAQ", "Russell2000", "Nikkei225", "USDJPY"]:
        if col in wd:
            r = wd[col].get('return_pct')
            print(f"    {col:<16} {r:+.2f}%" if r is not None else f"    {col:<16} N/A")

# 転換点
print("\n■ 3月 主要転換点（日次リターン上位/下位）")
for col, tp in turning_points.items():
    print(f"\n  {col}:")
    print(f"    急騰TOP3: ", end="")
    for g in tp["largest_gains"][:3]:
        print(f"  {g['date']} ({g['return_pct']:+.2f}%)", end="")
    print()
    print(f"    急落TOP3: ", end="")
    for l in tp["largest_losses"][:3]:
        print(f"  {l['date']} ({l['return_pct']:+.2f}%)", end="")
    print()

print("\n" + "=" * 70)
print("分析完了")
print("=" * 70)
