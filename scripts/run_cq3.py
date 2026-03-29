"""cq3: 信用買残変化率×米国ショック のクロスセクション回帰を実行する

Usage:
    python scripts/run_cq3.py
    python scripts/run_cq3.py --start 2020-01-01 --end 2026-03-28 --threshold 2.0
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from datetime import date

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import JQUANTS_API_KEY, MARKET_DATA_DIR
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="cq3: 信用買残×米国ショック検証")
    parser.add_argument("--start", default="2020-01-01", help="開始日")
    parser.add_argument("--end", default=date.today().strftime("%Y-%m-%d"), help="終了日")
    parser.add_argument("--threshold", type=float, default=2.0, help="イベント閾値(σ)")
    parser.add_argument("--n-stocks", type=int, default=200, help="JP銘柄数")
    args = parser.parse_args()

    # --- データ取得 ---
    cache = DataCache(MARKET_DATA_DIR)
    provider = JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)

    print(f"=== cq3: 信用買残×米国ショック検証 ===")
    print(f"期間: {args.start} - {args.end}")
    print(f"イベント閾値: {args.threshold}σ")
    print(f"JP銘柄数: {args.n_stocks}")
    print()

    # 1. S&P500リターン
    print("S&P500リターンを取得中...")
    import yfinance as yf
    sp500 = yf.download("^GSPC", start=args.start, end=args.end, auto_adjust=True, progress=False)
    if sp500.empty:
        print("エラー: S&P500データ取得失敗")
        return
    sp500_close = sp500["Close"].squeeze()
    if hasattr(sp500_close.index, "tz") and sp500_close.index.tz is not None:
        sp500_close.index = sp500_close.index.tz_localize(None)
    us_returns = sp500_close.pct_change().dropna()
    us_returns.name = "sp500"
    print(f"  S&P500: {len(us_returns)}日")

    # 2. JP銘柄リスト
    print("JP銘柄リストを取得中...")
    listed = provider.get_listed_stocks()
    from core.stock_allocation.event_study import get_jp_top_stocks
    jp_codes, jp_names = get_jp_top_stocks(provider, n=args.n_stocks)
    print(f"  JP銘柄: {len(jp_codes)}銘柄")

    # 3. JP株価
    print("JP株価を取得中（月単位チャンク）...")
    from core.stock_allocation import _fetch_stock_prices
    jp_prices_raw = _fetch_stock_prices(
        provider, jp_codes, args.start, args.end,
        progress_callback=lambda m, p: print(f"  {m}") if p % 0.2 < 0.01 else None,
    )
    if jp_prices_raw.empty:
        print("エラー: JP株価取得失敗")
        return

    close_col = "adj_close" if "adj_close" in jp_prices_raw.columns else "close"
    jp_close = jp_prices_raw.pivot(index="date", columns="code", values=close_col)
    jp_returns = jp_close.pct_change().dropna(how="all")
    print(f"  JP株価: {len(jp_returns)}日 × {len(jp_returns.columns)}銘柄")

    # 4. TOPIX
    import pandas as pd
    print("TOPIXリターンを取得中...")
    topix_returns = None
    try:
        topix = provider.get_index_prices("0000", start_date=args.start, end_date=args.end)
        topix["date"] = pd.to_datetime(topix["date"])
        topix = topix.set_index("date")["close"]
        topix_returns = topix.pct_change().dropna()
    except Exception as e:
        print(f"  J-Quants TOPIX取得失敗({e})。yfinanceで代替...")

    if topix_returns is None or len(topix_returns) == 0:
        # TOPIX連動ETF(1306)で代替
        topix_raw = yf.download("1306.T", start=args.start, end=args.end, auto_adjust=True, progress=False)
        if topix_raw.empty:
            print("エラー: TOPIX取得失敗")
            return
        topix_close = topix_raw["Close"].squeeze()
        if hasattr(topix_close.index, "tz") and topix_close.index.tz is not None:
            topix_close.index = topix_close.index.tz_localize(None)
        topix_returns = topix_close.pct_change().dropna()
    print(f"  TOPIX: {len(topix_returns)}日")

    # 5. 信用買残
    print("信用買残を取得中...")
    import pandas as pd
    margin_frames = []
    for code in jp_codes[:args.n_stocks]:
        try:
            df = provider.get_margin_trading(code=code, start_date=args.start, end_date=args.end)
            if df is not None and len(df) > 0:
                if "Date" in df.columns:
                    df = df.rename(columns={"Date": "date"})
                if "Code" in df.columns:
                    df = df.rename(columns={"Code": "code"})
                # 買残列を探す
                for col in ["LongVol", "LongBalance", "margin_buy_balance"]:
                    if col in df.columns:
                        df["margin_buy_balance"] = df[col]
                        break
                if "margin_buy_balance" in df.columns and "date" in df.columns and "code" in df.columns:
                    margin_frames.append(df[["date", "code", "margin_buy_balance"]].copy())
        except Exception:
            continue

    if not margin_frames:
        print("エラー: 信用買残データ取得失敗")
        return

    margin_data = pd.concat(margin_frames, ignore_index=True)
    margin_data["date"] = pd.to_datetime(margin_data["date"])
    print(f"  信用買残: {margin_data['code'].nunique()}銘柄, {len(margin_data)}行")

    # 6. 回帰実行
    print()
    print("=== クロスセクション回帰を実行中 ===")
    from core.stock_allocation.margin_event_study import run_margin_event_study

    result = run_margin_event_study(
        us_index_returns=us_returns,
        jp_stock_returns=jp_returns,
        jp_topix_returns=topix_returns,
        margin_data=margin_data,
        jp_stock_prices=jp_prices_raw,
        listed_stocks=listed,
        event_threshold_sigma=args.threshold,
        progress_callback=lambda m, p: print(f"  [{p:.0%}] {m}"),
    )

    # 7. 結果表示
    print()
    print("=" * 60)
    print("=== cq3 検証結果 ===")
    print("=" * 60)
    print(f"判定:          {result.verdict}")
    print(f"理由:          {result.rejection_reason}")
    print(f"イベント日数:  {result.n_events}")
    print(f"観測点数:      {result.n_observations}")
    print(f"平均銘柄数/日: {result.n_stocks_avg:.0f}")
    print()
    print(f"信用残変化率の係数: {result.b1_coef:+.6f}")
    print(f"  t統計量:         {result.b1_tstat:+.3f}")
    print(f"  p値:             {result.b1_pvalue:.4f}")
    print(f"  VIF:             {result.vif_margin_chg:.2f}")
    print()
    print(f"R² (フルモデル):   {result.r2_full:.4f}")
    print(f"R² (ベースモデル): {result.r2_base:.4f}")
    print(f"増分R²:            {result.incremental_r2:.4f}")
    print()

    if result.all_coefficients:
        print("全係数:")
        for name, vals in result.all_coefficients.items():
            if name == "const" or name.startswith("evt"):
                continue
            print(f"  {name:20s}: coef={vals['coef']:+.6f}  t={vals['tstat']:+.3f}  p={vals['pvalue']:.4f}")

    # 8. JSON保存
    output_path = Path("logs/iterations/cq3_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "verdict": result.verdict,
        "rejection_reason": result.rejection_reason,
        "b1_coef": result.b1_coef,
        "b1_pvalue": result.b1_pvalue,
        "b1_tstat": result.b1_tstat,
        "vif_margin_chg": result.vif_margin_chg,
        "n_events": result.n_events,
        "n_observations": result.n_observations,
        "n_stocks_avg": result.n_stocks_avg,
        "incremental_r2": result.incremental_r2,
        "r2_full": result.r2_full,
        "r2_base": result.r2_base,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果を保存: {output_path}")


if __name__ == "__main__":
    main()
