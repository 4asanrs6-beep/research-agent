"""日米セクターETFデータの取得・整列"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_benchmark_returns(
    start: str,
    end: str,
    jquants_provider=None,
    cache=None,
) -> pd.Series:
    """TOPIX連動ETF (1306) の日次リターンを取得する。J-Quants API優先、フォールバックでyfinance。

    Returns:
        Series indexed by date, values = daily close-to-close return
    """
    cache_key = f"benchmark_topix_{start}_{end}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info("TOPIX ベンチマーク キャッシュヒット")
            return cached.set_index("date")["ret"]

    close_series = None

    # --- J-Quants API で TOPIX ETF (1306) を取得 ---
    if jquants_provider is not None:
        try:
            logger.info("J-Quants: TOPIX ETF (1306) を取得中")
            start_year = int(start[:4])
            end_year = int(end[:4]) if end else 2026
            all_rows = []
            for year in range(start_year, end_year + 1):
                try:
                    df_year = jquants_provider.get_prices_daily_quotes(
                        code="13060", from_yyyymmdd=f"{year}0101", to_yyyymmdd=f"{year}1231"
                    )
                    if df_year is not None and len(df_year) > 0:
                        all_rows.append(df_year)
                except Exception:
                    pass
            if all_rows:
                df_all = pd.concat(all_rows, ignore_index=True)
                df_all["Date"] = pd.to_datetime(df_all["Date"])
                df_all = df_all.sort_values("Date").drop_duplicates(subset=["Date"])
                if "AdjustmentClose" in df_all.columns:
                    close_series = df_all.set_index("Date")["AdjustmentClose"]
                elif "Close" in df_all.columns:
                    close_series = df_all.set_index("Date")["Close"]
                if close_series is not None:
                    logger.info("J-Quants: TOPIX ETF 取得成功 (%d日)", len(close_series))
        except Exception as e:
            logger.warning("J-Quants TOPIX ETF 取得失敗: %s", e)

    # --- フォールバック: yfinance ---
    if close_series is None:
        try:
            import yfinance as yf
            logger.info("yfinance: TOPIX (1306.T) を取得中")
            raw = yf.download("1306.T", start=start, end=end, auto_adjust=True, progress=False)
            if raw.empty:
                logger.info("1306.T 取得失敗、^TPX にフォールバック")
                raw = yf.download("^TPX", start=start, end=end, auto_adjust=True, progress=False)
            if not raw.empty:
                close = raw["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close_series = close
        except Exception as e:
            logger.warning("yfinance TOPIX 取得失敗: %s", e)

    if close_series is None or len(close_series) == 0:
        logger.warning("TOPIX ベンチマークデータを取得できませんでした")
        return pd.Series(dtype=float)

    ret = close_series.pct_change().dropna()
    ret.index = pd.to_datetime(ret.index)
    ret.name = "TOPIX"

    if cache is not None:
        df = pd.DataFrame({"date": ret.index, "ret": ret.values})
        cache.put(cache_key, df)

    return ret


@dataclass
class AlignedData:
    """日米整列済みデータセット"""

    us_cc_returns: pd.DataFrame  # US close-to-close リターン (共通日付 × USティッカー)
    jp_cc_returns: pd.DataFrame  # JP close-to-close リターン (共通日付 × JPティッカー)
    jp_oc_returns: pd.DataFrame  # JP open-to-close リターン (共通日付 × JPティッカー)
    jp_close_prices: pd.DataFrame  # JP adj_close 価格 (共通日付 × JPティッカー)
    jp_open_prices: pd.DataFrame   # JP adj_open 価格 (共通日付 × JPティッカー)
    common_dates: pd.DatetimeIndex
    us_tickers: list[str]
    jp_tickers: list[str]
    jp_to_us_date_map: dict | None = None  # JP日付 → 実際のUS日付のマッピング
    us_trading_dates: list | None = None  # 全US営業日リスト


def fetch_us_etf_data(
    tickers: list[str],
    start: str,
    end: str,
    cache=None,
) -> pd.DataFrame:
    """yfinance で US セクター ETF の日次 OHLCV を取得する。

    Returns:
        DataFrame with columns: [date, ticker, open, close, adj_close]
    """
    cache_key = f"us_etf_{'_'.join(sorted(tickers))}_{start}_{end}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info("US ETF データキャッシュヒット (%d行)", len(cached))
            return cached

    import yfinance as yf

    logger.info("yfinance: US ETF %d銘柄を取得中 (%s ~ %s)", len(tickers), start, end)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)

    if raw.empty:
        raise ValueError("yfinance から US ETF データを取得できませんでした")

    rows = []
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                sub = raw[["Open", "Close", "Adj Close"]].copy()
            else:
                sub = raw[[("Open", ticker), ("Close", ticker), ("Adj Close", ticker)]].copy()
                sub.columns = ["Open", "Close", "Adj Close"]
            sub = sub.dropna()
            sub["ticker"] = ticker
            sub["date"] = sub.index
            sub = sub.rename(columns={"Open": "open", "Close": "close", "Adj Close": "adj_close"})
            rows.append(sub[["date", "ticker", "open", "close", "adj_close"]])
        except Exception as e:
            logger.warning("US ETF %s の取得に失敗: %s", ticker, e)

    if not rows:
        raise ValueError("US ETF データが1銘柄も取得できませんでした")

    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("US ETF データ取得完了: %d行, 銘柄数=%d", len(df), df["ticker"].nunique())

    if cache is not None:
        cache.put(cache_key, df)
    return df


def fetch_jp_etf_data(
    tickers: list[str],
    start: str,
    end: str,
    jquants_provider=None,
    cache=None,
) -> pd.DataFrame:
    """JP セクター ETF の日次 OHLCV を取得する。

    J-Quants API を優先し、失敗時は yfinance にフォールバック。

    Args:
        tickers: 4桁コード (例: ["1617", "1618", ...])

    Returns:
        DataFrame with columns: [date, ticker, open, close, adj_open, adj_close]
    """
    cache_key = f"jp_etf_{'_'.join(sorted(tickers))}_{start}_{end}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info("JP ETF データキャッシュヒット (%d行)", len(cached))
            return cached

    rows = []
    failed_tickers = []

    # --- J-Quants API で取得を試みる (年単位で分割リクエスト) ---
    if jquants_provider is not None:
        start_year = int(start[:4])
        end_year = int(end[:4])
        for ticker in tickers:
            code = f"{ticker}0"  # 5桁コード (例: 16170)
            ticker_dfs = []
            ok = True
            for year in range(start_year, end_year + 1):
                chunk_start = f"{year}-01-01" if year > start_year else start
                chunk_end = f"{year}-12-31" if year < end_year else end
                try:
                    df = jquants_provider.get_price_daily(
                        code=code,
                        start_date=chunk_start,
                        end_date=chunk_end,
                    )
                    if df is not None and not df.empty:
                        ticker_dfs.append(df)
                except Exception as e:
                    logger.debug("J-Quants: %s (%s) %s 取得失敗: %s", ticker, code, year, e)
                    ok = False
                    break
            if ok and ticker_dfs:
                sub = pd.concat(ticker_dfs, ignore_index=True)
                sub["ticker"] = ticker
                if "adj_open" not in sub.columns:
                    sub["adj_open"] = sub["open"]
                if "adj_close" not in sub.columns:
                    sub["adj_close"] = sub["close"]
                rows.append(sub[["date", "ticker", "open", "close", "adj_open", "adj_close"]])
                logger.info("J-Quants: %s (%s) %d行取得", ticker, code, len(sub))
                continue
            failed_tickers.append(ticker)
    else:
        failed_tickers = list(tickers)

    # --- yfinance フォールバック ---
    if failed_tickers:
        logger.info("yfinance フォールバック: %d銘柄", len(failed_tickers))
        try:
            import yfinance as yf

            yf_tickers = [f"{t}.T" for t in failed_tickers]
            raw = yf.download(yf_tickers, start=start, end=end, auto_adjust=False, progress=False)

            if not raw.empty:
                for ticker, yf_ticker in zip(failed_tickers, yf_tickers):
                    try:
                        if len(yf_tickers) == 1:
                            sub = raw[["Open", "Close", "Adj Close"]].copy()
                        else:
                            sub = raw[
                                [("Open", yf_ticker), ("Close", yf_ticker), ("Adj Close", yf_ticker)]
                            ].copy()
                            sub.columns = ["Open", "Close", "Adj Close"]
                        sub = sub.dropna()
                        sub["ticker"] = ticker
                        sub["date"] = sub.index
                        # ETF の場合 adj_open は近似として open * (adj_close / close) を使う
                        sub["adj_factor"] = np.where(
                            sub["Close"] != 0, sub["Adj Close"] / sub["Close"], 1.0
                        )
                        sub = sub.rename(columns={"Open": "open", "Close": "close", "Adj Close": "adj_close"})
                        sub["adj_open"] = sub["open"] * sub["adj_factor"]
                        rows.append(sub[["date", "ticker", "open", "close", "adj_open", "adj_close"]])
                        logger.info("yfinance: %s %d行取得", ticker, len(sub))
                    except Exception as e:
                        logger.warning("yfinance: %s 取得失敗: %s", ticker, e)
        except Exception as e:
            logger.error("yfinance フォールバック全体エラー: %s", e)

    if not rows:
        raise ValueError("JP ETF データが1銘柄も取得できませんでした")

    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    logger.info("JP ETF データ取得完了: %d行, 銘柄数=%d", len(df), df["ticker"].nunique())

    if cache is not None:
        cache.put(cache_key, df)
    return df


def build_aligned_dataset(
    us_data: pd.DataFrame,
    jp_data: pd.DataFrame,
    us_tickers: list[str],
    jp_tickers: list[str],
    accumulate_us_returns: bool = False,
) -> AlignedData:
    """日米データを共通取引日に整列し、リターンを計算する。

    整列ロジック:
    - JP取引日 t に対し、直前の US 取引日 t_us を紐付け (merge_asof)
    - US close-to-close: r^cc_{u,t_us} = close_{t_us} / close_{t_us-1} - 1
    - JP close-to-close: r^cc_{j,t} = adj_close_t / adj_close_{t-1} - 1
    - JP open-to-close: r^oc_{j,t} = adj_close_t / adj_open_t - 1
    """
    # --- US: ピボットして close-to-close リターン ---
    us_pivot = us_data.pivot(index="date", columns="ticker", values="adj_close")
    us_pivot = us_pivot[us_tickers].sort_index()
    us_cc = us_pivot.pct_change().iloc[1:]  # close-to-close

    # --- JP: ピボットして各種リターン ---
    jp_close_pivot = jp_data.pivot(index="date", columns="ticker", values="adj_close")
    jp_open_pivot = jp_data.pivot(index="date", columns="ticker", values="adj_open")
    jp_close_pivot = jp_close_pivot[jp_tickers].sort_index()
    jp_open_pivot = jp_open_pivot[jp_tickers].sort_index()

    jp_cc = jp_close_pivot.pct_change().iloc[1:]  # close-to-close
    jp_oc = (jp_close_pivot / jp_open_pivot - 1).dropna()  # open-to-close

    # --- 共通取引日の整列 ---
    # JP取引日 t に対し、直前の US 取引日を紐付け
    jp_dates = pd.DataFrame({"jp_date": jp_cc.index})
    us_dates = pd.DataFrame({"us_date": us_cc.index, "us_date_val": us_cc.index})

    # JP市場はUS市場より先に開くため、JP日tの寄付き時点で
    # 利用可能なのはt-1以前のUS終値。同日のUSはまだ始まっていない。
    merged = pd.merge_asof(
        jp_dates.sort_values("jp_date"),
        us_dates.sort_values("us_date"),
        left_on="jp_date",
        right_on="us_date",
        direction="backward",
        allow_exact_matches=False,
    )
    # US の取引日が5営業日以上前のものは除外 (長期休場など)
    merged = merged.dropna(subset=["us_date"])
    merged = merged[
        (merged["jp_date"] - merged["us_date"]).dt.days <= 7
    ]
    # 同じ US 日が複数の JP 日に紐付く場合、最初の1回だけ残す
    # → US 休場翌日の JP 営業日は古い US データの使い回しを防ぐ
    merged = merged.drop_duplicates(subset=["us_date"], keep="first")

    # 整列済みリターンを構成
    if accumulate_us_returns:
        # JP休場中のUSリターンを累積: 各JP日に紐付くUS日までの
        # 「前回使用したUS日の翌日〜今回のUS日」の累積リターンを計算
        us_aligned_rows = []
        prev_us_date = None
        for _, row in merged.iterrows():
            us_date = row["us_date"]
            if prev_us_date is not None:
                # 前回US日の翌日から今回US日までのリターンを累積
                mask = (us_cc.index > prev_us_date) & (us_cc.index <= us_date)
                us_window = us_cc.loc[mask]
                if len(us_window) > 1:
                    # 累積リターン: (1+r1)*(1+r2)*...-1
                    cumret = (1 + us_window).prod() - 1
                    us_aligned_rows.append(cumret)
                else:
                    us_aligned_rows.append(us_cc.loc[us_date])
            else:
                us_aligned_rows.append(us_cc.loc[us_date])
            prev_us_date = us_date
        us_aligned = pd.DataFrame(us_aligned_rows)
        us_aligned.index = merged["jp_date"].values
    else:
        us_aligned = us_cc.loc[merged["us_date"].values].copy()
        us_aligned.index = merged["jp_date"].values

    jp_cc_aligned = jp_cc.loc[jp_cc.index.isin(merged["jp_date"])]
    jp_oc_aligned = jp_oc.loc[jp_oc.index.isin(merged["jp_date"])]

    # 3つのインデックスの共通部分
    common = us_aligned.index.intersection(jp_cc_aligned.index).intersection(jp_oc_aligned.index)
    common = pd.DatetimeIndex(sorted(common))

    # NaN行を除去
    us_final = us_aligned.loc[common].dropna(how="any")
    jp_cc_final = jp_cc_aligned.loc[common].dropna(how="any")
    jp_oc_final = jp_oc_aligned.loc[common].dropna(how="any")

    common = us_final.index.intersection(jp_cc_final.index).intersection(jp_oc_final.index)
    common = pd.DatetimeIndex(sorted(common))

    # 利用可能なティッカーのみ残す
    actual_us = [t for t in us_tickers if t in us_final.columns and us_final[t].notna().sum() > 0]
    actual_jp = [t for t in jp_tickers if t in jp_cc_final.columns and jp_cc_final[t].notna().sum() > 0]

    logger.info(
        "整列完了: 共通日数=%d, US=%d銘柄, JP=%d銘柄, 期間=%s ~ %s",
        len(common), len(actual_us), len(actual_jp),
        common[0].strftime("%Y-%m-%d") if len(common) > 0 else "N/A",
        common[-1].strftime("%Y-%m-%d") if len(common) > 0 else "N/A",
    )

    # JP日付 → 実際のUS日付のマッピングを構築
    date_map = {}
    for _, row in merged.iterrows():
        jp_d = pd.Timestamp(row["jp_date"])
        us_d = pd.Timestamp(row["us_date"])
        if jp_d in common:
            date_map[jp_d] = us_d

    return AlignedData(
        us_cc_returns=us_final.loc[common, actual_us],
        jp_cc_returns=jp_cc_final.loc[common, actual_jp],
        jp_oc_returns=jp_oc_final.loc[common, actual_jp],
        jp_close_prices=jp_close_pivot.loc[common, actual_jp],
        jp_open_prices=jp_open_pivot.loc[common, actual_jp],
        common_dates=common,
        us_tickers=actual_us,
        jp_tickers=actual_jp,
        jp_to_us_date_map=date_map,
        us_trading_dates=sorted(us_cc.index.tolist()),
    )
