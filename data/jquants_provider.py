"""J-Quants API V2 データプロバイダ実装"""

import logging

import pandas as pd

from data.base_provider import MarketDataProvider
from data.cache import DataCache

logger = logging.getLogger(__name__)


class JQuantsProvider(MarketDataProvider):
    """J-Quants API V2 (Standard plan) を使用したデータプロバイダ"""

    def __init__(
        self,
        api_key: str,
        cache: DataCache | None = None,
    ):
        self.api_key = api_key
        self.cache = cache
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import jquantsapi

                self._client = jquantsapi.ClientV2(api_key=self.api_key)
            except Exception as e:
                logger.error("J-Quants APIクライアントの初期化に失敗: %s", e)
                raise
        return self._client

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            self._get_client()
            return True
        except Exception:
            return False

    def get_listed_stocks(self) -> pd.DataFrame:
        cache_key = "listed_stocks"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        client = self._get_client()
        df = client.get_list()

        column_map = {
            "Code": "code",
            "CoName": "name",
            "S17": "sector_17",
            "S33": "sector_33",
            "Mkt": "market",
            "S17Nm": "sector_17_name",
            "S33Nm": "sector_33_name",
            "MktNm": "market_name",
            "ScaleCat": "scale_category",
            "Mrgn": "margin_code",
            "MrgnNm": "margin_code_name",
        }
        available_cols = {k: v for k, v in column_map.items() if k in df.columns}
        df = df.rename(columns=available_cols)

        if self.cache:
            self.cache.put(cache_key, df)
        return df

    def get_trades_spec(self) -> pd.DataFrame:
        """貸借銘柄情報を取得（get_list() から margin_code を抽出）"""
        stocks = self.get_listed_stocks()
        cols = ["code"]
        if "margin_code" in stocks.columns:
            cols.append("margin_code")
        if "margin_code_name" in stocks.columns:
            cols.append("margin_code_name")
        return stocks[cols].drop_duplicates(subset=["code"])

    def get_price_daily(
        self,
        code: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        cache_key = f"price_daily_{code or 'all'}_{start_date}_{end_date}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        client = self._get_client()

        if code:
            # 個別銘柄指定: get_eq_bars_daily(code, from, to)
            kwargs = {"code": code}
            if start_date:
                kwargs["from_yyyymmdd"] = start_date.replace("-", "")
            if end_date:
                kwargs["to_yyyymmdd"] = end_date.replace("-", "")
            df = client.get_eq_bars_daily(**kwargs)
        else:
            # 全銘柄一括: get_eq_bars_daily_range(start_dt, end_dt)
            # V2 API は code 未指定の from/to クエリを受け付けないため
            start = start_date.replace("-", "") if start_date else "20170101"
            end = end_date.replace("-", "") if end_date else ""
            range_kwargs = {"start_dt": start}
            if end:
                range_kwargs["end_dt"] = end
            df = client.get_eq_bars_daily_range(**range_kwargs)

        column_map = {
            "Date": "date",
            "Code": "code",
            "O": "open",
            "H": "high",
            "L": "low",
            "C": "close",
            "Vo": "volume",
            "AdjFactor": "adjustment_factor",
            "AdjO": "adj_open",
            "AdjH": "adj_high",
            "AdjL": "adj_low",
            "AdjC": "adj_close",
            "AdjVo": "adj_volume",
        }
        available_cols = {k: v for k, v in column_map.items() if k in df.columns}
        df = df.rename(columns=available_cols)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        if self.cache:
            self.cache.put(cache_key, df)
        return df

    # カラム名変換（共有）
    _PRICE_COLUMN_MAP = {
        "Date": "date",
        "Code": "code",
        "O": "open",
        "H": "high",
        "L": "low",
        "C": "close",
        "Vo": "volume",
        "AdjFactor": "adjustment_factor",
        "AdjO": "adj_open",
        "AdjH": "adj_high",
        "AdjL": "adj_low",
        "AdjC": "adj_close",
        "AdjVo": "adj_volume",
    }

    def get_price_daily_by_date(self, date_str: str) -> pd.DataFrame:
        """特定日の全銘柄株価を取得（日付単位キャッシュ）

        Args:
            date_str: "YYYY-MM-DD" or "YYYYMMDD"
        """
        date_clean = date_str.replace("-", "")
        cache_key = f"price_alldate_{date_clean}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        client = self._get_client()
        df = client.get_eq_bars_daily(date_yyyymmdd=date_clean)

        available_cols = {k: v for k, v in self._PRICE_COLUMN_MAP.items() if k in df.columns}
        df = df.rename(columns=available_cols)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        if self.cache and not df.empty:
            self.cache.put(cache_key, df)
        return df

    def get_financial_summary(
        self,
        code: str | None = None,
    ) -> pd.DataFrame:
        cache_key = f"financial_{code or 'all'}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        client = self._get_client()
        kwargs = {}
        if code:
            kwargs["code"] = code

        df = client.get_fin_summary(**kwargs)

        if self.cache:
            self.cache.put(cache_key, df)
        return df

    def get_index_prices(
        self,
        index_code: str = "0000",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        cache_key = f"index_{index_code}_{start_date}_{end_date}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        client = self._get_client()
        kwargs = {"code": index_code}
        if start_date:
            kwargs["from_yyyymmdd"] = start_date.replace("-", "")
        if end_date:
            kwargs["to_yyyymmdd"] = end_date.replace("-", "")

        df = client.get_idx_bars_daily(**kwargs)

        column_map = {
            "Date": "date",
            "Code": "index_code",
            "O": "open",
            "H": "high",
            "L": "low",
            "C": "close",
        }
        available_cols = {k: v for k, v in column_map.items() if k in df.columns}
        df = df.rename(columns=available_cols)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        if self.cache:
            self.cache.put(cache_key, df)
        return df

    def get_margin_trading_by_date(self, date_str: str) -> pd.DataFrame:
        """特定日の全銘柄信用取引データを取得（日付単位キャッシュ）"""
        date_clean = date_str.replace("-", "")
        cache_key = f"margin_alldate_{date_clean}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        client = self._get_client()
        df = client.get_mkt_margin_interest(date_yyyymmdd=date_clean)

        if self.cache and df is not None and not df.empty:
            self.cache.put(cache_key, df)
        return df if df is not None else pd.DataFrame()

    def get_margin_trading(
        self,
        code: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        cache_key = f"margin_{code or 'all'}_{start_date}_{end_date}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        client = self._get_client()

        kwargs = {}
        if code:
            kwargs["code"] = code
        if start_date:
            kwargs["from_yyyymmdd"] = start_date.replace("-", "")
        if end_date:
            kwargs["to_yyyymmdd"] = end_date.replace("-", "")

        df = client.get_mkt_margin_interest(**kwargs)

        if self.cache:
            self.cache.put(cache_key, df)
        return df

    def get_short_selling(
        self,
        sector: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        cache_key = f"short_{sector or 'all'}_{start_date}_{end_date}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        client = self._get_client()
        kwargs = {}
        if sector:
            kwargs["sector_33_code"] = sector
        if start_date:
            kwargs["from_yyyymmdd"] = start_date.replace("-", "")
        if end_date:
            kwargs["to_yyyymmdd"] = end_date.replace("-", "")

        df = client.get_mkt_short_ratio(**kwargs)

        if self.cache:
            self.cache.put(cache_key, df)
        return df
