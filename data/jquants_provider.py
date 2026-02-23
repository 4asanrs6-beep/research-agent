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
        df = client.get_listed_info()

        column_map = {
            "Code": "code",
            "CompanyName": "name",
            "Sector17Code": "sector_17",
            "Sector33Code": "sector_33",
            "MarketCode": "market",
        }
        available_cols = {k: v for k, v in column_map.items() if k in df.columns}
        df = df.rename(columns=available_cols)

        if self.cache:
            self.cache.put(cache_key, df)
        return df

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
        kwargs = {}
        if code:
            kwargs["code"] = code
        if start_date:
            kwargs["from_yyyymmdd"] = start_date.replace("-", "")
        if end_date:
            kwargs["to_yyyymmdd"] = end_date.replace("-", "")

        df = client.get_prices_daily_quotes(**kwargs)

        column_map = {
            "Date": "date",
            "Code": "code",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "AdjustmentFactor": "adjustment_factor",
            "AdjustmentOpen": "adj_open",
            "AdjustmentHigh": "adj_high",
            "AdjustmentLow": "adj_low",
            "AdjustmentClose": "adj_close",
            "AdjustmentVolume": "adj_volume",
        }
        available_cols = {k: v for k, v in column_map.items() if k in df.columns}
        df = df.rename(columns=available_cols)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        if self.cache:
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
        if code:
            df = client.get_statements(code=code)
        else:
            df = client.get_statements()

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

        df = client.get_indices(**kwargs)

        column_map = {
            "Date": "date",
            "Code": "index_code",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
        }
        available_cols = {k: v for k, v in column_map.items() if k in df.columns}
        df = df.rename(columns=available_cols)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        if self.cache:
            self.cache.put(cache_key, df)
        return df

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

        df = client.get_markets_weekly_margin_interest(**kwargs)

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
            kwargs["sector33code"] = sector
        if start_date:
            kwargs["from_yyyymmdd"] = start_date.replace("-", "")
        if end_date:
            kwargs["to_yyyymmdd"] = end_date.replace("-", "")

        df = client.get_markets_short_selling(**kwargs)

        if self.cache:
            self.cache.put(cache_key, df)
        return df
