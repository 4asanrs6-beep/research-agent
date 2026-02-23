"""ユニバースフィルタ済みのデータプロバイダラッパー

FilteredProvider は MarketDataProvider をラップし、
get_listed_stocks() と get_trades_spec() がフィルタ済みの結果のみを返すようにする。
AI生成コードは何をしても、フィルタ済みの銘柄リストしか見えない。
"""

import pandas as pd

from data.base_provider import MarketDataProvider
from core.universe_filter import UniverseFilterConfig, apply_universe_filter


class FilteredProvider:
    """ユニバースフィルタ済みのデータプロバイダラッパー

    get_listed_stocks() → フィルタ済みの銘柄リストを返す
    get_trades_spec()   → フィルタ済みの銘柄のみを返す
    その他              → 内部プロバイダにそのまま委譲
    """

    def __init__(self, inner: MarketDataProvider, universe_config: UniverseFilterConfig):
        self.inner = inner
        self.universe_config = universe_config
        self._filtered_codes: set | None = None

    def get_listed_stocks(self) -> pd.DataFrame:
        df = self.inner.get_listed_stocks()
        df = apply_universe_filter(df, self.universe_config)
        self._filtered_codes = set(df["code"].tolist())
        return df

    def get_trades_spec(self) -> pd.DataFrame:
        df = self.inner.get_trades_spec()
        if self._filtered_codes is None:
            self.get_listed_stocks()  # codes を確定
        return df[df["code"].isin(self._filtered_codes)]

    # --- 以下は inner にそのまま委譲 ---

    def get_price_daily(
        self,
        code: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return self.inner.get_price_daily(code=code, start_date=start_date, end_date=end_date)

    def get_financial_summary(self, code: str | None = None) -> pd.DataFrame:
        return self.inner.get_financial_summary(code=code)

    def get_index_prices(
        self,
        index_code: str = "0000",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return self.inner.get_index_prices(index_code=index_code, start_date=start_date, end_date=end_date)

    def get_margin_trading(
        self,
        code: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return self.inner.get_margin_trading(code=code, start_date=start_date, end_date=end_date)

    def get_short_selling(
        self,
        sector: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return self.inner.get_short_selling(sector=sector, start_date=start_date, end_date=end_date)

    def is_available(self) -> bool:
        return self.inner.is_available()
