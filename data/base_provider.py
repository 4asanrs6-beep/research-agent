"""データプロバイダ抽象基底クラス"""

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataProvider(ABC):
    """市場データプロバイダの抽象基底クラス"""

    @abstractmethod
    def get_listed_stocks(self) -> pd.DataFrame:
        """上場銘柄一覧を取得
        Returns: columns=[code, name, sector_17, sector_33, market]
        """

    @abstractmethod
    def get_price_daily(
        self,
        code: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """株価日足（OHLCV）を取得
        Returns: columns=[date, code, open, high, low, close, volume, adjustment_factor]
        """

    @abstractmethod
    def get_financial_summary(
        self,
        code: str | None = None,
    ) -> pd.DataFrame:
        """財務サマリーを取得"""

    @abstractmethod
    def get_index_prices(
        self,
        index_code: str = "0000",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """指数四本値を取得（デフォルト: TOPIX）
        Returns: columns=[date, index_code, open, high, low, close]
        """

    @abstractmethod
    def get_margin_trading(
        self,
        code: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """信用取引残高を取得"""

    @abstractmethod
    def get_short_selling(
        self,
        sector: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """業種別空売り比率を取得"""

    def is_available(self) -> bool:
        """APIが利用可能かチェック"""
        return False
