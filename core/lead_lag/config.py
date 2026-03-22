"""日米業種リードラグ戦略のパラメータ定義"""

from dataclasses import dataclass, field


# =============================================================================
# ユニバース定義
# =============================================================================

# --- 米国: S&P 11セクター (Select Sector SPDR) ---
US_SECTOR_11 = {
    "XLB": "Materials", "XLC": "Communication Services", "XLE": "Energy",
    "XLF": "Financials", "XLI": "Industrials", "XLK": "Information Technology",
    "XLP": "Consumer Staples", "XLRE": "Real Estate", "XLU": "Utilities",
    "XLV": "Health Care", "XLY": "Consumer Discretionary",
}

# --- 米国: S&P 11セクター + 22インダストリー = 33種 ---
US_INDUSTRY_22 = {
    "XAR": "Aerospace & Defense", "XBI": "Biotech", "XHB": "Homebuilders",
    "XHE": "Health Care Equipment", "XHS": "Health Care Services",
    "XME": "Metals & Mining", "XOP": "Oil & Gas E&P",
    "XPH": "Pharmaceuticals", "XRT": "Retail", "XSD": "Semiconductor",
    "XSW": "Software & Services", "XTL": "Telecom", "XTN": "Transportation",
    "KBE": "Banks", "KCE": "Capital Markets", "KIE": "Insurance",
    "KRE": "Regional Banks", "XNTK": "NYSE Tech", "GNR": "Natural Resources",
    "HACK": "Cybersecurity", "IGV": "Software", "SOXX": "Semiconductor (SOXX)",
}

US_33 = {**US_SECTOR_11, **US_INDUSTRY_22}

# --- 日本: TOPIX-17 セクターETF ---
JP_TOPIX17 = {
    "1617": "食品", "1618": "エネルギー資源", "1619": "建設・資材",
    "1620": "素材・化学", "1621": "医薬品", "1622": "自動車・輸送機",
    "1623": "鉄鋼・非鉄", "1624": "機械", "1625": "電機・精密",
    "1626": "情報通信・サービスその他", "1627": "電力・ガス",
    "1628": "運輸・物流", "1629": "商社・卸売", "1630": "小売",
    "1631": "銀行", "1632": "金融(除く銀行)", "1633": "不動産",
}

# --- 日本: 東証33業種ETFは大部分が上場廃止済み ---
# 取得可能なのはTOPIX-17の17本のみ

# --- ユニバース選択肢 ---
UNIVERSE_OPTIONS = {
    "US_11": {"label": "米国 11セクター", "tickers": US_SECTOR_11},
    "US_33": {"label": "米国 33種 (11セクター+22インダストリー)", "tickers": US_33},
    "JP_17": {"label": "日本 TOPIX-17", "tickers": JP_TOPIX17},
    # JP_33: 東証33業種ETFは大部分が上場廃止済みのため選択不可
}

# --- デフォルト (後方互換) ---
US_TICKERS = list(US_SECTOR_11.keys())
US_TICKER_NAMES = US_SECTOR_11
JP_TICKERS = list(JP_TOPIX17.keys())
JP_TICKER_NAMES = JP_TOPIX17

# --- シクリカル / ディフェンシブ ラベル (論文 Section 4.1) ---
US_CYCLICAL = ["XLB", "XLE", "XLF", "XLRE"]
US_DEFENSIVE = ["XLK", "XLP", "XLU", "XLV"]
JP_CYCLICAL = ["1618", "1625", "1629", "1631"]
JP_DEFENSIVE = ["1617", "1621", "1627", "1630"]


@dataclass
class LeadLagConfig:
    """部分空間正則化PCA リードラグ戦略のパラメータ"""

    # 期間
    start_date: str = "2010-01-01"
    end_date: str = ""  # 空 = 今日まで
    prior_end_date: str = "2014-12-31"  # C_full 推定期間の終了日

    # 戦略パラメータ (論文デフォルト)
    rolling_window: int = 60  # L: ローリングウィンドウ長
    n_components: int = 3  # K: 使用する固有ベクトル数
    lambda_reg: float = 0.9  # λ: 正則化強度
    quantile_q: float = 0.3  # q: ロング/ショート分位点 (上下30%)

    # ユニバース選択
    us_universe: str = "US_11"  # "US_11" or "US_33"
    jp_universe: str = "JP_17"  # "JP_17" or "JP_33"

    # ティッカー (ユニバースから自動設定)
    us_tickers: list[str] = field(default_factory=lambda: list(US_TICKERS))
    jp_tickers: list[str] = field(default_factory=lambda: list(JP_TICKERS))

    # セクターラベル
    us_cyclical: list[str] = field(default_factory=lambda: list(US_CYCLICAL))
    us_defensive: list[str] = field(default_factory=lambda: list(US_DEFENSIVE))
    jp_cyclical: list[str] = field(default_factory=lambda: list(JP_CYCLICAL))
    jp_defensive: list[str] = field(default_factory=lambda: list(JP_DEFENSIVE))

    # ギャップフィルター閾値 (0=フィルターなし、0.1=10%消化でスキップ)
    gap_threshold: float = 0.0

    # 実行対象戦略
    run_mom: bool = True
    run_pca_plain: bool = True
    run_pca_sub: bool = True
    run_double: bool = True

    def __post_init__(self):
        """ユニバース選択に応じてティッカーリストを自動設定する。"""
        if self.us_universe in UNIVERSE_OPTIONS:
            self.us_tickers = list(UNIVERSE_OPTIONS[self.us_universe]["tickers"].keys())
        if self.jp_universe in UNIVERSE_OPTIONS:
            self.jp_tickers = list(UNIVERSE_OPTIONS[self.jp_universe]["tickers"].keys())


def get_ticker_name(ticker: str) -> str:
    """ティッカーコードから表示名 (コード付き) を取得する。"""
    for universe in [US_SECTOR_11, US_INDUSTRY_22, JP_TOPIX17]:
        if ticker in universe:
            return f"{universe[ticker]} ({ticker})"
    return ticker
