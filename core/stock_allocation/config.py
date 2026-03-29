"""個別株配賦戦略のパラメータ定義"""

import math
from dataclasses import dataclass, field

from core.lead_lag.config import JP_TOPIX17


# JP ETF コード → sector_17_name (J-Quants API の返り値に合わせる)
# 括弧の全角/半角差異を吸収するため、正規化関数で比較する
ETF_TO_SECTOR_17 = dict(JP_TOPIX17)  # {"1617": "食品", ...}

# 逆引き: sector_17_name → ETF コード
SECTOR_17_TO_ETF: dict[str, str] = {v: k for k, v in ETF_TO_SECTOR_17.items()}


def normalize_sector_name(name: str) -> str:
    """全角/半角括弧やスペースの差異を吸収する正規化"""
    return (
        name.replace("（", "(")
        .replace("）", ")")
        .replace("　", " ")
        .strip()
    )


def find_etf_for_sector(sector_name: str) -> str | None:
    """sector_17_name から ETF コードを検索する (正規化つき)"""
    if sector_name in SECTOR_17_TO_ETF:
        return SECTOR_17_TO_ETF[sector_name]
    norm = normalize_sector_name(sector_name)
    for name, code in SECTOR_17_TO_ETF.items():
        if normalize_sector_name(name) == norm:
            return code
    return None


def compute_n_sectors_from_q(quantile_q: float, n_jp_sectors: int = 17) -> int:
    """ETF版の分位点 q からロング/ショート対象セクター数を計算する。

    ETF版の build_portfolio と同じロジック: ceil(n_valid × q)
    """
    return max(1, math.ceil(n_jp_sectors * quantile_q))


@dataclass
class StockAllocationConfig:
    """個別株配賦戦略のパラメータ

    ETF版のLeadLagConfigと連動する設計:
      - quantile_q → n_sectors_long/short を自動計算
      - backtest期間 → ETF版のシグナル期間から自動取得
    """

    # --- ETF版との連動 ---
    # ETF版のquantile_q を使ってセクター数を自動計算する
    # 0より大きい場合、n_sectors_long/shortを上書きする
    quantile_q: float = 0.0  # 0 = 手動指定, >0 = ETF版と連動

    # --- セクター数 (quantile_q=0 のとき手動指定) ---
    n_sectors_long: int = 5
    n_sectors_short: int = 5

    # --- ユニバースフィルタ ---
    # 複数市場を同時に選択可能
    market_segments: list[str] = field(
        default_factory=lambda: ["プライム", "スタンダード", "グロース"]
    )
    scale_categories: list[str] = field(default_factory=list)
    margin_tradable_only: bool = True

    # --- コスト ---
    commission_rate: float = 0.001   # 片道 0.1%
    slippage_rate: float = 0.001     # 片道 0.1%

    # --- バックテスト期間 ---
    # None = ETF版BacktestResultのシグナル期間を自動使用
    backtest_start: str | None = None
    backtest_end: str | None = None

    def resolve_n_sectors(self, n_jp_sectors: int = 17) -> tuple[int, int]:
        """quantile_q が設定されている場合、セクター数を自動計算する。"""
        if self.quantile_q > 0:
            n = compute_n_sectors_from_q(self.quantile_q, n_jp_sectors)
            return n, n
        return self.n_sectors_long, self.n_sectors_short
