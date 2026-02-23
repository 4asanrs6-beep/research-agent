"""ユニバースフィルタリング設定モジュール

フィルタ条件をテキスト化してAIプロンプトに注入するための設定と変換機能を提供する。
機械的なフィルタリング（apply_universe_filter）も提供する。
"""

from dataclasses import dataclass, field

import pandas as pd

# --- 定数 ---

MARKET_SEGMENTS = ["プライム", "スタンダード", "グロース"]

TOPIX_SCALE_CATEGORIES = [
    "TOPIX Core30",
    "TOPIX Large70",
    "TOPIX Mid400",
    "TOPIX Small 1",
    "TOPIX Small 2",
]

SECTOR_17_LIST = [
    "食品",
    "エネルギー資源",
    "建設・資材",
    "素材・化学",
    "医薬品",
    "自動車・輸送機",
    "鉄鋼・非鉄",
    "機械",
    "電機・精密",
    "情報通信・サービスその他",
    "電気・ガス",
    "運輸・物流",
    "商社・卸売",
    "小売",
    "銀行",
    "金融（除く銀行）",
    "不動産",
]


@dataclass
class UniverseFilterConfig:
    """ユニバースフィルタリング条件"""

    market_segments: list[str] = field(default_factory=list)
    scale_categories: list[str] = field(default_factory=list)
    sector_filter_type: str = "none"  # "none" | "sector_17" | "sector_33"
    selected_sectors: list[str] = field(default_factory=list)
    margin_tradable_only: bool = False
    exclude_etf_reit: bool = True  # ETF・REIT・その他を除外（一般株式のみ）
    market_cap_min: float | None = None  # 億円
    market_cap_max: float | None = None  # 億円
    per_min: float | None = None
    per_max: float | None = None
    pbr_min: float | None = None
    pbr_max: float | None = None

    def is_empty(self) -> bool:
        """フィルタ条件が何も設定されていないか"""
        return (
            not self.market_segments
            and not self.scale_categories
            and self.sector_filter_type == "none"
            and not self.margin_tradable_only
            and not self.exclude_etf_reit
            and self.market_cap_min is None
            and self.market_cap_max is None
            and self.per_min is None
            and self.per_max is None
            and self.pbr_min is None
            and self.pbr_max is None
        )


# 一般株式の市場区分（ETF・REIT等はこれ以外の市場区分名を持つ）
_STOCK_MARKET_SEGMENTS = {"プライム", "スタンダード", "グロース"}


def apply_universe_filter(stocks_df: pd.DataFrame, config: UniverseFilterConfig) -> pd.DataFrame:
    """DataFrameに対してフィルタ条件を機械的に適用する

    Args:
        stocks_df: get_listed_stocks() の戻り値相当のDataFrame
        config: フィルタ条件

    Returns:
        フィルタ済みのDataFrame
    """
    df = stocks_df.copy()
    if config.exclude_etf_reit and "market_name" in df.columns:
        df = df[df["market_name"].isin(_STOCK_MARKET_SEGMENTS)]
    if config.market_segments:
        df = df[df["market_name"].isin(config.market_segments)]
    if config.scale_categories:
        df = df[df["scale_category"].isin(config.scale_categories)]
    if config.sector_filter_type == "sector_17" and config.selected_sectors:
        df = df[df["sector_17_name"].isin(config.selected_sectors)]
    if config.margin_tradable_only:
        df = df[df["margin_code"] == "2"]
    return df


def build_universe_description(config: UniverseFilterConfig) -> str:
    """フィルタ条件を自然言語テキストに変換する

    Args:
        config: フィルタ条件

    Returns:
        AIプロンプトに注入するテキスト。条件なしの場合は空文字列。
    """
    if config.is_empty():
        return ""

    lines = []

    # ETF・REIT除外
    if config.exclude_etf_reit:
        lines.append("- ETF・REIT・その他を除外（一般株式のみ）")

    # 市場区分
    if config.market_segments:
        lines.append(f"- 市場区分: {', '.join(config.market_segments)} のみ")

    # TOPIX規模区分
    if config.scale_categories:
        lines.append(f"- TOPIX規模区分: {', '.join(config.scale_categories)} のみ")

    # 業種フィルター
    if config.sector_filter_type == "sector_17" and config.selected_sectors:
        lines.append(f"- 業種（17業種区分）: {', '.join(config.selected_sectors)} のみ")
    elif config.sector_filter_type == "sector_33" and config.selected_sectors:
        lines.append(f"- 業種（33業種区分）: {', '.join(config.selected_sectors)} のみ")

    # 貸借銘柄
    if config.margin_tradable_only:
        lines.append("- 貸借銘柄のみ（空売り可能な銘柄に限定）")

    # 時価総額
    if config.market_cap_min is not None or config.market_cap_max is not None:
        cap_parts = []
        if config.market_cap_min is not None:
            cap_parts.append(f"{config.market_cap_min}億円以上")
        if config.market_cap_max is not None:
            cap_parts.append(f"{config.market_cap_max}億円以下")
        lines.append(f"- 時価総額: {' かつ '.join(cap_parts)}")

    # PER
    if config.per_min is not None or config.per_max is not None:
        per_parts = []
        if config.per_min is not None:
            per_parts.append(f"{config.per_min}倍以上")
        if config.per_max is not None:
            per_parts.append(f"{config.per_max}倍以下")
        lines.append(f"- PER: {' かつ '.join(per_parts)}")

    # PBR
    if config.pbr_min is not None or config.pbr_max is not None:
        pbr_parts = []
        if config.pbr_min is not None:
            pbr_parts.append(f"{config.pbr_min}倍以上")
        if config.pbr_max is not None:
            pbr_parts.append(f"{config.pbr_max}倍以下")
        lines.append(f"- PBR: {' かつ '.join(pbr_parts)}")

    if not lines:
        return ""

    return "\n".join(lines)
