"""セクターETFシグナル → 個別株マッピング (Phase 1)"""

import logging

import numpy as np
import pandas as pd

from .config import ETF_TO_SECTOR_17, find_etf_for_sector, normalize_sector_name

logger = logging.getLogger(__name__)


def build_sector_stock_map(
    listed_stocks: pd.DataFrame,
) -> dict[str, list[str]]:
    """sector_17_name → 銘柄コードのリストを構築する。

    Args:
        listed_stocks: JQuantsProvider.get_listed_stocks() の結果

    Returns:
        {"食品": ["2802", "2914", ...], "エネルギー資源": [...], ...}
    """
    if "sector_17_name" not in listed_stocks.columns:
        raise ValueError("listed_stocks に sector_17_name 列がありません")

    sector_map: dict[str, list[str]] = {}
    for _, row in listed_stocks.iterrows():
        sec = row["sector_17_name"]
        if pd.isna(sec) or not sec:
            continue
        sector_map.setdefault(sec, []).append(row["code"])

    logger.info(
        "セクター→銘柄マッピング: %d セクター, %d 銘柄",
        len(sector_map),
        sum(len(v) for v in sector_map.values()),
    )
    return sector_map


def build_etf_to_sector_mapping(
    listed_stocks: pd.DataFrame,
) -> dict[str, str]:
    """ETFコード → J-Quants sector_17_name の正確なマッピングを構築する。

    JP_TOPIX17 の名称と J-Quants の sector_17_name が微妙に異なる場合に対応する。
    """
    # J-Quants 側のユニークなセクター名を取得
    jquants_sectors = listed_stocks["sector_17_name"].dropna().unique()

    mapping: dict[str, str] = {}
    for etf_code, etf_sector_name in ETF_TO_SECTOR_17.items():
        # 完全一致
        if etf_sector_name in jquants_sectors:
            mapping[etf_code] = etf_sector_name
            continue
        # 正規化一致
        etf_norm = normalize_sector_name(etf_sector_name)
        matched = False
        for jq_sec in jquants_sectors:
            if normalize_sector_name(jq_sec) == etf_norm:
                mapping[etf_code] = jq_sec
                matched = True
                break
        if not matched:
            logger.warning(
                "ETF %s (%s) に対応するJ-Quantsセクターが見つかりません",
                etf_code, etf_sector_name,
            )

    return mapping


def assign_sector_signals(
    signals_df: pd.DataFrame,
    listed_stocks: pd.DataFrame,
    stock_codes: list[str] | None = None,
) -> pd.DataFrame:
    """PCA_SUBのセクターシグナルを個別株に配賦する。

    Args:
        signals_df: PCA_SUB.signals (index=date, columns=JP ETF tickers)
            例: columns=["1617", "1618", ..., "1633"]
        listed_stocks: JQuantsProvider.get_listed_stocks()
        stock_codes: 対象銘柄コードリスト (None = listed_stocks全体)

    Returns:
        DataFrame[date, code, sector_signal, sector_name, etf_code]
        各日付 × 各銘柄に、所属セクターのシグナル値を割り当て
    """
    etf_to_sector = build_etf_to_sector_mapping(listed_stocks)

    # 銘柄 → セクター → ETFコードのマッピング
    stock_to_etf: dict[str, str] = {}
    for _, row in listed_stocks.iterrows():
        code = row["code"]
        if stock_codes and code not in stock_codes:
            continue
        sec_name = row.get("sector_17_name", "")
        if pd.isna(sec_name) or not sec_name:
            continue
        # セクター名 → ETFコードを逆引き
        for etf_code, jq_sec in etf_to_sector.items():
            if jq_sec == sec_name or normalize_sector_name(jq_sec) == normalize_sector_name(sec_name):
                stock_to_etf[code] = etf_code
                break

    if not stock_to_etf:
        raise ValueError("銘柄→ETFマッピングが構築できませんでした")

    logger.info("個別株→ETFマッピング: %d 銘柄", len(stock_to_etf))

    # 日次でシグナルを配賦
    rows = []
    for dt in signals_df.index:
        sig_row = signals_df.loc[dt]
        for code, etf_code in stock_to_etf.items():
            if etf_code in sig_row.index:
                signal_val = sig_row[etf_code]
                if not np.isnan(signal_val):
                    rows.append({
                        "date": dt,
                        "code": code,
                        "sector_signal": signal_val,
                        "etf_code": etf_code,
                    })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        logger.info(
            "シグナル配賦完了: %d日 × 平均%.0f銘柄/日",
            result["date"].nunique(),
            len(result) / max(1, result["date"].nunique()),
        )
    return result
