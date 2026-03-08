"""銘柄リスト生成 — J-Quants株価データからQlib instruments/all.txtを構築"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def generate_instruments(
    price_df: pd.DataFrame,
    output_dir: Path,
    calendar_dates: list[str] | None = None,
) -> pd.DataFrame:
    """全銘柄の取引期間をQlib instruments形式で出力する。

    Args:
        price_df: 全銘柄株価 DataFrame (columns: code, date, adj_close, ...)
        output_dir: Qlibデータルート
        calendar_dates: カレンダー日付リスト (Noneなら制約なし)

    Returns:
        銘柄リスト DataFrame (columns: code, start_date, end_date)
    """
    logger.info("銘柄リスト生成中...")

    df = price_df.copy()
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    if calendar_dates is not None:
        cal_set = set(calendar_dates)
        df = df[df["date_str"].isin(cal_set)]

    instruments = (
        df.groupby("code")["date_str"]
        .agg(start_date="min", end_date="max")
        .reset_index()
    )

    # コードを文字列として保持
    instruments["code"] = instruments["code"].astype(str)

    inst_dir = output_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    inst_path = inst_dir / "all.txt"

    lines = []
    for _, row in instruments.iterrows():
        lines.append(f"{row['code']}\t{row['start_date']}\t{row['end_date']}")
    inst_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(
        "銘柄リスト出力: %d銘柄 → %s", len(instruments), inst_path
    )
    return instruments
