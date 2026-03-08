"""日本株営業日カレンダー生成 — TOPIXの取引日からQlibカレンダーを構築"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def generate_calendar(
    provider,
    output_dir: Path,
    start_date: str = "2017-01-01",
    end_date: str = "2026-03-07",
) -> list[str]:
    """TOPIXの取引日を使ってQlib用カレンダーファイルを生成する。

    Args:
        provider: JQuantsProvider インスタンス
        output_dir: Qlibデータルート (例: storage/qlib_data/jp_data)
        start_date: カレンダー開始日
        end_date: カレンダー終了日

    Returns:
        営業日リスト (YYYY-MM-DD)
    """
    logger.info("TOPIXデータから営業日カレンダーを生成中...")

    topix = provider.get_index_prices(
        index_code="0000",
        start_date=start_date,
        end_date=end_date,
    )

    if topix.empty:
        raise ValueError("TOPIXデータが取得できません。J-Quants APIを確認してください。")

    dates = sorted(topix["date"].dt.strftime("%Y-%m-%d").unique().tolist())
    logger.info("営業日カレンダー: %d日 (%s ~ %s)", len(dates), dates[0], dates[-1])

    cal_dir = output_dir / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_path = cal_dir / "day.txt"
    cal_path.write_text("\n".join(dates) + "\n", encoding="utf-8")

    logger.info("カレンダーファイル出力: %s", cal_path)
    return dates
