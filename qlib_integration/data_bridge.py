"""J-Quants → Qlib bin変換ブリッジ

Qlibバイナリ形式:
  各 {CODE}/{field}.day.bin は numpy float32 配列:
    array[0] = カレンダー上でデータ開始するインデックス (float32)
    array[1:] = カレンダーに整列した各日の値 (欠損はNaN)
"""

import logging
import time as _time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from qlib_integration.config import BIN_FIELDS, DATA_START_DATE, DATA_END_DATE, QLIB_DATA_DIR
from qlib_integration.calendar_jp import generate_calendar
from qlib_integration.instruments_jp import generate_instruments

logger = logging.getLogger(__name__)


def _api_call_with_retry(fn, max_retries: int = 5):
    """429エラー時にウェイト+リトライするラッパー"""
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if "429" in str(e) or "too many" in str(e).lower():
                wait = 15.0 * (attempt + 1)
                logger.warning(
                    "429レート制限 (試行%d/%d)、%d秒待機...",
                    attempt + 1, max_retries, int(wait),
                )
                _time.sleep(wait)
            else:
                raise
    raise last_err


def _fetch_all_prices_chunked(provider, start_date: str, end_date: str, on_progress=None):
    """全銘柄株価を月単位チャンクで取得（429対策）"""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.DateOffset(months=1) - pd.DateOffset(days=1), end)
        chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = cursor + pd.DateOffset(months=1)

    frames = []
    total = len(chunks)
    for i, (c_start, c_end) in enumerate(chunks):
        if on_progress:
            pct = 0.10 + 0.20 * (i / total)
            on_progress(f"株価取得中... ({i+1}/{total}チャンク: {c_start}~{c_end})", pct)

        df = _api_call_with_retry(
            lambda s=c_start, e=c_end: provider.get_price_daily(
                code=None, start_date=s, end_date=e,
            )
        )
        if df is not None and not df.empty:
            frames.append(df)

        # チャンク間ウェイト（429予防）
        if i < total - 1:
            _time.sleep(2.0)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _write_bin(path: Path, start_index: int, values: np.ndarray) -> None:
    """Qlib binファイルを書き出す。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    # header: start_index as float32, then values as float32
    header = np.array([start_index], dtype=np.float32)
    data = np.concatenate([header, values.astype(np.float32)])
    data.tofile(str(path))


def convert_full(
    provider,
    output_dir: Path | None = None,
    start_date: str = DATA_START_DATE,
    end_date: str = DATA_END_DATE,
    on_progress: Callable[[str, float], None] | None = None,
) -> dict:
    """J-Quantsデータを完全にQlibバイナリ形式へ変換する。

    Args:
        provider: JQuantsProvider インスタンス
        output_dir: Qlibデータ出力先 (デフォルト: QLIB_DATA_DIR)
        start_date: 変換開始日
        end_date: 変換終了日
        on_progress: 進捗コールバック (message, 0.0~1.0)

    Returns:
        変換結果サマリ dict
    """
    if output_dir is None:
        output_dir = QLIB_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    def progress(msg: str, pct: float):
        logger.info("[%.0f%%] %s", pct * 100, msg)
        if on_progress:
            on_progress(msg, pct)

    # Step 1: カレンダー生成
    progress("カレンダー生成中...", 0.0)
    calendar_dates = generate_calendar(provider, output_dir, start_date, end_date)
    date_to_idx = {d: i for i, d in enumerate(calendar_dates)}

    # Step 2: 全銘柄株価を月単位チャンクで取得（429対策）
    progress("株価データ取得中（月単位チャンク）...", 0.10)
    price_df = _fetch_all_prices_chunked(
        provider, start_date, end_date, on_progress=on_progress,
    )
    if price_df.empty:
        raise ValueError("株価データが取得できません。")

    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df["code"] = price_df["code"].astype(str)

    # VWAP近似: (open + high + low + close) / 4
    if "adj_open" in price_df.columns:
        price_df["vwap"] = (
            price_df["adj_open"]
            + price_df["adj_high"]
            + price_df["adj_low"]
            + price_df["adj_close"]
        ) / 4.0

    progress("株価データ取得完了: %d行" % len(price_df), 0.30)

    # Step 3: 銘柄リスト生成
    progress("銘柄リスト生成中...", 0.35)
    instruments_df = generate_instruments(price_df, output_dir, calendar_dates)

    # Step 4: 銘柄ごとにbinファイル変換
    codes = sorted(price_df["code"].unique())
    n_codes = len(codes)
    progress("binファイル変換開始: %d銘柄" % n_codes, 0.40)

    # フィールドマッピング (VWAP含む)
    field_map = dict(BIN_FIELDS)
    if "vwap" in price_df.columns:
        field_map["vwap"] = "vwap"

    features_dir = output_dir / "features"
    n_calendar = len(calendar_dates)
    converted_count = 0
    skipped_count = 0

    for i, code in enumerate(codes):
        code_df = price_df[price_df["code"] == code].copy()
        code_df["date_str"] = code_df["date"].dt.strftime("%Y-%m-%d")

        # カレンダーに存在する日付のみ
        code_df = code_df[code_df["date_str"].isin(date_to_idx)]
        if code_df.empty:
            skipped_count += 1
            continue

        code_df = code_df.sort_values("date_str")
        code_dates = code_df["date_str"].tolist()
        start_idx = date_to_idx[code_dates[0]]
        end_idx = date_to_idx[code_dates[-1]]
        span = end_idx - start_idx + 1

        code_dir = features_dir / code

        for src_col, bin_name in field_map.items():
            if src_col not in code_df.columns:
                continue

            # NaN配列を確保し、データがある日だけ埋める
            values = np.full(span, np.nan, dtype=np.float32)
            for _, row in code_df.iterrows():
                idx = date_to_idx[row["date_str"]] - start_idx
                val = row[src_col]
                if pd.notna(val):
                    values[idx] = float(val)

            bin_path = code_dir / f"{bin_name}.day.bin"
            _write_bin(bin_path, start_idx, values)

        converted_count += 1

        if (i + 1) % 500 == 0 or (i + 1) == n_codes:
            pct = 0.40 + 0.55 * (i + 1) / n_codes
            progress(
                "binファイル変換中: %d/%d銘柄" % (i + 1, n_codes),
                pct,
            )

    progress("変換完了", 1.0)

    result = {
        "output_dir": str(output_dir),
        "n_calendar_days": n_calendar,
        "calendar_range": f"{calendar_dates[0]} ~ {calendar_dates[-1]}",
        "n_instruments": len(instruments_df),
        "n_converted": converted_count,
        "n_skipped": skipped_count,
        "fields": list(field_map.values()),
        "n_price_rows": len(price_df),
    }
    logger.info("変換結果: %s", result)
    return result


def get_conversion_status(output_dir: Path | None = None) -> dict | None:
    """既存の変換データのステータスを返す。未変換ならNone。"""
    if output_dir is None:
        output_dir = QLIB_DATA_DIR

    cal_path = output_dir / "calendars" / "day.txt"
    inst_path = output_dir / "instruments" / "all.txt"
    features_dir = output_dir / "features"

    if not cal_path.exists():
        return None

    cal_lines = [l.strip() for l in cal_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    inst_lines = [l.strip() for l in inst_path.read_text(encoding="utf-8").splitlines() if l.strip()] if inst_path.exists() else []

    # 銘柄数 = featuresディレクトリの子ディレクトリ数
    n_features = len(list(features_dir.iterdir())) if features_dir.exists() else 0

    # サイズ計算
    total_size = 0
    if features_dir.exists():
        for f in features_dir.rglob("*.bin"):
            total_size += f.stat().st_size

    return {
        "n_calendar_days": len(cal_lines),
        "calendar_range": f"{cal_lines[0]} ~ {cal_lines[-1]}" if cal_lines else "",
        "n_instruments": len(inst_lines),
        "n_features_dirs": n_features,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }
