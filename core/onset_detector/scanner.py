"""全銘柄スキャン + レンジ分類 + 過熱フィルタ"""

import json
import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OnsetDetectorConfig

logger = logging.getLogger(__name__)

MARKET_CAP_CACHE_PATH = Path("storage/market_cap_cache.json")
CACHE_VALIDITY_SECONDS = 30 * 86400  # 30日


# ------------------------------------------------------------------
# 時価総額取得（yfinance — 逐次実行でレートリミット回避）
# ------------------------------------------------------------------
def _fetch_single_market_cap(code_str: str) -> tuple[str, float]:
    """yfinanceから1銘柄の時価総額（円）を取得"""
    import io
    import contextlib
    try:
        import yfinance as yf
        ticker_code = f"{code_str[:4]}.T"
        with contextlib.redirect_stderr(io.StringIO()):
            t = yf.Ticker(ticker_code)
            cap = t.fast_info.get("marketCap", 0)
        return code_str, float(cap) if cap else 0.0
    except Exception:
        return code_str, 0.0


def fetch_market_caps(
    codes: list[str],
    progress_callback=None,
    max_workers: int = 3,
) -> dict[str, float]:
    """全銘柄の現在時価総額を取得（30日キャッシュ付き）

    レートリミット対策: 逐次実行（1銘柄ずつ）。
    初回は時間がかかるが、キャッシュにより2回目以降は即時。

    Returns
    -------
    dict[str, float]  code -> 時価総額（円）
    """
    now = _time.time()

    # キャッシュ読み込み
    cache = {}
    if MARKET_CAP_CACHE_PATH.exists():
        try:
            with open(MARKET_CAP_CACHE_PATH, "r") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    # キャッシュ有効な銘柄と要取得銘柄を分離
    result = {}
    codes_to_fetch = []
    for code in codes:
        cached = cache.get(code)
        if cached and (now - cached.get("t", 0)) < CACHE_VALIDITY_SECONDS:
            result[code] = cached["cap"]
        else:
            codes_to_fetch.append(code)

    if not codes_to_fetch:
        logger.info(f"時価総額キャッシュ: {len(result)}銘柄全てキャッシュヒット")
        return result

    logger.info(
        f"時価総額取得: {len(codes_to_fetch)}銘柄をyfinanceから取得 "
        f"(キャッシュ: {len(result)}銘柄)"
    )

    # yfinanceのログ抑制
    for _logger_name in ("yfinance", "peewee"):
        logging.getLogger(_logger_name).setLevel(logging.CRITICAL)

    n_total = len(codes_to_fetch)
    consecutive_fails = 0

    # 逐次実行（レートリミット完全回避）
    for i, code in enumerate(codes_to_fetch):
        code, cap = _fetch_single_market_cap(code)
        result[code] = cap
        if cap > 0:
            cache[code] = {"cap": cap, "t": now}
            consecutive_fails = 0
        else:
            consecutive_fails += 1
            # 連続失敗が多い場合は少し待つ
            if consecutive_fails >= 5:
                _time.sleep(2.0)
                consecutive_fails = 0

        if progress_callback and (i + 1) % 20 == 0:
            progress_callback(i + 1, n_total, "時価総額取得中")

        # 50件ごとにキャッシュ中間保存
        if (i + 1) % 50 == 0:
            try:
                MARKET_CAP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(MARKET_CAP_CACHE_PATH, "w") as f:
                    json.dump(cache, f)
            except Exception:
                pass

    if progress_callback:
        progress_callback(n_total, n_total, "時価総額取得完了")

    # 取得結果サマリー
    n_success = sum(1 for c in codes_to_fetch if result.get(c, 0) > 0)
    n_fail = len(codes_to_fetch) - n_success
    logger.info(
        f"時価総額取得結果: 成功={n_success}, 失敗={n_fail} / {len(codes_to_fetch)}件"
    )

    # キャッシュ保存
    try:
        MARKET_CAP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MARKET_CAP_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        logger.warning(f"時価総額キャッシュ保存失敗: {e}")

    return result


@dataclass
class ScanResult:
    """スキャン結果"""
    code: str
    name: str
    sector: str
    scale_category: str
    # スコア
    p_event: float
    predicted_tte: float
    onset_score: float
    range_label: str             # "Early" / "Imminent"
    # 類型
    star_type: str
    # 説明
    top_factors: list = field(default_factory=list)
    # コンテキスト
    trailing_20d_excess: float = 0.0
    trailing_5d_return: float = 0.0
    recent_volume_change: float = 0.0


class OnsetScanner:
    """全銘柄スキャナー"""

    def __init__(self, model, feature_engine, config: OnsetDetectorConfig):
        self.model = model
        self.engine = feature_engine
        self.config = config

    def scan(
        self,
        all_prices: pd.DataFrame,
        topix: pd.DataFrame,
        listed_stocks: pd.DataFrame,
        progress_callback=None,
    ) -> list[ScanResult]:
        """全銘柄をスキャンしてTop-K候補を返す

        Pipeline:
        1. 時価総額フィルタ（yfinance）
        2. 全銘柄特徴量計算
        3. Stage 1: P(near_event) スクリーニング
        4. Stage 2: TTE予測
        5. 過熱フィルタ
        6. レンジ分類
        7. Top-K選定
        """
        cfg = self.config
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"

        # メタデータ
        meta_map = self._build_meta_map(listed_stocks)

        # 時価総額フィルタ（yfinance実値）
        scan_codes = None
        all_codes = [str(c) for c in all_prices["code"].unique()]

        if cfg.scan_min_market_cap > 0:
            min_cap_yen = cfg.scan_min_market_cap * 1e8  # 億円→円

            def _mcap_progress(done, total, msg):
                if progress_callback:
                    progress_callback(done, total, 0, phase="market_cap")

            market_caps = fetch_market_caps(
                all_codes, progress_callback=_mcap_progress,
            )
            exclude = {c for c, cap in market_caps.items() if 0 < cap < min_cap_yen}
            scan_codes = [c for c in all_codes if c not in exclude]
            logger.info(
                f"時価総額フィルタ: {len(all_codes)}銘柄 → "
                f"{len(scan_codes)}銘柄 (下限: {cfg.scan_min_market_cap:.0f}億円)"
            )

        # 1. 全銘柄特徴量
        logger.info("全銘柄特徴量計算中...")
        code_features, feature_names = self.engine.compute_features_universe(
            all_prices, topix, listed_stocks,
            scan_codes=scan_codes,
            progress_callback=progress_callback,
        )
        logger.info(f"特徴量計算完了: {len(code_features)}銘柄")

        if not code_features:
            return []

        # 配列構築
        codes = list(code_features.keys())
        X = np.stack([code_features[c] for c in codes])

        # 2. Stage 1 + Stage 2
        p_event, predicted_tte, onset_score = self.model.predict(X)

        # 3. コンテキスト情報計算 + 過熱フィルタ
        results = []
        for i, code in enumerate(codes):
            # Stage 1 スクリーニング
            if p_event[i] < cfg.p_event_min:
                continue

            # コンテキスト情報
            ctx = self._compute_context(code, all_prices, topix, close_col)

            # 過熱フィルタ
            if ctx["trailing_20d_excess"] > cfg.overheat_trailing_20d_excess:
                continue
            if ctx["trailing_5d_return"] > cfg.overheat_trailing_5d_return:
                continue

            # レンジ分類
            tte = predicted_tte[i]
            if cfg.range_imminent_min <= tte <= cfg.range_imminent_max:
                range_label = "Imminent"
            elif cfg.range_early_min <= tte <= cfg.range_early_max:
                range_label = "Early"
            else:
                range_label = "Early" if tte > cfg.range_early_max else "Imminent"

            # 特徴量寄与
            top_factors = self.model.explain_prediction(X[i:i + 1])

            meta = meta_map.get(code, {})
            results.append(ScanResult(
                code=code,
                name=meta.get("name", ""),
                sector=meta.get("sector_17_name", ""),
                scale_category=meta.get("scale_category", ""),
                p_event=float(p_event[i]),
                predicted_tte=float(tte),
                onset_score=float(onset_score[i]),
                range_label=range_label,
                star_type=str(meta.get("star_type", "unknown")),
                top_factors=[
                    {"name": n, "contribution": round(c, 4)}
                    for n, c in top_factors[:5]
                ],
                trailing_20d_excess=ctx["trailing_20d_excess"],
                trailing_5d_return=ctx["trailing_5d_return"],
                recent_volume_change=ctx["recent_volume_change"],
            ))

        # onset_score降順でソート + Top-K
        results.sort(key=lambda r: r.onset_score, reverse=True)
        results = results[:cfg.top_k]

        logger.info(
            f"スキャン完了: {len(results)}候補 "
            f"(Imminent={sum(1 for r in results if r.range_label == 'Imminent')}, "
            f"Early={sum(1 for r in results if r.range_label == 'Early')})"
        )
        return results

    # ------------------------------------------------------------------
    # ヘルパー
    # ------------------------------------------------------------------
    @staticmethod
    def _build_meta_map(listed_stocks: pd.DataFrame) -> dict:
        meta_map = {}
        cols = ["code", "name", "sector_17_name", "market_name", "scale_category"]
        available = [c for c in cols if c in listed_stocks.columns]
        if available:
            for _, row in listed_stocks[available].iterrows():
                meta_map[str(row["code"])] = {c: row.get(c, "") for c in available}
        return meta_map

    def _compute_context(
        self,
        code: str,
        all_prices: pd.DataFrame,
        topix: pd.DataFrame,
        close_col: str,
    ) -> dict:
        """銘柄のコンテキスト情報（過熱判定用）を計算"""
        grp = all_prices[all_prices["code"] == code].sort_values("date")
        if len(grp) < 5:
            return {"trailing_20d_excess": 0.0, "trailing_5d_return": 0.0, "recent_volume_change": 0.0}

        close = grp[close_col].astype(float).values
        volume = grp["volume"].astype(float).values

        # trailing 5d return
        if len(close) >= 6 and close[-6] > 0:
            trailing_5d = close[-1] / close[-6] - 1
        else:
            trailing_5d = 0.0

        # trailing 20d excess
        if len(close) >= 21 and close[-21] > 0:
            stock_ret_20d = close[-1] / close[-21] - 1
            # TOPIX 20d return
            topix_close_col = "close" if "close" in topix.columns else topix.columns[-1]
            topix_sorted = topix.sort_values("date")
            topix_vals = topix_sorted[topix_close_col].astype(float).values
            if len(topix_vals) >= 21 and topix_vals[-21] > 0:
                topix_ret_20d = topix_vals[-1] / topix_vals[-21] - 1
            else:
                topix_ret_20d = 0.0
            trailing_20d_excess = stock_ret_20d - topix_ret_20d
        else:
            trailing_20d_excess = 0.0

        # recent volume change
        if len(volume) >= 20:
            vol_recent = np.mean(volume[-5:])
            vol_base = np.mean(volume[-20:])
            vol_change = (vol_recent / vol_base - 1) if vol_base > 0 else 0.0
        else:
            vol_change = 0.0

        return {
            "trailing_20d_excess": float(trailing_20d_excess),
            "trailing_5d_return": float(trailing_5d),
            "recent_volume_change": float(vol_change),
        }
