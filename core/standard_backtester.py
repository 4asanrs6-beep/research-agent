"""標準バックテスト・オーケストレーター

データ取得 → シグナル生成 → ベクトル化イベントスタディ → 統計検定 → 評価
の全工程を一括実行する。
"""

import logging
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from core.evaluator import Evaluator
from core.signal_generator import SignalConfig, SignalGenerator
from core.universe_filter import UniverseFilterConfig, apply_universe_filter

logger = logging.getLogger(__name__)


@dataclass
class PreloadedData:
    """バックテスト間で再利用するキャッシュデータ"""
    prices_df: pd.DataFrame             # 全株価（テクニカル指標算出前）
    topix_df: pd.DataFrame              # TOPIX指数
    margin_df: pd.DataFrame | None      # 信用取引データ
    codes: list[str] = field(default_factory=list)
    code_name_map: dict = field(default_factory=dict)
    start_date: str = ""
    end_date: str = ""


# ======================================================================
# レートリミッター（Token Bucket + 429 サーキットブレーカー）
# ======================================================================
class _RateLimiter:
    """APIコール用レートリミッター

    - max_burst=1: トークンが蓄積されても1つまで（バースト防止）
    - 429検出時: 全ワーカーが一定時間停止（サーキットブレーカー）
    """

    def __init__(self, rate: float = 1.0, max_burst: int = 1):
        self._rate = rate
        self._max = max_burst
        self._lock = threading.Lock()
        self._tokens = 1.0
        self._last = _time.time()
        # サーキットブレーカー: 429検出時に全ワーカー停止
        self._pause_until = 0.0

    def acquire(self):
        while True:
            with self._lock:
                now = _time.time()
                # サーキットブレーカー: 停止中なら待つ
                if now < self._pause_until:
                    wait = self._pause_until - now
                    self._lock.release()
                    _time.sleep(wait)
                    self._lock.acquire()
                    now = _time.time()

                self._tokens = min(
                    self._max,
                    self._tokens + (now - self._last) * self._rate,
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            _time.sleep(0.1)

    def pause(self, seconds: float):
        """429検出時: 全ワーカーを指定秒数停止"""
        with self._lock:
            self._pause_until = max(
                self._pause_until,
                _time.time() + seconds,
            )
            self._tokens = 0.0


def _api_call_with_retry(fn, limiter: "_RateLimiter", max_retries: int = 2):
    """API呼び出しをレートリミッター + リトライ付きで実行

    jquantsapi ライブラリ内部にもリトライがあるため、
    ここでの429リトライは長めの待機 + 少ない回数に限定する。
    """
    limiter.acquire()
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if "429" in str(e):
                # ライブラリ内部リトライも全て失敗 → 長めに停止
                wait = 20.0 * (attempt + 1)  # 20s, 40s
                logger.info("429エラー、全ワーカー%s秒停止 (試行%d/%d)", wait, attempt + 1, max_retries)
                limiter.pause(wait)
                _time.sleep(wait)
                limiter.acquire()
            else:
                raise
    logger.warning("リトライ上限到達: %s", last_err)
    return None


class StandardBacktester:
    """パラメータベースの標準バックテスト実行エンジン"""

    def __init__(self, data_provider, db):
        self.provider = data_provider
        self.db = db
        self.signal_gen = SignalGenerator()
        self._limiter = _RateLimiter(rate=1.0, max_burst=1)

    # ------------------------------------------------------------------
    # メインエントリーポイント
    # ------------------------------------------------------------------
    def run(
        self,
        signal_config: SignalConfig,
        universe_config: UniverseFilterConfig,
        initial_capital: int = 10_000_000,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.001,
        start_date: str = "2021-01-01",
        end_date: str | None = None,
        max_stocks: int = 50,
        n_recent_examples: int = 10,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> dict:
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        def _progress(msg: str, pct: float = 0.0):
            if on_progress:
                on_progress(msg, pct)

        try:
            return self._run_inner(
                signal_config, universe_config,
                initial_capital, commission_rate, slippage_rate,
                start_date, end_date, max_stocks, n_recent_examples, _progress,
            )
        except Exception as e:
            logger.error("標準バックテストエラー: %s", e, exc_info=True)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # データプリロード（1回だけ実行し、複数バックテストで再利用）
    # ------------------------------------------------------------------
    def preload_data(
        self,
        universe_config: UniverseFilterConfig,
        start_date: str,
        end_date: str,
        max_stocks: int = 50,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> PreloadedData:
        """ユニバースフィルタ → 株価取得 → TOPIX取得 → 信用取得をまとめて実行

        この処理が70-200秒かかる重い部分。
        結果の PreloadedData を run_with_preloaded_data() に渡すことで
        2回目以降のバックテストではデータ取得をスキップできる。
        """
        def _progress(msg: str, pct: float = 0.0):
            if on_progress:
                on_progress(msg, pct)

        # 1. ユニバースフィルタ
        _progress("ユニバースをフィルタリング中...", 0.05)
        all_stocks = self.provider.get_listed_stocks()
        filtered = apply_universe_filter(all_stocks, universe_config)

        if filtered.empty:
            raise ValueError("フィルタ条件に合致する銘柄がありません")

        codes = filtered["code"].unique().tolist()
        if len(codes) > max_stocks:
            codes = list(np.random.choice(codes, max_stocks, replace=False))
        logger.info("対象銘柄数: %d", len(codes))

        code_name_map: dict[str, str] = {}
        if "name" in filtered.columns:
            for _, row in filtered[filtered["code"].isin(codes)].iterrows():
                code_name_map[row["code"]] = row["name"]

        # ウォームアップ期間
        warmup_days = 300
        start_dt = pd.Timestamp(start_date)
        warmup_start = (start_dt - timedelta(days=warmup_days + 100)).strftime("%Y-%m-%d")

        # 2. 株価データ取得
        _progress("株価データを取得中...", 0.10)
        biz_dates = pd.bdate_range(warmup_start, end_date)
        codes_set = set(codes)
        n_biz_dates = len(biz_dates)

        if len(codes) < n_biz_dates:
            logger.info("取得戦略: 銘柄別 (%d銘柄 < %d営業日)", len(codes), n_biz_dates)
            prices_df = self._fetch_prices_by_stock(codes, warmup_start, end_date, _progress)
        else:
            logger.info("取得戦略: 日付別 (%d銘柄 >= %d営業日)", len(codes), n_biz_dates)
            prices_df = self._fetch_prices_by_date(biz_dates, _progress)

        if prices_df is None or prices_df.empty:
            raise ValueError("株価データを取得できませんでした")

        if "date" not in prices_df.columns:
            raise ValueError("株価データにdate列がありません")
        prices_df["date"] = pd.to_datetime(prices_df["date"])
        prices_df = prices_df[prices_df["code"].isin(codes_set)].copy()
        logger.info("株価データ: %d行 (%d銘柄)", len(prices_df), prices_df["code"].nunique())

        if prices_df.empty:
            raise ValueError("対象銘柄の株価データが見つかりませんでした")

        # セクター情報を付与
        if "sector_17_name" in filtered.columns:
            sector_map = filtered.set_index("code")["sector_17_name"].to_dict()
            prices_df["sector_17_name"] = prices_df["code"].map(sector_map)

        prices_df = prices_df.sort_values(["code", "date"])

        # 3. TOPIX指数取得
        _progress("TOPIX指数を取得中...", 0.37)
        topix = pd.DataFrame()
        topix_raw = _api_call_with_retry(
            lambda: self.provider.get_index_prices("0000", start_date, end_date),
            self._limiter,
        )
        if topix_raw is not None and not topix_raw.empty:
            topix = topix_raw
            topix["date"] = pd.to_datetime(topix["date"])
            logger.info("TOPIX取得: %d行", len(topix))
        else:
            logger.warning("TOPIX取得失敗（リトライ後も取得不可）")

        # 4. 信用取引データ取得（全銘柄分を事前取得）
        _progress("信用取引データを取得中...", 0.38)
        fridays = [d for d in biz_dates if d.weekday() == 4]
        margin_df = self._fetch_margin_by_date(fridays, _progress)
        if margin_df is not None:
            logger.info("信用取引データ: %d行", len(margin_df))

        _progress("データプリロード完了", 0.45)

        return PreloadedData(
            prices_df=prices_df,
            topix_df=topix,
            margin_df=margin_df,
            codes=codes,
            code_name_map=code_name_map,
            start_date=start_date,
            end_date=end_date,
        )

    # ------------------------------------------------------------------
    # プリロード済みデータを使った高速バックテスト
    # ------------------------------------------------------------------
    def run_with_preloaded_data(
        self,
        signal_config: SignalConfig,
        preloaded: PreloadedData,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.001,
        n_recent_examples: int = 10,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> dict:
        """プリロード済みデータを使って高速バックテスト

        データ取得をスキップするため、15-30秒で完了する。
        """
        def _progress(msg: str, pct: float = 0.0):
            if on_progress:
                on_progress(msg, pct)

        try:
            start_date = preloaded.start_date
            end_date = preloaded.end_date

            # prices_df のコピーを使用（各実行で信用指標マージ等が変わるため）
            prices_df = preloaded.prices_df.copy()

            # 信用取引データのマージ（signal_config で必要な場合のみ）
            margin_df = preloaded.margin_df
            if signal_config.needs_margin_data() and margin_df is not None:
                _progress("信用取引指標を計算中...", 0.47)
                prices_df = self.signal_gen.compute_margin_indicators(
                    prices_df, margin_df, margin_type=signal_config.margin_type,
                )

            # シグナル生成
            _progress("シグナルを生成中...", 0.50)
            signals = self.signal_gen.generate_signals(prices_df, signal_config, margin_df=None)
            n_total_signals = len(signals)

            if not signals.empty:
                signals["date"] = pd.to_datetime(signals["date"])
                signals = signals[signals["date"] >= start_date]

            logger.info(
                "シグナル: 全期間=%d, 分析期間(%s〜)=%d",
                n_total_signals, start_date, len(signals),
            )

            if signals.empty:
                price_min_date = prices_df["date"].min().strftime("%Y-%m-%d")
                price_max_date = prices_df["date"].max().strftime("%Y-%m-%d")
                analysis_rows = len(prices_df[prices_df["date"] >= start_date])
                diag_parts = [
                    f"対象銘柄: {len(preloaded.codes)}社",
                    f"株価行数: {len(prices_df)} (日付範囲: {price_min_date}〜{price_max_date})",
                    f"分析期間内の株価行数: {analysis_rows}",
                    f"ウォームアップ含む全シグナル: {n_total_signals}件",
                    f"分析期間({start_date}〜)内: 0件",
                    f"条件: {', '.join(signal_config.get_active_conditions_summary())}",
                    f"結合: {signal_config.signal_logic}",
                ]
                return {"error": "シグナルが発生しませんでした。\n" + "\n".join(diag_parts)}

            # イベントスタディ
            _progress("イベントスタディを実行中...", 0.60)
            bt_result, signal_returns, excess_returns = self._vectorized_event_study(
                prices_df=prices_df,
                signals=signals,
                holding_days=signal_config.holding_period_days,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate,
                benchmark_df=preloaded.topix_df,
                code_name_map=preloaded.code_name_map,
            )

            # 統計検定
            _progress("統計検定を実行中...", 0.80)
            stats_result = self._compute_statistics(
                signal_returns=signal_returns,
                excess_returns=excess_returns,
                bt_result=bt_result,
            )

            # 自動評価
            _progress("結果を評価中...", 0.88)
            evaluator = Evaluator()
            evaluation = evaluator.evaluate(stats_result, bt_result)

            # 直近事例収集
            _progress("直近事例を収集中...", 0.93)
            recent_examples, pending_signals = self._collect_recent_examples(
                signals=signals, prices_df=prices_df,
                holding_days=signal_config.holding_period_days,
                n=n_recent_examples,
                code_name_map=preloaded.code_name_map,
            )

            _progress("完了", 1.0)

            config_snapshot = {
                "signal_config": _signal_config_to_dict(signal_config),
                "commission_rate": commission_rate,
                "slippage_rate": slippage_rate,
                "start_date": start_date,
                "end_date": end_date,
                "n_stocks_used": len(preloaded.codes),
                "n_signals": len(signals),
            }

            return {
                "statistics": stats_result,
                "backtest": bt_result,
                "evaluation": evaluation,
                "recent_examples": recent_examples,
                "pending_signals": pending_signals,
                "config_snapshot": config_snapshot,
            }

        except Exception as e:
            logger.error("プリロードバックテストエラー: %s", e, exc_info=True)
            return {"error": str(e)}

    def _run_inner(
        self,
        signal_config: SignalConfig,
        universe_config: UniverseFilterConfig,
        initial_capital: int,
        commission_rate: float,
        slippage_rate: float,
        start_date: str,
        end_date: str,
        max_stocks: int,
        n_recent_examples: int,
        progress: Callable,
    ) -> dict:
        # ============================================================
        # 1. ユニバースフィルタ
        # ============================================================
        progress("ユニバースをフィルタリング中...", 0.05)
        all_stocks = self.provider.get_listed_stocks()
        filtered = apply_universe_filter(all_stocks, universe_config)

        if filtered.empty:
            return {"error": "フィルタ条件に合致する銘柄がありません"}

        codes = filtered["code"].unique().tolist()
        if len(codes) > max_stocks:
            codes = list(np.random.choice(codes, max_stocks, replace=False))
        logger.info("対象銘柄数: %d", len(codes))

        # 銘柄コード→企業名マッピング
        code_name_map: dict[str, str] = {}
        if "name" in filtered.columns:
            for _, row in filtered[filtered["code"].isin(codes)].iterrows():
                code_name_map[row["code"]] = row["name"]

        # ウォームアップ期間（最大200日MA + α）
        warmup_days = 300
        start_dt = pd.Timestamp(start_date)
        warmup_start = (start_dt - timedelta(days=warmup_days + 100)).strftime("%Y-%m-%d")

        # ============================================================
        # 2. 株価データ取得（戦略的並行取得）
        # ============================================================
        progress("株価データを取得中...", 0.10)
        biz_dates = pd.bdate_range(warmup_start, end_date)
        codes_set = set(codes)
        n_biz_dates = len(biz_dates)

        if len(codes) < n_biz_dates:
            logger.info("取得戦略: 銘柄別 (%d銘柄 < %d営業日)", len(codes), n_biz_dates)
            prices_df = self._fetch_prices_by_stock(
                codes, warmup_start, end_date, progress,
            )
        else:
            logger.info("取得戦略: 日付別 (%d銘柄 >= %d営業日)", len(codes), n_biz_dates)
            prices_df = self._fetch_prices_by_date(biz_dates, progress)

        if prices_df is None or prices_df.empty:
            return {"error": "株価データを取得できませんでした"}

        if "date" not in prices_df.columns:
            return {"error": "株価データにdate列がありません"}
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        # 対象銘柄のみフィルタ（日付別取得では全銘柄含む）
        prices_df = prices_df[prices_df["code"].isin(codes_set)].copy()
        logger.info("株価データ: %d行 (%d銘柄)", len(prices_df), prices_df["code"].nunique())

        if prices_df.empty:
            return {"error": "対象銘柄の株価データが見つかりませんでした"}

        # セクター情報を付与
        if "sector_17_name" in filtered.columns:
            sector_map = filtered.set_index("code")["sector_17_name"].to_dict()
            prices_df["sector_17_name"] = prices_df["code"].map(sector_map)

        prices_df = prices_df.sort_values(["code", "date"])

        # ============================================================
        # 3. TOPIX指数取得（レートリミッター経由）
        # ============================================================
        progress("TOPIX指数を取得中...", 0.37)
        topix = pd.DataFrame()
        topix_raw = _api_call_with_retry(
            lambda: self.provider.get_index_prices("0000", start_date, end_date),
            self._limiter,
        )
        if topix_raw is not None and not topix_raw.empty:
            topix = topix_raw
            topix["date"] = pd.to_datetime(topix["date"])
            logger.info("TOPIX取得: %d行", len(topix))
        else:
            logger.warning("TOPIX取得失敗（リトライ後も取得不可）")

        # ============================================================
        # 4. 信用取引データ取得（週次・金曜のみ・並行取得）
        # ============================================================
        margin_df = None
        if signal_config.needs_margin_data():
            progress("信用取引データを取得中...", 0.38)
            fridays = [d for d in biz_dates if d.weekday() == 4]
            margin_df = self._fetch_margin_by_date(fridays, progress)
            if margin_df is not None:
                logger.info("信用取引データ: %d行", len(margin_df))
                # margin_ratio を prices_df に事前マージ（下流で参照するため）
                prices_df = self.signal_gen.compute_margin_indicators(
                    prices_df, margin_df, margin_type=signal_config.margin_type,
                )

        # ============================================================
        # 5. シグナル生成
        # ============================================================
        progress("シグナルを生成中...", 0.47)
        # margin_df=None: margin_ratio は既に prices_df にマージ済み
        signals = self.signal_gen.generate_signals(prices_df, signal_config, margin_df=None)
        n_total_signals = len(signals)

        if not signals.empty:
            signals["date"] = pd.to_datetime(signals["date"])
            signals = signals[signals["date"] >= start_date]

        logger.info(
            "シグナル: 全期間=%d, 分析期間(%s〜)=%d",
            n_total_signals, start_date, len(signals),
        )

        if signals.empty:
            # 株価データの日付カバレッジ診断
            price_min_date = prices_df["date"].min().strftime("%Y-%m-%d")
            price_max_date = prices_df["date"].max().strftime("%Y-%m-%d")
            analysis_rows = len(prices_df[prices_df["date"] >= start_date])
            diag_parts = [
                f"対象銘柄: {len(codes)}社",
                f"株価行数: {len(prices_df)} (日付範囲: {price_min_date}〜{price_max_date})",
                f"分析期間内の株価行数: {analysis_rows}",
                f"ウォームアップ含む全シグナル: {n_total_signals}件",
                f"分析期間({start_date}〜)内: 0件",
                f"条件: {', '.join(signal_config.get_active_conditions_summary())}",
                f"結合: {signal_config.signal_logic}",
            ]
            return {"error": "シグナルが発生しませんでした。\n" + "\n".join(diag_parts)}

        # ============================================================
        # 6. イベントスタディ（ベクトル化・高速）
        # ============================================================
        progress("イベントスタディを実行中...", 0.55)
        bt_result, signal_returns, excess_returns = self._vectorized_event_study(
            prices_df=prices_df,
            signals=signals,
            holding_days=signal_config.holding_period_days,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            benchmark_df=topix,
            code_name_map=code_name_map,
        )

        # ============================================================
        # 7. 統計検定（超過リターン）
        # ============================================================
        progress("統計検定を実行中...", 0.75)
        stats_result = self._compute_statistics(
            signal_returns=signal_returns,
            excess_returns=excess_returns,
            bt_result=bt_result,
        )

        # ============================================================
        # 8. 自動評価
        # ============================================================
        progress("結果を評価中...", 0.85)
        evaluator = Evaluator()
        evaluation = evaluator.evaluate(stats_result, bt_result)

        # ============================================================
        # 9. 直近事例収集
        # ============================================================
        progress("直近事例を収集中...", 0.90)
        recent_examples, pending_signals = self._collect_recent_examples(
            signals=signals, prices_df=prices_df,
            holding_days=signal_config.holding_period_days,
            n=n_recent_examples,
            code_name_map=code_name_map,
        )

        # ============================================================
        # 完了
        # ============================================================
        progress("完了", 1.0)

        config_snapshot = {
            "signal_config": _signal_config_to_dict(signal_config),
            "universe_config": _universe_config_to_dict(universe_config),
            "initial_capital": initial_capital,
            "commission_rate": commission_rate,
            "slippage_rate": slippage_rate,
            "start_date": start_date,
            "end_date": end_date,
            "max_stocks": max_stocks,
            "n_stocks_used": len(codes),
            "n_signals": len(signals),
        }

        return {
            "statistics": stats_result,
            "backtest": bt_result,
            "evaluation": evaluation,
            "recent_examples": recent_examples,
            "pending_signals": pending_signals,
            "config_snapshot": config_snapshot,
        }

    # ------------------------------------------------------------------
    # データ取得: 銘柄別並行取得
    # ------------------------------------------------------------------
    def _fetch_prices_by_stock(self, codes, start_date, end_date, progress):
        n_stocks = len(codes)
        limiter = self._limiter
        lock = threading.Lock()
        fetched_dfs: list[pd.DataFrame] = []
        completed = [0]
        errors = [0]

        est_sec = n_stocks / 1.0  # 1 req/sec
        if est_sec < 60:
            progress(f"株価取得中（銘柄別: {n_stocks}銘柄, 推定{est_sec:.0f}秒）", 0.10)
        else:
            progress(f"株価取得中（銘柄別: {n_stocks}銘柄, 推定{est_sec / 60:.0f}分）", 0.10)

        def fetch_one(code):
            return _api_call_with_retry(
                lambda: self.provider.get_price_daily(
                    code=code, start_date=start_date, end_date=end_date,
                ),
                limiter,
            )

        n_workers = min(2, max(1, n_stocks))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(fetch_one, c): c for c in codes}
            for future in as_completed(futures):
                df = future.result()
                with lock:
                    completed[0] += 1
                    if df is not None and not df.empty:
                        fetched_dfs.append(df)
                    else:
                        errors[0] += 1
                    if completed[0] % max(1, n_stocks // 10) == 0 or completed[0] == n_stocks:
                        pct = 0.10 + 0.25 * (completed[0] / n_stocks)
                        progress(f"株価取得中... ({completed[0]}/{n_stocks}銘柄)", pct)

        logger.info("株価取得完了(銘柄別): %d成功, %dエラー", len(fetched_dfs), errors[0])
        if not fetched_dfs:
            return None
        return pd.concat(fetched_dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # データ取得: 日付別並行取得
    # ------------------------------------------------------------------
    def _fetch_prices_by_date(self, biz_dates, progress):
        provider_cache = getattr(self.provider, "cache", None)
        limiter = self._limiter

        dates_to_fetch = []
        cached_dfs = []
        for d in biz_dates:
            date_clean = d.strftime("%Y%m%d")
            cache_key = f"price_alldate_{date_clean}"
            if provider_cache:
                cached = provider_cache.get(cache_key)
                if cached is not None:
                    cached_dfs.append(cached)
                    continue
            dates_to_fetch.append(d)

        n_cached = len(cached_dfs)
        n_to_fetch = len(dates_to_fetch)
        n_total = len(biz_dates)
        est_sec = n_to_fetch / 1.0  # 1 req/sec

        logger.info("株価(日付別): %d営業日中 %dキャッシュ済, %d要取得", n_total, n_cached, n_to_fetch)
        if n_to_fetch > 0:
            if est_sec < 60:
                progress(f"株価取得中（日付別: {n_cached}キャッシュ済, {n_to_fetch}日取得, 推定{est_sec:.0f}秒）", 0.10)
            else:
                progress(f"株価取得中（日付別: {n_cached}キャッシュ済, {n_to_fetch}日取得, 推定{est_sec / 60:.0f}分）", 0.10)
        else:
            progress(f"株価取得中（{n_cached}日分キャッシュ済）", 0.15)

        fetched_dfs = []
        failed_dates = []
        lock = threading.Lock()
        completed = [0]

        def fetch_one(d):
            date_str = d.strftime("%Y-%m-%d")
            result = _api_call_with_retry(
                lambda: self.provider.get_price_daily_by_date(date_str),
                limiter,
            )
            return d, result

        n_workers = min(2, max(1, n_to_fetch))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(fetch_one, d): d for d in dates_to_fetch}
            for future in as_completed(futures):
                d, df = future.result()
                with lock:
                    completed[0] += 1
                    if df is not None and not df.empty:
                        fetched_dfs.append(df)
                    else:
                        failed_dates.append(d)
                    if completed[0] % 10 == 0 or completed[0] == n_to_fetch:
                        done = n_cached + completed[0]
                        remaining = (n_to_fetch - completed[0]) / 1.0  # 1 req/sec
                        pct = 0.10 + 0.25 * (done / max(n_total, 1))
                        if remaining < 60:
                            progress(f"株価取得中... ({done}/{n_total}日, 残り約{remaining:.0f}秒)", pct)
                        else:
                            progress(f"株価取得中... ({done}/{n_total}日, 残り約{remaining / 60:.0f}分)", pct)

        if failed_dates:
            logger.warning(
                "株価取得失敗: %d日 (例: %s)",
                len(failed_dates),
                [d.strftime("%Y-%m-%d") for d in sorted(failed_dates)[:5]],
            )

        all_dfs = cached_dfs + fetched_dfs
        if not all_dfs:
            return None
        return pd.concat(all_dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # データ取得: 信用取引（日付別並行取得）
    # ------------------------------------------------------------------
    def _fetch_margin_by_date(self, fridays, progress):
        provider_cache = getattr(self.provider, "cache", None)
        limiter = self._limiter

        dates_to_fetch = []
        cached_dfs = []
        for d in fridays:
            date_clean = d.strftime("%Y%m%d")
            cache_key = f"margin_alldate_{date_clean}"
            if provider_cache:
                cached = provider_cache.get(cache_key)
                if cached is not None and not cached.empty:
                    cached_dfs.append(cached)
                    continue
            dates_to_fetch.append(d)

        n_cached = len(cached_dfs)
        n_to_fetch = len(dates_to_fetch)
        n_total = len(fridays)
        logger.info("信用取引: %d金曜中 %dキャッシュ済, %d要取得", n_total, n_cached, n_to_fetch)

        fetched_dfs = []
        lock = threading.Lock()
        completed = [0]

        def fetch_one(d):
            date_str = d.strftime("%Y-%m-%d")
            return _api_call_with_retry(
                lambda: self.provider.get_margin_trading_by_date(date_str),
                limiter,
            )

        n_workers = min(2, max(1, n_to_fetch))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(fetch_one, d): d for d in dates_to_fetch}
            for future in as_completed(futures):
                df = future.result()
                with lock:
                    completed[0] += 1
                    if df is not None and not df.empty:
                        fetched_dfs.append(df)
                    if completed[0] % 10 == 0 or completed[0] == n_to_fetch:
                        done = n_cached + completed[0]
                        pct = 0.38 + 0.07 * (done / max(n_total, 1))
                        progress(f"信用取引データ取得中... ({done}/{n_total}週)", pct)

        all_dfs = cached_dfs + fetched_dfs
        if not all_dfs:
            return None
        return pd.concat(all_dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # ベクトル化イベントスタディ（高速）
    # ------------------------------------------------------------------
    def _vectorized_event_study(
        self,
        prices_df: pd.DataFrame,
        signals: pd.DataFrame,
        holding_days: int,
        commission_rate: float,
        slippage_rate: float,
        benchmark_df: pd.DataFrame | None = None,
        code_name_map: dict[str, str] | None = None,
    ) -> tuple[dict, list[float], list[float]]:
        """ベクトル化イベントスタディ

        全シグナルの前方N日リターンを一括計算。
        ベンチマーク(TOPIX)の同期間リターンと比較して超過リターンを算出。

        Returns:
            (backtest_result_dict, signal_returns_list, excess_returns_list)
        """
        if code_name_map is None:
            code_name_map = {}
        close_col = "adj_close" if "adj_close" in prices_df.columns else "close"

        df = prices_df.sort_values(["code", "date"]).drop_duplicates(
            subset=["date", "code"], keep="last",
        ).copy()

        # --- 前方N日リターンとN日後の価格を一括計算 ---
        df["_fwd_close"] = df.groupby("code")[close_col].shift(-holding_days)
        df["_fwd_return"] = df["_fwd_close"] / df[close_col] - 1

        # --- シグナルと結合 ---
        sigs = signals[["date", "code"]].drop_duplicates().copy()
        sigs["date"] = pd.to_datetime(sigs["date"])

        merge_cols = ["date", "code", close_col, "_fwd_close", "_fwd_return"]
        if "margin_ratio" in df.columns:
            merge_cols.append("margin_ratio")

        merged = sigs.merge(
            df[merge_cols],
            on=["date", "code"],
            how="left",
        )

        # 有効なリターンのみ（保有期間分の株価がある）
        valid = merged.dropna(subset=["_fwd_return"]).copy()

        if valid.empty:
            return {
                "win_rate": 0.0,
                "total_trades": 0,
                "mean_return": 0.0,
                "n_signals": len(signals),
                "n_valid_signals": 0,
                "trade_log": [],
            }, [], []

        # --- 往復コスト調整 ---
        round_trip_cost = 2 * (commission_rate + slippage_rate)
        valid["net_return"] = valid["_fwd_return"] - round_trip_cost

        # --- ベンチマーク(TOPIX)のN日リターンを各シグナル日に紐付け ---
        valid["benchmark_return"] = np.nan
        valid["excess_return"] = np.nan

        if benchmark_df is not None and not benchmark_df.empty and "close" in benchmark_df.columns:
            bench = benchmark_df.sort_values("date").copy()
            bench["date"] = pd.to_datetime(bench["date"])
            bench = bench[["date", "close"]].drop_duplicates(subset=["date"], keep="last")
            bench["_bench_fwd"] = bench["close"].shift(-holding_days)
            bench["_bench_return"] = bench["_bench_fwd"] / bench["close"] - 1
            bench_map = bench.dropna(subset=["_bench_return"]).set_index("date")["_bench_return"]

            valid["benchmark_return"] = valid["date"].map(bench_map)
            valid["excess_return"] = valid["net_return"] - valid["benchmark_return"]
        else:
            # ベンチマークなし: 超過リターン = シグナルリターンそのもの
            valid["benchmark_return"] = 0.0
            valid["excess_return"] = valid["net_return"]

        # ベンチマークリターンが取得できたもののみ
        valid_excess = valid.dropna(subset=["excess_return"])

        signal_returns = valid["net_return"].values.tolist()
        excess_returns = valid_excess["excess_return"].values.tolist()
        benchmark_returns = valid_excess["benchmark_return"].values.tolist()

        returns_arr = np.array(signal_returns)
        excess_arr = np.array(excess_returns)

        win_rate = float(np.mean(returns_arr > 0))
        mean_return = float(np.mean(returns_arr))
        mean_excess = float(np.mean(excess_arr)) if len(excess_arr) > 0 else 0.0
        mean_benchmark = float(np.mean(benchmark_returns)) if benchmark_returns else 0.0
        excess_win_rate = float(np.mean(excess_arr > 0)) if len(excess_arr) > 0 else 0.0

        bt_result: dict = {
            "win_rate": win_rate,
            "total_trades": len(signal_returns),
            "mean_return": mean_return,
            "mean_excess_return": mean_excess,
            "mean_benchmark_return": mean_benchmark,
            "excess_win_rate": excess_win_rate,
            "n_signals": len(signals),
            "n_valid_signals": len(signal_returns),
            "holding_days": holding_days,
            "signal_returns": signal_returns,
            "benchmark_returns": benchmark_returns,
            "trade_log": [],
        }

        # --- シグナル結果ログ（直近200件） ---
        log_entries = valid.sort_values("date", ascending=False).head(200)
        trade_log = []
        for _, row in log_entries.iterrows():
            entry = {
                "date": str(row["date"])[:10],
                "code": row["code"],
                "name": code_name_map.get(row["code"], ""),
                "entry_price": round(float(row[close_col]), 1),
                "exit_price": round(float(row["_fwd_close"]), 1),
                "return_pct": round(float(row["net_return"] * 100), 2),
                "holding_days": holding_days,
            }
            if not np.isnan(row.get("excess_return", np.nan)):
                entry["excess_pct"] = round(float(row["excess_return"] * 100), 2)
            if "margin_ratio" in row.index and pd.notna(row.get("margin_ratio")):
                entry["margin_ratio"] = round(float(row["margin_ratio"]), 2)
            trade_log.append(entry)
        bt_result["trade_log"] = trade_log

        logger.info(
            "イベントスタディ完了: %d有効シグナル, 平均リターン=%.2f%%, 平均超過リターン=%.2f%%",
            len(signal_returns), mean_return * 100, mean_excess * 100,
        )

        return bt_result, signal_returns, excess_returns

    # ------------------------------------------------------------------
    # 統計検定（超過リターンの1サンプルt検定）
    # ------------------------------------------------------------------
    def _compute_statistics(
        self,
        signal_returns: list[float],
        excess_returns: list[float],
        bt_result: dict,
    ) -> dict:
        """超過リターン（シグナルリターン - ベンチマークリターン）の統計検定

        H0: 超過リターンの平均 = 0（シグナルに優位性なし）
        H1: 超過リターンの平均 ≠ 0（シグナルに優位性あり）
        """
        sig_arr = np.array(signal_returns)
        exc_arr = np.array(excess_returns)

        n_signals = len(sig_arr)
        n_excess = len(exc_arr)

        sig_mean = float(np.mean(sig_arr)) if n_signals > 0 else 0.0
        sig_std = float(np.std(sig_arr, ddof=1)) if n_signals > 1 else 0.0
        exc_mean = float(np.mean(exc_arr)) if n_excess > 0 else 0.0
        exc_std = float(np.std(exc_arr, ddof=1)) if n_excess > 1 else 0.0

        bench_mean = bt_result.get("mean_benchmark_return", 0.0)

        # 超過リターンの1サンプルt検定 (H0: mean = 0)
        if n_excess >= 2:
            t_stat, p_value = sp_stats.ttest_1samp(exc_arr, 0)
            t_stat = float(t_stat)
            p_value = float(p_value)
        else:
            t_stat, p_value = 0.0, 1.0

        # Cohen's d: 超過リターンの平均 / 超過リターンの標準偏差
        cohens_d = float(exc_mean / exc_std) if exc_std > 0 else 0.0

        win_rate = float(np.mean(sig_arr > 0)) if n_signals > 0 else 0.0
        excess_win_rate = float(np.mean(exc_arr > 0)) if n_excess > 0 else 0.0

        return {
            "test_name": "超過リターンの1サンプルt検定",
            "signal_mean": round(sig_mean, 6),
            "benchmark_mean": round(bench_mean, 6),
            "excess_mean": round(exc_mean, 6),
            "signal_std": round(sig_std, 6),
            "excess_std": round(exc_std, 6),
            "t_statistic": round(t_stat, 4),
            "p_value": p_value,
            "cohens_d": round(cohens_d, 4),
            "win_rate": round(win_rate, 4),
            "excess_win_rate": round(excess_win_rate, 4),
            "n_signals": n_signals,
            "n_excess": n_excess,
            "is_significant": p_value < 0.05 and n_excess >= 5,
        }

    # ------------------------------------------------------------------
    # 直近事例収集（完了済み + 進行中を分けて返す）
    # ------------------------------------------------------------------
    def _collect_recent_examples(
        self,
        signals: pd.DataFrame,
        prices_df: pd.DataFrame,
        holding_days: int,
        n: int = 10,
        code_name_map: dict[str, str] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """直近事例を収集

        Returns:
            (completed_examples, pending_examples)
            - completed: 測定期間が完了した事例
            - pending: 測定期間が未達の事例（進行中シグナル）
        """
        if code_name_map is None:
            code_name_map = {}
        close_col = "adj_close" if "adj_close" in prices_df.columns else "close"
        recent_signals = signals.sort_values("date", ascending=False).head(n * 5)
        completed = []
        pending = []

        for _, sig in recent_signals.iterrows():
            code = sig["code"]
            sig_date = pd.Timestamp(sig["date"])
            code_prices = prices_df[
                (prices_df["code"] == code) & (prices_df["date"] >= sig_date)
            ].sort_values("date")

            if len(code_prices) < 2:
                continue

            entry_price = code_prices.iloc[0][close_col]

            # 貸借倍率（シグナル日時点）
            mr_val = None
            if "margin_ratio" in code_prices.columns:
                mr_raw = code_prices.iloc[0].get("margin_ratio")
                if pd.notna(mr_raw):
                    mr_val = round(float(mr_raw), 2)

            if len(code_prices) >= holding_days + 1:
                # 測定完了
                exit_price = code_prices.iloc[holding_days][close_col]
                ret = (exit_price - entry_price) / entry_price * 100
                example = {
                    "signal_date": str(sig_date)[:10],
                    "code": code,
                    "name": code_name_map.get(code, ""),
                    "return_pct": round(ret, 2),
                    "entry_price": round(float(entry_price), 1),
                    "exit_price": round(float(exit_price), 1),
                    "holding_days": holding_days,
                }
                if mr_val is not None:
                    example["margin_ratio"] = mr_val
                completed.append(example)
            else:
                # 進行中（測定期間未達）
                latest_price = code_prices.iloc[-1][close_col]
                elapsed = len(code_prices) - 1
                interim_ret = (latest_price - entry_price) / entry_price * 100
                example = {
                    "signal_date": str(sig_date)[:10],
                    "code": code,
                    "name": code_name_map.get(code, ""),
                    "interim_return_pct": round(interim_ret, 2),
                    "entry_price": round(float(entry_price), 1),
                    "latest_price": round(float(latest_price), 1),
                    "elapsed_days": elapsed,
                    "remaining_days": holding_days - elapsed,
                }
                if mr_val is not None:
                    example["margin_ratio"] = mr_val
                pending.append(example)

            if len(completed) >= n:
                break

        return completed, pending


# ------------------------------------------------------------------
# ヘルパー
# ------------------------------------------------------------------
def _signal_config_to_dict(cfg: SignalConfig) -> dict:
    from dataclasses import fields
    d = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if v is not None and v != f.default:
            d[f.name] = v
    return d


def _universe_config_to_dict(cfg: UniverseFilterConfig) -> dict:
    from dataclasses import fields
    d = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, list) and v:
            d[f.name] = v
        elif isinstance(v, bool) and v:
            d[f.name] = v
        elif v is not None and not isinstance(v, (list, bool)):
            if f.name in ("sector_filter_type",) and v == "none":
                continue
            d[f.name] = v
    return d
