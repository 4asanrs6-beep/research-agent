"""日米業種リードラグ戦略モジュール

中川ら (SIG-FIN-036, 2026) の部分空間正則化PCAを用いた
日米業種リードラグ投資戦略の実装。
"""

import logging

from .config import LeadLagConfig
from .strategy import BacktestResult

logger = logging.getLogger(__name__)


def run_backtest(
    config: LeadLagConfig,
    jquants_provider=None,
    cache=None,
    progress_callback=None,
) -> BacktestResult:
    """全工程を一括実行するパイプライン関数。

    1. US/JP ETF データ取得
    2. 共通取引日への整列
    3. 全戦略の実行
    4. パフォーマンス指標の計算
    """
    from .data_fetcher import fetch_us_etf_data, fetch_jp_etf_data, build_aligned_dataset, fetch_benchmark_returns
    from .strategy import run_all_strategies

    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    end_date = config.end_date
    if not end_date:
        from datetime import date
        end_date = date.today().strftime("%Y-%m-%d")

    # --- Step 1: データ取得 ---
    _progress("US ETF データを取得中...", 0.0)
    us_data = fetch_us_etf_data(
        tickers=config.us_tickers,
        start=config.start_date,
        end=end_date,
        cache=cache,
    )

    _progress("JP ETF データを取得中...", 0.05)
    jp_data = fetch_jp_etf_data(
        tickers=config.jp_tickers,
        start=config.start_date,
        end=end_date,
        jquants_provider=jquants_provider,
        cache=cache,
    )

    # --- Step 2: 整列 ---
    _progress("日米データを整列中...", 0.08)
    aligned = build_aligned_dataset(us_data, jp_data, config.us_tickers, config.jp_tickers)

    if len(aligned.common_dates) < config.rolling_window + 10:
        raise ValueError(
            f"共通取引日が不足しています ({len(aligned.common_dates)}日)。"
            f"ローリングウィンドウ (L={config.rolling_window}) に対して最低 {config.rolling_window + 10} 日必要です。"
        )

    # --- Step 3: ベンチマーク (TOPIX) の取得 (戦略計算前に必要) ---
    benchmark_returns = None
    try:
        _progress("TOPIX ベンチマークを取得中...", 0.09)
        bm = fetch_benchmark_returns(config.start_date, end_date, cache=cache)
        if len(bm) > 0:
            common = bm.index.intersection(aligned.common_dates)
            benchmark_returns = bm.loc[common]
    except Exception as e:
        logger.warning("TOPIX ベンチマーク取得失敗: %s", e)

    # --- Step 4: 戦略実行 ---
    def _strategy_progress(msg, pct):
        _progress(msg, 0.10 + pct * 0.90)

    # overnightギャップ計算 (adj_open_t / adj_close_{t-1} - 1)
    import numpy as np
    jp_close = aligned.jp_cc_returns.values  # close-to-close
    jp_oc_vals = aligned.jp_oc_returns.values  # open-to-close
    # overnight_gap = cc / oc - 1 (近似: cc = gap * oc なので gap = (1+cc)/(1+oc) - 1)
    jp_gaps = np.full_like(jp_close, np.nan)
    for t in range(1, len(jp_close)):
        for j in range(jp_close.shape[1]):
            cc = jp_close[t, j]
            oc = jp_oc_vals[t, j]
            if not np.isnan(cc) and not np.isnan(oc) and abs(1 + oc) > 1e-10:
                jp_gaps[t, j] = (1 + cc) / (1 + oc) - 1

    from .strategy import BacktestResult
    result = run_all_strategies(
        us_cc=aligned.us_cc_returns.values,
        jp_cc=aligned.jp_cc_returns.values,
        jp_oc=aligned.jp_oc_returns.values,
        us_tickers=aligned.us_tickers,
        jp_tickers=aligned.jp_tickers,
        dates=aligned.common_dates,
        config=config,
        progress_callback=_strategy_progress,
        jp_overnight_gaps=jp_gaps,
    )
    result.benchmark_returns = benchmark_returns
    result.jp_close_prices = aligned.jp_close_prices
    result.jp_open_prices = aligned.jp_open_prices

    return result
