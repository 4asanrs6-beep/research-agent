"""個別株配賦モジュール (Phase 1-3)

ETF-to-ETF Lead-Lag戦略のシグナルを個別株に配賦し、
フィルター型/ランキング型のL/Sポートフォリオを構築・評価する。
"""

import logging

import pandas as pd

from .config import StockAllocationConfig
from .mapper import (
    build_sector_stock_map,
    build_etf_to_sector_mapping,
    assign_sector_signals,
)
from .portfolio import build_filter_portfolio, build_ranking_portfolio
from .backtester import run_stock_backtest, StockBacktestResult
from .residual_analysis import (
    decompose_returns,
    evaluate_signal_on_residuals,
    ResidualAnalysisResult,
)

logger = logging.getLogger(__name__)


def run_stock_allocation(
    backtest_result,
    config: StockAllocationConfig,
    jquants_provider,
    cache=None,
    progress_callback=None,
) -> StockBacktestResult:
    """Phase 1 パイプライン: ETFシグナル → 個別株配賦 → バックテスト

    Args:
        backtest_result: core.lead_lag.run_backtest() の結果
        config: StockAllocationConfig
        jquants_provider: JQuantsProvider
        cache: DataCache
        progress_callback: (msg, pct) コールバック

    Returns:
        StockBacktestResult
    """
    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    # --- 1. PCA_SUB シグナルの取得 ---
    _progress("PCA_SUBシグナルを取得中...", 0.0)
    pca_sub = backtest_result.strategies.get("PCA_SUB")
    if pca_sub is None or pca_sub.signals is None:
        raise ValueError("PCA_SUB の結果がありません")
    signals_df = pca_sub.signals
    etf_config = backtest_result.config

    # --- ETF版との連動: quantile_q → n_sectors 自動計算 ---
    if config.quantile_q > 0:
        n_jp = len(signals_df.columns)
        n_long, n_short = config.resolve_n_sectors(n_jp)
        logger.info("ETF版 q=%.2f → セクター数 L=%d / S=%d (JP %d)", config.quantile_q, n_long, n_short, n_jp)
    elif etf_config and etf_config.quantile_q > 0:
        # configで明示指定がなければ ETF版のqを自動引用
        n_jp = len(signals_df.columns)
        from .config import compute_n_sectors_from_q
        n_long = n_short = compute_n_sectors_from_q(etf_config.quantile_q, n_jp)
        logger.info("ETF版 q=%.2f を自動引用 → セクター数 L=%d / S=%d", etf_config.quantile_q, n_long, n_short)
    else:
        n_long, n_short = config.n_sectors_long, config.n_sectors_short

    # --- バックテスト期間: ETF版シグナル期間から自動取得 ---
    sig_dates = signals_df.dropna(how="all").index
    if len(sig_dates) == 0:
        raise ValueError("有効なシグナルがありません")
    sig_start = sig_dates[0]
    sig_end = sig_dates[-1]

    bt_start = config.backtest_start or sig_start.strftime("%Y-%m-%d")
    bt_end = config.backtest_end or sig_end.strftime("%Y-%m-%d")
    logger.info("バックテスト期間: %s — %s (シグナル範囲: %s — %s)",
                bt_start, bt_end, sig_start.strftime("%Y-%m-%d"), sig_end.strftime("%Y-%m-%d"))

    # --- 2. ユニバース取得 ---
    _progress("銘柄ユニバースを取得中...", 0.05)
    listed_stocks = jquants_provider.get_listed_stocks()

    from core.universe_filter import UniverseFilterConfig, apply_universe_filter
    uf_config = UniverseFilterConfig(
        market_segments=config.market_segments,
        scale_categories=config.scale_categories,
        margin_tradable_only=config.margin_tradable_only,
        exclude_etf_reit=True,
    )
    filtered_stocks = apply_universe_filter(listed_stocks, uf_config)
    stock_codes = filtered_stocks["code"].tolist()
    logger.info("ユニバース: %d 銘柄", len(stock_codes))

    # --- 3. シグナル配賦 ---
    _progress("セクターシグナルを個別株に配賦中...", 0.10)
    stock_signals = assign_sector_signals(signals_df, listed_stocks, stock_codes)
    if stock_signals.empty:
        raise ValueError("シグナル配賦に失敗しました")

    # --- 4. バックテスト期間の絞り込み ---
    stock_signals = stock_signals[
        (stock_signals["date"] >= bt_start) & (stock_signals["date"] <= bt_end)
    ]

    # --- 5. ポートフォリオ構築 ---
    _progress("ポートフォリオを構築中...", 0.20)
    portfolio = build_filter_portfolio(
        stock_signals,
        n_sectors_long=n_long,
        n_sectors_short=n_short,
    )

    if portfolio.empty:
        raise ValueError("ポートフォリオが空です")

    # --- 6. 個別株価格取得 ---
    _progress("個別株価格を取得中...", 0.30)
    trade_codes = portfolio["code"].unique().tolist()
    min_date = portfolio["date"].min().strftime("%Y-%m-%d")
    max_date = portfolio["date"].max().strftime("%Y-%m-%d")

    stock_prices = _fetch_stock_prices(
        jquants_provider, trade_codes, min_date, max_date,
        cache=cache, progress_callback=lambda msg, pct: _progress(msg, 0.30 + pct * 0.50),
    )

    if stock_prices.empty:
        raise ValueError("株価データの取得に失敗しました")

    # --- 7. バックテスト ---
    _progress("バックテスト実行中...", 0.85)
    result = run_stock_backtest(
        portfolio, stock_prices,
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate,
        benchmark_returns=backtest_result.benchmark_returns,
    )
    # ETF版のメトリクスを比較用に保存
    if pca_sub.metrics:
        result.etf_metrics = pca_sub.metrics

    _progress("完了", 1.0)
    return result


def _fetch_stock_prices(
    provider,
    codes: list[str],
    start_date: str,
    end_date: str,
    cache=None,
    progress_callback=None,
) -> pd.DataFrame:
    """個別株価格を月単位チャンクで一括取得する。

    全銘柄を月単位で一括取得 → 対象銘柄だけフィルタ。
    2年分なら24回のAPI呼び出しで済む。
    """
    import time as _time

    def _progress(msg, pct):
        if progress_callback:
            progress_callback(msg, pct)

    code_set = set(codes)
    keep_cols = ["date", "code", "open", "close", "adj_open", "adj_close", "volume"]

    # 月単位チャンクを生成
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.DateOffset(months=1) - pd.DateOffset(days=1), end)
        chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = cursor + pd.DateOffset(months=1)

    n_chunks = len(chunks)
    all_prices = []

    for i, (c_start, c_end) in enumerate(chunks):
        _progress(f"株価取得中... ({i+1}/{n_chunks}ヶ月: {c_start}〜{c_end})", i / max(1, n_chunks))

        success = False
        for attempt in range(4):
            try:
                df = provider.get_price_daily(code=None, start_date=c_start, end_date=c_end)
                if df is not None and len(df) > 0:
                    filtered = df[df["code"].isin(code_set)]
                    if len(filtered) > 0:
                        available = [c for c in keep_cols if c in filtered.columns]
                        all_prices.append(filtered[available].copy())
                success = True
                break
            except Exception as e:
                if "429" in str(e):
                    wait = 30 * (attempt + 1)  # 30s, 60s, 90s, 120s
                    logger.warning("429レートリミット、%d秒待機 (試行%d/4)", wait, attempt + 1)
                    _progress(f"レートリミット、{wait}秒待機中... ({c_start})", i / max(1, n_chunks))
                    _time.sleep(wait)
                else:
                    logger.warning("チャンク %s〜%s エラー: %s", c_start, c_end, e)
                    break

        if not success:
            logger.warning("チャンク %s〜%s をスキップ", c_start, c_end)

        # チャンク間ウェイト（429予防: 成功時も5秒空ける）
        if i < n_chunks - 1:
            _time.sleep(5.0)

    if not all_prices:
        return pd.DataFrame()

    result = pd.concat(all_prices, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    _progress("株価取得完了", 1.0)
    logger.info("株価取得: %d日 × %d銘柄 = %d行", result["date"].nunique(), result["code"].nunique(), len(result))
    return result
