"""日米業種リードラグ戦略モジュール

中川ら (SIG-FIN-036, 2026) の部分空間正則化PCAを用いた
日米業種リードラグ投資戦略の実装。
"""

import logging

from .config import LeadLagConfig
from .strategy import BacktestResult
from .signal_attribution import (
    AttributionResult,
    compute_signal_attribution,
    summarize_attribution,
    get_coupling_heatmap_data,
)

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
    aligned = build_aligned_dataset(
        us_data, jp_data, config.us_tickers, config.jp_tickers,
        accumulate_us_returns=config.accumulate_us_returns,
    )

    if len(aligned.common_dates) < config.rolling_window + 10:
        raise ValueError(
            f"共通取引日が不足しています ({len(aligned.common_dates)}日)。"
            f"ローリングウィンドウ (L={config.rolling_window}) に対して最低 {config.rolling_window + 10} 日必要です。"
        )

    # --- Step 3: ベンチマーク (TOPIX) の取得 (戦略計算前に必要) ---
    benchmark_returns = None
    try:
        _progress("TOPIX ベンチマークを取得中...", 0.09)
        bm = fetch_benchmark_returns(config.start_date, end_date, jquants_provider=jquants_provider, cache=cache)
        if len(bm) > 0:
            common = bm.index.intersection(aligned.common_dates)
            benchmark_returns = bm.loc[common]
    except Exception as e:
        import traceback as _tb
        logger.warning("TOPIX ベンチマーク取得失敗: %s\n%s", e, _tb.format_exc())

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
    result.jp_to_us_date_map = aligned.jp_to_us_date_map
    result.us_trading_dates = aligned.us_trading_dates

    # --- 最新USデータが整列済み期間の後にある場合、追加シグナルを計算 ---
    try:
        _progress("最新シグナルの確認中...", 0.99)
        from datetime import date as _today_date
        today_str = _today_date.today().strftime("%Y-%m-%d")
        us_all = fetch_us_etf_data(
            tickers=config.us_tickers, start=config.start_date, end=today_str, cache=None,  # キャッシュなしで最新取得
        )
        us_pivot = us_all.pivot(index="date", columns="ticker", values="adj_close")
        us_pivot = us_pivot[aligned.us_tickers].sort_index()
        last_common = aligned.common_dates[-1]
        # 整列済み期間後のUS営業日を探す
        us_after = us_pivot.index[us_pivot.index > last_common]
        if len(us_after) > 0:
            latest_us_date = us_after[-1]
            # 最新USリターン
            us_prev_idx = us_pivot.index.get_loc(latest_us_date) - 1
            if us_prev_idx >= 0:
                latest_us_ret = (us_pivot.iloc[us_pivot.index.get_loc(latest_us_date)] /
                                 us_pivot.iloc[us_prev_idx] - 1).values

                # 既存のシグナルDFに1行追加
                if "PCA_SUB" in result.strategies and result.strategies["PCA_SUB"].signals is not None:
                    sig_df = result.strategies["PCA_SUB"].signals
                    # 最後の行のシグナル計算と同じロジックで新しいシグナルを生成
                    from .strategy import rolling_pca_signal, build_prior_subspace, estimate_prior_correlation, _get_sector_indices
                    us_cc_all = np.vstack([aligned.us_cc_returns.values, latest_us_ret.reshape(1, -1)])
                    # JP側はダミー (NaN) を追加
                    jp_dummy = np.full((1, len(aligned.jp_tickers)), np.nan)
                    jp_cc_all = np.vstack([aligned.jp_cc_returns.values, jp_dummy])

                    us_cyc, us_def, jp_cyc, jp_def = _get_sector_indices(aligned.us_tickers, aligned.jp_tickers, config)
                    V0 = build_prior_subspace(len(aligned.us_tickers), len(aligned.jp_tickers), us_cyc, us_def, jp_cyc, jp_def)

                    import pandas as _pd
                    prior_end_ts = _pd.Timestamp(config.prior_end_date)
                    extended_dates = aligned.common_dates.append(_pd.DatetimeIndex([latest_us_date]))
                    prior_mask = extended_dates <= prior_end_ts
                    if prior_mask.sum() < config.rolling_window:
                        prior_mask = np.ones(len(extended_dates), dtype=bool)
                    combined_prior = np.hstack([us_cc_all[prior_mask], jp_cc_all[prior_mask]])
                    # NaN行を除外
                    valid_rows = ~np.isnan(combined_prior).any(axis=1)
                    if valid_rows.sum() >= config.rolling_window:
                        C0 = estimate_prior_correlation(V0, combined_prior[valid_rows])
                        signals_ext, _ = rolling_pca_signal(
                            us_cc_all, jp_cc_all, C0,
                            lambda_reg=config.lambda_reg,
                            rolling_window=config.rolling_window,
                            n_components=config.n_components,
                        )
                        # 最後の行のシグナルを取得
                        new_sig = signals_ext[-1]
                        if not np.isnan(new_sig).all():
                            new_row = _pd.DataFrame(
                                [new_sig], index=[latest_us_date], columns=sig_df.columns
                            )
                            result.strategies["PCA_SUB"].signals = _pd.concat([sig_df, new_row])
                            # daily_returnsにもNaN行を追加 (実績なし)
                            new_ret = _pd.Series([np.nan], index=[latest_us_date], name="PCA_SUB")
                            result.strategies["PCA_SUB"].daily_returns = _pd.concat([
                                result.strategies["PCA_SUB"].daily_returns, new_ret
                            ])
                            # US returnsにも追加
                            new_us_row = _pd.DataFrame(
                                [latest_us_ret], index=[latest_us_date], columns=aligned.us_tickers
                            )
                            result.us_cc_returns = _pd.concat([result.us_cc_returns, new_us_row])
                            # JP close pricesに最終終値を複製 (指値計算用)
                            if result.jp_close_prices is not None:
                                last_jp_close = result.jp_close_prices.iloc[-1:]
                                last_jp_close.index = [latest_us_date]
                                result.jp_close_prices = _pd.concat([result.jp_close_prices, last_jp_close])
                            logger.info("最新USデータ (%s) からの追加シグナルを生成しました", latest_us_date.strftime("%Y-%m-%d"))
    except Exception as e:
        import traceback as _tb
        logger.warning("最新シグナル計算失敗: %s\n%s", e, _tb.format_exc())

    return result
