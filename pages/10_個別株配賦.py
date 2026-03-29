"""個別株配賦ページ — ETFシグナルの個別株への展開 (Phase 1-3, 5)

ETF-to-ETF Lead-Lagシグナルを個別株に配賦し、
フィルター型/ランキング型のL/Sポートフォリオを構築・評価する。
"""

import threading
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import JQUANTS_API_KEY, MARKET_DATA_DIR
from core.styles import apply_reuters_style

st.set_page_config(page_title="個別株配賦", page_icon="S", layout="wide")


@st.cache_resource
def _get_provider_and_cache():
    from data.cache import DataCache
    from data.jquants_provider import JQuantsProvider
    cache = DataCache(MARKET_DATA_DIR)
    provider = JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache) if JQUANTS_API_KEY else None
    return provider, cache


# ---------------------------------------------------------------------------
# バックグラウンドスレッド
# ---------------------------------------------------------------------------
def _run_allocation_thread(progress_dict, backtest_result, config, provider, cache):
    """個別株配賦をバックグラウンドで実行。"""
    try:
        from core.stock_allocation import run_stock_allocation

        def on_progress(msg, pct):
            progress_dict["message"] = msg
            progress_dict["pct"] = pct

        result = run_stock_allocation(
            backtest_result=backtest_result,
            config=config,
            jquants_provider=provider,
            cache=cache,
            progress_callback=on_progress,
        )
        progress_dict["_result"] = result
        progress_dict["pct"] = 1.0
        progress_dict["message"] = "完了"
    except Exception as e:
        import traceback
        progress_dict["error"] = str(e)
        progress_dict["detail"] = traceback.format_exc()
        progress_dict["pct"] = 1.0


def _run_residual_thread(progress_dict, backtest_result, stock_signals, provider, config):
    """残差分析をバックグラウンドで実行。"""
    try:
        from core.stock_allocation.residual_analysis import decompose_returns, evaluate_signal_on_residuals
        from core.stock_allocation.mapper import build_etf_to_sector_mapping

        def on_progress(msg, pct):
            progress_dict["message"] = msg
            progress_dict["pct"] = pct

        on_progress("データ準備中...", 0.05)

        # 銘柄→ETFマッピング
        listed = provider.get_listed_stocks()
        etf_to_sector = build_etf_to_sector_mapping(listed)
        stock_to_etf = {}
        for _, row in listed.iterrows():
            sec = row.get("sector_17_name", "")
            if pd.isna(sec):
                continue
            for etf_code, jq_sec in etf_to_sector.items():
                if jq_sec == sec:
                    stock_to_etf[row["code"]] = etf_code
                    break

        # 個別株リターン — Phase 1 と同じ月単位チャンク取得
        unique_codes = set(stock_signals["code"].unique().tolist()[:200])  # 計算量制限
        start = config.backtest_start or "2024-01-01"
        end = config.backtest_end or date.today().strftime("%Y-%m-%d")
        on_progress(f"株価取得中 ({len(unique_codes)}銘柄)...", 0.10)

        from core.stock_allocation import _fetch_stock_prices
        prices_df = _fetch_stock_prices(
            provider, list(unique_codes), start, end,
            progress_callback=lambda m, p: on_progress(m, 0.10 + p * 0.25),
        )

        if prices_df.empty:
            raise ValueError("株価取得に失敗")

        close_col = "adj_close" if "adj_close" in prices_df.columns else "close"
        stock_ret = prices_df.pivot(index="date", columns="code", values=close_col).pct_change()

        # 市場リターン
        market_ret = backtest_result.benchmark_returns
        if market_ret is None:
            market_ret = pd.Series(0, index=stock_ret.index)

        # セクターETFリターン
        jp_close = backtest_result.jp_close_prices
        sec_ret = jp_close.pct_change() if jp_close is not None else pd.DataFrame()

        on_progress("残差分析実行中...", 0.40)
        decomposed = decompose_returns(
            stock_returns=stock_ret,
            market_returns=market_ret,
            sector_returns=sec_ret,
            stock_sectors=stock_to_etf,
            rolling_window=60,
            progress_callback=lambda m, p: on_progress(m, 0.40 + p * 0.40),
        )

        on_progress("シグナル評価中...", 0.85)
        residual_result = evaluate_signal_on_residuals(stock_signals, decomposed)

        progress_dict["_result"] = residual_result
        progress_dict["pct"] = 1.0
        progress_dict["message"] = "完了"
    except Exception as e:
        import traceback
        progress_dict["error"] = str(e)
        progress_dict["detail"] = traceback.format_exc()
        progress_dict["pct"] = 1.0


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def _ensure_ll_result():
    """セッションに ll_result がなければ、最新の保存結果を自動読み込みする。"""
    if "ll_result" in st.session_state:
        return
    from pathlib import Path
    import pickle
    result_dir = Path("storage/leadlag_results")
    if not result_dir.exists():
        return
    saved = sorted(result_dir.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
    if saved:
        try:
            with open(saved[0], "rb") as f:
                st.session_state["ll_result"] = pickle.load(f)
        except Exception:
            pass


def main():
    apply_reuters_style()

    st.markdown("# 個別株配賦")
    st.caption("ETF-to-ETF Lead-Lagシグナルを個別株に展開し、L/Sポートフォリオを構築・評価")

    # 前回の保存結果を自動読み込み
    _ensure_ll_result()

    # Lead-Lag結果の確認
    ll_result = st.session_state.get("ll_result")

    # スレッド完了の検出
    for key_thread, key_result, key_progress in [
        ("alloc_thread", "alloc_result", "alloc_progress"),
        ("resid_thread", "resid_result", "resid_progress"),
    ]:
        thread = st.session_state.get(key_thread)
        if thread is not None and not thread.is_alive() and key_result not in st.session_state:
            prog = st.session_state.get(key_progress, {})
            if "_result" in prog:
                st.session_state[key_result] = prog.pop("_result")

    tab_alloc, tab_residual, tab_compare = st.tabs([
        "Phase 1: 配賦バックテスト",
        "Phase 3: 残差分析",
        "Phase 2: US11 vs US33",
    ])

    with tab_alloc:
        _render_allocation_tab(ll_result)

    with tab_residual:
        _render_residual_tab(ll_result)

    with tab_compare:
        _render_comparison_tab()


# ---------------------------------------------------------------------------
# Phase 1: 配賦バックテスト
# ---------------------------------------------------------------------------
def _render_allocation_tab(ll_result):
    st.subheader("Phase 1: ETFシグナル → 個別株 配賦バックテスト")

    if ll_result is None:
        st.info("先に「日米リードラグ」ページでバックテストを実行してください。結果がセッションに保存されます。")
        return

    if "PCA_SUB" not in ll_result.strategies:
        st.warning("PCA_SUB の結果がありません。")
        return

    provider, cache = _get_provider_and_cache()
    if provider is None:
        st.error("J-Quants API キーが設定されていません。")
        return

    # --- ETF版パラメータの表示 ---
    etf_config = ll_result.config
    pca_signals = ll_result.strategies["PCA_SUB"].signals
    sig_dates = pca_signals.dropna(how="all").index
    n_jp = len(pca_signals.columns)

    from core.stock_allocation.config import compute_n_sectors_from_q
    etf_q = etf_config.quantile_q if etf_config else 0.3
    auto_n_sectors = compute_n_sectors_from_q(etf_q, n_jp)

    st.markdown("#### ETF版パラメータ (自動連動)")
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("分位点 q", f"{etf_q}")
        st.metric("→ セクター数 L/S", f"{auto_n_sectors} / {auto_n_sectors}")
    with col_info2:
        st.metric("JPセクター数", f"{n_jp}")
        st.metric("シグナル期間", f"{sig_dates[0].strftime('%Y-%m-%d')} — {sig_dates[-1].strftime('%Y-%m-%d')}")
    with col_info3:
        st.metric("λ", f"{etf_config.lambda_reg}" if etf_config else "-")
        st.metric("K", f"{etf_config.n_components}" if etf_config else "-")

    # --- 個別株側の設定 ---
    st.markdown("#### 個別株側の設定")
    col1, col2 = st.columns(2)

    with col1:
        use_etf_q = st.checkbox("ETF版の q と連動", value=True)
        if not use_etf_q:
            n_sectors = st.number_input("ロング/ショート対象セクター数", 1, n_jp // 2, auto_n_sectors)
        else:
            n_sectors = auto_n_sectors
            st.caption(f"q={etf_q} → L/S各{n_sectors}セクター")

    with col2:
        market = st.multiselect(
            "市場", ["プライム", "スタンダード", "グロース"],
            default=["プライム", "スタンダード", "グロース"],
        )
        margin_only = st.checkbox("信用取引可能銘柄のみ", value=True)

    # バックテスト期間
    st.markdown("#### バックテスト期間")
    st.caption(f"シグナル範囲: {sig_dates[0].strftime('%Y-%m-%d')} — {sig_dates[-1].strftime('%Y-%m-%d')}")

    # デフォルト: 直近2年 or シグナル開始日のどちらか遅い方
    from datetime import timedelta
    default_start = max(sig_dates[0].date(), sig_dates[-1].date() - timedelta(days=730))

    col4, col5 = st.columns(2)
    with col4:
        bt_start_val = st.date_input(
            "開始", value=default_start,
            min_value=sig_dates[0].date(), max_value=sig_dates[-1].date(),
            key="alloc_start",
        )
    with col5:
        bt_end_val = st.date_input(
            "終了", value=sig_dates[-1].date(),
            min_value=sig_dates[0].date(), max_value=sig_dates[-1].date(),
            key="alloc_end",
        )

    n_months_est = max(1, (bt_end_val.year - bt_start_val.year) * 12 + bt_end_val.month - bt_start_val.month)
    st.caption(f"約{n_months_est}ヶ月分 — 初回は株価取得に約{max(1, n_months_est * 5 // 60)}分かかります（2回目以降はキャッシュで即座）")

    # --- 実行 ---
    if st.button("配賦バックテストを実行", type="primary", key="run_alloc"):
        from core.stock_allocation.config import StockAllocationConfig
        config = StockAllocationConfig(
            quantile_q=etf_q if use_etf_q else 0,
            n_sectors_long=n_sectors,
            n_sectors_short=n_sectors,
            market_segments=market,
            margin_tradable_only=margin_only,
            backtest_start=bt_start_val.strftime("%Y-%m-%d"),
            backtest_end=bt_end_val.strftime("%Y-%m-%d"),
        )

        progress = {"message": "開始...", "pct": 0.0}
        st.session_state["alloc_progress"] = progress
        st.session_state.pop("alloc_result", None)

        t = threading.Thread(
            target=_run_allocation_thread,
            args=(progress, ll_result, config, provider, cache),
            daemon=True,
        )
        st.session_state["alloc_thread"] = t
        t.start()
        st.rerun()

    # --- 進捗 ---
    alloc_thread = st.session_state.get("alloc_thread")
    if alloc_thread is not None and alloc_thread.is_alive():
        prog = st.session_state.get("alloc_progress", {})
        import time
        st.progress(prog.get("pct", 0.0))
        st.caption(prog.get("message", "実行中..."))
        if "error" in prog:
            st.error(prog["error"])
        time.sleep(1)
        st.rerun()

    # --- エラー表示 ---
    prog = st.session_state.get("alloc_progress", {})
    if "error" in prog:
        st.error(f"エラー: {prog['error']}")
        if "detail" in prog:
            with st.expander("詳細"):
                st.code(prog["detail"])
        return

    # --- 結果表示 ---
    alloc_result = st.session_state.get("alloc_result")
    if alloc_result is None:
        return

    st.markdown("---")
    st.markdown("#### 結果")

    metrics = alloc_result.metrics
    etf_metrics = alloc_result.etf_metrics

    # メトリクス比較テーブル
    if etf_metrics:
        compare_keys = ["AR", "RISK", "R/R", "MDD", "monthly_win_rate"]
        labels = {
            "AR": "年率リターン (%)",
            "RISK": "リスク (%)",
            "R/R": "シャープ比",
            "MDD": "最大下落率 (%)",
            "monthly_win_rate": "月次勝率 (%)",
        }
        rows = []
        for k in compare_keys:
            rows.append({
                "指標": labels.get(k, k),
                "ETF版": etf_metrics.get(k, "-"),
                "個別株版": metrics.get(k, "-"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    # 累積リターンチャート
    if alloc_result.daily_returns is not None:
        ret = alloc_result.daily_returns.dropna()
        cum = (1 + ret).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name="個別株L/S", line={"width": 2}))

        # ETF版を重ねる
        pca_sub = ll_result.strategies.get("PCA_SUB")
        if pca_sub is not None:
            etf_ret = pca_sub.daily_returns.dropna()
            common = cum.index.intersection(etf_ret.index)
            if len(common) > 0:
                etf_cum = (1 + etf_ret.loc[common]).cumprod()
                fig.add_trace(go.Scatter(x=etf_cum.index, y=etf_cum.values, name="ETF版", line={"dash": "dash"}))

        fig.update_layout(title="累積リターン比較", yaxis_title="累積リターン", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ポジション数
    if alloc_result.daily_long_count is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("平均ロング数", f"{alloc_result.daily_long_count.mean():.0f}")
        with col2:
            st.metric("平均ショート数", f"{alloc_result.daily_short_count.mean():.0f}")


# ---------------------------------------------------------------------------
# Phase 3: 残差分析
# ---------------------------------------------------------------------------
def _render_residual_tab(ll_result):
    st.subheader("Phase 3: 残差化によるアルファ分解")
    st.caption(
        "Lead-Lagシグナルが、市場・業種方向感を超えた個別株アルファを持つかを検証。\n"
        "3A: 市場+業種残差、3B: +サイズ+モメンタム残差"
    )

    if ll_result is None:
        st.info("先にLead-Lagバックテストを実行してください。")
        return

    pca_sub = ll_result.strategies.get("PCA_SUB")
    if pca_sub is None or pca_sub.signals is None:
        st.warning("PCA_SUBの結果がありません。")
        return

    provider, cache = _get_provider_and_cache()
    if provider is None:
        st.error("J-Quants API キーが設定されていません。")
        return

    col1, col2 = st.columns(2)
    with col1:
        res_start = st.date_input("分析開始", value=date(2024, 1, 1), key="resid_start")
    with col2:
        res_end = st.date_input("分析終了", value=date.today(), key="resid_end")

    if st.button("残差分析を実行", type="primary", key="run_residual"):
        from core.stock_allocation.mapper import assign_sector_signals
        listed = provider.get_listed_stocks()

        from core.universe_filter import UniverseFilterConfig, apply_universe_filter
        filtered = apply_universe_filter(listed, UniverseFilterConfig(
            market_segments=["プライム"],
            margin_tradable_only=True,
            exclude_etf_reit=True,
        ))
        stock_codes = filtered["code"].tolist()

        stock_signals = assign_sector_signals(pca_sub.signals, listed, stock_codes)

        from core.stock_allocation.config import StockAllocationConfig
        config = StockAllocationConfig(
            backtest_start=res_start.strftime("%Y-%m-%d"),
            backtest_end=res_end.strftime("%Y-%m-%d"),
        )

        progress = {"message": "開始...", "pct": 0.0}
        st.session_state["resid_progress"] = progress
        st.session_state.pop("resid_result", None)

        t = threading.Thread(
            target=_run_residual_thread,
            args=(progress, ll_result, stock_signals, provider, config),
            daemon=True,
        )
        st.session_state["resid_thread"] = t
        t.start()
        st.rerun()

    # --- 進捗 ---
    resid_thread = st.session_state.get("resid_thread")
    if resid_thread is not None and resid_thread.is_alive():
        prog = st.session_state.get("resid_progress", {})
        import time
        st.progress(prog.get("pct", 0.0))
        st.caption(prog.get("message", "実行中..."))
        time.sleep(1)
        st.rerun()

    prog = st.session_state.get("resid_progress", {})
    if "error" in prog:
        st.error(f"エラー: {prog['error']}")
        if "detail" in prog:
            with st.expander("詳細"):
                st.code(prog["detail"])
        return

    # --- 結果表示 ---
    resid_result = st.session_state.get("resid_result")
    if resid_result is None:
        return

    st.markdown("---")
    st.markdown("#### 予測力比較")

    # メトリクスカード
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**生リターン**")
        st.metric("IC", f"{resid_result.raw_ic:+.4f}")
        st.metric("ICIR", f"{resid_result.raw_icir:+.3f}")
        st.metric("ヒット率", f"{resid_result.raw_hit_rate:.1%}")
    with col2:
        st.markdown("**3A: 市場+業種残差**")
        st.metric("IC", f"{resid_result.phase3a_ic:+.4f}")
        st.metric("ICIR", f"{resid_result.phase3a_icir:+.3f}")
        st.metric("ヒット率", f"{resid_result.phase3a_hit_rate:.1%}")
    with col3:
        st.markdown("**3B: +サイズ+モメンタム残差**")
        st.metric("IC", f"{resid_result.phase3b_ic:+.4f}")
        st.metric("ICIR", f"{resid_result.phase3b_icir:+.3f}")
        st.metric("ヒット率", f"{resid_result.phase3b_hit_rate:.1%}")

    # 解釈
    st.markdown("---")
    st.markdown("#### 判定")
    if not np.isnan(resid_result.phase3a_ic) and abs(resid_result.phase3a_ic) > 0.01:
        if not np.isnan(resid_result.phase3b_ic) and abs(resid_result.phase3b_ic) > 0.005:
            st.success(
                "3B残差でもIC > 0 → Lead-Lagシグナルは市場・業種・スタイルを超えた個別株アルファを含む可能性あり。"
                "Phase 2, 5 への展開に意義がある。"
            )
        else:
            st.warning(
                "3A残差ではIC > 0 だが、3B残差で消失 → シグナルの予測力はサイズ/モメンタムファクターで説明可能。"
                "個別株化の追加価値は限定的。"
            )
    else:
        st.error(
            "3A残差でIC ≈ 0 → シグナルの予測力は業種方向感のみ。"
            "ETF版で十分で、個別株への追加配賦は必要ない。"
        )

    # 因子分解の説明力
    st.markdown("#### 因子分解の説明力")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("市場要因 (R²)", f"{resid_result.r_squared_market:.1%}")
    with col2:
        st.metric("業種要因 (R²)", f"{resid_result.r_squared_sector:.1%}")

    # 日次IC推移
    if resid_result.daily_ic_raw is not None:
        st.markdown("#### 日次IC推移")
        fig = go.Figure()
        for series, name in [
            (resid_result.daily_ic_raw, "生リターン"),
            (resid_result.daily_ic_3a, "3A残差"),
            (resid_result.daily_ic_3b, "3B残差"),
        ]:
            if series is not None and len(series) > 0:
                rolling = series.rolling(20, min_periods=5).mean()
                fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values, name=name))
        fig.update_layout(title="日次IC (20日移動平均)", yaxis_title="IC", height=350)
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Phase 2: US11 vs US33
# ---------------------------------------------------------------------------
def _render_comparison_tab():
    st.subheader("Phase 2: US 11セクター vs US 33種 比較")
    st.caption(
        "US側ユニバースを11セクターから33種に拡張した場合の改善度を検証。"
        "「日米リードラグ」ページで両方のバックテスト結果が必要です。"
    )

    # セッションから両方の結果を探す
    ll_history = st.session_state.get("ll_history", [])
    if not ll_history:
        ll_result = st.session_state.get("ll_result")
        if ll_result:
            ll_history = [{"label": "current", "result": ll_result}]

    us11_results = []
    us33_results = []
    for h in ll_history:
        r = h.get("result")
        if r is None:
            continue
        cfg = r.config
        if cfg and cfg.us_universe == "US_11":
            us11_results.append(h)
        elif cfg and cfg.us_universe == "US_33":
            us33_results.append(h)

    if not us11_results or not us33_results:
        st.info(
            "US 11 と US 33 の両方のバックテスト結果が必要です。\n\n"
            "「日米リードラグ」ページの「設定・実行」タブで:\n"
            "1. USユニバース = US 11 で実行\n"
            "2. USユニバース = US 33 で実行\n\n"
            "両方の結果がセッションに保存されると、ここで比較できます。"
        )
        return

    from core.lead_lag.universe_comparison import compare_universes, build_comparison_table

    col1, col2 = st.columns(2)
    with col1:
        us11_idx = st.selectbox(
            "US 11 結果", range(len(us11_results)),
            format_func=lambda i: us11_results[i].get("label", f"#{i+1}"),
        )
    with col2:
        us33_idx = st.selectbox(
            "US 33 結果", range(len(us33_results)),
            format_func=lambda i: us33_results[i].get("label", f"#{i+1}"),
        )

    r11 = us11_results[us11_idx]["result"]
    r33 = us33_results[us33_idx]["result"]

    comp = compare_universes(r11, r33)
    df_comp = build_comparison_table(comp)

    st.markdown("#### メトリクス比較")
    st.dataframe(df_comp, hide_index=True)

    # シグナル相関
    if comp.signal_correlation:
        st.markdown("#### JPセクター別シグナル相関 (US11 vs US33)")
        from core.lead_lag.config import get_ticker_name

        corr_data = {get_ticker_name(k): v for k, v in comp.signal_correlation.items()}
        fig = go.Figure(data=go.Bar(
            x=list(corr_data.keys()),
            y=list(corr_data.values()),
            marker_color=["#2196F3" if v > 0.8 else "#FF9800" if v > 0.5 else "#F44336" for v in corr_data.values()],
        ))
        fig.update_layout(
            title="US11 vs US33: JPセクターシグナル相関",
            yaxis_title="相関係数",
            height=350,
            xaxis_tickangle=45,
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "相関が1.0に近い → US33の追加情報は少ない (ETF平均で十分)。"
            "相関が低い → US33が独自の情報を加えている可能性。"
        )


# ---------------------------------------------------------------------------
main()
