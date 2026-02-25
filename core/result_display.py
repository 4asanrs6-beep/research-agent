"""共有結果表示コンポーネント

研究ページと標準バックテストページで共通利用する結果表示。
- イベントスタディモード: 5タブ（分析結果/リターン分布/自動評価/パラメータ設定/直近事例）
- トレードモード（AI研究ページ）: 6タブ（概要/統計結果/バックテスト/自動評価/コード/直近事例）
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.styles import render_status_badge


def render_plan(plan: dict):
    """AI生成の分析計画を見やすく表示する（共通コンポーネント）"""

    hypothesis = plan.get("hypothesis", "")
    if hypothesis:
        st.markdown(f"> {hypothesis}")

    # 方法論
    methodology = plan.get("methodology", {})
    if methodology:
        st.markdown("#### 方法論")
        if methodology.get("approach"):
            st.markdown(f"**アプローチ:** {methodology['approach']}")
        steps = methodology.get("steps", [])
        if steps:
            for i, step in enumerate(steps, 1):
                st.markdown(f"{i}. {step}")
        tests = methodology.get("statistical_tests", [])
        metrics = methodology.get("metrics", [])
        if tests or metrics:
            tc1, tc2 = st.columns(2)
            if tests:
                tc1.markdown(f"**統計検定:** {', '.join(tests)}")
            if metrics:
                tc2.markdown(f"**評価指標:** {', '.join(metrics)}")

    # 対象・期間
    universe = plan.get("universe", {})
    period = plan.get("analysis_period", {})
    if universe or period:
        pc1, pc2 = st.columns(2)
        if universe:
            with pc1:
                st.markdown("#### ユニバース")
                st.markdown(f"**対象:** {universe.get('detail', 'N/A')}")
                if universe.get("reason"):
                    st.caption(universe["reason"])
        if period:
            with pc2:
                st.markdown("#### 分析期間")
                st.markdown(
                    f"**期間:** {period.get('start_date', '')} 〜 {period.get('end_date', '')}"
                )
                if period.get("reason"):
                    st.caption(period["reason"])

    # バックテスト
    backtest_cfg = plan.get("backtest", {})
    if backtest_cfg:
        st.markdown("#### バックテスト戦略")
        if backtest_cfg.get("strategy_description"):
            st.markdown(backtest_cfg["strategy_description"])
        bc1, bc2, bc3 = st.columns(3)
        if backtest_cfg.get("entry_rule"):
            bc1.markdown(f"**エントリー:** {backtest_cfg['entry_rule']}")
        if backtest_cfg.get("exit_rule"):
            bc2.markdown(f"**エグジット:** {backtest_cfg['exit_rule']}")
        if backtest_cfg.get("rebalance_frequency"):
            bc3.markdown(f"**リバランス:** {backtest_cfg['rebalance_frequency']}")

    # 期待される結果
    expected = plan.get("expected_outcome", "")
    if expected:
        st.markdown("#### 期待される結果")
        st.markdown(expected)


def plot_equity(backtest: dict) -> go.Figure | None:
    """エクイティカーブを描画する（トレードモード用）"""
    equity = backtest.get("equity_curve", [])
    if not equity:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[e["date"] for e in equity],
        y=[e["value"] for e in equity],
        name="Portfolio",
        line=dict(color="#FF8000", width=2),
    ))
    bench = backtest.get("benchmark_curve", [])
    if bench:
        fig.add_trace(go.Scatter(
            x=[b["date"] for b in bench],
            y=[b["value"] for b in bench],
            name="Benchmark",
            line=dict(color="#999", dash="dash"),
        ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Value",
        hovermode="x unified", height=380,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def _plot_return_distribution(backtest: dict) -> go.Figure | None:
    """リターン分布ヒストグラム（シグナル vs ベンチマーク）"""
    signal_returns = backtest.get("signal_returns", [])
    benchmark_returns = backtest.get("benchmark_returns", [])

    if not signal_returns:
        return None

    sig_arr = np.array(signal_returns) * 100  # %表示
    bench_arr = np.array(benchmark_returns) * 100 if benchmark_returns else None

    fig = go.Figure()

    # ベンチマーク分布（先に描画して背面に）
    if bench_arr is not None and len(bench_arr) > 0:
        fig.add_trace(go.Histogram(
            x=bench_arr,
            name="ベンチマーク (TOPIX)",
            marker_color="rgba(150,150,150,0.4)",
            nbinsx=50,
        ))

    # シグナル分布
    fig.add_trace(go.Histogram(
        x=sig_arr,
        name="シグナル銘柄",
        marker_color="rgba(255,128,0,0.6)",
        nbinsx=50,
    ))

    # 平均線
    sig_mean = float(np.mean(sig_arr))
    fig.add_vline(
        x=sig_mean, line_dash="dash", line_color="#FF8000", line_width=2,
        annotation_text=f"シグナル平均: {sig_mean:.2f}%",
        annotation_position="top right",
        annotation_font_color="#FF8000",
    )

    if bench_arr is not None and len(bench_arr) > 0:
        bench_mean = float(np.mean(bench_arr))
        fig.add_vline(
            x=bench_mean, line_dash="dash", line_color="#666", line_width=2,
            annotation_text=f"ベンチマーク平均: {bench_mean:.2f}%",
            annotation_position="top left",
            annotation_font_color="#666",
        )

    # 横軸レンジ: 外れ値を除外して見やすくする（1-99パーセンタイル）
    all_vals = sig_arr
    if bench_arr is not None and len(bench_arr) > 0:
        all_vals = np.concatenate([sig_arr, bench_arr])
    p1, p99 = float(np.percentile(all_vals, 1)), float(np.percentile(all_vals, 99))
    margin_x = (p99 - p1) * 0.1
    x_min = p1 - margin_x
    x_max = p99 + margin_x

    holding = backtest.get("holding_days", "N")
    fig.update_layout(
        xaxis_title=f"{holding}営業日リターン (%)",
        yaxis_title="度数",
        xaxis_range=[x_min, x_max],
        barmode="overlay",
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    return fig


def render_result_tabs(
    interpretation: dict,
    stats: dict,
    backtest: dict,
    code_or_config,
    recent_examples: list | None,
    code_tab_label: str = "コード",
    code_language: str = "python",
    pending_signals: list | None = None,
    key_prefix: str = "",
):
    """結果表示（イベントスタディモード / トレードモード自動切替）

    Args:
        interpretation: 評価結果 (label, confidence, summary, reasons, ...)
        stats: 統計結果 (p_value, cohens_d, ...)
        backtest: バックテスト結果
        code_or_config: コード文字列 or パラメータdict
        recent_examples: 直近事例リスト
        code_tab_label: コードタブのラベル（"コード" or "パラメータ設定"）
        code_language: コード表示言語（"python" or "json"）
        pending_signals: 測定期間未達の進行中シグナルリスト
        key_prefix: plotly_chart等の重複防止用プレフィックス
    """
    is_event_study = backtest.get("mean_return") is not None

    if is_event_study:
        _render_event_study_tabs(
            interpretation, stats, backtest,
            code_or_config, recent_examples,
            code_tab_label, code_language,
            pending_signals, key_prefix,
        )
    else:
        _render_trade_tabs(
            interpretation, stats, backtest,
            code_or_config, recent_examples,
            code_tab_label, code_language,
            key_prefix,
        )


# ======================================================================
# イベントスタディモード（5タブ）
# ======================================================================
def _render_event_study_tabs(
    interpretation, stats, backtest,
    code_or_config, recent_examples,
    code_tab_label, code_language,
    pending_signals=None, key_prefix="",
):
    tabs = st.tabs(["分析結果", "リターン分布", "自動評価", code_tab_label, "直近事例"])

    # --- 分析結果（概要 + 統計を統合） ---
    with tabs[0]:
        label = interpretation.get("evaluation_label", interpretation.get("label", "needs_review"))
        confidence = interpretation.get("confidence", 0)
        st.markdown(
            f"#### {render_status_badge(label)} &nbsp; 信頼度: {confidence:.0%}",
            unsafe_allow_html=True,
        )
        if interpretation.get("summary"):
            st.write(interpretation["summary"])

        st.markdown("---")

        # メイン指標（超過リターン中心）
        mean_excess = backtest.get("mean_excess_return", 0)
        mean_ret = backtest.get("mean_return", 0)
        mean_bench = backtest.get("mean_benchmark_return", 0)
        holding = backtest.get("holding_days", "?")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                f"平均超過リターン ({holding}日)",
                f"{mean_excess:.2%}",
                help="シグナル銘柄の平均リターン − ベンチマーク(TOPIX)の平均リターン",
            )
        with c2:
            st.metric(
                f"シグナル平均リターン ({holding}日)",
                f"{mean_ret:.2%}",
            )
        with c3:
            st.metric(
                f"ベンチマーク平均リターン ({holding}日)",
                f"{mean_bench:.2%}",
            )

        # 統計指標
        st.markdown("")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            n_valid = backtest.get("n_valid_signals", 0)
            st.metric("有効シグナル数", f"{n_valid:,}")
        with c2:
            wr = backtest.get("win_rate", 0)
            st.metric("勝率", f"{wr:.1%}")
        with c3:
            exc_wr = backtest.get("excess_win_rate", 0)
            st.metric("超過リターン勝率", f"{exc_wr:.1%}",
                      help="ベンチマークを上回った割合")
        with c4:
            pv = stats.get("p_value")
            st.metric("p値", f"{pv:.4f}" if isinstance(pv, (int, float)) else "N/A")
        with c5:
            cd = stats.get("cohens_d")
            st.metric("効果量 (d)", f"{cd:.3f}" if isinstance(cd, (int, float)) else "N/A")

        # 有意性判定
        is_sig = stats.get("is_significant", False)
        if is_sig:
            st.success("統計的に有意（p < 0.05）")
        else:
            st.warning("統計的に有意でない（p ≥ 0.05）")

        # 詳細統計（折りたたみ）
        with st.expander("詳細統計データ"):
            display_stats = {k: v for k, v in stats.items() if k != "recent_examples"}
            _JP_STAT_LABELS = {
                "test_name": "検定方法",
                "signal_mean": "シグナル平均リターン",
                "benchmark_mean": "ベンチマーク平均リターン",
                "excess_mean": "超過リターン平均",
                "signal_std": "シグナルリターン標準偏差",
                "excess_std": "超過リターン標準偏差",
                "t_statistic": "t統計量",
                "p_value": "p値",
                "cohens_d": "効果量 (Cohen's d)",
                "win_rate": "勝率",
                "excess_win_rate": "超過リターン勝率",
                "n_signals": "シグナル数",
                "n_excess": "超過リターン計算可能数",
                "is_significant": "統計的有意性",
            }
            rows = []
            for k, v in display_stats.items():
                label_jp = _JP_STAT_LABELS.get(k, k)
                if isinstance(v, bool):
                    v_str = "有意" if v else "非有意"
                elif isinstance(v, float):
                    v_str = f"{v:.6f}"
                else:
                    v_str = str(v)
                rows.append({"項目": label_jp, "値": v_str})
            if rows:
                st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # --- リターン分布 ---
    with tabs[1]:
        fig = _plot_return_distribution(backtest)
        if fig:
            chart_key = f"{key_prefix}_return_dist" if key_prefix else None
            st.plotly_chart(fig, width='stretch', key=chart_key)
        else:
            st.info("リターン分布データがありません")

        # シグナル結果ログ
        trade_log = backtest.get("trade_log", [])
        if trade_log:
            with st.expander(f"シグナル結果（直近{len(trade_log)}件）"):
                tl_df = pd.DataFrame(trade_log)
                _LOG_COL_MAP = {
                    "date": "シグナル日", "code": "銘柄コード", "name": "企業名",
                    "return_pct": "リターン(%)", "excess_pct": "超過リターン(%)",
                    "entry_price": "シグナル日終値", "exit_price": "測定日終値",
                    "holding_days": "測定期間(日)", "margin_ratio": "貸借倍率",
                }
                tl_df = tl_df.rename(columns={k: v for k, v in _LOG_COL_MAP.items() if k in tl_df.columns})
                tl_df = _reorder_columns(tl_df, [
                    "シグナル日", "銘柄コード", "企業名", "リターン(%)", "超過リターン(%)",
                    "シグナル日終値", "測定日終値", "測定期間(日)", "貸借倍率",
                ])
                st.dataframe(tl_df, width='stretch', hide_index=True)

    # --- 自動評価 ---
    with tabs[2]:
        _render_evaluation(interpretation)

    # --- パラメータ設定 / コード ---
    with tabs[3]:
        if isinstance(code_or_config, dict):
            _render_config_table(code_or_config)
        elif code_or_config:
            st.code(code_or_config, language=code_language, line_numbers=True)
        else:
            st.info("データがありません")

    # --- 直近事例 ---
    with tabs[4]:
        _render_recent_examples(recent_examples)
        if pending_signals:
            _render_pending_signals(pending_signals)


# ======================================================================
# トレードモード（6タブ — AI研究ページ互換）
# ======================================================================
def _render_trade_tabs(
    interpretation, stats, backtest,
    code_or_config, recent_examples,
    code_tab_label, code_language,
    key_prefix="",
):
    tabs = st.tabs(["概要", "統計結果", "バックテスト", "自動評価", code_tab_label, "直近事例"])

    # --- 概要 ---
    with tabs[0]:
        label = interpretation.get("evaluation_label", interpretation.get("label", "needs_review"))
        confidence = interpretation.get("confidence", 0)
        st.markdown(
            f"### {render_status_badge(label)} &nbsp; 信頼度: {confidence:.0%}",
            unsafe_allow_html=True,
        )
        if interpretation.get("summary"):
            st.write(interpretation["summary"])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sharpe = backtest.get("sharpe_ratio")
            st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe is not None else "N/A")
        with c2:
            cum = backtest.get("cumulative_return")
            st.metric("Cumulative Return", f"{cum:.1%}" if cum is not None else "N/A")
        with c3:
            pv = stats.get("p_value")
            st.metric("p-value", f"{pv:.4f}" if isinstance(pv, (int, float)) else "N/A")
        with c4:
            cd = stats.get("cohens_d")
            st.metric("Cohen's d", f"{cd:.3f}" if isinstance(cd, (int, float)) else "N/A")

    # --- 統計結果 ---
    with tabs[1]:
        if stats:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                pv = stats.get("p_value")
                st.metric("p-value", f"{pv:.4f}" if isinstance(pv, (int, float)) else "N/A")
            with c2:
                cd = stats.get("cohens_d")
                st.metric("Cohen's d", f"{cd:.3f}" if isinstance(cd, (int, float)) else "N/A")
            with c3:
                wr = stats.get("win_rate_condition")
                st.metric("Win Rate (Cond)", f"{wr:.1%}" if isinstance(wr, (int, float)) else "N/A")
            with c4:
                sig = stats.get("is_significant", False)
                st.metric("Significance", "Significant" if sig else "Not Significant")

            with st.expander("Raw JSON"):
                display_stats = {k: v for k, v in stats.items() if k != "recent_examples"}
                st.json(display_stats)
        else:
            st.info("統計結果がありません")

    # --- バックテスト ---
    with tabs[2]:
        if backtest:
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Cumulative Return", f"{backtest.get('cumulative_return', 0):.2%}")
            with c2:
                st.metric("Annual Return", f"{backtest.get('annual_return', 0):.2%}")
            with c3:
                st.metric("Sharpe Ratio", f"{backtest.get('sharpe_ratio', 0):.2f}")
            with c4:
                st.metric("Max Drawdown", f"{backtest.get('max_drawdown', 0):.2%}")
            with c5:
                st.metric("Trades", backtest.get("total_trades", 0))
            fig = plot_equity(backtest)
            if fig:
                chart_key = f"{key_prefix}_equity" if key_prefix else None
                st.plotly_chart(fig, width='stretch', key=chart_key)
            trade_log = backtest.get("trade_log", [])
            if trade_log:
                is_evt = "entry_price" in trade_log[0]
                log_label = "シグナル結果" if is_evt else "トレードログ"
                with st.expander(f"{log_label}（直近{len(trade_log)}件）"):
                    tl_df = pd.DataFrame(trade_log)
                    _LOG_COL_MAP = {
                        "date": "シグナル日", "code": "銘柄コード", "name": "企業名",
                        "entry_price": "エントリー価格", "exit_price": "決済価格",
                        "return_pct": "リターン(%)", "holding_days": "保有日数",
                        "action": "売買", "reason": "理由", "shares": "株数",
                        "price": "価格", "pnl": "損益",
                    }
                    tl_df = tl_df.rename(columns={k: v for k, v in _LOG_COL_MAP.items() if k in tl_df.columns})
                    st.dataframe(tl_df, width='stretch', hide_index=True)
        else:
            st.info("バックテスト結果がありません")

    # --- 自動評価 ---
    with tabs[3]:
        _render_evaluation(interpretation)

    # --- コード / パラメータ設定 ---
    with tabs[4]:
        if isinstance(code_or_config, dict):
            _render_config_table(code_or_config)
        elif code_or_config:
            st.code(code_or_config, language=code_language, line_numbers=True)
        else:
            st.info("データがありません")

    # --- 直近事例 ---
    with tabs[5]:
        _render_recent_examples(recent_examples)


# ======================================================================
# 共有サブコンポーネント
# ======================================================================
def _render_evaluation(interpretation: dict):
    """自動評価の表示"""
    if interpretation.get("reasons"):
        st.markdown("**判定理由:**")
        for r in interpretation["reasons"]:
            st.write(f"- {r}")
    for section, title in [("strengths", "強み"), ("weaknesses", "弱み"), ("suggestions", "改善提案")]:
        items = interpretation.get(section, [])
        if items:
            st.markdown(f"**{title}:**")
            for item in items:
                st.write(f"- {item}")


def _reorder_columns(df: pd.DataFrame, preferred_order: list[str]) -> pd.DataFrame:
    """DataFrameのカラムを指定順に並び替える（存在しないカラムはスキップ）"""
    ordered = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


def _render_recent_examples(recent_examples: list | None):
    """直近事例の表示"""
    if recent_examples:
        st.markdown("**測定完了済みのシグナル**")
        df = pd.DataFrame(recent_examples)
        if "return_pct" in df.columns:
            df["return_pct"] = df["return_pct"].apply(
                lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
            )
        _EXAMPLE_COL_MAP = {
            "signal_date": "シグナル日",
            "code": "銘柄コード",
            "name": "企業名",
            "return_pct": "リターン(%)",
            "entry_price": "シグナル日終値",
            "exit_price": "測定日終値",
            "holding_days": "測定期間(日)",
            "margin_ratio": "貸借倍率",
        }
        df = df.rename(columns={k: v for k, v in _EXAMPLE_COL_MAP.items() if k in df.columns})
        df = _reorder_columns(df, [
            "シグナル日", "銘柄コード", "企業名", "リターン(%)",
            "シグナル日終値", "測定日終値", "測定期間(日)", "貸借倍率",
        ])
        st.dataframe(df, width='stretch', hide_index=True)
    else:
        st.info("直近事例データがありません")


def _render_pending_signals(pending_signals: list):
    """進行中シグナル（測定期間未達）の表示"""
    if not pending_signals:
        return
    st.markdown("---")
    st.markdown(f"**進行中のシグナル（測定期間未達: {len(pending_signals)}件）**")
    st.caption("統計計算には含まれていません。現在進行中のため暫定リターンです。")
    df = pd.DataFrame(pending_signals)
    if "interim_return_pct" in df.columns:
        df["interim_return_pct"] = df["interim_return_pct"].apply(
            lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
        )
    _PENDING_COL_MAP = {
        "signal_date": "シグナル日",
        "code": "銘柄コード",
        "name": "企業名",
        "interim_return_pct": "暫定リターン(%)",
        "entry_price": "シグナル日終値",
        "latest_price": "直近終値",
        "elapsed_days": "経過日数",
        "remaining_days": "残り日数",
        "margin_ratio": "貸借倍率",
    }
    df = df.rename(columns={k: v for k, v in _PENDING_COL_MAP.items() if k in df.columns})
    df = _reorder_columns(df, [
        "シグナル日", "銘柄コード", "企業名", "暫定リターン(%)",
        "シグナル日終値", "直近終値", "経過日数", "残り日数", "貸借倍率",
    ])
    st.dataframe(df, width='stretch', hide_index=True)


def _render_config_table(config: dict):
    """パラメータ設定を表形式で表示"""
    import json as _json

    if not config:
        st.info("パラメータ設定がありません")
        return

    # 一括コピー用JSON
    with st.expander("設定を一括コピー（JSON）"):
        json_str = _json.dumps(config, ensure_ascii=False, indent=2, default=str)
        st.code(json_str, language="json")

    signal_cfg = config.get("signal_config", {})
    universe_cfg = config.get("universe_config", {})

    if signal_cfg:
        st.markdown("#### シグナル設定")
        rows = []
        _JP_LABELS = {
            "consecutive_bullish_days": "連続陽線日数",
            "consecutive_bearish_days": "連続陰線日数",
            "volume_surge_ratio": "出来高倍率閾値",
            "volume_surge_window": "出来高MA期間",
            "price_vs_ma25": "25日線との関係",
            "price_vs_ma75": "75日線との関係",
            "price_vs_ma200": "200日線との関係",
            "ma_deviation_pct": "移動平均乖離率(%)",
            "ma_deviation_window": "乖離率の基準MA",
            "rsi_window": "RSI期間",
            "rsi_lower": "RSI下限",
            "rsi_upper": "RSI上限",
            "bb_window": "ボリンジャーバンド幅",
            "bb_std": "BB標準偏差倍数",
            "bb_buy_below_lower": "BB下限タッチで買い",
            "ma_cross_short": "GC/DC 短期MA",
            "ma_cross_long": "GC/DC 長期MA",
            "ma_cross_type": "GC/DC方向",
            "macd_fast": "MACD 短期",
            "macd_slow": "MACD 長期",
            "macd_signal": "MACDシグナル",
            "atr_window": "ATR期間",
            "atr_max": "ATRフィルター",
            "ichimoku_cloud": "一目均衡表: 雲",
            "ichimoku_tenkan_above_kijun": "転換線>基準線",
            "sector_relative_strength_min": "セクター相対強度下限(%)",
            "sector_relative_lookback": "セクター相対強度期間",
            "margin_type": "信用取引データ種別",
            "margin_ratio_min": "貸借倍率下限",
            "margin_ratio_max": "貸借倍率上限",
            "short_selling_ratio_max": "空売り比率上限",
            "holding_period_days": "測定期間(日)",
            "signal_logic": "シグナル結合ロジック",
        }
        for k, v in signal_cfg.items():
            label = _JP_LABELS.get(k, k)
            rows.append({"パラメータ": label, "値": str(v)})
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    if universe_cfg:
        st.markdown("#### ユニバース設定")
        rows = []
        _UNI_LABELS = {
            "market_segments": "市場区分",
            "scale_categories": "TOPIX規模区分",
            "sector_filter_type": "業種フィルター",
            "selected_sectors": "選択業種",
            "margin_tradable_only": "貸借銘柄のみ",
            "market_cap_min": "時価総額下限(億円)",
            "market_cap_max": "時価総額上限(億円)",
            "per_min": "PER下限",
            "per_max": "PER上限",
            "pbr_min": "PBR下限",
            "pbr_max": "PBR上限",
        }
        for k, v in universe_cfg.items():
            label = _UNI_LABELS.get(k, k)
            rows.append({"パラメータ": label, "値": str(v)})
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # その他メタデータ
    meta_keys = [k for k in config if k not in ("signal_config", "universe_config")]
    if meta_keys:
        st.markdown("#### 実行設定")
        _META_LABELS = {
            "initial_capital": "初期資本(円)",
            "commission_rate": "売買手数料",
            "slippage_rate": "スリッページ",
            "start_date": "開始日",
            "end_date": "終了日",
            "max_stocks": "対象銘柄上限",
            "n_stocks_used": "実際の対象銘柄数",
            "n_signals": "シグナル数",
        }
        meta_rows = []
        for k in meta_keys:
            label = _META_LABELS.get(k, k)
            v = config[k]
            if k in ("commission_rate", "slippage_rate") and isinstance(v, (int, float)):
                v_str = f"{v:.2%}"
            else:
                v_str = str(v)
            meta_rows.append({"パラメータ": label, "値": v_str})
        st.dataframe(pd.DataFrame(meta_rows), width='stretch', hide_index=True)
