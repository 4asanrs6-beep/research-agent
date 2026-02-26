"""共有結果表示コンポーネント

研究ページと標準バックテストページで共通利用する結果表示。
- サマリーパネル（タブ外・常時表示）: 判定バッジ + キー指標 + パラメータ要約 + コピーボタン
- イベントスタディモード: 2タブ（分析詳細 / シグナル一覧）
- トレードモード（AI研究ページ）: 2タブ（分析詳細 / シグナル一覧）
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.styles import render_status_badge

# シグナルパラメータの日本語ラベルマッピング（共通定数）
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
    "margin_buy_change_pct_min": "買い残変化率 下限(%)",
    "margin_buy_change_pct_max": "買い残変化率 上限(%)",
    "margin_sell_change_pct_min": "売り残変化率 下限(%)",
    "margin_sell_change_pct_max": "売り残変化率 上限(%)",
    "margin_ratio_change_pct_min": "貸借倍率変化率 下限(%)",
    "margin_ratio_change_pct_max": "貸借倍率変化率 上限(%)",
    "margin_buy_turnover_days_min": "買い残回転日数 下限",
    "margin_buy_turnover_days_max": "買い残回転日数 上限",
    "margin_sell_turnover_days_min": "売り残回転日数 下限",
    "margin_sell_turnover_days_max": "売り残回転日数 上限",
    "margin_buy_vol_ratio_min": "買い残対出来高比率 下限",
    "margin_buy_vol_ratio_max": "買い残対出来高比率 上限",
    "margin_sell_vol_ratio_min": "売り残対出来高比率 下限",
    "margin_sell_vol_ratio_max": "売り残対出来高比率 上限",
    "margin_buy_vol_ratio_change_pct_min": "買い残対出来高比率変化率 下限(%)",
    "margin_buy_vol_ratio_change_pct_max": "買い残対出来高比率変化率 上限(%)",
    "margin_sell_vol_ratio_change_pct_min": "売り残対出来高比率変化率 下限(%)",
    "margin_sell_vol_ratio_change_pct_max": "売り残対出来高比率変化率 上限(%)",
    "holding_period_days": "測定期間(日)",
    "signal_logic": "シグナル結合ロジック",
}

# パラメータ要約用の短縮ラベル（1行サマリー向け）
_SHORT_LABELS = {
    "consecutive_bullish_days": "連続陽線{}日",
    "consecutive_bearish_days": "連続陰線{}日",
    "volume_surge_ratio": "出来高{}倍以上",
    "price_vs_ma25": "MA25: {}",
    "price_vs_ma75": "MA75: {}",
    "price_vs_ma200": "MA200: {}",
    "ma_deviation_pct": "乖離率{}%",
    "rsi_lower": "RSI≤{}",
    "rsi_upper": "RSI≥{}",
    "bb_buy_below_lower": "BB下限タッチ",
    "ma_cross_type": "{}",
    "ichimoku_cloud": "雲: {}",
    "margin_ratio_min": "貸借倍率≥{}",
    "margin_ratio_max": "貸借倍率≤{}",
    "short_selling_ratio_max": "空売り比率≤{}",
    "margin_buy_change_pct_min": "買い残変化≥{}%",
    "margin_sell_change_pct_min": "売り残変化≥{}%",
}


def _build_param_summary(config: dict) -> str:
    """アクティブなパラメータを1行サマリーにする"""
    signal_cfg = config.get("signal_config", {})
    parts = []

    # 主要パラメータを短縮表記で追加
    for key, fmt in _SHORT_LABELS.items():
        val = signal_cfg.get(key)
        if val is None or val == "" or val is False:
            continue
        if val is True:
            parts.append(fmt)  # bool型（例: BB下限タッチ）
        else:
            parts.append(fmt.format(val))

    # short_labelsに含まれない有効パラメータもフォールバック
    covered = set(_SHORT_LABELS.keys())
    skip = {"holding_period_days", "signal_logic", "rsi_window", "bb_window",
            "bb_std", "ma_cross_short", "ma_cross_long", "volume_surge_window",
            "ma_deviation_window", "macd_fast", "macd_slow", "macd_signal",
            "atr_window", "sector_relative_lookback", "margin_type"}
    for key, val in signal_cfg.items():
        if key in covered or key in skip:
            continue
        if val is None or val == "" or val is False:
            continue
        label = _JP_LABELS.get(key, key)
        parts.append(f"{label}: {val}")

    return " + ".join(parts) if parts else "カスタム設定"


def _build_copy_text(
    stats: dict, backtest: dict, interpretation: dict, config
) -> str:
    """結果全体をフォーマット済みテキストにする（一括コピー用）"""
    lines = []
    is_event_study = backtest.get("mean_return") is not None

    # --- 判定 ---
    label = interpretation.get("evaluation_label", interpretation.get("label", ""))
    confidence = interpretation.get("confidence", 0)
    lines.append("【バックテスト結果】")
    lines.append(f"判定: {label.upper()} (信頼度: {confidence:.0%})")

    if is_event_study:
        mean_excess = backtest.get("mean_excess_return", 0)
        mean_ret = backtest.get("mean_return", 0)
        mean_bench = backtest.get("mean_benchmark_return", 0)
        holding = backtest.get("holding_days", "?")
        pv = stats.get("p_value")
        cd = stats.get("cohens_d")
        n_valid = backtest.get("n_valid_signals", 0)
        wr = backtest.get("win_rate", 0)
        exc_wr = backtest.get("excess_win_rate", 0)
        lines.append(f"超過リターン: {mean_excess:+.2%} ({holding}日)")
        lines.append(f"シグナル平均: {mean_ret:+.2%} / ベンチマーク平均: {mean_bench:+.2%}")
        lines.append(
            f"p値: {pv:.4f} / 効果量d: {cd:+.3f}"
            if isinstance(pv, (int, float)) and isinstance(cd, (int, float))
            else f"p値: {pv} / 効果量d: {cd}"
        )
        lines.append(f"シグナル: {n_valid:,}件 / 勝率: {wr:.1%} / 超過勝率: {exc_wr:.1%}")
    else:
        sharpe = backtest.get("sharpe_ratio")
        cum = backtest.get("cumulative_return")
        mdd = backtest.get("max_drawdown")
        trades = backtest.get("total_trades", 0)
        lines.append(f"Sharpe: {sharpe:.2f}" if sharpe is not None else "Sharpe: N/A")
        lines.append(f"累積リターン: {cum:.1%}" if cum is not None else "累積リターン: N/A")
        lines.append(f"最大DD: {mdd:.1%}" if mdd is not None else "最大DD: N/A")
        lines.append(f"トレード数: {trades}")

    # --- パラメータ ---
    if isinstance(config, dict) and config:
        lines.append("")
        lines.append("【パラメータ】")
        signal_cfg = config.get("signal_config", {})
        for k, v in signal_cfg.items():
            if v is None or v == "" or v is False:
                continue
            label_jp = _JP_LABELS.get(k, k)
            lines.append(f"{label_jp}: {v}")

        # 測定期間・ロジック
        holding = signal_cfg.get("holding_period_days")
        logic = signal_cfg.get("signal_logic")
        if holding:
            lines.append(f"測定期間: {holding}日")
        if logic:
            lines.append(f"ロジック: {logic}")

        # ユニバース
        universe_cfg = config.get("universe_config", {})
        if universe_cfg:
            lines.append("")
            lines.append("【ユニバース】")
            for k, v in universe_cfg.items():
                if v is None or v == "" or v is False:
                    continue
                lines.append(f"{k}: {v}")

        start = config.get("start_date", "")
        end = config.get("end_date", "")
        if start or end:
            lines.append(f"期間: {start} ~ {end}")
        n_stocks = config.get("n_stocks_used")
        if n_stocks:
            lines.append(f"対象銘柄数: {n_stocks:,}社")

    return "\n".join(lines)


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
    """リターン分布（超過リターン + シグナル vs ベンチマーク比較）"""
    from plotly.subplots import make_subplots

    signal_returns = backtest.get("signal_returns", [])
    benchmark_returns = backtest.get("benchmark_returns", [])

    if not signal_returns:
        return None

    sig_arr = np.array(signal_returns) * 100  # %表示
    bench_arr = np.array(benchmark_returns) * 100 if benchmark_returns else None
    holding = backtest.get("holding_days", "N")

    has_bench = bench_arr is not None and len(bench_arr) > 0

    if has_bench:
        # 超過リターン分布を計算
        excess_arr = sig_arr - bench_arr[:len(sig_arr)]

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.55, 0.45],
            vertical_spacing=0.12,
            subplot_titles=[
                f"超過リターン分布（シグナル − ベンチマーク, {holding}日）",
                f"リターン比較（シグナル vs ベンチマーク, {holding}日）",
            ],
        )

        # --- 上段: 超過リターン分布 ---
        excess_mean = float(np.mean(excess_arr))
        # ゼロを境にプラス/マイナスで色分け
        fig.add_trace(go.Histogram(
            x=excess_arr,
            name="超過リターン",
            marker_color=[
                "rgba(46,125,50,0.6)" if v >= 0 else "rgba(198,40,40,0.5)"
                for v in np.sort(excess_arr)
            ] if len(excess_arr) <= 500 else "rgba(255,128,0,0.6)",
            nbinsx=40,
            showlegend=False,
        ), row=1, col=1)

        # ゼロライン
        fig.add_vline(
            x=0, line_color="#999", line_width=1, line_dash="dot",
            row=1, col=1,
        )
        # 平均線
        fig.add_vline(
            x=excess_mean, line_color="#FF8000", line_width=2, line_dash="dash",
            annotation_text=f"平均: {excess_mean:+.2f}%",
            annotation_position="top right",
            annotation_font=dict(color="#FF8000", size=12),
            row=1, col=1,
        )

        # 横軸レンジ（超過リターン）
        ep2, ep98 = float(np.percentile(excess_arr, 2)), float(np.percentile(excess_arr, 98))
        em = (ep98 - ep2) * 0.15
        fig.update_xaxes(range=[ep2 - em, ep98 + em], title_text="超過リターン (%)", row=1, col=1)
        fig.update_yaxes(title_text="度数", row=1, col=1)

        # --- 下段: シグナル vs ベンチマーク 比較 ---
        # 共通ビン幅を計算
        all_vals = np.concatenate([sig_arr, bench_arr])
        p2, p98 = float(np.percentile(all_vals, 2)), float(np.percentile(all_vals, 98))
        bin_width = (p98 - p2) / 35

        fig.add_trace(go.Histogram(
            x=bench_arr,
            name="ベンチマーク (TOPIX)",
            marker_color="rgba(150,150,150,0.45)",
            xbins=dict(size=bin_width),
        ), row=2, col=1)

        fig.add_trace(go.Histogram(
            x=sig_arr,
            name="シグナル銘柄",
            marker_color="rgba(255,128,0,0.55)",
            xbins=dict(size=bin_width),
        ), row=2, col=1)

        # 平均線（下段）
        sig_mean = float(np.mean(sig_arr))
        bench_mean = float(np.mean(bench_arr))
        fig.add_vline(
            x=sig_mean, line_color="#FF8000", line_width=1.5, line_dash="dash",
            annotation_text=f"シグナル: {sig_mean:.2f}%",
            annotation_position="top right",
            annotation_font=dict(color="#FF8000", size=11),
            row=2, col=1,
        )
        fig.add_vline(
            x=bench_mean, line_color="#666", line_width=1.5, line_dash="dash",
            annotation_text=f"BM: {bench_mean:.2f}%",
            annotation_position="top left",
            annotation_font=dict(color="#666", size=11),
            row=2, col=1,
        )

        vm = (p98 - p2) * 0.1
        fig.update_xaxes(range=[p2 - vm, p98 + vm], title_text=f"{holding}日リターン (%)", row=2, col=1)
        fig.update_yaxes(title_text="度数", row=2, col=1)

        fig.update_layout(
            barmode="overlay",
            height=620,
            margin=dict(l=50, r=20, t=40, b=40),
            legend=dict(
                yanchor="top", y=0.42, xanchor="right", x=0.98,
                bgcolor="rgba(255,255,255,0.85)",
            ),
        )
    else:
        # ベンチマークなし: シグナル分布のみ
        fig = go.Figure()
        sig_mean = float(np.mean(sig_arr))

        fig.add_trace(go.Histogram(
            x=sig_arr,
            name="シグナル銘柄",
            marker_color="rgba(255,128,0,0.6)",
            nbinsx=40,
        ))
        fig.add_vline(
            x=sig_mean, line_dash="dash", line_color="#FF8000", line_width=2,
            annotation_text=f"平均: {sig_mean:.2f}%",
            annotation_position="top right",
            annotation_font=dict(color="#FF8000", size=12),
        )

        p2, p98 = float(np.percentile(sig_arr, 2)), float(np.percentile(sig_arr, 98))
        vm = (p98 - p2) * 0.1
        fig.update_layout(
            xaxis_title=f"{holding}営業日リターン (%)",
            yaxis_title="度数",
            xaxis_range=[p2 - vm, p98 + vm],
            height=350,
            margin=dict(l=50, r=20, t=30, b=40),
        )

    return fig


def _metric_card(label: str, value: str, accent: bool = False) -> str:
    """1つのメトリクスカードHTML"""
    color = "var(--r-accent)" if accent else "var(--r-text)"
    return (
        f'<div style="background:var(--r-gray);border-radius:6px;'
        f'padding:0.6rem 0.7rem;text-align:center;min-width:0;">'
        f'<div style="font-size:1.35rem;font-weight:700;color:{color};'
        f'line-height:1.3;white-space:nowrap;">{value}</div>'
        f'<div style="font-size:0.75rem;color:#666;margin-top:2px;'
        f'white-space:nowrap;">{label}</div>'
        f'</div>'
    )


def _metrics_row(metrics: list[tuple[str, str, bool]]) -> str:
    """メトリクスカード群を横並びHTMLにする。各tupleは (label, value, accent)。"""
    cards = "".join(_metric_card(l, v, a) for l, v, a in metrics)
    return (
        f'<div style="display:grid;grid-template-columns:repeat({len(metrics)},1fr);'
        f'gap:0.5rem;margin:0.6rem 0;">'
        f'{cards}</div>'
    )


def _render_summary_panel(
    stats: dict, backtest: dict, interpretation: dict, code_or_config,
):
    """タブ外に常時表示されるサマリーパネル"""
    is_event_study = backtest.get("mean_return") is not None

    # --- ステータスバッジ + 信頼度 ---
    label = interpretation.get("evaluation_label", interpretation.get("label", "needs_review"))
    confidence = interpretation.get("confidence", 0)
    st.markdown(
        f"#### {render_status_badge(label)} &nbsp; 信頼度: {confidence:.0%}",
        unsafe_allow_html=True,
    )
    if interpretation.get("summary"):
        st.markdown(
            f'<p style="font-size:0.9rem;color:#444;line-height:1.6;margin:0.3rem 0 0.2rem;">'
            f'{interpretation["summary"]}</p>',
            unsafe_allow_html=True,
        )

    # --- キー指標（カスタムHTMLグリッド — 数値が途切れない） ---
    if is_event_study:
        mean_excess = backtest.get("mean_excess_return", 0)
        mean_ret = backtest.get("mean_return", 0)
        mean_bench = backtest.get("mean_benchmark_return", 0)
        holding = backtest.get("holding_days", "?")
        pv = stats.get("p_value")
        cd = stats.get("cohens_d")
        n_valid = backtest.get("n_valid_signals", 0)
        wr = backtest.get("win_rate", 0)
        exc_wr = backtest.get("excess_win_rate", 0)

        # 上段: リターン3指標
        return_metrics = [
            (f"超過リターン ({holding}日)", f"{mean_excess:+.2%}", True),
            (f"シグナル平均 ({holding}日)", f"{mean_ret:+.2%}", False),
            (f"ベンチマーク平均 ({holding}日)", f"{mean_bench:+.2%}", False),
        ]
        st.markdown(_metrics_row(return_metrics), unsafe_allow_html=True)

        # 下段: 統計指標
        metrics = [
            ("p値", f"{pv:.4f}" if isinstance(pv, (int, float)) else "N/A", False),
            ("効果量 d", f"{cd:+.3f}" if isinstance(cd, (int, float)) else "N/A", False),
            ("シグナル数", f"{n_valid:,}件", False),
            ("勝率", f"{wr:.1%}", False),
            ("超過勝率", f"{exc_wr:.1%}", False),
        ]
    else:
        sharpe = backtest.get("sharpe_ratio")
        cum = backtest.get("cumulative_return")
        mdd = backtest.get("max_drawdown")
        trades = backtest.get("total_trades", 0)
        pv = stats.get("p_value")

        metrics = [
            ("Sharpe", f"{sharpe:.2f}" if sharpe is not None else "N/A", True),
            ("累積リターン", f"{cum:.1%}" if cum is not None else "N/A", False),
            ("最大DD", f"{mdd:.1%}" if mdd is not None else "N/A", False),
            ("トレード数", f"{trades:,}", False),
            ("p値", f"{pv:.4f}" if isinstance(pv, (int, float)) else "N/A", False),
        ]

    st.markdown(_metrics_row(metrics), unsafe_allow_html=True)

    # --- パラメータ要約 + 期間（タグ風表示） ---
    if isinstance(code_or_config, dict) and code_or_config:
        param_summary = _build_param_summary(code_or_config)
        start = code_or_config.get("start_date", "")
        end = code_or_config.get("end_date", "")
        holding = (code_or_config.get("signal_config", {}).get("holding_period_days")
                   or backtest.get("holding_days", ""))
        n_stocks = code_or_config.get("n_stocks_used", "")

        meta_parts = []
        if start or end:
            meta_parts.append(f"{start} ~ {end}")
        if holding:
            meta_parts.append(f"測定 {holding}日")
        if n_stocks:
            meta_parts.append(f"対象 {n_stocks:,}社")

        # パラメータ部品をタグ風に分割表示
        param_tags = param_summary.split(" + ")
        tags_html = "".join(
            f'<span style="display:inline-block;background:#F0F0F0;border:1px solid #DDD;'
            f'border-radius:3px;padding:1px 8px;margin:2px 3px;font-size:0.78rem;'
            f'color:#333;white-space:nowrap;">{t.strip()}</span>'
            for t in param_tags
        )
        meta_html = ""
        if meta_parts:
            meta_spans = " / ".join(meta_parts)
            meta_html = (
                f'<div style="font-size:0.78rem;color:#888;margin-top:4px;">'
                f'{meta_spans}</div>'
            )

        st.markdown(
            f'<div style="margin:0.3rem 0 0.5rem;">'
            f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:0;">'
            f'<span style="font-size:0.78rem;color:#888;margin-right:4px;'
            f'white-space:nowrap;font-weight:600;">パラメータ:</span>'
            f'{tags_html}</div>'
            f'{meta_html}</div>',
            unsafe_allow_html=True,
        )

    # --- 一括コピー ---
    with st.expander("結果を一括コピー"):
        copy_text = _build_copy_text(stats, backtest, interpretation, code_or_config)
        st.code(copy_text, language="text")

    st.markdown("---")


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
        backtest: バックテスト結果 (trade_log含む)
        code_or_config: コード文字列 or パラメータdict
        recent_examples: (後方互換用・未使用)
        code_tab_label: コードタブのラベル（"コード" or "パラメータ設定"）
        code_language: コード表示言語（"python" or "json"）
        pending_signals: 測定期間未達のシグナルリスト
        key_prefix: plotly_chart等の重複防止用プレフィックス
    """
    is_event_study = backtest.get("mean_return") is not None

    # サマリーパネル（タブ外・常時表示）
    _render_summary_panel(stats, backtest, interpretation, code_or_config)

    # 簡素化された2タブ
    if is_event_study:
        _render_event_study_tabs(
            interpretation, stats, backtest,
            code_or_config, code_tab_label, code_language,
            pending_signals, key_prefix,
        )
    else:
        _render_trade_tabs(
            interpretation, stats, backtest,
            code_or_config, code_tab_label, code_language,
            key_prefix,
        )


# ======================================================================
# イベントスタディモード（2タブ）
# ======================================================================
def _render_event_study_tabs(
    interpretation, stats, backtest,
    code_or_config, code_tab_label, code_language,
    pending_signals=None, key_prefix="",
):
    tabs = st.tabs(["分析詳細", "シグナル一覧"])

    # --- Tab 1: 分析詳細 ---
    with tabs[0]:
        # リターン分布ヒストグラム（常時展開）
        fig = _plot_return_distribution(backtest)
        if fig:
            chart_key = f"{key_prefix}_return_dist" if key_prefix else None
            st.plotly_chart(fig, width='stretch', key=chart_key)

        # AI自動評価（expander）
        with st.expander("AI自動評価"):
            _render_evaluation(interpretation)

        # 詳細統計データ（expander）
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

        # パラメータ詳細（expander）
        with st.expander("パラメータ設定（全項目）"):
            if isinstance(code_or_config, dict):
                _render_config_table(code_or_config)
            elif code_or_config:
                st.code(code_or_config, language=code_language, line_numbers=True)
            else:
                st.info("データがありません")

    # --- Tab 2: シグナル一覧 ---
    with tabs[1]:
        trade_log = backtest.get("trade_log", [])
        has_completed = bool(trade_log)
        has_pending = bool(pending_signals)

        if not has_completed and not has_pending:
            st.info("シグナルデータがありません")

        # 測定完了シグナル
        if has_completed:
            st.markdown(f"**測定完了（{len(trade_log)}件）**")
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

        # 測定期間未達シグナル
        # pending_signals が None → 履歴表示（データなし）、[] → 実行直後で該当なし
        if has_pending:
            if has_completed:
                st.markdown("---")
            _render_pending_signals(pending_signals)
        elif pending_signals is not None and not has_pending:
            st.markdown("---")
            st.info("測定期間未達のシグナルはありません")


# ======================================================================
# トレードモード（2タブ）
# ======================================================================
def _render_trade_tabs(
    interpretation, stats, backtest,
    code_or_config, code_tab_label, code_language,
    key_prefix="",
):
    tabs = st.tabs(["分析詳細", "シグナル一覧"])

    # --- Tab 1: 分析詳細 ---
    with tabs[0]:
        # エクイティカーブ（常時展開）
        if backtest:
            fig = plot_equity(backtest)
            if fig:
                chart_key = f"{key_prefix}_equity" if key_prefix else None
                st.plotly_chart(fig, width='stretch', key=chart_key)

            # 統計詳細メトリクス
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("累積リターン", f"{backtest.get('cumulative_return', 0):.2%}")
            with c2:
                st.metric("年率リターン", f"{backtest.get('annual_return', 0):.2%}")
            with c3:
                st.metric("Sharpe", f"{backtest.get('sharpe_ratio', 0):.2f}")
            with c4:
                st.metric("最大DD", f"{backtest.get('max_drawdown', 0):.2%}")
            with c5:
                st.metric("トレード数", backtest.get("total_trades", 0))

        # AI自動評価（expander）
        with st.expander("AI自動評価"):
            _render_evaluation(interpretation)

        # 統計詳細（expander）
        if stats:
            with st.expander("詳細統計データ"):
                display_stats = {k: v for k, v in stats.items() if k != "recent_examples"}
                st.json(display_stats)

        # パラメータ詳細（expander）
        with st.expander("パラメータ設定（全項目）"):
            if isinstance(code_or_config, dict):
                _render_config_table(code_or_config)
            elif code_or_config:
                st.code(code_or_config, language=code_language, line_numbers=True)
            else:
                st.info("データがありません")

    # --- Tab 2: シグナル一覧 ---
    with tabs[1]:
        trade_log = backtest.get("trade_log", []) if backtest else []
        if trade_log:
            is_evt = "entry_price" in trade_log[0]
            log_label = "測定完了" if is_evt else "トレードログ"
            st.markdown(f"**{log_label}（{len(trade_log)}件）**")
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
            st.info("シグナルデータがありません")


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
