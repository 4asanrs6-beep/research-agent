"""分析結果閲覧・比較ページ"""

import json

import plotly.graph_objects as go
import streamlit as st

from config import DB_PATH
from db.database import Database

st.set_page_config(page_title="分析結果", page_icon="📊", layout="wide")


@st.cache_resource
def get_db():
    return Database(DB_PATH)


def plot_equity_curve(backtest: dict) -> go.Figure | None:
    """エクイティカーブをPlotlyで描画"""
    equity = backtest.get("equity_curve", [])
    if not equity:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[e["date"] for e in equity],
        y=[e["value"] for e in equity],
        name="ポートフォリオ",
        line=dict(color="#1f77b4"),
    ))

    bench = backtest.get("benchmark_curve", [])
    if bench:
        fig.add_trace(go.Scatter(
            x=[b["date"] for b in bench],
            y=[b["value"] for b in bench],
            name="ベンチマーク (TOPIX)",
            line=dict(color="#ff7f0e", dash="dash"),
        ))

    fig.update_layout(
        title="エクイティカーブ",
        xaxis_title="日付",
        yaxis_title="資産額 (円)",
        hovermode="x unified",
        height=400,
    )
    return fig


def main():
    st.title("📊 分析結果")

    db = get_db()
    runs = db.list_runs()

    if not runs:
        st.info("まだ分析結果がありません。「分析実行」または「AI研究」ページから分析を実行してください。")
        return

    # Run選択
    run_options = {}
    for r in runs:
        idea_snap = r.get("idea_snapshot", {})
        title = idea_snap.get("title", "不明") if isinstance(idea_snap, dict) else "不明"
        label_icon = {
            "valid": "✅", "invalid": "❌", "needs_review": "🔍",
        }.get(r.get("evaluation_label", ""), "⏳")
        status_icon = {
            "completed": "🟢", "running": "🔵", "failed": "🔴",
        }.get(r.get("status", ""), "⚪")
        key = f"{status_icon}{label_icon} Run #{r['id']} - {title}"
        run_options[key] = r

    selected_key = st.selectbox("分析結果を選択", list(run_options.keys()))
    run = run_options[selected_key]

    st.divider()

    # --- 基本情報 ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ステータス", run.get("status", "unknown"))
    with col2:
        st.metric("評価", run.get("evaluation_label", "---"))
    with col3:
        st.metric("実行日時", (run.get("started_at") or "")[:16])

    # --- アイデア・計画 ---
    with st.expander("アイデア・計画の詳細", expanded=False):
        idea_snap = run.get("idea_snapshot", {})
        plan_snap = run.get("plan_snapshot", {})

        if isinstance(idea_snap, dict):
            st.write(f"**アイデア**: {idea_snap.get('title', '不明')}")
            st.write(idea_snap.get("description", ""))
        if isinstance(plan_snap, dict):
            st.write(f"**分析手法**: {plan_snap.get('analysis_method', '不明')}")
            st.write(f"**期間**: {plan_snap.get('start_date', '')} ~ {plan_snap.get('end_date', '')}")
            st.json(plan_snap.get("parameters", {}))

    # --- 統計分析結果 ---
    stats = run.get("statistics_result")
    if stats and isinstance(stats, dict) and "error" not in stats:
        st.subheader("📐 統計分析結果")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("p値", f"{stats.get('p_value', 'N/A'):.4f}" if isinstance(stats.get('p_value'), (int, float)) else "N/A")
        with col2:
            st.metric("Cohen's d", f"{stats.get('cohens_d', 'N/A'):.3f}" if isinstance(stats.get('cohens_d'), (int, float)) else "N/A")
        with col3:
            st.metric("条件群勝率", f"{stats.get('win_rate_condition', 0):.1%}")
        with col4:
            st.metric("基準群勝率", f"{stats.get('win_rate_baseline', 0):.1%}")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("条件群平均", f"{stats.get('condition_mean', 0):.4%}")
        with col6:
            st.metric("基準群平均", f"{stats.get('baseline_mean', 0):.4%}")
        with col7:
            st.metric("条件群N", stats.get("n_condition", 0))
        with col8:
            sig = stats.get("is_significant", False)
            st.metric("有意性", "✅ 有意" if sig else "❌ 非有意")

        with st.expander("統計結果の詳細JSON"):
            st.json(stats)

    # --- バックテスト結果 ---
    bt = run.get("backtest_result")
    if bt and isinstance(bt, dict) and "error" not in bt:
        st.subheader("📈 バックテスト結果")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("累計リターン", f"{bt.get('cumulative_return', 0):.2%}")
        with col2:
            st.metric("年率リターン", f"{bt.get('annual_return', 0):.2%}")
        with col3:
            st.metric("シャープ比", f"{bt.get('sharpe_ratio', 0):.2f}")
        with col4:
            st.metric("最大DD", f"{bt.get('max_drawdown', 0):.2%}")
        with col5:
            st.metric("取引回数", bt.get("total_trades", 0))

        fig = plot_equity_curve(bt)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        bench_cols = st.columns(3)
        with bench_cols[0]:
            st.metric("BM累計リターン", f"{bt.get('benchmark_cumulative_return', 0):.2%}")
        with bench_cols[1]:
            st.metric("BM年率リターン", f"{bt.get('benchmark_annual_return', 0):.2%}")
        with bench_cols[2]:
            st.metric("BMシャープ比", f"{bt.get('benchmark_sharpe_ratio', 0):.2f}")

        with st.expander("取引ログ（直近20件）"):
            trade_log = bt.get("trade_log", [])
            if trade_log:
                st.dataframe(trade_log[-20:])

    # --- 評価 ---
    evaluation = run.get("evaluation")
    if evaluation and isinstance(evaluation, dict) and "error" not in evaluation:
        st.subheader("🔍 評価")

        label = evaluation.get("evaluation_label") or evaluation.get("label", "---")
        confidence = evaluation.get("confidence", 0)
        label_color = {"valid": "green", "invalid": "red", "needs_review": "orange"}.get(label, "gray")

        st.markdown(f"### 判定: :{label_color}[{label}] (信頼度: {confidence:.0%})")

        if evaluation.get("summary"):
            st.write(evaluation["summary"])

        reasons = evaluation.get("reasons", [])
        if reasons:
            st.write("**判断理由:**")
            for r in reasons:
                st.write(f"- {r}")

        # AI評価特有のフィールド
        for section, title in [("strengths", "強み"), ("weaknesses", "弱み"), ("suggestions", "提案")]:
            items = evaluation.get(section, [])
            if items:
                st.write(f"**{title}:**")
                for item in items:
                    st.write(f"- {item}")


if __name__ == "__main__":
    main()
