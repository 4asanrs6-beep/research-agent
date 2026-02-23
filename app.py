"""研究エージェント — Reuters風ダッシュボード"""

import streamlit as st

from config import DB_PATH, MARKET_DATA_DIR, JQUANTS_API_KEY
from db.database import Database
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider
from core.styles import apply_reuters_style, render_status_badge, render_card

st.set_page_config(
    page_title="研究エージェント",
    page_icon="R",
    layout="wide",
)


@st.cache_resource
def get_database() -> Database:
    return Database(DB_PATH)


@st.cache_resource
def get_data_provider():
    cache = DataCache(MARKET_DATA_DIR)
    return JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)


def main():
    apply_reuters_style()

    # --- サイドバー: 実行中インジケーター ---
    thread = st.session_state.get("research_thread")
    if thread and thread.is_alive():
        progress = st.session_state.get("research_progress", {})
        st.sidebar.markdown(
            '<div class="sidebar-running">'
            '<span class="pulse"></span> 研究を実行中...<br>'
            f'<small>{progress.get("message", "")}</small></div>',
            unsafe_allow_html=True,
        )

    # --- ヘッダー ---
    st.markdown("# Research Agent")
    st.caption("日本株データを用いた投資仮説の研究・検証プラットフォーム")

    db = get_database()
    runs = db.list_runs()
    knowledge_list = db.list_knowledge()
    completed_runs = [r for r in runs if r["status"] == "completed"]
    valid_k = [k for k in knowledge_list if k["validity"] == "valid"]

    provider = get_data_provider()
    api_ok = provider.is_available()

    # --- 3メトリクス ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="reuters-metric">'
            f'<div class="metric-value">{len(completed_runs)}</div>'
            '<div class="metric-label">完了した分析</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="reuters-metric">'
            f'<div class="metric-value">{len(valid_k)}</div>'
            '<div class="metric-label">有効な知見</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        api_text = "Online" if api_ok else "Offline"
        api_color = "#2E7D32" if api_ok else "#C62828"
        st.markdown(
            '<div class="reuters-metric">'
            f'<div class="metric-value" style="color:{api_color};">{api_text}</div>'
            '<div class="metric-label">J-Quants API</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- 最近の研究 10件 ---
    st.markdown("## Recent Research")
    recent_runs = runs[:10]
    if not recent_runs:
        st.info("まだ研究実行がありません。「研究」ページからアイデアを入力して実行してください。")
    else:
        for run in recent_runs:
            idea_snap = run.get("idea_snapshot", {})
            title = idea_snap.get("title", "不明") if isinstance(idea_snap, dict) else "不明"
            started = run.get("started_at", "")[:10]
            status = run.get("status", "unknown")
            eval_label = run.get("evaluation_label", "")

            bt = run.get("backtest_result") or {}
            sharpe = bt.get("sharpe_ratio")
            cum_ret = bt.get("cumulative_return")

            badge_html = render_status_badge(status)
            if eval_label and status == "completed":
                badge_html += " " + render_status_badge(eval_label)

            metrics_parts = []
            if sharpe is not None:
                metrics_parts.append(f"<span>Sharpe <strong>{sharpe:.2f}</strong></span>")
            if cum_ret is not None:
                metrics_parts.append(f"<span>Return <strong>{cum_ret:.1%}</strong></span>")
            metrics_html = "".join(metrics_parts)

            card_html = (
                f"<h4>{title}</h4>"
                f'<div class="card-meta">{started} &mdash; Run #{run["id"]} {badge_html}</div>'
                f'<div class="card-metrics">{metrics_html}</div>'
            )
            st.markdown(render_card(card_html, accent=(eval_label == "valid")), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
