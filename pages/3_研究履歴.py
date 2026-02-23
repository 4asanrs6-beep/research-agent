"""研究履歴ページ — 過去の研究結果一覧 + 詳細表示"""

import streamlit as st

from config import DB_PATH
from db.database import Database
from core.styles import apply_reuters_style, render_status_badge, render_card
from core.result_display import render_result_tabs

st.set_page_config(page_title="研究履歴", page_icon="R", layout="wide")


@st.cache_resource
def get_db():
    return Database(DB_PATH)


def main():
    apply_reuters_style()

    # --- サイドバー: 実行中インジケーター ---
    thread = st.session_state.get("research_thread")
    if thread and thread.is_alive():
        prog = st.session_state.get("research_progress", {})
        st.sidebar.markdown(
            '<div class="sidebar-running">'
            '<span class="pulse"></span> 研究を実行中...<br>'
            f'<small>{prog.get("message", "")}</small></div>',
            unsafe_allow_html=True,
        )

    st.markdown("# Research History")
    st.caption("過去の研究結果を閲覧")

    db = get_db()

    # --- 詳細表示モード ---
    selected_run_id = st.session_state.get("history_detail_run_id")
    if selected_run_id:
        _show_detail(db, selected_run_id)
        return

    # --- 一覧表示 ---
    _show_list(db)


def _show_list(db: Database):
    runs = db.list_runs()
    if not runs:
        st.info("まだ研究結果がありません。")
        return

    for run in runs:
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

        if st.button("詳細を表示", key=f"detail_{run['id']}"):
            st.session_state["history_detail_run_id"] = run["id"]
            st.rerun()


def _show_detail(db: Database, run_id: int):
    run = db.get_run(run_id)
    if not run:
        st.error("指定されたRunが見つかりません。")
        if st.button("一覧に戻る"):
            st.session_state.pop("history_detail_run_id", None)
            st.rerun()
        return

    # 戻るボタン
    if st.button("< 一覧に戻る"):
        st.session_state.pop("history_detail_run_id", None)
        st.rerun()

    idea_snap = run.get("idea_snapshot", {})
    title = idea_snap.get("title", "不明") if isinstance(idea_snap, dict) else "不明"
    started = run.get("started_at", "")[:16]
    status = run.get("status", "unknown")
    eval_label = run.get("evaluation_label", "")

    badge_html = render_status_badge(status)
    if eval_label and status == "completed":
        badge_html += " " + render_status_badge(eval_label)

    st.markdown(f"## {title}", unsafe_allow_html=True)
    st.markdown(f"Run #{run_id} &mdash; {started} &nbsp; {badge_html}", unsafe_allow_html=True)

    if status == "failed":
        evaluation = run.get("evaluation") or {}
        st.error(f"この研究はエラーで終了しました: {evaluation.get('error', '不明')}")
        return

    # データ取得
    stats = run.get("statistics_result") or {}
    backtest = run.get("backtest_result") or {}
    evaluation = run.get("evaluation") or {}
    generated_code = evaluation.get("generated_code", "")
    recent_examples = stats.get("recent_examples")

    # 標準BTの場合はパラメータ設定タブ
    plan_snap = run.get("plan_snapshot", {})
    is_standard_bt = (
        isinstance(plan_snap, dict)
        and plan_snap.get("analysis_method") == "standard_backtest"
    )

    if is_standard_bt:
        config_snapshot = plan_snap.get("parameters", {})
        render_result_tabs(
            evaluation, stats, backtest, config_snapshot, recent_examples,
            code_tab_label="パラメータ設定", code_language="json",
        )
    else:
        render_result_tabs(evaluation, stats, backtest, generated_code, recent_examples)


if __name__ == "__main__":
    main()
