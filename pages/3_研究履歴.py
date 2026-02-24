"""研究履歴ページ — 過去の研究結果一覧 + 詳細表示 + 計画編集・再実行"""

import json
import threading
from datetime import datetime

import streamlit as st

from config import DB_PATH, MARKET_DATA_DIR, JQUANTS_API_KEY
from db.database import Database
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider
from core.styles import apply_reuters_style, render_status_badge, render_card
from core.result_display import render_result_tabs, render_plan
from core.sidebar import render_sidebar_running_indicator
from core.ai_client import create_ai_client
from core.ai_researcher import AiResearcher

st.set_page_config(page_title="研究履歴", page_icon="R", layout="wide")


@st.cache_resource
def get_db():
    return Database(DB_PATH)


@st.cache_resource
def get_data_provider():
    cache = DataCache(MARKET_DATA_DIR)
    return JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)


def main():
    apply_reuters_style()
    render_sidebar_running_indicator()

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

        category = idea_snap.get("category", "") if isinstance(idea_snap, dict) else ""
        description = idea_snap.get("description", "") if isinstance(idea_snap, dict) else ""
        # 説明文は長い場合は80文字で切る
        desc_preview = (description[:80] + "...") if len(description) > 80 else description

        category_html = f"<span style='color:#888;font-size:0.85em;'>[{category}]</span> " if category else ""
        desc_html = f'<div style="color:#666;font-size:0.85em;margin-top:4px;">{desc_preview}</div>' if desc_preview else ""

        card_html = (
            f"<h4>{category_html}{title}</h4>"
            f'{desc_html}'
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

    # 研究条件の表示（スナップショットから復元）
    with st.expander("研究条件", expanded=False):
        if isinstance(idea_snap, dict):
            cat = idea_snap.get("category", "")
            desc = idea_snap.get("description", "")
            if cat:
                st.markdown(f"**カテゴリ:** {cat}")
            if desc:
                st.markdown("**アイデア詳細:**")
                st.info(desc)

        plan_snap = run.get("plan_snapshot", {})
        if isinstance(plan_snap, dict):
            s_date = plan_snap.get("start_date", "")
            e_date = plan_snap.get("end_date", "")
            uni_detail = plan_snap.get("universe_detail", "")
            if s_date or e_date:
                st.markdown(f"**分析期間:** {s_date} 〜 {e_date}")
            if uni_detail:
                st.markdown(f"**ユニバース条件:** {uni_detail}")

    # --- AI計画の表示 ---
    plan_snap = run.get("plan_snapshot", {})
    is_standard_bt = (
        isinstance(plan_snap, dict)
        and plan_snap.get("analysis_method") == "standard_backtest"
    )

    ai_plan = _extract_ai_plan(plan_snap)
    if ai_plan:
        with st.expander("AI生成計画", expanded=False):
            render_plan(ai_plan)

    if status == "failed":
        evaluation = run.get("evaluation") or {}
        st.error(f"この研究はエラーで終了しました: {evaluation.get('error', '不明')}")

    # データ取得
    stats = run.get("statistics_result") or {}
    backtest = run.get("backtest_result") or {}
    evaluation = run.get("evaluation") or {}
    generated_code = evaluation.get("generated_code", "")
    recent_examples = stats.get("recent_examples")

    if status != "failed":
        if is_standard_bt:
            config_snapshot = plan_snap.get("parameters", {})
            render_result_tabs(
                evaluation, stats, backtest, config_snapshot, recent_examples,
                code_tab_label="パラメータ設定", code_language="json",
            )
        else:
            render_result_tabs(evaluation, stats, backtest, generated_code, recent_examples)

    # --- 再実行セクション（AI研究のみ） ---
    if ai_plan and not is_standard_bt:
        _render_rerun_section(db, run, ai_plan, idea_snap, plan_snap)


# ===========================================================================
# AI計画の抽出
# ===========================================================================
def _extract_ai_plan(plan_snap: dict) -> dict | None:
    """plan_snapshot からAI計画dictを復元する。

    plan_snapshot は DB の plans テーブルの全カラムが入っている。
    AI計画の主要フィールド（methodology → parameters, backtest → backtest_config）を
    復元して返す。
    """
    if not isinstance(plan_snap, dict):
        return None

    # 標準BTの場合はAI計画なし
    if plan_snap.get("analysis_method") == "standard_backtest":
        return None

    # plan_snapshot に直接 AI 計画の構造があるか確認
    # （hypothesis, methodology, universe, backtest 等のキーがあれば直接計画）
    if "hypothesis" in plan_snap or "methodology" in plan_snap:
        return plan_snap

    # DB の plans テーブル構造から復元
    parameters = plan_snap.get("parameters", {})
    backtest_config = plan_snap.get("backtest_config", {})

    if not parameters and not backtest_config:
        return None

    # 復元: parameters → methodology, backtest_config → backtest
    restored = {}
    if parameters:
        restored["methodology"] = parameters
    if backtest_config:
        restored["backtest"] = backtest_config

    # その他の情報を追加
    if plan_snap.get("universe_detail"):
        restored["universe"] = {"detail": plan_snap["universe_detail"]}
    if plan_snap.get("start_date") or plan_snap.get("end_date"):
        restored["analysis_period"] = {
            "start_date": plan_snap.get("start_date", ""),
            "end_date": plan_snap.get("end_date", ""),
        }
    if plan_snap.get("name"):
        restored["plan_name"] = plan_snap["name"]

    return restored if restored else None


# ===========================================================================
# 再実行セクション
# ===========================================================================
def _render_rerun_section(
    db: Database,
    run: dict,
    ai_plan: dict,
    idea_snap: dict,
    plan_snap: dict,
):
    """計画の編集・再実行UIを表示する"""
    st.markdown("---")
    st.markdown("### 再実行")

    # 計画をJSON形式で編集可能にする
    plan_json = json.dumps(ai_plan, ensure_ascii=False, indent=2, default=str)

    edited_json = st.text_area(
        "計画を編集（JSON形式）",
        value=plan_json,
        height=300,
        key=f"rerun_plan_{run['id']}",
    )

    # 入力情報の復元
    idea_title = idea_snap.get("title", "") if isinstance(idea_snap, dict) else ""
    idea_text = idea_snap.get("description", "") if isinstance(idea_snap, dict) else ""
    category = idea_snap.get("category", "その他") if isinstance(idea_snap, dict) else "その他"

    col1, col2 = st.columns(2)

    with col1:
        if st.button("計画レビュー画面で編集", key=f"to_review_{run['id']}"):
            try:
                edited_plan = json.loads(edited_json)
            except json.JSONDecodeError as e:
                st.error(f"JSONの形式が正しくありません: {e}")
                return

            _navigate_to_plan_review(
                db=db,
                plan=edited_plan,
                idea_title=idea_title,
                idea_text=idea_text,
                category=category,
                plan_snap=plan_snap,
            )

    with col2:
        if st.button("この計画で直接実行", type="primary", key=f"direct_run_{run['id']}"):
            try:
                edited_plan = json.loads(edited_json)
            except json.JSONDecodeError as e:
                st.error(f"JSONの形式が正しくありません: {e}")
                return

            _navigate_to_direct_execute(
                db=db,
                plan=edited_plan,
                idea_title=idea_title,
                idea_text=idea_text,
                category=category,
                plan_snap=plan_snap,
            )


def _navigate_to_plan_review(
    db: Database,
    plan: dict,
    idea_title: str,
    idea_text: str,
    category: str,
    plan_snap: dict,
):
    """研究ページの plan_ready 状態へ遷移する。

    新しい idea + plan レコードを作成し、研究ページのセッション状態をセット。
    """
    # 新しい DB レコードを作成（元の履歴は不変）
    idea_id = db.create_idea(
        title=f"[再実行] {idea_title}",
        description=idea_text,
        category=category,
    )
    db.update_idea(idea_id, status="active")

    plan_id = db.create_plan(
        idea_id=idea_id,
        name=plan.get("plan_name", "AI生成プラン（再実行）"),
        analysis_method="ai_generated",
        universe=plan.get("universe", {}).get("type", "all"),
        universe_detail=plan.get("universe", {}).get("detail"),
        start_date=plan.get("analysis_period", {}).get("start_date"),
        end_date=plan.get("analysis_period", {}).get("end_date"),
        parameters=plan.get("methodology", {}),
        backtest_config=plan.get("backtest", {}),
    )
    db.update_plan(plan_id, status="ready")

    # 研究ページのセッション状態をセット
    universe_filter_text = plan_snap.get("universe_detail", "") if isinstance(plan_snap, dict) else ""
    start_date = plan.get("analysis_period", {}).get("start_date", "")
    end_date = plan.get("analysis_period", {}).get("end_date", "")

    st.session_state["rp_plan"] = plan
    st.session_state["rp_meta"] = {
        "idea_id": idea_id,
        "plan_id": plan_id,
        "idea_text": idea_text,
        "idea_title": idea_title,
        "category": category,
        "universe_filter_text": universe_filter_text,
        "universe_config": None,
        "start_date": start_date,
        "end_date": end_date,
    }
    st.session_state["rp_phase"] = "plan_ready"

    # 履歴の詳細表示をクリア
    st.session_state.pop("history_detail_run_id", None)

    st.switch_page("pages/1_研究.py")


def _navigate_to_direct_execute(
    db: Database,
    plan: dict,
    idea_title: str,
    idea_text: str,
    category: str,
    plan_snap: dict,
):
    """研究ページの executing 状態へ遷移し、バックグラウンド実行を開始する。

    新しい idea + plan レコードを作成して即座に実行スレッドを起動。
    """
    # 新しい DB レコードを作成
    idea_id = db.create_idea(
        title=f"[再実行] {idea_title}",
        description=idea_text,
        category=category,
    )
    db.update_idea(idea_id, status="active")

    plan_id = db.create_plan(
        idea_id=idea_id,
        name=plan.get("plan_name", "AI生成プラン（再実行）"),
        analysis_method="ai_generated",
        universe=plan.get("universe", {}).get("type", "all"),
        universe_detail=plan.get("universe", {}).get("detail"),
        start_date=plan.get("analysis_period", {}).get("start_date"),
        end_date=plan.get("analysis_period", {}).get("end_date"),
        parameters=plan.get("methodology", {}),
        backtest_config=plan.get("backtest", {}),
    )
    db.update_plan(plan_id, status="ready")

    universe_filter_text = plan_snap.get("universe_detail", "") if isinstance(plan_snap, dict) else ""
    start_date = plan.get("analysis_period", {}).get("start_date", "")
    end_date = plan.get("analysis_period", {}).get("end_date", "")

    meta = {
        "idea_id": idea_id,
        "plan_id": plan_id,
        "idea_text": idea_text,
        "idea_title": idea_title,
        "category": category,
        "universe_filter_text": universe_filter_text,
        "universe_config": None,
        "start_date": start_date,
        "end_date": end_date,
    }

    # 実行スレッドを起動（日本語ファイル名の pages/1_研究.py はimportできないため
    # スレッド起動関数をインラインで定義）
    shared = {"message": "開始中..."}
    st.session_state["rp_progress"] = shared
    st.session_state["rp_start_time"] = datetime.now()
    st.session_state["rp_phase"] = "executing"
    st.session_state["rp_plan"] = plan
    st.session_state["rp_meta"] = meta

    t = threading.Thread(
        target=_thread_execute_from_history,
        args=(shared, get_db(), get_data_provider(), plan, meta),
        daemon=True,
    )
    st.session_state["rp_thread"] = t
    t.start()

    # 履歴の詳細表示をクリア
    st.session_state.pop("history_detail_run_id", None)

    st.switch_page("pages/1_研究.py")


# ===========================================================================
# インラインスレッド関数（pages/1_研究.py の _thread_execute と同等）
# ===========================================================================
def _thread_execute_from_history(shared, db, provider, plan, meta):
    """履歴からの再実行用スレッド関数。

    pages/1_研究.py の _thread_execute と同じロジック。
    日本語ファイル名のため import できないのでここに定義。
    """
    try:
        ai_client = create_ai_client()
        researcher = AiResearcher(db=db, ai_client=ai_client, data_provider=provider)

        def on_status(msg):
            shared["message"] = msg

        result = researcher.execute_from_plan(
            plan=plan,
            idea_id=meta["idea_id"],
            plan_id=meta["plan_id"],
            idea_text=meta["idea_text"],
            idea_title=meta.get("idea_title", ""),
            category=meta.get("category", ""),
            universe_filter_text=meta.get("universe_filter_text", ""),
            universe_config=meta.get("universe_config"),
            start_date=meta.get("start_date"),
            end_date=meta.get("end_date"),
            on_status=on_status,
        )
        shared["_result"] = result
    except Exception as e:
        shared["error"] = str(e)


if __name__ == "__main__":
    main()
