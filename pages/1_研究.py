"""研究ページ — アイデア入力 → 計画レビュー → 実行 → 結果表示

全 AI 処理はバックグラウンドスレッドで実行。
ページ遷移しても処理は継続し、戻れば最新の状態が表示される。
待機画面は @st.fragment(run_every=2) でフラグメントのみ再描画し、
ページ全体の点滅を防止する。
"""

import threading
from datetime import date, datetime

import streamlit as st

from config import (
    DB_PATH, MARKET_DATA_DIR, ANALYSIS_CATEGORIES, JQUANTS_API_KEY,
)
from db.database import Database
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider
from core.ai_client import create_ai_client
from core.ai_researcher import AiResearcher, ResearchProgress
from core.universe_filter import (
    UniverseFilterConfig,
    build_universe_description,
    MARKET_SEGMENTS,
    TOPIX_SCALE_CATEGORIES,
    SECTOR_17_LIST,
)
from core.styles import apply_reuters_style, apply_waiting_overlay
from core.result_display import render_result_tabs, render_plan
from core.sidebar import render_sidebar_running_indicator

st.set_page_config(page_title="研究", page_icon="R", layout="wide")


@st.cache_resource
def get_db():
    return Database(DB_PATH)


@st.cache_resource
def get_data_provider():
    cache = DataCache(MARKET_DATA_DIR)
    return JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)


# ===========================================================================
# バックグラウンドスレッド関数（Streamlit API 使用禁止）
# ===========================================================================
def _thread_generate_plan(shared, db, provider, idea_text, idea_title, category,
                          universe_filter_text, universe_config, start_date, end_date):
    try:
        ai_client = create_ai_client()
        researcher = AiResearcher(db=db, ai_client=ai_client, data_provider=provider)
        shared["message"] = "AIが分析計画を作成しています..."
        plan, idea_id, plan_id = researcher.generate_plan_only(
            idea_text=idea_text,
            idea_title=idea_title,
            category=category,
            universe_filter_text=universe_filter_text,
            universe_config=universe_config,
            start_date=start_date,
            end_date=end_date,
        )
        shared["_result"] = (plan, idea_id, plan_id)
    except Exception as e:
        shared["error"] = str(e)


def _thread_refine_plan(shared, db, provider, plan, feedback, plan_id):
    try:
        ai_client = create_ai_client()
        researcher = AiResearcher(db=db, ai_client=ai_client, data_provider=provider)
        shared["message"] = "AIが計画を修正しています..."
        new_plan = researcher.refine_plan(plan, feedback, plan_id)
        shared["_result"] = new_plan
    except Exception as e:
        shared["error"] = str(e)


def _thread_execute(shared, db, provider, plan, meta):
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


# ===========================================================================
# メイン
# ===========================================================================
def main():
    apply_reuters_style()
    render_sidebar_running_indicator()
    st.markdown("# Research")
    st.caption("投資アイデアを入力してAIが自動で分析・検証します")

    _check_thread_completion()

    phase = st.session_state.get("rp_phase", "idle")

    if phase == "idle":
        _show_input_form()
    elif phase in ("generating_plan", "refining_plan"):
        apply_waiting_overlay()
        label = "計画を生成中..." if phase == "generating_plan" else "計画を修正中..."
        _waiting_fragment(label)
    elif phase == "plan_ready":
        _show_plan_review()
    elif phase == "executing":
        apply_waiting_overlay()
        _waiting_fragment("分析を実行中...")
    elif phase == "done":
        _show_result()


# ===========================================================================
# スレッド完了チェック（毎描画の先頭で呼ぶ）
# ===========================================================================
def _check_thread_completion():
    thread = st.session_state.get("rp_thread")
    if thread is None or thread.is_alive():
        return

    progress = st.session_state.get("rp_progress", {})
    phase = st.session_state.get("rp_phase", "idle")

    # スレッド参照をクリア
    st.session_state.pop("rp_thread", None)
    st.session_state.pop("rp_start_time", None)

    # --- エラー ---
    if "error" in progress:
        if phase == "generating_plan":
            st.session_state["rp_phase"] = "idle"
        elif phase == "refining_plan":
            st.session_state["rp_phase"] = "plan_ready"
        elif phase == "executing":
            meta = st.session_state.get("rp_meta", {})
            st.session_state["rp_result"] = ResearchProgress(
                phase="error",
                error=progress["error"],
                idea_title=meta.get("idea_title", ""),
                idea_text=meta.get("idea_text", ""),
                category=meta.get("category", ""),
                universe_filter_text=meta.get("universe_filter_text", ""),
                start_date=meta.get("start_date", ""),
                end_date=meta.get("end_date", ""),
                plan=st.session_state.get("rp_plan", {}),
            )
            st.session_state["rp_phase"] = "done"
        st.session_state["_rp_error"] = progress["error"]
        return

    # --- 成功 ---
    result = progress.get("_result")
    if result is None:
        return

    if phase == "generating_plan":
        plan, idea_id, plan_id = result
        st.session_state["rp_plan"] = plan
        meta = st.session_state.get("rp_meta", {})
        meta["idea_id"] = idea_id
        meta["plan_id"] = plan_id
        st.session_state["rp_meta"] = meta
        st.session_state["rp_phase"] = "plan_ready"
    elif phase == "refining_plan":
        st.session_state["rp_plan"] = result
        st.session_state["rp_phase"] = "plan_ready"
    elif phase == "executing":
        st.session_state["rp_result"] = result
        st.session_state["rp_phase"] = "done"


# ===========================================================================
# 待機画面（@st.fragment で部分再描画 — ページ全体は再描画しない）
# ===========================================================================
@st.fragment(run_every=2)
def _waiting_fragment(title: str):
    """フラグメント内で進捗を自動更新する。

    スレッドが完了したら st.rerun(scope="app") でページ全体を再描画し、
    次のフェーズへ遷移する。
    """
    thread = st.session_state.get("rp_thread")
    if thread is not None and not thread.is_alive():
        _check_thread_completion()
        st.rerun(scope="app")
        return

    progress = st.session_state.get("rp_progress", {})
    start_time = st.session_state.get("rp_start_time")

    st.markdown(f"### {title}")
    st.write(progress.get("message", "処理中..."))

    elapsed = ""
    if start_time:
        total_sec = int((datetime.now() - start_time).total_seconds())
        mm, ss = divmod(total_sec, 60)
        elapsed = f"経過時間: {mm:02d}:{ss:02d}"
    st.caption(elapsed)


# ===========================================================================
# ステージ 1: 入力フォーム
# ===========================================================================
def _show_input_form():
    error = st.session_state.pop("_rp_error", None)
    if error:
        st.error(f"計画生成エラー: {error}")

    from core.ai_client import ClaudeCodeClient
    if not ClaudeCodeClient().is_available():
        st.warning(
            "Claude Code CLI が検出されません。"
            "デモモード（ダミー応答）で動作します。"
        )

    idea_title = st.text_input(
        "タイトル",
        placeholder="例: 月曜日の株式リターンは低い（月曜効果）",
    )
    idea_text = st.text_area(
        "アイデアの詳細説明",
        height=150,
        placeholder=(
            "例: 月曜日は他の曜日と比較して平均リターンが低い傾向がある。"
            "週末効果とも呼ばれるこのアノマリーが日本市場でも有効かを検証したい。"
        ),
    )
    category = st.selectbox("カテゴリ", ANALYSIS_CATEGORIES)

    # --- ユニバース設定 ---
    with st.expander("ユニバース設定（分析対象銘柄の絞り込み）"):
        ucol1, ucol2 = st.columns(2)
        with ucol1:
            selected_markets = st.multiselect("市場区分", options=MARKET_SEGMENTS, help="空 = 全市場")
            selected_scales = st.multiselect("TOPIX規模区分", options=TOPIX_SCALE_CATEGORIES, help="空 = 全規模")
            margin_only = st.checkbox("貸借銘柄のみ（空売り可能）")
        with ucol2:
            sector_type = st.radio("業種フィルター", options=["なし", "17業種区分"], horizontal=True)
            selected_sectors = []
            if sector_type == "17業種区分":
                selected_sectors = st.multiselect("業種を選択", options=SECTOR_17_LIST)

        st.markdown("**財務スクリーニング**")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            cap_min = st.number_input("時価総額 下限（億円）", min_value=0.0, value=0.0, step=100.0, help="0 = 制限なし")
            cap_max = st.number_input("時価総額 上限（億円）", min_value=0.0, value=0.0, step=100.0, help="0 = 制限なし")
        with fc2:
            per_min = st.number_input("PER 下限（倍）", min_value=0.0, value=0.0, step=1.0, help="0 = 制限なし")
            per_max = st.number_input("PER 上限（倍）", min_value=0.0, value=0.0, step=1.0, help="0 = 制限なし")
        with fc3:
            pbr_min = st.number_input("PBR 下限（倍）", min_value=0.0, value=0.0, step=0.1, help="0 = 制限なし")
            pbr_max = st.number_input("PBR 上限（倍）", min_value=0.0, value=0.0, step=0.1, help="0 = 制限なし")

        st.markdown("**分析期間**")
        dc1, dc2 = st.columns(2)
        with dc1:
            start_date = st.date_input("開始日", value=date(2021, 1, 1))
        with dc2:
            end_date = st.date_input("終了日", value=date.today())

        universe_config = UniverseFilterConfig(
            market_segments=selected_markets,
            scale_categories=selected_scales,
            sector_filter_type="sector_17" if sector_type == "17業種区分" else "none",
            selected_sectors=selected_sectors,
            margin_tradable_only=margin_only,
            market_cap_min=cap_min if cap_min > 0 else None,
            market_cap_max=cap_max if cap_max > 0 else None,
            per_min=per_min if per_min > 0 else None,
            per_max=per_max if per_max > 0 else None,
            pbr_min=pbr_min if pbr_min > 0 else None,
            pbr_max=pbr_max if pbr_max > 0 else None,
        )
        universe_filter_text = build_universe_description(universe_config)
        if universe_filter_text:
            st.info(universe_filter_text)
        else:
            st.caption("フィルタ条件なし（全銘柄が対象）")

    st.markdown("---")

    # --- 計画生成ボタン ---
    if st.button("計画を生成", type="primary", width='stretch', disabled=not idea_text):
        if not idea_text:
            st.error("アイデアの説明を入力してください。")
            return

        shared = {"message": "開始中..."}
        meta = {
            "idea_text": idea_text,
            "idea_title": idea_title or idea_text[:50],
            "category": category,
            "universe_filter_text": universe_filter_text,
            "universe_config": universe_config,
            "start_date": str(start_date),
            "end_date": str(end_date),
        }
        st.session_state["rp_progress"] = shared
        st.session_state["rp_start_time"] = datetime.now()
        st.session_state["rp_phase"] = "generating_plan"
        st.session_state["rp_meta"] = meta

        t = threading.Thread(
            target=_thread_generate_plan,
            args=(
                shared, get_db(), get_data_provider(),
                idea_text, meta["idea_title"], category,
                universe_filter_text, universe_config,
                str(start_date), str(end_date),
            ),
            daemon=True,
        )
        st.session_state["rp_thread"] = t
        t.start()
        st.rerun()


# ===========================================================================
# ステージ 2: 計画レビュー
# ===========================================================================
def _show_plan_review():
    error = st.session_state.pop("_rp_error", None)
    if error:
        st.error(f"計画修正エラー: {error}")

    plan = st.session_state["rp_plan"]
    meta = st.session_state["rp_meta"]

    st.markdown("## 分析計画")

    # 入力条件
    with st.expander("入力条件", expanded=False):
        st.markdown(f"**タイトル:** {meta['idea_title']}")
        st.markdown(f"**カテゴリ:** {meta['category']}")
        st.markdown("**アイデア詳細:**")
        st.info(meta["idea_text"])
        if meta["start_date"] or meta["end_date"]:
            st.markdown(f"**分析期間:** {meta['start_date']} 〜 {meta['end_date']}")
        if meta["universe_filter_text"]:
            st.markdown(f"**ユニバース条件:** {meta['universe_filter_text']}")

    # 計画の表示（共通コンポーネント）
    render_plan(plan)

    st.markdown("---")

    # --- 計画修正 ---
    with st.expander("計画を修正する"):
        feedback = st.text_area(
            "修正したい内容を自由に記述してください",
            placeholder="例: バックテスト戦略をロングショートに変更してほしい / 分析期間を3年に短縮して",
            key="plan_feedback",
        )
        if st.button("修正して再生成", disabled=not feedback):
            shared = {"message": "開始中..."}
            st.session_state["rp_progress"] = shared
            st.session_state["rp_start_time"] = datetime.now()
            st.session_state["rp_phase"] = "refining_plan"

            t = threading.Thread(
                target=_thread_refine_plan,
                args=(shared, get_db(), get_data_provider(), plan, feedback, meta["plan_id"]),
                daemon=True,
            )
            st.session_state["rp_thread"] = t
            t.start()
            st.rerun()

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        execute = st.button("この計画で実行する", type="primary", width='stretch')
    with col2:
        if st.button("やり直す", width='stretch'):
            _clear_state()
            st.rerun()

    if execute:
        shared = {"message": "開始中..."}
        st.session_state["rp_progress"] = shared
        st.session_state["rp_start_time"] = datetime.now()
        st.session_state["rp_phase"] = "executing"

        t = threading.Thread(
            target=_thread_execute,
            args=(shared, get_db(), get_data_provider(), plan, meta),
            daemon=True,
        )
        st.session_state["rp_thread"] = t
        t.start()
        st.rerun()


# ===========================================================================
# ステージ 3: 結果表示
# ===========================================================================
def _show_result():
    result: ResearchProgress = st.session_state["rp_result"]

    if result.phase == "error":
        st.error(f"研究中にエラーが発生しました: {result.error}")
        _render_conditions_expander(result)
        if result.plan:
            with st.expander("AI生成計画"):
                render_plan(result.plan)
        if result.generated_code:
            with st.expander("生成されたコード"):
                st.code(result.generated_code, language="python")
        if st.button("新しい研究を開始"):
            _clear_state()
            st.rerun()
        return

    st.success(f"研究が完了しました (Run #{result.run_id})")

    _render_conditions_expander(result)

    if result.plan:
        with st.expander("AI生成計画"):
            render_plan(result.plan)

    # データ準備
    exec_result = result.execution_result.get("result", {}) if result.execution_result else {}
    stats = exec_result.get("statistics", {}) or {}
    backtest = exec_result.get("backtest", {}) or {}
    interpretation = result.interpretation or {}
    generated_code = result.generated_code or ""
    recent_examples = exec_result.get("recent_examples") or stats.get("recent_examples")

    render_result_tabs(interpretation, stats, backtest, generated_code, recent_examples)

    st.markdown("---")
    if st.button("新しい研究を開始", type="primary"):
        _clear_state()
        st.rerun()


def _render_conditions_expander(result: ResearchProgress):
    """研究条件の折りたたみ表示"""
    with st.expander("研究条件", expanded=False):
        if result.idea_title:
            st.markdown(f"**タイトル:** {result.idea_title}")
        if result.category:
            st.markdown(f"**カテゴリ:** {result.category}")
        if result.idea_text:
            st.markdown("**アイデア詳細:**")
            st.info(result.idea_text)
        if result.start_date or result.end_date:
            st.markdown(f"**分析期間:** {result.start_date} 〜 {result.end_date}")
        if result.universe_filter_text:
            st.markdown(f"**ユニバース条件:** {result.universe_filter_text}")


# ===========================================================================
# ユーティリティ
# ===========================================================================
def _clear_state():
    for key in list(st.session_state.keys()):
        if key.startswith("rp_") or key == "_rp_error":
            st.session_state.pop(key, None)
    # 旧バージョンのキーも掃除
    for key in ("research_thread", "research_progress", "research_start_time",
                "research_result", "research_plan", "research_plan_meta"):
        st.session_state.pop(key, None)


if __name__ == "__main__":
    main()
