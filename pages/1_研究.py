"""研究ページ — アイデア入力 + バックグラウンド実行 + 6タブ結果表示"""

import threading
import time
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
from core.styles import apply_reuters_style
from core.result_display import render_result_tabs

st.set_page_config(page_title="研究", page_icon="R", layout="wide")


@st.cache_resource
def get_db():
    return Database(DB_PATH)


@st.cache_resource
def get_data_provider():
    cache = DataCache(MARKET_DATA_DIR)
    return JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)


# ---------------------------------------------------------------------------
# バックグラウンド研究スレッド
# ---------------------------------------------------------------------------
def _run_research_thread(
    progress_dict: dict,
    db: Database,
    provider,
    idea_text: str,
    idea_title: str,
    category: str,
    universe_filter_text: str,
    universe_config: UniverseFilterConfig | None,
    start_date: str,
    end_date: str,
):
    """バックグラウンドスレッドで研究を実行する関数

    NOTE: st.session_state / @st.cache_resource はスレッド内から使えない。
    db, provider は呼び出し元（メインスレッド）で解決して渡す。
    progress_dict は通常の Python dict で、両スレッドが同じオブジェクトを共有する。
    """
    try:
        ai_client = create_ai_client()
        researcher = AiResearcher(db=db, ai_client=ai_client, data_provider=provider)

        def on_progress(p: ResearchProgress):
            progress_dict["phase"] = p.phase
            progress_dict["message"] = p.message
            progress_dict["run_id"] = p.run_id
            if p.error:
                progress_dict["error"] = p.error

        result = researcher.run_research(
            idea_text=idea_text,
            idea_title=idea_title,
            category=category,
            on_progress=on_progress,
            universe_filter_text=universe_filter_text,
            universe_config=universe_config,
            start_date=start_date,
            end_date=end_date,
        )
        # 結果もスレッドセーフな共有 dict に格納
        progress_dict["_result"] = result
    except Exception as e:
        progress_dict["phase"] = "error"
        progress_dict["error"] = str(e)


# ---------------------------------------------------------------------------
# メインページ
# ---------------------------------------------------------------------------
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

    st.markdown("# Research")
    st.caption("投資アイデアを入力してAIが自動で分析・検証します")

    # ==================================================================
    # 状態判定
    # ==================================================================
    thread = st.session_state.get("research_thread")
    is_running = thread is not None and thread.is_alive()

    # スレッド終了後: progress_dict["_result"] → session_state に昇格
    if thread is not None and not thread.is_alive() and "research_result" not in st.session_state:
        progress_dict = st.session_state.get("research_progress", {})
        if "_result" in progress_dict:
            st.session_state["research_result"] = progress_dict.pop("_result")
        elif progress_dict.get("phase") == "error":
            # エラー終了: ダミーの ResearchProgress を作成
            err_result = ResearchProgress()
            err_result.phase = "error"
            err_result.error = progress_dict.get("error", "不明なエラー")
            st.session_state["research_result"] = err_result

    has_result = "research_result" in st.session_state

    # ==================================================================
    # A) 実行中 → 進捗表示
    # ==================================================================
    if is_running:
        _show_progress()
        return

    # ==================================================================
    # B) 結果あり → 結果表示
    # ==================================================================
    if has_result:
        _show_result()
        return

    # ==================================================================
    # C) それ以外 → 入力フォーム
    # ==================================================================
    _show_input_form()


# ---------------------------------------------------------------------------
# 進捗表示
# ---------------------------------------------------------------------------
PHASE_PROGRESS = {
    "planning": 0.15,
    "coding": 0.35,
    "executing": 0.55,
    "interpreting": 0.75,
    "saving": 0.90,
    "done": 1.0,
    "error": 1.0,
}

PHASE_LABELS = {
    "planning": "分析計画を生成中...",
    "coding": "分析コードを生成中...",
    "executing": "コードを実行中...",
    "interpreting": "結果を解釈中...",
    "saving": "知見を保存中...",
    "done": "研究完了",
    "error": "エラーが発生しました",
}


def _show_progress():
    progress = st.session_state.get("research_progress", {})
    start_time = st.session_state.get("research_start_time")

    phase = progress.get("phase", "planning")
    pct = PHASE_PROGRESS.get(phase, 0.1)
    label = PHASE_LABELS.get(phase, phase)

    # 経過時間
    elapsed_str = ""
    if start_time:
        elapsed = datetime.now() - start_time
        total_sec = int(elapsed.total_seconds())
        mm, ss = divmod(total_sec, 60)
        elapsed_str = f"{mm:02d}:{ss:02d}"

    st.progress(pct)
    st.markdown(
        f"**{label}** &nbsp; <span style='color:#999;font-size:0.9em;'>{elapsed_str}</span>",
        unsafe_allow_html=True,
    )
    if progress.get("message"):
        st.caption(progress["message"])

    # 1秒後にリラン
    time.sleep(1)
    st.rerun()


# ---------------------------------------------------------------------------
# 結果表示
# ---------------------------------------------------------------------------
def _show_result():
    result: ResearchProgress = st.session_state["research_result"]

    if result.phase == "error":
        st.error(f"研究中にエラーが発生しました: {result.error}")
        if result.generated_code:
            with st.expander("生成されたコード"):
                st.code(result.generated_code, language="python")
        if st.button("新しい研究を開始"):
            _clear_state()
            st.rerun()
        return

    st.success(f"研究が完了しました (Run #{result.run_id})")

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


# ---------------------------------------------------------------------------
# 入力フォーム
# ---------------------------------------------------------------------------
def _show_input_form():
    # Claude Code CLI チェック
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

    # --- 実行ボタン ---
    if st.button("研究を実行", type="primary", use_container_width=True, disabled=not idea_text):
        if not idea_text:
            st.error("アイデアの説明を入力してください。")
            return

        # セッション初期化 — progress_dict は通常の dict
        progress_dict = {"phase": "planning", "message": "開始中..."}
        st.session_state["research_progress"] = progress_dict
        st.session_state["research_start_time"] = datetime.now()
        st.session_state.pop("research_result", None)

        t = threading.Thread(
            target=_run_research_thread,
            args=(
                progress_dict,
                get_db(),
                get_data_provider(),
                idea_text,
                idea_title or idea_text[:50],
                category,
                universe_filter_text,
                universe_config,
                str(start_date),
                str(end_date),
            ),
            daemon=True,
        )
        st.session_state["research_thread"] = t
        t.start()
        st.rerun()


def _clear_state():
    for key in ("research_thread", "research_progress", "research_start_time", "research_result"):
        st.session_state.pop(key, None)


if __name__ == "__main__":
    main()
