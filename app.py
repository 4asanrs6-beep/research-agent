"""研究エージェントアプリ - メインエントリ（ダッシュボード）"""

import streamlit as st

from config import DB_PATH, MARKET_DATA_DIR, JQUANTS_API_KEY
from db.database import Database
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider

st.set_page_config(
    page_title="研究エージェント",
    page_icon="🔬",
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
    st.title("🔬 研究エージェント")
    st.markdown("日本株データを用いた投資仮説の研究・検証プラットフォーム")

    st.divider()

    db = get_database()

    # ステータス表示
    col1, col2, col3, col4 = st.columns(4)

    ideas = db.list_ideas()
    runs = db.list_runs()
    knowledge_list = db.list_knowledge()

    with col1:
        st.metric("アイデア", len(ideas))
    with col2:
        completed_runs = [r for r in runs if r["status"] == "completed"]
        st.metric("完了した分析", len(completed_runs))
    with col3:
        valid_k = [k for k in knowledge_list if k["validity"] == "valid"]
        st.metric("有効な知見", len(valid_k))
    with col4:
        provider = get_data_provider()
        api_status = "接続済" if provider.is_available() else "未接続"
        st.metric("J-Quants API", api_status)

    st.divider()

    # 接続状態チェック
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("環境設定ステータス")
        from core.ai_client import ClaudeCodeClient
        claude_ok = ClaudeCodeClient().is_available()
        checks = {
            "J-Quants API": bool(JQUANTS_API_KEY),
            "Claude Code CLI": claude_ok,
            "データベース": True,
        }
        for name, ok in checks.items():
            icon = "✅" if ok else "⚠️"
            if not ok and name == "Claude Code CLI":
                status_text = "未検出 (claude コマンドを確認してください)"
            elif not ok:
                status_text = "未設定 (.envを確認してください)"
            else:
                status_text = "利用可能"
            st.write(f"{icon} **{name}**: {status_text}")

        cache = DataCache(MARKET_DATA_DIR)
        stats = cache.get_stats()
        st.write(f"📦 **キャッシュ**: {stats['file_count']}ファイル ({stats['total_size_mb']}MB)")

    with col_right:
        st.subheader("最近の分析")
        recent_runs = runs[:5]
        if recent_runs:
            for run in recent_runs:
                idea_snap = run.get("idea_snapshot", {})
                title = idea_snap.get("title", "不明") if isinstance(idea_snap, dict) else "不明"
                label = run.get("evaluation_label", "---")
                status = run.get("status", "unknown")

                label_icon = {"valid": "✅", "invalid": "❌", "needs_review": "🔍"}.get(label, "⏳")
                status_icon = {"completed": "🟢", "running": "🔵", "failed": "🔴"}.get(status, "⚪")

                st.write(f"{status_icon} {label_icon} **{title}** (Run #{run['id']})")
        else:
            st.info("まだ分析実行がありません。「アイデア管理」からアイデアを追加してください。")

    st.divider()

    # ページナビゲーション
    st.subheader("ナビゲーション")
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)

    with nav_col1:
        st.page_link("pages/1_💡_アイデア管理.py", label="💡 アイデア管理", use_container_width=True)
    with nav_col2:
        st.page_link("pages/2_🔬_分析実行.py", label="🔬 分析実行", use_container_width=True)
    with nav_col3:
        st.page_link("pages/3_📊_分析結果.py", label="📊 分析結果", use_container_width=True)
    with nav_col4:
        st.page_link("pages/4_📚_知見ベース.py", label="📚 知見ベース", use_container_width=True)
    with nav_col5:
        st.page_link("pages/5_🤖_AI研究.py", label="🤖 AI研究", use_container_width=True)


if __name__ == "__main__":
    main()
