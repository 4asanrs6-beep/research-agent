"""アイデア入力・管理ページ"""

import streamlit as st

from config import DB_PATH, ANALYSIS_CATEGORIES
from db.database import Database
from core.idea_manager import IdeaManager

st.set_page_config(page_title="アイデア管理", page_icon="💡", layout="wide")


@st.cache_resource
def get_db():
    return Database(DB_PATH)


def main():
    st.title("💡 アイデア管理")
    db = get_db()
    manager = IdeaManager(db)

    tab_new, tab_list = st.tabs(["新規作成", "アイデア一覧"])

    # --- 新規作成タブ ---
    with tab_new:
        st.subheader("新しい投資アイデアを追加")

        with st.form("new_idea_form"):
            title = st.text_input("タイトル", placeholder="例: 月曜日の株式リターンは低い")
            description = st.text_area(
                "アイデアの説明",
                height=150,
                placeholder="例: 月曜日は他の曜日と比較して平均リターンが低い傾向がある。"
                "週末効果とも呼ばれるこのアノマリーが日本市場でも有効かを検証したい。",
            )
            category = st.selectbox("カテゴリ", ANALYSIS_CATEGORIES)
            submitted = st.form_submit_button("アイデアを追加", use_container_width=True)

            if submitted:
                if not title or not description:
                    st.error("タイトルと説明は必須です。")
                else:
                    idea = manager.create(title, description, category)
                    st.success(f"アイデアを追加しました (ID: {idea.id})")
                    st.rerun()

    # --- 一覧タブ ---
    with tab_list:
        st.subheader("アイデア一覧")

        filter_status = st.selectbox(
            "ステータスで絞り込み",
            ["すべて", "draft", "active", "completed", "archived"],
        )
        status_filter = None if filter_status == "すべて" else filter_status

        ideas = manager.list_all(status=status_filter)

        if not ideas:
            st.info("アイデアがありません。「新規作成」タブからアイデアを追加してください。")
        else:
            for idea in ideas:
                status_icon = {
                    "draft": "📝", "active": "🔬",
                    "completed": "✅", "archived": "📦",
                }.get(idea.status, "⚪")

                with st.expander(f"{status_icon} [{idea.category}] {idea.title} (ID: {idea.id})"):
                    st.write(f"**ステータス**: {idea.status}")
                    st.write(f"**作成日**: {idea.created_at}")
                    st.write(f"**説明**:")
                    st.write(idea.description)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        new_status = st.selectbox(
                            "ステータス変更",
                            ["draft", "active", "completed", "archived"],
                            index=["draft", "active", "completed", "archived"].index(idea.status),
                            key=f"status_{idea.id}",
                        )
                        if st.button("更新", key=f"update_{idea.id}"):
                            manager.update(idea.id, status=new_status)
                            st.success("更新しました")
                            st.rerun()

                    with col3:
                        if st.button("🗑️ 削除", key=f"delete_{idea.id}", type="secondary"):
                            manager.delete(idea.id)
                            st.success("削除しました")
                            st.rerun()


if __name__ == "__main__":
    main()
