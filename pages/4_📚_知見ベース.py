"""知見検索・閲覧ページ"""

import streamlit as st

from config import DB_PATH
from db.database import Database
from core.knowledge_base import KnowledgeBase

st.set_page_config(page_title="知見ベース", page_icon="📚", layout="wide")


@st.cache_resource
def get_db():
    return Database(DB_PATH)


def main():
    st.title("📚 知見ベース")
    st.caption("分析から得られた知見の蓄積・検索")

    db = get_db()
    kb = KnowledgeBase(db)

    # --- 検索・フィルタ ---
    col1, col2, col3 = st.columns(3)
    with col1:
        validity_filter = st.selectbox(
            "有効性フィルタ",
            ["すべて", "valid", "invalid", "needs_review"],
        )
    with col2:
        search_query = st.text_input("キーワード検索", placeholder="例: モメンタム")
    with col3:
        tag_filter = st.text_input("タグ検索", placeholder="例: カレンダー効果")

    validity = None if validity_filter == "すべて" else validity_filter
    tag = tag_filter if tag_filter else None
    query = search_query if search_query else None

    knowledge_list = kb.search(validity=validity, tag=tag, query=query)

    st.divider()

    # --- 統計 ---
    all_knowledge = kb.search()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総知見数", len(all_knowledge))
    with col2:
        st.metric("有効", len([k for k in all_knowledge if k.validity == "valid"]))
    with col3:
        st.metric("無効", len([k for k in all_knowledge if k.validity == "invalid"]))
    with col4:
        st.metric("要レビュー", len([k for k in all_knowledge if k.validity == "needs_review"]))

    st.divider()

    # --- 知見一覧 ---
    if not knowledge_list:
        st.info("知見がありません。分析を実行すると自動的に知見が蓄積されます。")
        return

    st.write(f"**{len(knowledge_list)}件の知見が見つかりました**")

    for k in knowledge_list:
        icon = {"valid": "✅", "invalid": "❌", "needs_review": "🔍"}.get(k.validity, "⚪")

        with st.expander(f"{icon} {k.hypothesis[:80]} (ID: {k.id})"):
            col_l, col_r = st.columns([2, 1])

            with col_l:
                st.write(f"**仮説**: {k.hypothesis}")

                if k.summary:
                    st.write("**サマリー**:")
                    st.write(k.summary)

                if k.valid_conditions:
                    st.write("**有効条件**:")
                    st.write(k.valid_conditions)

                if k.invalid_conditions:
                    st.write("**無効条件**:")
                    st.write(k.invalid_conditions)

            with col_r:
                st.write(f"**有効性**: {k.validity}")
                st.write(f"**Run ID**: {k.run_id or 'N/A'}")
                st.write(f"**作成日**: {k.created_at[:10]}")
                if k.tags:
                    st.write("**タグ**: " + ", ".join(f"`{t}`" for t in k.tags))

                # 有効性の変更
                new_validity = st.selectbox(
                    "有効性を変更",
                    ["valid", "invalid", "needs_review"],
                    index=["valid", "invalid", "needs_review"].index(k.validity),
                    key=f"validity_{k.id}",
                )
                if st.button("更新", key=f"update_k_{k.id}"):
                    kb.update(k.id, validity=new_validity)
                    st.success("更新しました")
                    st.rerun()

                if st.button("🗑️ 削除", key=f"del_k_{k.id}", type="secondary"):
                    kb.delete(k.id)
                    st.success("削除しました")
                    st.rerun()


if __name__ == "__main__":
    main()
