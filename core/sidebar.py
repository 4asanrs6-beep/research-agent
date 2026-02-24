"""全ページ共通サイドバー — 実行中インジケーター"""

import streamlit as st


def render_sidebar_running_indicator():
    """研究・バックテストの実行中インジケーターをサイドバーに表示する。

    全ページの main() 先頭で呼び出すこと。
    alive なスレッドがあれば該当ページへの遷移リンク付きで表示する。
    """
    rp_thread = st.session_state.get("rp_thread")
    bt_thread = st.session_state.get("bt_thread")

    rp_alive = rp_thread is not None and rp_thread.is_alive()
    bt_alive = bt_thread is not None and bt_thread.is_alive()

    if not rp_alive and not bt_alive:
        return

    with st.sidebar:
        if rp_alive:
            prog = st.session_state.get("rp_progress", {})
            msg = prog.get("message", "処理中...")
            st.markdown(
                '<div class="sidebar-running">'
                '<span class="pulse"></span> 研究を実行中...<br>'
                f'<small>{msg}</small></div>',
                unsafe_allow_html=True,
            )
            st.page_link("pages/1_研究.py", label="研究ページへ移動", icon=":material/science:")

        if bt_alive:
            prog = st.session_state.get("bt_progress", {})
            msg = prog.get("message", "処理中...")
            st.markdown(
                '<div class="sidebar-running">'
                '<span class="pulse"></span> バックテスト実行中...<br>'
                f'<small>{msg}</small></div>',
                unsafe_allow_html=True,
            )
            st.page_link("pages/2_標準バックテスト.py", label="バックテストページへ移動", icon=":material/query_stats:")
