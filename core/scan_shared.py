"""異常スキャン — スレッド ↔ メインスレッド間の進捗共有用モジュール

st.session_state はバックグラウンドスレッドからアクセスすると
ScriptRunContext 警告が出るため、代わりにこのモジュールレベルの dict を使う。
"""

# スレッドが書き込み、メインスレッド（UI）が読み取る共有 dict
SCAN_SHARED: dict = {}
