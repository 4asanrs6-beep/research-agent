"""Qlib実験ページ — データ変換 + ML学習 + 結果分析"""

import threading
from datetime import date

import streamlit as st

from config import DB_PATH, MARKET_DATA_DIR, JQUANTS_API_KEY
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider
from core.styles import apply_reuters_style, apply_waiting_overlay
from core.sidebar import render_sidebar_running_indicator
from core.universe_filter import (
    UniverseFilterConfig,
    apply_universe_filter,
    MARKET_SEGMENTS,
    TOPIX_SCALE_CATEGORIES,
)
from qlib_integration.config import (
    QLIB_DATA_DIR,
    DATA_START_DATE,
    DATA_END_DATE,
    TRAIN_PERIOD,
    VALID_PERIOD,
    TEST_PERIOD,
    TOPK,
    N_DROP,
)

st.set_page_config(page_title="Qlib実験", page_icon="R", layout="wide")


@st.cache_resource
def get_data_provider():
    cache = DataCache(MARKET_DATA_DIR)
    return JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)


# ---------------------------------------------------------------------------
# バックグラウンドスレッド — データ変換
# ---------------------------------------------------------------------------
def _run_convert_thread(progress_dict: dict, provider, start_date: str, end_date: str):
    try:
        from qlib_integration.data_bridge import convert_full

        def on_progress(msg, pct):
            progress_dict["message"] = msg
            progress_dict["pct"] = pct

        result = convert_full(
            provider=provider,
            start_date=start_date,
            end_date=end_date,
            on_progress=on_progress,
        )
        progress_dict["_result"] = result
        progress_dict["pct"] = 1.0
        progress_dict["message"] = "完了"
    except Exception as e:
        progress_dict["error"] = str(e)
        progress_dict["pct"] = 1.0


# ---------------------------------------------------------------------------
# バックグラウンドスレッド — 実験実行
# ---------------------------------------------------------------------------
def _run_experiment_thread(
    progress_dict: dict,
    model_type: str,
    train_period: tuple,
    valid_period: tuple,
    test_period: tuple,
    topk: int,
    n_drop: int,
    instrument_codes: list[str] | None = None,
):
    try:
        from qlib_integration.workflow_runner import run_experiment

        def on_progress(msg, pct):
            progress_dict["message"] = msg
            progress_dict["pct"] = pct

        result = run_experiment(
            model_type=model_type,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
            topk=topk,
            n_drop=n_drop,
            instrument_codes=instrument_codes,
            on_progress=on_progress,
        )
        progress_dict["_result"] = result
        progress_dict["pct"] = 1.0
        progress_dict["message"] = "完了"
    except Exception as e:
        import traceback
        progress_dict["error"] = str(e)
        progress_dict["detail"] = traceback.format_exc()
        progress_dict["pct"] = 1.0


# ---------------------------------------------------------------------------
# メインページ
# ---------------------------------------------------------------------------
def main():
    apply_reuters_style()
    render_sidebar_running_indicator()

    st.markdown("# Qlib ML Ranking")
    st.caption("Alpha158特徴量 + MLモデルによるクロスセクショナル銘柄ランキング")

    tab_data, tab_experiment, tab_results = st.tabs([
        "DATA MANAGEMENT",
        "EXPERIMENT",
        "RESULTS",
    ])

    with tab_data:
        _render_data_tab()

    with tab_experiment:
        _render_experiment_tab()

    with tab_results:
        _render_results_tab()


# ---------------------------------------------------------------------------
# タブ1: データ管理
# ---------------------------------------------------------------------------
def _render_data_tab():
    from qlib_integration.data_bridge import get_conversion_status

    st.markdown("## Data Conversion")
    st.markdown("J-Quantsデータ → Qlibバイナリ形式への変換")

    # 既存データの状態表示
    status = get_conversion_status()
    if status:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("カレンダー日数", f"{status['n_calendar_days']:,}")
        with col2:
            st.metric("銘柄数", f"{status['n_instruments']:,}")
        with col3:
            st.metric("データ容量", f"{status['total_size_mb']:.1f} MB")
        with col4:
            st.metric("特徴量ディレクトリ", f"{status['n_features_dirs']:,}")

        st.info(f"カレンダー範囲: {status['calendar_range']}")
    else:
        st.warning("Qlibデータが未変換です。下のボタンで変換を開始してください。")

    st.markdown("---")

    # 変換実行中チェック
    convert_thread = st.session_state.get("qlib_convert_thread")
    is_converting = convert_thread is not None and convert_thread.is_alive()

    if is_converting:
        apply_waiting_overlay()
        prog = st.session_state.get("qlib_convert_progress", {})
        pct = prog.get("pct", 0)
        msg = prog.get("message", "変換中...")

        import time
        started_at = prog.get("_started_at")
        time_info = ""
        if started_at and pct > 0.01:
            elapsed = time.time() - started_at
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            if pct < 0.99:
                remaining = elapsed / pct * (1 - pct)
                remaining_min = int(remaining // 60)
                remaining_sec = int(remaining % 60)
                time_info = f" ({elapsed_min}:{elapsed_sec:02d}経過 / 残り約{remaining_min}:{remaining_sec:02d})"
            else:
                time_info = f" ({elapsed_min}:{elapsed_sec:02d}経過)"
        elif started_at:
            elapsed = time.time() - started_at
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            time_info = f" ({elapsed_min}:{elapsed_sec:02d}経過)"

        st.progress(pct, text=msg + time_info)

        if prog.get("error"):
            st.error(f"変換エラー: {prog['error']}")
        elif "_result" in prog:
            result = prog["_result"]
            st.success(
                f"変換完了: {result['n_converted']}銘柄, "
                f"{result['n_calendar_days']}営業日, "
                f"範囲: {result['calendar_range']}"
            )
            if st.button("OK", key="convert_done"):
                st.session_state.pop("qlib_convert_thread", None)
                st.session_state.pop("qlib_convert_progress", None)
                st.rerun()
        else:
            import time
            time.sleep(2)
            st.rerun()
        return

    # 変換完了結果の表示
    convert_prog = st.session_state.get("qlib_convert_progress", {})
    if convert_thread is not None and not convert_thread.is_alive():
        if "_result" in convert_prog:
            result = convert_prog["_result"]
            st.success(
                f"変換完了: {result['n_converted']}銘柄, "
                f"{result['n_calendar_days']}営業日"
            )
        elif convert_prog.get("error"):
            st.error(f"変換エラー: {convert_prog['error']}")
        if st.button("OK", key="convert_done_2"):
            st.session_state.pop("qlib_convert_thread", None)
            st.session_state.pop("qlib_convert_progress", None)
            st.rerun()
        return

    # 変換パラメータ + 実行
    with st.form("convert_form"):
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "開始日",
                value=date(2017, 1, 1),
                min_value=date(2014, 1, 1),
                max_value=date(2026, 12, 31),
            )
        with col2:
            end_date = st.date_input(
                "終了日",
                value=date(2026, 3, 7),
                min_value=date(2017, 1, 1),
                max_value=date(2026, 12, 31),
            )

        submitted = st.form_submit_button("データ変換を実行", type="primary")

    if submitted:
        provider = get_data_provider()
        if not provider.is_available():
            st.error("J-Quants APIが利用できません。APIキーを確認してください。")
            return

        import time as _time
        progress_dict = {"message": "開始中...", "pct": 0.0, "_started_at": _time.time()}
        st.session_state["qlib_convert_progress"] = progress_dict

        thread = threading.Thread(
            target=_run_convert_thread,
            args=(progress_dict, provider, str(start_date), str(end_date)),
            daemon=True,
        )
        thread.start()
        st.session_state["qlib_convert_thread"] = thread
        st.rerun()


# ---------------------------------------------------------------------------
# タブ2: 実験実行
# ---------------------------------------------------------------------------
def _get_calendar_range():
    """変換済みカレンダーの開始日・終了日をdateで返す。なければNone。"""
    from qlib_integration.data_bridge import get_conversion_status
    status = get_conversion_status()
    if not status or not status.get("calendar_range"):
        return None, None, status
    parts = status["calendar_range"].split(" ~ ")
    cal_start = date.fromisoformat(parts[0])
    cal_end = date.fromisoformat(parts[1])
    return cal_start, cal_end, status


def _auto_periods(cal_start: date, cal_end: date) -> dict:
    """カレンダー範囲からtrain/valid/testの自動分割を計算する。

    全期間を 6:2:2 で分割する。
    """
    total_days = (cal_end - cal_start).days
    train_days = int(total_days * 0.6)
    valid_days = int(total_days * 0.2)

    from datetime import timedelta
    train_end = cal_start + timedelta(days=train_days)
    valid_start = train_end + timedelta(days=1)
    valid_end = valid_start + timedelta(days=valid_days)
    test_start = valid_end + timedelta(days=1)

    return {
        "train_start": cal_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": cal_end,
    }


def _render_experiment_tab():
    cal_start, cal_end, status = _get_calendar_range()
    if not status:
        st.warning("先に「DATA MANAGEMENT」タブでデータ変換を実行してください。")
        return

    if cal_start is None:
        st.warning("カレンダーデータが不正です。データ変換を再実行してください。")
        return

    st.markdown("## Experiment Configuration")
    st.info(f"利用可能なデータ範囲: **{cal_start}** ~ **{cal_end}** ({status['n_calendar_days']}営業日)")

    # 実験実行中チェック
    exp_thread = st.session_state.get("qlib_exp_thread")
    is_running = exp_thread is not None and exp_thread.is_alive()

    if is_running:
        apply_waiting_overlay()
        prog = st.session_state.get("qlib_exp_progress", {})
        pct = prog.get("pct", 0)
        msg = prog.get("message", "実行中...")

        import time
        started_at = prog.get("_started_at")
        time_info = ""
        if started_at and pct > 0.01:
            elapsed = time.time() - started_at
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            if pct < 0.99:
                remaining = elapsed / pct * (1 - pct)
                remaining_min = int(remaining // 60)
                remaining_sec = int(remaining % 60)
                time_info = f" ({elapsed_min}:{elapsed_sec:02d}経過 / 残り約{remaining_min}:{remaining_sec:02d})"
            else:
                time_info = f" ({elapsed_min}:{elapsed_sec:02d}経過)"
        elif started_at:
            elapsed = time.time() - started_at
            elapsed_min = int(elapsed // 60)
            elapsed_sec = int(elapsed % 60)
            time_info = f" ({elapsed_min}:{elapsed_sec:02d}経過)"

        st.progress(pct, text=msg + time_info)
        time.sleep(2)
        st.rerun()
        return

    # スレッド完了後: 結果昇格 + 自動保存
    if exp_thread is not None and not exp_thread.is_alive() and "qlib_result" not in st.session_state:
        prog = st.session_state.get("qlib_exp_progress", {})
        if "_result" in prog:
            from qlib_integration.result_adapter import save_experiment
            result = prog.pop("_result")
            exp_id = save_experiment(result)
            result["experiment_id"] = exp_id
            st.session_state["qlib_result"] = result
        elif prog.get("error"):
            st.error(f"実験エラー: {prog['error']}")
            detail = prog.get("detail", "")
            if detail:
                with st.expander("詳細"):
                    st.code(detail)
            if st.button("OK", key="exp_error_ok"):
                st.session_state.pop("qlib_exp_thread", None)
                st.session_state.pop("qlib_exp_progress", None)
                st.rerun()
            return

    if "qlib_result" in st.session_state:
        st.success("実験完了。「RESULTS」タブで結果を確認できます。")
        if st.button("新しい実験を開始"):
            st.session_state.pop("qlib_result", None)
            st.session_state.pop("qlib_exp_thread", None)
            st.session_state.pop("qlib_exp_progress", None)
            st.rerun()
        return

    # データ範囲から自動分割
    auto = _auto_periods(cal_start, cal_end)

    # 実験パラメータ入力
    with st.form("experiment_form"):
        st.markdown("### ユニバース設定")
        st.caption("対象銘柄を絞り込みます。未選択の場合は全銘柄が対象です。")

        col1, col2 = st.columns(2)
        with col1:
            market_segs = st.multiselect("市場区分", MARKET_SEGMENTS, default=[])
            scale_cats = st.multiselect(
                "TOPIX規模区分", TOPIX_SCALE_CATEGORIES, default=[],
                help="Core30=超大型, Large70=大型, Mid400=中型, Small=小型",
            )
        with col2:
            exclude_etf = st.checkbox("ETF・REIT除外", value=True)
            margin_only = st.checkbox("貸借銘柄のみ", value=False)

        min_cap = st.number_input(
            "最低時価総額 (億円)", value=0, min_value=0, step=100,
            help="TOPIX規模区分から推定。0=制限なし。目安: Core30≒5000億, Large70≒2000億, Mid400≒500億, Small1≒100億",
        )

        st.markdown("### モデル設定")
        model_type = st.selectbox("モデル", ["lgb", "xgb"], format_func=lambda x: {"lgb": "LightGBM", "xgb": "XGBoost"}[x])

        st.markdown("### 期間設定")
        st.caption("データ範囲に基づいて自動分割（6:2:2）されています。変更も可能です。")
        col1, col2 = st.columns(2)
        with col1:
            train_start = st.date_input("学習開始", value=auto["train_start"], min_value=cal_start, max_value=cal_end)
            valid_start = st.date_input("検証開始", value=auto["valid_start"], min_value=cal_start, max_value=cal_end)
            test_start = st.date_input("テスト開始", value=auto["test_start"], min_value=cal_start, max_value=cal_end)
        with col2:
            train_end = st.date_input("学習終了", value=auto["train_end"], min_value=cal_start, max_value=cal_end)
            valid_end = st.date_input("検証終了", value=auto["valid_end"], min_value=cal_start, max_value=cal_end)
            test_end = st.date_input("テスト終了", value=auto["test_end"], min_value=cal_start, max_value=cal_end)

        st.markdown("### ポートフォリオ設定")
        col1, col2 = st.columns(2)
        with col1:
            topk = st.number_input("TopK銘柄数", value=TOPK, min_value=5, max_value=100)
        with col2:
            n_drop = st.number_input("毎日入替数", value=N_DROP, min_value=1, max_value=20)

        submitted = st.form_submit_button("実験を実行", type="primary")

    if submitted:
        # バリデーション: 期間の順序チェック
        if not (train_start < train_end < valid_start < valid_end < test_start < test_end):
            st.error("期間の順序が不正です。学習 → 検証 → テストの順に重ならないように設定してください。")
            return

        # ユニバースフィルタ適用
        universe_config = UniverseFilterConfig(
            market_segments=market_segs,
            scale_categories=scale_cats,
            exclude_etf_reit=exclude_etf,
            margin_tradable_only=margin_only,
        )
        has_cap_filter = min_cap > 0
        instrument_codes = None
        if not universe_config.is_empty() or has_cap_filter:
            provider = get_data_provider()
            stocks = provider.get_listed_stocks()
            filtered = apply_universe_filter(stocks, universe_config)

            # 時価総額フィルタ（scale_categoryから推定）
            if has_cap_filter and "scale_category" in filtered.columns:
                _SCALE_CAP_MAP = {
                    "TOPIX Core30": 5000.0,
                    "TOPIX Large70": 2000.0,
                    "TOPIX Mid400": 500.0,
                    "TOPIX Small 1": 100.0,
                    "TOPIX Small 2": 30.0,
                }
                filtered["_est_cap"] = filtered["scale_category"].map(_SCALE_CAP_MAP).fillna(30.0)
                filtered = filtered[filtered["_est_cap"] >= min_cap]

            instrument_codes = filtered["code"].astype(str).tolist()
            st.info(f"フィルタ適用: {len(stocks)}銘柄 → {len(instrument_codes)}銘柄")
            if len(instrument_codes) == 0:
                st.error("フィルタ条件に合致する銘柄がありません。条件を緩めてください。")
                return

        train_p = (str(train_start), str(train_end))
        valid_p = (str(valid_start), str(valid_end))
        test_p = (str(test_start), str(test_end))

        import time as _time
        progress_dict = {"message": "開始中...", "pct": 0.0, "_started_at": _time.time()}
        st.session_state["qlib_exp_progress"] = progress_dict

        thread = threading.Thread(
            target=_run_experiment_thread,
            args=(progress_dict, model_type, train_p, valid_p, test_p, topk, n_drop, instrument_codes),
            daemon=True,
        )
        thread.start()
        st.session_state["qlib_exp_thread"] = thread
        st.rerun()


# ---------------------------------------------------------------------------
# タブ3: 結果分析
# ---------------------------------------------------------------------------
def _render_results_tab():
    import numpy as np
    import pandas as pd
    from qlib_integration.result_adapter import (
        list_experiments, load_experiment, delete_experiment, summarize_experiment,
    )

    st.markdown("## Experiment Results")

    # 過去の実験一覧
    experiments = list_experiments()

    # セッションに最新結果がある場合、それを優先表示
    session_result = st.session_state.get("qlib_result")

    if not experiments and session_result is None:
        st.info("まだ実験結果がありません。「EXPERIMENT」タブから実験を実行してください。")
        return

    # 実験選択
    if experiments:
        options = []
        for exp in experiments:
            exp_id = exp.get("experiment_id", "?")
            model = exp.get("model_type", "?")
            ic = exp.get("ic_mean")
            ic_str = f"IC={ic:.4f}" if ic is not None else "IC=N/A"
            saved = exp.get("saved_at", "")[:16]
            options.append(f"{exp_id} | {model} | {ic_str} | {saved}")

        # 最新実験をデフォルト選択
        selected_idx = st.selectbox(
            "実験を選択",
            range(len(options)),
            format_func=lambda i: options[i],
            key="exp_selector",
        )

        selected_exp = experiments[selected_idx]
        exp_id = selected_exp["experiment_id"]

        # 完全なデータを読み込み（predictionsなど）
        result = load_experiment(exp_id)
        if result is None:
            st.error("実験データの読み込みに失敗しました。")
            return
    else:
        result = session_result

    # --- 結果表示 ---
    _render_result_detail(result)

    # 削除ボタン
    if experiments:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("この実験を削除", type="secondary"):
                delete_experiment(exp_id)
                st.session_state.pop("qlib_result", None)
                st.rerun()


def _get_code_name_map() -> dict[str, str]:
    """銘柄コード → 企業名の辞書を取得する（キャッシュ付き）。"""
    if "code_name_map" not in st.session_state:
        try:
            provider = get_data_provider()
            stocks = provider.get_listed_stocks()
            st.session_state["code_name_map"] = dict(
                zip(stocks["code"].astype(str), stocks["name"])
            )
        except Exception:
            st.session_state["code_name_map"] = {}
    return st.session_state["code_name_map"]


# Alpha158特徴量の日本語説明
_FEATURE_DESC = {
    "KMID": "ローソク足実体の位置",
    "KLEN": "ローソク足全長(ボラ)",
    "KMID2": "実体/全長比率",
    "KUP": "上ヒゲの長さ",
    "KUP2": "上ヒゲ/全長比率",
    "KLOW": "下ヒゲの長さ",
    "KLOW2": "下ヒゲ/全長比率",
    "KSFT": "終値の偏り",
    "KSFT2": "終値偏り(レンジ基準)",
    "OPEN": "始値/終値比率",
    "HIGH": "高値/終値比率",
    "LOW": "安値/終値比率",
    "VWAP": "VWAP/終値比率",
    "ROC": "N日変化率(モメンタム)",
    "MA": "N日移動平均乖離",
    "STD": "N日標準偏差(ボラ)",
    "BETA": "N日回帰傾き(トレンド強度)",
    "RSQR": "N日回帰R²(トレンド直線性)",
    "RESI": "N日回帰残差(トレンド乖離)",
    "RANK": "N日間の価格位置",
    "MAX": "N日最高値比率",
    "MIN": "N日最安値比率",
    "QTLU": "N日80%タイル",
    "QTLD": "N日20%タイル",
    "RSV": "ストキャスティクス的位置",
    "IMAX": "最高値が何日前か",
    "IMIN": "最安値が何日前か",
    "IMXD": "高安の時間差",
    "CNTP": "上昇日の割合",
    "CNTN": "下落日の割合",
    "CNTD": "上昇優位度",
    "SUMP": "上昇幅合計比率(RSI的)",
    "SUMN": "下落幅合計比率",
    "SUMD": "上昇-下落バランス",
    "VMA": "N日出来高移動平均",
    "VSTD": "N日出来高標準偏差",
    "WVMA": "出来高加重ボラ",
    "VSUMP": "出来高増加日割合",
    "VSUMN": "出来高減少日割合",
    "VSUMD": "出来高方向バランス",
    "CORR": "価格-出来高相関",
    "CORD": "リターン-出来高変化相関",
}


def _describe_feature(name: str) -> str:
    """特徴量名に日本語説明を付与する。"""
    # "KLEN5" → prefix="KLEN", window="5"
    # "(KLEN, 5)" や "('KLEN', 5)" 形式にも対応
    clean = name.replace("(", "").replace(")", "").replace("'", "").replace('"', "")
    parts = [p.strip() for p in clean.split(",")]
    if len(parts) >= 2:
        prefix, window = parts[0], parts[1]
    else:
        # KLEN5 → KLEN + 5
        import re
        m = re.match(r"([A-Z]+)(\d+)$", clean)
        if m:
            prefix, window = m.group(1), m.group(2)
        else:
            prefix, window = clean, ""

    desc = _FEATURE_DESC.get(prefix, "")
    if desc and window:
        return f"{name} ({desc}, {window}日)"
    elif desc:
        return f"{name} ({desc})"
    return name


def _ic_quality_label(ic: float) -> tuple[str, str]:
    """IC値から評価ラベルと色を返す。"""
    if ic > 0.05:
        return "良い（予測力が高い）", "#2E7D32"
    elif ic > 0.03:
        return "使える（有意な予測力あり）", "#FF8000"
    elif ic > 0:
        return "弱い（わずかな予測力）", "#666"
    else:
        return "なし（予測力なし）", "#C62828"


def _build_qlib_copy_text(result: dict, code_name_map: dict) -> str:
    """結果をフォーマット済みテキストにする（一括コピー用）。"""
    import numpy as np
    import pandas as pd

    lines = []
    lines.append("=" * 60)
    lines.append("Qlib ML Ranking 実験結果")
    lines.append("=" * 60)

    # 設定
    model_name = {"lgb": "LightGBM", "xgb": "XGBoost"}.get(
        result.get("model_type", ""), result.get("model_type", "不明")
    )
    lines.append(f"モデル: {model_name}")
    if "train_period" in result:
        lines.append(f"学習期間: {result['train_period'][0]} ~ {result['train_period'][1]}")
    if "test_period" in result:
        lines.append(f"テスト期間: {result['test_period'][0]} ~ {result['test_period'][1]}")
    n_inst = result.get("n_instruments", "全銘柄")
    lines.append(f"対象銘柄数: {n_inst}")
    lines.append(f"TopK: {result.get('topk', '不明')}")
    lines.append("")

    # 評価指標
    lines.append("--- 評価指標 ---")
    ic = result.get("ic_mean", float("nan"))
    icir = result.get("icir", float("nan"))
    rank_ic = result.get("rank_ic_mean", float("nan"))
    if isinstance(ic, float) and not np.isnan(ic):
        quality = "良い" if ic > 0.05 else ("使える" if ic > 0.03 else "弱い")
        lines.append(f"IC (平均): {ic:.4f} ({quality})")
    if isinstance(icir, float) and not np.isnan(icir):
        lines.append(f"ICIR: {icir:.4f} ({'安定' if icir > 0.5 else '不安定'})")
    if isinstance(rank_ic, float) and not np.isnan(rank_ic):
        lines.append(f"Rank IC: {rank_ic:.4f}")
    lines.append(f"テストサンプル数: {result.get('n_test_samples', 0):,}")
    lines.append("")

    # 特徴量重要度
    feat_imp = result.get("feature_importance", {})
    if feat_imp:
        lines.append("--- 特徴量重要度 (Top 30) ---")
        for i, (name, imp) in enumerate(feat_imp.items(), 1):
            desc = _describe_feature(name)
            lines.append(f"{i:2d}. {desc}: {imp:.1f}")
        lines.append("")

    # Top-K銘柄（最新日）
    predictions = result.get("predictions")
    if predictions is not None and isinstance(predictions.index, pd.MultiIndex):
        from qlib_integration.result_adapter import extract_topk_daily
        topk_val = result.get("topk", 30)
        topk_df = extract_topk_daily(predictions, topk=topk_val)
        latest_date = topk_df["date"].max()
        day_top = topk_df[topk_df["date"] == latest_date]

        lines.append(f"--- 推薦銘柄ランキング ({pd.Timestamp(latest_date).strftime('%Y-%m-%d')}) ---")
        for _, row in day_top.iterrows():
            code = str(row["code"])
            name = code_name_map.get(code, code_name_map.get(code.lstrip("0"), ""))
            lines.append(f"{int(row['rank']):2d}. {code} {name}: スコア {row['score']:.4f}")

    return "\n".join(lines)


def _render_result_detail(result: dict):
    """実験結果の詳細を表示する。"""
    import numpy as np
    import pandas as pd
    import plotly.express as px

    # --- 予測品質の総合評価 ---
    st.markdown("### 予測品質")

    ic = result.get("ic_mean", float("nan"))
    icir = result.get("icir", float("nan"))
    rank_ic = result.get("rank_ic_mean", float("nan"))
    n_samples = result.get("n_test_samples", 0)

    ic_valid = isinstance(ic, float) and not np.isnan(ic)
    icir_valid = isinstance(icir, float) and not np.isnan(icir)

    if ic_valid:
        label, color = _ic_quality_label(ic)
        st.markdown(
            f'<div style="background:#F5F5F5;border-left:4px solid {color};'
            f'padding:1rem 1.2rem;border-radius:4px;margin-bottom:1rem;">'
            f'<div style="font-size:1.4em;font-weight:700;color:{color};">'
            f'IC = {ic:.4f} &mdash; {label}</div>'
            f'<div style="font-size:0.85em;color:#666;margin-top:0.3em;">'
            f'ICはモデルの予測スコアと実際のリターンの相関です。'
            f'0.03以上で「使える」、0.05以上で「良い」とされます。</div></div>',
            unsafe_allow_html=True,
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("IC (平均)", f"{ic:.4f}" if ic_valid else "N/A",
                   help="予測スコアと実際のリターンのPearson相関。高いほど予測が正確。")
    with col2:
        st.metric("ICIR", f"{icir:.4f}" if icir_valid else "N/A",
                   help="IC / ICの標準偏差。予測の安定性を示す。0.5以上で安定。")
    with col3:
        ric_valid = isinstance(rank_ic, float) and not np.isnan(rank_ic)
        st.metric("Rank IC", f"{rank_ic:.4f}" if ric_valid else "N/A",
                   help="順位ベースの相関。外れ値に強い指標。")
    with col4:
        st.metric("テストサンプル数", f"{n_samples:,}",
                   help="テスト期間中の（銘柄x日数）のデータ点数。")

    st.markdown("---")

    # --- 実験設定 ---
    with st.expander("実験設定の詳細"):
        settings = {
            "モデル": {"lgb": "LightGBM", "xgb": "XGBoost"}.get(
                result.get("model_type", ""), result.get("model_type", "不明")
            ),
            "学習期間": " ~ ".join(result["train_period"]) if "train_period" in result else "不明",
            "検証期間": " ~ ".join(result["valid_period"]) if "valid_period" in result else "不明",
            "テスト期間": " ~ ".join(result["test_period"]) if "test_period" in result else "不明",
            "TopK銘柄数": str(result.get("topk", "不明")),
            "予測件数": f"{result.get('n_predictions', 0):,}",
        }
        settings_df = pd.DataFrame(
            [(k, v) for k, v in settings.items()],
            columns=["項目", "値"],
        )
        st.dataframe(settings_df, width="stretch", hide_index=True)

    # --- Top-K銘柄ランキング（メインコンテンツ） ---
    predictions = result.get("predictions")
    code_name_map = _get_code_name_map()

    if predictions is not None and len(predictions) > 0 and isinstance(predictions.index, pd.MultiIndex):
        st.markdown("### 推薦銘柄ランキング")
        st.caption("モデルが「上がりそう」と予測した銘柄の上位。スコアが高いほど有望と判定。")

        from qlib_integration.result_adapter import extract_topk_daily
        topk_val = result.get("topk", 30)
        topk_df = extract_topk_daily(predictions, topk=topk_val)

        # 銘柄名を付与
        topk_df["企業名"] = topk_df["code"].map(
            lambda c: code_name_map.get(c, code_name_map.get(c.lstrip("0"), ""))
        )

        # 日付選択
        available_dates = sorted(topk_df["date"].unique(), reverse=True)
        date_options = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in available_dates]

        selected_date_str = st.selectbox(
            "日付を選択", date_options, index=0,
            help="この日のモデル予測に基づくランキングです。",
        )
        selected_date = pd.Timestamp(selected_date_str)

        day_top = topk_df[topk_df["date"] == selected_date].copy()
        day_top = day_top.rename(columns={
            "rank": "順位",
            "code": "銘柄コード",
            "score": "予測スコア",
        })
        day_top["予測スコア"] = day_top["予測スコア"].round(4)

        st.dataframe(
            day_top[["順位", "銘柄コード", "企業名", "予測スコア"]],
            width="stretch",
            hide_index=True,
        )

    # --- 特徴量重要度 ---
    feat_imp = result.get("feature_importance", {})
    if feat_imp:
        st.markdown("### 特徴量重要度 (Top 30)")
        st.caption("モデルが予測に特に活用した特徴量。値が大きいほど予測への貢献度が高い。")

        imp_df = pd.DataFrame(
            [(_describe_feature(k), v) for k, v in feat_imp.items()],
            columns=["Feature", "Importance"],
        ).sort_values("Importance", ascending=True)

        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            height=max(400, len(imp_df) * 22),
        )
        fig.update_layout(
            margin=dict(l=0, r=20, t=10, b=10),
            yaxis_title="",
            xaxis_title="Gain (重要度)",
        )
        fig.update_traces(marker_color="#FF8000")
        st.plotly_chart(fig, width="stretch")

    # --- 予測スコア分布 ---
    if predictions is not None and len(predictions) > 0:
        with st.expander("予測スコアの分布"):
            st.caption("全銘柄に対するスコアの分布。右寄りの銘柄ほど「上がりそう」と予測されている。")
            pred_values = predictions.values
            pred_values = pred_values[~np.isnan(pred_values)]
            if len(pred_values) > 10000:
                pred_values = np.random.choice(pred_values, 10000, replace=False)

            fig = px.histogram(
                x=pred_values,
                nbins=100,
                labels={"x": "予測スコア", "y": "銘柄数"},
            )
            fig.update_traces(marker_color="#FF8000")
            fig.update_layout(margin=dict(l=0, r=20, t=10, b=10))
            st.plotly_chart(fig, width="stretch")

    # --- 一括コピー ---
    with st.expander("結果を一括コピー"):
        copy_text = _build_qlib_copy_text(result, code_name_map)
        st.code(copy_text, language="text")


if __name__ == "__main__":
    main()
