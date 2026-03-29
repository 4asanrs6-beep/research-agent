"""個別株ペア分析ページ — US→JP イベントスタディ

US個別株が大きく動いた日を起点に、JP個別株の翌日以降の
累積異常リターン（CAR）を計算し、銘柄間の伝播関係を発見する。
"""

import threading
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import JQUANTS_API_KEY, MARKET_DATA_DIR
from core.styles import apply_reuters_style

st.set_page_config(page_title="個別株ペア分析", page_icon="P", layout="wide")


@st.cache_resource
def _get_provider_and_cache():
    from data.cache import DataCache
    from data.jquants_provider import JQuantsProvider
    cache = DataCache(MARKET_DATA_DIR)
    provider = JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache) if JQUANTS_API_KEY else None
    return provider, cache


def _run_event_study_thread(progress_dict, config):
    """イベントスタディをバックグラウンドで実行。"""
    try:
        from core.stock_allocation.event_study import (
            run_event_study, fetch_us_stock_returns, SECTOR_PAIR_GUIDE,
        )

        def on_progress(msg, pct):
            progress_dict["message"] = msg
            progress_dict["pct"] = pct

        provider, cache = _get_provider_and_cache()

        # --- US株リターン取得 ---
        on_progress("US株価取得中...", 0.05)
        sector_pairs = config["sector_pairs"]
        us_tickers = []
        for sp in sector_pairs:
            us_tickers.extend(sp["us_tickers"])
        us_tickers = list(set(us_tickers))

        us_returns = fetch_us_stock_returns(
            us_tickers, config["start_date"], config["end_date"],
            progress_callback=lambda m, p: on_progress(m, 0.05 + p * 0.15),
        )

        if us_returns.empty:
            raise ValueError("US株価取得に失敗")

        # --- JP株リターン取得 ---
        on_progress("JP株価取得中...", 0.25)
        jp_codes = []
        for sp in sector_pairs:
            jp_codes.extend(sp["jp_codes"])
        jp_codes = list(set(jp_codes))

        from core.stock_allocation import _fetch_stock_prices
        jp_prices_raw = _fetch_stock_prices(
            provider, jp_codes, config["start_date"], config["end_date"],
            progress_callback=lambda m, p: on_progress(m, 0.25 + p * 0.25),
        )

        if jp_prices_raw.empty:
            raise ValueError("JP株価取得に失敗")

        close_col = "adj_close" if "adj_close" in jp_prices_raw.columns else "close"
        jp_close = jp_prices_raw.pivot(index="date", columns="code", values=close_col)
        jp_returns = jp_close.pct_change().dropna(how="all")

        # --- 銘柄名マッピング ---
        listed = provider.get_listed_stocks()
        code_names = dict(zip(listed["code"], listed["name"]))

        # --- イベントスタディ実行 ---
        on_progress("イベントスタディ実行中...", 0.55)
        result = run_event_study(
            us_prices=us_returns,
            jp_prices=jp_returns,
            sector_pairs=sector_pairs,
            lags=config.get("lags", [1, 2, 3, 5, 10]),
            event_threshold_sigma=config.get("threshold", 2.0),
            min_events=config.get("min_events", 10),
            isolated_only=config.get("isolated_only", False),
            isolated_window=config.get("isolated_window", 3),
            jp_code_names=code_names,
            progress_callback=lambda m, p: on_progress(m, 0.55 + p * 0.40),
        )

        progress_dict["_result"] = result
        progress_dict["pct"] = 1.0
        progress_dict["message"] = "完了"
    except Exception as e:
        import traceback
        progress_dict["error"] = str(e)
        progress_dict["detail"] = traceback.format_exc()
        progress_dict["pct"] = 1.0


def _run_cv_thread(progress_dict, config):
    """クロスバリデーションをバックグラウンドで実行。"""
    try:
        from core.stock_allocation.event_study import (
            run_cross_validated_scan, fetch_us_stock_returns,
            get_sp500_tickers, get_jp_top_stocks,
        )

        def on_progress(msg, pct):
            progress_dict["message"] = msg
            progress_dict["pct"] = pct

        provider, cache = _get_provider_and_cache()

        n_us = config["n_us"]
        n_jp = config["n_jp"]
        on_progress(f"S&P500上位{n_us}銘柄の株価取得中...", 0.02)
        us_tickers = get_sp500_tickers(n_us)
        us_returns = fetch_us_stock_returns(
            us_tickers, config["start_date"], config["end_date"],
            progress_callback=lambda m, p: on_progress(m, 0.02 + p * 0.10),
        )
        if us_returns.empty:
            raise ValueError("US株価取得に失敗")

        on_progress(f"JP上位{n_jp}銘柄の株価取得中...", 0.15)
        jp_codes, jp_names = get_jp_top_stocks(provider, n=n_jp)

        from core.stock_allocation import _fetch_stock_prices
        jp_prices_raw = _fetch_stock_prices(
            provider, jp_codes, config["start_date"], config["end_date"],
            progress_callback=lambda m, p: on_progress(m, 0.15 + p * 0.20),
        )
        if jp_prices_raw.empty:
            raise ValueError("JP株価取得に失敗")

        close_col = "adj_close" if "adj_close" in jp_prices_raw.columns else "close"
        jp_close = jp_prices_raw.pivot(index="date", columns="code", values=close_col)
        jp_returns = jp_close.pct_change().dropna(how="all")

        listed = provider.get_listed_stocks()
        jp_names_full = dict(zip(listed["code"], listed["name"]))

        on_progress("クロスバリデーション実行中...", 0.40)
        result = run_cross_validated_scan(
            us_prices=us_returns,
            jp_prices=jp_returns,
            sector_pairs=None,
            lags=config.get("lags", [1, 2, 3, 5, 10]),
            event_threshold_sigma=config.get("threshold", 2.0),
            min_events=config.get("min_events", 15),
            fdr_correction=True,
            jp_code_names=jp_names_full,
            progress_callback=lambda m, p: on_progress(m, 0.40 + p * 0.55),
        )

        progress_dict["_result"] = result
        progress_dict["pct"] = 1.0
        progress_dict["message"] = "完了"
    except Exception as e:
        import traceback
        progress_dict["error"] = str(e)
        progress_dict["detail"] = traceback.format_exc()
        progress_dict["pct"] = 1.0


def _run_fullscan_thread(progress_dict, config):
    """フルスキャンをバックグラウンドで実行。"""
    try:
        from core.stock_allocation.event_study import (
            run_event_study, fetch_us_stock_returns,
            get_sp500_tickers, get_jp_top_stocks,
        )

        def on_progress(msg, pct):
            progress_dict["message"] = msg
            progress_dict["pct"] = pct

        provider, cache = _get_provider_and_cache()

        # --- US株リターン取得 ---
        n_us = config["n_us"]
        n_jp = config["n_jp"]
        on_progress(f"S&P500上位{n_us}銘柄の株価取得中...", 0.02)
        us_tickers = get_sp500_tickers(n_us)
        us_returns = fetch_us_stock_returns(
            us_tickers, config["start_date"], config["end_date"],
            progress_callback=lambda m, p: on_progress(m, 0.02 + p * 0.13),
        )
        if us_returns.empty:
            raise ValueError("US株価取得に失敗")
        on_progress(f"US {len(us_returns.columns)}銘柄取得完了", 0.15)

        # --- JP株リターン取得 ---
        on_progress(f"JP上位{n_jp}銘柄の株価取得中...", 0.15)
        jp_codes, jp_names = get_jp_top_stocks(provider, n=n_jp)

        from core.stock_allocation import _fetch_stock_prices
        jp_prices_raw = _fetch_stock_prices(
            provider, jp_codes, config["start_date"], config["end_date"],
            progress_callback=lambda m, p: on_progress(m, 0.15 + p * 0.30),
        )
        if jp_prices_raw.empty:
            raise ValueError("JP株価取得に失敗")

        close_col = "adj_close" if "adj_close" in jp_prices_raw.columns else "close"
        jp_close = jp_prices_raw.pivot(index="date", columns="code", values=close_col)
        jp_returns = jp_close.pct_change().dropna(how="all")
        on_progress(f"JP {len(jp_returns.columns)}銘柄取得完了", 0.45)

        n_pairs = len(us_returns.columns) * len(jp_returns.columns)
        on_progress(f"フルスキャン実行中... ({n_pairs:,}ペア)", 0.50)

        # sector_pairs=None → 全ペアスキャン
        result = run_event_study(
            us_prices=us_returns,
            jp_prices=jp_returns,
            sector_pairs=None,  # 全ペアスキャン
            lags=config.get("lags", [1, 2, 3, 5]),
            event_threshold_sigma=config.get("threshold", 2.0),
            min_events=config.get("min_events", 15),
            isolated_only=config.get("isolated_only", False),
            fdr_correction=True,  # 大規模スキャンではFDR必須
            p_threshold=config.get("p_threshold", 0.05),
            jp_code_names=jp_names,
            progress_callback=lambda m, p: on_progress(m, 0.50 + p * 0.45),
        )

        progress_dict["_result"] = result
        progress_dict["pct"] = 1.0
        progress_dict["message"] = "完了"
    except Exception as e:
        import traceback
        progress_dict["error"] = str(e)
        progress_dict["detail"] = traceback.format_exc()
        progress_dict["pct"] = 1.0


def main():
    apply_reuters_style()

    st.markdown("# 個別株ペア分析")
    st.caption("US個別株のショック → JP個別株への伝播をイベントスタディで分析")

    # スレッド完了検出
    for key_t, key_r, key_p in [
        ("es_thread", "es_result", "es_progress"),
        ("fs_thread", "fs_result", "fs_progress"),
        ("cv_thread", "cv_result", "cv_progress"),
    ]:
        thread = st.session_state.get(key_t)
        if thread is not None and not thread.is_alive() and key_r not in st.session_state:
            prog = st.session_state.get(key_p, {})
            if "_result" in prog:
                st.session_state[key_r] = prog.pop("_result")

    tab_setup, tab_results, tab_fullscan, tab_cv, tab_network = st.tabs([
        "セクター指定", "結果", "フルスキャン", "クロスバリデーション", "ネットワーク",
    ])

    with tab_setup:
        _render_setup_tab()

    with tab_results:
        _render_results_tab()

    with tab_fullscan:
        _render_fullscan_tab()

    with tab_cv:
        _render_cv_tab()

    with tab_network:
        _render_network_tab()


def _render_setup_tab():
    from core.stock_allocation.event_study import SECTOR_PAIR_GUIDE

    st.subheader("分析設定")

    # --- セクターペア選択 ---
    st.markdown("#### セクターペア")
    st.caption("US業種 → JP業種の対応を選択。選択したペア内の銘柄同士をテストします。")

    provider, _ = _get_provider_and_cache()

    # JP側の銘柄を sector_33_name で取得
    jp_stocks_by_sector = {}
    if provider:
        listed = provider.get_listed_stocks()
        for _, row in listed.iterrows():
            sec33 = row.get("sector_33_name", "")
            if pd.isna(sec33) or not sec33:
                continue
            jp_stocks_by_sector.setdefault(sec33, []).append({
                "code": row["code"],
                "name": row.get("name", row["code"]),
            })

    selected_pairs = []
    for i, guide in enumerate(SECTOR_PAIR_GUIDE):
        col1, col2, col3 = st.columns([1, 3, 3])
        with col1:
            enabled = st.checkbox(guide["label"], value=True, key=f"sp_{i}")
        if not enabled:
            continue

        with col2:
            us_tickers = st.multiselect(
                f"US銘柄 ({guide['label']})",
                guide["us_tickers"],
                default=guide["us_tickers"][:5],
                key=f"us_{i}",
            )
        with col3:
            # JP側: hint_codes + sector_33に属する銘柄
            jp_options = list(guide.get("jp_hint_codes", []))
            for sec33 in guide.get("jp_sector_33", []):
                for stock in jp_stocks_by_sector.get(sec33, []):
                    if stock["code"] not in jp_options:
                        jp_options.append(stock["code"])
            jp_defaults = guide.get("jp_hint_codes", jp_options[:5])
            jp_codes = st.multiselect(
                f"JP銘柄 ({guide['label']})",
                jp_options,
                default=[c for c in jp_defaults if c in jp_options],
                key=f"jp_{i}",
            )

        if us_tickers and jp_codes:
            selected_pairs.append({
                "label": guide["label"],
                "us_tickers": us_tickers,
                "jp_codes": jp_codes,
            })

    # --- パラメータ ---
    st.markdown("#### パラメータ")
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("開始日", value=date(2020, 1, 1), key="es_start")
        end_date = st.date_input("終了日", value=date.today(), key="es_end")
    with col2:
        threshold = st.slider("イベント閾値 (σ)", 1.0, 4.0, 2.0, 0.5, key="es_threshold")
        min_events = st.number_input("最低イベント数", 5, 50, 10, key="es_min_events")
    with col3:
        isolated_only = st.checkbox("孤立イベントのみ", value=False, key="es_isolated")
        if isolated_only:
            isolated_window = st.number_input("孤立判定ウィンドウ (日)", 1, 10, 3, key="es_iso_window")
        else:
            isolated_window = 3
        lags_str = st.text_input("テストするラグ", "1,2,3,5,10", key="es_lags")

    lags = [int(x.strip()) for x in lags_str.split(",") if x.strip().isdigit()]

    # --- 実行 ---
    n_us = sum(len(sp["us_tickers"]) for sp in selected_pairs)
    n_jp = sum(len(sp["jp_codes"]) for sp in selected_pairs)
    n_pairs = sum(len(sp["us_tickers"]) * len(sp["jp_codes"]) for sp in selected_pairs)
    st.caption(f"US {n_us}銘柄 × JP {n_jp}銘柄 = {n_pairs}ペア")

    if st.button("イベントスタディを実行", type="primary", key="run_es"):
        if not selected_pairs:
            st.error("セクターペアを1つ以上選択してください")
            return

        config = {
            "sector_pairs": selected_pairs,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "threshold": threshold,
            "min_events": min_events,
            "isolated_only": isolated_only,
            "isolated_window": isolated_window,
            "lags": lags,
        }

        progress = {"message": "開始...", "pct": 0.0}
        st.session_state["es_progress"] = progress
        st.session_state.pop("es_result", None)

        t = threading.Thread(
            target=_run_event_study_thread,
            args=(progress, config),
            daemon=True,
        )
        st.session_state["es_thread"] = t
        t.start()
        st.rerun()

    # 進捗表示
    es_thread = st.session_state.get("es_thread")
    if es_thread is not None and es_thread.is_alive():
        prog = st.session_state.get("es_progress", {})
        import time
        st.progress(prog.get("pct", 0.0))
        st.caption(prog.get("message", "実行中..."))
        time.sleep(1)
        st.rerun()

    prog = st.session_state.get("es_progress", {})
    if "error" in prog:
        st.error(f"エラー: {prog['error']}")
        if "detail" in prog:
            with st.expander("詳細"):
                st.code(prog["detail"])


def _render_results_tab():
    result = st.session_state.get("es_result")
    if result is None:
        st.info("「設定・実行」タブでイベントスタディを実行してください。")
        return

    st.subheader(f"結果: {len(result.significant_pairs)}件の有意なペア / {len(result.pairs)}件中")
    st.caption(
        f"イベント閾値: {result.event_threshold}σ | "
        f"孤立イベントのみ: {'Yes' if result.isolated_only else 'No'} | "
        f"ラグ: {result.lags_tested}"
    )

    if not result.significant_pairs:
        st.warning("有意なペアが見つかりませんでした。閾値を下げるか期間を広げてください。")
        return

    # --- サマリーテーブル ---
    from core.stock_allocation.event_study import build_summary_table
    df = build_summary_table(result, top_n=50)

    st.markdown("#### 有意ペア一覧")

    # 表示列を選択
    display_cols = ["セクター", "US銘柄", "JP銘柄", "イベント数", "方向一致率", "ピークラグ", "ピークCAR"]
    for lag in result.lags_tested:
        display_cols.append(f"CAR(t+{lag})")
    available = [c for c in display_cols if c in df.columns]

    format_dict = {"方向一致率": "{:.1%}", "ピークCAR": "{:+.3%}"}
    for lag in result.lags_tested:
        col = f"CAR(t+{lag})"
        if col in df.columns:
            format_dict[col] = "{:+.3%}"

    st.dataframe(
        df[available].style.format(format_dict, na_rep="-"),
        height=min(800, 35 * len(df) + 40),
    )

    # --- インパルス応答チャート ---
    st.markdown("#### インパルス応答曲線 (CAR推移)")

    top_pairs = result.significant_pairs[:10]
    pair_labels = [f"{p.us_ticker}→{p.jp_name}" for p in top_pairs]

    selected_pair_idx = st.selectbox(
        "ペアを選択",
        range(len(top_pairs)),
        format_func=lambda i: pair_labels[i],
        key="es_pair_select",
    )

    pair = top_pairs[selected_pair_idx]

    fig = go.Figure()
    lags_sorted = sorted(pair.car.keys())
    cars = [pair.car[l] * 100 for l in lags_sorted]
    p_vals = [pair.car_p_value.get(l, 1.0) for l in lags_sorted]

    # CAR曲線
    fig.add_trace(go.Scatter(
        x=[f"t+{l}" for l in lags_sorted],
        y=cars,
        mode="lines+markers",
        name="CAR (%)",
        line={"width": 3},
        marker={"size": 10, "color": ["green" if p < 0.05 else "gray" for p in p_vals]},
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"{pair.us_ticker} → {pair.jp_name} ({pair.jp_code}) | イベント数: {pair.n_events} | 方向一致率: {pair.direction_hit_rate:.0%}",
        yaxis_title="累積異常リターン (%)",
        xaxis_title="ラグ",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # p値テーブル
    p_df = pd.DataFrame({
        "ラグ": [f"t+{l}" for l in lags_sorted],
        "CAR (%)": [f"{pair.car[l]*100:+.2f}" for l in lags_sorted],
        "t統計量": [f"{pair.car_t_stat.get(l, 0):.2f}" for l in lags_sorted],
        "p値": [f"{pair.car_p_value.get(l, 1):.4f}" for l in lags_sorted],
        "有意": ["*" if pair.car_p_value.get(l, 1) < 0.05 else "" for l in lags_sorted],
    })
    st.dataframe(p_df, hide_index=True)

    # --- セクター別集計 ---
    st.markdown("#### セクター別集計")
    sector_summary = {}
    for p in result.significant_pairs:
        label = p.sector_label or "その他"
        if label not in sector_summary:
            sector_summary[label] = {"count": 0, "avg_peak_car": [], "avg_hit_rate": []}
        sector_summary[label]["count"] += 1
        sector_summary[label]["avg_peak_car"].append(p.peak_car)
        sector_summary[label]["avg_hit_rate"].append(p.direction_hit_rate)

    sec_rows = []
    for label, data in sector_summary.items():
        sec_rows.append({
            "セクター": label,
            "有意ペア数": data["count"],
            "平均ピークCAR": np.mean(data["avg_peak_car"]),
            "平均方向一致率": np.mean(data["avg_hit_rate"]),
        })
    if sec_rows:
        sec_df = pd.DataFrame(sec_rows).sort_values("有意ペア数", ascending=False)
        st.dataframe(
            sec_df.style.format({"平均ピークCAR": "{:+.3%}", "平均方向一致率": "{:.1%}"}),
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# フルスキャンタブ
# ---------------------------------------------------------------------------
def _render_fullscan_tab():
    st.subheader("フルスキャン — US×JP 大規模ペア探索")
    st.caption(
        "S&P500上位 × JP大型株の全組み合わせをスキャン。"
        "Benjamini-Hochberg FDR補正で偽発見を制御。"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_us = st.selectbox("US銘柄数", [50, 100, 150, 200], index=1, key="fs_n_us")
        n_jp = st.selectbox("JP銘柄数", [100, 200, 300, 500], index=1, key="fs_n_jp")
    with col2:
        fs_start = st.date_input("開始日", value=date(2020, 1, 1), key="fs_start")
        fs_end = st.date_input("終了日", value=date.today(), key="fs_end")
    with col3:
        fs_threshold = st.slider("イベント閾値 (σ)", 1.5, 3.0, 2.0, 0.5, key="fs_threshold")
        fs_min_events = st.number_input("最低イベント数", 10, 30, 15, key="fs_min_events")

    n_pairs = n_us * n_jp
    st.caption(f"**{n_us} × {n_jp} = {n_pairs:,}ペア** をスキャン（FDR補正適用）")

    if st.button("フルスキャン実行", type="primary", key="run_fs"):
        config = {
            "n_us": n_us,
            "n_jp": n_jp,
            "start_date": fs_start.strftime("%Y-%m-%d"),
            "end_date": fs_end.strftime("%Y-%m-%d"),
            "threshold": fs_threshold,
            "min_events": fs_min_events,
            "lags": [1, 2, 3, 5, 10],
        }

        progress = {"message": "開始...", "pct": 0.0}
        st.session_state["fs_progress"] = progress
        st.session_state.pop("fs_result", None)

        t = threading.Thread(
            target=_run_fullscan_thread,
            args=(progress, config),
            daemon=True,
        )
        st.session_state["fs_thread"] = t
        t.start()
        st.rerun()

    # 進捗
    fs_thread = st.session_state.get("fs_thread")
    if fs_thread is not None and fs_thread.is_alive():
        prog = st.session_state.get("fs_progress", {})
        import time
        st.progress(prog.get("pct", 0.0))
        st.caption(prog.get("message", "実行中..."))
        time.sleep(2)
        st.rerun()

    # スレッド完了検出（タブ内でも再チェック）
    fs_thread2 = st.session_state.get("fs_thread")
    if fs_thread2 is not None and not fs_thread2.is_alive() and "fs_result" not in st.session_state:
        prog2 = st.session_state.get("fs_progress", {})
        if "_result" in prog2:
            st.session_state["fs_result"] = prog2.pop("_result")
            st.rerun()

    prog = st.session_state.get("fs_progress", {})
    if "error" in prog:
        st.error(f"エラー: {prog['error']}")
        if "detail" in prog:
            with st.expander("詳細"):
                st.code(prog["detail"])
        return

    # 結果
    result = st.session_state.get("fs_result")
    if result is None:
        return

    st.markdown("---")
    st.subheader(f"フルスキャン結果: {len(result.significant_pairs)}件 (FDR補正後) / {len(result.pairs)}件中")

    if not result.significant_pairs:
        st.warning("FDR補正後に有意なペアがありませんでした。閾値を下げるか期間を広げてください。")
        return

    from core.stock_allocation.event_study import build_summary_table
    df = build_summary_table(result, top_n=100)

    display_cols = ["US銘柄", "JP銘柄", "イベント数", "方向一致率", "ピークラグ", "ピークCAR"]
    for lag in result.lags_tested:
        display_cols.append(f"CAR(t+{lag})")
    available = [c for c in display_cols if c in df.columns]

    format_dict = {"方向一致率": "{:.1%}", "ピークCAR": "{:+.3%}"}
    for lag in result.lags_tested:
        col_name = f"CAR(t+{lag})"
        if col_name in df.columns:
            format_dict[col_name] = "{:+.3%}"

    st.dataframe(
        df[available].style.format(format_dict, na_rep="-"),
        height=min(800, 35 * len(df) + 40),
    )

    # インパルス応答チャート
    if result.significant_pairs:
        st.markdown("#### インパルス応答曲線")
        top_pairs = result.significant_pairs[:20]
        pair_labels = [f"{p.us_ticker}→{p.jp_name}" for p in top_pairs]
        idx = st.selectbox("ペア", range(len(top_pairs)), format_func=lambda i: pair_labels[i], key="fs_pair")
        pair = top_pairs[idx]

        fig = go.Figure()
        lags_sorted = sorted(pair.car.keys())
        cars = [pair.car[l] * 100 for l in lags_sorted]
        p_vals = [pair.car_p_value.get(l, 1.0) for l in lags_sorted]

        fig.add_trace(go.Scatter(
            x=[f"t+{l}" for l in lags_sorted], y=cars,
            mode="lines+markers", name="CAR (%)",
            line={"width": 3},
            marker={"size": 10, "color": ["green" if p < 0.05 else "gray" for p in p_vals]},
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title=f"{pair.us_ticker} → {pair.jp_name} | events={pair.n_events} | hit={pair.direction_hit_rate:.0%}",
            yaxis_title="CAR (%)", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# ネットワークタブ
# ---------------------------------------------------------------------------
def _render_cv_tab():
    st.subheader("クロスバリデーション — 前半発見 × 後半検証")
    st.caption(
        "全期間を前半(発見)と後半(検証)に分割。前半で有意なペアが後半でも再現するかテスト。"
        "両期間で有意なペアのみ「検証済み」として残す。"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        cv_n_us = st.selectbox("US銘柄数", [50, 100, 150], index=1, key="cv_n_us")
        cv_n_jp = st.selectbox("JP銘柄数", [100, 200, 300], index=1, key="cv_n_jp")
    with col2:
        cv_start = st.date_input("開始日", value=date(2018, 1, 1), key="cv_start")
        cv_end = st.date_input("終了日", value=date.today(), key="cv_end")
    with col3:
        cv_threshold = st.slider("イベント閾値 (σ)", 1.5, 3.0, 2.0, 0.5, key="cv_threshold")
        cv_min_events = st.number_input("最低イベント数(全期間)", 10, 30, 15, key="cv_min_events")

    n_pairs = cv_n_us * cv_n_jp
    st.caption(f"**{cv_n_us} × {cv_n_jp} = {n_pairs:,}ペア** | 前半で発見 → 後半で検証（FDR補正適用）")

    if st.button("クロスバリデーション実行", type="primary", key="run_cv"):
        config = {
            "n_us": cv_n_us,
            "n_jp": cv_n_jp,
            "start_date": cv_start.strftime("%Y-%m-%d"),
            "end_date": cv_end.strftime("%Y-%m-%d"),
            "threshold": cv_threshold,
            "min_events": cv_min_events,
            "lags": [1, 2, 3, 5, 10],
        }
        progress = {"message": "開始...", "pct": 0.0}
        st.session_state["cv_progress"] = progress
        st.session_state.pop("cv_result", None)
        t = threading.Thread(target=_run_cv_thread, args=(progress, config), daemon=True)
        st.session_state["cv_thread"] = t
        t.start()
        st.rerun()

    # 進捗
    cv_thread = st.session_state.get("cv_thread")
    if cv_thread is not None and cv_thread.is_alive():
        prog = st.session_state.get("cv_progress", {})
        import time
        st.progress(prog.get("pct", 0.0))
        st.caption(prog.get("message", "実行中..."))
        time.sleep(2)
        st.rerun()

    # スレッド完了検出
    cv_thread2 = st.session_state.get("cv_thread")
    if cv_thread2 is not None and not cv_thread2.is_alive() and "cv_result" not in st.session_state:
        prog2 = st.session_state.get("cv_progress", {})
        if "_result" in prog2:
            st.session_state["cv_result"] = prog2.pop("_result")
            st.rerun()

    prog = st.session_state.get("cv_progress", {})
    if "error" in prog:
        st.error(f"エラー: {prog['error']}")
        if "detail" in prog:
            with st.expander("詳細"):
                st.code(prog["detail"])
        return

    result = st.session_state.get("cv_result")
    if result is None:
        return

    # --- 結果表示 ---
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("前半で発見", f"{result.n_discovery_significant}件")
    with col2:
        st.metric("後半で再現", f"{result.n_validated}件")
    with col3:
        st.metric("再現率", f"{result.reproduction_rate:.0%}")

    st.caption(f"分割日: {result.split_date}")

    if not result.validated_pairs:
        st.warning("後半で再現するペアがありませんでした。期間を広げるか閾値を下げてください。")
        return

    # 検証済みペア一覧
    st.markdown("#### 検証済みペア")
    from core.stock_allocation.event_study import build_cv_summary_table
    df = build_cv_summary_table(result, top_n=50)

    format_dict = {
        "平均CAR": "{:+.3%}",
        "発見CAR": "{:+.3%}", "発見p": "{:.4f}", "発見一致率": "{:.1%}",
        "検証CAR": "{:+.3%}", "検証p": "{:.4f}", "検証一致率": "{:.1%}",
    }
    st.dataframe(
        df.style.format(format_dict, na_rep="-"),
        height=min(800, 35 * len(df) + 40),
    )

    # 前半/後半 CAR比較チャート
    if result.validated_pairs:
        st.markdown("#### 前半 vs 後半 CAR比較")
        top = result.validated_pairs[:15]
        labels = [f"{v.us_ticker}→{v.jp_name}" for v in top]
        disc_cars = [v.discovery_car.get(v.peak_lag, 0) * 100 for v in top]
        val_cars = [v.validation_car.get(v.peak_lag, 0) * 100 for v in top]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="前半(発見)", x=labels, y=disc_cars, marker_color="#2196F3"))
        fig.add_trace(go.Bar(name="後半(検証)", x=labels, y=val_cars, marker_color="#FF5722"))
        fig.update_layout(
            title="検証済みペア: 前半 vs 後半 CAR (%)",
            barmode="group", height=400, yaxis_title="CAR (%)",
            xaxis_tickangle=45,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)


def _render_network_tab():
    st.subheader("US → JP 銘柄ネットワーク")
    st.caption("イベントスタディで有意なペアをネットワークグラフとして可視化。ノードの大きさ=接続数、線の太さ=CAR強度。")

    # CV結果、フルスキャン結果、セクター指定結果のいずれかを使う
    cv_result = st.session_state.get("cv_result")
    result = st.session_state.get("fs_result") or st.session_state.get("es_result")

    # CV結果がある場合、validated_pairsからEventStudyResult互換オブジェクトを作る
    if cv_result and cv_result.validated_pairs:
        from core.stock_allocation.event_study import PairResult, EventStudyResult
        sig_pairs = []
        for vp in cv_result.validated_pairs:
            sig_pairs.append(PairResult(
                us_ticker=vp.us_ticker,
                jp_code=vp.jp_code,
                jp_name=vp.jp_name,
                sector_label=vp.sector_label,
                car=vp.validation_car,
                car_p_value=vp.validation_p,
                peak_lag=vp.peak_lag,
                peak_car=vp.avg_car,
                direction_hit_rate=vp.validation_hit_rate,
                n_events=vp.validation_n_events,
            ))
        result = EventStudyResult(
            pairs=sig_pairs,
            significant_pairs=sig_pairs,
            lags_tested=cv_result.lags_tested,
        )
        st.caption(f"クロスバリデーション検証済み {len(sig_pairs)}ペアを表示中")
    elif result is None:
        st.info("先に「セクター指定」「フルスキャン」「クロスバリデーション」のいずれかを実行してください。")
        return

    if not result.significant_pairs:
        st.warning("有意なペアがありません。")
        return

    from core.stock_allocation.network_analysis import (
        build_network, create_network_figure, build_hub_table,
    )

    # フィルタ
    st.markdown("#### フィルタ")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        min_hit = st.slider("方向一致率", 0.5, 1.0, 0.75, 0.05, key="net_min_hit")
    with col_f2:
        min_degree = st.slider("最低接続数", 1, 10, 2, 1, key="net_min_degree")
    with col_f3:
        max_edges = st.slider("表示エッジ上限", 10, 100, 30, 5, key="net_max_edges")

    # フィルタ適用: まず方向一致率で絞る
    filtered_pairs = [p for p in result.significant_pairs if p.direction_hit_rate >= min_hit]
    # 上限でカット（スコア上位）
    filtered_pairs = filtered_pairs[:max_edges]

    # 仮のresultを作ってbuild_network
    from core.stock_allocation.event_study import EventStudyResult
    filtered_result = EventStudyResult(
        pairs=filtered_pairs,
        significant_pairs=filtered_pairs,
        lags_tested=result.lags_tested,
    )
    network = build_network(filtered_result)

    # 最低接続数でノードを絞る
    if min_degree > 1:
        keep_ids = {n.id for n in network.nodes if n.degree >= min_degree}
        network.edges = [e for e in network.edges if e.source in keep_ids and e.target in keep_ids]
        network.nodes = [n for n in network.nodes if n.id in keep_ids]

    if not network.edges:
        st.warning("条件を満たすペアがありません。フィルタを緩めてください。")
        return

    n_us = len([n for n in network.nodes if n.side == "US"])
    n_jp = len([n for n in network.nodes if n.side == "JP"])
    st.caption(f"US {n_us}銘柄 + JP {n_jp}銘柄 | {len(network.edges)}本のエッジ")

    # --- ネットワークグラフ ---
    n_nodes = len(network.nodes)
    fig_height = max(500, min(1000, n_nodes * 25 + 100))
    fig = create_network_figure(network, height=fig_height)
    st.plotly_chart(fig, use_container_width=True)

    # --- ハブランキング ---
    st.markdown("#### 影響力ランキング")
    us_hub_df, jp_hub_df = build_hub_table(network)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**US（影響元）**")
        if not us_hub_df.empty:
            st.dataframe(
                us_hub_df.style.format({"加重強度": "{:.3f}"}),
                hide_index=True, height=min(400, 35 * len(us_hub_df) + 40),
            )
    with col2:
        st.markdown("**JP（影響先）**")
        if not jp_hub_df.empty:
            st.dataframe(
                jp_hub_df.style.format({"加重強度": "{:.3f}"}),
                hide_index=True, height=min(400, 35 * len(jp_hub_df) + 40),
            )

    # --- クラスター ---
    if network.clusters:
        st.markdown("#### クラスター（連結成分）")
        for i, cl in enumerate(network.clusters[:10]):
            sectors = ", ".join(cl["sectors"]) if cl["sectors"] else "混合"
            with st.expander(f"クラスター {i+1}: {sectors} ({cl['n_edges']}本)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**US**")
                    for name in cl["us"]:
                        st.text(f"  {name}")
                with col2:
                    st.markdown("**JP**")
                    for name in cl["jp"]:
                        st.text(f"  {name}")


main()
