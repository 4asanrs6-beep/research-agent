"""スター株リバースエンジニアリング分析ページ

直近1年で異常上昇した銘柄の共通特徴を定量分析し、
海外投資家フロー仮説を検証する。
"""

import json
import threading
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import (
    DB_PATH, MARKET_DATA_DIR, JQUANTS_API_KEY,
    STAR_STOCK_DEFAULTS,
)
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider
from core.ai_client import create_ai_client
from core.star_stock_analyzer import (
    StarStockAnalyzer, StarStockConfig, StarStockResult,
    _MULTI_ONSET_SIGNAL_NAMES, _ONSET_SIGNAL_SHORT,
    WIDE_FEATURE_KEYS, _WIDE_FEATURE_LABELS_JP,
)
from core.styles import apply_reuters_style, apply_waiting_overlay
from core.sidebar import render_sidebar_running_indicator

st.set_page_config(page_title="スター株分析", page_icon="R", layout="wide")

D = STAR_STOCK_DEFAULTS


def _normalize_code(code: str) -> str:
    """4桁コードを5桁に正規化する（J-Quants APIは5桁コードを使用）。
    例: '6920' → '69200', '285A' → '285A0', '69200' → '69200'
    """
    c = code.strip()
    if len(c) == 4 and c.isalnum():
        return c + "0"
    return c


@st.cache_resource
def get_data_provider():
    cache = DataCache(MARKET_DATA_DIR)
    return JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)


# ---------------------------------------------------------------------------
# バックグラウンドスレッド
# ---------------------------------------------------------------------------
def _run_ss_thread(
    progress_dict: dict,
    provider,
    config: StarStockConfig,
):
    """スター株分析をバックグラウンドで実行する"""
    try:
        ai_client = create_ai_client()
        analyzer = StarStockAnalyzer(data_provider=provider, ai_client=ai_client)

        def on_progress(msg: str, pct: float):
            progress_dict["message"] = msg
            progress_dict["pct"] = pct

        result = analyzer.run_analysis(config=config, on_progress=on_progress)
        progress_dict["_result"] = result
        progress_dict["pct"] = 1.0
        progress_dict["message"] = "完了"
    except Exception as e:
        progress_dict["error"] = str(e)
        progress_dict["pct"] = 1.0


# ---------------------------------------------------------------------------
# メインページ
# ---------------------------------------------------------------------------
def main():
    apply_reuters_style()
    render_sidebar_running_indicator()

    st.markdown("# Star Stock Analysis")
    st.caption("直近1年の異常上昇銘柄を逆引き分析 -- 共通特徴の抽出と海外フロー仮説の検証")

    # 状態判定
    thread = st.session_state.get("ss_thread")
    is_running = thread is not None and thread.is_alive()

    # スレッド完了後: 結果昇格
    if thread is not None and not thread.is_alive() and "ss_result" not in st.session_state:
        prog = st.session_state.get("ss_progress", {})
        if "_result" in prog:
            st.session_state["ss_result"] = prog.pop("_result")
        elif prog.get("error"):
            st.session_state["ss_result"] = StarStockResult(
                config={}, star_stocks=[], topix_return=0, n_auto_detected=0,
                n_user_specified=0, error=prog["error"],
            )

    has_result = "ss_result" in st.session_state

    if is_running:
        apply_waiting_overlay()
        _progress_fragment()
        return
    if has_result:
        _show_result()
        return
    _show_input_form()


# ---------------------------------------------------------------------------
# 進捗表示
# ---------------------------------------------------------------------------
@st.fragment(run_every=2)
def _progress_fragment():
    thread = st.session_state.get("ss_thread")
    if thread is not None and not thread.is_alive():
        prog = st.session_state.get("ss_progress", {})
        if "_result" in prog:
            st.session_state["ss_result"] = prog.pop("_result")
        elif prog.get("error"):
            st.session_state["ss_result"] = StarStockResult(
                config={}, star_stocks=[], topix_return=0, n_auto_detected=0,
                n_user_specified=0, error=prog["error"],
            )
        st.rerun(scope="app")
        return

    progress = st.session_state.get("ss_progress", {})
    start_time = st.session_state.get("ss_start_time")

    pct = progress.get("pct", 0.05)
    msg = progress.get("message", "開始中...")

    elapsed_str = ""
    if start_time:
        elapsed = datetime.now() - start_time
        total_sec = int(elapsed.total_seconds())
        mm, ss = divmod(total_sec, 60)
        elapsed_str = f"{mm:02d}:{ss:02d}"

    st.progress(min(pct, 0.99))
    st.markdown(
        f"**{msg}** &nbsp; <span style='color:#999;font-size:0.9em;'>{elapsed_str}</span>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 入力フォーム
# ---------------------------------------------------------------------------
def _show_input_form():
    # 銘柄コード入力
    user_codes_text = st.text_input(
        "銘柄コード（カンマ区切り、任意）",
        placeholder="6920, 5803, 7203",
        key="ss_user_codes",
        help="分析したい銘柄コードを入力（4桁でも5桁でもOK）。空欄の場合は自動検出のみ。",
    )

    # 企業名ルックアップ
    if user_codes_text.strip():
        codes_raw = [c.strip() for c in user_codes_text.split(",") if c.strip()]
        codes_normalized = [_normalize_code(c) for c in codes_raw]
        if codes_normalized:
            try:
                provider = get_data_provider()
                listed = provider.get_listed_stocks()
                matched = listed[listed["code"].astype(str).isin(codes_normalized)]
                if not matched.empty:
                    names = matched[["code", "name"]].drop_duplicates()
                    st.caption(" / ".join(f"{r['code']}: {r['name']}" for _, r in names.iterrows()))
                else:
                    st.caption("該当する銘柄が見つかりません。コードを確認してください。")
            except Exception:
                pass

    # 分析期間
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "分析開始日", value=date.today() - timedelta(days=365),
            key="ss_start_date",
            help="この日からの株価データを使って分析します。通常は1年前がおすすめです。",
        )
    with col2:
        end_date = st.date_input(
            "分析終了日", value=date.today(),
            key="ss_end_date",
            help="この日までの株価データを使って分析します。",
        )

    # 検出パラメータ
    with st.expander("検出パラメータ", expanded=False):
        st.caption("スター株として検出する条件を設定します。条件を満たす銘柄が自動的に検出されます。")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            min_return = st.number_input(
                "最低トータルリターン", min_value=0.0, max_value=10.0,
                value=D["min_total_return"], step=0.1, format="%.2f",
                key="ss_min_return",
                help="分析期間中に株価が何%以上上昇した銘柄をスター株とするか。0.50 = 50%以上の上昇。",
            )
        with pc2:
            min_excess = st.number_input(
                "最低超過リターン", min_value=0.0, max_value=10.0,
                value=D["min_excess_return"], step=0.1, format="%.2f",
                key="ss_min_excess",
                help="TOPIX（市場平均）を何%以上上回った銘柄を対象にするか。0.30 = 市場平均+30%以上。",
            )
        with pc3:
            min_vol_ratio = st.number_input(
                "最低出来高増加比", min_value=1.0, max_value=20.0,
                value=D["min_volume_increase_ratio"], step=0.1,
                key="ss_min_vol",
                help="期間後半の平均出来高が前半の何倍以上か。1.5 = 1.5倍以上に増加した銘柄。注目度が高まった証拠。",
            )

        auto_detect = st.checkbox(
            "自動検出を有効化", value=True, key="ss_auto_detect",
            help="ONにすると、上記の条件を満たす銘柄を全市場から自動的に探します。OFFにすると手動で指定した銘柄のみ分析します。",
        )
        max_auto = st.number_input(
            "自動検出上限数", min_value=5, max_value=200,
            value=D["max_auto_detect"], step=5,
            key="ss_max_auto",
            help="自動検出する銘柄の最大数。多すぎると分析時間が長くなります。",
        )

    # 仕手株フィルター
    with st.expander("仕手株フィルター", expanded=False):
        st.caption("値動きが不自然な「仕手株」を除外するためのフィルターです。自動検出された銘柄にのみ適用されます。")
        fc1, fc2 = st.columns(2)
        with fc1:
            min_cap = st.number_input(
                "最低時価総額（億円）", min_value=0.0, max_value=10000.0,
                value=D["min_market_cap_billion"], step=10.0,
                key="ss_min_cap",
                help="この金額未満の小型株を除外します。小型株は少額で株価操作されやすいため。50 = 時価総額50億円未満を除外。",
            )
            max_dd = st.number_input(
                "最大下落率（高値→終値）", min_value=0.0, max_value=1.0,
                value=D["max_drawdown_from_peak"], step=0.05, format="%.2f",
                key="ss_max_dd",
                help="期間中の最高値から最終日までの下落率がこの値を超えると除外。0.40 = 高値から40%以上下げた銘柄を除外（急騰急落の仕手パターン）。",
            )
        with fc2:
            max_day_ret = st.number_input(
                "1日最大騰落率", min_value=0.0, max_value=1.0,
                value=D["max_single_day_return"], step=0.05, format="%.2f",
                key="ss_max_day",
                help="1日で株価がこの率以上動いた銘柄を除外。0.20 = 1日で20%以上動いた銘柄を除外（仕手的な急騰を排除）。",
            )
            min_up_days = st.number_input(
                "最低上昇日比率", min_value=0.0, max_value=1.0,
                value=D["min_up_days_ratio"], step=0.05, format="%.2f",
                key="ss_min_up",
                help="全取引日のうち上昇した日の割合がこの値未満なら除外。0.45 = 45%未満を除外。健全な上昇株は上昇日が多い傾向。",
            )
        require_pos = st.checkbox(
            "終値が開始値以上を要求", value=D["require_positive_end"],
            key="ss_require_pos",
            help="ONにすると、期間終了時の株価が開始時より低い銘柄を除外します（途中で急騰しても最終的に下がった株を排除）。",
        )

    # 高度分析パラメータ
    with st.expander("高度分析パラメータ", expanded=False):
        st.caption("分析の詳細設定です。通常はデフォルト値のままで問題ありません。")
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            n_clusters = st.number_input(
                "クラスター数", min_value=2, max_value=10,
                value=D["n_clusters"], key="ss_n_clusters",
                help="スター株をいくつのグループに分類するか。銘柄数が少ない場合は2-3がおすすめ。多すぎると各グループの銘柄が少なくなります。",
            )
        with ac2:
            factor_window = st.number_input(
                "ファクター窓（日）", min_value=20, max_value=252,
                value=D["factor_window"], key="ss_factor_win",
                help="ファクター分解に使うローリング期間（日数）。60日 = 約3ヶ月の移動窓で分析。長いほど安定しますが変化への反応が遅れます。",
            )
        with ac3:
            lead_lag_max = st.number_input(
                "Lead-Lag最大ラグ（日）", min_value=1, max_value=30,
                value=D["lead_lag_max_lag"], key="ss_ll_lag",
                help="銘柄間の先行・追随関係を最大何日前まで調べるか。10 = 最大10営業日前の動きが影響しているかを検証。",
            )

    st.markdown("---")

    # 実行ボタン
    if st.button("分析開始", type="primary", use_container_width=True):
        user_codes = []
        if user_codes_text.strip():
            user_codes = [_normalize_code(c) for c in user_codes_text.split(",") if c.strip()]

        config = StarStockConfig(
            min_total_return=min_return,
            min_excess_return=min_excess,
            min_volume_increase_ratio=min_vol_ratio,
            auto_detect_enabled=auto_detect,
            max_auto_detect=max_auto,
            user_codes=user_codes,
            start_date=str(start_date),
            end_date=str(end_date),
            min_market_cap_billion=min_cap,
            max_drawdown_from_peak=max_dd,
            max_single_day_return=max_day_ret,
            min_up_days_ratio=min_up_days,
            require_positive_end=require_pos,
            n_clusters=n_clusters,
            factor_window=factor_window,
            lead_lag_max_lag=lead_lag_max,
        )

        progress_dict = {"pct": 0.0, "message": "開始中..."}
        st.session_state["ss_progress"] = progress_dict
        st.session_state["ss_start_time"] = datetime.now()
        st.session_state.pop("ss_result", None)

        t = threading.Thread(
            target=_run_ss_thread,
            args=(progress_dict, get_data_provider(), config),
            daemon=True,
        )
        st.session_state["ss_thread"] = t
        t.start()
        st.rerun()


# ---------------------------------------------------------------------------
# ヘルパー: 銘柄名表示フォーマット
# ---------------------------------------------------------------------------
def _fmt(s: dict) -> str:
    """コード + 企業名の短縮表示"""
    name = s.get("name", "")
    return f"{s['code']} {name}" if name else s["code"]


def _fmt_code_name(code: str, stocks: list[dict]) -> str:
    """コードから企業名付き表示を返す"""
    for s in stocks:
        if s["code"] == code:
            return _fmt(s)
    return code


# ---------------------------------------------------------------------------
# 結果表示
# ---------------------------------------------------------------------------
def _show_result():
    result: StarStockResult = st.session_state["ss_result"]

    if result.error:
        st.error(f"分析エラー: {result.error}")
        if st.button("新しい分析を開始"):
            _clear_state()
            st.rerun()
        return

    st.success(
        f"分析完了 -- スター株 {len(result.star_stocks)}銘柄 "
        f"（自動検出 {result.n_auto_detected} / ユーザー指定 {result.n_user_specified}） "
        f"TOPIX {result.topix_return:.1%}"
    )

    # 8タブ
    tabs = st.tabs([
        "スター株一覧",
        "共通点分析",
        "海外投資家フロー",
        "ファクター分析",
        "類型分類（深堀り）",
        "予測モデル",
        "早期発見指標",
        "今買える候補",
    ])

    with tabs[0]:
        _tab_star_stock_list(result)
    with tabs[1]:
        _tab_common_features(result)
    with tabs[2]:
        _tab_foreign_flow(result)
    with tabs[3]:
        _tab_factor_analysis(result)
    with tabs[4]:
        _tab_typology(result)
    with tabs[5]:
        _tab_prediction_model(result)
    with tabs[6]:
        _tab_early_detection(result)
    with tabs[7]:
        _tab_timing_candidates(result)

    st.markdown("---")
    if st.button("新しい分析を開始"):
        _clear_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Tab 1: スター株一覧
# ---------------------------------------------------------------------------
def _tab_star_stock_list(result: StarStockResult):
    stocks = result.star_stocks
    if not stocks:
        st.info("スター株が検出されませんでした。")
        return

    rows = []
    for s in stocks:
        tr = s.get("total_return", 0)
        er = s.get("excess_return", 0)
        onset_ret = s.get("onset_return")
        onset_ret_str = f"{onset_ret:.1%}" if onset_ret is not None else "-"
        # Onset検出方法の表示
        method = s.get("onset_detection_method", "")
        sig_score = s.get("onset_signal_score", 0)
        if method == "multi_signal":
            onset_method_str = f"マルチシグナル({sig_score}/10)"
        elif method == "CUSUM":
            onset_method_str = "CUSUM"
        else:
            onset_method_str = "-"
        # 発火シグナル名の短縮表示
        onset_sigs = s.get("onset_signals", [])
        onset_sigs_short = " ".join(_ONSET_SIGNAL_SHORT.get(n, n) for n in onset_sigs) if onset_sigs else "-"

        rows.append({
            "コード": s.get("code", ""),
            "企業名": s.get("name", ""),
            "セクター": s.get("sector", ""),
            "規模": s.get("scale_category", ""),
            "スター化後リターン": onset_ret_str,
            "期間リターン": f"{tr:.1%}" if isinstance(tr, (int, float)) else tr,
            "超過リターン": f"{er:.1%}" if isinstance(er, (int, float)) else er,
            "Onset検出": onset_method_str,
            "Onset日シグナル": onset_sigs_short,
            "フロースコア": f"{s.get('flow_score', 0):.3f}",
            "クラスター": s.get("cluster", ""),
            "Lead/Lag": s.get("lead_lag_role", ""),
            "スター化開始": s.get("star_onset_date", ""),
            "検出": s.get("source", ""),
            "仕手フラグ": ", ".join(s.get("pump_dump_flags", [])),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=min(len(df) * 38 + 40, 600))

    # サマリーメトリクス
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        avg_ret = np.mean([s.get("total_return", 0) for s in stocks])
        st.metric("平均リターン", f"{avg_ret:.1%}")
    with mc2:
        avg_excess = np.mean([s.get("excess_return", 0) for s in stocks])
        st.metric("平均超過リターン", f"{avg_excess:.1%}")
    with mc3:
        avg_flow = np.mean([s.get("flow_score", 0) for s in stocks])
        st.metric("平均フロースコア", f"{avg_flow:.3f}")
    with mc4:
        leaders = sum(1 for s in stocks if s.get("lead_lag_role") == "leader")
        st.metric("リーダー銘柄数", str(leaders))


# ---------------------------------------------------------------------------
# Tab 2: 共通点分析（定量シグナル）
# ---------------------------------------------------------------------------
_SIGNAL_FEATURES = [
    ("onset_return", "スター化後リターン"),
    ("total_return", "期間リターン"),
    ("excess_return", "超過リターン"),
    ("volume_change_ratio", "出来高変化比"),
    ("acceleration", "後半/前半加速度"),
    ("volume_surge_count", "出来高急増日数"),
    ("flow_score", "海外フロースコア"),
    ("vpin_increase", "VPIN変化"),
    ("factor_alpha", "ファクターα(日次)"),
    ("factor_r_squared", "ファクターR²"),
    ("realized_vol_change", "ボラティリティ変化率"),
    ("up_volume_ratio", "上昇日出来高比率"),
    ("accumulation_day_ratio", "アキュミュレーション日比率"),
    ("obv_trend_strength", "OBVトレンド強度"),
    ("beta_shift", "βシフト"),
    ("sharpe", "シャープレシオ"),
]


def _tab_common_features(result: StarStockResult):
    stocks = result.star_stocks
    if not stocks:
        st.info("スター株が検出されませんでした。")
        return

    # --- シグナル検証結果（反復的特徴量発見） ---
    sv = result.signal_validation
    if sv and "signals" in sv:
        # 収束インジケータ
        converged = sv.get("discovery_converged", False)
        iterations = sv.get("discovery_iterations", 0)
        if converged:
            st.success(f"反復探索が収束しました（{iterations}回反復）")
        else:
            best_prec = max((c.get("precision", 0) for c in sv.get("combo_signals", [{}])), default=0)
            st.warning(f"反復探索が収束しませんでした（{iterations}回反復、最良Precision: {best_prec:.1%}）")

        st.markdown("### 反復的特徴量発見（26特徴量×コンボ探索）")
        st.caption(
            "26個の短期ウィンドウ特徴量を網羅計算し、onset候補日×正例/負例で最適閾値(Youden's J)を決定。"
            "2-3特徴量のAND条件を探索し、反復的に閾値を精緻化。"
            "全ユニバースでFP率を検証しています。"
        )

        # --- 組み合わせシグナル結果（最重要） ---
        combo_sigs = sv.get("combo_signals", [])
        if combo_sigs:
            st.markdown("#### シグナル組み合わせ（AND条件）")
            combo_rows = []
            for cs in combo_sigs[:20]:
                prec = cs["precision"]
                is_ds = cs.get("is_doubly_specific", False)
                tag = (
                    "二重特異" if is_ds
                    else "強い" if prec >= 0.30
                    else "有効" if prec >= 0.20
                    else "やや有効" if prec >= 0.10
                    else "弱い"
                )
                combo_rows.append({
                    "判定": tag,
                    "シグナル組み合わせ": cs["labels"],
                    "シグナル数": cs["n_signals"],
                    "発火銘柄数": cs["total_hit"],
                    "うちスター": cs["star_hit"],
                    "Precision": f"{prec:.1%}",
                    "Recall": f"{cs['recall']:.1%}",
                    "Lift": f"{cs['lift']:.1f}x",
                    "タイミングLift": f"{cs.get('timing_lift', 0):.1f}x",
                    "検証FP率": f"{cs.get('validation_fp_rate', 0):.1%}" if cs.get('validation_fp_rate') is not None else "-",
                    "識別スコア": f"{cs.get('discriminative_score', 0):.3f}",
                    "二重特異": "Yes" if is_ds else "",
                })
            st.dataframe(pd.DataFrame(combo_rows), use_container_width=True, hide_index=True, key="ss_combo_validation")

            # Precision棒グラフ（組み合わせ）
            fig_combo = go.Figure()
            combo_labels = [cs["labels"][:40] for cs in combo_sigs[:15]]
            combo_precs = [cs["precision"] * 100 for cs in combo_sigs[:15]]
            combo_colors = [
                "#2E7D32" if p >= 30 else "#FF8000" if p >= 20 else "#FFC107" if p >= 10 else "#C62828"
                for p in combo_precs
            ]
            fig_combo.add_trace(go.Bar(
                y=combo_labels, x=combo_precs,
                marker_color=combo_colors, orientation="h",
            ))
            fig_combo.add_vline(x=30, line_dash="dash", line_color="#2E7D32",
                                annotation_text="30%（強い）")
            fig_combo.add_vline(x=20, line_dash="dash", line_color="#FF8000",
                                annotation_text="20%（有効）")
            base_pct = sv.get("base_rate", 0) * 100
            fig_combo.add_vline(x=base_pct, line_dash="dot", line_color="gray",
                                annotation_text=f"ベースレート({base_pct:.1f}%)")
            fig_combo.update_layout(
                title="シグナル組み合わせのPrecision（%）",
                xaxis_title="Precision (%)", height=max(350, len(combo_labels) * 25 + 100),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_combo, use_container_width=True, key="ss_combo_chart")

            st.markdown(
                "**読み方**: Precision 30% = この組み合わせ条件を満たす銘柄の30%がスター株。"
                "個別シグナルの数%と比べて大幅に向上しています。"
            )
        else:
            st.info("有効なシグナル組み合わせが見つかりませんでした。")

        # --- Lift × Timing Score 散布図（二重特異性マップ、上位15） ---
        sigs = sv.get("signals", [])[:15]
        if sigs and any(s.get("timing_score") is not None for s in sigs):
            st.markdown("#### 二重特異性マップ（Lift × タイミングスコア）")
            st.caption(
                "右上象限 = **二重特異**（銘柄にもタイミングにも特異的な最重要シグナル）。"
                "左上 = タイミング特異のみ（偶然）。右下 = 銘柄特異のみ（多くの株が持つ特徴で無意味）。"
            )
            sig_lifts = [s["lift"] for s in sigs]
            sig_ts = [s.get("timing_score", 0.5) for s in sigs]
            sig_labels = [s["label"] for s in sigs]
            sig_doubly = [s.get("is_doubly_specific", False) for s in sigs]
            sig_colors = [
                "#2E7D32" if d else "#FFC107" if l >= 1.5 or t >= 0.7 else "#C62828"
                for d, l, t in zip(sig_doubly, sig_lifts, sig_ts)
            ]
            fig_ds = go.Figure(data=go.Scatter(
                x=sig_lifts, y=sig_ts, mode="markers+text",
                text=sig_labels, textposition="top center", textfont=dict(size=9),
                marker=dict(size=14, color=sig_colors, line=dict(width=1, color="#333")),
            ))
            fig_ds.add_hline(
                y=0.70, line_dash="dash", line_color="#888",
                annotation_text="タイミング特異性閾値(0.70)",
            )
            fig_ds.add_vline(
                x=1.5, line_dash="dash", line_color="#888",
                annotation_text="銘柄特異性閾値(Lift=1.5)",
            )
            fig_ds.update_layout(
                title="二重特異性マップ",
                xaxis_title="Lift（銘柄特異性）",
                yaxis_title="タイミングスコア（タイミング特異性）",
                height=450,
            )
            st.plotly_chart(fig_ds, use_container_width=True, key="ss_double_specificity_map")

        # --- 個別シグナル（参考） ---
        with st.expander("個別シグナル詳細（参考）"):
            st.caption(
                f"onset日断面での個別シグナル評価。"
                f"全{sv.get('n_all', 0):,}断面中スター{sv.get('n_star_samples', sv.get('n_star', 0))}サンプル"
                f"（ベースレート{sv.get('base_rate', 0):.2%}）。"
            )
            sv_rows = []
            for sig in sv["signals"]:
                sv_rows.append({
                    "シグナル": sig["label"],
                    "閾値": f"{sig['threshold']:.3f}",
                    "発火数": sig["total_triggered"],
                    "うちスター": sig["star_triggered"],
                    "Precision": f"{sig['precision']:.1%}",
                    "Recall": f"{sig['recall']:.1%}",
                    "Lift": f"{sig['lift']:.1f}x",
                    "Youden J": f"{sig.get('youden_j', 0):.3f}",
                    "AUC": f"{sig.get('auc_approx', 0.5):.3f}",
                    "タイミングスコア": f"{sig.get('timing_score', 0):.2f}",
                    "識別スコア": f"{sig.get('discriminative_score', 0):.3f}",
                    "判定": sig["verdict"],
                })
            st.dataframe(pd.DataFrame(sv_rows), use_container_width=True, hide_index=True, key="ss_signal_validation")

            # Liftの棒グラフ
            fig_lift = go.Figure()
            labels = [sig["label"] for sig in sv["signals"]]
            lifts = [sig["lift"] for sig in sv["signals"]]
            colors = ["#2E7D32" if l >= 3 else "#FF8000" if l >= 2 else "#C62828" for l in lifts]
            fig_lift.add_trace(go.Bar(x=labels, y=lifts, marker_color=colors))
            fig_lift.add_hline(y=1.0, line_dash="dash", line_color="gray",
                               annotation_text="Lift=1（ランダム）")
            fig_lift.update_layout(
                title="個別シグナルのLift値", yaxis_title="Lift", height=300,
            )
            st.plotly_chart(fig_lift, use_container_width=True, key="ss_lift_chart")
    else:
        st.info("シグナル検証が実行されていません。")

    st.markdown("---")

    # --- 定量シグナルプロファイル ---
    with st.expander("シグナル統計詳細（スター株のみ）"):
        stat_rows = []
        for key, label in _SIGNAL_FEATURES:
            vals = [float(s.get(key, 0) or 0) for s in stocks if s.get(key) is not None]
            if not vals:
                continue
            arr = np.array(vals)
            stat_rows.append({
                "指標": label,
                "中央値": f"{np.median(arr):.4f}",
                "平均": f"{np.mean(arr):.4f}",
                "標準偏差": f"{np.std(arr):.4f}",
                "最小": f"{np.min(arr):.4f}",
                "25%": f"{np.percentile(arr, 25):.4f}",
                "75%": f"{np.percentile(arr, 75):.4f}",
                "最大": f"{np.max(arr):.4f}",
            })
        if stat_rows:
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True, key="ss_signal_stats")

    # --- セクター分布 ---
    st.markdown("### セクター構成")
    sectors = {}
    for s in stocks:
        sec = s.get("sector", "不明") or "不明"
        sectors[sec] = sectors.get(sec, 0) + 1
    sector_df = pd.DataFrame([
        {"セクター": k, "銘柄数": v, "比率": f"{v/len(stocks):.0%}"}
        for k, v in sorted(sectors.items(), key=lambda x: x[1], reverse=True)
    ])
    st.dataframe(sector_df, use_container_width=True, hide_index=True, key="ss_sector_dist")

    # --- 指標の分布（箱ひげ図） ---
    st.markdown("### 主要指標の分布")
    box_keys = ["excess_return", "volume_change_ratio", "flow_score", "vpin_increase", "factor_alpha"]
    box_labels = ["超過リターン", "出来高変化比", "フロースコア", "VPIN変化", "ファクターα"]

    fig_box = make_subplots(rows=1, cols=len(box_keys), subplot_titles=box_labels)
    for i, (key, label) in enumerate(zip(box_keys, box_labels)):
        vals = [float(s.get(key, 0) or 0) for s in stocks]
        fig_box.add_trace(
            go.Box(y=vals, name=label, marker_color="#FF8000", boxmean=True),
            row=1, col=i + 1,
        )
    fig_box.update_layout(height=350, showlegend=False, title_text="スター株の主要指標分布")
    st.plotly_chart(fig_box, use_container_width=True, key="ss_box_dist")

    # --- 特徴量ヒートマップ ---
    st.markdown("### 銘柄別シグナル一覧")
    feature_keys = [
        "onset_return", "excess_return", "volume_change_ratio",
        "acceleration", "flow_score", "vpin_increase",
        "factor_alpha", "realized_vol_change",
    ]
    feature_labels = [
        "スター化後リターン", "超過リターン", "出来高変化",
        "加速度", "フロースコア", "VPIN変化",
        "α", "ボラ変化",
    ]

    top_n = min(30, len(stocks))
    top_stocks = sorted(stocks, key=lambda x: x.get("excess_return", 0), reverse=True)[:top_n]

    z_data = []
    y_labels = []
    for s in top_stocks:
        row = [float(s.get(k, 0) or 0) for k in feature_keys]
        z_data.append(row)
        y_labels.append(_fmt(s))

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=feature_labels,
        y=y_labels,
        colorscale="RdYlGn",
        texttemplate="%{z:.2f}",
        textfont={"size": 10},
    ))
    fig.update_layout(
        title="銘柄×指標ヒートマップ",
        height=max(400, top_n * 28 + 120),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True, key="ss_heatmap")

    # AI要約（折りたたみ）
    if result.common_features_summary:
        with st.expander("AI解釈（参考）"):
            st.markdown(result.common_features_summary)


def _interpret_signal(key: str, p25: float, p50: float) -> str:
    """閾値の解釈を人間がわかる文にする"""
    interpretations = {
        "onset_return": f"スター化開始後に中央値{p50:.0%}上昇",
        "total_return": f"分析期間で中央値{p50:.0%}の上昇",
        "excess_return": f"市場平均を中央値{p50:.0%}上回る",
        "volume_change_ratio": f"出来高が後半に中央値{p50:.1f}倍に増加",
        "acceleration": f"後半の上昇ペースが前半の{p50:.1f}倍",
        "volume_surge_count": f"出来高急増が中央値{p50:.0f}日発生",
        "flow_score": f"海外フロースコア中央値{p50:.2f}（1.0が最大）",
        "vpin_increase": f"情報トレーダー参加率が{p50:.4f}pt上昇",
        "factor_alpha": f"市場で説明できない日次超過リターン{p50:.5f}",
        "factor_r_squared": f"市場要因で{p50:.0%}が説明可能（残りは個別要因）",
        "realized_vol_change": f"ボラティリティが{p50:.0%}変化",
        "up_volume_ratio": f"上昇日に全出来高の{p50:.0%}が集中",
        "accumulation_day_ratio": f"大量売買+価格安定の「静かな買い集め」日が{p50:.1%}",
        "obv_trend_strength": f"OBV（累積出来高）トレンド強度{p50:.4f}",
        "beta_shift": f"市場連動性が{p50:.3f}変化",
        "sharpe": f"リスク調整後リターン{p50:.2f}",
    }
    return interpretations.get(key, f"中央値 {p50:.4f}")


# ---------------------------------------------------------------------------
# Tab 3: 海外投資家フロー
# ---------------------------------------------------------------------------
def _tab_foreign_flow(result: StarStockResult):
    st.markdown("### 海外投資家フロー評価")
    st.markdown(result.foreign_flow_assessment or "_AI評価が生成されていません_")

    stocks = result.star_stocks
    if not stocks:
        return

    # フロースコア分布
    flow_scores = [s.get("flow_score", 0) for s in stocks]
    fig_dist = go.Figure(data=go.Histogram(
        x=flow_scores,
        nbinsx=20,
        marker_color="#FF8000",
        opacity=0.8,
    ))
    fig_dist.update_layout(
        title="フロースコア分布",
        xaxis_title="フロースコア",
        yaxis_title="銘柄数",
        height=350,
    )
    st.plotly_chart(fig_dist, use_container_width=True, key="ss_flow_dist")

    # プロキシ指標テーブル
    st.markdown("#### プロキシ指標別分析")
    indicator_keys = [
        "up_volume_ratio", "accumulation_day_ratio", "obv_trend_strength",
        "beta_shift", "vpin_increase",
    ]
    indicator_labels = [
        "アップVol比率", "アキュミュレーション日比率", "OBVトレンド",
        "β変化", "VPIN変化",
    ]

    rows = []
    for s in sorted(stocks, key=lambda x: x.get("flow_score", 0), reverse=True)[:30]:
        row = {"銘柄": _fmt(s)}
        fi = s.get("flow_indicators", {})
        for k, label in zip(indicator_keys, indicator_labels):
            val = fi.get(k, s.get(k, 0))
            row[label] = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
        row["合成スコア"] = f"{s.get('flow_score', 0):.3f}"
        rows.append(row)

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, key="ss_flow_table")

    # VPIN時系列チャート（上位5銘柄）
    st.markdown("#### VPIN時系列（上位5銘柄）")
    top5 = sorted(stocks, key=lambda x: abs(x.get("vpin_increase", 0)), reverse=True)[:5]
    fig_vpin = go.Figure()
    for s in top5:
        vpin_series = s.get("vpin_series", [])
        if vpin_series:
            fig_vpin.add_trace(go.Scatter(
                y=vpin_series,
                mode="lines",
                name=_fmt(s),
            ))
    fig_vpin.update_layout(
        title="VPIN推移（情報トレーダー参加率の変化）",
        yaxis_title="VPIN",
        xaxis_title="期間（サンプルポイント）",
        height=350,
    )
    st.plotly_chart(fig_vpin, use_container_width=True, key="ss_vpin_chart")


# ---------------------------------------------------------------------------
# Tab 4: ファクター分析
# ---------------------------------------------------------------------------
def _tab_factor_analysis(result: StarStockResult):
    fa = result.factor_analysis
    stocks = result.star_stocks
    if not fa or not stocks:
        st.info("ファクター分析結果がありません。")
        return

    # 解説
    r2 = fa.get('avg_r_squared', 0)
    st.markdown(
        f"4ファクターモデル（市場・規模・バリュー・モメンタム）で各銘柄のリターンを分解。"
        f"**R² = {r2:.2f}**（＝既知ファクターで{r2:.0%}しか説明できない → "
        f"残り{1-r2:.0%}は銘柄固有の要因）"
    )

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.metric("平均日次α", f"{fa.get('avg_alpha', 0):.5f}")
    with fc2:
        st.metric("α有意率", f"{fa.get('alpha_significant_pct', 0):.0%}")
    with fc3:
        st.metric("平均R²", f"{r2:.2f}")

    # α分布ヒストグラム
    alphas = [s.get("factor_alpha", 0) for s in stocks if s.get("factor_alpha") is not None]
    if alphas:
        fig_alpha = go.Figure(data=go.Histogram(
            x=alphas, nbinsx=20, marker_color="#1565C0", opacity=0.8,
        ))
        fig_alpha.update_layout(
            title="ファクターα分布（正の値 = ファクターで説明できない超過リターン）",
            xaxis_title="日次α", yaxis_title="銘柄数", height=350,
        )
        st.plotly_chart(fig_alpha, use_container_width=True, key="ss_alpha_dist")

    # ファクター露出散布図
    st.markdown("#### ファクター露出マップ")
    mkt_betas, alpha_vals, labels_list, flow_vals = [], [], [], []
    for s in stocks:
        betas = s.get("factor_betas", {})
        if "MKT" in betas:
            mkt_betas.append(betas["MKT"])
            alpha_vals.append(s.get("factor_alpha", 0))
            labels_list.append(_fmt(s))
            flow_vals.append(s.get("flow_score", 0))

    if mkt_betas:
        fig_scatter = go.Figure(data=go.Scatter(
            x=mkt_betas, y=alpha_vals,
            mode="markers+text", text=labels_list,
            textposition="top center", textfont=dict(size=8),
            marker=dict(size=10, color=flow_vals, colorscale="YlOrRd",
                        showscale=True, colorbar=dict(title="フロースコア")),
        ))
        fig_scatter.update_layout(
            title="マーケットβ vs α（色: フロースコア）",
            xaxis_title="マーケットβ（市場感応度）",
            yaxis_title="日次α（説明できないリターン）",
            height=450,
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="ss_factor_scatter")

    # テーブル
    st.markdown("#### ファクター分解詳細")
    factor_rows = []
    for s in stocks[:30]:
        betas = s.get("factor_betas", {})
        factor_rows.append({
            "銘柄": _fmt(s),
            "α": f"{s.get('factor_alpha', 0):.5f}",
            "α t値": f"{s.get('factor_alpha_tstat', 0):.2f}",
            "MKT β": f"{betas.get('MKT', 0):.3f}",
            "SMB β": f"{betas.get('SMB', 0):.3f}",
            "WML β": f"{betas.get('WML', 0):.3f}",
            "R²": f"{s.get('factor_r_squared', 0):.3f}",
        })
    if factor_rows:
        st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, key="ss_factor_table")


# ---------------------------------------------------------------------------
# Tab 5: 類型分類（深堀り）
# ---------------------------------------------------------------------------
_FEATURE_LABELS_JP = {
    "total_return": "リターン",
    "excess_return": "超過リターン",
    "max_drawdown": "最大DD",
    "acceleration": "加速度",
    "volume_change_ratio": "出来高変化",
    "volume_surge_count": "出来高急増回数",
    "realized_vol_change": "ボラ変化",
    "flow_score": "フロースコア",
    "vpin_increase": "VPIN変化",
    "factor_alpha": "α",
    "up_volume_ratio": "アップVol比率",
    "accumulation_day_ratio": "アキュミュレーション",
    "obv_trend_strength": "OBVトレンド",
    "beta_shift": "β変化",
}


def _tab_typology(result: StarStockResult):
    cluster = result.cluster_analysis
    stocks = result.star_stocks

    if not cluster or "error" in cluster:
        st.info(cluster.get("error", "クラスター分析結果がありません。"))
        return

    # PCA散布図
    pca_coords = cluster.get("pca_coords", [])
    labels_arr = cluster.get("labels", [])
    n_clusters = cluster.get("n_clusters", 4)
    colors = ["#FF8000", "#1565C0", "#2E7D32", "#C62828", "#7B1FA2",
               "#F57C00", "#0277BD", "#558B2F", "#AD1457", "#4527A0"]

    if pca_coords and labels_arr:
        fig_pca = go.Figure()
        for c_id in range(n_clusters):
            indices = [i for i, l in enumerate(labels_arr) if l == c_id]
            if not indices:
                continue
            xs = [pca_coords[i][0] for i in indices]
            ys = [pca_coords[i][1] if len(pca_coords[i]) > 1 else 0 for i in indices]
            names = [_fmt(stocks[i]) for i in indices if i < len(stocks)]
            fig_pca.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers+text",
                text=names, textposition="top center", textfont=dict(size=8),
                name=f"クラスター{c_id}",
                marker=dict(size=12, color=colors[c_id % len(colors)]),
            ))
        variance = cluster.get("pca_explained_variance", [0, 0])
        fig_pca.update_layout(
            title="PCA散布図（クラスター色分け）",
            xaxis_title=f"PC1 ({variance[0]:.1%})" if variance else "PC1",
            yaxis_title=f"PC2 ({variance[1]:.1%})" if len(variance) > 1 else "PC2",
            height=500,
        )
        st.plotly_chart(fig_pca, use_container_width=True, key="ss_pca")

    # --- 各クラスターの深堀り ---
    st.markdown("### クラスター別 詳細分析")
    profiles = cluster.get("cluster_profiles", [])
    typology = result.pattern_typology

    for cp in profiles:
        c_id = cp["cluster_id"]
        color = colors[c_id % len(colors)]
        n_mem = cp["n_members"]

        # AIパターン名があれば使用
        ai_name = ""
        ai_desc = ""
        if typology:
            for pt in typology:
                if pt.get("cluster_id") == c_id:
                    ai_name = pt.get("pattern_name", "")
                    ai_desc = pt.get("description", "")
                    break

        header = f"タイプ{c_id}" if not ai_name else ai_name
        st.markdown(
            f'<h4 style="color:{color};border-left:4px solid {color};padding-left:8px;">'
            f'{header}（{n_mem}銘柄）</h4>',
            unsafe_allow_html=True,
        )

        if ai_desc:
            st.markdown(ai_desc)

        # メンバー銘柄テーブル
        member_codes = cp.get("member_codes", [])
        member_stocks = [s for s in stocks if s["code"] in member_codes]

        if member_stocks:
            mem_rows = []
            for s in member_stocks:
                onset_ret = s.get("onset_return")
                onset_ret_str = f"{onset_ret:.1%}" if onset_ret is not None else "-"
                mem_rows.append({
                    "コード": s["code"],
                    "企業名": s.get("name", ""),
                    "セクター": s.get("sector", ""),
                    "スター化後リターン": onset_ret_str,
                    "期間リターン": f"{s.get('total_return', 0):.1%}",
                    "超過リターン": f"{s.get('excess_return', 0):.1%}",
                    "フロースコア": f"{s.get('flow_score', 0):.3f}",
                    "VPIN変化": f"{s.get('vpin_increase', 0):.4f}",
                    "Lead/Lag": s.get("lead_lag_role", ""),
                    "スター化開始": s.get("star_onset_date", ""),
                })
            st.dataframe(
                pd.DataFrame(mem_rows), use_container_width=True, hide_index=True,
                key=f"ss_cluster_{c_id}_members",
            )

            # クラスター統計サマリー
            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                avg_r = np.mean([s.get("total_return", 0) for s in member_stocks])
                st.metric("平均リターン", f"{avg_r:.1%}")
            with cc2:
                avg_f = np.mean([s.get("flow_score", 0) for s in member_stocks])
                st.metric("平均フロースコア", f"{avg_f:.3f}")
            with cc3:
                avg_v = np.mean([s.get("vpin_increase", 0) for s in member_stocks])
                st.metric("平均VPIN変化", f"{avg_v:.4f}")
            with cc4:
                sec_counts = {}
                for s in member_stocks:
                    sec = s.get("sector", "不明") or "不明"
                    sec_counts[sec] = sec_counts.get(sec, 0) + 1
                top_sec = max(sec_counts, key=sec_counts.get) if sec_counts else "不明"
                st.metric("最多セクター", top_sec)

        # Centroidのレーダーチャート
        try:
            profile = cp.get("centroid_profile", {})
            if profile:
                feat_keys = list(profile.keys())[:8]
                feat_vals = [float(profile[k]) for k in feat_keys]
                feat_names = [_FEATURE_LABELS_JP.get(k, k) for k in feat_keys]

                # 値が全てゼロだとチャートが崩れるのでスキップ
                if any(v != 0 for v in feat_vals):
                    fig_radar = go.Figure(data=go.Scatterpolar(
                        r=feat_vals + [feat_vals[0]],
                        theta=feat_names + [feat_names[0]],
                        fill="toself",
                        fillcolor=_hex_to_rgba(color, 0.15),
                        line=dict(color=color),
                        name=header,
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        height=350,
                        title=f"{header} 特徴量プロファイル",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_radar, use_container_width=True, key=f"ss_radar_{c_id}")
        except Exception as e:
            st.caption(f"レーダーチャート描画エラー: {e}")

        st.markdown("---")

    # Lead-Lag ネットワーク
    lead_lag = result.lead_lag_analysis
    pairs = lead_lag.get("pairs", [])
    if pairs:
        st.markdown("### Lead-Lag ネットワーク（どの銘柄が先行して動くか）")

        leaders_list = []
        followers_list = []
        values = []
        link_labels = []
        node_set = set()

        for p in pairs[:30]:
            leaders_list.append(p["leader"])
            followers_list.append(p["follower"])
            values.append(1)
            link_labels.append(f"lag={p['lag_days']}日 (p={p['p_value']:.3f})")
            node_set.add(p["leader"])
            node_set.add(p["follower"])

        node_list = sorted(node_set)
        node_map = {n: i for i, n in enumerate(node_list)}
        node_labels = [_fmt_code_name(n, stocks) for n in node_list]

        fig_sankey = go.Figure(data=go.Sankey(
            node=dict(pad=15, thickness=20, label=node_labels, color="#FF8000"),
            link=dict(
                source=[node_map[l] for l in leaders_list],
                target=[node_map[f] for f in followers_list],
                value=values, label=link_labels,
                color="rgba(255, 128, 0, 0.3)",
            ),
        ))
        fig_sankey.update_layout(title="Lead-Lag関係（Granger因果）", height=500)
        st.plotly_chart(fig_sankey, use_container_width=True, key="ss_sankey")

        # Lead-Lagテーブル
        ll_rows = []
        for p in pairs[:20]:
            ll_rows.append({
                "リーダー": _fmt_code_name(p["leader"], stocks),
                "フォロワー": _fmt_code_name(p["follower"], stocks),
                "ラグ（日）": p["lag_days"],
                "F値": f"{p['f_stat']:.2f}",
                "p値": f"{p['p_value']:.4f}",
            })
        if ll_rows:
            st.dataframe(pd.DataFrame(ll_rows), use_container_width=True, key="ss_ll_table")


# ---------------------------------------------------------------------------
# Tab 6: 予測モデル
# ---------------------------------------------------------------------------
def _tab_prediction_model(result: StarStockResult):
    cs = result.cross_sectional
    if not cs or "error" in cs:
        st.info(cs.get("error", "クロスセクショナル回帰結果がありません。"))
        return

    st.markdown("### Fama-MacBeth横断面回帰")
    st.markdown(
        f"毎月の横断面回帰を **{cs.get('n_months', 0)}ヶ月** 繰り返し、"
        f"「どの指標が翌月リターンを予測するか」を統計検定。"
    )

    features = cs.get("features", [])
    coeffs = cs.get("coefficients", [])
    t_stats = cs.get("t_statistics", [])
    p_values = cs.get("p_values", [])
    sig = cs.get("significant_predictors", [])

    feature_labels_map = {
        "volume_surge_3m": "出来高急増回数（3ヶ月）",
        "beta_change": "β変化量",
        "realized_vol_ratio": "ボラティリティ変化率",
        "momentum_3m": "3ヶ月モメンタム",
        "up_volume_ratio": "アップボリューム比率",
    }

    if features:
        rows = []
        for i, f in enumerate(features):
            is_sig = f in sig
            rows.append({
                "指標": feature_labels_map.get(f, f),
                "係数": f"{coeffs[i]:.6f}" if i < len(coeffs) else "",
                "t値": f"{t_stats[i]:.3f}" if i < len(t_stats) else "",
                "p値": f"{p_values[i]:.4f}" if i < len(p_values) else "",
                "判定": "有意" if is_sig else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, key="ss_famamacbeth")

    if sig:
        sig_jp = [feature_labels_map.get(f, f) for f in sig]
        st.success(f"統計的に有意な予測変数: {', '.join(sig_jp)}")
        st.markdown(
            "上記の指標が高い銘柄は、翌月のリターンが統計的に高い傾向があります。"
            "「今買える候補」タブでこれらの条件に合う銘柄を確認できます。"
        )
    else:
        st.warning("統計的に有意な予測変数は見つかりませんでした。")


# ---------------------------------------------------------------------------
# Tab 7: 早期発見指標
# ---------------------------------------------------------------------------
def _tab_early_detection(result: StarStockResult):
    st.markdown("### 早期発見ルール")
    st.markdown(result.detection_rules or "_ルールが生成されていません_")

    stocks = result.star_stocks
    if not stocks:
        return

    # --- 全スター株のスター化開始サマリー ---
    st.markdown("### 全スター株のスター化タイミング")
    st.caption(f"全{len(stocks)}銘柄のスター化開始日と変化点を表示しています。")

    onset_rows = []
    for s in stocks:
        onset = s.get("star_onset_date", "")
        cps = s.get("change_points", [])
        n_cps = len(cps)
        cp_types = ", ".join(f"{cp['type']}({cp.get('confidence', 0):.0%})" for cp in cps) if cps else "検出なし"
        onset_ret = s.get("onset_return")
        method = s.get("onset_detection_method", "")
        sig_score = s.get("onset_signal_score", 0)
        combo_score = s.get("onset_combo_score")
        if combo_score is not None:
            method_str = f"反復探索(コンボ{combo_score})"
        elif method == "multi_signal":
            method_str = f"マルチシグナル({sig_score}/10)"
        elif method == "CUSUM":
            method_str = "CUSUM"
        else:
            method_str = "-"
        onset_sigs = s.get("onset_signals", [])
        sigs_short = " ".join(_ONSET_SIGNAL_SHORT.get(n, n) for n in onset_sigs) if onset_sigs else "-"
        onset_rows.append({
            "銘柄": _fmt(s),
            "スター化開始日": onset or "未検出",
            "検出方法": method_str,
            "発火シグナル": sigs_short,
            "スター化後リターン": f"{onset_ret:.1%}" if onset_ret is not None else "-",
            "変化点数": n_cps,
            "変化点詳細": cp_types,
            "レジーム数": s.get("n_regimes", 1),
        })
    st.dataframe(pd.DataFrame(onset_rows), use_container_width=True, hide_index=True, key="ss_onset_summary")

    # --- Onset候補テーブル ---
    sv = result.signal_validation
    onset_candidates = sv.get("onset_candidates", {}) if sv else {}
    if onset_candidates:
        with st.expander("Onset候補日一覧（反復探索で生成）"):
            st.caption("各スター株の全onset候補日。コンボスコアが高い候補が最終的なonset日として選択されます。")
            cand_rows = []
            for s in stocks:
                code = s["code"]
                cands = onset_candidates.get(code, [])
                for ci, c in enumerate(cands):
                    is_selected = (s.get("star_onset_date", "") == c.get("date", ""))
                    cand_rows.append({
                        "銘柄": _fmt(s),
                        "候補日": c.get("date", ""),
                        "直前20日リターン": f"{c.get('pre_20d_return', 0):.1%}",
                        "60日先ピークリターン": f"{c.get('peak_60d_return', 0):.1%}",
                        "60日先リターン": f"{c.get('fwd_60d_return', 0):.1%}",
                        "コンボスコア": c.get("combo_score", "-"),
                        "選択": "Yes" if is_selected else "",
                    })
            if cand_rows:
                st.dataframe(pd.DataFrame(cand_rows), use_container_width=True, hide_index=True, key="ss_onset_candidates")

    # --- 変化点詳細テーブル ---
    st.markdown("### 変化点詳細")
    cp_rows = []
    for s in stocks:
        cps = s.get("change_points", [])
        for cp in cps:
            forced = cp.get("forced", False)
            cp_rows.append({
                "銘柄": _fmt(s),
                "日付": cp.get("date", ""),
                "タイプ": cp.get("type", ""),
                "信頼度": f"{cp.get('confidence', 0):.0%}",
                "前ボラ": f"{cp.get('before_vol', 0):.5f}",
                "後ボラ": f"{cp.get('after_vol', 0):.5f}",
                "備考": "強制検出" if forced else "",
            })

    if cp_rows:
        st.dataframe(pd.DataFrame(cp_rows), use_container_width=True, hide_index=True, key="ss_changepoints")
    else:
        st.info("変化点は検出されませんでした。")

    # スター化開始日の分布
    onset_dates = [s.get("star_onset_date", "") for s in stocks if s.get("star_onset_date")]
    if onset_dates:
        st.markdown("#### スター化開始日の分布")
        onset_ts = pd.to_datetime(onset_dates, errors="coerce").dropna()
        if len(onset_ts) > 0:
            fig_onset = go.Figure(data=go.Histogram(
                x=onset_ts, nbinsx=12, marker_color="#2E7D32", opacity=0.8,
            ))
            fig_onset.update_layout(
                title="スター化はいつ始まったか",
                xaxis_title="日付", yaxis_title="銘柄数", height=300,
            )
            st.plotly_chart(fig_onset, use_container_width=True, key="ss_onset_hist")

    # --- シグナルヒートマップ（onset日前後） ---
    st.markdown("### Onsetシグナル・ヒートマップ")
    st.caption("各スター株のonset日前後のシグナル発火状況。onset日に赤い縦線を表示。")

    # マルチシグナルで検出された銘柄のみ
    ms_stocks = [s for s in stocks if s.get("onset_detection_method") == "multi_signal"
                 and s.get("onset_signal_matrix") is not None
                 and s.get("star_onset_date")]
    if ms_stocks:
        hm_target = st.selectbox(
            "銘柄を選択",
            options=[_fmt(s) for s in ms_stocks],
            key="ss_hm_select",
        )
        hm_idx = next((i for i, s in enumerate(ms_stocks) if _fmt(s) == hm_target), 0)
        sel = ms_stocks[hm_idx]
        sig_matrix = sel["onset_signal_matrix"]
        onset_date_str = sel["star_onset_date"]

        if isinstance(sig_matrix, pd.DataFrame) and not sig_matrix.empty:
            try:
                onset_ts = pd.Timestamp(onset_date_str)
                sig_dates = pd.to_datetime(sig_matrix.index, errors="coerce")
                # onset日のインデックスを探す
                onset_pos = None
                for i, d in enumerate(sig_dates):
                    if d >= onset_ts:
                        onset_pos = i
                        break
                if onset_pos is None:
                    onset_pos = len(sig_dates) - 1

                # onset-20日 ～ onset+5日 の範囲
                start_pos = max(0, onset_pos - 20)
                end_pos = min(len(sig_matrix), onset_pos + 6)
                window = sig_matrix.iloc[start_pos:end_pos]

                if len(window) > 0:
                    # ヒートマップ用データ (行=シグナル、列=日付)
                    z_data = window[_MULTI_ONSET_SIGNAL_NAMES].T.astype(int).values
                    y_labels = [_ONSET_SIGNAL_SHORT.get(n, n) for n in _MULTI_ONSET_SIGNAL_NAMES]
                    x_labels = [str(d)[:10] for d in window.index]

                    fig_hm = go.Figure(data=go.Heatmap(
                        z=z_data,
                        x=x_labels,
                        y=y_labels,
                        colorscale=[[0, "#F5F5F5"], [1, "#E53935"]],
                        showscale=False,
                        xgap=1, ygap=1,
                    ))
                    # onset日の縦線（カテゴリカル軸対応: shapeで描画）
                    onset_label = onset_date_str[:10]
                    if onset_label in x_labels:
                        onset_x_pos = x_labels.index(onset_label)
                        fig_hm.add_shape(
                            type="line",
                            x0=onset_x_pos, x1=onset_x_pos,
                            y0=-0.5, y1=len(y_labels) - 0.5,
                            xref="x", yref="y",
                            line=dict(color="#1565C0", width=2, dash="dash"),
                        )
                        fig_hm.add_annotation(
                            x=onset_x_pos, y=len(y_labels) - 0.5,
                            text="onset", showarrow=False, yshift=12,
                            font=dict(color="#1565C0", size=11),
                        )
                    fig_hm.update_layout(
                        title=f"{_fmt(sel)} — Onset前後のシグナル発火",
                        xaxis_title="日付", yaxis_title="シグナル",
                        height=400,
                    )
                    st.plotly_chart(fig_hm, use_container_width=True, key="ss_onset_heatmap")

                    # onset日のスコアサマリー
                    score = sel.get("onset_signal_score", 0)
                    fired = sel.get("onset_signals", [])
                    fired_short = " ".join(_ONSET_SIGNAL_SHORT.get(n, n) for n in fired)
                    st.info(f"Onset日スコア: **{score}/10** — 発火: {fired_short}")
            except Exception as e:
                st.warning(f"ヒートマップ描画エラー: {e}")
    else:
        st.info("マルチシグナルで検出された銘柄がありません。")


# ---------------------------------------------------------------------------
# Tab 8: 今買える候補（買いタイミング近接度）
# ---------------------------------------------------------------------------
def _tab_timing_candidates(result: StarStockResult):
    candidates = result.timing_candidates
    stocks = result.star_stocks
    sv = result.signal_validation

    st.markdown("### 今、スター化直前に近い銘柄")

    # Discoveryサマリーカード
    if sv:
        converged = sv.get("discovery_converged", False)
        best_combos = sv.get("best_combos", [])
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.metric("探索収束", "Yes" if converged else "No")
        with dc2:
            n_best = len(best_combos)
            st.metric("有効コンボ数", f"{n_best}")
        with dc3:
            if best_combos:
                best_p = max(c.get("precision", 0) for c in best_combos)
                st.metric("ベストPrecision", f"{best_p:.1%}")
        if best_combos:
            st.caption(
                "ベストコンボ: " + best_combos[0].get("labels", "")
                + f" (Precision {best_combos[0].get('precision', 0):.1%}, "
                + f"Recall {best_combos[0].get('recall', 0):.1%})"
            )

    st.markdown(
        "26個のワイド特徴量テンプレート + discoveredコンボ（data-driven閾値）で候補をスコアリング。"
        "ETF・REIT等・時価総額下限未満は除外済みです。"
    )

    if not candidates:
        st.warning(
            "候補が見つかりませんでした。スター株のonsetパターンが不十分か、"
            "類似銘柄が現在のユニバースに存在しない可能性があります。"
        )
        return

    # メトリクス
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("候補数", f"{len(candidates)}銘柄")
    with mc2:
        top_score = candidates[0].get("composite_score", candidates[0]["similarity"])
        st.metric("最高スコア", f"{top_score:.3f}")
    with mc3:
        avg_sim = np.mean([c["similarity"] for c in candidates[:10]])
        st.metric("上位10類似度平均", f"{avg_sim:.1%}")
    with mc4:
        avg_combo = np.mean([c.get("combos_fired", 0) for c in candidates[:10]])
        total_combo = candidates[0].get("combos_total", 0) if candidates else 0
        st.metric("上位10コンボ発火平均", f"{avg_combo:.1f} / {total_combo}")

    # メインテーブル
    rows = []
    for i, c in enumerate(candidates[:30]):
        combos_fired = c.get("combos_fired", 0)
        ds_fired = c.get("doubly_specific_fired", 0)
        combos_total = c.get("combos_total", 0)
        fired_names = c.get("fired_combo_names", [])
        # 初動シグナル表示
        os_score = c.get("onset_signal_score", 0)
        os_fired = c.get("onset_signals_fired", {})
        os_fired_names = [_ONSET_SIGNAL_SHORT.get(k, k) for k, v in os_fired.items() if v]
        os_str = f"{os_score}/10 ({' '.join(os_fired_names)})" if os_fired_names else f"{os_score}/10"
        rows.append({
            "順位": i + 1,
            "コード": c["code"],
            "企業名": c.get("name", ""),
            "セクター": c.get("sector", ""),
            "規模": c.get("scale_category", ""),
            "総合スコア": f"{c.get('composite_score', c['similarity']):.3f}",
            "初動シグナル": os_str,
            "類似度": f"{c['similarity']:.1%}",
            "コンボ発火": f"{combos_fired}/{combos_total}" if combos_total > 0 else "-",
            "二重特異コンボ": f"{ds_fired}" if ds_fired > 0 else "-",
            "発火条件": "; ".join(fired_names[:2]) if fired_names else "-",
            "直近60日リターン": f"{c.get('recent_return_60d', 0):.1%}",
            "出来高変化": f"{c.get('recent_volume_change', 1):.1f}x",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, key="ss_timing_table")

    # 類似度分布
    sims = [c["similarity"] for c in candidates]
    fig_sim = go.Figure(data=go.Histogram(
        x=sims, nbinsx=20, marker_color="#7B1FA2", opacity=0.8,
    ))
    fig_sim.update_layout(
        title="全候補の類似度分布",
        xaxis_title="コサイン類似度（1.0 = スター化直前パターンと完全一致）",
        yaxis_title="銘柄数", height=300,
    )
    st.plotly_chart(fig_sim, use_container_width=True, key="ss_sim_dist")

    # 上位候補のdiscoveredコンボ特徴量を横棒グラフで比較
    if len(candidates) >= 3 and sv:
        best_combo_keys = set()
        for combo in sv.get("best_combos", sv.get("combo_signals", []))[:3]:
            for k in combo.get("keys", []):
                best_combo_keys.add(k)

        if best_combo_keys:
            st.markdown("#### 上位候補のDiscoveredコンボ特徴量比較")
            display_keys = sorted(best_combo_keys)
            display_labels = [_WIDE_FEATURE_LABELS_JP.get(k, k) for k in display_keys]

            top_cands = candidates[:5]
            fig_bar = go.Figure()
            for c in top_cands:
                feats = c.get("features", {})
                vals = [feats.get(k, 0) for k in display_keys]
                fig_bar.add_trace(go.Bar(
                    y=display_labels,
                    x=vals,
                    name=f"{c['code']} {c.get('name', '')[:8]}",
                    orientation="h",
                ))
            fig_bar.update_layout(
                barmode="group",
                height=max(300, len(display_keys) * 40 + 100),
                title="Discoveredコンボの特徴量値（上位5候補）",
                xaxis_title="特徴量値",
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="ss_timing_bar")
        else:
            st.info("有効なコンボが見つかっていないため、特徴量比較を表示できません。")

    st.markdown(
        "---\n"
        "**注意**: 類似度が高い ≠ 必ずスター化する。過去のパターンとの統計的な類似性を示しているに過ぎません。"
        "投資判断の一参考としてご利用ください。"
    )


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """#FF8000 → rgba(255,128,0,0.15) に変換"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _clear_state():
    for key in ("ss_thread", "ss_progress", "ss_start_time", "ss_result"):
        st.session_state.pop(key, None)


if __name__ == "__main__":
    main()
