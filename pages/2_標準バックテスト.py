"""標準バックテストページ — パラメータ設定 + 実行 + 結果表示 + 保存"""

import json
import threading
from datetime import date, datetime

import streamlit as st

from config import (
    DB_PATH, MARKET_DATA_DIR, JQUANTS_API_KEY,
    STANDARD_BACKTEST_DEFAULTS,
)
from db.database import Database
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider
from core.signal_generator import SignalConfig
from core.standard_backtester import StandardBacktester
from core.universe_filter import (
    UniverseFilterConfig,
    build_universe_description,
    MARKET_SEGMENTS,
    TOPIX_SCALE_CATEGORIES,
    SECTOR_17_LIST,
)
from core.styles import apply_reuters_style, apply_waiting_overlay
from core.result_display import render_result_tabs
from core.sidebar import render_sidebar_running_indicator

st.set_page_config(page_title="標準バックテスト", page_icon="R", layout="wide")

D = STANDARD_BACKTEST_DEFAULTS


@st.cache_resource
def get_db():
    return Database(DB_PATH)


@st.cache_resource
def get_data_provider():
    cache = DataCache(MARKET_DATA_DIR)
    return JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)


# ---------------------------------------------------------------------------
# バックグラウンドスレッド
# ---------------------------------------------------------------------------
def _run_bt_thread(
    progress_dict: dict,
    db: Database,
    provider,
    signal_config: SignalConfig,
    universe_config: UniverseFilterConfig,
    initial_capital: int,
    commission_rate: float,
    slippage_rate: float,
    start_date: str,
    end_date: str,
    max_stocks: int,
    n_recent_examples: int = 10,
):
    try:
        bt = StandardBacktester(data_provider=provider, db=db)

        def on_progress(msg: str, pct: float):
            progress_dict["message"] = msg
            progress_dict["pct"] = pct

        result = bt.run(
            signal_config=signal_config,
            universe_config=universe_config,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            start_date=start_date,
            end_date=end_date,
            max_stocks=max_stocks,
            n_recent_examples=n_recent_examples,
            on_progress=on_progress,
        )
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

    st.markdown("# Standard Backtest")
    st.caption("パラメータ設定による定型バックテスト")

    # 状態判定
    thread = st.session_state.get("bt_thread")
    is_running = thread is not None and thread.is_alive()

    # スレッド完了後: 結果昇格
    if thread is not None and not thread.is_alive() and "bt_result" not in st.session_state:
        prog = st.session_state.get("bt_progress", {})
        if "_result" in prog:
            st.session_state["bt_result"] = prog.pop("_result")
        elif prog.get("error"):
            st.session_state["bt_result"] = {"error": prog["error"]}

    has_result = "bt_result" in st.session_state

    if is_running:
        apply_waiting_overlay()
        _progress_fragment()
        return
    if has_result:
        _show_result()
        return
    _show_input_form()


# ---------------------------------------------------------------------------
# 進捗表示（@st.fragment で部分再描画）
# ---------------------------------------------------------------------------
@st.fragment(run_every=1)
def _progress_fragment():
    """フラグメント内で進捗を自動更新する。"""
    thread = st.session_state.get("bt_thread")
    if thread is not None and not thread.is_alive():
        # 結果を昇格させてページ全体を再描画
        prog = st.session_state.get("bt_progress", {})
        if "_result" in prog:
            st.session_state["bt_result"] = prog.pop("_result")
        elif prog.get("error"):
            st.session_state["bt_result"] = {"error": prog["error"]}
        st.rerun(scope="app")
        return

    progress = st.session_state.get("bt_progress", {})
    start_time = st.session_state.get("bt_start_time")

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
# 結果表示
# ---------------------------------------------------------------------------
def _show_result():
    result = st.session_state["bt_result"]

    if "error" in result:
        st.error(f"バックテストエラー: {result['error']}")
        if st.button("新しいバックテストを開始"):
            _clear_state()
            st.rerun()
        return

    stats = result.get("statistics", {}) or {}
    backtest = result.get("backtest", {}) or {}
    evaluation = result.get("evaluation", {}) or {}
    recent_examples = result.get("recent_examples")
    pending_signals = result.get("pending_signals")
    config_snapshot = result.get("config_snapshot", {})

    st.success(
        f"バックテスト完了 — "
        f"シグナル {config_snapshot.get('n_signals', 0)}件 / "
        f"銘柄 {config_snapshot.get('n_stocks_used', 0)}社"
    )

    render_result_tabs(
        interpretation=evaluation,
        stats=stats,
        backtest=backtest,
        code_or_config=config_snapshot,
        recent_examples=recent_examples,
        code_tab_label="パラメータ設定",
        code_language="json",
        pending_signals=pending_signals,
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("結果を保存", type="primary"):
            _save_result(result)
    with col2:
        if st.button("新しいバックテストを開始"):
            _clear_state()
            st.rerun()


def _save_result(result: dict):
    """結果をDBに保存する"""
    db = get_db()
    config_snapshot = result.get("config_snapshot", {})
    signal_cfg = config_snapshot.get("signal_config", {})

    # タイトル自動生成
    parts = []
    if signal_cfg.get("consecutive_bullish_days"):
        parts.append(f"連続陽線{signal_cfg['consecutive_bullish_days']}日")
    if signal_cfg.get("consecutive_bearish_days"):
        parts.append(f"連続陰線{signal_cfg['consecutive_bearish_days']}日")
    if signal_cfg.get("volume_surge_ratio"):
        parts.append(f"出来高{signal_cfg['volume_surge_ratio']}倍")
    if signal_cfg.get("price_vs_ma25"):
        parts.append(f"25日線{'上' if signal_cfg['price_vs_ma25'] == 'above' else '下'}")
    if signal_cfg.get("rsi_lower"):
        parts.append(f"RSI<{signal_cfg['rsi_lower']}")
    if signal_cfg.get("rsi_upper"):
        parts.append(f"RSI>{signal_cfg['rsi_upper']}")
    if signal_cfg.get("bb_buy_below_lower"):
        parts.append("BB下限")
    if signal_cfg.get("ma_cross_short"):
        parts.append("GC/DC")
    if signal_cfg.get("macd_fast"):
        parts.append("MACD")
    if signal_cfg.get("margin_ratio_min") or signal_cfg.get("margin_ratio_max"):
        parts.append("貸借倍率")
    title = "標準BT: " + "+".join(parts[:5]) if parts else "標準BT: カスタム設定"

    idea_id = db.create_idea(
        title=title,
        description=f"標準バックテスト ({config_snapshot.get('start_date', '')} 〜 {config_snapshot.get('end_date', '')})",
        category="テクニカル",
    )
    plan_id = db.create_plan(
        idea_id=idea_id,
        name=title,
        analysis_method="standard_backtest",
        start_date=config_snapshot.get("start_date"),
        end_date=config_snapshot.get("end_date"),
        parameters=config_snapshot,
    )
    run_id = db.create_run(
        plan_id=plan_id,
        idea_snapshot={"title": title},
        plan_snapshot={"analysis_method": "standard_backtest", "parameters": config_snapshot},
    )

    evaluation = result.get("evaluation", {})
    db.update_run(
        run_id,
        statistics_result=result.get("statistics"),
        backtest_result=result.get("backtest"),
        evaluation=evaluation,
        evaluation_label=evaluation.get("label"),
        status="completed",
        finished_at=datetime.now().isoformat(),
    )
    st.success(f"保存しました (Run #{run_id})")


# ---------------------------------------------------------------------------
# 入力フォーム
# ---------------------------------------------------------------------------
def _show_input_form():
    # ==================================================================
    # セクション1: ユニバース・期間
    # ==================================================================
    with st.expander("ユニバース・期間", expanded=True):
        ucol1, ucol2 = st.columns(2)
        with ucol1:
            selected_markets = st.multiselect(
                "市場区分", options=MARKET_SEGMENTS, help="空 = 全市場",
                key="sbt_markets",
            )
            selected_scales = st.multiselect(
                "TOPIX規模区分", options=TOPIX_SCALE_CATEGORIES, help="空 = 全規模",
                key="sbt_scales",
            )
            margin_only = st.checkbox("貸借銘柄のみ", key="sbt_margin")
            exclude_etf_reit = st.checkbox(
                "ETF・REITを除外", value=True, key="sbt_exclude_etf",
                help="一般株式のみ対象（ETF・REIT・インフラファンド等を除外）",
            )
        with ucol2:
            sector_type = st.radio(
                "業種フィルター", options=["なし", "17業種区分"],
                horizontal=True, key="sbt_sector_type",
            )
            selected_sectors = []
            if sector_type == "17業種区分":
                selected_sectors = st.multiselect(
                    "業種を選択", options=SECTOR_17_LIST, key="sbt_sectors",
                )

        st.markdown("**財務スクリーニング**")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            cap_min = st.number_input("時価総額 下限（億円）", min_value=0.0, value=0.0, step=100.0, key="sbt_cap_min")
            cap_max = st.number_input("時価総額 上限（億円）", min_value=0.0, value=0.0, step=100.0, key="sbt_cap_max")
        with fc2:
            per_min = st.number_input("PER 下限", min_value=0.0, value=0.0, step=1.0, key="sbt_per_min")
            per_max = st.number_input("PER 上限", min_value=0.0, value=0.0, step=1.0, key="sbt_per_max")
        with fc3:
            pbr_min = st.number_input("PBR 下限", min_value=0.0, value=0.0, step=0.1, key="sbt_pbr_min")
            pbr_max = st.number_input("PBR 上限", min_value=0.0, value=0.0, step=0.1, key="sbt_pbr_max")

        st.markdown("**分析期間**")
        dc1, dc2 = st.columns(2)
        with dc1:
            start_date = st.date_input("開始日", value=date(2021, 1, 1), key="sbt_start")
        with dc2:
            end_date = st.date_input("終了日", value=date.today(), key="sbt_end")

        st.markdown("**実行設定**")
        n_recent_examples = st.number_input(
            "直近事例の表示数", min_value=5, max_value=1000,
            value=10, step=5, key="sbt_n_examples",
            help="結果の「直近事例」タブに表示するシグナル発動事例の件数。",
        )
        max_stocks = 4000  # 全銘柄対象

    # ==================================================================
    # セクション2: テクニカルシグナル
    # ==================================================================
    with st.expander("テクニカルシグナル"):
        st.markdown("##### 連続陽線/陰線・出来高")
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            use_bullish = st.checkbox("連続陽線", key="sbt_use_bullish")
            bullish_days = 3
            if use_bullish:
                bullish_days = st.number_input(
                    "連続陽線日数", min_value=2, max_value=20, value=3,
                    key="sbt_bullish_days",
                )
        with tc2:
            use_bearish = st.checkbox("連続陰線", key="sbt_use_bearish")
            bearish_days = 3
            if use_bearish:
                bearish_days = st.number_input(
                    "連続陰線日数", min_value=2, max_value=20, value=3,
                    key="sbt_bearish_days",
                )
        with tc3:
            use_vol = st.checkbox("出来高倍率", key="sbt_use_vol")
            vol_ratio = 1.5
            vol_window = D["volume_surge_window"]
            if use_vol:
                vol_ratio = st.number_input(
                    "出来高倍率閾値", min_value=1.0, max_value=10.0, value=1.5, step=0.1,
                    key="sbt_vol_ratio",
                )
                vol_window = st.number_input(
                    "出来高MA期間", min_value=5, max_value=60, value=D["volume_surge_window"],
                    key="sbt_vol_window",
                )

        st.markdown("##### 移動平均")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            ma25_rel = st.selectbox("25日線との関係", ["なし", "上", "下"], key="sbt_ma25")
        with mc2:
            ma75_rel = st.selectbox("75日線との関係", ["なし", "上", "下"], key="sbt_ma75")
        with mc3:
            ma200_rel = st.selectbox("200日線との関係", ["なし", "上", "下"], key="sbt_ma200")

        use_deviation = st.checkbox("移動平均乖離率", key="sbt_use_dev")
        dev_pct = 5.0
        dev_base = 25
        if use_deviation:
            mc4, mc5 = st.columns(2)
            with mc4:
                dev_pct = st.number_input(
                    "乖離率閾値(%)", min_value=0.1, max_value=50.0, value=5.0, step=0.5,
                    key="sbt_dev_pct",
                )
            with mc5:
                dev_base = st.selectbox(
                    "乖離率の基準MA", [25, 75, 200], key="sbt_dev_base",
                )

        st.markdown("##### RSI")
        rc1, rc2 = st.columns(2)
        rsi_window = D["rsi_window"]
        rsi_lower = 30.0
        rsi_upper = 70.0
        with rc1:
            use_rsi_lower = st.checkbox("RSI下限（売られすぎ）", key="sbt_use_rsi_lo")
            if use_rsi_lower:
                rsi_lower = st.number_input(
                    "RSI下限", min_value=0.0, max_value=100.0, value=30.0, step=5.0,
                    key="sbt_rsi_lo",
                )
        with rc2:
            use_rsi_upper = st.checkbox("RSI上限（買われすぎ）", key="sbt_use_rsi_hi")
            if use_rsi_upper:
                rsi_upper = st.number_input(
                    "RSI上限", min_value=0.0, max_value=100.0, value=70.0, step=5.0,
                    key="sbt_rsi_hi",
                )
        if use_rsi_lower or use_rsi_upper:
            rsi_window = st.number_input(
                "RSI期間", min_value=5, max_value=50, value=D["rsi_window"], key="sbt_rsi_win",
            )

        st.markdown("##### ボリンジャーバンド")
        bb_buy_below = st.checkbox("BB下限タッチで買い", key="sbt_bb_buy")
        bb_window = D["bb_window"]
        bb_std = D["bb_std"]
        if bb_buy_below:
            bc1, bc2 = st.columns(2)
            with bc1:
                bb_window = st.number_input(
                    "BB期間", min_value=5, max_value=60, value=D["bb_window"], key="sbt_bb_win",
                )
            with bc2:
                bb_std = st.number_input(
                    "BB標準偏差倍数", min_value=0.5, max_value=5.0, value=D["bb_std"], step=0.5,
                    key="sbt_bb_std",
                )

        st.markdown("##### ゴールデンクロス / デッドクロス")
        use_cross = st.checkbox("MA交差シグナル", key="sbt_use_cross")
        cross_short = 5
        cross_long = 25
        cross_dir = "ゴールデンクロス"
        if use_cross:
            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                cross_short = st.number_input(
                    "短期MA", min_value=2, max_value=100, value=5,
                    key="sbt_cross_short",
                )
            with gc2:
                cross_long = st.number_input(
                    "長期MA", min_value=5, max_value=300, value=25,
                    key="sbt_cross_long",
                )
            with gc3:
                cross_dir = st.selectbox(
                    "方向", ["ゴールデンクロス", "デッドクロス"],
                    key="sbt_cross_dir",
                )

        st.markdown("##### MACD")
        use_macd = st.checkbox("MACDクロス", key="sbt_use_macd")
        macd_fast = 12
        macd_slow = 26
        macd_signal = 9
        if use_macd:
            md1, md2, md3 = st.columns(3)
            with md1:
                macd_fast = st.number_input(
                    "MACD短期", min_value=2, max_value=50, value=12,
                    key="sbt_macd_fast",
                )
            with md2:
                macd_slow = st.number_input(
                    "MACD長期", min_value=5, max_value=100, value=26,
                    key="sbt_macd_slow",
                )
            with md3:
                macd_signal = st.number_input(
                    "MACDシグナル", min_value=2, max_value=50, value=9,
                    key="sbt_macd_sig",
                )

        st.markdown("##### ATR・一目均衡表・セクター相対強度")
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            use_atr = st.checkbox("ATRフィルター", key="sbt_use_atr")
            atr_max = 3.0
            if use_atr:
                atr_max = st.number_input(
                    "ATR上限(%)", min_value=0.1, max_value=20.0, value=3.0, step=0.5,
                    key="sbt_atr_max",
                )
        with ac2:
            ichimoku_cloud = st.selectbox(
                "一目均衡表: 雲", ["なし", "上", "下"], key="sbt_ichimoku",
            )
            ichimoku_tk = st.checkbox("転換線>基準線", key="sbt_ichimoku_tk")
        with ac3:
            use_sector_rs = st.checkbox("セクター相対強度", key="sbt_use_sector_rs")
            sector_rs_min = 70.0
            if use_sector_rs:
                sector_rs_min = st.number_input(
                    "下限パーセンタイル(%)", min_value=0.0, max_value=100.0, value=70.0, step=5.0,
                    key="sbt_sector_rs_min",
                )

    # ==================================================================
    # セクション3: 信用取引・センチメント
    # ==================================================================
    with st.expander("信用取引・センチメント"):
        margin_type_label = st.selectbox(
            "信用取引データの種類",
            ["合算（週次）", "制度信用（週次）", "一般信用（週次）"],
            key="sbt_margin_type",
            help="J-Quants APIの信用取引データは全て週次（毎週金曜更新）。"
                 "合算=制度+一般の合計、制度信用=取引所ルールに基づく信用取引、"
                 "一般信用=証券会社独自ルールの信用取引。",
        )
        cr1, cr2, cr3 = st.columns(3)
        with cr1:
            use_margin_min = st.checkbox("貸借倍率 下限", key="sbt_use_mr_min")
            margin_ratio_min = 1.0
            if use_margin_min:
                margin_ratio_min = st.number_input(
                    "貸借倍率 下限", min_value=0.0, max_value=100.0, value=1.0, step=0.5,
                    key="sbt_mr_min",
                )
        with cr2:
            use_margin_max = st.checkbox("貸借倍率 上限", key="sbt_use_mr_max")
            margin_ratio_max = 5.0
            if use_margin_max:
                margin_ratio_max = st.number_input(
                    "貸借倍率 上限", min_value=0.0, max_value=100.0, value=5.0, step=0.5,
                    key="sbt_mr_max",
                )
        with cr3:
            use_short_ratio = st.checkbox("空売り比率 上限", key="sbt_use_sr")
            short_ratio_max = 40.0
            if use_short_ratio:
                short_ratio_max = st.number_input(
                    "空売り比率 上限(%)", min_value=0.0, max_value=100.0, value=40.0, step=5.0,
                    key="sbt_sr_max",
                )

        st.markdown("**回転日数フィルター（信用残÷20日平均出来高）**")
        td1, td2 = st.columns(2)
        with td1:
            use_buy_td_min = st.checkbox("買い残回転日数 下限", key="sbt_use_buy_td_min")
            buy_td_min = 5.0
            if use_buy_td_min:
                buy_td_min = st.number_input(
                    "買い残回転日数 下限", min_value=0.0, max_value=200.0, value=5.0, step=1.0,
                    key="sbt_buy_td_min",
                )
            use_buy_td_max = st.checkbox("買い残回転日数 上限", key="sbt_use_buy_td_max")
            buy_td_max = 30.0
            if use_buy_td_max:
                buy_td_max = st.number_input(
                    "買い残回転日数 上限", min_value=0.0, max_value=200.0, value=30.0, step=1.0,
                    key="sbt_buy_td_max",
                )
        with td2:
            use_sell_td_min = st.checkbox("売り残回転日数 下限", key="sbt_use_sell_td_min")
            sell_td_min = 5.0
            if use_sell_td_min:
                sell_td_min = st.number_input(
                    "売り残回転日数 下限", min_value=0.0, max_value=200.0, value=5.0, step=1.0,
                    key="sbt_sell_td_min",
                )
            use_sell_td_max = st.checkbox("売り残回転日数 上限", key="sbt_use_sell_td_max")
            sell_td_max = 30.0
            if use_sell_td_max:
                sell_td_max = st.number_input(
                    "売り残回転日数 上限", min_value=0.0, max_value=200.0, value=30.0, step=1.0,
                    key="sbt_sell_td_max",
                )

        st.markdown("**対出来高比率フィルター（信用残÷当日出来高）**")
        vr1, vr2 = st.columns(2)
        with vr1:
            use_buy_vr_min = st.checkbox("買い残対出来高比率 下限", key="sbt_use_buy_vr_min")
            buy_vr_min = 1.0
            if use_buy_vr_min:
                buy_vr_min = st.number_input(
                    "買い残対出来高比率 下限", min_value=0.0, max_value=100.0, value=1.0, step=0.5,
                    key="sbt_buy_vr_min",
                )
            use_buy_vr_max = st.checkbox("買い残対出来高比率 上限", key="sbt_use_buy_vr_max")
            buy_vr_max = 10.0
            if use_buy_vr_max:
                buy_vr_max = st.number_input(
                    "買い残対出来高比率 上限", min_value=0.0, max_value=100.0, value=10.0, step=0.5,
                    key="sbt_buy_vr_max",
                )
        with vr2:
            use_sell_vr_min = st.checkbox("売り残対出来高比率 下限", key="sbt_use_sell_vr_min")
            sell_vr_min = 1.0
            if use_sell_vr_min:
                sell_vr_min = st.number_input(
                    "売り残対出来高比率 下限", min_value=0.0, max_value=100.0, value=1.0, step=0.5,
                    key="sbt_sell_vr_min",
                )
            use_sell_vr_max = st.checkbox("売り残対出来高比率 上限", key="sbt_use_sell_vr_max")
            sell_vr_max = 10.0
            if use_sell_vr_max:
                sell_vr_max = st.number_input(
                    "売り残対出来高比率 上限", min_value=0.0, max_value=100.0, value=10.0, step=0.5,
                    key="sbt_sell_vr_max",
                )

        st.markdown("**前週比変化率フィルター**")
        ch1, ch2, ch3 = st.columns(3)
        with ch1:
            use_buy_chg_min = st.checkbox("買い残変化率 下限", key="sbt_use_buy_chg_min")
            buy_chg_min = 10.0
            if use_buy_chg_min:
                buy_chg_min = st.number_input(
                    "買い残変化率 下限(%)", min_value=-100.0, max_value=500.0, value=10.0, step=5.0,
                    key="sbt_buy_chg_min",
                )
            use_buy_chg_max = st.checkbox("買い残変化率 上限", key="sbt_use_buy_chg_max")
            buy_chg_max = -5.0
            if use_buy_chg_max:
                buy_chg_max = st.number_input(
                    "買い残変化率 上限(%)", min_value=-100.0, max_value=500.0, value=-5.0, step=5.0,
                    key="sbt_buy_chg_max",
                )
        with ch2:
            use_sell_chg_min = st.checkbox("売り残変化率 下限", key="sbt_use_sell_chg_min")
            sell_chg_min = 10.0
            if use_sell_chg_min:
                sell_chg_min = st.number_input(
                    "売り残変化率 下限(%)", min_value=-100.0, max_value=500.0, value=10.0, step=5.0,
                    key="sbt_sell_chg_min",
                )
            use_sell_chg_max = st.checkbox("売り残変化率 上限", key="sbt_use_sell_chg_max")
            sell_chg_max = -5.0
            if use_sell_chg_max:
                sell_chg_max = st.number_input(
                    "売り残変化率 上限(%)", min_value=-100.0, max_value=500.0, value=-5.0, step=5.0,
                    key="sbt_sell_chg_max",
                )
        with ch3:
            use_mr_chg_min = st.checkbox("貸借倍率変化率 下限", key="sbt_use_mr_chg_min")
            mr_chg_min = 10.0
            if use_mr_chg_min:
                mr_chg_min = st.number_input(
                    "貸借倍率変化率 下限(%)", min_value=-100.0, max_value=500.0, value=10.0, step=5.0,
                    key="sbt_mr_chg_min",
                )
            use_mr_chg_max = st.checkbox("貸借倍率変化率 上限", key="sbt_use_mr_chg_max")
            mr_chg_max = -5.0
            if use_mr_chg_max:
                mr_chg_max = st.number_input(
                    "貸借倍率変化率 上限(%)", min_value=-100.0, max_value=500.0, value=-5.0, step=5.0,
                    key="sbt_mr_chg_max",
                )

        st.markdown("**対出来高比率の前週比変化率フィルター**")
        vc1, vc2 = st.columns(2)
        with vc1:
            use_buy_vrc_min = st.checkbox("買い残対出来高比率変化率 下限", key="sbt_use_buy_vrc_min")
            buy_vrc_min = 10.0
            if use_buy_vrc_min:
                buy_vrc_min = st.number_input(
                    "買い残対出来高比率変化率 下限(%)", min_value=-100.0, max_value=500.0, value=10.0, step=5.0,
                    key="sbt_buy_vrc_min",
                )
            use_buy_vrc_max = st.checkbox("買い残対出来高比率変化率 上限", key="sbt_use_buy_vrc_max")
            buy_vrc_max = -5.0
            if use_buy_vrc_max:
                buy_vrc_max = st.number_input(
                    "買い残対出来高比率変化率 上限(%)", min_value=-100.0, max_value=500.0, value=-5.0, step=5.0,
                    key="sbt_buy_vrc_max",
                )
        with vc2:
            use_sell_vrc_min = st.checkbox("売り残対出来高比率変化率 下限", key="sbt_use_sell_vrc_min")
            sell_vrc_min = 10.0
            if use_sell_vrc_min:
                sell_vrc_min = st.number_input(
                    "売り残対出来高比率変化率 下限(%)", min_value=-100.0, max_value=500.0, value=10.0, step=5.0,
                    key="sbt_sell_vrc_min",
                )
            use_sell_vrc_max = st.checkbox("売り残対出来高比率変化率 上限", key="sbt_use_sell_vrc_max")
            sell_vrc_max = -5.0
            if use_sell_vrc_max:
                sell_vrc_max = st.number_input(
                    "売り残対出来高比率変化率 上限(%)", min_value=-100.0, max_value=500.0, value=-5.0, step=5.0,
                    key="sbt_sell_vrc_max",
                )

    # ==================================================================
    # セクション4: 測定・コスト設定
    # ==================================================================
    with st.expander("測定期間・コスト設定"):
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            holding_days = st.number_input(
                "測定期間（営業日）", min_value=1, max_value=252, value=D["holding_period_days"],
                key="sbt_hold",
                help="シグナル発生後、何営業日後のリターンを測定するか。",
            )
        with mc2:
            commission_pct = st.number_input(
                "売買手数料(%)", min_value=0.0, max_value=5.0,
                value=D["commission_rate"] * 100, step=0.01, key="sbt_comm",
                help="往復コストとしてリターンから控除されます。",
            )
        with mc3:
            slippage_pct = st.number_input(
                "スリッページ(%)", min_value=0.0, max_value=5.0,
                value=D["slippage_rate"] * 100, step=0.01, key="sbt_slip",
                help="往復コストとしてリターンから控除されます。",
            )
        initial_capital = D["initial_capital"]

    # ==================================================================
    # シグナル結合ロジック
    # ==================================================================
    signal_logic = st.radio(
        "シグナル結合ロジック",
        ["AND（全条件一致）", "OR（いずれか一致）"],
        horizontal=True, key="sbt_logic",
    )

    st.markdown("---")

    # ==================================================================
    # 実行ボタン
    # ==================================================================
    if st.button("バックテスト実行", type="primary", width='stretch'):
        # --- SignalConfig 構築 ---
        sig_cfg = SignalConfig(
            consecutive_bullish_days=bullish_days if use_bullish else None,
            consecutive_bearish_days=bearish_days if use_bearish else None,
            volume_surge_ratio=vol_ratio if use_vol else None,
            volume_surge_window=vol_window if use_vol else D["volume_surge_window"],
            price_vs_ma25=_ma_rel(ma25_rel),
            price_vs_ma75=_ma_rel(ma75_rel),
            price_vs_ma200=_ma_rel(ma200_rel),
            ma_deviation_pct=dev_pct if use_deviation else None,
            ma_deviation_window=dev_base if use_deviation else 25,
            rsi_window=rsi_window,
            rsi_lower=rsi_lower if use_rsi_lower else None,
            rsi_upper=rsi_upper if use_rsi_upper else None,
            bb_window=bb_window,
            bb_std=bb_std,
            bb_buy_below_lower=bb_buy_below,
            ma_cross_short=cross_short if use_cross else None,
            ma_cross_long=cross_long if use_cross else None,
            ma_cross_type="golden_cross" if cross_dir == "ゴールデンクロス" else "dead_cross",
            macd_fast=macd_fast if use_macd else None,
            macd_slow=macd_slow if use_macd else None,
            macd_signal=macd_signal if use_macd else 9,
            atr_window=14,
            atr_max=atr_max if use_atr else None,
            ichimoku_cloud=_ichimoku_rel(ichimoku_cloud),
            ichimoku_tenkan_above_kijun=ichimoku_tk,
            sector_relative_strength_min=sector_rs_min if use_sector_rs else None,
            sector_relative_lookback=D["sector_relative_lookback"],
            margin_type=_margin_type_value(margin_type_label),
            margin_ratio_min=margin_ratio_min if use_margin_min else None,
            margin_ratio_max=margin_ratio_max if use_margin_max else None,
            short_selling_ratio_max=short_ratio_max if use_short_ratio else None,
            margin_buy_change_pct_min=buy_chg_min if use_buy_chg_min else None,
            margin_buy_change_pct_max=buy_chg_max if use_buy_chg_max else None,
            margin_sell_change_pct_min=sell_chg_min if use_sell_chg_min else None,
            margin_sell_change_pct_max=sell_chg_max if use_sell_chg_max else None,
            margin_ratio_change_pct_min=mr_chg_min if use_mr_chg_min else None,
            margin_ratio_change_pct_max=mr_chg_max if use_mr_chg_max else None,
            margin_buy_turnover_days_min=buy_td_min if use_buy_td_min else None,
            margin_buy_turnover_days_max=buy_td_max if use_buy_td_max else None,
            margin_sell_turnover_days_min=sell_td_min if use_sell_td_min else None,
            margin_sell_turnover_days_max=sell_td_max if use_sell_td_max else None,
            margin_buy_vol_ratio_min=buy_vr_min if use_buy_vr_min else None,
            margin_buy_vol_ratio_max=buy_vr_max if use_buy_vr_max else None,
            margin_sell_vol_ratio_min=sell_vr_min if use_sell_vr_min else None,
            margin_sell_vol_ratio_max=sell_vr_max if use_sell_vr_max else None,
            margin_buy_vol_ratio_change_pct_min=buy_vrc_min if use_buy_vrc_min else None,
            margin_buy_vol_ratio_change_pct_max=buy_vrc_max if use_buy_vrc_max else None,
            margin_sell_vol_ratio_change_pct_min=sell_vrc_min if use_sell_vrc_min else None,
            margin_sell_vol_ratio_change_pct_max=sell_vrc_max if use_sell_vrc_max else None,
            holding_period_days=holding_days,
            signal_logic="AND" if "AND" in signal_logic else "OR",
        )

        if not sig_cfg.has_any_signal():
            st.error("少なくとも1つのシグナル条件を有効にしてください。")
            return

        # --- UniverseFilterConfig ---
        uni_cfg = UniverseFilterConfig(
            market_segments=selected_markets,
            scale_categories=selected_scales,
            sector_filter_type="sector_17" if sector_type == "17業種区分" else "none",
            selected_sectors=selected_sectors,
            margin_tradable_only=margin_only,
            exclude_etf_reit=exclude_etf_reit,
            market_cap_min=cap_min if cap_min > 0 else None,
            market_cap_max=cap_max if cap_max > 0 else None,
            per_min=per_min if per_min > 0 else None,
            per_max=per_max if per_max > 0 else None,
            pbr_min=pbr_min if pbr_min > 0 else None,
            pbr_max=pbr_max if pbr_max > 0 else None,
        )

        # --- スレッド起動 ---
        progress_dict = {"pct": 0.0, "message": "開始中..."}
        st.session_state["bt_progress"] = progress_dict
        st.session_state["bt_start_time"] = datetime.now()
        st.session_state.pop("bt_result", None)

        t = threading.Thread(
            target=_run_bt_thread,
            args=(
                progress_dict,
                get_db(),
                get_data_provider(),
                sig_cfg,
                uni_cfg,
                initial_capital,
                commission_pct / 100,
                slippage_pct / 100,
                str(start_date),
                str(end_date),
                max_stocks,
                n_recent_examples,
            ),
            daemon=True,
        )
        st.session_state["bt_thread"] = t
        t.start()
        st.rerun()


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------
def _ma_rel(val: str) -> str | None:
    if val == "上":
        return "above"
    elif val == "下":
        return "below"
    return None


def _ichimoku_rel(val: str) -> str | None:
    if val == "上":
        return "above"
    elif val == "下":
        return "below"
    return None


def _margin_type_value(label: str) -> str:
    if "制度" in label:
        return "standard"
    elif "一般" in label:
        return "negotiable"
    return "combined"


def _clear_state():
    for key in ("bt_thread", "bt_progress", "bt_start_time", "bt_result"):
        st.session_state.pop(key, None)


if __name__ == "__main__":
    main()
