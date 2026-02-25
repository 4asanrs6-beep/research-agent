"""研究ページ — 仮説入力 → パラメータ選択 → Phase 1 探索 + グリッドサーチ → 結果表示

3フェーズ構造:
  Phase 1: 探索 — 5つの多様なアプローチを一気に試す
  Phase 2: グリッド設計 — AIが最有望アプローチを分析、チューニングパラメータを選定
  Phase 3: グリッド実行 — パラメータの全組み合わせを自動試行（最大30通り）

全 AI 処理はバックグラウンドスレッドで実行。
待機画面は @st.fragment(run_every=2) でフラグメントのみ再描画。
"""

import json
import threading
from datetime import date, datetime

import pandas as pd
import streamlit as st

from config import (
    DB_PATH, MARKET_DATA_DIR, ANALYSIS_CATEGORIES, JQUANTS_API_KEY,
    AI_RESEARCH_PHASE1_CONFIGS,
    AI_RESEARCH_GRID_MAX_COMBINATIONS,
)
from db.database import Database
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider
from core.ai_client import create_ai_client
from core.ai_researcher import AiResearcher, ResearchProgress
from core.ai_parameter_selector import ParameterSelectionResult, dict_to_signal_config
from core.universe_filter import (
    UniverseFilterConfig,
    build_universe_description,
    MARKET_SEGMENTS,
    TOPIX_SCALE_CATEGORIES,
    SECTOR_17_LIST,
)
from core.styles import apply_reuters_style, apply_waiting_overlay, render_status_badge
from core.result_display import render_result_tabs
from core.sidebar import render_sidebar_running_indicator

st.set_page_config(page_title="研究", page_icon="R", layout="wide")

_PHASE_LABELS = {1: "探索", 2: "グリッド設計", 3: "グリッド実行"}


@st.cache_resource
def get_db():
    return Database(DB_PATH)


@st.cache_resource
def get_data_provider():
    cache = DataCache(MARKET_DATA_DIR)
    return JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)


# ===========================================================================
# バックグラウンドスレッド関数（Streamlit API 使用禁止）
# ===========================================================================
def _thread_generate_params(shared, hypothesis, universe_desc, start_date, end_date):
    """AIにSignalConfigパラメータを選択させる"""
    try:
        ai_client = create_ai_client()
        from core.ai_parameter_selector import AiParameterSelector
        selector = AiParameterSelector(ai_client)
        shared["message"] = "AIがパラメータを選択しています..."
        result = selector.select_parameters(
            hypothesis=hypothesis,
            universe_desc=universe_desc,
            start_date=start_date,
            end_date=end_date,
        )
        shared["_result"] = result
    except Exception as e:
        shared["error"] = str(e)


def _thread_execute_loop(shared, db, provider, hypothesis, idea_title, category,
                         signal_config_dict, universe_config, start_date, end_date,
                         universe_filter_text):
    """3フェーズ研究ループを実行"""
    try:
        ai_client = create_ai_client()
        researcher = AiResearcher(
            db=db, ai_client=ai_client, data_provider=provider,
        )

        def on_progress(prog):
            shared["message"] = prog.message
            shared["current_iteration"] = prog.current_iteration
            shared["max_iterations"] = prog.max_iterations
            shared["current_phase"] = prog.current_phase
            shared["phase_label"] = prog.phase_label
            shared["current_config_in_phase"] = prog.current_config_in_phase
            shared["total_configs_in_phase"] = prog.total_configs_in_phase

        result = researcher.run_research_loop(
            hypothesis=hypothesis,
            idea_title=idea_title,
            category=category,
            universe_config=universe_config,
            start_date=start_date,
            end_date=end_date,
            initial_config_dict=signal_config_dict,
            on_progress=on_progress,
            universe_filter_text=universe_filter_text,
        )
        shared["_result"] = result
    except Exception as e:
        shared["error"] = str(e)


# ===========================================================================
# メイン
# ===========================================================================
def main():
    apply_reuters_style()
    render_sidebar_running_indicator()
    st.markdown("# Research")
    st.caption("投資仮説を入力 → AIがパラメータ選択 → Phase 1 探索 + グリッドサーチで検証")

    _check_thread_completion()

    phase = st.session_state.get("rp_phase", "idle")

    if phase == "idle":
        _show_input_form()
    elif phase == "generating_params":
        apply_waiting_overlay()
        _waiting_fragment("パラメータを生成中...")
    elif phase == "params_ready":
        _show_params_review()
    elif phase == "executing":
        apply_waiting_overlay()
        _waiting_fragment("3フェーズ研究実行中...")
    elif phase == "done":
        _show_result()


# ===========================================================================
# スレッド完了チェック
# ===========================================================================
def _check_thread_completion():
    thread = st.session_state.get("rp_thread")
    if thread is None or thread.is_alive():
        return

    progress = st.session_state.get("rp_progress", {})
    phase = st.session_state.get("rp_phase", "idle")

    st.session_state.pop("rp_thread", None)
    st.session_state.pop("rp_start_time", None)

    # --- エラー ---
    if "error" in progress:
        if phase == "generating_params":
            st.session_state["rp_phase"] = "idle"
        elif phase == "executing":
            meta = st.session_state.get("rp_meta", {})
            st.session_state["rp_result"] = ResearchProgress(
                phase="error",
                error=progress["error"],
                idea_title=meta.get("idea_title", ""),
                idea_text=meta.get("idea_text", ""),
                category=meta.get("category", ""),
                universe_filter_text=meta.get("universe_filter_text", ""),
                start_date=meta.get("start_date", ""),
                end_date=meta.get("end_date", ""),
            )
            st.session_state["rp_phase"] = "done"
        st.session_state["_rp_error"] = progress["error"]
        return

    # --- 成功 ---
    result = progress.get("_result")
    if result is None:
        return

    if phase == "generating_params":
        st.session_state["rp_param_result"] = result
        st.session_state["rp_phase"] = "params_ready"
    elif phase == "executing":
        st.session_state["rp_result"] = result
        st.session_state["rp_phase"] = "done"


# ===========================================================================
# 待機画面
# ===========================================================================
@st.fragment(run_every=2)
def _waiting_fragment(title: str):
    thread = st.session_state.get("rp_thread")
    if thread is not None and not thread.is_alive():
        _check_thread_completion()
        st.rerun(scope="app")
        return

    progress = st.session_state.get("rp_progress", {})
    start_time = st.session_state.get("rp_start_time")

    st.markdown(f"### {title}")
    st.write(progress.get("message", "処理中..."))

    # フェーズ進捗
    current_phase = progress.get("current_phase", 0)
    phase_label = progress.get("phase_label", "")
    config_in_phase = progress.get("current_config_in_phase", 0)
    total_in_phase = progress.get("total_configs_in_phase", 0)

    if current_phase > 0:
        st.markdown(f"**Phase {current_phase}/3: {phase_label}**")

        # フェーズ内プログレスバー
        if total_in_phase > 0 and config_in_phase > 0:
            pct = min(config_in_phase / total_in_phase, 1.0)
            st.progress(pct, text=f"パラメータセット {config_in_phase}/{total_in_phase} 実行中...")

    # 全体プログレスバー
    cur = progress.get("current_iteration", 0)
    mx = progress.get("max_iterations", 0)
    if cur > 0 and mx > 0:
        pct = min(cur / mx, 1.0)
        st.progress(pct, text=f"全体進捗: {cur}/{mx} パターン完了")

    elapsed = ""
    if start_time:
        total_sec = int((datetime.now() - start_time).total_seconds())
        mm, ss = divmod(total_sec, 60)
        elapsed = f"経過時間: {mm:02d}:{ss:02d}"
    st.caption(elapsed)


# ===========================================================================
# ステージ 1: 入力フォーム
# ===========================================================================
def _show_input_form():
    error = st.session_state.pop("_rp_error", None)
    if error:
        st.error(f"エラー: {error}")

    from core.ai_client import ClaudeCodeClient
    if not ClaudeCodeClient().is_available():
        st.warning(
            "Claude Code CLI が検出されません。"
            "デモモード（ダミー応答）で動作します。"
        )

    idea_title = st.text_input(
        "タイトル",
        placeholder="例: 3日連続陽線+出来高急増銘柄の20日後リターン",
    )
    idea_text = st.text_area(
        "投資仮説の詳細",
        height=150,
        placeholder=(
            "例: 3日連続で陽線を付け、かつ直近の出来高が過去20日平均の2倍以上に急増した銘柄は、"
            "20営業日後にTOPIXを上回るリターンを示すはずだ。"
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

    # --- パラメータ生成ボタン ---
    if st.button("AIにパラメータを選択させる", type="primary", width='stretch', disabled=not idea_text):
        if not idea_text:
            st.error("仮説の説明を入力してください。")
            return

        shared = {"message": "開始中..."}
        meta = {
            "idea_text": idea_text,
            "idea_title": idea_title or idea_text[:50],
            "category": category,
            "universe_filter_text": universe_filter_text,
            "universe_config": universe_config,
            "start_date": str(start_date),
            "end_date": str(end_date),
        }
        st.session_state["rp_progress"] = shared
        st.session_state["rp_start_time"] = datetime.now()
        st.session_state["rp_phase"] = "generating_params"
        st.session_state["rp_meta"] = meta

        t = threading.Thread(
            target=_thread_generate_params,
            args=(
                shared, idea_text, universe_filter_text,
                str(start_date), str(end_date),
            ),
            daemon=True,
        )
        st.session_state["rp_thread"] = t
        t.start()
        st.rerun()


# ===========================================================================
# ステージ 2: パラメータレビュー
# ===========================================================================
def _show_params_review():
    error = st.session_state.pop("_rp_error", None)
    if error:
        st.error(f"エラー: {error}")

    param_result: ParameterSelectionResult = st.session_state["rp_param_result"]
    meta = st.session_state["rp_meta"]

    st.markdown("## AIが選択したパラメータ")

    # 入力条件
    with st.expander("入力条件", expanded=False):
        st.markdown(f"**タイトル:** {meta['idea_title']}")
        st.markdown(f"**カテゴリ:** {meta['category']}")
        st.markdown("**仮説:**")
        st.info(meta["idea_text"])
        if meta["start_date"] or meta["end_date"]:
            st.markdown(f"**分析期間:** {meta['start_date']} 〜 {meta['end_date']}")
        if meta["universe_filter_text"]:
            st.markdown(f"**ユニバース条件:** {meta['universe_filter_text']}")

    # パラメータ選択理由
    if param_result.reasoning:
        st.markdown("#### パラメータ選択理由")
        st.write(param_result.reasoning)

    # 仮説↔パラメータ対応
    if param_result.hypothesis_mapping:
        st.markdown("#### 仮説とパラメータの対応")
        st.write(param_result.hypothesis_mapping)

    # テスト不可能な側面
    if param_result.unmappable_aspects:
        st.warning("**この仮説ではテストできない側面:**\n" +
                   "\n".join(f"- {a}" for a in param_result.unmappable_aspects))

    # パラメータテーブル
    st.markdown("#### シグナルパラメータ")
    cfg_dict = param_result.signal_config_dict
    _JP_LABELS = {
        "consecutive_bullish_days": "連続陽線日数",
        "consecutive_bearish_days": "連続陰線日数",
        "volume_surge_ratio": "出来高倍率閾値",
        "volume_surge_window": "出来高MA期間",
        "price_vs_ma25": "25日線との関係",
        "price_vs_ma75": "75日線との関係",
        "price_vs_ma200": "200日線との関係",
        "ma_deviation_pct": "移動平均乖離率(%)",
        "rsi_lower": "RSI下限",
        "rsi_upper": "RSI上限",
        "bb_buy_below_lower": "BB下限タッチで買い",
        "ma_cross_short": "GC/DC 短期MA",
        "ma_cross_long": "GC/DC 長期MA",
        "ma_cross_type": "GC/DC方向",
        "macd_fast": "MACD短期",
        "macd_slow": "MACD長期",
        "atr_max": "ATRフィルター(%)",
        "ichimoku_cloud": "一目均衡表: 雲",
        "ichimoku_tenkan_above_kijun": "転換線>基準線",
        "sector_relative_strength_min": "セクター相対強度下限(%)",
        "margin_ratio_min": "貸借倍率下限",
        "margin_ratio_max": "貸借倍率上限",
        "short_selling_ratio_max": "空売り比率上限",
        "margin_buy_change_pct_min": "買い残変化率 下限(%)",
        "margin_buy_change_pct_max": "買い残変化率 上限(%)",
        "margin_sell_change_pct_min": "売り残変化率 下限(%)",
        "margin_sell_change_pct_max": "売り残変化率 上限(%)",
        "margin_ratio_change_pct_min": "貸借倍率変化率 下限(%)",
        "margin_ratio_change_pct_max": "貸借倍率変化率 上限(%)",
        "margin_buy_turnover_days_min": "買い残回転日数 下限",
        "margin_buy_turnover_days_max": "買い残回転日数 上限",
        "margin_sell_turnover_days_min": "売り残回転日数 下限",
        "margin_sell_turnover_days_max": "売り残回転日数 上限",
        "margin_buy_vol_ratio_min": "買い残対出来高比率 下限",
        "margin_buy_vol_ratio_max": "買い残対出来高比率 上限",
        "margin_sell_vol_ratio_min": "売り残対出来高比率 下限",
        "margin_sell_vol_ratio_max": "売り残対出来高比率 上限",
        "margin_buy_vol_ratio_change_pct_min": "買い残対出来高比率変化率 下限(%)",
        "margin_buy_vol_ratio_change_pct_max": "買い残対出来高比率変化率 上限(%)",
        "margin_sell_vol_ratio_change_pct_min": "売り残対出来高比率変化率 下限(%)",
        "margin_sell_vol_ratio_change_pct_max": "売り残対出来高比率変化率 上限(%)",
        "holding_period_days": "測定期間(営業日)",
        "signal_logic": "シグナル結合ロジック",
    }
    rows = []
    for k, v in cfg_dict.items():
        label = _JP_LABELS.get(k, k)
        rows.append({"パラメータ": label, "キー": k, "値": str(v)})
    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    else:
        st.warning("パラメータが空です")

    # 生のJSON（折りたたみ）
    with st.expander("JSON（コピー用）"):
        st.code(json.dumps(cfg_dict, ensure_ascii=False, indent=2), language="json")

    st.markdown("---")

    # フェーズ説明
    st.info(
        f"**Phase 1 探索 + グリッドサーチで検証します**\n\n"
        f"- Phase 1 探索: AIが{AI_RESEARCH_PHASE1_CONFIGS}つの異なるアプローチを試行\n"
        f"- Phase 2 グリッド設計: AIが最も有望なアプローチを分析、チューニングパラメータを選定\n"
        f"- Phase 3 グリッド実行: パラメータの全組み合わせを自動試行（最大{AI_RESEARCH_GRID_MAX_COMBINATIONS}通り）\n"
        f"- データは初回のみ取得、2回目以降は高速"
    )

    col1, col2 = st.columns(2)
    with col1:
        execute = st.button("この設定で実行", type="primary", width='stretch')
    with col2:
        if st.button("やり直す", width='stretch'):
            _clear_state()
            st.rerun()

    if execute:
        shared = {
            "message": "開始中...",
            "current_iteration": 0,
            "max_iterations": AI_RESEARCH_PHASE1_CONFIGS,  # Phase 2完了時に動的更新
            "current_phase": 0,
            "phase_label": "",
            "current_config_in_phase": 0,
            "total_configs_in_phase": 0,
        }
        st.session_state["rp_progress"] = shared
        st.session_state["rp_start_time"] = datetime.now()
        st.session_state["rp_phase"] = "executing"

        t = threading.Thread(
            target=_thread_execute_loop,
            args=(
                shared, get_db(), get_data_provider(),
                meta["idea_text"], meta["idea_title"], meta["category"],
                cfg_dict, meta["universe_config"],
                meta["start_date"], meta["end_date"],
                meta["universe_filter_text"],
            ),
            daemon=True,
        )
        st.session_state["rp_thread"] = t
        t.start()
        st.rerun()


# ===========================================================================
# イテレーション比較サマリー
# ===========================================================================
def _render_iteration_comparison(iterations, best_idx):
    """イテレーション比較テーブルを表示する"""
    if not iterations or len(iterations) < 1:
        return

    rows = []
    for i, it in enumerate(iterations):
        bt = it.backtest_result
        is_best = (best_idx is not None and i == best_idx)
        phase_label = _PHASE_LABELS.get(it.phase, "?")

        if "error" in bt:
            rows.append({
                "#": f"{it.iteration}{'★' if is_best else ''}",
                "Phase": phase_label,
                "アプローチ": it.approach_name or it.changes_description or "初回",
                "シグナル": 0,
                "超過リターン": "エラー",
                "p値": "-",
                "効果量d": "-",
                "エッジ": "-",
            })
            continue

        stats = bt.get("statistics", {})
        backtest_data = bt.get("backtest", bt)
        n_valid = backtest_data.get("n_valid_signals", 0)
        mean_excess = stats.get("excess_mean") or backtest_data.get("mean_excess_return") or 0
        p_value = stats.get("p_value")
        p_value = p_value if p_value is not None else 1.0
        cohens_d = stats.get("cohens_d") or 0

        if mean_excess > 0:
            edge = "ロング"
        elif mean_excess < 0:
            edge = "ショート"
        else:
            edge = "-"

        rows.append({
            "#": f"{it.iteration}{'★' if is_best else ''}",
            "Phase": phase_label,
            "アプローチ": it.approach_name or it.changes_description or "初回",
            "シグナル": n_valid,
            "超過リターン": f"{mean_excess:+.2%}",
            "p値": f"{p_value:.4f}",
            "効果量d": f"{cohens_d:+.3f}",
            "エッジ": edge,
        })

    if not rows:
        return

    st.markdown("#### 全パターン比較")
    df = pd.DataFrame(rows)

    def _highlight_best(row):
        if "★" in str(row["#"]):
            return ["background-color: rgba(255, 128, 0, 0.15)"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_highlight_best, axis=1)
    st.dataframe(styled, width='stretch', hide_index=True)
    st.caption("★ = ベスト結果")


# ===========================================================================
# ステージ 3: 結果表示
# ===========================================================================
def _show_result():
    result: ResearchProgress = st.session_state["rp_result"]

    if result.phase == "error":
        st.error(f"研究中にエラーが発生しました: {result.error}")
        _render_conditions_expander(result)
        if st.button("新しい研究を開始"):
            _clear_state()
            st.rerun()
        return

    st.success(f"研究が完了しました (Run #{result.run_id})")

    _render_conditions_expander(result)

    iterations = result.iterations or []
    best_idx = result.best_iteration_index

    if not iterations:
        st.warning("イテレーション結果がありません")
        if st.button("新しい研究を開始"):
            _clear_state()
            st.rerun()
        return

    # --- 全パターン比較サマリー ---
    _render_iteration_comparison(iterations, best_idx)

    # --- フェーズ別にグループ化 ---
    phase1_its = [it for it in iterations if it.phase == 1]
    phase3_its = [it for it in iterations if it.phase == 3]  # グリッド実行結果

    # --- タブ構成: ベスト結果 | 探索(5件) | グリッド設計 | グリッド結果(N件) | AI総合評価 ---
    tab_labels = ["ベスト結果"]
    if phase1_its:
        tab_labels.append(f"Phase 1 探索 ({len(phase1_its)}件)")
    tab_labels.append("Phase 2 グリッド設計")
    if phase3_its:
        tab_labels.append(f"Phase 3 グリッド結果 ({len(phase3_its)}件)")
    tab_labels.append("AI総合評価")

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    # --- ベスト結果タブ ---
    with tabs[tab_idx]:
        if best_idx is not None and best_idx < len(iterations):
            best_it = iterations[best_idx]
            phase_name = _PHASE_LABELS.get(best_it.phase, "?")
            st.markdown(f"**ベスト: #{best_it.iteration} — {best_it.approach_name}** (Phase {best_it.phase}: {phase_name})")
            _render_iteration_result(best_it, key_suffix="best")
        else:
            st.warning("ベスト結果を特定できませんでした")
    tab_idx += 1

    # --- Phase 1 タブ ---
    if phase1_its:
        with tabs[tab_idx]:
            st.markdown("### Phase 1: 探索")
            st.caption("全く異なるアプローチを幅広く試行")
            for i, it in enumerate(phase1_its):
                is_best = (best_idx is not None and iterations.index(it) == best_idx)
                with st.expander(
                    f"{'★ ' if is_best else ''}{it.approach_name} — "
                    f"{_get_result_summary(it)}",
                    expanded=is_best,
                ):
                    if is_best:
                        st.info("このパターンがベスト結果に選ばれました")
                    _render_iteration_result(it, key_suffix=f"p1_{i}")
        tab_idx += 1

    # --- Phase 2 グリッド設計タブ ---
    with tabs[tab_idx]:
        st.markdown("### Phase 2: グリッド設計")
        st.caption("AIがPhase 1の結果を分析し、グリッドサーチのパラメータを設計")
        grid_spec = result.grid_spec
        if grid_spec:
            if grid_spec.get("analysis"):
                st.markdown("#### AI分析")
                st.write(grid_spec["analysis"])
            if grid_spec.get("selected_approach"):
                st.markdown(f"**ベースアプローチ:** {grid_spec['selected_approach']}")
            if grid_spec.get("reasoning"):
                st.markdown(f"**設計理由:** {grid_spec['reasoning']}")
            st.markdown("---")
            if grid_spec.get("base_config"):
                st.markdown("#### ベース設定")
                st.code(json.dumps(grid_spec["base_config"], ensure_ascii=False, indent=2), language="json")
            if grid_spec.get("sweep_parameters"):
                st.markdown("#### スイープパラメータ")
                sweep = grid_spec["sweep_parameters"]
                sweep_rows = []
                for param, values in sweep.items():
                    sweep_rows.append({
                        "パラメータ": param,
                        "候補値": str(values),
                        "候補数": len(values),
                    })
                st.dataframe(pd.DataFrame(sweep_rows), width='stretch', hide_index=True)
                # 組み合わせ数
                total_combos = 1
                for values in sweep.values():
                    total_combos *= len(values)
                st.metric("全組み合わせ数", total_combos)
        else:
            st.info("グリッド設計データがありません")
    tab_idx += 1

    # --- Phase 3 グリッド結果タブ ---
    if phase3_its:
        with tabs[tab_idx]:
            st.markdown("### Phase 3: グリッド結果")
            st.caption("パラメータの全組み合わせを自動試行した結果")

            # グリッド結果テーブル（スイープパラメータ値を独立列として表示）
            grid_rows = []
            sweep_param_names = []
            if grid_spec and grid_spec.get("sweep_parameters"):
                sweep_param_names = list(grid_spec["sweep_parameters"].keys())

            for i, it in enumerate(phase3_its):
                bt = it.backtest_result
                is_best = (best_idx is not None and iterations.index(it) == best_idx)

                row = {"#": f"{it.iteration}{'★' if is_best else ''}"}

                # スイープパラメータ値を個別列に
                combo = it.grid_combo or {}
                for param in sweep_param_names:
                    row[param] = combo.get(param, "-")

                if "error" in bt:
                    row["シグナル"] = 0
                    row["超過リターン"] = "エラー"
                    row["p値"] = "-"
                    row["効果量d"] = "-"
                    row["エッジ"] = "-"
                else:
                    stats = bt.get("statistics", {})
                    backtest_data = bt.get("backtest", bt)
                    n_valid = backtest_data.get("n_valid_signals", 0)
                    mean_excess = stats.get("excess_mean") or backtest_data.get("mean_excess_return") or 0
                    p_value = stats.get("p_value")
                    p_value = p_value if p_value is not None else 1.0
                    cohens_d = stats.get("cohens_d") or 0

                    row["シグナル"] = n_valid
                    row["超過リターン"] = f"{mean_excess:+.2%}"
                    row["p値"] = f"{p_value:.4f}"
                    row["効果量d"] = f"{cohens_d:+.3f}"
                    row["エッジ"] = "ロング" if mean_excess > 0 else ("ショート" if mean_excess < 0 else "-")

                grid_rows.append(row)

            if grid_rows:
                st.markdown("#### 全件テーブル")
                df_grid = pd.DataFrame(grid_rows)

                def _highlight_best_grid(row):
                    if "★" in str(row["#"]):
                        return ["background-color: rgba(255, 128, 0, 0.15)"] * len(row)
                    return [""] * len(row)

                styled_grid = df_grid.style.apply(_highlight_best_grid, axis=1)
                st.dataframe(styled_grid, width='stretch', hide_index=True)
                st.caption("★ = ベスト結果")

            # 上位5件のみexpander詳細
            scored_phase3 = []
            for i, it in enumerate(phase3_its):
                bt = it.backtest_result
                if "error" in bt:
                    scored_phase3.append((-999, i, it))
                    continue
                stats = bt.get("statistics", {})
                backtest_data = bt.get("backtest", bt)
                n_valid = backtest_data.get("n_valid_signals", 0)
                mean_excess = stats.get("excess_mean") or backtest_data.get("mean_excess_return") or 0
                p_value = stats.get("p_value")
                p_value = p_value if p_value is not None else 1.0
                cohens_d = abs(stats.get("cohens_d") or 0)
                score = cohens_d * 3 + (2 if p_value < 0.05 else 0) + (1 if n_valid >= 20 else 0) + abs(mean_excess) * 10
                scored_phase3.append((score, i, it))
            scored_phase3.sort(key=lambda x: x[0], reverse=True)

            st.markdown("#### 上位5件の詳細")
            for rank, (score, orig_idx, it) in enumerate(scored_phase3[:5]):
                is_best = (best_idx is not None and iterations.index(it) == best_idx)
                combo_desc = it.changes_description or f"Grid {orig_idx+1}"
                with st.expander(
                    f"{'★ ' if is_best else ''}#{rank+1} {combo_desc} — "
                    f"{_get_result_summary(it)}",
                    expanded=(rank == 0),
                ):
                    if is_best:
                        st.info("このパターンがベスト結果に選ばれました")
                    _render_iteration_result(it, key_suffix=f"grid_{orig_idx}")
        tab_idx += 1

    # --- AI総合評価タブ ---
    with tabs[tab_idx]:
        interp = result.interpretation
        if interp:
            label = interp.get("evaluation_label", "needs_review")
            confidence = interp.get("confidence", 0)
            st.markdown(
                f"### {render_status_badge(label)} &nbsp; 信頼度: {confidence:.0%}",
                unsafe_allow_html=True,
            )
            if interp.get("summary"):
                st.write(interp["summary"])
            if interp.get("reasons"):
                st.markdown("**判定理由:**")
                for r in interp["reasons"]:
                    st.write(f"- {r}")
            for section, title in [("strengths", "強み"), ("weaknesses", "弱み"), ("suggestions", "改善提案")]:
                items = interp.get(section, [])
                if items:
                    st.markdown(f"**{title}:**")
                    for item in items:
                        st.write(f"- {item}")
        else:
            st.info("AI評価データがありません")

    st.markdown("---")
    if st.button("新しい研究を開始", type="primary"):
        _clear_state()
        st.rerun()


def _get_result_summary(it) -> str:
    """イテレーション結果の1行サマリーを返す"""
    bt = it.backtest_result
    if "error" in bt:
        return "エラー"

    stats = bt.get("statistics", {})
    backtest_data = bt.get("backtest", bt)
    n_valid = backtest_data.get("n_valid_signals", 0)
    mean_excess = stats.get("excess_mean") or backtest_data.get("mean_excess_return") or 0
    p_value = stats.get("p_value")
    p_value = p_value if p_value is not None else 1.0

    return f"シグナル{n_valid}件, 超過リターン{mean_excess:+.2%}, p={p_value:.4f}"


def _render_iteration_result(it, key_suffix=""):
    """1つのイテレーション結果を表示"""
    bt = it.backtest_result

    # エラーの場合
    if "error" in bt:
        st.error(f"バックテストエラー: {bt['error']}")
        if it.ai_reasoning:
            st.write(f"**AI判断:** {it.ai_reasoning}")
        return

    # アプローチの説明
    if it.ai_reasoning:
        st.write(f"**AI判断:** {it.ai_reasoning}")

    # StandardBacktester結果を render_result_tabs で表示
    stats = bt.get("statistics", {})
    backtest = bt.get("backtest", bt)
    evaluation = bt.get("evaluation", {})
    recent_examples = bt.get("recent_examples")
    pending_signals = bt.get("pending_signals")
    config_snapshot = bt.get("config_snapshot", {})

    render_result_tabs(
        interpretation=evaluation,
        stats=stats,
        backtest=backtest,
        code_or_config=config_snapshot,
        recent_examples=recent_examples,
        code_tab_label="パラメータ設定",
        code_language="json",
        pending_signals=pending_signals,
        key_prefix=f"iter_{it.iteration}_{key_suffix}",
    )


def _render_conditions_expander(result: ResearchProgress):
    """研究条件の折りたたみ表示"""
    with st.expander("研究条件", expanded=False):
        if result.idea_title:
            st.markdown(f"**タイトル:** {result.idea_title}")
        if result.category:
            st.markdown(f"**カテゴリ:** {result.category}")
        if result.idea_text:
            st.markdown("**仮説:**")
            st.info(result.idea_text)
        if result.start_date or result.end_date:
            st.markdown(f"**分析期間:** {result.start_date} 〜 {result.end_date}")
        if result.universe_filter_text:
            st.markdown(f"**ユニバース条件:** {result.universe_filter_text}")


# ===========================================================================
# ユーティリティ
# ===========================================================================
def _clear_state():
    for key in list(st.session_state.keys()):
        if key.startswith("rp_") or key == "_rp_error":
            st.session_state.pop(key, None)


if __name__ == "__main__":
    main()
