"""Onset検出 Phase 1 — スター株共通点発見 + 追加スター株発見 + 初動特定

ユーザーがスター株を指定 → 共通特徴量を発見 → 追加スター株を発見 → 初動日を特定
"""

import json
import threading
import time as _time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import MARKET_DATA_DIR, JQUANTS_API_KEY
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider
from core.styles import apply_reuters_style
from core.onset_detector.discoverer import (
    WIDE_FEATURE_DESCRIPTIONS_JP,
    ONSET_SIGNAL_NAMES_JP,
)

# 初動シグナルの略称（表示用）
SIGNAL_JP_SHORT = {
    "volume_surge": "出来高急増",
    "quiet_accumulation": "静的買い集め",
    "consecutive_accumulation": "継続的蓄積",
    "obv_breakout": "出来高累計突破",
    "bb_squeeze": "価格帯収縮",
    "volatility_compression": "値動き収縮",
    "higher_lows": "安値切り上げ",
    "range_breakout": "高値ブレイク",
    "ma_crossover": "平均線クロス",
    "up_volume_dominance": "買い出来高優勢",
}

st.set_page_config(page_title="Onset検出", page_icon="R", layout="wide")
apply_reuters_style()

RESULTS_DIR = Path("storage/onset_results")


def _save_results(result: dict) -> Path:
    """結果をJSON保存"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"phase1_{ts}.json"

    serializable = {}
    for key in ("common_features", "additional_stars", "onset_dates",
                "ai_interpretation", "warnings"):
        val = result.get(key)
        if val is not None:
            serializable[key] = val

    # star_stocks / all_stars
    for key in ("star_stocks", "all_stars"):
        val = result.get(key)
        if val:
            serializable[key] = val

    serializable["saved_at"] = ts

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
    return path


def _load_results(path: Path) -> dict:
    """保存済み結果を読み込み"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_saved_results() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("phase1_*.json"), reverse=True)


@st.cache_resource
def get_data_provider():
    cache = DataCache(MARKET_DATA_DIR)
    return JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache)


# ---------------------------------------------------------------------------
# バックグラウンド実行
# ---------------------------------------------------------------------------
def _run_phase1_thread(progress_dict: dict, provider, config):
    """Phase 1パイプラインをバックグラウンド実行"""
    try:
        from core.onset_detector import run_phase1_discovery

        def on_progress(step, total, msg):
            progress_dict["step"] = step
            progress_dict["total"] = total
            progress_dict["message"] = msg
            progress_dict["pct"] = step / total if total > 0 else 0

        result = run_phase1_discovery(
            data_provider=provider,
            config=config,
            progress_callback=on_progress,
        )
        progress_dict["_result"] = result
        progress_dict["pct"] = 1.0
        progress_dict["message"] = "完了"
    except Exception as e:
        import traceback
        progress_dict["error"] = str(e)
        progress_dict["traceback"] = traceback.format_exc()
        progress_dict["pct"] = 1.0


# ---------------------------------------------------------------------------
# メインページ
# ---------------------------------------------------------------------------
def main():
    st.markdown(
        '<p style="font-size:18px;font-weight:700;margin-bottom:2px;">'
        'Onset検出 Phase 1 '
        '<span style="font-size:12px;color:#888;font-weight:400;">'
        '— 共通点発見 + 追加スター株 + 初動特定</span></p>',
        unsafe_allow_html=True,
    )

    # --- サイドバー ---
    with st.sidebar:
        st.markdown("### Onset検出 設定")

        st.markdown("**分析期間**")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input(
                "開始日", value=date.today() - timedelta(days=365),
                key="onset_start_date",
                help="分析対象の開始日",
            )
        with col_d2:
            end_date = st.date_input(
                "終了日", value=date.today(),
                key="onset_end_date",
                help="分析対象の終了日",
            )

        st.markdown("**対象フィルタ**")
        scan_min_market_cap = st.number_input(
            "最低時価総額（億円）",
            min_value=0, max_value=10000, value=0, step=50,
            help="yfinanceで実際の時価総額を取得しフィルタ。0で無効。"
            "初回取得は時間がかかりますが30日間キャッシュされます。"
            "ユーザー指定スター株は除外されません",
        )

        st.markdown("**発見パラメータ**")
        discovery_min_precision = st.slider(
            "コンボ最低精度", 0.05, 0.50, 0.15, 0.05,
            help="特徴量コンボの最低精度。低いほど多くのコンボが候補になります",
        )
        discovery_min_recall = st.slider(
            "コンボ最低再現率", 0.10, 0.80, 0.30, 0.05,
            help="特徴量コンボの最低再現率。高いほど多くのスター株をカバーするコンボのみ採用",
        )
        discovery_max_additional = st.slider(
            "追加スター株上限", 5, 100, 30,
            help="コンボ条件で発見する追加スター株の最大数",
        )

        st.markdown("**信用取引データ**")
        use_margin = st.checkbox(
            "信用取引データを使用", value=True,
            help="約52回の追加APIコール（キャッシュ有効）。信用買い残・貸借倍率等の特徴量を追加",
        )

        st.markdown("**AI解釈**")
        use_ai = st.checkbox(
            "Claude CLIで結果を解釈", value=True,
            help="Claude Code CLIを使って発見結果を自然言語で解釈・説明します",
        )

        # 過去結果読込
        st.markdown("---")
        saved_list = _list_saved_results()
        st.markdown(f"**過去の実行結果** ({len(saved_list)}件)")
        if saved_list:
            selected_file = st.selectbox(
                "読み込む結果を選択",
                ["（選択してください）"] + [sp.name for sp in saved_list[:20]],
                key="onset_history_select",
            )
            if st.button("結果を読み込む", key="onset_load_history"):
                if selected_file != "（選択してください）":
                    sp = RESULTS_DIR / selected_file
                    if sp.exists():
                        loaded = _load_results(sp)
                        st.session_state["onset_result"] = loaded
                        st.rerun()
        else:
            st.caption("実行後に自動保存されます")

    # --- スター株入力 ---
    st.markdown("### スター株の指定")

    # スター株分析結果の有無
    ss_result = st.session_state.get("ss_result")
    has_ss_result = (
        ss_result is not None
        and hasattr(ss_result, "star_stocks")
        and ss_result.star_stocks
    )

    source_options = []
    if has_ss_result:
        source_options.append(
            f"スター株分析の結果を使用（{len(ss_result.star_stocks)}件）"
        )
    source_options.append("手動でコード指定")

    star_source = st.radio(
        "スター株の指定方法",
        source_options,
        horizontal=True,
        help="「スター株分析」ページの結果がある場合はそのまま引き継げます",
    )

    user_star_codes = []

    if star_source.startswith("スター株分析の結果"):
        star_stocks_input = ss_result.star_stocks
        user_star_codes = [str(s["code"])[:4] for s in star_stocks_input]
        preview_df = pd.DataFrame([
            {
                "コード": s["code"],
                "銘柄名": s.get("name", ""),
                "超過リターン": f"{s.get('excess_return', 0):.1%}",
            }
            for s in star_stocks_input[:10]
        ])
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
        if len(star_stocks_input) > 10:
            st.caption(f"...他 {len(star_stocks_input) - 10}件")
    else:
        codes_text = st.text_input(
            "スター株コード（カンマ区切り）",
            placeholder="6920, 5803, 7203, 6526, 6857",
            help="過去に大きく上昇した銘柄のコードを入力（4桁でも5桁でもOK）",
        )
        if codes_text.strip():
            user_star_codes = [c.strip() for c in codes_text.split(",") if c.strip()]
            st.caption(f"{len(user_star_codes)}件指定")

    # --- Config構築 ---
    from core.onset_detector.config import OnsetDetectorConfig
    config = OnsetDetectorConfig(
        start_date=str(start_date),
        end_date=str(end_date),
        user_star_codes=user_star_codes,
        scan_min_market_cap=float(scan_min_market_cap),
        discovery_min_precision=discovery_min_precision,
        discovery_min_recall=discovery_min_recall,
        discovery_max_additional_stars=discovery_max_additional,
        use_ai_interpretation=use_ai,
        use_margin_features=use_margin,
    )

    # --- 実行ボタン ---
    st.markdown("---")
    col1, col2 = st.columns([2, 8])
    with col1:
        run_clicked = st.button(
            "Phase 1 実行", type="primary", use_container_width=True,
        )

    # --- パイプライン実行（session_stateでスレッド管理 → 画面復帰時も継続） ---
    if run_clicked:
        if not user_star_codes:
            st.warning("スター株コードを入力してください")
            return

        progress_dict = {
            "step": 0, "total": 8, "message": "開始...", "pct": 0.0,
            "_start_time": _time.time(),
        }
        provider = get_data_provider()

        thread = threading.Thread(
            target=_run_phase1_thread,
            args=(progress_dict, provider, config),
            daemon=True,
        )
        thread.start()

        # session_stateに保持 → 画面復帰時にも参照可能
        st.session_state["onset_thread"] = thread
        st.session_state["onset_progress"] = progress_dict

    # --- 実行中スレッドの監視（ページリロード/復帰時も動作） ---
    thread = st.session_state.get("onset_thread")
    progress_dict = st.session_state.get("onset_progress")

    if thread is not None and progress_dict is not None:
        if thread.is_alive():
            # まだ実行中 → 進捗表示してリフレッシュ
            pct = progress_dict.get("pct", 0.0)
            msg = progress_dict.get("message", "")
            step = progress_dict.get("step", 0)
            total = progress_dict.get("total", 7)
            start_time = progress_dict.get("_start_time", _time.time())
            elapsed = int(_time.time() - start_time)
            if elapsed >= 60:
                elapsed_str = f"{elapsed // 60}分{elapsed % 60:02d}秒"
            else:
                elapsed_str = f"{elapsed}秒"

            st.progress(min(pct, 0.99))
            st.markdown(
                f"**Step {step}/{total}**: {msg}　（経過: {elapsed_str}）"
            )
            _time.sleep(1.0)
            st.rerun()
        else:
            # スレッド完了 → 結果を取り出す
            thread.join()

            # session_stateからクリア
            del st.session_state["onset_thread"]
            del st.session_state["onset_progress"]

            if "error" in progress_dict:
                st.error(f"エラー: {progress_dict['error']}")
                if "traceback" in progress_dict:
                    with st.expander("詳細"):
                        st.code(progress_dict["traceback"])
            else:
                result = progress_dict.get("_result", {})
                if "error" in result:
                    st.warning(result["error"])
                else:
                    st.session_state["onset_result"] = result
                    saved_path = _save_results(result)
                    st.success(f"結果保存完了: {saved_path.name}")
                    st.rerun()

            return

    # --- 結果表示 ---
    result = st.session_state.get("onset_result")
    if result is None:
        st.info("「Phase 1 実行」をクリックして開始してください")
        return

    _display_results(result)


def _normalize_ai_headers(text: str) -> str:
    """AI解釈のMarkdownヘッダー（#, ##, ###）をインラインの太字に変換し、フォントを均一にする"""
    import re
    text = re.sub(r'^#### (.+)$', r'**▸ \1**', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'**▸ \1**', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'\n**■ \1**', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'\n**◆ \1**', text, flags=re.MULTILINE)
    # 表（| ... |）はそのまま残す
    return text


def _section(title: str):
    """統一された小見出し（h4相当、フォントサイズ抑制）"""
    st.markdown(
        f'<p style="font-size:13px;font-weight:700;margin:14px 0 4px 0;'
        f'border-bottom:1px solid #ddd;padding-bottom:3px;">{title}</p>',
        unsafe_allow_html=True,
    )


def _tag(label: str, color: str) -> str:
    """インラインタグHTML"""
    return (
        f'<span style="background:{color};color:#fff;padding:1px 6px;'
        f'border-radius:3px;font-size:11px;font-weight:600;">{label}</span>'
    )


TAG_INPUT = _tag("入力", "#1565C0")
TAG_FOUND = _tag("発見", "#E65100")


# ---------------------------------------------------------------------------
# 結果テキスト生成（コピペ用）
# ---------------------------------------------------------------------------
def _build_result_text(result: dict) -> str:
    """結果全体をプレーンテキストに変換"""
    lines = ["=" * 60, "Onset検出 Phase 1 結果サマリー", "=" * 60, ""]

    star_stocks = result.get("star_stocks", [])
    additional_stars = result.get("additional_stars", [])
    common_features = result.get("common_features", {})
    onset_dates = result.get("onset_dates", {})
    all_stars = result.get("all_stars", [])
    ai_interp = result.get("ai_interpretation", "")

    # サマリー
    n_onset = sum(1 for od in onset_dates.values() if od.get("onset_date"))
    best_combos = common_features.get("best_combos", [])
    base_rate = common_features.get("base_rate", 0)
    lines.append(f"入力スター株: {len(star_stocks)}件")
    lines.append(f"追加発見: {len(additional_stars)}件")
    lines.append(f"初動特定: {n_onset}/{len(all_stars)}件")
    lines.append(f"ベストコンボ: {len(best_combos)}件")
    lines.append("")

    # 確率・リターンサマリー
    base_rate_universe = common_features.get("base_rate_universe", base_rate)
    n_universe = common_features.get("n_universe", 0)
    if best_combos:
        best = best_combos[0]
        best_names = " かつ ".join(best.get("features_jp", best["features"]))
        lines.append("--- スター株になる確率 ---")
        # 母集団精度が計算済みか確認
        u_prec = best.get("universe_precision")
        if u_prec is not None and n_universe > 0:
            u_hits = best.get("universe_n_hits", 0)
            u_stars = best.get("universe_n_stars", 0)
            u_lift = u_prec / base_rate_universe if base_rate_universe > 0 else 0
            lines.append(f"  分析対象: {n_universe:,}銘柄（時価総額フィルタ済）")
            lines.append(f"  スター株ベースレート: {base_rate_universe:.2%}（何もしない場合）")
            lines.append(f"  条件「{best_names}」を対象期間中に満たした銘柄: {u_hits:,}件")
            lines.append(f"  そのうちスター株になった: {u_stars}件 → 確率 {u_prec:.1%}（{u_lift:.1f}倍）")
        else:
            lines.append(f"  市場ベースレート（条件なし）: {base_rate:.1%}")
            lines.append(f"  条件「{best_names}」を満たした場合: {best['precision']:.0%}（{best['lift']:.1f}倍）")
        lines.append(f"  カバー率: {best['recall']:.0%}のスター株をカバー")
        lines.append("")

    max_returns = [
        od.get("max_return") or od.get("max_return_60d") or od.get("fwd_return_60d")
        for od in onset_dates.values()
        if od.get("onset_date") and (
            od.get("max_return") is not None or
            od.get("max_return_60d") is not None or
            od.get("fwd_return_60d") is not None
        )
    ]
    max_returns = [r for r in max_returns if r is not None]
    if max_returns:
        mean_ret = sum(max_returns) / len(max_returns)
        sorted_rets = sorted(max_returns)
        median_ret = sorted_rets[len(sorted_rets) // 2]
        lines.append("--- 初動後 最大到達リターン統計 ---")
        lines.append(f"  平均 最大リターン: {mean_ret:.1%}")
        lines.append(f"  中央値 最大リターン: {median_ret:.1%}")
        lines.append(f"  最大: {max(max_returns):.1%}  最小: {min(max_returns):.1%}")
        lines.append(f"  対象銘柄数: {len(max_returns)}")
        lines.append("")

    # AI解釈
    if ai_interp:
        lines += ["--- AI解釈 ---", ai_interp, ""]

    # 共通特徴量
    signals = common_features.get("signals", [])
    useful = [s for s in signals if s.get("verdict") != "meaningless"]
    if useful:
        lines.append("--- 判別特徴量 (スター株 vs 非スター株) ---")
        for s in useful[:10]:
            name = s.get("feature_jp", s["feature"])
            lines.append(
                f"  {name}: スター={_fmt_val(s['pos_mean'])}  "
                f"非スター={_fmt_val(s['neg_mean'])}  "
                f"閾値>={_fmt_val(s['threshold'])}  "
                f"J={s['j_stat']:.3f}  Lift={s['lift']:.1f}x"
            )
        lines.append("")

    # コンボ（母集団精度優先、なければ訓練精度）
    if best_combos:
        lines.append("--- ベストコンボ（母集団ベース） ---")
        for i, c in enumerate(best_combos[:5]):
            names = " AND ".join(c.get("features_jp", c["features"]))
            u_prec = c.get("universe_precision")
            u_hits = c.get("universe_n_hits", 0)
            u_stars = c.get("universe_n_stars", 0)
            if u_prec is not None and u_hits > 0:
                u_lift = u_prec / base_rate_universe if base_rate_universe > 0 else 0
                lines.append(
                    f"  {i+1}. {names}  "
                    f"母集団確率={u_prec:.1%}（{u_lift:.1f}倍） "
                    f"条件合致={u_hits}件中{u_stars}件スター株  "
                    f"カバー率={c['recall']:.0%}"
                )
            else:
                tp = c.get("tp", 0)
                total_hits = c.get("total_hits", c.get("n_combo", "?"))
                lines.append(
                    f"  {i+1}. {names}  "
                    f"訓練精度={c['precision']:.0%}（参考）  "
                    f"カバー率={c['recall']:.0%}  合致={total_hits}件中{tp}件スター株"
                )
        lines.append("")

    # 全スター株 + 初動
    lines.append("--- 全スター株 + 初動日 ---")
    lines.append(f"{'コード':<8} {'銘柄名':<16} {'種別':<4} {'初動日':<12} "
                 f"{'Sig':>3} {'60d後':>7} 発火シグナル")
    lines.append("-" * 90)
    for star in all_stars:
        code = str(star["code"])
        od = onset_dates.get(code, {})
        src = "入力" if star.get("source") == "user" else "発見"
        onset = od.get("onset_date", "-")
        score = od.get("score", 0)
        max_r = od.get("max_return") or od.get("max_return_60d") or od.get("fwd_return_60d")
        fwd = f"{max_r:.1%}" if od.get("onset_date") and max_r is not None else "-"
        sigs = "、".join(SIGNAL_JP_SHORT.get(s, s) for s in od.get("signals", []))
        name = star.get("name", "")[:14]
        lines.append(f"{code:<8} {name:<16} {src:<4} {onset:<12} {score:>3} {fwd:>7} {sigs}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 確率・リターンサマリー
# ---------------------------------------------------------------------------
def _display_probability_summary(result: dict):
    """スター株確率と初動後リターンを直感的に表示"""
    common_features = result.get("common_features", {})
    onset_dates = result.get("onset_dates", {})

    base_rate = common_features.get("base_rate", 0)
    base_rate_universe = common_features.get("base_rate_universe", base_rate)
    n_universe = common_features.get("n_universe", 0)
    best_combos = common_features.get("best_combos", [])

    # 初動後リターン: onset_date がある銘柄のmax_returnを優先収集
    ret_pairs = []  # (code, return値)
    for code, od in onset_dates.items():
        if not od.get("onset_date"):
            continue
        # max_return（新） > max_return_60d（旧） > fwd_return_60d の優先順で取得
        ret = od.get("max_return") or od.get("max_return_60d") or od.get("fwd_return_60d")
        if ret is None:
            continue
        ret_pairs.append((code, float(ret)))

    if not best_combos and not ret_pairs:
        return

    st.markdown("---")

    # ================================================================
    # ブロック1: このシグナルが出たら何%がスター株になったか？
    # ================================================================
    st.markdown(
        '<p style="font-size:14px;font-weight:700;margin:4px 0 8px 0;">'
        '❓ このシグナルが出た銘柄は、何%がスター株になったか？</p>',
        unsafe_allow_html=True,
    )

    if best_combos:
        n_star_total = common_features.get("n_star", 0)
        has_universe = best_combos[0].get("universe_precision") is not None

        if has_universe and n_universe > 0:
            st.caption(
                f"📊 分析対象 {n_universe:,}銘柄のうち、対象期間中にいつかでも条件を満たした銘柄の何%がスター株になったか"
                f"（スター株ベースレート = {base_rate_universe:.2%}）"
            )

            # 0件合致（過学習）とそれ以外に分離
            valid_combos = [c for c in best_combos[:5] if c.get("universe_n_hits", 0) > 0]
            overfit_combos = [c for c in best_combos[:5] if c.get("universe_n_hits", 0) == 0]

            for i, c in enumerate(valid_combos):
                u_prec = c.get("universe_precision", 0)
                u_hits = c.get("universe_n_hits", 0)
                u_stars = c.get("universe_n_stars", 0)
                u_lift = u_prec / base_rate_universe if base_rate_universe > 0 else 0
                recall = c.get("recall", 0)

                # 信頼度
                if u_stars >= 5:
                    rel_icon, rel_label = "🟢", "信頼度: 高（N≥5）"
                elif u_stars >= 3:
                    rel_icon, rel_label = "🟡", "信頼度: 中（N=3〜4）"
                else:
                    rel_icon, rel_label = "🔴", "信頼度: 低（N≤2 — 参考値）"

                # 条件行を閾値付きで生成
                cond_lines = _combo_cond_lines(c, numbered=True)

                with st.expander(
                    f"{rel_icon} **#{i+1}** — スター株確率 **{u_prec:.1%}**（ベースレートの{u_lift:.1f}倍）　"
                    f"条件合致: {u_hits}件 / うちスター株: {u_stars}件　カバー率: {recall:.0%}",
                    expanded=(i == 0),
                ):
                    st.markdown("**以下の条件を全て同時に満たしている銘柄:**")
                    for line in cond_lines:
                        st.markdown(line)
                    st.caption(rel_label)

            if overfit_combos:
                with st.expander(f"⚠️ 条件が厳しすぎるコンボ（母集団で0件合致 — 実用外） {len(overfit_combos)}件", expanded=False):
                    st.warning(
                        "以下のコンボは訓練データでは高精度でしたが、全銘柄×複数時点のスキャンで"
                        "一度も条件を満たした銘柄が見つかりませんでした。"
                        "条件が厳しすぎて実際の相場では発動しないと考えられます。"
                    )
                    for i, c in enumerate(overfit_combos):
                        cond_lines = _combo_cond_lines(c, numbered=True)
                        with st.expander(
                            f"過学習コンボ #{i+1}  訓練精度={c['precision']:.0%}（参考のみ）　カバー率: {c['recall']:.0%}",
                            expanded=False,
                        ):
                            for line in cond_lines:
                                st.markdown(line)

            # 参考: 同一データ内精度（過学習注意）
            with st.expander("参考: 同一データ内精度（過学習注意）", expanded=False):
                st.warning(
                    "以下は**同じデータで条件を発見・評価した**精度（訓練データ精度）です。"
                    "スター株が少ない場合、100%になりやすく**信頼性が低い**です。"
                    "上の母集団精度の方が実態に近い値です。"
                )
                ref_rows = []
                for i, c in enumerate(best_combos[:5]):
                    features = c.get("features", [])
                    features_jp = c.get("features_jp", features)
                    names = " かつ ".join(features_jp)
                    tp = c.get("tp", 0)
                    total_hits = c.get("total_hits", c.get("n_combo", 0))
                    ref_rows.append({
                        "順位": f"#{i+1}",
                        "条件": names,
                        "訓練精度": f"{c['precision']:.0%}",
                        "Lift": f"{c['lift']:.1f}倍",
                        "合致": f"{total_hits}件中{tp}件スター株",
                    })
                st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)

        else:
            # 母集団精度がない場合（旧データ互換）
            small_sample_combos = [c for c in best_combos[:5] if c.get("tp", 0) < 5]
            if small_sample_combos:
                st.warning(
                    f"⚠️ **サンプルサイズ警告**: 同一データで条件を発見・評価しているため"
                    f"精度が過大評価されている可能性があります。"
                    f"母集団精度は再実行すると計算されます。"
                )

            st.caption(f"※ 再実行すると母集団精度（全銘柄×複数時点スキャン）が計算されます")
            for i, c in enumerate(best_combos[:5]):
                tp = c.get("tp", 0)
                total_hits = c.get("total_hits", c.get("n_combo", 0))
                if tp >= 5:
                    rel_icon, rel_label = "🟢", "信頼度: 高（N≥5）"
                elif tp >= 3:
                    rel_icon, rel_label = "🟡", "信頼度: 中（N=3〜4）"
                else:
                    rel_icon, rel_label = "🔴", "信頼度: 低（N≤2 — 参考値）"
                cond_lines = _combo_cond_lines(c, numbered=True)
                with st.expander(
                    f"{rel_icon} **#{i+1}** — 訓練精度 **{c['precision']:.0%}**（{c['lift']:.1f}倍）　"
                    f"カバー率: {c['recall']:.0%}",
                    expanded=(i == 0),
                ):
                    st.markdown("**以下の条件を全て同時に満たしている銘柄:**")
                    for line in cond_lines:
                        st.markdown(line)
                    st.caption(rel_label)

    st.markdown("")

    # ================================================================
    # ブロック2: 初動後に買ったら何%リターン？
    # ================================================================
    st.markdown(
        '<p style="font-size:14px;font-weight:700;margin:12px 0 8px 0;">'
        '💰 初動後の最大到達リターン（保有中のピーク時点）</p>',
        unsafe_allow_html=True,
    )

    if ret_pairs:
        returns = [r for _, r in ret_pairs]
        n = len(returns)
        mean_ret = sum(returns) / n
        sorted_r = sorted(returns)
        median_ret = sorted_r[n // 2]
        max_ret = max(returns)
        min_ret = min(returns)
        # 60日内に一度でも+0%以上になった銘柄の割合
        win_rate = sum(1 for r in returns if r > 0) / n

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("平均 最大リターン", f"{mean_ret:.1%}")
        c2.metric("中央値 最大リターン", f"{median_ret:.1%}")
        c3.metric("最大 / 最小", f"{max_ret:.0%} / {min_ret:.0%}")
        c4.metric("プラスに到達した割合", f"{win_rate:.0%}")

        st.caption(f"※ {n}銘柄の初動後60日間での株価ピーク到達リターン。実際の売却タイミングは問わない。")

        # 銘柄別リターン棒グラフ（銘柄名を含める）
        all_stars = result.get("all_stars", [])
        code_to_name = {str(s["code"]): s.get("name", str(s["code"])) for s in all_stars}

        paired_sorted = sorted(ret_pairs, key=lambda x: x[1], reverse=True)
        bar_x = [f"{code_to_name.get(c, c)}" for c, _ in paired_sorted]
        bar_y = [r * 100 for _, r in paired_sorted]
        bar_colors = ["#D32F2F" if v >= 0 else "#1565C0" for v in bar_y]
        bar_text = [f"{v:.0f}%" for v in bar_y]

        # yaxis range に余裕を持たせてテキスト切れを防ぐ
        y_max = max(bar_y) if bar_y else 0
        y_min = min(bar_y) if bar_y else 0
        y_top = y_max * 1.35 + 10
        y_bot = min(y_min * 1.2 - 5, -5)

        fig = go.Figure(go.Bar(
            x=bar_x,
            y=bar_y,
            marker_color=bar_colors,
            text=bar_text,
            textposition="outside",
            textfont=dict(size=9),
        ))
        fig.add_hline(
            y=mean_ret * 100,
            line_dash="dash", line_color="#FF6F00", line_width=2,
            annotation=dict(text=f"平均 {mean_ret:.1%}", font_size=10, font_color="#FF6F00"),
        )
        fig.add_hline(y=0, line_color="#999", line_width=1)
        fig.update_layout(
            height=270,
            margin=dict(l=10, r=10, t=20, b=60),
            yaxis=dict(
                title="最大リターン（%）",
                range=[y_bot, y_top],
            ),
            font=dict(size=10),
            xaxis=dict(tickangle=-35),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("初動後リターンのデータがありません（初動日が特定されていない可能性があります）")


# ---------------------------------------------------------------------------
# 結果表示
# ---------------------------------------------------------------------------
def _display_results(result: dict):
    """Phase 1結果を表示"""

    star_stocks = result.get("star_stocks", [])
    additional_stars = result.get("additional_stars", [])
    common_features = result.get("common_features", {})
    onset_dates = result.get("onset_dates", {})
    all_stars = result.get("all_stars", [])
    ai_interp = result.get("ai_interpretation", "")
    warnings = result.get("warnings", [])

    for w in warnings:
        st.warning(w)

    # --- サマリーメトリクス ---
    n_onset = sum(1 for od in onset_dates.values() if od.get("onset_date"))
    best_combos = common_features.get("best_combos", [])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("入力スター株", f"{len(star_stocks)}件")
    col2.metric("追加発見", f"{len(additional_stars)}件")
    col3.metric("初動特定", f"{n_onset}/{len(all_stars)}件")
    col4.metric("ベストコンボ", f"{len(best_combos)}件")

    # --- 確率・リターンサマリー（最重要セクション） ---
    _display_probability_summary(result)

    # --- AI解釈（折りたたみ可能） ---
    if ai_interp:
        with st.expander("AI解釈（クリックで展開）", expanded=True):
            normalized = _normalize_ai_headers(ai_interp)
            st.markdown(normalized)

    # --- タブ構成 ---
    tabs = st.tabs([
        "共通特徴量",
        f"スター株一覧 ({len(all_stars)})",
        "初動タイムライン",
        "個別詳細",
        "結果コピー",
    ])

    with tabs[0]:
        _display_common_features(common_features)

    with tabs[1]:
        _display_star_list(star_stocks, additional_stars, all_stars, onset_dates)

    with tabs[2]:
        _display_onset_timeline(all_stars, onset_dates)

    with tabs[3]:
        _display_individual_detail(all_stars, onset_dates, common_features)

    with tabs[4]:
        _display_copy_text(result)


def _fmt_val(v: float) -> str:
    """値の桁に応じて自動フォーマット"""
    av = abs(v)
    if av == 0:
        return "0"
    if av >= 100:
        return f"{v:,.0f}"
    if av >= 1:
        return f"{v:.2f}"
    if av >= 0.01:
        return f"{v:.4f}"
    return f"{v:.6f}"


# 特徴量キー → 閾値フォーマット区分
_FEAT_PCT = {
    # リターン・騰落率系（× 100 して %表示）
    "ret_5d", "ret_20d", "up_days_ratio_10d", "quiet_accum_rate_20d",
    "higher_lows_slope_10d", "range_position_20d",
    "sector_rel_ret_10d", "ma5_ma20_gap", "price_vs_ma20_pct",
    "ma_deviation_25d", "ma_deviation_75d",
    "spread_proxy_5d", "max_gap_up_5d", "gap_frequency_20d",
    "higher_highs_ratio_10d", "proximity_52w_high",
    "margin_buy_change_pct", "margin_ratio_change_pct",
    "bb_width_pctile_60d", "up_volume_ratio_10d",
}
_FEAT_RATIO = {
    # 倍率系（X倍以上）
    "vol_ratio_5d_20d", "vol_ratio_5d_60d", "vol_acceleration",
    "atr_ratio_5d_20d", "intraday_range_ratio_5d", "realized_vol_5d_vs_20d",
    "topix_beta_20d", "residual_vol_ratio", "vol_vs_market_vol",
    "margin_ratio", "margin_buy_vol_ratio", "turnover_change_10d_20d",
    "obv_slope_10d",
}
_FEAT_DAYS = {
    # 日数系（X日以上）
    "vol_surge_count_10d", "consecutive_up_days", "margin_buy_turnover_days",
}


def _fmt_threshold(feat_key: str, threshold: float) -> str:
    """閾値を人間が読める形式にフォーマット"""
    sign = "+" if threshold > 0 else ""
    if feat_key in _FEAT_PCT:
        pct = threshold * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%以上"
    elif feat_key in _FEAT_DAYS:
        return f"{int(round(threshold))}日以上"
    elif feat_key in _FEAT_RATIO:
        return f"{threshold:.2f}倍以上"
    else:
        # その他（生の値）
        if abs(threshold) < 10:
            return f"{sign}{threshold:.3f}以上"
        return f"{sign}{threshold:.1f}以上"


def _combo_cond_lines(c: dict, numbered: bool = False) -> list[str]:
    """コンボの各条件を「説明 が 閾値以上」の形式で列挙して返す"""
    features = c.get("features", [])
    features_jp = c.get("features_jp", features)
    thresholds = c.get("thresholds", [0.0] * len(features))
    lines = []
    for idx, (feat_key, feat_label, th) in enumerate(zip(features, features_jp, thresholds)):
        desc = WIDE_FEATURE_DESCRIPTIONS_JP.get(feat_key, feat_label)
        th_str = _fmt_threshold(feat_key, th)
        prefix = f"{idx+1}. " if numbered else "  ✅ "
        lines.append(f"{prefix}**{feat_label}**（{desc}）が **{th_str}**")
    return lines


def _display_common_features(common_features: dict):
    """共通特徴量の表示 — スター株 vs 非スター株の比較を重視"""

    if common_features.get("error"):
        st.warning(common_features["error"])
        return

    signals = common_features.get("signals", [])
    best_combos = common_features.get("best_combos", [])
    combo_signals = common_features.get("combo_signals", [])
    base_rate = common_features.get("base_rate", 0)
    n_star = common_features.get("n_star", 0)
    n_non_star = common_features.get("n_non_star", 0)

    useful_signals = [s for s in signals if s.get("verdict") != "meaningless"][:12]
    if not useful_signals:
        st.info("有意な特徴量が見つかりませんでした")
        return

    # --- Cohen's d計算（stdデータがある場合のみ） ---
    has_std = any(s.get("pos_std") for s in useful_signals)
    if has_std:
        for s in useful_signals:
            pos_std = s.get("pos_std") or 0.01
            neg_std = s.get("neg_std") or 0.01
            pooled = max(np.sqrt((pos_std**2 + neg_std**2) / 2), 0.0001)
            s["cohens_d"] = round((s["pos_mean"] - s["neg_mean"]) / pooled, 2)
        sorted_by_d = sorted(useful_signals, key=lambda s: abs(s.get("cohens_d", 0)), reverse=True)
    else:
        # stdデータなし（旧フォーマット）→ J-stat順
        sorted_by_d = sorted(useful_signals, key=lambda s: s["j_stat"], reverse=True)

    # ================================================================
    # Section 1: Small multiples — スター株 vs 非スター株 直接比較
    # ================================================================
    _section("スター株 vs 非スター株 — 特徴量の直接比較")
    st.caption(
        f"スター株 {n_star}件 vs ランダム非スター株 {n_non_star}件。"
        f"各グラフは「スター株はこの指標が非スター株の何倍か」を示す。"
        f"倍率が高いほどスター株に特有の状態。"
    )

    # Top 6 features in 3-column grid（倍率表示に変更）
    top_for_chart = sorted_by_d[:6]
    cols = st.columns(3)
    for i, s in enumerate(top_for_chart):
        with cols[i % 3]:
            pos_m = s["pos_mean"]
            neg_m = s["neg_mean"]
            # 倍率計算（分母0対策）
            ratio = pos_m / neg_m if abs(neg_m) > 1e-9 else float("inf")

            feat_desc = WIDE_FEATURE_DESCRIPTIONS_JP.get(s["feature"], s.get("feature_jp", s["feature"]))
            # 説明文をタイトルに使用（40文字まで）
            desc_short = feat_desc[:38] + "…" if len(feat_desc) > 38 else feat_desc

            # 倍率を棒グラフで表示
            bar_vals = [1.0, ratio if ratio != float("inf") else 0]
            bar_labels = ["市場平均（基準）", f"スター株 {ratio:.1f}倍" if ratio != float("inf") else "スター株"]
            bar_colors = ["#9E9E9E", "#D32F2F"]
            bar_texts = ["基準 1.0倍", f"+{ratio:.1f}倍" if ratio > 0 else f"{ratio:.1f}倍"]

            # yaxis range — textposition="outside" のクリップ対策
            y_top = max(bar_vals) * 1.5 if max(bar_vals) > 0 else 2.0

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=bar_labels,
                y=bar_vals,
                marker_color=bar_colors,
                text=bar_texts,
                textposition="outside",
                textfont=dict(size=11, color=["#555", "#C62828"]),
                width=0.55,
            ))
            # 基準ライン
            fig.add_hline(y=1.0, line_dash="dash", line_color="#999", line_width=1)
            ratio_label = f"{ratio:.1f}倍" if ratio != float("inf") else "∞"
            fig.update_layout(
                title=dict(
                    text=f"<b>{desc_short}</b><br><sup>スター株は市場平均の {ratio_label}</sup>",
                    font_size=11,
                ),
                height=280,
                margin=dict(l=20, r=10, t=75, b=15),
                showlegend=False,
                font=dict(size=10),
                yaxis=dict(
                    title="市場平均比（倍）",
                    range=[0, y_top],
                    zeroline=True, zerolinecolor="#E0E0E0",
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ================================================================
    # Section 2: ランキングチャート（Youden's J — 判別力）
    # ================================================================
    _section("特徴量ランキング — スター株を識別する力（上位10件）")
    st.caption(
        "Youden's J = （スター株でこの条件を満たした割合）-（非スター株でこの条件を満たした割合）。"
        "J=1.0 なら完璧に区別でき、J=0.0 なら意味なし。"
        "値が大きいほど「スター株に特有の状態」を表す特徴量です。"
    )

    # J-stat順にソート（上位10件）
    top10_j = sorted(useful_signals, key=lambda s: s["j_stat"], reverse=True)[:10]
    top10_j_rev = list(reversed(top10_j))

    chart_names = [s.get("feature_jp", s["feature"]) for s in top10_j_rev]
    chart_vals = [s["j_stat"] for s in top10_j_rev]
    chart_labels = [
        f"J={v:.2f} ({WIDE_FEATURE_DESCRIPTIONS_JP.get(s['feature'], '')[:12]}…)"
        if len(WIDE_FEATURE_DESCRIPTIONS_JP.get(s['feature'], '')) > 12
        else f"J={v:.2f}"
        for s, v in zip(top10_j_rev, chart_vals)
    ]
    chart_colors = [
        "#C62828" if s["verdict"] == "strong" else
        "#EF6C00" if s["verdict"] == "weak_useful" else "#9E9E9E"
        for s in top10_j_rev
    ]

    fig_rank = go.Figure(go.Bar(
        x=chart_vals, y=chart_names, orientation="h",
        marker_color=chart_colors,
        text=[f"J={v:.2f}" for v in chart_vals],
        textposition="auto",
        textfont=dict(size=10),
        customdata=[
            [WIDE_FEATURE_DESCRIPTIONS_JP.get(s["feature"], ""), s["pos_mean"], s["neg_mean"]]
            for s in top10_j_rev
        ],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "説明: %{customdata[0]}<br>"
            "J値: %{x:.3f}<br>"
            "スター株平均: %{customdata[1]:.4f}<br>"
            "市場平均: %{customdata[2]:.4f}<extra></extra>"
        ),
    ))
    fig_rank.add_vline(x=0.5, line_dash="dot", line_color="#999",
                       annotation_text="中程度", annotation_position="top right",
                       annotation_font_size=9)
    fig_rank.update_layout(
        height=max(220, len(top10_j) * 30 + 50),
        margin=dict(l=220, r=60, t=20, b=20),
        xaxis=dict(title="判別力 Youden's J（0.0〜1.0）", range=[0, 1.05]),
        font=dict(size=11),
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    # 全特徴量ランキング（折りたたみ）
    with st.expander(f"全特徴量ランキング（{len(useful_signals)}件）", expanded=False):
        rank_rows = []
        for rank_i, s in enumerate(sorted(useful_signals, key=lambda s: s["j_stat"], reverse=True), 1):
            pos_m = s["pos_mean"]
            neg_m = s["neg_mean"]
            ratio_str = f"{pos_m/neg_m:.1f}倍" if abs(neg_m) > 1e-9 and neg_m > 0 else "−"
            rank_rows.append({
                "順位": rank_i,
                "特徴量": s.get("feature_jp", s["feature"]),
                "説明": WIDE_FEATURE_DESCRIPTIONS_JP.get(s["feature"], ""),
                "J値": f"{s['j_stat']:.3f}",
                "スター株平均": _fmt_val(pos_m),
                "市場平均": _fmt_val(neg_m),
                "倍率": ratio_str,
                "判定": s.get("verdict", ""),
            })
        st.dataframe(pd.DataFrame(rank_rows), use_container_width=True, hide_index=True)

    # ================================================================
    # Section 3: 詳細比較テーブル
    # ================================================================
    with st.expander("詳細比較テーブル（全特徴量）", expanded=False):
        rows = []
        for s in sorted_by_d:
            pos_m = s["pos_mean"]
            neg_m = s["neg_mean"]
            row = {
                "特徴量": s.get("feature_jp", s["feature"]),
                "スター株": _fmt_val(pos_m),
                "非スター株": _fmt_val(neg_m),
                "差": f"{pos_m - neg_m:+.4f}",
            }
            if has_std:
                row["効果量(d)"] = f"{s.get('cohens_d', 0):.1f}"
            row.update({
                "閾値": f">={_fmt_val(s['threshold'])}",
                "J値": f"{s['j_stat']:.3f}",
                "Lift": f"{s['lift']:.1f}x",
                "精度": f"{s['precision']:.0%}",
                "再現率": f"{s['recall']:.0%}",
            })
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ================================================================
    # Section 4: ベスト特徴量コンボ
    # ================================================================
    _section("ベスト特徴量コンボ — 複数条件を同時に満たす銘柄の絞り込み")
    if best_combos:
        base_rate_u = common_features.get("base_rate_universe", base_rate)
        n_univ = common_features.get("n_universe", 0)

        # 母集団で条件合致0件（過学習）のコンボを分離
        valid_bc = [c for c in best_combos[:5] if c.get("universe_n_hits", 0) > 0 or c.get("universe_precision") is None]
        overfit_bc = [c for c in best_combos[:5] if c.get("universe_precision") is not None and c.get("universe_n_hits", 0) == 0]

        for i, c in enumerate(valid_bc):
            u_prec = c.get("universe_precision")
            u_hits = c.get("universe_n_hits", 0)
            u_stars = c.get("universe_n_stars", 0)
            cond_lines = _combo_cond_lines(c, numbered=True)

            if u_prec is not None and n_univ > 0:
                u_lift = u_prec / base_rate_u if base_rate_u > 0 else 0
                header = (
                    f"**#{i+1}**  スター株確率 **{u_prec:.1%}**（{u_lift:.1f}倍）　"
                    f"合致: {u_hits}件 / スター株: {u_stars}件　カバー率: {c['recall']:.0%}"
                )
            else:
                header = (
                    f"**#{i+1}**  精度 **{c['precision']:.0%}**（{c['lift']:.1f}倍）　"
                    f"カバー率: {c['recall']:.0%}"
                )

            with st.expander(header, expanded=(i == 0)):
                st.markdown("**以下の条件を全て同時に満たしている銘柄:**")
                for line in cond_lines:
                    st.markdown(line)

        if overfit_bc:
            with st.expander(f"⚠️ 過学習コンボ（母集団で0件合致） {len(overfit_bc)}件", expanded=False):
                st.warning("訓練データ内では高精度でしたが、全銘柄×複数時点のスキャンで一度も合致しませんでした。条件が厳しすぎるため実用外です。")
                for i, c in enumerate(overfit_bc):
                    cond_lines = _combo_cond_lines(c, numbered=True)
                    with st.expander(f"過学習 #{i+1}  訓練精度={c['precision']:.0%}（参考のみ）", expanded=False):
                        for line in cond_lines:
                            st.markdown(line)
    elif combo_signals:
        st.info(
            "精度・再現率の閾値を満たすコンボがありません。"
            "サイドバーの「コンボ最低精度」「コンボ最低再現率」を下げてみてください"
        )
        with st.expander("参考: 上位コンボ（閾値未達）"):
            rows = []
            for c in combo_signals[:10]:
                names = " AND ".join(c.get("features_jp", c["features"]))
                rows.append({
                    "コンボ": names,
                    "精度": f"{c['precision']:.0%}",
                    "再現率": f"{c['recall']:.0%}",
                    "Lift": f"{c['lift']:.1f}x",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("特徴量コンボが見つかりませんでした")


def _display_star_list(
    star_stocks: list, additional_stars: list,
    all_stars: list, onset_dates: dict,
):
    """全スター株を入力/発見の区別付きで一覧表示"""

    st.markdown(
        f'{TAG_INPUT} ユーザーが指定したスター株　'
        f'{TAG_FOUND} 共通特徴量コンボ条件で自動発見された銘柄',
        unsafe_allow_html=True,
    )
    st.caption(
        "60日後リターン(参考): 初動から60日後時点での株価変化率　｜　"
        "最大リターン(期間末まで): 初動後〜分析期間終了までのピーク到達リターン　｜　"
        "最大DD: 初動後〜期間末でのピークからの最大下落率（ドローダウン）"
    )

    rows = []
    for star in all_stars:
        code = str(star["code"])
        od = onset_dates.get(code, {})
        is_input = star.get("source") == "user"
        has_onset = bool(od.get("onset_date"))
        rows.append({
            "種別": "入力" if is_input else "発見",
            "コード": code,
            "銘柄名": star.get("name", ""),
            "セクター": star.get("sector", star.get("sector_17_name", "")),
            "超過リターン": f"{star.get('excess_return', 0):.1%}"
                if star.get("excess_return") is not None else "-",
            "初動日": od.get("onset_date", "-"),
            "シグナル数": od.get("score", 0) if has_onset else "-",
            "60日後リターン(参考)": f"{od['fwd_return_60d']:.1%}" if has_onset and od.get("fwd_return_60d") is not None else "-",
            "最大リターン(期間末まで)": (
                f"{(od.get('max_return') or od.get('max_return_60d')):.1%}"
                if has_onset and (od.get("max_return") or od.get("max_return_60d")) is not None else "-"
            ),
            "最大DD": (
                f"{(od.get('max_drawdown') or od.get('max_drawdown_60d')):.1%}"
                if has_onset and (od.get("max_drawdown") or od.get("max_drawdown_60d")) is not None else "-"
            ),
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.apply(
                lambda row: [
                    "background-color: #E3F2FD" if row["種別"] == "入力"
                    else "background-color: #FFF3E0"
                ] * len(row),
                axis=1,
            ),
            use_container_width=True, hide_index=True,
        )


def _fmt_signal_with_qty(sig_key: str, sig_qty: dict) -> str:
    """シグナル名に数値情報を付加（例: 出来高急増（平均の3.2倍））"""
    base = SIGNAL_JP_SHORT.get(sig_key, sig_key)
    if sig_key not in sig_qty:
        return base
    qty = sig_qty[sig_key]
    if sig_key == "volume_surge":
        return f"{base}（平均の{qty}倍）"
    elif sig_key == "higher_lows":
        sign = "+" if qty >= 0 else ""
        return f"{base}（10日で{sign}{qty}%上昇）"
    elif sig_key == "range_breakout":
        sign = "+" if qty >= 0 else ""
        return f"{base}（20日高値を{sign}{qty}%突破）"
    elif sig_key == "up_volume_dominance":
        return f"{base}（買い日の出来高{qty:.0f}%）"
    elif sig_key == "obv_breakout":
        sign = "+" if qty >= 0 else ""
        return f"{base}（10日で{sign}{qty}%上昇）"
    elif sig_key == "quiet_accumulation":
        return f"{base}（5日平均出来高は20日平均の{qty}倍）"
    elif sig_key == "ma_crossover":
        sign = "+" if qty >= 0 else ""
        return f"{base}（短期線が中期線を{sign}{qty}%上回る）"
    return base


def _display_onset_timeline(all_stars: list, onset_dates: dict):
    """初動タイムラインの表示"""

    rows = []
    for star in all_stars:
        code = str(star["code"])
        od = onset_dates.get(code, {})
        is_input = star.get("source") == "user"
        sig_qty = od.get("signal_quantities", {})
        sigs_jp = "、".join(
            _fmt_signal_with_qty(s, sig_qty) for s in od.get("signals", [])
        )
        # max_return（新） > max_return_60d（旧） > fwd_return_60d の優先順
        max_ret = od.get("max_return") or od.get("max_return_60d") or od.get("fwd_return_60d")
        rows.append({
            "種別": "入力" if is_input else "発見",
            "コード": code,
            "銘柄名": star.get("name", ""),
            "初動日": od.get("onset_date", "-"),
            "シグナル数": od.get("score", 0),
            "発火シグナル": sigs_jp,
            "最大リターン(期間末)": f"{max_ret:.1%}" if od.get("onset_date") and max_ret is not None else "-",
        })

    if rows:
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values("初動日", ascending=True)
        st.dataframe(
            df_sorted.style.apply(
                lambda row: [
                    "background-color: #E3F2FD" if row["種別"] == "入力"
                    else "background-color: #FFF3E0"
                ] * len(row),
                axis=1,
            ),
            use_container_width=True, hide_index=True,
        )

    # シグナル頻度
    signal_counts = {}
    for od in onset_dates.values():
        for sig in od.get("signals", []):
            signal_counts[sig] = signal_counts.get(sig, 0) + 1

    if signal_counts:
        _section("初動シグナル頻度")
        sorted_signals = sorted(signal_counts.items(), key=lambda x: -x[1])
        n_total = len(onset_dates)
        # シグナル名を日本語に変換（短縮名＋英語名）
        sig_labels = [
            f"{SIGNAL_JP_SHORT.get(s[0], s[0])}"
            for s in sorted_signals
        ]
        fig = go.Figure(go.Bar(
            x=[s[1] for s in sorted_signals],
            y=sig_labels,
            orientation="h",
            marker_color="#1E88E5",
            text=[f"{s[1]}/{n_total}" for s in sorted_signals],
            textposition="auto",
        ))
        fig.update_layout(
            height=max(200, len(sorted_signals) * 28 + 40),
            margin=dict(l=200, r=40, t=10, b=20),
            xaxis_title="発火回数",
            font=dict(size=11),
        )
        st.plotly_chart(fig, use_container_width=True)


def _display_individual_detail(all_stars: list, onset_dates: dict, common_features: dict):
    """個別銘柄の詳細表示"""

    if not all_stars:
        st.info("表示する銘柄がありません")
        return

    options = [
        f"{s['code']} {s.get('name', '')} ({'入力' if s.get('source') == 'user' else '発見'})"
        for s in all_stars
    ]
    selected = st.selectbox("銘柄を選択", options)
    if not selected:
        return

    code = selected.split(" ")[0]
    star = next((s for s in all_stars if str(s["code"]) == code), None)
    if star is None:
        return

    od = onset_dates.get(code, {})
    is_input = star.get("source") == "user"
    tag_html = TAG_INPUT if is_input else TAG_FOUND

    st.markdown(
        f'{tag_html} **{code} {star.get("name", "")}**'
        f' — {star.get("sector", star.get("sector_17_name", ""))}',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    if star.get("total_return") is not None:
        col1.metric("期間リターン", f"{star['total_return']:.1%}")
    if star.get("excess_return") is not None:
        col2.metric("超過リターン", f"{star['excess_return']:.1%}")
    if od.get("onset_date"):
        col3.metric("初動日", od["onset_date"])

    if od.get("onset_date"):
        _section("初動シグナル")
        sig_qty = od.get("signal_quantities", {})
        sigs_detail = []
        for s in od.get("signals", []):
            with_qty = _fmt_signal_with_qty(s, sig_qty)
            full_desc = ONSET_SIGNAL_NAMES_JP.get(s, s)
            sigs_detail.append(f"{with_qty}（{full_desc}）")
        max_ret = od.get("max_return") or od.get("max_return_60d") or od.get("fwd_return_60d")
        sig_text = f"シグナルスコア: **{od['score']}/10** ｜ "
        if max_ret is not None:
            sig_text += f"最大到達リターン: **{max_ret:.1%}** ｜ "
        sig_text += "発火: " + "、".join(sigs_detail)
        st.markdown(sig_text)
    else:
        st.info("初動日を特定できませんでした")

    if star.get("matched_combo"):
        _section("発見条件（合致コンボ）")
        combo = star["matched_combo"]
        parts = []
        for fn, val, th in zip(
            combo.get("features", []),
            combo.get("values", []),
            combo.get("thresholds", []),
        ):
            parts.append(f"`{fn}` = {val:.4f} (>={th:.4f})")
        st.markdown(" AND ".join(parts))


def _display_copy_text(result: dict):
    """結果を一括コピー可能なテキストで表示"""
    text = _build_result_text(result)
    st.text_area(
        "以下のテキストを全選択してコピーできます（Ctrl+A → Ctrl+C）",
        value=text,
        height=500,
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__" or True:
    main()
