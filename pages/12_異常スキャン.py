"""異常スキャンページ — ルール作成・スキャン・履歴の3タブ統合"""

import json
import threading
import time
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from config import ANOMALY_DEFAULTS, DB_PATH, JQUANTS_API_KEY, MARKET_DATA_DIR
from core.anomaly_engine import (
    UI_FEATURE_CATEGORIES,
    UI_FEATURE_KEYS,
    UI_FEATURE_LABELS_JP,
    OPERATOR_LABELS_JP,
    scan_universe,
)
from core.sidebar import render_sidebar_running_indicator
from core.styles import apply_reuters_style, render_status_badge
from core.universe_filter import MARKET_SEGMENTS, SECTOR_17_LIST, UniverseFilterConfig, apply_universe_filter
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider
from db.database import Database

st.set_page_config(page_title="異常スキャン", page_icon="R", layout="wide")
apply_reuters_style()
render_sidebar_running_indicator()

_cache = DataCache(MARKET_DATA_DIR)
_provider = JQuantsProvider(api_key=JQUANTS_API_KEY, cache=_cache)
_db = Database(DB_PATH)

# =========================================================================
# ヘルパー
# =========================================================================

def _nearest_trading_day(
    target_str: str,
    upper_bound: str,
    max_tries: int = 8,
) -> tuple[str | None, pd.DataFrame]:
    target_ts = pd.Timestamp(target_str)
    for i in range(max_tries):
        candidate = (target_ts - timedelta(days=i)).strftime("%Y-%m-%d")
        if candidate > upper_bound:
            continue
        df = _provider.get_price_daily_by_date(candidate)
        if not df.empty:
            return candidate, df
    return None, pd.DataFrame()


def _build_rule_config_from_ui(conditions_list: list[dict], logic: str) -> dict:
    """UI入力からルールConfigを構築"""
    return {
        "conditions": conditions_list,
        "logic": logic,
        "forward_eval_days": ANOMALY_DEFAULTS["forward_eval_days"],
    }


def _run_scan_thread(
    shared: dict,
    rule_config: dict,
    universe_df: pd.DataFrame,
    all_prices: pd.DataFrame,
    scan_date_str: str,
):
    """バックグラウンドスレッド — 計算のみ（API呼び出しなし）。

    shared は普通の dict で、メインスレッドから参照される。
    st.session_state には一切触らない。
    """
    try:
        t_start = time.time()
        total_stocks = len(universe_df)

        shared["message"] = f"スキャン中... 0/{total_stocks:,}銘柄"
        shared["processed"] = 0
        shared["total"] = total_stocks
        shared["started_at"] = t_start
        shared["phase"] = "scanning"

        sector_returns_10d = {}

        def _progress(processed, total):
            elapsed = time.time() - t_start
            if processed > 0:
                speed = processed / elapsed
                remaining = (total - processed) / speed if speed > 0 else 0
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                eta_str = f"残り約{eta_min}分{eta_sec}秒" if eta_min > 0 else f"残り約{eta_sec}秒"
            else:
                eta_str = "推定中..."

            pct = int(processed / total * 100) if total > 0 else 0
            shared["message"] = f"スキャン中... {processed:,}/{total:,}銘柄 ({pct}%) — {eta_str}"
            shared["processed"] = processed
            shared["total"] = total
            shared["pct"] = pct
            shared["eta_str"] = eta_str

        result_df = scan_universe(
            rule_config=rule_config,
            universe_df=universe_df,
            all_prices=all_prices,
            scan_date=scan_date_str,
            sector_returns_10d=sector_returns_10d,
            progress_callback=_progress,
        )

        elapsed_total = time.time() - t_start
        shared["phase"] = "done"
        shared["message"] = f"完了 ({elapsed_total:.0f}秒)"
        shared["pct"] = 100
        shared["_result"] = result_df
        shared["_scan_date"] = scan_date_str
        shared["_rule_config"] = rule_config

    except Exception as e:
        shared["phase"] = "error"
        shared["error"] = str(e)
        shared["message"] = f"エラー: {e}"


def _check_scan_completion():
    """スレッド完了時に結果を session_state に移す"""
    thread = st.session_state.get("an_thread")
    if thread is None or thread.is_alive():
        return
    shared = st.session_state.get("an_progress", {})
    st.session_state.pop("an_thread", None)
    st.session_state["an_running"] = False
    if "error" in shared:
        st.session_state["an_scan_error"] = shared["error"]
        st.session_state["an_scan_result"] = pd.DataFrame()
    elif "_result" in shared:
        st.session_state["an_scan_result"] = shared["_result"]
        st.session_state["an_scan_error"] = None
        st.session_state["an_result_date"] = shared.get("_scan_date", "")
        st.session_state["an_scan_rule_config"] = shared.get("_rule_config", {})


@st.fragment(run_every=2)
def _scan_waiting_fragment():
    """2秒ごとに自動更新して進捗を表示する"""
    thread = st.session_state.get("an_thread")
    if thread is not None and not thread.is_alive():
        _check_scan_completion()
        st.rerun(scope="app")
        return
    shared = st.session_state.get("an_progress", {})
    phase = shared.get("phase", "")
    msg = shared.get("message", "処理中...")
    pct = shared.get("pct", 0)
    processed = shared.get("processed", 0)
    total = shared.get("total", 0)

    if phase == "scanning" and total > 0:
        st.progress(min(pct / 100.0, 1.0), text=msg)
        if processed > 0:
            elapsed = time.time() - shared.get("started_at", time.time())
            speed = processed / elapsed if elapsed > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.caption(f"処理速度: {speed:.0f} 銘柄/秒")
            c2.caption(f"経過時間: {elapsed:.0f}秒")
            c3.caption(f"推定残り: {shared.get('eta_str', '—')}")
    else:
        st.info(msg)


def _color_ret(val) -> str:
    if pd.isna(val):
        return "color: #999"
    if isinstance(val, str):
        return "color: #555"
    if val > 0:
        return "color: #2E7D32; font-weight: 600"
    if val < 0:
        return "color: #C62828; font-weight: 600"
    return "color: #555"


# =========================================================================
# メインエリア
# =========================================================================

st.title("異常スキャン")

if not JQUANTS_API_KEY:
    st.error("J-Quants APIキーが未設定（`.env` に `JQUANTS_API_KEY` を設定）")
    st.stop()

tab_create, tab_scan, tab_history = st.tabs(["ルール作成・編集", "スキャン実行", "スキャン履歴"])


# =========================================================================
# TAB 1: ルール作成・編集
# =========================================================================
with tab_create:
    st.subheader("ルール定義")

    # テンプレート選択
    template_names = ["（新規作成）"] + [t["name"] for t in ANOMALY_DEFAULTS["template_rules"]]
    selected_template = st.selectbox("テンプレート", template_names, key="an_template")

    # テンプレートロード
    if selected_template != "（新規作成）":
        tmpl = next(t for t in ANOMALY_DEFAULTS["template_rules"] if t["name"] == selected_template)
        if st.button("テンプレートをロード", key="an_load_tmpl"):
            st.session_state["an_rule_name"] = tmpl["name"]
            st.session_state["an_rule_desc"] = tmpl["description"]
            st.session_state["an_rule_logic"] = tmpl["logic"]
            st.session_state["an_rule_conditions"] = tmpl["conditions"]
            st.rerun()

    st.divider()

    # ルール名・説明
    rule_name = st.text_input(
        "ルール名",
        value=st.session_state.get("an_rule_name", ""),
        key="an_rule_name_input",
    )
    rule_desc = st.text_area(
        "説明",
        value=st.session_state.get("an_rule_desc", ""),
        key="an_rule_desc_input",
        height=80,
    )

    # 結合ロジック
    logic = st.radio(
        "結合ロジック",
        ["AND", "OR"],
        index=0 if st.session_state.get("an_rule_logic", "AND") == "AND" else 1,
        horizontal=True,
        key="an_logic_radio",
    )

    st.divider()

    # 条件編集
    st.subheader("条件")

    # セッションから条件リスト取得
    if "an_rule_conditions" not in st.session_state:
        st.session_state["an_rule_conditions"] = []

    conditions = st.session_state["an_rule_conditions"]

    # 特徴量選択
    feature_options = {k: f"{UI_FEATURE_LABELS_JP[k]} [{UI_FEATURE_CATEGORIES[k]}]" for k in UI_FEATURE_KEYS}

    new_feature = st.selectbox(
        "特徴量を追加",
        ["（選択してください）"] + list(feature_options.keys()),
        format_func=lambda k: feature_options.get(k, k),
        key="an_new_feature",
    )

    if new_feature != "（選択してください）":
        nc1, nc2, nc3 = st.columns([2, 1, 1])
        with nc1:
            new_op = st.selectbox(
                "演算子",
                list(OPERATOR_LABELS_JP.keys()),
                format_func=lambda k: OPERATOR_LABELS_JP[k],
                key="an_new_op",
            )
        with nc2:
            new_val = st.number_input("閾値", value=0.0, format="%.4f", key="an_new_val")
        with nc3:
            new_val_upper = None
            if new_op == "between":
                new_val_upper = st.number_input("上限値", value=0.0, format="%.4f", key="an_new_val_upper")

        if st.button("条件を追加", key="an_add_cond"):
            cond = {
                "feature_key": new_feature,
                "operator": new_op,
                "value": new_val,
            }
            if new_val_upper is not None:
                cond["value_upper"] = new_val_upper
            conditions.append(cond)
            st.session_state["an_rule_conditions"] = conditions
            st.rerun()

    # 現在の条件一覧
    if conditions:
        st.write("**現在の条件:**")
        for i, cond in enumerate(conditions):
            fkey = cond.get("feature_key", "")
            label = UI_FEATURE_LABELS_JP.get(fkey, fkey)
            op_label = OPERATOR_LABELS_JP.get(cond.get("operator", ""), cond.get("operator", ""))
            val = cond.get("value", "")
            val_upper = cond.get("value_upper")

            display_text = f"{label} {op_label} {val}"
            if val_upper is not None:
                display_text += f" 〜 {val_upper}"

            c1, c2 = st.columns([5, 1])
            with c1:
                st.write(f"{i+1}. {display_text}")
            with c2:
                if st.button("削除", key=f"an_del_cond_{i}"):
                    conditions.pop(i)
                    st.session_state["an_rule_conditions"] = conditions
                    st.rerun()

        st.write(f"結合: **{logic}**")
    else:
        st.info("条件がまだ追加されていません。上から特徴量を選択して条件を追加してください。")

    st.divider()

    # 保存ボタン
    col_save, col_clear = st.columns(2)
    with col_save:
        if st.button("ルールを保存", type="primary", key="an_save_rule", disabled=not (rule_name and conditions)):
            rule_config = _build_rule_config_from_ui(conditions, logic)
            rule_id = _db.create_anomaly_rule(
                name=rule_name,
                description=rule_desc,
                features_config=rule_config,
                status="active",
            )
            st.success(f"ルール「{rule_name}」を保存しました (ID: {rule_id})")

    with col_clear:
        if st.button("クリア", key="an_clear_rule"):
            st.session_state["an_rule_name"] = ""
            st.session_state["an_rule_desc"] = ""
            st.session_state["an_rule_logic"] = "AND"
            st.session_state["an_rule_conditions"] = []
            st.rerun()

    # 保存済みルール一覧
    st.divider()
    st.subheader("保存済みルール")

    saved_rules = _db.list_anomaly_rules()
    if saved_rules:
        for rule in saved_rules:
            badge = render_status_badge(rule["status"])
            fc = rule.get("features_config", {})
            n_conds = len(fc.get("conditions", []))

            with st.expander(f'{rule["name"]} ({n_conds}条件)', expanded=False):
                st.markdown(badge, unsafe_allow_html=True)
                st.write(f"**説明:** {rule.get('description', '—')}")
                st.write(f"**ロジック:** {fc.get('logic', 'AND')}")

                for cond in fc.get("conditions", []):
                    fk = cond.get("feature_key", "")
                    st.write(f"- {UI_FEATURE_LABELS_JP.get(fk, fk)} {cond.get('operator', '')} {cond.get('value', '')}")

                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("編集にロード", key=f"an_edit_{rule['id']}"):
                        st.session_state["an_rule_name"] = rule["name"]
                        st.session_state["an_rule_desc"] = rule.get("description", "")
                        st.session_state["an_rule_logic"] = fc.get("logic", "AND")
                        st.session_state["an_rule_conditions"] = fc.get("conditions", [])
                        st.rerun()
                with c2:
                    new_status = "archived" if rule["status"] == "active" else "active"
                    label = "アーカイブ" if rule["status"] == "active" else "有効化"
                    if st.button(label, key=f"an_toggle_{rule['id']}"):
                        _db.update_anomaly_rule(rule["id"], status=new_status)
                        st.rerun()
                with c3:
                    if st.button("削除", key=f"an_delete_{rule['id']}"):
                        _db.delete_anomaly_rule(rule["id"])
                        st.rerun()
    else:
        st.info("保存済みルールはありません。テンプレートから作成してみてください。")


# =========================================================================
# TAB 2: スキャン実行
# =========================================================================
with tab_scan:
    st.subheader("スキャン実行")

    # ルール選択
    scan_source = st.radio(
        "スキャンするルール",
        ["保存済みルール", "現在編集中のルール"],
        horizontal=True,
        key="an_scan_source",
    )

    scan_rule_config = None
    scan_rule_name = "（未選択）"

    if scan_source == "保存済みルール":
        active_rules = _db.list_anomaly_rules()
        if active_rules:
            rule_options = {r["id"]: f'{r["name"]} ({r["status"]})' for r in active_rules}
            selected_rule_id = st.selectbox(
                "ルール選択",
                list(rule_options.keys()),
                format_func=lambda k: rule_options[k],
                key="an_scan_rule_id",
            )
            selected_rule = _db.get_anomaly_rule(selected_rule_id)
            if selected_rule:
                scan_rule_config = selected_rule["features_config"]
                scan_rule_name = selected_rule["name"]
        else:
            st.info("保存済みルールがありません。「ルール作成・編集」タブでルールを作成してください。")
    else:
        if conditions:
            scan_rule_config = _build_rule_config_from_ui(conditions, logic)
            scan_rule_name = rule_name or "（未保存ルール）"
        else:
            st.info("「ルール作成・編集」タブで条件を設定してください。")

    st.divider()

    # 日付選択
    today = date.today()
    scan_date = st.date_input("スキャン日付", value=today, max_value=today, key="an_scan_date")

    # ユニバースフィルタ
    with st.expander("ユニバースフィルタ"):
        an_markets = st.multiselect("市場区分", MARKET_SEGMENTS, default=MARKET_SEGMENTS, key="an_markets")
        an_sectors = st.multiselect("業種", SECTOR_17_LIST, default=[], key="an_sectors")

    universe_config = {
        "market_segments": an_markets,
        "exclude_etf_reit": True,
    }
    if an_sectors:
        universe_config["sector_filter_type"] = "sector_17"
        universe_config["selected_sectors"] = an_sectors

    # スキャン実行ボタン
    an_thread = st.session_state.get("an_thread")
    an_running = an_thread is not None and an_thread.is_alive()

    if an_running:
        _scan_waiting_fragment()
    else:
        # 前回完了チェック
        if an_thread is not None:
            _check_scan_completion()

        if st.button(
            "スキャン実行",
            type="primary",
            key="an_run_scan",
            disabled=scan_rule_config is None,
        ):
            st.session_state["an_scan_result"] = None
            st.session_state["an_scan_error"] = None

            # ① データ取得はメインスレッドで（API はスレッド非対応のため）
            scan_date_str = scan_date.strftime("%Y-%m-%d")
            with st.spinner("銘柄リスト取得中..."):
                listed_stocks = _provider.get_listed_stocks()
                ufc = UniverseFilterConfig(**universe_config) if universe_config else UniverseFilterConfig(exclude_etf_reit=True)
                universe_df = apply_universe_filter(listed_stocks, ufc)

            st.info(f"対象: {len(universe_df):,}銘柄 — 価格データ取得中...")
            target_ts = pd.Timestamp(scan_date_str)
            frames = []
            bar = st.progress(0, text="価格データ取得中... 0/65日分")
            for days_back in range(0, 90):
                d = (target_ts - timedelta(days=days_back)).strftime("%Y-%m-%d")
                df = _provider.get_price_daily_by_date(d)
                if not df.empty:
                    df["date"] = d
                    frames.append(df)
                pct = min(len(frames) / 65, 1.0)
                bar.progress(pct, text=f"価格データ取得中... {len(frames)}/65日分 ({d})")
                if len(frames) >= 65:
                    break

            if not frames:
                st.error("価格データが取得できませんでした。日付を確認してください。")
                st.stop()

            all_prices = pd.concat(frames, ignore_index=True)

            # ② 計算のみバックグラウンドスレッドで
            shared = {"message": "スキャン開始中...", "phase": "scanning"}
            st.session_state["an_progress"] = shared
            st.session_state["an_running"] = True

            t = threading.Thread(
                target=_run_scan_thread,
                args=(shared, scan_rule_config, universe_df, all_prices, scan_date_str),
                daemon=True,
            )
            st.session_state["an_thread"] = t
            t.start()
            st.rerun()

    # スキャン結果表示
    scan_result = st.session_state.get("an_scan_result")
    scan_error = st.session_state.get("an_scan_error")

    if scan_error:
        st.error(f"スキャンエラー: {scan_error}")

    if scan_result is not None and not scan_result.empty:
        result_date = st.session_state.get("an_scan_date", "")
        st.success(f"**{len(scan_result)}銘柄** が検出されました（{result_date}）")

        # 結果テーブル（重要な列を左、セクターは右）
        display_df = pd.DataFrame()
        display_df["コード"] = scan_result["code"]
        display_df["銘柄名"] = scan_result["name"]

        # 主要特徴量のみ表示（ルールで使われている特徴量を優先）
        scan_config = st.session_state.get("an_scan_rule_config", {})
        used_keys = [c.get("feature_key", "") for c in scan_config.get("conditions", [])]
        show_keys = used_keys + [k for k in ["daily_return", "vol_ratio_5d_20d", "ret_5d"] if k not in used_keys]

        features_data = scan_result["features"].tolist()
        for fkey in show_keys:
            if fkey in UI_FEATURE_LABELS_JP:
                label = UI_FEATURE_LABELS_JP[fkey]
                vals = [f.get(fkey) if isinstance(f, dict) else None for f in features_data]
                display_df[label] = [round(v, 2) if isinstance(v, float) else v for v in vals]

        # 検知理由
        display_df["検知理由"] = scan_result["reasons"].apply(
            lambda rs: " / ".join(rs) if isinstance(rs, list) else str(rs)
        )

        # セクターは右端
        display_df["セクター"] = scan_result["sector"]

        # 色付け + フォーマット
        numeric_cols = [c for c in display_df.columns if c not in ("コード", "銘柄名", "セクター", "検知理由")]
        fmt_dict = {c: "{:.2f}" for c in numeric_cols}
        styled = display_df.style.format(fmt_dict, na_rep="—")
        if numeric_cols:
            try:
                styled = styled.map(_color_ret, subset=numeric_cols)
            except AttributeError:
                styled = styled.applymap(_color_ret, subset=numeric_cols)

        st.dataframe(styled, hide_index=True, width="stretch", height=500)

        # 結果保存
        st.divider()
        if st.button("結果を保存", key="an_save_result"):
            rule_id_for_save = None
            if scan_source == "保存済みルール":
                rule_id_for_save = st.session_state.get("an_scan_rule_id")

            if rule_id_for_save is None:
                rule_id_for_save = _db.create_anomaly_rule(
                    name=scan_rule_name,
                    description="スキャン結果保存時に自動作成",
                    features_config=st.session_state.get("an_scan_rule_config", {}),
                    status="active",
                )

            results_for_save = []
            for _, row in scan_result.iterrows():
                results_for_save.append({
                    "code": row["code"],
                    "name": row["name"],
                    "sector": row.get("sector", ""),
                    "reasons": row.get("reasons", []),
                    "features": row.get("features", {}),
                })

            scan_id = _db.create_anomaly_scan(
                rule_id=rule_id_for_save,
                scan_date=result_date,
                results=results_for_save,
                summary={"n_flagged": len(scan_result)},
            )
            st.success(f"スキャン結果を保存しました (ID: {scan_id})")

    elif scan_result is not None and scan_result.empty and not an_running:
        st.info("条件に合致する銘柄はありませんでした。")


# =========================================================================
# TAB 3: スキャン履歴
# =========================================================================
with tab_history:
    st.subheader("スキャン履歴")

    all_scans = _db.list_anomaly_scans()

    if not all_scans:
        st.info("スキャン履歴がありません。「スキャン実行」タブからスキャンを実行して結果を保存してください。")
    else:
        all_rules = {r["id"]: r["name"] for r in _db.list_anomaly_rules()}

        scan_list = []
        for scan in all_scans:
            rule_name_h = all_rules.get(scan["rule_id"], f"(削除済み #{scan['rule_id']})")
            summary = scan.get("summary", {}) or {}
            scan_list.append({
                "ID": scan["id"],
                "日付": scan["scan_date"],
                "ルール": rule_name_h,
                "検出数": summary.get("n_flagged", len(scan.get("results", []))),
                "実行日時": scan["created_at"],
            })

        scan_overview = pd.DataFrame(scan_list)
        st.dataframe(scan_overview, hide_index=True, width="stretch")

        # 詳細表示
        selected_scan_id = st.selectbox(
            "詳細を表示",
            [s["id"] for s in all_scans],
            format_func=lambda sid: f"#{sid} - {next((s['scan_date'] for s in all_scans if s['id'] == sid), '')}",
            key="an_hist_select",
        )

        if selected_scan_id:
            scan_detail = _db.get_anomaly_scan(selected_scan_id)
            if scan_detail:
                results = scan_detail.get("results", [])
                if results:
                    detail_df = pd.DataFrame(results)

                    # 主要カラムを先に
                    display = pd.DataFrame()
                    display["コード"] = detail_df["code"]
                    display["銘柄名"] = detail_df["name"]

                    # 特徴量展開（主要のみ）
                    show_keys = ["daily_return", "vol_ratio_5d_20d", "ret_5d", "range_position_20d"]
                    if "features" in detail_df.columns:
                        for fkey in show_keys:
                            if fkey in UI_FEATURE_LABELS_JP:
                                label = UI_FEATURE_LABELS_JP[fkey]
                                display[label] = detail_df["features"].apply(
                                    lambda f: f.get(fkey) if isinstance(f, dict) else None
                                )

                    if "reasons" in detail_df.columns:
                        display["検知理由"] = detail_df["reasons"].apply(
                            lambda rs: " / ".join(rs) if isinstance(rs, list) else str(rs)
                        )

                    if "sector" in detail_df.columns:
                        display["セクター"] = detail_df["sector"]

                    st.dataframe(display, hide_index=True, width="stretch", height=400)
                else:
                    st.info("結果データがありません。")

                if st.button("このスキャン結果を削除", key=f"an_del_scan_{selected_scan_id}"):
                    _db.delete_anomaly_scan(selected_scan_id)
                    st.rerun()
