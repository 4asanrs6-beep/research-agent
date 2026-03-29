"""日米リードラグ戦略ページ — 部分空間正則化PCAによるセクターETF戦略

Reference:
    中川ら (SIG-FIN-036, 2026)
    "部分空間正則化付き主成分分析を用いた日米業種リードラグ投資戦略"
"""

import threading
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import JQUANTS_API_KEY, MARKET_DATA_DIR
from core.styles import apply_reuters_style, apply_waiting_overlay
from core.lead_lag.config import (
    LeadLagConfig,
    JP_TICKER_NAMES,
    US_TICKER_NAMES,
    JP_TICKERS,
    US_TICKERS,
    UNIVERSE_OPTIONS,
    get_ticker_name,
)

st.set_page_config(page_title="日米リードラグ", page_icon="R", layout="wide")

# 戦略の日本語表示名
STRATEGY_LABELS = {
    "PCA_SUB": "提案手法 (正則化PCA)",
    "PCA_PLAIN": "通常PCA (正則化なし)",
    "MOM": "モメンタム",
    "DOUBLE": "ダブルソート",
    "HYBRID": "HYBRID",
    "K3K4_ENS": "K3/K4アンサンブル",
    "GAP_-10": "GAP_-10%",
    "GAP_0": "GAP_0%",
    "GAP_5": "GAP_5%",
    "GAP_10": "GAP_10%",
    "GAP_20": "GAP_20%",
    "GAP_30": "GAP_30%",
    "K3K4_GAP": "K3K4+GAP",
    "GAP_CUSTOM": "GAP (手動設定)",
}


@st.cache_resource
def _get_provider_and_cache():
    from data.cache import DataCache
    from data.jquants_provider import JQuantsProvider

    cache = DataCache(MARKET_DATA_DIR)
    provider = JQuantsProvider(api_key=JQUANTS_API_KEY, cache=cache) if JQUANTS_API_KEY else None
    return provider, cache


# ---------------------------------------------------------------------------
# バックグラウンドスレッド
# ---------------------------------------------------------------------------
def _run_leadlag_thread(progress_dict: dict, provider, cache, config: LeadLagConfig):
    try:
        from core.lead_lag import run_backtest

        def on_progress(msg, pct):
            progress_dict["message"] = msg
            progress_dict["pct"] = pct

        result = run_backtest(
            config=config,
            jquants_provider=provider,
            cache=cache,
            progress_callback=on_progress,
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
# メイン
# ---------------------------------------------------------------------------
def main():
    apply_reuters_style()

    st.markdown("# 日米セクター リードラグ戦略")
    st.caption("米国セクターETFの当日リターンから、翌営業日の日本セクターETFの寄引リターンを予測 (中川ら, 2026)")

    # 前回の保存結果を自動読み込み
    _ensure_ll_result()

    # スレッド完了の早期検出 (全タブで結果を使えるようにする)
    ll_thread = st.session_state.get("ll_thread")
    if ll_thread is not None and not ll_thread.is_alive() and "ll_result" not in st.session_state:
        prog = st.session_state.get("ll_progress", {})
        if "_result" in prog:
            st.session_state["ll_result"] = prog.pop("_result")

    tab_trade, tab_setup, tab_auto, tab_results, tab_signals, tab_param_compare, tab_attribution = st.tabs([
        "本日の売買",
        "設定・実行",
        "自動探索",
        "結果比較",
        "シグナル分析",
        "パラメータ比較",
        "寄与分解",
    ])

    with tab_trade:
        _render_daily_trade_tab()

    with tab_setup:
        _render_setup_tab()

    with tab_auto:
        _render_auto_search_tab()

    with tab_results:
        _render_results_tab()

    with tab_signals:
        _render_signals_tab()

    with tab_param_compare:
        _render_param_compare_tab()

    with tab_attribution:
        _render_attribution_tab()


# ---------------------------------------------------------------------------
# AI提案パラメータの取得・追加実行
# ---------------------------------------------------------------------------
def _build_suggest_prompt(history: list) -> str:
    """既存結果 + パラメータ比較AI分析を踏まえて、次のパラメータをJSON形式で提案させる。"""
    lines = []
    for i, h in enumerate(history):
        r = h["result"]
        if "PCA_SUB" not in r.strategies:
            continue
        m = r.strategies["PCA_SUB"].metrics
        cfg = r.config
        lines.append(
            f"#{i+1} L={cfg.rolling_window} λ={cfg.lambda_reg} K={cfg.n_components} q={cfg.quantile_q} "
            f"学習期間~{cfg.prior_end_date} → "
            f"年率{m['AR']:+.1f}%, シャープ比{m['R/R']:.2f}, MDD{m['MDD']:.1f}%, "
            f"月次勝率{m.get('monthly_win_rate',0):.0f}%, 月次ブレ幅{m.get('monthly_std',0):.2f}%, "
            f"最悪月{m.get('monthly_worst',0):+.1f}%"
        )
    results_text = "\n".join(lines)

    # パラメータ比較タブのAI分析結果があればそれも渡す
    prior_analysis = st.session_state.get("ll_param_ai", "")
    prior_section = ""
    if prior_analysis:
        # 長すぎる場合は先頭4000文字に切る
        truncated = prior_analysis[:4000]
        prior_section = f"""
## 前回のAI分析レポート (パラメータ比較タブで実施済み)
以下はこれまでの結果を別のAIが詳細に分析した結果です。この分析の知見・提案・警告を踏まえて次の提案をしてください。

{truncated}
"""

    return f"""あなたは定量投資戦略のパラメータ最適化の専門家です。
以下は日米セクターETFリードラグ戦略（部分空間正則化PCA）を異なるパラメータで実行した結果です。

## これまでの結果
{results_text}
{prior_section}
## パラメータの意味
- L (ウィンドウ長): 相関行列推定の日数。20〜252の整数
- λ (正則化強度): 事前知識の信頼度。0.0〜1.0
- K (主成分数): 共通因子の数。1〜10の整数
- q (売買比率): ロング/ショートする割合。0.1〜0.5
- prior_end (学習期間終了日): 事前共変動構造を推定する期間の終了日。"YYYY-MM-DD"形式。開始日以降〜終了日以前で設定。より最近の日付にすると最近の市場構造を反映する

## 重視する指標
- 毎月安定してプラスを出すこと（月次勝率、月次ブレ幅の小ささ）
- 対TOPIX超過リターンの安定性（特に2023年以降のTOPIX強気局面でも勝てるか）
- 最大下落率の抑制

## あなたのタスク
前回のAI分析レポートの知見を活かし、まだ試していない有望なパラメータの組み合わせを8つ提案してください。
前回の分析で指摘された傾向を踏まえ、その延長線上でさらに良い設定を探索してください。

重要: 学習期間 (prior_end) も探索してください。古い学習期間 (2014年等) では2023年以降のTOPIX強気局面に対応できていない可能性がある。
学習期間を変えることで、最近の市場構造（半導体・AI主導の相場等）を反映し、対TOPIX超過が改善する可能性があります。

**必ず以下のJSON形式のみで回答してください。説明文は不要です:**
```json
[
  {{"label": "提案名", "L": 80, "lambda": 0.88, "K": 3, "q": 0.3, "prior_end": "2020-12-31", "reason": "理由"}},
  ...
]
```"""


def _run_ai_suggest_thread(progress_dict: dict, prompt: str):
    """AIにパラメータ提案を依頼。"""
    try:
        from core.ai_client import create_ai_client
        import json

        progress_dict["message"] = "AI にパラメータを提案させています..."
        progress_dict["pct"] = 0.3
        client = create_ai_client()
        # JSON出力を求めるのでデフォルトのシステムプロンプト使用
        response = client.send_message(prompt)

        # JSONを抽出
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        # [ ] を探す
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        suggestions = json.loads(text)
        progress_dict["_result"] = suggestions
        progress_dict["pct"] = 1.0
        progress_dict["message"] = "完了"
    except Exception as e:
        import traceback
        progress_dict["error"] = str(e)
        progress_dict["detail"] = traceback.format_exc()
        progress_dict["pct"] = 1.0


def _render_ai_suggest_and_run():
    """AI提案パラメータの取得 → 追加実行 UI。"""

    # --- AI提案取得中 ---
    suggest_thread = st.session_state.get("ll_ai_suggest_thread")
    is_suggesting = suggest_thread is not None and suggest_thread.is_alive()

    if is_suggesting:
        prog = st.session_state.get("ll_ai_suggest_progress", {})
        st.warning("AI がパラメータを分析・提案中...")
        st.progress(prog.get("pct", 0), text=prog.get("message", ""))
        import time
        time.sleep(2)
        st.rerun()
        return

    # 提案取得完了
    if suggest_thread is not None and not suggest_thread.is_alive() and "ll_ai_suggestions" not in st.session_state:
        prog = st.session_state.get("ll_ai_suggest_progress", {})
        if "_result" in prog:
            st.session_state["ll_ai_suggestions"] = prog.pop("_result")
            st.session_state.pop("ll_ai_suggest_thread", None)
            st.session_state.pop("ll_ai_suggest_progress", None)
            st.rerun()
        elif prog.get("error"):
            st.error(f"AI提案エラー: {prog['error']}")
            if st.button("OK", key="suggest_err_ok"):
                st.session_state.pop("ll_ai_suggest_thread", None)
                st.session_state.pop("ll_ai_suggest_progress", None)
                st.rerun()
            return

    # --- 追加探索実行中 ---
    extra_thread = st.session_state.get("ll_extra_thread")
    is_extra_running = extra_thread is not None and extra_thread.is_alive()

    if is_extra_running:
        prog = st.session_state.get("ll_extra_progress", {})
        import time
        started_at = prog.get("_started_at")
        elapsed = time.time() - started_at if started_at else 0
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        st.warning(f"AI提案パラメータで追加探索中... ({elapsed_min}:{elapsed_sec:02d} 経過)")
        st.progress(prog.get("pct", 0), text=prog.get("message", ""))
        time.sleep(3)
        st.rerun()
        return

    # 追加探索完了
    if extra_thread is not None and not extra_thread.is_alive() and "ll_extra_done" not in st.session_state:
        prog = st.session_state.get("ll_extra_progress", {})
        if "_result" in prog:
            all_results = prog.pop("_result")
            history = st.session_state.get("ll_history", [])
            for item in all_results:
                result = item["result"]
                cfg = result.config
                param_key = f"L{cfg.rolling_window}_λ{cfg.lambda_reg}_K{cfg.n_components}_q{cfg.quantile_q}_G{cfg.gap_threshold}_P{cfg.prior_end_date}_{result.period_start}_{result.period_end}"
                existing_keys = [h.get("_param_key") for h in history]
                if param_key not in existing_keys:
                    history.append({
                        "_param_key": param_key,
                        "label": f"L={cfg.rolling_window} λ={cfg.lambda_reg} K={cfg.n_components} q={cfg.quantile_q}" + (f" GAP{int(cfg.gap_threshold*100)}%" if cfg.gap_threshold < 1.0 else "") + f" ~{cfg.prior_end_date[:4]}",
                        "period": f"{result.period_start}~{result.period_end}",
                        "result": result,
                    })
            st.session_state["ll_extra_done"] = True
            st.session_state.pop("ll_extra_thread", None)
            st.session_state.pop("ll_extra_progress", None)
            # 提案をクリアして次のラウンドを可能に
            st.session_state.pop("ll_ai_suggestions", None)
            st.rerun()
        elif prog.get("error"):
            st.error(f"追加探索エラー: {prog['error']}")
            st.session_state.pop("ll_extra_thread", None)
            st.session_state.pop("ll_extra_progress", None)
            return

    if "ll_extra_done" in st.session_state:
        n = len(st.session_state.get("ll_history", []))
        st.success(f"AI提案パラメータの追加探索が完了しました (履歴: {n}件)")
        st.info("「パラメータ比較」タブで全結果を比較できます。さらにAI提案で探索を続けることもできます。")
        st.session_state.pop("ll_extra_done", None)

    # --- 提案がある場合: 表示 + 実行ボタン ---
    suggestions = st.session_state.get("ll_ai_suggestions")
    if suggestions:
        st.markdown("### AI が提案するパラメータ")
        sug_rows = []
        for i, s in enumerate(suggestions):
            row = {
                "#": i + 1,
                "提案名": s.get("label", f"提案{i+1}"),
                "L": s.get("L", 60),
                "λ": s.get("lambda", 0.9),
                "K": s.get("K", 3),
                "q": s.get("q", 0.3),
            }
            if "prior_end" in s:
                row["学習期間"] = s["prior_end"]
            row["理由"] = s.get("reason", "")
            sug_rows.append(row)
        st.dataframe(pd.DataFrame(sug_rows), hide_index=True)

        if st.button("この提案パラメータで追加探索を実行", type="primary", key="run_ai_suggestions"):
            # 提案をグリッド形式に変換 (prior_end含む)
            grid = []
            for i, s in enumerate(suggestions):
                g = {"label": s.get("label", f"AI提案{i+1}"), "L": s["L"], "lambda": s["lambda"], "K": s["K"], "q": s["q"]}
                if "prior_end" in s:
                    g["prior_end"] = s["prior_end"]
                grid.append(g)
            # 期間は直近の履歴から取得
            history = st.session_state.get("ll_history", [])
            if history:
                last_cfg = history[-1]["result"].config
                start_d = last_cfg.start_date
                end_d = last_cfg.end_date
                prior_d = last_cfg.prior_end_date
            else:
                start_d, end_d, prior_d = "2015-01-01", "2025-12-31", "2018-12-31"

            provider, cache = _get_provider_and_cache()
            import time as _time
            progress_dict = {"message": "開始中...", "pct": 0.0, "_started_at": _time.time()}
            st.session_state["ll_extra_progress"] = progress_dict
            thread = threading.Thread(
                target=_run_auto_search_thread,
                args=(progress_dict, provider, cache, start_d, end_d if end_d else "2025-12-31", prior_d, grid),
                daemon=True,
            )
            thread.start()
            st.session_state["ll_extra_thread"] = thread
            st.rerun()
        return

    # --- 提案がない場合: 取得ボタン ---
    history = st.session_state.get("ll_history", [])
    if len(history) >= 2:
        st.markdown("### AI による次の探索提案")
        st.caption("これまでの結果を分析して、まだ試していない有望なパラメータをAIが提案します。")
        if st.button("AI に次のパラメータを提案させる", type="primary", key="get_ai_suggestions"):
            prompt = _build_suggest_prompt(history)
            progress_dict = {"message": "開始中...", "pct": 0.0}
            st.session_state["ll_ai_suggest_progress"] = progress_dict
            thread = threading.Thread(
                target=_run_ai_suggest_thread,
                args=(progress_dict, prompt),
                daemon=True,
            )
            thread.start()
            st.session_state["ll_ai_suggest_thread"] = thread
            st.rerun()


# ---------------------------------------------------------------------------
# 自動探索: パラメータグリッド定義
# ---------------------------------------------------------------------------
# ベース: L=110 λ=0.85 K=3 q=0.5 GAP5% NE50%復活 累積
_BASE = {"L": 110, "lambda": 0.85, "K": 3, "q": 0.5, "gap": 0.05, "net_exposure": 0.5, "ne_mode": "fill", "accumulate": True}

def _grid(label, **overrides):
    """ベースパラメータを上書きしてグリッド項目を生成。"""
    entry = {**_BASE, "label": label}
    entry.update(overrides)
    return entry

AUTO_SEARCH_GRID = [
    # --- 0. ベースライン ---
    _grid("ベース (最適)"),
    # --- 1. ウィンドウ長 L ---
    _grid("L=80",   L=80),
    _grid("L=90",   L=90),
    _grid("L=100",  L=100),
    _grid("L=120",  L=120),
    _grid("L=130",  L=130),
    # --- 2. 正則化強度 λ ---
    _grid("λ=0.80", **{"lambda": 0.80}),
    _grid("λ=0.83", **{"lambda": 0.83}),
    _grid("λ=0.87", **{"lambda": 0.87}),
    _grid("λ=0.90", **{"lambda": 0.90}),
    # --- 3. 主成分数 K ---
    _grid("K=2",    K=2),
    _grid("K=4",    K=4),
    _grid("K=5",    K=5),
    # --- 4. 売買比率 q ---
    _grid("q=0.3",  q=0.3),
    _grid("q=0.4",  q=0.4),
    _grid("q=0.45", q=0.45),
    # --- 5. GAPフィルター閾値 ---
    _grid("GAP-10%", gap=-0.1),
    _grid("GAP0%",   gap=0.0),
    _grid("GAP10%",  gap=0.1),
    _grid("GAP20%",  gap=0.2),
    # --- 6. ネットエクスポージャー ---
    _grid("NE0%復活 (完全バランス)",  net_exposure=0.0),
    _grid("NE25%復活",                net_exposure=0.25),
    _grid("NE100% (制限なし)",        net_exposure=1.0, ne_mode="trim"),
    _grid("NE50%削減",                ne_mode="trim"),
    _grid("NE0%削減",                 net_exposure=0.0, ne_mode="trim"),
    # --- 7. |NE|スキップ ---
    _grid("|NE|≧9 skip",             ne_skip_long=9, ne_skip_short=9, net_exposure=1.0, ne_mode="trim"),
    _grid("|NE|≧8 skip",             ne_skip_long=8, ne_skip_short=8, net_exposure=1.0, ne_mode="trim"),
    _grid("|NE|≧7 skip",             ne_skip_long=7, ne_skip_short=7, net_exposure=1.0, ne_mode="trim"),
    # --- 8. 累積の有無 ---
    _grid("累積なし",                 accumulate=False),
    # --- 9. 学習期間 ---
    _grid("学習~2020",  prior_end="2020-12-31"),
    _grid("学習~2022",  prior_end="2022-12-31"),
    _grid("学習~2024",  prior_end="2024-06-30"),
    # --- 10. 複合 ---
    _grid("L=100 λ=0.83 K=4",        L=100, **{"lambda": 0.83}, K=4),
    _grid("L=120 GAP0% |NE|≧9skip",  L=120, gap=0.0, ne_skip_long=9, ne_skip_short=9, net_exposure=1.0, ne_mode="trim"),
]


def _run_auto_search_thread(progress_dict: dict, provider, cache, start_date: str, end_date: str, prior_end: str, grid: list):
    """自動探索: 複数パラメータを順次実行する。"""
    try:
        from core.lead_lag import run_backtest

        results = []
        total = len(grid)
        for i, params in enumerate(grid):
            progress_dict["message"] = f"[{i+1}/{total}] {params['label']} (L={params['L']}, λ={params['lambda']})"
            progress_dict["pct"] = i / total

            p_prior = params.get("prior_end", prior_end)
            p_gap = params.get("gap", 1.0)  # 1.0 = フィルターなし
            p_ne = params.get("net_exposure", 1.0)
            p_ne_mode = params.get("ne_mode", "trim")
            p_ne_skip_long = params.get("ne_skip_long", 0)
            p_ne_skip_short = params.get("ne_skip_short", 0)
            p_accum = params.get("accumulate", False)
            config = LeadLagConfig(
                start_date=start_date,
                end_date=end_date,
                prior_end_date=p_prior,
                rolling_window=params["L"],
                n_components=params["K"],
                lambda_reg=params["lambda"],
                quantile_q=params["q"],
                gap_threshold=p_gap,
                net_exposure_limit=p_ne,
                net_exposure_mode=p_ne_mode,
                net_exposure_skip_long=p_ne_skip_long,
                net_exposure_skip_short=p_ne_skip_short,
                accumulate_us_returns=p_accum,
                run_pca_sub=True,
                run_pca_plain=False,
                run_mom=False,
                run_double=False,
            )

            result = run_backtest(
                config=config,
                jquants_provider=provider,
                cache=cache,
            )
            results.append({
                "params": params,
                "result": result,
            })

        progress_dict["_result"] = results
        progress_dict["pct"] = 1.0
        progress_dict["message"] = "完了"
    except Exception as e:
        import traceback
        progress_dict["error"] = str(e)
        progress_dict["detail"] = traceback.format_exc()
        progress_dict["pct"] = 1.0


# ---------------------------------------------------------------------------
# タブ: 自動探索
# ---------------------------------------------------------------------------
def _render_auto_search_tab():
    st.markdown("## 自動パラメータ探索")
    st.caption(
        "期間を指定するだけで、12パターンのパラメータを自動で実行し、"
        "最適な組み合わせをAIが分析します。"
    )

    # 実行中チェック
    auto_thread = st.session_state.get("ll_auto_thread")
    is_running = auto_thread is not None and auto_thread.is_alive()

    if is_running:
        prog = st.session_state.get("ll_auto_progress", {})
        pct = prog.get("pct", 0)
        msg = prog.get("message", "実行中...")

        import time
        started_at = prog.get("_started_at")
        elapsed = time.time() - started_at if started_at else 0
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)

        st.warning(f"自動探索 実行中... ({elapsed_min}:{elapsed_sec:02d} 経過)")
        st.progress(pct, text=msg)

        if pct > 0.01 and pct < 0.99:
            remaining = elapsed / pct * (1 - pct)
            remaining_min = int(remaining // 60)
            remaining_sec = int(remaining % 60)
            st.caption(f"残り約 {remaining_min}:{remaining_sec:02d}")

        time.sleep(3)
        st.rerun()
        return

    # 完了後: 結果を履歴に登録
    if auto_thread is not None and not auto_thread.is_alive() and "ll_auto_done" not in st.session_state:
        prog = st.session_state.get("ll_auto_progress", {})
        if "_result" in prog:
            all_results = prog.pop("_result")
            if "ll_history" not in st.session_state:
                st.session_state["ll_history"] = []
            history = st.session_state["ll_history"]

            for item in all_results:
                params = item["params"]
                result = item["result"]
                cfg = result.config
                suffix = ""
                pass  # 拡張戦略は自動計算
                param_key = f"L{cfg.rolling_window}_λ{cfg.lambda_reg}_K{cfg.n_components}_q{cfg.quantile_q}_G{cfg.gap_threshold}_P{cfg.prior_end_date}{suffix}_{result.period_start}_{result.period_end}"
                label = f"L={cfg.rolling_window} λ={cfg.lambda_reg} K={cfg.n_components} q={cfg.quantile_q}" + (f" GAP{int(cfg.gap_threshold*100)}%" if cfg.gap_threshold < 1.0 else "") + f" ~{cfg.prior_end_date[:4]}"
                pass
                existing_keys = [h.get("_param_key") for h in history]
                if param_key not in existing_keys:
                    history.append({
                        "_param_key": param_key,
                        "label": label,
                        "period": f"{result.period_start}~{result.period_end}",
                        "result": result,
                    })

            # シャープ比最高の結果を現在の結果としてセット
            if all_results:
                def _get_sharpe(item):
                    s = item["result"].strategies
                    # アンサンブルや動的qの結果も考慮
                    best_sr = 0
                    for key in ["PCA_SUB", "ENSEMBLE", "DYNAMIC_Q"]:
                        if key in s:
                            best_sr = max(best_sr, s[key].metrics.get("R/R", 0))
                    return best_sr
                    return 0
                best = max(all_results, key=_get_sharpe)
                st.session_state["ll_result"] = best["result"]

            st.session_state["ll_auto_done"] = True
            st.session_state.pop("ll_auto_thread", None)
            st.session_state.pop("ll_auto_progress", None)
            st.rerun()

        elif prog.get("error"):
            st.error(f"自動探索エラー: {prog['error']}")
            detail = prog.get("detail", "")
            if detail:
                with st.expander("詳細"):
                    st.code(detail)
            if st.button("OK", key="auto_error_ok"):
                st.session_state.pop("ll_auto_thread", None)
                st.session_state.pop("ll_auto_progress", None)
                st.rerun()
            return

    if "ll_auto_done" in st.session_state:
        n = len(st.session_state.get("ll_history", []))
        st.success(f"自動探索完了 - {n} パターンの結果が「パラメータ比較」タブに保存されました。")

        # --- AI提案による追加探索 ---
        _render_ai_suggest_and_run()

        st.markdown("---")
        if st.button("最初からやり直す", key="auto_restart"):
            st.session_state.pop("ll_auto_done", None)
            st.session_state.pop("ll_ai_suggestions", None)
            st.session_state.pop("ll_ai_suggest_thread", None)
            st.session_state.pop("ll_ai_suggest_progress", None)
            st.rerun()
        return

    # --- 入力フォーム ---
    with st.form("auto_search_form"):
        st.markdown("### 分析期間")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "開始日", value=date(2015, 1, 1),
                min_value=date(2005, 1, 1), max_value=date(2026, 12, 31),
                key="auto_start",
            )
        with col2:
            end_date = st.date_input(
                "終了日", value=date(2025, 12, 31),
                min_value=date(2010, 1, 1), max_value=date(2026, 12, 31),
                key="auto_end",
            )

        st.markdown("### 事前知識の学習期間")
        st.caption("開始日からこの日付までのデータで日米の共変動パターンを学習します。")
        prior_end = st.date_input(
            "学習期間の終了日", value=date(2018, 12, 31),
            min_value=date(2010, 1, 1), max_value=date(2025, 12, 31),
            key="auto_prior",
        )

        submitted = st.form_submit_button(f"自動探索を開始 ({len(AUTO_SEARCH_GRID)}パターン)", type="primary")

    if submitted:
        provider, cache = _get_provider_and_cache()

        import time as _time
        progress_dict = {"message": "開始中...", "pct": 0.0, "_started_at": _time.time()}
        st.session_state["ll_auto_progress"] = progress_dict

        thread = threading.Thread(
            target=_run_auto_search_thread,
            args=(progress_dict, provider, cache, str(start_date), str(end_date), str(prior_end), AUTO_SEARCH_GRID),
            daemon=True,
        )
        thread.start()
        st.session_state["ll_auto_thread"] = thread
        st.session_state.pop("ll_auto_done", None)
        st.rerun()

    # --- 探索パラメータ一覧 ---
    with st.expander(f"探索するパラメータ一覧 ({len(AUTO_SEARCH_GRID)}パターン)"):
        grid_df = pd.DataFrame([
            {
                "#": i + 1,
                "パターン名": p["label"],
                "ウィンドウ長 (L)": p["L"],
                "正則化 (λ)": p["lambda"],
                "主成分数 (K)": p["K"],
                "売買比率 (q)": p["q"],
                "学習期間終了": p.get("prior_end", "(フォーム指定)"),
            }
            for i, p in enumerate(AUTO_SEARCH_GRID)
        ])
        st.dataframe(grid_df, hide_index=True)
        st.caption(
            "各パラメータの意味: "
            "**L** = 相関計算の日数 (大きいほど安定、小さいほど直近重視) / "
            "**λ** = 経済理論をどの程度信用するか / "
            "**K** = 日米共通パターンの数 / "
            "**q** = 上位/下位何%を売買するか"
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 結果の保存/読み込み (pickle)
# ---------------------------------------------------------------------------
import json
import pickle
from pathlib import Path

RESULT_DIR = Path("storage/leadlag_results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _save_result_to_disk(result, name: str | None = None) -> str:
    """BacktestResult を pickle で保存する。"""
    cfg = result.config
    if name is None:
        name = (
            f"L{cfg.rolling_window}_lam{cfg.lambda_reg}_K{cfg.n_components}"
            f"_q{cfg.quantile_q}_{cfg.us_universe}"
        )
    path = RESULT_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(path)


def _load_result_from_disk(path: str | Path):
    """pickle から BacktestResult を読み込む。"""
    with open(path, "rb") as f:
        return pickle.load(f)


def _list_saved_results() -> list[dict]:
    """保存済み結果の一覧を返す。"""
    results = []
    for p in sorted(RESULT_DIR.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True):
        results.append({
            "name": p.stem,
            "path": str(p),
            "size_mb": p.stat().st_size / 1024 / 1024,
            "modified": pd.Timestamp.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    return results


def _ensure_ll_result():
    """セッションに ll_result がなければ、最新の保存結果を自動読み込みする。"""
    if "ll_result" in st.session_state:
        return
    # 「新しい実験を開始」でクリアされた直後は復元しない
    if st.session_state.get("ll_cleared"):
        return
    saved = _list_saved_results()
    if saved:
        try:
            result = _load_result_from_disk(saved[0]["path"])
            st.session_state["ll_result"] = result
        except Exception:
            pass


# ---------------------------------------------------------------------------
# プリセット (設定の保存/読み込み)
# ---------------------------------------------------------------------------
PRESETS_PATH = Path("storage/leadlag_presets.json")


def _load_presets() -> dict:
    if PRESETS_PATH.exists():
        try:
            return json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_presets(presets: dict):
    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRESETS_PATH.write_text(json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")


def _render_presets():
    presets = _load_presets()

    with st.expander("設定の保存/読み込み"):
        # 読み込み
        if presets:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected = st.selectbox(
                    "保存済み設定",
                    list(presets.keys()),
                    key="preset_select",
                )
            with col2:
                if st.button("読み込む", key="load_preset"):
                    p = presets[selected]
                    # 古い結果をクリア（新しい条件で再実行が必要）
                    st.session_state.pop("ll_result", None)
                    st.session_state.pop("ll_interpretation", None)
                    st.session_state["ll_cleared"] = True
                    # フォームウィジェットのsession_stateを直接上書き
                    from datetime import date as _d
                    for k in ["form_L", "form_lambda", "form_K", "form_q", "form_gap",
                              "form_prior", "form_start", "form_end", "form_us",
                              "form_accumulate", "form_net_exposure", "form_ne_mode",
                              "form_ne_skip_long", "form_ne_skip_short"]:
                        st.session_state.pop(k, None)
                    st.session_state["form_L"] = p["L"]
                    st.session_state["form_lambda"] = p["lambda"]
                    st.session_state["form_K"] = p["K"]
                    st.session_state["form_q"] = p["q"]
                    st.session_state["form_gap"] = p.get("gap", 100)
                    st.session_state["form_prior"] = _d.fromisoformat(p.get("prior_end", "2018-12-31"))
                    st.session_state["form_start"] = _d.fromisoformat(p.get("start", "2015-01-01"))
                    st.session_state["form_end"] = _d.fromisoformat(p.get("end", "2025-12-31"))
                    st.session_state["form_us"] = p.get("us_universe", "US_11")
                    st.session_state["form_accumulate"] = p.get("accumulate", False)
                    st.session_state["form_net_exposure"] = p.get("net_exposure", 100)
                    st.session_state["form_ne_mode"] = p.get("ne_mode", "trim")
                    st.session_state["form_ne_skip_long"] = p.get("ne_skip_long", p.get("ne_skip", 0))
                    st.session_state["form_ne_skip_short"] = p.get("ne_skip_short", p.get("ne_skip", 0))
                    st.success(f"「{selected}」を読み込みました")
                    st.rerun()
                if st.button("削除", key="delete_preset"):
                    del presets[selected]
                    _save_presets(presets)
                    st.rerun()
        else:
            st.caption("保存済みの設定はありません")

        # 保存
        result = st.session_state.get("ll_result")
        if result:
            cfg = result.config
            col1, col2 = st.columns([3, 1])
            with col1:
                preset_name = st.text_input(
                    "名前を付けて保存",
                    value=f"L={cfg.rolling_window} λ={cfg.lambda_reg} K={cfg.n_components} q={cfg.quantile_q}",
                    key="preset_name",
                )
            with col2:
                if st.button("保存", key="save_preset"):
                    gap_pct = int(cfg.gap_threshold * 100) if cfg.gap_threshold < 1.0 else 100
                    presets[preset_name] = {
                        "L": cfg.rolling_window,
                        "lambda": cfg.lambda_reg,
                        "K": cfg.n_components,
                        "q": cfg.quantile_q,
                        "gap": gap_pct,
                        "prior_end": cfg.prior_end_date,
                        "start": cfg.start_date,
                        "end": cfg.end_date,
                        "us_universe": cfg.us_universe,
                        "accumulate": cfg.accumulate_us_returns,
                        "net_exposure": int(cfg.net_exposure_limit * 100),
                        "ne_mode": cfg.net_exposure_mode,
                        "ne_skip_long": cfg.net_exposure_skip_long,
                        "ne_skip_short": cfg.net_exposure_skip_short,
                    }
                    _save_presets(presets)
                    st.success(f"「{preset_name}」を保存しました")


# ---------------------------------------------------------------------------
# タブ: 本日の売買
# ---------------------------------------------------------------------------
def _render_daily_trade_tab():
    result = st.session_state.get("ll_result")
    if result is None:
        st.info("「設定・実行」タブでバックテストを実行してください。結果に基づいて売買シグナルを表示します。")
        return

    strategies = result.strategies
    config = result.config
    if "PCA_SUB" not in strategies or strategies["PCA_SUB"].signals is None:
        st.warning("PCA_SUBの結果がありません。")
        return

    sig_df = strategies["PCA_SUB"].signals.dropna(how="all")
    us_ret = result.us_cc_returns
    close_df = result.jp_close_prices
    open_df = result.jp_open_prices
    gap_df = result.jp_overnight_gaps

    if len(sig_df) == 0:
        return

    st.markdown("## 売買シグナル")

    # --- パラメータサマリー ---
    gap_pct = int(config.gap_threshold * 100) if config.gap_threshold < 1.0 else None
    param_text = f"L={config.rolling_window}  λ={config.lambda_reg}  K={config.n_components}  q={config.quantile_q}"
    if gap_pct is not None:
        param_text += f"  GAP={gap_pct}%"
    st.caption(f"設定: {param_text}  |  期間: {result.period_start}〜{result.period_end}")

    # --- 日付選択 (選んだ日 = 日本の売買日) ---
    available_dates = sig_df.index.tolist()
    from datetime import date as _date
    today = _date.today()

    # 売買可能な日 = データの2日目以降 (前日のシグナルが必要) + 最新シグナルの翌日
    trade_dates = available_dates[1:]  # signals[t-1] で t に売買

    selected_input = st.date_input(
        "売買する日を選択",
        value=today,
        min_value=trade_dates[0].date() if hasattr(trade_dates[0], 'date') else trade_dates[0],
        max_value=today,
        key="trade_date_input",
    )

    # 最も近い売買可能日を探す
    selected_ts = pd.Timestamp(selected_input)
    # 選択日以前の最も近い売買日を探す (未来日を使わない)
    candidates = [d for d in trade_dates if d <= selected_ts]
    if not candidates:
        candidates = trade_dates[:1]
    jp_trade_date = candidates[-1]

    # シグナル日の決定
    # 新アライメント: sig_df[t] は前日US終値ベース → t日のJP市場で売買。
    # よって売買日 jp_trade_date に対するシグナル日は同日。
    if jp_trade_date in available_dates:
        signal_date = jp_trade_date
    else:
        # jp_trade_dateがavailable_datesにない場合 (最新シグナルの翌営業日等)
        signal_date = available_dates[-1]

    # 最新シグナルの翌日 (まだ売買していない日) も選択可能にする
    last_signal = available_dates[-1]
    is_future = selected_ts > last_signal
    if is_future:
        signal_date = last_signal
        jp_trade_date = selected_ts

    if selected_ts != jp_trade_date and not is_future:
        st.caption(f"※ {selected_input} は非営業日のため {jp_trade_date.strftime('%Y-%m-%d')} を表示")

    # シグナルデータ
    sig = sig_df.loc[signal_date]
    q = config.quantile_q
    n_long = max(1, int(np.ceil(len(sig.dropna()) * q)))
    sorted_sig = sig.dropna().sort_values(ascending=False)

    # 米国データ日 (JP日付→実際のUS日付マッピングを使用)
    date_map = result.jp_to_us_date_map or {}
    actual_us_date = date_map.get(signal_date)
    if actual_us_date is None:
        # フォールバック: us_retのインデックスから取得
        us_dates_before = us_ret.index[us_ret.index <= signal_date]
        actual_us_date = us_dates_before[-1] if len(us_dates_before) > 0 else None

    jp_date_str = jp_trade_date.strftime('%Y-%m-%d') if hasattr(jp_trade_date, 'strftime') else str(jp_trade_date)
    us_date_str = actual_us_date.strftime('%Y-%m-%d') if actual_us_date is not None else "不明"

    # 今日以降 or データ範囲外は未来扱い
    from datetime import date as _today_date
    today_ts = pd.Timestamp(_today_date.today())
    is_today_or_future = jp_trade_date >= today_ts

    if is_future or is_today_or_future:
        st.info(f"**米国 {us_date_str} の終値に基づく → 日本 {jp_date_str} の売買指示**")
    else:
        st.markdown(f"**米国 {us_date_str} の終値 → 日本 {jp_date_str} の売買 (実績)**")

    if signal_date in us_ret.index:
        us_day_ret = us_ret.loc[signal_date]

        # 累積の場合、どの期間の累積かを表示
        us_title_suffix = ""
        if config.accumulate_us_returns and date_map:
            sig_idx = available_dates.index(signal_date) if signal_date in available_dates else -1
            if sig_idx > 0:
                prev_us = date_map.get(available_dates[sig_idx - 1])
                curr_us = date_map.get(signal_date)
                if prev_us is not None and curr_us is not None:
                    # 全US営業日リストから、prev_usの翌日〜curr_usの期間を取得
                    us_all = result.us_trading_dates or []
                    accum_dates = [d for d in us_all if d > prev_us and d <= curr_us]
                    if len(accum_dates) > 1:
                        us_title_suffix = f" [累積: {accum_dates[0].strftime('%m/%d')}〜{accum_dates[-1].strftime('%m/%d')} ({len(accum_dates)}日分)]"

        st.markdown(f"### 米国セクター騰落率 ({us_date_str}){us_title_suffix}")
        us_rows = []
        for t in us_day_ret.index:
            us_rows.append({
                "セクター": get_ticker_name(t),
                "騰落率 (%)": round(us_day_ret[t] * 100, 2),
            })
        us_display = pd.DataFrame(us_rows).sort_values("騰落率 (%)", ascending=False)
        st.dataframe(
            us_display.style.format({"騰落率 (%)": "{:+.1f}"}).map(
                lambda v: "color: #2E7D32" if isinstance(v, (int, float)) and v > 0
                else "color: #C62828" if isinstance(v, (int, float)) and v < 0 else "",
                subset=["騰落率 (%)"],
            ),
            hide_index=True, height=min(420, len(us_rows) * 35 + 40),
        )

    # --- 売買シグナルテーブル ---
    has_gap = config.gap_threshold < 1.0
    has_price = close_df is not None and open_df is not None

    # is_future は既に上で設定済み

    st.markdown(f"### 日本セクターETF 売買判断 ({jp_date_str})")

    trade_rows = []
    for rank, (ticker, sig_val) in enumerate(sorted_sig.items()):
        sector = get_ticker_name(ticker)
        if rank < n_long:
            position = "ロング"
        elif rank >= len(sorted_sig) - n_long:
            position = "ショート"
        else:
            position = "-"

        # 前日終値 = 売買日の1つ前のclose_dfインデックスの終値
        last_close = None
        if has_price and ticker in close_df.columns:
            close_dates_before = close_df.index[close_df.index < jp_trade_date]
            if len(close_dates_before) > 0:
                last_close = close_df.loc[close_dates_before[-1], ticker]

        lc = int(last_close) if last_close is not None and not np.isnan(last_close) else None

        row = {
            "セクター": sector,
            "コード": ticker,
            "判定": position,
            "シグナル": round(sig_val, 2),
        }

        if lc:
            row["前日終値"] = f"{lc:,}"

        # 指値目安
        if has_gap and lc and abs(sig_val) > 1e-10 and position in ("ロング", "ショート"):
            if position == "ロング":
                limit = int(lc * (1 + abs(sig_val) * config.gap_threshold))
                row["指値目安"] = f"{limit:,}以下で買い"
            else:
                limit = int(lc * (1 - abs(sig_val) * config.gap_threshold))
                row["指値目安"] = f"{limit:,}以上で売り"
        elif position in ("ロング", "ショート"):
            row["指値目安"] = "成行"

        # 売買日のデータがある場合は実績を表示（今日以降は除外）
        jp_td = jp_trade_date if not is_future and not is_today_or_future else None
        if jp_td is not None and has_price and hasattr(jp_td, 'strftime'):
            jp_td_ts = pd.Timestamp(jp_td) if not isinstance(jp_td, pd.Timestamp) else jp_td
            if jp_td_ts in open_df.index and ticker in open_df.columns:
                t_open = open_df.loc[jp_td_ts, ticker]
                t_close = close_df.loc[jp_td_ts, ticker] if jp_td_ts in close_df.index else None
                op = int(t_open) if t_open is not None and not np.isnan(t_open) else None
                cl = int(t_close) if t_close is not None and not np.isnan(t_close) else None
                if op:
                    row["寄付き"] = f"{op:,}"
                if cl:
                    row["終値"] = f"{cl:,}"
                if op and cl and op > 0:
                    row["当日騰落(%)"] = round((cl / op - 1) * 100, 1)

                # GAP判定 (実績)
                if has_gap and lc and op and abs(sig_val) > 1e-10 and position in ("ロング", "ショート"):
                    actual_gap = (op - lc) / lc
                    if position == "ロング":
                        can_trade = actual_gap <= abs(sig_val) * config.gap_threshold
                    else:
                        can_trade = actual_gap >= -abs(sig_val) * config.gap_threshold
                    row["GAP判定"] = "エントリー" if can_trade else "スキップ"

        trade_rows.append(row)

    trade_df = pd.DataFrame(trade_rows)

    # ネットエクスポージャー上限判定
    if config.net_exposure_limit < 1.0:
        gap_col = "GAP判定"
        has_gap_col = gap_col in trade_df.columns

        # GAPフィルター後のエントリー数を数える
        if has_gap_col:
            long_mask = (trade_df["判定"] == "ロング") & (trade_df[gap_col] == "エントリー")
            short_mask = (trade_df["判定"] == "ショート") & (trade_df[gap_col] == "エントリー")
        else:
            long_mask = trade_df["判定"] == "ロング"
            short_mask = trade_df["判定"] == "ショート"

        n_long_entry = long_mask.sum()
        n_short_entry = short_mask.sum()
        max_diff = max(1, int(np.ceil(n_long * config.net_exposure_limit)))

        ne_results = ["-"] * len(trade_df)
        mode_label = "復活" if config.net_exposure_mode == "fill" else "削減"

        if config.net_exposure_mode == "fill":
            # fillモード: 少ない側にGAPスキップ銘柄を復活
            if has_gap_col:
                long_skipped = (trade_df["判定"] == "ロング") & (trade_df[gap_col] == "スキップ")
                short_skipped = (trade_df["判定"] == "ショート") & (trade_df[gap_col] == "スキップ")
            else:
                long_skipped = pd.Series([False] * len(trade_df))
                short_skipped = pd.Series([False] * len(trade_df))

            if n_long_entry - n_short_entry > max_diff and short_skipped.sum() > 0:
                # ショートが少ない → スキップされたショートをシグナル強い順に復活
                skipped_indices = trade_df[short_skipped].sort_values("シグナル", ascending=True).index
                n_to_fill = min(len(skipped_indices), n_long_entry - n_short_entry - max_diff)
                for idx in skipped_indices[:n_to_fill]:
                    ne_results[idx] = "NE復活"
            elif n_short_entry - n_long_entry > max_diff and long_skipped.sum() > 0:
                # ロングが少ない → スキップされたロングをシグナル強い順に復活
                skipped_indices = trade_df[long_skipped].sort_values("シグナル", ascending=False).index
                n_to_fill = min(len(skipped_indices), n_short_entry - n_long_entry - max_diff)
                for idx in skipped_indices[:n_to_fill]:
                    ne_results[idx] = "NE復活"
        else:
            # trimモード: 多い側を削減
            if n_long_entry - n_short_entry > max_diff:
                long_indices = trade_df[long_mask].sort_values("シグナル", ascending=True).index
                n_to_skip = n_long_entry - (n_short_entry + max_diff)
                for idx in long_indices[:n_to_skip]:
                    ne_results[idx] = "NE超過"
            elif n_short_entry - n_long_entry > max_diff:
                short_indices = trade_df[short_mask].sort_values("シグナル", ascending=False).index
                n_to_skip = n_short_entry - (n_long_entry + max_diff)
                for idx in short_indices[:n_to_skip]:
                    ne_results[idx] = "NE超過"

        trade_df["NE判定"] = ne_results
        ne_pct = int(config.net_exposure_limit * 100)
        st.caption(f"ネットエクスポージャー上限: {ne_pct}% [{mode_label}] (L{n_long_entry} vs S{n_short_entry}, 許容差{max_diff})")

    # ネットエクスポージャー絶対値スキップ判定
    if config.net_exposure_skip_long > 0 or config.net_exposure_skip_short > 0:
        gap_col = "GAP判定"
        has_gap_col = gap_col in trade_df.columns
        if has_gap_col:
            n_l = ((trade_df["判定"] == "ロング") & (trade_df[gap_col] == "エントリー")).sum()
            n_s = ((trade_df["判定"] == "ショート") & (trade_df[gap_col] == "エントリー")).sum()
        else:
            n_l = (trade_df["判定"] == "ロング").sum()
            n_s = (trade_df["判定"] == "ショート").sum()

        net_val = n_l - n_s
        skip_reason = None
        if config.net_exposure_skip_long > 0 and net_val >= config.net_exposure_skip_long:
            skip_reason = f"NE = {net_val} ≧ {config.net_exposure_skip_long} (ロング偏り)"
        elif config.net_exposure_skip_short > 0 and net_val <= -config.net_exposure_skip_short:
            skip_reason = f"NE = {net_val} ≦ -{config.net_exposure_skip_short} (ショート偏り)"
        if skip_reason:
            st.warning(f"{skip_reason} → **本日は全銘柄見送り**")

    # スタイリング
    style_cols = ["判定"]
    if "GAP判定" in trade_df.columns:
        style_cols.append("GAP判定")
    if "NE判定" in trade_df.columns:
        style_cols.append("NE判定")

    def _style(v):
        s = str(v)
        if "ロング" in s or "エントリー" in s:
            return "color: #2E7D32; font-weight: bold"
        if "ショート" in s:
            return "color: #C62828; font-weight: bold"
        if "スキップ" in s or "NE超過" in s:
            return "color: #999; text-decoration: line-through"
        if "NE復活" in s:
            return "color: #E65100; font-weight: bold"
        return ""

    fmt = {"シグナル": "{:+.2f}"}
    if "当日騰落(%)" in trade_df.columns:
        fmt["当日騰落(%)"] = "{:+.1f}"

    st.dataframe(
        trade_df.style.map(_style, subset=style_cols).format(fmt, na_rep="-"),
        hide_index=True,
        height=min(660, len(trade_df) * 35 + 40),
    )

    # --- ティッカーコピー用 ---
    with st.expander("ティッカーをコピー"):
        # ロング/ショート/全てのティッカーを縦一列で表示
        long_tickers = trade_df[trade_df["判定"] == "ロング"]["コード"].tolist()
        short_tickers = trade_df[trade_df["判定"] == "ショート"]["コード"].tolist()
        all_tickers = trade_df["コード"].tolist()
        col_l, col_s, col_a = st.columns(3)
        with col_l:
            st.caption(f"ロング ({len(long_tickers)})")
            st.text_area("ロング", "\n".join(long_tickers), height=150, key="copy_long_tickers", label_visibility="collapsed")
        with col_s:
            st.caption(f"ショート ({len(short_tickers)})")
            st.text_area("ショート", "\n".join(short_tickers), height=150, key="copy_short_tickers", label_visibility="collapsed")
        with col_a:
            st.caption(f"全て ({len(all_tickers)})")
            st.text_area("全て", "\n".join(all_tickers), height=150, key="copy_all_tickers", label_visibility="collapsed")

    # --- 当日リターンサマリー ---
    if not is_future and not is_today_or_future:
        st.markdown("### 当日の戦略リターン")
        jp_td_ts = pd.Timestamp(jp_trade_date) if not isinstance(jp_trade_date, pd.Timestamp) else jp_trade_date

        # 表示対象: フィルターなし → GAP(NE制限なし) → GAP(NE制限込み) の順
        # フィルタリングが厳しい順に右へ
        display_strats = [("PCA_SUB", "フィルターなし")]
        gap_pct_val = int(config.gap_threshold * 100) if config.gap_threshold < 1.0 else None
        # NE制限なしのGAP感応度テスト
        for gap_key in ["GAP_-10", "GAP_0", "GAP_5", "GAP_10", "GAP_20", "GAP_30"]:
            if gap_key in strategies:
                gap_label = gap_key.replace("GAP_", "GAP ")
                gap_rate = {"GAP_-10": -10, "GAP_0": 0, "GAP_5": 5, "GAP_10": 10, "GAP_20": 20, "GAP_30": 30}.get(gap_key)
                if gap_rate == gap_pct_val:
                    display_strats.append((gap_key, f"GAP_{gap_rate}% (NE制限なし)"))
                else:
                    display_strats.append((gap_key, f"{gap_label}%"))
        # NE制限込みのGAP_CUSTOM（最も厳しいフィルター）
        if "GAP_CUSTOM" in strategies:
            display_strats.append(("GAP_CUSTOM", f"GAP_{gap_pct}% (NE制限込み)"))

        ret_cols = []
        for strat_name, label in display_strats:
            if strat_name in strategies:
                ret = strategies[strat_name].daily_returns
                if jp_td_ts in ret.index and not np.isnan(ret.loc[jp_td_ts]):
                    ret_cols.append((label, f"{ret.loc[jp_td_ts] * 100:+.2f}%"))
                else:
                    ret_cols.append((label, "売買なし"))

        if ret_cols:
            for i in range(0, len(ret_cols), 4):
                chunk = ret_cols[i:i+4]
                cols = st.columns(len(chunk))
                for col, (label, val) in zip(cols, chunk):
                    with col:
                        st.metric(label, val)

    # --- パフォーマンスサマリー ---
    st.markdown("---")
    st.markdown("### パフォーマンスサマリー")
    # TOPIX年率
    bm_ar = None
    if result.benchmark_returns is not None and len(result.benchmark_returns) > 0:
        bm_valid = result.benchmark_returns.dropna()
        if len(bm_valid) > 0:
            bm_ar = bm_valid.mean() * 252 * 100

    summary_keys = ["PCA_SUB"]
    if "GAP_CUSTOM" in strategies:
        summary_keys.append("GAP_CUSTOM")
    summary_rows = []
    for key in summary_keys:
        if key not in strategies:
            continue
        m = strategies[key].metrics
        label = "フィルターなし" if key == "PCA_SUB" else f"GAP_{gap_pct}%"
        row = {
            "戦略": label,
            "年率リターン (%)": round(m["AR"], 1),
            "シャープ比": round(m["R/R"], 2),
            "最大下落率 (%)": round(m["MDD"], 1),
            "エントリー率 (%)": round(m.get("entry_rate", 100), 0),
            "月次勝率 (%)": round(m.get("monthly_win_rate", 0), 0),
            "最悪月 (%)": round(m.get("monthly_worst", 0), 1),
        }
        if bm_ar is not None:
            row["対TOPIX超過 (%)"] = round(m["AR"] - bm_ar, 1)
        summary_rows.append(row)
    if summary_rows:
        fmt = {
            "年率リターン (%)": "{:+.1f}",
            "シャープ比": "{:.2f}",
            "最大下落率 (%)": "{:.1f}",
            "エントリー率 (%)": "{:.0f}",
            "月次勝率 (%)": "{:.0f}",
            "最悪月 (%)": "{:+.1f}",
        }
        if "対TOPIX超過 (%)" in summary_rows[0]:
            fmt["対TOPIX超過 (%)"] = "{:+.1f}"
        st.dataframe(
            pd.DataFrame(summary_rows).style.format(fmt),
            hide_index=True,
        )
        if bm_ar is not None:
            st.caption(f"TOPIX年率: {bm_ar:+.1f}% (同期間)")


# ---------------------------------------------------------------------------
# タブ1: 設定・実行
# ---------------------------------------------------------------------------
def _render_setup_tab():
    # 実行中チェック
    ll_thread = st.session_state.get("ll_thread")
    is_running = ll_thread is not None and ll_thread.is_alive()

    if is_running:
        prog = st.session_state.get("ll_progress", {})
        pct = prog.get("pct", 0)
        msg = prog.get("message", "実行中...")

        import time

        started_at = prog.get("_started_at")
        elapsed = time.time() - started_at if started_at else 0
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)

        # --- 明確な実行中 UI ---
        st.warning(f"バックテスト実行中... ({elapsed_min}:{elapsed_sec:02d} 経過)")
        st.progress(pct, text=msg)

        # ステップ説明
        step_labels = {
            "US ETF": "Step 1/5: 米国ETFデータ取得 (yfinance)",
            "JP ETF": "Step 2/5: 日本ETFデータ取得 (J-Quants / yfinance)",
            "整列": "Step 3/5: 日米データの取引日整列",
            "PCA_SUB": "Step 4/5: PCA シグナル計算",
            "PCA_PLAIN": "Step 4/5: PCA シグナル計算",
            "MOM": "Step 5/5: ベースライン戦略計算",
            "DOUBLE": "Step 5/5: ベースライン戦略計算",
            "事前": "Step 3/5: 事前部分空間・相関行列の推定",
            "ポートフォリオ": "Step 4/5: ポートフォリオ構築",
            "完了": "完了",
        }
        step_desc = msg
        for key, label in step_labels.items():
            if key in msg:
                step_desc = label
                break
        st.info(step_desc)

        if pct > 0.01 and pct < 0.99:
            remaining = elapsed / pct * (1 - pct)
            remaining_min = int(remaining // 60)
            remaining_sec = int(remaining % 60)
            st.caption(f"残り約 {remaining_min}:{remaining_sec:02d}")

        time.sleep(2)
        st.rerun()
        return

    # スレッド完了後
    if ll_thread is not None and not ll_thread.is_alive() and "ll_result" not in st.session_state:
        prog = st.session_state.get("ll_progress", {})
        if "_result" in prog:
            st.session_state["ll_result"] = prog.pop("_result")
        elif prog.get("error"):
            st.error(f"実行エラー: {prog['error']}")
            detail = prog.get("detail", "")
            if detail:
                with st.expander("詳細"):
                    st.code(detail)
            if st.button("OK", key="ll_error_ok"):
                st.session_state.pop("ll_thread", None)
                st.session_state.pop("ll_progress", None)
                st.rerun()
            return

    if "ll_result" in st.session_state:
        result = st.session_state["ll_result"]
        st.success("実行完了 - 「結果比較」「シグナル分析」タブで結果を確認できます。")

        # ディスクに自動保存
        try:
            saved_path = _save_result_to_disk(result)
        except Exception as e:
            pass

        # 履歴に自動保存
        if "ll_history" not in st.session_state:
            st.session_state["ll_history"] = []
        history = st.session_state["ll_history"]
        # 同一パラメータの重複チェック
        cfg = result.config
        param_key = f"L{cfg.rolling_window}_λ{cfg.lambda_reg}_K{cfg.n_components}_q{cfg.quantile_q}_G{cfg.gap_threshold}_NE{cfg.net_exposure_limit}_{cfg.net_exposure_mode}_NSL{cfg.net_exposure_skip_long}_NSS{cfg.net_exposure_skip_short}_AC{cfg.accumulate_us_returns}_P{cfg.prior_end_date}_{result.period_start}_{result.period_end}"
        existing_keys = [h.get("_param_key") for h in history]
        if param_key not in existing_keys:
            _label_parts = [f"L={cfg.rolling_window} λ={cfg.lambda_reg} K={cfg.n_components} q={cfg.quantile_q}"]
            if cfg.gap_threshold < 1.0:
                _label_parts.append(f"GAP{int(cfg.gap_threshold*100)}%")
            if cfg.net_exposure_limit < 1.0:
                ne_mode_label = "復活" if cfg.net_exposure_mode == "fill" else "削減"
                _label_parts.append(f"NE{int(cfg.net_exposure_limit*100)}%{ne_mode_label}")
            if cfg.net_exposure_skip_long > 0 or cfg.net_exposure_skip_short > 0:
                skip_parts = []
                if cfg.net_exposure_skip_long > 0:
                    skip_parts.append(f"L≧{cfg.net_exposure_skip_long}")
                if cfg.net_exposure_skip_short > 0:
                    skip_parts.append(f"S≧{cfg.net_exposure_skip_short}")
                _label_parts.append(f"NE{'/'.join(skip_parts)}skip")
            if cfg.accumulate_us_returns:
                _label_parts.append("累積")
            _label_parts.append(f"~{cfg.prior_end_date[:4]}")
            history.append({
                "_param_key": param_key,
                "label": " ".join(_label_parts),
                "period": f"{result.period_start}~{result.period_end}",
                "result": result,
            })
            st.caption(f"パラメータ比較タブに保存しました (履歴: {len(history)}件)")

        # プリセット保存
        presets = _load_presets()
        gap_pct_save = int(cfg.gap_threshold * 100) if cfg.gap_threshold < 1.0 else 100
        default_name = f"L={cfg.rolling_window} λ={cfg.lambda_reg} K={cfg.n_components} q={cfg.quantile_q}"
        if gap_pct_save < 100:
            default_name += f" GAP{gap_pct_save}%"
        if cfg.net_exposure_limit < 1.0:
            ne_mode_label = "復活" if cfg.net_exposure_mode == "fill" else "削減"
            default_name += f" NE{int(cfg.net_exposure_limit*100)}%{ne_mode_label}"
        if cfg.net_exposure_skip_long > 0 or cfg.net_exposure_skip_short > 0:
            skip_parts = []
            if cfg.net_exposure_skip_long > 0:
                skip_parts.append(f"L≧{cfg.net_exposure_skip_long}")
            if cfg.net_exposure_skip_short > 0:
                skip_parts.append(f"S≧{cfg.net_exposure_skip_short}")
            default_name += f" NE{'/'.join(skip_parts)}skip"
        if cfg.accumulate_us_returns:
            default_name += " 累積"
        col1, col2 = st.columns([3, 1])
        with col1:
            preset_name = st.text_input("名前を付けて保存", value=default_name, key="save_preset_name")
        with col2:
            if st.button("保存", key="save_preset_btn"):
                presets[preset_name] = {
                    "L": cfg.rolling_window, "lambda": cfg.lambda_reg,
                    "K": cfg.n_components, "q": cfg.quantile_q,
                    "gap": gap_pct_save, "prior_end": cfg.prior_end_date,
                    "start": cfg.start_date, "end": cfg.end_date,
                    "us_universe": cfg.us_universe,
                    "accumulate": cfg.accumulate_us_returns,
                    "net_exposure": int(cfg.net_exposure_limit * 100),
                    "ne_mode": cfg.net_exposure_mode,
                    "ne_skip_long": cfg.net_exposure_skip_long,
                        "ne_skip_short": cfg.net_exposure_skip_short,
                }
                _save_presets(presets)
                st.success(f"「{preset_name}」を保存しました")

        col_new, col_save = st.columns(2)
        with col_new:
            if st.button("新しい実験を開始"):
                st.session_state.pop("ll_result", None)
                st.session_state.pop("ll_thread", None)
                st.session_state.pop("ll_progress", None)
                st.session_state.pop("ll_interpretation", None)
                st.session_state["ll_cleared"] = True
                st.rerun()
        return

    # --- 保存済み結果の読み込み ---
    saved = _list_saved_results()
    if saved:
        with st.expander(f"保存済みの結果を読み込む ({len(saved)}件)"):
            for item in saved:
                col1, col2, col3 = st.columns([4, 2, 1])
                with col1:
                    st.text(f"{item['name']}")
                with col2:
                    st.caption(f"{item['modified']}  ({item['size_mb']:.1f} MB)")
                with col3:
                    if st.button("読込", key=f"load_{item['name']}"):
                        try:
                            result = _load_result_from_disk(item["path"])
                            st.session_state["ll_result"] = result
                            st.success(f"「{item['name']}」を読み込みました")
                            st.rerun()
                        except Exception as e:
                            st.error(f"読み込みエラー: {e}")

    # --- プリセット保存/読み込み ---
    _render_presets()

    # --- パラメータ入力 ---
    st.markdown("## パラメータ設定")
    st.info(
        "**仮説**: 米国市場で時点tに確定した業種別情報が、"
        "日本市場の翌営業日t+1の日中リターン (Open-to-Close) に波及する"
    )

    # デフォルト値をsession_stateから初期化 (初回のみ)
    if "form_L" not in st.session_state:
        st.session_state.setdefault("form_L", 60)
        st.session_state.setdefault("form_lambda", 0.9)
        st.session_state.setdefault("form_K", 3)
        st.session_state.setdefault("form_q", 0.3)
        st.session_state.setdefault("form_gap", 100)
        st.session_state.setdefault("form_prior", date(2018, 12, 31))
        st.session_state.setdefault("form_start", date(2015, 1, 1))
        st.session_state.setdefault("form_end", date(2025, 12, 31))
        st.session_state.setdefault("form_us", "US_11")

    with st.form("leadlag_form"):
        st.markdown("### ユニバース選択")
        col1, col2 = st.columns(2)
        with col1:
            us_options = ["US_11", "US_33"]
            us_universe = st.selectbox(
                "米国側",
                us_options,
                index=us_options.index(st.session_state["form_us"]) if st.session_state["form_us"] in us_options else 0,
                format_func=lambda x: {"US_11": "11セクター (SPDR)", "US_33": "33種 (11セクター+22インダストリー)"}[x],
                key="form_us",
            )
        with col2:
            jp_universe = "JP_17"
            st.info("日本側: TOPIX-17 (17業種)")

        st.markdown("### 分析期間")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "開始日",
                min_value=date(2005, 1, 1),
                max_value=date(2026, 12, 31),
                key="form_start",
            )
        with col2:
            end_date = st.date_input(
                "終了日",
                min_value=date(2010, 1, 1),
                max_value=date(2026, 12, 31),
                key="form_end",
            )

        st.markdown("### PCA パラメータ")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rolling_window = st.number_input(
                "ウィンドウ長 (L)",
                min_value=20, max_value=252,
                key="form_L",
                help="相関行列を計算する直近の営業日数。60 = 約3ヶ月分。",
            )
        with col2:
            lambda_reg = st.slider(
                "正則化強度",
                min_value=0.0, max_value=1.0, step=0.01,
                key="form_lambda",
                help="経済理論の事前知識をどの程度信用するか。0.9 = 90%事前知識 + 10%データ。論文推奨値は0.9。",
            )
        with col3:
            n_components = st.number_input(
                "主成分数 (K)",
                min_value=1, max_value=10,
                key="form_K",
                help="日米共通の変動パターン数。3 = グローバル景気・国別差・景気循環。",
            )
        with col4:
            quantile_q = st.slider(
                "売買比率",
                min_value=0.1, max_value=0.5, step=0.05,
                key="form_q",
                help="17セクター中の上位/下位何%をロング/ショートするか。0.3 = 上下各5本。",
            )

        st.markdown("### 事前知識の学習期間")
        st.caption(
            "開始日から下記の日付までのデータで、日米セクター間の長期的な共変動パターン（グローバル景気連動など）を学習します。"
            "残りの期間がテスト期間になります。分析期間の前半〜中盤に設定してください。"
        )
        prior_end = st.date_input(
            "学習期間の終了日",
            min_value=date(2010, 1, 1),
            max_value=date(2025, 12, 31),
            key="form_prior",
        )

        st.markdown("### 比較戦略")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            run_pca_sub = st.checkbox("提案手法 (正則化PCA)", value=True, disabled=True)
        with col2:
            run_pca_plain = st.checkbox("通常PCA (正則化なし)", value=True)
        with col3:
            run_mom = st.checkbox("モメンタム", value=True)
        with col4:
            run_double = st.checkbox("ダブルソート", value=True)

        st.markdown("### 休場処理")
        if "form_accumulate" not in st.session_state:
            st.session_state["form_accumulate"] = False
        accumulate_us = st.checkbox(
            "JP休場中の米国リターンを累積してシグナルに反映",
            key="form_accumulate",
            help="JP休場中に発生した複数日分の米国リターンを累積し、休場明けのシグナル入力に使用します。OFFの場合は直前1日のみ使用。",
        )

        st.markdown("### ギャップフィルター")
        st.caption("シグナル強度に対してovernightギャップが閾値以上消化済みの銘柄をスキップ")
        gap_threshold = st.slider(
            "ギャップ消化率の閾値 (%)",
            min_value=-100, max_value=100, step=5,
            key="form_gap",
            help="100%=フィルターなし (全エントリー)。10%=予測の10%消化でスキップ。0%=予測方向に少しでも動いたらスキップ。-20%=予測と逆に20%以上動いた銘柄のみエントリー",
        )

        ne_col1, ne_col2, ne_col3 = st.columns([2, 1, 1])
        with ne_col1:
            if "form_net_exposure" not in st.session_state:
                st.session_state["form_net_exposure"] = 100
            net_exposure = st.slider(
                "ネットエクスポージャー上限 (%)",
                min_value=0, max_value=100, step=5,
                key="form_net_exposure",
                help="GAPフィルター後のL/S偏りを制限。0%=完全バランス。100%=制限なし。",
            )
        with ne_col2:
            if "form_ne_mode" not in st.session_state:
                st.session_state["form_ne_mode"] = "trim"
            ne_mode = st.radio(
                "調整モード",
                options=["trim", "fill"],
                format_func=lambda x: "多い側を削減" if x == "trim" else "少ない側を復活",
                key="form_ne_mode",
                help="trim: 多い側のシグナル弱い銘柄をスキップ。fill: 少ない側にGAPフィルターで外された銘柄をシグナル強い順に復活。",
            )
        with ne_col3:
            if "form_ne_skip_long" not in st.session_state:
                st.session_state["form_ne_skip_long"] = 0
            ne_skip_long = st.number_input(
                "NE≧Nで見送り(L偏り)",
                min_value=0, max_value=20, step=1,
                key="form_ne_skip_long",
                help="ロング数-ショート数≧Nなら見送り。0=制限なし。",
            )
            if "form_ne_skip_short" not in st.session_state:
                st.session_state["form_ne_skip_short"] = 0
            ne_skip_short = st.number_input(
                "NE≦-Nで見送り(S偏り)",
                min_value=0, max_value=20, step=1,
                key="form_ne_skip_short",
                help="ショート数-ロング数≧Nなら見送り。0=制限なし。",
            )

        st.caption("拡張戦略 (HYBRID, K3/K4アンサンブル) は自動的に計算されます")

        submitted = st.form_submit_button("バックテストを実行", type="primary")

    if submitted:
        st.session_state.pop("ll_cleared", None)
        config = LeadLagConfig(
            start_date=str(start_date),
            end_date=str(end_date),
            prior_end_date=str(prior_end),
            us_universe=us_universe,
            jp_universe=jp_universe,
            rolling_window=rolling_window,
            n_components=n_components,
            lambda_reg=lambda_reg,
            quantile_q=quantile_q,
            gap_threshold=gap_threshold / 100.0,
            accumulate_us_returns=accumulate_us,
            net_exposure_limit=net_exposure / 100.0,
            net_exposure_mode=ne_mode,
            net_exposure_skip_long=ne_skip_long,
            net_exposure_skip_short=ne_skip_short,
            run_pca_sub=True,
            run_pca_plain=run_pca_plain,
            run_mom=run_mom,
            run_double=run_double,
        )

        provider, cache = _get_provider_and_cache()

        import time as _time

        progress_dict = {"message": "開始中...", "pct": 0.0, "_started_at": _time.time()}
        st.session_state["ll_progress"] = progress_dict

        thread = threading.Thread(
            target=_run_leadlag_thread,
            args=(progress_dict, provider, cache, config),
            daemon=True,
        )
        thread.start()
        st.session_state["ll_thread"] = thread
        st.rerun()

    # --- ETF一覧表示 ---
    with st.expander("対象ETF一覧"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**米国 S&P 500 セクターETF (11本)**")
            us_df = pd.DataFrame([
                {"ティッカー": t, "セクター": US_TICKER_NAMES.get(t, "")}
                for t in US_TICKERS
            ])
            st.dataframe(us_df, hide_index=True, height=420)
        with col2:
            st.markdown("**日本 TOPIX-17 セクターETF (17本)**")
            jp_df = pd.DataFrame([
                {"コード": t, "セクター": JP_TICKER_NAMES.get(t, "")}
                for t in JP_TICKERS
            ])
            st.dataframe(jp_df, hide_index=True, height=620)


# ---------------------------------------------------------------------------
# タブ2: 結果比較
# ---------------------------------------------------------------------------
def _render_results_tab():
    result = st.session_state.get("ll_result")
    if result is None:
        st.info("「設定・実行」タブでバックテストを実行してください。")
        return

    st.markdown("## 戦略パフォーマンス比較")
    st.caption(f"分析期間: {result.period_start} ~ {result.period_end} ({result.n_common_days} 営業日)")

    strategies = result.strategies
    if not strategies:
        st.warning("戦略結果がありません。")
        return

    # --- 対TOPIX超過リターン計算 (全箇所で使用) ---
    has_bm = result.benchmark_returns is not None and len(result.benchmark_returns) > 0
    excess_metrics = {}
    bm_ar = None
    if has_bm:
        bm_valid = result.benchmark_returns.dropna()
        bm_ar = bm_valid.mean() * 252 * 100 if len(bm_valid) > 0 else None
        for name, strat in strategies.items():
            # NaN（エントリーなし）= 0%リターンとして扱い、全日でTOPIXと比較
            strat_ret_filled = strat.daily_returns.reindex(bm_valid.index).fillna(0)
            if len(strat_ret_filled) > 0:
                from core.lead_lag.strategy import compute_metrics as _cm
                excess_daily = strat_ret_filled - bm_valid
                excess_metrics[name] = _cm(excess_daily.values, bm_valid.index)

    # --- KPI メトリクス (提案手法のハイライト) ---
    if "PCA_SUB" in strategies:
        m = strategies["PCA_SUB"].metrics
        em = excess_metrics.get("PCA_SUB", {})
        st.markdown("**提案手法 (正則化PCA) — 絶対パフォーマンス:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("年率リターン", f"{m['AR']:+.1f}%")
        with col2:
            st.metric("シャープ比", f"{m['R/R']:.2f}")
        with col3:
            st.metric("月次勝率", f"{m.get('monthly_win_rate', 0):.0f}%")
        with col4:
            st.metric("最大下落率", f"{m['MDD']:.1f}%")

        if em:
            st.markdown("**提案手法 — 対TOPIX:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("超過リターン", f"{em.get('AR', 0):+.1f}%")
            with col2:
                st.metric("超過シャープ比", f"{em.get('R/R', 0):.2f}")
            with col3:
                st.metric("対TOPIX月次勝率", f"{em.get('monthly_win_rate', 0):.0f}%")
            with col4:
                st.metric("対TOPIX最悪月", f"{em.get('monthly_worst', 0):+.1f}%")

    # --- ギャップフィルター KPI ---
    if "GAP_CUSTOM" in strategies:
        gm = strategies["GAP_CUSTOM"].metrics
        threshold_pct = int(result.config.gap_threshold * 100)
        st.markdown(f"**ギャップフィルター ({threshold_pct}%消化でスキップ):**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("年率リターン", f"{gm['AR']:+.1f}%")
        with col2:
            st.metric("シャープ比", f"{gm['R/R']:.2f}")
        with col3:
            st.metric("エントリー率", f"{gm.get('entry_rate', 100):.0f}%")
        with col4:
            st.metric("月次勝率", f"{gm.get('monthly_win_rate', 0):.0f}%")

    # --- パフォーマンス比較表 ---
    st.markdown("### 全戦略の比較 — 絶対パフォーマンス")
    rows = []
    show_keys = ["PCA_SUB", "GAP_CUSTOM",
                 "GAP_-10", "GAP_0", "GAP_5", "GAP_10", "GAP_20", "GAP_30",
                 "K3K4_ENS", "K3K4_GAP", "HYBRID",
                 "PCA_PLAIN", "MOM", "DOUBLE"]

    def _make_perf_row(m, label):
        return {
            "戦略": label,
            "年率リターン (%)": round(m["AR"], 2),
            "シャープ比": round(m["R/R"], 2),
            "最大下落率 (%)": round(m["MDD"], 2),
            "エントリー率 (%)": round(m.get("entry_rate", 100), 0),
            "月次勝率 (%)": round(m.get("monthly_win_rate", 0), 0),
            "月次平均 (%)": round(m.get("monthly_mean", 0), 2),
            "月次ブレ幅 (%)": round(m.get("monthly_std", 0), 2),
            "最悪月 (%)": round(m.get("monthly_worst", 0), 1),
        }

    # TOPIX行
    if has_bm and len(bm_valid) > 0:
        from core.lead_lag.strategy import compute_metrics as _cm_bm
        bm_m = _cm_bm(bm_valid.values, bm_valid.index)
        rows.append(_make_perf_row(bm_m, "TOPIX (ベンチマーク)"))
    elif not has_bm:
        st.caption("TOPIX ベンチマークデータが取得できませんでした。")

    gap_pct_display = int(result.config.gap_threshold * 100) if result.config.gap_threshold < 1.0 else None
    for name in show_keys:
        if name in strategies:
            m = strategies[name].metrics
            label = STRATEGY_LABELS.get(name, name)
            if name == "GAP_CUSTOM" and gap_pct_display is not None:
                label = f"GAP_{gap_pct_display}% (NE制限込み)"
            elif name == "K3K4_GAP" and gap_pct_display is not None:
                label = f"K3K4+GAP_{gap_pct_display}% (NE制限込み)"
            rows.append(_make_perf_row(m, label))

    if rows:
        df_perf = pd.DataFrame(rows)
        fmt = {
            "年率リターン (%)": "{:+.1f}",
            "シャープ比": "{:.2f}",
            "最大下落率 (%)": "{:.1f}",
            "エントリー率 (%)": "{:.0f}",
            "月次勝率 (%)": "{:.0f}",
            "月次平均 (%)": "{:+.2f}",
            "月次ブレ幅 (%)": "{:.2f}",
            "最悪月 (%)": "{:+.1f}",
        }
        st.dataframe(
            df_perf.style.apply(_highlight_best, axis=0).format(fmt),
            hide_index=True,
        )

    # --- 対TOPIX比較表 ---
    if excess_metrics:
        st.markdown("### 全戦略の比較 — 対TOPIX (超過リターンベース)")
        st.caption(
            "戦略リターン - TOPIXリターン を日次で計算した超過リターンの指標。"
            "TOPIXを上回った月の割合や、TOPIXに対する最悪月がわかる"
        )
        ex_rows = []
        ex_keys = ["PCA_SUB", "GAP_CUSTOM",
                   "GAP_-10", "GAP_0", "GAP_5", "GAP_10", "GAP_20", "GAP_30",
                   "K3K4_ENS", "K3K4_GAP", "HYBRID",
                   "PCA_PLAIN", "MOM", "DOUBLE"]
        # 不足している戦略の超過メトリクスを計算
        from core.lead_lag.strategy import compute_metrics as _cm
        for name in ex_keys:
            if name in strategies and name not in excess_metrics and has_bm:
                s_ret_filled = strategies[name].daily_returns.reindex(bm_valid.index).fillna(0)
                if len(s_ret_filled) > 0:
                    s_excess = s_ret_filled - bm_valid
                    excess_metrics[name] = _cm(s_excess.values, bm_valid.index)
        for name in ex_keys:
            if name not in excess_metrics:
                continue
            em = excess_metrics[name]
            label = STRATEGY_LABELS.get(name, name)
            if name == "GAP_CUSTOM" and gap_pct_display is not None:
                label = f"GAP_{gap_pct_display}% (NE制限込み)"
            elif name == "K3K4_GAP" and gap_pct_display is not None:
                label = f"K3K4+GAP_{gap_pct_display}% (NE制限込み)"
            ex_rows.append({
                "戦略": label,
                "超過リターン (%)": round(em.get("AR", 0), 1),
                "超過シャープ比": round(em.get("R/R", 0), 2),
                "対TOPIX月次勝率 (%)": round(em.get("monthly_win_rate", 0), 0),
                "超過月次平均 (%)": round(em.get("monthly_mean", 0), 2),
                "超過月次ブレ幅 (%)": round(em.get("monthly_std", 0), 2),
                "対TOPIX最悪月 (%)": round(em.get("monthly_worst", 0), 1),
            })
        if ex_rows:
            df_ex = pd.DataFrame(ex_rows)
            st.dataframe(
                df_ex.style.apply(_highlight_best, axis=0).format({
                    "超過リターン (%)": "{:+.1f}",
                    "超過シャープ比": "{:.2f}",
                    "対TOPIX月次勝率 (%)": "{:.0f}",
                    "超過月次平均 (%)": "{:+.2f}",
                    "超過月次ブレ幅 (%)": "{:.2f}",
                    "対TOPIX最悪月 (%)": "{:+.1f}",
                }),
                hide_index=True,
            )

    # --- 累積リターンチャート ---
    st.markdown("### 累積リターン推移")
    st.caption("1円を投資した場合の資産額の推移")
    fig = go.Figure()

    colors = {
        "PCA_SUB": "#FF8000",
        "GAP_CUSTOM": "#E91E63",
        "PCA_PLAIN": "#888888",
        "MOM": "#1565C0",
        "DOUBLE": "#2E7D32",
    }
    dashes = {
        "PCA_SUB": "solid",
        "GAP_CUSTOM": "solid",
        "PCA_PLAIN": "dot",
        "MOM": "dash",
        "DOUBLE": "dashdot",
    }

    for name in ["PCA_SUB", "GAP_CUSTOM", "DOUBLE", "PCA_PLAIN", "MOM"]:
        if name not in strategies:
            continue
        s = strategies[name]
        ret = s.daily_returns.dropna()
        cum = (1 + ret).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index,
            y=cum.values,
            name=STRATEGY_LABELS.get(name, name),
            line=dict(color=colors.get(name, "#333"), dash=dashes.get(name, "solid"), width=2),
        ))

    # TOPIX ベンチマーク
    if result.benchmark_returns is not None and len(result.benchmark_returns) > 0:
        bm = result.benchmark_returns.dropna()
        bm_cum = (1 + bm).cumprod()
        fig.add_trace(go.Scatter(
            x=bm_cum.index,
            y=bm_cum.values,
            name="TOPIX (ベンチマーク)",
            line=dict(color="#999999", dash="dot", width=1.5),
        ))

    fig.update_layout(
        height=500,
        xaxis_title="日付",
        yaxis_title="累積リターン (1起点)",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, width="stretch")

    # --- 年別リターン ---
    st.markdown("### 年別リターン (%)")
    st.caption("各年の通年リターン。緑=プラス、赤=マイナス。TOPIX列は同期間の指数リターン")
    if "PCA_SUB" in strategies:
        _render_annual_returns(strategies, result.benchmark_returns)

    # --- 月次リターン ---
    st.markdown("### 月次リターン (%)")
    if "PCA_SUB" in strategies:
        _render_monthly_heatmap(strategies, result.benchmark_returns)

    # --- 直近の売買サンプル ---
    st.markdown("### 直近の売買サンプル")
    _render_trade_samples(result)

    # --- L/S エクスポージャー分析 ---
    _render_ls_exposure_analysis(result)

    # --- AI 解釈 ---
    st.markdown("---")
    _render_ai_interpretation(result)


def _highlight_best(s):
    """パフォーマンス表の最良値をハイライト"""
    higher_better = {
        "年率リターン (%)", "シャープ比", "最大下落率 (%)",
        "月次勝率 (%)", "月次平均 (%)", "対TOPIX超過 (%)",
        "超過リターン (%)", "超過シャープ比", "対TOPIX月次勝率 (%)",
        "超過月次平均 (%)",
    }
    lower_better = {"月次ブレ幅 (%)", "超過月次ブレ幅 (%)"}
    max_better = {"最悪月 (%)", "対TOPIX最悪月 (%)"}
    if s.name not in higher_better | lower_better | max_better:
        return [""] * len(s)
    try:
        vals = s.astype(float)
    except (ValueError, TypeError):
        return [""] * len(s)
    if s.name in higher_better:
        best = vals == vals.max()
    elif s.name in lower_better:
        best = vals == vals.min()
    else:
        best = vals == vals.max()
    return ["font-weight: bold; color: #FF8000" if v else "" for v in best]


def _render_annual_returns(strategies, benchmark_returns=None):
    """年別リターンの表を表示 (TOPIX + 対TOPIX超過)"""
    annual_data = {}
    for name, strat in strategies.items():
        ret = strat.daily_returns.dropna()
        if len(ret) == 0:
            continue
        yearly = ret.groupby(ret.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
        annual_data[STRATEGY_LABELS.get(name, name)] = yearly

    # TOPIX + 対TOPIX超過
    bm_yearly = None
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        bm = benchmark_returns.dropna()
        bm_yearly = bm.groupby(bm.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
        annual_data["TOPIX"] = bm_yearly

    if annual_data:
        df_annual = pd.DataFrame(annual_data)
        df_annual.index.name = "年"

        # 対TOPIX超過列を戦略ごとに追加
        if bm_yearly is not None:
            for name, strat in strategies.items():
                label = STRATEGY_LABELS.get(name, name)
                if label in df_annual.columns:
                    excess_col = f"{label} 対TOPIX"
                    df_annual[excess_col] = df_annual[label] - df_annual["TOPIX"]

        # 年平均行を追加
        avg_row = df_annual.mean()
        avg_row.name = "平均"
        df_annual.index = df_annual.index.astype(str)
        df_annual = pd.concat([df_annual, avg_row.to_frame().T])

        st.dataframe(
            df_annual.style.format("{:+.1f}", na_rep="-").map(
                lambda v: "color: #2E7D32" if isinstance(v, (int, float)) and v > 0
                else "color: #C62828" if isinstance(v, (int, float)) and v < 0 else ""
            ),
        )


def _render_trade_samples(result, n_days: int = 10):
    """直近n日の売買詳細を表示する。"""
    strategies = result.strategies
    if "PCA_SUB" not in strategies or strategies["PCA_SUB"].signals is None:
        return
    if result.us_cc_returns is None:
        return

    sig_df = strategies["PCA_SUB"].signals.dropna(how="all")
    us_ret = result.us_cc_returns
    gap_df = result.jp_overnight_gaps
    close_df = result.jp_close_prices
    open_df = result.jp_open_prices
    if len(sig_df) == 0:
        return

    last_dates = sig_df.index[-n_days:]
    config = result.config
    q = config.quantile_q
    has_gap = config.gap_threshold > 0 and gap_df is not None

    gap_pct = int(config.gap_threshold * 100)
    st.caption(
        f"分析期間の最後{n_days}日間の売買判断。"
        + (f"ギャップ閾値: {gap_pct}% (シグナルの{gap_pct}%以上がギャップで消化→スキップ)" if has_gap else "ギャップフィルター: なし")
    )

    for date in last_dates:
        sig = sig_df.loc[date]
        n_long = max(1, int(np.ceil(len(sig.dropna()) * q)))

        us_dates_before = us_ret.index[us_ret.index <= date]
        us_day_ret = us_ret.loc[us_dates_before[-1]] if len(us_dates_before) > 0 else None

        # ギャップデータ
        gap_row = None
        if gap_df is not None and date in gap_df.index:
            gap_row = gap_df.loc[date]

        sorted_sig = sig.dropna().sort_values(ascending=False)

        # 当日のリターンサマリー
        ret_summary = ""
        gap_pct_val_int = int(config.gap_threshold * 100) if config.gap_threshold < 1.0 else None
        # フィルターなし → GAP(NE無) → GAP(NE込) の順
        summary_strats = [("PCA_SUB", "フィルターなし")]
        for gk in ["GAP_-10", "GAP_0", "GAP_5", "GAP_10", "GAP_20", "GAP_30"]:
            gr = {"GAP_-10": -10, "GAP_0": 0, "GAP_5": 5, "GAP_10": 10, "GAP_20": 20, "GAP_30": 30}.get(gk)
            if gk in strategies and gr == gap_pct_val_int:
                summary_strats.append((gk, f"GAP_{gr}%(NE無)"))
        if "GAP_CUSTOM" in strategies:
            summary_strats.append(("GAP_CUSTOM", f"GAP_{gap_pct}%(NE込)"))
        for strat_name, label in summary_strats:
            if strat_name in strategies:
                ret = strategies[strat_name].daily_returns
                if date in ret.index and not np.isnan(ret.loc[date]):
                    ret_summary += f" | {label}: {ret.loc[date]*100:+.2f}%"
                else:
                    ret_summary += f" | {label}: 売買なし"

        with st.expander(f"{date.strftime('%Y-%m-%d')}{ret_summary}"):
            # 米国ETF騰落率
            if us_day_ret is not None:
                st.markdown("**前日の米国セクター騰落率:**")
                us_rows = []
                for t in us_day_ret.index:
                    us_rows.append({
                        "セクター": US_TICKER_NAMES.get(t, t),
                        "騰落率 (%)": round(us_day_ret[t] * 100, 2),
                    })
                us_display = pd.DataFrame(us_rows).sort_values("騰落率 (%)", ascending=False)
                st.dataframe(
                    us_display.style.format({"騰落率 (%)": "{:+.2f}"}).map(
                        lambda v: "color: #2E7D32" if isinstance(v, (int, float)) and v > 0
                        else "color: #C62828" if isinstance(v, (int, float)) and v < 0 else "",
                        subset=["騰落率 (%)"],
                    ),
                    hide_index=True, height=200,
                )

            # 日本ETF売買判断
            st.markdown("**日本セクターETF — シグナル・ギャップ・売買判断:**")
            trade_rows = []
            for rank, (ticker, sig_val) in enumerate(sorted_sig.items()):
                sector = JP_TICKER_NAMES.get(ticker, ticker)

                # ロング/ショート判定 (フィルターなし)
                if rank < n_long:
                    base_position = "ロング"
                elif rank >= len(sorted_sig) - n_long:
                    base_position = "ショート"
                else:
                    base_position = "-"

                # 価格情報
                prev_close = None
                today_open = None
                today_close = None
                # 前日終値
                if close_df is not None and ticker in close_df.columns:
                    date_idx = close_df.index.get_loc(date) if date in close_df.index else None
                    if date_idx is not None and date_idx > 0:
                        prev_close = close_df.iloc[date_idx - 1][ticker]
                # 当日始値・終値
                if open_df is not None and date in open_df.index and ticker in open_df.columns:
                    today_open = open_df.loc[date, ticker]
                if close_df is not None and date in close_df.index and ticker in close_df.columns:
                    today_close = close_df.loc[date, ticker]

                # 各値を安全に取得
                pc = int(prev_close) if prev_close is not None and not np.isnan(prev_close) else None
                op = int(today_open) if today_open is not None and not np.isnan(today_open) else None
                cl = int(today_close) if today_close is not None and not np.isnan(today_close) else None
                oc_ret = round((today_close / today_open - 1) * 100, 2) if op and cl and today_open > 0 else None

                row = {
                    "セクター": sector,
                    "判定": base_position,
                    "シグナル": round(sig_val, 4),
                    "前日終値": pc,
                }

                # ギャップフィルター列 (閾値設定時のみ)
                if has_gap:
                    gap_val = gap_row[ticker] if gap_row is not None and ticker in gap_row.index else np.nan
                    gap_pct_val = round(gap_val * 100, 2) if not np.isnan(gap_val) else None

                    # 指値目安 (ロング=この価格以下で買い、ショート=この価格以上で売り)
                    limit_price = None
                    if pc and abs(sig_val) > 1e-10 and base_position in ("ロング", "ショート"):
                        if base_position == "ロング":
                            limit_price = int(pc * (1 + abs(sig_val) * config.gap_threshold))
                        else:
                            limit_price = int(pc * (1 - abs(sig_val) * config.gap_threshold))

                    # 消化率・判定
                    absorbed = 0
                    can_trade = True
                    if not np.isnan(gap_val) and abs(sig_val) > 1e-10 and base_position in ("ロング", "ショート"):
                        if base_position == "ロング":
                            absorbed = max(0, gap_val / abs(sig_val) * 100) if gap_val > 0 else 0
                            can_trade = gap_val <= abs(sig_val) * config.gap_threshold
                        else:
                            absorbed = max(0, -gap_val / abs(sig_val) * 100) if gap_val < 0 else 0
                            can_trade = gap_val >= -abs(sig_val) * config.gap_threshold

                    limit_label = "以下で買い" if base_position == "ロング" else "以上で売り" if base_position == "ショート" else ""
                    row["指値目安"] = f"{limit_price:,}{limit_label}" if limit_price else "-"
                    row["寄付き"] = f"{op:,}" if op else "-"
                    row["前日比(%)"] = gap_pct_val
                    row["織込済(%)"] = str(round(absorbed)) if base_position in ("ロング", "ショート") else "-"
                    row["GAP判定"] = ("エントリー" if can_trade else "スキップ") if base_position in ("ロング", "ショート") else "-"
                    row["終値"] = f"{cl:,}" if cl else "-"
                    row["当日騰落(%)"] = oc_ret
                else:
                    row["寄付き"] = f"{op:,}" if op else "-"
                    row["終値"] = f"{cl:,}" if cl else "-"
                    row["当日騰落(%)"] = oc_ret

                trade_rows.append(row)

            trade_df = pd.DataFrame(trade_rows)

            # スタイリング
            style_cols = ["判定"]
            if "GAP判定" in trade_df.columns:
                style_cols.append("GAP判定")

            def _style_trade(v):
                s = str(v)
                if "ロング" in s or "エントリー" in s:
                    return "color: #2E7D32; font-weight: bold"
                if "ショート" in s:
                    return "color: #C62828; font-weight: bold"
                if "スキップ" in s:
                    return "color: #999; text-decoration: line-through"
                return ""

            fmt = {"シグナル": "{:+.2f}"}
            if "当日騰落(%)" in trade_df.columns:
                fmt["当日騰落(%)"] = "{:+.1f}"
            if "前日比(%)" in trade_df.columns:
                fmt["前日比(%)"] = "{:+.1f}"

            st.dataframe(
                trade_df.style.map(_style_trade, subset=style_cols).format(fmt, na_rep="-"),
                hide_index=True,
            )


def _calc_monthly_returns(daily_returns: pd.Series) -> pd.DataFrame:
    """日次リターンから年×月のヒートマップ用DataFrameを作成する。"""
    ret = daily_returns.dropna()
    if len(ret) == 0:
        return pd.DataFrame()
    monthly = ret.groupby([ret.index.year, ret.index.month]).apply(
        lambda x: ((1 + x).prod() - 1) * 100
    )
    monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=["年", "月"])
    pivot = monthly.unstack(level="月")
    pivot.columns = [f"{m}月" for m in pivot.columns]
    # 年間合計列を追加
    annual = ret.groupby(ret.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
    pivot["年間"] = annual
    return pivot


def _render_monthly_heatmap(strategies, benchmark_returns=None):
    """月次リターンのヒートマップ + テーブルを表示する。"""
    # 戦略選択 (TOPIX, 対TOPIX も追加)
    strat_names = [n for n in ["PCA_SUB", "GAP_CUSTOM", "PCA_PLAIN", "MOM", "DOUBLE"] if n in strategies]
    choices = strat_names.copy()
    has_bm = benchmark_returns is not None and len(benchmark_returns) > 0
    if has_bm:
        choices.append("TOPIX")
        for n in strat_names:
            choices.append(f"{n}_vs_TOPIX")
    if not choices:
        return

    def _format_choice(x):
        if x == "TOPIX":
            return "TOPIX (ベンチマーク)"
        if x.endswith("_vs_TOPIX"):
            base = x.replace("_vs_TOPIX", "")
            return f"{STRATEGY_LABELS.get(base, base)} 対TOPIX超過"
        return STRATEGY_LABELS.get(x, x)

    selected = st.selectbox(
        "表示する戦略",
        choices,
        format_func=_format_choice,
        key="monthly_strat_select",
    )

    if selected == "TOPIX":
        ret = benchmark_returns
    elif selected.endswith("_vs_TOPIX"):
        base = selected.replace("_vs_TOPIX", "")
        strat_ret = strategies[base].daily_returns.dropna()
        bm_ret = benchmark_returns.dropna()
        common_idx = strat_ret.index.intersection(bm_ret.index)
        ret = strat_ret.loc[common_idx] - bm_ret.loc[common_idx]
    else:
        ret = strategies[selected].daily_returns
    pivot = _calc_monthly_returns(ret)
    if pivot.empty:
        return

    # --- ヒートマップ (Plotly) ---
    month_cols = [c for c in pivot.columns if c != "年間"]
    z_data = pivot[month_cols].values

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=[c for c in month_cols],
        y=[str(y) for y in pivot.index],
        colorscale=[
            [0.0, "#C62828"],
            [0.5, "#FFFFFF"],
            [1.0, "#2E7D32"],
        ],
        zmid=0,
        text=[[f"{v:+.1f}%" if not np.isnan(v) else "" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="リターン(%)"),
        hovertemplate="年: %{y}<br>月: %{x}<br>リターン: %{z:+.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(250, len(pivot) * 35 + 80),
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=50, r=20, t=10, b=40),
    )
    st.plotly_chart(fig, width="stretch")

    # --- テーブル (年間列付き) ---
    with st.expander("月次リターン テーブル"):
        st.dataframe(
            pivot.style.format("{:+.1f}", na_rep="-").map(
                lambda v: "color: #2E7D32; font-weight: 600" if isinstance(v, (int, float)) and v > 0
                else "color: #C62828; font-weight: 600" if isinstance(v, (int, float)) and v < 0
                else ""
            ),
        )

    # --- 月別の平均・勝率 ---
    monthly_ret = ret.dropna()
    if len(monthly_ret) > 0:
        by_month = monthly_ret.groupby(monthly_ret.index.month)
        stats_rows = []
        for m in range(1, 13):
            if m not in by_month.groups:
                continue
            group = by_month.get_group(m)
            monthly_agg = group.groupby([group.index.year, group.index.month]).apply(
                lambda x: ((1 + x).prod() - 1) * 100
            )
            stats_rows.append({
                "月": f"{m}月",
                "平均リターン (%)": round(monthly_agg.mean(), 1),
                "勝率 (%)": round((monthly_agg > 0).mean() * 100, 0),
                "最大 (%)": round(monthly_agg.max(), 1),
                "最小 (%)": round(monthly_agg.min(), 1),
                "回数": len(monthly_agg),
            })
        if stats_rows:
            st.markdown("**月別の傾向 (全期間平均)**")
            st.caption("勝率 = その月がプラスだった年の割合")
            df_stats = pd.DataFrame(stats_rows)
            st.dataframe(
                df_stats.style.format({
                    "平均リターン (%)": "{:+.1f}",
                    "勝率 (%)": "{:.0f}",
                    "最大 (%)": "{:+.1f}",
                    "最小 (%)": "{:+.1f}",
                }).map(
                    lambda v: "color: #2E7D32; font-weight: 600" if isinstance(v, (int, float)) and v > 0
                    else "color: #C62828; font-weight: 600" if isinstance(v, (int, float)) and v < 0
                    else "",
                    subset=["平均リターン (%)"],
                ),
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# AI 解釈
# ---------------------------------------------------------------------------
def _build_interpretation_prompt(result) -> str:
    """バックテスト結果からAI解釈用プロンプトを構築する。"""
    strategies = result.strategies
    config = result.config

    # --- パラメータ説明 ---
    param_parts = [
        f"L={config.rolling_window}, λ={config.lambda_reg}, K={config.n_components}, q={config.quantile_q}",
    ]
    if config.gap_threshold < 1.0:
        param_parts.append(f"GAPフィルター閾値={int(config.gap_threshold*100)}%")
    if config.net_exposure_limit < 1.0:
        ne_mode = "復活" if config.net_exposure_mode == "fill" else "削減"
        param_parts.append(f"NE上限={int(config.net_exposure_limit*100)}%({ne_mode})")
    if config.net_exposure_skip_long > 0 or config.net_exposure_skip_short > 0:
        skip_p = []
        if config.net_exposure_skip_long > 0:
            skip_p.append(f"L偏り≧{config.net_exposure_skip_long}")
        if config.net_exposure_skip_short > 0:
            skip_p.append(f"S偏り≧{config.net_exposure_skip_short}")
        param_parts.append(f"NE{'/'.join(skip_p)}スキップ")
    if config.accumulate_us_returns:
        param_parts.append("US休場累積あり")
    param_summary = ", ".join(param_parts)

    # --- 全戦略パフォーマンス表 ---
    perf_lines = [
        "| 戦略 | 年率R(%) | リスク(%) | SR | MDD(%) | エントリー率(%) | 月次勝率(%) | 月次平均(%) | 月次ブレ幅(%) | 最悪月(%) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    strat_order = ["PCA_SUB", "GAP_CUSTOM", "PCA_PLAIN", "MOM", "DOUBLE",
                   "HYBRID", "K3K4_ENS", "K3K4_GAP",
                   "GAP_-10", "GAP_0", "GAP_5", "GAP_10", "GAP_20", "GAP_30"]
    for name in strat_order:
        if name not in strategies:
            continue
        m = strategies[name].metrics
        label = STRATEGY_LABELS.get(name, name)
        if name == "GAP_CUSTOM":
            label = f"GAP_{int(config.gap_threshold*100)}% (設定値)"
        perf_lines.append(
            f"| {label} | {m['AR']:.1f} | {m['RISK']:.1f} | {m['R/R']:.2f} | {m['MDD']:.1f} "
            f"| {m.get('entry_rate', 100):.0f} "
            f"| {m.get('monthly_win_rate', 0):.0f} | {m.get('monthly_mean', 0):+.2f} "
            f"| {m.get('monthly_std', 0):.2f} | {m.get('monthly_worst', 0):+.1f} |"
        )
    perf_table = "\n".join(perf_lines)

    # --- メイン戦略の選択 (GAP_CUSTOM優先) ---
    main_key = "GAP_CUSTOM" if "GAP_CUSTOM" in strategies else "PCA_SUB"

    # --- 年別リターン (主要戦略) ---
    annual_lines = []
    for name in strat_order:
        if name not in strategies:
            continue
        ret = strategies[name].daily_returns.dropna()
        if len(ret) == 0:
            continue
        label = STRATEGY_LABELS.get(name, name)
        if name == "GAP_CUSTOM":
            label = f"GAP_{int(config.gap_threshold*100)}%"
        yearly = ret.groupby(ret.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
        for year, val in yearly.items():
            annual_lines.append(f"- {label} {year}年: {val:+.1f}%")
    annual_text = "\n".join(annual_lines) if annual_lines else "(データなし)"

    # --- L/Sエクスポージャー分析 (NE見送り日を含む) ---
    ls_text = "(データなし)"
    ls_strat = strategies.get("GAP_CUSTOM") or strategies.get("GAP_5")
    # NE見送り日の仮想リターン用にGAP感応度テストを取得
    gap_pct_int = int(config.gap_threshold * 100) if config.gap_threshold < 1.0 else None
    gap_ne_free_key = None
    for gk, gr in [("GAP_-10", -10), ("GAP_0", 0), ("GAP_5", 5), ("GAP_10", 10), ("GAP_20", 20), ("GAP_30", 30)]:
        if gr == gap_pct_int and gk in strategies:
            gap_ne_free_key = gk
            break

    if ls_strat and ls_strat.daily_long_count is not None and ls_strat.daily_short_count is not None:
        lc = ls_strat.daily_long_count.dropna()
        sc = ls_strat.daily_short_count.dropna()
        ret_s = ls_strat.daily_returns
        common = lc.index.intersection(sc.index)
        if len(common) > 0:
            lc = lc.loc[common].astype(int)
            sc = sc.loc[common].astype(int)
            net = lc - sc
            ret_aligned = ret_s.reindex(common)
            ls_lines = [
                "| ネット(L-S) | エントリー日数 | 勝率(%) | 平均R(%) | NE見送り | 見送り仮想R(%) | 見送り仮想勝率(%) |",
                "|---|---|---|---|---|---|---|",
            ]
            analysis = pd.DataFrame({"ネット": net, "リターン": ret_aligned, "売買": ~ret_aligned.isna()})
            for net_val, group in sorted(analysis.groupby("ネット"), key=lambda x: x[0]):
                traded = group[group["売買"]]
                skipped = group[~group["売買"]]
                n_traded = len(traded)
                n_skipped = len(skipped)
                win_rate = (traded["リターン"] > 0).sum() / n_traded * 100 if n_traded > 0 else 0
                avg_ret = traded["リターン"].mean() * 100 if n_traded > 0 else 0
                virt_r = "-"
                virt_wr = "-"
                if n_skipped > 0 and gap_ne_free_key:
                    virt_ret = strategies[gap_ne_free_key].daily_returns.reindex(skipped.index).dropna()
                    if len(virt_ret) > 0:
                        virt_r = f"{virt_ret.mean() * 100:+.3f}"
                        virt_wr = f"{(virt_ret > 0).sum() / len(virt_ret) * 100:.1f}"
                ls_lines.append(
                    f"| {int(net_val)} | {n_traded} | {win_rate:.1f} | {avg_ret:+.3f} "
                    f"| {n_skipped} | {virt_r} | {virt_wr} |"
                )
            ls_text = "\n".join(ls_lines)

    # --- 直近シグナル ---
    signal_text = "(データなし)"
    if "PCA_SUB" in strategies and strategies["PCA_SUB"].signals is not None:
        sig = strategies["PCA_SUB"].signals.dropna(how="all")
        if len(sig) > 0:
            latest = sig.iloc[-1].sort_values(ascending=False)
            sig_lines = [f"日付: {sig.index[-1].strftime('%Y-%m-%d')}"]
            for ticker, val in latest.items():
                sector = JP_TICKER_NAMES.get(ticker, ticker)
                direction = "ロング (買い)" if val > 0 else "ショート (売り)"
                sig_lines.append(f"- {sector} ({ticker}): シグナル={val:+.4f} -> {direction}")
            signal_text = "\n".join(sig_lines)

    # --- TOPIX ---
    topix_annual_text = "(データなし)"
    excess_text = "(データなし)"
    if result.benchmark_returns is not None and len(result.benchmark_returns) > 0:
        bm = result.benchmark_returns.dropna()
        bm_yearly = bm.groupby(bm.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
        topix_annual_text = "\n".join([f"- TOPIX {y}年: {v:+.1f}%" for y, v in bm_yearly.items()])

        # 対TOPIX超過 (全戦略)
        from core.lead_lag.strategy import compute_metrics as _cm
        ex_lines = [
            "| 戦略 | 超過年率R(%) | 超過SR | 対TOPIX月次勝率(%) |",
            "|---|---|---|---|",
        ]
        for name in strat_order:
            if name not in strategies:
                continue
            # NaN=0%として全日でTOPIXと比較
            strat_ret_filled = strategies[name].daily_returns.reindex(bm.index).fillna(0)
            excess_daily = strat_ret_filled - bm
            em = _cm(excess_daily.values, bm.index)
            label = STRATEGY_LABELS.get(name, name)
            if name == "GAP_CUSTOM":
                label = f"GAP_{int(config.gap_threshold*100)}%(NE込)"
            ex_lines.append(
                f"| {label} | {em['AR']:+.1f} | {em['R/R']:.2f} | {em.get('monthly_win_rate',0):.0f} |"
            )
        # メイン戦略の年別超過
        if main_key in strategies:
            strat_ret_filled = strategies[main_key].daily_returns.reindex(bm.index).fillna(0)
            excess_daily = strat_ret_filled - bm
            if len(excess_daily) > 0:
                excess_yearly = excess_daily.groupby(excess_daily.index.year).apply(
                    lambda x: ((1 + x).prod() - 1) * 100
                )
                main_label = f"GAP_{int(config.gap_threshold*100)}%" if main_key == "GAP_CUSTOM" else "PCA_SUB"
                ex_lines.append(f"\n{main_label} 年別 対TOPIX超過:")
                for y, v in excess_yearly.items():
                    bm_y = bm_yearly.get(y, 0)
                    ex_lines.append(f"  {y}年: 戦略{v+bm_y:+.1f}% - TOPIX{bm_y:+.1f}% = 超過{v:+.1f}%")
        excess_text = "\n".join(ex_lines)

    prompt = f"""あなたは定量投資戦略の専門アナリストです。
以下の日米セクターETFリードラグ戦略のバックテスト結果を分析してください。

## 戦略概要
- 手法: 部分空間正則化PCA (中川ら, SIG-FIN-036, 2026)
- 仮説: 米国セクターETFの当日リターンが、日本セクターETFの翌日寄引リターンを予測する
- 設定: {param_summary}
- 期間: {result.period_start} ~ {result.period_end} ({result.n_common_days}営業日)
- 米国側: S&P 500 セクターETF {len(result.us_tickers)}本
- 日本側: TOPIX-17 セクターETF {len(result.jp_tickers)}本

## パラメータの意味
- **L**: ローリングウィンドウ長。長いほど安定。
- **λ**: 正則化強度。1に近いほど事前知識重視。
- **K**: 主成分数。日米共通パターン数。
- **q**: 売買比率。0.5=全セクター。
- **GAPフィルター閾値**: 値が小さいほど厳しい。-10%=最厳格(逆方向10%以上動いた銘柄のみ)。0%=シグナル方向のギャップは全てスキップ。5%=5%まで許容。100%=フィルターなし。
- **NE上限**: GAPフィルター後のL/S偏りを制限。復活=少ない側にGAPで外された銘柄を復活。削減=多い側を削除。
- **|NE|≧Nスキップ**: |ロング数-ショート数|がN以上の日は全銘柄見送り。

## 戦略間の関係 (重要: 正しく理解すること)
- **正則化PCA (PCA_SUB)**: ベースとなるシグナル生成。GAPフィルターなしで全日エントリー。
- **GAP_X% (GAP_-10, GAP_0, GAP_5, GAP_10, GAP_20, GAP_30)**: すべて**正則化PCAのシグナルを使用**し、GAPフィルターを適用したもの。通常PCAのシグナルではない。つまりGAP_5%は「正則化PCA + GAPフィルター5%」であり、両者は既に組み合わさっている。
- **GAP_CUSTOM**: ユーザーが手動設定した閾値でのGAPフィルター。シグナルは正則化PCA。
- **K3K4_ENS**: K=3とK=4のアンサンブル。GAPフィルターなし。
- **K3K4_GAP**: K3K4_ENSのシグナルにGAP_50%を適用。
- **通常PCA (PCA_PLAIN)**: 正則化なし(λ=0)のPCA。比較用ベースライン。
- **HYBRID**: VIX/円高でポジション縮小する正則化PCA。

## 全戦略パフォーマンス比較
{perf_table}

## TOPIX 年別リターン
{topix_annual_text}

## 対TOPIX超過パフォーマンス
{excess_text}

## 年別リターン (各戦略)
{annual_text}

## L/Sエクスポージャー分析 (ネットポジション別の日数・勝率・平均リターン)
{ls_text}

## 直近シグナル
{signal_text}

以下の観点で分析してください:

1. **GAPフィルター分析**: 閾値別のSR・エントリー率・MDD・年別リターンの比較。最適閾値の推定。GAP_CUSTOMはNE制限込みの実運用設定、GAP_X%はNE制限なしの純粋GAPフィルター。両者の差がNE制限の効果。

2. **L/Sエクスポージャー分析**: ネット別のエントリー日数・勝率・平均リターンとNE見送り日数・仮想リターンから、NE制限のロング偏り/ショート偏り各方向の最適閾値を分析。見送り日の仮想リターンがプラスならNE制限が過剰、マイナスなら適切。

3. **対TOPIX分析**: 年別の対TOPIX超過の安定性。

4. **改善提案 (最重要)**: 提供データの中でSRが最も高い戦略を起点とし、そのSRをさらに向上させるための具体的な改善案を提示すること。SR最良の戦略がなぜ高SRを達成しているのかを分析し、その強みを維持・強化しつつ弱点を補う方向で提案すること。特にNE制限の方向別閾値の最適化、GAPフィルター閾値の微調整について具体的に提案すること。"""

    return prompt


def _run_interpretation_thread(progress_dict: dict, prompt: str):
    """バックグラウンドでClaude CLIを呼び出す。"""
    try:
        from core.ai_client import create_ai_client

        progress_dict["message"] = "Claude に問い合わせ中..."
        progress_dict["pct"] = 0.3
        client = create_ai_client()
        response = client.send_message(prompt, system_prompt="")
        progress_dict["_result"] = response
        progress_dict["pct"] = 1.0
        progress_dict["message"] = "完了"
    except Exception as e:
        import traceback

        progress_dict["error"] = str(e)
        progress_dict["detail"] = traceback.format_exc()
        progress_dict["pct"] = 1.0


def _render_ls_exposure_analysis(result):
    """L/Sエクスポージャー分析: ネットポジション別の日数・勝率・平均リターン。"""
    # daily_long_count を持つ GAP 系戦略を探す
    gap_strats = {}
    for key, strat in result.strategies.items():
        if strat.daily_long_count is not None and strat.daily_short_count is not None:
            gap_strats[key] = strat

    if not gap_strats:
        return

    st.markdown("### L/S エクスポージャー分析")

    # 対象戦略を選択
    target_key = None
    if "GAP_CUSTOM" in gap_strats:
        target_key = "GAP_CUSTOM"
    else:
        # 最初のGAP戦略
        target_key = next(iter(gap_strats))

    strat_options = {k: STRATEGY_LABELS.get(k, k) for k in gap_strats}
    if len(strat_options) > 1:
        target_key = st.selectbox(
            "分析対象の戦略",
            options=list(strat_options.keys()),
            format_func=lambda k: strat_options[k],
            index=list(strat_options.keys()).index(target_key) if target_key in strat_options else 0,
            key="ls_analysis_target",
        )

    strat = gap_strats[target_key]
    lc = strat.daily_long_count.dropna()
    sc = strat.daily_short_count.dropna()
    ret = strat.daily_returns

    # L/S数がある日全てを対象（NEスキップ日はリターンがNaN）
    common = lc.index.intersection(sc.index)
    if len(common) == 0:
        st.info("エクスポージャーデータがありません。")
        return

    lc = lc.loc[common].astype(int)
    sc = sc.loc[common].astype(int)
    net = lc - sc
    ret_aligned = ret.reindex(common)  # NaN = NEスキップで売買しなかった日

    # --- ネットエクスポージャー別集計 ---
    analysis_df = pd.DataFrame({
        "ロング数": lc,
        "ショート数": sc,
        "ネット": net,
        "リターン": ret_aligned,
        "売買": ~ret_aligned.isna(),
    })

    # ネット別集計（エントリー日とNEスキップ日を分けて表示）
    net_groups = analysis_df.groupby("ネット")
    net_summary_rows = []
    for net_val, group in sorted(net_groups, key=lambda x: x[0]):
        traded = group[group["売買"]]
        skipped = group[~group["売買"]]
        n_traded = len(traded)
        n_skipped = len(skipped)
        n_total = n_traded + n_skipped
        wins = (traded["リターン"] > 0).sum() if n_traded > 0 else 0
        win_rate = wins / n_traded * 100 if n_traded > 0 else 0
        avg_ret = traded["リターン"].mean() * 100 if n_traded > 0 else 0
        row = {
            "ネット": int(net_val),
            "エントリー": n_traded,
            "平均リターン (%)": f"{avg_ret:+.3f}" if n_traded > 0 else "-",
            "勝率 (%)": f"{win_rate:.1f}" if n_traded > 0 else "-",
        }
        if n_skipped > 0:
            row["NE見送り"] = n_skipped
            # NE見送り日の仮想リターン（GAPのみ戦略から取得）
            gap_key_for_virtual = None
            gap_pct_val = int(result.config.gap_threshold * 100) if result.config.gap_threshold < 1.0 else None
            for gk, gr in [("GAP_-10", -10), ("GAP_0", 0), ("GAP_5", 5), ("GAP_10", 10), ("GAP_20", 20), ("GAP_30", 30)]:
                if gr == gap_pct_val and gk in result.strategies:
                    gap_key_for_virtual = gk
                    break
            if gap_key_for_virtual:
                virt_ret = result.strategies[gap_key_for_virtual].daily_returns
                skipped_rets = virt_ret.reindex(skipped.index).dropna()
                if len(skipped_rets) > 0:
                    virt_avg = skipped_rets.mean() * 100
                    virt_wins = (skipped_rets > 0).sum()
                    row["見送り仮想R (%)"] = f"{virt_avg:+.3f}"
                    row["見送り仮想勝率 (%)"] = f"{virt_wins / len(skipped_rets) * 100:.1f}"
        else:
            row["NE見送り"] = 0
        net_summary_rows.append(row)
    net_summary = pd.DataFrame(net_summary_rows)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ネットエクスポージャー別**")
        fmt_cols = {}
        for c in ["平均リターン (%)", "勝率 (%)", "見送り仮想R (%)", "見送り仮想勝率 (%)"]:
            if c in net_summary.columns:
                fmt_cols[c] = lambda v: f"{v:+.3f}" if isinstance(v, (int, float)) else str(v)
        st.dataframe(
            net_summary,
            hide_index=True,
            height=min(500, len(net_summary) * 35 + 40),
        )

    with col2:
        # ヒストグラム: エントリー日数(青) + NE見送り日数(赤) + 平均リターン(折線)
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=net_summary["ネット"],
                y=net_summary["エントリー"],
                name="エントリー日数",
                marker_color="#1565C0",
                opacity=0.7,
            ),
            secondary_y=False,
        )

        if "NE見送り" in net_summary.columns and net_summary["NE見送り"].sum() > 0:
            fig.add_trace(
                go.Bar(
                    x=net_summary["ネット"],
                    y=net_summary["NE見送り"],
                    name="NE見送り日数",
                    marker_color="#C62828",
                    opacity=0.5,
                ),
                secondary_y=False,
            )
            fig.update_layout(barmode="stack")

        # 文字列→数値変換ヘルパー
        def _to_float(v):
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        # 平均リターン（エントリー日のみ）
        ret_vals = [_to_float(v) for v in net_summary["平均リターン (%)"]]
        fig.add_trace(
            go.Scatter(
                x=net_summary["ネット"],
                y=ret_vals,
                name="平均リターン (%)",
                mode="lines+markers",
                line=dict(color="#FF8000", width=2),
                marker=dict(size=8),
            ),
            secondary_y=True,
        )

        # 見送り仮想リターン（ある場合）
        if "見送り仮想R (%)" in net_summary.columns:
            virt_vals = [_to_float(v) for v in net_summary["見送り仮想R (%)"]]
            if any(v is not None for v in virt_vals):
                fig.add_trace(
                    go.Scatter(
                        x=net_summary["ネット"],
                        y=virt_vals,
                        name="見送り仮想R (%)",
                        mode="markers",
                        marker=dict(size=10, color="#C62828", symbol="x"),
                    ),
                    secondary_y=True,
                )

        fig.update_layout(
            height=400,
            xaxis_title="ネットエクスポージャー (L-S)",
            template="plotly_white",
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=60, r=60, t=30, b=40),
        )
        fig.update_yaxes(title_text="日数", secondary_y=False)
        fig.update_yaxes(title_text="平均リターン (%)", secondary_y=True)
        st.plotly_chart(fig, width="stretch")

    # --- L数・S数別の詳細 ---
    with st.expander("ロング数・ショート数の詳細分布"):
        ls_groups = analysis_df.groupby(["ロング数", "ショート数"])
        ls_rows = []
        for (l_val, s_val), group in ls_groups:
            n_days = len(group)
            wins = (group["リターン"] > 0).sum()
            win_rate = wins / n_days * 100 if n_days > 0 else 0
            avg_ret = group["リターン"].mean() * 100
            ls_rows.append({
                "ロング数": int(l_val),
                "ショート数": int(s_val),
                "ネット": int(l_val - s_val),
                "日数": n_days,
                "勝率 (%)": round(win_rate, 1),
                "平均リターン (%)": round(avg_ret, 3),
            })
        ls_detail = pd.DataFrame(ls_rows).sort_values(["ネット", "ロング数"], ascending=[True, True])
        st.dataframe(
            ls_detail.style.format({
                "勝率 (%)": "{:.1f}",
                "平均リターン (%)": "{:+.3f}",
            }),
            hide_index=True,
            height=min(600, len(ls_detail) * 35 + 40),
        )


def _render_ai_interpretation(result):
    """AI解釈セクションのレンダリング。"""
    st.markdown("### AI による結果分析")

    # 実行中チェック
    interp_thread = st.session_state.get("ll_interp_thread")
    is_running = interp_thread is not None and interp_thread.is_alive()

    if is_running:
        prog = st.session_state.get("ll_interp_progress", {})
        msg = prog.get("message", "解釈中...")
        pct = prog.get("pct", 0)
        st.warning("AI が結果を分析中です... (数十秒かかります)")
        st.progress(pct, text=msg)

        import time
        time.sleep(2)
        st.rerun()
        return

    # スレッド完了後
    if interp_thread is not None and not interp_thread.is_alive():
        prog = st.session_state.get("ll_interp_progress", {})
        if "_result" in prog and "ll_interpretation" not in st.session_state:
            st.session_state["ll_interpretation"] = prog["_result"]
            st.session_state.pop("ll_interp_thread", None)
            st.session_state.pop("ll_interp_progress", None)
            st.rerun()
            return
        elif prog.get("error"):
            st.error(f"AI解釈エラー: {prog['error']}")
            detail = prog.get("detail", "")
            if detail:
                with st.expander("詳細"):
                    st.code(detail)
            if st.button("OK", key="interp_error_ok"):
                st.session_state.pop("ll_interp_thread", None)
                st.session_state.pop("ll_interp_progress", None)
                st.rerun()
            return

    # 既存の解釈結果を表示
    if "ll_interpretation" in st.session_state:
        st.markdown(st.session_state["ll_interpretation"])
        col1, col2 = st.columns(2)
        with col1:
            if st.button("再分析する", key="reinterpret"):
                st.session_state.pop("ll_interpretation", None)
                st.rerun()
        with col2:
            with st.expander("テキストをコピー"):
                st.text_area("", st.session_state["ll_interpretation"], height=400, key="copy_interpretation")
        return

    # 実行ボタン
    st.caption("バックテスト結果をClaude に渡して、戦略の有効性・リスク・直近シグナルの意味を分析します。")
    if st.button("AI で結果を分析する", type="primary", key="run_interpretation"):
        prompt = _build_interpretation_prompt(result)
        progress_dict = {"message": "開始中...", "pct": 0.0}
        st.session_state["ll_interp_progress"] = progress_dict

        import threading as _threading

        thread = _threading.Thread(
            target=_run_interpretation_thread,
            args=(progress_dict, prompt),
            daemon=True,
        )
        thread.start()
        st.session_state["ll_interp_thread"] = thread
        st.rerun()


# ---------------------------------------------------------------------------
# 仕組みの可視化
# ---------------------------------------------------------------------------
def _render_mechanism_explanation(result):
    """直近日の米国セクター変動 → 日本セクター予測の流れを可視化する。"""
    strategies = result.strategies
    if "PCA_SUB" not in strategies or strategies["PCA_SUB"].signals is None:
        return
    if result.us_cc_returns is None:
        return

    sig = strategies["PCA_SUB"].signals.dropna(how="all")
    us_ret = result.us_cc_returns
    if len(sig) == 0 or len(us_ret) == 0:
        return

    st.markdown("## 提案手法の仕組み (直近日の例)")

    st.markdown("""
この戦略は以下の3ステップで動きます:

**Step 1.** 前日の米国11セクターETFの騰落率を取得する\n
**Step 2.** PCAで「日米に共通する変動パターン」を抽出し、米国の動きをそのパターンに分解する\n
**Step 3.** 同じパターンを使って「日本の各セクターがどう動くか」を予測し、上位をロング・下位をショートする
""")

    # 直近日を取得
    latest_date = sig.index[-1]
    latest_sig = sig.iloc[-1]

    # 対応する米国リターン (直近)
    us_dates_before = us_ret.index[us_ret.index <= latest_date]
    if len(us_dates_before) == 0:
        return
    latest_us_date = us_dates_before[-1]
    latest_us_ret = us_ret.loc[latest_us_date]

    st.markdown(f"#### 直近の例 ({latest_us_date.strftime('%Y-%m-%d')})")

    # --- Step 1: 米国セクターの騰落率 ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Step 1: 前日の米国セクター騰落率**")
        us_display = pd.DataFrame({
            "セクター": [US_TICKER_NAMES.get(t, t) for t in latest_us_ret.index],
            "騰落率 (%)": [round(v * 100, 2) for v in latest_us_ret.values],
        }).sort_values("騰落率 (%)", ascending=False)

        st.dataframe(
            us_display.style.map(
                lambda v: "color: #2E7D32; font-weight: 600" if isinstance(v, (int, float)) and v > 0
                else "color: #C62828; font-weight: 600" if isinstance(v, (int, float)) and v < 0
                else "",
                subset=["騰落率 (%)"],
            ).format({"騰落率 (%)": "{:+.2f}"}),
            hide_index=True,
            height=430,
        )

    # --- Step 3: 日本セクターの予測 ---
    with col_right:
        st.markdown("**Step 3: 翌日の日本セクター予測 → 売買判断**")
        q = result.config.quantile_q
        n_long = max(1, int(np.ceil(len(latest_sig) * q)))

        sorted_sig = latest_sig.sort_values(ascending=False)
        jp_display_rows = []
        for i, (ticker, val) in enumerate(sorted_sig.items()):
            if i < n_long:
                pos = "ロング (買い)"
            elif i >= len(sorted_sig) - n_long:
                pos = "ショート (売り)"
            else:
                pos = "-"
            jp_display_rows.append({
                "セクター": JP_TICKER_NAMES.get(ticker, ticker),
                "予測スコア": round(val, 4),
                "売買": pos,
            })
        jp_display = pd.DataFrame(jp_display_rows)

        st.dataframe(
            jp_display.style.map(
                lambda v: "color: #2E7D32; font-weight: bold"
                if "ロング" in str(v) else "color: #C62828; font-weight: bold"
                if "ショート" in str(v) else "",
                subset=["売買"],
            ).format({"予測スコア": "{:+.4f}"}),
            hide_index=True,
            height=630,
        )

    # --- Step 2 の説明 (簡潔に) ---
    st.markdown("""
> **Step 2 (内部処理)**: 過去60日間の日米セクターの相関構造から、
> 「グローバル景気」「日米の差」「景気敏感 vs 防衛的」の3つの共通パターンを抽出。
> 米国側の騰落率をこの3パターンに分解し、同じパターンの日本側への影響度で予測スコアを算出しています。
> 正則化 (λ=0.9) により、短期のノイズに振り回されず安定した予測が可能になっています。
""")


# ---------------------------------------------------------------------------
# タブ3: シグナル分析
# ---------------------------------------------------------------------------
def _render_signals_tab():
    result = st.session_state.get("ll_result")
    if result is None:
        st.info("「設定・実行」タブでバックテストを実行してください。")
        return

    strategies = result.strategies

    # --- 仕組みの可視化: 直近日の米国→日本の伝播 ---
    _render_mechanism_explanation(result)

    st.markdown("---")

    # --- シグナルヒートマップ ---
    if "PCA_SUB" in strategies and strategies["PCA_SUB"].signals is not None:
        st.markdown("## シグナル ヒートマップ")
        st.caption("各日の日本セクターETFへの予測シグナル。赤=ロング (上昇予測)、青=ショート (下落予測)")

        sig = strategies["PCA_SUB"].signals.dropna(how="all")

        # 表示期間の選択
        n_months = st.selectbox(
            "表示期間",
            [3, 6, 12, 24, 0],
            format_func=lambda x: f"直近{x}ヶ月" if x > 0 else "全期間",
            index=2,
        )
        if n_months > 0:
            cutoff = sig.index[-1] - pd.DateOffset(months=n_months)
            sig = sig[sig.index >= cutoff]

        col_names = [JP_TICKER_NAMES.get(c, c) for c in sig.columns]

        fig = go.Figure(data=go.Heatmap(
            z=sig.values.T,
            x=sig.index,
            y=col_names,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="シグナル"),
        ))
        fig.update_layout(
            height=600,
            xaxis_title="日付",
            yaxis_title="セクター",
            template="plotly_white",
            margin=dict(l=150, r=20, t=30, b=40),
        )
        st.plotly_chart(fig, width="stretch")

    # --- 固有値推移 ---
    if result.eigenvalue_history is not None:
        st.markdown("## 主成分の固有値推移")
        st.caption("上位K個の固有値の時系列。安定していれば日米共通の変動構造が持続していることを示す")
        eigvals = result.eigenvalue_history
        dates = result.strategies.get("PCA_SUB", next(iter(strategies.values()))).daily_returns.index
        # 追加シグナルでdatesが1行多い場合がある
        if len(dates) > len(eigvals):
            dates = dates[:len(eigvals)]

        valid_mask = ~np.isnan(eigvals[:, 0])
        if valid_mask.sum() > 0:
            fig = go.Figure()
            for k in range(eigvals.shape[1]):
                fig.add_trace(go.Scatter(
                    x=dates[valid_mask],
                    y=eigvals[valid_mask, k],
                    name=f"第{k+1}主成分",
                    line=dict(width=1.5),
                ))
            fig.update_layout(
                height=350,
                xaxis_title="日付",
                yaxis_title="固有値",
                template="plotly_white",
                legend=dict(x=0.02, y=0.98),
                margin=dict(l=60, r=20, t=30, b=40),
            )
            st.plotly_chart(fig, width="stretch")

    # --- 直近シグナル ---
    if "PCA_SUB" in strategies and strategies["PCA_SUB"].signals is not None:
        st.markdown("## 直近のシグナル一覧")
        sig = strategies["PCA_SUB"].signals.dropna(how="all")
        if len(sig) > 0:
            latest = sig.iloc[-1].sort_values(ascending=False)
            n_long = max(1, int(np.ceil(len(latest) * result.config.quantile_q)))

            latest_df = pd.DataFrame({
                "セクター": [JP_TICKER_NAMES.get(t, t) for t in latest.index],
                "コード": latest.index,
                "シグナル強度": latest.values,
                "ポジション": [
                    "ロング (買い)" if i < n_long
                    else "ショート (売り)" if i >= len(latest) - n_long
                    else "-"
                    for i in range(len(latest))
                ],
            })

            st.dataframe(
                latest_df.style.map(
                    lambda v: "color: #2E7D32; font-weight: bold"
                    if "ロング" in str(v) else "color: #C62828; font-weight: bold"
                    if "ショート" in str(v) else "",
                    subset=["ポジション"],
                ).format({"シグナル強度": "{:+.4f}"}),
                hide_index=True,
            )
            st.caption(f"基準日: {sig.index[-1].strftime('%Y-%m-%d')}")


# ---------------------------------------------------------------------------
# タブ4: パラメータ比較
# ---------------------------------------------------------------------------
def _render_param_compare_tab():
    history = st.session_state.get("ll_history", [])

    st.markdown("## パラメータ比較")
    st.caption(
        "異なるパラメータで実行した結果を並べて比較できます。"
        "「設定・実行」タブでパラメータを変えて複数回実行してください。"
    )

    if len(history) == 0:
        st.info("まだ実行履歴がありません。「設定・実行」タブでバックテストを実行すると、自動的にここに保存されます。")
        return

    if len(history) == 1:
        st.warning("比較するには2つ以上の実行結果が必要です。パラメータを変えてもう一度実行してください。")

    # --- パフォーマンス比較表 (提案手法同士) ---
    st.markdown("### パラメータ別パフォーマンス")
    rows = []
    for i, h in enumerate(history):
        r = h["result"]
        cfg = r.config
        has_custom_gap = cfg.gap_threshold < 1.0

        for key, strat in r.strategies.items():
            # GAP閾値付き設定の場合: PCA_SUB (ベースライン) と GAP_CUSTOM のみ表示
            # 他の戦略 (HYBRID等) は同じパラメータのGAP無し設定と重複するので除外
            if has_custom_gap and key not in ("PCA_SUB", "GAP_CUSTOM"):
                continue

            m = strat.metrics
            strat_label = STRATEGY_LABELS.get(key, key)
            if key == "GAP_CUSTOM":
                gap_pct = int(cfg.gap_threshold * 100)
                strat_label = f"GAP_{gap_pct}%"

            row = {
                "#": i + 1,
                "パラメータ": h["label"],
                "戦略": strat_label,
                "年率リターン (%)": round(m["AR"], 1),
                "シャープ比": round(m["R/R"], 2),
                "最大下落率 (%)": round(m["MDD"], 1),
            }
            # エントリー率はGAP系のみ表示
            if key in ("GAP_CUSTOM", "GAP_0", "GAP_10", "GAP_20", "GAP_30", "GAP_50", "GAP_70", "K3K4_GAP"):
                row["エントリー率 (%)"] = round(m.get("entry_rate", 100), 0)

            row["月次勝率 (%)"] = round(m.get("monthly_win_rate", 0), 0)
            row["最悪月 (%)"] = round(m.get("monthly_worst", 0), 1)
            rows.append(row)

    if rows:
        df_compare = pd.DataFrame(rows)
        st.dataframe(
            df_compare.style.apply(_highlight_best_compare, axis=0).format({
                "年率リターン (%)": "{:+.1f}",
                "シャープ比": "{:.2f}",
                "最大下落率 (%)": "{:.1f}",
                "エントリー率 (%)": "{:.0f}",
                "月次勝率 (%)": "{:.0f}",
                "月次ブレ幅 (%)": "{:.2f}",
                "最悪月 (%)": "{:+.1f}",
            }),
            hide_index=True,
            height=min(800, len(df_compare) * 35 + 40),
        )

    # --- 累積リターンの重ね描き ---
    if len(history) >= 2:
        st.markdown("### 累積リターンの比較")
        fig = go.Figure()

        palette = ["#FF8000", "#1565C0", "#2E7D32", "#9C27B0", "#E91E63",
                    "#00BCD4", "#795548", "#607D8B"]
        for i, h in enumerate(history):
            r = h["result"]
            if "PCA_SUB" not in r.strategies:
                continue
            ret = r.strategies["PCA_SUB"].daily_returns.dropna()
            cum = (1 + ret).cumprod()
            fig.add_trace(go.Scatter(
                x=cum.index,
                y=cum.values,
                name=f"#{i+1} {h['label']}",
                line=dict(color=palette[i % len(palette)], width=2),
            ))

        fig.update_layout(
            height=500,
            xaxis_title="日付",
            yaxis_title="累積リターン (1起点)",
            template="plotly_white",
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=60, r=20, t=30, b=40),
        )
        st.plotly_chart(fig, width="stretch")

    # --- 年別リターンの比較 ---
    if len(history) >= 2:
        st.markdown("### 年別リターン比較 (%)")
        annual_data = {}
        for i, h in enumerate(history):
            r = h["result"]
            if "PCA_SUB" not in r.strategies:
                continue
            ret = r.strategies["PCA_SUB"].daily_returns.dropna()
            if len(ret) == 0:
                continue
            yearly = ret.groupby(ret.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
            annual_data[f"#{i+1} {h['label']}"] = yearly

        if annual_data:
            df_annual = pd.DataFrame(annual_data)
            df_annual.index.name = "年"
            st.dataframe(
                df_annual.style.format("{:+.1f}").map(
                    lambda v: "color: #2E7D32" if v > 0 else "color: #C62828" if v < 0 else ""
                ),
            )

    # --- 月次リターン比較 ---
    if len(history) >= 2:
        st.markdown("### 月別平均リターン比較 (%)")
        st.caption("各月の平均リターンをパラメータ設定ごとに比較")
        monthly_avg_data = {}
        for i, h in enumerate(history):
            r = h["result"]
            if "GAP_CUSTOM" in r.strategies:
                ret = r.strategies["GAP_CUSTOM"].daily_returns.dropna()
            elif "PCA_SUB" in r.strategies:
                ret = r.strategies["PCA_SUB"].daily_returns.dropna()
            else:
                continue
            if len(ret) == 0:
                continue
            by_month = ret.groupby(ret.index.month)
            avgs = {}
            for m_num in range(1, 13):
                if m_num not in by_month.groups:
                    continue
                group = by_month.get_group(m_num)
                m_agg = group.groupby([group.index.year, group.index.month]).apply(
                    lambda x: ((1 + x).prod() - 1) * 100
                )
                avgs[f"{m_num}月"] = round(m_agg.mean(), 1)
            monthly_avg_data[f"#{i+1} {h['label']}"] = avgs

        if monthly_avg_data:
            df_monthly = pd.DataFrame(monthly_avg_data).T
            df_monthly.index.name = "パラメータ"
            st.dataframe(
                df_monthly.style.format("{:+.1f}", na_rep="-").map(
                    lambda v: "color: #2E7D32" if isinstance(v, (int, float)) and v > 0
                    else "color: #C62828" if isinstance(v, (int, float)) and v < 0
                    else ""
                ),
            )

    # --- AI によるパラメータ比較分析 ---
    if len(history) >= 2:
        st.markdown("---")
        _render_param_compare_ai(history)

    # --- AI分析を基に追加探索 (このタブ内で完結) ---
    if len(history) >= 2 and "ll_param_ai" in st.session_state:
        st.markdown("---")
        _render_explore_from_analysis(history)

    # --- 履歴クリア ---
    st.markdown("---")
    if st.button("履歴をすべてクリア", key="clear_history"):
        st.session_state["ll_history"] = []
        st.session_state.pop("ll_param_ai", None)
        st.session_state.pop("ll_param_suggestions", None)
        st.rerun()


def _build_param_compare_prompt(history: list) -> str:
    """パラメータ比較用のAI分析プロンプトを構築する。新機能 (DYNAMIC_K, REGIME, BLEND等) を含む。"""
    # 全戦略の名前リスト
    all_strat_keys = ["PCA_SUB", "GAP_CUSTOM", "HYBRID", "K3K4_ENS", "GAP_-10", "GAP_0", "GAP_5", "GAP_10", "GAP_20", "GAP_30", "K3K4_GAP"]

    # --- パラメータ一覧 ---
    param_changes = []
    for i, h in enumerate(history):
        cfg = h["result"].config
        extras = []
        if cfg.gap_threshold < 1.0:
            extras.append(f"GAP{int(cfg.gap_threshold*100)}%")
        if cfg.net_exposure_limit < 1.0:
            ne_ml = "復活" if cfg.net_exposure_mode == "fill" else "削減"
            extras.append(f"NE{int(cfg.net_exposure_limit*100)}%{ne_ml}")
        if cfg.net_exposure_skip_long > 0 or cfg.net_exposure_skip_short > 0:
            sp = []
            if cfg.net_exposure_skip_long > 0:
                sp.append(f"L≧{cfg.net_exposure_skip_long}")
            if cfg.net_exposure_skip_short > 0:
                sp.append(f"S≧{cfg.net_exposure_skip_short}")
            extras.append(f"NE{'/'.join(sp)}skip")
        if cfg.accumulate_us_returns:
            extras.append("US累積あり")
        ext_str = f", {', '.join(extras)}" if extras else ""
        param_changes.append(
            f"- #{i+1}: L={cfg.rolling_window}, λ={cfg.lambda_reg}, K={cfg.n_components}, "
            f"q={cfg.quantile_q}, 学習~{cfg.prior_end_date}{ext_str}"
        )
    param_text = "\n".join(param_changes)

    # --- メイン比較表 (GAP_CUSTOMがあればそちらを優先、なければPCA_SUB) ---
    table_lines = [
        "| # | パラメータ | 戦略 | 年率R(%) | SR | MDD(%) | エントリー率(%) | 月次勝率(%) | 月次平均(%) | 月次ブレ幅(%) | 最悪月(%) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, h in enumerate(history):
        r = h["result"]
        # GAP_CUSTOMがあればそちらを表示 (GAPフィルター適用後の実際のパフォーマンス)
        if "GAP_CUSTOM" in r.strategies:
            m = r.strategies["GAP_CUSTOM"].metrics
            strat_name = f"GAP_{int(r.config.gap_threshold*100)}%"
        elif "PCA_SUB" in r.strategies:
            m = r.strategies["PCA_SUB"].metrics
            strat_name = "PCA_SUB"
        else:
            continue
        table_lines.append(
            f"| {i+1} | {h['label']} | {strat_name} | {m['AR']:.1f} | {m['R/R']:.2f} | {m['MDD']:.1f} "
            f"| {m.get('entry_rate', 100):.0f} "
            f"| {m.get('monthly_win_rate', 0):.0f} | {m.get('monthly_mean', 0):+.2f} "
            f"| {m.get('monthly_std', 0):.2f} | {m.get('monthly_worst', 0):+.1f} |"
        )
    table_text = "\n".join(table_lines)

    # --- 拡張戦略の結果 (DYNAMIC_K, REGIME, BLEND等) ---
    ext_lines = ["各パラメータ設定で自動計算された拡張戦略の結果:"]
    ext_lines.append("| # | 戦略 | 年率R(%) | SR | MDD(%) | エントリー率(%) | 月次勝率(%) | 月次ブレ幅(%) | 最悪月(%) |")
    ext_lines.append("|---|---|---|---|---|---|---|---|---|")
    has_ext = False
    for i, h in enumerate(history):
        r = h["result"]
        cfg = r.config
        has_custom_gap = cfg.gap_threshold < 1.0
        for key in all_strat_keys:
            if key == "PCA_SUB":
                continue
            if key not in r.strategies:
                continue
            # GAP閾値付き設定ではGAP_CUSTOMのみ。他は重複
            if has_custom_gap and key != "GAP_CUSTOM":
                continue
            has_ext = True
            m = r.strategies[key].metrics
            label = STRATEGY_LABELS.get(key, key)
            if key == "GAP_CUSTOM":
                label = f"GAP_{int(cfg.gap_threshold*100)}%"
            ext_lines.append(
                f"| {i+1} | {label} | {m['AR']:.1f} | {m['R/R']:.2f} | {m['MDD']:.1f} "
                f"| {m.get('entry_rate', 100):.0f} "
                f"| {m.get('monthly_win_rate', 0):.0f} | {m.get('monthly_std', 0):.2f} "
                f"| {m.get('monthly_worst', 0):+.1f} |"
            )
    ext_text = "\n".join(ext_lines) if has_ext else "(拡張戦略なし)"

    # --- 年別リターン (GAP_CUSTOM優先、なければPCA_SUB) ---
    annual_lines = []
    for i, h in enumerate(history):
        r = h["result"]
        if "GAP_CUSTOM" in r.strategies:
            ret = r.strategies["GAP_CUSTOM"].daily_returns.dropna()
        elif "PCA_SUB" in r.strategies:
            ret = r.strategies["PCA_SUB"].daily_returns.dropna()
        else:
            continue
        if len(ret) == 0:
            continue
        yearly = ret.groupby(ret.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
        for year, val in yearly.items():
            annual_lines.append(f"- #{i+1} {year}年: {val:+.1f}%")
    annual_text = "\n".join(annual_lines) if annual_lines else "(データなし)"

    # --- 拡張戦略の年別リターン (BLEND, REGIME等 - 代表設定のみ) ---
    ext_annual_lines = []
    # 最初の設定だけでも拡張戦略の年別を出す
    for i, h in enumerate(history[:3]):  # 上位3件
        r = h["result"]
        for key in ["BLEND_50", "REGIME", "DYNAMIC_K"]:
            if key not in r.strategies:
                continue
            ret = r.strategies[key].daily_returns.dropna()
            if len(ret) == 0:
                continue
            label = STRATEGY_LABELS.get(key, key)
            yearly = ret.groupby(ret.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
            for year, val in yearly.items():
                ext_annual_lines.append(f"- #{i+1} {label} {year}年: {val:+.1f}%")
    ext_annual_text = "\n".join(ext_annual_lines) if ext_annual_lines else "(データなし)"

    # --- TOPIX ---
    topix_text = "(データなし)"
    excess_compare_text = "(データなし)"
    bm = None
    for h in history:
        if h["result"].benchmark_returns is not None and len(h["result"].benchmark_returns) > 0:
            bm = h["result"].benchmark_returns.dropna()
            break
    if bm is not None and len(bm) > 0:
        bm_yearly = bm.groupby(bm.index.year).apply(lambda x: ((1 + x).prod() - 1) * 100)
        topix_text = "\n".join([f"- TOPIX {y}年: {v:+.1f}%" for y, v in bm_yearly.items()])

        from core.lead_lag.strategy import compute_metrics as _cm
        ex_table_lines = [
            "| # | 戦略 | 超過R(%) | 超過SR | 対TOPIX月次勝率(%) | 超過ブレ幅(%) |",
            "|---|---|---|---|---|---|",
        ]
        for i, h in enumerate(history):
            r = h["result"]
            for key in ["PCA_SUB", "BLEND_50", "BLEND_70", "REGIME", "DYNAMIC_K"]:
                if key not in r.strategies:
                    continue
                strat_ret_filled = r.strategies[key].daily_returns.reindex(bm.index).fillna(0)
                excess_daily = strat_ret_filled - bm
                em = _cm(excess_daily.values, bm.index)
                label = STRATEGY_LABELS.get(key, key)
                ex_table_lines.append(
                    f"| {i+1} | {label} | {em['AR']:+.1f} | {em['R/R']:.2f} "
                    f"| {em.get('monthly_win_rate',0):.0f} | {em.get('monthly_std',0):.2f} |"
                )
        excess_compare_text = "\n".join(ex_table_lines)

    prompt = f"""あなたは定量投資戦略の専門アナリストです。
日米セクターETFリードラグ戦略を、異なるパラメータ設定で{len(history)}回実行した結果を比較分析してください。

## 戦略の概要
米国セクターETFの当日リターンから、翌営業日の日本セクターETFの寄引リターンを予測するロング・ショート戦略。

## パラメータの意味
- **L**: ローリングウィンドウ長（日数）。長いほど安定、短いほど追従性が高い。
- **λ**: 正則化強度。1に近いほど事前知識を重視、0に近いほど直近データを重視。
- **K**: 主成分数。日米共通の変動パターン数。
- **q**: 売買比率。17セクター中の上位/下位何%をロング/ショートするか。0.5=全セクター。
- **GAP閾値**: オーバーナイトギャップフィルターの閾値。**値が小さいほど厳しい（エントリー率が下がる）。**
  - GAP=-10%: 最も厳しい。シグナルと逆方向に10%以上動いた銘柄のみエントリー。
  - GAP=0%: 厳しい。シグナル方向に少しでもギャップアップしたらスキップ。
  - GAP=5%: やや厳しい。シグナルの5%までのギャップは許容。
  - GAP=10%: 標準。シグナルの10%までのギャップは許容。
  - GAP=20%: 緩い。ほとんどエントリーする。
  - GAP=100%: フィルターなし。
- **NE上限**: GAPフィルター後のロング数とショート数の偏りを制限。復活=少ない側にGAPで外された銘柄を復活。削減=多い側を削除。
- **|NE|≧Nスキップ**: GAPフィルター後の|ロング数-ショート数|がN以上ならその日は全銘柄見送り。
- **US累積**: JP休場中の複数日分の米国リターンを累積してシグナル入力に使用。

## 戦略間の関係 (重要: 正しく理解すること)
- **正則化PCA (PCA_SUB)**: ベースとなるシグナル生成。GAPフィルターなしで全日エントリー。
- **GAP_X% (GAP_-10〜GAP_30, GAP_CUSTOM)**: すべて**正則化PCAのシグナルを使用**し、GAPフィルターを適用したもの。通常PCAのシグナルではない。つまりGAP_5%は「正則化PCA + GAPフィルター5%」であり、両者は既に組み合わさっている。
- **K3K4_ENS**: K=3とK=4のアンサンブル。GAPフィルターなし。
- **K3K4_GAP**: K3K4_ENSシグナルにGAP_50%を適用。
- **通常PCA (PCA_PLAIN)**: 正則化なし(λ=0)のPCA。比較用ベースライン。
- **HYBRID**: VIX/円高でポジション縮小する正則化PCA。

## 各実行のパラメータ
{param_text}

## メイン戦略パフォーマンス比較 (GAP_CUSTOMがあればGAPフィルター適用後の値)
{table_text}

## 拡張戦略の結果
{ext_text}

## 対TOPIX超過パフォーマンス比較 (PCA_SUB + 拡張戦略)
{excess_compare_text}

## TOPIX 年別リターン
{topix_text}

## メイン戦略 年別リターン (GAP_CUSTOMがあればGAPフィルター適用後の値)
{annual_text}

## 拡張戦略 年別リターン (代表設定)
{ext_annual_text}

以下の観点で分析してください:

1. **拡張戦略の効果分析 (最重要)**: DYNAMIC_K, REGIME, BLEND_50/70 の各戦略がPCA_SUBに対してどう改善しているか。特に:
   - BLENDが2023年以降のTOPIX強気局面での対TOPIX劣後をどの程度補完できているか
   - REGIMEが2026年のマイナスをどの程度抑制できているか
   - DYNAMIC_Kがパフォーマンスの安定性に寄与しているか
   - 拡張戦略間で最も有望なのはどれか

2. **総合最優秀設定**: PCA_SUBと拡張戦略を含めた全体の中で、最優秀の「パラメータ + 戦略タイプ」の組み合わせ

3. **月次安定性**: 絶対リターンと対TOPIX超過の両方で毎月安定しているか

4. **対TOPIX分析**: 拡張戦略がTOPIX超過の安定性をどう変えたか

5. **次の探索提案**: パラメータと拡張戦略の組み合わせで、次に試すべきものを2-3個提案

6. **ギャップフィルターの分析 (重要)**: GAP_50/K3K4_GAPの効果を分析してください。特に:
   - シグナル相対閾値 (absorption_rate=50%) は適切か。もっと厳しく(30%)すべきか緩く(70%)すべきか
   - ギャップフィルターによる取引頻度の低下とリターン向上のトレードオフ
   - 「オーバーナイトリバーサル」アノマリーとの重複取り: 純粋なリードラグ効果なのか、ギャップリバーサルも乗せているのか
   - SR/リターンが過大評価されている可能性はあるか (実運用でのスリッページ等)

7. **2026年パフォーマンス改善**: 拡張戦略が2026年の劣化をどの程度緩和できているか

8. **実運用推奨・改善提案 (最重要)**: 提供データの中でSRが最も高い設定を起点とし、そのSRをさらに向上させるための具体的な改善案を提示すること。SR最良の設定がなぜ高SRを達成しているかを分析し、その強みを維持・強化しつつ弱点を補う方向で次に試すべきパラメータの組み合わせを3つ提案すること。"""

    return prompt


def _run_param_ai_thread(progress_dict: dict, prompt: str):
    """パラメータ比較AI分析をバックグラウンドで実行。"""
    try:
        from core.ai_client import create_ai_client

        progress_dict["message"] = "Claude にパラメータ比較を分析させています..."
        progress_dict["pct"] = 0.3
        client = create_ai_client()
        response = client.send_message(prompt, system_prompt="")
        progress_dict["_result"] = response
        progress_dict["pct"] = 1.0
        progress_dict["message"] = "完了"
    except Exception as e:
        import traceback

        progress_dict["error"] = str(e)
        progress_dict["detail"] = traceback.format_exc()
        progress_dict["pct"] = 1.0


def _render_explore_from_analysis(history: list):
    """AI分析結果を基にした追加探索セクション (パラメータ比較タブ内)。"""
    st.markdown("### AI 分析を基に追加探索")
    st.caption("上の分析結果を踏まえて、AIが新たなパラメータを提案し、そのまま実行できます。")

    # --- AI提案取得中 ---
    sug_thread = st.session_state.get("ll_param_sug_thread")
    is_suggesting = sug_thread is not None and sug_thread.is_alive()

    if is_suggesting:
        prog = st.session_state.get("ll_param_sug_progress", {})
        st.warning("AI が分析結果を基にパラメータを提案中...")
        st.progress(prog.get("pct", 0), text=prog.get("message", ""))
        import time
        time.sleep(2)
        st.rerun()
        return

    # 取得完了
    if sug_thread is not None and not sug_thread.is_alive():
        prog = st.session_state.get("ll_param_sug_progress", {})
        if "_result" in prog and "ll_param_suggestions" not in st.session_state:
            st.session_state["ll_param_suggestions"] = prog.pop("_result")
            st.session_state.pop("ll_param_sug_thread", None)
            st.session_state.pop("ll_param_sug_progress", None)
            st.rerun()
        elif prog.get("error"):
            st.error(f"AI提案エラー: {prog['error']}")
            if st.button("OK", key="param_sug_err"):
                st.session_state.pop("ll_param_sug_thread", None)
                st.session_state.pop("ll_param_sug_progress", None)
                st.rerun()
            return

    # --- 追加探索実行中 ---
    run_thread = st.session_state.get("ll_param_run_thread")
    is_running = run_thread is not None and run_thread.is_alive()

    if is_running:
        prog = st.session_state.get("ll_param_run_progress", {})
        import time
        started_at = prog.get("_started_at")
        elapsed = time.time() - started_at if started_at else 0
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)
        st.warning(f"提案パラメータで追加探索中... ({elapsed_min}:{elapsed_sec:02d} 経過)")
        st.progress(prog.get("pct", 0), text=prog.get("message", ""))
        time.sleep(3)
        st.rerun()
        return

    # 実行完了
    if run_thread is not None and not run_thread.is_alive():
        prog = st.session_state.get("ll_param_run_progress", {})
        if "_result" in prog:
            all_results = prog.pop("_result")
            for item in all_results:
                result = item["result"]
                cfg = result.config
                suffix = ""
                pass  # 拡張戦略は自動計算
                param_key = f"L{cfg.rolling_window}_λ{cfg.lambda_reg}_K{cfg.n_components}_q{cfg.quantile_q}_G{cfg.gap_threshold}_P{cfg.prior_end_date}{suffix}_{result.period_start}_{result.period_end}"
                label = f"L={cfg.rolling_window} λ={cfg.lambda_reg} K={cfg.n_components} q={cfg.quantile_q}" + (f" GAP{int(cfg.gap_threshold*100)}%" if cfg.gap_threshold < 1.0 else "") + f" ~{cfg.prior_end_date[:4]}"
                pass
                existing_keys = [h.get("_param_key") for h in history]
                if param_key not in existing_keys:
                    history.append({
                        "_param_key": param_key,
                        "label": label,
                        "period": f"{result.period_start}~{result.period_end}",
                        "result": result,
                    })
            st.session_state.pop("ll_param_run_thread", None)
            st.session_state.pop("ll_param_run_progress", None)
            st.session_state.pop("ll_param_suggestions", None)
            st.session_state.pop("ll_param_ai", None)  # AI分析も再実行を促す
            st.success(f"追加探索完了 (履歴: {len(history)}件)。上のAI分析を再実行して新しい結果を含めた分析ができます。")
            st.rerun()
        elif prog.get("error"):
            st.error(f"探索エラー: {prog['error']}")
            st.session_state.pop("ll_param_run_thread", None)
            st.session_state.pop("ll_param_run_progress", None)
        return

    # --- 提案がある場合: 表示 + 実行ボタン ---
    suggestions = st.session_state.get("ll_param_suggestions")
    if suggestions:
        st.markdown(f"**AI が提案する追加探索パラメータ ({len(suggestions)}件):**")
        sug_rows = []
        for i, s in enumerate(suggestions):
            row = {
                "#": i + 1,
                "提案名": s.get("label", f"提案{i+1}"),
                "L": s.get("L", 100),
                "λ": s.get("lambda", 0.85),
                "K": s.get("K", 3),
                "q": s.get("q", 0.25),
            }
            if "prior_end" in s:
                row["学習期間"] = s["prior_end"]
            row["理由"] = s.get("reason", "")
            sug_rows.append(row)
        st.dataframe(pd.DataFrame(sug_rows), hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("この提案で追加探索を実行", type="primary", key="run_param_suggestions"):
                grid = []
                for i, s in enumerate(suggestions):
                    g = {
                        "label": s.get("label", f"AI提案{i+1}"),
                        "L": s["L"], "lambda": s["lambda"], "K": s["K"], "q": s["q"],
                    }
                    if "prior_end" in s:
                        g["prior_end"] = s["prior_end"]
                    grid.append(g)

                # 期間は直近の履歴から
                last_cfg = history[-1]["result"].config
                start_d = last_cfg.start_date
                end_d = last_cfg.end_date or "2025-12-31"
                prior_d = last_cfg.prior_end_date

                provider, cache = _get_provider_and_cache()
                import time as _time
                progress_dict = {"message": "開始中...", "pct": 0.0, "_started_at": _time.time()}
                st.session_state["ll_param_run_progress"] = progress_dict
                thread = threading.Thread(
                    target=_run_auto_search_thread,
                    args=(progress_dict, provider, cache, start_d, end_d, prior_d, grid),
                    daemon=True,
                )
                thread.start()
                st.session_state["ll_param_run_thread"] = thread
                st.rerun()
        with col2:
            if st.button("提案を破棄", key="discard_param_suggestions"):
                st.session_state.pop("ll_param_suggestions", None)
                st.rerun()
        return

    # --- 提案取得ボタン ---
    if st.button("この分析を基に追加探索パラメータを生成", type="primary", key="gen_param_suggestions"):
        prompt = _build_suggest_prompt(history)
        progress_dict = {"message": "開始中...", "pct": 0.0}
        st.session_state["ll_param_sug_progress"] = progress_dict
        thread = threading.Thread(
            target=_run_ai_suggest_thread,
            args=(progress_dict, prompt),
            daemon=True,
        )
        thread.start()
        st.session_state["ll_param_sug_thread"] = thread
        st.rerun()


def _render_param_compare_ai(history: list):
    """パラメータ比較のAI分析セクション。"""
    st.markdown("### AI によるパラメータ比較分析")

    # 実行中チェック
    ai_thread = st.session_state.get("ll_param_ai_thread")
    is_running = ai_thread is not None and ai_thread.is_alive()

    if is_running:
        prog = st.session_state.get("ll_param_ai_progress", {})
        msg = prog.get("message", "分析中...")
        pct = prog.get("pct", 0)
        st.warning("AI がパラメータ比較を分析中です... (数十秒かかります)")
        st.progress(pct, text=msg)

        import time
        time.sleep(2)
        st.rerun()
        return

    # スレッド完了後
    if ai_thread is not None and not ai_thread.is_alive():
        prog = st.session_state.get("ll_param_ai_progress", {})
        if "_result" in prog:
            st.session_state["ll_param_ai"] = prog["_result"]
            st.session_state.pop("ll_param_ai_thread", None)
            st.session_state.pop("ll_param_ai_progress", None)
            st.rerun()
            return
        elif prog.get("error"):
            st.error(f"AI分析エラー: {prog['error']}")
            detail = prog.get("detail", "")
            if detail:
                with st.expander("詳細"):
                    st.code(detail)
            if st.button("OK", key="param_ai_error_ok"):
                st.session_state.pop("ll_param_ai_thread", None)
                st.session_state.pop("ll_param_ai_progress", None)
                st.rerun()
            return
        else:
            # スレッド終了したが結果もエラーもない (異常)
            st.session_state.pop("ll_param_ai_thread", None)
            st.session_state.pop("ll_param_ai_progress", None)

    # 既存の結果を表示
    if "ll_param_ai" in st.session_state:
        st.markdown(st.session_state["ll_param_ai"])
        col1, col2 = st.columns(2)
        with col1:
            if st.button("再分析する", key="param_ai_rerun"):
                st.session_state.pop("ll_param_ai", None)
                st.rerun()
        with col2:
            with st.expander("テキストをコピー"):
                st.text_area("", st.session_state["ll_param_ai"], height=400, key="copy_param_ai")
        return

    # 実行ボタン
    st.caption(
        f"現在 {len(history)} 件のパラメータ設定を比較できます。"
        "AI にどのパラメータが最適か、次に何を試すべきかを分析させます。"
    )
    if st.button("AI でパラメータ比較を分析する", type="primary", key="run_param_ai"):
        prompt = _build_param_compare_prompt(history)
        progress_dict = {"message": "開始中...", "pct": 0.0}
        st.session_state["ll_param_ai_progress"] = progress_dict

        import threading as _threading

        thread = _threading.Thread(
            target=_run_param_ai_thread,
            args=(progress_dict, prompt),
            daemon=True,
        )
        thread.start()
        st.session_state["ll_param_ai_thread"] = thread
        st.rerun()


def _highlight_best_compare(s):
    """比較表の最良値をハイライト"""
    higher_better = {"年率リターン (%)", "シャープ比", "最大下落率 (%)", "月次勝率 (%)", "月次平均 (%)"}
    lower_better = {"月次ブレ幅 (%)"}
    max_better = {"最悪月 (%)"}
    if s.name not in higher_better | lower_better | max_better:
        return [""] * len(s)
    try:
        vals = s.astype(float)
    except (ValueError, TypeError):
        return [""] * len(s)
    if s.name in higher_better:
        best = vals == vals.max()
    elif s.name in lower_better:
        best = vals == vals.min()
    else:
        best = vals == vals.max()
    return ["font-weight: bold; color: #FF8000" if v else "" for v in best]


# ---------------------------------------------------------------------------
# 寄与分解タブ (Phase 0)
# ---------------------------------------------------------------------------
def _run_attribution_thread(progress_dict: dict, backtest_result, config, sub_periods):
    """寄与分解をバックグラウンドで実行するスレッド。"""
    try:
        from core.lead_lag.signal_attribution import compute_signal_attribution

        def on_progress(msg, pct):
            progress_dict["message"] = msg
            progress_dict["pct"] = pct

        attr = compute_signal_attribution(
            backtest_result=backtest_result,
            config=config,
            sub_periods=sub_periods,
            progress_callback=on_progress,
        )
        progress_dict["_result"] = attr
        progress_dict["pct"] = 1.0
        progress_dict["message"] = "完了"
    except Exception as e:
        import traceback
        progress_dict["error"] = str(e)
        progress_dict["detail"] = traceback.format_exc()
        progress_dict["pct"] = 1.0


def _render_attribution_tab():
    """寄与分解 (Signal Attribution) タブ — Phase 0: ETF伝播ルート候補分析"""
    st.subheader("寄与分解 — US→JP 因子空間上の結合強度")
    st.caption(
        "PCAシグナルを各USセクター→各JPセクターへの寄与に分解します。"
        "これは因子空間での結合強度であり、経済的因果を直接示すものではありません。"
    )

    result = st.session_state.get("ll_result")
    if result is None:
        st.info("先に「設定・実行」タブでバックテストを実行してください。")
        return

    if "PCA_SUB" not in result.strategies or result.strategies["PCA_SUB"].signals is None:
        st.warning("PCA_SUB の結果がありません。run_pca_sub=True で実行してください。")
        return

    config = result.config

    # --- 期間設定 ---
    col1, col2 = st.columns(2)
    with col1:
        period_years = st.selectbox("期間分割 (年)", [3, 5, 7, 10], index=1, key="attr_period_years")
    with col2:
        st.markdown(f"**分析期間:** {result.period_start} — {result.period_end}")

    # --- 実行ボタン ---
    attr_thread = st.session_state.get("attr_thread")
    if attr_thread is not None and not attr_thread.is_alive() and "attr_result" not in st.session_state:
        prog = st.session_state.get("attr_progress", {})
        if "_result" in prog:
            st.session_state["attr_result"] = prog.pop("_result")
        elif "error" in prog:
            st.error(f"寄与分解エラー: {prog['error']}")
            if "detail" in prog:
                with st.expander("詳細"):
                    st.code(prog["detail"])

    if st.button("寄与分解を実行", type="primary", key="run_attribution"):
        progress = {"message": "開始...", "pct": 0.0}
        st.session_state["attr_progress"] = progress
        st.session_state.pop("attr_result", None)

        t = threading.Thread(
            target=_run_attribution_thread,
            args=(progress, result, config, None),
            daemon=True,
        )
        st.session_state["attr_thread"] = t
        t.start()
        st.rerun()

    # --- 実行中の進捗表示 ---
    if attr_thread is not None and attr_thread.is_alive():
        prog = st.session_state.get("attr_progress", {})
        import time
        bar = st.progress(prog.get("pct", 0.0))
        st.caption(prog.get("message", "実行中..."))
        time.sleep(1)
        st.rerun()

    # --- 結果表示 ---
    attr = st.session_state.get("attr_result")
    if attr is None:
        st.info("「寄与分解を実行」ボタンを押してください。")
        return

    from core.lead_lag.signal_attribution import (
        summarize_attribution,
        get_coupling_heatmap_data,
    )

    # --- 1. 結合強度ヒートマップ ---
    st.markdown("### 結合強度ヒートマップ")
    period_options = ["全期間"] + list(attr.period_stability.keys())
    selected_period = st.selectbox("期間", period_options, key="attr_heatmap_period")

    period_key = None if selected_period == "全期間" else selected_period
    matrix, us_labels, jp_labels = get_coupling_heatmap_data(attr, period=period_key)

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=matrix,
        x=jp_labels,
        y=us_labels,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 9},
        colorbar={"title": "結合強度"},
    ))
    fig_heatmap.update_layout(
        title=f"US→JP 結合強度 ({selected_period})",
        xaxis_title="JP セクター",
        yaxis_title="US セクター",
        height=max(400, len(us_labels) * 25 + 150),
        margin={"l": 200, "r": 50, "t": 50, "b": 150},
        xaxis={"tickangle": 45},
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- 2. 安定ペアランキング ---
    st.markdown("### 安定ペアランキング")
    st.caption("複数期間で符号が一致し、結合強度が安定しているペア上位")

    top_n = st.slider("表示件数", 10, 50, 20, key="attr_top_n")
    df_summary = summarize_attribution(attr, top_n=top_n)
    if not df_summary.empty:
        st.dataframe(
            df_summary.style.format({
                "mean_coupling": "{:+.4f}",
                "stability_score": "{:.3f}",
                "period_consistency": "{:.0%}",
                "avg_IC": "{:+.4f}",
                "avg_hit_rate": "{:.1%}",
            }),
            height=min(700, 35 * len(df_summary) + 40),
        )
    else:
        st.warning("安定ペアが見つかりませんでした。")

    # --- 3. 期間別比較 ---
    if len(attr.period_stability) >= 2:
        st.markdown("### 期間別 結合強度の変化")

        period_labels = list(attr.period_stability.keys())
        n_periods = len(period_labels)
        cols = st.columns(min(n_periods, 3))

        for idx, label in enumerate(period_labels):
            with cols[idx % len(cols)]:
                pdata = attr.period_stability[label]
                p_matrix = pdata["mean_coupling"]
                fig_p = go.Figure(data=go.Heatmap(
                    z=p_matrix,
                    x=[get_ticker_name(t) for t in attr.jp_tickers],
                    y=[get_ticker_name(t) for t in attr.us_tickers],
                    colorscale="RdBu_r",
                    zmid=0,
                    zmin=-0.3,
                    zmax=0.3,
                    colorbar={"title": ""},
                ))
                fig_p.update_layout(
                    title=f"{label} ({pdata['n_days']}日)",
                    height=350,
                    margin={"l": 10, "r": 10, "t": 40, "b": 80},
                    xaxis={"tickangle": 45, "tickfont": {"size": 8}},
                    yaxis={"tickfont": {"size": 8}},
                )
                st.plotly_chart(fig_p, use_container_width=True)

    # --- 4. ペア別IC ヒートマップ ---
    if attr.period_stability:
        st.markdown("### ペア別 予測力 (IC)")
        ic_period = st.selectbox(
            "期間", list(attr.period_stability.keys()),
            key="attr_ic_period",
        )
        ic_data = attr.period_stability[ic_period]["ic_by_pair"]

        fig_ic = go.Figure(data=go.Heatmap(
            z=ic_data,
            x=[get_ticker_name(t) for t in attr.jp_tickers],
            y=[get_ticker_name(t) for t in attr.us_tickers],
            colorscale="RdYlGn",
            zmid=0,
            text=np.where(np.isnan(ic_data), "", np.round(ic_data, 3).astype(str)),
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar={"title": "IC"},
        ))
        fig_ic.update_layout(
            title=f"ペア別 寄与→実現リターン IC ({ic_period})",
            xaxis_title="JP セクター",
            yaxis_title="US セクター",
            height=max(400, len(us_labels) * 25 + 150),
            margin={"l": 200, "r": 50, "t": 50, "b": 150},
            xaxis={"tickangle": 45},
        )
        st.plotly_chart(fig_ic, use_container_width=True)


# ---------------------------------------------------------------------------
main()
