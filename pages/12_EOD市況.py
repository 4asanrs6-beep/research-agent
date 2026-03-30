"""EOD市況ページ — 引け後の市場俯瞰 + ランキング + フィルタ + 簡易複合条件"""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from config import JQUANTS_API_KEY, MARKET_DATA_DIR
from core.eod_dashboard import (
    apply_compound_filter,
    compute_advance_decline,
    compute_eod_rankings,
    compute_market_segment_strength,
    compute_sector_summary,
    detect_new_highs_lows,
)
from core.sidebar import render_sidebar_running_indicator
from core.styles import apply_reuters_style, render_ranking_html
from core.universe_filter import MARKET_SEGMENTS, SECTOR_17_LIST
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider

st.set_page_config(page_title="EOD市況", page_icon="R", layout="wide")
apply_reuters_style()
render_sidebar_running_indicator()

_cache = DataCache(MARKET_DATA_DIR)
_provider = JQuantsProvider(api_key=JQUANTS_API_KEY, cache=_cache)


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




def _fmt_vol(v):
    if pd.isna(v):
        return "—"
    if v >= 1e8:
        return f"{v / 1e8:.1f}億"
    if v >= 1e4:
        return f"{v / 1e4:.0f}万"
    return f"{v:,.0f}"


def _fmt_val(v):
    if pd.isna(v):
        return "—"
    if v >= 1e12:
        return f"{v / 1e12:.1f}兆円"
    if v >= 1e8:
        return f"{v / 1e8:.0f}億円"
    if v >= 1e4:
        return f"{v / 1e4:.0f}万円"
    return f"{v:,.0f}円"


def _fmt_cap(v):
    if pd.isna(v):
        return "—"
    if v >= 10000:
        return f"{v / 10000:.1f}兆円"
    return f"{v:,.0f}億円"


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def _fetch_market_caps_yf(codes: tuple) -> pd.Series:
    """Yahoo Finance から時価総額（億円）を並列取得"""
    import yfinance as yf

    def _get_cap(code: str) -> tuple[str, float | None]:
        try:
            fi = yf.Ticker(f"{str(code)[:4]}.T").fast_info
            cap = getattr(fi, "market_cap", None)
            return code, float(cap) / 1e8 if cap else None
        except Exception:
            return code, None

    caps: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for code, cap in ex.map(_get_cap, codes):
            if cap is not None:
                caps[code] = cap
    return pd.Series(caps, dtype=float)


def _get_caps_for_codes(codes_list: list) -> pd.Series:
    """表示用の銘柄コードリストに対して時価総額を取得する（最大100件）"""
    if not codes_list:
        return pd.Series(dtype=float)
    codes_tuple = tuple(sorted(set(str(c) for c in codes_list[:100])))
    return _fetch_market_caps_yf(codes_tuple)


def _build_ranking_table(
    df: pd.DataFrame,
    sort_col: str,
    ascending: bool = False,
    n: int = 50,
    show_volume_change: bool = False,
    show_tv: bool = False,
    show_tv_change: bool = False,
    show_vol_surge: bool = False,
    fetch_caps: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """ランキングテーブルを構築。整形済みDataFrame + 色付け対象列を返す。"""
    if ascending:
        subset = df.nsmallest(n, sort_col)
    else:
        subset = df.nlargest(n, sort_col)

    # 表示対象の銘柄のみ時価総額を取得
    caps = _get_caps_for_codes(subset["code"].tolist()) if fetch_caps else pd.Series(dtype=float)

    rows = []
    for _, r in subset.iterrows():
        row = {
            "#": 0,
            "コード": r["code"],
            "銘柄名": r["name"],
            "終値": f"{r['adj_close']:,.0f}" if pd.notna(r.get("adj_close")) else "—",
            "騰落率(%)": round(r["daily_return"] * 100, 1) if pd.notna(r.get("daily_return")) else None,
            "時価総額": _fmt_cap(caps.get(r["code"])) if fetch_caps else "—",
        }
        if show_volume_change:
            row["出来高"] = _fmt_vol(r.get("volume"))
            vc = r.get("volume_change")
            row["前日比"] = f"{vc:.1f}x" if pd.notna(vc) else "—"
        if show_vol_surge:
            row["出来高"] = _fmt_vol(r.get("volume"))
            vc = r.get("volume_change")
            row["急増倍率"] = f"{vc:.1f}x" if pd.notna(vc) else "—"
        if show_tv:
            row["売買代金"] = _fmt_val(r.get("trading_value"))
        if show_tv_change:
            tvc = r.get("trading_value_change")
            row["前日比"] = f"{tvc:.1f}x" if pd.notna(tvc) else "—"
        row["市場"] = r.get("market_name", "")
        row["セクター"] = r.get("sector_17_name", "")
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result["#"] = range(1, len(result) + 1)

    # 色付け対象
    ret_cols = [c for c in ["騰落率(%)"] if c in result.columns]
    change_cols = [c for c in ["前日比", "急増倍率"] if c in result.columns]
    return result, ret_cols, change_cols


# =========================================================================
# サイドバー
# =========================================================================

with st.sidebar:
    st.header("フィルタ設定")

    today = date.today()
    eod_target_date = st.date_input("基準日", value=today, max_value=today, key="eod_date")

    st.divider()

    eod_market_segments = st.multiselect(
        "上場区分",
        options=MARKET_SEGMENTS,
        default=MARKET_SEGMENTS,
        key="eod_market",
    )

    eod_sectors = st.multiselect(
        "業種フィルタ",
        options=SECTOR_17_LIST,
        default=[],
        key="eod_sectors",
    )

    st.divider()

    eod_cap_col1, eod_cap_col2 = st.columns(2)
    with eod_cap_col1:
        eod_cap_min = st.number_input(
            "時価総額 下限(億円)",
            min_value=0,
            value=0,
            step=100,
            key="eod_cap_min",
        )
    with eod_cap_col2:
        eod_cap_max = st.number_input(
            "時価総額 上限(億円)",
            min_value=0,
            value=0,
            step=1000,
            key="eod_cap_max",
            help="0 = 制限なし",
        )

    eod_min_tv = st.number_input(
        "売買代金下限（百万円）",
        min_value=0,
        value=0,
        step=100,
        key="eod_min_tv",
    )

    st.divider()
    st.caption("データソース: J-Quants API + Yahoo Finance(時価総額)")

# =========================================================================
# メインエリア
# =========================================================================

st.title("EOD市況")

if not JQUANTS_API_KEY:
    st.error("J-Quants APIキーが未設定（`.env` に `JQUANTS_API_KEY` を設定）")
    st.stop()

target_str = eod_target_date.strftime("%Y-%m-%d")

with st.spinner("データ読み込み中..."):
    ref_date, prices_today = _nearest_trading_day(target_str, target_str)
    if not ref_date:
        st.warning("データが取得できませんでした。日付を変更してください。")
        st.stop()

    prev_target = (pd.Timestamp(ref_date) - timedelta(days=1)).strftime("%Y-%m-%d")
    prev_date, prices_prev = _nearest_trading_day(prev_target, ref_date)
    if not prev_date:
        st.warning("前営業日のデータが取得できませんでした。")
        st.stop()

    listed_stocks = _provider.get_listed_stocks()

st.caption(f"基準日: **{ref_date}** / 前営業日: **{prev_date}**")


# --- ランキング計算 ---
@st.cache_data(ttl=3600)
def _compute_rankings(ref_date_key: str, prev_date_key: str):
    """キャッシュ用ラッパー"""
    _, p_today = _nearest_trading_day(ref_date_key, ref_date_key)
    prev_t = (pd.Timestamp(ref_date_key) - timedelta(days=1)).strftime("%Y-%m-%d")
    _, p_prev = _nearest_trading_day(prev_t, ref_date_key)
    stocks = _provider.get_listed_stocks()
    return compute_eod_rankings(p_today, p_prev, stocks)


ranking_df = _compute_rankings(ref_date, prev_date)

# --- フィルタ適用 ---
filtered = ranking_df.copy()

# 一般株式のみ
_STOCK_SEGMENTS = {"プライム", "スタンダード", "グロース"}
filtered = filtered[filtered["market_name"].isin(_STOCK_SEGMENTS)]

if eod_market_segments:
    filtered = filtered[filtered["market_name"].isin(eod_market_segments)]

if eod_sectors:
    filtered = filtered[filtered["sector_17_name"].isin(eod_sectors)]

if eod_min_tv > 0:
    filtered = filtered[filtered["trading_value"] >= eod_min_tv * 1e6]

# --- 時価総額フィルタ（設定時のみ取得） ---
if not filtered.empty and (eod_cap_min > 0 or eod_cap_max > 0):
    with st.spinner("時価総額フィルタ適用中..."):
        all_caps = _fetch_market_caps_yf(tuple(sorted(filtered["code"].unique().tolist())))
        codes_with_cap = set()
        for code in filtered["code"].unique():
            cap = all_caps.get(code)
            if cap is None:
                continue
            if eod_cap_min > 0 and cap < eod_cap_min:
                continue
            if eod_cap_max > 0 and cap > eod_cap_max:
                continue
            codes_with_cap.add(code)
        filtered = filtered[filtered["code"].isin(codes_with_cap)]


# =========================================================================
# タブ表示
# =========================================================================

tab_overview, tab_return, tab_volume, tab_tv, tab_newhighlow, tab_compound = st.tabs([
    "概況", "騰落率", "出来高", "売買代金", "新高値・安値", "複合条件",
])


# --- TAB 1: 概況 ---
with tab_overview:
    ad = compute_advance_decline(prices_today, prices_prev)

    col1, col2, col3, col4 = st.columns(4)
    adv_color = "#2E7D32" if ad["advance"] > ad["decline"] else "#C62828"
    with col1:
        st.markdown(
            f'<div class="reuters-metric">'
            f'<div class="metric-value" style="color:{adv_color}">{ad["advance"]:,}</div>'
            f'<div class="metric-label">値上がり</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="reuters-metric">'
            f'<div class="metric-value" style="color:#C62828">{ad["decline"]:,}</div>'
            f'<div class="metric-label">値下がり</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="reuters-metric">'
            f'<div class="metric-value">{ad["unchanged"]:,}</div>'
            f'<div class="metric-label">変わらず</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        ratio_pct = ad["advance_ratio"] * 100
        r_color = "#2E7D32" if ratio_pct > 50 else "#C62828"
        st.markdown(
            f'<div class="reuters-metric">'
            f'<div class="metric-value" style="color:{r_color}">{ratio_pct:.1f}%</div>'
            f'<div class="metric-label">騰落比率</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # セクターヒートマップ
    st.subheader("セクター別騰落率")
    sector_summary = compute_sector_summary(filtered)

    if not sector_summary.empty:
        treemap_data = sector_summary.reset_index()
        treemap_data.columns = ["sector", "mean_return", "median_return", "advance", "decline", "count"]
        treemap_data["abs_return"] = treemap_data["mean_return"].abs()
        treemap_data["display"] = treemap_data.apply(
            lambda r: f"{r['sector']}<br>{r['mean_return']:+.2%}", axis=1
        )

        fig = px.treemap(
            treemap_data,
            path=["display"],
            values="count",
            color="mean_return",
            color_continuous_scale=["#C62828", "#FFFFFF", "#2E7D32"],
            color_continuous_midpoint=0,
        )
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=10))
        fig.update_coloraxes(colorbar_title="騰落率")
        st.plotly_chart(fig, width="stretch")

        st.page_link("pages/6_セクター別騰落率.py", label="セクター別騰落率ページへ", icon=":material/bar_chart:")

    st.divider()

    # 上場区分別
    st.subheader("上場区分別")
    segment_df = compute_market_segment_strength(filtered)
    if not segment_df.empty:
        seg_display = segment_df.reset_index()
        seg_display.columns = ["区分", "平均騰落率(%)", "値上がり", "値下がり", "銘柄数"]
        seg_display["平均騰落率(%)"] = (seg_display["平均騰落率(%)"] * 100).round(2)
        seg_display["値上がり"] = seg_display["値上がり"].astype(int)
        seg_display["値下がり"] = seg_display["値下がり"].astype(int)
        seg_display["銘柄数"] = seg_display["銘柄数"].astype(int)
        render_ranking_html(seg_display, ret_cols=["平均騰落率(%)"], max_height=200)


# --- TAB 2: 騰落率 ---
with tab_return:
    st.subheader("値上がり上位")
    tbl, ret_cols, chg_cols = _build_ranking_table(
        filtered, "daily_return", ascending=False,         show_volume_change=True,
    )
    render_ranking_html(tbl, ret_cols=ret_cols, change_cols=chg_cols, max_height=500)

    st.divider()

    st.subheader("値下がり上位")
    tbl2, ret_cols2, chg_cols2 = _build_ranking_table(
        filtered, "daily_return", ascending=True,         show_volume_change=True,
    )
    render_ranking_html(tbl2, ret_cols=ret_cols2, change_cols=chg_cols2, max_height=500)


# --- TAB 3: 出来高 ---
with tab_volume:
    st.subheader("出来高ランキング")
    tbl3, ret3, chg3 = _build_ranking_table(
        filtered, "volume", ascending=False,         show_volume_change=True,
    )
    render_ranking_html(tbl3, ret_cols=ret3, change_cols=chg3, max_height=500)

    st.divider()

    st.subheader("出来高急増銘柄")
    vol_surge = filtered.dropna(subset=["volume_change"])
    vol_surge = vol_surge[vol_surge["volume_change"] > 2.0]
    tbl4, ret4, chg4 = _build_ranking_table(
        vol_surge, "volume_change", ascending=False,         show_vol_surge=True,
    )
    render_ranking_html(tbl4, ret_cols=ret4, change_cols=chg4, max_height=500)


# --- TAB 4: 売買代金 ---
with tab_tv:
    st.subheader("売買代金ランキング")
    tbl5, ret5, chg5 = _build_ranking_table(
        filtered, "trading_value", ascending=False,         show_tv=True, show_tv_change=True,
    )
    render_ranking_html(tbl5, ret_cols=ret5, change_cols=chg5, max_height=500)


# --- TAB 5: 新高値・安値 ---
with tab_newhighlow:
    st.subheader("52週新高値・新安値")
    st.caption("過去252営業日の高値/安値と比較")

    @st.cache_data(ttl=3600)
    def _load_52w_data(ref_date_str: str):
        ref_ts = pd.Timestamp(ref_date_str)
        frames = []
        for months_back in range(1, 13):
            target = (ref_ts - pd.DateOffset(months=months_back)).strftime("%Y-%m-%d")
            d, df = _nearest_trading_day(target, ref_date_str)
            if d and not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    hist_data = _load_52w_data(ref_date)

    if not hist_data.empty:
        new_highs, new_lows = detect_new_highs_lows(prices_today, hist_data)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="reuters-metric">'
                f'<div class="metric-value" style="color:#2E7D32">{len(new_highs)}</div>'
                f'<div class="metric-label">新高値銘柄数</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="reuters-metric">'
                f'<div class="metric-value" style="color:#C62828">{len(new_lows)}</div>'
                f'<div class="metric-label">新安値銘柄数</div></div>',
                unsafe_allow_html=True,
            )

        stock_info = listed_stocks.set_index("code")

        if not new_highs.empty:
            st.subheader("新高値")
            nh = new_highs.copy()
            nh["銘柄名"] = nh["code"].map(stock_info["name"].to_dict())
            nh["セクター"] = nh["code"].map(stock_info["sector_17_name"].to_dict())
            nh.rename(columns={"code": "コード"}, inplace=True)
            render_ranking_html(nh[["コード", "銘柄名", "セクター"]], max_height=400)

        if not new_lows.empty:
            st.subheader("新安値")
            nl = new_lows.copy()
            nl["銘柄名"] = nl["code"].map(stock_info["name"].to_dict())
            nl["セクター"] = nl["code"].map(stock_info["sector_17_name"].to_dict())
            nl.rename(columns={"code": "コード"}, inplace=True)
            render_ranking_html(nl[["コード", "銘柄名", "セクター"]], max_height=400)
    else:
        st.info("52週の履歴データがまだ蓄積されていません。")


# --- TAB 6: 複合条件 ---
with tab_compound:
    st.subheader("複合条件フィルタ")
    st.caption("保存はされません。一時的な絞り込み用です。")

    preset = st.selectbox(
        "プリセット条件",
        [
            "カスタム",
            "値上がり + 出来高急増",
            "値下がり + 出来高急増",
            "値上がり + 高売買代金",
        ],
        key="eod_preset",
    )

    if preset == "値上がり + 出来高急増":
        conditions = [
            {"column": "daily_return", "operator": "gt", "value": 0.0},
            {"column": "volume_change", "operator": "gt", "value": 2.0},
        ]
    elif preset == "値下がり + 出来高急増":
        conditions = [
            {"column": "daily_return", "operator": "lt", "value": 0.0},
            {"column": "volume_change", "operator": "gt", "value": 2.0},
        ]
    elif preset == "値上がり + 高売買代金":
        conditions = [
            {"column": "daily_return", "operator": "gt", "value": 0.0},
            {"column": "trading_value", "operator": "gt", "value": 1e9},
        ]
    else:
        conditions = []

        num_conds = st.number_input("条件数", min_value=0, max_value=5, value=0, key="eod_ncond")
        available_cols = {
            "daily_return": "騰落率",
            "volume": "出来高",
            "volume_change": "出来高変化(前日比)",
            "trading_value": "売買代金",
            "trading_value_change": "売買代金変化(前日比)",
        }
        for ci in range(int(num_conds)):
            c1, c2, c3 = st.columns(3)
            with c1:
                col_key = st.selectbox(
                    "項目", list(available_cols.keys()),
                    format_func=lambda k: available_cols[k],
                    key=f"eod_ccol_{ci}",
                )
            with c2:
                op = st.selectbox("演算子", ["gt", "lt", "gte", "lte"], key=f"eod_cop_{ci}")
            with c3:
                val = st.number_input("値", value=0.0, key=f"eod_cval_{ci}")
            conditions.append({"column": col_key, "operator": op, "value": val})

    if conditions:
        compound_result = apply_compound_filter(filtered, conditions)
        st.write(f"**{len(compound_result)}銘柄** がヒット")

        if not compound_result.empty:
            tbl_c, retc, chgc = _build_ranking_table(
                compound_result, "daily_return", ascending=False,
                n=len(compound_result),                 show_volume_change=True, show_tv=True,
            )
            render_ranking_html(tbl_c, ret_cols=retc, change_cols=chgc, max_height=550)
