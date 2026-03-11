"""セクター詳細ページ — セクター内銘柄の騰落率・時価総額一覧

ソート: 列ヘッダーをクリック
時価総額: Yahoo Finance (yfinance) から取得（J-Quants APIにないため）
"""

import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import JQUANTS_API_KEY, MARKET_DATA_DIR
from core.sidebar import render_sidebar_running_indicator
from core.styles import apply_reuters_style
from core.universe_filter import MARKET_SEGMENTS, SECTOR_17_LIST
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider

st.set_page_config(page_title="セクター詳細", page_icon="R", layout="wide")
apply_reuters_style()
render_sidebar_running_indicator()

_cache = DataCache(MARKET_DATA_DIR)
_provider = JQuantsProvider(api_key=JQUANTS_API_KEY, cache=_cache)


# =============================================================================
# データ取得
# =============================================================================

def _nearest_trading_day(target_str: str, upper_bound: str, max_tries: int = 8):
    target_ts = pd.Timestamp(target_str)
    for i in range(max_tries):
        candidate = (target_ts - timedelta(days=i)).strftime("%Y-%m-%d")
        if candidate > upper_bound:
            continue
        df = _provider.get_price_daily_by_date(candidate)
        if not df.empty:
            return candidate, df
    return None, pd.DataFrame()


@st.cache_data(ttl=3600)
def load_stock_returns(
    sector: str,
    end_date: str,
    use_custom: bool,
    custom_start: str,
    market_segments_tuple: tuple,
) -> tuple[pd.DataFrame, str]:
    """セクター内銘柄の騰落率テーブルを計算する（時価総額は別途取得）。

    Returns:
        result_df   : index=コード, columns=[銘柄名, 1日, 5日, 1ヶ月, ...]
        ref_date_str: 基準日
    """
    stocks_df = _provider.get_listed_stocks()

    mask = stocks_df["sector_17_name"] == sector
    if market_segments_tuple and "market_name" in stocks_df.columns:
        mask &= stocks_df["market_name"].isin(set(market_segments_tuple))
    sector_stocks = stocks_df[mask].copy()

    if sector_stocks.empty:
        return pd.DataFrame(), ""

    sector_codes = set(sector_stocks["code"].astype(str))
    name_map = (
        sector_stocks.set_index("code")["name"].to_dict()
        if "name" in sector_stocks.columns
        else {}
    )
    scale_map = (
        sector_stocks.set_index("code")["scale_category"].to_dict()
        if "scale_category" in sector_stocks.columns
        else {}
    )

    def filter_to_sector(raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["code"] = df["code"].astype(str)
        df = df[df["code"].isin(sector_codes)]
        return df[["code", "adj_close"]].dropna()

    def calc_ret(start_df: pd.DataFrame, end_df: pd.DataFrame) -> pd.Series:
        s = start_df.set_index("code")["adj_close"]
        e = end_df.set_index("code")["adj_close"]
        common = s.index.intersection(e.index)
        if common.empty:
            return pd.Series(dtype=float)
        return (e[common] / s[common] - 1) * 100

    ref_date_str, ref_raw = _nearest_trading_day(end_date, end_date)
    if not ref_date_str:
        return pd.DataFrame(), ""
    ref_df = filter_to_sector(ref_raw)
    ref_ts = pd.Timestamp(ref_date_str)

    result = pd.DataFrame(index=sorted(sector_codes))
    result.index.name = "コード"
    result["銘柄名"] = pd.Series(name_map)
    result["規模区分"] = pd.Series(scale_map)

    d1_str, raw1d = _nearest_trading_day(
        (ref_ts - timedelta(days=1)).strftime("%Y-%m-%d"), ref_date_str
    )
    if not raw1d.empty:
        result["1日"] = calc_ret(filter_to_sector(raw1d), ref_df)

        # 前日比: 2営業日前→1営業日前の変動
        if d1_str:
            d1_ts = pd.Timestamp(d1_str)
            _, raw2d = _nearest_trading_day(
                (d1_ts - timedelta(days=1)).strftime("%Y-%m-%d"), d1_str
            )
            if not raw2d.empty:
                result["前日騰落"] = calc_ret(
                    filter_to_sector(raw2d), filter_to_sector(raw1d)
                )

    _, raw5d = _nearest_trading_day(
        (ref_ts - timedelta(days=7)).strftime("%Y-%m-%d"), ref_date_str
    )
    if not raw5d.empty:
        result["5日"] = calc_ret(filter_to_sector(raw5d), ref_df)

    target_1m = (ref_ts - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    _, raw1m = _nearest_trading_day(target_1m, ref_date_str)
    if not raw1m.empty:
        result["1ヶ月"] = calc_ret(filter_to_sector(raw1m), ref_df)

    if use_custom and custom_start:
        _, rawc = _nearest_trading_day(custom_start, ref_date_str)
        if not rawc.empty:
            result[f"カスタム({custom_start}〜)"] = calc_ret(
                filter_to_sector(rawc), ref_df
            )

    ret_cols = [
        c for c in result.columns if c not in ("銘柄名", "規模区分")
    ]
    result = result.dropna(subset=ret_cols, how="all")
    return result, ref_date_str


@st.cache_data(ttl=3600 * 6)
def fetch_market_caps_yf(codes: tuple) -> pd.Series:
    """Yahoo Finance から時価総額（億円）を並列取得する。

    - 最大10並列で HTTP リクエストを発行
    - TTL 6時間でキャッシュ（時価総額は頻繁に変わらないため）
    - 取得失敗コードは NaN
    """
    import yfinance as yf

    def _get_cap(code: str) -> tuple[str, float | None]:
        try:
            fi = yf.Ticker(f"{code[:4]}.T").fast_info
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


# =============================================================================
# スタイリング（page 6 と同方式）
# =============================================================================

def _color(val) -> str:
    if pd.isna(val):
        return "color: #999"
    if val > 0:
        return "color: #2E7D32; font-weight: 600"
    if val < 0:
        return "color: #C62828; font-weight: 600"
    return "color: #555"


def _styler_color(styler, cols: list[str]):
    try:
        return styler.map(_color, subset=cols)
    except AttributeError:
        return styler.applymap(_color, subset=cols)


# =============================================================================
# サイドバー
# =============================================================================

with st.sidebar:
    st.header("設定")

    today = date.today()
    end_date = st.date_input("基準日", value=today, max_value=today)

    selected_markets = st.multiselect(
        "市場区分",
        options=MARKET_SEGMENTS,
        default=MARKET_SEGMENTS,
    )

    st.divider()

    with st.expander("カスタム期間を追加"):
        use_custom = st.checkbox("カスタム期間列を表示", value=False)
        custom_max = end_date - timedelta(days=1)
        custom_default = min(today - timedelta(days=90), custom_max)
        custom_start = st.date_input(
            "開始日",
            value=custom_default,
            max_value=custom_max,
            disabled=not use_custom,
        )

    st.divider()
    st.caption("列ヘッダークリック → ソート")
    st.caption("データソース: J-Quants / Yahoo Finance")

# =============================================================================
# メインエリア
# =============================================================================

# セクター解決: query_params → session_state → セレクタ
sector = st.query_params.get("sector", "") or st.session_state.pop("detail_sector", "")

if not sector or sector not in SECTOR_17_LIST:
    st.title("セクター詳細")
    sector = st.selectbox("セクターを選択", SECTOR_17_LIST)
    if not sector:
        st.stop()

# 戻るリンク
back_url = f"/{urllib.parse.quote('セクター別騰落率')}"
st.markdown(
    f'<a href="{back_url}" style="color:#1565C0;font-size:13px;text-decoration:none;">'
    f"← セクター別騰落率に戻る</a>",
    unsafe_allow_html=True,
)
st.title(sector)

if not JQUANTS_API_KEY:
    st.error("J-Quants APIキーが未設定（`.env` に `JQUANTS_API_KEY` を設定）")
    st.stop()

end_str = end_date.strftime("%Y-%m-%d")
custom_start_str = custom_start.strftime("%Y-%m-%d") if use_custom else ""
markets_tuple = (
    tuple(sorted(selected_markets)) if selected_markets else tuple(sorted(MARKET_SEGMENTS))
)

# ── 騰落率テーブル読み込み（DataCache 済みなら瞬時）──────────────────────
with st.spinner(f"{sector} の騰落率を読み込み中..."):
    try:
        result_df, ref_date_str = load_stock_returns(
            sector, end_str, use_custom, custom_start_str, markets_tuple
        )
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        st.stop()

if result_df is None or result_df.empty:
    st.warning("データが取得できませんでした。")
    st.stop()

# ── 時価総額を Yahoo Finance から取得（並列 / 6h キャッシュ）────────────
codes_tuple = tuple(sorted(result_df.index.tolist()))
with st.spinner("時価総額を Yahoo Finance から取得中..."):
    try:
        market_caps = fetch_market_caps_yf(codes_tuple)
        result_df["時価総額(億円)"] = market_caps.reindex(result_df.index)
    except Exception as e:
        st.warning(f"時価総額の取得に失敗しました: {e}")
        result_df["時価総額(億円)"] = pd.NA

st.caption(
    f"基準日: **{ref_date_str}**　|　{len(result_df)} 銘柄　"
    f"|　列ヘッダークリックでソート"
)

# ── テーブル表示 ────────────────────────────────────────────────────────
ret_cols = [
    c for c in result_df.columns
    if c not in ("銘柄名", "規模区分", "時価総額(億円)")
]

def _fmt_ret(v):
    return f"{v:+.2f}%" if pd.notna(v) else "—"

def _fmt_cap(v):
    if pd.isna(v):
        return "—"
    if v >= 10000:
        return f"{v / 10000:.1f}兆円"
    return f"{v:,.0f}億円"

fmt_dict = {c: _fmt_ret for c in ret_cols}
fmt_dict["時価総額(億円)"] = _fmt_cap

styled = _styler_color(
    result_df.style.format(fmt_dict),
    ret_cols,
)

st.dataframe(styled, width="stretch", height=700)

st.divider()

# ── チャート ─────────────────────────────────────────────────────────────
st.subheader("騰落率チャート")

chart_col = st.radio(
    "表示する期間",
    ret_cols,
    horizontal=True,
    label_visibility="collapsed",
)

chart_data = result_df[chart_col].dropna().sort_values()

# 銘柄が多い場合は上位・下位10件に絞る
if len(chart_data) > 30:
    chart_data = pd.concat([chart_data.head(10), chart_data.tail(10)]).sort_values()
    st.caption("30銘柄超のため上位10件・下位10件を表示")

bar_colors = ["#C62828" if v < 0 else "#2E7D32" for v in chart_data.values]
labels = [
    f"{code}  {str(result_df.at[code, '銘柄名'])[:10] if pd.notna(result_df.at[code, '銘柄名']) else ''}"
    for code in chart_data.index
]

fig = go.Figure(
    go.Bar(
        x=chart_data.values,
        y=labels,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.2f}%" for v in chart_data.values],
        textposition="outside",
        textfont=dict(size=10),
    )
)

x_abs = max(abs(chart_data.min()), abs(chart_data.max())) * 1.4 if not chart_data.empty else 5
fig.update_layout(
    height=max(420, len(chart_data) * 26 + 80),
    xaxis=dict(
        title="騰落率（%）",
        tickformat="+.1f",
        range=[-x_abs, x_abs],
        zeroline=True,
        zerolinecolor="#333",
        zerolinewidth=1.5,
    ),
    yaxis=dict(tickfont=dict(size=11)),
    margin=dict(l=10, r=120, t=10, b=40),
)

st.plotly_chart(fig, width="stretch")
