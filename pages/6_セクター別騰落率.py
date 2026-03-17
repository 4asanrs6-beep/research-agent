"""セクター別騰落率ページ — TOPIX 17業種 / 33業種を期間別テーブル＋棒グラフで比較

【高速化戦略】
  get_price_daily_by_date(date) → 必要な日付だけ取得
  各日付はDataCacheに個別永続保存されるため、過去データは2回目以降ほぼ瞬時。
"""

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import JQUANTS_API_KEY, MARKET_DATA_DIR
from core.sidebar import render_sidebar_running_indicator
from core.styles import apply_reuters_style
from core.universe_filter import MARKET_SEGMENTS, SECTOR_17_LIST, SECTOR_33_LIST
from data.cache import DataCache
from data.jquants_provider import JQuantsProvider

st.set_page_config(page_title="セクター別騰落率", page_icon="R", layout="wide")
apply_reuters_style()
render_sidebar_running_indicator()

_cache = DataCache(MARKET_DATA_DIR)
_provider = JQuantsProvider(api_key=JQUANTS_API_KEY, cache=_cache)


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


@st.cache_data(ttl=3600)
def load_sector_returns_table(
    end_date: str,
    use_custom: bool,
    custom_start: str,
    market_segments_tuple: tuple,
    sector_type: str = "17",
) -> tuple[pd.DataFrame, str]:
    sector_col = "sector_17_name" if sector_type == "17" else "sector_33_name"
    sector_list = SECTOR_17_LIST if sector_type == "17" else SECTOR_33_LIST

    stocks_df = _provider.get_listed_stocks()
    sector_map = stocks_df.set_index("code")[sector_col].to_dict()
    market_map = (
        stocks_df.set_index("code")["market_name"].to_dict()
        if "market_name" in stocks_df.columns
        else {}
    )

    def filter_df(raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df["sector"] = df["code"].map(sector_map)
        df["market_name"] = df["code"].map(market_map)
        df = df[df["sector"].isin(sector_list)]
        if market_segments_tuple:
            df = df[df["market_name"].isin(set(market_segments_tuple))]
        return df[["code", "adj_close", "sector"]].dropna()

    def calc_return(start_df: pd.DataFrame, end_df: pd.DataFrame) -> pd.Series:
        s = start_df.set_index("code")["adj_close"]
        e = end_df.set_index("code")["adj_close"]
        common = s.index.intersection(e.index)
        if common.empty:
            return pd.Series(dtype=float)
        ret = (e[common] / s[common] - 1) * 100
        sec = end_df.set_index("code")["sector"].reindex(common)
        return (
            pd.DataFrame({"ret": ret, "sector": sec})
            .dropna()
            .groupby("sector")["ret"]
            .mean()
        )

    ref_date_str, ref_raw = _nearest_trading_day(end_date, end_date)
    if not ref_date_str:
        return pd.DataFrame(), ""
    ref_df = filter_df(ref_raw)
    ref_ts = pd.Timestamp(ref_date_str)

    result = pd.DataFrame(index=sector_list)
    result.index.name = "セクター"

    d1_str, raw1d = _nearest_trading_day(
        (ref_ts - timedelta(days=1)).strftime("%Y-%m-%d"), ref_date_str
    )
    if not raw1d.empty:
        result["1日"] = calc_return(filter_df(raw1d), ref_df).reindex(sector_list)

        # 前日比: 2営業日前→1営業日前の変動
        if d1_str:
            d1_ts = pd.Timestamp(d1_str)
            _, raw2d = _nearest_trading_day(
                (d1_ts - timedelta(days=1)).strftime("%Y-%m-%d"), d1_str
            )
            if not raw2d.empty:
                result["前日騰落"] = calc_return(
                    filter_df(raw2d), filter_df(raw1d)
                ).reindex(sector_list)

    _, raw5d = _nearest_trading_day(
        (ref_ts - timedelta(days=7)).strftime("%Y-%m-%d"), ref_date_str
    )
    if not raw5d.empty:
        result["5日"] = calc_return(filter_df(raw5d), ref_df).reindex(sector_list)

    target_1m = (ref_ts - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    _, raw1m = _nearest_trading_day(target_1m, ref_date_str)
    if not raw1m.empty:
        result["1ヶ月"] = calc_return(filter_df(raw1m), ref_df).reindex(sector_list)

    if use_custom and custom_start:
        _, rawc = _nearest_trading_day(custom_start, ref_date_str)
        if not rawc.empty:
            result[f"カスタム({custom_start}〜)"] = (
                calc_return(filter_df(rawc), ref_df).reindex(sector_list)
            )

    return result, ref_date_str


# =============================================================================
# サイドバー
# =============================================================================

with st.sidebar:
    st.header("設定")

    sector_type = st.radio(
        "業種分類",
        ["17業種", "33業種"],
        horizontal=True,
    )
    sector_type_key = "17" if sector_type == "17業種" else "33"

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
    st.caption("セクター名クリック → 銘柄詳細")
    st.caption("列ヘッダークリック → ソート")
    st.caption("データソース: J-Quants API")

# =============================================================================
# メインエリア
# =============================================================================

st.title("セクター別騰落率")

if not JQUANTS_API_KEY:
    st.error("J-Quants APIキーが未設定（`.env` に `JQUANTS_API_KEY` を設定）")
    st.stop()

end_str = end_date.strftime("%Y-%m-%d")
custom_start_str = custom_start.strftime("%Y-%m-%d") if use_custom else ""
markets_tuple = (
    tuple(sorted(selected_markets)) if selected_markets else tuple(sorted(MARKET_SEGMENTS))
)

with st.spinner("データ読み込み中..."):
    try:
        result_df, ref_date_str = load_sector_returns_table(
            end_str, use_custom, custom_start_str, markets_tuple, sector_type_key
        )
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        st.stop()

if result_df is None or result_df.empty:
    st.warning("データが取得できませんでした。期間や市場区分を変更してください。")
    st.stop()

st.caption(f"基準日: **{ref_date_str}**")

# --- テーブル ---
numeric_cols = result_df.columns.tolist()

# セクター列をリンクURLに変換してから reset_index
display_df = result_df.reset_index()
display_df["セクター"] = display_df["セクター"].apply(
    lambda s: f"セクター詳細?sector={s}&sector_type={sector_type_key}"
)


def _color(val) -> str:
    if pd.isna(val):
        return "color: #999"
    if val > 0:
        return "color: #2E7D32; font-weight: 600"
    if val < 0:
        return "color: #C62828; font-weight: 600"
    return "color: #555"


try:
    styled = display_df.style.map(_color, subset=numeric_cols)
except AttributeError:
    styled = display_df.style.applymap(_color, subset=numeric_cols)

# column_config を動的に構築
col_config: dict = {
    "セクター": st.column_config.LinkColumn(
        "セクター",
        display_text=r"sector=([^&]+)",  # URL から sector 名を抽出して表示
        width="medium",
    ),
}
for col in numeric_cols:
    col_config[col] = st.column_config.NumberColumn(col, format="%+.2f%%")

st.dataframe(
    styled,
    column_config=col_config,
    width="stretch",
    height=min(660, max(400, len(result_df) * 36 + 60)),
    hide_index=True,
)

st.divider()

# --- チャート ---
st.subheader("チャート")

chart_col = st.radio(
    "表示する期間",
    numeric_cols,
    horizontal=True,
    label_visibility="collapsed",
)

bar_data = result_df[chart_col].dropna().sort_values()
colors = ["#C62828" if v < 0 else "#2E7D32" for v in bar_data.values]

fig = go.Figure(
    go.Bar(
        x=bar_data.values,
        y=bar_data.index.tolist(),
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in bar_data.values],
        textposition="outside",
        textfont=dict(size=11),
    )
)

x_abs = max(abs(bar_data.min()), abs(bar_data.max())) * 1.4 if not bar_data.empty else 5
fig.update_layout(
    height=max(520, len(bar_data) * 26 + 80),
    xaxis=dict(
        title="騰落率（%）",
        tickformat="+.1f",
        range=[-x_abs, x_abs],
        zeroline=True,
        zerolinecolor="#333",
        zerolinewidth=1.5,
    ),
    yaxis=dict(tickfont=dict(size=12)),
    margin=dict(l=10, r=120, t=10, b=40),
)

st.plotly_chart(fig, width="stretch")
