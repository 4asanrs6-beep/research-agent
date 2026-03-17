"""Reuters風UIテーマ — CSS注入 + HTMLヘルパー関数"""

from __future__ import annotations

import pandas as pd
import streamlit as st

# --- 配色定数 ---
COLOR_BG = "#FFFFFF"
COLOR_TEXT = "#1A1A2E"
COLOR_ACCENT = "#FF8000"
COLOR_ACCENT_LIGHT = "#FFF3E6"
COLOR_GRAY_LIGHT = "#F5F5F5"
COLOR_GRAY_BORDER = "#E0E0E0"
COLOR_GREEN = "#2E7D32"
COLOR_RED = "#C62828"
COLOR_BLUE = "#1565C0"


def apply_reuters_style() -> None:
    """Reuters風CSSを注入（各ページ冒頭で呼ぶ）"""
    st.markdown(_REUTERS_CSS, unsafe_allow_html=True)


def render_status_badge(label: str) -> str:
    """ステータスバッジのHTML文字列を返す"""
    style_map = {
        "valid": f"background:{COLOR_GREEN};color:#fff;",
        "invalid": f"background:{COLOR_RED};color:#fff;",
        "needs_review": f"background:{COLOR_ACCENT};color:#fff;",
        "running": f"background:{COLOR_BLUE};color:#fff;",
        "completed": f"background:{COLOR_GREEN};color:#fff;",
        "failed": f"background:{COLOR_RED};color:#fff;",
    }
    label_text_map = {
        "valid": "Valid",
        "invalid": "Invalid",
        "needs_review": "Review",
        "running": "Running",
        "completed": "Completed",
        "failed": "Failed",
    }
    style = style_map.get(label, f"background:{COLOR_GRAY_BORDER};color:{COLOR_TEXT};")
    text = label_text_map.get(label, label)
    return (
        f'<span style="{style}padding:2px 10px;border-radius:3px;'
        f'font-size:0.78em;font-weight:600;letter-spacing:0.03em;">'
        f"{text}</span>"
    )


def render_card(html: str, accent: bool = False) -> str:
    """カード風divラッパーのHTML文字列を返す"""
    border_color = COLOR_ACCENT if accent else COLOR_GRAY_BORDER
    return (
        f'<div class="reuters-card" style="border-left:3px solid {border_color};">'
        f"{html}</div>"
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_REUTERS_CSS = """\
<style>
/* ---------- Reuters-style global theme ---------- */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap');

/* Root variables */
:root {
    --r-accent: #FF8000;
    --r-text: #1A1A2E;
    --r-bg: #FFFFFF;
    --r-gray: #F5F5F5;
    --r-border: #E0E0E0;
}

/* Base */
.stApp {
    font-family: 'Noto Sans JP', 'Helvetica Neue', Arial, sans-serif;
    color: var(--r-text);
}

/* Headers with orange underline */
.stApp h1 {
    border-bottom: 3px solid var(--r-accent);
    padding-bottom: 0.35em;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.stApp h2, .stApp h3 {
    font-weight: 600;
}

/* Card component */
.reuters-card {
    background: var(--r-bg);
    border: 1px solid var(--r-border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    transition: box-shadow 0.15s;
}
.reuters-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.reuters-card h4 {
    margin: 0 0 0.3em 0;
    font-size: 1.05em;
    font-weight: 600;
}
.reuters-card .card-meta {
    font-size: 0.82em;
    color: #666;
    margin-bottom: 0.3em;
}
.reuters-card .card-metrics {
    display: flex;
    gap: 1.2rem;
    font-size: 0.88em;
    margin-top: 0.4em;
}
.reuters-card .card-metrics span {
    color: #555;
}
.reuters-card .card-metrics strong {
    color: var(--r-text);
}

/* Metric boxes */
.reuters-metric {
    background: var(--r-gray);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.reuters-metric .metric-value {
    font-size: 1.8em;
    font-weight: 700;
    color: var(--r-accent);
    line-height: 1.2;
}
.reuters-metric .metric-label {
    font-size: 0.82em;
    color: #666;
    margin-top: 0.2em;
}

/* Primary button override */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background-color: var(--r-accent) !important;
    border: none !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background-color: #E67300 !important;
}

/* Sidebar running indicator */
.sidebar-running {
    background: #FFF3E6;
    border: 1px solid var(--r-accent);
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.8rem;
    font-size: 0.85em;
}
.sidebar-running .pulse {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--r-accent);
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse-anim 1.2s infinite;
}
@keyframes pulse-anim {
    0%   { opacity: 1; }
    50%  { opacity: 0.3; }
    100% { opacity: 1; }
}

/* Progress bar accent */
.stProgress > div > div > div {
    background-color: var(--r-accent) !important;
}

/* Tab styling — institutional dark */
.stTabs [data-baseweb="tab-list"] button {
    font-size: 0.82em !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    border-bottom-color: var(--r-text) !important;
    color: var(--r-text) !important;
}

/* Expander — clean borders */
.stExpander {
    border: 1px solid var(--r-border) !important;
    border-radius: 2px !important;
}

/* Divider */
.stApp hr {
    border-color: var(--r-border);
}

/* Links — accent color */
.stApp .stMarkdown a { color: var(--r-accent); }

/* Metric value override — darker text */
[data-testid="stMetricValue"] {
    color: var(--r-text) !important;
    font-weight: 700 !important;
}

/* ---------- Ranking table ---------- */
.rank-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82em;
    line-height: 1.5;
    margin-bottom: 1rem;
}
.rank-table thead th {
    background: #1A1A2E;
    color: #F5F5F5;
    font-weight: 600;
    padding: 8px 12px;
    text-align: left;
    white-space: nowrap;
    border-bottom: 2px solid var(--r-accent);
    letter-spacing: 0.02em;
    font-size: 0.92em;
}
.rank-table thead th.num {
    text-align: right;
}
.rank-table tbody tr {
    border-bottom: 1px solid #EBEBEB;
    transition: background 0.12s;
}
.rank-table tbody tr:nth-child(even) {
    background: #FAFAFA;
}
.rank-table tbody tr:hover {
    background: #FFF3E6;
}
.rank-table tbody td {
    padding: 6px 12px;
    white-space: nowrap;
}
.rank-table tbody td.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
}
.rank-table tbody td.rank-cell {
    text-align: center;
    font-weight: 700;
    color: #999;
    width: 36px;
}
.rank-table tbody td.code-cell {
    font-weight: 600;
    color: #1565C0;
}
.rank-table tbody td.name-cell {
    max-width: 160px;
    overflow: hidden;
    text-overflow: ellipsis;
}
.rank-table .val-pos {
    color: #2E7D32;
    font-weight: 600;
}
.rank-table .val-neg {
    color: #C62828;
    font-weight: 600;
}
.rank-table .val-hot {
    color: #FF8000;
    font-weight: 700;
}
.rank-table .val-muted {
    color: #999;
}
.rank-table .tag-sector {
    background: #F0F0F0;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 0.88em;
    color: #555;
}
.rank-table .tag-market {
    background: #E8EAF6;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 0.88em;
    color: #3949AB;
}

/* Waiting overlay — dim page header while background task runs */
.waiting-overlay .stApp h1,
.waiting-overlay .stApp .stCaption {
    opacity: 0.4;
}
.waiting-dimmed h1,
.waiting-dimmed [data-testid="stCaptionContainer"] {
    opacity: 0.4;
    pointer-events: none;
}
</style>
"""


def render_ranking_html(
    df: pd.DataFrame,
    *,
    ret_cols: list[str] | None = None,
    change_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    max_height: int = 500,
) -> None:
    """DataFrameをプロ仕様のHTMLランキングテーブルとして描画する。

    Parameters
    ----------
    df : 表示するDataFrame（整形済み）
    ret_cols : 騰落率列名リスト（+/- で色分け）
    change_cols : 変化倍率列名リスト（2x超でオレンジ、1.5x超で緑）
    num_cols : 右寄せにする数値列名リスト
    max_height : テーブルの最大高さ（px）
    """
    if df.empty:
        st.info("該当データなし")
        return

    ret_cols = ret_cols or []
    change_cols = change_cols or []
    num_cols = num_cols or []

    # 自動検出: 数値列は右寄せ候補
    auto_num = set(num_cols) | set(ret_cols)
    for col in df.columns:
        if col in ("#", "コード", "銘柄名", "市場", "セクター"):
            continue
        auto_num.add(col)

    # ヘッダー生成
    header_cells = []
    for col in df.columns:
        cls = ' class="num"' if col in auto_num else ""
        header_cells.append(f"<th{cls}>{col}</th>")
    thead = "<thead><tr>" + "".join(header_cells) + "</tr></thead>"

    # ボディ生成
    tbody_rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]

            # ランク列
            if col == "#":
                cells.append(f'<td class="rank-cell">{val}</td>')
                continue

            # コード列
            if col == "コード":
                cells.append(f'<td class="code-cell">{val}</td>')
                continue

            # 銘柄名列
            if col == "銘柄名":
                name_str = str(val) if pd.notna(val) else ""
                cells.append(f'<td class="name-cell">{name_str}</td>')
                continue

            # セクター列
            if col == "セクター":
                s = str(val) if pd.notna(val) else ""
                cells.append(f'<td><span class="tag-sector">{s}</span></td>')
                continue

            # 市場列
            if col == "市場":
                m = str(val) if pd.notna(val) else ""
                cells.append(f'<td><span class="tag-market">{m}</span></td>')
                continue

            # 騰落率列
            if col in ret_cols:
                if pd.isna(val):
                    cells.append('<td class="num val-muted">—</td>')
                elif isinstance(val, (int, float)):
                    cls = "val-pos" if val > 0 else ("val-neg" if val < 0 else "")
                    cells.append(f'<td class="num {cls}">{val:+.2f}%</td>')
                else:
                    cells.append(f'<td class="num">{val}</td>')
                continue

            # 変化倍率列 (前日比, 急増倍率)
            if col in change_cols:
                if pd.isna(val):
                    cells.append('<td class="num val-muted">—</td>')
                elif isinstance(val, str):
                    # Already formatted like "2.5x"
                    try:
                        num_val = float(val.replace("x", ""))
                        cls = "val-hot" if num_val > 2.0 else ("val-pos" if num_val > 1.5 else "")
                        cells.append(f'<td class="num {cls}">{val}</td>')
                    except ValueError:
                        cells.append(f'<td class="num">{val}</td>')
                else:
                    cells.append(f'<td class="num">{val}</td>')
                continue

            # 一般数値列
            if col in auto_num:
                if pd.isna(val):
                    cells.append('<td class="num val-muted">—</td>')
                else:
                    cells.append(f'<td class="num">{val}</td>')
                continue

            # テキスト列
            cells.append(f"<td>{val if pd.notna(val) else ''}</td>")

        tbody_rows.append("<tr>" + "".join(cells) + "</tr>")

    tbody = "<tbody>" + "".join(tbody_rows) + "</tbody>"

    html = (
        f'<div style="max-height:{max_height}px;overflow-y:auto;border:1px solid #E0E0E0;'
        f'border-radius:4px;">'
        f'<table class="rank-table">{thead}{tbody}</table></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def apply_waiting_overlay() -> None:
    """待機中のページヘッダーを薄く表示するCSSを注入する"""
    st.markdown(
        """<style>
        .stApp h1, .stApp [data-testid="stCaptionContainer"] {
            opacity: 0.4;
            pointer-events: none;
        }
        </style>""",
        unsafe_allow_html=True,
    )
