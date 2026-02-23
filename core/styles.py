"""Reuters風UIテーマ — CSS注入 + HTMLヘルパー関数"""

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

/* Tab styling */
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    border-bottom-color: var(--r-accent) !important;
    color: var(--r-accent) !important;
}

/* Divider */
.stApp hr {
    border-color: var(--r-border);
}

/* Remove emoji-heavy default styling */
.stApp .stMarkdown a { color: var(--r-accent); }
</style>
"""
