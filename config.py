"""アプリケーション設定"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# パス設定
BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"
DB_PATH = STORAGE_DIR / "research.db"
MARKET_DATA_DIR = STORAGE_DIR / "market_data"

# ディレクトリ作成
STORAGE_DIR.mkdir(exist_ok=True)
MARKET_DATA_DIR.mkdir(exist_ok=True)

# J-Quants API V2
JQUANTS_API_KEY = os.getenv("JQUANTS_API_KEY", "")

# 分析カテゴリ
ANALYSIS_CATEGORIES = [
    "カレンダー効果",
    "モメンタム",
    "バリュー",
    "リバーサル",
    "ボラティリティ",
    "セクターローテーション",
    "イベントドリブン",
    "テクニカル",
    "ファンダメンタル",
    "その他",
]

# デフォルト分析パラメータ
DEFAULT_ANALYSIS_PARAMS = {
    "lookback_years": 5,
    "max_lookback_years": 10,
    "significance_level": 0.05,
    "min_samples": 30,
}

# バックテスト設定
BACKTEST_DEFAULTS = {
    "initial_capital": 10_000_000,  # 1000万円
    "commission_rate": 0.001,  # 0.1%
    "slippage_rate": 0.001,  # 0.1%
    "rebalance_frequency": "monthly",  # daily, weekly, monthly
}

# AI研究エージェント設定
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "claude-sonnet-4-20250514")
AI_API_BASE_URL = os.getenv("AI_API_BASE_URL", "https://api.anthropic.com")
AI_MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", "4096"))

# コード実行設定
CODE_EXECUTION_TIMEOUT = int(os.getenv("CODE_EXECUTION_TIMEOUT", "120"))  # 秒
