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
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

# ディレクトリ作成
STORAGE_DIR.mkdir(exist_ok=True)
MARKET_DATA_DIR.mkdir(exist_ok=True)
for _d in [KNOWLEDGE_DIR, KNOWLEDGE_DIR / "alpha", KNOWLEDGE_DIR / "failed", KNOWLEDGE_DIR / "notes"]:
    _d.mkdir(exist_ok=True)

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

# AI研究エージェント設定（Claude Code CLI経由）
CLAUDE_CLI_TIMEOUT = int(os.getenv("CLAUDE_CLI_TIMEOUT", "600"))  # CLI呼び出しタイムアウト秒

# コード実行設定
CODE_EXECUTION_TIMEOUT = int(os.getenv("CODE_EXECUTION_TIMEOUT", "120"))  # 秒

# 標準バックテスト設定
STANDARD_BACKTEST_DEFAULTS = {
    # ユニバース
    "max_stocks": 50,
    # テクニカルシグナル
    "volume_surge_window": 20,
    "ma_deviation_window": 25,
    "rsi_window": 14,
    "bb_window": 20,
    "bb_std": 2.0,
    "atr_window": 14,
    "sector_relative_lookback": 20,
    # ポジション管理
    "holding_period_days": 20,
    "max_positions": 10,
    "allocation_method": "equal_weight",
    # コスト・資本
    "initial_capital": 10_000_000,
    "commission_rate": 0.001,
    "slippage_rate": 0.001,
    # シグナル結合
    "signal_logic": "AND",
}
