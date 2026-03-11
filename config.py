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
CLAUDE_CLI_MODEL = os.getenv("CLAUDE_CLI_MODEL", "claude-opus-4-6")  # 計画生成・結果解釈用
CLAUDE_CLI_CODE_MODEL = os.getenv("CLAUDE_CLI_CODE_MODEL", "sonnet")  # コード生成用（高速）
CLAUDE_CLI_TIMEOUT = int(os.getenv("CLAUDE_CLI_TIMEOUT", "600"))  # CLI呼び出しタイムアウト秒
CLAUDE_CLI_MODEL = os.getenv("CLAUDE_CLI_MODEL", "sonnet")  # 使用モデル（sonnet/opus/haiku）

# コード実行設定
CODE_EXECUTION_TIMEOUT = int(os.getenv("CODE_EXECUTION_TIMEOUT", "120"))  # 秒

# 標準バックテスト設定
# AI研究イテレーション設定
AI_RESEARCH_PHASE1_CONFIGS = 5              # Phase 1: AI構造探索
AI_RESEARCH_GRID_MAX_COMBINATIONS = 50      # Phase 3: グリッド最大組み合わせ数
AI_RESEARCH_GRID_MAX_PARAMS = 3             # Phase 2: AIが指定する最大パラメータ数
AI_RESEARCH_GRID_MAX_VALUES_PER_PARAM = 6   # Phase 2: パラメータあたり最大候補値数
AI_RESEARCH_MIN_SIGNALS = 20        # 最小シグナル数（これ未満なら条件緩和を促す）
AI_RESEARCH_MAX_STOCKS = 4000        # AI研究で使用する最大銘柄数（実質上限なし）

STAR_STOCK_DEFAULTS = {
    "min_total_return": 0.50,
    "min_excess_return": 0.30,
    "min_volume_increase_ratio": 1.5,
    "max_auto_detect": 50,
    # 仕手株フィルター
    "min_market_cap_billion": 50.0,
    "max_drawdown_from_peak": 0.40,
    "max_single_day_return": 0.20,
    "min_up_days_ratio": 0.45,
    "require_positive_end": True,
    # 高度分析パラメータ
    "rolling_beta_window": 60,
    "volume_surge_threshold": 2.0,
    "accumulation_price_threshold": 0.005,
    "accumulation_volume_threshold": 1.5,
    "obv_trend_window": 60,
    "sector_correlation_window": 60,
    "factor_window": 60,
    "vpin_bucket_size": 20,
    "lead_lag_max_lag": 10,
    "n_clusters": 4,
    # 反復的特徴量発見パラメータ
    "discovery_max_iterations": 5,
    "discovery_target_precision": 0.20,
    "discovery_min_recall": 0.30,
    "discovery_neg_sample_size": 200,
    "onset_min_forward_return": 0.10,
    "onset_max_candidates": 5,
}

ANOMALY_DEFAULTS = {
    "forward_eval_days": [1, 3, 5, 10, 20],
    "default_logic": "AND",
    "template_rules": [
        {
            "name": "出来高ブレイクアウト",
            "description": "出来高急増 + レンジ上限ブレイク",
            "conditions": [
                {"feature_key": "vol_ratio_5d_20d", "operator": "gt", "value": 2.0},
                {"feature_key": "range_position_20d", "operator": "gt", "value": 0.8},
            ],
            "logic": "AND",
        },
        {
            "name": "ボラ圧縮後の上放れ",
            "description": "ボリンジャーバンド収縮後に上方ブレイク",
            "conditions": [
                {"feature_key": "bb_width_pctile_60d", "operator": "lt", "value": 20.0},
                {"feature_key": "ret_5d", "operator": "gt", "value": 0.05},
            ],
            "logic": "AND",
        },
        {
            "name": "セクター逆行高",
            "description": "セクターに逆行して強い銘柄",
            "conditions": [
                {"feature_key": "sector_rel_ret_10d", "operator": "gt", "value": 0.05},
                {"feature_key": "vol_ratio_5d_20d", "operator": "gt", "value": 1.5},
            ],
            "logic": "AND",
        },
        {
            "name": "大型株の異常出来高",
            "description": "売買回転急増 + ATR低下（静かな異常出来高）",
            "conditions": [
                {"feature_key": "turnover_change_5d_20d", "operator": "gt", "value": 1.5},
                {"feature_key": "atr_ratio_5d_20d", "operator": "lt", "value": 0.8},
            ],
            "logic": "AND",
        },
    ],
}

STANDARD_BACKTEST_DEFAULTS = {
    # ユニバース
    "max_stocks": 4000,
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
