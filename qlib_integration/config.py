"""Qlib統合用の設定・定数"""

from pathlib import Path

from config import STORAGE_DIR

# データディレクトリ
QLIB_DATA_DIR = STORAGE_DIR / "qlib_data" / "jp_data"

# 実験結果保存ディレクトリ
QLIB_RESULTS_DIR = STORAGE_DIR / "qlib_data" / "results"

# データ変換デフォルト
DATA_START_DATE = "2017-01-01"
DATA_END_DATE = "2026-03-07"

# 実験期間デフォルト
TRAIN_PERIOD = ("2018-01-01", "2023-12-31")
VALID_PERIOD = ("2024-01-01", "2024-12-31")
TEST_PERIOD = ("2025-01-01", "2026-03-07")

# ポートフォリオ設定
TOPK = 30       # 保有銘柄数
N_DROP = 3      # 毎日の入替数
BENCHMARK = "0000"  # TOPIX

# bin変換対象フィールド
BIN_FIELDS = {
    "adj_close": "close",
    "adj_open": "open",
    "adj_high": "high",
    "adj_low": "low",
    "adj_volume": "volume",
    "adjustment_factor": "factor",
}

# LightGBMデフォルトパラメータ
LGB_DEFAULT_PARAMS = {
    "loss": "mse",
    "colsample_bytree": 0.8879,
    "learning_rate": 0.0421,
    "subsample": 0.8789,
    "lambda_l1": 205.6999,
    "lambda_l2": 580.9768,
    "max_depth": 8,
    "num_leaves": 210,
    "num_threads": 20,
}
