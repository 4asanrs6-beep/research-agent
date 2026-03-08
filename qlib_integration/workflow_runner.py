"""Qlib実験オーケストレータ — Alpha158 + LightGBM を1本通すワークフロー"""

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from qlib_integration.config import (
    QLIB_DATA_DIR,
    TRAIN_PERIOD,
    VALID_PERIOD,
    TEST_PERIOD,
    TOPK,
    N_DROP,
    LGB_DEFAULT_PARAMS,
)

logger = logging.getLogger(__name__)

_qlib_initialized = False

# Alpha158の158特徴量名（Qlibソースから確認済みの正確な順序）
_ALPHA158_NAMES: list[str] = []
# K線特徴量 (9)
_ALPHA158_NAMES += ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
# 価格特徴量 (4)
_ALPHA158_NAMES += ["OPEN0", "HIGH0", "LOW0", "VWAP0"]
# ローリング特徴量 (29カテゴリ × 5ウィンドウ = 145)
_ROLLING_CATEGORIES = [
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI",
    "MAX", "MIN", "QTLU", "QTLD", "RANK", "RSV",
    "IMAX", "IMIN", "IMXD",
    "CORR", "CORD",
    "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD",
    "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD",
]
for _cat in _ROLLING_CATEGORIES:
    for _w in [5, 10, 20, 30, 60]:
        _ALPHA158_NAMES.append(f"{_cat}{_w}")


def init_qlib_jp(data_dir: Path | None = None) -> None:
    """Qlibを日本株データで初期化する。"""
    global _qlib_initialized
    if _qlib_initialized:
        return

    if data_dir is None:
        data_dir = QLIB_DATA_DIR

    cal_path = data_dir / "calendars" / "day.txt"
    if not cal_path.exists():
        raise FileNotFoundError(
            f"Qlibデータが見つかりません: {cal_path}\n"
            "先にデータ変換を実行してください。"
        )

    import qlib
    try:
        qlib.init(provider_uri=str(data_dir))
    except Exception as e:
        if "reinitialize" in str(e).lower() or "already activated" in str(e).lower():
            logger.info("Qlib既に初期化済み（スキップ）")
        else:
            raise
    _qlib_initialized = True
    logger.info("Qlib初期化完了: %s", data_dir)


def _build_alpha158_jp_handler(
    instruments: str,
    start_time: str,
    end_time: str,
    fit_start_time: str,
    fit_end_time: str,
):
    """日本株用Alpha158 DataHandlerを構築する。

    ラベル: Ref($close, -1) / $close - 1 (中国のT+1売買制限が日本にはないため1日先リターン)
    """
    from qlib.contrib.data.handler import Alpha158

    label_config = [
        ["Ref($close, -1) / $close - 1"],
        ["LABEL0"],
    ]

    handler = Alpha158(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        label=label_config,
    )
    return handler


def _write_filtered_instruments(data_dir: Path, codes: list[str]) -> str:
    """フィルタ済み銘柄リストをQlib instruments形式で書き出す。

    instruments/filtered.txt を生成し、instrument名 "filtered" を返す。
    """
    inst_dir = data_dir / "instruments"
    all_path = inst_dir / "all.txt"

    # all.txt から対象銘柄の行だけ抽出
    code_set = set(codes)
    lines = []
    for line in all_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if parts[0] in code_set:
            lines.append(line)

    filtered_path = inst_dir / "filtered.txt"
    filtered_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("フィルタ済み銘柄リスト: %d / %d銘柄", len(lines), len(code_set))
    return "filtered"


def run_experiment(
    data_dir: Path | None = None,
    model_type: str = "lgb",
    train_period: tuple[str, str] | None = None,
    valid_period: tuple[str, str] | None = None,
    test_period: tuple[str, str] | None = None,
    topk: int = TOPK,
    n_drop: int = N_DROP,
    lgb_params: dict | None = None,
    instrument_codes: list[str] | None = None,
    on_progress: Callable[[str, float], None] | None = None,
) -> dict:
    """Qlib実験を実行する。

    Args:
        data_dir: Qlibデータディレクトリ
        model_type: "lgb" (LightGBM) or "xgb" (XGBoost)
        train_period: 学習期間 (start, end)
        valid_period: 検証期間
        test_period: テスト期間
        topk: TopK銘柄数
        n_drop: 毎日入替数
        lgb_params: LightGBMパラメータ
        instrument_codes: フィルタ済み銘柄コードリスト (Noneなら全銘柄)
        on_progress: 進捗コールバック

    Returns:
        実験結果 dict
    """
    if train_period is None:
        train_period = TRAIN_PERIOD
    if valid_period is None:
        valid_period = VALID_PERIOD
    if test_period is None:
        test_period = TEST_PERIOD

    def progress(msg: str, pct: float):
        logger.info("[%.0f%%] %s", pct * 100, msg)
        if on_progress:
            on_progress(msg, pct)

    if data_dir is None:
        data_dir = QLIB_DATA_DIR

    # Step 1: Qlib初期化
    progress("Qlib初期化中...", 0.0)
    init_qlib_jp(data_dir)

    # Step 1.5: フィルタ済み銘柄リスト作成
    if instrument_codes is not None and len(instrument_codes) > 0:
        progress(f"ユニバースフィルタ適用中... ({len(instrument_codes)}銘柄)", 0.03)
        instruments_name = _write_filtered_instruments(data_dir, instrument_codes)
    else:
        instruments_name = "all"

    # Step 2: DataHandler構築
    progress("Alpha158 DataHandler構築中...", 0.05)
    handler = _build_alpha158_jp_handler(
        instruments=instruments_name,
        start_time=train_period[0],
        end_time=test_period[1],
        fit_start_time=train_period[0],
        fit_end_time=train_period[1],
    )

    # Step 3: Dataset構築
    progress("Dataset構築中...", 0.15)
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": train_period,
            "valid": valid_period,
            "test": test_period,
        },
    )

    # Step 4: モデル学習
    progress("モデル学習中...", 0.25)
    if model_type == "lgb":
        from qlib.contrib.model.gbdt import LGBModel
        params = lgb_params if lgb_params else dict(LGB_DEFAULT_PARAMS)
        model = LGBModel(**params)
    elif model_type == "xgb":
        try:
            from qlib.contrib.model.xgboost import XGBModel
        except ImportError:
            raise ImportError("XGBoostを使うには xgboost パッケージをインストールしてください: pip install xgboost")
        model = XGBModel()
    else:
        raise ValueError(f"未対応のモデルタイプ: {model_type}")

    model.fit(dataset)
    progress("モデル学習完了", 0.60)

    # Step 5: 予測
    progress("予測実行中...", 0.65)
    pred = model.predict(dataset)
    progress("予測完了: %d件" % len(pred), 0.75)

    # Step 6: IC / ICIR 計算
    progress("評価指標計算中...", 0.80)
    metrics = _compute_ic_metrics(pred, dataset, test_period)

    # Step 7: 特徴量重要度
    progress("特徴量重要度取得中...", 0.90)
    feature_importance = _get_feature_importance(model, handler)

    progress("実験完了", 1.0)

    return {
        "model_type": model_type,
        "train_period": train_period,
        "valid_period": valid_period,
        "test_period": test_period,
        "topk": topk,
        "n_drop": n_drop,
        "instruments": instruments_name,
        "n_instruments": len(instrument_codes) if instrument_codes else "all",
        **metrics,
        "feature_importance": feature_importance,
        "predictions": pred,
        "n_predictions": len(pred),
    }


def _compute_ic_metrics(pred: pd.Series, dataset, test_period: tuple[str, str]) -> dict:
    """IC / ICIR / RankIC を計算する。"""
    try:
        test_data = dataset.prepare("test", col_set="label")
        if isinstance(test_data, pd.DataFrame):
            label = test_data.iloc[:, 0]
        else:
            label = test_data

        # pred と label の共通インデックスで整列
        common_idx = pred.index.intersection(label.index)
        if len(common_idx) == 0:
            logger.warning("予測とラベルの共通インデックスがありません")
            return {
                "ic_mean": float("nan"),
                "icir": float("nan"),
                "rank_ic_mean": float("nan"),
                "n_test_samples": 0,
            }

        pred_aligned = pred.loc[common_idx]
        label_aligned = label.loc[common_idx]

        # マルチインデックス (datetime, instrument) の場合、日次でIC計算
        if isinstance(common_idx, pd.MultiIndex):
            dates = common_idx.get_level_values(0).unique()
            ics = []
            rank_ics = []
            for dt in dates:
                try:
                    p = pred_aligned.loc[dt]
                    l = label_aligned.loc[dt]
                    if len(p) < 5:
                        continue
                    ic = p.corr(l)
                    rank_ic = p.rank().corr(l.rank())
                    if not np.isnan(ic):
                        ics.append(ic)
                    if not np.isnan(rank_ic):
                        rank_ics.append(rank_ic)
                except Exception:
                    continue

            ic_series = pd.Series(ics)
            rank_ic_series = pd.Series(rank_ics)

            ic_mean = ic_series.mean() if len(ic_series) > 0 else float("nan")
            ic_std = ic_series.std() if len(ic_series) > 1 else float("nan")
            icir = ic_mean / ic_std if ic_std > 0 else float("nan")
            rank_ic_mean = rank_ic_series.mean() if len(rank_ic_series) > 0 else float("nan")
        else:
            # フラットインデックスの場合
            ic_mean = pred_aligned.corr(label_aligned)
            rank_ic_mean = pred_aligned.rank().corr(label_aligned.rank())
            icir = float("nan")

        return {
            "ic_mean": float(ic_mean) if not np.isnan(ic_mean) else float("nan"),
            "icir": float(icir) if not np.isnan(icir) else float("nan"),
            "rank_ic_mean": float(rank_ic_mean) if not np.isnan(rank_ic_mean) else float("nan"),
            "n_test_samples": len(common_idx),
        }
    except Exception as e:
        logger.error("IC計算エラー: %s", e)
        return {
            "ic_mean": float("nan"),
            "icir": float("nan"),
            "rank_ic_mean": float("nan"),
            "n_test_samples": 0,
        }


def _is_generic_name(name: str) -> bool:
    """column_0, feature_0 のような汎用名かどうか判定する。"""
    return (
        name.startswith("column_")
        or name.startswith("feature_")
        or name.startswith("Column_")
    )


def _get_feature_importance(model, handler) -> dict:
    """モデルの特徴量重要度を取得する (Top 30)。"""
    try:
        # LGBModel の内部 model へアクセス
        inner_model = getattr(model, "model", None)
        if inner_model is None:
            return {}

        importance = inner_model.feature_importance(importance_type="gain")
        n_features = len(importance)

        # 特徴量名の取得を複数手段で試行
        feature_names = None

        # 方法1: handler.get_cols() から取得
        try:
            cols = handler.get_cols()
            if isinstance(cols, pd.MultiIndex):
                feature_names = ["_".join(str(x) for x in tup) for tup in cols]
            elif isinstance(cols, pd.Index):
                feature_names = cols.tolist()
        except Exception:
            pass

        # 方法2: LightGBMモデルのfeature_name()から取得
        if feature_names is None or len(feature_names) != n_features:
            try:
                feature_names = inner_model.feature_name()
            except Exception:
                pass

        # 汎用名（column_0等）の場合、Alpha158の既知名にマッピング
        if feature_names is not None and len(feature_names) == n_features:
            if all(_is_generic_name(str(n)) for n in feature_names):
                if n_features == len(_ALPHA158_NAMES):
                    feature_names = list(_ALPHA158_NAMES)
                    logger.info("Alpha158特徴量名を適用 (%d個)", n_features)
                elif n_features < len(_ALPHA158_NAMES):
                    # プロセッサで一部が除外された場合、先頭からマッピング
                    feature_names = _ALPHA158_NAMES[:n_features]
                    logger.info("Alpha158特徴量名を部分適用 (%d/%d個)", n_features, len(_ALPHA158_NAMES))

        # 方法3: フォールバック — それでも汎用名ならAlpha158名を使う
        if feature_names is None or len(feature_names) != n_features:
            if n_features == len(_ALPHA158_NAMES):
                feature_names = list(_ALPHA158_NAMES)
            else:
                feature_names = [f"feature_{i}" for i in range(n_features)]

        # 特徴量名を文字列に統一
        feature_names = [str(n) for n in feature_names]

        imp_dict = dict(zip(feature_names, importance.tolist()))
        # Top 30 を重要度降順でソート
        sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:30]
        return dict(sorted_imp)
    except Exception as e:
        logger.warning("特徴量重要度取得失敗: %s", e)
        return {}
