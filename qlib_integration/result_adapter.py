"""Qlib結果 → 既存システム形式への変換アダプター + 保存・読込"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qlib_integration.config import QLIB_RESULTS_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 実験結果の保存・読込
# ---------------------------------------------------------------------------
def save_experiment(result: dict) -> str:
    """実験結果をディスクに保存する。

    保存先: QLIB_RESULTS_DIR/{experiment_id}/
      - meta.json   : 指標・パラメータ（人間可読）
      - predictions.pkl : 予測スコア（pd.Series）
      - feature_importance.json : 特徴量重要度

    Returns:
        experiment_id (タイムスタンプベース)
    """
    exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = QLIB_RESULTS_DIR / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # メタデータ（JSON化できるもの）
    meta = {}
    for k, v in result.items():
        if k == "predictions":
            continue  # 別ファイルに保存
        if k == "feature_importance":
            continue  # 別ファイルに保存
        if isinstance(v, float) and np.isnan(v):
            meta[k] = None
        elif isinstance(v, (str, int, float, bool, list)):
            meta[k] = v
        elif isinstance(v, tuple):
            meta[k] = list(v)
        else:
            meta[k] = str(v)
    meta["experiment_id"] = exp_id
    meta["saved_at"] = datetime.now().isoformat()

    (exp_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 予測スコア
    predictions = result.get("predictions")
    if predictions is not None:
        with open(exp_dir / "predictions.pkl", "wb") as f:
            pickle.dump(predictions, f)

    # 特徴量重要度
    feat_imp = result.get("feature_importance", {})
    if feat_imp:
        (exp_dir / "feature_importance.json").write_text(
            json.dumps(feat_imp, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logger.info("実験結果を保存: %s", exp_dir)
    return exp_id


def load_experiment(experiment_id: str) -> dict | None:
    """保存済み実験結果を読み込む。"""
    exp_dir = QLIB_RESULTS_DIR / experiment_id
    meta_path = exp_dir / "meta.json"
    if not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # tupleに戻す
    for key in ("train_period", "valid_period", "test_period"):
        if key in meta and isinstance(meta[key], list):
            meta[key] = tuple(meta[key])

    # Noneをnanに戻す
    for key in ("ic_mean", "icir", "rank_ic_mean"):
        if key in meta and meta[key] is None:
            meta[key] = float("nan")

    # 予測スコア
    pred_path = exp_dir / "predictions.pkl"
    if pred_path.exists():
        with open(pred_path, "rb") as f:
            meta["predictions"] = pickle.load(f)

    # 特徴量重要度
    fi_path = exp_dir / "feature_importance.json"
    if fi_path.exists():
        meta["feature_importance"] = json.loads(fi_path.read_text(encoding="utf-8"))

    return meta


def list_experiments() -> list[dict]:
    """保存済み実験の一覧を返す（新しい順）。"""
    if not QLIB_RESULTS_DIR.exists():
        return []

    experiments = []
    for d in sorted(QLIB_RESULTS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["experiment_id"] = d.name
            experiments.append(meta)
        except Exception:
            continue

    return experiments


def delete_experiment(experiment_id: str) -> bool:
    """保存済み実験を削除する。"""
    exp_dir = QLIB_RESULTS_DIR / experiment_id
    if not exp_dir.exists():
        return False
    import shutil
    shutil.rmtree(exp_dir)
    return True


def extract_topk_daily(predictions: pd.Series, topk: int = 30) -> pd.DataFrame:
    """Qlib予測スコアから日次Top-K銘柄リストを抽出する。

    Args:
        predictions: Qlibモデルの予測スコア (MultiIndex: datetime, instrument)
        topk: 上位何銘柄を抽出するか

    Returns:
        DataFrame (columns: date, rank, code, score)
    """
    if not isinstance(predictions.index, pd.MultiIndex):
        raise ValueError("predictionsはMultiIndex (datetime, instrument) 形式である必要があります")

    records = []
    dates = predictions.index.get_level_values(0).unique()

    for dt in dates:
        day_scores = predictions.loc[dt].sort_values(ascending=False)
        top = day_scores.head(topk)
        for rank, (code, score) in enumerate(top.items(), 1):
            records.append({
                "date": dt,
                "rank": rank,
                "code": str(code),
                "score": float(score),
            })

    return pd.DataFrame(records)


def compute_topk_returns(
    topk_df: pd.DataFrame,
    provider,
    holding_days: int = 1,
) -> pd.DataFrame:
    """Top-K銘柄の実際のフォワードリターンを計算する。

    Args:
        topk_df: extract_topk_daily() の出力
        provider: JQuantsProvider インスタンス
        holding_days: 保有日数

    Returns:
        日次の等配分ポートフォリオリターン DataFrame
    """
    dates = sorted(topk_df["date"].unique())
    if len(dates) < 2:
        return pd.DataFrame(columns=["date", "portfolio_return"])

    start_date = pd.Timestamp(dates[0]).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")

    price_df = provider.get_price_daily(start_date=start_date, end_date=end_date)
    if price_df.empty:
        return pd.DataFrame(columns=["date", "portfolio_return"])

    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df["code"] = price_df["code"].astype(str)

    # 銘柄ごとの日次リターンを計算
    price_df = price_df.sort_values(["code", "date"])
    price_df["return"] = price_df.groupby("code")["adj_close"].pct_change()

    # Top-K銘柄のリターンを日次で集計
    topk_df = topk_df.copy()
    topk_df["date"] = pd.to_datetime(topk_df["date"])

    results = []
    for dt in dates:
        top_codes = topk_df[topk_df["date"] == dt]["code"].tolist()
        day_returns = price_df[
            (price_df["date"] == dt) & (price_df["code"].isin(top_codes))
        ]["return"]

        if len(day_returns) > 0:
            port_ret = day_returns.mean()  # 等配分
        else:
            port_ret = 0.0

        results.append({"date": dt, "portfolio_return": port_ret})

    return pd.DataFrame(results)


def compute_excess_return(
    portfolio_returns: pd.DataFrame,
    provider,
    benchmark_code: str = "0000",
) -> pd.DataFrame:
    """ポートフォリオリターンとベンチマーク（TOPIX）の超過リターンを計算する。

    Args:
        portfolio_returns: compute_topk_returns() の出力
        provider: JQuantsProvider
        benchmark_code: ベンチマーク指数コード

    Returns:
        DataFrame (columns: date, portfolio_return, benchmark_return, excess_return,
                   cum_portfolio, cum_benchmark, cum_excess)
    """
    if portfolio_returns.empty:
        return portfolio_returns

    dates = portfolio_returns["date"]
    start_date = dates.min().strftime("%Y-%m-%d")
    end_date = dates.max().strftime("%Y-%m-%d")

    topix = provider.get_index_prices(
        index_code=benchmark_code,
        start_date=start_date,
        end_date=end_date,
    )
    if topix.empty:
        portfolio_returns["benchmark_return"] = 0.0
        portfolio_returns["excess_return"] = portfolio_returns["portfolio_return"]
        return portfolio_returns

    topix["date"] = pd.to_datetime(topix["date"])
    topix = topix.sort_values("date")
    topix["benchmark_return"] = topix["close"].pct_change()

    merged = portfolio_returns.merge(
        topix[["date", "benchmark_return"]],
        on="date",
        how="left",
    )
    merged["benchmark_return"] = merged["benchmark_return"].fillna(0)
    merged["excess_return"] = merged["portfolio_return"] - merged["benchmark_return"]

    # 累積リターン
    merged["cum_portfolio"] = (1 + merged["portfolio_return"]).cumprod() - 1
    merged["cum_benchmark"] = (1 + merged["benchmark_return"]).cumprod() - 1
    merged["cum_excess"] = merged["cum_portfolio"] - merged["cum_benchmark"]

    return merged


def summarize_experiment(result: dict) -> dict:
    """実験結果のサマリを人間可読な形式に変換する。"""
    summary = {
        "モデル": result.get("model_type", "不明"),
        "学習期間": f"{result.get('train_period', ('?', '?'))[0]} ~ {result.get('train_period', ('?', '?'))[1]}",
        "テスト期間": f"{result.get('test_period', ('?', '?'))[0]} ~ {result.get('test_period', ('?', '?'))[1]}",
        "テストサンプル数": result.get("n_test_samples", 0),
        "予測件数": result.get("n_predictions", 0),
    }

    ic = result.get("ic_mean")
    icir = result.get("icir")
    rank_ic = result.get("rank_ic_mean")

    if ic is not None and not np.isnan(ic):
        summary["IC (平均)"] = f"{ic:.4f}"
        quality = "良い" if ic > 0.05 else ("使える" if ic > 0.03 else "弱い")
        summary["IC評価"] = quality

    if icir is not None and not np.isnan(icir):
        summary["ICIR"] = f"{icir:.4f}"
        stability = "安定" if icir > 0.5 else "不安定"
        summary["安定性"] = stability

    if rank_ic is not None and not np.isnan(rank_ic):
        summary["Rank IC (平均)"] = f"{rank_ic:.4f}"

    return summary
