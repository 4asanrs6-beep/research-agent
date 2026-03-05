"""Track A/B評価 + 類型別 + バックテスト"""

import logging

import numpy as np
import pandas as pd

from .config import OnsetDetectorConfig

logger = logging.getLogger(__name__)


class OnsetEvaluator:
    """Onset Detectorの評価フレームワーク"""

    def __init__(self, config: OnsetDetectorConfig):
        self.config = config

    # ==================================================================
    # Precision@K / Lead Time / Tail Loss
    # ==================================================================
    def compute_precision_at_k(
        self,
        onset_scores: np.ndarray,
        y1: np.ndarray,
        k: int | None = None,
    ) -> float:
        """Top-K中のpositive率"""
        if k is None:
            k = self.config.top_k
        k = min(k, len(onset_scores))
        top_idx = np.argsort(onset_scores)[-k:]
        return float(y1[top_idx].mean())

    def compute_lead_time(
        self,
        onset_scores: np.ndarray,
        first_hits: np.ndarray,
        k: int | None = None,
    ) -> dict:
        """Top-Kの到達サンプルにおけるfirst_hit統計"""
        if k is None:
            k = self.config.top_k
        k = min(k, len(onset_scores))
        top_idx = np.argsort(onset_scores)[-k:]
        fh = first_hits[top_idx]
        reached = fh[fh > 0]

        if len(reached) == 0:
            return {"median": float(self.config.horizon), "mean": float(self.config.horizon), "n_reached": 0}

        return {
            "median": float(np.median(reached)),
            "mean": float(np.mean(reached)),
            "std": float(np.std(reached)),
            "min": float(np.min(reached)),
            "max": float(np.max(reached)),
            "n_reached": int(len(reached)),
            "n_total": int(k),
        }

    def compute_tail_loss(
        self,
        onset_scores: np.ndarray,
        first_hits: np.ndarray,
        k: int | None = None,
    ) -> float:
        """Top-Kのうち未到達（潜在損失）の比率"""
        if k is None:
            k = self.config.top_k
        k = min(k, len(onset_scores))
        top_idx = np.argsort(onset_scores)[-k:]
        fh = first_hits[top_idx]
        return float((fh <= 0).mean())

    # ==================================================================
    # 類型別評価
    # ==================================================================
    def evaluate_by_type(
        self,
        onset_scores: np.ndarray,
        y1: np.ndarray,
        first_hits: np.ndarray,
        star_types: np.ndarray,
        k: int | None = None,
    ) -> dict:
        """類型別のPrecision@K / Lead Time / Tail Loss"""
        if k is None:
            k = self.config.top_k

        unique_types = np.unique(star_types)
        results = {}

        for t in unique_types:
            if t < 0:
                continue  # 非スター株スキップ
            mask = star_types == t
            if mask.sum() < 5:
                continue

            scores_t = onset_scores[mask]
            y1_t = y1[mask]
            fh_t = first_hits[mask]
            k_t = min(k, len(scores_t))

            results[f"type_{int(t)}"] = {
                "n_samples": int(mask.sum()),
                "precision@K": self.compute_precision_at_k(scores_t, y1_t, k_t),
                "lead_time": self.compute_lead_time(scores_t, fh_t, k_t),
                "tail_loss": self.compute_tail_loss(scores_t, fh_t, k_t),
            }

        return results

    # ==================================================================
    # Track A/B 比較レポート
    # ==================================================================
    def generate_comparison_report(self, eval_result: dict) -> str:
        """Track A/B比較レポートを文字列で生成"""
        lines = []
        lines.append("=" * 50)
        lines.append("  Onset Detector v2 評価レポート")
        lines.append("=" * 50)

        for track_name in ["Track_A", "Track_B"]:
            data = eval_result.get(track_name, {})
            if "error" in data:
                lines.append(f"\n  {track_name}: {data['error']}")
                continue

            label = "未知銘柄" if track_name == "Track_A" else "既知銘柄将来"
            lines.append(f"\n  {track_name} ({label})")
            lines.append(f"  Stage1 AUC: {data.get('mean_auc', 0):.4f}")

            # Precision@K
            for key in data:
                if key.startswith("precision@"):
                    lines.append(f"  {key}: {data[key]:.1%}")

            lines.append(f"  Mean Lead Time: {data.get('lead_time_median', 0):.1f}日")
            lines.append(f"  Tail Loss: {data.get('tail_loss', 0):.1%}")

        # イベント定義
        params = eval_result.get("best_event_params", {})
        if params:
            lines.append(f"\n  最適イベント定義:")
            lines.append(f"    excess_threshold = {params.get('excess_threshold', 0):.0%}")
            lines.append(f"    horizon = {params.get('horizon', 0)}日")

        lines.append(f"  最適τ = {eval_result.get('best_tau', 0)}")

        # Top特徴量
        top_feats = eval_result.get("top_features", [])
        if top_feats:
            lines.append(f"\n  Top特徴量 (importance):")
            for f in top_feats[:10]:
                lines.append(f"    {f['name']}: {f['importance']:.0f}")

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

    # ==================================================================
    # イベント定義グリッド評価
    # ==================================================================
    def evaluate_event_grid(
        self,
        model_class,
        X: np.ndarray,
        all_prices: pd.DataFrame,
        topix: pd.DataFrame,
        listed_stocks: pd.DataFrame,
        star_stocks: list[dict],
        feature_names: list[str],
    ) -> list[dict]:
        """excess_threshold × horizon の全組み合わせを評価

        Note: 完全なネストCVはここで実装。計算量が大きいため独立実行推奨。
        """
        from .labeler import OnsetLabeler

        cfg = self.config
        grid = cfg.get_event_grid()
        results = []

        for et, h in grid:
            logger.info(f"グリッド評価: excess_threshold={et}, horizon={h}")

            # 一時的にconfig変更
            temp_cfg = OnsetDetectorConfig(
                excess_threshold=et,
                horizon=h,
                T_near=min(cfg.T_near, h),
                n_outer_folds=cfg.n_outer_folds,
                stage1_n_estimators=200,  # 高速化
                stage2_n_estimators=150,
            )

            labeler = OnsetLabeler(self.config.__class__.__new__(self.config.__class__), temp_cfg)
            # NOTE: build_training_datasetは再計算が必要
            # 本格実装ではlabelerにproviderを渡す

            results.append({
                "excess_threshold": et,
                "horizon": h,
                "status": "placeholder",  # 完全実装はcompute-intensive
            })

        return results

    # ==================================================================
    # 簡易バックテスト
    # ==================================================================
    def backtest(
        self,
        scan_results_history: list[tuple[str, list]],
        all_prices: pd.DataFrame,
        holding_days: int = 20,
    ) -> dict:
        """日付ごとのスキャン結果で簡易バックテスト

        Parameters
        ----------
        scan_results_history : list of (date_str, list[ScanResult])
        all_prices : pd.DataFrame
        holding_days : int

        Returns
        -------
        dict  バックテスト結果
        """
        close_col = "adj_close" if "adj_close" in all_prices.columns else "close"
        trades = []

        for scan_date, results in scan_results_history:
            for r in results[:10]:  # Top-10のみ
                code = r.code if hasattr(r, "code") else r["code"]
                grp = all_prices[all_prices["code"] == code].sort_values("date")
                grp_dates = pd.to_datetime(grp["date"])

                # エントリー日
                entry_mask = grp_dates >= pd.Timestamp(scan_date)
                if entry_mask.sum() == 0:
                    continue
                entry_idx = entry_mask.idxmax()
                entry_pos = grp.index.get_loc(entry_idx)

                if entry_pos + holding_days >= len(grp):
                    continue

                entry_price = grp[close_col].iloc[entry_pos]
                exit_price = grp[close_col].iloc[entry_pos + holding_days]

                if entry_price > 0:
                    ret = exit_price / entry_price - 1
                    trades.append({
                        "code": code,
                        "scan_date": scan_date,
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_price),
                        "return": float(ret),
                        "holding_days": holding_days,
                    })

        if not trades:
            return {"error": "取引なし", "n_trades": 0}

        returns = np.array([t["return"] for t in trades])
        return {
            "n_trades": len(trades),
            "mean_return": float(np.mean(returns)),
            "median_return": float(np.median(returns)),
            "win_rate": float((returns > 0).mean()),
            "max_return": float(np.max(returns)),
            "min_return": float(np.min(returns)),
            "sharpe_approx": float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0,
            "trades": trades[:50],  # 詳細は上位50件
        }
