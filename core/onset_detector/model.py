"""二段階TTEモデル + PurgedGroupCV (Track A/B)"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OnsetDetectorConfig

logger = logging.getLogger(__name__)


# ======================================================================
# PurgedGroupTimeSeriesCV
# ======================================================================
class PurgedGroupTimeSeriesCV:
    """時系列CV + 銘柄グループ分離 + embargo

    Track A (strict): val銘柄をtrainから完全除外
    Track B (relaxed): 同一銘柄可（異なる期間ならOK）
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 60,
        strict_group: bool = True,
    ):
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.strict_group = strict_group

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None,
        dates: np.ndarray = None,
    ):
        """
        Parameters
        ----------
        X : array-like
        y : array-like
        groups : array-like  銘柄コード
        dates : array-like   日付（整数インデックスまたは文字列日付）

        Yields
        ------
        train_idx, val_idx
        """
        n = len(X)
        if dates is None:
            dates = np.arange(n)

        # 日付でソート
        sort_idx = np.argsort(dates)
        sorted_dates = dates[sort_idx]
        sorted_groups = groups[sort_idx] if groups is not None else None

        # 時系列分割
        fold_size = n // (self.n_splits + 1)

        for fold in range(self.n_splits):
            # train: 先頭〜fold_size*(fold+1)
            # val: fold_size*(fold+1)〜fold_size*(fold+2)
            train_end = fold_size * (fold + 1)
            val_start = train_end
            val_end = min(val_start + fold_size, n)

            if val_end <= val_start:
                continue

            train_sorted = sort_idx[:train_end]
            val_sorted = sort_idx[val_start:val_end]

            # Embargo: train末尾のembargo_days日分を除外
            if self.embargo_days > 0 and len(train_sorted) > 0:
                train_dates_vals = sorted_dates[:train_end]
                val_dates_vals = sorted_dates[val_start:val_end]

                # 日付がstr/datetime型ならint indexで代替
                if isinstance(train_dates_vals[0], (str, np.str_)):
                    # 簡易embargo: 末尾embargo_days個を除外
                    embargo_n = min(self.embargo_days, len(train_sorted) // 4)
                    train_sorted = train_sorted[:-embargo_n] if embargo_n > 0 else train_sorted
                else:
                    embargo_cutoff = sorted_dates[train_end - 1] - self.embargo_days
                    train_sorted = train_sorted[sorted_dates[:train_end] <= embargo_cutoff]

            # Track A: val銘柄をtrainから完全除外
            if self.strict_group and sorted_groups is not None:
                val_groups = set(sorted_groups[val_start:val_end])
                # train_sortedは元配列へのインデックス → groups[idx]でグループ取得
                mask = np.array([groups[idx] not in val_groups for idx in train_sorted])
                train_sorted = train_sorted[mask]

            if len(train_sorted) < 10 or len(val_sorted) < 5:
                continue

            yield train_sorted, val_sorted


# ======================================================================
# TwoStageOnsetModel
# ======================================================================
class TwoStageOnsetModel:
    """二段階モデル: Stage 1 (binary) + Stage 2 (Huber regression)"""

    def __init__(self, config: OnsetDetectorConfig):
        self.config = config
        self.stage1_model = None
        self.stage2_model = None
        self.best_excess_threshold: float = config.excess_threshold
        self.best_horizon: int = config.horizon
        self.best_tau: float = 20.0
        self.feature_names: list[str] = []
        self._eval_results: dict = {}

    def train(
        self,
        X: np.ndarray,
        dataset: dict,
        feature_names: list[str],
    ) -> dict:
        """ネストCV + 二段階モデル学習

        Returns
        -------
        dict  評価結果 (Track A/B metrics)
        """
        import lightgbm as lgb

        self.feature_names = feature_names
        cfg = self.config

        samples = dataset["samples"]
        y1 = dataset["y1"]
        y2 = dataset["y2"]
        weights = dataset["sample_weights"]
        stage2_mask = dataset["stage2_mask"]
        first_hits = dataset["first_hits"]

        # グループ（銘柄コード）と日付
        groups = np.array([s["code"] for s in samples])
        dates = np.array([s.get("date", s.get("date_idx", i)) for i, s in enumerate(samples)])

        # =========================================================
        # ネストCV: イベント定義グリッド探索
        # =========================================================
        event_grid = cfg.get_event_grid()
        logger.info(f"イベント定義グリッド: {len(event_grid)}パターン")

        # 簡易探索: デフォルトパラメータで実行（フルネストは計算量大）
        # フルネストは evaluator.py で独立に実行可能
        best_score = -1
        best_params = (cfg.excess_threshold, cfg.horizon)

        # =========================================================
        # Track A (strict) + Track B (relaxed) 評価
        # =========================================================
        eval_results = {}
        for track_name, strict in [("Track_A", True), ("Track_B", False)]:
            logger.info(f"{track_name} CV開始...")
            cv = PurgedGroupTimeSeriesCV(
                n_splits=cfg.n_outer_folds,
                embargo_days=cfg.embargo_days,
                strict_group=strict,
            )

            oof_p_event = np.full(len(y1), np.nan)
            oof_tte = np.full(len(y1), np.nan)
            fold_metrics = []

            for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y1, groups, dates)):
                logger.info(f"  {track_name} Fold {fold_i}: train={len(train_idx)}, val={len(val_idx)}")

                X_tr, X_val = X[train_idx], X[val_idx]
                y1_tr, y1_val = y1[train_idx], y1[val_idx]
                w_tr = weights[train_idx]

                # Stage 1: binary classification
                pos_count = y1_tr.sum()
                neg_count = len(y1_tr) - pos_count
                scale_pos = neg_count / max(pos_count, 1)

                stage1 = lgb.LGBMClassifier(
                    objective="binary",
                    n_estimators=cfg.stage1_n_estimators,
                    num_leaves=cfg.stage1_num_leaves,
                    min_child_samples=cfg.stage1_min_child_samples,
                    learning_rate=cfg.stage1_learning_rate,
                    feature_fraction=cfg.stage1_feature_fraction,
                    bagging_fraction=cfg.stage1_bagging_fraction,
                    bagging_freq=5,
                    scale_pos_weight=scale_pos,
                    verbose=-1,
                    random_state=42,
                )
                stage1.fit(
                    X_tr, y1_tr,
                    sample_weight=w_tr,
                    eval_set=[(X_val, y1_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )

                p_event = stage1.predict_proba(X_val)[:, 1]
                oof_p_event[val_idx] = p_event

                # Stage 2: TTE regression (到達サンプルのみ)
                s2_train_mask = stage2_mask[train_idx]
                s2_val_mask = stage2_mask[val_idx]

                if s2_train_mask.sum() >= 10:
                    X_s2_tr = X_tr[s2_train_mask]
                    y_s2_tr = y2[train_idx][s2_train_mask]

                    stage2 = lgb.LGBMRegressor(
                        objective="huber",
                        n_estimators=cfg.stage2_n_estimators,
                        num_leaves=cfg.stage2_num_leaves,
                        min_child_samples=cfg.stage2_min_child_samples,
                        learning_rate=cfg.stage2_learning_rate,
                        feature_fraction=cfg.stage2_feature_fraction,
                        verbose=-1,
                        random_state=42,
                    )

                    if s2_val_mask.sum() >= 5:
                        X_s2_val = X_val[s2_val_mask]
                        y_s2_val = y2[val_idx][s2_val_mask]
                        stage2.fit(
                            X_s2_tr, y_s2_tr,
                            eval_set=[(X_s2_val, y_s2_val)],
                            callbacks=[lgb.early_stopping(50, verbose=False)],
                        )
                    else:
                        stage2.fit(X_s2_tr, y_s2_tr)

                    pred_log_tte = stage2.predict(X_val)
                    oof_tte[val_idx] = np.expm1(pred_log_tte)
                else:
                    oof_tte[val_idx] = cfg.horizon / 2  # fallback

                # Fold metrics
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(y1_val, p_event)
                except Exception:
                    auc = 0.5
                fold_metrics.append({"fold": fold_i, "auc": auc, "n_val": len(val_idx)})

            eval_results[track_name] = {
                "fold_metrics": fold_metrics,
                "oof_p_event": oof_p_event,
                "oof_tte": oof_tte,
                "mean_auc": np.mean([m["auc"] for m in fold_metrics]),
            }
            logger.info(f"  {track_name} Mean AUC: {eval_results[track_name]['mean_auc']:.4f}")

        self._eval_results = eval_results

        # =========================================================
        # τ最適化 (Precision@K × Lead Time)
        # =========================================================
        self.best_tau = self._optimize_tau(
            eval_results, y1, first_hits, dataset
        )

        # =========================================================
        # 全データで最終モデル学習
        # =========================================================
        logger.info("最終モデル学習中...")

        pos_count = y1.sum()
        neg_count = len(y1) - pos_count
        scale_pos = neg_count / max(pos_count, 1)

        self.stage1_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=cfg.stage1_n_estimators,
            num_leaves=cfg.stage1_num_leaves,
            min_child_samples=cfg.stage1_min_child_samples,
            learning_rate=cfg.stage1_learning_rate,
            feature_fraction=cfg.stage1_feature_fraction,
            bagging_fraction=cfg.stage1_bagging_fraction,
            bagging_freq=5,
            scale_pos_weight=scale_pos,
            verbose=-1,
            random_state=42,
        )
        self.stage1_model.fit(X, y1, sample_weight=weights)

        # Stage 2
        s2_mask = stage2_mask
        if s2_mask.sum() >= 10:
            self.stage2_model = lgb.LGBMRegressor(
                objective="huber",
                n_estimators=cfg.stage2_n_estimators,
                num_leaves=cfg.stage2_num_leaves,
                min_child_samples=cfg.stage2_min_child_samples,
                learning_rate=cfg.stage2_learning_rate,
                feature_fraction=cfg.stage2_feature_fraction,
                verbose=-1,
                random_state=42,
            )
            self.stage2_model.fit(X[s2_mask], y2[s2_mask])

        logger.info("最終モデル学習完了")

        # 評価メトリクス集約
        return self._compile_eval_report(eval_results, y1, first_hits, dataset)

    # ------------------------------------------------------------------
    # τ最適化
    # ------------------------------------------------------------------
    def _optimize_tau(
        self,
        eval_results: dict,
        y1: np.ndarray,
        first_hits: np.ndarray,
        dataset: dict,
    ) -> float:
        """Precision@K × Lead Time でτを最適化"""
        cfg = self.config
        best_tau = 20.0
        best_objective = -np.inf

        # Track Bのoof結果を使用
        track = eval_results.get("Track_B", eval_results.get("Track_A", {}))
        oof_p = track.get("oof_p_event", np.array([]))
        oof_tte = track.get("oof_tte", np.array([]))

        if len(oof_p) == 0:
            return best_tau

        valid = ~np.isnan(oof_p) & ~np.isnan(oof_tte)
        if valid.sum() < 50:
            return best_tau

        for tau in range(cfg.tau_range[0], cfg.tau_range[1] + 1):
            scores = oof_p[valid] * np.exp(-np.clip(oof_tte[valid], 0, 200) / tau)
            top_k = min(cfg.top_k, int(valid.sum() * 0.1))
            if top_k < 5:
                top_k = 5

            top_indices = np.argsort(scores)[-top_k:]

            # Precision: top-Kのうちy1=1の比率
            precision = y1[np.where(valid)[0][top_indices]].mean()

            # Lead Time: 到達サンプルのfirst_hit中央値
            fh_top = first_hits[np.where(valid)[0][top_indices]]
            fh_reached = fh_top[fh_top > 0]
            lead_time = np.median(fh_reached) if len(fh_reached) > 0 else cfg.horizon

            # objective: precision × lead_time_bonus (短いTTEは高得点)
            lead_bonus = max(0, 1 - lead_time / cfg.horizon)
            objective = precision * (1 + lead_bonus)

            if objective > best_objective:
                best_objective = objective
                best_tau = tau

        logger.info(f"最適τ = {best_tau} (objective={best_objective:.4f})")
        return float(best_tau)

    # ------------------------------------------------------------------
    # 推論
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        p_event : np.ndarray     Stage 1 確率
        predicted_tte : np.ndarray  Stage 2 予測TTE
        onset_score : np.ndarray    複合スコア
        """
        if self.stage1_model is None:
            raise RuntimeError("モデルが学習されていません")

        p_event = self.stage1_model.predict_proba(X)[:, 1]

        if self.stage2_model is not None:
            pred_log_tte = self.stage2_model.predict(X)
            predicted_tte = np.expm1(pred_log_tte)
            predicted_tte = np.clip(predicted_tte, 1, self.config.horizon)
        else:
            predicted_tte = np.full(len(X), self.config.horizon / 2)

        onset_score = p_event * np.exp(-predicted_tte / self.best_tau)

        return p_event, predicted_tte, onset_score

    def explain_prediction(self, X_single: np.ndarray) -> list[tuple[str, float]]:
        """単一サンプルのSHAP-like特徴量寄与を返す（簡易版: feature importance × 値）"""
        if self.stage1_model is None:
            return []

        importances = self.stage1_model.feature_importances_
        if len(importances) != X_single.shape[-1]:
            return []

        contributions = importances * np.abs(X_single.flatten())
        top_indices = np.argsort(contributions)[-10:][::-1]

        return [
            (self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}",
             float(contributions[i]))
            for i in top_indices
        ]

    # ------------------------------------------------------------------
    # 保存/読み込み
    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        """モデルを保存"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "stage1_model": self.stage1_model,
            "stage2_model": self.stage2_model,
            "best_excess_threshold": self.best_excess_threshold,
            "best_horizon": self.best_horizon,
            "best_tau": self.best_tau,
            "feature_names": self.feature_names,
            "config": self.config,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"モデル保存: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TwoStageOnsetModel":
        """モデルを読み込み"""
        with open(path, "rb") as f:
            state = pickle.load(f)
        model = cls(state["config"])
        model.stage1_model = state["stage1_model"]
        model.stage2_model = state["stage2_model"]
        model.best_excess_threshold = state["best_excess_threshold"]
        model.best_horizon = state["best_horizon"]
        model.best_tau = state["best_tau"]
        model.feature_names = state["feature_names"]
        return model

    # ------------------------------------------------------------------
    # 評価レポート集約
    # ------------------------------------------------------------------
    def _compile_eval_report(
        self,
        eval_results: dict,
        y1: np.ndarray,
        first_hits: np.ndarray,
        dataset: dict,
    ) -> dict:
        """Track A/B の評価メトリクスを集約"""
        report = {}
        cfg = self.config

        for track_name, track_data in eval_results.items():
            oof_p = track_data["oof_p_event"]
            oof_tte = track_data["oof_tte"]
            valid = ~np.isnan(oof_p) & ~np.isnan(oof_tte)

            if valid.sum() < 10:
                report[track_name] = {"error": "サンプル不足"}
                continue

            # onset_score計算
            scores = oof_p[valid] * np.exp(-np.clip(oof_tte[valid], 0, 200) / self.best_tau)

            # Precision@K
            top_k = min(cfg.top_k, max(5, int(valid.sum() * 0.05)))
            top_idx = np.argsort(scores)[-top_k:]
            valid_indices = np.where(valid)[0]
            precision_k = y1[valid_indices[top_idx]].mean()

            # Lead Time
            fh_top = first_hits[valid_indices[top_idx]]
            reached = fh_top[fh_top > 0]
            lead_time_median = float(np.median(reached)) if len(reached) > 0 else float(cfg.horizon)

            # Tail Loss (negative return ratio)
            negative_mask = fh_top <= 0  # 未到達 = 潜在的損失
            tail_loss = float(negative_mask.mean())

            report[track_name] = {
                "mean_auc": track_data["mean_auc"],
                "fold_metrics": track_data["fold_metrics"],
                f"precision@{top_k}": float(precision_k),
                "lead_time_median": lead_time_median,
                "tail_loss": float(tail_loss),
                "n_valid": int(valid.sum()),
                "top_k": top_k,
            }

        report["best_tau"] = self.best_tau
        report["best_event_params"] = {
            "excess_threshold": self.best_excess_threshold,
            "horizon": self.best_horizon,
        }

        # Feature importance
        if self.stage1_model is not None:
            imp = self.stage1_model.feature_importances_
            top_20 = np.argsort(imp)[-20:][::-1]
            report["top_features"] = [
                {
                    "name": self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}",
                    "importance": float(imp[i]),
                }
                for i in top_20
            ]

        return report
