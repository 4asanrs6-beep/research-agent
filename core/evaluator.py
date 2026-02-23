"""結果評価モジュール"""

import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """統計的有意性 + バックテスト結果を総合判定"""

    def __init__(
        self,
        significance_level: float = 0.05,
        min_sharpe: float = 0.3,
        min_cohens_d: float = 0.2,
    ):
        self.significance_level = significance_level
        self.min_sharpe = min_sharpe
        self.min_cohens_d = min_cohens_d

    def evaluate(
        self,
        statistics_result: dict | None,
        backtest_result: dict | None,
    ) -> dict:
        """統計分析結果とバックテスト結果を総合評価

        Returns:
            {
                "label": "valid" | "invalid" | "needs_review",
                "confidence": float (0-1),
                "reasons": list[str],
                "stat_score": float,
                "backtest_score": float,
            }
        """
        reasons = []
        stat_score = 0.0
        bt_score = 0.0

        # --- 統計分析の評価 ---
        if statistics_result and "error" not in statistics_result:
            p_value = statistics_result.get("p_value", 1.0)
            cohens_d = abs(statistics_result.get("cohens_d", 0.0))
            is_sig = statistics_result.get("is_significant", False)
            n_cond = statistics_result.get("n_condition", 0)

            if is_sig:
                stat_score += 0.4
                reasons.append(f"統計的に有意 (p={p_value:.4f})")
            else:
                reasons.append(f"統計的に有意ではない (p={p_value:.4f})")

            if cohens_d >= self.min_cohens_d:
                stat_score += 0.3
                reasons.append(f"効果量が十分 (d={cohens_d:.3f})")
            else:
                reasons.append(f"効果量が小さい (d={cohens_d:.3f})")

            if n_cond >= 30:
                stat_score += 0.1
            else:
                reasons.append(f"サンプル数が少ない (n={n_cond})")

            cond_wr = statistics_result.get("win_rate_condition", 0)
            base_wr = statistics_result.get("win_rate_baseline", 0)
            if cond_wr > base_wr:
                stat_score += 0.2
                reasons.append(f"勝率改善 ({base_wr:.1%} → {cond_wr:.1%})")
        else:
            reasons.append("統計分析結果なし")

        # --- バックテストの評価 ---
        if backtest_result and "error" not in backtest_result:
            sharpe = backtest_result.get("sharpe_ratio", 0.0)
            annual_ret = backtest_result.get("annual_return", 0.0)
            max_dd = backtest_result.get("max_drawdown", 1.0)
            bench_annual = backtest_result.get("benchmark_annual_return", 0.0)

            if sharpe >= self.min_sharpe:
                bt_score += 0.3
                reasons.append(f"シャープ比良好 ({sharpe:.2f})")
            else:
                reasons.append(f"シャープ比低い ({sharpe:.2f})")

            if annual_ret > bench_annual:
                bt_score += 0.3
                reasons.append(f"ベンチマーク超過 ({annual_ret:.1%} vs {bench_annual:.1%})")
            else:
                reasons.append(f"ベンチマーク未達 ({annual_ret:.1%} vs {bench_annual:.1%})")

            if max_dd < 0.3:
                bt_score += 0.2
                reasons.append(f"ドローダウン許容範囲 ({max_dd:.1%})")
            else:
                reasons.append(f"ドローダウン大きい ({max_dd:.1%})")

            if backtest_result.get("win_rate", 0) > 0.5:
                bt_score += 0.2
        else:
            reasons.append("バックテスト結果なし")

        # --- 総合判定 ---
        total = (stat_score + bt_score) / 2
        if total >= 0.6:
            label = "valid"
        elif total >= 0.3:
            label = "needs_review"
        else:
            label = "invalid"

        return {
            "label": label,
            "confidence": round(total, 3),
            "reasons": reasons,
            "stat_score": round(stat_score, 3),
            "backtest_score": round(bt_score, 3),
        }
