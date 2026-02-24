"""自動研究実行モジュール（パラメータ選択 + イテレーション方式）

研究サイクル:
  仮説入力 → AIがSignalConfigパラメータ選択 → StandardBacktesterで実行
  → 結果が悪ければAIがパラメータ調整して再実行（最大N回）
  → ベスト結果を選択 → AI解釈 → 知見保存

コード生成は行わない。AIはJSON出力のみ。
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config import AI_RESEARCH_MAX_ITERATIONS, AI_RESEARCH_MIN_SIGNALS, AI_RESEARCH_MAX_STOCKS
from db.database import Database
from core.ai_parameter_selector import (
    AiParameterSelector,
    ParameterSelectionResult,
    dict_to_signal_config,
    signal_config_to_dict,
)
from core.ai_interpreter import AiInterpreter
from core.knowledge_base import KnowledgeBase
from core.standard_backtester import StandardBacktester
from core.signal_generator import SignalConfig
from core.universe_filter import UniverseFilterConfig

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    """1回のイテレーション結果"""
    iteration: int
    signal_config_dict: dict
    backtest_result: dict           # StandardBacktester.run() の戻り値
    ai_reasoning: str
    changes_description: str


@dataclass
class ResearchProgress:
    """研究実行の進捗状態"""
    phase: str = "idle"  # idle, generating_params, params_ready, executing, interpreting, saving, done, error
    message: str = ""
    # パラメータ選択結果
    parameter_selection: dict = field(default_factory=dict)
    signal_config_dict: dict = field(default_factory=dict)
    # イテレーション
    current_iteration: int = 0
    max_iterations: int = AI_RESEARCH_MAX_ITERATIONS
    iterations: list = field(default_factory=list)  # list[IterationResult]
    # ベスト結果
    best_iteration_index: int | None = None
    best_result: dict = field(default_factory=dict)
    # AI解釈
    interpretation: dict = field(default_factory=dict)
    # DB参照
    run_id: int | None = None
    knowledge_id: int | None = None
    error: str | None = None
    # 表示用メタデータ
    idea_title: str = ""
    idea_text: str = ""
    category: str = ""
    start_date: str = ""
    end_date: str = ""
    universe_filter_text: str = ""


def _compute_composite_score(bt_result: dict, stats: dict) -> float:
    """イテレーション結果のスコアを算出（ベスト選択用）

    スコア = 統計的有意(+2) + |Cohen's d|*3 + シグナル>=20(+1) + 超過リターン*10
    """
    score = 0.0
    if stats.get("is_significant"):
        score += 2.0
    cohens_d = abs(stats.get("cohens_d", 0))
    score += cohens_d * 3.0
    n_valid = bt_result.get("n_valid_signals", 0)
    if n_valid >= AI_RESEARCH_MIN_SIGNALS:
        score += 1.0
    mean_excess = bt_result.get("mean_excess_return", 0)
    score += mean_excess * 10.0
    return score


class AiResearcher:
    """AI研究エージェント: パラメータ選択 + イテレーション方式"""

    def __init__(
        self,
        db: Database,
        ai_client: Any,
        data_provider: Any,
        max_iterations: int = AI_RESEARCH_MAX_ITERATIONS,
    ):
        self.db = db
        self.data_provider = data_provider
        self.max_iterations = max_iterations

        self.param_selector = AiParameterSelector(ai_client)
        self.interpreter = AiInterpreter(ai_client)
        self.knowledge_base = KnowledgeBase(db)
        self.backtester = StandardBacktester(data_provider, db)

    def select_parameters(
        self,
        hypothesis: str,
        universe_desc: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> ParameterSelectionResult:
        """仮説からSignalConfigパラメータを選択する（UIに表示してユーザー確認を得るためのステップ）"""
        return self.param_selector.select_parameters(
            hypothesis=hypothesis,
            universe_desc=universe_desc,
            start_date=start_date,
            end_date=end_date,
        )

    def run_research_loop(
        self,
        hypothesis: str,
        idea_title: str,
        category: str,
        universe_config: UniverseFilterConfig,
        start_date: str,
        end_date: str,
        initial_config_dict: dict,
        max_iterations: int | None = None,
        on_progress: Any = None,
        universe_filter_text: str = "",
    ) -> ResearchProgress:
        """イテレーションループを実行

        1. initial_config_dict で StandardBacktester.run()
        2. AI が結果を評価しパラメータ調整
        3. 改善見込みなし or 最大回数到達まで繰り返し
        4. ベスト結果を選択し AI 解釈
        5. DB・知見保存
        """
        if max_iterations is None:
            max_iterations = self.max_iterations

        progress = ResearchProgress(
            idea_title=idea_title,
            idea_text=hypothesis,
            category=category,
            start_date=start_date,
            end_date=end_date,
            universe_filter_text=universe_filter_text,
            signal_config_dict=initial_config_dict,
            max_iterations=max_iterations,
        )

        def notify(phase: str, message: str):
            progress.phase = phase
            progress.message = message
            logger.info("[%s] %s", phase, message)
            if on_progress:
                on_progress(progress)

        try:
            # --- 0. DB にアイデア・プラン保存 ---
            notify("executing", "アイデアを保存中...")
            if not idea_title:
                idea_title = hypothesis[:50]
            idea_id = self.db.create_idea(idea_title, hypothesis, category)
            self.db.update_idea(idea_id, status="active")

            plan_data = {
                "plan_name": f"AI研究: {idea_title}",
                "hypothesis": hypothesis,
                "methodology": {"approach": "パラメータ選択 + イテレーション"},
                "analysis_period": {"start_date": start_date, "end_date": end_date},
            }
            plan_id = self.db.create_plan(
                idea_id=idea_id,
                name=plan_data["plan_name"],
                analysis_method="ai_parameter_selection",
                start_date=start_date,
                end_date=end_date,
                parameters={"signal_config": initial_config_dict},
            )
            self.db.update_plan(plan_id, status="ready")

            idea_snapshot = self.db.get_idea(idea_id)
            plan_snapshot = self.db.get_plan(plan_id)
            run_id = self.db.create_run(plan_id, idea_snapshot, plan_snapshot)
            progress.run_id = run_id

            # --- 1. イテレーションループ ---
            current_config_dict = dict(initial_config_dict)
            previous_iterations_summary: list[dict] = []

            for i in range(max_iterations):
                iteration_num = i + 1
                progress.current_iteration = iteration_num
                notify("executing", f"イテレーション {iteration_num}/{max_iterations}: バックテスト実行中...")

                # バックテスト実行
                signal_config = dict_to_signal_config(current_config_dict)
                bt_result = self.backtester.run(
                    signal_config=signal_config,
                    universe_config=universe_config,
                    start_date=start_date,
                    end_date=end_date,
                    max_stocks=AI_RESEARCH_MAX_STOCKS,
                    on_progress=lambda msg, pct: notify(
                        "executing",
                        f"イテレーション {iteration_num}/{max_iterations}: {msg}",
                    ),
                )

                # エラーチェック
                if "error" in bt_result:
                    error_msg = bt_result["error"]
                    logger.warning("イテレーション%d バックテストエラー: %s", iteration_num, error_msg)
                    # エラーでも IterationResult として記録
                    it_result = IterationResult(
                        iteration=iteration_num,
                        signal_config_dict=dict(current_config_dict),
                        backtest_result=bt_result,
                        ai_reasoning=f"バックテストエラー: {error_msg}",
                        changes_description="初回" if i == 0 else "調整後",
                    )
                    progress.iterations.append(it_result)

                    # 最初のイテレーションでエラーなら条件緩和を試みる
                    if i == 0:
                        # シグナル0件の場合、AIに条件緩和を依頼
                        result_summary = {
                            "n_signals": 0, "n_valid": 0,
                            "mean_excess": 0.0, "p_value": 1.0,
                            "cohens_d": 0.0, "win_rate": 0.0,
                            "is_significant": False,
                            "error": error_msg,
                        }
                        previous_iterations_summary.append({
                            "iteration": iteration_num,
                            "n_signals": 0,
                            "mean_excess": 0.0,
                            "p_value": 1.0,
                            "cohens_d": 0.0,
                            "changes_description": "初回（エラー）",
                        })
                        if i < max_iterations - 1:
                            notify("executing", f"イテレーション {iteration_num}/{max_iterations}: パラメータ調整中...")
                            adjusted = self.param_selector.adjust_parameters(
                                hypothesis=hypothesis,
                                current_config_dict=current_config_dict,
                                result_summary=result_summary,
                                iteration=iteration_num + 1,
                                max_iterations=max_iterations,
                                previous_iterations=previous_iterations_summary,
                            )
                            if adjusted is not None:
                                current_config_dict = adjusted.signal_config_dict
                                continue
                        break
                    break

                # 成功したイテレーション結果を記録
                stats = bt_result.get("statistics", {})
                changes_desc = "初回" if i == 0 else "AIによるパラメータ調整"

                it_result = IterationResult(
                    iteration=iteration_num,
                    signal_config_dict=dict(current_config_dict),
                    backtest_result=bt_result,
                    ai_reasoning="",
                    changes_description=changes_desc,
                )
                progress.iterations.append(it_result)

                # 結果サマリー
                n_valid = bt_result.get("backtest", bt_result).get("n_valid_signals", 0)
                result_summary = {
                    "n_signals": stats.get("n_signals", n_valid),
                    "n_valid": stats.get("n_excess", n_valid),
                    "mean_excess": stats.get("excess_mean", 0.0),
                    "p_value": stats.get("p_value", 1.0),
                    "cohens_d": stats.get("cohens_d", 0.0),
                    "win_rate": stats.get("win_rate", 0.0),
                    "is_significant": stats.get("is_significant", False),
                }

                previous_iterations_summary.append({
                    "iteration": iteration_num,
                    "n_signals": result_summary["n_signals"],
                    "mean_excess": result_summary["mean_excess"],
                    "p_value": result_summary["p_value"],
                    "cohens_d": result_summary["cohens_d"],
                    "changes_description": changes_desc,
                })

                # 最終イテレーション → ループ終了
                if i >= max_iterations - 1:
                    break

                # AI にパラメータ調整を依頼
                notify("executing", f"イテレーション {iteration_num}/{max_iterations}: AIがパラメータ調整中...")
                adjusted = self.param_selector.adjust_parameters(
                    hypothesis=hypothesis,
                    current_config_dict=current_config_dict,
                    result_summary=result_summary,
                    iteration=iteration_num + 1,
                    max_iterations=max_iterations,
                    previous_iterations=previous_iterations_summary,
                )

                if adjusted is None:
                    # AI判断: これ以上改善なし
                    logger.info("AI判断により早期停止 (イテレーション%d)", iteration_num)
                    break

                current_config_dict = adjusted.signal_config_dict
                # 次のイテレーションに理由を記録
                it_result.ai_reasoning = adjusted.reasoning

            # --- 2. ベスト結果選択 ---
            notify("interpreting", "ベスト結果を選択中...")
            best_idx = self._select_best_iteration(progress.iterations)
            progress.best_iteration_index = best_idx

            if best_idx is not None:
                best_it = progress.iterations[best_idx]
                progress.best_result = best_it.backtest_result
            elif progress.iterations:
                # スコア計算不能でも最後の結果を使用
                progress.best_iteration_index = len(progress.iterations) - 1
                progress.best_result = progress.iterations[-1].backtest_result
            else:
                raise RuntimeError("イテレーション結果がありません")

            # --- 3. AI解釈（ベスト結果に対して） ---
            best_bt = progress.best_result
            if "error" not in best_bt:
                notify("interpreting", "AIが結果を解釈中...")
                best_stats = best_bt.get("statistics", {})
                best_backtest = best_bt.get("backtest", best_bt)

                interpretation = self.interpreter.interpret(
                    idea_text=hypothesis,
                    plan=plan_data,
                    statistics_result=best_stats,
                    backtest_result=best_backtest,
                )
                progress.interpretation = interpretation
            else:
                progress.interpretation = {
                    "evaluation_label": "needs_review",
                    "confidence": 0.0,
                    "summary": f"バックテストエラー: {best_bt.get('error', '')}",
                    "reasons": [best_bt.get("error", "不明なエラー")],
                    "strengths": [],
                    "weaknesses": [],
                    "suggestions": ["パラメータを見直してください"],
                    "knowledge_entry": {"hypothesis": hypothesis, "tags": [category]},
                }

            # --- 4. DB保存 ---
            notify("saving", "結果を保存中...")
            best_stats_to_save = best_bt.get("statistics", {})
            best_backtest_to_save = best_bt.get("backtest", best_bt)

            self.db.update_run(
                run_id,
                statistics_result=best_stats_to_save,
                backtest_result=best_backtest_to_save,
                evaluation=progress.interpretation,
                evaluation_label=progress.interpretation.get("evaluation_label", "needs_review"),
                status="completed",
                finished_at=datetime.now().isoformat(),
            )

            # 知見保存
            ke = progress.interpretation.get("knowledge_entry", {})
            knowledge = self.knowledge_base.save_from_run(
                run_id=run_id,
                hypothesis=ke.get("hypothesis", hypothesis),
                evaluation=progress.interpretation,
                tags=ke.get("tags", [category]),
                plan=plan_data,
                statistics_result=best_stats_to_save,
                backtest_result=best_backtest_to_save,
            )
            progress.knowledge_id = knowledge.id

            self.db.update_idea(idea_id, status="completed")
            self.db.update_plan(plan_id, status="completed")

            notify("done", "研究完了")
            return progress

        except Exception as e:
            progress.phase = "error"
            progress.error = str(e)
            progress.message = f"エラー: {e}"
            logger.error("研究実行エラー: %s", e, exc_info=True)

            if progress.run_id:
                self.db.update_run(
                    progress.run_id,
                    status="failed",
                    evaluation={"error": str(e)},
                    finished_at=datetime.now().isoformat(),
                )

            if on_progress:
                on_progress(progress)
            return progress

    def _select_best_iteration(self, iterations: list) -> int | None:
        """イテレーション結果からベストを選択"""
        if not iterations:
            return None

        best_idx = None
        best_score = -float("inf")

        for idx, it in enumerate(iterations):
            bt = it.backtest_result
            if "error" in bt:
                continue
            stats = bt.get("statistics", {})
            score = _compute_composite_score(bt.get("backtest", bt), stats)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx
