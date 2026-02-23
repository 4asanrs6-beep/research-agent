"""自動研究実行モジュール（AI研究サイクル）

研究サイクル全体を管理:
  アイデア入力 → 分析計画生成 → コード生成 → コード実行 → 結果解釈 → 知見保存

Phase1: ユーザー入力のアイデアを使用
Phase2（将来）: AIが自動でアイデア生成
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from db.database import Database
from core.ai_planner import AiPlanner
from core.ai_code_generator import AiCodeGenerator
from core.ai_executor import AiExecutor
from core.ai_interpreter import AiInterpreter
from core.knowledge_base import KnowledgeBase
from core.filtered_provider import FilteredProvider
from core.universe_filter import UniverseFilterConfig

logger = logging.getLogger(__name__)


@dataclass
class ResearchProgress:
    """研究実行の進捗状態"""
    phase: str = "idle"  # idle, planning, coding, executing, interpreting, saving, done, error
    message: str = ""
    plan: dict = field(default_factory=dict)
    generated_code: str = ""
    execution_result: dict = field(default_factory=dict)
    interpretation: dict = field(default_factory=dict)
    run_id: int | None = None
    knowledge_id: int | None = None
    error: str | None = None


class AiResearcher:
    """AI研究エージェント: 研究サイクル全体を自動実行"""

    def __init__(
        self,
        db: Database,
        ai_client: Any,
        data_provider: Any,
        max_code_fix_attempts: int = 2,
    ):
        """
        Args:
            db: データベースインスタンス
            ai_client: AIモデルクライアント
            data_provider: MarketDataProviderインスタンス
            max_code_fix_attempts: コードエラー時の最大修正回数
        """
        self.db = db
        self.data_provider = data_provider
        self.max_code_fix_attempts = max_code_fix_attempts

        self.planner = AiPlanner(ai_client)
        self.code_generator = AiCodeGenerator(ai_client)
        self.executor = AiExecutor()
        self.interpreter = AiInterpreter(ai_client)
        self.knowledge_base = KnowledgeBase(db)

    def run_research(
        self,
        idea_text: str,
        idea_title: str = "",
        category: str = "その他",
        on_progress: Any = None,
        universe_filter_text: str = "",
        universe_config: UniverseFilterConfig | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> ResearchProgress:
        """研究サイクルを実行

        Args:
            idea_text: 投資アイデアのテキスト
            idea_title: アイデアのタイトル（未指定時は自動生成）
            category: 分析カテゴリ
            on_progress: 進捗コールバック (ResearchProgress -> None)
            universe_filter_text: ユニバースフィルタ条件のテキスト
            universe_config: 機械的フィルタ用のユニバース設定
            start_date: 分析開始日（例: "2021-01-01"）
            end_date: 分析終了日（例: "2026-02-23"）

        Returns:
            最終的なResearchProgress
        """
        progress = ResearchProgress()

        def notify(phase: str, message: str):
            progress.phase = phase
            progress.message = message
            logger.info("[%s] %s", phase, message)
            if on_progress:
                on_progress(progress)

        try:
            # --- 0. FilteredProvider でラップ ---
            if universe_config and not universe_config.is_empty():
                effective_provider = FilteredProvider(self.data_provider, universe_config)
            else:
                effective_provider = self.data_provider

            # --- 1. アイデア保存 ---
            notify("planning", "アイデアを保存中...")
            if not idea_title:
                idea_title = idea_text[:50]
            idea_id = self.db.create_idea(idea_title, idea_text, category)
            self.db.update_idea(idea_id, status="active")

            # --- 2. 分析計画生成 ---
            notify("planning", "AIが分析計画を生成中...")
            plan = self.planner.generate_plan(
                idea_text,
                universe_filter_text=universe_filter_text,
                start_date=start_date,
                end_date=end_date,
            )
            if "error" in plan:
                raise RuntimeError(f"計画生成エラー: {plan['error']}")
            progress.plan = plan

            # DB にプラン保存
            plan_id = self.db.create_plan(
                idea_id=idea_id,
                name=plan.get("plan_name", "AI生成プラン"),
                analysis_method="ai_generated",
                universe=plan.get("universe", {}).get("type", "all"),
                universe_detail=plan.get("universe", {}).get("detail"),
                start_date=plan.get("analysis_period", {}).get("start_date"),
                end_date=plan.get("analysis_period", {}).get("end_date"),
                parameters=plan.get("methodology", {}),
                backtest_config=plan.get("backtest", {}),
            )
            self.db.update_plan(plan_id, status="ready")

            # Run 作成
            idea_snapshot = self.db.get_idea(idea_id)
            plan_snapshot = self.db.get_plan(plan_id)
            run_id = self.db.create_run(plan_id, idea_snapshot, plan_snapshot)
            progress.run_id = run_id

            # --- 3. コード生成 ---
            notify("coding", "AIが分析コードを生成中...")
            code = self.code_generator.generate_code(
                plan,
                universe_filter_text=universe_filter_text,
                start_date=start_date,
                end_date=end_date,
            )
            progress.generated_code = code

            # --- 4. コード実行（エラー時はリトライ） ---
            notify("executing", "分析コードを実行中...")
            exec_result = self.executor.execute(code, effective_provider)

            for attempt in range(self.max_code_fix_attempts):
                if exec_result["success"]:
                    break
                notify(
                    "coding",
                    f"コード修正中 (試行{attempt + 2}/{self.max_code_fix_attempts + 1})...",
                )
                try:
                    code = self.code_generator.fix_code(code, exec_result["error"])
                except Exception as fix_err:
                    logger.warning("コード修正失敗 (試行%d): %s", attempt + 2, fix_err)
                    break
                progress.generated_code = code
                notify("executing", "修正コードを実行中...")
                exec_result = self.executor.execute(code, effective_provider)

            progress.execution_result = exec_result

            if not exec_result["success"]:
                raise RuntimeError(f"コード実行失敗: {exec_result['error']}")

            # 結果をRunに保存
            result_data = exec_result["result"] or {}
            statistics_to_save = result_data.get("statistics") or {}
            if result_data.get("recent_examples"):
                statistics_to_save["recent_examples"] = result_data["recent_examples"]
            self.db.update_run(
                run_id,
                statistics_result=statistics_to_save,
                backtest_result=result_data.get("backtest"),
                data_period=result_data.get("metadata", {}).get("data_period"),
                universe_snapshot=result_data.get("metadata", {}).get("universe_codes"),
            )

            # --- 5. 結果解釈 ---
            notify("interpreting", "AIが結果を解釈中...")
            interpretation = self.interpreter.interpret(
                idea_text=idea_text,
                plan=plan,
                statistics_result=result_data.get("statistics"),
                backtest_result=result_data.get("backtest"),
            )
            interpretation["generated_code"] = code
            progress.interpretation = interpretation

            # 評価をRunに保存
            self.db.update_run(
                run_id,
                evaluation=interpretation,
                evaluation_label=interpretation.get("evaluation_label", "needs_review"),
                status="completed",
                finished_at=datetime.now().isoformat(),
            )

            # --- 6. 知見保存（DB + Markdown） ---
            notify("saving", "知見を保存中...")
            ke = interpretation.get("knowledge_entry", {})
            knowledge = self.knowledge_base.save_from_run(
                run_id=run_id,
                hypothesis=ke.get("hypothesis", idea_text),
                evaluation=interpretation,
                tags=ke.get("tags", [category]),
                plan=plan,
                statistics_result=result_data.get("statistics"),
                backtest_result=result_data.get("backtest"),
                generated_code=code,
            )
            progress.knowledge_id = knowledge.id

            # アイデアのステータス更新
            self.db.update_idea(idea_id, status="completed")
            self.db.update_plan(plan_id, status="completed")

            notify("done", "研究完了")
            return progress

        except Exception as e:
            progress.phase = "error"
            progress.error = str(e)
            progress.message = f"エラー: {e}"
            logger.error("研究実行エラー: %s", e, exc_info=True)

            # Runが作成済みなら失敗ステータスに更新
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

    # === Phase2用: AIアイデア生成（将来拡張） ===

    def generate_ideas(self, context: str = "", n_ideas: int = 3) -> list[dict]:
        """AIが自動で研究アイデアを生成（将来実装）

        Args:
            context: ヒントやコンテキスト情報
            n_ideas: 生成するアイデアの数

        Returns:
            [{"title": str, "description": str, "category": str}, ...]
        """
        # Phase2で実装予定
        # 既存の知見ベースを参照し、まだ検証されていない仮説を自動生成
        raise NotImplementedError(
            "AIアイデア自動生成はPhase2で実装予定です。"
            "現在はユーザー入力のアイデアを使用してください。"
        )
