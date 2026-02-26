"""自動研究実行モジュール（Phase 1 探索 + グリッドサーチ方式）

研究サイクル:
  仮説入力 → AIがSignalConfigパラメータ選択（ユーザー確認）
  → Phase 1: 探索（5つの多様なアプローチを一気に試す）
  → Phase 2: グリッド設計（AIが最有望アプローチを分析 → ベースconfig + スイープパラメータを指定）
  → Phase 3: グリッド実行（itertools.productで全組み合わせを自動実行）
  → ベスト結果を選択 → AI解釈 → 知見保存

コード生成は行わない。AIはJSON出力のみ。
データはPhase開始前に1回だけプリロードし、各バックテストでは再利用する。
"""

import itertools
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config import (
    AI_RESEARCH_PHASE1_CONFIGS,
    AI_RESEARCH_GRID_MAX_COMBINATIONS,
    AI_RESEARCH_MIN_SIGNALS,
    AI_RESEARCH_MAX_STOCKS,
)
from db.database import Database
from core.ai_parameter_selector import (
    AiParameterSelector,
    GridSpecResult,
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
    backtest_result: dict           # StandardBacktester結果
    ai_reasoning: str
    changes_description: str
    phase: int = 0                  # 1=探索, 2=グリッド設計, 3=グリッド実行
    approach_name: str = ""         # アプローチ名
    grid_combo: dict | None = None  # Phase 3用、スイープ値を記録


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
    max_iterations: int = AI_RESEARCH_PHASE1_CONFIGS
    iterations: list = field(default_factory=list)  # list[IterationResult]
    # フェーズ進捗
    current_phase: int = 0          # 1=探索, 2=グリッド設計, 3=グリッド実行
    phase_label: str = ""           # "探索", "グリッド設計", "グリッド実行"
    current_config_in_phase: int = 0
    total_configs_in_phase: int = 0
    # グリッドサーチ関連
    grid_spec: dict = field(default_factory=dict)   # AIのグリッド設計結果（UI表示用）
    grid_total_combos: int = 0                       # 全組み合わせ数
    # ベスト結果
    best_iteration_index: int | None = None
    best_result: dict = field(default_factory=dict)
    # AI解釈
    interpretation: dict = field(default_factory=dict)
    best_analysis: str = ""                              # ベスト結果の深い分析（Markdown）
    next_param_suggestions: str = ""                     # 次のパラメータ提案（Markdown）
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


_PARAM_JP_LABELS = {
    "consecutive_bullish_days": "連続陽線",
    "consecutive_bearish_days": "連続陰線",
    "volume_surge_ratio": "出来高倍率",
    "volume_surge_window": "出来高MA期間",
    "price_vs_ma25": "25日線",
    "price_vs_ma75": "75日線",
    "price_vs_ma200": "200日線",
    "ma_deviation_pct": "MA乖離率",
    "rsi_lower": "RSI下限",
    "rsi_upper": "RSI上限",
    "bb_buy_below_lower": "BB下限",
    "ma_cross_short": "GC/DC短期",
    "ma_cross_long": "GC/DC長期",
    "ma_cross_type": "GC/DC方向",
    "macd_fast": "MACD短期",
    "macd_slow": "MACD長期",
    "atr_max": "ATR上限",
    "ichimoku_cloud": "一目雲",
    "ichimoku_tenkan_above_kijun": "転換線>基準線",
    "sector_relative_strength_min": "セクター相対強度",
    "margin_ratio_min": "貸借倍率下限",
    "margin_ratio_max": "貸借倍率上限",
    "margin_buy_change_pct_min": "買い残変化率下限",
    "margin_buy_change_pct_max": "買い残変化率上限",
    "margin_sell_change_pct_min": "売り残変化率下限",
    "margin_sell_change_pct_max": "売り残変化率上限",
    "margin_ratio_change_pct_min": "倍率変化率下限",
    "margin_ratio_change_pct_max": "倍率変化率上限",
    "short_selling_ratio_max": "空売り比率上限",
    "margin_buy_turnover_days_min": "買い残回転日数下限",
    "margin_buy_turnover_days_max": "買い残回転日数上限",
    "margin_sell_turnover_days_min": "売り残回転日数下限",
    "margin_sell_turnover_days_max": "売り残回転日数上限",
    "margin_buy_vol_ratio_min": "買い残対出来高比率下限",
    "margin_buy_vol_ratio_max": "買い残対出来高比率上限",
    "margin_sell_vol_ratio_min": "売り残対出来高比率下限",
    "margin_sell_vol_ratio_max": "売り残対出来高比率上限",
    "margin_buy_vol_ratio_change_pct_min": "買い残対出来高比率変化率下限",
    "margin_buy_vol_ratio_change_pct_max": "買い残対出来高比率変化率上限",
    "margin_sell_vol_ratio_change_pct_min": "売り残対出来高比率変化率下限",
    "margin_sell_vol_ratio_change_pct_max": "売り残対出来高比率変化率上限",
    "holding_period_days": "測定期間",
    "signal_logic": "ロジック",
}


def _describe_config_diff(prev: dict, curr: dict) -> str:
    """2つのSignalConfig辞書の差分を日本語で記述する"""
    all_keys = sorted(set(list(prev.keys()) + list(curr.keys())))
    parts = []
    for k in all_keys:
        label = _PARAM_JP_LABELS.get(k, k)
        old_val = prev.get(k)
        new_val = curr.get(k)
        if old_val == new_val:
            continue
        if old_val is None and new_val is not None:
            parts.append(f"{label}: 追加({new_val})")
        elif old_val is not None and new_val is None:
            parts.append(f"{label}: 削除")
        else:
            parts.append(f"{label}: {old_val}→{new_val}")
    return ", ".join(parts) if parts else "変更なし"


def _compute_composite_score(bt_result: dict, stats: dict) -> float:
    """イテレーション結果のスコアを算出（ベスト選択用）

    スコア = 統計的有意(+2) + |Cohen's d|*3 + シグナル>=20(+1) + |超過リターン|*10
    エッジの方向（ロング/ショート）に依存せず、効果の強さで評価する。
    """
    score = 0.0
    p_value = stats.get("p_value")
    p_value = p_value if p_value is not None else 1.0
    if p_value < 0.05 and bt_result.get("n_valid_signals", 0) >= 5:
        score += 2.0
    cohens_d = abs(stats.get("cohens_d", 0))
    score += cohens_d * 3.0
    n_valid = bt_result.get("n_valid_signals", 0)
    if n_valid >= AI_RESEARCH_MIN_SIGNALS:
        score += 1.0
    mean_excess = bt_result.get("mean_excess_return", 0)
    score += abs(mean_excess) * 10.0
    return score


def _iteration_to_result_dict(it: IterationResult) -> dict:
    """IterationResult を AI プロンプト用の辞書に変換する"""
    return {
        "approach_name": it.approach_name or f"Config {it.iteration}",
        "signal_config_dict": it.signal_config_dict,
        "backtest_result": it.backtest_result,
    }


def generate_grid_configs(
    grid_spec: GridSpecResult,
    max_combos: int = AI_RESEARCH_GRID_MAX_COMBINATIONS,
) -> list[dict]:
    """GridSpecResultからitertools.productで全組み合わせを生成する

    Returns:
        list[dict] — 各要素は {**base_config, **combo} のSignalConfig辞書
    """
    base = dict(grid_spec.base_config)
    sweep = grid_spec.sweep_parameters

    if not sweep:
        return [base]

    param_names = list(sweep.keys())
    value_lists = [sweep[name] for name in param_names]

    all_combos = list(itertools.product(*value_lists))
    if len(all_combos) > max_combos:
        logger.warning(
            "グリッド組み合わせ数(%d)が上限(%d)を超過。先頭%d個に絞ります。",
            len(all_combos), max_combos, max_combos,
        )
        all_combos = all_combos[:max_combos]

    configs = []
    for combo in all_combos:
        config = {**base, **dict(zip(param_names, combo))}
        configs.append(config)

    return configs


class AiResearcher:
    """AI研究エージェント: Phase 1 探索 + グリッドサーチ方式"""

    def __init__(
        self,
        db: Database,
        ai_client: Any,
        data_provider: Any,
    ):
        self.db = db
        self.data_provider = data_provider

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
        """Phase 1 探索 + グリッドサーチ研究ループを実行

        Phase 1: 探索 — 5つの多様なアプローチを一気に試す
        Phase 2: グリッド設計 — AIが最有望アプローチを分析 → ベースconfig + スイープパラメータを指定
        Phase 3: グリッド実行 — itertools.productで全組み合わせを自動実行（最大30通り）
        """
        n_phase1 = AI_RESEARCH_PHASE1_CONFIGS

        progress = ResearchProgress(
            idea_title=idea_title,
            idea_text=hypothesis,
            category=category,
            start_date=start_date,
            end_date=end_date,
            universe_filter_text=universe_filter_text,
            signal_config_dict=initial_config_dict,
            max_iterations=n_phase1,  # Phase 2完了時に動的更新
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
                "methodology": {"approach": "Phase 1 探索 + グリッドサーチ（探索→グリッド設計→グリッド実行）"},
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

            # --- データプリロード（1回だけ） ---
            notify("executing", "データをプリロード中（初回のみ・数分かかります）...")
            preloaded = self.backtester.preload_data(
                universe_config=universe_config,
                start_date=start_date,
                end_date=end_date,
                max_stocks=AI_RESEARCH_MAX_STOCKS,
                on_progress=lambda msg, pct: notify("executing", f"データプリロード: {msg}"),
            )

            iteration_counter = 0  # 全体通し番号

            # =============================================================
            # Phase 1: 探索
            # =============================================================
            progress.current_phase = 1
            progress.phase_label = "探索"
            progress.total_configs_in_phase = n_phase1
            notify("executing", f"Phase 1/3 探索: AIが{n_phase1}つの多様なアプローチを生成中...")

            configs_phase1 = self.param_selector.generate_diverse_configs(
                hypothesis=hypothesis,
                universe_desc=universe_filter_text,
                start_date=start_date,
                end_date=end_date,
                n=n_phase1,
            )

            for idx, cfg in enumerate(configs_phase1):
                iteration_counter += 1
                progress.current_iteration = iteration_counter
                progress.current_config_in_phase = idx + 1
                approach = cfg.hypothesis_mapping or f"アプローチ{idx+1}"
                notify(
                    "executing",
                    f"Phase 1/3 探索: [{idx+1}/{len(configs_phase1)}] {approach} を実行中...",
                )

                signal_config = dict_to_signal_config(cfg.signal_config_dict)
                bt_result = self.backtester.run_with_preloaded_data(
                    signal_config=signal_config,
                    preloaded=preloaded,
                    on_progress=lambda msg, pct, _a=approach, _i=idx: notify(
                        "executing",
                        f"Phase 1/3 探索: [{_i+1}/{len(configs_phase1)}] {_a} — {msg}",
                    ),
                )

                it_result = IterationResult(
                    iteration=iteration_counter,
                    signal_config_dict=cfg.signal_config_dict,
                    backtest_result=bt_result,
                    ai_reasoning=cfg.reasoning,
                    changes_description=approach,
                    phase=1,
                    approach_name=approach,
                )
                progress.iterations.append(it_result)

            # =============================================================
            # Phase 2: グリッド設計
            # =============================================================
            progress.current_phase = 2
            progress.phase_label = "グリッド設計"
            progress.total_configs_in_phase = 0
            notify("executing", "Phase 2/3 グリッド設計: AIがPhase 1の全結果を分析中...")

            phase1_results = [
                _iteration_to_result_dict(it) for it in progress.iterations if it.phase == 1
            ]
            grid_spec = self.param_selector.specify_grid(
                hypothesis=hypothesis,
                all_results=phase1_results,
                max_combos=AI_RESEARCH_GRID_MAX_COMBINATIONS,
            )

            # グリッド設計結果をprogressに保存（UI表示用）
            progress.grid_spec = {
                "analysis": grid_spec.analysis,
                "selected_approach": grid_spec.selected_approach,
                "base_config": grid_spec.base_config,
                "sweep_parameters": grid_spec.sweep_parameters,
                "reasoning": grid_spec.reasoning,
            }

            # グリッド組み合わせ生成
            grid_configs = generate_grid_configs(grid_spec)
            n_grid = len(grid_configs)
            progress.grid_total_combos = n_grid
            progress.max_iterations = n_phase1 + n_grid  # 動的更新
            notify(
                "executing",
                f"Phase 2/3 グリッド設計完了: {n_grid}通りの組み合わせを生成 "
                f"(ベース: {grid_spec.selected_approach})",
            )

            # =============================================================
            # Phase 3: グリッド実行
            # =============================================================
            progress.current_phase = 3
            progress.phase_label = "グリッド実行"
            progress.total_configs_in_phase = n_grid
            notify("executing", f"Phase 3/3 グリッド実行: {n_grid}通りの組み合わせを自動試行中...")

            sweep_param_names = list(grid_spec.sweep_parameters.keys())

            for idx, config_dict in enumerate(grid_configs):
                iteration_counter += 1
                progress.current_iteration = iteration_counter
                progress.current_config_in_phase = idx + 1

                # スイープ値を記録
                combo = {k: config_dict[k] for k in sweep_param_names if k in config_dict}
                combo_desc = ", ".join(f"{k}={v}" for k, v in combo.items())
                approach = f"Grid [{idx+1}/{n_grid}] {combo_desc}"

                notify(
                    "executing",
                    f"Phase 3/3 グリッド実行: [{idx+1}/{n_grid}] {combo_desc}",
                )

                signal_config = dict_to_signal_config(config_dict)
                bt_result = self.backtester.run_with_preloaded_data(
                    signal_config=signal_config,
                    preloaded=preloaded,
                    on_progress=lambda msg, pct, _desc=combo_desc, _i=idx: notify(
                        "executing",
                        f"Phase 3/3 グリッド実行: [{_i+1}/{n_grid}] {_desc} — {msg}",
                    ),
                )

                it_result = IterationResult(
                    iteration=iteration_counter,
                    signal_config_dict=config_dict,
                    backtest_result=bt_result,
                    ai_reasoning=grid_spec.reasoning,
                    changes_description=combo_desc,
                    phase=3,
                    approach_name=approach,
                    grid_combo=combo,
                )
                progress.iterations.append(it_result)

            # =============================================================
            # ベスト結果選択 + AI解釈
            # =============================================================
            notify("interpreting", "ベスト結果を選択中...")
            best_idx = self._select_best_iteration(progress.iterations)
            progress.best_iteration_index = best_idx

            if best_idx is not None:
                best_it = progress.iterations[best_idx]
                progress.best_result = best_it.backtest_result
            elif progress.iterations:
                progress.best_iteration_index = len(progress.iterations) - 1
                progress.best_result = progress.iterations[-1].backtest_result
            else:
                raise RuntimeError("イテレーション結果がありません")

            # AI解釈（ベスト結果に対して）
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

                # ベスト結果の深い分析（パラメータ設計ロジック・有効要因の言語化）
                notify("interpreting", "AIがベスト結果を詳細分析中...")
                best_it = progress.iterations[best_idx]
                progress.best_analysis = self.interpreter.analyze_best_result(
                    hypothesis=hypothesis,
                    iterations=progress.iterations,
                    best_iteration=best_it,
                    grid_spec=progress.grid_spec or None,
                )

                # 次のパラメータ提案（現在の体系では測れない条件の提案）
                notify("interpreting", "AIが追加パラメータを提案中...")
                from core.ai_parameter_selector import _SIGNAL_CONFIG_SCHEMA
                progress.next_param_suggestions = self.interpreter.suggest_next_parameters(
                    hypothesis=hypothesis,
                    iterations=progress.iterations,
                    best_iteration=best_it,
                    best_analysis=progress.best_analysis,
                    available_params_schema=_SIGNAL_CONFIG_SCHEMA,
                )
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

            # --- DB保存 ---
            notify("saving", "結果を保存中...")
            best_stats_to_save = best_bt.get("statistics", {})
            best_backtest_to_save = best_bt.get("backtest", best_bt)

            self.db.update_run(
                run_id,
                statistics_result=best_stats_to_save,
                backtest_result=best_backtest_to_save,
                evaluation=progress.interpretation,
                evaluation_label=progress.interpretation.get("evaluation_label", "needs_review"),
                best_analysis=progress.best_analysis or None,
                next_param_suggestions=progress.next_param_suggestions or None,
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
