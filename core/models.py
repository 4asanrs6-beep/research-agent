"""データモデル（dataclass）"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Idea:
    id: int | None = None
    title: str = ""
    description: str = ""
    category: str = ""
    status: str = "draft"  # draft, active, completed, archived
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Plan:
    id: int | None = None
    idea_id: int = 0
    name: str = ""
    analysis_method: str = ""
    universe: str = "all"  # all, sector, individual
    universe_detail: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    parameters: dict = field(default_factory=dict)
    backtest_config: dict = field(default_factory=dict)
    status: str = "draft"  # draft, ready, running, completed
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Run:
    id: int | None = None
    plan_id: int = 0
    idea_snapshot: dict = field(default_factory=dict)
    plan_snapshot: dict = field(default_factory=dict)
    data_period: str | None = None
    universe_snapshot: list | None = None
    statistics_result: dict | None = None
    backtest_result: dict | None = None
    evaluation: dict | None = None
    evaluation_label: str | None = None  # valid, invalid, needs_review
    best_analysis: str | None = None  # AI深層分析（Markdown）
    next_param_suggestions: str | None = None  # 追加パラメータ提案（Markdown）
    status: str = "running"  # running, completed, failed
    started_at: str = ""
    finished_at: str | None = None


@dataclass
class Knowledge:
    id: int | None = None
    run_id: int | None = None
    hypothesis: str = ""
    validity: str = "needs_review"  # valid, invalid, needs_review
    valid_conditions: str | None = None
    invalid_conditions: str | None = None
    summary: str | None = None
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class StatisticsResult:
    """統計分析結果"""
    test_name: str = ""
    condition_returns: list = field(default_factory=list)
    baseline_returns: list = field(default_factory=list)
    condition_mean: float = 0.0
    baseline_mean: float = 0.0
    condition_std: float = 0.0
    baseline_std: float = 0.0
    t_statistic: float = 0.0
    p_value: float = 1.0
    cohens_d: float = 0.0
    win_rate_condition: float = 0.0
    win_rate_baseline: float = 0.0
    n_condition: int = 0
    n_baseline: int = 0
    is_significant: bool = False


@dataclass
class BacktestResult:
    """バックテスト結果"""
    cumulative_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    benchmark_cumulative_return: float = 0.0
    benchmark_annual_return: float = 0.0
    benchmark_sharpe_ratio: float = 0.0
    equity_curve: list = field(default_factory=list)
    benchmark_curve: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)
