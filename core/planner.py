"""分析計画作成"""

from datetime import datetime, timedelta

from config import DEFAULT_ANALYSIS_PARAMS, BACKTEST_DEFAULTS
from db.database import Database
from core.models import Plan


# カテゴリごとの分析テンプレート
ANALYSIS_TEMPLATES = {
    "カレンダー効果": {
        "analysis_method": "calendar_effect",
        "description": "特定の日・曜日・月における株価リターンの偏りを検証",
        "parameters": {
            "effect_type": "month_of_year",  # month_of_year, day_of_week, turn_of_month
            "target_period": None,  # e.g., 1 for January
            "return_horizon": 1,  # 日数
        },
    },
    "モメンタム": {
        "analysis_method": "momentum",
        "description": "過去のリターンが将来のリターンを予測するか検証",
        "parameters": {
            "lookback_days": 20,
            "holding_days": 20,
            "n_quantiles": 5,
            "long_quantile": 5,
            "short_quantile": 1,
        },
    },
    "バリュー": {
        "analysis_method": "value",
        "description": "バリュー指標（PER, PBR等）に基づくリターン差を検証",
        "parameters": {
            "metric": "per",  # per, pbr, dividend_yield
            "n_quantiles": 5,
            "long_quantile": 1,  # 最も割安
            "holding_days": 60,
        },
    },
    "リバーサル": {
        "analysis_method": "reversal",
        "description": "短期的な株価の反転効果を検証",
        "parameters": {
            "lookback_days": 5,
            "holding_days": 5,
            "n_quantiles": 5,
            "long_quantile": 1,  # 最も下落した銘柄
        },
    },
    "ボラティリティ": {
        "analysis_method": "volatility",
        "description": "ボラティリティに基づく投資戦略の有効性を検証",
        "parameters": {
            "vol_lookback_days": 20,
            "holding_days": 20,
            "n_quantiles": 5,
            "long_quantile": 1,  # 低ボラ
        },
    },
    "セクターローテーション": {
        "analysis_method": "sector_rotation",
        "description": "セクター間のリターン差やローテーション効果を検証",
        "parameters": {
            "lookback_days": 20,
            "holding_days": 20,
            "sector_type": "sector_33",
        },
    },
    "イベントドリブン": {
        "analysis_method": "event_driven",
        "description": "特定のイベント前後のリターンを検証",
        "parameters": {
            "event_type": "earnings",  # earnings, ex_dividend, etc.
            "pre_event_days": 5,
            "post_event_days": 5,
        },
    },
    "テクニカル": {
        "analysis_method": "technical",
        "description": "テクニカル指標に基づく売買シグナルの有効性を検証",
        "parameters": {
            "indicator": "sma_cross",  # sma_cross, rsi, bollinger
            "short_window": 5,
            "long_window": 25,
            "holding_days": 10,
        },
    },
    "ファンダメンタル": {
        "analysis_method": "fundamental",
        "description": "ファンダメンタルデータに基づく投資戦略を検証",
        "parameters": {
            "metric": "roe",
            "n_quantiles": 5,
            "holding_days": 60,
        },
    },
    "その他": {
        "analysis_method": "custom",
        "description": "カスタム分析",
        "parameters": {},
    },
}


class Planner:
    def __init__(self, db: Database):
        self.db = db

    def get_template(self, category: str) -> dict:
        """カテゴリに対応する分析テンプレートを返す"""
        return ANALYSIS_TEMPLATES.get(category, ANALYSIS_TEMPLATES["その他"])

    def create_plan(
        self,
        idea_id: int,
        name: str,
        category: str,
        universe: str = "all",
        universe_detail: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        parameters: dict | None = None,
        backtest_config: dict | None = None,
    ) -> Plan:
        template = self.get_template(category)

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            years = DEFAULT_ANALYSIS_PARAMS["lookback_years"]
            start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

        merged_params = {**template["parameters"]}
        if parameters:
            merged_params.update(parameters)

        merged_backtest = {**BACKTEST_DEFAULTS}
        if backtest_config:
            merged_backtest.update(backtest_config)

        plan_id = self.db.create_plan(
            idea_id=idea_id,
            name=name,
            analysis_method=template["analysis_method"],
            universe=universe,
            universe_detail=universe_detail,
            start_date=start_date,
            end_date=end_date,
            parameters=merged_params,
            backtest_config=merged_backtest,
        )
        return self._to_model(self.db.get_plan(plan_id))

    def get(self, plan_id: int) -> Plan | None:
        row = self.db.get_plan(plan_id)
        return self._to_model(row) if row else None

    def list_plans(self, idea_id: int | None = None) -> list[Plan]:
        rows = self.db.list_plans(idea_id)
        return [self._to_model(r) for r in rows]

    def update(self, plan_id: int, **kwargs) -> Plan | None:
        self.db.update_plan(plan_id, **kwargs)
        row = self.db.get_plan(plan_id)
        return self._to_model(row) if row else None

    def delete(self, plan_id: int) -> None:
        self.db.delete_plan(plan_id)

    @staticmethod
    def _to_model(row: dict) -> Plan:
        return Plan(
            id=row["id"],
            idea_id=row["idea_id"],
            name=row["name"],
            analysis_method=row["analysis_method"],
            universe=row["universe"],
            universe_detail=row.get("universe_detail"),
            start_date=row.get("start_date"),
            end_date=row.get("end_date"),
            parameters=row.get("parameters", {}),
            backtest_config=row.get("backtest_config", {}),
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
