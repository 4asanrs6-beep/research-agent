"""AIクライアント - Claude Code CLI経由

Anthropic APIを直接呼び出すのではなく、
ローカルの Claude Code CLI を subprocess 経由で呼び出す。
これにより Claude Code が knowledge/ フォルダやワークスペースを
参照しながら分析を行える。
"""

import json
import logging
import subprocess
import os
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseAiClient(ABC):
    """AIクライアントの抽象基底クラス"""

    @abstractmethod
    def send_message(self, prompt: str) -> str:
        """プロンプトを送信してテキスト応答を返す"""


class ClaudeCodeClient(BaseAiClient):
    """Claude Code CLI を subprocess 経由で呼び出すクライアント

    claude -p <prompt> を実行し、標準出力を応答として返す。
    作業ディレクトリを research-agent に設定することで、
    Claude Code が knowledge/ フォルダ等を参照可能。
    """

    def __init__(self, cwd: str | Path | None = None, timeout: int = 300):
        """
        Args:
            cwd: Claude Code を実行する作業ディレクトリ
                 (デフォルト: research-agent プロジェクトルート)
            timeout: 実行タイムアウト秒数
        """
        if cwd is None:
            from config import BASE_DIR
            self.cwd = str(BASE_DIR)
        else:
            self.cwd = str(cwd)
        self.timeout = timeout

    def send_message(self, prompt: str) -> str:
        """Claude Code CLI にプロンプトを送信して応答を得る

        プロンプトは stdin 経由で渡す（Windowsのコマンドライン長制限を回避）。
        """
        cmd = ["claude", "-p", "--output-format", "text"]

        # Claude Code セッション内から呼ぶ場合のネスト防止を回避
        env = {**os.environ}
        env.pop("CLAUDECODE", None)

        logger.info("Claude Code CLI 呼び出し (cwd=%s, timeout=%ds, prompt=%d文字)",
                     self.cwd, self.timeout, len(prompt))

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                cwd=self.cwd,
                timeout=self.timeout,
                env=env,
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if result.returncode != 0:
                logger.error("Claude Code CLI エラー (rc=%d): %s", result.returncode, stderr.strip())
                raise RuntimeError(f"Claude Code CLI エラー (rc={result.returncode}): {stderr.strip()}")

            response = stdout.strip()
            if not response:
                # stdout が空の場合、stderr にヒントがないか確認
                logger.error("Claude Code CLI: 空の応答。stderr=%s", stderr.strip())
                raise RuntimeError(
                    f"Claude Code CLI が空の応答を返しました。"
                    f"{(' stderr: ' + stderr.strip()) if stderr.strip() else ''}"
                )

            logger.info("Claude Code CLI 応答取得 (%d文字)", len(response))
            return response

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Claude Code CLI タイムアウト ({self.timeout}秒)")
        except FileNotFoundError:
            raise RuntimeError(
                "claude コマンドが見つかりません。"
                "Claude Code CLI がインストールされていることを確認してください。"
            )

    def is_available(self) -> bool:
        """Claude Code CLI が利用可能かチェック"""
        try:
            env = {**os.environ}
            env.pop("CLAUDECODE", None)
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


class DummyAiClient(BaseAiClient):
    """テスト・デモ用ダミーAIクライアント

    Claude Code CLI が利用不可の場合に使用。固定レスポンスを返す。
    """

    def send_message(self, prompt: str) -> str:
        if "分析計画" in prompt and "投資アイデア" in prompt:
            return self._dummy_plan_response()
        elif "run_analysis" in prompt or "Python分析コード" in prompt:
            return self._dummy_code_response()
        elif "修正" in prompt and "エラー" in prompt:
            return self._dummy_code_response()
        elif "評価" in prompt and "分析結果" in prompt:
            return self._dummy_interpretation_response()
        else:
            return '{"message": "ダミーレスポンス"}'

    def _dummy_plan_response(self) -> str:
        plan = {
            "plan_name": "デモ分析計画",
            "hypothesis": "月曜日の株式リターンは他の曜日と比較して有意に異なる",
            "data_requirements": {
                "price_data": True,
                "financial_data": False,
                "index_data": True,
                "margin_data": False,
                "short_selling_data": False,
                "description": "株価日足データとTOPIX指数データ",
            },
            "universe": {
                "type": "all",
                "detail": "全上場銘柄",
                "reason": "カレンダー効果は市場全体に影響するため",
            },
            "analysis_period": {
                "start_date": "2019-01-01",
                "end_date": "2024-12-31",
                "reason": "5年間の十分なサンプル期間",
            },
            "methodology": {
                "approach": "曜日別リターンの比較分析",
                "steps": [
                    "株価データ取得・前処理",
                    "曜日別リターン計算",
                    "統計検定の実施",
                    "バックテスト実行",
                ],
                "statistical_tests": ["t検定", "Mann-Whitney U検定"],
                "metrics": ["平均リターン", "勝率", "効果量"],
            },
            "backtest": {
                "strategy_description": "月曜日に買い、火曜日に売る戦略",
                "entry_rule": "月曜始値で購入",
                "exit_rule": "火曜始値で売却",
                "rebalance_frequency": "weekly",
                "benchmark": "TOPIX",
            },
            "expected_outcome": "月曜日のリターンが他の曜日と統計的に有意に異なるかを判定",
        }
        return f"```json\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n```"

    def _dummy_code_response(self) -> str:
        return '''```python
import pandas as pd
import numpy as np
from scipy import stats

def run_analysis(data_provider):
    """月曜効果の分析（デモ）"""
    try:
        prices = data_provider.get_price_daily(
            start_date="2019-01-01",
            end_date="2024-12-31"
        )
    except Exception:
        dates = pd.bdate_range("2019-01-01", "2024-12-31")
        np.random.seed(42)
        codes = ["7203", "9984", "6758"]
        rows = []
        for code in codes:
            base = 1000
            for d in dates:
                ret = np.random.normal(0.0003, 0.015)
                base *= (1 + ret)
                rows.append({"date": d, "code": code, "adj_close": base, "close": base})
        prices = pd.DataFrame(rows)

    if len(prices) == 0:
        return {
            "statistics": {"error": "データが取得できませんでした"},
            "backtest": {"error": "データなし"},
            "metadata": {"description": "データ取得失敗"}
        }

    close_col = "adj_close" if "adj_close" in prices.columns else "close"
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["code", "date"])

    prices["return"] = prices.groupby("code")[close_col].pct_change()
    prices["day_of_week"] = prices["date"].dt.dayofweek
    prices = prices.dropna(subset=["return"])

    monday = prices[prices["day_of_week"] == 0]["return"].values
    others = prices[prices["day_of_week"] != 0]["return"].values

    t_stat, p_value = stats.ttest_ind(monday, others, equal_var=False)
    pooled_std = np.sqrt(
        ((len(monday)-1)*np.std(monday,ddof=1)**2 + (len(others)-1)*np.std(others,ddof=1)**2)
        / (len(monday)+len(others)-2)
    )
    cohens_d = (np.mean(monday) - np.mean(others)) / pooled_std if pooled_std > 0 else 0

    statistics = {
        "test_name": "monday_effect",
        "condition_mean": float(np.mean(monday)),
        "baseline_mean": float(np.mean(others)),
        "condition_std": float(np.std(monday, ddof=1)),
        "baseline_std": float(np.std(others, ddof=1)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "win_rate_condition": float(np.mean(monday > 0)),
        "win_rate_baseline": float(np.mean(others > 0)),
        "n_condition": len(monday),
        "n_baseline": len(others),
        "is_significant": p_value < 0.05,
    }

    initial_capital = 10_000_000
    capital = initial_capital
    equity = []
    trades = []
    sample_code = prices["code"].unique()[0]
    code_prices = prices[prices["code"] == sample_code].copy()

    for _, row in code_prices.iterrows():
        if row["day_of_week"] == 0:
            daily_return = row["return"] * 0.5
            capital *= (1 + daily_return)
            trades.append({
                "date": str(row["date"])[:10],
                "code": sample_code,
                "action": "buy_sell",
                "shares": 100,
                "price": row[close_col],
            })
        equity.append({"date": str(row["date"])[:10], "value": capital})

    values = [e["value"] for e in equity]
    daily_rets = np.diff(values) / values[:-1] if len(values) > 1 else np.array([0])
    cum_return = (capital / initial_capital) - 1
    n_days = len(equity)
    annual_return = (1 + cum_return) ** (252 / max(n_days, 1)) - 1
    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
    peak = values[0]
    max_dd = 0
    for v in values:
        if v > peak: peak = v
        dd = (peak - v) / peak
        if dd > max_dd: max_dd = dd

    backtest = {
        "cumulative_return": float(cum_return),
        "annual_return": float(annual_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(np.mean(daily_rets > 0)) if len(daily_rets) > 0 else 0,
        "total_trades": len(trades),
        "benchmark_cumulative_return": 0.0,
        "benchmark_annual_return": 0.0,
        "benchmark_sharpe_ratio": 0.0,
        "equity_curve": equity[-252:],
        "benchmark_curve": [],
        "trade_log": trades[-20:],
    }

    metadata = {
        "universe_codes": list(prices["code"].unique()[:10]),
        "data_period": f"{prices['date'].min().strftime('%Y-%m-%d')} ~ {prices['date'].max().strftime('%Y-%m-%d')}",
        "description": "月曜効果の検証: 月曜日のリターンが他の曜日と比較して異なるかを統計的に検定",
    }

    return {"statistics": statistics, "backtest": backtest, "metadata": metadata}
```'''

    def _dummy_interpretation_response(self) -> str:
        interp = {
            "evaluation_label": "needs_review",
            "confidence": 0.45,
            "summary": "月曜効果について統計的に有意な差は確認されましたが、効果量が小さく、実運用での収益性には疑問が残ります。追加検証が推奨されます。",
            "reasons": [
                "統計的有意性はあるがp値が境界域",
                "効果量(Cohen's d)が小さい",
                "バックテストのシャープ比が低い",
                "サンプル数は十分",
            ],
            "strengths": [
                "十分なサンプル数で検証",
                "複数銘柄での検証",
            ],
            "weaknesses": [
                "取引コスト考慮後の収益性が不明",
                "市場環境による変動を未考慮",
            ],
            "suggestions": [
                "セクター別に効果を検証",
                "時期別のサブサンプル分析",
                "取引コスト込みのバックテスト",
            ],
            "knowledge_entry": {
                "hypothesis": "月曜日の株式リターンは他の曜日と統計的に異なる",
                "valid_conditions": "大型株において弱い月曜効果が観察される",
                "invalid_conditions": "取引コスト考慮後は収益性が低い",
                "tags": ["カレンダー効果", "曜日効果", "月曜効果"],
            },
        }
        return f"```json\n{json.dumps(interp, ensure_ascii=False, indent=2)}\n```"


def create_ai_client(cwd: str | Path | None = None) -> BaseAiClient:
    """AIクライアントを生成するファクトリ

    Claude Code CLI が利用可能ならそれを使用。
    利用不可の場合は DummyAiClient にフォールバック。
    """
    client = ClaudeCodeClient(cwd=cwd)
    if client.is_available():
        logger.info("Claude Code CLI を使用します")
        return client
    else:
        logger.warning("Claude Code CLI が利用不可のため、DummyAiClient を使用します")
        return DummyAiClient()
