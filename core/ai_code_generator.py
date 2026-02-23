"""AI による分析コード生成モジュール

分析計画を入力として受け取り、実行可能なPython分析コードを生成する。
固定ロジックではなく、AIが毎回異なる分析コードを生成できる構造。
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

CODE_GENERATION_PROMPT = """\
あなたは日本株市場の定量分析プログラマーです。
以下の分析計画に基づいて、実行可能なPython分析コードを生成してください。

## 分析計画
```json
{plan_json}
```

## データ取得API
データは `data_provider` オブジェクトを通じて取得します。以下のメソッドが利用可能です:

```python
# 上場銘柄一覧 → columns: [code, name, sector_17, sector_33, market]
stocks_df = data_provider.get_listed_stocks()

# 株価日足 → columns: [date, code, open, high, low, close, volume, adj_close, ...]
prices_df = data_provider.get_price_daily(
    code="7203",          # 省略時: 全銘柄
    start_date="2020-01-01",
    end_date="2024-12-31"
)

# 指数四本値（TOPIX） → columns: [date, index_code, open, high, low, close]
index_df = data_provider.get_index_prices(
    index_code="0000",    # 0000=TOPIX
    start_date="2020-01-01",
    end_date="2024-12-31"
)

# 財務サマリー
fin_df = data_provider.get_financial_summary(code="7203")

# 信用取引残高
margin_df = data_provider.get_margin_trading(code="7203", start_date=..., end_date=...)

# 業種別空売り比率
short_df = data_provider.get_short_selling(sector="...", start_date=..., end_date=...)
```

## コード生成ルール

1. コードは1つの関数 `run_analysis(data_provider)` として記述してください
2. この関数は以下の辞書を返してください:
```python
{{
    "statistics": {{
        "test_name": str,         # 分析手法名
        "condition_mean": float,   # 条件群の平均リターン
        "baseline_mean": float,    # 基準群の平均リターン
        "condition_std": float,    # 条件群の標準偏差
        "baseline_std": float,     # 基準群の標準偏差
        "t_statistic": float,      # t統計量
        "p_value": float,          # p値
        "cohens_d": float,         # 効果量
        "win_rate_condition": float,# 条件群の勝率
        "win_rate_baseline": float, # 基準群の勝率
        "n_condition": int,        # 条件群のサンプル数
        "n_baseline": int,         # 基準群のサンプル数
        "is_significant": bool,    # 有意かどうか (p < 0.05)
        # その他、分析固有の結果もここに追加可能
    }},
    "backtest": {{
        "cumulative_return": float,
        "annual_return": float,
        "sharpe_ratio": float,
        "max_drawdown": float,
        "win_rate": float,
        "total_trades": int,
        "benchmark_cumulative_return": float,
        "benchmark_annual_return": float,
        "benchmark_sharpe_ratio": float,
        "equity_curve": [  # 日次の資産推移
            {{"date": "YYYY-MM-DD", "value": float}}, ...
        ],
        "benchmark_curve": [
            {{"date": "YYYY-MM-DD", "value": float}}, ...
        ],
        "trade_log": [
            {{"date": str, "code": str, "action": str, "shares": int, "price": float}}, ...
        ],
    }},
    "metadata": {{
        "universe_codes": list,    # 分析対象銘柄コード
        "data_period": str,        # "YYYY-MM-DD ~ YYYY-MM-DD"
        "description": str,        # 分析の説明
    }}
}}
```
3. 利用可能なライブラリ: pandas, numpy, scipy.stats のみ
4. データ取得には必ず data_provider のメソッドを使用すること
5. エラーハンドリングを含めること
6. コードのみを出力してください。説明文は不要です。

## 出力形式
```python
import pandas as pd
import numpy as np
from scipy import stats

def run_analysis(data_provider):
    # ここにコードを記述
    ...
    return result
```
"""


class AiCodeGenerator:
    """AIを使用して分析コードを生成"""

    def __init__(self, ai_client: Any):
        """
        Args:
            ai_client: AIモデルクライアント。send_message(prompt) -> str を持つオブジェクト。
        """
        self.ai_client = ai_client

    def generate_code(self, plan: dict) -> str:
        """分析計画からPython分析コードを生成

        Args:
            plan: 分析計画の辞書（AiPlannerの出力）

        Returns:
            実行可能なPythonコード文字列
        """
        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
        prompt = CODE_GENERATION_PROMPT.format(plan_json=plan_json)

        try:
            response = self.ai_client.send_message(prompt)
            code = self._extract_code(response)
            self._validate_code(code)
            logger.info("分析コード生成完了 (%d文字)", len(code))
            return code
        except Exception as e:
            logger.error("AIコード生成エラー: %s", e)
            raise

    def fix_code(self, code: str, error_message: str) -> str:
        """実行エラーが発生したコードを修正

        Args:
            code: エラーが発生した元のコード
            error_message: エラーメッセージ

        Returns:
            修正されたPythonコード文字列
        """
        prompt = f"""\
以下のPython分析コードを実行したところエラーが発生しました。
エラーを修正したコードを出力してください。コードのみを出力し、説明は不要です。

## エラーメッセージ
```
{error_message}
```

## 元のコード
```python
{code}
```
"""
        try:
            response = self.ai_client.send_message(prompt)
            fixed_code = self._extract_code(response)
            self._validate_code(fixed_code)
            logger.info("コード修正完了")
            return fixed_code
        except Exception as e:
            logger.error("AIコード修正エラー: %s", e)
            raise

    def _extract_code(self, response: str | None) -> str:
        """AIレスポンスからPythonコードを抽出"""
        if not response:
            raise ValueError("AIから空の応答を受信しました")
        text = response.strip()

        # ```python ... ``` ブロックを抽出
        if "```python" in text:
            start = text.index("```python") + 9
            end = text.index("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            return text[start:end].strip()

        # コードブロックがない場合はそのまま
        return text

    def _validate_code(self, code: str) -> None:
        """生成コードの基本的な安全性チェック"""
        # 構文チェック
        compile(code, "<ai_generated>", "exec")

        # 危険な操作を禁止
        forbidden = [
            "os.system", "subprocess", "eval(", "exec(",
            "__import__", "open(", "pathlib",
            "shutil", "socket", "requests.get", "urllib",
        ]
        for token in forbidden:
            if token in code:
                raise ValueError(f"禁止された操作が含まれています: {token}")

        # run_analysis関数の存在を確認
        if "def run_analysis" not in code:
            raise ValueError("run_analysis関数が定義されていません")
