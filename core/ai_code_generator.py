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
# 上場銘柄一覧 → columns: [code, name, sector_17, sector_33, market,
#   sector_17_name, sector_33_name, market_name, scale_category,
#   margin_code, margin_code_name]
# market_name: "プライム" / "スタンダード" / "グロース"
# scale_category: "TOPIX Core30" / "TOPIX Large70" / "TOPIX Mid400" / "TOPIX Small 1" / "TOPIX Small 2" 等
# margin_code: "1"=制度信用, "2"=貸借銘柄(空売り可)
stocks_df = data_provider.get_listed_stocks()

# 株価日足 → columns: [date, code, open, high, low, close, volume,
#   adjustment_factor, adj_open, adj_high, adj_low, adj_close, adj_volume]
# ※ code は必須指定を推奨（code 省略時は全銘柄一括取得で非常に低速）
# ※ 複数銘柄のデータが必要な場合は銘柄ごとにループして取得すること
prices_df = data_provider.get_price_daily(
    code="7203",          # 必ず指定すること（省略すると全銘柄一括で低速）
    start_date="2020-01-01",
    end_date="2024-12-31"
)

# 【推奨パターン】複数銘柄の株価データ取得:
# all_prices = []
# for code in universe_codes:
#     df = data_provider.get_price_daily(code=code, start_date=..., end_date=...)
#     all_prices.append(df)
# prices_df = pd.concat(all_prices, ignore_index=True)

# 指数四本値（TOPIX） → columns: [date, index_code, open, high, low, close]
index_df = data_provider.get_index_prices(
    index_code="0000",    # 0000=TOPIX
    start_date="2020-01-01",
    end_date="2024-12-31"
)

# 財務サマリー
fin_df = data_provider.get_financial_summary(code="7203")

# 貸借銘柄情報 → columns: [code, margin_code, margin_code_name]
# margin_code: "1"=制度信用, "2"=貸借銘柄(空売り可)
trades_spec_df = data_provider.get_trades_spec()

# 信用取引残高
margin_df = data_provider.get_margin_trading(code="7203", start_date=..., end_date=...)

# 業種別空売り比率
short_df = data_provider.get_short_selling(sector="...", start_date=..., end_date=...)
```

## データ取得上の注意
- 分析期間は直近5年以内を推奨（例: 2021-01-01〜2026-02-23）。10年以上はデータ量過大でタイムアウトの原因になる
- 株価日足の取得は **必ず銘柄コード（code）を指定** して1銘柄ずつ取得すること。code省略の全銘柄一括取得は非常に低速
- ユニバースが大量（100銘柄超）の場合はサンプリングを検討すること（例: ランダムに50銘柄選択）
- コード実行タイムアウトは120秒。データ取得に時間がかかりすぎないよう注意すること

{universe_filter_section}
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
    "recent_examples": [
        # 条件に合致した直近10件（日付降順）
        {{"date": "YYYY-MM-DD", "description": "条件の説明", "return_pct": float}},
        ...
    ],
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
7. recent_examplesには条件に合致した直近10件を日付降順で含めること。各要素は {{"date": "YYYY-MM-DD", "description": "条件の簡潔な説明", "return_pct": float（リターン%）}} の形式とする。

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

    def generate_code(
        self,
        plan: dict,
        universe_filter_text: str = "",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        """分析計画からPython分析コードを生成

        Args:
            plan: 分析計画の辞書（AiPlannerの出力）
            universe_filter_text: ユニバースフィルタ条件のテキスト
            start_date: 分析開始日（例: "2021-01-01"）
            end_date: 分析終了日（例: "2026-02-23"）

        Returns:
            実行可能なPythonコード文字列
        """
        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)

        universe_filter_section = ""
        if universe_filter_text:
            universe_filter_section += (
                "## ユニバースフィルター条件\n"
                "以下の条件で分析対象銘柄をフィルタリングしてください。"
                "get_listed_stocks() や get_trades_spec() を使用して銘柄を絞り込むコードを生成してください:\n"
                f"{universe_filter_text}\n\n"
            )
        if start_date and end_date:
            universe_filter_section += (
                "## 分析期間の制約（必須）\n"
                "分析対象期間は固定です。コード内の日付を以下に合わせてください:\n"
                f"- 開始日: {start_date}\n"
                f"- 終了日: {end_date}\n"
                "※ 指標計算のウォームアップ（例: 20日移動平均のための追加期間）は"
                "必要に応じて開始日より前のデータを取得して構いません。\n\n"
            )

        prompt = CODE_GENERATION_PROMPT.format(
            plan_json=plan_json,
            universe_filter_section=universe_filter_section,
        )

        try:
            response = self.ai_client.send_message(prompt)
            code = self._extract_code(response)
            self._validate_code(code)
            logger.info("分析コード生成完了 (%d文字)", len(code))
            return code
        except SyntaxError as e:
            # 構文エラーは実行時にリトライ可能なのでコードを返す
            logger.warning("生成コードに構文エラーあり（実行時に修正試行）: %s", e)
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
        except SyntaxError as e:
            logger.warning("修正コードに構文エラーあり（実行時に再検出）: %s", e)
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
