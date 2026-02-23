"""生成された分析コードの安全な実行モジュール

AIが生成したPythonコードを制限された環境で実行し、結果を取得する。
"""

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from config import CODE_EXECUTION_TIMEOUT

logger = logging.getLogger(__name__)

# コード実行時に提供する安全なグローバル名前空間
SAFE_GLOBALS = {
    "__builtins__": {
        # 基本組み込み
        "True": True,
        "False": False,
        "None": None,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "isinstance": isinstance,
        "type": type,
        "print": print,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "Exception": Exception,
        "hasattr": hasattr,
        "getattr": getattr,
        "setattr": setattr,
    },
    "pd": pd,
    "np": np,
    "stats": scipy_stats,
    "scipy": type("module", (), {"stats": scipy_stats}),
}


class AiExecutor:
    """生成されたPython分析コードを安全に実行"""

    def __init__(self, timeout: int | None = None):
        self.timeout = timeout or CODE_EXECUTION_TIMEOUT

    def execute(self, code: str, data_provider: Any) -> dict:
        """生成コードを実行し結果を返す

        Args:
            code: run_analysis(data_provider) を含むPythonコード
            data_provider: MarketDataProviderインスタンス

        Returns:
            {
                "success": bool,
                "result": dict | None,  # run_analysisの戻り値
                "error": str | None,
                "code": str,            # 実行したコード
            }
        """
        logger.info("コード実行開始 (timeout=%ds)", self.timeout)

        try:
            result = self._run_with_timeout(code, data_provider)
            logger.info("コード実行完了")
            return {
                "success": True,
                "result": self._sanitize_result(result),
                "error": None,
                "code": code,
            }
        except FuturesTimeoutError:
            msg = f"実行タイムアウト ({self.timeout}秒)"
            logger.error(msg)
            return {"success": False, "result": None, "error": msg, "code": code}
        except Exception as e:
            msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error("コード実行エラー: %s", msg)
            return {"success": False, "result": None, "error": msg, "code": code}

    def _run_with_timeout(self, code: str, data_provider: Any) -> dict:
        """タイムアウト付きでコードを実行"""
        namespace = {**SAFE_GLOBALS, "data_provider": data_provider}

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._execute_code, code, namespace, data_provider)
            return future.result(timeout=self.timeout)

    @staticmethod
    def _execute_code(code: str, namespace: dict, data_provider: Any) -> dict:
        """コードを実行してrun_analysisを呼び出す"""
        exec(code, namespace)

        run_analysis = namespace.get("run_analysis")
        if run_analysis is None:
            raise ValueError("run_analysis関数が見つかりません")

        result = run_analysis(data_provider)

        if not isinstance(result, dict):
            raise TypeError(f"run_analysisはdictを返す必要があります (got {type(result).__name__})")

        return result

    def _sanitize_result(self, result: dict) -> dict:
        """結果をJSON互換に変換"""
        return self._convert_to_serializable(result)

    def _convert_to_serializable(self, obj: Any) -> Any:
        """numpy/pandas型をPython標準型に変換"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)[:10]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
