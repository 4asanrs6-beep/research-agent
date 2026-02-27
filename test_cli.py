"""Claude CLI サブプロセス呼び出しの切り分けテスト

テスト1: PIPE方式（従来） - デッドロックの可能性あり
テスト2: ファイルstdin + PIPE stdout（前回の修正）
テスト3: 全ファイル方式（新方式） - パイプ一切不使用
テスト4: テスト3 をバックグラウンドスレッドで実行
"""
import subprocess
import os
import shutil
import tempfile
import threading
import time

# 実際のコード生成と同規模の大きいプロンプト
BIG_PROMPT = """あなたは日本株市場の定量分析プログラマーです。
以下の分析計画に基づいて、実行可能なPython分析コードを生成してください。

分析計画: 月曜日のリターンが他の曜日と統計的に有意に異なるかを検証する。
対象: 日本株全銘柄、期間: 2021-01-01 〜 2025-12-31
方法: t検定、効果量計算、バックテスト

以下のPython関数を出力してください（コードのみ、説明不要）:

```python
import pandas as pd
import numpy as np
from scipy import stats

def run_analysis(data_provider):
    # ここにコードを記述
    ...
    return {"statistics": {...}, "backtest": {...}, "metadata": {...}}
```
""" * 3  # 3倍に増やして実際のプロンプトサイズに近づける

CMD = [
    "claude", "-p",
    "--model", "sonnet",
    "--max-turns", "1",
    "--output-format", "text",
    "--tools", "",
    "--no-session-persistence",
]

ENV = {**os.environ}
ENV.pop("CLAUDECODE", None)
ENV.pop("CLAUDE_CODE_SESSION_ID", None)

TIMEOUT = 180  # 3分


def test_pipe():
    """テスト1: subprocess.run + PIPE（従来方式）"""
    print(f"[テスト1] subprocess.run + stdin pipe + capture_output (prompt={len(BIG_PROMPT)}文字)")
    start = time.time()
    try:
        result = subprocess.run(
            CMD, input=BIG_PROMPT,
            capture_output=True, encoding="utf-8", errors="replace",
            timeout=TIMEOUT, env=ENV,
        )
        elapsed = time.time() - start
        print(f"  OK: rc={result.returncode}, stdout={len(result.stdout)}文字, {elapsed:.1f}秒")
        if result.stderr:
            print(f"  stderr: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT ({TIMEOUT}秒)")
    except Exception as e:
        print(f"  ERROR: {e}")


def test_file_stdin_pipe_stdout():
    """テスト2: ファイルstdin + PIPE stdout（前回の修正）"""
    print(f"[テスト2] Popen + file stdin + PIPE stdout (prompt={len(BIG_PROMPT)}文字)")
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".txt")
    start = time.time()
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(BIG_PROMPT)
        with open(tmp_path, "r", encoding="utf-8") as sf:
            result = subprocess.run(
                CMD, stdin=sf,
                capture_output=True, encoding="utf-8", errors="replace",
                timeout=TIMEOUT, env=ENV,
            )
        elapsed = time.time() - start
        print(f"  OK: rc={result.returncode}, stdout={len(result.stdout)}文字, {elapsed:.1f}秒")
        if result.stderr:
            print(f"  stderr: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT ({TIMEOUT}秒)")
    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        os.unlink(tmp_path)


def test_all_files():
    """テスト3: stdin/stdout/stderr 全てファイル（新方式・パイプ不使用）"""
    print(f"[テスト3] Popen + ALL files (stdin/stdout/stderr) (prompt={len(BIG_PROMPT)}文字)")
    tmp_dir = tempfile.mkdtemp(prefix="claude_test_")
    stdin_path = os.path.join(tmp_dir, "stdin.txt")
    stdout_path = os.path.join(tmp_dir, "stdout.txt")
    stderr_path = os.path.join(tmp_dir, "stderr.txt")
    start = time.time()
    try:
        with open(stdin_path, "w", encoding="utf-8") as f:
            f.write(BIG_PROMPT)

        with open(stdin_path, "r", encoding="utf-8") as f_in, \
             open(stdout_path, "w", encoding="utf-8") as f_out, \
             open(stderr_path, "w", encoding="utf-8") as f_err:

            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW

            proc = subprocess.Popen(
                CMD, stdin=f_in, stdout=f_out, stderr=f_err,
                env=ENV, creationflags=creationflags,
            )
            try:
                proc.wait(timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
                print(f"  TIMEOUT ({TIMEOUT}秒)")
                return

        elapsed = time.time() - start
        with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
            stdout = f.read()
        with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
            stderr = f.read()

        print(f"  OK: rc={proc.returncode}, stdout={len(stdout)}文字, {elapsed:.1f}秒")
        if stderr.strip():
            print(f"  stderr: {stderr.strip()[:200]}")
    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_all_files_thread():
    """テスト4: テスト3 と同じ方式をバックグラウンドスレッドで実行"""
    print(f"[テスト4] テスト3 をバックグラウンドスレッドで実行")
    result_holder = {}

    def worker():
        tmp_dir = tempfile.mkdtemp(prefix="claude_test_")
        stdin_path = os.path.join(tmp_dir, "stdin.txt")
        stdout_path = os.path.join(tmp_dir, "stdout.txt")
        stderr_path = os.path.join(tmp_dir, "stderr.txt")
        start = time.time()
        try:
            with open(stdin_path, "w", encoding="utf-8") as f:
                f.write(BIG_PROMPT)

            with open(stdin_path, "r", encoding="utf-8") as f_in, \
                 open(stdout_path, "w", encoding="utf-8") as f_out, \
                 open(stderr_path, "w", encoding="utf-8") as f_err:

                creationflags = 0
                if os.name == "nt":
                    creationflags = subprocess.CREATE_NO_WINDOW

                proc = subprocess.Popen(
                    CMD, stdin=f_in, stdout=f_out, stderr=f_err,
                    env=ENV, creationflags=creationflags,
                )
                try:
                    proc.wait(timeout=TIMEOUT)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
                    result_holder["error"] = f"TIMEOUT ({TIMEOUT}秒)"
                    return

            elapsed = time.time() - start
            with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
                stdout = f.read()
            with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                stderr = f.read()

            result_holder["ok"] = f"rc={proc.returncode}, stdout={len(stdout)}文字, {elapsed:.1f}秒"
            if stderr.strip():
                result_holder["stderr"] = stderr.strip()[:200]
        except Exception as e:
            result_holder["error"] = str(e)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=TIMEOUT + 30)
    if t.is_alive():
        print("  スレッドが応答なし")
    elif "ok" in result_holder:
        print(f"  OK: {result_holder['ok']}")
        if "stderr" in result_holder:
            print(f"  stderr: {result_holder['stderr']}")
    else:
        print(f"  ERROR: {result_holder.get('error', '不明')}")


if __name__ == "__main__":
    print("=" * 60)
    print("Claude CLI サブプロセス切り分けテスト")
    print(f"タイムアウト: {TIMEOUT}秒")
    print("=" * 60)
    print()
    test_pipe()
    print()
    test_file_stdin_pipe_stdout()
    print()
    test_all_files()
    print()
    test_all_files_thread()
    print()
    print("完了")
