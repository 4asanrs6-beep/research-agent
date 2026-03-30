#!/usr/bin/env python3
"""Phase完了時の成果物チェッカー。

使い方:
    python scripts/complete_phase.py --phase 1 --question T1-Q05
    python scripts/complete_phase.py --phase 5 --question T1-Q04

各Phaseで必要な成果物が揃っているかを検証し、
不足があればエラーで止まる。
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DIARY = ROOT / "logs" / "research_diary.md"
SUMMARY = ROOT / "logs" / "iterations" / "decision_summary.md"
KNOWLEDGE = ROOT / "logs" / "experiments" / "knowledge.md"
ITERATIONS = ROOT / "logs" / "iterations"
EXPERIMENTS = ROOT / "logs" / "experiments"


def check_file_updated_recently(path: Path, minutes: int = 30) -> bool:
    """ファイルが直近N分以内に更新されたかを確認。"""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() < minutes * 60


def check_file_contains(path: Path, text: str) -> bool:
    """ファイルに特定のテキストが含まれるか。"""
    if not path.exists():
        return False
    return text in path.read_text(encoding="utf-8")


def check_phase_1(question_id: str) -> list[str]:
    """Phase 1（生成）の成果物チェック。"""
    errors = []

    # brainstorm JSONが存在するか
    brainstorm_files = list(ITERATIONS.glob("brainstorm*.json"))
    if not brainstorm_files:
        errors.append("[NG] brainstorm JSONが存在しない (logs/iterations/brainstorm*.json)")

    # brainstorm_discussion.mdに記録があるか
    discussion = ITERATIONS / "brainstorm_discussion.md"
    if not discussion.exists():
        errors.append("[NG] brainstorm_discussion.md が存在しない")

    # research_diary.mdに「選定待ち」の記述があるか
    if not check_file_contains(DIARY, "選定待ち"):
        errors.append("[NG] research_diary.md に「選定待ち」の解説がない（Phase 1完了時に書くルール）")

    # decision_summary.mdに Phase 1 完了の記録があるか
    if not check_file_contains(SUMMARY, "Phase 1"):
        errors.append("[NG] decision_summary.md に Phase 1 完了の記録がない")

    return errors


def check_phase_2(question_id: str) -> list[str]:
    """Phase 2（選定）の成果物チェック。"""
    errors = []

    # research_diary.mdに選定結果の記述があるか
    if not check_file_contains(DIARY, "選んだ") and not check_file_contains(DIARY, "選定"):
        errors.append("[NG] research_diary.md に選定結果の解説がない（Phase 2完了時に書くルール）")

    # decision_summary.mdに選定の記録があるか
    if not check_file_contains(SUMMARY, "選定"):
        errors.append("[NG] decision_summary.md に選定の記録がない")

    return errors


def check_phase_3(question_id: str) -> list[str]:
    """Phase 3（設計）の成果物チェック。"""
    errors = []
    q_slug = question_id.replace("T1-", "").lower()

    # ideas.json
    ideas_files = list(ITERATIONS.glob(f"*{q_slug}*ideas*"))
    if not ideas_files:
        errors.append(f"[NG] ideas.json が存在しない (*{q_slug}*ideas*)")

    # plan.json
    plan_files = list(ITERATIONS.glob(f"*{q_slug}*plan*"))
    if not plan_files:
        errors.append(f"[NG] plan.json が存在しない (*{q_slug}*plan*)")

    # conversation.mdにPhase 3の記録があるか
    conv = ITERATIONS / "conversation.md"
    if conv.exists() and not check_file_contains(conv, "Phase 3") and not check_file_contains(conv, "設計"):
        errors.append("[NG] conversation.md に Phase 3 の記録がない")

    # decision_summary.md更新
    if not check_file_updated_recently(SUMMARY):
        errors.append("[WARN] decision_summary.md が直近30分以内に更新されていない")

    return errors


def check_phase_4(question_id: str) -> list[str]:
    """Phase 4（実験）の成果物チェック。"""
    errors = []
    q_slug = question_id.replace("T1-", "").lower()

    # result.json
    result_files = list(ITERATIONS.glob(f"*{q_slug}*result*"))
    if not result_files:
        errors.append(f"[NG] result.json が存在しない (*{q_slug}*result*)")

    # Phase 5の成果物を先取りしていないか
    review_files = list(EXPERIMENTS.glob(f"*{q_slug}*review*"))
    if review_files:
        errors.append("[NG] Phase 5の成果物（review.md）がPhase 4完了前に存在する。先取り違反の可能性")

    # decision_summary.md更新
    if not check_file_updated_recently(SUMMARY):
        errors.append("[WARN] decision_summary.md が直近30分以内に更新されていない")

    return errors


def check_phase_5(question_id: str) -> list[str]:
    """Phase 5（議論）の成果物チェック。"""
    errors = []
    q_slug = question_id.replace("T1-", "").lower()

    # 1. review.md
    review_files = list(EXPERIMENTS.glob(f"*{q_slug}*review*"))
    if not review_files:
        errors.append(f"[NG] experiment review が存在しない (logs/experiments/*{q_slug}*review*)")

    # 2. Codex裁定の記録
    if review_files:
        review_text = review_files[0].read_text(encoding="utf-8")
        if "裁定" not in review_text and "verdict" not in review_text.lower():
            errors.append("[NG] review.md にCodex裁定の記録がない")

    # 3. knowledge.md更新
    if not check_file_contains(KNOWLEDGE, question_id) and not check_file_contains(KNOWLEDGE, q_slug):
        errors.append(f"[NG] knowledge.md に {question_id} の知見が記録されていない")

    # 4. research_diary.md更新
    if not check_file_contains(DIARY, question_id) and not check_file_contains(DIARY, q_slug):
        errors.append(f"[NG] research_diary.md に {question_id} のエントリがない")

    # 5. decision_summary.md更新
    if not check_file_updated_recently(SUMMARY):
        errors.append("[WARN] decision_summary.md が直近30分以内に更新されていない")

    return errors


PHASE_CHECKERS = {
    1: check_phase_1,
    2: check_phase_2,
    3: check_phase_3,
    4: check_phase_4,
    5: check_phase_5,
}


def main():
    parser = argparse.ArgumentParser(description="Phase完了チェッカー")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--question", type=str, required=True, help="問いのID (例: T1-Q05)")
    args = parser.parse_args()

    checker = PHASE_CHECKERS[args.phase]
    errors = checker(args.question)

    print(f"\n=== Phase {args.phase} 完了チェック: {args.question} ===\n")

    if errors:
        for e in errors:
            print(f"  {e}")
        print(f"\n{'='*50}")
        print(f"  NG: {len(errors)}件の不足があります。Phase {args.phase} は未完了です。")
        print(f"{'='*50}\n")
        sys.exit(1)
    else:
        print("  すべての成果物が揃っています。")
        print(f"\n{'='*50}")
        print(f"  OK: Phase {args.phase} 完了。")
        print(f"{'='*50}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
