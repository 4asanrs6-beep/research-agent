"""知見蓄積・検索モジュール

DB保存に加え、knowledge/ フォルダにMarkdown形式でも保存する。
フォルダ構造:
  knowledge/
    alpha/     … valid と判定された知見
    failed/    … invalid と判定された知見
    notes/     … needs_review / その他
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

from config import KNOWLEDGE_DIR
from db.database import Database
from core.models import Knowledge

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """知見の蓄積・検索・再利用（DB + Markdown二重保存）"""

    def __init__(self, db: Database, knowledge_dir: Path | None = None):
        self.db = db
        self.knowledge_dir = knowledge_dir or KNOWLEDGE_DIR
        for sub in ("alpha", "failed", "notes"):
            (self.knowledge_dir / sub).mkdir(parents=True, exist_ok=True)

    # === 保存 ===

    def save_from_run(
        self,
        run_id: int,
        hypothesis: str,
        evaluation: dict,
        tags: list[str] | None = None,
        plan: dict | None = None,
        statistics_result: dict | None = None,
        backtest_result: dict | None = None,
        generated_code: str | None = None,
    ) -> Knowledge:
        """分析実行結果から知見を自動生成・保存（DB + Markdown）"""
        label = evaluation.get("label") or evaluation.get("evaluation_label", "needs_review")
        reasons = evaluation.get("reasons", [])

        valid_conditions = None
        invalid_conditions = None
        if label == "valid":
            valid_conditions = "\n".join(
                r for r in reasons if "良好" in r or "超過" in r or "有意" in r or "改善" in r
            )
        elif label == "invalid":
            invalid_conditions = "\n".join(
                r for r in reasons if "低い" in r or "未達" in r or "ではない" in r or "小さい" in r
            )

        summary = f"判定: {label} (信頼度: {evaluation.get('confidence', 0):.1%})\n"
        summary += "\n".join(f"- {r}" for r in reasons)

        kid = self.db.create_knowledge(
            hypothesis=hypothesis,
            validity=label,
            run_id=run_id,
            valid_conditions=valid_conditions,
            invalid_conditions=invalid_conditions,
            summary=summary,
            tags=tags or [],
        )
        knowledge = self._to_model(self.db.get_knowledge(kid))

        # Markdown保存
        run_data = self.db.get_run(run_id)
        self._write_markdown(
            knowledge=knowledge,
            run_data=run_data,
            plan=plan,
            evaluation=evaluation,
            statistics_result=statistics_result,
            backtest_result=backtest_result,
            generated_code=generated_code,
        )

        return knowledge

    def save(
        self,
        hypothesis: str,
        validity: str = "needs_review",
        run_id: int | None = None,
        valid_conditions: str | None = None,
        invalid_conditions: str | None = None,
        summary: str | None = None,
        tags: list[str] | None = None,
    ) -> Knowledge:
        kid = self.db.create_knowledge(
            hypothesis=hypothesis,
            validity=validity,
            run_id=run_id,
            valid_conditions=valid_conditions,
            invalid_conditions=invalid_conditions,
            summary=summary,
            tags=tags or [],
        )
        knowledge = self._to_model(self.db.get_knowledge(kid))

        run_data = self.db.get_run(run_id) if run_id else None
        self._write_markdown(knowledge=knowledge, run_data=run_data)

        return knowledge

    # === 読み取り・検索 ===

    def get(self, knowledge_id: int) -> Knowledge | None:
        row = self.db.get_knowledge(knowledge_id)
        return self._to_model(row) if row else None

    def search(
        self,
        validity: str | None = None,
        tag: str | None = None,
        query: str | None = None,
    ) -> list[Knowledge]:
        rows = self.db.list_knowledge(validity=validity, tag=tag, search=query)
        return [self._to_model(r) for r in rows]

    def update(self, knowledge_id: int, **kwargs) -> Knowledge | None:
        self.db.update_knowledge(knowledge_id, **kwargs)
        row = self.db.get_knowledge(knowledge_id)
        if row:
            knowledge = self._to_model(row)
            self._write_markdown(knowledge=knowledge)
            return knowledge
        return None

    def delete(self, knowledge_id: int) -> None:
        knowledge = self.get(knowledge_id)
        if knowledge:
            md_path = self._get_md_path(knowledge)
            if md_path.exists():
                md_path.unlink()
        self.db.delete_knowledge(knowledge_id)

    def load_all_markdown(self) -> list[dict]:
        """knowledge/ 内の全Markdownファイルを読み込み、辞書のリストで返す。
        将来AIが知見フォルダを一括参照するためのメソッド。
        """
        results = []
        for md_file in sorted(self.knowledge_dir.rglob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            results.append({
                "path": str(md_file.relative_to(self.knowledge_dir)),
                "folder": md_file.parent.name,
                "content": text,
            })
        return results

    # === Markdown書き出し ===

    def _get_md_path(self, knowledge: Knowledge) -> Path:
        """知見のvalidityに応じたフォルダにMarkdownパスを返す"""
        folder_map = {
            "valid": "alpha",
            "invalid": "failed",
            "needs_review": "notes",
        }
        folder = folder_map.get(knowledge.validity, "notes")
        safe_name = self._safe_filename(knowledge.hypothesis, knowledge.id)
        return self.knowledge_dir / folder / f"{safe_name}.md"

    def _write_markdown(
        self,
        knowledge: Knowledge,
        run_data: dict | None = None,
        plan: dict | None = None,
        evaluation: dict | None = None,
        statistics_result: dict | None = None,
        backtest_result: dict | None = None,
        generated_code: str | None = None,
    ) -> Path:
        """知見をMarkdownファイルとして書き出す"""
        md_path = self._get_md_path(knowledge)

        lines = []
        lines.append(f"# {knowledge.hypothesis}")
        lines.append("")
        lines.append(f"- **Knowledge ID**: {knowledge.id}")
        lines.append(f"- **Run ID**: {knowledge.run_id or 'N/A'}")
        lines.append(f"- **判定**: {knowledge.validity}")
        lines.append(f"- **作成日**: {knowledge.created_at}")
        if knowledge.tags:
            lines.append(f"- **タグ**: {', '.join(knowledge.tags)}")
        lines.append("")

        # --- 分析条件 ---
        lines.append("## 分析条件")
        lines.append("")
        if plan:
            universe = plan.get("universe", {})
            period = plan.get("analysis_period", {})
            methodology = plan.get("methodology", {})
            lines.append(f"- **対象**: {universe.get('detail', universe.get('type', 'N/A'))}")
            lines.append(f"- **期間**: {period.get('start_date', '?')} ~ {period.get('end_date', '?')}")
            lines.append(f"- **アプローチ**: {methodology.get('approach', 'N/A')}")
            steps = methodology.get("steps", [])
            if steps:
                for i, s in enumerate(steps, 1):
                    lines.append(f"  {i}. {s}")
        elif run_data:
            snap = run_data.get("plan_snapshot", {})
            if isinstance(snap, dict):
                lines.append(f"- **分析手法**: {snap.get('analysis_method', 'N/A')}")
                lines.append(f"- **期間**: {snap.get('start_date', '?')} ~ {snap.get('end_date', '?')}")
                lines.append(f"- **ユニバース**: {snap.get('universe', 'N/A')}")
        else:
            lines.append("（分析条件情報なし）")
        lines.append("")

        # --- 統計結果 ---
        stats = statistics_result
        if not stats and run_data:
            stats = run_data.get("statistics_result")
        if stats and isinstance(stats, dict) and "error" not in stats:
            lines.append("## 統計結果")
            lines.append("")
            lines.append("| 指標 | 値 |")
            lines.append("|------|-----|")
            stat_fields = [
                ("p値", "p_value", ".4f"),
                ("t統計量", "t_statistic", ".3f"),
                ("Cohen's d", "cohens_d", ".3f"),
                ("条件群平均", "condition_mean", ".4%"),
                ("基準群平均", "baseline_mean", ".4%"),
                ("条件群勝率", "win_rate_condition", ".1%"),
                ("基準群勝率", "win_rate_baseline", ".1%"),
                ("条件群N", "n_condition", "d"),
                ("基準群N", "n_baseline", "d"),
            ]
            for label, key, fmt in stat_fields:
                val = stats.get(key)
                if val is not None:
                    lines.append(f"| {label} | {val:{fmt}} |")
            sig = stats.get("is_significant")
            if sig is not None:
                lines.append(f"| 有意性 | {'有意' if sig else '非有意'} |")
            lines.append("")

        # --- バックテスト結果 ---
        bt = backtest_result
        if not bt and run_data:
            bt = run_data.get("backtest_result")
        if bt and isinstance(bt, dict) and "error" not in bt:
            lines.append("## バックテスト結果")
            lines.append("")
            lines.append("| 指標 | 値 |")
            lines.append("|------|-----|")
            bt_fields = [
                ("累計リターン", "cumulative_return", ".2%"),
                ("年率リターン", "annual_return", ".2%"),
                ("シャープ比", "sharpe_ratio", ".2f"),
                ("最大DD", "max_drawdown", ".2%"),
                ("勝率", "win_rate", ".1%"),
                ("取引回数", "total_trades", "d"),
                ("BM累計リターン", "benchmark_cumulative_return", ".2%"),
            ]
            for label, key, fmt in bt_fields:
                val = bt.get(key)
                if val is not None:
                    lines.append(f"| {label} | {val:{fmt}} |")
            lines.append("")

        # --- 解釈 ---
        interp = evaluation
        if not interp and run_data:
            interp = run_data.get("evaluation")
        if interp and isinstance(interp, dict) and "error" not in interp:
            lines.append("## 解釈")
            lines.append("")
            if interp.get("summary"):
                lines.append(interp["summary"])
                lines.append("")
            reasons = interp.get("reasons", [])
            if reasons:
                lines.append("**判断理由:**")
                for r in reasons:
                    lines.append(f"- {r}")
                lines.append("")
            for section, title in [("strengths", "強み"), ("weaknesses", "弱み"), ("suggestions", "改善提案")]:
                items = interp.get(section, [])
                if items:
                    lines.append(f"**{title}:**")
                    for item in items:
                        lines.append(f"- {item}")
                    lines.append("")

        # --- 結論 ---
        lines.append("## 結論")
        lines.append("")
        if knowledge.validity == "valid":
            lines.append("**この仮説は有効と判定されました。**")
        elif knowledge.validity == "invalid":
            lines.append("**この仮説は無効と判定されました。**")
        else:
            lines.append("**この仮説は追加検証が必要です。**")
        lines.append("")
        if knowledge.valid_conditions:
            lines.append(f"有効条件: {knowledge.valid_conditions}")
            lines.append("")
        if knowledge.invalid_conditions:
            lines.append(f"無効条件: {knowledge.invalid_conditions}")
            lines.append("")

        # --- 生成コード ---
        if generated_code:
            lines.append("## 分析コード")
            lines.append("")
            lines.append("```python")
            lines.append(generated_code)
            lines.append("```")
            lines.append("")

        # --- メタデータ ---
        lines.append("---")
        lines.append("")
        lines.append(f"_Knowledge ID: {knowledge.id} | Run ID: {knowledge.run_id or 'N/A'} | "
                      f"生成日: {knowledge.created_at}_")

        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown知見保存: %s", md_path)

        return md_path

    @staticmethod
    def _safe_filename(text: str, knowledge_id: int | None) -> str:
        """ファイル名に使える安全な文字列を生成"""
        safe = re.sub(r'[\\/*?:"<>|]', "", text)
        safe = safe.replace("\n", " ").strip()[:60]
        prefix = f"k{knowledge_id:04d}" if knowledge_id else "k0000"
        return f"{prefix}_{safe}" if safe else prefix

    @staticmethod
    def _to_model(row: dict) -> Knowledge:
        return Knowledge(
            id=row["id"],
            run_id=row.get("run_id"),
            hypothesis=row["hypothesis"],
            validity=row["validity"],
            valid_conditions=row.get("valid_conditions"),
            invalid_conditions=row.get("invalid_conditions"),
            summary=row.get("summary"),
            tags=row.get("tags", []),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
