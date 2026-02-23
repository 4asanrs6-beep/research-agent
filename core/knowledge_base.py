"""知見蓄積・検索モジュール"""

import logging

from db.database import Database
from core.models import Knowledge

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """知見の蓄積・検索・再利用"""

    def __init__(self, db: Database):
        self.db = db

    def save_from_run(
        self,
        run_id: int,
        hypothesis: str,
        evaluation: dict,
        tags: list[str] | None = None,
    ) -> Knowledge:
        """分析実行結果から知見を自動生成・保存"""
        label = evaluation.get("label", "needs_review")
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
        return self._to_model(self.db.get_knowledge(kid))

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
        return self._to_model(self.db.get_knowledge(kid))

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
        return self._to_model(row) if row else None

    def delete(self, knowledge_id: int) -> None:
        self.db.delete_knowledge(knowledge_id)

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
