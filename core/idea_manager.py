"""アイデアCRUD管理"""

from db.database import Database
from core.models import Idea


class IdeaManager:
    def __init__(self, db: Database):
        self.db = db

    def create(self, title: str, description: str, category: str) -> Idea:
        idea_id = self.db.create_idea(title, description, category)
        return self._to_model(self.db.get_idea(idea_id))

    def get(self, idea_id: int) -> Idea | None:
        row = self.db.get_idea(idea_id)
        return self._to_model(row) if row else None

    def list_all(self, status: str | None = None) -> list[Idea]:
        rows = self.db.list_ideas(status)
        return [self._to_model(r) for r in rows]

    def update(self, idea_id: int, **kwargs) -> Idea | None:
        self.db.update_idea(idea_id, **kwargs)
        row = self.db.get_idea(idea_id)
        return self._to_model(row) if row else None

    def delete(self, idea_id: int) -> None:
        self.db.delete_idea(idea_id)

    @staticmethod
    def _to_model(row: dict) -> Idea:
        return Idea(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            category=row["category"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
