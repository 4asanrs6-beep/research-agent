"""SQLiteデータベース操作・スキーマ管理"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ideas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'draft',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    idea_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    analysis_method TEXT NOT NULL,
                    universe TEXT NOT NULL DEFAULT 'all',
                    universe_detail TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    parameters TEXT NOT NULL DEFAULT '{}',
                    backtest_config TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'draft',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (idea_id) REFERENCES ideas(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id INTEGER NOT NULL,
                    idea_snapshot TEXT NOT NULL,
                    plan_snapshot TEXT NOT NULL,
                    data_period TEXT,
                    universe_snapshot TEXT,
                    statistics_result TEXT,
                    backtest_result TEXT,
                    evaluation TEXT,
                    evaluation_label TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    started_at TEXT NOT NULL DEFAULT (datetime('now')),
                    finished_at TEXT,
                    FOREIGN KEY (plan_id) REFERENCES plans(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    hypothesis TEXT NOT NULL,
                    validity TEXT NOT NULL DEFAULT 'needs_review',
                    valid_conditions TEXT,
                    invalid_conditions TEXT,
                    summary TEXT,
                    tags TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_plans_idea_id ON plans(idea_id);
                CREATE INDEX IF NOT EXISTS idx_runs_plan_id ON runs(plan_id);
                CREATE INDEX IF NOT EXISTS idx_knowledge_run_id ON knowledge(run_id);
                CREATE INDEX IF NOT EXISTS idx_knowledge_validity ON knowledge(validity);
            """)

    # === Ideas CRUD ===

    def create_idea(self, title: str, description: str, category: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO ideas (title, description, category) VALUES (?, ?, ?)",
                (title, description, category),
            )
            return cursor.lastrowid

    def get_idea(self, idea_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
            return dict(row) if row else None

    def list_ideas(self, status: str | None = None) -> list[dict]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM ideas WHERE status = ? ORDER BY updated_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM ideas ORDER BY updated_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def update_idea(self, idea_id: int, **kwargs) -> None:
        allowed = {"title", "description", "category", "status"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        fields["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [idea_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE ideas SET {set_clause} WHERE id = ?", values)

    def delete_idea(self, idea_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM ideas WHERE id = ?", (idea_id,))

    # === Plans CRUD ===

    def create_plan(
        self,
        idea_id: int,
        name: str,
        analysis_method: str,
        universe: str = "all",
        universe_detail: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        parameters: dict | None = None,
        backtest_config: dict | None = None,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO plans
                   (idea_id, name, analysis_method, universe, universe_detail,
                    start_date, end_date, parameters, backtest_config)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    idea_id,
                    name,
                    analysis_method,
                    universe,
                    universe_detail,
                    start_date,
                    end_date,
                    json.dumps(parameters or {}, ensure_ascii=False),
                    json.dumps(backtest_config or {}, ensure_ascii=False),
                ),
            )
            return cursor.lastrowid

    def get_plan(self, plan_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            d["parameters"] = json.loads(d["parameters"])
            d["backtest_config"] = json.loads(d["backtest_config"])
            return d

    def list_plans(self, idea_id: int | None = None) -> list[dict]:
        with self._connect() as conn:
            if idea_id:
                rows = conn.execute(
                    "SELECT * FROM plans WHERE idea_id = ? ORDER BY created_at DESC",
                    (idea_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM plans ORDER BY created_at DESC"
                ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["parameters"] = json.loads(d["parameters"])
                d["backtest_config"] = json.loads(d["backtest_config"])
                result.append(d)
            return result

    def update_plan(self, plan_id: int, **kwargs) -> None:
        allowed = {
            "name", "analysis_method", "universe", "universe_detail",
            "start_date", "end_date", "parameters", "backtest_config", "status",
        }
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        for json_field in ("parameters", "backtest_config"):
            if json_field in fields and isinstance(fields[json_field], dict):
                fields[json_field] = json.dumps(fields[json_field], ensure_ascii=False)
        fields["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [plan_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE plans SET {set_clause} WHERE id = ?", values)

    def delete_plan(self, plan_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM plans WHERE id = ?", (plan_id,))

    # === Runs CRUD ===

    def create_run(
        self,
        plan_id: int,
        idea_snapshot: dict,
        plan_snapshot: dict,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO runs (plan_id, idea_snapshot, plan_snapshot)
                   VALUES (?, ?, ?)""",
                (
                    plan_id,
                    json.dumps(idea_snapshot, ensure_ascii=False),
                    json.dumps(plan_snapshot, ensure_ascii=False),
                ),
            )
            return cursor.lastrowid

    def update_run(self, run_id: int, **kwargs) -> None:
        allowed = {
            "data_period", "universe_snapshot", "statistics_result",
            "backtest_result", "evaluation", "evaluation_label", "status", "finished_at",
        }
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        for json_field in (
            "universe_snapshot", "statistics_result", "backtest_result", "evaluation",
        ):
            if json_field in fields and isinstance(fields[json_field], (dict, list)):
                fields[json_field] = json.dumps(fields[json_field], ensure_ascii=False)
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [run_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE runs SET {set_clause} WHERE id = ?", values)

    def get_run(self, run_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
            for field in (
                "idea_snapshot", "plan_snapshot", "universe_snapshot",
                "statistics_result", "backtest_result", "evaluation",
            ):
                if d.get(field):
                    d[field] = json.loads(d[field])
            return d

    def list_runs(self, plan_id: int | None = None) -> list[dict]:
        with self._connect() as conn:
            if plan_id:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE plan_id = ? ORDER BY started_at DESC",
                    (plan_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY started_at DESC"
                ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                for field in (
                    "idea_snapshot", "plan_snapshot", "universe_snapshot",
                    "statistics_result", "backtest_result", "evaluation",
                ):
                    if d.get(field):
                        d[field] = json.loads(d[field])
                result.append(d)
            return result

    # === Knowledge CRUD ===

    def create_knowledge(
        self,
        hypothesis: str,
        validity: str = "needs_review",
        run_id: int | None = None,
        valid_conditions: str | None = None,
        invalid_conditions: str | None = None,
        summary: str | None = None,
        tags: list[str] | None = None,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO knowledge
                   (run_id, hypothesis, validity, valid_conditions,
                    invalid_conditions, summary, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    hypothesis,
                    validity,
                    valid_conditions,
                    invalid_conditions,
                    summary,
                    json.dumps(tags or [], ensure_ascii=False),
                ),
            )
            return cursor.lastrowid

    def get_knowledge(self, knowledge_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge WHERE id = ?", (knowledge_id,)
            ).fetchone()
            if not row:
                return None
            d = dict(row)
            d["tags"] = json.loads(d["tags"])
            return d

    def list_knowledge(
        self,
        validity: str | None = None,
        tag: str | None = None,
        search: str | None = None,
    ) -> list[dict]:
        with self._connect() as conn:
            query = "SELECT * FROM knowledge WHERE 1=1"
            params: list[Any] = []
            if validity:
                query += " AND validity = ?"
                params.append(validity)
            if search:
                query += " AND (hypothesis LIKE ? OR summary LIKE ? OR valid_conditions LIKE ? OR invalid_conditions LIKE ?)"
                term = f"%{search}%"
                params.extend([term, term, term, term])
            query += " ORDER BY updated_at DESC"
            rows = conn.execute(query, params).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["tags"] = json.loads(d["tags"])
                if tag and tag not in d["tags"]:
                    continue
                result.append(d)
            return result

    def update_knowledge(self, knowledge_id: int, **kwargs) -> None:
        allowed = {
            "hypothesis", "validity", "valid_conditions",
            "invalid_conditions", "summary", "tags",
        }
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        if "tags" in fields and isinstance(fields["tags"], list):
            fields["tags"] = json.dumps(fields["tags"], ensure_ascii=False)
        fields["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [knowledge_id]
        with self._connect() as conn:
            conn.execute(f"UPDATE knowledge SET {set_clause} WHERE id = ?", values)

    def delete_knowledge(self, knowledge_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
