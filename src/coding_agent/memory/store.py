"""SQLite-backed persistent memory store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from coding_agent.memory.models import CommitRecord, MemoryRecord, RunRecord, ToolCallRecord


class MemoryStore:
    """Persistent storage for runs, memories, tool calls, and commits."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._ensure_writable()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS commits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    commit_hash TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def _ensure_writable(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(run_id, goal, status, summary)
                VALUES('__memory_write_check__', 'write check', 'completed', '')
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    summary=excluded.summary
                """
            )
            conn.execute("DELETE FROM runs WHERE run_id = '__memory_write_check__'")

    def add_memory(self, record: MemoryRecord) -> MemoryRecord:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO memories(kind, content, tags_json) VALUES(?, ?, ?)",
                (record.kind, record.content, json.dumps(record.tags)),
            )
            record.id = int(cursor.lastrowid)
        return record

    def search_memory(self, query: str, limit: int = 10) -> list[MemoryRecord]:
        sql = """
        SELECT * FROM memories
        WHERE content LIKE ? OR tags_json LIKE ?
        ORDER BY created_at DESC
        LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (f"%{query}%", f"%{query}%", limit)).fetchall()
        return [self._memory_from_row(row) for row in rows]

    def recent_memory(self, limit: int = 10) -> list[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._memory_from_row(row) for row in rows]

    def record_run(self, record: RunRecord) -> RunRecord:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(run_id, goal, status, summary)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    summary=excluded.summary
                """,
                (record.run_id, record.goal, record.status, record.summary),
            )
        return record

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [RunRecord(**dict(row)) for row in rows]

    def record_tool_call(self, record: ToolCallRecord) -> ToolCallRecord:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO tool_calls(run_id, tool_name, input_json, output_json) VALUES(?, ?, ?, ?)",
                (record.run_id, record.tool_name, record.input_json, record.output_json),
            )
        return record

    def record_commit(self, record: CommitRecord) -> CommitRecord:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO commits(run_id, commit_hash, message) VALUES(?, ?, ?)",
                (record.run_id, record.commit_hash, record.message),
            )
        return record

    @staticmethod
    def _memory_from_row(row: sqlite3.Row) -> MemoryRecord:
        payload = dict(row)
        payload["tags"] = json.loads(payload.pop("tags_json"))
        return MemoryRecord(**payload)
