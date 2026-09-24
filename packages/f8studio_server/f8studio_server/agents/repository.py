from __future__ import annotations

import sqlite3
from pathlib import Path

import msgspec

from .models import AgentRunStatus, AgentSessionRecord, AgentSessionSummary


def _bytes(value: object) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"expected database bytes, got {type(value).__name__}")
    return value


class AgentRepository:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path.resolve()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10.0)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    record BLOB NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS agent_sessions_project_updated
                    ON agent_sessions(project_id, updated_at DESC);
                """
            )

    def save(self, record: AgentSessionRecord) -> AgentSessionRecord:
        payload = msgspec.json.encode(record)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_sessions(session_id, project_id, status, updated_at, record)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    project_id = excluded.project_id,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    record = excluded.record
                """,
                (record.session_id, record.project_id, record.status.value, record.updated_at, payload),
            )
        return record

    def get(self, session_id: str) -> AgentSessionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT record FROM agent_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return msgspec.json.decode(_bytes(row[0]), type=AgentSessionRecord)

    def list(self, project_id: str | None = None) -> tuple[AgentSessionSummary, ...]:
        query = "SELECT record FROM agent_sessions"
        parameters: tuple[str, ...] = ()
        if project_id is not None:
            query += " WHERE project_id = ?"
            parameters = (project_id,)
        query += " ORDER BY updated_at DESC, session_id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        records = tuple(msgspec.json.decode(_bytes(row[0]), type=AgentSessionRecord) for row in rows)
        return tuple(
            AgentSessionSummary(
                session_id=record.session_id,
                project_id=record.project_id,
                title=record.title,
                provider_id=record.provider_id,
                model_id=record.model_id,
                status=record.status,
                updated_at=record.updated_at,
                message_count=len(record.messages),
            )
            for record in records
        )

    def interrupted(self) -> tuple[AgentSessionRecord, ...]:
        statuses = (AgentRunStatus.running.value, AgentRunStatus.waiting_for_approval.value)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT record FROM agent_sessions WHERE status IN (?, ?)",
                statuses,
            ).fetchall()
        return tuple(msgspec.json.decode(_bytes(row[0]), type=AgentSessionRecord) for row in rows)


__all__ = ["AgentRepository"]
