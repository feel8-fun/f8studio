from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from f8studio_core.graph import PatchResult, StudioDocument, decode_document, encode_document

from .models import ProjectRecord, ProjectSummary


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StoredRequest:
    action: str
    fingerprint: str
    result: PatchResult


def utc_now_text() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected database text, got {type(value).__name__}")
    return value


def _integer(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected database integer, got {type(value).__name__}")
    return value


def _bytes(value: object) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"expected database bytes, got {type(value).__name__}")
    return value


class ProjectRepository:
    def __init__(self, database_path: Path, *, request_history_limit: int = 2048) -> None:
        if request_history_limit < 1:
            raise ValueError("request_history_limit must be positive")
        self._database_path = database_path.resolve()
        self._request_history_limit = request_history_limit
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def database_path(self) -> Path:
        return self._database_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10.0)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    graph_revision INTEGER NOT NULL,
                    layout_revision INTEGER NOT NULL,
                    document BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS processed_requests (
                    project_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    graph_changed INTEGER NOT NULL,
                    layout_changed INTEGER NOT NULL,
                    document BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (project_id, request_id),
                    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS processed_requests_created
                    ON processed_requests(project_id, created_at DESC);
                """
            )
            row = connection.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif _text(row[0]) != str(SCHEMA_VERSION):
                raise RuntimeError(f"unsupported project database schema: {_text(row[0])}")

    def create_project(self, record: ProjectRecord) -> ProjectRecord:
        document_bytes = encode_document(record.document)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO projects(
                        project_id, name, description, created_at, updated_at,
                        graph_revision, layout_revision, document
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.project_id,
                        record.name,
                        record.description,
                        record.created_at,
                        record.updated_at,
                        record.document.graph_revision,
                        record.document.layout_revision,
                        document_bytes,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise FileExistsError(f"project already exists: {record.project_id}") from exc
        return record

    def list_projects(self) -> tuple[ProjectSummary, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT project_id, name, description, created_at, updated_at,
                       graph_revision, layout_revision
                FROM projects
                ORDER BY lower(name), project_id
                """
            ).fetchall()
        return tuple(
            ProjectSummary(
                project_id=_text(row[0]),
                name=_text(row[1]),
                description=_text(row[2]),
                created_at=_text(row[3]),
                updated_at=_text(row[4]),
                graph_revision=_integer(row[5]),
                layout_revision=_integer(row[6]),
            )
            for row in rows
        )

    def get_project(self, project_id: str) -> ProjectRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT project_id, name, description, created_at, updated_at, document
                FROM projects WHERE project_id = ?
                """,
                (project_id,),
            ).fetchone()
        if row is None:
            return None
        return ProjectRecord(
            project_id=_text(row[0]),
            name=_text(row[1]),
            description=_text(row[2]),
            created_at=_text(row[3]),
            updated_at=_text(row[4]),
            document=decode_document(_bytes(row[5])),
        )

    def update_metadata(self, project_id: str, *, name: str, description: str) -> ProjectRecord:
        timestamp = utc_now_text()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE projects SET name = ?, description = ?, updated_at = ? WHERE project_id = ?",
                (name, description, timestamp, project_id),
            )
            if cursor.rowcount != 1:
                raise FileNotFoundError(f"project not found: {project_id}")
        record = self.get_project(project_id)
        if record is None:
            raise FileNotFoundError(f"project not found after update: {project_id}")
        return record

    def delete_project(self, project_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
            if cursor.rowcount != 1:
                raise FileNotFoundError(f"project not found: {project_id}")
            # Older hotkey tables have no foreign key, so remove their bindings explicitly.
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'global_hotkeys'"
            ).fetchone() is not None:
                connection.execute("DELETE FROM global_hotkeys WHERE project_id = ?", (project_id,))

    def replace_document(self, project_id: str, document: StudioDocument) -> ProjectRecord:
        timestamp = utc_now_text()
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE projects
                   SET updated_at = ?, graph_revision = ?, layout_revision = ?, document = ?
                   WHERE project_id = ?""",
                (
                    timestamp,
                    document.graph_revision,
                    document.layout_revision,
                    encode_document(document),
                    project_id,
                ),
            )
            if cursor.rowcount != 1:
                raise FileNotFoundError(f"project not found: {project_id}")
        record = self.get_project(project_id)
        if record is None:
            raise FileNotFoundError(f"project not found after restore: {project_id}")
        return record

    def lookup_request(self, project_id: str, request_id: str) -> StoredRequest | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT action, fingerprint, graph_changed, layout_changed, document
                FROM processed_requests
                WHERE project_id = ? AND request_id = ?
                """,
                (project_id, request_id),
            ).fetchone()
        if row is None:
            return None
        return StoredRequest(
            action=_text(row[0]),
            fingerprint=_text(row[1]),
            result=PatchResult(
                request_id=request_id,
                document=decode_document(_bytes(row[4])),
                graph_changed=bool(_integer(row[2])),
                layout_changed=bool(_integer(row[3])),
            ),
        )

    def commit_result(
        self,
        project_id: str,
        *,
        action: str,
        fingerprint: str,
        result: PatchResult,
    ) -> None:
        document = result.document
        document_bytes = encode_document(document)
        timestamp = utc_now_text()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE projects
                SET updated_at = ?, graph_revision = ?, layout_revision = ?, document = ?
                WHERE project_id = ?
                """,
                (
                    timestamp,
                    document.graph_revision,
                    document.layout_revision,
                    document_bytes,
                    project_id,
                ),
            )
            if cursor.rowcount != 1:
                raise FileNotFoundError(f"project not found: {project_id}")
            connection.execute(
                """
                INSERT INTO processed_requests(
                    project_id, request_id, action, fingerprint,
                    graph_changed, layout_changed, document, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    result.request_id,
                    action,
                    fingerprint,
                    int(result.graph_changed),
                    int(result.layout_changed),
                    document_bytes,
                    timestamp,
                ),
            )
            connection.execute(
                """
                DELETE FROM processed_requests
                WHERE project_id = ? AND request_id NOT IN (
                    SELECT request_id FROM processed_requests
                    WHERE project_id = ?
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT ?
                )
                """,
                (project_id, project_id, self._request_history_limit),
            )


__all__ = ["ProjectRepository", "StoredRequest", "utc_now_text"]
