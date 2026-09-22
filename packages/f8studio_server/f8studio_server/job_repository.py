from __future__ import annotations

import sqlite3
from pathlib import Path

import msgspec

from .models import DeployJob, JobStatus, ServiceDeployResult


_RESULTS_DECODER = msgspec.json.Decoder(tuple[ServiceDeployResult, ...])


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


class JobRepository:
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
                CREATE TABLE IF NOT EXISTS deploy_jobs (
                    job_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    source_graph_revision INTEGER NOT NULL,
                    source_semantic_revision TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    service_results BLOB NOT NULL,
                    error_message TEXT NOT NULL,
                    UNIQUE(project_id, request_id),
                    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS deploy_jobs_semantic
                    ON deploy_jobs(project_id, source_semantic_revision, status, updated_at DESC);
                """
            )

    def create(self, job: DeployJob, *, request_fingerprint: str) -> DeployJob:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO deploy_jobs(
                    job_id, request_id, request_fingerprint, project_id,
                    source_graph_revision, source_semantic_revision, status,
                    created_at, updated_at, service_results, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.request_id,
                    request_fingerprint,
                    job.project_id,
                    job.source_graph_revision,
                    job.source_semantic_revision,
                    job.status.value,
                    job.created_at,
                    job.updated_at,
                    msgspec.json.encode(job.service_results),
                    job.error_message,
                ),
            )
        return job

    def update(self, job: DeployJob) -> DeployJob:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE deploy_jobs
                SET status = ?, updated_at = ?, service_results = ?, error_message = ?
                WHERE job_id = ?
                """,
                (
                    job.status.value,
                    job.updated_at,
                    msgspec.json.encode(job.service_results),
                    job.error_message,
                    job.job_id,
                ),
            )
            if cursor.rowcount != 1:
                raise FileNotFoundError(f"deploy job not found: {job.job_id}")
        return job

    def get(self, job_id: str) -> DeployJob | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, request_id, project_id, source_graph_revision,
                       source_semantic_revision, status, created_at, updated_at,
                       service_results, error_message
                FROM deploy_jobs WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        return None if row is None else self._job_from_row(row)

    def find_request(self, project_id: str, request_id: str) -> tuple[str, DeployJob] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_fingerprint, job_id, request_id, project_id,
                       source_graph_revision, source_semantic_revision, status,
                       created_at, updated_at, service_results, error_message
                FROM deploy_jobs WHERE project_id = ? AND request_id = ?
                """,
                (project_id, request_id),
            ).fetchone()
        if row is None:
            return None
        return _text(row[0]), self._job_from_row(row[1:])

    def find_successful(self, project_id: str, semantic_revision: str) -> DeployJob | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, request_id, project_id, source_graph_revision,
                       source_semantic_revision, status, created_at, updated_at,
                       service_results, error_message
                FROM deploy_jobs
                WHERE project_id = ? AND source_semantic_revision = ? AND status = ?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (project_id, semantic_revision, JobStatus.succeeded.value),
            ).fetchone()
        return None if row is None else self._job_from_row(row)

    def latest_for_project(self, project_id: str) -> DeployJob | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, request_id, project_id, source_graph_revision,
                       source_semantic_revision, status, created_at, updated_at,
                       service_results, error_message
                FROM deploy_jobs
                WHERE project_id = ?
                ORDER BY updated_at DESC, job_id DESC LIMIT 1
                """,
                (project_id,),
            ).fetchone()
        return None if row is None else self._job_from_row(row)

    def mark_interrupted_jobs_failed(self, *, timestamp: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE deploy_jobs
                SET status = ?, updated_at = ?, error_message = ?
                WHERE status IN (?, ?)
                """,
                (
                    JobStatus.failed.value,
                    timestamp,
                    "server restarted before deployment completed",
                    JobStatus.queued.value,
                    JobStatus.running.value,
                ),
            )
            return cursor.rowcount

    @staticmethod
    def _job_from_row(row: tuple[object, ...]) -> DeployJob:
        return DeployJob(
            job_id=_text(row[0]),
            request_id=_text(row[1]),
            project_id=_text(row[2]),
            source_graph_revision=_integer(row[3]),
            source_semantic_revision=_text(row[4]),
            status=JobStatus(_text(row[5])),
            created_at=_text(row[6]),
            updated_at=_text(row[7]),
            service_results=_RESULTS_DECODER.decode(_bytes(row[8])),
            error_message=_text(row[9]),
        )


__all__ = ["JobRepository"]
