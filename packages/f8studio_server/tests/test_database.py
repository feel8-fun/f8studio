import sqlite3
from pathlib import Path

import pytest

from f8studio_server.database import StudioDatabase
from f8studio_server.project_repository import ProjectRepository


def test_connections_commit_rollback_and_close(tmp_path: Path) -> None:
    database = StudioDatabase(tmp_path / "studio.sqlite3")
    with database.connection() as connection:
        connection.execute("CREATE TABLE sample (value INTEGER)")
        connection.execute("INSERT INTO sample VALUES (1)")
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with pytest.raises(ValueError, match="rollback"):
        with database.connection() as connection:
            connection.execute("INSERT INTO sample VALUES (2)")
            raise ValueError("rollback")
    with database.connection() as connection:
        assert connection.execute("SELECT value FROM sample").fetchall() == [(1,)]


def test_existing_project_metadata_remains_compatible(tmp_path: Path) -> None:
    path = tmp_path / "studio.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
                             "INSERT INTO metadata VALUES ('schema_version', '1');")
    connection.close()
    repository = ProjectRepository(path)
    assert repository.list_projects() == ()
