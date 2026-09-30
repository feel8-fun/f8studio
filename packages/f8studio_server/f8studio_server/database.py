from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


class StudioDatabase:
    """Shared connection policy, with one transaction/connection per operation.

    Connections never cross worker threads and are closed even when a repository
    rolls back. Existing tables and project schema metadata remain compatible.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10.0)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA busy_timeout = 10000")
            with connection:
                yield connection
        finally:
            connection.close()


def database_text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected database text, got {type(value).__name__}")
    return value


def database_integer(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError(f"expected database integer, got {type(value).__name__}")
    return value


def database_bytes(value: object) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"expected database bytes, got {type(value).__name__}")
    return value
