from __future__ import annotations

import errno
import os
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Generator


class StudioServerAlreadyRunningError(RuntimeError):
    pass


def _lock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def single_server_instance(lock_path: Path | None = None) -> Generator[None]:
    path = lock_path or Path.home() / ".f8studio-web-server.lock"
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        try:
            _lock_file(handle)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            raise StudioServerAlreadyRunningError("A Web Studio server is already running for this user") from exc
        try:
            yield
        finally:
            _unlock_file(handle)


__all__ = ["StudioServerAlreadyRunningError", "single_server_instance"]
