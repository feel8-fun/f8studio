import os
from pathlib import Path
import subprocess
import sys

import pytest

from f8studio_server.server_instance import single_server_instance


def test_server_lock_rejects_second_instance_and_releases_on_exit(tmp_path: Path) -> None:
    lock_path = tmp_path / "server.lock"
    with single_server_instance(lock_path):
        with pytest.raises(RuntimeError, match="already running"):
            with single_server_instance(lock_path):
                pytest.fail("second instance acquired the lock")
    with single_server_instance(lock_path):
        pass


def test_server_entrypoint_rejects_second_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    with single_server_instance():
        result = subprocess.run(
            [sys.executable, "-m", "f8studio_server", "--port", "8259"],
            capture_output=True,
            text=True,
            timeout=10,
            env=os.environ.copy(),
            check=False,
        )
    assert result.returncode != 0
    assert "A Web Studio server is already running for this user" in result.stderr
