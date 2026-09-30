from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from f8pysdk.service_runtime_tools.deploy import ServiceProcessConfig, ServiceProcessManager
from f8pysdk.service_runtime_tools.deploy import process_manager as process_manager_module
from f8pysdk.service_runtime_tools.deploy.process_manager import SUPERVISOR_GRACEFUL_STOP_TIMEOUT_S


class _Catalog:
    def __init__(self, entry_path: Path) -> None:
        self.entry_path = entry_path

    def service_entry(self, service_class: str):
        if service_class == "f8.test":
            from f8pysdk.service_runtime_tools.inventory.entry import load_service_entry
            return load_service_entry(self.entry_path.parent)
        return None


class _FakeStdin:
    def __init__(self) -> None:
        self.value = ""
        self.closed = False

    def write(self, text: str) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        self.value += str(text)
        return len(str(text))

    def flush(self) -> None:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return None

    def close(self) -> None:
        self.closed = True
        return None


class _FakeStdout:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True
        return None


class _FakeProcess:
    pid = 4242

    def __init__(self) -> None:
        self.stdout = None
        self.stdin = _FakeStdin()
        self.wait_timeouts: list[float] = []
        self.killed = False
        self.terminated = False

    def poll(self) -> int | None:
        if self.killed:
            return 0
        if self.terminated:
            return 0
        return None

    def terminate(self) -> None:
        self.terminated = True
        return None

    def wait(self, timeout: float | None = None) -> None:
        self.wait_timeouts.append(float(timeout or 0.0))
        if "stop\n" in self.stdin.value:
            self.terminated = True
            return None
        if self.terminated:
            return None
        raise subprocess.TimeoutExpired(cmd="fake", timeout=float(timeout or 0.0))

    def kill(self) -> None:
        self.killed = True
        return None


def _write_service_entry(tmp_path: Path) -> Path:
    service_dir = tmp_path / "svc"
    service_dir.mkdir()
    entry_path = service_dir / "service.yml"
    entry_path.write_text(
        "\n".join(
            [
                "schemaVersion: f8serviceEntry/1",
                "serviceClass: f8.test",
                "label: Test Service",
                "version: 0.0.1",
                "launch:",
                "  command: python",
                "  args: ['-m', 'f8_test_service']",
                "  env:",
                "    F8_ZENOH_CONNECT: tcp/launch-stale",
                "  workdir: .",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return entry_path


def test_process_manager_zenoh_env_installs_zenoh_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry_path = _write_service_entry(tmp_path)
    captured: dict[str, Any] = {}

    def fake_popen(cmd: list[str], **kwargs: Any) -> _FakeProcess:
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        captured["stdin"] = kwargs["stdin"]
        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    manager = ServiceProcessManager(_Catalog(entry_path))
    manager.start(
        ServiceProcessConfig(
            service_class="f8.test",
            service_id="svc1",
            bus_backend="zenoh",
            zenoh_config_path="/tmp/zenoh.json5",
            zenoh_connect=("tcp/127.0.0.1:7447",),
            zenoh_listen=("tcp/0.0.0.0:0",),
            zenoh_shm_pool_bytes=123456,
        )
    )

    cmd = captured["cmd"]
    env = captured["env"]
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("f8pysdk/service_runtime_tools/deploy/supervised_child.py")
    assert "--parent-pid" in cmd
    assert "--" in cmd
    assert captured["stdin"] == subprocess.PIPE
    child_cmd = cmd[cmd.index("--") + 1 :]
    assert "--bus-backend" in child_cmd
    assert child_cmd[child_cmd.index("--bus-backend") + 1] == "zenoh"
    assert "--zenoh-config" in child_cmd
    assert "--bus-backend" in cmd
    assert env["F8_BUS_BACKEND"] == "zenoh"
    assert env["F8_ZENOH_CONFIG"] == "/tmp/zenoh.json5"
    assert env["F8_ZENOH_CONNECT"] == "tcp/127.0.0.1:7447"
    assert env["F8_ZENOH_LISTEN"] == "tcp/0.0.0.0:0"
    assert env["F8_ZENOH_SHM_POOL_BYTES"] == "123456"


def test_process_manager_detached_starts_service_command_directly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry_path = _write_service_entry(tmp_path)
    captured: dict[str, Any] = {}

    def fake_popen(cmd: list[str], **kwargs: Any) -> _FakeProcess:
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        captured["stdin"] = kwargs["stdin"]
        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    manager = ServiceProcessManager(_Catalog(entry_path))
    manager.start(
        ServiceProcessConfig(
            service_class="f8.test",
            service_id="svc1",
            supervision_mode="detached",
            bus_backend="zenoh",
        )
    )

    cmd = captured["cmd"]
    assert cmd[:3] == ["python", "-m", "f8_test_service"]
    assert not any(str(part).endswith("supervised_child.py") for part in cmd)
    assert captured["stdin"] is None
    assert "--bus-backend" in cmd
    assert cmd[cmd.index("--bus-backend") + 1] == "zenoh"


def test_process_manager_rejects_invalid_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry_path = _write_service_entry(tmp_path)

    manager = ServiceProcessManager(_Catalog(entry_path))
    with pytest.raises(ValueError, match="Invalid process bus_backend"):
        manager.start(
            ServiceProcessConfig(
                service_class="f8.test",
                service_id="svc1",
                bus_backend="invalid",
            )
        )


def test_process_manager_mem_env_clears_transport_specific_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry_path = _write_service_entry(tmp_path)
    captured: dict[str, Any] = {}

    def fake_popen(cmd: list[str], **kwargs: Any) -> _FakeProcess:
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return _FakeProcess()

    monkeypatch.setenv("F8_ZENOH_CONFIG", "/tmp/stale.json5")
    monkeypatch.setenv("F8_ZENOH_CONNECT", "tcp/parent-stale")
    monkeypatch.setenv("F8_ZENOH_LISTEN", "tcp/parent-listen-stale")
    monkeypatch.setenv("F8_ZENOH_SHM_POOL_BYTES", "999")
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    manager = ServiceProcessManager(_Catalog(entry_path))
    manager.start(
        ServiceProcessConfig(
            service_class="f8.test",
            service_id="svc1",
            bus_backend="mem",
            zenoh_config_path="/tmp/ignored.json5",
            zenoh_connect=("tcp/ignored",),
            zenoh_listen=("tcp/ignored-listen",),
            zenoh_shm_pool_bytes=123456,
        )
    )

    cmd = captured["cmd"]
    env = captured["env"]
    assert "--bus-backend" in cmd
    assert cmd[cmd.index("--bus-backend") + 1] == "mem"
    assert "--zenoh-config" not in cmd
    assert "--zenoh-connect" not in cmd
    assert "--zenoh-listen" not in cmd
    assert "--zenoh-shm-pool-bytes" not in cmd
    assert env["F8_BUS_BACKEND"] == "mem"
    assert "F8_ZENOH_CONFIG" not in env
    assert "F8_ZENOH_CONNECT" not in env
    assert "F8_ZENOH_LISTEN" not in env
    assert "F8_ZENOH_SHM_POOL_BYTES" not in env


def test_process_manager_stop_requests_supervisor_graceful_stop_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry_path = _write_service_entry(tmp_path)
    proc = _FakeProcess()

    def fake_popen(cmd: list[str], **kwargs: Any) -> _FakeProcess:
        _ = (cmd, kwargs)
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    manager = ServiceProcessManager(_Catalog(entry_path))
    manager.start(ServiceProcessConfig(service_class="f8.test", service_id="svc1"))

    manager.stop("svc1")

    assert proc.stdin.value == "stop\n"
    assert proc.stdin.closed is True
    assert proc.wait_timeouts
    assert proc.wait_timeouts[0] == SUPERVISOR_GRACEFUL_STOP_TIMEOUT_S
    assert proc.terminated is True
    assert proc.killed is False


def test_process_manager_stop_closes_stdio_handles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry_path = _write_service_entry(tmp_path)
    proc = _FakeProcess()
    stdout = _FakeStdout()
    proc.stdout = stdout

    def fake_popen(cmd: list[str], **kwargs: Any) -> _FakeProcess:
        _ = (cmd, kwargs)
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    manager = ServiceProcessManager(_Catalog(entry_path))
    manager.start(ServiceProcessConfig(service_class="f8.test", service_id="svc1"))

    manager.stop("svc1")

    assert proc.stdin.closed is True
    assert stdout.closed is True


def test_process_manager_stop_falls_back_to_terminate_when_supervisor_stdin_is_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entry_path = _write_service_entry(tmp_path)
    proc = _FakeProcess()
    proc.stdin.close()

    def fake_popen(cmd: list[str], **kwargs: Any) -> _FakeProcess:
        _ = (cmd, kwargs)
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    manager = ServiceProcessManager(_Catalog(entry_path))
    manager.start(ServiceProcessConfig(service_class="f8.test", service_id="svc1"))

    manager.stop("svc1")

    assert proc.terminated is True


def test_windows_service_process_scan_matches_service_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_manager_module.os, "name", "nt")
    monkeypatch.setattr(
        process_manager_module,
        "_windows_process_command_rows",
        lambda: [
            {"ProcessId": 111, "CommandLine": '"svc.exe" --service-id player'},
            {"ProcessId": 222, "CommandLine": '"svc.exe" --service-id other'},
            {"ProcessId": 333, "CommandLine": '"svc.exe" --service-id=player'},
            {"ProcessId": 444, "CommandLine": ""},
        ],
    )

    matches = process_manager_module.find_service_processes_by_service_id("player", current_pid=111)

    assert [match.pid for match in matches] == [333]


def test_windows_service_process_scan_cache_reuses_process_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_manager_module.os, "name", "nt")
    monkeypatch.setattr(process_manager_module, "_WINDOWS_PROCESS_ROWS_CACHE", None)
    monkeypatch.setattr(process_manager_module, "_WINDOWS_PROCESS_ROWS_CACHE_TS_S", 0.0)
    calls: list[str] = []

    def fake_rows() -> list[dict[str, object]]:
        calls.append("scan")
        return [
            {"ProcessId": 101, "CommandLine": '"engine.exe" --service-id engine'},
            {"ProcessId": 202, "CommandLine": '"player.exe" --service-id player'},
        ]

    monkeypatch.setattr(process_manager_module, "_windows_process_command_rows", fake_rows)

    engine_matches = process_manager_module.find_service_processes_by_service_id(
        "engine",
        current_pid=999,
        use_cached_windows_rows=True,
    )
    player_matches = process_manager_module.find_service_processes_by_service_id(
        "player",
        current_pid=999,
        use_cached_windows_rows=True,
    )

    assert calls == ["scan"]
    assert [match.pid for match in engine_matches] == [101]
    assert [match.pid for match in player_matches] == [202]


def test_windows_service_process_scan_default_uses_fresh_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(process_manager_module.os, "name", "nt")
    monkeypatch.setattr(
        process_manager_module,
        "_WINDOWS_PROCESS_ROWS_CACHE",
        [{"ProcessId": 101, "CommandLine": '"engine.exe" --service-id stale'}],
    )
    calls: list[str] = []

    def fake_rows() -> list[dict[str, object]]:
        calls.append("scan")
        return [{"ProcessId": 202, "CommandLine": '"engine.exe" --service-id engine'}]

    monkeypatch.setattr(process_manager_module, "_windows_process_command_rows", fake_rows)

    matches = process_manager_module.find_service_processes_by_service_id("engine", current_pid=999)

    assert calls == ["scan"]
    assert [match.pid for match in matches] == [202]
