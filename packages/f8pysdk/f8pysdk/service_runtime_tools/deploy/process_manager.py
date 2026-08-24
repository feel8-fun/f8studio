from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import msgspec

from ...bus import BusBackend
from .._internal.error_reporting import ExceptionLogOnce, fingerprint_exception
from ..inventory.catalog import ServiceCatalog
from ..inventory.entry import load_service_entry


logger = logging.getLogger(__name__)
SUPERVISOR_GRACEFUL_STOP_TIMEOUT_S = 0.8
TERMINATE_WAIT_TIMEOUT_S = 0.8
TASKKILL_TIMEOUT_S = 1.0
FINAL_EXIT_WAIT_TIMEOUT_S = 0.8
READER_THREAD_JOIN_TIMEOUT_S = 0.5
WINDOWS_PROCESS_SCAN_CACHE_TTL_S = 0.75
_WINDOWS_PROCESS_ROWS_CACHE_LOCK = threading.Lock()
_WINDOWS_PROCESS_ROWS_CACHE_TS_S = 0.0
_WINDOWS_PROCESS_ROWS_CACHE: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class ServiceProcessConfig:
    service_class: str
    service_id: str
    supervision_mode: Literal["studio_owned", "detached"] = "studio_owned"
    bus_backend: BusBackend = "zenoh"
    zenoh_config_path: str | None = None
    zenoh_connect: tuple[str, ...] = ()
    zenoh_listen: tuple[str, ...] = ()
    zenoh_shm_pool_bytes: int = 256 * 1024 * 1024


@dataclass(frozen=True)
class ServiceProcessMatch:
    pid: int
    cmdline: tuple[str, ...]

    def display_command(self) -> str:
        return " ".join(shlex.quote(part) for part in self.cmdline)


@dataclass(frozen=True)
class ServiceProcessTerminateResult:
    service_id: str
    matched_pids: tuple[int, ...]
    terminated_pids: tuple[int, ...]
    remaining_pids: tuple[int, ...]

    @property
    def success(self) -> bool:
        return not self.remaining_pids


def _cmdline_matches_service_id(cmdline: tuple[str, ...], service_id: str) -> bool:
    sid = str(service_id or "").strip()
    if not sid:
        return False
    for index, token in enumerate(cmdline):
        text = str(token)
        if text == "--service-id":
            value_index = index + 1
            if value_index < len(cmdline) and str(cmdline[value_index]) == sid:
                return True
            continue
        prefix = "--service-id="
        if text.startswith(prefix) and text[len(prefix) :] == sid:
            return True
    return False


def _read_proc_cmdline(path: Path) -> tuple[str, ...]:
    try:
        raw = path.read_bytes()
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return ()
    return tuple(part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part)


def find_service_processes_by_service_id(
    service_id: str,
    *,
    current_pid: int | None = None,
    use_cached_windows_rows: bool = False,
) -> list[ServiceProcessMatch]:
    sid = str(service_id or "").strip()
    if not sid:
        return []
    own_pid = os.getpid() if current_pid is None else int(current_pid)
    if os.name == "nt":
        return _find_windows_service_processes_by_service_id(
            sid,
            current_pid=own_pid,
            rows=(_cached_windows_process_command_rows() if use_cached_windows_rows else None),
        )
    if os.name != "posix":
        return []
    matches: list[ServiceProcessMatch] = []
    proc_root = Path("/proc")
    try:
        entries = list(proc_root.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError) as exc:
        logger.debug("Process scan failed: cannot list /proc", exc_info=exc)
        return []
    for entry in entries:
        name = entry.name
        if not name.isdecimal():
            continue
        pid = int(name)
        if pid == own_pid:
            continue
        cmdline = _read_proc_cmdline(entry / "cmdline")
        if not cmdline:
            continue
        if _cmdline_matches_service_id(cmdline, sid):
            matches.append(ServiceProcessMatch(pid=pid, cmdline=cmdline))
    return sorted(matches, key=lambda item: item.pid)


def _windows_process_command_rows() -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3.0,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        logger.debug("Windows process command line scan failed", exc_info=exc)
        return []
    if int(completed.returncode or 0) != 0:
        return []
    raw = str(completed.stdout or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.debug("Windows process command line JSON decode failed", exc_info=exc)
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _cached_windows_process_command_rows() -> list[dict[str, Any]]:
    global _WINDOWS_PROCESS_ROWS_CACHE
    global _WINDOWS_PROCESS_ROWS_CACHE_TS_S

    now_s = time.monotonic()
    with _WINDOWS_PROCESS_ROWS_CACHE_LOCK:
        cached = _WINDOWS_PROCESS_ROWS_CACHE
        if cached is not None and (now_s - float(_WINDOWS_PROCESS_ROWS_CACHE_TS_S)) <= WINDOWS_PROCESS_SCAN_CACHE_TTL_S:
            return [dict(row) for row in cached]

    rows = _windows_process_command_rows()
    with _WINDOWS_PROCESS_ROWS_CACHE_LOCK:
        _WINDOWS_PROCESS_ROWS_CACHE = [dict(row) for row in rows]
        _WINDOWS_PROCESS_ROWS_CACHE_TS_S = time.monotonic()
        return [dict(row) for row in rows]


def _find_windows_service_processes_by_service_id(
    service_id: str,
    *,
    current_pid: int,
    rows: list[dict[str, Any]] | None = None,
) -> list[ServiceProcessMatch]:
    sid = str(service_id or "").strip()
    if not sid:
        return []
    matches: list[ServiceProcessMatch] = []
    command_rows = _windows_process_command_rows() if rows is None else list(rows)
    for row in command_rows:
        pid_raw = row.get("ProcessId")
        try:
            pid = int(pid_raw)
        except (TypeError, ValueError):
            continue
        if int(pid) == int(current_pid):
            continue
        command_line = str(row.get("CommandLine") or "").strip()
        if not command_line:
            continue
        try:
            cmdline = tuple(shlex.split(command_line, posix=False))
        except ValueError:
            cmdline = (command_line,)
        if _cmdline_matches_service_id(cmdline, sid):
            matches.append(ServiceProcessMatch(pid=int(pid), cmdline=cmdline))
    return sorted(matches, key=lambda item: item.pid)


def _process_group_ids(matches: list[ServiceProcessMatch]) -> list[int]:
    own_pgid = os.getpgrp() if os.name == "posix" else -1
    groups: set[int] = set()
    for match in matches:
        try:
            pgid = os.getpgid(int(match.pid))
        except ProcessLookupError:
            continue
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Process group lookup failed pid=%s", match.pid, exc_info=exc)
            continue
        if pgid <= 0 or pgid == own_pgid:
            continue
        groups.add(int(pgid))
    return sorted(groups)


def _wait_until_service_processes_exit(service_id: str, *, deadline_s: float) -> list[ServiceProcessMatch]:
    remaining = find_service_processes_by_service_id(service_id)
    while remaining and time.monotonic() < deadline_s:
        time.sleep(0.05)
        remaining = find_service_processes_by_service_id(service_id)
    return remaining


def terminate_service_processes_by_service_id(
    service_id: str,
    *,
    grace_s: float = 2.0,
    kill_s: float = 2.0,
) -> ServiceProcessTerminateResult:
    sid = str(service_id or "").strip()
    matches = find_service_processes_by_service_id(sid)
    matched_pids = tuple(match.pid for match in matches)
    if not matches:
        return ServiceProcessTerminateResult(
            service_id=sid,
            matched_pids=(),
            terminated_pids=(),
            remaining_pids=(),
        )

    if os.name == "posix":
        groups = _process_group_ids(matches)
        for pgid in groups:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("SIGTERM service process group failed service_id=%s pgid=%s", sid, pgid, exc_info=exc)
        remaining = _wait_until_service_processes_exit(sid, deadline_s=time.monotonic() + max(0.0, float(grace_s)))
        if remaining:
            for pgid in _process_group_ids(remaining):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    continue
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    logger.debug("SIGKILL service process group failed service_id=%s pgid=%s", sid, pgid, exc_info=exc)
            remaining = _wait_until_service_processes_exit(
                sid,
                deadline_s=time.monotonic() + max(0.0, float(kill_s)),
            )
        remaining_pids = tuple(match.pid for match in remaining)
        return ServiceProcessTerminateResult(
            service_id=sid,
            matched_pids=matched_pids,
            terminated_pids=tuple(pid for pid in matched_pids if pid not in set(remaining_pids)),
            remaining_pids=remaining_pids,
        )

    remaining_pids: tuple[int, ...] = matched_pids
    for match in matches:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(int(match.pid)), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
                check=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
            logger.debug("taskkill external service process failed service_id=%s pid=%s", sid, match.pid, exc_info=exc)
    remaining = _wait_until_service_processes_exit(sid, deadline_s=time.monotonic() + max(0.0, float(kill_s)))
    remaining_pids = tuple(match.pid for match in remaining)
    return ServiceProcessTerminateResult(
        service_id=sid,
        matched_pids=matched_pids,
        terminated_pids=tuple(pid for pid in matched_pids if pid not in set(remaining_pids)),
        remaining_pids=remaining_pids,
    )


class ServiceProcessManager:
    """
    Launch/track local service processes based on discovery `service.yml`.
    """

    def __init__(self, catalog: ServiceCatalog | None = None) -> None:
        self._catalog = catalog or ServiceCatalog.instance()
        self._procs: dict[str, subprocess.Popen[Any]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._entries_lock = threading.Lock()
        self._exception_log_once = ExceptionLogOnce()

    def service_ids(self) -> list[str]:
        with self._entries_lock:
            return list(self._procs.keys())

    def _start_reader(self, *, service_id: str, proc: subprocess.Popen[Any], on_output: Any | None) -> None:
        if on_output is None:
            return

        def _run() -> None:
            try:
                stream = proc.stdout
                if stream is None:
                    return
                for line in iter(stream.readline, ""):
                    if line == "" and proc.poll() is not None:
                        break
                    try:
                        on_output(str(service_id), str(line))
                    except Exception as exc:
                        fp = fingerprint_exception(context="service_process_manager.on_output", exc=exc)
                        if self._exception_log_once.should_log(fp):
                            logger.error("Service output callback raised (service_id=%s)", service_id, exc_info=exc)
            except Exception as exc:
                fp = fingerprint_exception(context="service_process_manager.reader_thread", exc=exc)
                if self._exception_log_once.should_log(fp):
                    logger.error("Service output reader thread failed (service_id=%s)", service_id, exc_info=exc)

        t = threading.Thread(target=_run, name=f"svc-log:{service_id}", daemon=True)
        with self._entries_lock:
            self._threads[service_id] = t
        t.start()

    def is_running(self, service_id: str) -> bool:
        sid = str(service_id)
        with self._entries_lock:
            proc = self._procs.get(sid)
        if proc is None:
            return False
        if proc.poll() is None:
            return True
        self._cleanup_entry(sid)
        return False

    def _cleanup_entry(self, service_id: str) -> None:
        sid = str(service_id)
        with self._entries_lock:
            proc = self._procs.pop(sid, None)
            thread = self._threads.pop(sid, None)
        if proc is None:
            return
        try:
            stdin = proc.stdin
            if stdin:
                stdin.close()
        except (AttributeError, BrokenPipeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Service process stdin close failed (service_id=%s)", sid, exc_info=exc)
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=READER_THREAD_JOIN_TIMEOUT_S)
            except RuntimeError as exc:
                logger.debug("Service process reader thread join failed (service_id=%s)", sid, exc_info=exc)
        try:
            stdout = proc.stdout
            if stdout:
                stdout.close()
        except (AttributeError, BrokenPipeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Service process stdout close failed (service_id=%s)", sid, exc_info=exc)

    def _request_graceful_stop(self, *, service_id: str, proc: subprocess.Popen[Any]) -> bool:
        try:
            stdin = proc.stdin
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Service process stdin unavailable (service_id=%s)", service_id, exc_info=exc)
            return False
        if stdin is None:
            return False
        try:
            stdin.write("stop\n")
            stdin.flush()
            return True
        except (BrokenPipeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Service process graceful stop request failed (service_id=%s)", service_id, exc_info=exc)
            return False

    def _wait_for_exit(self, *, service_id: str, proc: subprocess.Popen[Any], timeout_s: float) -> bool:
        try:
            proc.wait(timeout=max(0.0, float(timeout_s)))
            return True
        except subprocess.TimeoutExpired:
            return False
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Service process wait failed (service_id=%s)", service_id, exc_info=exc)
            return False

    def stop(self, service_id: str) -> bool:
        sid = str(service_id)
        with self._entries_lock:
            proc = self._procs.get(sid)
        if proc is None:
            return True

        try:
            pid = proc.pid
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Service process pid read failed (service_id=%s)", sid, exc_info=exc)
            pid = None

        if self._request_graceful_stop(service_id=sid, proc=proc) and self._wait_for_exit(
            service_id=sid,
            proc=proc,
            timeout_s=SUPERVISOR_GRACEFUL_STOP_TIMEOUT_S,
        ):
            self._cleanup_entry(sid)
            return True

        try:
            proc.terminate()
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Service process terminate failed (service_id=%s)", sid, exc_info=exc)
        if not self._wait_for_exit(service_id=sid, proc=proc, timeout_s=TERMINATE_WAIT_TIMEOUT_S):
            logger.debug("Service process wait after terminate timed out (service_id=%s)", sid)

        if os.name == "nt" and pid:
            try:
                creationflags = subprocess.CREATE_NO_WINDOW
            except AttributeError as exc:
                logger.debug("subprocess.CREATE_NO_WINDOW unavailable", exc_info=exc)
                creationflags = 0
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=TASKKILL_TIMEOUT_S,
                    check=False,
                    creationflags=creationflags,
                )
            except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
                logger.debug("taskkill failed for service process (service_id=%s pid=%s)", sid, pid, exc_info=exc)
        else:
            try:
                if proc.poll() is None:
                    proc.kill()
            except (AttributeError, RuntimeError, TypeError) as exc:
                logger.debug("Service process kill failed (service_id=%s)", sid, exc_info=exc)

        deadline = time.monotonic() + FINAL_EXIT_WAIT_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                if proc.poll() is not None:
                    break
            except (AttributeError, RuntimeError, TypeError):
                break
            time.sleep(0.05)

        if proc.poll() is None:
            return False
        self._cleanup_entry(sid)
        return True

    def start(self, cfg: ServiceProcessConfig, *, on_output: Any | None = None) -> None:
        service_class = str(cfg.service_class).strip()
        service_id = str(cfg.service_id).strip()
        supervision_mode = str(cfg.supervision_mode or "studio_owned").strip().lower()
        if supervision_mode not in {"studio_owned", "detached"}:
            raise ValueError("Invalid supervision_mode; expected 'studio_owned' or 'detached'.")
        bus_backend = str(cfg.bus_backend or "zenoh").strip().lower()
        if bus_backend not in {"zenoh", "mem"}:
            raise ValueError("Invalid process bus_backend; expected 'zenoh' or 'mem'.")

        entry_path = self._catalog.service_entry_path(service_class)
        if entry_path is None:
            raise ValueError(f"Missing discovery entry path for serviceClass={service_class!r}")
        service_dir = Path(entry_path).resolve()
        try:
            if service_dir.is_file() and service_dir.name.lower() == "service.yml":
                service_dir = service_dir.parent.resolve()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug(
                "Service discovery entry path normalization failed serviceClass=%s path=%s",
                service_class,
                entry_path,
                exc_info=exc,
            )
        entry = load_service_entry(service_dir)

        if self.is_running(service_id):
            return
        self._cleanup_entry(service_id)

        launch = entry.launch
        cmd = [str(launch.command), *[str(a) for a in (launch.args or [])]]
        cmd += ["--service-id", service_id, "--bus-backend", bus_backend]
        if bus_backend == "zenoh":
            zenoh_config_path = str(cfg.zenoh_config_path or "").strip()
            if zenoh_config_path:
                cmd += ["--zenoh-config", zenoh_config_path]
            for endpoint in tuple(str(item).strip() for item in cfg.zenoh_connect if str(item).strip()):
                cmd += ["--zenoh-connect", endpoint]
            for endpoint in tuple(str(item).strip() for item in cfg.zenoh_listen if str(item).strip()):
                cmd += ["--zenoh-listen", endpoint]
            cmd += ["--zenoh-shm-pool-bytes", str(max(0, int(cfg.zenoh_shm_pool_bytes)))]

        child_cmd = list(cmd)
        if supervision_mode == "studio_owned":
            supervisor_path = Path(__file__).with_name("supervised_child.py").resolve()
            if not supervisor_path.exists():
                raise FileNotFoundError(f"Missing service supervisor wrapper: {supervisor_path}")
            cmd = [
                os.environ.get("PYTHON", "") or sys.executable,
                supervisor_path.as_posix(),
                "--parent-pid",
                str(os.getpid()),
                "--",
                *child_cmd,
            ]

        env = os.environ.copy()
        launch_env = launch.env
        if isinstance(launch_env, dict):
            try:
                env.update({str(k): str(v) for k, v in launch_env.items()})
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Service launch env normalization failed service_id=%s", service_id, exc_info=exc)
        env["F8_SERVICE_ID"] = service_id
        env["F8_BUS_BACKEND"] = bus_backend
        if bus_backend == "zenoh":
            zenoh_config_path = str(cfg.zenoh_config_path or "").strip()
            if zenoh_config_path:
                env["F8_ZENOH_CONFIG"] = zenoh_config_path
            else:
                env.pop("F8_ZENOH_CONFIG", None)
            if cfg.zenoh_connect:
                env["F8_ZENOH_CONNECT"] = ",".join(str(item).strip() for item in cfg.zenoh_connect if str(item).strip())
            else:
                env.pop("F8_ZENOH_CONNECT", None)
            if cfg.zenoh_listen:
                env["F8_ZENOH_LISTEN"] = ",".join(str(item).strip() for item in cfg.zenoh_listen if str(item).strip())
            else:
                env.pop("F8_ZENOH_LISTEN", None)
            env["F8_ZENOH_SHM_POOL_BYTES"] = str(max(0, int(cfg.zenoh_shm_pool_bytes)))
        else:
            env.pop("F8_ZENOH_CONFIG", None)
            env.pop("F8_ZENOH_CONNECT", None)
            env.pop("F8_ZENOH_LISTEN", None)
            env.pop("F8_ZENOH_SHM_POOL_BYTES", None)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        workdir_value = launch.workdir
        workdir_raw = (
            "./" if workdir_value is None or isinstance(workdir_value, msgspec.UnsetType) else str(workdir_value)
        )
        workdir = Path(workdir_raw).expanduser()
        if not workdir.is_absolute():
            workdir = (service_dir / workdir).resolve()
        else:
            workdir = workdir.resolve()

        creationflags = 0
        if os.name == "nt":
            try:
                creationflags = subprocess.CREATE_NO_WINDOW
            except AttributeError:
                logger.debug("subprocess.CREATE_NO_WINDOW unavailable for service_id=%s", service_id, exc_info=True)

        proc = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            env=env,
            stdin=subprocess.PIPE if supervision_mode == "studio_owned" else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        with self._entries_lock:
            self._procs[service_id] = proc
        if on_output is not None:
            try:
                pid_txt = str(proc.pid)
            except (AttributeError, RuntimeError, TypeError) as exc:
                logger.debug("Service process pid string conversion failed service_id=%s", service_id, exc_info=exc)
                pid_txt = "?"
            on_output(
                service_id,
                f"[proc] started pid={pid_txt} serviceClass={service_class} supervision={supervision_mode}\n",
            )
        self._start_reader(service_id=service_id, proc=proc, on_output=on_output)
