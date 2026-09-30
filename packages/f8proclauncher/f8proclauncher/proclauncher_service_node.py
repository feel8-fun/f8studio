from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import platform
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from f8pysdk.capabilities import ClosableNode
from f8pysdk.specs import F8RuntimeNode
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import ServiceNode

logger = logging.getLogger(__name__)

_FIELD_PROGRAM_PATH: Final[str] = "programPath"
_FIELD_SINGLETON: Final[str] = "singleton"
_FIELD_DETACHED: Final[str] = "detached"
_FIELD_ACTIVE: Final[str] = "active"


def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
    return None


def _split_command_line(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return shlex.split(text, posix=not _is_windows())


def _expand_arg(raw: str) -> str:
    return os.path.expanduser(os.path.expandvars(str(raw)))


def _resolve_executable(exe: str) -> str:
    expanded = _expand_arg(exe)
    p = Path(expanded)
    if p.exists():
        try:
            return str(p.resolve())
        except OSError:
            return str(p)
    hit = shutil.which(expanded)
    if hit:
        return str(Path(hit).resolve())
    return expanded


def _argv_from_program_path(program_path: str) -> list[str]:
    argv = [_expand_arg(a) for a in _split_command_line(program_path)]
    if not argv:
        return []
    argv[0] = _resolve_executable(argv[0])
    return argv


def _signature(argv: list[str]) -> str:
    payload = json.dumps(argv, ensure_ascii=False, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class _PidRecord:
    pid: int
    argv: list[str]
    created_ts_ms: int


def _pidfile_path_for(argv: list[str]) -> Path:
    sig = _signature(argv)
    return Path(tempfile.gettempdir()) / f"f8-proclauncher-{sig}.json"


def _read_pid_record(path: Path) -> _PidRecord | None:
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.debug("pidfile parse failed: %s", path, exc_info=exc)
        return None

    if not isinstance(obj, dict):
        return None
    pid = obj.get("pid")
    argv = obj.get("argv")
    created_ts_ms = obj.get("created_ts_ms")
    if not isinstance(pid, int):
        return None
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        return None
    if not isinstance(created_ts_ms, int):
        created_ts_ms = 0
    return _PidRecord(pid=pid, argv=list(argv), created_ts_ms=int(created_ts_ms))


def _write_pid_record(path: Path, record: _PidRecord) -> None:
    try:
        path.write_text(
            json.dumps({"pid": int(record.pid), "argv": list(record.argv), "created_ts_ms": int(record.created_ts_ms)}),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.debug("pidfile write failed: %s", path, exc_info=exc)


def _remove_pidfile(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("pidfile unlink failed: %s", path, exc_info=exc)


def _is_pid_running(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    if _is_windows():
        try:
            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except OSError:
            return False
        out = (proc.stdout or "").strip()
        if not out or out.lower().startswith("info:"):
            return False
        return f"\"{int(pid)}\"" in out or f",{int(pid)}," in out
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _terminate_pid(pid: int, *, timeout_s: float = 2.0) -> bool:
    target_pid = int(pid)
    if target_pid <= 0:
        return True
    if not _is_pid_running(target_pid):
        return True

    if _is_windows():
        try:
            proc = subprocess.run(
                ["taskkill", "/PID", str(target_pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return False
        if int(proc.returncode) == 0:
            return True
        return not _is_pid_running(target_pid)

    try:
        os.kill(target_pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError as exc:
        logger.debug("SIGTERM failed pid=%s error_type=%s", target_pid, type(exc).__name__, exc_info=exc)

    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
        if not _is_pid_running(target_pid):
            return True
        time.sleep(0.05)

    try:
        os.kill(target_pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError as exc:
        logger.debug("SIGKILL failed pid=%s error_type=%s", target_pid, type(exc).__name__, exc_info=exc)
        return not _is_pid_running(target_pid)

    deadline_kill = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline_kill:
        if not _is_pid_running(target_pid):
            return True
        time.sleep(0.05)
    return not _is_pid_running(target_pid)


class ProcLauncherServiceNode(ServiceNode, ClosableNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[],
            data_out_ports=[],
            state_fields=[str(s.name) for s in list(node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._lock = asyncio.Lock()
        # `ServiceBus` starts with active=True and persists it on startup; however, lifecycle
        # callbacks are only emitted on changes. Default to active=True so state writes can
        # trigger launches immediately.
        self._active = True

        self._program_path = str(self._initial_state.get(_FIELD_PROGRAM_PATH) or "").strip()
        initial_singleton = _coerce_bool(self._initial_state.get(_FIELD_SINGLETON, True))
        self._singleton = True if initial_singleton is None else bool(initial_singleton)
        initial_detached = _coerce_bool(self._initial_state.get(_FIELD_DETACHED, True))
        self._detached = True if initial_detached is None else bool(initial_detached)

        self._proc: subprocess.Popen[bytes] | None = None
        self._pidfile: Path | None = None
        self._last_error_sig: str | None = None

    async def close(self) -> None:
        async with self._lock:
            await self._stop_if_managed(reason="close")

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == _FIELD_PROGRAM_PATH:
            return str(value or "").strip()
        if name == _FIELD_ACTIVE:
            b = _coerce_bool(value)
            if b is None:
                raise ValueError("invalid active (expected boolean)")
            return bool(b)
        if name in (_FIELD_SINGLETON, _FIELD_DETACHED):
            b = _coerce_bool(value)
            if b is None:
                raise ValueError(f"invalid {name} (expected boolean)")
            return bool(b)
        return value

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        async with self._lock:
            self._active = bool(active)
            # Launch/stop is driven by the persisted `active` state field via `on_state("active", ...)`.
            # `ServiceBus` emits lifecycle callbacks first, then persists the state; handling in one
            # place avoids double-start when `singleton=false`.

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name == _FIELD_ACTIVE:
            b = _coerce_bool(value)
            if b is None:
                return
            async with self._lock:
                self._active = bool(b)
                if self._active:
                    await self._apply_desired_state(reason="state:active")
                else:
                    await self._stop_if_managed(reason="state:active_false")
            return
        if name not in {_FIELD_PROGRAM_PATH, _FIELD_SINGLETON, _FIELD_DETACHED}:
            return
        async with self._lock:
            if not self._active:
                await self._refresh_cached_config()
                return
            await self._apply_desired_state(reason=f"state:{name}")

    async def _refresh_cached_config(self) -> None:
        program_path = await self._read_state_str(_FIELD_PROGRAM_PATH, default=self._program_path)
        singleton = await self._read_state_bool(_FIELD_SINGLETON, default=self._singleton)
        detached = await self._read_state_bool(_FIELD_DETACHED, default=self._detached)
        self._program_path = program_path
        self._singleton = singleton
        self._detached = detached

    async def _read_state_str(self, field: str, *, default: str) -> str:
        v = await self.get_state_value(field)
        if v is None:
            v = self._initial_state.get(field, None)
        if v is None:
            return str(default or "")
        return str(v or "").strip()

    async def _read_state_bool(self, field: str, *, default: bool) -> bool:
        v = await self.get_state_value(field)
        if v is None:
            v = self._initial_state.get(field, None)
        b = _coerce_bool(v)
        if b is None:
            return bool(default)
        return bool(b)

    async def _apply_desired_state(self, *, reason: str) -> None:
        await self._refresh_cached_config()
        program_path = self._program_path
        detached = bool(self._detached)
        singleton = bool(self._singleton)

        if not program_path:
            await self._stop_if_managed(reason=f"{reason}:empty_programPath")
            return

        argv = _argv_from_program_path(program_path)
        if not argv:
            self._log_once(
                f"invalid argv for programPath={program_path!r}",
                sig=f"argv:{program_path}",
                level="error",
            )
            return

        pidfile = _pidfile_path_for(argv)
        self._pidfile = pidfile

        if singleton:
            rec = _read_pid_record(pidfile)
            if rec is not None and _is_pid_running(rec.pid):
                logger.info("skip launch (singleton): pid=%s argv=%s", rec.pid, rec.argv)
                if detached:
                    self._proc = None
                return
            if rec is not None and not _is_pid_running(rec.pid):
                _remove_pidfile(pidfile)

        if not detached:
            # If we already have a live managed process, keep it.
            if self._proc is not None and self._proc.poll() is None:
                return
            # If a previous managed pidfile exists and is alive (e.g. we lost the handle),
            # treat it as running and skip.
            rec = _read_pid_record(pidfile)
            if rec is not None and _is_pid_running(rec.pid):
                return

        # If switching to detached while a managed process exists, stop managing it (best effort).
        if detached and self._proc is not None and self._proc.poll() is None:
            logger.info("detached=true: releasing managed process pid=%s", self._proc.pid)
            self._proc = None

        await self._spawn(argv, detached=detached)

    async def _spawn(self, argv: list[str], *, detached: bool) -> None:
        creationflags = 0
        if detached and _is_windows():
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
        try:
            proc = subprocess.Popen([str(x) for x in argv], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True, creationflags=creationflags, start_new_session=detached and not _is_windows())
        except FileNotFoundError:
            self._log_once(f"program not found argv={argv!r}", sig=f"spawn:fnf:{argv!r}", level="error")
            return
        except OSError as exc:
            self._log_once(
                f"spawn failed argv={argv!r} err={type(exc).__name__}:{exc}",
                sig=f"spawn:os:{argv!r}:{type(exc).__name__}",
                level="error",
            )
            return

        pid = int(proc.pid)
        pidfile = _pidfile_path_for(argv)
        _write_pid_record(pidfile, _PidRecord(pid=pid, argv=list(argv), created_ts_ms=_now_ms()))

        if detached:
            logger.info("launched (detached) pid=%s argv=%s", pid, argv)
            self._proc = None
            return

        logger.info("launched pid=%s argv=%s", pid, argv)
        self._proc = proc

    async def _stop_if_managed(self, *, reason: str) -> None:
        await self._refresh_cached_config()
        if bool(self._detached):
            return

        proc = self._proc
        self._proc = None

        pid: int | None = None
        if proc is not None:
            pid = int(proc.pid)
        else:
            rec = _read_pid_record(self._pidfile) if self._pidfile is not None else None
            if rec is not None:
                pid = int(rec.pid)

        if pid is None:
            return

        ok = await asyncio.to_thread(_terminate_pid, pid)
        if ok:
            logger.info("stopped pid=%s reason=%s", pid, reason)
        else:
            logger.warning("stop failed pid=%s reason=%s", pid, reason)

        if self._pidfile is not None:
            _remove_pidfile(self._pidfile)

    def _log_once(self, message: str, *, sig: str, level: str) -> None:
        if self._last_error_sig == sig:
            return
        self._last_error_sig = sig
        if level == "error":
            logger.error("%s node_id=%s", message, self.node_id)
        else:
            logger.warning("%s node_id=%s", message, self.node_id)
