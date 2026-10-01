from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PyScriptPermissionContext:
    local_exec_granted: bool
    expires_ts_ms: int | None
    grant_ts_ms: int
    session_id: str


class PyScriptLocalExec:
    def __init__(self, *, node_id: str, now_ms: Callable[[], int]) -> None:
        self._node_id = str(node_id)
        self._now_ms = now_ms
        self._local_exec_granted = False
        self._grant_session_id = ""
        self._grant_ts_ms = 0
        self._grant_expires_ts_ms: int | None = None

    @staticmethod
    def coerce_ttl_ms(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return max(1, int(value))
        except (TypeError, ValueError) as exc:
            raise ValueError("ttlMs must be an integer") from exc

    def permission_context(self) -> PyScriptPermissionContext:
        allowed = self.is_allowed()
        return PyScriptPermissionContext(
            local_exec_granted=bool(allowed),
            expires_ts_ms=int(self._grant_expires_ts_ms) if self._grant_expires_ts_ms is not None else None,
            grant_ts_ms=int(self._grant_ts_ms or 0),
            session_id=str(self._grant_session_id or ""),
        )

    def permission_view(self) -> dict[str, Any]:
        permission = self.permission_context()
        return {
            "localExecGranted": bool(permission.local_exec_granted),
            "expiresTsMs": permission.expires_ts_ms,
            "grantTsMs": int(permission.grant_ts_ms),
            "sessionId": str(permission.session_id),
        }

    def is_allowed(self) -> bool:
        if not self._local_exec_granted:
            return False
        expiry = self._grant_expires_ts_ms
        if expiry is not None and self._now_ms() > int(expiry):
            self._local_exec_granted = False
            self._grant_expires_ts_ms = None
            return False
        return True

    def grant(self, *, ttl_ms: int | None, session_id: object | None) -> dict[str, Any]:
        self._local_exec_granted = True
        self._grant_ts_ms = self._now_ms()
        self._grant_session_id = str(session_id or self._grant_ts_ms)
        self._grant_expires_ts_ms = (self._grant_ts_ms + ttl_ms) if ttl_ms is not None else None
        return {"ok": True, "result": self.permission_view()}

    def revoke(self) -> dict[str, Any]:
        self._local_exec_granted = False
        self._grant_expires_ts_ms = None
        return {"ok": True, "result": self.permission_view()}

    async def exec_local(
        self,
        command: str,
        args: list[str] | tuple[str, ...] | None = None,
        *,
        timeout_ms: int | None = None,
        cwd: str | None = None,
        env: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.is_allowed():
            raise PermissionError("local execution is not granted")

        cmd = str(command or "").strip()
        if not cmd:
            raise ValueError("exec_local command is empty")

        argv = [cmd]
        if args is not None:
            for item in list(args):
                argv.append(str(item))

        run_cwd = str(cwd).strip() if cwd is not None else None
        proc_env: dict[str, str] | None = None
        if env is not None:
            proc_env = dict(os.environ)
            for key, value in dict(env).items():
                proc_env[str(key)] = str(value)

        logger.info("[%s:pyscript] exec_local command=%s args=%s", self._node_id, cmd, argv[1:])
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=run_cwd,
            env=proc_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        timeout_s: float | None
        if timeout_ms is None:
            timeout_s = None
        else:
            timeout_s = max(0.001, float(timeout_ms) / 1000.0)

        try:
            if timeout_s is None:
                stdout_raw, stderr_raw = await proc.communicate()
            else:
                stdout_raw, stderr_raw = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"exec_local timeout command={cmd}") from exc

        stdout_text = (stdout_raw or b"").decode("utf-8", errors="replace")
        stderr_text = (stderr_raw or b"").decode("utf-8", errors="replace")
        return {
            "ok": bool(proc.returncode == 0),
            "returncode": int(proc.returncode or 0),
            "stdout": stdout_text,
            "stderr": stderr_text,
            "command": cmd,
            "args": argv[1:],
        }


__all__ = ["PyScriptLocalExec", "PyScriptPermissionContext"]
