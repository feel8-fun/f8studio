from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .local_exec import PyScriptLocalExec
from .script_runtime_values import normalize_script_output_value


@dataclass(frozen=True, slots=True)
class PyScriptCommandCall:
    name: str
    args: dict[str, Any]
    meta: dict[str, Any]


class PyScriptCommandDispatcher:
    def __init__(
        self,
        *,
        local_exec: PyScriptLocalExec,
        run_script_command: Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[Any]],
    ) -> None:
        self._local_exec = local_exec
        self._run_script_command = run_script_command

    async def dispatch(self, name: str, args: dict[str, Any] | None = None, *, meta: dict[str, Any] | None = None) -> Any:
        command = self._build_call(name, args, meta)

        if command.name == "grant_local_exec":
            return self._grant_local_exec(command)

        if command.name == "revoke_local_exec":
            return self._local_exec.revoke()

        result = await self._run_script_command(command.name, command.args, command.meta)
        return {"ok": True, "result": normalize_script_output_value(result)}

    def _build_call(self, name: str, args: dict[str, Any] | None, meta: dict[str, Any] | None) -> PyScriptCommandCall:
        command_name = str(name or "").strip()
        if not command_name:
            raise ValueError("empty command name")
        return PyScriptCommandCall(name=command_name, args=dict(args or {}), meta=dict(meta or {}))

    def _grant_local_exec(self, command: PyScriptCommandCall) -> dict[str, Any]:
        ttl_ms = self._local_exec.coerce_ttl_ms(command.args.get("ttlMs"))
        session_id = command.meta.get("reqId") or command.meta.get("sessionId")
        return self._local_exec.grant(ttl_ms=ttl_ms, session_id=session_id)


__all__ = ["PyScriptCommandCall", "PyScriptCommandDispatcher"]
