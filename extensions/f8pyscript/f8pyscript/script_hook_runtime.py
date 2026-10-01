from __future__ import annotations

from typing import Any

from .hook_invoker import PyScriptHookInvoker
from .script_runtime import PyScriptHookSet


class PyScriptHookRuntime:
    def __init__(self, *, hooks: PyScriptHookSet, invoker: PyScriptHookInvoker) -> None:
        self._hooks = hooks
        self._invoker = invoker

    @classmethod
    def empty(cls, *, invoker: PyScriptHookInvoker) -> "PyScriptHookRuntime":
        return cls(hooks=PyScriptHookSet.empty(), invoker=invoker)

    @property
    def has_command_hook(self) -> bool:
        return self._hooks.on_command is not None

    def is_data_hook_async(self) -> bool:
        return bool(self._hooks.on_data_is_async)

    def is_command_hook_async(self) -> bool:
        return bool(self._hooks.on_command_is_async)

    def invoke_start_sync(self) -> Any:
        return self._invoker.invoke_sync(self._hooks.on_start, self._hooks.on_start_is_async, "onStart")

    def invoke_stop_sync(self) -> Any:
        return self._invoker.invoke_sync(self._hooks.on_stop, self._hooks.on_stop_is_async, "onStop")

    async def invoke_stop(self) -> Any:
        result, _ = await self._invoker.invoke_async(
            self._hooks.on_stop,
            self._hooks.on_stop_is_async,
            True,
            "onStop",
        )
        return result

    async def invoke_pause(self, meta: dict[str, Any]) -> Any:
        result, _ = await self._invoker.invoke_async(
            self._hooks.on_pause,
            self._hooks.on_pause_is_async,
            True,
            "onPause",
            meta,
        )
        return result

    async def invoke_resume(self, meta: dict[str, Any]) -> Any:
        result, _ = await self._invoker.invoke_async(
            self._hooks.on_resume,
            self._hooks.on_resume_is_async,
            True,
            "onResume",
            meta,
        )
        return result

    async def invoke_state(self, name: str, value: Any, ts_ms: int | None) -> Any:
        result, self._hooks.on_state_maybe_awaitable = await self._invoker.invoke_async(
            self._hooks.on_state,
            self._hooks.on_state_is_async,
            self._hooks.on_state_maybe_awaitable,
            "onState",
            name,
            value,
            ts_ms,
        )
        return result

    async def invoke_data(self, port: str, value: Any, ts_ms: int | None) -> Any:
        result, self._hooks.on_data_maybe_awaitable = await self._invoker.invoke_async(
            self._hooks.on_data,
            self._hooks.on_data_is_async,
            self._hooks.on_data_maybe_awaitable,
            "onData",
            port,
            value,
            ts_ms,
        )
        return result

    async def invoke_tick(self, tick_payload: dict[str, int]) -> Any:
        result, self._hooks.on_tick_maybe_awaitable = await self._invoker.invoke_async(
            self._hooks.on_tick,
            self._hooks.on_tick_is_async,
            self._hooks.on_tick_maybe_awaitable,
            "onTick",
            tick_payload,
        )
        return result

    async def invoke_command(self, name: str, args: dict[str, Any], meta: dict[str, Any]) -> Any:
        result, _ = await self._invoker.invoke_async(
            self._hooks.on_command,
            self._hooks.on_command_is_async,
            True,
            "onCommand",
            name,
            args,
            meta,
        )
        return result


__all__ = ["PyScriptHookRuntime"]
