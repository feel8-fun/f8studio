from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from .script_context import PyScriptServiceContext


_HOOK_AWAITABLE_SCHEDULE_ERRORS = (RuntimeError, TypeError, ValueError)
_SCRIPT_USER_HOOK_ERRORS = (Exception,)


class PyScriptHookInvoker:
    def __init__(
        self,
        *,
        node_id: str,
        build_context: Callable[[], PyScriptServiceContext],
        set_error: Callable[[str, BaseException], None],
        task_prefix: str = "pyscript",
    ) -> None:
        self._node_id = str(node_id)
        self._build_context = build_context
        self._set_error = set_error
        self._task_prefix = str(task_prefix)

    def invoke_sync(self, hook: Callable[..., Any] | None, hook_is_async: bool, stage: str, *args: Any) -> Any:
        if hook is None:
            return None
        try:
            invoke_ctx = self._build_context()
            if hook_is_async:
                coroutine = hook(invoke_ctx, *args)
                self._schedule_sync_hook_awaitable(stage, coroutine)
                return None

            result = hook(invoke_ctx, *args)
            if inspect.isawaitable(result):
                self._schedule_sync_hook_awaitable(stage, result)
                return None
            return result
        except _SCRIPT_USER_HOOK_ERRORS as exc:
            self._set_error(stage, exc)
            return None

    async def invoke_async(
        self,
        hook: Callable[..., Any] | None,
        hook_is_async: bool,
        hook_maybe_awaitable: bool,
        stage: str,
        *args: Any,
    ) -> tuple[Any, bool]:
        if hook is None:
            return None, hook_maybe_awaitable
        try:
            invoke_ctx = self._build_context()
            if hook_is_async:
                return await hook(invoke_ctx, *args), hook_maybe_awaitable
            result = hook(invoke_ctx, *args)
            if not hook_maybe_awaitable:
                return result, False
            if inspect.isawaitable(result):
                return await result, True
            return result, False
        except _SCRIPT_USER_HOOK_ERRORS as exc:
            self._set_error(stage, exc)
            raise

    def _close_unscheduled_hook_awaitable(self, stage: str, result: Any) -> None:
        if not inspect.iscoroutine(result):
            return
        try:
            result.close()
        except RuntimeError as exc:
            self._set_error(f"{stage}:schedule", exc)

    def _schedule_sync_hook_awaitable(self, stage: str, result: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            self._close_unscheduled_hook_awaitable(stage, result)
            self._set_error(f"{stage}:schedule", exc)
            return
        try:
            task = asyncio.ensure_future(result, loop=loop)
        except _HOOK_AWAITABLE_SCHEDULE_ERRORS as exc:
            self._close_unscheduled_hook_awaitable(stage, result)
            self._set_error(f"{stage}:schedule", exc)
            return
        if isinstance(task, asyncio.Task):
            task.set_name(f"{self._task_prefix}:{stage}:{self._node_id}")


__all__ = ["PyScriptHookInvoker"]
