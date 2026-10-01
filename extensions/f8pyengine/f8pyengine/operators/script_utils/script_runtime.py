from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

_SCRIPT_RUNTIME_COMPILE_ERRORS = (Exception,)


@dataclass(slots=True)
class HookSet:
    runtime: dict[str, Callable[..., Any]]
    on_msg: Callable[..., Any] | None
    on_exec: Callable[..., Any] | None
    on_state: Callable[..., Any] | None
    on_msg_is_async: bool
    on_exec_is_async: bool
    on_state_is_async: bool
    on_msg_maybe_awaitable: bool
    on_exec_maybe_awaitable: bool
    on_state_maybe_awaitable: bool

    @property
    def on_msg_only_mode(self) -> bool:
        return self.on_msg is not None and self.on_exec is None


class ScriptRuntimeCompiler:
    def __init__(self, *, set_error: Callable[[str, BaseException], None]) -> None:
        self._set_error = set_error

    def compile(self, code: str) -> HookSet:
        env: dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(code, env, env)
        except _SCRIPT_RUNTIME_COMPILE_ERRORS as exc:
            self._set_error("compile", exc)
            return HookSet(
                runtime={},
                on_msg=None,
                on_exec=None,
                on_state=None,
                on_msg_is_async=False,
                on_exec_is_async=False,
                on_state_is_async=False,
                on_msg_maybe_awaitable=False,
                on_exec_maybe_awaitable=False,
                on_state_maybe_awaitable=False,
            )

        runtime: dict[str, Callable[..., Any]] = {}
        for hook in ("onStart", "onState", "onMsg", "onExec", "onStop"):
            fn = env.get(hook)
            if callable(fn):
                runtime[hook] = fn

        on_msg = runtime.get("onMsg")
        on_exec = runtime.get("onExec")
        on_state = runtime.get("onState")
        on_msg_is_async = bool(on_msg and inspect.iscoroutinefunction(on_msg))
        on_exec_is_async = bool(on_exec and inspect.iscoroutinefunction(on_exec))
        on_state_is_async = bool(on_state and inspect.iscoroutinefunction(on_state))
        return HookSet(
            runtime=runtime,
            on_msg=on_msg if callable(on_msg) else None,
            on_exec=on_exec if callable(on_exec) else None,
            on_state=on_state if callable(on_state) else None,
            on_msg_is_async=on_msg_is_async,
            on_exec_is_async=on_exec_is_async,
            on_state_is_async=on_state_is_async,
            on_msg_maybe_awaitable=bool(on_msg and not on_msg_is_async),
            on_exec_maybe_awaitable=bool(on_exec and not on_exec_is_async),
            on_state_maybe_awaitable=bool(on_state and not on_state_is_async),
        )
