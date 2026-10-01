from __future__ import annotations

import builtins
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


_SAFE_MODULES: set[str] = {
    "asyncio",
    "collections",
    "datetime",
    "functools",
    "itertools",
    "json",
    "math",
    "random",
    "re",
    "statistics",
    "time",
    "typing",
}


@dataclass(slots=True)
class PyScriptHookSet:
    on_start: Callable[..., Any] | None
    on_stop: Callable[..., Any] | None
    on_pause: Callable[..., Any] | None
    on_resume: Callable[..., Any] | None
    on_state: Callable[..., Any] | None
    on_data: Callable[..., Any] | None
    on_tick: Callable[..., Any] | None
    on_command: Callable[..., Any] | None
    on_start_is_async: bool
    on_stop_is_async: bool
    on_pause_is_async: bool
    on_resume_is_async: bool
    on_state_is_async: bool
    on_data_is_async: bool
    on_tick_is_async: bool
    on_command_is_async: bool
    on_state_maybe_awaitable: bool
    on_data_maybe_awaitable: bool
    on_tick_maybe_awaitable: bool

    @classmethod
    def empty(cls) -> "PyScriptHookSet":
        return cls(
            on_start=None,
            on_stop=None,
            on_pause=None,
            on_resume=None,
            on_state=None,
            on_data=None,
            on_tick=None,
            on_command=None,
            on_start_is_async=False,
            on_stop_is_async=False,
            on_pause_is_async=False,
            on_resume_is_async=False,
            on_state_is_async=False,
            on_data_is_async=False,
            on_tick_is_async=False,
            on_command_is_async=False,
            on_state_maybe_awaitable=False,
            on_data_maybe_awaitable=False,
            on_tick_maybe_awaitable=False,
        )


class PyScriptRuntimeCompiler:
    def __init__(self, *, is_local_exec_allowed: Callable[[], bool]) -> None:
        self._is_local_exec_allowed = is_local_exec_allowed

    def build_builtins(self) -> dict[str, Any]:
        runtime_builtins: dict[str, Any] = {
            "abs": builtins.abs,
            "all": builtins.all,
            "any": builtins.any,
            "bool": builtins.bool,
            "bytes": builtins.bytes,
            "callable": builtins.callable,
            "dict": builtins.dict,
            "enumerate": builtins.enumerate,
            "float": builtins.float,
            "int": builtins.int,
            "isinstance": builtins.isinstance,
            "len": builtins.len,
            "list": builtins.list,
            "max": builtins.max,
            "min": builtins.min,
            "pow": builtins.pow,
            "print": builtins.print,
            "range": builtins.range,
            "round": builtins.round,
            "set": builtins.set,
            "slice": builtins.slice,
            "sorted": builtins.sorted,
            "str": builtins.str,
            "sum": builtins.sum,
            "tuple": builtins.tuple,
            "zip": builtins.zip,
            "Exception": builtins.Exception,
            "ValueError": builtins.ValueError,
            "TypeError": builtins.TypeError,
            "RuntimeError": builtins.RuntimeError,
            "KeyError": builtins.KeyError,
            "IndexError": builtins.IndexError,
            "PermissionError": builtins.PermissionError,
        }
        runtime_builtins["__import__"] = self._guarded_import
        return runtime_builtins

    def compile(self, code: str) -> PyScriptHookSet:
        env: dict[str, Any] = {"__builtins__": self.build_builtins()}
        exec(code, env, env)

        on_start_raw = env.get("onStart")
        on_stop_raw = env.get("onStop")
        on_pause_raw = env.get("onPause")
        on_resume_raw = env.get("onResume")
        on_state_raw = env.get("onState")
        on_data_raw = env.get("onData")
        on_tick_raw = env.get("onTick")
        on_command_raw = env.get("onCommand")

        on_start = on_start_raw if callable(on_start_raw) else None
        on_stop = on_stop_raw if callable(on_stop_raw) else None
        on_pause = on_pause_raw if callable(on_pause_raw) else None
        on_resume = on_resume_raw if callable(on_resume_raw) else None
        on_state = on_state_raw if callable(on_state_raw) else None
        on_data = on_data_raw if callable(on_data_raw) else None
        on_tick = on_tick_raw if callable(on_tick_raw) else None
        on_command = on_command_raw if callable(on_command_raw) else None

        on_start_is_async = inspect.iscoroutinefunction(on_start) if on_start is not None else False
        on_stop_is_async = inspect.iscoroutinefunction(on_stop) if on_stop is not None else False
        on_pause_is_async = inspect.iscoroutinefunction(on_pause) if on_pause is not None else False
        on_resume_is_async = inspect.iscoroutinefunction(on_resume) if on_resume is not None else False
        on_state_is_async = inspect.iscoroutinefunction(on_state) if on_state is not None else False
        on_data_is_async = inspect.iscoroutinefunction(on_data) if on_data is not None else False
        on_tick_is_async = inspect.iscoroutinefunction(on_tick) if on_tick is not None else False
        on_command_is_async = inspect.iscoroutinefunction(on_command) if on_command is not None else False

        return PyScriptHookSet(
            on_start=on_start,
            on_stop=on_stop,
            on_pause=on_pause,
            on_resume=on_resume,
            on_state=on_state,
            on_data=on_data,
            on_tick=on_tick,
            on_command=on_command,
            on_start_is_async=on_start_is_async,
            on_stop_is_async=on_stop_is_async,
            on_pause_is_async=on_pause_is_async,
            on_resume_is_async=on_resume_is_async,
            on_state_is_async=on_state_is_async,
            on_data_is_async=on_data_is_async,
            on_tick_is_async=on_tick_is_async,
            on_command_is_async=on_command_is_async,
            on_state_maybe_awaitable=bool(on_state is not None and not on_state_is_async),
            on_data_maybe_awaitable=bool(on_data is not None and not on_data_is_async),
            on_tick_maybe_awaitable=bool(on_tick is not None and not on_tick_is_async),
        )

    def _guarded_import(
        self,
        name: str,
        globals_obj: Any = None,
        locals_obj: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        module_name = str(name or "").strip()
        if not module_name:
            raise ImportError("empty module name")
        root_name = module_name.split(".")[0]
        if not self._is_local_exec_allowed() and root_name not in _SAFE_MODULES:
            raise PermissionError(f"import blocked without local exec grant: {module_name}")
        return builtins.__import__(module_name, globals_obj, locals_obj, fromlist, int(level))


__all__ = ["PyScriptHookSet", "PyScriptRuntimeCompiler"]
