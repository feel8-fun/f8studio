import asyncio
from dataclasses import dataclass
from typing import cast

from f8pyscript.hook_invoker import PyScriptHookInvoker
from f8pyscript.script_context import PyScriptServiceContext
from f8pyscript.script_hook_runtime import PyScriptHookRuntime
from f8pyscript.script_runtime import PyScriptHookSet


@dataclass
class _Ctx:
    value: int


def test_hook_runtime_invokes_data_and_tracks_maybe_awaitable() -> None:
    async def _run() -> tuple[object, bool]:
        errors: list[str] = []
        invoker = PyScriptHookInvoker(
            node_id="svcA",
            build_context=lambda: cast(PyScriptServiceContext, _Ctx(value=5)),
            set_error=lambda stage, exc: errors.append(f"{stage}:{exc}"),
        )

        async def _data_hook(ctx: _Ctx, port: str, value: int, ts_ms: int | None = None) -> dict[str, object]:
            assert port == "in"
            assert ts_ms == 123
            return {"outputs": {"out": ctx.value + value}}

        hooks = PyScriptHookSet.empty()
        hooks.on_data = _data_hook
        hooks.on_data_is_async = True
        hooks.on_data_maybe_awaitable = False
        runtime = PyScriptHookRuntime(hooks=hooks, invoker=invoker)

        result = await runtime.invoke_data("in", 7, 123)
        assert errors == []
        return result, runtime.is_data_hook_async()

    result, is_async = asyncio.run(_run())

    assert result == {"outputs": {"out": 12}}
    assert is_async is True


def test_empty_hook_runtime_has_no_command_hook() -> None:
    invoker = PyScriptHookInvoker(
        node_id="svcA",
        build_context=lambda: cast(PyScriptServiceContext, _Ctx(value=5)),
        set_error=lambda stage, exc: None,
    )

    runtime = PyScriptHookRuntime.empty(invoker=invoker)

    assert runtime.has_command_hook is False
    assert runtime.is_data_hook_async() is False
    assert runtime.is_command_hook_async() is False
