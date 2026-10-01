import asyncio
from dataclasses import dataclass
from typing import Any, cast

from f8pyscript.hook_invoker import PyScriptHookInvoker
from f8pyscript.script_context import PyScriptServiceContext


@dataclass
class _Ctx:
    value: int


def test_invoke_sync_calls_regular_hook() -> None:
    errors: list[str] = []
    invoker = PyScriptHookInvoker(
        node_id="svcA",
        build_context=lambda: cast(PyScriptServiceContext, _Ctx(value=3)),
        set_error=lambda stage, exc: errors.append(f"{stage}:{exc}"),
    )

    def _hook(ctx: _Ctx, value: int) -> int:
        return ctx.value + value

    assert invoker.invoke_sync(_hook, False, "onStart", 4) == 7
    assert errors == []


def test_invoke_sync_reports_regular_hook_failure() -> None:
    errors: list[str] = []
    invoker = PyScriptHookInvoker(
        node_id="svcA",
        build_context=lambda: cast(PyScriptServiceContext, _Ctx(value=3)),
        set_error=lambda stage, exc: errors.append(f"{stage}:{type(exc).__name__}:{exc}"),
    )

    def _hook(ctx: _Ctx) -> None:
        del ctx
        raise RuntimeError("boom")

    assert invoker.invoke_sync(_hook, False, "onStart") is None
    assert errors == ["onStart:RuntimeError:boom"]


def test_invoke_async_awaits_coroutine_hook() -> None:
    async def _run() -> tuple[Any, bool]:
        invoker = PyScriptHookInvoker(
            node_id="svcA",
            build_context=lambda: cast(PyScriptServiceContext, _Ctx(value=5)),
            set_error=lambda stage, exc: None,
        )

        async def _hook(ctx: _Ctx, value: int) -> int:
            return ctx.value + value

        return await invoker.invoke_async(_hook, True, False, "onData", 6)

    assert asyncio.run(_run()) == (11, False)
