import asyncio
from typing import Any

import pytest

from f8pyscript.command_dispatcher import PyScriptCommandDispatcher
from f8pyscript.local_exec import PyScriptLocalExec


def test_dispatcher_routes_local_exec_grant_and_revoke() -> None:
    async def _run() -> tuple[dict[str, Any], dict[str, Any], bool]:
        local_exec = PyScriptLocalExec(node_id="svcA", now_ms=lambda: 1000)

        async def _run_script_command(name: str, args: dict[str, Any], meta: dict[str, Any]) -> Any:
            del name, args, meta
            raise AssertionError("built-in command should not call script command")

        dispatcher = PyScriptCommandDispatcher(
            local_exec=local_exec,
            run_script_command=_run_script_command,
        )

        grant_reply = await dispatcher.dispatch("grant_local_exec", {"ttlMs": "50"}, meta={"reqId": "r1"})
        assert local_exec.is_allowed()

        revoke_reply = await dispatcher.dispatch("revoke_local_exec", {})
        return grant_reply, revoke_reply, local_exec.is_allowed()

    grant_reply, revoke_reply, allowed = asyncio.run(_run())

    assert grant_reply["ok"] is True
    assert grant_reply["result"]["localExecGranted"] is True
    assert grant_reply["result"]["expiresTsMs"] == 1050
    assert grant_reply["result"]["sessionId"] == "r1"
    assert revoke_reply["ok"] is True
    assert revoke_reply["result"]["localExecGranted"] is False
    assert allowed is False


def test_dispatcher_wraps_and_normalizes_script_command_result() -> None:
    async def _run() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        local_exec = PyScriptLocalExec(node_id="svcA", now_ms=lambda: 1000)
        received_args: dict[str, Any] = {}
        received_meta: dict[str, Any] = {}

        async def _run_script_command(name: str, args: dict[str, Any], meta: dict[str, Any]) -> Any:
            assert name == "custom"
            received_args.update(args)
            received_meta.update(meta)
            args["x"] = 99
            meta["reqId"] = "mutated"
            return {1: [True]}

        dispatcher = PyScriptCommandDispatcher(
            local_exec=local_exec,
            run_script_command=_run_script_command,
        )
        original_args: dict[str, Any] = {"x": 1}
        original_meta: dict[str, Any] = {"reqId": "r1"}

        reply = await dispatcher.dispatch(" custom ", original_args, meta=original_meta)
        assert original_args == {"x": 1}
        assert original_meta == {"reqId": "r1"}
        return reply, received_args, received_meta

    reply, received_args, received_meta = asyncio.run(_run())

    assert received_args == {"x": 1}
    assert received_meta == {"reqId": "r1"}
    assert reply == {"ok": True, "result": {"1": [True]}}


def test_dispatcher_rejects_empty_command_name() -> None:
    async def _run() -> None:
        local_exec = PyScriptLocalExec(node_id="svcA", now_ms=lambda: 1000)

        async def _run_script_command(name: str, args: dict[str, Any], meta: dict[str, Any]) -> Any:
            del name, args, meta
            return None

        dispatcher = PyScriptCommandDispatcher(
            local_exec=local_exec,
            run_script_command=_run_script_command,
        )

        with pytest.raises(ValueError, match="empty command name"):
            await dispatcher.dispatch("  ", {})

    asyncio.run(_run())
