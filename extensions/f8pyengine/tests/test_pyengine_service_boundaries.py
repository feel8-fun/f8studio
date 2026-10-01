from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode

from f8pyengine.constants import SERVICE_CLASS
from f8pyengine.pyengine_service import PyEngineService


@dataclass
class _RuntimeStub:
    bus: object


class _BrokenTeardownBus:
    def __init__(self) -> None:
        self.exec_emitter_cleared = False

    def unregister_rungraph_hook(self, hook: object) -> None:
        _ = hook
        raise ValueError("rungraph hook missing")

    def unregister_service_hook(self, hook: object) -> None:
        _ = hook
        raise RuntimeError("service hook missing")

    def set_exec_emitter(self, emitter: object) -> None:
        assert emitter is None
        self.exec_emitter_cleared = True


class _BrokenAutoSampler:
    async def close(self) -> None:
        raise RuntimeError("auto sampler close failed")


class _BrokenExecutor:
    async def set_active(self, active: bool) -> None:
        _ = active
        raise RuntimeError("set active failed")

    async def stop_all_entrypoints(self) -> None:
        raise RuntimeError("stop entrypoints failed")


def test_teardown_logs_each_boundary_failure_and_continues(caplog) -> None:
    bus = _BrokenTeardownBus()
    service = PyEngineService()
    service._auto_sampler = _BrokenAutoSampler()  # type: ignore[assignment]
    service._executor = _BrokenExecutor()  # type: ignore[assignment]
    service._runtime = object()  # type: ignore[assignment]

    async def _run() -> None:
        await service.teardown(_RuntimeStub(bus=bus))  # type: ignore[arg-type]

    with caplog.at_level("ERROR", logger="f8pyengine.pyengine_service"):
        asyncio.run(_run())

    assert bus.exec_emitter_cleared is True
    assert service._runtime is None
    assert service._auto_sampler is None
    assert "unregister_rungraph_hook failed during teardown" in caplog.text
    assert "unregister_service_hook failed during teardown" in caplog.text
    assert "auto sampler close failed during teardown" in caplog.text
    assert "set_active(False) failed during teardown" in caplog.text
    assert "stop_all_entrypoints failed during teardown" in caplog.text


class _BrokenRegistryExecutor:
    def register_node(self, node: object) -> None:
        _ = node
        raise RuntimeError("register failed")

    def unregister_node(self, node_id: str) -> None:
        _ = node_id
        raise ValueError("unregister failed")


class _ExecNode:
    node_id = "exec1"

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        _ = exec_id
        _ = in_port
        return []


class _ExecBus:
    def get_node(self, node_id: str) -> Any:
        assert node_id == "exec1"
        return _ExecNode()


def test_sync_exec_nodes_logs_registry_failures_and_continues(caplog) -> None:
    service = PyEngineService()
    service._executor = _BrokenRegistryExecutor()  # type: ignore[assignment]
    service._exec_node_ids = {"stale"}

    graph = F8RuntimeGraph(
        graphId="g1",
        revision="r1",
        nodes=[
            F8RuntimeNode(
                nodeId="exec1",
                serviceId="svcA",
                serviceClass=SERVICE_CLASS,
                operatorClass="f8.test.exec",
                execInPorts=["exec"],
                execOutPorts=[],
            )
        ],
        edges=[],
    )

    async def _run() -> None:
        await service._sync_exec_nodes(_RuntimeStub(bus=_ExecBus()), graph)  # type: ignore[arg-type]

    with caplog.at_level("ERROR", logger="f8pyengine.pyengine_service"):
        asyncio.run(_run())

    assert service._exec_node_ids == set()
    assert "unregister exec node failed: stale" in caplog.text
    assert "register exec node failed: exec1" in caplog.text
