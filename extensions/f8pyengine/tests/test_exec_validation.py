import asyncio
import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.specs import (  # noqa: E402
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8RuntimeGraph,
    F8RuntimeNode,
)
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.nodes import OperatorNode  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.pyengine_service import PyEngineService  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402


@dataclass
class _RuntimeStub:
    bus: object


class _ProbeRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        del initial_state
        super().__init__(
            node_id=node_id,
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
            exec_in_ports=list(node.execInPorts or []),
            exec_out_ports=list(node.execOutPorts or []),
        )
        self.calls = 0
        self.inflight = 0
        self.max_inflight = 0

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        _ = exec_id
        _ = in_port
        self.calls += 1
        self.inflight += 1
        if self.inflight > self.max_inflight:
            self.max_inflight = self.inflight
        try:
            await asyncio.sleep(0.01)
        finally:
            self.inflight -= 1
        return []


def _node(*, node_id: str, operator_class: str, exec_in: list[str], exec_out: list[str]) -> F8RuntimeNode:
    return F8RuntimeNode(
        nodeId=node_id,
        serviceId="svcA",
        serviceClass=SERVICE_CLASS,
        operatorClass=operator_class,
        execInPorts=list(exec_in),
        execOutPorts=list(exec_out),
        stateFields=[],
    )


def _exec_edge(*, edge_id: str, from_node: str, from_port: str, to_node: str, to_port: str) -> F8Edge:
    return F8Edge(
        edgeId=edge_id,
        fromServiceId="svcA",
        fromOperatorId=from_node,
        fromPort=from_port,
        toServiceId="svcA",
        toOperatorId=to_node,
        toPort=to_port,
        kind=F8EdgeKindEnum.exec,
        strategy=F8EdgeStrategyEnum.latest,
    )


class ExecValidationTests(unittest.IsolatedAsyncioTestCase):
    async def _setup_service(self) -> tuple[object, PyEngineService, _RuntimeStub, object]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        service = PyEngineService()
        runtime = _RuntimeStub(bus=bus)
        await service.setup(runtime)  # type: ignore[arg-type]
        return bus, service, runtime, reg

    async def _teardown_service(self, service: PyEngineService, runtime: _RuntimeStub) -> None:
        await service.teardown(runtime)  # type: ignore[arg-type]

    async def test_allows_multiple_exec_entrypoints(self) -> None:
        bus, service, runtime, _registry = await self._setup_service()
        try:
            n1 = _node(node_id="tick1", operator_class="f8.tick", exec_in=[], exec_out=["exec"])
            n2 = _node(node_id="tick2", operator_class="f8.tick", exec_in=[], exec_out=["exec"])
            graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[n1, n2], edges=[])
            await bus.set_rungraph(graph)  # type: ignore[attr-defined]
        finally:
            await self._teardown_service(service, runtime)

    async def test_rejects_exec_cycle(self) -> None:
        bus, service, runtime, _registry = await self._setup_service()
        try:
            tick = _node(node_id="tick1", operator_class="f8.tick", exec_in=[], exec_out=["exec"])
            seq = _node(node_id="seq1", operator_class="f8.exec_sequence", exec_in=["exec"], exec_out=["exec"])
            edges = [
                _exec_edge(edge_id="e1", from_node="tick1", from_port="exec", to_node="seq1", to_port="exec"),
                _exec_edge(edge_id="e2", from_node="seq1", from_port="exec", to_node="seq1", to_port="exec"),
            ]
            graph = F8RuntimeGraph(graphId="g2", revision="r1", nodes=[tick, seq], edges=edges)
            with self.assertRaises(RuntimeError):
                await bus.set_rungraph(graph)  # type: ignore[attr-defined]
        finally:
            await self._teardown_service(service, runtime)

    async def test_rejects_multi_connected_exec_out_port(self) -> None:
        bus, service, runtime, _registry = await self._setup_service()
        try:
            tick = _node(node_id="tick1", operator_class="f8.tick", exec_in=[], exec_out=["exec"])
            seq = _node(node_id="seq1", operator_class="f8.exec_sequence", exec_in=["exec"], exec_out=["exec"])
            a = _node(node_id="a", operator_class="f8.exec_sequence", exec_in=["exec"], exec_out=["exec"])
            b = _node(node_id="b", operator_class="f8.exec_sequence", exec_in=["exec"], exec_out=["exec"])
            edges = [
                _exec_edge(edge_id="e1", from_node="tick1", from_port="exec", to_node="seq1", to_port="exec"),
                _exec_edge(edge_id="e2", from_node="seq1", from_port="exec", to_node="a", to_port="exec"),
                _exec_edge(edge_id="e3", from_node="seq1", from_port="exec", to_node="b", to_port="exec"),
            ]
            graph = F8RuntimeGraph(graphId="g3", revision="r1", nodes=[tick, seq, a, b], edges=edges)
            with self.assertRaises(RuntimeError):
                await bus.set_rungraph(graph)  # type: ignore[attr-defined]
        finally:
            await self._teardown_service(service, runtime)

    async def test_multiple_ticks_share_single_serial_exec_worker(self) -> None:
        bus, service, runtime, registry = await self._setup_service()
        try:
            registry.register_operator_factory(
                SERVICE_CLASS,
                "f8.test_probe",
                lambda node_id, node, initial_state: _ProbeRuntimeNode(
                    node_id=node_id, node=node, initial_state=initial_state
                ),
                overwrite=True,
            )

            tick1 = _node(node_id="tick1", operator_class="f8.tick", exec_in=[], exec_out=["exec"])
            tick2 = _node(node_id="tick2", operator_class="f8.tick", exec_in=[], exec_out=["exec"])
            probe = _node(node_id="probe", operator_class="f8.test_probe", exec_in=["fromA", "fromB"], exec_out=[])

            edges = [
                _exec_edge(edge_id="e1", from_node="tick1", from_port="exec", to_node="probe", to_port="fromA"),
                _exec_edge(edge_id="e2", from_node="tick2", from_port="exec", to_node="probe", to_port="fromB"),
            ]
            graph = F8RuntimeGraph(graphId="g4", revision="r1", nodes=[tick1, tick2, probe], edges=edges)
            await bus.set_rungraph(graph)  # type: ignore[attr-defined]

            await asyncio.sleep(0.2)
            node = bus.get_node("probe")
            self.assertIsInstance(node, _ProbeRuntimeNode)
            assert isinstance(node, _ProbeRuntimeNode)
            self.assertGreater(node.calls, 0)
            self.assertEqual(node.max_inflight, 1)
        finally:
            await self._teardown_service(service, runtime)


if __name__ == "__main__":
    unittest.main()
