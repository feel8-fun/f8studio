import os
import sys
import unittest
from dataclasses import dataclass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.generated import (  # noqa: E402
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8RuntimeGraph,
    F8RuntimeNode,
)
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry  # noqa: E402
from f8pysdk.service_host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.pyengine_service import PyEngineService  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402


@dataclass
class _RuntimeStub:
    bus: object


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
    async def _setup_service(self) -> tuple[object, PyEngineService, _RuntimeStub]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_pyengine_specs(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        service = PyEngineService()
        runtime = _RuntimeStub(bus=bus)
        await service.setup(runtime)  # type: ignore[arg-type]
        return bus, service, runtime

    async def _teardown_service(self, service: PyEngineService, runtime: _RuntimeStub) -> None:
        await service.teardown(runtime)  # type: ignore[arg-type]

    async def test_rejects_multiple_exec_entrypoints(self) -> None:
        bus, service, runtime = await self._setup_service()
        try:
            n1 = _node(node_id="tick1", operator_class="f8.tick", exec_in=[], exec_out=["exec"])
            n2 = _node(node_id="tick2", operator_class="f8.tick", exec_in=[], exec_out=["exec"])
            graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[n1, n2], edges=[])
            with self.assertRaises(RuntimeError):
                await bus.set_rungraph(graph)  # type: ignore[attr-defined]
        finally:
            await self._teardown_service(service, runtime)

    async def test_rejects_exec_cycle(self) -> None:
        bus, service, runtime = await self._setup_service()
        try:
            tick = _node(node_id="tick1", operator_class="f8.tick", exec_in=[], exec_out=["exec"])
            seq = _node(node_id="seq1", operator_class="f8.sequence", exec_in=["exec"], exec_out=["exec"])
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
        bus, service, runtime = await self._setup_service()
        try:
            tick = _node(node_id="tick1", operator_class="f8.tick", exec_in=[], exec_out=["exec"])
            seq = _node(node_id="seq1", operator_class="f8.sequence", exec_in=["exec"], exec_out=["exec"])
            a = _node(node_id="a", operator_class="f8.sequence", exec_in=["exec"], exec_out=["exec"])
            b = _node(node_id="b", operator_class="f8.sequence", exec_in=["exec"], exec_out=["exec"])
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


if __name__ == "__main__":
    unittest.main()
