import asyncio
import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.bus import ServiceBus, ServiceBusConfig  # noqa: E402
from f8pysdk.specs import (  # noqa: E402
    F8AutoSampleRequest,
    F8DataPortSpec,
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8RuntimeService,
    any_schema,
)
from f8pysdk.testing import InMemoryCluster, InMemoryTransport  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.pyengine_service import PyEngineService  # noqa: E402


@dataclass
class _RuntimeStub:
    bus: object


class _FixedComputableNode:
    def __init__(self, *, node_id: str, value: Any) -> None:
        self.node_id = node_id
        self._value = value
        self.compute_calls: list[tuple[str, str | int | None]] = []

    def attach(self, bus: object) -> None:
        self._bus = bus

    async def validate_state(self, field: str, value: object, *, ts_ms: int, meta: dict[str, object]) -> object:
        _ = field
        _ = ts_ms
        _ = meta
        return value

    async def on_state(self, field: str, value: object, *, ts_ms: int | None = None) -> None:
        _ = field
        _ = value
        _ = ts_ms
        return

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        self.compute_calls.append((str(port), ctx_id))
        return self._value


def _graph(*, auto_sample_requests: list[F8AutoSampleRequest]) -> F8RuntimeGraph:
    return F8RuntimeGraph(
        graphId="g_auto_sample",
        revision="r1",
        services=[
            F8RuntimeService(
                serviceId="svcA",
                serviceClass=SERVICE_CLASS,
                autoSampleRequests=list(auto_sample_requests),
            ),
            F8RuntimeService(
                serviceId="studio",
                serviceClass="f8.pystudio",
            ),
        ],
        nodes=[
            F8RuntimeNode(
                nodeId="src",
                serviceId="svcA",
                serviceClass=SERVICE_CLASS,
                operatorClass="f8.test.source",
                dataInPorts=[],
                dataOutPorts=[F8DataPortSpec(name="out", description="", valueSchema=any_schema(), definitionProtected=False)],
                execInPorts=[],
                execOutPorts=[],
            ),
            F8RuntimeNode(
                nodeId="viz_text_1",
                serviceId="studio",
                serviceClass="f8.pystudio",
                operatorClass="f8.viz.text",
                dataInPorts=[F8DataPortSpec(name="inputData", description="", valueSchema=any_schema(), definitionProtected=False)],
                dataOutPorts=[],
                execInPorts=[],
                execOutPorts=[],
            ),
        ],
        edges=[
            F8Edge(
                edgeId="edge_cross_viz",
                fromServiceId="svcA",
                fromOperatorId="src",
                fromPort="out",
                toServiceId="studio",
                toOperatorId="viz_text_1",
                toPort="inputData",
                kind=F8EdgeKindEnum.data,
                strategy=F8EdgeStrategyEnum.latest,
            )
        ],
    )


class AutoSamplerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        cluster = InMemoryCluster()
        self.engine_bus = ServiceBus(
            ServiceBusConfig(service_id="svcA", data_delivery="buffered"),
            transport=InMemoryTransport(cluster=cluster),
        )
        self.studio_bus = ServiceBus(
            ServiceBusConfig(service_id="studio", data_delivery="callback"),
            transport=InMemoryTransport(cluster=cluster),
        )
        self.service = PyEngineService()
        self.runtime = _RuntimeStub(bus=self.engine_bus)
        await self.service.setup(self.runtime)  # type: ignore[arg-type]

    async def asyncTearDown(self) -> None:
        await self.service.teardown(self.runtime)  # type: ignore[arg-type]

    async def test_pyengine_service_auto_sample_request_republishes_to_cross_service_consumers(self) -> None:
        sample = {"value": 42, "label": "sampled"}
        source = _FixedComputableNode(node_id="src", value=sample)
        self.engine_bus.register_node(source)

        graph = _graph(
            auto_sample_requests=[
                F8AutoSampleRequest(
                    sourceNodeId="src",
                    sourcePort="out",
                    intervalMs=25,
                    deliverLocal=False,
                    publishCrossService=True,
                )
            ]
        )

        await self.studio_bus.set_rungraph(graph)
        await self.engine_bus.set_rungraph(graph)
        await asyncio.sleep(0.12)

        pulled = await self.studio_bus.pull_data("viz_text_1", "inputData")
        self.assertEqual(pulled, sample)
        self.assertGreaterEqual(len(source.compute_calls), 1)

    async def test_pyengine_service_redeploy_without_auto_sample_request_stops_periodic_sampling(self) -> None:
        source = _FixedComputableNode(node_id="src", value={"value": 1})
        self.engine_bus.register_node(source)

        graph_v1 = _graph(
            auto_sample_requests=[
                F8AutoSampleRequest(
                    sourceNodeId="src",
                    sourcePort="out",
                    intervalMs=25,
                    deliverLocal=False,
                    publishCrossService=True,
                )
            ]
        )
        graph_v2 = _graph(auto_sample_requests=[])

        await self.studio_bus.set_rungraph(graph_v1)
        await self.engine_bus.set_rungraph(graph_v1)
        await asyncio.sleep(0.10)
        self.assertGreaterEqual(len(source.compute_calls), 1)

        await self.engine_bus.set_rungraph(graph_v2)
        count_after_redeploy = len(source.compute_calls)
        await asyncio.sleep(0.10)
        self.assertEqual(len(source.compute_calls), count_after_redeploy)

    async def test_pyengine_service_deactivate_and_activate_pause_and_resume_auto_sampling(self) -> None:
        source = _FixedComputableNode(node_id="src", value={"value": 1})
        self.engine_bus.register_node(source)

        graph = _graph(
            auto_sample_requests=[
                F8AutoSampleRequest(
                    sourceNodeId="src",
                    sourcePort="out",
                    intervalMs=25,
                    deliverLocal=False,
                    publishCrossService=True,
                )
            ]
        )

        await self.studio_bus.set_rungraph(graph)
        await self.engine_bus.set_rungraph(graph)
        await asyncio.sleep(0.10)
        self.assertGreaterEqual(len(source.compute_calls), 1)

        await self.service.on_deactivate(self.engine_bus, {"case": "pause"})
        count_after_deactivate = len(source.compute_calls)
        await asyncio.sleep(0.10)
        self.assertEqual(len(source.compute_calls), count_after_deactivate)

        await self.service.on_activate(self.engine_bus, {"case": "resume"})
        await asyncio.sleep(0.10)
        self.assertGreater(len(source.compute_calls), count_after_deactivate)


if __name__ == "__main__":
    unittest.main()
