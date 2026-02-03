import json
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.generated import (  # noqa: E402
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
)
from f8pysdk.nats_naming import kv_key_node_state  # noqa: E402
from f8pysdk.schema_helpers import string_schema  # noqa: E402
from f8pysdk.runtime_node import RuntimeNode  # noqa: E402
from f8pysdk.service_bus import ServiceBus, ServiceBusConfig  # noqa: E402
from f8pysdk.state_write import StateWriteContext, StateWriteError, StateWriteOrigin  # noqa: E402
from f8pysdk.testing import InMemoryCluster, InMemoryTransport, ServiceBusHarness  # noqa: E402


class _DummyNode:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.allow_unknown_state_fields = False

    def attach(self, bus: object) -> None:
        self._bus = bus

    async def validate_state(self, field: str, value: object, *, ts_ms: int, meta: dict[str, object]) -> object:
        return value

    async def on_state(self, field: str, value: object, *, ts_ms: int | None = None) -> None:
        return


class _RejectingNode(_DummyNode):
    def validate_state(self, field: str, value: object, *, ts_ms: int, meta: dict[str, object]) -> object:
        raise StateWriteError("CONFLICT", f"reject {field}")


class StateWriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_external_cannot_write_ro(self) -> None:
        bus = ServiceBus(ServiceBusConfig(service_id="svc"))
        bus._graph = object()
        bus.register_node(_DummyNode("svc"))
        bus._state_access_by_node_field[("svc", "status")] = F8StateAccess.ro
        ctx = StateWriteContext(origin=StateWriteOrigin.external, source="endpoint")
        with self.assertRaises(StateWriteError) as cm:
            await bus._validate_state_update(
                node_id="svc",
                field="status",
                value=1,
                ts_ms=1,
                meta={"source": "endpoint"},
                ctx=ctx,
            )
        self.assertEqual(cm.exception.code, "FORBIDDEN")

    async def test_runtime_can_write_ro(self) -> None:
        bus = ServiceBus(ServiceBusConfig(service_id="svc"))
        bus._graph = object()
        bus.register_node(_DummyNode("svc"))
        bus._state_access_by_node_field[("svc", "status")] = F8StateAccess.ro
        ctx = StateWriteContext(origin=StateWriteOrigin.runtime, source="runtime")
        out = await bus._validate_state_update(
            node_id="svc",
            field="status",
            value=1,
            ts_ms=1,
            meta={"source": "runtime"},
            ctx=ctx,
        )
        self.assertEqual(out, 1)

    async def test_rejecting_node_propagates_error(self) -> None:
        bus = ServiceBus(ServiceBusConfig(service_id="svc"))
        bus.register_node(_RejectingNode("svc"))
        ctx = StateWriteContext(origin=StateWriteOrigin.runtime, source="runtime")
        with self.assertRaises(StateWriteError) as cm:
            await bus._validate_state_update(
                node_id="svc",
                field="status",
                value=1,
                ts_ms=1,
                meta={"source": "runtime"},
                ctx=ctx,
            )
        self.assertEqual(cm.exception.code, "CONFLICT")

    async def test_publish_state_sets_origin(self) -> None:
        cluster = InMemoryCluster()
        transport = InMemoryTransport(cluster=cluster, kv_bucket="kv.svc")
        bus = ServiceBus(ServiceBusConfig(service_id="svc"), transport=transport)
        bus._state_access_by_node_field[("svc", "status")] = F8StateAccess.ro
        await bus.publish_state_runtime("svc", "status", 7, ts_ms=42)
        key = kv_key_node_state(node_id="svc", field="status")
        raw = await transport.kv_get(key)
        payload = json.loads(raw.decode("utf-8")) if raw else {}
        self.assertEqual(payload.get("source"), "runtime")
        self.assertEqual(payload.get("origin"), "runtime")

    async def test_cross_service_state_edge(self) -> None:
        harness = ServiceBusHarness()
        bus_a = harness.create_bus("svcA")
        bus_b = harness.create_bus("svcB")

        node_a = F8RuntimeNode(
            nodeId="opA",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpA",
            stateFields=[
                F8StateSpec(name="out", valueSchema=string_schema(), access=F8StateAccess.rw),
            ],
        )
        node_b = F8RuntimeNode(
            nodeId="opB",
            serviceId="svcB",
            serviceClass="svcB",
            operatorClass="OpB",
            stateFields=[
                F8StateSpec(name="input", valueSchema=string_schema(), access=F8StateAccess.rw),
            ],
        )
        edge = F8Edge(
            edgeId="e1",
            fromServiceId="svcA",
            fromOperatorId="opA",
            fromPort="out",
            toServiceId="svcB",
            toOperatorId="opB",
            toPort="input",
            kind=F8EdgeKindEnum.state,
            strategy=F8EdgeStrategyEnum.latest,
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[node_a, node_b], edges=[edge])

        await bus_a.set_rungraph(graph)
        await bus_b.set_rungraph(graph)

        await bus_a.publish_state_runtime("opA", "out", "v1", ts_ms=1)
        out = await bus_b.get_state("opB", "input")
        self.assertEqual(out, "v1")

    async def test_intra_state_edge_propagation(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        bus.register_node(RuntimeNode(node_id="opA"))
        bus.register_node(RuntimeNode(node_id="opB"))

        node_a = F8RuntimeNode(
            nodeId="opA",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpA",
            stateFields=[
                F8StateSpec(name="out", valueSchema=string_schema(), access=F8StateAccess.rw),
            ],
        )
        node_b = F8RuntimeNode(
            nodeId="opB",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpB",
            stateFields=[
                F8StateSpec(name="input", valueSchema=string_schema(), access=F8StateAccess.rw),
            ],
        )
        edge = F8Edge(
            edgeId="e1",
            fromServiceId="svcA",
            fromOperatorId="opA",
            fromPort="out",
            toServiceId="svcA",
            toOperatorId="opB",
            toPort="input",
            kind=F8EdgeKindEnum.state,
            strategy=F8EdgeStrategyEnum.latest,
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[node_a, node_b], edges=[edge])
        await bus.set_rungraph(graph)

        await bus.publish_state_runtime("opA", "out", "v1", ts_ms=1)
        out = await bus.get_state("opB", "input")
        self.assertEqual(out, "v1")

        await bus.publish_state_runtime("opA", "out", "v2", ts_ms=2)
        out = await bus.get_state("opB", "input")
        self.assertEqual(out, "v2")

    async def test_intra_state_edge_blocked_by_ro(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        bus.register_node(RuntimeNode(node_id="opA"))
        bus.register_node(RuntimeNode(node_id="opB"))

        node_a = F8RuntimeNode(
            nodeId="opA",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpA",
            stateFields=[
                F8StateSpec(name="out", valueSchema=string_schema(), access=F8StateAccess.rw),
            ],
        )
        node_b = F8RuntimeNode(
            nodeId="opB",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpB",
            stateFields=[
                F8StateSpec(name="input", valueSchema=string_schema(), access=F8StateAccess.ro),
            ],
        )
        edge = F8Edge(
            edgeId="e1",
            fromServiceId="svcA",
            fromOperatorId="opA",
            fromPort="out",
            toServiceId="svcA",
            toOperatorId="opB",
            toPort="input",
            kind=F8EdgeKindEnum.state,
            strategy=F8EdgeStrategyEnum.latest,
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[node_a, node_b], edges=[edge])
        await bus.set_rungraph(graph)

        await bus.publish_state_runtime("opA", "out", "v1", ts_ms=1)
        out = await bus.get_state("opB", "input")
        self.assertIsNone(out)

    async def test_intra_state_edge_initial_sync(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        bus.register_node(RuntimeNode(node_id="opA"))
        bus.register_node(RuntimeNode(node_id="opB"))

        await bus.publish_state_runtime("opA", "out", "pre", ts_ms=1)

        node_a = F8RuntimeNode(
            nodeId="opA",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpA",
            stateFields=[
                F8StateSpec(name="out", valueSchema=string_schema(), access=F8StateAccess.rw),
            ],
        )
        node_b = F8RuntimeNode(
            nodeId="opB",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpB",
            stateFields=[
                F8StateSpec(name="input", valueSchema=string_schema(), access=F8StateAccess.rw),
            ],
        )
        edge = F8Edge(
            edgeId="e1",
            fromServiceId="svcA",
            fromOperatorId="opA",
            fromPort="out",
            toServiceId="svcA",
            toOperatorId="opB",
            toPort="input",
            kind=F8EdgeKindEnum.state,
            strategy=F8EdgeStrategyEnum.latest,
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[node_a, node_b], edges=[edge])
        await bus.set_rungraph(graph)

        out = await bus.get_state("opB", "input")
        self.assertEqual(out, "pre")

    async def test_rungraph_state_values_apply_and_update(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")

        node = F8RuntimeNode(
            nodeId="opA",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpA",
            stateFields=[
                F8StateSpec(name="cfg", valueSchema=string_schema(), access=F8StateAccess.rw),
                F8StateSpec(name="mode", valueSchema=string_schema(), access=F8StateAccess.wo),
            ],
            stateValues={"cfg": "v1", "mode": "m1"},
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[node], edges=[])
        await bus.set_rungraph(graph)

        v_cfg = await bus.get_state("opA", "cfg")
        v_mode = await bus.get_state("opA", "mode")
        self.assertEqual(v_cfg, "v1")
        self.assertEqual(v_mode, "m1")

        node2 = F8RuntimeNode(
            nodeId="opA",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpA",
            stateFields=node.stateFields,
            stateValues={"cfg": "v2", "mode": "m2"},
        )
        graph2 = F8RuntimeGraph(graphId="g1", revision="r2", nodes=[node2], edges=[])
        await bus.set_rungraph(graph2)

        v_cfg = await bus.get_state("opA", "cfg")
        v_mode = await bus.get_state("opA", "mode")
        self.assertEqual(v_cfg, "v2")
        self.assertEqual(v_mode, "m2")

    async def test_system_identity_seeds_and_protect(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")

        service_node = F8RuntimeNode(
            nodeId="svcA",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass=None,
            stateFields=[
                F8StateSpec(name="svcId", valueSchema=string_schema(), access=F8StateAccess.ro),
            ],
        )
        op_node = F8RuntimeNode(
            nodeId="opA",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpA",
            stateFields=[
                F8StateSpec(name="svcId", valueSchema=string_schema(), access=F8StateAccess.ro),
                F8StateSpec(name="operatorId", valueSchema=string_schema(), access=F8StateAccess.ro),
            ],
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[service_node, op_node], edges=[])
        await bus.set_rungraph(graph)

        svc_id = await bus.get_state("svcA", "svcId")
        op_svc_id = await bus.get_state("opA", "svcId")
        op_id = await bus.get_state("opA", "operatorId")
        self.assertEqual(svc_id, "svcA")
        self.assertEqual(op_svc_id, "svcA")
        self.assertEqual(op_id, "opA")

        with self.assertRaises(StateWriteError):
            await bus._publish_state("opA", "operatorId", "x", origin=StateWriteOrigin.external, source="endpoint", ts_ms=2)
        with self.assertRaises(StateWriteError):
            await bus._publish_state("svcA", "svcId", "x", origin=StateWriteOrigin.external, source="endpoint", ts_ms=2)


if __name__ == "__main__":
    unittest.main()
