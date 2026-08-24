import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.specs import (  # noqa: E402
    Code,
    F8Command,
    F8CommandInvokeRequest,
    F8CommandInvokeReply,
    F8CommandParam,
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8ServiceSpec,
    F8SetStateReply,
    F8SetStateRequest,
    F8StateAccess,
    F8StateSpec,
)
from f8pysdk.command import command_input_state_field, command_output_state_field, hidden_command_state_specs  # noqa: E402
from f8pysdk.command import CommandExecutionErrorKind, CommandOutputPolicy  # noqa: E402
from f8pysdk.zenoh_naming import zenoh_state_key  # noqa: E402
from f8pysdk.nodes import RuntimeNode  # noqa: E402
from f8pysdk.specs import string_schema  # noqa: E402
from f8pysdk.bus import ServiceBus, ServiceBusConfig  # noqa: E402
from f8pysdk.service_bus.internal.micro import ServiceBusControlHandlers  # noqa: E402
from f8pysdk.service_bus.state.options import StatePublishOptions  # noqa: E402
from f8pysdk.service_bus.state.pipeline import publish_state, validate_state_update  # noqa: E402
from f8pysdk.state import StateWriteContext, StateWriteError, StateWriteOrigin  # noqa: E402
from f8pysdk.codec import decode_as, decode_obj, dump_json, encode_obj, validate_as  # noqa: E402
from f8pysdk.testing import InMemoryCluster, InMemoryTransport, ServiceBusHarness  # noqa: E402


class _DummyNode:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id

    def attach(self, bus: object) -> None:
        self._bus = bus

    async def validate_state(self, field: str, value: object, *, ts_ms: int, meta: dict[str, object]) -> object:
        return value

    async def on_state(self, field: str, value: object, *, ts_ms: int | None = None) -> None:
        return


class _RejectingNode(_DummyNode):
    def validate_state(self, field: str, value: object, *, ts_ms: int, meta: dict[str, object]) -> object:
        raise StateWriteError("CONFLICT", f"reject {field}")


class _OnStateFailNode(_DummyNode):
    async def on_state(self, field: str, value: object, *, ts_ms: int | None = None) -> None:
        raise RuntimeError("on_state failed")


class _RecordingNode(_DummyNode):
    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.state_calls: list[tuple[str, object]] = []

    async def on_state(self, field: str, value: object, *, ts_ms: int | None = None) -> None:
        self.state_calls.append((field, value))


class _CommandServiceNode(_DummyNode):
    def __init__(self, node_id: str, *, event: unittest.IsolatedAsyncioTestCase | None = None) -> None:
        del event
        super().__init__(node_id)
        self.spec = F8ServiceSpec(
            serviceClass="svc.test.command",
            label="Command Service",
            commands=[
                F8Command(
                    name="run",
                    params=[
                        F8CommandParam(name="a", valueSchema=string_schema()),
                        F8CommandParam(name="b", valueSchema=string_schema()),
                    ],
                ),
                F8Command(name="nop", params=[]),
            ],
        )
        self.on_state_calls: list[tuple[str, object]] = []
        self.command_calls: list[tuple[str, dict[str, object]]] = []
        self.command_meta_calls: list[dict[str, object]] = []
        self._block_event: asyncio.Event | None = None

    async def on_state(self, field: str, value: object, *, ts_ms: int | None = None) -> None:
        self.on_state_calls.append((field, value))

    async def on_command(
        self,
        name: str,
        args: dict[str, object] | None = None,
        *,
        meta: dict[str, object] | None = None,
    ) -> object:
        self.command_meta_calls.append(dict(meta or {}))
        call_args = dict(args or {})
        self.command_calls.append((str(name), call_args))
        if self._block_event is not None and str(name) == "run":
            await self._block_event.wait()
        if str(name) == "nop":
            return {"called": "nop"}
        return {"echo": call_args}


class _NonCommandServiceNode(_DummyNode):
    def __init__(self, node_id: str) -> None:
        super().__init__(node_id)
        self.spec = F8ServiceSpec(
            serviceClass="svc.test.command",
            label="Command Service",
            commands=[
                F8Command(
                    name="run",
                    params=[
                        F8CommandParam(name="a", valueSchema=string_schema()),
                        F8CommandParam(name="b", valueSchema=string_schema()),
                    ],
                ),
                F8Command(name="nop", params=[]),
            ],
        )


class _FailingCommandServiceNode(_CommandServiceNode):
    async def on_command(
        self,
        name: str,
        args: dict[str, object] | None = None,
        *,
        meta: dict[str, object] | None = None,
    ) -> object:
        self.command_meta_calls.append(dict(meta or {}))
        call_args = dict(args or {})
        self.command_calls.append((str(name), call_args))
        raise RuntimeError("command failed")


class _FakeReq:
    def __init__(self, payload: object) -> None:
        self.data = encode_obj(payload)
        self.response: bytes | None = None

    async def respond(self, payload: bytes) -> None:
        self.response = payload


class StateWriteTests(unittest.IsolatedAsyncioTestCase):
    async def test_set_state_endpoint_uses_wire_field_name_across_generated_aliases(self) -> None:
        bus = AsyncMock()
        endpoint = ServiceBusControlHandlers(bus)
        req = _FakeReq(
            validate_as(
                F8SetStateRequest,
                {
                    "reqId": "r1",
                    "args": {"nodeId": "sink", "field": "input", "value": "endpoint"},
                    "meta": {"source": "test"},
                },
            )
        )

        await endpoint._set_state(req)

        bus.publish_state_external.assert_awaited_once()
        call = bus.publish_state_external.await_args
        self.assertEqual(call.args[:3], ("sink", "input", "endpoint"))
        self.assertIsNotNone(req.response)
        reply = decode_as(req.response or b"", F8SetStateReply)
        self.assertTrue(reply.ok)
        self.assertEqual(dump_json(reply.result), {"nodeId": "sink", "field": "input"})

    async def test_external_cannot_write_ro(self) -> None:
        bus = ServiceBus(ServiceBusConfig(service_id="svc"))
        bus._graph = object()
        bus.register_node(_DummyNode("svc"))
        bus.state_store.access_by_node_field[("svc", "status")] = F8StateAccess.ro
        ctx = StateWriteContext(origin=StateWriteOrigin.external, source="endpoint")
        with self.assertRaises(StateWriteError) as cm:
            await validate_state_update(
                bus,
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
        bus.state_store.access_by_node_field[("svc", "status")] = F8StateAccess.ro
        ctx = StateWriteContext(origin=StateWriteOrigin.runtime, source="runtime")
        out = await validate_state_update(
            bus,
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
            await validate_state_update(
                bus,
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
        transport = InMemoryTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="svc"), transport=transport)
        bus.state_store.access_by_node_field[("svc", "status")] = F8StateAccess.ro
        await bus.publish_state_runtime("svc", "status", 7, ts_ms=42)
        key = zenoh_state_key("svc", node_id="svc", field="status")
        raw = await transport.retained_get(key)
        payload = decode_obj(raw) if raw else {}
        self.assertEqual(payload.get("source"), "runtime")
        self.assertEqual(payload.get("origin"), "runtime")

    async def test_publish_state_runtime_force_publish_rewrites_same_value(self) -> None:
        cluster = InMemoryCluster()
        transport = InMemoryTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="svc"), transport=transport)
        bus.state_store.access_by_node_field[("svc", "status")] = F8StateAccess.ro

        await bus.publish_state_runtime("svc", "status", 7, ts_ms=42)
        await bus.publish_state_runtime("svc", "status", 7, ts_ms=43)
        key = zenoh_state_key("svc", node_id="svc", field="status")
        raw = await transport.retained_get(key)
        payload = decode_obj(raw) if raw else {}
        self.assertEqual(payload.get("tsMs"), 42)

        await bus.publish_state_runtime("svc", "status", 7, ts_ms=44, force_publish=True)
        raw2 = await transport.retained_get(key)
        payload2 = decode_obj(raw2) if raw2 else {}
        self.assertEqual(payload2.get("tsMs"), 44)

    async def test_publish_state_options_disable_intra_state_fanout(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        bus.register_node(RuntimeNode(node_id="opA"))
        sink = _RecordingNode("opB")
        bus.register_node(sink)

        graph = F8RuntimeGraph(
            graphId="g_options",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="opA",
                    serviceId="svcA",
                    serviceClass="svcA",
                    operatorClass="OpA",
                    stateFields=[F8StateSpec(name="out", valueSchema=string_schema(), access=F8StateAccess.rw)],
                ),
                F8RuntimeNode(
                    nodeId="opB",
                    serviceId="svcA",
                    serviceClass="svcA",
                    operatorClass="OpB",
                    stateFields=[F8StateSpec(name="input", valueSchema=string_schema(), access=F8StateAccess.rw)],
                ),
            ],
            edges=[
                F8Edge(
                    edgeId="e_options",
                    fromServiceId="svcA",
                    fromOperatorId="opA",
                    fromPort="out",
                    toServiceId="svcA",
                    toOperatorId="opB",
                    toPort="input",
                    kind=F8EdgeKindEnum.state,
                    strategy=F8EdgeStrategyEnum.latest,
                )
            ],
        )
        await bus.set_rungraph(graph)

        await publish_state(
            bus,
            "opA",
            "out",
            "v1",
            origin=StateWriteOrigin.runtime,
            source="test",
            options=StatePublishOptions(fanout_intra_state_edges=False),
        )

        self.assertEqual((await bus.get_state("opA", "out")).value, "v1")
        self.assertFalse((await bus.get_state("opB", "input")).found)
        self.assertEqual(sink.state_calls, [])

    async def test_no_state_fanout_meta_is_preserved_as_plain_meta(self) -> None:
        cluster = InMemoryCluster()
        transport = InMemoryTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="svc"), transport=transport)
        bus.register_node(_RecordingNode("svc"))
        bus.state_store.access_by_node_field[("svc", "status")] = F8StateAccess.rw

        await publish_state(
            bus,
            "svc",
            "status",
            7,
            origin=StateWriteOrigin.runtime,
            source="test",
            meta={"tag": "x", "_noStateFanout": True},
        )

        key = zenoh_state_key("svc", node_id="svc", field="status")
        raw = await transport.retained_get(key)
        payload = decode_obj(raw) if raw else {}
        self.assertEqual(payload.get("tag"), "x")
        self.assertTrue(bool(payload.get("_noStateFanout")))

    async def test_publish_state_persists_even_if_local_callback_fails(self) -> None:
        cluster = InMemoryCluster()
        transport = InMemoryTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="svc"), transport=transport)
        bus.state_store.access_by_node_field[("svc", "status")] = F8StateAccess.rw
        bus.register_node(_OnStateFailNode("svc"))

        await bus.publish_state_runtime("svc", "status", 7, ts_ms=42)

        key = zenoh_state_key("svc", node_id="svc", field="status")
        raw = await transport.retained_get(key)
        payload = decode_obj(raw) if raw else {}
        self.assertEqual(payload.get("value"), 7)

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
        out = (await bus_b.get_state("opB", "input")).value
        self.assertEqual(out, "v1")

    async def test_cross_service_state_new_target_gets_initial_value(self) -> None:
        """
        If a new downstream target is added for an already-watched remote key,
        it should receive the current value even if the upstream doesn't change.
        """
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
        graph_v1 = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[node_a, node_b], edges=[edge])

        await bus_a.set_rungraph(graph_v1)
        await bus_b.set_rungraph(graph_v1)

        await bus_a.publish_state_runtime("opA", "out", "v1", ts_ms=1)
        out = (await bus_b.get_state("opB", "input")).value
        self.assertEqual(out, "v1")

        # Add a second downstream binding for the same remote key.
        node_c = F8RuntimeNode(
            nodeId="opC",
            serviceId="svcB",
            serviceClass="svcB",
            operatorClass="OpC",
            stateFields=[
                F8StateSpec(name="input2", valueSchema=string_schema(), access=F8StateAccess.rw),
            ],
        )
        edge2 = F8Edge(
            edgeId="e2",
            fromServiceId="svcA",
            fromOperatorId="opA",
            fromPort="out",
            toServiceId="svcB",
            toOperatorId="opC",
            toPort="input2",
            kind=F8EdgeKindEnum.state,
            strategy=F8EdgeStrategyEnum.latest,
        )
        graph_v2 = F8RuntimeGraph(graphId="g1", revision="r2", nodes=[node_a, node_b, node_c], edges=[edge, edge2])

        await bus_a.set_rungraph(graph_v2)
        await bus_b.set_rungraph(graph_v2)

        out2 = (await bus_b.get_state("opC", "input2")).value
        self.assertEqual(out2, "v1")

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
        out = (await bus.get_state("opB", "input")).value
        self.assertEqual(out, "v1")

        await bus.publish_state_runtime("opA", "out", "v2", ts_ms=2)
        out = (await bus.get_state("opB", "input")).value
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
        with self.assertRaises(RuntimeError):
            await bus.set_rungraph(graph)

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

        out = (await bus.get_state("opB", "input")).value
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

        v_cfg = (await bus.get_state("opA", "cfg")).value
        v_mode = (await bus.get_state("opA", "mode")).value
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

        v_cfg = (await bus.get_state("opA", "cfg")).value
        v_mode = (await bus.get_state("opA", "mode")).value
        self.assertEqual(v_cfg, "v2")
        self.assertEqual(v_mode, "m2")

    async def test_rungraph_state_values_do_not_overwrite_equal_timestamp_state(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")

        node = F8RuntimeNode(
            nodeId="opA",
            serviceId="svcA",
            serviceClass="svcA",
            operatorClass="OpA",
            stateFields=[
                F8StateSpec(name="cfg", valueSchema=string_schema(), access=F8StateAccess.rw),
            ],
            stateValues={"cfg": "rungraph"},
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[node], edges=[])
        await bus.publish_state_runtime("opA", "cfg", "runtime", ts_ms=100)

        with patch("f8pysdk.service_bus.workflow.rungraph.now_ms", return_value=100):
            await bus.set_rungraph(graph)

        state = await bus.get_state("opA", "cfg")
        self.assertTrue(state.found)
        self.assertEqual(state.value, "runtime")
        self.assertEqual(state.ts_ms, 100)

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

        svc_id = (await bus.get_state("svcA", "svcId")).value
        op_svc_id = (await bus.get_state("opA", "svcId")).value
        op_id = (await bus.get_state("opA", "operatorId")).value
        self.assertEqual(svc_id, "svcA")
        self.assertEqual(op_svc_id, "svcA")
        self.assertEqual(op_id, "opA")

        with self.assertRaises(StateWriteError):
            await bus.publish_state_external("opA", "operatorId", "x", ts_ms=2)
        with self.assertRaises(StateWriteError):
            await bus.publish_state_external("svcA", "svcId", "x", ts_ms=2)

    async def test_hidden_command_input_dispatches_and_fanouts_output(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _CommandServiceNode("svc")
        sink_node = _RecordingNode("sink")
        bus.register_node(service_node)
        bus.register_node(sink_node)

        hidden_fields = hidden_command_state_specs(list(service_node.spec.commands or []))
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="svc",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    stateFields=hidden_fields,
                ),
                F8RuntimeNode(
                    nodeId="sink",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    operatorClass="Sink",
                    stateFields=[F8StateSpec(name="value", valueSchema=string_schema(), access=F8StateAccess.rw)],
                ),
            ],
            edges=[
                F8Edge(
                    edgeId="e1",
                    fromServiceId="svc",
                    fromOperatorId="svc",
                    fromPort=command_output_state_field("run"),
                    toServiceId="svc",
                    toOperatorId="sink",
                    toPort="value",
                    kind=F8EdgeKindEnum.state,
                    strategy=F8EdgeStrategyEnum.latest,
                )
            ],
        )
        await bus.set_rungraph(graph)

        await bus.publish_state_external("svc", command_input_state_field("run"), [1, 2, 3], ts_ms=10)

        self.assertEqual(service_node.command_calls, [("run", {"a": 1, "b": 2})])
        self.assertEqual(
            service_node.command_meta_calls,
            [
                {
                    "source": "endpoint",
                    "tsMs": 10,
                    "commandInputField": command_input_state_field("run"),
                }
            ],
        )
        self.assertEqual(service_node.on_state_calls, [])
        self.assertEqual((await bus.get_state("sink", "value")).value, {"echo": {"a": 1, "b": 2}})
        self.assertEqual(sink_node.state_calls, [("value", {"echo": {"a": 1, "b": 2}})])

    async def test_hidden_command_input_repeated_same_value_dispatches_each_write(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _CommandServiceNode("svc")
        bus.register_node(service_node)

        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="svc",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    stateFields=hidden_command_state_specs(list(service_node.spec.commands or [])),
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        await bus.publish_state_external("svc", command_input_state_field("nop"), {}, ts_ms=10)
        await bus.publish_state_external("svc", command_input_state_field("nop"), {}, ts_ms=11)

        self.assertEqual(service_node.command_calls, [("nop", {}), ("nop", {})])

    async def test_hidden_command_busy_policy_keeps_latest_pending_value(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _CommandServiceNode("svc")
        service_node._block_event = asyncio.Event()
        bus.register_node(service_node)

        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="svc",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    stateFields=hidden_command_state_specs(list(service_node.spec.commands or [])),
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        task1 = asyncio.create_task(bus.publish_state_external("svc", command_input_state_field("run"), 1, ts_ms=1))
        while service_node.command_calls != [("run", {"a": 1})]:
            await asyncio.sleep(0)
        task2 = asyncio.create_task(bus.publish_state_external("svc", command_input_state_field("run"), 2, ts_ms=2))
        task3 = asyncio.create_task(bus.publish_state_external("svc", command_input_state_field("run"), 3, ts_ms=3))
        await asyncio.sleep(0)
        service_node._block_event.set()
        await asyncio.gather(task1, task2, task3)

        self.assertEqual(service_node.command_calls, [("run", {"a": 1}), ("run", {"a": 3})])
        self.assertEqual((await bus.get_state("svc", command_output_state_field("run"))).value, {"echo": {"a": 3}})

    async def test_command_endpoint_success_is_reply_first(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _CommandServiceNode("svc")
        bus.register_node(service_node)
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="svc",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    stateFields=hidden_command_state_specs(list(service_node.spec.commands or [])),
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        endpoint = ServiceBusControlHandlers(bus)
        req = _FakeReq(
            F8CommandInvokeRequest(
                reqId="r1",
                call="nop",
                args={},
                meta={"source": "ui"},
            )
        )
        await endpoint._cmd(req)

        self.assertIsNotNone(req.response)
        reply = decode_as(req.response or b"", F8CommandInvokeReply)
        self.assertTrue(reply.ok)
        hidden_output = await bus.get_state("svc", command_output_state_field("nop"))
        self.assertFalse(hidden_output.found)

    async def test_service_bus_invoke_command_is_reply_first_by_default(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _CommandServiceNode("svc")
        bus.register_node(service_node)
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="svc",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    stateFields=hidden_command_state_specs(list(service_node.spec.commands or [])),
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        result = await bus.invoke_command("svc", "nop", {}, meta={"source": "test"})

        self.assertTrue(result.ok)
        self.assertEqual(result.value, {"called": "nop"})
        hidden_output = await bus.get_state("svc", command_output_state_field("nop"))
        self.assertFalse(hidden_output.found)

    async def test_service_bus_invoke_command_can_opt_into_hidden_output_writeback(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _CommandServiceNode("svc")
        bus.register_node(service_node)
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="svc",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    stateFields=hidden_command_state_specs(list(service_node.spec.commands or [])),
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        result = await bus.invoke_command("svc", "nop", {}, output_policy=CommandOutputPolicy.hidden_state)

        self.assertTrue(result.ok)
        self.assertEqual(result.value, {"called": "nop"})
        self.assertEqual((await bus.get_state("svc", command_output_state_field("nop"))).value, {"called": "nop"})

    async def test_service_bus_invoke_command_can_skip_hidden_output_writeback(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _CommandServiceNode("svc")
        bus.register_node(service_node)
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="svc",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    stateFields=hidden_command_state_specs(list(service_node.spec.commands or [])),
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        result = await bus.invoke_command("svc", "nop", {}, output_policy=CommandOutputPolicy.none)

        self.assertTrue(result.ok)
        self.assertEqual(result.value, {"called": "nop"})
        hidden_output = await bus.get_state("svc", command_output_state_field("nop"))
        self.assertFalse(hidden_output.found)

    async def test_service_bus_invoke_command_reports_missing_target(self) -> None:
        bus = ServiceBus(ServiceBusConfig(service_id="svc"))

        result = await bus.invoke_command("svc", "nop", {})

        self.assertFalse(result.ok)
        self.assertEqual(result.error_kind, CommandExecutionErrorKind.missing_target)
        self.assertEqual(result.error_message, "unknown call: nop")

    async def test_command_paths_share_missing_target_handling(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _NonCommandServiceNode("svc")
        bus.register_node(service_node)
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="svc",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    stateFields=hidden_command_state_specs(list(service_node.spec.commands or [])),
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        await bus.publish_state_external("svc", command_input_state_field("run"), 7, ts_ms=10)
        hidden_output = await bus.get_state("svc", command_output_state_field("run"))
        self.assertFalse(hidden_output.found)

        endpoint = ServiceBusControlHandlers(bus)
        req = _FakeReq(F8CommandInvokeRequest(reqId="r1", call="run", args={"a": 7}, meta={"source": "ui"}))
        await endpoint._cmd(req)

        self.assertIsNotNone(req.response)
        reply = decode_as(req.response or b"", F8CommandInvokeReply)
        self.assertFalse(reply.ok)
        self.assertIsNotNone(reply.error)
        self.assertEqual(reply.error.code, Code.UNKNOWN_CALL)

    async def test_command_paths_share_handler_failure_handling(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _FailingCommandServiceNode("svc")
        bus.register_node(service_node)
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="svc",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    stateFields=hidden_command_state_specs(list(service_node.spec.commands or [])),
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        await bus.publish_state_external("svc", command_input_state_field("run"), 7, ts_ms=10)
        hidden_output = await bus.get_state("svc", command_output_state_field("run"))
        self.assertFalse(hidden_output.found)

        endpoint = ServiceBusControlHandlers(bus)
        req = _FakeReq(F8CommandInvokeRequest(reqId="r1", call="run", args={"a": 7}, meta={"source": "ui"}))
        await endpoint._cmd(req)

        self.assertEqual(service_node.command_calls, [("run", {"a": 7}), ("run", {"a": 7})])
        self.assertIsNotNone(req.response)
        reply = decode_as(req.response or b"", F8CommandInvokeReply)
        self.assertFalse(reply.ok)
        self.assertIsNotNone(reply.error)
        self.assertEqual(reply.error.code, Code.INTERNAL)

    async def test_hidden_writeback_failure_does_not_fail_command_invocation(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _CommandServiceNode("svc")
        bus.register_node(service_node)
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="svc",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    stateFields=hidden_command_state_specs(list(service_node.spec.commands or [])),
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        endpoint = ServiceBusControlHandlers(bus)
        with patch(
            "f8pysdk.service_bus.internal.command.CommandGateway.write_output",
            new=AsyncMock(side_effect=RuntimeError("writeback failed")),
        ):
            await bus.publish_state_external("svc", command_input_state_field("nop"), {}, ts_ms=10)
            invoke_result = await bus.invoke_command("svc", "nop", {}, output_policy=CommandOutputPolicy.hidden_state)
            req = _FakeReq(F8CommandInvokeRequest(reqId="r1", call="nop", args={}, meta={"source": "ui"}))
            await endpoint._cmd(req)

        self.assertEqual(service_node.command_calls, [("nop", {}), ("nop", {}), ("nop", {})])
        hidden_output = await bus.get_state("svc", command_output_state_field("nop"))
        self.assertFalse(hidden_output.found)
        self.assertTrue(invoke_result.ok)
        self.assertEqual(invoke_result.value, {"called": "nop"})
        self.assertIsNotNone(req.response)
        reply = decode_as(req.response or b"", F8CommandInvokeReply)
        self.assertTrue(reply.ok)
        self.assertEqual(reply.result, {"called": "nop"})

    async def test_hidden_state_command_and_micro_command_share_scalar_arg_semantics(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _CommandServiceNode("svc")
        bus.register_node(service_node)
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="svc",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    stateFields=hidden_command_state_specs(list(service_node.spec.commands or [])),
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        await bus.publish_state_external("svc", command_input_state_field("run"), 7, ts_ms=10)

        endpoint = ServiceBusControlHandlers(bus)
        req = _FakeReq(
            {
                "reqId": "r1",
                "call": "run",
                "args": 7,
                "meta": {"source": "ui"},
            }
        )
        await endpoint._cmd(req)
        reply = decode_as(req.response or b"", F8CommandInvokeReply)

        self.assertEqual(
            service_node.command_calls,
            [
                ("run", {"a": 7}),
                ("run", {"a": 7}),
            ],
        )
        self.assertTrue(reply.ok)

    async def test_hidden_state_command_and_micro_command_share_list_arg_semantics(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _CommandServiceNode("svc")
        bus.register_node(service_node)
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="svc",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    stateFields=hidden_command_state_specs(list(service_node.spec.commands or [])),
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        await bus.publish_state_external("svc", command_input_state_field("run"), [7, 8], ts_ms=10)

        endpoint = ServiceBusControlHandlers(bus)
        req = _FakeReq(
            {
                "reqId": "r1",
                "call": "run",
                "args": [7, 8],
                "meta": {"source": "ui"},
            }
        )
        await endpoint._cmd(req)
        reply = decode_as(req.response or b"", F8CommandInvokeReply)

        self.assertEqual(
            service_node.command_calls,
            [
                ("run", {"a": 7, "b": 8}),
                ("run", {"a": 7, "b": 8}),
            ],
        )
        self.assertTrue(reply.ok)

    async def test_unregister_node_refreshes_command_hidden_bindings(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _CommandServiceNode("svc")
        bus.register_node(service_node)
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="svc",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    stateFields=hidden_command_state_specs(list(service_node.spec.commands or [])),
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        self.assertTrue(bus.command_gateway.is_hidden_field(node_id="svc", field=command_input_state_field("run")))
        self.assertIsNotNone(bus.command_gateway.input_binding(node_id="svc", field=command_input_state_field("run")))

        bus.unregister_node("svc")

        self.assertFalse(bus.command_gateway.is_hidden_field(node_id="svc", field=command_input_state_field("run")))
        self.assertIsNone(bus.command_gateway.input_binding(node_id="svc", field=command_input_state_field("run")))

    async def test_micro_command_rejects_positional_args_without_declared_params(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _CommandServiceNode("svc")
        bus.register_node(service_node)
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="svc",
                    serviceId="svc",
                    serviceClass="svc.test.command",
                    stateFields=hidden_command_state_specs(list(service_node.spec.commands or [])),
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        req = _FakeReq({"reqId": "r1", "call": "dynamic", "args": [7], "meta": {"source": "ui"}})
        endpoint = ServiceBusControlHandlers(bus)
        await endpoint._cmd(req)
        reply = decode_as(req.response or b"", F8CommandInvokeReply)

        self.assertEqual(service_node.command_calls, [])
        self.assertFalse(reply.ok)
        self.assertIsNotNone(reply.error)
        self.assertEqual(reply.error.code, Code.INVALID_ARGS)

    async def test_service_bus_invoke_command_rejects_positional_args_without_declared_params(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svc")
        service_node = _CommandServiceNode("svc")
        bus.register_node(service_node)

        result = await bus.invoke_command("svc", "dynamic", [7])

        self.assertFalse(result.ok)
        self.assertEqual(result.error_kind, CommandExecutionErrorKind.invalid_args)
        self.assertEqual(service_node.command_calls, [])


if __name__ == "__main__":
    unittest.main()
