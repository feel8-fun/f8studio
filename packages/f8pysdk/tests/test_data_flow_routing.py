from __future__ import annotations

import asyncio
import logging
import os
import sys
import unittest
from typing import Any

from msgspec import UNSET

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.specs import (  # noqa: E402
    F8ComplexObjectTypeSchema,
    F8DataPortDelivery,
    F8DataPortPayloadKind,
    F8DataPortSpec,
    F8DataStreamCongestion,
    F8DataStreamPriority,
    F8DataStreamReliability,
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8RuntimeGraph,
    F8RuntimeNode,
    audio_chunk_port,
    data_port_payload_kind,
    data_port_stream_delivery,
    integer_schema,
    string_schema,
    video_frame_port,
    video_frame_metadata_schema,
)
from f8pysdk.f8_naming import data_key  # noqa: E402
from f8pysdk.codec import decode_obj, encode_obj  # noqa: E402
from f8pysdk.bus import ServiceBus, ServiceBusConfig  # noqa: E402
from f8pysdk.service_bus.data.emit import DataEmitOptions  # noqa: E402
from f8pysdk.service_bus.internal.micro import ServiceBusControlHandlers  # noqa: E402
from f8pysdk.testing import InMemoryCluster, InMemoryTransport, push_input  # noqa: E402


class _RecordingTransport(InMemoryTransport):
    def __init__(self, *, cluster: InMemoryCluster) -> None:
        super().__init__(cluster=cluster)
        self.published_keys: list[str] = []

    async def publish(self, key: str, payload: bytes) -> None:
        self.published_keys.append(str(key))
        await super().publish(key, payload)


class _DataReceiverNode:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.data_calls: list[tuple[str, object, int | None]] = []

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

    async def on_data(self, port: str, value: object, *, ts_ms: int | None = None) -> None:
        self.data_calls.append((str(port), value, ts_ms))


class _ComputableNode:
    def __init__(self, node_id: str, value: Any) -> None:
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


class _ControlEndpointRequest:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.response = b""

    async def respond(self, payload: bytes) -> None:
        self.response = bytes(payload)


def _runtime_node(*, node_id: str, service_id: str, data_in: list[str] | None = None, data_out: list[str] | None = None) -> F8RuntimeNode:
    return F8RuntimeNode(
        nodeId=node_id,
        serviceId=service_id,
        serviceClass=service_id,
        operatorClass=f"data_flow.{node_id}",
        dataInPorts=list(data_in or []),
        dataOutPorts=list(data_out or []),
    )


def _video_port(name: str) -> F8DataPortSpec:
    return video_frame_port(
        name=name,
        definition_protected=True,
    )


def _legacy_video_metadata_schema() -> F8ComplexObjectTypeSchema:
    return F8ComplexObjectTypeSchema(
        properties={
            "schemaVersion": integer_schema(),
            "format": string_schema(enum=["bgra32", "bgr24", "flow2_f16", "scalar1_f32"]),
            "width": integer_schema(),
            "height": integer_schema(),
            "pitch": integer_schema(),
            "frameId": integer_schema(),
            "tsMs": integer_schema(),
        },
        required=["schemaVersion", "format", "width", "height", "pitch", "frameId", "tsMs"],
        additionalProperties=False,
    )


def _legacy_video_port(name: str) -> F8DataPortSpec:
    return F8DataPortSpec(
        name=name,
        valueSchema=_legacy_video_metadata_schema(),
        definitionProtected=True,
    )


def _legacy_audio_metadata_schema() -> F8ComplexObjectTypeSchema:
    return F8ComplexObjectTypeSchema(
        properties={
            "schemaVersion": integer_schema(),
            "format": string_schema(enum=["f32le"]),
            "sampleRate": integer_schema(),
            "channels": integer_schema(),
            "frames": integer_schema(),
            "bytesPerFrame": integer_schema(),
            "seq": integer_schema(),
            "frameIndex": integer_schema(),
            "tsMs": integer_schema(),
        },
        required=[
            "schemaVersion",
            "format",
            "sampleRate",
            "channels",
            "frames",
            "bytesPerFrame",
            "seq",
            "frameIndex",
            "tsMs",
        ],
        additionalProperties=False,
    )


def _legacy_audio_port(name: str) -> F8DataPortSpec:
    return F8DataPortSpec(
        name=name,
        valueSchema=_legacy_audio_metadata_schema(),
        definitionProtected=True,
    )


def _data_edge(
    *,
    edge_id: str,
    from_service: str,
    from_node: str,
    from_port: str,
    to_service: str,
    to_node: str,
    to_port: str,
) -> F8Edge:
    return F8Edge(
        edgeId=edge_id,
        fromServiceId=from_service,
        fromOperatorId=from_node,
        fromPort=from_port,
        toServiceId=to_service,
        toOperatorId=to_node,
        toPort=to_port,
        kind=F8EdgeKindEnum.data,
        strategy=F8EdgeStrategyEnum.latest,
    )


async def _sleep_ticks(ticks: int) -> None:
    for _ in range(max(0, int(ticks))):
        await asyncio.sleep(0)


class DataFlowRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_video_frame_port_uses_explicit_payload_and_stream_specs(self) -> None:
        port = video_frame_port(name="video", definition_protected=True)

        self.assertEqual(port.payload.kind, F8DataPortPayloadKind.video_frame)
        self.assertEqual(port.stream.delivery, F8DataPortDelivery.latest)
        self.assertEqual(port.stream.reliability, F8DataStreamReliability.best_effort)
        self.assertEqual(port.stream.congestion, F8DataStreamCongestion.drop)
        self.assertEqual(port.stream.priority, F8DataStreamPriority.real_time)
        self.assertEqual(port.payloadKind, F8DataPortPayloadKind.video_frame)
        self.assertEqual(port.delivery, F8DataPortDelivery.latest)
        self.assertEqual(port.payload.schemaVersion, 2)
        self.assertEqual(list(port.payload.formats), ["bgra32", "bgr24", "flow2_f16", "scalar1_f32"])
        self.assertIs(port.valueSchema, port.payload.metadataSchema)
        self.assertEqual(
            port.valueSchema.required,
            ["schemaVersion", "format", "width", "height", "pitch", "frameId", "tsMs", "streamEpoch"],
        )
        self.assertEqual(port.valueSchema.title, "F8 Video Frame Stream Metadata")
        self.assertIn("video_frame data stream", str(port.valueSchema.description))
        self.assertIs(port.valueSchema.additionalProperties, False)
        self.assertIs(port.valueSchema.field_comment, UNSET)

    async def test_audio_chunk_port_uses_explicit_payload_and_stream_specs(self) -> None:
        port = audio_chunk_port(name="audio", definition_protected=True)

        self.assertEqual(port.payload.kind, F8DataPortPayloadKind.audio_chunk)
        self.assertEqual(port.stream.delivery, F8DataPortDelivery.latest)
        self.assertEqual(port.stream.priority, F8DataStreamPriority.real_time)
        self.assertEqual(port.payloadKind, F8DataPortPayloadKind.audio_chunk)
        self.assertEqual(port.delivery, F8DataPortDelivery.latest)
        self.assertEqual(port.payload.schemaVersion, 1)
        self.assertEqual(list(port.payload.formats), ["f32le"])
        self.assertIs(port.valueSchema, port.payload.metadataSchema)
        self.assertIs(port.valueSchema.additionalProperties, False)
        self.assertIs(port.valueSchema.field_comment, UNSET)

    async def test_video_frame_metadata_schema_has_no_payload_kind_comment(self) -> None:
        schema = video_frame_metadata_schema()

        self.assertIs(schema.field_comment, UNSET)

    async def test_legacy_video_metadata_schema_infers_stream_payload_kind(self) -> None:
        port = _legacy_video_port("video")

        self.assertEqual(data_port_payload_kind(port), F8DataPortPayloadKind.video_frame)
        self.assertEqual(data_port_stream_delivery(port), F8DataPortDelivery.latest)

    async def test_legacy_audio_metadata_schema_infers_stream_payload_kind(self) -> None:
        port = _legacy_audio_port("audio")

        self.assertEqual(data_port_payload_kind(port), F8DataPortPayloadKind.audio_chunk)
        self.assertEqual(data_port_stream_delivery(port), F8DataPortDelivery.latest)

    async def test_legacy_video_metadata_schema_resolves_stream_key_in_split_graph(self) -> None:
        cluster = InMemoryCluster()
        transport = _RecordingTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="sink"), transport=transport)
        subject = data_key("player", from_node_id="player", port_id="video")

        graph = F8RuntimeGraph(
            graphId="g-legacy-video-stream-edge",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="sink",
                    serviceId="sink",
                    serviceClass="f8.sink",
                    dataInPorts=[_legacy_video_port("video")],
                ),
            ],
            edges=[
                _data_edge(
                    edge_id="video",
                    from_service="player",
                    from_node="player",
                    from_port="video",
                    to_service="sink",
                    to_node="sink",
                    to_port="video",
                ),
            ],
        )

        await bus.set_rungraph(graph)

        self.assertEqual(bus.data_input_zenoh_key("sink", "video"), subject)
        self.assertEqual(bus.data_router.cross_in_by_key, {})
        self.assertEqual(bus.data_router.input_buffers, {})

    async def test_video_frame_edge_resolves_stream_key_without_json_subscription(self) -> None:
        cluster = InMemoryCluster()
        transport = _RecordingTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="sink"), transport=transport)
        subject = data_key("player", from_node_id="player", port_id="video")

        graph = F8RuntimeGraph(
            graphId="g-video-stream-edge",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="player",
                    serviceId="player",
                    serviceClass="f8.implayer",
                    dataOutPorts=[_video_port("video")],
                ),
                F8RuntimeNode(
                    nodeId="sink",
                    serviceId="sink",
                    serviceClass="f8.sink",
                    dataInPorts=[_video_port("video")],
                ),
            ],
            edges=[
                _data_edge(
                    edge_id="video",
                    from_service="player",
                    from_node="player",
                    from_port="video",
                    to_service="sink",
                    to_node="sink",
                    to_port="video",
                ),
            ],
        )

        await bus.set_rungraph(graph)

        self.assertEqual(bus.data_input_zenoh_key("sink", "video"), subject)
        self.assertEqual(bus.data_router.cross_in_by_key, {})
        self.assertEqual(bus.data_router.input_buffers, {})

    async def test_video_frame_service_node_edge_uses_service_id_as_stream_node(self) -> None:
        cluster = InMemoryCluster()
        transport = _RecordingTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="sink"), transport=transport)
        subject = data_key("player", from_node_id="player", port_id="video")

        graph = F8RuntimeGraph(
            graphId="g-video-service-node-edge",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="player",
                    serviceId="player",
                    serviceClass="f8.implayer",
                    dataOutPorts=[_video_port("video")],
                ),
                F8RuntimeNode(
                    nodeId="sink",
                    serviceId="sink",
                    serviceClass="f8.sink",
                    dataInPorts=[_video_port("video")],
                ),
            ],
            edges=[
                F8Edge(
                    edgeId="video",
                    fromServiceId="player",
                    fromPort="video",
                    toServiceId="sink",
                    toPort="video",
                    kind=F8EdgeKindEnum.data,
                    strategy=F8EdgeStrategyEnum.latest,
                ),
            ],
        )

        await bus.set_rungraph(graph)

        self.assertEqual(bus.data_input_zenoh_key("sink", "video"), subject)
        self.assertEqual(bus.data_router.cross_in_by_key, {})
        self.assertEqual(bus.data_router.input_buffers, {})

    async def test_default_cross_publish_policy_is_routed(self) -> None:
        cluster = InMemoryCluster()
        transport = _RecordingTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="svc"), transport=transport)

        await bus.emit_data("node1", "out", {"x": 1}, ts_ms=1)

        self.assertEqual(bus.cross_publish_policy, "routed")
        self.assertEqual(transport.published_keys, [])

    async def test_callback_delivery_invokes_callback_and_keeps_pull_buffer(self) -> None:
        cluster = InMemoryCluster()
        transport = InMemoryTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="svc", data_delivery="callback"), transport=transport)
        node = _DataReceiverNode("node1")
        bus.register_node(node)

        push_input(bus, "node1", "in", 123, ts_ms=5)
        await _sleep_ticks(2)
        pulled = await bus.pull_data("node1", "in")

        self.assertEqual(node.data_calls, [("in", 123, 5)])
        self.assertEqual(pulled, 123)

    async def test_buffered_delivery_keeps_pull_buffer_without_callback(self) -> None:
        cluster = InMemoryCluster()
        transport = InMemoryTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="svc", data_delivery="buffered"), transport=transport)
        node = _DataReceiverNode("node1")
        bus.register_node(node)

        push_input(bus, "node1", "in", 123, ts_ms=5)
        await _sleep_ticks(2)
        pulled = await bus.pull_data("node1", "in")

        self.assertEqual(node.data_calls, [])
        self.assertEqual(pulled, 123)

    async def test_legacy_data_delivery_aliases_fail_fast(self) -> None:
        cluster = InMemoryCluster()
        transport = InMemoryTransport(cluster=cluster)

        for mode in ("push", "pull", "both"):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    ServiceBus(ServiceBusConfig(service_id="svc", data_delivery=mode), transport=transport)

    async def test_pending_push_callbacks_are_dropped_when_bus_deactivates(self) -> None:
        cluster = InMemoryCluster()
        transport = InMemoryTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="svc", data_delivery="callback"), transport=transport)
        node = _DataReceiverNode("node1")
        bus.register_node(node)

        push_input(bus, "node1", "in", 123, ts_ms=5)
        await bus.set_active(False)
        await _sleep_ticks(2)

        self.assertEqual(node.data_calls, [])

    async def test_data_router_stop_flush_task_logs_unexpected_failure(self) -> None:
        cluster = InMemoryCluster()
        transport = InMemoryTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="svc", data_delivery="callback"), transport=transport)

        async def _broken_flush() -> None:
            raise RuntimeError("flush failed")

        task = asyncio.create_task(_broken_flush())
        await _sleep_ticks(1)
        bus.data_router._on_data_flush_task = task

        with self.assertLogs("f8pysdk.service_bus.data.router", level=logging.ERROR) as captured:
            await bus.data_router._stop_flush_task()

        self.assertIsNone(bus.data_router._on_data_flush_task)
        self.assertTrue(any("on_data flush task stop failed" in line for line in captured.output))

    async def test_push_input_is_ignored_while_bus_inactive(self) -> None:
        cluster = InMemoryCluster()
        transport = InMemoryTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="svc", data_delivery="callback"), transport=transport)
        node = _DataReceiverNode("node1")
        bus.register_node(node)

        await bus.set_active(False)
        push_input(bus, "node1", "in", 123, ts_ms=5)
        await _sleep_ticks(2)

        self.assertEqual(node.data_calls, [])

    async def test_pull_triggered_compute_stays_local_even_when_cross_publish_is_all(self) -> None:
        cluster = InMemoryCluster()
        transport = _RecordingTransport(cluster=cluster)
        bus = ServiceBus(
            ServiceBusConfig(service_id="svcA", cross_publish_policy="all", data_delivery="buffered"),
            transport=transport,
        )
        source = _ComputableNode("src", 42)
        bus.register_node(source)

        graph = F8RuntimeGraph(
            graphId="g-data-flow-routing",
            revision="r1",
            nodes=[
                _runtime_node(node_id="src", service_id="svcA", data_out=["out"]),
                _runtime_node(node_id="local_dst", service_id="svcA", data_in=["in"]),
                _runtime_node(node_id="remote_dst", service_id="svcB", data_in=["in"]),
            ],
            edges=[
                _data_edge(
                    edge_id="local",
                    from_service="svcA",
                    from_node="src",
                    from_port="out",
                    to_service="svcA",
                    to_node="local_dst",
                    to_port="in",
                ),
                _data_edge(
                    edge_id="remote",
                    from_service="svcA",
                    from_node="src",
                    from_port="out",
                    to_service="svcB",
                    to_node="remote_dst",
                    to_port="in",
                ),
            ],
        )

        await bus.set_rungraph(graph)
        pulled = await bus.pull_data("local_dst", "in", ctx_id="ctx-1")

        self.assertEqual(pulled, 42)
        self.assertEqual(source.compute_calls, [("out", "ctx-1")])
        self.assertEqual(transport.published_keys, [])
        self.assertEqual(bus.data_router.input_buffers[("local_dst", "in")].last_seen_value, 42)

    async def test_cross_publish_policy_all_publishes_without_cross_routes(self) -> None:
        cluster = InMemoryCluster()
        transport = _RecordingTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="svc", cross_publish_policy="all"), transport=transport)

        await bus.emit_data("node1", "out", {"x": 1}, ts_ms=1)

        self.assertEqual(bus.cross_publish_policy, "all")
        self.assertEqual(
            transport.published_keys,
            [data_key("svc", from_node_id="node1", port_id="out")],
        )

    async def test_data_emit_options_can_force_local_only_emit_without_cross_publish(self) -> None:
        cluster = InMemoryCluster()
        transport = _RecordingTransport(cluster=cluster)
        bus = ServiceBus(
            ServiceBusConfig(service_id="svcA", cross_publish_policy="all", data_delivery="buffered"),
            transport=transport,
        )

        graph = F8RuntimeGraph(
            graphId="g-local-only-emit",
            revision="r1",
            nodes=[
                _runtime_node(node_id="src", service_id="svcA", data_out=["out"]),
                _runtime_node(node_id="local_dst", service_id="svcA", data_in=["in"]),
                _runtime_node(node_id="remote_dst", service_id="svcB", data_in=["in"]),
            ],
            edges=[
                _data_edge(
                    edge_id="local",
                    from_service="svcA",
                    from_node="src",
                    from_port="out",
                    to_service="svcA",
                    to_node="local_dst",
                    to_port="in",
                ),
                _data_edge(
                    edge_id="remote",
                    from_service="svcA",
                    from_node="src",
                    from_port="out",
                    to_service="svcB",
                    to_node="remote_dst",
                    to_port="in",
                ),
            ],
        )

        await bus.set_rungraph(graph)
        await bus.data_router.emit_data(
            "src",
            "out",
            99,
            ts_ms=11,
            options=DataEmitOptions.local_compute_only(),
        )

        self.assertEqual(transport.published_keys, [])
        self.assertEqual(bus.data_router.input_buffers[("local_dst", "in")].last_seen_value, 99)

    async def test_data_router_debug_input_buffers_reports_local_values(self) -> None:
        cluster = InMemoryCluster()
        transport = _RecordingTransport(cluster=cluster)
        bus = ServiceBus(
            ServiceBusConfig(service_id="svcA", cross_publish_policy="none", data_delivery="buffered"),
            transport=transport,
        )
        graph = F8RuntimeGraph(
            graphId="g-debug-input-buffers",
            revision="r1",
            nodes=[
                _runtime_node(node_id="src", service_id="svcA", data_out=["out"]),
                _runtime_node(node_id="local_dst", service_id="svcA", data_in=["in"]),
            ],
            edges=[
                _data_edge(
                    edge_id="local",
                    from_service="svcA",
                    from_node="src",
                    from_port="out",
                    to_service="svcA",
                    to_node="local_dst",
                    to_port="in",
                ),
            ],
        )

        await bus.set_rungraph(graph)
        await bus.emit_data("src", "out", {"axis": 0.5}, ts_ms=12, ctx_id="ctx-a")
        pulled = await bus.pull_data("local_dst", "in", ctx_id="ctx-b")

        self.assertEqual(pulled, {"axis": 0.5})
        snapshot = bus.data_router.debug_input_buffers(node_id="local_dst", port="in", include_value=True)

        self.assertEqual(snapshot["serviceId"], "svcA")
        self.assertEqual(snapshot["matched"], 1)
        self.assertEqual(snapshot["truncated"], False)
        buffers = snapshot["buffers"]
        self.assertEqual(len(buffers), 1)
        buffer = buffers[0]
        self.assertEqual(buffer["nodeId"], "local_dst")
        self.assertEqual(buffer["port"], "in")
        self.assertEqual(buffer["queueLength"], 0)
        self.assertEqual(buffer["edgeId"], "local")
        self.assertEqual(buffer["lastSeenTs"], 12)
        self.assertEqual(buffer["lastSeenCtxId"], "ctx-a")
        self.assertEqual(buffer["lastSeenValue"], {"axis": 0.5})
        self.assertEqual(buffer["lastPulledCtxId"], "ctx-b")
        self.assertEqual(buffer["lastPulledValue"], {"axis": 0.5})

    async def test_data_router_debug_input_buffers_reports_output_snapshots(self) -> None:
        cluster = InMemoryCluster()
        transport = _RecordingTransport(cluster=cluster)
        bus = ServiceBus(
            ServiceBusConfig(service_id="svcA", cross_publish_policy="none", data_delivery="buffered"),
            transport=transport,
        )
        graph = F8RuntimeGraph(
            graphId="g-debug-output-snapshots",
            revision="r1",
            nodes=[
                _runtime_node(node_id="src", service_id="svcA", data_out=["out"]),
                _runtime_node(node_id="local_dst", service_id="svcA", data_in=["in"]),
            ],
            edges=[
                _data_edge(
                    edge_id="local",
                    from_service="svcA",
                    from_node="src",
                    from_port="out",
                    to_service="svcA",
                    to_node="local_dst",
                    to_port="in",
                ),
            ],
        )

        await bus.set_rungraph(graph)
        await bus.emit_data("src", "out", {"axis": 0.75}, ts_ms=33, ctx_id="ctx-out")

        snapshot = bus.data_router.debug_input_buffers(node_id="src", port="out", include_value=True)

        self.assertEqual(snapshot["matched"], 0)
        self.assertEqual(snapshot["buffers"], [])
        self.assertEqual(snapshot["outputMatched"], 1)
        outputs = snapshot["outputs"]
        self.assertEqual(len(outputs), 1)
        output = outputs[0]
        self.assertEqual(output["nodeId"], "src")
        self.assertEqual(output["port"], "out")
        self.assertEqual(output["tsMs"], 33)
        self.assertEqual(output["ctxId"], "ctx-out")
        self.assertEqual(output["deliveredLocal"], True)
        self.assertEqual(output["crossPublishDecision"], "local_only")
        self.assertEqual(output["lastEmittedValue"], {"axis": 0.75})

    async def test_data_router_debug_input_buffers_can_omit_large_values(self) -> None:
        cluster = InMemoryCluster()
        transport = _RecordingTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="svcA", cross_publish_policy="none"), transport=transport)

        await bus.set_active(True)
        bus.data_router.push_input("node", "in", {"large": "x" * 32}, ts_ms=1)
        snapshot = bus.data_router.debug_input_buffers(
            node_id="node",
            port="in",
            include_value=True,
            max_value_bytes=8,
        )

        buffer = snapshot["buffers"][0]
        self.assertEqual(buffer["lastSeenPayloadKind"], "json_object")
        self.assertEqual(buffer["lastSeenOmitReason"], "value_too_large")
        self.assertNotIn("lastSeenValue", buffer)

    async def test_debug_data_control_endpoint_returns_input_buffer_snapshot(self) -> None:
        cluster = InMemoryCluster()
        transport = _RecordingTransport(cluster=cluster)
        bus = ServiceBus(ServiceBusConfig(service_id="svcA", cross_publish_policy="none"), transport=transport)
        await bus.set_active(True)
        bus.data_router.push_input("node", "in", 123, ts_ms=2)
        await bus.emit_data("node", "out", 456, ts_ms=3)
        request = _ControlEndpointRequest(
            encode_obj(
                {
                    "reqId": "req-debug-input",
                    "args": {"nodeId": "node", "port": "in", "limit": 5, "includeValue": True},
                }
            )
        )
        output_request = _ControlEndpointRequest(
            encode_obj(
                {
                    "reqId": "req-debug-output",
                    "args": {"nodeId": "node", "port": "out", "limit": 5, "includeValue": True},
                }
            )
        )

        handlers = ServiceBusControlHandlers(bus)
        await handlers._debug_data(request)
        await handlers._debug_data(output_request)

        response = decode_obj(request.response)
        self.assertEqual(response["reqId"], "req-debug-input")
        self.assertEqual(response["ok"], True)
        result = response["result"]
        self.assertEqual(result["serviceId"], "svcA")
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["buffers"][0]["lastSeenValue"], 123)

        output_response = decode_obj(output_request.response)
        self.assertEqual(output_response["reqId"], "req-debug-output")
        self.assertEqual(output_response["ok"], True)
        output_result = output_response["result"]
        self.assertEqual(output_result["serviceId"], "svcA")
        self.assertEqual(output_result["matched"], 0)
        self.assertEqual(output_result["outputMatched"], 1)
        self.assertEqual(output_result["outputs"][-1]["lastEmittedValue"], 456)


if __name__ == "__main__":
    unittest.main()
