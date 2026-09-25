import asyncio
from typing import Any

from f8pysdk.registry import Registry
from f8pysdk.host import ServiceHost, ServiceHostConfig
from f8pysdk.specs import (
    F8DataPortSpec,
    F8Edge,
    F8EdgeKindEnum,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    audio_chunk_port,
    number_schema,
    video_frame_port,
)
from f8pysdk.testing import ServiceBusHarness, buffer_input
from f8pysdk.time_utils import now_ms
from f8studio_server.events import EventJournal
from f8studio_server.studio_runtime import (
    EventPresentationOutlet,
    SERVICE_CLASS,
    StudioRuntimeConfig,
    StudioRuntimeService,
    create_studio_registry,
)
from f8studio_server.studio_runtime.operators import (
    DataExprRuntimeNode,
    StateExprRuntimeNode,
    VizAudioRuntimeNode,
    VizThreeDRuntimeNode,
    VizTCodeRuntimeNode,
    VizTrackRuntimeNode,
    VizVideoRuntimeNode,
)


class CapturingPresentationOutlet:
    def __init__(self) -> None:
        self.commands: list[tuple[str, str, dict[str, Any], int | None]] = []

    def emit(
        self,
        node_id: str,
        command: str,
        payload: dict[str, Any],
        *,
        ts_ms: int | None = None,
    ) -> None:
        self.commands.append((node_id, command, payload, ts_ms))


def test_event_presentation_outlet_retains_latest_commands_until_detach() -> None:
    async def scenario() -> None:
        events = EventJournal(server_epoch="test")
        outlet = EventPresentationOutlet(events)
        outlet.emit("video1", "viz.video.set", {"videoStreamKey": "f8/test/video"}, ts_ms=10)
        outlet.emit("text1", "viz.text.update", {"value": "ready"}, ts_ms=11)
        assert [(item.node_id, item.command) for item in outlet.snapshot()] == [
            ("video1", "viz.video.set"),
            ("text1", "viz.text.update"),
        ]

        outlet.emit("video1", "viz.video.detach", {}, ts_ms=12)
        assert [(item.node_id, item.command) for item in outlet.snapshot()] == [("text1", "viz.text.update")]
        await outlet.close()

    asyncio.run(scenario())


def test_studio_registry_is_isolated_and_injects_presentation_outlet() -> None:
    first_outlet = CapturingPresentationOutlet()
    second_outlet = CapturingPresentationOutlet()
    first = create_studio_registry(presentation=first_outlet)
    second = create_studio_registry(presentation=second_outlet)
    first_registry = Registry.wrap(first)
    describe = first_registry.describe(SERVICE_CLASS)

    assert len(describe.operators) == 14
    assert first is not second
    audio_spec = next(spec for spec in describe.operators if str(spec.operatorClass) == "f8.viz.audio")
    node = first_registry.create_operator_node(
        node_id="audio1",
        node=F8RuntimeNode(
            nodeId="audio1",
            serviceId="studio",
            serviceClass=SERVICE_CLASS,
            operatorClass="f8.viz.audio",
            stateFields=audio_spec.stateFields,
            dataInPorts=audio_spec.dataInPorts,
            dataOutPorts=audio_spec.dataOutPorts,
        ),
    )

    assert isinstance(node, VizAudioRuntimeNode)
    assert node.presentation is first_outlet


def test_video_viz_publishes_implayer_stream_key_after_rungraph_routes_are_ready() -> None:
    async def scenario() -> None:
        outlet = CapturingPresentationOutlet()
        harness = ServiceBusHarness()
        bus = harness.create_bus("studio")
        _ = ServiceHost(
            bus,
            config=ServiceHostConfig(service_class=SERVICE_CLASS),
            registry=create_studio_registry(presentation=outlet),
        )
        graph = F8RuntimeGraph(
            graphId="implayer-video",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="player",
                    serviceId="player",
                    serviceClass="f8.implayer",
                    operatorClass="f8.implayer",
                    dataOutPorts=[video_frame_port(name="video")],
                ),
                F8RuntimeNode(
                    nodeId="viewer",
                    serviceId="studio",
                    serviceClass=SERVICE_CLASS,
                    operatorClass=VizVideoRuntimeNode.SPEC.operatorClass,
                    dataInPorts=[video_frame_port(name="video")],
                    stateFields=list(VizVideoRuntimeNode.SPEC.stateFields),
                ),
            ],
            edges=[F8Edge(
                edgeId="player-video",
                fromServiceId="player",
                fromOperatorId="player",
                fromPort="video",
                toServiceId="studio",
                toOperatorId="viewer",
                toPort="video",
                kind=F8EdgeKindEnum.data,
            )],
        )

        await bus.set_rungraph(graph)
        await asyncio.sleep(0)
        video_commands = [payload for _, command, payload, _ in outlet.commands if command == "viz.video.set"]
        assert video_commands
        assert video_commands[-1]["videoStreamKey"] == "f8/svc/player/nodes/player/data/video"

    asyncio.run(scenario())


def test_audio_viz_publishes_audiocap_stream_key_after_rungraph_routes_are_ready() -> None:
    async def scenario() -> None:
        outlet = CapturingPresentationOutlet()
        harness = ServiceBusHarness()
        bus = harness.create_bus("studio")
        _ = ServiceHost(
            bus,
            config=ServiceHostConfig(service_class=SERVICE_CLASS),
            registry=create_studio_registry(presentation=outlet),
        )
        graph = F8RuntimeGraph(
            graphId="audiocap-audio",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="capture",
                    serviceId="capture",
                    serviceClass="f8.audiocap",
                    operatorClass="f8.audiocap",
                    dataOutPorts=[audio_chunk_port(name="audio")],
                ),
                F8RuntimeNode(
                    nodeId="viewer",
                    serviceId="studio",
                    serviceClass=SERVICE_CLASS,
                    operatorClass=VizAudioRuntimeNode.SPEC.operatorClass,
                    dataInPorts=[audio_chunk_port(name="audio")],
                    stateFields=list(VizAudioRuntimeNode.SPEC.stateFields),
                ),
            ],
            edges=[F8Edge(
                edgeId="capture-audio",
                fromServiceId="capture",
                fromOperatorId="capture",
                fromPort="audio",
                toServiceId="studio",
                toOperatorId="viewer",
                toPort="audio",
                kind=F8EdgeKindEnum.data,
            )],
        )

        await bus.set_rungraph(graph)
        await asyncio.sleep(0)
        audio_commands = [payload for _, command, payload, _ in outlet.commands if command == "viz.audio.set"]
        assert audio_commands
        assert audio_commands[-1]["audioStreamKey"] == "f8/svc/capture/nodes/capture/data/audio"

    asyncio.run(scenario())


def test_tcode_operator_is_static_and_emits_local_renderer_commands() -> None:
    async def scenario() -> None:
        outlet = CapturingPresentationOutlet()
        registry = Registry.wrap(create_studio_registry(presentation=outlet))
        node = registry.create_operator_node(
            node_id="tcode1",
            node=F8RuntimeNode(
                nodeId="tcode1",
                serviceId="studio",
                serviceClass=SERVICE_CLASS,
                operatorClass=VizTCodeRuntimeNode.SPEC.operatorClass,
                dataInPorts=list(VizTCodeRuntimeNode.SPEC.dataInPorts),
                stateFields=list(VizTCodeRuntimeNode.SPEC.stateFields),
            ),
            initial_state={"model": "SR6"},
        )
        assert isinstance(node, VizTCodeRuntimeNode)
        await node.on_data("tcode", "L05000 R09999", ts_ms=123)
        assert [command for _, command, _, _ in outlet.commands] == [
            "viz.tcode.set_model",
            "viz.tcode.write",
        ]
        assert outlet.commands[-1][2] == {"line": "L05000 R09999\n"}

    asyncio.run(scenario())


def test_presentation_outlet_publishes_unreliable_scoped_event() -> None:
    async def scenario() -> None:
        events = EventJournal(server_epoch="epoch1")
        stream = await events.open_stream(client_epoch=None, after_sequence=None)
        outlet = EventPresentationOutlet(events)

        outlet.emit("text1", "viz.text.update", {"value": "hello"}, ts_ms=123)
        event = await asyncio.wait_for(stream.queue.get(), timeout=1.0)

        assert event.type == "presentation.command"
        assert event.scope == "node:text1"
        assert event.payload == {
            "nodeId": "text1",
            "command": "viz.text.update",
            "payload": {"value": "hello"},
            "tsMs": 123,
        }
        await outlet.close()
        await events.close_stream(stream.subscription_id)

    asyncio.run(scenario())


def test_expression_operators_are_registered_with_editable_specs() -> None:
    registry = Registry.wrap(create_studio_registry(presentation=CapturingPresentationOutlet()))
    describe = registry.describe(SERVICE_CLASS)
    specs = {str(spec.operatorClass): spec for spec in describe.operators}

    assert {"f8.data_expr", "f8.state_expr"} <= specs.keys()
    data_code = next(field for field in specs["f8.data_expr"].stateFields if field.name == "code")
    state_out = next(field for field in specs["f8.state_expr"].stateFields if field.name == "out")
    assert data_code.valueRequired is True
    assert state_out.valueRequired is True
    assert state_out.access == F8StateAccess.ro


def test_data_expression_dynamic_outputs_and_context_cache() -> None:
    async def scenario() -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("studio")
        registry = create_studio_registry(presentation=CapturingPresentationOutlet())
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=registry)
        node_spec = F8RuntimeNode(
            nodeId="expr1",
            serviceId="studio",
            serviceClass=SERVICE_CLASS,
            operatorClass=DataExprRuntimeNode.SPEC.operatorClass,
            stateFields=list(DataExprRuntimeNode.SPEC.stateFields),
            stateValues={"code": "{'sum': a + b, 'product': a * b}", "unpackDictOutputs": True},
            dataInPorts=[
                F8DataPortSpec(name="a", valueSchema=any_schema(), definitionProtected=False),
                F8DataPortSpec(name="b", valueSchema=any_schema(), definitionProtected=False),
            ],
            dataOutPorts=[
                F8DataPortSpec(name="sum", valueSchema=any_schema(), definitionProtected=False),
                F8DataPortSpec(name="product", valueSchema=any_schema(), definitionProtected=False),
            ],
        )
        await bus.set_rungraph(F8RuntimeGraph(graphId="expr", revision="r1", nodes=[node_spec], edges=[]))
        runtime = bus.get_node("expr1")
        assert isinstance(runtime, DataExprRuntimeNode)

        buffer_input(bus, "expr1", "a", 2, ts_ms=0, edge=None, ctx_id=None)
        buffer_input(bus, "expr1", "b", 5, ts_ms=0, edge=None, ctx_id=None)
        assert await runtime.compute_output("sum", ctx_id=1) == 7
        buffer_input(bus, "expr1", "a", 100, ts_ms=1, edge=None, ctx_id=None)
        assert await runtime.compute_output("product", ctx_id=1) == 10
        assert await runtime.compute_output("product", ctx_id=2) == 500

    asyncio.run(scenario())


def test_data_expression_reports_unavailable_numpy() -> None:
    async def scenario() -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("studio")
        registry = create_studio_registry(presentation=CapturingPresentationOutlet())
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=registry)
        node_spec = F8RuntimeNode(
            nodeId="expr_numpy",
            serviceId="studio",
            serviceClass=SERVICE_CLASS,
            operatorClass=DataExprRuntimeNode.SPEC.operatorClass,
            stateFields=list(DataExprRuntimeNode.SPEC.stateFields),
            stateValues={"code": "np.mean(x)", "allowNumpy": True},
            dataInPorts=[F8DataPortSpec(name="x", valueSchema=any_schema(), definitionProtected=False)],
            dataOutPorts=[F8DataPortSpec(name="out", valueSchema=any_schema(), definitionProtected=False)],
        )
        await bus.set_rungraph(F8RuntimeGraph(graphId="expr_numpy", revision="r1", nodes=[node_spec], edges=[]))
        runtime = bus.get_node("expr_numpy")
        assert isinstance(runtime, DataExprRuntimeNode)
        assert await runtime.compute_output("out", ctx_id=1) is None

        snapshot = bus.monitor_collector._build_snapshot(ts_ms=int(now_ms()))
        assert "NumPy expressions are unavailable" in str(snapshot.error.currentMessage or "")

    asyncio.run(scenario())


def test_state_expression_publishes_changes_and_monitor_errors() -> None:
    async def scenario() -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("studio")
        registry = create_studio_registry(presentation=CapturingPresentationOutlet())
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=registry)
        node_spec = F8RuntimeNode(
            nodeId="state_expr1",
            serviceId="studio",
            serviceClass=SERVICE_CLASS,
            operatorClass=StateExprRuntimeNode.SPEC.operatorClass,
            stateFields=[
                *list(StateExprRuntimeNode.SPEC.stateFields),
                F8StateSpec(
                    name="a",
                    valueSchema=number_schema(default=1.5),
                    access=F8StateAccess.rw,
                    valueRequired=False,
                ),
                F8StateSpec(
                    name="b",
                    valueSchema=number_schema(default=2.5),
                    access=F8StateAccess.rw,
                    valueRequired=False,
                ),
            ],
            stateValues={"code": "a + b", "a": 1.5, "b": 2.5},
        )
        await bus.set_rungraph(F8RuntimeGraph(graphId="state_expr", revision="r1", nodes=[node_spec], edges=[]))
        runtime = bus.get_node("state_expr1")
        assert isinstance(runtime, StateExprRuntimeNode)

        await runtime.on_state("a", 4.0, ts_ms=1)
        assert await runtime.get_state_value("out") == 6.5
        await runtime.on_state("code", "a +", ts_ms=2)
        assert await runtime.get_state_value("out") is None
        snapshot = bus.monitor_collector._build_snapshot(ts_ms=int(now_ms()))
        assert "syntax error" in str(snapshot.error.currentMessage or "")
        await runtime.on_state("code", "a * b", ts_ms=3)
        assert await runtime.get_state_value("out") == 10.0
        cleared = bus.monitor_collector._build_snapshot(ts_ms=int(now_ms()))
        assert str(cleared.error.currentMessage or "") == ""

    asyncio.run(scenario())


def test_track_visualization_normalizes_detection_history() -> None:
    async def scenario() -> None:
        outlet = CapturingPresentationOutlet()
        registry = Registry.wrap(create_studio_registry(presentation=outlet))
        runtime = registry.create_operator_node(
            node_id="track1",
            node=F8RuntimeNode(
                nodeId="track1",
                serviceId="studio",
                serviceClass=SERVICE_CLASS,
                operatorClass=VizTrackRuntimeNode.SPEC.operatorClass,
                dataInPorts=[F8DataPortSpec(name="detections", valueSchema=any_schema())],
            ),
            initial_state={"throttleMs": 0},
        )
        assert isinstance(runtime, VizTrackRuntimeNode)

        await runtime.on_data(
            "detections",
            {
                "schemaVersion": "f8visionDetections/1",
                "width": 640,
                "height": 480,
                "skeletonProtocol": "coco17",
                "detections": [
                    {
                        "id": "42",
                        "bbox": ["1", "2", "3", "4"],
                        "keypoints": [{"x": 1, "y": 2}, "invalid"],
                    }
                ],
            },
            ts_ms=1_000,
        )

        _, command, payload, _ = outlet.commands[-1]
        assert command == "viz.track.set"
        assert payload["width"] == 640
        assert payload["height"] == 480
        assert payload["tracks"] == [
            {
                "id": 42,
                "history": [
                    {
                        "tsMs": 1_000,
                        "kind": "det",
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                        "keypoints": [{"x": 1.0, "y": 2.0}],
                        "skeletonProtocol": "coco17",
                    }
                ],
            }
        ]

    asyncio.run(scenario())


def test_three_d_visualization_aggregates_ports_and_single_bones() -> None:
    async def scenario() -> None:
        outlet = CapturingPresentationOutlet()
        registry = Registry.wrap(create_studio_registry(presentation=outlet))
        runtime = registry.create_operator_node(
            node_id="three1",
            node=F8RuntimeNode(
                nodeId="three1",
                serviceId="studio",
                serviceClass=SERVICE_CLASS,
                operatorClass=VizThreeDRuntimeNode.SPEC.operatorClass,
                dataInPorts=[
                    F8DataPortSpec(name="camA", valueSchema=any_schema()),
                    F8DataPortSpec(name="camB", valueSchema=any_schema()),
                ],
            ),
            initial_state={"throttleMs": 0},
        )
        assert isinstance(runtime, VizThreeDRuntimeNode)

        await runtime.on_data(
            "camB",
            {"modelName": "Avatar", "skeletonProtocol": "none", "bones": [{"name": "root", "pos": [0, 1, 2]}]},
            ts_ms=100,
        )
        await runtime.on_data(
            "camA",
            {"name": "Head", "pos": [1, 2, 3], "rot": [1, 0, 0, 0]},
            ts_ms=101,
        )

        _, command, payload, _ = outlet.commands[-1]
        assert command == "viz.three_d.set"
        people = payload["people"]
        assert isinstance(people, list)
        assert [person["name"] for person in people] == ["camA:camA", "camB:Avatar"]
        assert people[0]["nodes"][0]["pos"] == [1.0, 2.0, 3.0]
        assert payload["worldUp"] == "+y"

        await runtime.on_state("worldUp", "-z", ts_ms=102)
        assert any(command == "viz.three_d.world_up" and data == {"worldUp": "-z"} for _, command, data, _ in outlet.commands)

    asyncio.run(scenario())


def test_studio_runtime_starts_and_stops_without_qt() -> None:
    async def scenario() -> None:
        outlet = CapturingPresentationOutlet()
        service = StudioRuntimeService(
            StudioRuntimeConfig(bus_backend="mem"),
            presentation=outlet,
        )

        await service.start()
        await service.stop()

    asyncio.run(scenario())
