import asyncio
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.specs import F8DataPortSpec, F8RuntimeGraph, F8RuntimeNode, F8StateAccess, F8StateSpec, any_schema, integer_schema  # noqa: E402
from f8pysdk.specs import F8Edge, F8EdgeKindEnum, F8EdgeStrategyEnum  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.nodes import OperatorNode  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import buffer_input  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402
from f8pysdk.time_utils import now_ms  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.recorder import RecorderRuntimeNode  # noqa: E402
from f8pyengine.operators.replayer import ReplayerRuntimeNode  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402
from f8pyengine.pyengine_service import PyEngineService  # noqa: E402
from f8pyengine.recording import (  # noqa: E402
    FORMAT_VERSION,
    RecordingHeader,
    RecordingReader,
    RecordingWriter,
    TIME_MODE_OFFSET_FROM_PLAY,
    TIME_MODE_RECORDED_EPOCH,
)


_PASSIVE_SINK_OPERATOR_CLASS = "f8.test.passive_sink"
_PULL_PROBE_OPERATOR_CLASS = "f8.test.pull_probe"


class _PassiveSinkRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, object] | None = None) -> None:
        del initial_state
        super().__init__(
            node_id=node_id,
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
            exec_in_ports=list(node.execInPorts or []),
            exec_out_ports=list(node.execOutPorts or []),
        )

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        _ = exec_id
        _ = in_port
        return []


class _PullProbeRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, object] | None = None) -> None:
        del initial_state
        super().__init__(
            node_id=node_id,
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
            exec_in_ports=list(node.execInPorts or []),
            exec_out_ports=list(node.execOutPorts or []),
        )
        self.values: list[Any] = []
        self.exec_ids: list[str] = []

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        _ = in_port
        value = await self.pull("value", ctx_id=exec_id)
        self.values.append(value)
        self.exec_ids.append(str(exec_id))
        return []


@dataclass
class _RuntimeStub:
    bus: object


def _data_edge(*, edge_id: str, from_node: str, from_port: str, to_node: str, to_port: str) -> F8Edge:
    return F8Edge(
        edgeId=edge_id,
        fromServiceId="svcA",
        fromOperatorId=from_node,
        fromPort=from_port,
        toServiceId="svcA",
        toOperatorId=to_node,
        toPort=to_port,
        kind=F8EdgeKindEnum.data,
        strategy=F8EdgeStrategyEnum.latest,
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


def _monitor_error_message(bus: object) -> str:
    snapshot = bus.monitor_collector._build_snapshot(ts_ms=int(now_ms()))
    return str(snapshot.error.currentMessage or "")


class RecorderReplayerTests(unittest.IsolatedAsyncioTestCase):
    async def _build_runtime(
        self,
        *,
        nodes: list[F8RuntimeNode],
        edges: list[F8Edge] | None = None,
    ) -> tuple[ServiceBusHarness, object]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        reg.register_operator_factory(
            SERVICE_CLASS,
            _PASSIVE_SINK_OPERATOR_CLASS,
            lambda node_id, node, initial_state: _PassiveSinkRuntimeNode(
                node_id=node_id,
                node=node,
                initial_state=initial_state,
            ),
            overwrite=True,
        )
        reg.register_operator_factory(
            SERVICE_CLASS,
            _PULL_PROBE_OPERATOR_CLASS,
            lambda node_id, node, initial_state: _PullProbeRuntimeNode(
                node_id=node_id,
                node=node,
                initial_state=initial_state,
            ),
            overwrite=True,
        )
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)
        graph = F8RuntimeGraph(graphId="g_recording", revision="r1", nodes=list(nodes), edges=list(edges or []))
        await bus.set_rungraph(graph)
        return harness, bus

    async def _build_runtime_with_service(
        self,
        *,
        nodes: list[F8RuntimeNode],
        edges: list[F8Edge] | None = None,
    ) -> tuple[ServiceBusHarness, object, PyEngineService, _RuntimeStub]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        reg.register_operator_factory(
            SERVICE_CLASS,
            _PASSIVE_SINK_OPERATOR_CLASS,
            lambda node_id, node, initial_state: _PassiveSinkRuntimeNode(
                node_id=node_id,
                node=node,
                initial_state=initial_state,
            ),
            overwrite=True,
        )
        reg.register_operator_factory(
            SERVICE_CLASS,
            _PULL_PROBE_OPERATOR_CLASS,
            lambda node_id, node, initial_state: _PullProbeRuntimeNode(
                node_id=node_id,
                node=node,
                initial_state=initial_state,
            ),
            overwrite=True,
        )
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)
        service = PyEngineService()
        runtime = _RuntimeStub(bus=bus)
        await service.setup(runtime)  # type: ignore[arg-type]
        graph = F8RuntimeGraph(graphId="g_recording", revision="r1", nodes=list(nodes), edges=list(edges or []))
        await bus.set_rungraph(graph)
        return harness, bus, service, runtime

    def _recorder_node(self, *, path: str, node_id: str = "rec1", data_ports: list[str] | None = None) -> F8RuntimeNode:
        state_fields = list(RecorderRuntimeNode.SPEC.stateFields or [])
        state_fields.append(
            F8StateSpec(
                name="alpha",
                label="Alpha",
                description="Custom sparse state.",
                valueSchema=integer_schema(),
                access=F8StateAccess.rw,
                valueRequired=False,
                showOnNode=True,
            )
        )
        return F8RuntimeNode(
            nodeId=node_id,
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=RecorderRuntimeNode.SPEC.operatorClass,
            stateFields=state_fields,
            stateValues={"path": path, "enabled": True, "append": True},
            execInPorts=["record"],
            execOutPorts=[],
            dataInPorts=[F8DataPortSpec(name=name, description="", valueSchema=any_schema(), definitionProtected=False) for name in list(data_ports or ["a", "b"])],
            dataOutPorts=[],
        )

    def _replayer_node(self, *, path: str, node_id: str = "rep1", time_mode: str = TIME_MODE_OFFSET_FROM_PLAY) -> F8RuntimeNode:
        state_fields = list(ReplayerRuntimeNode.SPEC.stateFields or [])
        state_fields.append(
            F8StateSpec(
                name="alpha",
                label="Alpha",
                description="Replayed sparse state.",
                valueSchema=integer_schema(),
                access=F8StateAccess.rw,
                valueRequired=False,
                showOnNode=True,
            )
        )
        data_out_ports = list(ReplayerRuntimeNode.SPEC.dataOutPorts or [])
        data_out_ports.append(F8DataPortSpec(name="outA", description="", valueSchema=any_schema(), definitionProtected=False))
        return F8RuntimeNode(
            nodeId=node_id,
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=ReplayerRuntimeNode.SPEC.operatorClass,
            stateFields=state_fields,
            stateValues={"path": path, "loop": False, "timeMode": time_mode, "playing": False},
            execInPorts=list(ReplayerRuntimeNode.SPEC.execInPorts or []),
            execOutPorts=list(ReplayerRuntimeNode.SPEC.execOutPorts or []),
            dataInPorts=[],
            dataOutPorts=data_out_ports,
        )

    def _pull_sink_node(self, *, node_id: str = "sink1") -> F8RuntimeNode:
        return F8RuntimeNode(
            nodeId=node_id,
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=_PASSIVE_SINK_OPERATOR_CLASS,
            stateFields=[],
            dataInPorts=[
                F8DataPortSpec(name="outA", description="", valueSchema=any_schema(), definitionProtected=False),
                F8DataPortSpec(name="positionMs", description="", valueSchema=any_schema(), definitionProtected=False),
            ],
            dataOutPorts=[],
            execInPorts=[],
            execOutPorts=[],
        )

    def _pull_probe_node(self, *, node_id: str = "probe1") -> F8RuntimeNode:
        return F8RuntimeNode(
            nodeId=node_id,
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=_PULL_PROBE_OPERATOR_CLASS,
            stateFields=[],
            dataInPorts=[F8DataPortSpec(name="value", description="", valueSchema=any_schema(), definitionProtected=False)],
            dataOutPorts=[],
            execInPorts=["exec"],
            execOutPorts=[],
        )

    async def test_recorder_records_arrival_time_for_data_samples_and_sparse_state_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "capture.f8rec")
            _, bus = await self._build_runtime(nodes=[self._recorder_node(path=path)])
            node = bus.get_node("rec1")
            self.assertIsInstance(node, RecorderRuntimeNode)
            try:
                buffer_input(bus, "rec1", "a", 11, ts_ms=1000, edge=None, ctx_id=1000)
                buffer_input(bus, "rec1", "b", {"x": 2}, ts_ms=1000, edge=None, ctx_id=1000)
                with patch("f8pyengine.operators.recorder.now_ms", side_effect=[5000, 5010]):
                    await node.on_exec(1000, "record")
                    await bus.publish_state_runtime("rec1", "alpha", 7, ts_ms=1010)

                reader = RecordingReader(path)
                events = list(reader.iter_events())
                self.assertEqual(events[0].type, "header")
                self.assertEqual(events[1].type, "data_sample")
                self.assertEqual(events[2].type, "state_change")
                self.assertEqual(events[0].header.created_ts_ms, 5000)
                self.assertEqual(events[1].tick_ts_ms, 5000)
                self.assertEqual(events[1].relative_offset_ms, 0)
                self.assertEqual(events[1].data["a"], 11)
                self.assertEqual(events[1].data["b"], {"x": 2})
                self.assertEqual(events[2].state_ts_ms, 5010)
                self.assertEqual(events[2].relative_offset_ms, 10)
                self.assertEqual(events[2].field, "alpha")
                self.assertEqual(events[2].value, 7)
            finally:
                await node.close()

    async def test_recorder_append_mode_allows_compatible_and_rejects_incompatible_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "append.f8rec")
            _, bus = await self._build_runtime(nodes=[self._recorder_node(path=path, node_id="rec1", data_ports=["a"])])
            node = bus.get_node("rec1")
            self.assertIsInstance(node, RecorderRuntimeNode)
            try:
                buffer_input(bus, "rec1", "a", 1, ts_ms=1000, edge=None, ctx_id=1000)
                await node.on_exec(1000, "record")

                _, bus2 = await self._build_runtime(nodes=[self._recorder_node(path=path, node_id="rec2", data_ports=["a"])])
                node2 = bus2.get_node("rec2")
                self.assertIsInstance(node2, RecorderRuntimeNode)
                try:
                    buffer_input(bus2, "rec2", "a", 2, ts_ms=1010, edge=None, ctx_id=1010)
                    await node2.on_exec(1010, "record")

                    info = RecordingReader(path).read_info()
                    self.assertEqual(info.header.data_ports, ("a",))
                    self.assertEqual(info.event_count, 3)

                    _, bus3 = await self._build_runtime(nodes=[self._recorder_node(path=path, node_id="rec3", data_ports=["a", "b"])])
                    node3 = bus3.get_node("rec3")
                    self.assertIsInstance(node3, RecorderRuntimeNode)
                    try:
                        buffer_input(bus3, "rec3", "a", 3, ts_ms=1020, edge=None, ctx_id=1020)
                        buffer_input(bus3, "rec3", "b", 4, ts_ms=1020, edge=None, ctx_id=1020)
                        await node3.on_exec(1020, "record")
                        self.assertIn("header mismatch", _monitor_error_message(bus3))
                    finally:
                        await node3.close()
                finally:
                    await node2.close()
            finally:
                await node.close()

    async def test_replayer_replays_data_state_and_position_offset_mode(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".f8rec")
        os.close(fd)
        try:
            writer = RecordingWriter(
                path,
                header=RecordingHeader(
                    format_version=FORMAT_VERSION,
                    created_ts_ms=1000,
                    data_ports=("outA",),
                    state_fields=("alpha",),
                ),
                append=False,
            )
            writer.open()
            writer.write_data_sample(tick_ts_ms=1000, relative_offset_ms=0, data={"outA": 1})
            writer.write_state_change(state_ts_ms=1010, relative_offset_ms=10, field="alpha", value=9)
            writer.write_data_sample(tick_ts_ms=1020, relative_offset_ms=20, data={"outA": 2})
            writer.close()

            replayer = self._replayer_node(path=path, time_mode=TIME_MODE_OFFSET_FROM_PLAY)
            sink = self._pull_sink_node()
            _, bus = await self._build_runtime(
                nodes=[replayer, sink],
                edges=[
                    _data_edge(edge_id="d1", from_node="rep1", from_port="outA", to_node="sink1", to_port="outA"),
                    _data_edge(edge_id="d2", from_node="rep1", from_port="positionMs", to_node="sink1", to_port="positionMs"),
                ],
            )
            node = bus.get_node("rep1")
            self.assertIsInstance(node, ReplayerRuntimeNode)
            await asyncio.sleep(0.02)
            self.assertEqual((await bus.get_state("rep1", "durationMs")).value, 20)
            await node.on_exec(1, "play")
            await asyncio.sleep(0.08)

            out_a = await bus.pull_data("sink1", "outA", ctx_id="final")
            pos = await bus.pull_data("sink1", "positionMs", ctx_id="final")
            alpha = (await bus.get_state("rep1", "alpha")).value
            self.assertEqual(out_a, 2)
            self.assertEqual(alpha, 9)
            self.assertGreaterEqual(int(pos), 20)
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    async def test_replayer_supports_recorded_epoch_mode(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".f8rec")
        os.close(fd)
        try:
            now_ms = int(asyncio.get_running_loop().time() * 1000.0)
            wall_now_ms = int(__import__("time").time() * 1000.0)
            start_ts_ms = wall_now_ms + 20
            writer = RecordingWriter(
                path,
                header=RecordingHeader(
                    format_version=FORMAT_VERSION,
                    created_ts_ms=start_ts_ms,
                    data_ports=("outA",),
                    state_fields=(),
                ),
                append=False,
            )
            writer.open()
            writer.write_data_sample(tick_ts_ms=start_ts_ms, relative_offset_ms=0, data={"outA": 3})
            writer.write_data_sample(tick_ts_ms=start_ts_ms + 20, relative_offset_ms=20, data={"outA": 4})
            writer.close()

            replayer = self._replayer_node(path=path, time_mode=TIME_MODE_RECORDED_EPOCH)
            sink = self._pull_sink_node()
            _, bus = await self._build_runtime(
                nodes=[replayer, sink],
                edges=[_data_edge(edge_id="d1", from_node="rep1", from_port="outA", to_node="sink1", to_port="outA")],
            )
            node = bus.get_node("rep1")
            self.assertIsInstance(node, ReplayerRuntimeNode)
            await asyncio.sleep(0.02)
            await node.on_exec(1, "play")
            await asyncio.sleep(0.08)
            out_a = await bus.pull_data("sink1", "outA", ctx_id="epoch")
            self.assertEqual(out_a, 4)
            self.assertGreaterEqual(now_ms, 0)
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    async def test_replayer_emits_sample_exec_with_matching_ctx_for_downstream_pull(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".f8rec")
        os.close(fd)
        try:
            writer = RecordingWriter(
                path,
                header=RecordingHeader(
                    format_version=FORMAT_VERSION,
                    created_ts_ms=1000,
                    data_ports=("outA",),
                    state_fields=(),
                ),
                append=False,
            )
            writer.open()
            writer.write_data_sample(tick_ts_ms=1000, relative_offset_ms=0, data={"outA": 1})
            writer.write_data_sample(tick_ts_ms=1010, relative_offset_ms=10, data={"outA": 2})
            writer.close()

            replayer = self._replayer_node(path=path, time_mode=TIME_MODE_OFFSET_FROM_PLAY)
            probe = self._pull_probe_node()
            _, bus, service, runtime = await self._build_runtime_with_service(
                nodes=[replayer, probe],
                edges=[
                    _data_edge(edge_id="d1", from_node="rep1", from_port="outA", to_node="probe1", to_port="value"),
                    _exec_edge(edge_id="e1", from_node="rep1", from_port="sample", to_node="probe1", to_port="exec"),
                ],
            )
            try:
                node = bus.get_node("rep1")
                self.assertIsInstance(node, ReplayerRuntimeNode)
                pull_probe = bus.get_node("probe1")
                self.assertIsInstance(pull_probe, _PullProbeRuntimeNode)
                assert isinstance(pull_probe, _PullProbeRuntimeNode)

                await asyncio.sleep(0.02)
                await node.on_exec(1, "play")

                end = asyncio.get_running_loop().time() + 1.0
                while asyncio.get_running_loop().time() < end:
                    if len(pull_probe.values) >= 2:
                        break
                    await asyncio.sleep(0.01)

                self.assertEqual(pull_probe.values, [1, 2])
                self.assertEqual(len(pull_probe.exec_ids), 2)
                self.assertNotEqual(pull_probe.exec_ids[0], pull_probe.exec_ids[1])
            finally:
                await service.teardown(runtime)  # type: ignore[arg-type]
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    async def test_replayer_emits_started_exec_when_playback_starts_from_state(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".f8rec")
        os.close(fd)
        try:
            writer = RecordingWriter(
                path,
                header=RecordingHeader(
                    format_version=FORMAT_VERSION,
                    created_ts_ms=1000,
                    data_ports=("outA",),
                    state_fields=(),
                ),
                append=False,
            )
            writer.open()
            writer.write_data_sample(tick_ts_ms=1000, relative_offset_ms=0, data={"outA": 1})
            writer.close()

            replayer = self._replayer_node(path=path, time_mode=TIME_MODE_OFFSET_FROM_PLAY)
            probe = self._pull_probe_node()
            _, bus, service, runtime = await self._build_runtime_with_service(
                nodes=[replayer, probe],
                edges=[_exec_edge(edge_id="e1", from_node="rep1", from_port="started", to_node="probe1", to_port="exec")],
            )
            try:
                pull_probe = bus.get_node("probe1")
                self.assertIsInstance(pull_probe, _PullProbeRuntimeNode)
                assert isinstance(pull_probe, _PullProbeRuntimeNode)

                await asyncio.sleep(0.02)
                await bus.publish_state_runtime("rep1", "playing", True, ts_ms=1)

                end = asyncio.get_running_loop().time() + 1.0
                while asyncio.get_running_loop().time() < end:
                    if pull_probe.exec_ids:
                        break
                    await asyncio.sleep(0.01)

                self.assertEqual(len(pull_probe.exec_ids), 1)
            finally:
                await service.teardown(runtime)  # type: ignore[arg-type]
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    async def test_replayer_bad_file_sets_last_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "broken.f8rec")
            with open(path, "wb") as fh:
                fh.write(b"not-a-valid-recording")

            _, bus = await self._build_runtime(nodes=[self._replayer_node(path=path)])
            await bus.publish_state_runtime("rep1", "path", path, ts_ms=1)
            await asyncio.sleep(0.05)
            loaded = (await bus.get_state("rep1", "loaded")).value
            self.assertEqual(bool(loaded), False)
            snapshot = bus.monitor_collector._build_snapshot(ts_ms=int(now_ms()))
            self.assertTrue(str(snapshot.error.lastMessage or ""))

    def test_position_ms_is_not_a_state_field(self) -> None:
        state_names = [str(field.name) for field in list(ReplayerRuntimeNode.SPEC.stateFields or [])]
        data_names = [str(port.name) for port in list(ReplayerRuntimeNode.SPEC.dataOutPorts or [])]
        self.assertNotIn("positionMs", state_names)
        self.assertIn("positionMs", data_names)

    def test_recorder_sample_counters_are_not_state_fields(self) -> None:
        state_names = [str(field.name) for field in list(RecorderRuntimeNode.SPEC.stateFields or [])]
        self.assertNotIn("sampleCount", state_names)
        self.assertNotIn("stateEventCount", state_names)


def test_recorder_and_replayer_paths_are_publish_redacted() -> None:
    recorder_path = next(field for field in list(RecorderRuntimeNode.SPEC.stateFields or []) if str(field.name) == "path")
    replayer_path = next(field for field in list(ReplayerRuntimeNode.SPEC.stateFields or []) if str(field.name) == "path")

    assert bool(recorder_path.redactOnPublish) is True
    assert bool(replayer_path.redactOnPublish) is True


if __name__ == "__main__":
    unittest.main()
