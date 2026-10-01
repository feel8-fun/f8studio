import asyncio
import os
import sys
import tempfile
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
from f8pysdk.nodes import OperatorNode  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.replayer import ReplayerRuntimeNode  # noqa: E402
from f8pyengine.operators.state_trigger import StateTriggerRuntimeNode  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402
from f8pyengine.pyengine_service import PyEngineService  # noqa: E402
from f8pyengine.recording import FORMAT_VERSION, RecordingHeader, RecordingWriter, TIME_MODE_OFFSET_FROM_PLAY  # noqa: E402


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


@dataclass
class _RuntimeStub:
    bus: object


class _ProbeExecRuntimeNode(OperatorNode):
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
        self.exec_ids: list[str] = []
        self.delay_s = 0.05

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        _ = in_port
        self.calls += 1
        self.exec_ids.append(str(exec_id))
        await asyncio.sleep(float(self.delay_s))
        return []


class StateTriggerTests(unittest.IsolatedAsyncioTestCase):
    async def _setup_runtime(
        self,
        *,
        trigger_state_values: dict[str, Any] | None = None,
        extra_nodes: list[F8RuntimeNode] | None = None,
        extra_edges: list[F8Edge] | None = None,
        include_probe: bool = True,
    ) -> tuple[object, PyEngineService, _RuntimeStub]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        reg.register_operator_factory(
            SERVICE_CLASS,
            "f8.test_probe_exec",
            lambda node_id, node, initial_state: _ProbeExecRuntimeNode(
                node_id=node_id, node=node, initial_state=initial_state
            ),
            overwrite=True,
        )
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)
        service = PyEngineService()
        runtime = _RuntimeStub(bus=bus)
        await service.setup(runtime)  # type: ignore[arg-type]

        state_trigger = F8RuntimeNode(
            nodeId="trigger",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=StateTriggerRuntimeNode.SPEC.operatorClass,
            stateFields=list(StateTriggerRuntimeNode.SPEC.stateFields or []),
            stateValues={"enabled": True, **dict(trigger_state_values or {})},
            execInPorts=list(StateTriggerRuntimeNode.SPEC.execInPorts or []),
            execOutPorts=list(StateTriggerRuntimeNode.SPEC.execOutPorts or []),
        )
        probe = F8RuntimeNode(
            nodeId="probe",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass="f8.test_probe_exec",
            execInPorts=["exec"],
            execOutPorts=[],
        )
        nodes = [state_trigger, *( [probe] if include_probe else [] ), *(list(extra_nodes or []))]
        edges = [*( [_exec_edge(edge_id="e1", from_node="trigger", from_port="changed", to_node="probe", to_port="exec")] if include_probe else [] ), *(list(extra_edges or []))]
        graph = F8RuntimeGraph(
            graphId="g_state_trigger",
            revision="r1",
            nodes=nodes,
            edges=edges,
        )
        await bus.set_rungraph(graph)  # type: ignore[attr-defined]

        entrypoint_ready = False
        for _ in range(100):
            node = bus.get_node("trigger")
            if isinstance(node, StateTriggerRuntimeNode) and node._entrypoint_ctx is not None:
                entrypoint_ready = True
                break
            await asyncio.sleep(0.01)
        self.assertTrue(entrypoint_ready)

        return bus, service, runtime

    async def _teardown_runtime(self, service: PyEngineService, runtime: _RuntimeStub) -> None:
        await service.teardown(runtime)  # type: ignore[arg-type]

    async def _wait_probe_calls(self, bus: object, *, at_least: int, timeout_s: float = 1.0) -> _ProbeExecRuntimeNode:
        end = asyncio.get_running_loop().time() + timeout_s
        while True:
            node = bus.get_node("probe")
            if isinstance(node, _ProbeExecRuntimeNode) and node.calls >= at_least:
                return node
            if asyncio.get_running_loop().time() >= end:
                self.fail(f"timed out waiting for probe calls >= {at_least}")
            await asyncio.sleep(0.01)

    async def test_value_changed_triggers_changed_exec_once(self) -> None:
        bus, service, runtime = await self._setup_runtime()
        try:
            await bus.publish_state_external("trigger", "value", 1, source="test")  # type: ignore[attr-defined]
            probe = await self._wait_probe_calls(bus, at_least=1, timeout_s=1.2)
            self.assertEqual(probe.calls, 1)
        finally:
            await self._teardown_runtime(service, runtime)

    async def test_same_value_does_not_trigger(self) -> None:
        bus, service, runtime = await self._setup_runtime()
        try:
            await bus.publish_state_external("trigger", "value", "abc", source="test")  # type: ignore[attr-defined]
            probe = await self._wait_probe_calls(bus, at_least=1, timeout_s=1.2)
            await bus.publish_state_external("trigger", "value", "abc", source="test")  # type: ignore[attr-defined]
            await asyncio.sleep(0.25)
            self.assertEqual(probe.calls, 1)
        finally:
            await self._teardown_runtime(service, runtime)

    async def test_disabled_state_suppresses_trigger(self) -> None:
        bus, service, runtime = await self._setup_runtime()
        try:
            await bus.publish_state_external("trigger", "enabled", False, source="test")  # type: ignore[attr-defined]
            await bus.publish_state_external("trigger", "value", 1, source="test")  # type: ignore[attr-defined]
            await asyncio.sleep(0.25)

            node = bus.get_node("probe")
            self.assertIsInstance(node, _ProbeExecRuntimeNode)
            assert isinstance(node, _ProbeExecRuntimeNode)
            self.assertEqual(node.calls, 0)
        finally:
            await self._teardown_runtime(service, runtime)

    async def test_burst_value_updates_coalesce_to_latest_pending_trigger(self) -> None:
        bus, service, runtime = await self._setup_runtime()
        try:
            node = bus.get_node("probe")
            self.assertIsInstance(node, _ProbeExecRuntimeNode)
            assert isinstance(node, _ProbeExecRuntimeNode)
            node.delay_s = 0.08

            tasks: list[asyncio.Task[None]] = []
            for i in range(1, 31):
                task = asyncio.create_task(
                    bus.publish_state_external("trigger", "value", i, source="test")  # type: ignore[attr-defined]
                )
                tasks.append(task)
            await asyncio.gather(*tasks)

            # Wait until call count settles, instead of sleeping a fixed long window.
            end = asyncio.get_running_loop().time() + 2.0
            last_calls = -1
            stable_ticks = 0
            while asyncio.get_running_loop().time() < end:
                current = node.calls
                if current > 0 and current == last_calls:
                    stable_ticks += 1
                    if stable_ticks >= 5:
                        break
                else:
                    stable_ticks = 0
                    last_calls = current
                await asyncio.sleep(0.02)
            self.assertGreater(node.calls, 0)
            self.assertLess(node.calls, 30)
            self.assertEqual(node.exec_ids[-1], "30")
        finally:
            await self._teardown_runtime(service, runtime)

    async def test_initial_value_does_not_trigger_by_default(self) -> None:
        bus, service, runtime = await self._setup_runtime(trigger_state_values={"value": 5})
        try:
            await asyncio.sleep(0.2)
            node = bus.get_node("probe")
            self.assertIsInstance(node, _ProbeExecRuntimeNode)
            assert isinstance(node, _ProbeExecRuntimeNode)
            self.assertEqual(node.calls, 0)
        finally:
            await self._teardown_runtime(service, runtime)

    async def test_fire_on_start_emits_once_when_enabled_and_initial_value_present(self) -> None:
        bus, service, runtime = await self._setup_runtime(trigger_state_values={"value": 5, "fireOnStart": True})
        try:
            probe = await self._wait_probe_calls(bus, at_least=1, timeout_s=1.2)
            self.assertEqual(probe.calls, 1)
        finally:
            await self._teardown_runtime(service, runtime)

    async def test_value_change_can_drive_replayer_play(self) -> None:
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
            writer.write_data_sample(tick_ts_ms=1500, relative_offset_ms=500, data={"outA": 2})
            writer.close()

            replayer = F8RuntimeNode(
                nodeId="rep1",
                serviceId="svcA",
                serviceClass=SERVICE_CLASS,
                operatorClass=ReplayerRuntimeNode.SPEC.operatorClass,
                stateFields=list(ReplayerRuntimeNode.SPEC.stateFields or []),
                stateValues={"path": path, "loop": False, "timeMode": TIME_MODE_OFFSET_FROM_PLAY, "playing": False},
                execInPorts=list(ReplayerRuntimeNode.SPEC.execInPorts or []),
                execOutPorts=list(ReplayerRuntimeNode.SPEC.execOutPorts or []),
                dataInPorts=[],
                dataOutPorts=list(ReplayerRuntimeNode.SPEC.dataOutPorts or []),
            )

            bus, service, runtime = await self._setup_runtime(
                extra_nodes=[replayer],
                extra_edges=[_exec_edge(edge_id="e2", from_node="trigger", from_port="changed", to_node="rep1", to_port="play")],
                include_probe=False,
            )
            try:
                await asyncio.sleep(0.05)
                await bus.publish_state_external("trigger", "value", 1, source="test")  # type: ignore[attr-defined]
                end = asyncio.get_running_loop().time() + 1.2
                while asyncio.get_running_loop().time() < end:
                    playing = (await bus.get_state("rep1", "playing")).value
                    if bool(playing):
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(bool((await bus.get_state("rep1", "playing")).value))
            finally:
                await self._teardown_runtime(service, runtime)
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    unittest.main()
