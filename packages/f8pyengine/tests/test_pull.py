import asyncio
import os
import sys
import unittest
from typing import Any


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.generated import F8DataPortSpec, F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry  # noqa: E402
from f8pysdk.schema_helpers import any_schema  # noqa: E402
from f8pysdk.service_host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.pull import PullRuntimeNode, register_operator  # noqa: E402


class PullNodeTests(unittest.IsolatedAsyncioTestCase):
    async def _build_bus_and_host(self) -> tuple[Any, ServiceHost]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        host = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)
        return bus, host

    async def _install_graph(self, bus: Any, *, enabled: bool, hz: int) -> None:
        op = F8RuntimeNode(
            nodeId="pull1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=PullRuntimeNode.SPEC.operatorClass,
            stateFields=list(PullRuntimeNode.SPEC.stateFields or []),
            stateValues={
                "autoTriggerEnabled": bool(enabled),
                "autoTriggerHz": int(hz),
            },
            dataInPorts=[
                F8DataPortSpec(name="value", description="", valueSchema=any_schema(), required=False),
            ],
            dataOutPorts=[],
            execInPorts=["exec"],
            execOutPorts=[],
        )
        graph = F8RuntimeGraph(graphId="g_pull", revision="1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

    async def test_auto_trigger_periodically_pulls_inputs(self) -> None:
        bus, _host = await self._build_bus_and_host()
        await self._install_graph(bus, enabled=True, hz=30)

        node = bus.get_node("pull1")
        self.assertIsInstance(node, PullRuntimeNode)
        assert isinstance(node, PullRuntimeNode)

        count = 0

        async def _count_pull(_port: str, *, ctx_id: str | int | None = None) -> Any:
            nonlocal count
            _ = ctx_id
            count += 1
            return None

        node.pull = _count_pull  # type: ignore[method-assign]
        await asyncio.sleep(0.22)
        self.assertGreaterEqual(count, 2)

    async def test_auto_trigger_disabled_does_not_pull(self) -> None:
        bus, _host = await self._build_bus_and_host()
        await self._install_graph(bus, enabled=False, hz=30)

        node = bus.get_node("pull1")
        self.assertIsInstance(node, PullRuntimeNode)
        assert isinstance(node, PullRuntimeNode)

        count = 0

        async def _count_pull(_port: str, *, ctx_id: str | int | None = None) -> Any:
            nonlocal count
            _ = ctx_id
            count += 1
            return None

        node.pull = _count_pull  # type: ignore[method-assign]
        await asyncio.sleep(0.2)
        self.assertEqual(count, 0)

    async def test_lifecycle_pause_and_resume_periodic_pull(self) -> None:
        bus, _host = await self._build_bus_and_host()
        await self._install_graph(bus, enabled=True, hz=20)

        node = bus.get_node("pull1")
        self.assertIsInstance(node, PullRuntimeNode)
        assert isinstance(node, PullRuntimeNode)
        self.assertIsNotNone(node._task)
        assert node._task is not None
        self.assertFalse(node._task.done())

        await node.on_lifecycle(False, {"case": "pause"})
        await asyncio.sleep(0.05)
        self.assertIsNone(node._task)

        await node.on_lifecycle(True, {"case": "resume"})
        await asyncio.sleep(0.05)
        self.assertIsNotNone(node._task)
        assert node._task is not None
        self.assertFalse(node._task.done())

    async def test_spec_is_hidden_and_has_no_exec_inputs(self) -> None:
        spec = PullRuntimeNode.SPEC
        tags = [str(t) for t in list(spec.tags or [])]
        self.assertIn("__hidden__", tags)
        self.assertEqual(list(spec.execInPorts or []), [])


if __name__ == "__main__":
    unittest.main()
