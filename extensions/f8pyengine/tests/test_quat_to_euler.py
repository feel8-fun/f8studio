import math
import os
import sys
import unittest
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.testing import buffer_input  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.quat_to_euler import QuatToEulerRuntimeNode, register_operator  # noqa: E402


class QuatToEulerTests(unittest.IsolatedAsyncioTestCase):
    async def _build_node(self, *, state_values: dict[str, Any] | None = None) -> tuple[Any, QuatToEulerRuntimeNode]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="q2e1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=QuatToEulerRuntimeNode.SPEC.operatorClass,
            stateFields=list(QuatToEulerRuntimeNode.SPEC.stateFields or []),
            stateValues=dict(state_values or {}),
            dataInPorts=list(QuatToEulerRuntimeNode.SPEC.dataInPorts or []),
            dataOutPorts=list(QuatToEulerRuntimeNode.SPEC.dataOutPorts or []),
        )
        graph = F8RuntimeGraph(graphId="g_q2e", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)
        node = bus.get_node("q2e1")
        self.assertIsInstance(node, QuatToEulerRuntimeNode)
        assert isinstance(node, QuatToEulerRuntimeNode)
        return bus, node

    async def _step(self, bus: Any, node: QuatToEulerRuntimeNode, *, quat: list[float], idx: int) -> Any:
        buffer_input(bus, "q2e1", "quat", quat, ts_ms=idx, edge=None, ctx_id=None)
        return await node.compute_output("euler", ctx_id=idx)

    async def test_identity_quaternion_is_zero(self) -> None:
        bus, node = await self._build_node()
        out = await self._step(bus, node, quat=[1.0, 0.0, 0.0, 0.0], idx=1)
        self.assertIsInstance(out, list)
        assert isinstance(out, list)
        self.assertAlmostEqual(out[0], 0.0, places=6)
        self.assertAlmostEqual(out[1], 0.0, places=6)
        self.assertAlmostEqual(out[2], 0.0, places=6)

    async def test_zyx_z_rotation_90_deg(self) -> None:
        bus, node = await self._build_node(state_values={"order": "ZYX", "degrees": True})
        c = math.sqrt(0.5)
        out = await self._step(bus, node, quat=[c, 0.0, 0.0, c], idx=2)
        self.assertAlmostEqual(out[0], 0.0, places=4)
        self.assertAlmostEqual(out[1], 0.0, places=4)
        self.assertAlmostEqual(out[2], 90.0, places=3)

    async def test_order_change_takes_effect(self) -> None:
        bus, node = await self._build_node(state_values={"order": "ZYX", "degrees": True})
        out_zyx = await self._step(bus, node, quat=[0.8, 0.2, 0.4, 0.38], idx=3)
        await node.on_state("order", "XYZ")
        out_xyz = await self._step(bus, node, quat=[0.8, 0.2, 0.4, 0.38], idx=4)
        self.assertNotEqual([round(v, 4) for v in out_zyx], [round(v, 4) for v in out_xyz])

    async def test_radians_mode(self) -> None:
        bus, node = await self._build_node(state_values={"order": "ZYX", "degrees": False})
        c = math.sqrt(0.5)
        out = await self._step(bus, node, quat=[c, 0.0, 0.0, c], idx=5)
        self.assertAlmostEqual(out[2], math.pi / 2.0, places=5)

    async def test_invalid_input_keeps_last_output(self) -> None:
        bus, node = await self._build_node()
        good = await self._step(bus, node, quat=[1.0, 0.0, 0.0, 0.0], idx=6)
        bad = await self._step(bus, node, quat=[1.0, 0.0, 0.0], idx=7)
        self.assertEqual(bad, good)


if __name__ == "__main__":
    unittest.main()
