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
from f8pyengine.operators.bone_selector import (  # noqa: E402
    BoneSelectorRuntimeNode,
    register_operator,
)


def _skeleton_payload(bones: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "skeleton_binary",
        "modelName": "VMC",
        "timestampMs": 1,
        "schema": "f8.skeleton.v1",
        "skeletonProtocol": "unity_humanoid",
        "boneCount": len(bones),
        "bones": list(bones),
        "trailer": None,
    }


class BoneSelectorTests(unittest.IsolatedAsyncioTestCase):
    async def _build_node(self, *, state_values: dict[str, Any] | None = None) -> tuple[Any, BoneSelectorRuntimeNode]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="selector1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=BoneSelectorRuntimeNode.SPEC.operatorClass,
            stateFields=list(BoneSelectorRuntimeNode.SPEC.stateFields or []),
            stateValues=dict(state_values or {}),
            dataInPorts=list(BoneSelectorRuntimeNode.SPEC.dataInPorts or []),
            dataOutPorts=list(BoneSelectorRuntimeNode.SPEC.dataOutPorts or []),
        )
        graph = F8RuntimeGraph(graphId="g_bone_selector", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("selector1")
        self.assertIsInstance(node, BoneSelectorRuntimeNode)
        assert isinstance(node, BoneSelectorRuntimeNode)
        return bus, node

    async def test_updates_available_bones_and_outputs_first_when_target_empty(self) -> None:
        bus, node = await self._build_node(state_values={"target": ""})
        payload = _skeleton_payload(
            bones=[
                {"name": "Hips", "pos": [1.0, 2.0, 3.0], "rot": [1.0, 0.0, 0.0, 0.0]},
                {"name": "Spine", "pos": [4.0, 5.0, 6.0], "rot": [1.0, 0.0, 0.0, 0.0]},
            ]
        )
        buffer_input(bus, "selector1", "skeleton", payload, ts_ms=1, edge=None, ctx_id=1)
        out = await node.compute_output("bone", ctx_id=1)

        self.assertEqual(out["name"], "Hips")
        self.assertEqual([float(v) for v in out["pos"]], [1.0, 2.0, 3.0])

        available = await bus.get_state("selector1", "availableBones")
        self.assertTrue(available.found)
        self.assertEqual(list(available.value or []), ["Hips", "Spine"])

        target = await bus.get_state("selector1", "target")
        self.assertTrue(target.found)
        self.assertEqual(str(target.value or ""), "Hips")

    async def test_target_selects_matching_bone(self) -> None:
        bus, node = await self._build_node(state_values={"target": "Spine"})
        payload = _skeleton_payload(
            bones=[
                {"name": "Hips", "pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]},
                {"name": "Spine", "pos": [7.0, 8.0, 9.0], "rot": [0.9, 0.1, 0.0, 0.0]},
            ]
        )
        buffer_input(bus, "selector1", "skeleton", payload, ts_ms=2, edge=None, ctx_id=2)
        out = await node.compute_output("bone", ctx_id=2)

        self.assertEqual(out["name"], "Spine")
        self.assertEqual([round(float(v), 3) for v in out["pos"]], [7.0, 8.0, 9.0])

    async def test_missing_target_auto_falls_back_to_first_bone(self) -> None:
        bus, node = await self._build_node(state_values={"target": "Spine"})
        payload = _skeleton_payload(
            bones=[
                {"name": "Head", "pos": [10.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]},
            ]
        )
        buffer_input(bus, "selector1", "skeleton", payload, ts_ms=3, edge=None, ctx_id=3)
        out = await node.compute_output("bone", ctx_id=3)

        self.assertEqual(out["name"], "Head")
        target = await bus.get_state("selector1", "target")
        self.assertTrue(target.found)
        self.assertEqual(str(target.value or ""), "Head")

    async def test_empty_or_invalid_skeleton_clears_available_and_outputs_none(self) -> None:
        bus, node = await self._build_node(state_values={"target": "Hips"})
        buffer_input(bus, "selector1", "skeleton", {"bones": []}, ts_ms=4, edge=None, ctx_id=4)
        out = await node.compute_output("bone", ctx_id=4)
        self.assertIsNone(out)

        available = await bus.get_state("selector1", "availableBones")
        self.assertTrue(available.found)
        self.assertEqual(list(available.value or []), [])

        target = await bus.get_state("selector1", "target")
        self.assertTrue(target.found)
        self.assertEqual(str(target.value or ""), "")


if __name__ == "__main__":
    unittest.main()
