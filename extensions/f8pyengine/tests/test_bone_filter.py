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
from f8pyengine.operators.bone_filter import (  # noqa: E402
    BoneFilterRuntimeNode,
    _Bone,
    register_operator,
)


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _quat_conjugate(q: list[float]) -> list[float]:
    return [q[0], -q[1], -q[2], -q[3]]


def _quat_rotate(q: list[float], v: list[float]) -> list[float]:
    p = [0.0, v[0], v[1], v[2]]
    qp = _quat_mul(q, p)
    qpq = _quat_mul(qp, _quat_conjugate(q))
    return [qpq[1], qpq[2], qpq[3]]


def _compose_pose(filtered: dict[str, list[float]], relative: dict[str, list[float]]) -> dict[str, list[float]]:
    f_pos = filtered["pos"]
    f_rot = filtered["rot"]
    r_pos = relative["pos"]
    r_rot = relative["rot"]
    world_delta = _quat_rotate(f_rot, r_pos)
    return {
        "pos": [
            f_pos[0] + world_delta[0],
            f_pos[1] + world_delta[1],
            f_pos[2] + world_delta[2],
        ],
        "rot": _quat_mul(f_rot, r_rot),
    }


class BoneFilterTests(unittest.IsolatedAsyncioTestCase):
    async def _build_node(self, *, state_values: dict[str, Any] | None = None) -> tuple[Any, BoneFilterRuntimeNode]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="bone1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=BoneFilterRuntimeNode.SPEC.operatorClass,
            stateFields=list(BoneFilterRuntimeNode.SPEC.stateFields or []),
            stateValues=dict(state_values or {}),
            dataInPorts=list(BoneFilterRuntimeNode.SPEC.dataInPorts or []),
            dataOutPorts=list(BoneFilterRuntimeNode.SPEC.dataOutPorts or []),
        )
        graph = F8RuntimeGraph(graphId="g_bone", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("bone1")
        self.assertIsInstance(node, BoneFilterRuntimeNode)
        assert isinstance(node, BoneFilterRuntimeNode)
        return bus, node

    async def _step(self, bus: Any, node: BoneFilterRuntimeNode, *, bone: dict[str, Any], idx: int, port: str) -> Any:
        buffer_input(bus, "bone1", "bone", bone, ts_ms=idx, edge=None, ctx_id=None)
        return await node.compute_output(port, ctx_id=idx)

    async def test_relative_pose_composition_consistency(self) -> None:
        bus, node = await self._build_node(state_values={"filter_type": "EMA", "ema_alpha": 0.5, "jumpEnabled": False})
        _ = await self._step(bus, node, bone={"pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]}, idx=1, port="filtered")
        raw = {"pos": [1.0, 0.0, 0.0], "rot": [0.9238795, 0.0, 0.0, 0.3826834]}
        filtered = await self._step(bus, node, bone=raw, idx=2, port="filtered")
        relative = await node.compute_output("relative", ctx_id=2)
        self.assertIsInstance(filtered, dict)
        self.assertIsInstance(relative, dict)
        assert isinstance(filtered, dict)
        assert isinstance(relative, dict)

        rebuilt = _compose_pose(filtered, relative)
        self.assertAlmostEqual(rebuilt["pos"][0], raw["pos"][0], places=4)
        self.assertAlmostEqual(rebuilt["pos"][1], raw["pos"][1], places=4)
        self.assertAlmostEqual(rebuilt["pos"][2], raw["pos"][2], places=4)
        dot = abs(sum(rebuilt["rot"][i] * raw["rot"][i] for i in range(4)))
        self.assertGreater(dot, 0.999)

    async def test_relative_pos_is_local_space(self) -> None:
        _bus, node = await self._build_node()
        filtered = _Bone(pos=(0.0, 0.0, 0.0), rot=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))
        raw = _Bone(pos=(1.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0))
        relative = node._compute_relative(filtered, raw)
        self.assertAlmostEqual(relative.pos[0], 0.0, places=4)
        self.assertAlmostEqual(relative.pos[1], -1.0, places=4)
        self.assertAlmostEqual(relative.pos[2], 0.0, places=4)

    async def test_jump_single_outlier_not_triggered(self) -> None:
        bus, node = await self._build_node(
            state_values={
                "filter_type": "EMA",
                "ema_alpha": 0.4,
                "jumpEnabled": True,
                "jumpPosThreshold": 0.5,
                "jumpConsecutiveFrames": 3,
                "jumpCooldownFrames": 4,
            }
        )
        for idx in range(1, 12):
            bone = {"pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]}
            await self._step(bus, node, bone=bone, idx=idx, port="filtered")
        outlier = {"pos": [5.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]}
        await self._step(bus, node, bone=outlier, idx=20, port="filtered")
        baseline = {"pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]}
        await self._step(bus, node, bone=baseline, idx=21, port="filtered")
        self.assertEqual(node._jump_count, 0)

    async def test_jump_cooldown_blocks_retrigger(self) -> None:
        bus, node = await self._build_node(
            state_values={
                "filter_type": "EMA",
                "ema_alpha": 0.4,
                "jumpEnabled": True,
                "jumpPosThreshold": 0.2,
                "jumpConsecutiveFrames": 2,
                "jumpCooldownFrames": 5,
            }
        )
        for idx in range(1, 8):
            baseline = {"pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]}
            await self._step(bus, node, bone=baseline, idx=idx, port="filtered")

        for idx in range(10, 12):
            jump_a = {"pos": [3.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]}
            await self._step(bus, node, bone=jump_a, idx=idx, port="filtered")
        self.assertEqual(node._jump_count, 1)

        for idx in range(12, 15):
            jump_b = {"pos": [6.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]}
            await self._step(bus, node, bone=jump_b, idx=idx, port="filtered")
        self.assertEqual(node._jump_count, 1)

    async def test_invalid_input_keeps_last_output(self) -> None:
        bus, node = await self._build_node(state_values={"jumpEnabled": False})
        valid = {"pos": [0.2, 0.1, -0.3], "rot": [1.0, 0.0, 0.0, 0.0]}
        filtered_ok = await self._step(bus, node, bone=valid, idx=1, port="filtered")
        filtered_bad = await self._step(bus, node, bone={"pos": [1.0, 2.0]}, idx=2, port="filtered")
        self.assertEqual(filtered_bad, filtered_ok)

    async def test_runtime_state_updates_apply(self) -> None:
        bus, node = await self._build_node()
        await node.on_state("filter_type", "DEMA")
        await node.on_state("dema_alpha", 0.2)
        await node.on_state("jumpPosThreshold", 0.6)
        await node.on_state("jumpConsecutiveFrames", 4)
        await node.on_state("jumpCooldownFrames", 7)

        self.assertEqual(node._filter_type, "DEMA")
        self.assertAlmostEqual(node._dema_alpha, 0.2)
        self.assertAlmostEqual(node._jump_pos_threshold, 0.6)
        self.assertEqual(node._jump_consecutive_frames, 4)
        self.assertEqual(node._jump_cooldown_frames, 7)

        out = await self._step(
            bus,
            node,
            bone={"pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]},
            idx=10,
            port="relative",
        )
        self.assertIsInstance(out, dict)


if __name__ == "__main__":
    unittest.main()
