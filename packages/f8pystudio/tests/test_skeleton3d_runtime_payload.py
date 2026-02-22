import asyncio
import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any


PKG_STUDIO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_STUDIO, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)


from f8pystudio.operators.skeleton3d import PyStudioSkeleton3DRuntimeNode  # noqa: E402
from f8pystudio.ui_bus import UiCommand, set_ui_command_sink  # noqa: E402


@dataclass(frozen=True)
class _FakePort:
    name: str


@dataclass(frozen=True)
class _FakeState:
    name: str


@dataclass(frozen=True)
class _FakeNode:
    dataInPorts: list[_FakePort]
    dataOutPorts: list[_FakePort]
    stateFields: list[_FakeState]


def _make_runtime(*, initial_state: dict[str, Any] | None = None) -> PyStudioSkeleton3DRuntimeNode:
    fake = _FakeNode(dataInPorts=[_FakePort(name="skeletons")], dataOutPorts=[], stateFields=[])
    return PyStudioSkeleton3DRuntimeNode(node_id="n1", node=fake, initial_state=initial_state or {})


class Skeleton3DRuntimePayloadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        set_ui_command_sink(None)

    async def test_accepts_single_dict_and_list_payload(self) -> None:
        cmds: list[UiCommand] = []
        set_ui_command_sink(lambda c: cmds.append(c))

        n = _make_runtime(initial_state={"throttleMs": 0})
        single = {
            "modelName": "Alice",
            "skeletonProtocol": "mediapipe_pose_33",
            "bones": [
                {"name": "hip", "pos": [0.0, 1.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]},
                {"name": "head", "pos": [0.0, 1.7, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]},
            ],
        }
        await n.on_data("skeletons", single, ts_ms=1000)
        self.assertGreaterEqual(len(cmds), 1)
        first = cmds[-1]
        self.assertEqual(first.command, "skeleton3d.set")
        self.assertEqual(len(first.payload.get("people") or []), 1)
        person0 = (first.payload.get("people") or [])[0]
        self.assertEqual(person0.get("name"), "Alice")
        self.assertEqual(person0.get("skeletonProtocol"), "mediapipe_pose_33")
        self.assertIsInstance(person0.get("skeletonEdges"), list)
        self.assertEqual(len(person0.get("nodes") or []), 2)
        node0 = (person0.get("nodes") or [])[0]
        self.assertEqual(int(node0.get("index")), 0)
        self.assertIsNotNone(person0.get("bbox"))

        second_person = {
            "modelName": "Bob",
            "bones": [{"name": "hip", "pos": [2.0, 0.9, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]}],
        }
        await n.on_data("skeletons", [single, second_person], ts_ms=1010)
        second = cmds[-1]
        self.assertEqual(len(second.payload.get("people") or []), 2)
        await n.close()

    async def test_applies_limits(self) -> None:
        cmds: list[UiCommand] = []
        set_ui_command_sink(lambda c: cmds.append(c))

        n = _make_runtime(initial_state={"throttleMs": 0, "maxPeople": 1, "maxBonesPerPerson": 1})
        payload = [
            {
                "modelName": "Alice",
                "bones": [
                    {"name": "hip", "pos": [0.0, 1.0, 0.0]},
                    {"name": "head", "pos": [0.0, 1.7, 0.0]},
                ],
            },
            {
                "modelName": "Bob",
                "bones": [
                    {"name": "hip", "pos": [2.0, 0.9, 0.0]},
                    {"name": "head", "pos": [2.0, 1.4, 0.0]},
                ],
            },
        ]
        await n.on_data("skeletons", payload, ts_ms=1000)
        self.assertGreaterEqual(len(cmds), 1)
        out = cmds[-1].payload
        people = out.get("people") or []
        self.assertEqual(len(people), 1)
        nodes = people[0].get("nodes") or []
        self.assertEqual(len(nodes), 1)
        await n.close()

    async def test_throttle_and_state_change_refresh(self) -> None:
        cmds: list[UiCommand] = []
        set_ui_command_sink(lambda c: cmds.append(c))

        n = _make_runtime(initial_state={"throttleMs": 100})
        payload = {
            "modelName": "Alice",
            "bones": [{"name": "hip", "pos": [0.0, 1.0, 0.0]}],
        }
        await n.on_data("skeletons", payload, ts_ms=1000)
        self.assertEqual(len(cmds), 1)

        await n.on_data("skeletons", payload, ts_ms=1010)
        self.assertEqual(len(cmds), 1)
        await asyncio.sleep(0.13)
        self.assertGreaterEqual(len(cmds), 2)

        await n.on_state("showPersonNames", True, ts_ms=1200)
        await asyncio.sleep(0.13)
        self.assertGreaterEqual(len(cmds), 3)
        last = cmds[-1].payload
        flags = dict(last.get("renderFlags") or {})
        self.assertTrue(bool(flags.get("showPersonNames")))

        await n.on_state("showSkeletonLines", False, ts_ms=1250)
        await asyncio.sleep(0.13)
        self.assertGreaterEqual(len(cmds), 4)
        last = cmds[-1].payload
        flags = dict(last.get("renderFlags") or {})
        self.assertFalse(bool(flags.get("showSkeletonLines")))

        await n.on_state("markerScale", 2.5, ts_ms=1300)
        await asyncio.sleep(0.13)
        self.assertGreaterEqual(len(cmds), 5)
        last = cmds[-1].payload
        flags = dict(last.get("renderFlags") or {})
        self.assertAlmostEqual(float(flags.get("markerScale") or 0.0), 2.5, places=4)
        await n.close()

    async def test_unknown_protocol_does_not_emit_edges(self) -> None:
        cmds: list[UiCommand] = []
        set_ui_command_sink(lambda c: cmds.append(c))

        n = _make_runtime(initial_state={"throttleMs": 0})
        payload = {
            "modelName": "Alice",
            "skeletonProtocol": "custom_proto",
            "bones": [{"name": "hip", "pos": [0.0, 1.0, 0.0]}],
        }
        await n.on_data("skeletons", payload, ts_ms=1000)
        self.assertGreaterEqual(len(cmds), 1)
        out = cmds[-1].payload
        people = out.get("people") or []
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0].get("skeletonProtocol"), "custom_proto")
        self.assertIsNone(people[0].get("skeletonEdges"))
        await n.close()


if __name__ == "__main__":
    unittest.main()
