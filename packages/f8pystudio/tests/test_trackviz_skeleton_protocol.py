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


from f8pystudio.operators.trackviz import PyStudioTrackVizRuntimeNode  # noqa: E402
from f8pystudio.skeleton_protocols import skeleton_edges_for_protocol  # noqa: E402
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


def _make_runtime(*, initial_state: dict[str, Any] | None = None) -> PyStudioTrackVizRuntimeNode:
    fake = _FakeNode(dataInPorts=[_FakePort(name="detections")], dataOutPorts=[], stateFields=[])
    return PyStudioTrackVizRuntimeNode(node_id="n_trackviz", node=fake, initial_state=initial_state or {})


class TrackVizSkeletonProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        set_ui_command_sink(None)

    async def test_detection_protocol_override_and_payload_fallback(self) -> None:
        cmds: list[UiCommand] = []
        set_ui_command_sink(lambda c: cmds.append(c))
        node = _make_runtime(initial_state={"throttleMs": 0})

        payload = {
            "tsMs": 1000,
            "width": 1280,
            "height": 720,
            "skeletonProtocol": "coco17",
            "detections": [
                {
                    "id": 1,
                    "bbox": [100, 100, 200, 300],
                    "keypoints": [{"x": 120.0, "y": 140.0, "score": 0.9}],
                    "skeletonProtocol": "mediapipe_pose_33",
                },
                {
                    "id": 2,
                    "bbox": [300, 200, 360, 380],
                    "keypoints": [{"x": 320.0, "y": 250.0, "score": 0.8}],
                },
            ],
        }
        await node.on_data("detections", payload, ts_ms=1000)
        await asyncio.sleep(0)
        self.assertGreaterEqual(len(cmds), 1)

        tracks = cmds[-1].payload["tracks"]
        by_id = {int(t["id"]): t for t in tracks}
        hist1 = by_id[1]["history"][-1]
        hist2 = by_id[2]["history"][-1]
        self.assertEqual(hist1.get("skeletonProtocol"), "mediapipe_pose_33")
        self.assertEqual(hist2.get("skeletonProtocol"), "coco17")
        await node.close()

    async def test_missing_protocol_is_not_injected(self) -> None:
        cmds: list[UiCommand] = []
        set_ui_command_sink(lambda c: cmds.append(c))
        node = _make_runtime(initial_state={"throttleMs": 0})

        payload = {
            "tsMs": 1000,
            "width": 640,
            "height": 480,
            "detections": [
                {
                    "id": 1,
                    "bbox": [10, 20, 100, 200],
                    "keypoints": [{"x": 20.0, "y": 30.0, "score": 0.5}],
                }
            ],
        }
        await node.on_data("detections", payload, ts_ms=1000)
        await asyncio.sleep(0)
        self.assertGreaterEqual(len(cmds), 1)
        sample = cmds[-1].payload["tracks"][0]["history"][-1]
        self.assertNotIn("skeletonProtocol", sample)
        await node.close()

    def test_edge_lookup(self) -> None:
        coco = skeleton_edges_for_protocol("coco17")
        self.assertIsNotNone(coco)
        self.assertTrue(bool(coco))

        mp33 = skeleton_edges_for_protocol("mediapipe_pose_33")
        self.assertIsNotNone(mp33)
        self.assertTrue(bool(mp33))

        human36m = skeleton_edges_for_protocol("human36m_17")
        self.assertIsNotNone(human36m)
        self.assertTrue(bool(human36m))

        unknown = skeleton_edges_for_protocol("unknown_protocol")
        self.assertIsNone(unknown)
        missing = skeleton_edges_for_protocol("")
        self.assertIsNone(missing)


if __name__ == "__main__":
    unittest.main()
