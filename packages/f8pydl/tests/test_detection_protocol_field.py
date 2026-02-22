import os
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_PYDL, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)


from f8pydl.service_node import (  # noqa: E402
    OnnxVisionServiceNode,
)


@dataclass(frozen=True)
class _FakeKeypoint:
    x: float
    y: float
    score: float | None


@dataclass(frozen=True)
class _FakeDetection:
    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]
    keypoints: list[_FakeKeypoint] | None = None
    obb: list[tuple[float, float]] | None = None


class _PayloadBuilder:
    _build_detection_payload = OnnxVisionServiceNode._build_detection_payload

    def __init__(self, *, skeleton_protocol: str) -> None:
        self._service_task = "humandetector"
        self._model = SimpleNamespace(model_id="demo_model", task="yolo_pose", skeleton_protocol=skeleton_protocol)

    def _apply_detection_filters(self, detections: list[Any]) -> list[Any]:
        return list(detections)


class DetectionProtocolFieldTests(unittest.TestCase):
    def test_protocol_emits_coco17(self) -> None:
        builder = _PayloadBuilder(skeleton_protocol="coco17")
        detections = [
            _FakeDetection(
                cls="person",
                conf=0.95,
                xyxy=(10.0, 20.0, 80.0, 160.0),
                keypoints=[_FakeKeypoint(x=20.0, y=30.0, score=0.8)],
                obb=[],
            )
        ]
        payload = builder._build_detection_payload(width=1920, height=1080, frame_id=10, ts_ms=1234, detections=detections)
        self.assertEqual(payload["skeletonProtocol"], "coco17")
        self.assertEqual(payload["detections"][0]["skeletonProtocol"], "coco17")

    def test_protocol_emits_none(self) -> None:
        builder = _PayloadBuilder(skeleton_protocol="none")
        detections = [
            _FakeDetection(
                cls="person",
                conf=0.9,
                xyxy=(10.0, 20.0, 80.0, 160.0),
                keypoints=[],
                obb=[],
            )
        ]
        payload = builder._build_detection_payload(width=1920, height=1080, frame_id=10, ts_ms=1234, detections=detections)
        self.assertEqual(payload["skeletonProtocol"], "none")
        self.assertEqual(payload["detections"][0]["skeletonProtocol"], "none")

    def test_protocol_emits_custom_string(self) -> None:
        builder = _PayloadBuilder(skeleton_protocol="my_custom_19")
        detections = [
            _FakeDetection(
                cls="person",
                conf=0.9,
                xyxy=(10.0, 20.0, 80.0, 160.0),
                keypoints=[],
                obb=[],
            )
        ]
        payload = builder._build_detection_payload(width=1920, height=1080, frame_id=10, ts_ms=1234, detections=detections)
        self.assertEqual(payload["skeletonProtocol"], "my_custom_19")
        self.assertEqual(payload["detections"][0]["skeletonProtocol"], "my_custom_19")


if __name__ == "__main__":
    unittest.main()
