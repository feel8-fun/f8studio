import os
import sys
import unittest
from dataclasses import dataclass


PKG_MPPOSE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_MPPOSE, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)


from f8pymppose.constants import SKELETON_PROTOCOL_MEDIAPIPE_POSE_33  # noqa: E402
from f8pymppose.service_node import (  # noqa: E402
    _tasks_model_spec_for_complexity,
    build_pose_detection_payload,
    build_pose_skeleton_payload,
    compute_bbox_from_keypoints,
    extract_pose_keypoints,
    extract_pose_world_keypoints,
    should_run_inference,
)


@dataclass(frozen=True)
class _FakeLandmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass(frozen=True)
class _FakeTasksResult:
    pose_landmarks: list[list[_FakeLandmark]]
    pose_world_landmarks: list[list[_FakeLandmark]] | None = None


class PosePayloadTests(unittest.TestCase):
    def test_should_run_inference_every_n(self) -> None:
        self.assertTrue(should_run_inference(None, 100, 3))
        self.assertFalse(should_run_inference(100, 102, 3))
        self.assertTrue(should_run_inference(100, 103, 3))

    def test_extract_pose_keypoints_with_visibility_threshold(self) -> None:
        result = _FakeTasksResult(
            pose_landmarks=[
                [
                    _FakeLandmark(x=0.1, y=0.2, z=0.3, visibility=0.9),
                    _FakeLandmark(x=0.3, y=0.4, z=0.1, visibility=0.2),
                ]
            ]
        )
        keypoints = extract_pose_keypoints(result, width=200, height=100, visibility_threshold=0.5)
        self.assertEqual(len(keypoints), 2)
        self.assertAlmostEqual(float(keypoints[0]["x"] or 0.0), 20.0, places=3)
        self.assertAlmostEqual(float(keypoints[0]["y"] or 0.0), 20.0, places=3)
        self.assertIsNone(keypoints[1]["x"])
        self.assertIsNone(keypoints[1]["y"])

    def test_extract_pose_keypoints_tasks_result_shape(self) -> None:
        result = _FakeTasksResult(
            pose_landmarks=[
                [
                    _FakeLandmark(x=0.5, y=0.5, z=0.0, visibility=0.9),
                    _FakeLandmark(x=0.25, y=0.2, z=-0.1, visibility=0.8),
                ]
            ]
        )
        keypoints = extract_pose_keypoints(result, width=100, height=200, visibility_threshold=0.5)
        self.assertEqual(len(keypoints), 2)
        self.assertAlmostEqual(float(keypoints[0]["x"] or 0.0), 50.0, places=3)
        self.assertAlmostEqual(float(keypoints[0]["y"] or 0.0), 100.0, places=3)

    def test_extract_pose_world_keypoints(self) -> None:
        result = _FakeTasksResult(
            pose_landmarks=[[]],
            pose_world_landmarks=[
                [
                    _FakeLandmark(x=0.05, y=-0.03, z=0.12, visibility=0.9),
                    _FakeLandmark(x=0.02, y=0.01, z=-0.08, visibility=0.2),
                ]
            ],
        )
        world_keypoints = extract_pose_world_keypoints(result, visibility_threshold=0.5)
        self.assertEqual(len(world_keypoints), 2)
        self.assertAlmostEqual(float(world_keypoints[0]["x"] or 0.0), 0.05, places=6)
        self.assertAlmostEqual(float(world_keypoints[0]["y"] or 0.0), -0.03, places=6)
        self.assertAlmostEqual(float(world_keypoints[0]["z"] or 0.0), 0.12, places=6)
        self.assertIsNone(world_keypoints[1]["x"])

    def test_compute_bbox_from_keypoints(self) -> None:
        keypoints = [
            {"x": 10.0, "y": 20.0, "z": 0.0, "score": 0.9},
            {"x": 110.0, "y": 70.0, "z": 0.1, "score": 0.8},
            {"x": None, "y": None, "z": None, "score": 0.1},
        ]
        bbox = compute_bbox_from_keypoints(keypoints, width=200, height=100)
        self.assertEqual(bbox, [10, 20, 110, 70])

    def test_build_payload_with_33_keypoints(self) -> None:
        keypoints = [
            {"x": float(i + 1), "y": float(i + 2), "z": float(i) * 0.1, "score": 0.95}
            for i in range(33)
        ]
        payload = build_pose_detection_payload(
            frame_id=12,
            ts_ms=3456,
            width=1920,
            height=1080,
            keypoints=keypoints,
        )
        self.assertEqual(payload["schemaVersion"], "f8visionDetections/1")
        self.assertEqual(payload["skeletonProtocol"], SKELETON_PROTOCOL_MEDIAPIPE_POSE_33)
        self.assertEqual(len(payload["detections"]), 1)
        det = payload["detections"][0]
        self.assertEqual(det["id"], 1)
        self.assertEqual(det["cls"], "person")
        self.assertEqual(det["skeletonProtocol"], SKELETON_PROTOCOL_MEDIAPIPE_POSE_33)
        self.assertEqual(len(det["keypoints"]), 33)

    def test_build_payload_empty_when_no_visible_points(self) -> None:
        keypoints = [{"x": None, "y": None, "z": None, "score": 0.1} for _ in range(33)]
        payload = build_pose_detection_payload(
            frame_id=12,
            ts_ms=3456,
            width=1920,
            height=1080,
            keypoints=keypoints,
        )
        self.assertEqual(payload["skeletonProtocol"], SKELETON_PROTOCOL_MEDIAPIPE_POSE_33)
        self.assertEqual(payload["detections"], [])

    def test_build_skeleton_payload_udp_compatible_shape(self) -> None:
        keypoints = [
            {"x": float(i + 1), "y": float(i + 2), "z": 0.1 * float(i + 1), "score": 0.95}
            for i in range(33)
        ]
        payload = build_pose_skeleton_payload(
            frame_id=99,
            ts_ms=2345,
            keypoints=keypoints,
            world_keypoints=None,
            width=200,
            height=100,
        )
        self.assertEqual(payload["type"], "skeleton_binary")
        self.assertEqual(payload["schema"], "f8.skeleton.v1")
        self.assertEqual(payload["modelName"], "MediaPipePose")
        self.assertEqual(payload["skeletonProtocol"], SKELETON_PROTOCOL_MEDIAPIPE_POSE_33)
        self.assertEqual(payload["boneCount"], 33)
        bones = payload["bones"]
        self.assertEqual(len(bones), 33)
        self.assertEqual(bones[0]["name"], "nose")
        self.assertEqual(len(bones[0]["rot"]), 4)
        self.assertEqual(bones[0]["pos"], [0.005, 0.02, 0.1])
        self.assertEqual(bones[11]["name"], "left_shoulder")
        self.assertEqual(payload["trailer"], None)

    def test_build_skeleton_payload_skips_hidden_points(self) -> None:
        keypoints = [{"x": None, "y": None, "z": None, "score": 0.0} for _ in range(33)]
        payload = build_pose_skeleton_payload(
            frame_id=100,
            ts_ms=3456,
            keypoints=keypoints,
            world_keypoints=None,
            width=1280,
            height=720,
        )
        self.assertEqual(payload["boneCount"], 0)
        self.assertEqual(payload["bones"], [])

    def test_bone_orientation_identity_when_direction_is_positive_y(self) -> None:
        keypoints = [{"x": None, "y": None, "z": None, "score": 0.0} for _ in range(33)]
        keypoints[11] = {"x": 100.0, "y": 100.0, "z": 0.0, "score": 1.0}
        keypoints[13] = {"x": 100.0, "y": 200.0, "z": 0.0, "score": 1.0}
        payload = build_pose_skeleton_payload(
            frame_id=101,
            ts_ms=4567,
            keypoints=keypoints,
            world_keypoints=None,
            width=1000,
            height=1000,
        )
        by_name = {str(item["name"]): item for item in payload["bones"]}
        shoulder = by_name["left_shoulder"]
        self.assertEqual(shoulder["rot"], [1.0, 0.0, 0.0, 0.0])

    def test_bone_orientation_rotates_toward_positive_x(self) -> None:
        keypoints = [{"x": None, "y": None, "z": None, "score": 0.0} for _ in range(33)]
        keypoints[11] = {"x": 100.0, "y": 100.0, "z": 0.0, "score": 1.0}
        keypoints[13] = {"x": 200.0, "y": 100.0, "z": 0.0, "score": 1.0}
        payload = build_pose_skeleton_payload(
            frame_id=102,
            ts_ms=5678,
            keypoints=keypoints,
            world_keypoints=None,
            width=1000,
            height=1000,
        )
        by_name = {str(item["name"]): item for item in payload["bones"]}
        shoulder = by_name["left_shoulder"]
        rot = shoulder["rot"]
        self.assertAlmostEqual(float(rot[0]), 0.707106, places=4)
        self.assertAlmostEqual(float(rot[1]), 0.0, places=4)
        self.assertAlmostEqual(float(rot[2]), 0.0, places=4)
        self.assertAlmostEqual(float(rot[3]), -0.707106, places=4)

    def test_build_skeleton_payload_prefers_world_keypoints(self) -> None:
        keypoints = [
            {"x": float(i + 1), "y": float(i + 2), "z": 0.1 * float(i + 1), "score": 0.95}
            for i in range(33)
        ]
        world_keypoints = [
            {"x": 0.01 * float(i), "y": -0.02 * float(i), "z": 0.03 * float(i), "score": 0.95}
            for i in range(33)
        ]
        payload = build_pose_skeleton_payload(
            frame_id=120,
            ts_ms=6789,
            keypoints=keypoints,
            world_keypoints=world_keypoints,
            width=1920,
            height=1080,
        )
        bones = payload["bones"]
        self.assertEqual(bones[10]["name"], "mouth_right")
        self.assertEqual(bones[10]["pos"], [0.1, 0.2, 0.3])

    def test_tasks_model_spec_mapping(self) -> None:
        lite = _tasks_model_spec_for_complexity("lite")
        full = _tasks_model_spec_for_complexity("full")
        heavy = _tasks_model_spec_for_complexity("heavy")
        self.assertIn("lite", lite.filename)
        self.assertIn("full", full.filename)
        self.assertIn("heavy", heavy.filename)


if __name__ == "__main__":
    unittest.main()
