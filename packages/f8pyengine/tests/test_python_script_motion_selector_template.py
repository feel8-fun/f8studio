import unittest
from pathlib import Path


def _load_motion_selector_script() -> str:
    script_path = Path(__file__).resolve().parents[3] / "docs" / "scenarios" / "scripts" / "motion_selector.py"
    return script_path.read_text(encoding="utf-8")


def _build_env() -> dict[str, object]:
    env: dict[str, object] = {}
    exec(_load_motion_selector_script(), env, env)
    return env


class MotionSelectorTemplateTests(unittest.TestCase):
    def test_selects_highest_motion_bbox(self) -> None:
        env = _build_env()
        on_start = env["onStart"]
        on_msg = env["onMsg"]
        assert callable(on_start)
        assert callable(on_msg)

        logs: list[str] = []
        ctx = {"locals": {}, "log": logs.append}
        on_start(ctx)

        flow_payload = {
            "schemaVersion": "f8visionFlowField/1",
            "frameId": 10,
            "tsMs": 1000,
            "stats": {"meanMag": 0.15},
            "vectors": [
                {"x": 10, "y": 10, "mag": 0.20},
                {"x": 12, "y": 14, "mag": 0.22},
                {"x": 14, "y": 18, "mag": 0.20},
                {"x": 16, "y": 20, "mag": 0.23},
                {"x": 18, "y": 22, "mag": 0.21},
                {"x": 20, "y": 24, "mag": 0.20},
                {"x": 60, "y": 60, "mag": 0.95},
                {"x": 62, "y": 64, "mag": 0.92},
                {"x": 64, "y": 68, "mag": 0.91},
                {"x": 66, "y": 70, "mag": 0.90},
                {"x": 68, "y": 72, "mag": 0.89},
                {"x": 70, "y": 74, "mag": 0.88},
            ],
        }
        detections_payload = {
            "schemaVersion": "f8visionDetections/1",
            "frameId": 10,
            "tsMs": 1000,
            "width": 1920,
            "height": 1080,
            "detections": [
                {"cls": "a", "score": 0.3, "bbox": [0, 0, 40, 40]},
                {"cls": "b", "score": 0.4, "bbox": [50, 50, 90, 90]},
            ],
        }

        self.assertIsNone(on_msg(ctx, {"flowField": flow_payload}))
        out = on_msg(ctx, {"detections": detections_payload})
        self.assertIsInstance(out, dict)
        assert isinstance(out, dict)
        outputs = out.get("outputs")
        self.assertIsInstance(outputs, dict)
        assert isinstance(outputs, dict)
        selected = outputs.get("selected")
        self.assertIsInstance(selected, dict)
        assert isinstance(selected, dict)
        selected_dets = selected.get("detections")
        self.assertIsInstance(selected_dets, list)
        assert isinstance(selected_dets, list)
        self.assertEqual(len(selected_dets), 1)
        det = selected_dets[0]
        self.assertEqual(det.get("cls"), "b")
        self.assertGreater(float(det.get("score") or 0.0), 0.05)

    def test_returns_none_when_frame_gap_too_large(self) -> None:
        env = _build_env()
        on_start = env["onStart"]
        on_msg = env["onMsg"]
        assert callable(on_start)
        assert callable(on_msg)

        ctx = {"locals": {}, "log": lambda _msg: None}
        on_start(ctx)

        flow_payload = {
            "schemaVersion": "f8visionFlowField/1",
            "frameId": 100,
            "tsMs": 5000,
            "stats": {"meanMag": 0.1},
            "vectors": [
                {"x": 10, "y": 10, "mag": 0.6},
                {"x": 12, "y": 10, "mag": 0.6},
                {"x": 14, "y": 10, "mag": 0.6},
                {"x": 16, "y": 10, "mag": 0.6},
                {"x": 18, "y": 10, "mag": 0.6},
                {"x": 20, "y": 10, "mag": 0.6},
            ],
        }
        detections_payload = {
            "schemaVersion": "f8visionDetections/1",
            "frameId": 1,
            "tsMs": 100,
            "width": 640,
            "height": 360,
            "detections": [
                {"cls": "x", "score": 0.8, "bbox": [0, 0, 64, 64]},
            ],
        }

        self.assertIsNone(on_msg(ctx, {"detections": detections_payload}))
        self.assertIsNone(on_msg(ctx, {"flowField": flow_payload}))


if __name__ == "__main__":
    unittest.main()
