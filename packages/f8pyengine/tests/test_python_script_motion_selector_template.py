import unittest
from pathlib import Path


def _load_motion_selector_script() -> str:
    script_path = Path(__file__).resolve().parents[3] / "docs" / "scenarios" / "scripts" / "motion_selector.py"
    return script_path.read_text(encoding="utf-8")


def _build_env() -> dict[str, object]:
    env: dict[str, object] = {}
    exec(_load_motion_selector_script(), env, env)
    return env


class MotionSelectorTemplateTests(unittest.IsolatedAsyncioTestCase):
    def _build_ctx(self, *, flow_packet: dict[str, object] | None) -> dict[str, object]:
        state: dict[str, object] = {"flow_packet": flow_packet, "subscribed": None}

        async def _get_state(_field: str) -> object:
            return "test.flow.shm"

        def _subscribe_video_shm(key: str, shm_name: str, *, decode: str = "auto", use_event: bool = False) -> None:
            state["subscribed"] = {
                "key": key,
                "shm_name": shm_name,
                "decode": decode,
                "use_event": use_event,
            }

        def _get_video_shm(_key: str) -> dict[str, object] | None:
            packet = state.get("flow_packet")
            return packet if isinstance(packet, dict) else None

        return {
            "locals": {},
            "log": lambda _msg: None,
            "get_state": _get_state,
            "subscribe_video_shm": _subscribe_video_shm,
            "get_video_shm": _get_video_shm,
        }

    async def test_selects_highest_motion_bbox(self) -> None:
        env = _build_env()
        on_start = env["onStart"]
        on_msg = env["onMsg"]
        assert callable(on_start)
        assert callable(on_msg)
        np_module = env.get("np")
        if np_module is None:
            self.skipTest("numpy not available for motion_selector template")

        flow_array = np_module.zeros((96, 96, 2), dtype=np_module.float32)
        flow_array[50:90, 50:90, 0] = 4.0
        flow_packet: dict[str, object] = {
            "header": {"frameId": 10, "tsMs": 1000},
            "decoded": {"kind": "flow2_f16", "data": flow_array},
            "meta": {},
            "raw": b"",
        }
        ctx = self._build_ctx(flow_packet=flow_packet)
        await on_start(ctx)

        detections_payload = {
            "schemaVersion": "f8visionDetections/1",
            "frameId": 10,
            "tsMs": 1000,
            "width": 96,
            "height": 96,
            "detections": [
                {"cls": "a", "score": 0.3, "bbox": [0, 0, 40, 40]},
                {"cls": "b", "score": 0.4, "bbox": [50, 50, 90, 90]},
            ],
        }

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

    async def test_returns_none_when_frame_gap_too_large(self) -> None:
        env = _build_env()
        on_start = env["onStart"]
        on_msg = env["onMsg"]
        assert callable(on_start)
        assert callable(on_msg)
        np_module = env.get("np")
        if np_module is None:
            self.skipTest("numpy not available for motion_selector template")

        flow_array = np_module.zeros((64, 64, 2), dtype=np_module.float32)
        flow_array[8:32, 8:32, 0] = 2.0
        flow_packet: dict[str, object] = {
            "header": {"frameId": 100, "tsMs": 5000},
            "decoded": {"kind": "flow2_f16", "data": flow_array},
            "meta": {},
            "raw": b"",
        }
        ctx = self._build_ctx(flow_packet=flow_packet)
        await on_start(ctx)

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


if __name__ == "__main__":
    unittest.main()
