from __future__ import annotations

import asyncio
import os
import re
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_PYDL, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)

from f8pysdk.state import StateRead
from f8pysdk.video_transport import VIDEO_FORMAT_BGRA32, VIDEO_FORMAT_FLOW2_F16, VIDEO_FORMAT_SCALAR1_F32

from f8pydl.detection_sorter_service_node import (
    DetectionSorterServiceNode,
    aggregate_roi_score,
    decode_score_map_from_frame,
    rescale_bbox_to_score_map,
    sort_detection_payload,
)
from f8pydl.node_registry import create_pydl_registry


class _BusStub:
    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        self.state: dict[str, Any] = dict(initial_state or {})
        self.errors: list[tuple[str, str, str]] = []
        self.emitted: list[tuple[str, str, Any, int | None, str | int | None]] = []
        self.published_state: list[tuple[str, str, Any, int | None]] = []
        self.timings: list[tuple[str, float, float, int | None, bool]] = []

    async def emit_data(
        self,
        node_id: str,
        port: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        ctx_id: str | int | None = None,
    ) -> None:
        self.emitted.append((node_id, port, value, ts_ms, ctx_id))

    async def publish_state_runtime(self, node_id: str, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        self.state[field] = value
        self.published_state.append((node_id, field, value, ts_ms))

    async def get_state(self, node_id: str, field: str) -> StateRead:
        del node_id
        if field in self.state:
            return StateRead(found=True, value=self.state[field], ts_ms=None)
        return StateRead(found=False, value=None, ts_ms=None)

    def get_state_cached(self, node_id: str, field: str, default: Any = None) -> Any:
        del node_id
        return self.state.get(field, default)

    def report_error(
        self,
        node_id: str,
        code: str,
        message: str,
        severity: str = "error",
        fingerprint: str | None = None,
        ts_ms: int | None = None,
    ) -> None:
        del severity, fingerprint, ts_ms
        self.errors.append((str(node_id), str(code), str(message)))

    def clear_error(self, node_id: str, fingerprint: str | None = None, ts_ms: int | None = None) -> None:
        del node_id, fingerprint, ts_ms
        self.errors.clear()

    def record_monitor_timing(
        self,
        *,
        port: str,
        process_ms: float,
        latency_ms: float,
        ts_ms: int | None = None,
    ) -> None:
        self.timings.append((str(port), float(process_ms), float(latency_ms), ts_ms, True))

    def data_input_zenoh_key(self, node_id: str, port: str) -> str | None:
        del node_id
        if str(port) == "score":
            return "f8/test/detsorter/score"
        return None


def _make_detection_payload(
    detections: list[dict[str, Any]],
    *,
    frame_id: int = 1,
    ts_ms: int = 1000,
    width: int = 4,
    height: int = 4,
) -> dict[str, Any]:
    return {
        "schemaVersion": "f8visionDetections/1",
        "frameId": frame_id,
        "tsMs": ts_ms,
        "width": width,
        "height": height,
        "model": "demo",
        "task": "detector",
        "skeletonProtocol": "none",
        "detections": detections,
    }


class _FrameSourceStub:
    def __init__(self, frame: Any | None) -> None:
        self._frame = frame

    def close(self) -> None:
        return

    def reset(self) -> None:
        return

    def read_latest(self, *, stream_key: str, timeout_ms: int, dedupe: bool = True) -> Any | None:
        del stream_key, timeout_ms, dedupe
        return self._frame


def _make_scalar_frame(array: np.ndarray) -> Any:
    values = np.asarray(array, dtype=np.float32)
    height = int(values.shape[0])
    width = int(values.shape[1])
    pitch = width * 4
    return SimpleNamespace(
        width=width,
        height=height,
        pitch=pitch,
        fmt=VIDEO_FORMAT_SCALAR1_F32,
        frame_id=1,
        ts_ms=1000,
        payload=memoryview(values.tobytes(order="C")),
    )


def _make_flow_frame(flow: np.ndarray) -> Any:
    flow_values = np.asarray(flow, dtype=np.float32)
    height = int(flow_values.shape[0])
    width = int(flow_values.shape[1])
    payload = np.ascontiguousarray(flow_values.astype(np.float16).view(np.uint8)).tobytes(order="C")
    pitch = width * 4
    return SimpleNamespace(
        width=width,
        height=height,
        pitch=pitch,
        fmt=VIDEO_FORMAT_FLOW2_F16,
        frame_id=1,
        ts_ms=1000,
        payload=memoryview(payload),
    )


class DetectionSorterHelpersTests(unittest.TestCase):
    def test_rescale_bbox_to_score_map(self) -> None:
        bbox = rescale_bbox_to_score_map(
            [50, 25, 150, 75],
            detections_width=200,
            detections_height=100,
            score_width=100,
            score_height=50,
        )
        self.assertEqual(bbox, (25, 12, 75, 38))

    def test_aggregate_roi_score_variants(self) -> None:
        roi = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        self.assertEqual(aggregate_roi_score(roi, "max"), 7.0)
        self.assertEqual(aggregate_roi_score(roi, "sum"), 16.0)
        self.assertEqual(aggregate_roi_score(roi, "median"), 4.0)
        self.assertEqual(aggregate_roi_score(roi, "mean"), 4.0)

    def test_sort_detection_payload_mean_desc_keeps_original_score(self) -> None:
        score_map = np.asarray(
            [
                [1.0, 1.0, 9.0, 9.0],
                [1.0, 1.0, 9.0, 9.0],
                [2.0, 2.0, 3.0, 3.0],
                [2.0, 2.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
        payload = _make_detection_payload(
            [
                {"cls": "low", "score": 0.11, "bbox": [0, 0, 2, 2]},
                {"cls": "high", "score": 0.22, "bbox": [2, 0, 4, 2]},
            ]
        )

        sorted_payload = sort_detection_payload(
            payload,
            score_map=score_map,
            sort_direction="desc",
            score_aggregation="mean",
        )

        self.assertIsNotNone(sorted_payload)
        assert sorted_payload is not None
        self.assertEqual([item["cls"] for item in sorted_payload["detections"]], ["high", "low"])
        self.assertEqual([item["score"] for item in sorted_payload["detections"]], [0.22, 0.11])

    def test_sort_detection_payload_flow_magnitude(self) -> None:
        flow = np.zeros((4, 4, 2), dtype=np.float32)
        flow[0:2, 0:2, 0] = 1.0
        flow[2:4, 2:4, 0] = 3.0
        magnitude = np.sqrt((flow[:, :, 0] * flow[:, :, 0]) + (flow[:, :, 1] * flow[:, :, 1]))
        payload = _make_detection_payload(
            [
                {"cls": "weak", "score": 0.5, "bbox": [0, 0, 2, 2]},
                {"cls": "strong", "score": 0.6, "bbox": [2, 2, 4, 4]},
            ]
        )

        sorted_payload = sort_detection_payload(
            payload,
            score_map=magnitude,
            sort_direction="desc",
            score_aggregation="mean",
        )

        self.assertIsNotNone(sorted_payload)
        assert sorted_payload is not None
        self.assertEqual([item["cls"] for item in sorted_payload["detections"]], ["strong", "weak"])

    def test_sort_detection_payload_ascending_order(self) -> None:
        score_map = np.asarray([[10.0, 10.0], [1.0, 1.0]], dtype=np.float32)
        payload = _make_detection_payload(
            [
                {"cls": "high", "score": 0.9, "bbox": [0, 0, 2, 1]},
                {"cls": "low", "score": 0.8, "bbox": [0, 1, 2, 2]},
            ],
            width=2,
            height=2,
        )

        sorted_payload = sort_detection_payload(
            payload,
            score_map=score_map,
            sort_direction="asc",
            score_aggregation="mean",
        )

        self.assertIsNotNone(sorted_payload)
        assert sorted_payload is not None
        self.assertEqual([item["cls"] for item in sorted_payload["detections"]], ["low", "high"])

    def test_sort_detection_payload_applies_cls_weights_exact(self) -> None:
        score_map = np.asarray([[5.0, 5.0], [4.0, 4.0]], dtype=np.float32)
        payload = _make_detection_payload(
            [
                {"cls": "person", "score": 0.1, "bbox": [0, 0, 2, 1]},
                {"cls": "car", "score": 0.2, "bbox": [0, 1, 2, 2]},
            ],
            width=2,
            height=2,
        )

        sorted_payload = sort_detection_payload(
            payload,
            score_map=score_map,
            sort_direction="desc",
            score_aggregation="mean",
            cls_weights_exact={"person": 0.1, "car": 2.0},
            cls_weights_regex=[],
        )

        self.assertIsNotNone(sorted_payload)
        assert sorted_payload is not None
        self.assertEqual([item["cls"] for item in sorted_payload["detections"]], ["car", "person"])

    def test_sort_detection_payload_applies_cls_weights_regex(self) -> None:
        score_map = np.asarray([[5.0, 5.0], [4.0, 4.0]], dtype=np.float32)
        payload = _make_detection_payload(
            [
                {"cls": "cat", "score": 0.1, "bbox": [0, 0, 2, 1]},
                {"cls": "dog_1", "score": 0.2, "bbox": [0, 1, 2, 2]},
            ],
            width=2,
            height=2,
        )

        sorted_payload = sort_detection_payload(
            payload,
            score_map=score_map,
            sort_direction="desc",
            score_aggregation="mean",
            cls_weights_exact={},
            cls_weights_regex=[(re.compile("^dog_.*$"), 2.0)],
        )

        self.assertIsNotNone(sorted_payload)
        assert sorted_payload is not None
        self.assertEqual([item["cls"] for item in sorted_payload["detections"]], ["dog_1", "cat"])

    def test_sort_detection_payload_multiplies_all_matching_cls_weights(self) -> None:
        score_map = np.asarray([[4.0, 4.0], [5.0, 5.0]], dtype=np.float32)
        payload = _make_detection_payload(
            [
                {"cls": "FEMALE_GENITALIA_EXPOSED", "score": 0.1, "bbox": [0, 0, 2, 1]},
                {"cls": "neutral", "score": 0.2, "bbox": [0, 1, 2, 2]},
            ],
            width=2,
            height=2,
        )

        sorted_payload = sort_detection_payload(
            payload,
            score_map=score_map,
            sort_direction="desc",
            score_aggregation="mean",
            cls_weights_exact={},
            cls_weights_regex=[
                (re.compile("^.*_GENITALIA_.*$"), 3.0),
                (re.compile("^.*_EXPOSED$"), 2.0),
            ],
        )

        self.assertIsNotNone(sorted_payload)
        assert sorted_payload is not None
        self.assertEqual([item["cls"] for item in sorted_payload["detections"]], ["FEMALE_GENITALIA_EXPOSED", "neutral"])

    def test_sort_detection_payload_all_aggregations(self) -> None:
        score_map = np.asarray(
            [
                [1.0, 5.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        )
        payload = _make_detection_payload(
            [
                {"cls": "peaky", "score": 0.1, "bbox": [0, 0, 2, 2]},
                {"cls": "steady", "score": 0.2, "bbox": [2, 0, 4, 2]},
            ],
            width=4,
            height=2,
        )

        self.assertEqual(
            [item["cls"] for item in sort_detection_payload(payload, score_map=score_map, sort_direction="desc", score_aggregation="max")["detections"]],
            ["peaky", "steady"],
        )
        self.assertEqual(
            [item["cls"] for item in sort_detection_payload(payload, score_map=score_map, sort_direction="desc", score_aggregation="sum")["detections"]],
            ["peaky", "steady"],
        )
        self.assertEqual(
            [item["cls"] for item in sort_detection_payload(payload, score_map=score_map, sort_direction="desc", score_aggregation="median")["detections"]],
            ["steady", "peaky"],
        )

    def test_sort_detection_payload_invalid_bbox_kept_at_end_stably(self) -> None:
        score_map = np.asarray([[3.0, 3.0], [4.0, 4.0]], dtype=np.float32)
        payload = _make_detection_payload(
            [
                {"cls": "valid", "score": 0.7, "bbox": [0, 0, 2, 2]},
                {"cls": "bad-a", "score": 0.5, "bbox": [1, 1, 1, 2]},
                {"cls": "bad-b", "score": 0.4, "bbox": [0, 0, 0, 0]},
            ],
            width=2,
            height=2,
        )

        sorted_payload = sort_detection_payload(
            payload,
            score_map=score_map,
            sort_direction="desc",
            score_aggregation="mean",
        )

        self.assertIsNotNone(sorted_payload)
        assert sorted_payload is not None
        self.assertEqual([item["cls"] for item in sorted_payload["detections"]], ["valid", "bad-a", "bad-b"])

    def test_sort_detection_payload_zero_temperature_is_deterministic(self) -> None:
        score_map = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        payload = _make_detection_payload(
            [
                {"cls": "low", "bbox": [0, 0, 1, 1]},
                {"cls": "medium", "bbox": [1, 0, 2, 1]},
                {"cls": "high", "bbox": [2, 0, 3, 1]},
            ],
            width=3,
            height=1,
        )

        sorted_payload = sort_detection_payload(
            payload,
            score_map=score_map,
            sort_direction="desc",
            score_aggregation="mean",
            temperature=0.0,
            random_generator=np.random.default_rng(2),
        )

        self.assertIsNotNone(sorted_payload)
        assert sorted_payload is not None
        self.assertEqual([item["cls"] for item in sorted_payload["detections"]], ["high", "medium", "low"])

    def test_sort_detection_payload_temperature_samples_reproducible_order(self) -> None:
        score_map = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        payload = _make_detection_payload(
            [
                {"cls": "low", "bbox": [0, 0, 1, 1]},
                {"cls": "medium", "bbox": [1, 0, 2, 1]},
                {"cls": "high", "bbox": [2, 0, 3, 1]},
            ],
            width=3,
            height=1,
        )

        first = sort_detection_payload(
            payload,
            score_map=score_map,
            sort_direction="desc",
            score_aggregation="mean",
            temperature=1.0,
            random_generator=np.random.default_rng(2),
        )
        second = sort_detection_payload(
            payload,
            score_map=score_map,
            sort_direction="desc",
            score_aggregation="mean",
            temperature=1.0,
            random_generator=np.random.default_rng(2),
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None
        assert second is not None
        first_order = [item["cls"] for item in first["detections"]]
        second_order = [item["cls"] for item in second["detections"]]
        self.assertEqual(first_order, ["medium", "low", "high"])
        self.assertEqual(second_order, first_order)

    def test_sort_detection_payload_temperature_respects_ascending_preference(self) -> None:
        score_map = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        payload = _make_detection_payload(
            [
                {"cls": "low", "bbox": [0, 0, 1, 1]},
                {"cls": "medium", "bbox": [1, 0, 2, 1]},
                {"cls": "high", "bbox": [2, 0, 3, 1]},
            ],
            width=3,
            height=1,
        )

        top_counts = {"low": 0, "medium": 0, "high": 0}
        random_generator = np.random.default_rng(12345)
        for _ in range(2000):
            sorted_payload = sort_detection_payload(
                payload,
                score_map=score_map,
                sort_direction="asc",
                score_aggregation="mean",
                temperature=0.5,
                random_generator=random_generator,
            )
            assert sorted_payload is not None
            top_counts[sorted_payload["detections"][0]["cls"]] += 1

        self.assertGreater(top_counts["low"], top_counts["medium"])
        self.assertGreater(top_counts["medium"], top_counts["high"])


class DetectionSorterRegistryTests(unittest.TestCase):
    def test_detection_sorter_exposes_non_negative_temperature_state(self) -> None:
        registry = create_pydl_registry()
        spec = registry.service_spec("f8.dl.detsorter")

        self.assertIsNotNone(spec)
        assert spec is not None
        temperature_state = next(state for state in spec.stateFields if state.name == "temperature")
        self.assertEqual(temperature_state.valueSchema.default, 0.0)
        self.assertEqual(temperature_state.valueSchema.minimum, 0.0)


class DetectionSorterServiceNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_service_node_keeps_event_loop_responsive_during_sort(self) -> None:
        started = threading.Event()

        class _SlowSorterNode(DetectionSorterServiceNode):
            def _sort_latest_detections(self) -> dict[str, Any] | None:
                started.set()
                time.sleep(0.2)
                return self._latest_detections

        bus = _BusStub()
        node = _SlowSorterNode(node_id="sorterResponsive", node=SimpleNamespace(stateFields=[]), initial_state=None)
        node.attach(bus)
        payload = _make_detection_payload(
            [{"cls": "x", "score": 0.8, "bbox": [0, 0, 2, 2]}],
            frame_id=1,
            width=2,
            height=2,
        )

        sort_task = asyncio.create_task(node.on_data("detections", payload))
        try:
            t0 = time.perf_counter()
            started_ok = await asyncio.to_thread(started.wait, 1.0)
            self.assertTrue(started_ok)
            elapsed_s = time.perf_counter() - t0

            self.assertLess(elapsed_s, 0.15)
            self.assertEqual(bus.emitted, [])
        finally:
            await sort_task
            node._close_score_reader()

        self.assertEqual(len(bus.emitted), 1)

    async def test_validate_state_cls_weights_rejects_invalid_json(self) -> None:
        node = DetectionSorterServiceNode(node_id="sorterZ", node=SimpleNamespace(stateFields=[]), initial_state=None)
        with self.assertRaises(ValueError):
            _ = await node.validate_state("clsWeights", "{", ts_ms=0, meta={})

    async def test_validate_state_temperature_accepts_non_negative_finite_number(self) -> None:
        node = DetectionSorterServiceNode(node_id="sorterTemperature", node=SimpleNamespace(stateFields=[]), initial_state=None)

        validated = await node.validate_state("temperature", 0.75, ts_ms=0, meta={})

        self.assertEqual(validated, 0.75)

    async def test_validate_state_temperature_rejects_invalid_values(self) -> None:
        node = DetectionSorterServiceNode(node_id="sorterTemperature", node=SimpleNamespace(stateFields=[]), initial_state=None)

        for invalid_value in (-0.01, float("nan"), float("inf"), True, "invalid"):
            with self.subTest(value=invalid_value), self.assertRaises(ValueError):
                _ = await node.validate_state("temperature", invalid_value, ts_ms=0, meta={})

    async def test_service_node_loads_and_updates_temperature(self) -> None:
        node = DetectionSorterServiceNode(
            node_id="sorterTemperature",
            node=SimpleNamespace(stateFields=[]),
            initial_state={"temperature": 0.25},
        )

        await node._ensure_config_loaded()
        self.assertEqual(node._temperature, 0.25)

        await node.on_state("temperature", 1.5)
        self.assertEqual(node._temperature, 1.5)

    async def test_service_node_sorts_scalar_map_and_emits(self) -> None:
        frame = _make_scalar_frame(
            np.asarray(
                [
                    [1.0, 1.0, 9.0, 9.0],
                    [1.0, 1.0, 9.0, 9.0],
                    [0.0, 0.0, 2.0, 2.0],
                    [0.0, 0.0, 2.0, 2.0],
                ],
                dtype=np.float32,
            )
        )
        bus = _BusStub()
        node = DetectionSorterServiceNode(node_id="sorterA", node=SimpleNamespace(stateFields=[]), initial_state=None)
        node.attach(bus)
        node._score_source = _FrameSourceStub(frame)
        payload = _make_detection_payload(
            [
                {"cls": "left", "score": 0.3, "bbox": [0, 0, 2, 2]},
                {"cls": "right", "score": 0.4, "bbox": [2, 0, 4, 2]},
            ],
            frame_id=1,
        )

        await node.on_data("detections", payload)

        self.assertEqual(len(bus.emitted), 1)
        emitted_payload = bus.emitted[0][2]
        self.assertEqual([item["cls"] for item in emitted_payload["detections"]], ["right", "left"])
        self.assertEqual(bus.errors, [])
        node._close_score_reader()

    async def test_service_node_inactive_does_not_emit(self) -> None:
        bus = _BusStub()
        node = DetectionSorterServiceNode(node_id="sorterInactive", node=SimpleNamespace(stateFields=[]), initial_state=None)
        node.attach(bus)
        node._score_source = _FrameSourceStub(_make_scalar_frame(np.ones((2, 2), dtype=np.float32)))
        await node.on_lifecycle(False, {})
        payload = _make_detection_payload(
            [{"cls": "x", "score": 0.8, "bbox": [0, 0, 2, 2]}],
            frame_id=1,
            width=2,
            height=2,
        )

        await node.on_data("detections", payload)

        self.assertEqual(bus.emitted, [])
        self.assertEqual(bus.errors, [])
        node._close_score_reader()

    async def test_service_node_supports_flow2f16_magnitude(self) -> None:
        flow = np.zeros((4, 4, 2), dtype=np.float32)
        flow[0:2, 0:2, 0] = 1.0
        flow[2:4, 2:4, 1] = 4.0
        bus = _BusStub()
        node = DetectionSorterServiceNode(node_id="sorterB", node=SimpleNamespace(stateFields=[]), initial_state=None)
        node.attach(bus)
        node._score_source = _FrameSourceStub(_make_flow_frame(flow))
        payload = _make_detection_payload(
            [
                {"cls": "weak", "score": 0.3, "bbox": [0, 0, 2, 2]},
                {"cls": "strong", "score": 0.4, "bbox": [2, 2, 4, 4]},
            ],
            frame_id=1,
        )

        await node.on_data("detections", payload)

        self.assertEqual(len(bus.emitted), 1)
        emitted_payload = bus.emitted[0][2]
        self.assertEqual([item["cls"] for item in emitted_payload["detections"]], ["strong", "weak"])
        node._close_score_reader()

    async def test_service_node_rescales_bboxes(self) -> None:
        frame = _make_scalar_frame(
            np.asarray(
                [
                    [0.0, 0.0, 9.0, 9.0],
                    [0.0, 0.0, 9.0, 9.0],
                ],
                dtype=np.float32,
            )
        )
        bus = _BusStub()
        node = DetectionSorterServiceNode(node_id="sorterC", node=SimpleNamespace(stateFields=[]), initial_state=None)
        node.attach(bus)
        node._score_source = _FrameSourceStub(frame)
        payload = _make_detection_payload(
            [
                {"cls": "left", "score": 0.3, "bbox": [0, 0, 2, 4]},
                {"cls": "right", "score": 0.4, "bbox": [2, 0, 4, 4]},
            ],
            frame_id=1,
            width=4,
            height=4,
        )

        await node.on_data("detections", payload)

        self.assertEqual([item["cls"] for item in bus.emitted[0][2]["detections"]], ["right", "left"])
        node._close_score_reader()

    async def test_service_node_unsupported_format_sets_last_error(self) -> None:
        bgra = np.zeros((2, 2, 4), dtype=np.uint8)
        frame = SimpleNamespace(
            width=2,
            height=2,
            pitch=8,
            fmt=VIDEO_FORMAT_BGRA32,
            frame_id=1,
            ts_ms=1000,
            payload=memoryview(bgra.tobytes(order="C")),
        )
        bus = _BusStub()
        node = DetectionSorterServiceNode(node_id="sorterD", node=SimpleNamespace(stateFields=[]), initial_state=None)
        node.attach(bus)
        node._score_source = _FrameSourceStub(frame)
        payload = _make_detection_payload([{"cls": "x", "score": 0.8, "bbox": [0, 0, 2, 2]}], frame_id=1, width=2, height=2)

        await node.on_data("detections", payload)

        self.assertEqual(len(bus.emitted), 1)
        emitted_payload = bus.emitted[0][2]
        self.assertEqual([item["cls"] for item in emitted_payload["detections"]], ["x"])
        self.assertIn("score source unavailable", bus.errors[-1][2])
        node._close_score_reader()

    async def test_service_node_emits_even_when_frame_id_differs(self) -> None:
        bus = _BusStub()
        node = DetectionSorterServiceNode(node_id="sorterE", node=SimpleNamespace(stateFields=[]), initial_state=None)
        node.attach(bus)
        node._score_source = _FrameSourceStub(_make_scalar_frame(np.ones((2, 2), dtype=np.float32)))
        payload = _make_detection_payload([{"cls": "x", "score": 0.8, "bbox": [0, 0, 2, 2]}], frame_id=10, width=2, height=2)

        await node.on_data("detections", payload)

        self.assertEqual(len(bus.emitted), 1)
        self.assertEqual(bus.errors, [])
        node._close_score_reader()

    async def test_service_node_score_stream_without_frame_passes_through(self) -> None:
        bus = _BusStub()
        node = DetectionSorterServiceNode(node_id="sorterF", node=SimpleNamespace(stateFields=[]), initial_state=None)
        node.attach(bus)
        node._score_source = _FrameSourceStub(None)
        payload = _make_detection_payload(
            [
                {"cls": "a", "score": 0.1, "bbox": [0, 0, 2, 2]},
                {"cls": "b", "score": 0.9, "bbox": [0, 0, 2, 2]},
            ],
            frame_id=1,
            width=2,
            height=2,
        )

        await node.on_data("detections", payload)

        self.assertEqual(len(bus.emitted), 1)
        emitted_payload = bus.emitted[0][2]
        self.assertEqual([item["cls"] for item in emitted_payload["detections"]], ["a", "b"])
        self.assertEqual(bus.errors, [])

    async def test_service_node_invalid_score_frame_sets_last_error(self) -> None:
        frame = SimpleNamespace(
            width=0,
            height=2,
            pitch=8,
            fmt=VIDEO_FORMAT_SCALAR1_F32,
            frame_id=1,
            ts_ms=1000,
            payload=memoryview(b"\x00" * 16),
        )
        bus = _BusStub()
        node = DetectionSorterServiceNode(node_id="sorterH", node=SimpleNamespace(stateFields=[]), initial_state=None)
        node.attach(bus)
        node._score_source = _FrameSourceStub(frame)
        payload = _make_detection_payload(
            [{"cls": "x", "score": 0.8, "bbox": [0, 0, 2, 2]}],
            frame_id=1,
            width=2,
            height=2,
        )

        await node.on_data("detections", payload)

        self.assertEqual(len(bus.emitted), 1)
        self.assertIn("score source unavailable", bus.errors[-1][2])
        node._close_score_reader()

    def test_decode_score_map_from_frame_scalar(self) -> None:
        values = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        header = SimpleNamespace(
            width=2,
            height=2,
            pitch=8,
            fmt=VIDEO_FORMAT_SCALAR1_F32,
        )

        score_map = decode_score_map_from_frame(header=header, payload=memoryview(values.tobytes(order="C")))

        np.testing.assert_allclose(score_map, values)


if __name__ == "__main__":
    unittest.main()
