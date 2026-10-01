import asyncio
import os
import sys
import threading
import time
import unittest
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (PKG_PYDL,):
    if p not in sys.path:
        sys.path.insert(0, p)


from f8pydl.service_node import OnnxVisionServiceNode  # noqa: E402
from f8pysdk.bus import ServiceBus, ServiceBusConfig  # noqa: E402
from f8pysdk.codec import decode_obj  # noqa: E402
from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode, F8StateAccess, F8StateSpec  # noqa: E402
from f8pysdk.specs import array_schema  # noqa: E402
from f8pysdk.specs import string_schema  # noqa: E402
from f8pysdk.testing import InMemoryCluster, InMemoryTransport  # noqa: E402
from f8pysdk.video_transport import VIDEO_FORMAT_BGRA32  # noqa: E402
from f8pysdk.zenoh_naming import zenoh_state_key  # noqa: E402


class _BusStub:
    def __init__(self) -> None:
        self.errors: list[tuple[str, str, str]] = []
        self.state_updates: list[tuple[str, str, Any]] = []
        self.timings: list[tuple[str, float, float, int | None, bool]] = []

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

    async def publish_state_runtime(self, node_id: str, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        self.state_updates.append((str(node_id), str(field), value))

    def record_monitor_timing(
        self,
        *,
        port: str,
        process_ms: float,
        latency_ms: float,
        ts_ms: int | None = None,
    ) -> None:
        self.timings.append((str(port), float(process_ms), float(latency_ms), ts_ms, True))


@dataclass(frozen=True)
class _FakeDetection:
    cls: str
    conf: float
    xyxy: tuple[float, float, float, float]
    keypoints: list[Any] | None = None
    obb: list[tuple[float, float]] | None = None


@dataclass(frozen=True)
class _FakeTemporalRuntime:
    clip_length: int
    sampling_rate: int

    @property
    def buffer_span(self) -> int:
        return (int(self.clip_length) - 1) * int(self.sampling_rate) + 1


class _TemporalHarness:
    _reset_temporal_buffer = OnnxVisionServiceNode._reset_temporal_buffer
    _append_temporal_frame = OnnxVisionServiceNode._append_temporal_frame
    _temporal_window_ready = OnnxVisionServiceNode._temporal_window_ready
    _should_infer_temporal = OnnxVisionServiceNode._should_infer_temporal
    _build_temporal_sequence = OnnxVisionServiceNode._build_temporal_sequence
    _build_detection_payload = OnnxVisionServiceNode._build_detection_payload

    def __init__(self, *, runtime: _FakeTemporalRuntime, infer_every_n: int = 1) -> None:
        self._temporal_det_runtime = runtime
        self._temporal_frame_buffer: deque[Any] = deque(maxlen=runtime.buffer_span)
        self._temporal_frame_counter = 0
        self._infer_every_n = infer_every_n
        self._last_infer_frame_id: int | None = None
        self._enabled_classes: list[str] = []
        self._per_class_k = 0
        self._model = SimpleNamespace(model_id="demo_model", skeleton_protocol="none")
        self._service_task = "detector"

    def _apply_detection_filters(self, detections: list[Any]) -> list[Any]:
        return list(detections)


class YowoTemporalServiceNodeTests(unittest.TestCase):
    def test_temporal_window_warms_up_before_ready(self) -> None:
        harness = _TemporalHarness(runtime=_FakeTemporalRuntime(clip_length=4, sampling_rate=2))
        for frame_id in range(1, 7):
            harness._append_temporal_frame(
                prepared_frame=np.full((3, 1, 1), frame_id, dtype=np.float32),
                frame_id=frame_id,
                ts_ms=frame_id * 10,
            )
            self.assertFalse(harness._temporal_window_ready())
        harness._append_temporal_frame(
            prepared_frame=np.full((3, 1, 1), 7, dtype=np.float32),
            frame_id=7,
            ts_ms=70,
        )
        self.assertTrue(harness._temporal_window_ready())

    def test_sampling_rate_selects_sparse_frames(self) -> None:
        harness = _TemporalHarness(runtime=_FakeTemporalRuntime(clip_length=4, sampling_rate=2))
        for frame_id in range(1, 8):
            harness._append_temporal_frame(
                prepared_frame=np.full((3, 1, 1), frame_id, dtype=np.float32),
                frame_id=frame_id,
                ts_ms=frame_id * 10,
            )
        sequence = harness._build_temporal_sequence()
        self.assertEqual(sequence.shape, (4, 3, 1, 1))
        self.assertEqual(sequence[:, 0, 0, 0].tolist(), [1.0, 3.0, 5.0, 7.0])

    def test_infer_every_n_throttles_temporal_inference(self) -> None:
        harness = _TemporalHarness(runtime=_FakeTemporalRuntime(clip_length=4, sampling_rate=1), infer_every_n=2)
        for frame_id in range(1, 5):
            harness._append_temporal_frame(
                prepared_frame=np.full((3, 1, 1), frame_id, dtype=np.float32),
                frame_id=frame_id,
                ts_ms=frame_id * 10,
            )
        self.assertTrue(harness._should_infer_temporal())
        harness._last_infer_frame_id = 4
        harness._append_temporal_frame(
            prepared_frame=np.full((3, 1, 1), 5, dtype=np.float32),
            frame_id=5,
            ts_ms=50,
        )
        self.assertFalse(harness._should_infer_temporal())
        harness._append_temporal_frame(
            prepared_frame=np.full((3, 1, 1), 6, dtype=np.float32),
            frame_id=6,
            ts_ms=60,
        )
        self.assertTrue(harness._should_infer_temporal())

    def test_reset_temporal_buffer_clears_history(self) -> None:
        harness = _TemporalHarness(runtime=_FakeTemporalRuntime(clip_length=4, sampling_rate=1))
        harness._append_temporal_frame(
            prepared_frame=np.ones((3, 1, 1), dtype=np.float32),
            frame_id=1,
            ts_ms=10,
        )
        harness._reset_temporal_buffer()
        self.assertEqual(len(harness._temporal_frame_buffer), 0)
        self.assertEqual(harness._temporal_frame_counter, 0)

    def test_temporal_detection_payload_stays_on_standard_schema(self) -> None:
        harness = _TemporalHarness(runtime=_FakeTemporalRuntime(clip_length=4, sampling_rate=1))
        payload = harness._build_detection_payload(
            width=640,
            height=360,
            frame_id=9,
            ts_ms=1234,
            detections=[_FakeDetection(cls="insertive_actor", conf=0.8, xyxy=(10.0, 20.0, 100.0, 120.0))],
        )
        self.assertEqual(payload["schemaVersion"], "f8visionDetections/1")
        self.assertEqual(payload["frameId"], 9)
        self.assertEqual(payload["detections"][0]["cls"], "insertive_actor")
        self.assertEqual(payload["detections"][0]["bbox"], [10, 20, 100, 120])


class OnnxVisionServiceNodeLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_loop_keeps_event_loop_responsive_during_detector_inference(self) -> None:
        class _Frame:
            fmt = VIDEO_FORMAT_BGRA32
            frame_id = 1
            ts_ms = 100
            width = 1
            height = 1
            pitch = 4
            payload = bytes([0, 0, 0, 255])

            def release(self) -> None:
                return None

        class _Source:
            def __init__(self) -> None:
                self._returned = False

            def read_latest(self, *, stream_key: str, timeout_ms: int) -> _Frame | None:
                del stream_key, timeout_ms
                if self._returned:
                    return None
                self._returned = True
                return _Frame()

        class _SlowDetectorRuntime:
            def __init__(self, *, started: threading.Event) -> None:
                self._started = started

            def infer(self, frame_bgr: object) -> tuple[list[Any], dict[str, object]]:
                del frame_bgr
                self._started.set()
                time.sleep(0.2)
                return [], {}

        class _ResponsiveNode(OnnxVisionServiceNode):
            def __init__(self, *, runtime: _SlowDetectorRuntime, source: _Source) -> None:
                super().__init__(
                    node_id="detector",
                    node=SimpleNamespace(stateFields=[]),
                    initial_state=None,
                    service_class="f8.dl.detector",
                    service_task="detector",
                    output_port="detections",
                    allowed_tasks={"yolo_det"},
                )
                self._runtime = runtime
                self._source = source
                self.emitted: list[tuple[str, Any]] = []

            async def _ensure_config_loaded(self) -> None:
                return None

            async def _ensure_runtime(self) -> bool:
                self._det_runtime = self._runtime  # type: ignore[assignment]
                self._temporal_det_runtime = None
                self._cls_runtime = None
                return True

            def _ensure_video_source(self) -> _Source:
                return self._source

            def _resolve_video_stream_key(self) -> str:
                return "video-stream"

            async def emit(
                self,
                port: str,
                value: Any,
                *,
                ts_ms: int | None = None,
                ctx_id: str | int | None = None,
            ) -> None:
                del ts_ms, ctx_id
                self.emitted.append((str(port), value))

        started = threading.Event()
        node = _ResponsiveNode(runtime=_SlowDetectorRuntime(started=started), source=_Source())
        node._bus = _BusStub()
        loop_task = asyncio.create_task(node._loop())
        try:
            t0 = time.perf_counter()
            started_ok = await asyncio.to_thread(started.wait, 1.0)
            self.assertTrue(started_ok)
            elapsed_s = time.perf_counter() - t0

            self.assertLess(elapsed_s, 0.15)
            self.assertEqual(node.emitted, [])
        finally:
            loop_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await loop_task

    async def test_publish_model_index_scans_local_detector_models(self) -> None:
        bus = _BusStub()
        node = OnnxVisionServiceNode(
            node_id="detector",
            node=SimpleNamespace(stateFields=[]),
            initial_state=None,
            service_class="f8.dl.detector",
            service_task="detector",
            output_port="detections",
            allowed_tasks={"yolo_det", "yolo_obb", "yowo_temporal_det"},
        )
        node._bus = bus

        await node._publish_model_index()

        updates = {(field, str(node_id)): value for node_id, field, value in bus.state_updates}
        available = updates[("availableModels", "detector")]
        self.assertIsInstance(available, list)
        self.assertIn("nudenet-320n", available)
        self.assertIn("erax_nsfw_yolo11n", available)
        self.assertIn("f8_motion_yowov3", available)
        self.assertEqual(updates[("modelId", "detector")], available[0])

    async def test_detector_model_index_persists_runtime_state_on_bus(self) -> None:
        transport = InMemoryTransport(cluster=InMemoryCluster())
        bus = ServiceBus(ServiceBusConfig(service_id="detector", service_class="f8.dl.detector"), transport=transport)
        node = OnnxVisionServiceNode(
            node_id="detector",
            node=F8RuntimeNode(nodeId="detector", serviceId="detector", serviceClass="f8.dl.detector"),
            initial_state=None,
            service_class="f8.dl.detector",
            service_task="detector",
            output_port="detections",
            allowed_tasks={"yolo_det", "yolo_obb", "yowo_temporal_det"},
        )
        bus.register_node(node)
        bus.state_store.access_by_node_field[("detector", "availableModels")] = F8StateAccess.ro
        bus.state_store.access_by_node_field[("detector", "modelId")] = F8StateAccess.rw

        await node._publish_model_index()

        raw = await transport.retained_get(zenoh_state_key("detector", node_id="detector", field="availableModels"))
        payload = decode_obj(raw) if raw is not None else {}
        self.assertEqual(payload.get("origin"), "runtime")
        self.assertIn("nudenet-320n", payload.get("value") or [])

    async def test_detector_republishes_model_index_after_rungraph_apply(self) -> None:
        transport = InMemoryTransport(cluster=InMemoryCluster())
        bus = ServiceBus(ServiceBusConfig(service_id="detector", service_class="f8.dl.detector"), transport=transport)
        node = OnnxVisionServiceNode(
            node_id="detector",
            node=F8RuntimeNode(nodeId="detector", serviceId="detector", serviceClass="f8.dl.detector"),
            initial_state=None,
            service_class="f8.dl.detector",
            service_task="detector",
            output_port="detections",
            allowed_tasks={"yolo_det", "yolo_obb", "yowo_temporal_det"},
        )
        bus.register_node(node)
        bus.register_rungraph_hook(node)
        graph = F8RuntimeGraph(
            graphId="g-detector-models",
            revision="r1",
            nodes=[
                F8RuntimeNode(
                    nodeId="detector",
                    serviceId="detector",
                    serviceClass="f8.dl.detector",
                    operatorClass=None,
                    stateFields=[
                        F8StateSpec(
                            name="availableModels",
                            valueSchema=array_schema(items=string_schema()),
                            access=F8StateAccess.ro,
                        ),
                        F8StateSpec(name="modelId", valueSchema=string_schema(), access=F8StateAccess.rw),
                        F8StateSpec(
                            name="modelClasses",
                            valueSchema=array_schema(items=string_schema()),
                            access=F8StateAccess.ro,
                        ),
                        F8StateSpec(
                            name="enabledClasses",
                            valueSchema=array_schema(items=string_schema()),
                            access=F8StateAccess.rw,
                        ),
                    ],
                )
            ],
            edges=[],
        )

        await bus.set_rungraph(graph)
        key = zenoh_state_key("detector", node_id="detector", field="availableModels")
        raw = await transport.retained_get(key)
        payload = decode_obj(raw) if raw is not None else {}

        self.assertEqual(payload.get("origin"), "runtime")
        self.assertIn("nudenet-320n", payload.get("value") or [])

    async def test_loop_retries_when_runtime_is_reset_after_ensure(self) -> None:
        class _RuntimeResetNode(OnnxVisionServiceNode):
            async def _ensure_config_loaded(self) -> None:
                return None

            async def _ensure_runtime(self) -> bool:
                self._det_runtime = None
                self._temporal_det_runtime = None
                self._cls_runtime = None
                return True

            def _ensure_video_source(self) -> Any:
                raise AssertionError("video source should not be opened without runtime")

        bus = _BusStub()
        node = _RuntimeResetNode(
            node_id="visionRace",
            node=SimpleNamespace(stateFields=[]),
            initial_state=None,
            service_class="f8.dl.detector",
            service_task="detector",
            output_port="detections",
            allowed_tasks={"yolo_det"},
        )
        node._bus = bus

        task = asyncio.create_task(node._loop())
        await asyncio.sleep(0.08)
        if task.done():
            task.result()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(bus.errors, [])

    async def test_close_cancels_attach_init_task(self) -> None:
        started = asyncio.Event()
        finalized = asyncio.Event()

        class _SlowInitNode(OnnxVisionServiceNode):
            async def _ensure_config_loaded(self) -> None:
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    finalized.set()

            async def _loop(self) -> None:
                await asyncio.Event().wait()

        bus = ServiceBus(ServiceBusConfig(service_id="detector", service_class="f8.dl.detector", bus_backend="mem"))
        node = _SlowInitNode(
            node_id="detector",
            node=SimpleNamespace(stateFields=[]),
            initial_state=None,
            service_class="f8.dl.detector",
            service_task="detector",
            output_port="detections",
            allowed_tasks={"yolo_det"},
        )

        node.attach(bus)
        await started.wait()
        await node.close()

        self.assertIsNone(node._init_task)
        self.assertIsNone(node._task)
        self.assertTrue(finalized.is_set())


if __name__ == "__main__":
    unittest.main()
