import asyncio
import os
import sys
import time
import unittest
from types import SimpleNamespace
from typing import Any


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (PKG_PYDL,):
    if p not in sys.path:
        sys.path.insert(0, p)


from f8pydl.optflow_service_node import (  # noqa: E402
    OnnxOptflowServiceNode,
    OptflowFramePairCache,
    PreparedFlowFrame,
    pack_flow2_f16_payload,
)
from f8pysdk.bus import ServiceBus, ServiceBusConfig  # noqa: E402
from f8pysdk.state import StateRead  # noqa: E402
from f8pysdk.zenoh_naming import zenoh_data_key  # noqa: E402


class _BusStub:
    def __init__(self, initial_state: dict[str, Any] | None = None, *, has_rungraph: bool = True) -> None:
        self.state: dict[str, Any] = dict(initial_state or {})
        self.errors: list[tuple[str, str, str]] = []
        self.timings: list[tuple[str, float, float, int | None, bool]] = []
        self.clear_count = 0
        self._has_rungraph = bool(has_rungraph)

    def has_rungraph(self) -> bool:
        return self._has_rungraph

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
        self.clear_count += 1

    async def publish_state_runtime(self, node_id: str, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        self.state[field] = value

    async def get_state(self, node_id: str, field: str) -> StateRead:
        del node_id
        if field in self.state:
            return StateRead(found=True, value=self.state[field], ts_ms=None)
        return StateRead(found=False, value=None, ts_ms=None)

    def record_monitor_timing(
        self,
        *,
        port: str,
        process_ms: float,
        latency_ms: float,
        ts_ms: int | None = None,
    ) -> None:
        self.timings.append((str(port), float(process_ms), float(latency_ms), ts_ms, True))


class OptflowPackAndCacheTests(unittest.TestCase):
    def test_pack_flow2_f16_payload_roundtrip(self) -> None:
        import numpy as np  # type: ignore

        flow = np.asarray(
            [
                [[1.25, -2.5], [0.0, 3.125]],
                [[-4.0, 5.5], [6.0, -7.75]],
            ],
            dtype=np.float32,
        )
        pitch, payload = pack_flow2_f16_payload(flow)
        self.assertEqual(pitch, 8)
        self.assertEqual(len(payload), 16)

        decoded = np.frombuffer(payload, dtype=np.float16).reshape((2, 2, 2)).astype(np.float32)
        self.assertTrue(np.allclose(decoded, flow, atol=1e-2))

    def test_frame_pair_cache_reuses_previous_tensor(self) -> None:
        cache = OptflowFramePairCache()
        tensor1 = object()
        tensor2 = object()
        tensor3 = object()
        f1 = PreparedFlowFrame(frame_id=1, width=640, height=480, tensor=tensor1)
        f2 = PreparedFlowFrame(frame_id=2, width=640, height=480, tensor=tensor2)
        f3 = PreparedFlowFrame(frame_id=3, width=640, height=480, tensor=tensor3)

        pair1 = cache.push_and_get_pair(f1)
        self.assertIsNone(pair1)

        pair2 = cache.push_and_get_pair(f2)
        assert pair2 is not None
        self.assertIs(pair2[0].tensor, tensor1)
        self.assertIs(pair2[1].tensor, tensor2)

        pair3 = cache.push_and_get_pair(f3)
        assert pair3 is not None
        self.assertIs(pair3[0].tensor, tensor2)
        self.assertIs(pair3[1].tensor, tensor3)

    def test_frame_pair_cache_resets_on_resolution_change(self) -> None:
        cache = OptflowFramePairCache()
        f1 = PreparedFlowFrame(frame_id=1, width=640, height=480, tensor=object())
        f2 = PreparedFlowFrame(frame_id=2, width=320, height=240, tensor=object())
        f3 = PreparedFlowFrame(frame_id=3, width=320, height=240, tensor=object())

        self.assertIsNone(cache.push_and_get_pair(f1))
        self.assertIsNone(cache.push_and_get_pair(f2))
        self.assertIsNotNone(cache.push_and_get_pair(f3))


class OptflowServiceNodeErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_publish_model_index_scans_local_optflow_models(self) -> None:
        bus = _BusStub()
        node = OnnxOptflowServiceNode(
            node_id="optflowA",
            node=SimpleNamespace(stateFields=[]),
            initial_state=None,
            service_class="f8.dl.optflow",
            allowed_tasks={"optflow_neuflowv2"},
        )
        node._bus = bus

        await node._publish_model_index()

        available = bus.state["availableModels"]
        self.assertEqual(available, ["neuflow_mixed", "neuflow_sintel", "neuflow_things"])
        self.assertEqual(bus.state["modelId"], "neuflow_mixed")

    async def test_set_last_error_dedupes_repeated_message(self) -> None:
        bus = _BusStub()
        node = OnnxOptflowServiceNode(
            node_id="optflowA",
            node=SimpleNamespace(stateFields=[]),
            initial_state=None,
            service_class="f8.dl.optflow",
            allowed_tasks={"optflow_neuflowv2"},
        )
        node._bus = bus

        await node._set_last_error("missing video data input")
        await node._set_last_error("missing video data input")
        await node._set_last_error("")
        await node._set_last_error("")

        self.assertEqual(bus.errors, [("optflowA", "DL_OPTFLOW_RUNTIME", "missing video data input")])
        self.assertEqual(bus.clear_count, 1)

    async def test_loop_retries_when_runtime_is_reset_after_ensure(self) -> None:
        class _RuntimeResetNode(OnnxOptflowServiceNode):
            async def _ensure_config_loaded(self) -> None:
                return None

            def _resolve_input_stream_key(self) -> str:
                return "f8/svc/source/nodes/camera/data/video"

            async def _ensure_runtime(self) -> bool:
                self._runtime = None
                return True

            def _ensure_video_source(self) -> Any:
                raise AssertionError("video source should not be opened without runtime")

        bus = _BusStub()
        node = _RuntimeResetNode(
            node_id="optflowRace",
            node=SimpleNamespace(stateFields=[]),
            initial_state=None,
            service_class="f8.dl.optflow",
            allowed_tasks={"optflow_neuflowv2"},
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

    async def test_missing_input_stream_key_ignores_startup_before_rungraph(self) -> None:
        node = OnnxOptflowServiceNode(
            node_id="optflowGrace",
            node=SimpleNamespace(stateFields=[]),
            initial_state=None,
            service_class="f8.dl.optflow",
            allowed_tasks={"optflow_neuflowv2"},
        )
        node._bus = _BusStub(has_rungraph=False)
        node._missing_input_since_monotonic = time.monotonic() - 30.0

        await node._handle_missing_input_stream_key()

        self.assertEqual(node._bus.errors, [])
        self.assertIsNone(node._missing_input_since_monotonic)

    async def test_missing_input_stream_key_starts_grace_after_rungraph_arrives(self) -> None:
        node = OnnxOptflowServiceNode(
            node_id="optflowGraphGrace",
            node=SimpleNamespace(stateFields=[]),
            initial_state=None,
            service_class="f8.dl.optflow",
            allowed_tasks={"optflow_neuflowv2"},
        )
        node._bus = _BusStub(has_rungraph=True)

        await node._handle_missing_input_stream_key()

        self.assertEqual(node._bus.errors, [])
        self.assertIsNotNone(node._missing_input_since_monotonic)

    async def test_missing_input_stream_key_reports_after_grace_period(self) -> None:
        node = OnnxOptflowServiceNode(
            node_id="optflowMissing",
            node=SimpleNamespace(stateFields=[]),
            initial_state=None,
            service_class="f8.dl.optflow",
            allowed_tasks={"optflow_neuflowv2"},
        )
        bus = _BusStub()
        node._bus = bus
        node._missing_input_since_monotonic = time.monotonic() - 3.0

        await node._handle_missing_input_stream_key()

        self.assertEqual(bus.errors, [("optflowMissing", "DL_OPTFLOW_RUNTIME", "missing video data input")])

    async def test_resolve_input_stream_key_reuses_cached_route_during_grace_period(self) -> None:
        class _RouteFlapNode(OnnxOptflowServiceNode):
            def __init__(self) -> None:
                super().__init__(
                    node_id="optflowRouteFlap",
                    node=SimpleNamespace(stateFields=[]),
                    initial_state=None,
                    service_class="f8.dl.optflow",
                    allowed_tasks={"optflow_neuflowv2"},
                )
                self.current_input_key = "f8/svc/source/nodes/camera/data/video"

            def _current_input_stream_key(self) -> str:
                return self.current_input_key

        node = _RouteFlapNode()

        self.assertEqual(node._resolve_input_stream_key(), "f8/svc/source/nodes/camera/data/video")
        self.assertFalse(node._using_cached_input_stream_key)

        node.current_input_key = ""
        self.assertEqual(node._resolve_input_stream_key(), "f8/svc/source/nodes/camera/data/video")
        self.assertTrue(node._using_cached_input_stream_key)

    async def test_resolve_input_stream_key_reports_missing_after_cached_route_expires(self) -> None:
        class _RouteMissingNode(OnnxOptflowServiceNode):
            def _current_input_stream_key(self) -> str:
                return ""

        node = _RouteMissingNode(
            node_id="optflowRouteExpired",
            node=SimpleNamespace(stateFields=[]),
            initial_state=None,
            service_class="f8.dl.optflow",
            allowed_tasks={"optflow_neuflowv2"},
        )
        node._last_input_stream_key = "f8/svc/source/nodes/camera/data/video"
        node._missing_input_since_monotonic = time.monotonic() - 3.0

        self.assertEqual(node._resolve_input_stream_key(), "")
        self.assertFalse(node._using_cached_input_stream_key)

    async def test_attach_uses_service_id_for_flow_zenoh_key(self) -> None:
        bus = ServiceBus(ServiceBusConfig(service_id="dl_service", bus_backend="mem"))
        node = OnnxOptflowServiceNode(
            node_id="optflowF",
            node=SimpleNamespace(stateFields=[]),
            initial_state=None,
            service_class="f8.dl.optflow",
            allowed_tasks={"optflow_neuflowv2"},
        )

        node.attach(bus)
        try:
            self.assertEqual(
                node._flow_key,
                zenoh_data_key("dl_service", node_id="optflowF", port_id="flow"),
            )
        finally:
            await node.close()

    async def test_close_cancels_attach_init_task(self) -> None:
        started = asyncio.Event()
        finalized = asyncio.Event()

        class _SlowInitNode(OnnxOptflowServiceNode):
            async def _ensure_config_loaded(self) -> None:
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    finalized.set()

            async def _loop(self) -> None:
                await asyncio.Event().wait()

        bus = ServiceBus(ServiceBusConfig(service_id="dl_service", bus_backend="mem"))
        node = _SlowInitNode(
            node_id="optflowF",
            node=SimpleNamespace(stateFields=[]),
            initial_state=None,
            service_class="f8.dl.optflow",
            allowed_tasks={"optflow_neuflowv2"},
        )

        node.attach(bus)
        await started.wait()
        await node.close()

        self.assertIsNone(node._init_task)
        self.assertIsNone(node._task)
        self.assertTrue(finalized.is_set())


if __name__ == "__main__":
    unittest.main()
