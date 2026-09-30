from __future__ import annotations

import asyncio
import os
import sys
import unittest
from typing import Any
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk import monitoring as monitoring_module  # noqa: E402
from f8pysdk.codec import decode_obj, dump_json  # noqa: E402
from f8pysdk.f8_naming import data_key  # noqa: E402
from f8pysdk.monitoring import MonitorCollector, MonitorCollectorConfig  # noqa: E402
from f8pysdk.time_utils import now_ms  # noqa: E402


class _FakeTransport:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes]] = []

    async def publish(self, key: str, payload: bytes) -> None:
        self.published.append((str(key), bytes(payload)))


class _FakeDataRouter:
    def __init__(self) -> None:
        self._depth = 3

    def queue_depth(self) -> int:
        return self._depth


class _FlakyDataRouter:
    def __init__(self) -> None:
        self.calls = 0

    def queue_depth(self) -> int:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("queue unavailable")
        return 3


class _FailingTransport:
    async def publish(self, key: str, payload: bytes) -> None:
        _ = key
        _ = payload
        raise OSError("publish unavailable")


class _FakeBus:
    def __init__(self) -> None:
        self.service_id = "svcA"
        self._service_class = "f8.tests.service"
        self._active = True
        self._ready = False
        self._transport = _FakeTransport()
        self.data_router = _FakeDataRouter()


class MonitorCollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_and_publish_snapshot(self) -> None:
        bus = _FakeBus()
        collector = MonitorCollector(
            bus, MonitorCollectorConfig(enabled=True, interval_ms=200, window_ms=2000, gpu_enabled=False)
        )
        ts = int(now_ms())
        collector.record_ready(True)
        collector.record_observed(port="in")
        collector.record_processed(port="out", emit_ts_ms=ts - 12, now_ts_ms=ts)
        collector.record_timing(port="out", process_ms=12.0, latency_ms=25.0, ts_ms=ts)
        collector.record_input_sample_ts(node_id="n1", sample_ts_ms=ts - 25)
        collector.record_emit_completed(node_id="n1", now_ts_ms=ts)
        collector.record_wait_ms(wait_ms=8.0)
        collector.record_dropped(dropped_count=2)
        collector.record_local_only_emit()
        collector.record_routed_cross_emit()
        collector.record_suppressed_cross_publish()
        collector.record_callback_delivery()
        collector.record_buffer_pull_delivery()
        collector.report_error(
            node_id="n1",
            code="X_ERR",
            message="boom",
            severity="warning",
            fingerprint="fp-boom",
            ts_ms=ts,
        )

        snapshot = collector._build_snapshot(ts_ms=ts)
        payload = dump_json(snapshot)
        self.assertEqual(payload.get("schemaVersion"), "f8monitor/1")
        self.assertEqual(str(snapshot.serviceId), "svcA")
        self.assertTrue(bool(snapshot.ready))
        self.assertEqual(int(snapshot.frame.observed), 1)
        self.assertEqual(int(snapshot.frame.processed), 1)
        self.assertEqual(int(snapshot.frame.dropped), 2)
        self.assertEqual(int(snapshot.frame.localOnlyEmits), 1)
        self.assertEqual(int(snapshot.frame.routedCrossEmits), 1)
        self.assertEqual(int(snapshot.frame.suppressedCrossPublishes), 1)
        self.assertEqual(int(snapshot.frame.callbackDeliveries), 1)
        self.assertEqual(int(snapshot.frame.bufferPullDeliveries), 1)
        self.assertEqual(int(snapshot.queue.depth), 3)
        self.assertEqual(str(snapshot.error.lastNodeId), "n1")
        self.assertEqual(str(snapshot.error.lastCode), "X_ERR")
        self.assertEqual(str(snapshot.error.lastMessage), "boom")
        self.assertEqual(str(snapshot.error.lastSeverity.value), "warning")
        self.assertEqual(str(snapshot.error.lastFingerprint), "fp-boom")
        self.assertEqual(int(snapshot.error.lastRepeatCount), 1)
        self.assertEqual(str(snapshot.error.currentNodeId), "n1")
        self.assertEqual(str(snapshot.error.currentCode), "X_ERR")
        self.assertEqual(str(snapshot.error.currentMessage), "boom")
        self.assertEqual(str(snapshot.error.currentSeverity.value), "warning")
        self.assertEqual(int(snapshot.error.currentTsMs or 0), ts)
        self.assertFalse(bool(snapshot.gpu.available))
        self.assertEqual(float(snapshot.timing.processMsAvg or 0.0), 12.0)
        self.assertEqual(float(snapshot.timing.latencyMsAvg or 0.0), 25.0)
        self.assertGreaterEqual(float(snapshot.timing.latencyMsP95 or 0.0), 0.0)

        await collector._publish_snapshot(snapshot)
        self.assertEqual(len(bus._transport.published), 1)
        key, raw = bus._transport.published[0]
        self.assertEqual(key, data_key("svcA", from_node_id="svcA", port_id="monitor"))
        envelope = decode_obj(raw)
        self.assertEqual(envelope.get("value", {}).get("schemaVersion"), "f8monitor/1")

    async def test_explicit_timing_does_not_count_processed(self) -> None:
        bus = _FakeBus()
        collector = MonitorCollector(
            bus, MonitorCollectorConfig(enabled=True, interval_ms=200, window_ms=2000, gpu_enabled=False)
        )
        ts = int(now_ms())

        collector.record_timing(port="flow", process_ms=7.5, latency_ms=41.0, ts_ms=ts)

        snapshot = collector._build_snapshot(ts_ms=ts)
        self.assertEqual(int(snapshot.frame.processed), 0)
        self.assertEqual(float(snapshot.timing.processMsAvg or 0.0), 7.5)
        self.assertEqual(float(snapshot.timing.latencyMsAvg or 0.0), 41.0)

    async def test_report_error_repeats_and_clear_current(self) -> None:
        bus = _FakeBus()
        collector = MonitorCollector(
            bus, MonitorCollectorConfig(enabled=True, interval_ms=200, window_ms=2000, gpu_enabled=False)
        )
        ts = int(now_ms())

        collector.report_error(node_id="nodeA", code="E_A", message="first", fingerprint="same", ts_ms=ts)
        collector.report_error(node_id="nodeA", code="E_A", message="first", fingerprint="same", ts_ms=ts + 1)
        snapshot = collector._build_snapshot(ts_ms=ts + 1)

        self.assertEqual(int(snapshot.error.countWindow), 2)
        self.assertEqual(str(snapshot.error.lastFingerprint), "same")
        self.assertEqual(int(snapshot.error.lastRepeatCount), 2)
        self.assertEqual(str(snapshot.error.currentMessage), "first")

        collector.clear_error(node_id="nodeA", fingerprint="same", ts_ms=ts + 2)
        cleared = collector._build_snapshot(ts_ms=ts + 2)

        self.assertEqual(int(cleared.error.countWindow), 2)
        self.assertEqual(str(cleared.error.lastFingerprint), "same")
        self.assertEqual(int(cleared.error.lastRepeatCount), 2)
        self.assertEqual(str(cleared.error.currentNodeId), "")
        self.assertEqual(str(cleared.error.currentMessage), "")
        self.assertIsNone(cleared.error.currentTsMs)

        collector.report_error(node_id="nodeA", code="E_B", message="second", fingerprint="other", ts_ms=ts + 3)
        next_snapshot = collector._build_snapshot(ts_ms=ts + 3)
        self.assertEqual(str(next_snapshot.error.lastFingerprint), "other")
        self.assertEqual(int(next_snapshot.error.lastRepeatCount), 1)

    async def test_monitor_loop_emits_periodically(self) -> None:
        bus = _FakeBus()
        collector = MonitorCollector(
            bus, MonitorCollectorConfig(enabled=True, interval_ms=200, window_ms=2000, gpu_enabled=False)
        )
        collector.record_ready(True)
        await collector.start()
        try:
            await asyncio.sleep(0.45)
        finally:
            await collector.stop()
        self.assertGreaterEqual(len(bus._transport.published), 1)

    async def test_publish_once_records_build_failure(self) -> None:
        bus = _FakeBus()
        bus.data_router = _FlakyDataRouter()
        collector = MonitorCollector(
            bus, MonitorCollectorConfig(enabled=True, interval_ms=200, window_ms=2000, gpu_enabled=False)
        )

        await collector._publish_once()
        snapshot = collector._build_snapshot(ts_ms=int(now_ms()))

        self.assertEqual(str(snapshot.error.lastCode), "MONITOR_BUILD_ERROR")
        self.assertIn("RuntimeError: queue unavailable", str(snapshot.error.lastMessage))

    async def test_publish_once_records_publish_failure(self) -> None:
        bus = _FakeBus()
        bus._transport = _FailingTransport()
        collector = MonitorCollector(
            bus, MonitorCollectorConfig(enabled=True, interval_ms=200, window_ms=2000, gpu_enabled=False)
        )

        await collector._publish_once()
        snapshot = collector._build_snapshot(ts_ms=int(now_ms()))

        self.assertEqual(str(snapshot.error.lastCode), "MONITOR_PUBLISH_ERROR")
        self.assertIn("OSError: publish unavailable", str(snapshot.error.lastMessage))

    async def test_repeated_error_publish_is_throttled_to_summary(self) -> None:
        bus = _FakeBus()
        collector = MonitorCollector(
            bus, MonitorCollectorConfig(enabled=True, interval_ms=200, window_ms=2000, gpu_enabled=False)
        )
        ts = int(now_ms())

        with patch.object(monitoring_module, "_ERROR_REPEAT_PUBLISH_INTERVAL_MS", 50):
            await collector.start()
            try:
                collector.report_error(node_id="nodeA", code="E_A", message="first", fingerprint="same", ts_ms=ts)
                await asyncio.sleep(0.02)
                self.assertEqual(len(bus._transport.published), 1)

                for index in range(2, 51):
                    collector.report_error(
                        node_id="nodeA",
                        code="E_A",
                        message="first",
                        fingerprint="same",
                        ts_ms=ts + index,
                    )
                await asyncio.sleep(0.02)
                self.assertEqual(len(bus._transport.published), 1)

                await asyncio.sleep(0.08)
                self.assertEqual(len(bus._transport.published), 2)
                envelope = decode_obj(bus._transport.published[-1][1])
                error = envelope.get("value", {}).get("error", {})
                self.assertEqual(int(error.get("lastRepeatCount")), 50)
                self.assertEqual(str(error.get("lastFingerprint")), "same")
            finally:
                await collector.stop()

    async def test_clear_error_cancels_pending_repeat_publish(self) -> None:
        bus = _FakeBus()
        collector = MonitorCollector(
            bus, MonitorCollectorConfig(enabled=True, interval_ms=200, window_ms=2000, gpu_enabled=False)
        )
        ts = int(now_ms())

        with patch.object(monitoring_module, "_ERROR_REPEAT_PUBLISH_INTERVAL_MS", 100):
            await collector.start()
            try:
                collector.report_error(node_id="nodeA", code="E_A", message="first", fingerprint="same", ts_ms=ts)
                await asyncio.sleep(0.02)
                collector.report_error(node_id="nodeA", code="E_A", message="first", fingerprint="same", ts_ms=ts + 1)
                collector.clear_error(node_id="nodeA", fingerprint="same", ts_ms=ts + 2)
                await asyncio.sleep(0.02)

                self.assertEqual(len(bus._transport.published), 2)
                envelope = decode_obj(bus._transport.published[-1][1])
                error = envelope.get("value", {}).get("error", {})
                self.assertEqual(str(error.get("currentNodeId") or ""), "")
                self.assertEqual(str(error.get("currentMessage") or ""), "")
            finally:
                await collector.stop()


if __name__ == "__main__":
    unittest.main()
