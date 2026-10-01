import asyncio
import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (PKG_PYDL,):
    if p not in sys.path:
        sys.path.insert(0, p)


from f8pydl.tcnwave_service_node import OnnxTcnWaveServiceNode  # noqa: E402
from f8pysdk.nodes import RuntimeNode  # noqa: E402
from f8pysdk.state import StateRead  # noqa: E402


@dataclass(frozen=True)
class _StateField:
    name: str


@dataclass(frozen=True)
class _NodeStub:
    stateFields: list[_StateField]


class _FakeBus:
    def __init__(self) -> None:
        self.state_values: dict[str, Any] = {}
        self.errors: list[tuple[str, str, str]] = []
        self.emits: list[tuple[str, str, Any, int | None, str | int | None]] = []
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
        self.emits.append((node_id, port, value, ts_ms, ctx_id))

    async def publish_state_runtime(self, node_id: str, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        _ = node_id
        _ = ts_ms
        self.state_values[str(field)] = value

    async def get_state(self, node_id: str, field: str) -> StateRead:
        _ = node_id
        key = str(field)
        if key in self.state_values:
            return StateRead(found=True, value=self.state_values[key], ts_ms=0)
        return StateRead(found=False, value=None, ts_ms=None)

    def get_state_cached(self, node_id: str, field: str, default: Any) -> Any:
        _ = node_id
        return self.state_values.get(str(field), default)

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


class TcnServiceNodeTests(unittest.TestCase):
    def test_missing_default_video_input_requests_zenoh_key(self) -> None:
        async def _run() -> None:
            node = OnnxTcnWaveServiceNode(
                node_id="tcn_node",
                node=_NodeStub(stateFields=[]),
                initial_state={},
                service_class="f8.dl.tcnwave",
                allowed_tasks={"tcn_wave"},
            )
            bus = _FakeBus()
            RuntimeNode.attach(node, bus)
            await node._ensure_config_loaded()
            await node._handle_missing_video_input()
            self.assertEqual(bus.errors[-1][2], "missing video data input")

        asyncio.run(_run())

    def test_output_values_are_float(self) -> None:
        values = OnnxTcnWaveServiceNode._to_float_list([1, 2.5, "3.25"])
        self.assertEqual(values, [1.0, 2.5, 3.25])
        self.assertTrue(all(isinstance(v, float) for v in values))

    def test_runtime_temporal_params_come_from_state(self) -> None:
        async def _run() -> None:
            node = OnnxTcnWaveServiceNode(
                node_id="tcn_node",
                node=_NodeStub(stateFields=[]),
                initial_state={"outputScale": 7.0, "outputBias": -2.5},
                service_class="f8.dl.tcnwave",
                allowed_tasks={"tcn_wave"},
            )
            bus = _FakeBus()
            RuntimeNode.attach(node, bus)
            await node._ensure_config_loaded()
            self.assertEqual(node._output_scale, 7.0)
            self.assertEqual(node._output_bias, -2.5)

        asyncio.run(_run())

    def test_loop_retries_when_runtime_is_reset_after_ensure(self) -> None:
        class _RuntimeResetNode(OnnxTcnWaveServiceNode):
            async def _ensure_config_loaded(self) -> None:
                return None

            async def _ensure_runtime(self) -> bool:
                self._runtime = None
                return True

            def _resolve_video_stream_key(self) -> str:
                return "f8/svc/source/nodes/camera/data/video"

            def _ensure_video_source(self) -> Any:
                raise AssertionError("video source should not be opened without runtime")

        async def _run() -> None:
            node = _RuntimeResetNode(
                node_id="tcn_node",
                node=_NodeStub(stateFields=[]),
                initial_state={},
                service_class="f8.dl.tcnwave",
                allowed_tasks={"tcn_wave"},
            )
            bus = _FakeBus()
            node._bus = bus

            task = asyncio.create_task(node._loop())
            await asyncio.sleep(0.08)
            if task.done():
                task.result()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            self.assertEqual(bus.errors, [])

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
