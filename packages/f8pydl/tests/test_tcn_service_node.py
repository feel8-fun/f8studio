import asyncio
import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_PYDL, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)


from f8pydl.tcnwave_service_node import OnnxTcnWaveServiceNode  # noqa: E402
from f8pysdk.runtime_node import RuntimeNode  # noqa: E402
from f8pysdk.service_bus.state_read import StateRead  # noqa: E402


@dataclass(frozen=True)
class _StateField:
    name: str


@dataclass(frozen=True)
class _NodeStub:
    stateFields: list[_StateField]


class _FakeBus:
    def __init__(self) -> None:
        self.state_values: dict[str, Any] = {}
        self.emits: list[tuple[str, str, Any, int | None]] = []

    async def emit_data(self, node_id: str, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        self.emits.append((node_id, port, value, ts_ms))

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


class TcnServiceNodeTests(unittest.TestCase):
    def test_missing_shm_name_sets_last_error_and_does_not_crash(self) -> None:
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
            await node._handle_missing_shm_name(now_ms=1234)
            self.assertEqual(bus.state_values.get("lastError"), "missing shmName")

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


if __name__ == "__main__":
    unittest.main()
