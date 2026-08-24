from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest.mock import patch

from f8pysdk.specs import F8DataPortSpec, F8RuntimeNode, number_schema

from f8pystudio.operators.viz_wave import OPERATOR_CLASS, VizWaveRuntimeNode
from f8pystudio.studio_specs.identifiers import SERVICE_CLASS


def _runtime_node() -> F8RuntimeNode:
    return F8RuntimeNode(
        nodeId="vizwave1",
        serviceId="svc_studio",
        serviceClass=SERVICE_CLASS,
        operatorClass=OPERATOR_CLASS,
        dataInPorts=[],
        dataOutPorts=[],
        stateFields=[],
        stateValues={},
    )


class _StateUnavailableVizWaveRuntimeNode(VizWaveRuntimeNode):
    async def get_state_value(self, field: str) -> Any:
        raise RuntimeError(f"state store unavailable for {field}")


class _InlineStateVizWaveRuntimeNode(VizWaveRuntimeNode):
    def __init__(
        self,
        *,
        node_id: str,
        node: F8RuntimeNode,
        runtime_state: dict[str, Any],
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(node_id=node_id, node=node, initial_state=initial_state)
        self._runtime_state = dict(runtime_state)

    async def get_state_value(self, field: str) -> Any:
        return self._runtime_state.get(field)


class VizWaveRuntimeNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_config_reads_fall_back_to_initial_state_when_runtime_state_fails(self) -> None:
        runtime = _StateUnavailableVizWaveRuntimeNode(
            node_id="vizwave1",
            node=_runtime_node(),
            initial_state={
                "throttleMs": "250",
                "bufferLimit": "999999",
                "showLegend": 1,
                "minVal": "-2.5",
            },
        )

        with patch("f8pystudio.operators.viz_wave.logger.debug") as debug_log:
            throttle_ms = await runtime._get_int_state("throttleMs", default=100, minimum=0, maximum=60000)
            buffer_limit = await runtime._get_int_state("bufferLimit", default=200, minimum=10, maximum=5000)
            show_legend = await runtime._get_bool_state("showLegend", default=False)
            min_value = await runtime._get_float_state_optional("minVal")

        self.assertEqual(throttle_ms, 250)
        self.assertEqual(buffer_limit, 5000)
        self.assertTrue(show_legend)
        self.assertEqual(min_value, -2.5)
        self.assertEqual(debug_log.call_count, 4)

    async def test_runtime_state_wins_over_initial_state(self) -> None:
        runtime = _InlineStateVizWaveRuntimeNode(
            node_id="vizwave1",
            node=_runtime_node(),
            runtime_state={"windowMs": "1500"},
            initial_state={"windowMs": 100},
        )

        window_ms = await runtime._get_int_state("windowMs", default=10000, minimum=100, maximum=600000)

        self.assertEqual(window_ms, 1500)

    async def test_float_state_rejects_empty_invalid_and_nan_values(self) -> None:
        runtime = _InlineStateVizWaveRuntimeNode(
            node_id="vizwave1",
            node=_runtime_node(),
            runtime_state={"minVal": "", "maxVal": "nan", "badVal": object()},
        )

        self.assertIsNone(await runtime._get_float_state_optional("minVal"))
        self.assertIsNone(await runtime._get_float_state_optional("maxVal"))
        self.assertIsNone(await runtime._get_float_state_optional("badVal"))

    async def test_close_cancels_pending_refresh_task(self) -> None:
        runtime = _InlineStateVizWaveRuntimeNode(
            node_id="vizwave1",
            node=_runtime_node(),
            runtime_state={},
        )
        task = asyncio.create_task(asyncio.sleep(60))
        runtime._refresh_task = task
        runtime._scheduled_refresh_ms = 123

        await runtime.close()

        self.assertIsNone(runtime._refresh_task)
        self.assertIsNone(runtime._scheduled_refresh_ms)
        self.assertTrue(task.cancelled())

    async def test_atomic_axis_frame_is_expanded_and_filtered_by_declared_ports(self) -> None:
        node = _runtime_node()
        node.dataInPorts = [
            F8DataPortSpec(name=axis, description="", valueSchema=number_schema())
            for axis in ("L0", "L1", "L2")
        ]
        runtime = _InlineStateVizWaveRuntimeNode(
            node_id="vizwave1",
            node=node,
            runtime_state={},
            initial_state={"throttleMs": 100},
        )

        with patch("f8pystudio.operators.viz_wave.emit_ui_command"):
            await runtime.on_data(
                "L0",
                {"axes": {"L0": 0.1, "L1": 0.2, "L2": 0.3, "R0": 0.9}},
                ts_ms=123,
            )

        self.assertEqual(runtime._series, {"L0": [(123, 0.1)], "L1": [(123, 0.2)], "L2": [(123, 0.3)]})


if __name__ == "__main__":
    unittest.main()
