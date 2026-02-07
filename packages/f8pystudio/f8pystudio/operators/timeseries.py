from __future__ import annotations

import asyncio
import time
from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    boolean_schema,
    integer_schema,
    number_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS
from ..color_table import series_colors
from ..ui_bus import emit_ui_command


OPERATOR_CLASS = "f8.timeseries"
RENDERER_CLASS = "pystudio_timeseries"


class PyStudioTimeSeriesRuntimeNode(RuntimeNode):
    """
    Studio-side runtime node for time-series plotting.

    Pulls numeric input values, records timestamps, and publishes a UI command
    containing a bounded buffer of points.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._refresh_task: asyncio.Task[object] | None = None
        self._config_loaded = False
        self._series: dict[str, list[tuple[int, float]]] = {}
        self._last_refresh_ms: int | None = None
        self._dirty: bool = False
        self._throttle_ms: int = 100
        self._window_ms: int = 10000
        self._buffer_limit: int = 200
        self._show_legend: bool = False
        self._y_min: float | None = None
        self._y_max: float | None = None
        self._scheduled_refresh_ms: int | None = None

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        return

    async def close(self) -> None:
        try:
            t = self._refresh_task
            self._refresh_task = None
            self._scheduled_refresh_ms = None
            if t is None:
                return
            t.cancel()
        except Exception:
            pass
        try:
            await asyncio.gather(t, return_exceptions=True)
        except Exception:
            pass

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        # Timeseries supports arbitrary editable data-in ports.
        port = str(port or "").strip()
        if not port:
            return
        await self._ensure_config_loaded()
        try:
            val = float(value)
        except Exception:
            return
        ts = int(ts_ms) if ts_ms is not None else int(time.time() * 1000)
        buf = self._series.get(port)
        if buf is None:
            buf = []
            self._series[port] = buf
        buf.append((ts, val))
        self._dirty = True
        self._prune_points(window_ms=self._window_ms, buffer_limit=self._buffer_limit, now_ms=ts)
        await self._schedule_refresh(now_ms=ts)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        f = str(field or "").strip()
        if f == "throttleMs":
            self._throttle_ms = await self._get_int_state("throttleMs", default=100, minimum=0, maximum=60000)
        elif f == "windowMs":
            self._window_ms = await self._get_int_state("windowMs", default=10000, minimum=100, maximum=600000)
        elif f == "bufferLimit":
            self._buffer_limit = await self._get_int_state("bufferLimit", default=200, minimum=10, maximum=5000)
        elif f == "showLegend":
            self._show_legend = await self._get_bool_state("showLegend", default=False)
        elif f == "minVal":
            self._y_min = await self._get_float_state_optional("minVal")
        elif f == "maxVal":
            self._y_max = await self._get_float_state_optional("maxVal")
        else:
            return
        await self._schedule_refresh(now_ms=int(ts_ms) if ts_ms is not None else int(time.time() * 1000))

    async def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return
        self._throttle_ms = await self._get_int_state("throttleMs", default=100, minimum=0, maximum=60000)
        self._window_ms = await self._get_int_state("windowMs", default=10000, minimum=100, maximum=600000)
        self._buffer_limit = await self._get_int_state("bufferLimit", default=200, minimum=10, maximum=5000)
        self._show_legend = await self._get_bool_state("showLegend", default=False)
        self._y_min = await self._get_float_state_optional("minVal")
        self._y_max = await self._get_float_state_optional("maxVal")
        self._config_loaded = True

    async def _schedule_refresh(self, *, now_ms: int) -> None:
        throttle_ms = max(0, int(self._throttle_ms))
        last_refresh = int(self._last_refresh_ms or 0)
        if throttle_ms <= 0 or last_refresh <= 0:
            await self._flush(now_ms=now_ms)
            return

        target_ms = last_refresh + throttle_ms
        if int(now_ms) >= int(target_ms):
            await self._flush(now_ms=now_ms)
            return

        if self._refresh_task is not None and not self._refresh_task.done():
            return

        delay_ms = max(0, int(target_ms) - int(now_ms))
        self._scheduled_refresh_ms = int(target_ms)
        try:
            loop = asyncio.get_running_loop()
        except Exception:
            return
        self._refresh_task = loop.create_task(self._flush_after(delay_ms), name=f"pystudio:timeseries:flush:{self.node_id}")

    async def _flush_after(self, delay_ms: int) -> None:
        try:
            await asyncio.sleep(float(max(0, int(delay_ms))) / 1000.0)
        except Exception:
            return
        await self._flush(now_ms=int(time.time() * 1000))

    async def _flush(self, *, now_ms: int) -> None:
        self._scheduled_refresh_ms = None
        changed = False
        if self._prune_points(window_ms=self._window_ms, buffer_limit=self._buffer_limit, now_ms=int(now_ms)):
            changed = True
        if self._dirty:
            changed = True

        any_points = any(bool(v) for v in (self._series or {}).values())
        if changed or any_points:
            preferred = list(self.data_in_ports or [])
            keys = list(self._series.keys())
            ordered_keys = [k for k in preferred if k in self._series] + [k for k in keys if k not in set(preferred)]
            colors = series_colors(ordered_keys)
            emit_ui_command(
                self.node_id,
                "timeseries.set",
                {
                    "series": {k: list(v) for k, v in (self._series or {}).items() if v},
                    "colors": {k: list(rgb) for k, rgb in colors.items()},
                    "windowMs": int(self._window_ms),
                    "nowMs": int(now_ms),
                    "showLegend": bool(self._show_legend),
                    "minVal": self._y_min,
                    "maxVal": self._y_max,
                },
                ts_ms=int(now_ms),
            )

        self._last_refresh_ms = int(now_ms)
        self._dirty = False

    async def _get_int_state(self, name: str, *, default: int, minimum: int, maximum: int) -> int:
        v = None
        try:
            v = await self.get_state_value(name)
        except Exception:
            v = None
        if v is None:
            try:
                v = self._initial_state.get(name)
            except Exception:
                v = None

        try:
            out = int(v) if v is not None else int(default)
        except Exception:
            out = int(default)
        if out < minimum:
            out = minimum
        if out > maximum:
            out = maximum
        return out

    async def _get_bool_state(self, name: str, *, default: bool) -> bool:
        v = None
        try:
            v = await self.get_state_value(name)
        except Exception:
            v = None
        if v is None:
            try:
                v = self._initial_state.get(name)
            except Exception:
                v = None
        try:
            return bool(v) if v is not None else bool(default)
        except Exception:
            return bool(default)

    async def _get_float_state_optional(self, name: str) -> float | None:
        v = None
        try:
            v = await self.get_state_value(name)
        except Exception:
            v = None
        if v is None:
            try:
                v = self._initial_state.get(name)
            except Exception:
                v = None
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        try:
            out = float(v)
        except Exception:
            return None
        if out != out:  # NaN
            return None
        return out

    def _prune_points(self, *, window_ms: int, buffer_limit: int, now_ms: int) -> bool:
        changed = False
        if not self._series:
            return False
        cutoff = int(now_ms) - int(window_ms) if window_ms > 0 else None
        for k in list(self._series.keys()):
            pts = self._series.get(k) or []
            n0 = len(pts)
            if cutoff is not None and pts:
                pts = [(ts, v) for (ts, v) in pts if int(ts) >= int(cutoff)]
            if buffer_limit > 0 and len(pts) > buffer_limit:
                pts = pts[-int(buffer_limit) :]
            if len(pts) != n0:
                changed = True
            if pts:
                self._series[k] = pts
            else:
                # Drop empty series to keep payload small and allow UI to remove curves.
                self._series.pop(k, None)
        return changed


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return PyStudioTimeSeriesRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=SERVICE_CLASS,
            operatorClass=OPERATOR_CLASS,
            version="0.0.1",
            label="Time Series Plot",
            description="Plot numeric values over time (UI-only).",
            tags=["plot", "timeseries", "ui"],
            dataInPorts=[
                F8DataPortSpec(
                    name="value",
                    description="Numeric input value (y-axis).",
                    valueSchema=number_schema(),
                ),
            ],
            dataOutPorts=[],
            editableDataInPorts=True,
            rendererClass=RENDERER_CLASS,
            stateFields=[
                F8StateSpec(
                    name="bufferLimit",
                    label="Buffer Limit",
                    description="Maximum number of points kept in memory.",
                    valueSchema=integer_schema(default=200, minimum=10, maximum=5000),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="windowMs",
                    label="Time Window (ms)",
                    description="Only keep data within this time window.",
                    valueSchema=integer_schema(default=10000, minimum=100, maximum=600000),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="throttleMs",
                    label="Refresh (ms)",
                    description="UI refresh interval in milliseconds.",
                    valueSchema=integer_schema(default=100, minimum=0, maximum=60000),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="showLegend",
                    label="Legend",
                    description="Toggle plot legend visibility.",
                    valueSchema=boolean_schema(default=False),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="minVal",
                    label="Min",
                    description="Fixed y-axis minimum (leave empty for auto).",
                    valueSchema=number_schema(default=None),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="maxVal",
                    label="Max",
                    description="Fixed y-axis maximum (leave empty for auto).",
                    valueSchema=number_schema(default=None),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
            ],
        ),
        overwrite=True,
    )
    return reg
