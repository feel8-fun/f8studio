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
    integer_schema,
    number_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS
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
        self._task: asyncio.Task[object] | None = None
        self._points: list[tuple[int, float]] = []
        self._data_event: asyncio.Event | None = None
        self._last_refresh_ms: int | None = None
        self._dirty: bool = False
        self._window_ms: int = 10000
        self._buffer_limit: int = 200

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        if self._task is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except Exception:
            return
        self._data_event = asyncio.Event()
        add_listener = getattr(bus, "add_data_listener", None)
        if callable(add_listener):
            try:
                add_listener(self.node_id, "value", self._on_data)
            except Exception:
                pass
        self._task = loop.create_task(self._run(), name=f"pystudio:timeseries:{self.node_id}")

    async def close(self) -> None:
        bus = self._bus
        if bus is not None:
            remove_listener = getattr(bus, "remove_data_listener", None)
            if callable(remove_listener):
                try:
                    remove_listener(self.node_id, "value", self._on_data)
                except Exception:
                    pass
        t = self._task
        self._task = None
        if t is None:
            return
        try:
            t.cancel()
        except Exception:
            pass
        try:
            await asyncio.gather(t, return_exceptions=True)
        except Exception:
            pass

    async def _on_data(self, _node_id: str, _port: str, value: Any, ts_ms: int) -> None:
        try:
            val = float(value)
        except Exception:
            return
        ts = int(ts_ms) if ts_ms is not None else int(time.time() * 1000)
        self._points.append((ts, val))
        self._dirty = True
        self._prune_points(window_ms=self._window_ms, buffer_limit=self._buffer_limit, now_ms=ts)
        ev = self._data_event
        if ev is not None:
            ev.set()

    async def _run(self) -> None:
        while True:
            ev = self._data_event
            if ev is None:
                await asyncio.sleep(0.05)
                continue
            await ev.wait()
            ev.clear()

            throttle_ms = await self._get_int_state("throttleMs", default=100, minimum=0, maximum=60000)
            window_ms = await self._get_int_state("windowMs", default=10000, minimum=100, maximum=600000)
            buffer_limit = await self._get_int_state("bufferLimit", default=200, minimum=10, maximum=5000)
            self._window_ms = int(window_ms)
            self._buffer_limit = int(buffer_limit)

            now_ms = int(time.time() * 1000)
            last_refresh = self._last_refresh_ms or 0
            if throttle_ms > 0 and last_refresh > 0:
                wait_ms = max(0, (last_refresh + int(throttle_ms)) - now_ms)
                if wait_ms > 0:
                    await asyncio.sleep(float(wait_ms) / 1000.0)

            changed = False
            now_ms = int(time.time() * 1000)

            if self._prune_points(window_ms=window_ms, buffer_limit=buffer_limit, now_ms=now_ms):
                changed = True
            if self._dirty:
                changed = True

            if changed or self._points:
                emit_ui_command(
                    self.node_id,
                    "timeseries.set",
                    {"points": list(self._points), "windowMs": int(window_ms)},
                    ts_ms=now_ms,
                )

            self._last_refresh_ms = int(now_ms)
            self._dirty = False

    async def _get_int_state(self, name: str, *, default: int, minimum: int, maximum: int) -> int:
        v = None
        try:
            v = await self.get_state(name)
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

    def _prune_points(self, *, window_ms: int, buffer_limit: int, now_ms: int) -> bool:
        changed = False
        if window_ms > 0 and self._points:
            cutoff = int(now_ms) - int(window_ms)
            n0 = len(self._points)
            self._points = [(ts, v) for (ts, v) in self._points if ts >= cutoff]
            if len(self._points) != n0:
                changed = True
        if buffer_limit > 0 and len(self._points) > buffer_limit:
            self._points = self._points[-int(buffer_limit) :]
            changed = True
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
            rendererClass=RENDERER_CLASS,
            stateFields=[
                F8StateSpec(
                    name="bufferLimit",
                    label="Buffer Limit",
                    description="Maximum number of points kept in memory.",
                    valueSchema=integer_schema(default=200, minimum=10, maximum=5000),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="windowMs",
                    label="Time Window (ms)",
                    description="Only keep data within this time window.",
                    valueSchema=integer_schema(default=10000, minimum=100, maximum=600000),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="throttleMs",
                    label="Refresh (ms)",
                    description="UI refresh interval in milliseconds.",
                    valueSchema=integer_schema(default=100, minimum=0, maximum=60000),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
            ],
        ),
        overwrite=True,
    )
    return reg
