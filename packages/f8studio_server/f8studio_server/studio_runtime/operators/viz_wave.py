from __future__ import annotations

from ._throttled_flusher import ThrottledFlusher

from ..presentation import PresentationOutlet

from ...presentation_models import WaveScene

import msgspec

import logging
import time
from typing import Any

from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8SpecEditPolicy,
    F8StateAccess,
    F8StateSpec,
    boolean_schema,
    editable_collection_edit_policy,
    integer_schema,
    number_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.registry import Registry

from ..identifiers import SERVICE_CLASS
from ..visualization.colors import series_colors
from .categories import PALETTE_CATEGORY_VIZ
from ._viz_base import StudioVizRuntimeNodeBase, viz_sampling_state_fields


OPERATOR_CLASS = "f8.viz.wave"
RENDERER_CLASS = "viz_wave"

logger = logging.getLogger(__name__)



class VizWaveRuntimeNode(StudioVizRuntimeNodeBase):
    """
    Studio-side runtime node for time-series plotting.

    Pulls numeric input values, records timestamps, and publishes a presentation command
    containing a bounded buffer of points.
    """

    def __init__(
        self,
        *,
        node_id: str,
        node: F8RuntimeNode,
        initial_state: dict[str, Any] | None = None,
        presentation: PresentationOutlet | None = None,
    ) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[],
            state_fields=[s.name for s in (node.stateFields or [])],
            initial_state=initial_state,
            presentation=presentation,
        )
        self._flusher = ThrottledFlusher(self._flush, name=f"viz:wave:{self.node_id}")
        self._config_loaded = False
        self._series: dict[str, list[tuple[int, float]]] = {}
        self._dirty: bool = False
        self._throttle_ms: int = 100
        self._window_ms: int = 10000
        self._buffer_limit: int = 200
        self._show_legend: bool = False
        self._y_min: float | None = None
        self._y_max: float | None = None

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        return

    async def close(self) -> None:
        await self._flusher.close()
        self.presentation.emit(self.node_id, "viz.wave.detach", {}, ts_ms=int(time.time() * 1000))

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        # Timeseries supports arbitrary editable data-in ports.
        port = str(port or "").strip()
        if not port:
            return
        await self._ensure_config_loaded()
        try:
            val = float(value)
        except (TypeError, ValueError):
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
        elif f == "clearNonce":
            await self._get_int_state("clearNonce", default=0, minimum=0, maximum=2147483647)
            self._series.clear()
            self._dirty = True
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

    async def _schedule_refresh(self, now_ms: int) -> None:
        await self._flusher.schedule(now_ms=now_ms, throttle_ms=self._throttle_ms)

    async def _flush(self, now_ms: int) -> None:
        changed = False
        if self._prune_points(window_ms=self._window_ms, buffer_limit=self._buffer_limit, now_ms=int(now_ms)):
            changed = True
        if self._dirty:
            changed = True

        any_points = any(bool(v) for v in (self._series or {}).values())
        if changed or any_points:
            preferred = list(self.data_in_ports or [])
            keys = list(self._series.keys())
            preferred_set = set(preferred)
            unknown_order = [k for k in keys if k not in preferred_set]
            color_order = preferred + unknown_order
            colors = series_colors(color_order)
            self.presentation.emit(
                self.node_id,
                "viz.wave.set",
                msgspec.to_builtins(
                    WaveScene(
                        series={k: list(v) for k, v in (self._series or {}).items() if v},
                        colors={k: list(rgb) for k, rgb in colors.items()},
                        windowMs=int(self._window_ms),
                        nowMs=int(now_ms),
                        showLegend=bool(self._show_legend),
                        minVal=self._y_min,
                        maxVal=self._y_max,
                    )
                ),
                ts_ms=int(now_ms),
            )

        self._dirty = False

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


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=SERVICE_CLASS,
            paletteCategory=PALETTE_CATEGORY_VIZ,
            operatorClass=OPERATOR_CLASS,
            version="0.0.1",
            label="Wave Viz",
            description="Plot numeric values over time (UI-only).",
            tags=["plot", "timeseries", "ui"],
            dataInPorts=[
                F8DataPortSpec(
                    name="x",
                    description="Numeric input value (y-axis).",
                    valueSchema=number_schema(),
                ),
                F8DataPortSpec(
                    name="y",
                    description="Numeric input value (y-axis).",
                    valueSchema=number_schema(),
                    showOnNode=False,
                ),
                F8DataPortSpec(
                    name="z",
                    description="Numeric input value (y-axis).",
                    valueSchema=number_schema(),
                    showOnNode=False,
                ),
            ],
            dataOutPorts=[],
            editPolicy=F8SpecEditPolicy(dataInPorts=editable_collection_edit_policy()),
            rendererClass=RENDERER_CLASS,
            stateFields=[
                F8StateSpec(
                    name="uiUpdate",
                    label="UI Update",
                    description="Pause/resume embedded chart updates in the editor.",
                    valueSchema=boolean_schema(default=True),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="bufferLimit",
                    label="Buffer Limit",
                    description="Maximum number of points kept in memory.",
                    valueSchema=integer_schema(default=200, minimum=10, maximum=5000),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="clearNonce",
                    label="Clear Nonce",
                    description="Increment to clear accumulated series buffer.",
                    valueSchema=integer_schema(default=0, minimum=0, maximum=2147483647),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="windowMs",
                    label="Time Window (ms)",
                    description="Only keep data within this time window.",
                    valueSchema=integer_schema(default=10000, minimum=100, maximum=600000),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="throttleMs",
                    label="Refresh (ms)",
                    description="UI refresh interval in milliseconds.",
                    valueSchema=integer_schema(default=100, minimum=0, maximum=60000),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="showLegend",
                    label="Legend",
                    description="Toggle plot legend visibility.",
                    valueSchema=boolean_schema(default=False),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="minVal",
                    label="Min",
                    description="Fixed y-axis minimum (leave empty for auto).",
                    valueSchema=number_schema(default=None),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="maxVal",
                    label="Max",
                    description="Fixed y-axis maximum (leave empty for auto).",
                    valueSchema=number_schema(default=None),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
                *viz_sampling_state_fields(show_on_node=False),
            ],
        ),
        VizWaveRuntimeNode,
        overwrite=True,
    )
    return registry
