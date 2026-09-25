from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, cast

from f8pysdk.specs import (
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    boolean_schema,
    integer_schema,
    number_schema,
    string_schema,
    video_frame_port,
)
from f8pysdk.capabilities import RungraphHookBus
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..identifiers import SERVICE_CLASS
from ..presentation import PresentationOutlet
from .categories import PALETTE_CATEGORY_VIZ


OPERATOR_CLASS = "f8.viz.video"
RENDERER_CLASS = "viz_video"
log = logging.getLogger(__name__)

_STATE_READ_ERRORS = (RuntimeError, OSError, TypeError, ValueError)
_NUMERIC_PARSE_ERRORS = (TypeError, ValueError, OverflowError)


class VizVideoRuntimeNode(OperatorNode):
    """
    Studio visualization node for Zenoh latest-frame video.

    This runtime node sends stream configuration through the presentation outlet;
    frame payloads remain on the media/data transport.
    """

    presentation: PresentationOutlet

    SPEC = F8OperatorSpec(
        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
        serviceClass=SERVICE_CLASS,
        paletteCategory=PALETTE_CATEGORY_VIZ,
        operatorClass=OPERATOR_CLASS,
        version="0.0.1",
        label="Video Viz",
        description="Display frames from Zenoh latest-frame video streams.",
        tags=["ui", "zenoh", "video", "viewer"],
        dataInPorts=[
            video_frame_port(
                name="video",
                description="Input video frame stream.",
                definition_protected=True,
            ),
            video_frame_port(
                name="flow",
                description="Optional dense optical-flow frame stream.",
                definition_protected=False,
            ),
            video_frame_port(
                name="scalar",
                description="Optional scalar metric frame stream.",
                definition_protected=False,
            ),
        ],
        dataOutPorts=[],
        rendererClass=RENDERER_CLASS,
        stateFields=[
            F8StateSpec(
                name="uiUpdate",
                label="UI Update",
                description="Pause/resume embedded viewer updates in the editor.",
                valueSchema=boolean_schema(default=True),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="throttleMs",
                label="Refresh (ms)",
                description="UI refresh interval in milliseconds (0 = as fast as possible).",
                valueSchema=integer_schema(default=33, minimum=0, maximum=60000),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="flowDisplayMode",
                label="Flow Display",
                description="Flow rendering mode: off, hsv, or arrows.",
                valueSchema=string_schema(default="off", enum=["off", "hsv", "arrows"]),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="flowMagScale",
                label="Flow Mag Scale",
                description="Reference max magnitude for HSV/value and arrow scaling.",
                valueSchema=number_schema(default=20.0, minimum=0.1, maximum=500.0),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="flowStride",
                label="Flow Stride",
                description="Sampling stride for arrow rendering.",
                valueSchema=integer_schema(default=12, minimum=2, maximum=128),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="scaleMode",
                label="Scale Mode",
                description="Video scaling mode: native (1:1) or fit.",
                valueSchema=string_schema(default="fit", enum=["native", "fit"]),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="scalarDisplayMode",
                label="Scalar Display",
                description="Scalar rendering mode: off or colormap.",
                valueSchema=string_schema(default="off", enum=["off", "colormap"]),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="scalarColormap",
                label="Scalar Colormap",
                description="Colormap for scalar rendering.",
                valueSchema=string_schema(default="turbo", enum=["gray", "turbo", "viridis", "magma"]),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="scalarRangeMode",
                label="Scalar Range Mode",
                description="Scalar normalization mode: auto or manual.",
                valueSchema=string_schema(default="auto", enum=["auto", "manual"]),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="scalarMin",
                label="Scalar Min",
                description="Manual min for scalar normalization.",
                valueSchema=number_schema(default=-1.0, minimum=-1_000_000_000.0, maximum=1_000_000_000.0),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="scalarMax",
                label="Scalar Max",
                description="Manual max for scalar normalization.",
                valueSchema=number_schema(default=1.0, minimum=-1_000_000_000.0, maximum=1_000_000_000.0),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="scalarAutoPercentileLo",
                label="Scalar Auto Lo %",
                description="Lower percentile for auto scalar normalization.",
                valueSchema=number_schema(default=2.0, minimum=0.0, maximum=100.0),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="scalarAutoPercentileHi",
                label="Scalar Auto Hi %",
                description="Upper percentile for auto scalar normalization.",
                valueSchema=number_schema(default=98.0, minimum=0.0, maximum=100.0),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="scalarInvert",
                label="Scalar Invert",
                description="Invert normalized scalar values before colormap.",
                valueSchema=boolean_schema(default=False),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="scalarNanMode",
                label="Scalar NaN Mode",
                description="NaN/Inf handling for scalar values.",
                valueSchema=string_schema(default="transparent", enum=["transparent", "zero", "min", "max"]),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
        ],
    )

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[],
            data_out_ports=[],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._config_loaded = False
        self._throttle_ms = 33
        self._flow_display_mode = "off"
        self._flow_mag_scale = 20.0
        self._flow_stride = 12
        self._scale_mode = "native"
        self._scalar_display_mode = "off"
        self._scalar_colormap = "turbo"
        self._scalar_range_mode = "auto"
        self._scalar_min = -1.0
        self._scalar_max = 1.0
        self._scalar_auto_percentile_lo = 2.0
        self._scalar_auto_percentile_hi = 98.0
        self._scalar_invert = False
        self._scalar_nan_mode = "transparent"
        self._pending_task: asyncio.Task[object] | None = None
        self._rungraph_bus: RungraphHookBus | None = None

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        rungraph_bus = cast(RungraphHookBus, bus)
        rungraph_bus.register_rungraph_hook(self)
        self._rungraph_bus = rungraph_bus
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._ensure_config_loaded(), name=f"pystudio:video:init:{self.node_id}")
        except RuntimeError:
            log.debug("viz video config init deferred; no running event loop node_id=%s", self.node_id, exc_info=True)

    async def close(self) -> None:
        try:
            rungraph_bus = self._rungraph_bus
            self._rungraph_bus = None
            if rungraph_bus is not None:
                rungraph_bus.unregister_rungraph_hook(self)
            t = self._pending_task
            self._pending_task = None
            if t is not None:
                t.cancel()
                await asyncio.gather(t, return_exceptions=True)
        except (RuntimeError, TypeError, ValueError):
            log.debug("viz video node close cleanup failed node_id=%s", self.node_id, exc_info=True)
        self.presentation.emit(self.node_id, "viz.video.detach", {}, ts_ms=int(time.time() * 1000))

    async def validate_rungraph(self, graph: F8RuntimeGraph) -> None:
        del graph

    async def on_rungraph(self, graph: F8RuntimeGraph) -> None:
        del graph
        await self._ensure_config_loaded()
        await self._push_config(now_ms=int(time.time() * 1000))

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        f = str(field or "").strip()
        if f not in (
            "throttleMs",
            "flowDisplayMode",
            "flowMagScale",
            "flowStride",
            "scaleMode",
            "scalarDisplayMode",
            "scalarColormap",
            "scalarRangeMode",
            "scalarMin",
            "scalarMax",
            "scalarAutoPercentileLo",
            "scalarAutoPercentileHi",
            "scalarInvert",
            "scalarNanMode",
        ):
            return
        await self._ensure_config_loaded()
        if f == "throttleMs":
            self._throttle_ms = await self._get_int_state("throttleMs", default=self._throttle_ms, minimum=0, maximum=60000)
        elif f == "flowDisplayMode":
            mode = str(await self._get_str_state("flowDisplayMode", default=self._flow_display_mode)).strip().lower()
            self._flow_display_mode = mode if mode in ("off", "hsv", "arrows") else "off"
        elif f == "flowMagScale":
            self._flow_mag_scale = await self._get_float_state("flowMagScale", default=self._flow_mag_scale, minimum=0.1, maximum=500.0)
        elif f == "flowStride":
            self._flow_stride = await self._get_int_state("flowStride", default=self._flow_stride, minimum=2, maximum=128)
        elif f == "scaleMode":
            mode = str(await self._get_str_state("scaleMode", default=self._scale_mode)).strip().lower()
            self._scale_mode = mode if mode in ("native", "fit") else "native"
        elif f == "scalarDisplayMode":
            mode = str(await self._get_str_state("scalarDisplayMode", default=self._scalar_display_mode)).strip().lower()
            self._scalar_display_mode = self._normalize_scalar_display_mode(mode)
        elif f == "scalarColormap":
            cmap = str(await self._get_str_state("scalarColormap", default=self._scalar_colormap)).strip().lower()
            self._scalar_colormap = self._normalize_scalar_colormap(cmap)
        elif f == "scalarRangeMode":
            mode = str(await self._get_str_state("scalarRangeMode", default=self._scalar_range_mode)).strip().lower()
            self._scalar_range_mode = self._normalize_scalar_range_mode(mode)
        elif f == "scalarMin":
            self._scalar_min = await self._get_float_state(
                "scalarMin",
                default=self._scalar_min,
                minimum=-1_000_000_000.0,
                maximum=1_000_000_000.0,
            )
        elif f == "scalarMax":
            self._scalar_max = await self._get_float_state(
                "scalarMax",
                default=self._scalar_max,
                minimum=-1_000_000_000.0,
                maximum=1_000_000_000.0,
            )
        elif f == "scalarAutoPercentileLo":
            self._scalar_auto_percentile_lo = await self._get_float_state(
                "scalarAutoPercentileLo",
                default=self._scalar_auto_percentile_lo,
                minimum=0.0,
                maximum=100.0,
            )
        elif f == "scalarAutoPercentileHi":
            self._scalar_auto_percentile_hi = await self._get_float_state(
                "scalarAutoPercentileHi",
                default=self._scalar_auto_percentile_hi,
                minimum=0.0,
                maximum=100.0,
            )
        elif f == "scalarInvert":
            self._scalar_invert = await self._get_bool_state("scalarInvert", default=self._scalar_invert)
        elif f == "scalarNanMode":
            mode = str(await self._get_str_state("scalarNanMode", default=self._scalar_nan_mode)).strip().lower()
            self._scalar_nan_mode = self._normalize_scalar_nan_mode(mode)
        await self._push_config(now_ms=int(ts_ms) if ts_ms is not None else int(time.time() * 1000))

    async def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return
        self._throttle_ms = await self._get_int_state("throttleMs", default=33, minimum=0, maximum=60000)
        flow_mode = (await self._get_str_state("flowDisplayMode", default="off")).strip().lower()
        self._flow_display_mode = flow_mode if flow_mode in ("off", "hsv", "arrows") else "off"
        self._flow_mag_scale = await self._get_float_state("flowMagScale", default=20.0, minimum=0.1, maximum=500.0)
        self._flow_stride = await self._get_int_state("flowStride", default=12, minimum=2, maximum=128)
        mode = (await self._get_str_state("scaleMode", default="native")).strip().lower()
        self._scale_mode = mode if mode in ("native", "fit") else "native"
        scalar_display = (await self._get_str_state("scalarDisplayMode", default="off")).strip().lower()
        self._scalar_display_mode = self._normalize_scalar_display_mode(scalar_display)
        scalar_colormap = (await self._get_str_state("scalarColormap", default="turbo")).strip().lower()
        self._scalar_colormap = self._normalize_scalar_colormap(scalar_colormap)
        scalar_range_mode = (await self._get_str_state("scalarRangeMode", default="auto")).strip().lower()
        self._scalar_range_mode = self._normalize_scalar_range_mode(scalar_range_mode)
        self._scalar_min = await self._get_float_state("scalarMin", default=-1.0, minimum=-1_000_000_000.0, maximum=1_000_000_000.0)
        self._scalar_max = await self._get_float_state("scalarMax", default=1.0, minimum=-1_000_000_000.0, maximum=1_000_000_000.0)
        self._scalar_auto_percentile_lo = await self._get_float_state(
            "scalarAutoPercentileLo",
            default=2.0,
            minimum=0.0,
            maximum=100.0,
        )
        self._scalar_auto_percentile_hi = await self._get_float_state(
            "scalarAutoPercentileHi",
            default=98.0,
            minimum=0.0,
            maximum=100.0,
        )
        self._scalar_invert = await self._get_bool_state(
            "scalarInvert",
            default=False,
        )
        scalar_nan_mode = (await self._get_str_state("scalarNanMode", default="transparent")).strip().lower()
        self._scalar_nan_mode = self._normalize_scalar_nan_mode(scalar_nan_mode)
        self._config_loaded = True
        await self._push_config(now_ms=int(time.time() * 1000))

    async def _push_config(self, *, now_ms: int) -> None:
        if self._pending_task is not None and not self._pending_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._pending_task = loop.create_task(self._push_config_async(now_ms), name=f"pystudio:video:cfg:{self.node_id}")

    async def _push_config_async(self, now_ms: int) -> None:
        video_stream_key = str(self.input_zenoh_key("video") or "").strip()
        flow_stream_key = str(self.input_zenoh_key("flow") or "").strip()
        scalar_stream_key = str(self.input_zenoh_key("scalar") or "").strip()
        payload: dict[str, object] = {
            "videoStreamKey": video_stream_key,
            "throttleMs": int(self._throttle_ms),
            "flowStreamKey": flow_stream_key,
            "flowDisplayMode": str(self._flow_display_mode or "off"),
            "flowMagScale": float(self._flow_mag_scale),
            "flowStride": int(self._flow_stride),
            "scaleMode": str(self._scale_mode or "native"),
            "scalarStreamKey": scalar_stream_key,
            "scalarDisplayMode": self._normalize_scalar_display_mode(self._scalar_display_mode),
            "scalarColormap": self._normalize_scalar_colormap(self._scalar_colormap),
            "scalarRangeMode": self._normalize_scalar_range_mode(self._scalar_range_mode),
            "scalarMin": float(self._scalar_min),
            "scalarMax": float(self._scalar_max),
            "scalarAutoPercentileLo": float(self._scalar_auto_percentile_lo),
            "scalarAutoPercentileHi": float(self._scalar_auto_percentile_hi),
            "scalarInvert": bool(self._scalar_invert),
            "scalarNanMode": self._normalize_scalar_nan_mode(self._scalar_nan_mode),
        }
        self.presentation.emit(
            self.node_id,
            "viz.video.set",
            payload,
            ts_ms=int(now_ms),
        )

    async def _get_int_state(self, name: str, *, default: int, minimum: int, maximum: int) -> int:
        v = await self._config_state_value(name)
        try:
            out = int(v) if v is not None else int(default)
        except _NUMERIC_PARSE_ERRORS:
            out = int(default)
        if out < minimum:
            out = minimum
        if out > maximum:
            out = maximum
        return out

    async def _get_str_state(self, name: str, *, default: str) -> str:
        v = await self._config_state_value(name)
        return str(v) if v is not None else str(default)

    async def _get_bool_state(self, name: str, *, default: bool) -> bool:
        v = await self._config_state_value(name, default=default)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        text = str(v or "").strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        return bool(default)

    async def _get_float_state(self, name: str, *, default: float, minimum: float, maximum: float) -> float:
        v = await self._config_state_value(name)
        try:
            out = float(v) if v is not None else float(default)
        except _NUMERIC_PARSE_ERRORS:
            out = float(default)
        if out < minimum:
            out = minimum
        if out > maximum:
            out = maximum
        return out

    async def _config_state_value(self, name: str, *, default: Any = None) -> Any:
        try:
            value = await self.get_state_value(name)
        except _STATE_READ_ERRORS:
            log.debug(
                "viz video state read failed; falling back to initial state node_id=%s field=%s",
                self.node_id,
                name,
                exc_info=True,
            )
            value = None
        if value is not None:
            return value
        return self._initial_state.get(name, default)

    @staticmethod
    def _normalize_scalar_display_mode(mode: str) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized in ("off", "colormap"):
            return normalized
        return "off"

    @staticmethod
    def _normalize_scalar_colormap(colormap: str) -> str:
        normalized = str(colormap or "").strip().lower()
        if normalized in ("gray", "turbo", "viridis", "magma"):
            return normalized
        return "turbo"

    @staticmethod
    def _normalize_scalar_range_mode(mode: str) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized in ("auto", "manual"):
            return normalized
        return "auto"

    @staticmethod
    def _normalize_scalar_nan_mode(mode: str) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized in ("transparent", "zero", "min", "max"):
            return normalized
        return "transparent"


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(VizVideoRuntimeNode.SPEC, VizVideoRuntimeNode, overwrite=True)
    return registry
