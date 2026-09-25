from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator, interp1d

from f8pysdk.specs import (
    F8UiControlKind,
    F8UiControlSpec,
    F8ArrayTypeSchema,
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8SpecEditPolicy,
    F8StateAccess,
    F8StateSpec,
    editable_collection_edit_policy,
    number_schema,
    string_schema,
)
from f8pysdk.specs import UNSET
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry
from f8pysdk.specs import array_schema, number_schema as helper_number_schema

from ..constants import SERVICE_CLASS
from ._runtime_errors import OPERATOR_EVAL_ERRORS, OPERATOR_STATE_PUBLISH_ERRORS
from .wave_loop_sampler import LoopingLinearSampler

OPERATOR_CLASS = "f8.wave_funscript"
_TOPLEVEL_AXIS = "TopLevel"
_DEFAULT_MAX_T = 10.0
_DEFAULT_INTERP = "linear"
_INTERP_METHODS = ("linear", "pchip", "akima", "cubic_spline")
_HEATMAP_BINS = 128
_DEFAULT_HEATMAP = [0.0] * _HEATMAP_BINS

logger = logging.getLogger(__name__)


def _schema_default_value(schema: Any) -> Any:
    if isinstance(schema, dict):
        return schema.get("default")
    try:
        default_value = schema.default
    except AttributeError:
        return None
    if default_value is UNSET:
        return None
    return default_value


def _to_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _coerce_max_t(value: Any) -> float:
    numeric = _to_float_or_none(value)
    if numeric is None:
        raise ValueError("maxT must be numeric")
    if numeric <= 0.0:
        raise ValueError("maxT must be > 0")
    return float(numeric)


def _coerce_path(value: Any) -> str:
    return str(value or "").strip()


def _coerce_axis(value: Any) -> str:
    return str(value or "").strip() or _TOPLEVEL_AXIS


def _coerce_interp(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text not in _INTERP_METHODS:
        raise ValueError(f"interp must be one of: {', '.join(_INTERP_METHODS)}")
    return text


def _serialize_points(points: list[tuple[float, float]]) -> list[list[float]]:
    return [[float(t_value), float(y_value)] for t_value, y_value in points]


def _normalize_actions(actions: Any) -> list[tuple[float, float]]:
    if not isinstance(actions, list):
        return []
    deduped: dict[float, tuple[int, float]] = {}
    for index, item in enumerate(actions):
        if not isinstance(item, dict):
            continue
        at_ms = _to_float_or_none(item.get("at"))
        pos_value = _to_float_or_none(item.get("pos"))
        if at_ms is None or pos_value is None:
            continue
        time_sec = float(at_ms) / 1000.0
        pos_norm = float(pos_value) / 100.0
        deduped[time_sec] = (index, pos_norm)
    ordered = sorted(deduped.items(), key=lambda item: item[0])
    return [(float(t_value), float(index_and_value[1])) for t_value, index_and_value in ordered]


def _parse_duration_time_seconds(raw_value: Any) -> float | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 3:
        return None
    hours_value = _to_float_or_none(parts[0])
    minutes_value = _to_float_or_none(parts[1])
    seconds_value = _to_float_or_none(parts[2])
    if hours_value is None or minutes_value is None or seconds_value is None:
        return None
    return float(hours_value * 3600.0 + minutes_value * 60.0 + seconds_value)


def _fallback_duration_from_points(points: list[tuple[float, float]]) -> float:
    if not points:
        return _DEFAULT_MAX_T
    return max(_DEFAULT_MAX_T, float(points[-1][0]))


def _load_funscript_file(file_path: str) -> dict[str, Any]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"funscript file does not exist: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("funscript root must be a JSON object")
    return raw


def _make_constant_model(value: float) -> Callable[[np.ndarray], np.ndarray]:
    constant_value = float(value)

    def _evaluate(xs: np.ndarray) -> np.ndarray:
        return np.full(xs.shape, constant_value, dtype=np.float64)

    return _evaluate


def _make_linear_model_from_active_points(
    active_points: list[tuple[float, float]], *, max_t: float
) -> Callable[[np.ndarray], np.ndarray]:
    if len(active_points) == 0:
        return _make_constant_model(0.0)
    if len(active_points) == 1:
        return _make_constant_model(active_points[0][1])

    t_values = np.asarray([point[0] for point in active_points], dtype=np.float64)
    y_values = np.asarray([point[1] for point in active_points], dtype=np.float64)
    if t_values[0] > 0.0:
        t_values = np.concatenate(([0.0], t_values))
        y_values = np.concatenate(([y_values[-1]], y_values))
    if t_values[-1] < float(max_t):
        t_values = np.concatenate((t_values, [float(max_t)]))
        y_values = np.concatenate((y_values, [y_values[0]]))

    linear = interp1d(t_values, y_values, kind="linear", bounds_error=False, fill_value="extrapolate", assume_sorted=True)

    def _evaluate(xs: np.ndarray) -> np.ndarray:
        return np.asarray(linear(xs), dtype=np.float64)

    return _evaluate


def _make_periodic_model_from_active_points(
    active_points: list[tuple[float, float]], *, method: str, max_t: float
) -> Callable[[np.ndarray], np.ndarray]:
    if len(active_points) == 0:
        return _make_constant_model(0.0)
    if len(active_points) == 1:
        return _make_constant_model(active_points[0][1])
    if len(active_points) == 2 or method == "linear":
        return _make_linear_model_from_active_points(active_points, max_t=max_t)

    first_t, first_y = active_points[0]
    last_t, last_y = active_points[-1]
    x_values = np.asarray([last_t - float(max_t), *[point[0] for point in active_points], first_t + float(max_t)], dtype=np.float64)
    y_values = np.asarray([last_y, *[point[1] for point in active_points], first_y], dtype=np.float64)

    if method == "pchip":
        interpolator = PchipInterpolator(x_values, y_values, extrapolate=True)
    elif method == "akima":
        interpolator = Akima1DInterpolator(x_values, y_values)
    elif method == "cubic_spline":
        interpolator = CubicSpline(x_values, y_values, extrapolate=True)
    else:
        raise ValueError(f"unsupported interpolation method: {method}")

    def _evaluate(xs: np.ndarray) -> np.ndarray:
        return np.asarray(interpolator(xs), dtype=np.float64)

    return _evaluate


def _extract_axis_map(document: dict[str, Any]) -> tuple[list[str], dict[str, list[tuple[float, float]]], list[tuple[float, float]], float]:
    top_actions = _normalize_actions(document.get("actions"))
    all_axes = [_TOPLEVEL_AXIS]
    axis_points: dict[str, list[tuple[float, float]]] = {_TOPLEVEL_AXIS: top_actions}

    axes_raw = document.get("axes")
    if isinstance(axes_raw, list):
        for entry in axes_raw:
            if not isinstance(entry, dict):
                continue
            axis_id = str(entry.get("id") or "").strip()
            if not axis_id or axis_id in axis_points:
                continue
            all_axes.append(axis_id)
            axis_points[axis_id] = _normalize_actions(entry.get("actions"))

    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    duration_value = _to_float_or_none(metadata.get("duration"))
    duration_from_text = _parse_duration_time_seconds(metadata.get("durationTime"))
    selected_for_fallback = top_actions
    file_duration = duration_value or duration_from_text or _fallback_duration_from_points(selected_for_fallback)
    return all_axes, axis_points, top_actions, float(file_duration)




def _segment_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _accumulate_heat_segment(
    bins: np.ndarray,
    *,
    start_t: float,
    end_t: float,
    distance: float,
    max_t: float,
) -> None:
    if distance <= 0.0:
        return
    if end_t <= start_t:
        return
    if end_t > max_t:
        duration = end_t - start_t
        before_wrap = max_t - start_t
        if before_wrap > 0.0:
            _accumulate_heat_segment(
                bins,
                start_t=start_t,
                end_t=max_t,
                distance=distance * (before_wrap / duration),
                max_t=max_t,
            )
        _accumulate_heat_segment(
            bins,
            start_t=0.0,
            end_t=end_t - max_t,
            distance=distance * ((end_t - max_t) / duration),
            max_t=max_t,
        )
        return

    num_bins = int(bins.shape[0])
    bin_width = float(max_t) / float(num_bins)
    if bin_width <= 0.0:
        return
    duration = end_t - start_t
    start_index = max(0, min(num_bins - 1, int(math.floor(start_t / bin_width))))
    end_index = max(0, min(num_bins - 1, int(math.floor(max(0.0, end_t - 1e-12) / bin_width))))
    for index in range(start_index, end_index + 1):
        bin_start = float(index) * bin_width
        bin_end = bin_start + bin_width
        overlap = _segment_overlap(start_t, end_t, bin_start, bin_end)
        if overlap <= 0.0:
            continue
        bins[index] += distance * (overlap / duration)


def _compute_heatmap(points: list[tuple[float, float]], *, max_t: float) -> list[float]:
    active = LoopingLinearSampler.from_points(points, max_t=max_t).active_points()
    bins = np.zeros((_HEATMAP_BINS,), dtype=np.float64)
    if len(active) <= 1 or max_t <= 0.0:
        return [0.0] * _HEATMAP_BINS

    loop_points = list(active)
    loop_points.append((active[0][0] + float(max_t), active[0][1]))
    for index in range(len(loop_points) - 1):
        start_t, start_y = loop_points[index]
        end_t, end_y = loop_points[index + 1]
        _accumulate_heat_segment(
            bins,
            start_t=float(start_t),
            end_t=float(end_t),
            distance=abs(float(end_y) - float(start_y)),
            max_t=float(max_t),
        )

    peak = float(np.max(bins)) if bins.size else 0.0
    if peak <= 0.0:
        return [0.0] * _HEATMAP_BINS
    return [float(value / peak) for value in bins.tolist()]


class WaveFunscriptRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        self._runtime_node = node
        data_in_ports = [str(port.name) for port in list(node.dataInPorts or [])] or ["t"]
        data_out_ports = [str(port.name) for port in list(node.dataOutPorts or [])] or ["value"]
        state_fields = [str(field.name) for field in list(node.stateFields or [])] or [
            "funscriptPath",
            "allAxes",
            "selectedAxis",
            "points",
            "maxT",
            "interp",
            "heatmap",
        ]
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=data_in_ports,
            data_out_ports=data_out_ports,
            state_fields=state_fields,
        )

        self._state_values: dict[str, Any] = {}
        for field in list(node.stateFields or []):
            field_name = str(field.name or "").strip()
            if not field_name:
                continue
            default_value = _schema_default_value(field.valueSchema)
            if default_value is not None:
                self._state_values[field_name] = default_value
        self._state_values.update(dict(initial_state or {}))

        self._path = _coerce_path(self._state_values.get("funscriptPath", ""))
        self._selected_axis = _coerce_axis(self._state_values.get("selectedAxis", _TOPLEVEL_AXIS))
        self._interp = _coerce_interp(self._state_values.get("interp", _DEFAULT_INTERP))
        max_t_raw = self._state_values.get("maxT", _DEFAULT_MAX_T)
        max_t_value = _to_float_or_none(max_t_raw)
        self._max_t = float(max_t_value) if max_t_value is not None and max_t_value > 0.0 else float(_DEFAULT_MAX_T)
        self._auto_max_t = max_t_value is None or math.isclose(float(self._max_t), float(_DEFAULT_MAX_T))
        self._file_max_t = float(_DEFAULT_MAX_T)
        self._all_axes: list[str] = [_TOPLEVEL_AXIS]
        self._axis_points: dict[str, list[tuple[float, float]]] = {_TOPLEVEL_AXIS: []}
        self._points: list[tuple[float, float]] = []
        self._linear_sampler = LoopingLinearSampler.from_points([], max_t=self._max_t)
        self._interp_model = _make_constant_model(0.0)
        self._use_linear_sampler_for_output = True
        self._heatmap: list[float] = list(_DEFAULT_HEATMAP)
        self._last_error = ""
        self._last_output: float | None = None
        self._publish_pending = True
        self._eval_error_sig = ""
        self._eval_error_ts_ms = 0
        self._internal_max_t_sync = False

        self._reload_document()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        if active:
            await self._publish_public_state(force=True)

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "funscriptPath":
            return _coerce_path(value)
        if name == "selectedAxis":
            return _coerce_axis(value)
        if name == "maxT":
            return _coerce_max_t(value)
        if name == "interp":
            return _coerce_interp(value)
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name in {"allAxes", "points", "heatmap"}:
            return
        if name == "funscriptPath":
            self._path = _coerce_path(value)
            self._state_values[name] = self._path
            self._reload_document()
            await self._publish_public_state(force=False)
            return
        if name == "selectedAxis":
            self._selected_axis = _coerce_axis(value)
            self._state_values[name] = self._selected_axis
            self._apply_selected_axis()
            await self._publish_public_state(force=False)
            return
        if name == "maxT":
            self._max_t = _coerce_max_t(value)
            self._state_values[name] = self._max_t
            if not self._internal_max_t_sync:
                self._auto_max_t = False
            self._rebuild_outputs()
            await self._publish_public_state(force=False)
            return
        if name == "interp":
            self._interp = _coerce_interp(value)
            self._state_values[name] = self._interp
            self._rebuild_outputs()
            await self._publish_public_state(force=False)
            return
        self._state_values[name] = value

    def _reload_document(self) -> None:
        self._all_axes = [_TOPLEVEL_AXIS]
        self._axis_points = {_TOPLEVEL_AXIS: []}
        self._file_max_t = float(_DEFAULT_MAX_T)
        if not self._path:
            self._set_last_error("")
            self._apply_selected_axis()
            return
        try:
            document = _load_funscript_file(self._path)
            self._all_axes, self._axis_points, _top_actions, self._file_max_t = _extract_axis_map(document)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
            self._all_axes = [_TOPLEVEL_AXIS]
            self._axis_points = {_TOPLEVEL_AXIS: []}
            self._file_max_t = float(_DEFAULT_MAX_T)
            self._points = []
            self._heatmap = list(_DEFAULT_HEATMAP)
            self._set_last_error(f"failed to load funscript: {type(exc).__name__}: {exc}")
            self._publish_pending = True
            return

        if self._auto_max_t:
            self._max_t = float(self._file_max_t)
            self._state_values["maxT"] = self._max_t
        self._clear_last_error()
        self._apply_selected_axis()

    def _apply_selected_axis(self) -> None:
        if self._selected_axis not in self._all_axes:
            self._selected_axis = _TOPLEVEL_AXIS
            self._state_values["selectedAxis"] = self._selected_axis
        self._points = list(self._axis_points.get(self._selected_axis, []))
        self._rebuild_outputs()

    def _rebuild_outputs(self) -> None:
        self._linear_sampler = LoopingLinearSampler.from_points(self._points, max_t=self._max_t)
        active_points = self._linear_sampler.active_points()
        self._use_linear_sampler_for_output = self._interp == "linear" or len(active_points) <= 2
        self._interp_model = _make_periodic_model_from_active_points(active_points, method=self._interp, max_t=self._max_t)
        self._heatmap = _compute_heatmap(self._points, max_t=self._max_t)
        self._publish_pending = True

    def _set_last_error(self, message: str) -> None:
        text = str(message or "").strip()
        if text == self._last_error:
            return
        self._last_error = text
        self._publish_pending = True

    def _clear_last_error(self) -> None:
        if not self._last_error:
            return
        self._last_error = ""
        self._publish_pending = True

    async def _publish_public_state(self, *, force: bool) -> None:
        if not force and not self._publish_pending:
            return
        self._publish_pending = False
        await self._safe_set_state("allAxes", list(self._all_axes))
        if self._state_values.get("selectedAxis") != self._selected_axis:
            self._state_values["selectedAxis"] = self._selected_axis
        await self._safe_set_state("selectedAxis", str(self._selected_axis))
        self._internal_max_t_sync = True
        try:
            await self._safe_set_state("maxT", float(self._max_t))
        finally:
            self._internal_max_t_sync = False
        await self._safe_set_state("points", _serialize_points(self._points))
        await self._safe_set_state("interp", str(self._interp))
        await self._safe_set_state("heatmap", list(self._heatmap))
        await self._safe_publish_monitor_error(str(self._last_error))

    async def _safe_publish_monitor_error(self, message: str) -> None:
        try:
            if message:
                await self.report_error(
                    "WAVE_FUNSCRIPT_ERROR",
                    message,
                    severity="error",
                    fingerprint=f"wave-funscript:{message}",
                )
                return
            await self.clear_error()
        except OPERATOR_STATE_PUBLISH_ERRORS:
            logger.exception("[%s:wave_funscript] failed to publish monitor error", self.node_id)

    async def _safe_set_state(self, field: str, value: Any) -> None:
        try:
            await self.set_state(field, value)
        except OPERATOR_STATE_PUBLISH_ERRORS:
            logger.exception("[%s:wave_funscript] failed to publish state: %s", self.node_id, field)

    def _should_log_repeating_eval_error(self, sig: str, *, now_ms: int) -> bool:
        if sig != self._eval_error_sig:
            self._eval_error_sig = sig
            self._eval_error_ts_ms = int(now_ms)
            return True
        if (int(now_ms) - int(self._eval_error_ts_ms)) >= 5000:
            self._eval_error_ts_ms = int(now_ms)
            return True
        return False

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if str(port or "") != "value":
            return None
        if self._publish_pending:
            await self._publish_public_state(force=False)
        t_raw = await self.pull("t", ctx_id=ctx_id)
        t_value = _to_float_or_none(t_raw)
        if t_value is None:
            return self._last_output
        try:
            out = self._linear_sampler.sample(float(t_value)) if self._use_linear_sampler_for_output else float(np.asarray(self._interp_model(np.asarray([math.fmod(float(t_value), float(self._max_t)) if math.fmod(float(t_value), float(self._max_t)) >= 0.0 else math.fmod(float(t_value), float(self._max_t)) + float(self._max_t)], dtype=np.float64)), dtype=np.float64)[0])
        except OPERATOR_EVAL_ERRORS as exc:
            now_ms = int(time.time() * 1000.0)
            sig = f"{type(exc).__name__}:{exc}"
            if self._should_log_repeating_eval_error(sig, now_ms=now_ms):
                logger.exception("[%s:wave_funscript] eval failed", self.node_id)
            return self._last_output
        self._last_output = float(out)
        return self._last_output


WaveFunscriptRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.motion",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Wave Funscript",
    description=(
        "Load a .funscript JSON file and expose one axis as a looping linear waveform.\n"
        "`t` is in seconds and evaluation uses `t % maxT`."
    ),
    tags=["wave", "funscript", "heatmap", "script"],
    dataInPorts=[
        F8DataPortSpec(
            name="t",
            description="Scalar time input in seconds. Runtime evaluation uses `t % maxT`.",
            valueSchema=number_schema(),
            required=True,
            showOnNode=True,
        ),
    ],
    dataOutPorts=[
        F8DataPortSpec(
            name="value",
            description="Normalized output from the selected funscript axis using the chosen interpolation mode.",
            valueSchema=number_schema(),
            required=True,
            showOnNode=True,
        ),
    ],
    stateFields=[
        F8StateSpec(
            name="funscriptPath",
            label="Funscript Path",
            description="Path to a .funscript JSON file. Cleared when exporting publish JSON.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
            redactOnPublish=True,
        ),
        F8StateSpec(
            name="allAxes",
            label="All Axes",
            description="Available funscript channels including the TopLevel pseudo-axis.",
            valueSchema=array_schema(items=string_schema()),
            access=F8StateAccess.ro,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="selectedAxis",
            label="Selected Axis",
            description="Axis id to load from the funscript file.",
            valueSchema=string_schema(default=_TOPLEVEL_AXIS),
            access=F8StateAccess.rw,
            required=True,
            control=F8UiControlSpec(kind=F8UiControlKind.select, optionsFromState="allAxes"),
            showOnNode=True,
        ),
        F8StateSpec(
            name="points",
            label="Points",
            description="Normalized `[timeSec, pos01]` points loaded from the selected axis.",
            valueSchema=F8ArrayTypeSchema(items=F8ArrayTypeSchema(items=helper_number_schema()), default=[]),
            access=F8StateAccess.ro,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="maxT",
            label="Max T",
            description="Loop period in seconds. Initialized from the funscript duration and user-overridable.",
            valueSchema=number_schema(default=_DEFAULT_MAX_T),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="interp",
            label="Interp",
            description="Interpolation method used for runtime output. Heatmap remains linear.",
            valueSchema=string_schema(default=_DEFAULT_INTERP, enum=list(_INTERP_METHODS)),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="heatmap",
            label="Heatmap",
            description="Per-time-bin activity heatmap derived from the selected axis.",
            valueSchema=F8ArrayTypeSchema(items=helper_number_schema(), default=_DEFAULT_HEATMAP),
            access=F8StateAccess.ro,
            required=True,
            control=F8UiControlSpec(kind=F8UiControlKind.custom, rendererKey="wave_heatmap"),
            showOnNode=True,
        ),
    ],
    editPolicy=F8SpecEditPolicy(stateFields=editable_collection_edit_policy()),
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(WaveFunscriptRuntimeNode.SPEC, WaveFunscriptRuntimeNode, overwrite=True)
    return registry
