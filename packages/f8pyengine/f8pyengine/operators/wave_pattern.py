from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator, interp1d

from f8pysdk.specs import (
    F8UiControlKind,
    F8UiControlSpec,
    F8ArrayTypeSchema,
    F8DataPortSpec,
    F8JsonValue,
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
from ._runtime_errors import OPERATOR_EVAL_ERRORS, OPERATOR_MODEL_ERRORS, OPERATOR_STATE_PUBLISH_ERRORS
from .wave_loop_sampler import LoopingLinearSampler


OPERATOR_CLASS = "f8.wave_pattern"
_DEFAULT_MAX_T = 10.0
_DEFAULT_MIN_VALUE = 0.0
_DEFAULT_MAX_VALUE = 1.0
_DEFAULT_INTERP = "pchip"
_INTERP_METHODS = ("linear", "pchip", "akima", "cubic_spline")
_PREVIEW_SAMPLES = 256
_PERIODIC_EPSILON = 1e-9
_DEFAULT_POINTS = [[0.0, 0.0], [_DEFAULT_MAX_T, 0.0]]

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


def _coerce_preview_bound(field_name: str, value: Any) -> float:
    numeric = _to_float_or_none(value)
    if numeric is None:
        raise ValueError(f"{field_name} must be numeric")
    return float(numeric)


def _coerce_interp(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text not in _INTERP_METHODS:
        raise ValueError(f"interp must be one of: {', '.join(_INTERP_METHODS)}")
    return text


def _serialize_points(points: list[tuple[float, float]]) -> list[list[float]]:
    return [[float(t_value), float(y_value)] for t_value, y_value in points]


def _normalize_points(value: Any, *, max_t: float) -> list[tuple[float, float]]:
    del max_t
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("points must be a list of [t, value] pairs")

    deduped: dict[float, tuple[int, float]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        t_value = _to_float_or_none(item[0])
        y_value = _to_float_or_none(item[1])
        if t_value is None or y_value is None:
            continue
        deduped[float(t_value)] = (index, float(y_value))

    ordered = sorted(deduped.items(), key=lambda item: item[0])
    return [(float(t_value), float(index_and_y[1])) for t_value, index_and_y in ordered]


def _active_cycle_points(points: list[tuple[float, float]], *, max_t: float) -> list[tuple[float, float]]:
    deduped: dict[float, float] = {}
    for t_value, y_value in points:
        if t_value < 0.0 or t_value > float(max_t):
            continue
        cycle_t = 0.0 if math.isclose(float(t_value), float(max_t), abs_tol=_PERIODIC_EPSILON) else float(t_value)
        deduped[cycle_t] = float(y_value)
    ordered = sorted(deduped.items(), key=lambda item: item[0])
    return [(float(t_value), float(y_value)) for t_value, y_value in ordered]


def _make_constant_model(value: float) -> Callable[[np.ndarray], np.ndarray]:
    constant_value = float(value)

    def _evaluate(xs: np.ndarray) -> np.ndarray:
        return np.full(xs.shape, constant_value, dtype=np.float64)

    return _evaluate


def _make_linear_model(points: list[tuple[float, float]], *, max_t: float) -> Callable[[np.ndarray], np.ndarray]:
    cycle_points = _active_cycle_points(points, max_t=max_t)
    if len(cycle_points) == 0:
        return _make_constant_model(0.0)
    if len(cycle_points) == 1:
        return _make_constant_model(cycle_points[0][1])

    t_values = np.asarray([point[0] for point in cycle_points], dtype=np.float64)
    y_values = np.asarray([point[1] for point in cycle_points], dtype=np.float64)
    if t_values[0] > 0.0:
        t_values = np.concatenate(([0.0], t_values))
        y_values = np.concatenate(([y_values[-1]], y_values))
    if t_values[-1] < float(max_t):
        t_values = np.concatenate((t_values, [float(max_t)]))
        y_values = np.concatenate((y_values, [y_values[0]]))

    linear = interp1d(t_values, y_values, kind="linear", bounds_error=False, fill_value="extrapolate", assume_sorted=True)

    def _evaluate(xs: np.ndarray) -> np.ndarray:
        out = linear(xs)
        return np.asarray(out, dtype=np.float64)

    return _evaluate


def _make_periodic_model(
    points: list[tuple[float, float]],
    *,
    method: str,
    max_t: float,
) -> Callable[[np.ndarray], np.ndarray]:
    cycle_points = _active_cycle_points(points, max_t=max_t)
    if len(cycle_points) == 0:
        return _make_constant_model(0.0)
    if len(cycle_points) == 1:
        return _make_constant_model(cycle_points[0][1])
    if len(cycle_points) == 2 or method == "linear":
        return _make_linear_model(cycle_points, max_t=max_t)

    first_t, first_y = cycle_points[0]
    last_t, last_y = cycle_points[-1]
    x_values = np.asarray(
        [last_t - float(max_t), *[point[0] for point in cycle_points], first_t + float(max_t)],
        dtype=np.float64,
    )
    y_values = np.asarray([last_y, *[point[1] for point in cycle_points], first_y], dtype=np.float64)

    if method == "pchip":
        interpolator = PchipInterpolator(x_values, y_values, extrapolate=True)
    elif method == "akima":
        interpolator = Akima1DInterpolator(x_values, y_values)
    elif method == "cubic_spline":
        interpolator = CubicSpline(x_values, y_values, extrapolate=True)
    else:
        raise ValueError(f"unsupported interpolation method: {method}")

    def _evaluate(xs: np.ndarray) -> np.ndarray:
        out = interpolator(xs)
        return np.asarray(out, dtype=np.float64)

    return _evaluate


class WavePatternRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        self._runtime_node = node
        data_in_ports = [str(p.name) for p in list(node.dataInPorts or [])] or ["t"]
        data_out_ports = [str(p.name) for p in list(node.dataOutPorts or [])] or ["value"]
        state_fields = [str(s.name) for s in list(node.stateFields or [])] or [
            "points",
            "maxT",
            "minValue",
            "maxValue",
            "interp",
            "preview",
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

        max_t_raw = self._state_values.get("maxT", _DEFAULT_MAX_T)
        max_t_value = _to_float_or_none(max_t_raw)
        self._max_t = float(max_t_value) if max_t_value is not None and max_t_value > 0.0 else float(_DEFAULT_MAX_T)

        min_value_raw = self._state_values.get("minValue", _DEFAULT_MIN_VALUE)
        max_value_raw = self._state_values.get("maxValue", _DEFAULT_MAX_VALUE)
        min_value = _to_float_or_none(min_value_raw)
        max_value = _to_float_or_none(max_value_raw)
        self._min_value = float(min_value) if min_value is not None else float(_DEFAULT_MIN_VALUE)
        self._max_value = float(max_value) if max_value is not None else float(_DEFAULT_MAX_VALUE)
        self._interp = _coerce_interp(self._state_values.get("interp", _DEFAULT_INTERP))
        default_points = [[0.0, 0.0], [float(self._max_t), 0.0]]
        self._points = _normalize_points(self._state_values.get("points", default_points), max_t=self._max_t)
        self._preview_cycle: list[tuple[float, float]] = []
        self._linear_sampler = LoopingLinearSampler.from_points([], max_t=self._max_t)
        self._use_linear_sampler_for_output = False
        self._last_error = ""
        self._last_output: float | None = None
        self._publish_pending = True
        self._model: Callable[[np.ndarray], np.ndarray] = _make_constant_model(0.0)
        self._eval_error_sig = ""
        self._eval_error_ts_ms = 0

        self._state_values["maxT"] = self._max_t
        self._state_values["minValue"] = self._min_value
        self._state_values["maxValue"] = self._max_value
        self._state_values["interp"] = self._interp
        self._state_values["points"] = _serialize_points(self._points)
        self._sync_runtime_node_state_values()

        self._rebuild_model()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        if active:
            await self._publish_public_state(force=True)

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "points":
            return _serialize_points(_normalize_points(value, max_t=self._max_t))
        if name == "maxT":
            return _coerce_max_t(value)
        if name == "minValue":
            return _coerce_preview_bound("minValue", value)
        if name == "maxValue":
            return _coerce_preview_bound("maxValue", value)
        if name == "interp":
            return _coerce_interp(value)
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name == "preview":
            return

        if name == "points":
            self._points = _normalize_points(value, max_t=self._max_t)
            self._state_values[name] = _serialize_points(self._points)
            self._sync_runtime_node_state_values()
            self._rebuild_model()
            await self._publish_public_state(force=False)
            return
        if name == "maxT":
            self._max_t = _coerce_max_t(value)
            self._points = _normalize_points(self._state_values.get("points"), max_t=self._max_t)
            self._state_values[name] = self._max_t
            self._state_values["points"] = _serialize_points(self._points)
            self._sync_runtime_node_state_values()
            self._rebuild_model()
            await self._publish_public_state(force=False)
            return
        if name == "minValue":
            self._min_value = _coerce_preview_bound("minValue", value)
            self._state_values[name] = self._min_value
            self._sync_runtime_node_state_values()
            return
        if name == "maxValue":
            self._max_value = _coerce_preview_bound("maxValue", value)
            self._state_values[name] = self._max_value
            self._sync_runtime_node_state_values()
            return
        if name == "interp":
            self._interp = _coerce_interp(value)
            self._state_values[name] = self._interp
            self._sync_runtime_node_state_values()
            self._rebuild_model()
            await self._publish_public_state(force=False)
            return

        self._state_values[name] = value

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

    def _rebuild_model(self) -> None:
        active_points = _active_cycle_points(self._points, max_t=self._max_t)
        self._linear_sampler = LoopingLinearSampler.from_points(self._points, max_t=self._max_t)
        self._use_linear_sampler_for_output = self._interp == "linear" or len(active_points) <= 2
        try:
            self._model = _make_periodic_model(self._points, method=self._interp, max_t=self._max_t)
            t_preview = np.linspace(0.0, float(self._max_t), num=_PREVIEW_SAMPLES, endpoint=False, dtype=np.float64)
            preview = np.asarray(self._model(t_preview), dtype=np.float64)
            if preview.ndim == 0:
                preview = np.full((_PREVIEW_SAMPLES,), float(preview), dtype=np.float64)
            elif preview.ndim != 1 or preview.shape[0] != _PREVIEW_SAMPLES:
                raise ValueError("preview result must be 1D and match preview sample count")
        except OPERATOR_MODEL_ERRORS as exc:
            self._model = _make_constant_model(0.0)
            t_preview = np.linspace(0.0, float(self._max_t), num=_PREVIEW_SAMPLES, endpoint=False, dtype=np.float64)
            preview = np.zeros((_PREVIEW_SAMPLES,), dtype=np.float64)
            self._set_last_error(f"wave pattern rebuild failed: {type(exc).__name__}: {exc}")
        else:
            self._clear_last_error()

        preview_values = [float(v) for v in preview.tolist()]
        preview_times = [float(v) for v in t_preview.tolist()]
        self._preview_cycle = list(zip(preview_times, preview_values, strict=True))
        self._publish_pending = True

    def _sync_runtime_node_state_values(self) -> None:
        current = cast(dict[str, F8JsonValue], dict(self._runtime_node.stateValues or {}))
        current["points"] = _serialize_points(self._points)
        current["maxT"] = float(self._max_t)
        current["minValue"] = float(self._min_value)
        current["maxValue"] = float(self._max_value)
        current["interp"] = str(self._interp)
        self._runtime_node.stateValues = current

    async def _publish_public_state(self, *, force: bool) -> None:
        if not force and not self._publish_pending:
            return
        self._publish_pending = False
        await self._safe_set_state("points", _serialize_points(self._points))
        await self._safe_set_state("preview", [list(point) for point in self._preview_cycle])
        await self._safe_publish_monitor_error(str(self._last_error))

    async def _safe_publish_monitor_error(self, message: str) -> None:
        try:
            if message:
                await self.report_error(
                    "WAVE_PATTERN_ERROR",
                    message,
                    severity="error",
                    fingerprint=f"wave-pattern:{message}",
                )
                return
            await self.clear_error()
        except OPERATOR_STATE_PUBLISH_ERRORS:
            logger.exception("[%s:wave_pattern] failed to publish monitor error", self.node_id)

    async def _safe_set_state(self, field: str, value: Any) -> None:
        try:
            await self.set_state(field, value)
        except OPERATOR_STATE_PUBLISH_ERRORS:
            logger.exception("[%s:wave_pattern] failed to publish state: %s", self.node_id, field)

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
            if self._use_linear_sampler_for_output:
                out = self._linear_sampler.sample(float(t_value))
            else:
                wrapped_t = math.fmod(float(t_value), float(self._max_t))
                if wrapped_t < 0.0:
                    wrapped_t += float(self._max_t)
                out_arr = np.asarray(self._model(np.asarray([wrapped_t], dtype=np.float64)), dtype=np.float64)
                out = float(out_arr[0]) if out_arr.size else 0.0
        except OPERATOR_EVAL_ERRORS as exc:
            now_ms = int(time.time() * 1000.0)
            sig = f"{type(exc).__name__}:{exc}"
            if self._should_log_repeating_eval_error(sig, now_ms=now_ms):
                logger.exception("[%s:wave_pattern] eval failed", self.node_id)
            return self._last_output

        self._last_output = out
        return self._last_output


WavePatternRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.motion",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Wave Pattern",
    description=(
        "Interactive periodic waveform node.\n"
        "\n"
        "Core\n"
        "- `t` is cycle-domain input and runtime output evaluates at `t % maxT`.\n"
        "- `points` stores editable control points as `[t, value]` pairs.\n"
        "- `interp` selects the interpolation method for preview and runtime evaluation.\n"
        "- Preview samples the periodic waveform over `[0, maxT)`."
    ),
    tags=["wave", "pattern", "interp", "signal"],
    dataInPorts=[
        F8DataPortSpec(
            name="t",
            description="Scalar cycle-domain input. Runtime evaluation uses `t % maxT`.",
            valueSchema=number_schema(),
            required=True,
            showOnNode=True,
        ),
    ],
    dataOutPorts=[
        F8DataPortSpec(
            name="value",
            description="Interpolated waveform output for the current wrapped `t` sample.",
            valueSchema=number_schema(),
            required=True,
            showOnNode=True,
        ),
    ],
    stateFields=[
        F8StateSpec(
            name="points",
            label="Points",
            description="Editable control points as `[t, value]` pairs.",
            valueSchema=F8ArrayTypeSchema(
                items=F8ArrayTypeSchema(items=helper_number_schema()),
                default=_DEFAULT_POINTS,
            ),
            access=F8StateAccess.rw,
            required=True,
            control=F8UiControlSpec(kind=F8UiControlKind.custom, rendererKey="wave_pattern_editor"),
            showOnNode=True,
        ),
        F8StateSpec(
            name="maxT",
            label="Max T",
            description="Cycle horizon for wrapping and preview sampling.",
            valueSchema=number_schema(default=_DEFAULT_MAX_T),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="minValue",
            label="Min Value",
            description="Editor and preview lower Y bound.",
            valueSchema=number_schema(default=_DEFAULT_MIN_VALUE),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="maxValue",
            label="Max Value",
            description="Editor and preview upper Y bound.",
            valueSchema=number_schema(default=_DEFAULT_MAX_VALUE),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="interp",
            label="Interp",
            description="Interpolation method used to generate the periodic waveform.",
            valueSchema=string_schema(default=_DEFAULT_INTERP, enum=list(_INTERP_METHODS)),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="preview",
            label="Preview",
            description="Preview waveform samples as `[t, value]` pairs over `[0, maxT)`.",
            valueSchema=array_schema(items=array_schema(items=helper_number_schema())),
            access=F8StateAccess.ro,
            required=True,
            control=F8UiControlSpec(kind=F8UiControlKind.custom, rendererKey="wave_preview"),
            showOnNode=False,
        ),
    ],
    editPolicy=F8SpecEditPolicy(stateFields=editable_collection_edit_policy()),
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(WavePatternRuntimeNode.SPEC, WavePatternRuntimeNode, overwrite=True)
    return registry
