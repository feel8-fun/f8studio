from __future__ import annotations

import math
from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    number_schema,
    string_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS
from .envelope import EnvelopeTracker
from ._ports import exec_out_ports

OPERATOR_CLASS = "f8.axis_envelope"

_METHODS = ("EMA", "DEMA", "SMA")
_EPS = 1e-9


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _normalize_method(value: Any, *, default: str = "EMA") -> str:
    method = str(value or "").strip().upper()
    if method in _METHODS:
        return method
    return default


class AxisEnvelopeRuntimeNode(OperatorNode):
    """
    Convert 2D oscillation (x,y) into normalized amplitudes along major/minor axes.

    - Tracks center and covariance using EMA (ema_alpha)
    - Projects onto principal axes
    - Uses envelope trackers for normalized outputs (0..1)
    - If only one input is provided, degrades to 1D envelope on that value and
      outputs minor axis as a constant 0.5.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})

        self._ema_alpha = self._coerce_alpha(self._initial_state.get("ema_alpha"), default=0.2)
        self._method = _normalize_method(self._initial_state.get("method"), default="EMA")
        self._rise_alpha = float(_coerce_number(self._initial_state.get("rise_alpha")) or 0.4)
        self._fall_alpha = float(_coerce_number(self._initial_state.get("fall_alpha")) or 0.05)
        self._min_span = max(0.0, float(_coerce_number(self._initial_state.get("min_span")) or 0.25))
        self._sma_window = self._coerce_window(self._initial_state.get("sma_window"), default=10)
        self._margin = float(_coerce_number(self._initial_state.get("margin")) or 0.0)
        # Reset protection:
        # - reset_z: proportional threshold using normalized distance (sigma units) from the current estimate.
        # - reset_abs_max: legacy absolute threshold (0 disables); kept for backwards compatibility.
        self._reset_z = max(0.0, float(_coerce_number(self._initial_state.get("reset_z")) or 8.0))
        self._reset_abs_max = max(0.0, float(_coerce_number(self._initial_state.get("reset_abs_max")) or 0.0))

        self._center_x: float | None = None
        self._center_y: float | None = None
        self._cov_xx = 0.0
        self._cov_xy = 0.0
        self._cov_yy = 0.0

        self._major_env = EnvelopeTracker(
            method=self._method,
            rise_alpha=self._rise_alpha,
            fall_alpha=self._fall_alpha,
            min_span=self._min_span,
            sma_window=self._sma_window,
        )
        self._minor_env = EnvelopeTracker(
            method=self._method,
            rise_alpha=self._rise_alpha,
            fall_alpha=self._fall_alpha,
            min_span=self._min_span,
            sma_window=self._sma_window,
        )

        self._last_outputs: dict[str, float | None] = {"major": None, "minor": None}
        self._last_ctx_id: str | int | None = None
        self._last_input: tuple[float | None, float | None] | None = None
        self._last_mode: str | None = None
        self._dirty = True

    @staticmethod
    def _coerce_alpha(value: Any, *, default: float) -> float:
        numeric = _coerce_number(value)
        if numeric is None:
            return float(default)
        return max(0.0, min(1.0, float(numeric)))

    @staticmethod
    def _coerce_window(value: Any, *, default: int) -> int:
        numeric = _coerce_number(value)
        if numeric is None:
            return int(default)
        return max(1, int(round(float(numeric))))

    def _reset_cache(self) -> None:
        self._last_outputs = {"major": None, "minor": None}
        self._last_ctx_id = None
        self._last_input = None
        self._last_mode = None
        self._dirty = True

    def _reset_axis_state(self) -> None:
        self._center_x = None
        self._center_y = None
        self._cov_xx = 0.0
        self._cov_xy = 0.0
        self._cov_yy = 0.0

    def _reset_estimator(self) -> None:
        self._reset_axis_state()
        self._major_env.reset()
        self._minor_env.reset()

    def _is_outlier_single(self, value: float) -> bool:
        """
        Proportional outlier detection for 1D mode based on current envelope span.

        Treat as outlier when the sample is far outside the current [lower, upper]
        range by a large multiple of the span.
        """
        reset_z = float(self._reset_z)
        if reset_z <= 0.0:
            return False
        lower = self._major_env.lower
        upper = self._major_env.upper
        if lower is None or upper is None:
            return False
        span = float(upper - lower)
        if span <= _EPS:
            return False
        mid = 0.5 * (float(upper) + float(lower))
        z = abs(float(value) - mid) / span
        return bool(z >= reset_z)

    def _principal_axes_and_vars(self) -> tuple[tuple[float, float], tuple[float, float], float, float]:
        """
        Return (major_axis, minor_axis, var_major, var_minor) from current covariance.
        """
        sxx = float(self._cov_xx)
        sxy = float(self._cov_xy)
        syy = float(self._cov_yy)

        if sxx + syy <= _EPS:
            return (1.0, 0.0), (0.0, 1.0), 0.0, 0.0

        diff = sxx - syy
        temp = math.sqrt(max(0.0, diff * diff + 4.0 * sxy * sxy))
        lambda1 = 0.5 * (sxx + syy + temp)
        lambda2 = 0.5 * (sxx + syy - temp)

        if abs(sxy) > _EPS:
            vx = lambda1 - syy
            vy = sxy
        else:
            if sxx >= syy:
                vx, vy = 1.0, 0.0
            else:
                vx, vy = 0.0, 1.0

        norm = math.hypot(vx, vy)
        if norm <= _EPS:
            major = (1.0, 0.0)
        else:
            major = (vx / norm, vy / norm)
        minor = (-major[1], major[0])
        return major, minor, float(max(0.0, lambda1)), float(max(0.0, lambda2))

    def _is_outlier_dual(self, x: float, y: float) -> bool:
        """
        Proportional outlier detection in 2D mode using principal-axis z-scores.

        If a single sample lands many sigma away from the current estimate, we reset
        to avoid poisoning the covariance + envelopes.
        """
        reset_z = float(self._reset_z)
        if reset_z <= 0.0:
            return False
        if self._center_x is None or self._center_y is None:
            return False
        if float(self._cov_xx) + float(self._cov_yy) <= _EPS:
            return False

        major_axis, minor_axis, var_major, var_minor = self._principal_axes_and_vars()
        dx = float(x) - float(self._center_x)
        dy = float(y) - float(self._center_y)
        major_value = major_axis[0] * dx + major_axis[1] * dy
        minor_value = minor_axis[0] * dx + minor_axis[1] * dy

        std_major = math.sqrt(float(var_major) + _EPS)
        std_minor = math.sqrt(float(var_minor) + _EPS)
        z_major = abs(float(major_value)) / std_major
        z_minor = abs(float(minor_value)) / std_minor
        return bool(max(z_major, z_minor) >= reset_z)

    def _is_outlier_abs_legacy(self, x: float | None, y: float | None) -> bool:
        limit = float(self._reset_abs_max)
        if limit <= 0.0:
            return False
        if x is not None and abs(float(x)) >= limit:
            return True
        if y is not None and abs(float(y)) >= limit:
            return True
        return False

    def _apply_state_values(self, values: dict[str, Any]) -> None:
        tracker_changed = False

        if "ema_alpha" in values:
            self._ema_alpha = self._coerce_alpha(values.get("ema_alpha"), default=self._ema_alpha)

        if "method" in values:
            method = _normalize_method(values.get("method"), default=self._method)
            if method != self._method:
                self._method = method
                tracker_changed = True

        if "rise_alpha" in values:
            numeric = _coerce_number(values.get("rise_alpha"))
            if numeric is not None and float(numeric) != self._rise_alpha:
                self._rise_alpha = float(numeric)
                tracker_changed = True

        if "fall_alpha" in values:
            numeric = _coerce_number(values.get("fall_alpha"))
            if numeric is not None and float(numeric) != self._fall_alpha:
                self._fall_alpha = float(numeric)
                tracker_changed = True

        if "min_span" in values:
            numeric = _coerce_number(values.get("min_span"))
            if numeric is not None:
                numeric = max(0.0, float(numeric))
                if numeric != self._min_span:
                    self._min_span = numeric
                    tracker_changed = True

        if "sma_window" in values:
            window = self._coerce_window(values.get("sma_window"), default=self._sma_window)
            if window != self._sma_window:
                self._sma_window = window
                tracker_changed = True

        if "margin" in values:
            numeric = _coerce_number(values.get("margin"))
            if numeric is not None:
                self._margin = float(numeric)

        if "reset_z" in values:
            numeric = _coerce_number(values.get("reset_z"))
            if numeric is not None:
                self._reset_z = max(0.0, float(numeric))

        if "reset_abs_max" in values:
            numeric = _coerce_number(values.get("reset_abs_max"))
            if numeric is not None:
                self._reset_abs_max = max(0.0, float(numeric))

        if tracker_changed:
            for tracker in (self._major_env, self._minor_env):
                tracker.set_parameters(
                    method=self._method,
                    rise_alpha=self._rise_alpha,
                    fall_alpha=self._fall_alpha,
                    min_span=self._min_span,
                    sma_window=self._sma_window,
                )
            self._reset_cache()

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        name = str(field or "")
        if name in {"ema_alpha", "method", "rise_alpha", "fall_alpha", "min_span", "sma_window", "margin", "reset_z", "reset_abs_max"}:
            self._apply_state_values({name: value})

    def _update_center_and_cov(self, x: float, y: float) -> None:
        if self._center_x is None or self._center_y is None:
            self._center_x = float(x)
            self._center_y = float(y)
            self._cov_xx = 0.0
            self._cov_xy = 0.0
            self._cov_yy = 0.0
            return

        alpha = self._ema_alpha
        self._center_x += alpha * (x - self._center_x)
        self._center_y += alpha * (y - self._center_y)

        dx = x - self._center_x
        dy = y - self._center_y
        inv = 1.0 - alpha
        self._cov_xx = inv * self._cov_xx + alpha * dx * dx
        self._cov_xy = inv * self._cov_xy + alpha * dx * dy
        self._cov_yy = inv * self._cov_yy + alpha * dy * dy

    def _principal_axes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        sxx = float(self._cov_xx)
        sxy = float(self._cov_xy)
        syy = float(self._cov_yy)

        if sxx + syy <= _EPS:
            return (1.0, 0.0), (0.0, 1.0)

        diff = sxx - syy
        temp = math.sqrt(max(0.0, diff * diff + 4.0 * sxy * sxy))
        lambda1 = 0.5 * (sxx + syy + temp)

        if abs(sxy) > _EPS:
            vx = lambda1 - syy
            vy = sxy
        else:
            if sxx >= syy:
                vx, vy = 1.0, 0.0
            else:
                vx, vy = 0.0, 1.0

        norm = math.hypot(vx, vy)
        if norm <= _EPS:
            major = (1.0, 0.0)
        else:
            major = (vx / norm, vy / norm)
        minor = (-major[1], major[0])
        return major, minor

    def _normalize_with_margin(self, tracker: EnvelopeTracker, value: float) -> float:
        lower = tracker.lower
        upper = tracker.upper
        if lower is None or upper is None:
            return 0.5
        margin = float(self._margin)
        if margin:
            lower -= margin
            upper += margin
        span = upper - lower
        if span <= 0:
            return 0.5
        normalized = (value - lower) / span
        if normalized < 0.0:
            return 0.0
        if normalized > 1.0:
            return 1.0
        return float(normalized)

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        port_s = str(port)
        if port_s not in self._last_outputs:
            return None

        x = _coerce_number(await self.pull("x", ctx_id=ctx_id))
        y = _coerce_number(await self.pull("y", ctx_id=ctx_id))

        if x is None and y is None:
            return self._last_outputs.get(port_s)

        mode = "dual" if x is not None and y is not None else "single"
        if self._is_outlier_abs_legacy(x, y):
            self._reset_estimator()
            self._last_outputs = {"major": 0.5, "minor": 0.5}
            self._last_ctx_id = ctx_id
            self._last_input = (x, y)
            self._last_mode = mode
            self._dirty = False
            return self._last_outputs.get(port_s)

        if not self._dirty:
            if ctx_id is not None and ctx_id == self._last_ctx_id and mode == self._last_mode:
                return self._last_outputs.get(port_s)
            if ctx_id is None and self._last_input == (x, y) and mode == self._last_mode:
                return self._last_outputs.get(port_s)

        if mode == "single":
            value = x if x is not None else y
            if value is None:
                return self._last_outputs.get(port_s)
            if self._is_outlier_single(float(value)):
                self._reset_estimator()
                self._last_outputs = {"major": 0.5, "minor": 0.5}
                self._last_ctx_id = ctx_id
                self._last_input = (x, y)
                self._last_mode = mode
                self._dirty = False
                return self._last_outputs.get(port_s)
            self._major_env.update(float(value))
            major = self._normalize_with_margin(self._major_env, float(value))
            self._last_outputs = {"major": major, "minor": 0.5}
        else:
            if self._is_outlier_dual(float(x), float(y)):
                self._reset_estimator()
                self._last_outputs = {"major": 0.5, "minor": 0.5}
                self._last_ctx_id = ctx_id
                self._last_input = (x, y)
                self._last_mode = mode
                self._dirty = False
                return self._last_outputs.get(port_s)
            self._update_center_and_cov(float(x), float(y))
            if self._center_x is None or self._center_y is None:
                return self._last_outputs.get(port_s)
            major_axis, minor_axis, _var_major, _var_minor = self._principal_axes_and_vars()
            dx = float(x) - self._center_x
            dy = float(y) - self._center_y
            major_value = major_axis[0] * dx + major_axis[1] * dy
            minor_value = minor_axis[0] * dx + minor_axis[1] * dy

            self._major_env.update(major_value)
            self._minor_env.update(minor_value)
            major = self._normalize_with_margin(self._major_env, major_value)
            minor = self._normalize_with_margin(self._minor_env, minor_value)
            self._last_outputs = {"major": major, "minor": minor}

        self._last_ctx_id = ctx_id
        self._last_input = (x, y)
        self._last_mode = mode
        self._dirty = False
        return self._last_outputs.get(port_s)


AxisEnvelopeRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Axis Envelope",
    description="Estimate normalized amplitudes along major/minor axes from 2D input.",
    tags=["signal", "envelope", "ellipse", "amplitude", "transform"],
    dataInPorts=[
        F8DataPortSpec(name="x", description="X coordinate.", valueSchema=number_schema(), required=False),
        F8DataPortSpec(name="y", description="Y coordinate.", valueSchema=number_schema(), required=False),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="major", description="Normalized amplitude along the major axis (0..1).", valueSchema=number_schema()),
        F8DataPortSpec(name="minor", description="Normalized amplitude along the minor axis (0..1).", valueSchema=number_schema()),
    ],
    stateFields=[
        F8StateSpec(
            name="ema_alpha",
            label="EMA Alpha",
            description="EMA smoothing for center/covariance tracking.",
            valueSchema=number_schema(default=0.2, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="method",
            label="Method",
            description="Envelope filter method.",
            valueSchema=string_schema(default="EMA", enum=["EMA", "DEMA", "SMA"]),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="rise_alpha",
            label="Rise Alpha",
            description="Smoothing factor when moving toward the envelope.",
            valueSchema=number_schema(default=0.4, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="fall_alpha",
            label="Fall Alpha",
            description="Smoothing factor when moving away from the envelope.",
            valueSchema=number_schema(default=0.05, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="min_span",
            label="Min Span",
            description="Minimum enforced envelope span.",
            valueSchema=number_schema(default=0.25, minimum=0.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="sma_window",
            label="SMA Window",
            description="Window size for SMA mode.",
            valueSchema=number_schema(default=10, minimum=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="margin",
            label="Margin",
            description="Extra margin added to the envelopes before normalization.",
            valueSchema=number_schema(default=0.0, minimum=0.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="reset_z",
            label="Reset Z",
            description="Proportional outlier threshold. If a sample is >= this many sigma (2D) or spans (1D) away, reset and ignore it (0 disables).",
            valueSchema=number_schema(default=8.0, minimum=0.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="reset_abs_max",
            label="Reset Abs Max",
            description="Legacy absolute outlier threshold; if abs(x) or abs(y) exceeds this value, reset and ignore the sample (0 disables).",
            valueSchema=number_schema(default=0.0, minimum=0.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return AxisEnvelopeRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(AxisEnvelopeRuntimeNode.SPEC, overwrite=True)
    return reg
