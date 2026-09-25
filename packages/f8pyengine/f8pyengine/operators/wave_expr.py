from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np

from f8pysdk.specs import (
    F8UiControlKind,
    F8UiControlSpec,
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
from f8pysdk.specs import schema_type

from ..constants import SERVICE_CLASS
from ..wave_expr_lang import RESERVED_NAMES, compile_expr, eval_compiled, eval_scalar, render_expression
from ._runtime_errors import OPERATOR_EVAL_ERRORS, OPERATOR_MODEL_ERRORS, OPERATOR_STATE_PUBLISH_ERRORS


OPERATOR_CLASS = "f8.wave_expr"
_DEFAULT_TEMPLATE = "0.5 + 0.5 * cos(t)"
_DEFAULT_MAX_T = 10.0
_DEFAULT_MIN_VALUE = 0.0
_DEFAULT_MAX_VALUE = 0.0
_PREVIEW_SAMPLES = 256

_PROTECTED_STATE_FIELDS = {
    "template",
    "maxT",
    "minValue",
    "maxValue",
    "express",
    "preview",
    "svcId",
    "operatorId",
}

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


def _normalize_template(value: Any) -> str:
    text = str("" if value is None else value)
    if "\n" not in text and "\r" not in text:
        return text.strip()
    parts = [part.strip() for part in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return " ".join([part for part in parts if part]).strip()


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


def _access_text(access: F8StateAccess | str | None) -> str:
    if isinstance(access, F8StateAccess):
        raw = access.value
    else:
        raw = access
    return str(raw or "").strip().lower()


def _is_variable_field(field: F8StateSpec) -> bool:
    name = str(field.name or "").strip()
    if not name or name in _PROTECTED_STATE_FIELDS:
        return False

    access = _access_text(field.access)
    if access not in {"rw", "wo"}:
        return False

    s_type = schema_type(field.valueSchema)
    return s_type in {"number", "integer"}


class WaveExprRuntimeNode(OperatorNode):
    """Template-driven waveform expression node."""

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        data_in_ports = [str(p.name) for p in list(node.dataInPorts or [])] or ["t"]
        data_out_ports = [str(p.name) for p in list(node.dataOutPorts or [])] or ["value"]
        state_fields = [str(s.name) for s in list(node.stateFields or [])] or [
            "template",
            "maxT",
            "minValue",
            "maxValue",
            "express",
            "preview",
        ]

        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=data_in_ports,
            data_out_ports=data_out_ports,
            state_fields=state_fields,
        )

        self._variable_field_names = [
            str(field.name)
            for field in list(node.stateFields or [])
            if _is_variable_field(field)
        ]

        self._state_values: dict[str, Any] = {}
        for field in list(node.stateFields or []):
            field_name = str(field.name or "").strip()
            if not field_name:
                continue
            default_value = _schema_default_value(field.valueSchema)
            if default_value is not None:
                self._state_values[field_name] = default_value
        self._state_values.update(dict(initial_state or {}))
        self._template = _normalize_template(self._state_values.get("template") or _DEFAULT_TEMPLATE)

        max_t_raw = self._state_values.get("maxT", _DEFAULT_MAX_T)
        max_t_value = _to_float_or_none(max_t_raw)
        self._max_t = float(max_t_value) if max_t_value is not None and max_t_value > 0.0 else float(_DEFAULT_MAX_T)
        min_value_raw = self._state_values.get("minValue", _DEFAULT_MIN_VALUE)
        max_value_raw = self._state_values.get("maxValue", _DEFAULT_MAX_VALUE)
        min_value = _to_float_or_none(min_value_raw)
        max_value = _to_float_or_none(max_value_raw)
        self._min_value = float(min_value) if min_value is not None else float(_DEFAULT_MIN_VALUE)
        self._max_value = float(max_value) if max_value is not None else float(_DEFAULT_MAX_VALUE)
        self._state_values["minValue"] = self._min_value
        self._state_values["maxValue"] = self._max_value

        self._compiled = None
        self._eval_variables: dict[str, float] = {}
        self._express = ""
        self._preview_cycle: list[tuple[float, float]] = []
        self._last_error = ""
        self._last_output: float | None = None
        self._publish_pending = True

        self._eval_error_sig = ""
        self._eval_error_ts_ms = 0

        self._rebuild_model()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        if active:
            await self._publish_public_state(force=True)

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "template":
            return _normalize_template(value)
        if name == "maxT":
            return _coerce_max_t(value)
        if name == "minValue":
            return _coerce_preview_bound("minValue", value)
        if name == "maxValue":
            return _coerce_preview_bound("maxValue", value)
        if name in self._variable_field_names:
            numeric = _to_float_or_none(value)
            if numeric is None:
                raise ValueError(f"{name} must be numeric")
            return float(numeric)
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()

        if name in {"express", "preview"}:
            return

        if name == "template":
            normalized = _normalize_template(value)
            self._state_values[name] = normalized
            self._template = normalized
            self._rebuild_model()
            await self._publish_public_state(force=False)
            return

        if name == "maxT":
            self._max_t = _coerce_max_t(value)
            self._state_values[name] = self._max_t
            self._rebuild_model()
            await self._publish_public_state(force=False)
            return
        if name == "minValue":
            self._min_value = _coerce_preview_bound("minValue", value)
            self._state_values[name] = self._min_value
            return
        if name == "maxValue":
            self._max_value = _coerce_preview_bound("maxValue", value)
            self._state_values[name] = self._max_value
            return

        if name in self._variable_field_names:
            numeric = _to_float_or_none(value)
            if numeric is None:
                self._state_values[name] = value
            else:
                self._state_values[name] = float(numeric)
            self._rebuild_model()
            await self._publish_public_state(force=False)
            return

        # Keep a best-effort snapshot for editable non-variable states.
        self._state_values[name] = value

    def _collect_eval_variables(self) -> tuple[dict[str, float], list[str]]:
        variables: dict[str, float] = {}
        warnings: list[str] = []

        for name in self._variable_field_names:
            if name in RESERVED_NAMES:
                warnings.append(f"variable name collides with reserved symbol: {name}")
                continue

            numeric = _to_float_or_none(self._state_values.get(name))
            if numeric is None:
                warnings.append(f"variable is not numeric: {name}")
                continue
            variables[name] = float(numeric)

        return variables, warnings

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
        variables, variable_warnings = self._collect_eval_variables()

        replacements = dict(variables)
        replacements["maxT"] = float(self._max_t)
        replacements["maxt"] = float(self._max_t)

        rendered, render_error = render_expression(self._template, replacements=replacements)
        compiled, compile_error = compile_expr(self._template)

        if render_error:
            messages = [str(render_error)]
            messages.extend(variable_warnings)
            self._set_last_error("; ".join(messages[:3]))
            return

        if compile_error or compiled is None:
            messages = [str(compile_error or "invalid expression")]
            messages.extend(variable_warnings)
            self._set_last_error("; ".join(messages[:3]))
            return

        try:
            t_preview = np.linspace(0.0, float(self._max_t), num=_PREVIEW_SAMPLES, endpoint=False, dtype=np.float64)
            preview_raw = eval_compiled(
                compiled,
                t=t_preview,
                maxt=float(self._max_t),
                variables=variables,
            )
            preview_arr = np.asarray(preview_raw, dtype=np.float64)
            if preview_arr.ndim == 0:
                preview = np.full((_PREVIEW_SAMPLES,), float(preview_arr), dtype=np.float64)
            elif preview_arr.ndim == 1:
                if preview_arr.shape[0] == _PREVIEW_SAMPLES:
                    preview = preview_arr
                elif preview_arr.shape[0] == 1:
                    preview = np.full((_PREVIEW_SAMPLES,), float(preview_arr[0]), dtype=np.float64)
                else:
                    raise ValueError(
                        f"preview result length mismatch: {preview_arr.shape[0]} != {_PREVIEW_SAMPLES}"
                    )
            else:
                raise ValueError("preview result must be scalar or 1D")
        except OPERATOR_MODEL_ERRORS as exc:
            self._set_last_error(f"preview eval failed: {type(exc).__name__}: {exc}")
            return

        self._compiled = compiled
        self._eval_variables = variables
        self._express = str(rendered or "")
        preview_values = [float(v) for v in np.asarray(preview, dtype=np.float64).tolist()]
        preview_times = [float(v) for v in np.asarray(t_preview, dtype=np.float64).tolist()]
        self._preview_cycle = list(zip(preview_times, preview_values, strict=True))
        self._publish_pending = True
        if variable_warnings:
            self._set_last_error("; ".join(variable_warnings[:3]))
        else:
            self._clear_last_error()

    async def _publish_public_state(self, *, force: bool) -> None:
        if not force and not self._publish_pending:
            return

        self._publish_pending = False
        await self._safe_set_state("express", str(self._express))
        await self._safe_set_state("preview", [list(point) for point in self._preview_cycle])
        await self._safe_publish_monitor_error(str(self._last_error))

    async def _safe_publish_monitor_error(self, message: str) -> None:
        try:
            if message:
                await self.report_error(
                    "WAVE_EXPR_ERROR",
                    message,
                    severity="error",
                    fingerprint=f"wave-expr:{message}",
                )
                return
            await self.clear_error()
        except OPERATOR_STATE_PUBLISH_ERRORS:
            logger.exception("[%s:wave_expr] failed to publish monitor error", self.node_id)

    async def _safe_set_state(self, field: str, value: Any) -> None:
        try:
            await self.set_state(field, value)
        except OPERATOR_STATE_PUBLISH_ERRORS:
            logger.exception("[%s:wave_expr] failed to publish state: %s", self.node_id, field)

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

        if self._compiled is None:
            return self._last_output

        t_raw = await self.pull("t", ctx_id=ctx_id)
        t_value = _to_float_or_none(t_raw)
        if t_value is None:
            return self._last_output

        try:
            wrapped_t = math.fmod(float(t_value), float(self._max_t))
            if wrapped_t < 0.0:
                wrapped_t += float(self._max_t)
            out = eval_scalar(
                self._compiled,
                t=wrapped_t,
                maxt=float(self._max_t),
                variables=self._eval_variables,
            )
        except OPERATOR_EVAL_ERRORS as exc:
            now_ms = int(time.time() * 1000.0)
            sig = f"{type(exc).__name__}:{exc}"
            if self._should_log_repeating_eval_error(sig, now_ms=now_ms):
                logger.exception("[%s:wave_expr] eval failed", self.node_id)
            return self._last_output

        self._last_output = float(out)
        return self._last_output


WaveExprRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.expr",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Wave Expr",
    description=(
        "Template-based waveform expression node.\n"
        "\n"
        "Core\n"
        "- `t` is cycle-domain input, not radians.\n"
        "- Runtime output evaluates with `t % maxT`.\n"
        "- Any numeric RW/WO state field can be referenced by name.\n"
        "- `express` shows the final formula after numeric state substitution.\n"
        "- `Preview` shows sampled `[t, value]` pairs over `[0, maxT)`.\n"
        "\n"
        "Oscillators\n"
        "- Phase trig: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`\n"
        "- Shape helpers: `saw`, `tri`, `pulse`\n"
        "- Tempest: `tempest(t, p, c)` where `t` is 0..1 circle phase, `p` is phase, `c` is eccentricity\n"
        "\n"
        "Utility\n"
        "- Blend and range: `clamp`, `lerp`, `smoothstep`, `saturate`\n"
        "- Phase helpers: `frac`, `wrap`\n"
        "- Selection: `cond(condition, a, b)`\n"
        "- Sequence: `sequence([a, b, c])` uses `int(t) % len(sequence)`\n"
        "- Numeric helpers: `abs`, `min`, `max`, `round`, `floor`, `ceil`, `sqrt`, `exp`, `log`, `log10`\n"
        "\n"
        "Examples\n"
        "- `0.5 + 0.5 * cos(t)`\n"
        "- `sequence([10, 20, 30, 20])`\n"
        "- `tempest(t, 0, c)`\n"
        "- `cond(t > 4, 1, 0)`"
    ),
    tags=["expr", "wave", "template", "signal"],
    dataInPorts=[
        F8DataPortSpec(
            name="t",
            description="Scalar cycle-domain input. 1.0 means one period; output evaluation uses `t % maxT`.",
            valueSchema=number_schema(),
            definitionProtected=True,
            showOnNode=True,
        ),
    ],
    dataOutPorts=[
        F8DataPortSpec(
            name="value",
            description="Expression output value for the current wrapped `t` sample.",
            valueSchema=number_schema(),
            definitionProtected=True,
            showOnNode=True,
        ),
    ],
    stateFields=[
        F8StateSpec(
            name="template",
            label="Template",
            description=(
                "Waveform expression template. `t` is cycle-domain, numeric state fields can be referenced by name, "
                "and helpers include `cond`, `sequence([...])`, `tempest(t, p, c)`, phase trig, and shaping functions."
            ),
            valueSchema=string_schema(default=_DEFAULT_TEMPLATE),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
            control=F8UiControlSpec(kind=F8UiControlKind.textarea, language="python"),
        ),
        F8StateSpec(
            name="maxT",
            label="Max T",
            description="Cycle horizon for wrapping and preview sampling. Runtime output uses `t % maxT`; preview samples `[0, maxT)`.",
            valueSchema=number_schema(default=_DEFAULT_MAX_T, minimum=1e-6),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="minValue",
            label="Min Value",
            description="Preview window lower bound (Y-axis). Auto zoom when minValue >= maxValue.",
            valueSchema=number_schema(default=_DEFAULT_MIN_VALUE),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="maxValue",
            label="Max Value",
            description="Preview window upper bound (Y-axis). Auto zoom when minValue >= maxValue.",
            valueSchema=number_schema(default=_DEFAULT_MAX_VALUE),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="express",
            label="Express",
            description="Rendered expression after numeric state substitution. `t` remains symbolic so you can inspect the final formula.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="preview",
            label="Preview",
            description="Preview waveform samples as `[t, value]` pairs over `[0, maxT)`. Changes in preview coordinates trigger redraw.",
            valueSchema={
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "default": [],
            },
            access=F8StateAccess.ro,
            valueRequired=True,
            control=F8UiControlSpec(kind=F8UiControlKind.custom, rendererKey="wave_preview"),
            showOnNode=True,
        ),
    ],
    editPolicy=F8SpecEditPolicy(stateFields=editable_collection_edit_policy()),
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(WaveExprRuntimeNode.SPEC, WaveExprRuntimeNode, overwrite=True)
    return registry
