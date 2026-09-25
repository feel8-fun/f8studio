from __future__ import annotations

from typing import Any

from f8pysdk.codec import coerce_flag
from f8pysdk.specs import (
    F8CollectionEditPolicy,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8SpecEditPolicy,
    F8StateAccess,
    F8StateSpec,
    F8UiControlKind,
    F8UiControlSpec,
    boolean_schema,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..identifiers import SERVICE_CLASS
from .categories import PALETTE_CATEGORY_CONTROL

OPERATOR_CLASS = "f8.value_stepper"
_STEP_MODE_FIXED = "fixed"
_STEP_MODE_ACCELERATED = "accelerated"
_STEP_MODE_ADAPTIVE = "adaptive"
_STEP_MODE_VALUES = [_STEP_MODE_FIXED, _STEP_MODE_ACCELERATED, _STEP_MODE_ADAPTIVE]
_ADAPTIVE_TRIGGER_WINDOW_MS = 250


class ValueStepperRuntimeNode(OperatorNode):
    """
    Studio-only numeric stepper that converts trigger-style state changes into a
    clamped numeric value.

    This is intended as a small glue operator for ControlPanel/button hotkey
    workflows, without depending on pyengine or expression nodes.
    """

    SPEC = F8OperatorSpec(
        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
        serviceClass=SERVICE_CLASS,
        paletteCategory=PALETTE_CATEGORY_CONTROL,
        operatorClass=OPERATOR_CLASS,
        version="0.0.1",
        label="Value Stepper",
        description="Studio-only numeric stepper for hotkey-driven increase/decrease control with clamp support.",
        tags=["studio", "state", "stepper", "slider", "hotkey"],
        dataInPorts=[],
        dataOutPorts=[],
        execInPorts=[],
        execOutPorts=[],
        rendererClass="default_op",
        editPolicy=F8SpecEditPolicy(
            stateFields=F8CollectionEditPolicy(canAdd=False, canDelete=False, canEditExisting=True)
        ),
        stateFields=[
            F8StateSpec(
                name="value",
                label="Value",
                description="Current output value after clamp and trigger processing.",
                valueSchema=number_schema(default=0.0),
                access=F8StateAccess.rw,
                required=True,
                control=F8UiControlSpec(kind=F8UiControlKind.slider),
                showOnNode=True,
            ),
            F8StateSpec(
                name="min",
                label="Min",
                description="Lower clamp bound.",
                valueSchema=number_schema(default=0.0),
                access=F8StateAccess.rw,
                required=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="max",
                label="Max",
                description="Upper clamp bound.",
                valueSchema=number_schema(default=1.0),
                access=F8StateAccess.rw,
                required=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="step",
                label="Step",
                description="Fixed increment/decrement size.",
                valueSchema=number_schema(default=0.01, minimum=0.0),
                access=F8StateAccess.rw,
                required=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="loop",
                label="Loop",
                description="When enabled, trigger steps wrap around the min/max range instead of clamping.",
                valueSchema=boolean_schema(default=False),
                access=F8StateAccess.rw,
                required=True,
                control=F8UiControlSpec(kind=F8UiControlKind.toggle),
                showOnNode=False,
            ),
            F8StateSpec(
                name="increaseTrigger",
                label="Increase",
                description="Increment trigger input, typically driven by a button state edge.",
                valueSchema=integer_schema(default=0),
                access=F8StateAccess.rw,
                required=True,
                control=F8UiControlSpec(kind=F8UiControlKind.button),
                showOnNode=False,
            ),
            F8StateSpec(
                name="decreaseTrigger",
                label="Decrease",
                description="Decrement trigger input, typically driven by a button state edge.",
                valueSchema=integer_schema(default=0),
                access=F8StateAccess.rw,
                required=True,
                control=F8UiControlSpec(kind=F8UiControlKind.button),
                showOnNode=False,
            ),
            F8StateSpec(
                name="stepMode",
                label="Step Mode",
                description="How trigger presses choose between the fixed and accelerated step sizes.",
                valueSchema=string_schema(default=_STEP_MODE_FIXED, enum=list(_STEP_MODE_VALUES)),
                access=F8StateAccess.rw,
                required=True,
                control=F8UiControlSpec(kind=F8UiControlKind.select),
                showOnNode=False,
            ),
            F8StateSpec(
                name="acceleratedStep",
                label="Accelerated Step",
                description="Larger step size used by accelerated mode, or by adaptive mode during rapid repeated triggers.",
                valueSchema=number_schema(default=0.05, minimum=0.0),
                access=F8StateAccess.rw,
                required=True,
                showOnNode=False,
            ),
        ],
    )

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        initial = dict(initial_state or {})
        minimum = self._convert_float_or_default(initial.get("min"), 0.0)
        maximum = self._convert_float_or_default(initial.get("max"), 1.0)
        self._minimum, self._maximum = self._ordered_bounds(minimum, maximum)
        self._step = self._convert_non_negative_float_or_default(initial.get("step"), 0.01)
        self._accelerated_step = self._convert_non_negative_float_or_default(initial.get("acceleratedStep"), 0.05)
        self._step_mode = self._coerce_step_mode(initial.get("stepMode"))
        self._loop = coerce_flag(initial.get("loop"), default=False)
        self._value = self._coerce_value_in_bounds(self._convert_float_or_default(initial.get("value"), 0.0))
        self._increase_trigger = self._convert_int_or_default(initial.get("increaseTrigger"), 0)
        self._decrease_trigger = self._convert_int_or_default(initial.get("decreaseTrigger"), 0)
        self._last_trigger_ts_ms = 0

        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
            exec_in_ports=list(node.execInPorts or []),
            exec_out_ports=list(node.execOutPorts or []),
        )

    @staticmethod
    def _convert_float_or_default(value: Any, default: float) -> float:
        if value is None:
            return float(default)
        return float(value)

    @staticmethod
    def _convert_non_negative_float_or_default(value: Any, default: float) -> float:
        coerced = ValueStepperRuntimeNode._convert_float_or_default(value, default)
        return coerced if coerced >= 0.0 else 0.0

    @staticmethod
    def _convert_int_or_default(value: Any, default: int) -> int:
        if value is None:
            return int(default)
        return int(value)

    @staticmethod
    def _ordered_bounds(minimum: float, maximum: float) -> tuple[float, float]:
        if minimum <= maximum:
            return minimum, maximum
        return maximum, minimum

    @staticmethod
    def _coerce_step_mode(value: Any) -> str:
        text = str(value or _STEP_MODE_FIXED).strip().lower()
        if text not in _STEP_MODE_VALUES:
            return _STEP_MODE_FIXED
        return text

    def _clamp_value(self, value: float) -> float:
        lower, upper = self._ordered_bounds(self._minimum, self._maximum)
        if value < lower:
            return lower
        if value > upper:
            return upper
        return value

    def _coerce_value_in_bounds(self, value: float) -> float:
        if self._loop:
            return self._wrap_value(value)
        return self._clamp_value(value)

    def _wrap_value(self, value: float) -> float:
        lower, upper = self._ordered_bounds(self._minimum, self._maximum)
        span = upper - lower
        if span <= 0.0:
            return lower
        wrapped = (float(value) - lower) % span
        return lower + wrapped

    def _effective_step(self, *, ts_ms: int | None) -> float:
        if self._step_mode == _STEP_MODE_ACCELERATED:
            return self._accelerated_step
        if self._step_mode == _STEP_MODE_ADAPTIVE:
            trigger_ts = 0 if ts_ms is None else int(ts_ms)
            if self._is_adaptive_accelerated(trigger_ts):
                return self._accelerated_step
        return self._step

    def _is_adaptive_accelerated(self, trigger_ts_ms: int) -> bool:
        last_trigger_ts_ms = int(self._last_trigger_ts_ms)
        if trigger_ts_ms <= 0 or last_trigger_ts_ms <= 0:
            return False
        elapsed_ms = trigger_ts_ms - last_trigger_ts_ms
        return 0 < elapsed_ms <= _ADAPTIVE_TRIGGER_WINDOW_MS

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms
        del meta

        if field == "value":
            return self._coerce_value_in_bounds(self._convert_float_or_default(value, self._value))
        if field == "min":
            candidate = self._convert_float_or_default(value, self._minimum)
            return candidate if candidate <= self._maximum else self._maximum
        if field == "max":
            candidate = self._convert_float_or_default(value, self._maximum)
            return candidate if candidate >= self._minimum else self._minimum
        if field == "step":
            return self._convert_non_negative_float_or_default(value, self._step)
        if field == "acceleratedStep":
            return self._convert_non_negative_float_or_default(value, self._accelerated_step)
        if field == "loop":
            return coerce_flag(value, default=self._loop)
        if field == "increaseTrigger":
            return self._convert_int_or_default(value, self._increase_trigger)
        if field == "decreaseTrigger":
            return self._convert_int_or_default(value, self._decrease_trigger)
        if field == "stepMode":
            return self._coerce_step_mode(value)
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        if field == "value":
            self._value = self._clamp_value(self._convert_float_or_default(value, self._value))
            return

        if field == "min":
            self._minimum = self._convert_float_or_default(value, self._minimum)
            await self._sync_value_after_bounds_change(ts_ms=ts_ms)
            return

        if field == "max":
            self._maximum = self._convert_float_or_default(value, self._maximum)
            await self._sync_value_after_bounds_change(ts_ms=ts_ms)
            return

        if field == "step":
            self._step = self._convert_non_negative_float_or_default(value, self._step)
            return

        if field == "acceleratedStep":
            self._accelerated_step = self._convert_non_negative_float_or_default(value, self._accelerated_step)
            return

        if field == "loop":
            self._loop = coerce_flag(value, default=self._loop)
            coerced_value = self._coerce_value_in_bounds(self._value)
            if coerced_value == self._value:
                return
            self._value = coerced_value
            await self.set_state("value", coerced_value, ts_ms=ts_ms)
            return

        if field == "stepMode":
            self._step_mode = self._coerce_step_mode(value)
            return

        if field == "increaseTrigger":
            self._increase_trigger = self._convert_int_or_default(value, self._increase_trigger)
            await self._apply_trigger(delta=1.0, ts_ms=ts_ms)
            return

        if field == "decreaseTrigger":
            self._decrease_trigger = self._convert_int_or_default(value, self._decrease_trigger)
            await self._apply_trigger(delta=-1.0, ts_ms=ts_ms)
            return

    async def _sync_value_after_bounds_change(self, *, ts_ms: int | None) -> None:
        self._minimum, self._maximum = self._ordered_bounds(self._minimum, self._maximum)
        next_value = self._coerce_value_in_bounds(self._value)
        if next_value == self._value:
            return
        self._value = next_value
        await self.set_state("value", next_value, ts_ms=ts_ms)

    async def _apply_trigger(self, *, delta: float, ts_ms: int | None) -> None:
        next_value = self._coerce_value_in_bounds(self._value + (delta * self._effective_step(ts_ms=ts_ms)))
        self._value = next_value
        await self.set_state("value", next_value, ts_ms=ts_ms)
        trigger_ts = 0 if ts_ms is None else int(ts_ms)
        self._last_trigger_ts_ms = trigger_ts


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(ValueStepperRuntimeNode.SPEC, ValueStepperRuntimeNode, overwrite=True)
    return registry
