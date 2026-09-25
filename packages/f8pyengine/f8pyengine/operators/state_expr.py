from __future__ import annotations

import logging
import time
from typing import Any

from f8pysdk.codec import coerce_bool
from f8pysdk.specs import (
    F8UiControlKind,
    F8UiControlSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8SpecEditPolicy,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    editable_collection_edit_policy,
    string_schema,
)
from f8pysdk.specs import UNSET
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS
from ._py_expr_eval import (
    compile_expr,
    is_identifier,
    normalize_expr_code,
    np,
    safe_eval_compiled,
    unwrap_wrapped_value,
    wrap_value,
)
from ._runtime_errors import OPERATOR_EVAL_ERRORS, OPERATOR_STATE_PUBLISH_ERRORS, OPERATOR_VALUE_COMPARE_ERRORS


OPERATOR_CLASS = "f8.state_expr"

logger = logging.getLogger(__name__)

_PROTECTED_STATE_FIELDS = {
    "allowNumpy",
    "code",
    "operatorId",
    "out",
    "svcId",
}

_UNPUBLISHED = object()


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


def _access_text(access: F8StateAccess | str | None) -> str:
    if isinstance(access, F8StateAccess):
        raw_access = access.value
    else:
        raw_access = access
    return str(raw_access or "").strip().lower()


def _is_symbol_state_field(field: F8StateSpec) -> bool:
    name = str(field.name or "").strip()
    if not name or name in _PROTECTED_STATE_FIELDS:
        return False
    return _access_text(field.access) in {"rw", "wo"}


class StateExprRuntimeNode(OperatorNode):
    """
    State-driven expression operator.

    - Editable state fields become expression symbols.
    - Result is published to read-only state `out`.
    - No data ports are involved.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        state_fields = [str(field.name) for field in list(node.stateFields or [])] or [
            "allowNumpy",
            "code",
            "out",
        ]
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[],
            data_out_ports=[],
            state_fields=state_fields,
        )

        self._symbol_state_names = [
            str(field.name)
            for field in list(node.stateFields or [])
            if _is_symbol_state_field(field)
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

        self._allow_numpy = coerce_bool(self._state_values.get("allowNumpy"), default=False)
        self._code = normalize_expr_code(self._state_values.get("code") or "0")
        self._compiled = None
        self._compile_error: str | None = None
        self._out_value: Any = None
        self._last_error = ""
        self._published_out_value: Any = _UNPUBLISHED
        self._published_last_error: Any = _UNPUBLISHED

        self._last_eval_exc_sig = ""
        self._last_eval_exc_log_ts_ms = 0
        self._last_publish_exc_sig = ""
        self._last_publish_exc_log_ts_ms = 0

        self._recompile()
        self._evaluate_current_output()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        if not active:
            return
        await self._publish_public_state(force=True)

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "allowNumpy":
            return coerce_bool(value, default=False)
        if name == "code":
            return normalize_expr_code(value)
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name == "out":
            return

        if name == "allowNumpy":
            self._allow_numpy = coerce_bool(value, default=self._allow_numpy)
            self._state_values[name] = self._allow_numpy
            self._recompile()
            self._evaluate_current_output()
            await self._publish_public_state(force=False)
            return

        if name == "code":
            self._code = normalize_expr_code(value)
            self._state_values[name] = self._code
            self._recompile()
            self._evaluate_current_output()
            await self._publish_public_state(force=False)
            return

        self._state_values[name] = value
        if name not in self._symbol_state_names:
            return

        self._evaluate_current_output()
        await self._publish_public_state(force=False)

    def _should_log_repeating_error(self, sig: str, *, now_ms: int, kind: str) -> bool:
        if kind == "eval":
            if sig != self._last_eval_exc_sig:
                self._last_eval_exc_sig = sig
                self._last_eval_exc_log_ts_ms = int(now_ms)
                return True
            if (int(now_ms) - int(self._last_eval_exc_log_ts_ms)) >= 5000:
                self._last_eval_exc_log_ts_ms = int(now_ms)
                return True
            return False

        if kind == "publish":
            if sig != self._last_publish_exc_sig:
                self._last_publish_exc_sig = sig
                self._last_publish_exc_log_ts_ms = int(now_ms)
                return True
            if (int(now_ms) - int(self._last_publish_exc_log_ts_ms)) >= 5000:
                self._last_publish_exc_log_ts_ms = int(now_ms)
                return True
            return False

        return True

    def _recompile(self) -> None:
        compiled, err = compile_expr(self._code, allow_numpy=self._allow_numpy)
        self._compiled = compiled
        self._compile_error = err

    def _build_eval_names(self) -> dict[str, Any]:
        symbol_values: dict[str, Any] = {}
        for name in self._symbol_state_names:
            symbol_values[name] = self._state_values.get(name)

        env: dict[str, Any] = {}
        env["states"] = {name: wrap_value(value) for name, value in symbol_values.items()}
        for name, value in symbol_values.items():
            if is_identifier(name):
                env[name] = wrap_value(value)
        return env

    def _set_last_error(self, message: str) -> None:
        self._last_error = str(message or "").strip()

    def _clear_last_error(self) -> None:
        self._last_error = ""

    def _evaluate_current_output(self) -> None:
        if self._compiled is None:
            self._out_value = None
            self._set_last_error(self._compile_error or "invalid expression")
            return

        try:
            result = safe_eval_compiled(
                self._compiled,
                names=self._build_eval_names(),
                allow_numpy=self._allow_numpy,
            )
        except OPERATOR_EVAL_ERRORS as exc:
            now_ms = int(time.time() * 1000.0)
            sig = f"{type(exc).__name__}:{exc}"
            if self._should_log_repeating_error(sig, now_ms=now_ms, kind="eval"):
                logger.warning("[%s:state_expr] eval failed: %s", self.node_id, exc, exc_info=True)
            self._out_value = None
            self._set_last_error(f"{type(exc).__name__}: {exc}")
            return

        self._out_value = unwrap_wrapped_value(result)
        self._clear_last_error()

    async def _publish_public_state(self, *, force: bool) -> None:
        if force or not self._values_equal(self._published_out_value, self._out_value):
            await self._safe_set_state("out", self._out_value)
            self._published_out_value = self._out_value

        if force or not self._values_equal(self._published_last_error, self._last_error):
            await self._safe_publish_monitor_error(self._last_error)
            self._published_last_error = self._last_error

    async def _safe_publish_monitor_error(self, message: str) -> None:
        try:
            if message:
                await self.report_error(
                    "STATE_EXPR_ERROR",
                    message,
                    severity="error",
                    fingerprint=f"state-expr:{message}",
                )
                return
            await self.clear_error()
        except OPERATOR_STATE_PUBLISH_ERRORS as exc:
            now_ms = int(time.time() * 1000.0)
            sig = f"monitor:{type(exc).__name__}:{exc}"
            if self._should_log_repeating_error(sig, now_ms=now_ms, kind="publish"):
                logger.exception("[%s:state_expr] failed to publish monitor error", self.node_id)

    async def _safe_set_state(self, field: str, value: Any) -> None:
        try:
            await self.set_state(field, value)
        except OPERATOR_STATE_PUBLISH_ERRORS as exc:
            now_ms = int(time.time() * 1000.0)
            sig = f"{field}:{type(exc).__name__}:{exc}"
            if self._should_log_repeating_error(sig, now_ms=now_ms, kind="publish"):
                logger.exception("[%s:state_expr] failed to publish state: %s", self.node_id, field)

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if left is _UNPUBLISHED or right is _UNPUBLISHED:
            return False
        try:
            return bool(left == right)
        except OPERATOR_VALUE_COMPARE_ERRORS:
            return False


StateExprRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.expr",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="State Expr",
    description=(
        "Evaluate a small Python expression using state fields as symbols.\n"
        "\n"
        "Core\n"
        "- Editable RW/WO state fields become expression symbols directly.\n"
        "- The expression result is published to read-only state `out`.\n"
        "- State names that are not valid Python identifiers remain available through `states['field-name']`.\n"
        "- No data ports are involved.\n"
        "\n"
        "Available\n"
        "- Builtins: `abs`, `min`, `max`, `round`, `float`, `int`, `len`, `sum`, `sorted`, `range`, `any`, `all`, `sigmoid`\n"
        "\n"
        "Examples\n"
        "- `sigmoid(x)`\n"
        "- `a + b`\n"
        "- `config.center.x`\n"
        "- `states['left-value'] * gain`\n"
        "- `math.sin(phase)`"
    ),
    tags=["expr", "state", "logic", "transform"],
    stateFields=[
        F8StateSpec(
            name="allowNumpy",
            label="Allow Numpy",
            description="Enable `np.*` and `numpy.*` inside the expression.",
            control=F8UiControlSpec(kind=F8UiControlKind.toggle),
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=False,
            required=False,
        ),
        F8StateSpec(
            name="code",
            label="Expr",
            description=(
                "Single Python expression. Editable RW/WO state fields are available directly by name; "
                "non-identifier names remain available through `states[...]`."
            ),
            control=F8UiControlSpec(kind=F8UiControlKind.textarea, language="python"),
            valueSchema=string_schema(default="0"),
            access=F8StateAccess.rw,
            showOnNode=True,
            required=True,
        ),
        F8StateSpec(
            name="out",
            label="Out",
            description="Expression result published by the node.",
            valueSchema=any_schema(),
            access=F8StateAccess.ro,
            showOnNode=True,
            required=True,
        ),
    ],
    editPolicy=F8SpecEditPolicy(stateFields=editable_collection_edit_policy()),
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(StateExprRuntimeNode.SPEC, StateExprRuntimeNode, overwrite=True)
    return registry
