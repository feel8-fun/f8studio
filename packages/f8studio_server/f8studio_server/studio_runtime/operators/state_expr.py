from __future__ import annotations

import logging
import time
from types import CodeType
from typing import Any

from f8pysdk.codec import coerce_bool
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry
from f8pysdk.specs import (
    F8DataTypeSchema,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8SpecEditPolicy,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    editable_collection_edit_policy,
    schema_default,
    string_schema,
)

from ..identifiers import SERVICE_CLASS
from ._py_expr_eval import compile_expr, is_identifier, normalize_expr_code, safe_eval_compiled, unwrap_value, wrap_value
from .categories import PALETTE_CATEGORY_EXPR

OPERATOR_CLASS = "f8.state_expr"
_PROTECTED_STATE_FIELDS = frozenset({"allowNumpy", "code", "operatorId", "out", "svcId"})
_UNPUBLISHED = object()
_ERROR_LOG_INTERVAL_MS = 5_000

logger = logging.getLogger(__name__)


def _default_value(schema: F8DataTypeSchema) -> object:
    return schema_default(schema)


def _is_symbol_field(field: F8StateSpec) -> bool:
    return bool(field.name) and field.name not in _PROTECTED_STATE_FIELDS and field.access in {
        F8StateAccess.rw,
        F8StateAccess.wo,
    }


class StateExprRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        fields = list(node.stateFields or [])
        state_fields = [field.name for field in fields] or ["allowNumpy", "code", "out"]
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[],
            data_out_ports=[],
            state_fields=state_fields,
        )
        self._symbol_state_names = [field.name for field in fields if _is_symbol_field(field)]
        self._state_values: dict[str, object] = {
            field.name: _default_value(field.valueSchema)
            for field in fields
            if _default_value(field.valueSchema) is not None
        }
        self._state_values.update(dict(initial_state or {}))
        self._allow_numpy = coerce_bool(self._state_values.get("allowNumpy"), default=False)
        self._code = normalize_expr_code(self._state_values.get("code") or "0")
        self._compiled: CodeType | None = None
        self._compile_error: str | None = None
        self._out_value: object = None
        self._last_error = ""
        self._published_out_value: object = _UNPUBLISHED
        self._published_error: object = _UNPUBLISHED
        self._last_error_signature = ""
        self._last_error_log_ms = 0
        self._recompile()
        self._evaluate()

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        if field == "allowNumpy":
            return coerce_bool(value, default=False)
        if field == "code":
            return normalize_expr_code(value)
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        if field == "out":
            return
        if field == "allowNumpy":
            self._allow_numpy = coerce_bool(value, default=False)
            self._state_values[field] = self._allow_numpy
            self._recompile()
        elif field == "code":
            self._code = normalize_expr_code(value)
            self._state_values[field] = self._code
            self._recompile()
        elif field in self._symbol_state_names:
            self._state_values[field] = value
        else:
            return
        self._evaluate()
        await self._publish_public_state(force=False)

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        if active:
            await self._publish_public_state(force=True)

    def _recompile(self) -> None:
        self._compiled, self._compile_error = compile_expr(self._code, allow_numpy=self._allow_numpy)

    def _build_eval_names(self) -> dict[str, object]:
        symbols = {name: self._state_values.get(name) for name in self._symbol_state_names}
        wrapped = {name: wrap_value(value) for name, value in symbols.items()}
        names: dict[str, object] = {"states": wrapped}
        names.update({name: value for name, value in wrapped.items() if is_identifier(name)})
        return names

    def _log_eval_failure(self, exc: Exception) -> None:
        now_ms = int(time.time() * 1_000)
        signature = f"{type(exc).__name__}:{exc}"
        if signature == self._last_error_signature and now_ms - self._last_error_log_ms < _ERROR_LOG_INTERVAL_MS:
            return
        self._last_error_signature = signature
        self._last_error_log_ms = now_ms
        logger.exception("[%s:state_expr] expression evaluation failed", self.node_id, exc_info=exc)

    def _evaluate(self) -> None:
        if self._compiled is None:
            self._out_value = None
            self._last_error = self._compile_error or "invalid expression"
            return
        try:
            self._out_value = unwrap_value(safe_eval_compiled(self._compiled, names=self._build_eval_names()))
        except Exception as exc:
            # This is the boundary for arbitrary user-authored expressions.
            self._log_eval_failure(exc)
            self._out_value = None
            self._last_error = f"{type(exc).__name__}: {exc}"
            return
        self._last_error = ""

    async def _publish_public_state(self, *, force: bool) -> None:
        if force or not self._values_equal(self._published_out_value, self._out_value):
            await self.set_state("out", self._out_value)
            self._published_out_value = self._out_value
        if force or not self._values_equal(self._published_error, self._last_error):
            if self._last_error:
                await self.report_error(
                    "STATE_EXPR_ERROR",
                    self._last_error,
                    fingerprint=f"web-studio-state-expr:{self._last_error}",
                )
            else:
                await self.clear_error()
            self._published_error = self._last_error

    @staticmethod
    def _values_equal(left: object, right: object) -> bool:
        if left is _UNPUBLISHED or right is _UNPUBLISHED:
            return False
        try:
            result = left == right
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("state expression output comparison failed", exc_info=exc)
            return False
        return result


StateExprRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=PALETTE_CATEGORY_EXPR,
    operatorClass=OPERATOR_CLASS,
    version="0.1.0",
    label="Studio State Expr",
    description="Publish a derived state value from writable state symbols in the Web Studio runtime.",
    tags=["studio", "web", "expr", "state", "transform"],
    stateFields=[
        F8StateSpec(
            name="allowNumpy",
            label="Allow NumPy",
            description="Reserved for a runtime with an explicitly installed NumPy capability.",
            uiControl="toggle",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=False,
            required=False,
        ),
        F8StateSpec(
            name="x",
            description="Starter writable expression symbol.",
            valueSchema=any_schema(),
            access=F8StateAccess.rw,
            showOnNode=True,
            required=False,
        ),
        F8StateSpec(
            name="code",
            label="Expr",
            description="Restricted Python expression using writable state names or the states mapping.",
            uiControl="wrapline[python]",
            valueSchema=string_schema(default="x"),
            access=F8StateAccess.rw,
            showOnNode=True,
            required=True,
        ),
        F8StateSpec(
            name="out",
            label="Out",
            description="Read-only expression result.",
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


__all__ = ["StateExprRuntimeNode", "register_operator"]
