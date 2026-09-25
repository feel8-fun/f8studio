from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from types import CodeType
from typing import Any, cast

from f8pysdk.codec import coerce_bool
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry
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
    any_schema,
    boolean_schema,
    editable_collection_edit_policy,
    string_schema,
)

from ..identifiers import SERVICE_CLASS
from ._py_expr_eval import compile_expr, is_identifier, normalize_expr_code, safe_eval_compiled, unwrap_value, wrap_value
from .categories import PALETTE_CATEGORY_EXPR

OPERATOR_CLASS = "f8.data_expr"
_ERROR_LOG_INTERVAL_MS = 5_000

logger = logging.getLogger(__name__)


class DataExprRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        data_in_ports = [port.name for port in (node.dataInPorts or [])] or ["input"]
        data_out_ports = [port.name for port in (node.dataOutPorts or [])] or ["out"]
        state_fields = [field.name for field in (node.stateFields or [])] or ["code"]
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=data_in_ports,
            data_out_ports=data_out_ports,
            state_fields=state_fields,
        )
        state = dict(initial_state or {})
        self._code = normalize_expr_code(state.get("code") or "input")
        self._allow_numpy = coerce_bool(state.get("allowNumpy"), default=False)
        self._unpack_dict_outputs = coerce_bool(state.get("unpackDictOutputs"), default=False)
        self._compiled: CodeType | None = None
        self._compile_error: str | None = None
        self._last_ctx_id: str | int | None = None
        self._last_outputs: dict[str, object] = {}
        self._dirty = True
        self._last_error_signature = ""
        self._last_error_log_ms = 0
        self._recompile()

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        if field in {"allowNumpy", "unpackDictOutputs"}:
            return coerce_bool(value, default=False)
        if field == "code":
            return normalize_expr_code(value)
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        if field == "allowNumpy":
            self._allow_numpy = coerce_bool(value, default=False)
            self._recompile()
        elif field == "unpackDictOutputs":
            self._unpack_dict_outputs = coerce_bool(value, default=False)
        elif field == "code":
            self._code = normalize_expr_code(value)
            self._recompile()
        else:
            return
        self._dirty = True

    def _recompile(self) -> None:
        self._compiled, self._compile_error = compile_expr(self._code, allow_numpy=self._allow_numpy)

    def _build_eval_names(self, inputs: Mapping[str, object]) -> dict[str, object]:
        wrapped_inputs = {name: wrap_value(value) for name, value in inputs.items()}
        names: dict[str, object] = {"inputs": wrapped_inputs}
        names.update({name: value for name, value in wrapped_inputs.items() if is_identifier(name)})
        return names

    def _default_output_port(self) -> str | None:
        if "out" in self.data_out_ports:
            return "out"
        return self.data_out_ports[0] if self.data_out_ports else None

    def _extract_outputs(self, result: object) -> dict[str, object]:
        if isinstance(result, Mapping) and self._unpack_dict_outputs:
            outputs: dict[str, object] = {}
            result_mapping = cast(Mapping[object, object], result)
            for raw_name, value in result_mapping.items():
                name = str(raw_name)
                if name in self.data_out_ports:
                    outputs[name] = unwrap_value(value)
                else:
                    logger.warning("[%s:data_expr] unpack key has no matching output port: %s", self.node_id, name)
            return outputs
        default_port = self._default_output_port()
        return {} if default_port is None else {default_port: unwrap_value(cast(object, result))}

    def _log_eval_failure(self, exc: Exception) -> None:
        now_ms = int(time.time() * 1_000)
        signature = f"{type(exc).__name__}:{exc}"
        if signature == self._last_error_signature and now_ms - self._last_error_log_ms < _ERROR_LOG_INTERVAL_MS:
            return
        self._last_error_signature = signature
        self._last_error_log_ms = now_ms
        logger.exception("[%s:data_expr] expression evaluation failed", self.node_id, exc_info=exc)

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if port not in self.data_out_ports:
            return None
        if not self._dirty and ctx_id is not None and ctx_id == self._last_ctx_id:
            return self._last_outputs.get(port)

        inputs: dict[str, object] = {}
        for input_port in self.data_in_ports:
            try:
                inputs[input_port] = await self.pull(input_port, ctx_id=ctx_id)
            except (LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.exception("[%s:data_expr] input pull failed for port %s", self.node_id, input_port, exc_info=exc)
                inputs[input_port] = None

        try:
            if self._compiled is None:
                raise ValueError(self._compile_error or "invalid expression")
            result = safe_eval_compiled(self._compiled, names=self._build_eval_names(inputs))
        except Exception as exc:
            # This is the boundary for arbitrary user-authored expressions.
            self._log_eval_failure(exc)
            await self.report_error(
                "DATA_EXPR_ERROR",
                f"{type(exc).__name__}: {exc}",
                fingerprint=f"web-studio-data-expr:{type(exc).__name__}:{exc}",
            )
            result = None
        else:
            await self.clear_error()

        self._last_outputs = self._extract_outputs(result)
        self._last_ctx_id = ctx_id
        self._dirty = False
        return self._last_outputs.get(port)


DataExprRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=PALETTE_CATEGORY_EXPR,
    operatorClass=OPERATOR_CLASS,
    version="0.1.0",
    label="Studio Data Expr",
    description="Evaluate a restricted expression over dynamic data inputs in the Web Studio runtime.",
    tags=["studio", "web", "expr", "data", "transform"],
    dataInPorts=[F8DataPortSpec(name="x", description="Expression input.", valueSchema=any_schema(), required=False)],
    dataOutPorts=[F8DataPortSpec(name="out", description="Expression result.", valueSchema=any_schema(), required=False)],
    editPolicy=F8SpecEditPolicy(
        dataInPorts=editable_collection_edit_policy(),
        dataOutPorts=editable_collection_edit_policy(),
    ),
    stateFields=[
        F8StateSpec(
            name="allowNumpy",
            label="Allow NumPy",
            description="Reserved for a runtime with an explicitly installed NumPy capability.",
            control=F8UiControlSpec(kind=F8UiControlKind.toggle),
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=False,
            required=False,
        ),
        F8StateSpec(
            name="unpackDictOutputs",
            label="Unpack Dict Outputs",
            description="Map dictionary keys to output ports with the same names.",
            control=F8UiControlSpec(kind=F8UiControlKind.toggle),
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=False,
            required=False,
        ),
        F8StateSpec(
            name="code",
            label="Expr",
            description="Restricted Python expression using input names or the inputs mapping.",
            control=F8UiControlSpec(kind=F8UiControlKind.textarea, language="python"),
            valueSchema=string_schema(default="x"),
            access=F8StateAccess.rw,
            showOnNode=True,
            required=True,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(DataExprRuntimeNode.SPEC, DataExprRuntimeNode, overwrite=True)
    return registry


__all__ = ["DataExprRuntimeNode", "register_operator"]
