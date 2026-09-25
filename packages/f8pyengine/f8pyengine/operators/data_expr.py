from __future__ import annotations

import logging
import time
from types import CodeType
from typing import Any

from f8pysdk.codec import coerce_bool
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
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS
from ._py_expr_eval import (
    compile_expr as _compile_expr,
    is_identifier as _is_identifier,
    normalize_expr_code,
    np,
    safe_eval_compiled as _safe_eval_compiled,
    unwrap_wrapped_value,
    wrap_value as _wrap_value,
)


OPERATOR_CLASS = "f8.data_expr"

logger = logging.getLogger(__name__)


class DataExprRuntimeNode(OperatorNode):
    """
    Lightweight expression operator.

    - Multiple data inputs (editable)
    - Single output: `out`
    - Single state field: `code`
    - No external state access; expression only sees input values

    Recommended usage:
      input.center.x
      a + b - c**2
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        data_in_ports = [p.name for p in (node.dataInPorts or [])] or ["input"]
        data_out_ports = [p.name for p in (node.dataOutPorts or [])] or ["out"]
        state_fields = [s.name for s in (node.stateFields or [])] or ["code"]
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=data_in_ports,
            data_out_ports=data_out_ports,
            state_fields=state_fields,
        )
        self._initial_state = dict(initial_state or {})
        self._code = self._normalize_code(self._initial_state.get("code") or "input")
        self._allow_numpy = coerce_bool(self._initial_state.get("allowNumpy"), default=False)
        self._unpack_dict_outputs = coerce_bool(self._initial_state.get("unpackDictOutputs"), default=False)
        self._compiled: CodeType | None = None
        self._compile_error: str | None = None
        self._recompile()

        self._last_ctx_id: str | int | None = None
        self._last_outputs: dict[str, Any] = {}
        self._dirty: bool = True
        self._last_eval_exc_sig: str = ""
        self._last_eval_exc_log_ts_ms: int = 0
        self._last_pull_exc_sig: str = ""
        self._last_pull_exc_log_ts_ms: int = 0
        self._last_unmatched_exc_sig: str = ""
        self._last_unmatched_exc_log_ts_ms: int = 0

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

        if kind == "pull":
            if sig != self._last_pull_exc_sig:
                self._last_pull_exc_sig = sig
                self._last_pull_exc_log_ts_ms = int(now_ms)
                return True
            if (int(now_ms) - int(self._last_pull_exc_log_ts_ms)) >= 5000:
                self._last_pull_exc_log_ts_ms = int(now_ms)
                return True
            return False
        if kind == "unmatched":
            if sig != self._last_unmatched_exc_sig:
                self._last_unmatched_exc_sig = sig
                self._last_unmatched_exc_log_ts_ms = int(now_ms)
                return True
            if (int(now_ms) - int(self._last_unmatched_exc_log_ts_ms)) >= 5000:
                self._last_unmatched_exc_log_ts_ms = int(now_ms)
                return True
            return False

        return True

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        _ = ts_ms
        name = str(field or "")
        if name == "allowNumpy":
            self._allow_numpy = coerce_bool(value, default=False)
            self._recompile()
            self._dirty = True
            return
        if name == "unpackDictOutputs":
            self._unpack_dict_outputs = coerce_bool(value, default=False)
            self._dirty = True
            return
        if name != "code":
            return
        self._code = self._normalize_code(value)
        self._recompile()
        self._dirty = True

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "allowNumpy":
            return coerce_bool(value, default=False)
        if name == "unpackDictOutputs":
            return coerce_bool(value, default=False)
        if name == "code":
            return self._normalize_code(value)
        return value

    @staticmethod
    def _normalize_code(value: Any) -> str:
        """
        Expr is a single-line expression. Normalize multi-line edits (paste, UI)
        into a single line to avoid hidden '\n' semantics.
        """
        return normalize_expr_code(value)

    def _recompile(self) -> None:
        compiled, err = _compile_expr(self._code, allow_numpy=self._allow_numpy)
        self._compiled = compiled
        self._compile_error = err

    def _build_eval_names(self, inputs: dict[str, Any]) -> dict[str, Any]:
        env: dict[str, Any] = {}
        # Always provide `inputs` mapping for non-identifier port names.
        env["inputs"] = {k: _wrap_value(v) for k, v in inputs.items()}
        for k, v in inputs.items():
            if _is_identifier(k):
                env[k] = _wrap_value(v)
        return env

    def _default_output_port(self) -> str | None:
        if "out" in self.data_out_ports:
            return "out"
        if self.data_out_ports:
            return str(self.data_out_ports[0])
        return None

    def _normalize_output_value(self, value: Any) -> Any:
        return unwrap_wrapped_value(value)

    def _extract_outputs(self, result: Any) -> dict[str, Any]:
        outputs: dict[str, Any] = {}

        if isinstance(result, dict) and self._unpack_dict_outputs:
            for key, value in result.items():
                key_s = str(key)
                if key_s in self.data_out_ports:
                    outputs[key_s] = self._normalize_output_value(value)
                    continue
                now_ms = int(time.time() * 1000.0)
                sig = f"unmatched_port:{key_s}"
                if self._should_log_repeating_error(sig, now_ms=now_ms, kind="unmatched"):
                    logger.warning("[%s:data_expr] unpack key has no matching output port: %s", self.node_id, key_s)
            return outputs

        default_port = self._default_output_port()
        if default_port is None:
            return outputs
        outputs[default_port] = self._normalize_output_value(result)
        return outputs

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        out_port = str(port or "")
        if out_port not in self.data_out_ports:
            return None
        if not self._dirty and ctx_id is not None and ctx_id == self._last_ctx_id:
            return self._last_outputs.get(out_port)

        pulled: dict[str, Any] = {}
        for p in list(self.data_in_ports or []):
            try:
                pulled[str(p)] = await self.pull(str(p), ctx_id=ctx_id)
            except Exception as exc:
                pulled[str(p)] = None
                now_ms = int(time.time() * 1000.0)
                sig = f"{type(exc).__name__}:{exc}:port={p}"
                if self._should_log_repeating_error(sig, now_ms=now_ms, kind="pull"):
                    logger.exception("[%s:data_expr] pull failed (port=%s)", self.node_id, p)

        try:
            if self._compiled is None:
                raise ValueError(self._compile_error or "invalid expression")
            out = _safe_eval_compiled(
                self._compiled,
                names=self._build_eval_names(pulled),
                allow_numpy=self._allow_numpy,
            )
        except Exception as exc:
            now_ms = int(time.time() * 1000.0)
            sig = f"{type(exc).__name__}:{exc}"
            if self._should_log_repeating_error(sig, now_ms=now_ms, kind="eval"):
                logger.warning("[%s:data_expr] eval failed: %s", self.node_id, exc)
            out = None

        self._last_outputs = self._extract_outputs(out)
        self._last_ctx_id = ctx_id
        self._dirty = False
        return self._last_outputs.get(out_port)


DataExprRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.expr",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Data Expr",
    description=(
        "Evaluate a small Python expression using input values.\n"
        "\n"
        "Core\n"
        "- The main input is `x`, and any additional input port name can be referenced directly.\n"
        "- The expression is a single Python expression, not a statement block.\n"
        "- The result is emitted on `out`.\n"
        "- If `Unpack Dict Outputs` is enabled and the result is a dict, matching output ports receive matching keys.\n"
        "\n"
        "Available\n"
        "- Builtins: `abs`, `min`, `max`, `round`, `float`, `int`, `len`, `sum`, `sorted`, `range`, `any`, `all`, `sigmoid`\n"
        "- Python expressions: indexing, dict/list/tuple literals, comprehensions, conditionals\n"
        "- Math namespace: `math.*`\n"
        "- Optional numpy namespace: `np.*` and `numpy.*` when `Allow Numpy` is enabled\n"
        "\n"
        "Examples\n"
        "- `x * 0.5`\n"
        "- `max(0, x)`\n"
        "- `{'left': x[0], 'right': x[1]}`\n"
        "- `[value * 2 for value in x]`"
    ),
    tags=["expr", "math", "logic", "transform", "lightweight"],
    dataInPorts=[
        F8DataPortSpec(name="x", description="Input value for the expression.", valueSchema=any_schema(), required=False),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="out", description="Expression result.", valueSchema=any_schema(), required=False),
    ],
    editPolicy=F8SpecEditPolicy(
        dataInPorts=editable_collection_edit_policy(),
        dataOutPorts=editable_collection_edit_policy(),
    ),
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
            name="unpackDictOutputs",
            label="Unpack Dict Outputs",
            description="When enabled, dict results are unpacked into output ports with matching names.",
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
                "Single Python expression. Reference `x` and any extra input port names directly. Supports literals, "
                "indexing, comprehensions, conditionals, `math.*`, and optional `np.*` / `numpy.*` when `Allow Numpy` is enabled."
            ),
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
