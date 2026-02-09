from __future__ import annotations

import ast
import logging
import math
from dataclasses import dataclass
from types import CodeType
from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    string_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS


OPERATOR_CLASS = "f8.expr"

logger = logging.getLogger(__name__)


_ALLOWED_GLOBAL_FNS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
}

_ALLOWED_MATH_FNS: set[str] = {
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sqrt",
    "log",
    "log10",
    "exp",
    "floor",
    "ceil",
}


def _is_identifier(name: str) -> bool:
    try:
        return bool(name) and name.isidentifier()
    except Exception:
        return False


def _wrap_value(v: Any) -> Any:
    if isinstance(v, (dict, list, tuple)):
        return _JsonRef(v)
    return v


@dataclass(frozen=True)
class _JsonRef:
    """
    Wrapper for JSON-like values that supports attribute access for dict keys.

    Example:
      input.center.x  <=>  input["center"]["x"]

    This intentionally blocks dunder/private access to avoid escaping into python internals.
    """

    value: Any

    def _deny_attr(self, name: str) -> bool:
        s = str(name or "")
        return not s or s.startswith("_")

    def __getattr__(self, name: str) -> "_JsonRef":
        if self._deny_attr(name):
            raise AttributeError(name)
        v = self.value
        if isinstance(v, dict) and name in v:
            return _JsonRef(v[name])
        raise AttributeError(name)

    def __getitem__(self, key: Any) -> "_JsonRef":
        v = self.value
        if isinstance(v, dict):
            if isinstance(key, str) and key.startswith("_"):
                raise KeyError(key)
            return _JsonRef(v[key])
        if isinstance(v, (list, tuple)):
            return _JsonRef(v[int(key)])
        raise TypeError(f"not indexable: {type(v).__name__}")

    def unwrap(self) -> Any:
        v = self.value
        if isinstance(v, dict):
            return {str(k): _JsonRef(x).unwrap() for k, x in v.items()}
        if isinstance(v, list):
            return [_JsonRef(x).unwrap() for x in v]
        if isinstance(v, tuple):
            return tuple(_JsonRef(x).unwrap() for x in v)
        return v


class _ExprValidator(ast.NodeVisitor):
    """
    Validate a Python expression AST for safe evaluation.

    Allows:
    - literals, names, indexing, attribute access (for _JsonRef), arithmetic, boolean ops, comparisons
    - calls to a small allowlist: abs/min/max/round, and math.<fn> (where fn is allowlisted)
    """

    def __init__(self) -> None:
        super().__init__()
        self._errors: list[str] = []

    def error(self, msg: str) -> None:
        self._errors.append(str(msg))

    def validate(self, expr: str) -> tuple[ast.Expression | None, str | None]:
        try:
            tree = ast.parse(str(expr or ""), mode="eval")
        except SyntaxError as exc:
            return None, f"syntax error: {exc.msg}"
        try:
            self.visit(tree)
        except Exception as exc:
            return None, f"validation error: {exc}"
        if self._errors:
            return None, "; ".join(self._errors[:3])
        if not isinstance(tree, ast.Expression):
            return None, "not an expression"
        return tree, None

    def generic_visit(self, node: ast.AST) -> Any:
        allowed: tuple[type[ast.AST], ...] = (
            ast.Expression,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Attribute,
            ast.Subscript,
            ast.Slice,
            ast.Tuple,
            ast.List,
            ast.Dict,
            ast.UnaryOp,
            ast.UAdd,
            ast.USub,
            ast.Not,
            ast.BinOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Is,
            ast.IsNot,
            ast.In,
            ast.NotIn,
            ast.IfExp,
            ast.Call,
            ast.keyword,
        )
        index_node = getattr(ast, "Index", None)
        if index_node is not None:
            allowed = (*allowed, index_node)
        if not isinstance(node, allowed):
            self.error(f"disallowed syntax: {type(node).__name__}")
            return None
        return super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if str(node.attr or "").startswith("_"):
            self.error("private/dunder attribute access is not allowed")
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        # Allow abs/min/max/round(...)
        if isinstance(node.func, ast.Name):
            if str(node.func.id) not in _ALLOWED_GLOBAL_FNS:
                self.error(f"call not allowed: {node.func.id}")
                return None
        # Allow math.<fn>(...)
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
            fn = str(node.func.attr or "")
            if fn not in _ALLOWED_MATH_FNS:
                self.error(f"math call not allowed: math.{fn}")
                return None
        else:
            self.error("call target not allowed")
            return None
        return self.generic_visit(node)


def _compile_expr(expr: str) -> tuple[CodeType | None, str | None]:
    validator = _ExprValidator()
    tree, err = validator.validate(expr)
    if tree is None:
        return None, str(err or "invalid expression")
    try:
        return compile(tree, "<f8.expr>", "eval"), None
    except Exception as exc:
        return None, str(exc)


def _safe_eval_compiled(code: CodeType, *, names: dict[str, Any]) -> Any:
    # No builtins; only allowlisted math/functions.
    safe_globals: dict[str, Any] = {"__builtins__": {}}
    safe_globals.update(_ALLOWED_GLOBAL_FNS)
    safe_globals["math"] = math
    return eval(code, safe_globals, names)  # noqa: S307 (controlled eval)


class ExprRuntimeNode(OperatorNode):
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
        exec_in_ports = list(node.execInPorts or []) or ["exec"]
        exec_out_ports = list(node.execOutPorts or []) or ["exec"]
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=data_in_ports,
            data_out_ports=data_out_ports,
            state_fields=state_fields,
            exec_in_ports=exec_in_ports,
            exec_out_ports=exec_out_ports,
        )
        self._initial_state = dict(initial_state or {})
        self._code = self._normalize_code(self._initial_state.get("code") or "input")
        self._compiled: CodeType | None = None
        self._compile_error: str | None = None
        self._recompile()

        self._last_ctx_id: str | int | None = None
        self._last_out: Any = None
        self._dirty: bool = True

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        _ = ts_ms
        if str(field or "") != "code":
            return
        self._code = self._normalize_code(value)
        self._recompile()
        self._dirty = True

    @staticmethod
    def _normalize_code(value: Any) -> str:
        """
        Expr is a single-line expression. Normalize multi-line edits (paste, UI)
        into a single line to avoid hidden '\n' semantics.
        """
        s = str("" if value is None else value)
        if "\n" not in s and "\r" not in s:
            return s.strip()
        # Join lines with spaces so "a\n+b" behaves like "a +b".
        parts = [p.strip() for p in s.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        return " ".join([p for p in parts if p]).strip()

    def _recompile(self) -> None:
        compiled, err = _compile_expr(self._code)
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

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if str(port or "") != "out":
            return None
        if not self._dirty and ctx_id is not None and ctx_id == self._last_ctx_id:
            return self._last_out

        pulled: dict[str, Any] = {}
        for p in list(self.data_in_ports or []):
            try:
                pulled[str(p)] = await self.pull(str(p), ctx_id=ctx_id)
            except Exception:
                pulled[str(p)] = None

        try:
            if self._compiled is None:
                raise ValueError(self._compile_error or "invalid expression")
            out = _safe_eval_compiled(self._compiled, names=self._build_eval_names(pulled))
            if isinstance(out, _JsonRef):
                out = out.unwrap()
        except Exception as exc:
            logger.warning("[%s:expr] eval failed: %s", self.node_id, exc)
            out = None

        self._last_out = out
        self._last_ctx_id = ctx_id
        self._dirty = False
        return out

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        _ = (exec_id, in_port)
        out = await self.compute_output("out", ctx_id=exec_id)
        try:
            await self.emit("out", out)
        except Exception:
            pass
        ports = list(self.exec_out_ports or [])
        return ports or ["exec"]


ExprRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Expr",
    description="Evaluate a small expression using input values (math/logic/extraction).",
    tags=["expr", "math", "logic", "transform", "lightweight"],
    execInPorts=["exec"],
    execOutPorts=["exec"],
    editableExecInPorts=True,
    editableExecOutPorts=True,
    dataInPorts=[
        F8DataPortSpec(name="input", description="Default input value.", valueSchema=any_schema(), required=False),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="out", description="Expression result.", valueSchema=any_schema(), required=False),
    ],
    editableDataInPorts=True,
    editableDataOutPorts=False,
    stateFields=[
        F8StateSpec(
            name="code",
            label="Expr",
            description="Single-line expression (no statements). Examples: input.center.x ; a+b-c**2",
            uiControl="wrapline",
            uiLanguage="python",
            valueSchema=string_schema(default="input"),
            access=F8StateAccess.rw,
            showOnNode=True,
            required=False,
        ),
    ],
    editableStateFields=False,
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return ExprRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(ExprRuntimeNode.SPEC, overwrite=True)
    return reg
