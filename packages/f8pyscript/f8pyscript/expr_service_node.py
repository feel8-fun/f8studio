from __future__ import annotations

import ast
import keyword
import logging
import math
import time
from dataclasses import dataclass
from types import CodeType
from typing import Any

from f8pysdk.capabilities import ClosableNode
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import ServiceNode

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_CODE = "msg"

_ALLOWED_GLOBAL_FNS: dict[str, Any] = {
    "abs": abs,
    "float": float,
    "int": int,
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
    text = str(name or "")
    return bool(text) and text.isidentifier() and not keyword.iskeyword(text)


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"{field_name} expects a boolean")


def _normalize_code(value: Any) -> str:
    text = str("" if value is None else value)
    if "\n" not in text and "\r" not in text:
        return text.strip()
    parts = [part.strip() for part in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return " ".join([part for part in parts if part]).strip()


def _wrap_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return _JsonRef(value)
    return value


@dataclass(frozen=True)
class _JsonRef:
    value: Any

    def __getattr__(self, name: str) -> Any:
        attr = str(name or "")
        if not attr or attr.startswith("_"):
            raise AttributeError(name)
        if isinstance(self.value, dict) and attr in self.value:
            return _wrap_value(self.value[attr])
        raise AttributeError(name)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(self.value, dict):
            if isinstance(key, str) and key.startswith("_"):
                raise KeyError(key)
            return _wrap_value(self.value[key])
        if isinstance(self.value, (list, tuple)):
            return _wrap_value(self.value[int(key)])
        raise TypeError(f"not indexable: {type(self.value).__name__}")

    def __iter__(self):
        if isinstance(self.value, dict):
            for key in self.value:
                yield key
            return
        if isinstance(self.value, (list, tuple)):
            for item in self.value:
                yield _wrap_value(item)
            return
        raise TypeError(f"not iterable: {type(self.value).__name__}")

    def unwrap(self) -> Any:
        if isinstance(self.value, dict):
            return {str(key): _JsonRef(item).unwrap() for key, item in self.value.items()}
        if isinstance(self.value, list):
            return [_JsonRef(item).unwrap() for item in self.value]
        if isinstance(self.value, tuple):
            return tuple(_JsonRef(item).unwrap() for item in self.value)
        return self.value


class _ExprValidator(ast.NodeVisitor):
    def __init__(self, *, allow_numpy: bool) -> None:
        super().__init__()
        self._allow_numpy = bool(allow_numpy)
        self._errors: list[str] = []

    def error(self, message: str) -> None:
        self._errors.append(str(message))

    def validate(self, expr: str) -> tuple[ast.Expression | None, str | None]:
        try:
            tree = ast.parse(str(expr or ""), mode="eval")
        except SyntaxError as exc:
            return None, f"syntax error: {exc.msg}"
        self.visit(tree)
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
            ast.comprehension,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.Store,
            ast.Call,
            ast.keyword,
        )
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
        if isinstance(node.func, ast.Name):
            if str(node.func.id) not in _ALLOWED_GLOBAL_FNS:
                self.error(f"call not allowed: {node.func.id}")
                return None
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
            fn = str(node.func.attr or "")
            if fn not in _ALLOWED_MATH_FNS:
                self.error(f"math call not allowed: math.{fn}")
                return None
        elif isinstance(node.func, ast.Attribute):
            base: ast.AST = node.func.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if not self._allow_numpy:
                self.error("numpy calls are disabled")
                return None
            if not (isinstance(base, ast.Name) and base.id in ("np", "numpy")):
                self.error("call target not allowed")
                return None
        else:
            self.error("call target not allowed")
            return None
        return self.generic_visit(node)


def _compile_expr(expr: str, *, allow_numpy: bool) -> tuple[CodeType | None, str | None]:
    validator = _ExprValidator(allow_numpy=allow_numpy)
    tree, error = validator.validate(expr)
    if tree is None:
        return None, str(error or "invalid expression")
    try:
        return compile(tree, "<f8.pyexpr>", "eval"), None
    except (SyntaxError, TypeError, ValueError) as exc:
        return None, str(exc)


def _safe_eval_compiled(code: CodeType, *, names: dict[str, Any], allow_numpy: bool) -> Any:
    safe_globals: dict[str, Any] = {"__builtins__": {}}
    safe_globals.update(_ALLOWED_GLOBAL_FNS)
    safe_globals["math"] = math
    if allow_numpy:
        if np is None:
            raise RuntimeError("numpy is not available")
        safe_globals["np"] = np
        safe_globals["numpy"] = np
    return eval(code, safe_globals, names)  # noqa: S307


class PythonExprServiceNode(ServiceNode, ClosableNode):
    def __init__(self, *, node_id: str, node: Any, initial_state: dict[str, Any] | None = None) -> None:
        data_in_ports = [str(p.name) for p in list(node.dataInPorts or [])] or ["in"]
        data_out_ports = [str(p.name) for p in list(node.dataOutPorts or [])] or ["out"]
        state_fields = [str(s.name) for s in list(node.stateFields or [])] or [
            "code",
            "allowNumpy",
            "unpackDictOutputs",
            "lastError",
            "active",
        ]
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=data_in_ports,
            data_out_ports=data_out_ports,
            state_fields=state_fields,
        )
        self._initial_state = dict(initial_state or {})
        self._code = _normalize_code(self._initial_state.get("code") or DEFAULT_CODE)
        self._allow_numpy = _coerce_bool(self._initial_state.get("allowNumpy"), field_name="allowNumpy")
        self._unpack_dict_outputs = _coerce_bool(
            self._initial_state.get("unpackDictOutputs"), field_name="unpackDictOutputs"
        )
        self._compiled: CodeType | None = None
        self._compile_error: str | None = None
        self._last_error: str = ""
        self._latest_inputs: dict[str, Any] = {}
        self._active = True
        self._warn_unmatched_sig: str = ""
        self._warn_unmatched_ts_ms = 0
        self._eval_error_sig: str = ""
        self._eval_error_ts_ms = 0
        self._recompile()

    async def close(self) -> None:
        return

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "code":
            return _normalize_code(value)
        if name == "allowNumpy":
            return _coerce_bool(value, field_name="allowNumpy")
        if name == "unpackDictOutputs":
            return _coerce_bool(value, field_name="unpackDictOutputs")
        if name == "active":
            return _coerce_bool(value, field_name="active")
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name == "code":
            self._code = _normalize_code(value)
            self._recompile()
            return
        if name == "allowNumpy":
            self._allow_numpy = _coerce_bool(value, field_name="allowNumpy")
            self._recompile()
            return
        if name == "unpackDictOutputs":
            self._unpack_dict_outputs = _coerce_bool(value, field_name="unpackDictOutputs")
            return
        if name == "active":
            self._active = _coerce_bool(value, field_name="active")

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        if not self._active:
            return
        in_port = str(port or "").strip()
        if not in_port:
            return
        self._latest_inputs[in_port] = value
        result = await self._eval_latest()
        if result is None and self._compiled is None:
            return
        await self._emit_result(result)

    def _recompile(self) -> None:
        compiled, error = _compile_expr(self._code, allow_numpy=self._allow_numpy)
        self._compiled = compiled
        self._compile_error = error

    def _build_eval_names(self) -> dict[str, Any]:
        wrapped_inputs = {key: _wrap_value(item) for key, item in self._latest_inputs.items()}
        env: dict[str, Any] = {"inputs": wrapped_inputs}
        for key, item in wrapped_inputs.items():
            if _is_identifier(key):
                env[key] = item
        return env

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000.0)

    async def _set_last_error(self, message: str) -> None:
        self._last_error = str(message)
        try:
            await self.set_state("lastError", self._last_error)
        except Exception as exc:
            logger.debug("[%s:pyexpr] failed to set lastError", self.node_id, exc_info=exc)

    async def _clear_last_error(self) -> None:
        if not self._last_error:
            return
        self._last_error = ""
        try:
            await self.set_state("lastError", "")
        except Exception as exc:
            logger.debug("[%s:pyexpr] failed to clear lastError", self.node_id, exc_info=exc)

    def _should_log_error(self, sig: str, *, kind: str, now_ms: int) -> bool:
        if kind == "eval":
            if sig != self._eval_error_sig:
                self._eval_error_sig = sig
                self._eval_error_ts_ms = int(now_ms)
                return True
            if (int(now_ms) - int(self._eval_error_ts_ms)) >= 5000:
                self._eval_error_ts_ms = int(now_ms)
                return True
            return False
        if kind == "unmatched":
            if sig != self._warn_unmatched_sig:
                self._warn_unmatched_sig = sig
                self._warn_unmatched_ts_ms = int(now_ms)
                return True
            if (int(now_ms) - int(self._warn_unmatched_ts_ms)) >= 5000:
                self._warn_unmatched_ts_ms = int(now_ms)
                return True
            return False
        return True

    async def _eval_latest(self) -> Any:
        if self._compiled is None:
            await self._set_last_error(self._compile_error or "invalid expression")
            return None
        try:
            out = _safe_eval_compiled(
                self._compiled,
                names=self._build_eval_names(),
                allow_numpy=self._allow_numpy,
            )
            if isinstance(out, _JsonRef):
                out = out.unwrap()
        except Exception as exc:
            now_ms = self._now_ms()
            sig = f"{type(exc).__name__}:{exc}"
            if self._should_log_error(sig, kind="eval", now_ms=now_ms):
                logger.warning("[%s:pyexpr] eval failed: %s", self.node_id, exc)
            await self._set_last_error(f"eval: {exc}")
            return None
        await self._clear_last_error()
        return out

    def _default_output_port(self) -> str | None:
        if "out" in self.data_out_ports:
            return "out"
        if self.data_out_ports:
            return str(self.data_out_ports[0])
        return None

    async def _emit_result(self, result: Any) -> None:
        if isinstance(result, dict) and self._unpack_dict_outputs:
            matched = False
            for raw_key, raw_value in result.items():
                out_port = str(raw_key)
                if out_port not in self.data_out_ports:
                    now_ms = self._now_ms()
                    sig = f"unmatched:{out_port}"
                    if self._should_log_error(sig, kind="unmatched", now_ms=now_ms):
                        logger.warning("[%s:pyexpr] unpack output key has no port: %s", self.node_id, out_port)
                    continue
                matched = True
                await self.emit(out_port, raw_value)
            if matched:
                return
        default_out = self._default_output_port()
        if default_out is None:
            return
        await self.emit(default_out, result)
