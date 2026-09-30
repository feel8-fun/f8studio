from __future__ import annotations

import ast
from f8pysdk.expr_policy import numpy_attribute_allowed
import math
from dataclasses import dataclass
from types import CodeType
from typing import Any, cast

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None


def sigmoid(x: Any) -> Any:
    """Standard sigmoid function: 1 / (1 + exp(-x))"""
    if np is not None and isinstance(x, (np.ndarray, np.generic)):
        x_np = cast(Any, x)
        return 1.0 / (1.0 + np.exp(-x_np))
    try:
        val = float(x)
        return 1.0 / (1.0 + math.exp(-val))
    except OverflowError:
        return 0.0 if float(x) < 0 else 1.0
    except (ValueError, TypeError):
        return 0.0


ALLOWED_GLOBAL_FNS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "sigmoid": sigmoid,
    "sorted": sorted,
    "sum": sum,
}

ALLOWED_MATH_FNS: set[str] = {
    "acos",
    "asin",
    "atan",
    "atan2",
    "ceil",
    "cos",
    "exp",
    "floor",
    "log",
    "log10",
    "sin",
    "sqrt",
    "tan",
}


def is_identifier(name: str) -> bool:
    return isinstance(name, str) and name.isidentifier()
def normalize_expr_code(value: Any) -> str:
    text = str("" if value is None else value)
    if "\n" not in text and "\r" not in text:
        return text.strip()
    parts = [part.strip() for part in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return " ".join([part for part in parts if part]).strip()


def wrap_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return JsonRef(value)
    return value


def unwrap_wrapped_value(value: Any) -> Any:
    if isinstance(value, JsonRef):
        return value.unwrap()
    return value


@dataclass(frozen=True)
class JsonRef:
    """
    Wrapper for JSON-like values that supports attribute access for dict keys.

    Example:
      input.center.x  <=>  input["center"]["x"]

    This intentionally blocks dunder/private access to avoid escaping into python internals.
    """

    value: Any

    def _deny_attr(self, name: str) -> bool:
        text = str(name or "")
        return not text or text.startswith("_")

    def __getattr__(self, name: str) -> Any:
        if self._deny_attr(name):
            raise AttributeError(name)
        raw_value = self.value
        if isinstance(raw_value, dict) and name in raw_value:
            return wrap_value(raw_value[name])
        raise AttributeError(name)

    def __getitem__(self, key: Any) -> Any:
        raw_value = self.value
        if isinstance(raw_value, dict):
            if isinstance(key, str) and key.startswith("_"):
                raise KeyError(key)
            return wrap_value(raw_value[key])
        if isinstance(raw_value, (list, tuple)):
            return wrap_value(raw_value[int(key)])
        raise TypeError(f"not indexable: {type(raw_value).__name__}")

    def __iter__(self):
        raw_value = self.value
        if isinstance(raw_value, dict):
            for key in raw_value:
                yield key
            return
        if isinstance(raw_value, (list, tuple)):
            for item in raw_value:
                yield wrap_value(item)
            return
        raise TypeError(f"not iterable: {type(raw_value).__name__}")

    def unwrap(self) -> Any:
        raw_value = self.value
        if isinstance(raw_value, dict):
            return {str(key): JsonRef(item).unwrap() for key, item in raw_value.items()}
        if isinstance(raw_value, list):
            return [JsonRef(item).unwrap() for item in raw_value]
        if isinstance(raw_value, tuple):
            return tuple(JsonRef(item).unwrap() for item in raw_value)
        return raw_value


class ExprValidator(ast.NodeVisitor):
    """
    Validate a Python expression AST for safe evaluation.

    Allows:
    - literals, names, indexing, attribute access (for JsonRef), arithmetic, boolean ops, comparisons
    - comprehensions (list/set/dict/generator)
    - calls to a small allowlist: abs/min/max/round, and math.<fn> (where fn is allowlisted)
    """

    def __init__(self, *, allow_numpy: bool) -> None:
        super().__init__()
        self._errors: list[str] = []
        self._allow_numpy = bool(allow_numpy)

    def error(self, msg: str) -> None:
        self._errors.append(str(msg))

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
            ast.Add,
            ast.And,
            ast.Attribute,
            ast.BinOp,
            ast.BoolOp,
            ast.Call,
            ast.Compare,
            ast.comprehension,
            ast.Constant,
            ast.Dict,
            ast.DictComp,
            ast.Div,
            ast.Eq,
            ast.Expression,
            ast.FloorDiv,
            ast.GeneratorExp,
            ast.Gt,
            ast.GtE,
            ast.IfExp,
            ast.In,
            ast.Is,
            ast.IsNot,
            ast.keyword,
            ast.List,
            ast.ListComp,
            ast.Load,
            ast.Lt,
            ast.LtE,
            ast.Mod,
            ast.Mult,
            ast.Name,
            ast.Not,
            ast.NotEq,
            ast.NotIn,
            ast.Or,
            ast.Pow,
            ast.SetComp,
            ast.Slice,
            ast.Store,
            ast.Sub,
            ast.Subscript,
            ast.Tuple,
            ast.UAdd,
            ast.UnaryOp,
            ast.USub,
        )
        if not isinstance(node, allowed):
            self.error(f"disallowed syntax: {type(node).__name__}")
            return None
        return super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if not numpy_attribute_allowed(node):
            self.error("numpy member is not allowed in numeric expressions")
            return None
        if str(node.attr or "").startswith("_"):
            self.error("private/dunder attribute access is not allowed")
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name):
            if str(node.func.id) not in ALLOWED_GLOBAL_FNS:
                self.error(f"call not allowed: {node.func.id}")
                return None
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
            fn_name = str(node.func.attr or "")
            if fn_name not in ALLOWED_MATH_FNS:
                self.error(f"math call not allowed: math.{fn_name}")
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


def compile_expr(expr: str, *, allow_numpy: bool) -> tuple[CodeType | None, str | None]:
    validator = ExprValidator(allow_numpy=allow_numpy)
    tree, err = validator.validate(expr)
    if tree is None:
        return None, str(err or "invalid expression")
    try:
        return compile(tree, "<f8.py_expr>", "eval"), None
    except (SyntaxError, TypeError, ValueError) as exc:
        return None, str(exc)


def safe_eval_compiled(code: CodeType, *, names: dict[str, Any], allow_numpy: bool) -> Any:
    safe_globals: dict[str, Any] = {"__builtins__": {}}
    safe_globals.update(ALLOWED_GLOBAL_FNS)
    safe_globals["math"] = math
    if allow_numpy:
        if np is None:
            raise RuntimeError("numpy is not available")
        safe_globals["np"] = np
        safe_globals["numpy"] = np
    return eval(code, safe_globals, names)  # noqa: S307 (controlled eval)
