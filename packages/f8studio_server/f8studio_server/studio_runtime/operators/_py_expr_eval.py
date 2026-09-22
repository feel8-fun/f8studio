from __future__ import annotations

import ast
import math
from collections.abc import Iterator, Mapping, Sequence
from types import CodeType
from typing import cast


RuntimeValue = object


def sigmoid(value: object) -> float:
    """Return a scalar sigmoid without importing the optional NumPy stack."""
    try:
        numeric = float(cast(float | int | str, value))
    except (TypeError, ValueError):
        return 0.0
    try:
        return 1.0 / (1.0 + math.exp(-numeric))
    except OverflowError:
        return 0.0 if numeric < 0 else 1.0


ALLOWED_GLOBAL_FNS: dict[str, object] = {
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

ALLOWED_MATH_FNS = frozenset(
    {
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
)


def is_identifier(name: str) -> bool:
    return bool(name) and name.isidentifier()


def normalize_expr_code(value: object) -> str:
    text = "" if value is None else str(value)
    if "\n" not in text and "\r" not in text:
        return text.strip()
    parts = [part.strip() for part in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return " ".join(part for part in parts if part).strip()


class JsonRef:
    """Read-only view for expression values with a runtime-defined JSON schema."""

    __slots__ = ("_value",)

    def __init__(self, value: Mapping[object, object] | Sequence[object]) -> None:
        self._value = value

    def __getattr__(self, name: str) -> object:
        # Attribute names are user data and therefore cannot be declared statically.
        if not name or name.startswith("_"):
            raise AttributeError(name)
        if isinstance(self._value, Mapping) and name in self._value:
            return wrap_value(self._value[name])
        raise AttributeError(name)

    def __getitem__(self, key: object) -> object:
        if isinstance(self._value, Mapping):
            if isinstance(key, str) and key.startswith("_"):
                raise KeyError(key)
            return wrap_value(self._value[key])
        if isinstance(key, bool) or not isinstance(key, int):
            raise TypeError("sequence index must be an integer")
        return wrap_value(self._value[key])

    def __iter__(self) -> Iterator[object]:
        if isinstance(self._value, Mapping):
            yield from self._value
            return
        for item in self._value:
            yield wrap_value(item)

    def __len__(self) -> int:
        return len(self._value)

    def unwrap(self) -> object:
        return unwrap_value(self._value)


def wrap_value(value: object) -> object:
    if isinstance(value, Mapping):
        return JsonRef(cast(Mapping[object, object], value))
    if isinstance(value, (list, tuple)):
        return JsonRef(cast(Sequence[object], value))
    return value


def unwrap_value(value: object) -> object:
    if isinstance(value, JsonRef):
        return value.unwrap()
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(key): unwrap_value(item) for key, item in mapping.items()}
    if isinstance(value, list):
        return [unwrap_value(item) for item in cast(list[object], value)]
    if isinstance(value, tuple):
        return tuple(unwrap_value(item) for item in cast(tuple[object, ...], value))
    return value


class ExprValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self._errors: list[str] = []

    def validate(self, expression: str) -> tuple[ast.Expression | None, str | None]:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            return None, f"syntax error: {exc.msg}"
        self.visit(tree)
        if self._errors:
            return None, "; ".join(self._errors[:3])
        return tree, None

    def generic_visit(self, node: ast.AST) -> None:
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
            self._errors.append(f"disallowed syntax: {type(node).__name__}")
            return
        super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            self._errors.append("private/dunder attribute access is not allowed")
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id not in ALLOWED_GLOBAL_FNS:
                self._errors.append(f"call not allowed: {node.func.id}")
                return
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "math"
        ):
            if node.func.attr not in ALLOWED_MATH_FNS:
                self._errors.append(f"math call not allowed: math.{node.func.attr}")
                return
        else:
            self._errors.append("call target not allowed")
            return
        self.generic_visit(node)


def compile_expr(expression: str, *, allow_numpy: bool) -> tuple[CodeType | None, str | None]:
    if allow_numpy:
        return None, "NumPy expressions are unavailable in the Qt-free Web Studio runtime"
    tree, error = ExprValidator().validate(expression)
    if tree is None:
        return None, error or "invalid expression"
    try:
        return compile(tree, "<f8.web_studio_expr>", "eval"), None
    except (TypeError, ValueError) as exc:
        return None, f"compile error: {exc}"


def safe_eval_compiled(code: CodeType, *, names: Mapping[str, object]) -> object:
    safe_globals: dict[str, object] = {"__builtins__": {}}
    safe_globals.update(ALLOWED_GLOBAL_FNS)
    safe_globals["math"] = math
    return cast(object, eval(code, safe_globals, dict(names)))  # noqa: S307


__all__ = [
    "RuntimeValue",
    "compile_expr",
    "is_identifier",
    "normalize_expr_code",
    "safe_eval_compiled",
    "unwrap_value",
    "wrap_value",
]
