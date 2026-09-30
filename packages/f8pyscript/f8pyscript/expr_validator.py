from __future__ import annotations

import ast
from f8pysdk.expr_policy import EXPRESSION_AST_NODES, numpy_attribute_allowed
from types import CodeType
from typing import Any


PYEXPR_ALLOWED_GLOBAL_FNS: dict[str, Any] = {
    "abs": abs,
    "float": float,
    "int": int,
    "min": min,
    "max": max,
    "round": round,
}

PYEXPR_ALLOWED_MATH_FNS: frozenset[str] = frozenset(
    {
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
)


class PyExprValidator(ast.NodeVisitor):
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
        if not isinstance(node, EXPRESSION_AST_NODES):
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
            if str(node.func.id) not in PYEXPR_ALLOWED_GLOBAL_FNS:
                self.error(f"call not allowed: {node.func.id}")
                return None
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "math":
            fn = str(node.func.attr or "")
            if fn not in PYEXPR_ALLOWED_MATH_FNS:
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


def compile_pyexpr(expr: str, *, allow_numpy: bool) -> tuple[CodeType | None, str | None]:
    validator = PyExprValidator(allow_numpy=allow_numpy)
    tree, error = validator.validate(expr)
    if tree is None:
        return None, str(error or "invalid expression")
    try:
        return compile(tree, "<f8.pyexpr>", "eval"), None
    except (SyntaxError, TypeError, ValueError) as exc:
        return None, str(exc)


__all__ = [
    "PYEXPR_ALLOWED_GLOBAL_FNS",
    "PYEXPR_ALLOWED_MATH_FNS",
    "PyExprValidator",
    "compile_pyexpr",
]
