from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from types import CodeType
from typing import Any, Callable, Mapping

import numpy as np

_TWO_PI = 2.0 * math.pi


class _ExprValidator(ast.NodeVisitor):
    """Validate expression AST for a constrained waveform expression language."""

    _ALLOWED_NODE_TYPES: tuple[type[ast.AST], ...] = (
        ast.Expression,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Tuple,
        ast.List,
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
        ast.IfExp,
        ast.Call,
        ast.Attribute,
        ast.keyword,
    )

    def __init__(self, *, allowed_call_names: set[str], allowed_math_names: set[str]) -> None:
        super().__init__()
        self._allowed_call_names = allowed_call_names
        self._allowed_math_names = allowed_math_names
        self._errors: list[str] = []

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

    def _error(self, message: str) -> None:
        self._errors.append(str(message))

    def generic_visit(self, node: ast.AST) -> Any:
        if not isinstance(node, self._ALLOWED_NODE_TYPES):
            self._error(f"disallowed syntax: {type(node).__name__}")
            return None
        return super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        name = str(node.id or "")
        if name.startswith("_"):
            self._error("private names are not allowed")
            return None
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        attr = str(node.attr or "")
        if attr.startswith("_"):
            self._error("private attributes are not allowed")
            return None
        if not isinstance(node.value, ast.Name) or node.value.id != "math":
            self._error("only math.<fn> attribute access is allowed")
            return None
        if attr not in self._allowed_math_names:
            self._error(f"math attribute not allowed: math.{attr}")
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name):
            fn = str(node.func.id or "")
            if fn not in self._allowed_call_names:
                self._error(f"call not allowed: {fn}")
                return None
            return self.generic_visit(node)

        if isinstance(node.func, ast.Attribute):
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "math":
                self._error("call target not allowed")
                return None
            fn = str(node.func.attr or "")
            if fn not in self._allowed_math_names:
                self._error(f"math call not allowed: math.{fn}")
                return None
            return self.generic_visit(node)

        self._error("call target not allowed")
        return None


@dataclass(frozen=True)
class _MathProxy:
    """Math-like namespace that keeps trig functions phase-based."""

    e: float = math.e
    pi: float = math.pi
    tau: float = _TWO_PI

    def sin(self, x: Any) -> Any:
        return _phase_sin(x)

    def cos(self, x: Any) -> Any:
        return _phase_cos(x)

    def tan(self, x: Any) -> Any:
        return _phase_tan(x)

    def asin(self, x: Any) -> Any:
        return _phase_asin(x)

    def acos(self, x: Any) -> Any:
        return _phase_acos(x)

    def atan(self, x: Any) -> Any:
        return _phase_atan(x)

    def atan2(self, y: Any, x: Any) -> Any:
        return _phase_atan2(y, x)

    def exp(self, x: Any) -> Any:
        return np.exp(x)

    def log(self, x: Any) -> Any:
        return np.log(x)

    def log10(self, x: Any) -> Any:
        return np.log10(x)

    def sqrt(self, x: Any) -> Any:
        return np.sqrt(x)

    def floor(self, x: Any) -> Any:
        return np.floor(x)

    def ceil(self, x: Any) -> Any:
        return np.ceil(x)


def _phase_sin(x: Any) -> Any:
    return np.sin(_TWO_PI * np.asarray(x))


def _phase_cos(x: Any) -> Any:
    return np.cos(_TWO_PI * np.asarray(x))


def _phase_tan(x: Any) -> Any:
    return np.tan(_TWO_PI * np.asarray(x))


def _phase_asin(x: Any) -> Any:
    return np.arcsin(np.asarray(x)) / _TWO_PI


def _phase_acos(x: Any) -> Any:
    return np.arccos(np.asarray(x)) / _TWO_PI


def _phase_atan(x: Any) -> Any:
    return np.arctan(np.asarray(x)) / _TWO_PI


def _phase_atan2(y: Any, x: Any) -> Any:
    return np.arctan2(np.asarray(y), np.asarray(x)) / _TWO_PI


def _round_value(x: Any, ndigits: int = 0) -> Any:
    return np.round(np.asarray(x), int(ndigits))


def _min_value(a: Any, b: Any) -> Any:
    return np.minimum(np.asarray(a), np.asarray(b))


def _max_value(a: Any, b: Any) -> Any:
    return np.maximum(np.asarray(a), np.asarray(b))


def cond(condition: Any, when_true: Any, when_false: Any) -> Any:
    return np.where(np.asarray(condition, dtype=bool), when_true, when_false)


def clamp(value: Any, lo: Any, hi: Any) -> Any:
    return np.clip(np.asarray(value), lo, hi)


def lerp(a: Any, b: Any, t: Any) -> Any:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    t_arr = np.asarray(t)
    return a_arr + (b_arr - a_arr) * t_arr


def frac(value: Any) -> Any:
    value_arr = np.asarray(value)
    return value_arr - np.floor(value_arr)


def wrap(value: Any, lo: Any = 0.0, hi: Any = 1.0) -> Any:
    value_arr = np.asarray(value)
    lo_arr = np.asarray(lo)
    hi_arr = np.asarray(hi)
    span = hi_arr - lo_arr
    if np.any(span == 0):
        raise ValueError("wrap expects hi != lo")
    return lo_arr + np.mod(value_arr - lo_arr, span)


def saw(value: Any) -> Any:
    return 2.0 * frac(value) - 1.0


def tri(value: Any) -> Any:
    return 1.0 - 4.0 * np.abs(frac(value) - 0.5)


def pulse(value: Any, duty: Any = 0.5, low: Any = 0.0, high: Any = 1.0) -> Any:
    duty_arr = np.asarray(duty)
    if np.any((duty_arr < 0.0) | (duty_arr > 1.0)):
        raise ValueError("pulse duty must be in [0, 1]")
    return np.where(frac(value) < duty_arr, high, low)


def smoothstep(edge0: Any, edge1: Any, x: Any) -> Any:
    e0 = np.asarray(edge0)
    e1 = np.asarray(edge1)
    x_arr = np.asarray(x)
    denom = e1 - e0
    safe_denom = np.where(denom == 0, 1.0, denom)
    t = saturate((x_arr - e0) / safe_denom)
    out = t * t * (3.0 - 2.0 * t)
    if np.any(denom == 0):
        out = np.where(x_arr < e0, 0.0, 1.0)
    return out


def saturate(x: Any) -> Any:
    return clamp(x, 0.0, 1.0)


def tempest(t: Any, p: Any, c: Any) -> Any:
    t_arr = np.asarray(t, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)
    theta = _TWO_PI * t_arr
    phase = theta + ((math.pi * p_arr) / 2.0)
    return -np.cos(phase + (c_arr * np.sin(phase)))


def _fadein_value(t: Any, duration: Any) -> Any:
    duration_arr = np.asarray(duration, dtype=np.float64)
    if np.any(duration_arr <= 0.0):
        raise ValueError("fadein expects duration > 0")
    t_arr = np.asarray(t, dtype=np.float64)
    return saturate(t_arr / duration_arr)


def fadein(t: Any, duration: Any) -> Any:
    return _fadein_value(t, duration)


def _fadeout_value(t: Any, duration: Any, maxt: Any) -> Any:
    duration_arr = np.asarray(duration, dtype=np.float64)
    if np.any(duration_arr <= 0.0):
        raise ValueError("fadeout expects duration > 0")
    t_arr = np.asarray(t, dtype=np.float64)
    maxt_arr = np.asarray(maxt, dtype=np.float64)
    return saturate((maxt_arr - t_arr) / duration_arr)


def fadeout(t: Any, duration: Any, maxt: Any) -> Any:
    return _fadeout_value(t, duration, maxt)


def _sequence_value(t: Any, values: Any) -> Any:
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.ndim != 1:
        raise ValueError("sequence expects a 1D list or tuple")
    if values_arr.shape[0] == 0:
        raise ValueError("sequence expects at least one value")

    t_arr = np.asarray(t, dtype=np.float64)
    index_arr = np.trunc(t_arr).astype(np.int64, copy=False)
    wrapped_index_arr = np.mod(index_arr, values_arr.shape[0])
    selected = np.take(values_arr, wrapped_index_arr)

    if np.asarray(selected).ndim == 0:
        return float(selected)
    return selected


def sequence(t: Any, values: Any) -> Any:
    return _sequence_value(t, values)


_ALLOWED_DIRECT_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": np.abs,
    "float": float,
    "int": int,
    "min": _min_value,
    "max": _max_value,
    "round": _round_value,
    "sin": _phase_sin,
    "cos": _phase_cos,
    "tan": _phase_tan,
    "asin": _phase_asin,
    "acos": _phase_acos,
    "atan": _phase_atan,
    "atan2": _phase_atan2,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "floor": np.floor,
    "ceil": np.ceil,
    "cond": cond,
    "clamp": clamp,
    "lerp": lerp,
    "frac": frac,
    "wrap": wrap,
    "saw": saw,
    "tri": tri,
    "pulse": pulse,
    "smoothstep": smoothstep,
    "saturate": saturate,
    "tempest": tempest,
    "fadein": lambda duration: duration,
    "fadeout": lambda duration: duration,
    "sequence": lambda values: values,
}

_ALLOWED_MATH_NAMES: set[str] = {
    "e",
    "pi",
    "tau",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "exp",
    "log",
    "log10",
    "sqrt",
    "floor",
    "ceil",
}

_RESERVED_NAMES: set[str] = {
    "t",
    "maxt",
    "maxT",
    "p",
    "cycle",
    "math",
    "pi",
    "tau",
    "e",
    *set(_ALLOWED_DIRECT_FUNCTIONS.keys()),
}


class _NameSubstituter(ast.NodeTransformer):
    def __init__(self, *, replacements: Mapping[str, float]) -> None:
        super().__init__()
        self._replacements = replacements

    def visit_Name(self, node: ast.Name) -> ast.AST:
        key = str(node.id or "")
        if not isinstance(node.ctx, ast.Load):
            return node
        if key not in self._replacements:
            return node
        value = float(self._replacements[key])
        return ast.copy_location(ast.Constant(value=value), node)


def _parse_and_validate(expr: str) -> tuple[ast.Expression | None, str | None]:
    validator = _ExprValidator(
        allowed_call_names=set(_ALLOWED_DIRECT_FUNCTIONS.keys()),
        allowed_math_names=_ALLOWED_MATH_NAMES,
    )
    return validator.validate(expr)


def compile_expr(expr: str) -> tuple[CodeType | None, str | None]:
    """Compile a waveform expression after AST safety validation."""

    tree, error = _parse_and_validate(expr)
    if tree is None:
        return None, str(error or "invalid expression")

    try:
        return compile(tree, "<wave_expr>", "eval"), None
    except (SyntaxError, TypeError, ValueError) as exc:
        return None, str(exc)


def render_expression(expr: str, *, replacements: Mapping[str, float]) -> tuple[str | None, str | None]:
    """Render a normalized expression string with selected names replaced by constants."""

    tree, error = _parse_and_validate(expr)
    if tree is None:
        return None, str(error or "invalid expression")

    try:
        transformer = _NameSubstituter(replacements=replacements)
        replaced = transformer.visit(tree)
        ast.fix_missing_locations(replaced)
        return ast.unparse(replaced), None
    except (SyntaxError, TypeError, ValueError, AttributeError) as exc:
        return None, str(exc)


def _build_eval_locals(
    *,
    t: float | np.ndarray,
    maxt: float,
    variables: Mapping[str, float] | None,
) -> dict[str, Any]:
    names: dict[str, Any] = {}
    names.update(_ALLOWED_DIRECT_FUNCTIONS)
    names["math"] = _MathProxy()
    names["pi"] = math.pi
    names["tau"] = _TWO_PI
    names["e"] = math.e

    names["t"] = t
    names["maxt"] = float(maxt)
    names["maxT"] = float(maxt)
    names["p"] = frac(t)
    names["cycle"] = np.floor(t)
    names["fadein"] = lambda duration: _fadein_value(t, duration)
    names["fadeout"] = lambda duration: _fadeout_value(t, duration, maxt)
    names["sequence"] = lambda values: _sequence_value(t, values)

    if variables is not None:
        for key, value in variables.items():
            key_text = str(key)
            if not key_text.isidentifier():
                raise ValueError(f"invalid variable name: {key_text}")
            if key_text.startswith("_"):
                raise ValueError(f"private variable name is not allowed: {key_text}")
            if key_text in _RESERVED_NAMES:
                raise ValueError(f"variable name collides with reserved symbol: {key_text}")
            names[key_text] = float(value)

    return names


def eval_compiled(
    code: CodeType,
    *,
    t: float | np.ndarray,
    maxt: float,
    variables: Mapping[str, float] | None = None,
) -> Any:
    """Evaluate compiled expression with scalar or vector `t`."""

    if not isinstance(code, CodeType):
        raise TypeError("code must be compiled with compile_expr")

    maxt_value = float(maxt)
    if not np.isfinite(maxt_value) or maxt_value <= 0.0:
        raise ValueError("maxt must be a finite positive number")

    names = _build_eval_locals(t=t, maxt=maxt_value, variables=variables)
    safe_globals: dict[str, Any] = {"__builtins__": {}}
    return eval(code, safe_globals, names)  # noqa: S307 - constrained globals/locals after AST validation


def eval_scalar(
    code: CodeType,
    *,
    t: float,
    maxt: float,
    variables: Mapping[str, float] | None = None,
) -> float:
    """Evaluate to a scalar float value."""

    raw = eval_compiled(code, t=float(t), maxt=float(maxt), variables=variables)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1 and arr.shape[0] == 1:
        return float(arr[0])
    raise ValueError("scalar evaluation produced non-scalar result")


def _coerce_result_to_wave(result: Any, *, samples: int) -> np.ndarray:
    arr = np.asarray(result, dtype=np.float64)

    if arr.ndim == 0:
        return np.full((samples,), float(arr), dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError("expression result must be scalar or 1D")

    if arr.shape[0] != samples:
        if arr.shape[0] == 1:
            return np.full((samples,), float(arr[0]), dtype=np.float64)
        raise ValueError(f"expression result length mismatch: got {arr.shape[0]}, expected {samples}")

    return arr.astype(np.float64, copy=False)


def eval_wave(
    code: CodeType,
    *,
    maxt: float,
    samples: int,
    variables: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Evaluate a compiled waveform expression over `t in [0, maxt)`."""

    maxt_value = float(maxt)
    if not np.isfinite(maxt_value) or maxt_value <= 0.0:
        raise ValueError("maxt must be a finite positive number")

    samples_value = int(samples)
    if samples_value < 2:
        raise ValueError("samples must be >= 2")

    t = np.linspace(0.0, maxt_value, num=samples_value, endpoint=False, dtype=np.float64)
    result = eval_compiled(code, t=t, maxt=maxt_value, variables=variables)
    return _coerce_result_to_wave(result, samples=samples_value)


__all__ = [
    "RESERVED_NAMES",
    "compile_expr",
    "render_expression",
    "eval_compiled",
    "eval_scalar",
    "eval_wave",
    "cond",
    "clamp",
    "lerp",
    "frac",
    "wrap",
    "saw",
    "tri",
    "pulse",
    "smoothstep",
    "saturate",
    "tempest",
    "fadein",
    "fadeout",
    "sequence",
]

RESERVED_NAMES = _RESERVED_NAMES
