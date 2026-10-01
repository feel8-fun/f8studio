from __future__ import annotations

import math
from dataclasses import dataclass
from types import CodeType
from typing import Any

from .expr_json_ref import PyExprJsonRef
from .expr_validator import PYEXPR_ALLOWED_GLOBAL_FNS

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


_PYEXPR_EVAL_ERRORS = (Exception,)
def _safe_eval_compiled(code: CodeType, *, names: dict[str, Any], allow_numpy: bool) -> Any:
    safe_globals: dict[str, Any] = {"__builtins__": {}}
    safe_globals.update(PYEXPR_ALLOWED_GLOBAL_FNS)
    safe_globals["math"] = math
    if allow_numpy:
        if np is None:
            raise RuntimeError("numpy is not available")
        safe_globals["np"] = np
        safe_globals["numpy"] = np
    return eval(code, safe_globals, names)  # noqa: S307


@dataclass(frozen=True, slots=True)
class PyExprEvalResult:
    value: Any = None
    error: BaseException | None = None


class PyExprEvaluator:
    def evaluate(self, code: CodeType, *, names: dict[str, Any], allow_numpy: bool) -> PyExprEvalResult:
        try:
            value = _safe_eval_compiled(code, names=names, allow_numpy=allow_numpy)
            if isinstance(value, PyExprJsonRef):
                value = value.unwrap()
            return PyExprEvalResult(value=value)
        except _PYEXPR_EVAL_ERRORS as exc:
            return PyExprEvalResult(error=exc)


__all__ = ["PyExprEvalResult", "PyExprEvaluator", "np"]
