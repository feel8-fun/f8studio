"""Shared numeric expression policy; expressions cannot access NumPy I/O.

This is an AST language restriction, not process isolation for untrusted Python.
Python script nodes intentionally have a different execution contract.
"""
from __future__ import annotations

import ast

NUMPY_EXPRESSION_MEMBERS = frozenset({
    'abs', 'absolute', 'all', 'any', 'arange', 'arccos', 'arcsin', 'arctan', 'arctan2',
    'argmax', 'argmin', 'argsort', 'array', 'asarray', 'ceil', 'clip', 'concatenate',
    'cos', 'cosh', 'cross', 'cumsum', 'diff', 'dot', 'e', 'exp', 'expm1', 'eye',
    'float32', 'float64', 'floor', 'full', 'hypot', 'inf', 'int32', 'int64',
    'interp', 'isfinite', 'isinf', 'isnan', 'linspace', 'log', 'log10', 'log1p',
    'max', 'maximum', 'mean', 'median', 'min', 'minimum', 'nan', 'nan_to_num',
    'ones', 'ones_like', 'percentile', 'pi', 'power', 'prod', 'quantile', 'radians',
    'reshape', 'round', 'sign', 'sin', 'sinh', 'sort', 'sqrt', 'square', 'squeeze',
    'stack', 'std', 'sum', 'tan', 'tanh', 'transpose', 'unique', 'var', 'where',
    'zeros', 'zeros_like', 'linalg', 'linalg.norm', 'linalg.det', 'linalg.solve',
    'linalg.inv', 'linalg.svd', 'fft', 'fft.fft', 'fft.ifft', 'fft.rfft', 'fft.irfft',
})


def numpy_attribute_allowed(node: ast.Attribute) -> bool:
    parts: list[str] = []
    base: ast.expr = node
    while isinstance(base, ast.Attribute):
        parts.append(base.attr)
        base = base.value
    if not isinstance(base, ast.Name) or base.id not in {'np', 'numpy'}:
        return True
    return '.'.join(reversed(parts)) in NUMPY_EXPRESSION_MEMBERS


# Syntax is shared; function allowlists and value semantics belong to each caller.
EXPRESSION_AST_NODES: tuple[type[ast.AST], ...] = (
    ast.Add,
    ast.And,
    ast.Attribute,
    ast.BinOp,
    ast.BoolOp,
    ast.Call,
    ast.Compare,
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
    ast.USub,
    ast.UnaryOp,
    ast.comprehension,
    ast.keyword,
)
