from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def wrap_pyexpr_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return PyExprJsonRef(value)
    return value


@dataclass(frozen=True)
class PyExprJsonRef:
    value: Any

    def __getattr__(self, name: str) -> Any:
        attr = str(name or "")
        if not attr or attr.startswith("_"):
            raise AttributeError(name)
        if isinstance(self.value, dict) and attr in self.value:
            return wrap_pyexpr_value(self.value[attr])
        raise AttributeError(name)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(self.value, dict):
            if isinstance(key, str) and key.startswith("_"):
                raise KeyError(key)
            return wrap_pyexpr_value(self.value[key])
        if isinstance(self.value, (list, tuple)):
            return wrap_pyexpr_value(self.value[int(key)])
        raise TypeError(f"not indexable: {type(self.value).__name__}")

    def __iter__(self):
        if isinstance(self.value, dict):
            for key in self.value:
                yield key
            return
        if isinstance(self.value, (list, tuple)):
            for item in self.value:
                yield wrap_pyexpr_value(item)
            return
        raise TypeError(f"not iterable: {type(self.value).__name__}")

    def unwrap(self) -> Any:
        if isinstance(self.value, dict):
            return {str(key): PyExprJsonRef(item).unwrap() for key, item in self.value.items()}
        if isinstance(self.value, list):
            return [PyExprJsonRef(item).unwrap() for item in self.value]
        if isinstance(self.value, tuple):
            return tuple(PyExprJsonRef(item).unwrap() for item in self.value)
        return self.value


__all__ = ["PyExprJsonRef", "wrap_pyexpr_value"]
