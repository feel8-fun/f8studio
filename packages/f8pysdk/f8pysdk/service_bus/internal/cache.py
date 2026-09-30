"""Internal cache primitives owned by `service_bus`."""
from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar, overload

_K = TypeVar("_K")
_V = TypeVar("_V")
_D = TypeVar("_D")


class CappedOrderedDict(OrderedDict[_K, _V], Generic[_K, _V]):
    """
    Ordered mapping with max-entry cap.

    - `get` and `__getitem__` refresh recency.
    - `__setitem__` enforces max size.
    """

    def __init__(self, *, max_entries: int) -> None:
        super().__init__()
        self._max_entries = max(0, int(max_entries))

    def __getitem__(self, key: _K) -> _V:
        value = super().__getitem__(key)
        super().move_to_end(key)
        return value

    @overload
    def get(self, key: _K, default: None = None) -> _V | None: ...

    @overload
    def get(self, key: _K, default: _V) -> _V: ...

    @overload
    def get(self, key: _K, default: _D) -> _V | _D: ...

    def get(self, key: _K, default: _D | None = None) -> _V | _D | None:
        if key in self:
            return self[key]
        return default

    def __setitem__(self, key: _K, value: _V) -> None:
        exists = key in self
        super().__setitem__(key, value)
        if exists:
            super().move_to_end(key)
        if self._max_entries > 0:
            while len(self) > self._max_entries:
                self.popitem(last=False)


__all__ = ["CappedOrderedDict"]
