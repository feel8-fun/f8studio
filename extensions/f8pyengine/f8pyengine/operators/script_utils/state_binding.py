from __future__ import annotations

import keyword
from typing import Any

_MISSING = object()


class ValueAdapter:
    @classmethod
    def wrap(cls, value: Any) -> Any:
        value_t = type(value)
        if value_t in (str, int, float, bool, type(None)):
            return value
        if isinstance(value, _PyEngineObjectView):
            return value
        if value_t is dict:
            return _PyEngineObjectView(value)
        if isinstance(value, list):
            return [cls.wrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls.wrap(item) for item in value)
        return value

    @classmethod
    def unwrap(cls, value: Any) -> Any:
        if isinstance(value, _PyEngineObjectView):
            return {k: cls.unwrap(v) for k, v in value._data.items()}
        if isinstance(value, dict):
            return {str(k): cls.unwrap(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls.unwrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls.unwrap(item) for item in value)
        return value


class _PyEngineObjectView:
    __slots__ = ("_data", "_attr_to_key")

    def __init__(
        self,
        data: dict[str, Any],
        *,
        copy_data: bool = False,
        build_attr_index: bool = False,
    ) -> None:
        self._data = dict(data) if copy_data else data
        if build_attr_index:
            self._attr_to_key: dict[str, str] | None = self._build_attr_to_key(self._data)
        else:
            self._attr_to_key = None

    @staticmethod
    def _build_attr_to_key(data: dict[str, Any]) -> dict[str, str]:
        attr_to_key: dict[str, str] = {}
        for raw_key in data.keys():
            key = str(raw_key or "")
            if key.isidentifier() and not keyword.iskeyword(key):
                attr_to_key[key] = key
        return attr_to_key

    def __getitem__(self, key: str) -> Any:
        return ValueAdapter.wrap(self._data[str(key)])

    def get(self, key: str, default: Any = None) -> Any:
        key_s = str(key)
        value = self._data.get(key_s, _MISSING)
        if value is _MISSING:
            return default
        return ValueAdapter.wrap(value)

    def keys(self):
        return self._data.keys()

    def items(self):
        return ((k, ValueAdapter.wrap(v)) for k, v in self._data.items())

    def values(self):
        return (ValueAdapter.wrap(v) for v in self._data.values())

    def __contains__(self, key: object) -> bool:
        return str(key or "") in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self.to_dict())

    def __str__(self) -> str:
        return str(self.to_dict())

    def __getattr__(self, name: str) -> Any:
        attr_to_key = self._attr_to_key
        if attr_to_key is None:
            attr_to_key = self._build_attr_to_key(self._data)
            self._attr_to_key = attr_to_key
        key = attr_to_key.get(str(name or ""))
        if key is None:
            raise AttributeError(f"Unknown attribute: {name}")
        return ValueAdapter.wrap(self._data.get(key))

    def to_dict(self) -> dict[str, Any]:
        return ValueAdapter.unwrap(self._data)


class PyEngineInputsView(_PyEngineObjectView):
    pass


class PyEngineStatesView(_PyEngineObjectView):
    pass
