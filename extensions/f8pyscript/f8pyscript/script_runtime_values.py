from __future__ import annotations

import keyword
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


_MISSING = object()


class _PyScriptObjectView:
    __slots__ = ("_data", "_attr_to_key")

    def __init__(
        self,
        data: dict[str, Any],
        *,
        copy_data: bool = False,
        build_attr_index: bool = False,
    ) -> None:
        if copy_data:
            self._data = dict(data)
        else:
            self._data = data
        self._attr_to_key: dict[str, str] | None
        if build_attr_index:
            self._attr_to_key = self._build_attr_to_key(self._data)
        else:
            self._attr_to_key = None

    @staticmethod
    def _build_attr_to_key(data: dict[str, Any]) -> dict[str, str]:
        attr_to_key: dict[str, str] = {}
        for raw_key in data.keys():
            key = str(raw_key or "")
            if key.isidentifier() and not keyword.iskeyword(key):
                attr_to_key[key] = key
        for raw_key in data.keys():
            key = str(raw_key or "")
            if not key.isidentifier() or not keyword.iskeyword(key):
                continue
            alias = f"{key}_"
            if alias.isidentifier() and not keyword.iskeyword(alias) and alias not in attr_to_key:
                attr_to_key[alias] = key
        return attr_to_key

    def __getitem__(self, key: str) -> Any:
        return self._wrap_value(self._data[str(key)])

    def get(self, key: str, default: Any = None) -> Any:
        key_s = str(key)
        value = self._data.get(key_s, _MISSING)
        if value is _MISSING:
            return default
        return self._wrap_value(value)

    def keys(self):
        return self._data.keys()

    def items(self):
        return ((k, self._wrap_value(v)) for k, v in self._data.items())

    def values(self):
        return (self._wrap_value(v) for v in self._data.values())

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
        return self._wrap_value(self._data.get(key))

    def to_dict(self) -> dict[str, Any]:
        return self._unwrap_value(self._data)

    @classmethod
    def _wrap_value(cls, value: Any) -> Any:
        value_t = type(value)
        if value_t in (str, int, float, bool, type(None)):
            return value
        if isinstance(value, _PyScriptObjectView):
            return value
        if value_t is dict:
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._wrap_value(item) for item in value)
        return value

    @classmethod
    def _unwrap_value(cls, value: Any) -> Any:
        if isinstance(value, _PyScriptObjectView):
            return {k: cls._unwrap_value(v) for k, v in value._data.items()}
        if isinstance(value, dict):
            return {str(k): cls._unwrap_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._unwrap_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._unwrap_value(item) for item in value)
        return value


class PyScriptStatesView(_PyScriptObjectView):
    pass


def normalize_script_output_value(value: Any, *, _seen: set[int] | None = None) -> Any:
    if isinstance(value, _PyScriptObjectView):
        value = value.to_dict()
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, dict):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return None
        _seen.add(value_id)
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[str(key)] = normalize_script_output_value(item, _seen=_seen)
        _seen.discard(value_id)
        return out
    if isinstance(value, list):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return []
        _seen.add(value_id)
        out_list = [normalize_script_output_value(item, _seen=_seen) for item in value]
        _seen.discard(value_id)
        return out_list
    if isinstance(value, tuple):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return ()
        _seen.add(value_id)
        out_tuple = tuple(normalize_script_output_value(item, _seen=_seen) for item in value)
        _seen.discard(value_id)
        return out_tuple
    if isinstance(value, set):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return []
        _seen.add(value_id)
        out_set = [normalize_script_output_value(item, _seen=_seen) for item in value]
        _seen.discard(value_id)
        return out_set
    return value


def normalize_script_output_value_fast(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return normalize_script_output_value(value)


@dataclass(frozen=True)
class ScriptOutputPorts:
    data_out_ports: frozenset[str]
    single_data_out_port: str | None
    has_out_port: bool


def build_script_output_ports(data_out_ports: Iterable[str]) -> ScriptOutputPorts:
    port_names = frozenset(str(name) for name in data_out_ports)
    single_data_out_port: str | None = None
    if len(port_names) == 1:
        for port_name in port_names:
            single_data_out_port = port_name
    return ScriptOutputPorts(
        data_out_ports=port_names,
        single_data_out_port=single_data_out_port,
        has_out_port="out" in port_names,
    )


def extract_script_outputs(
    result: Any,
    *,
    ports: ScriptOutputPorts,
    normalize_one: Callable[[Any], Any] = normalize_script_output_value_fast,
) -> dict[str, Any]:
    if result is None:
        return {}

    if isinstance(result, dict):
        raw_outputs = result.get("outputs")
        if isinstance(raw_outputs, dict):
            single_out_port = ports.single_data_out_port
            if single_out_port is not None and len(raw_outputs) == 1 and single_out_port in raw_outputs:
                return {single_out_port: normalize_one(raw_outputs.get(single_out_port))}
            if len(raw_outputs) == 1:
                for key, value in raw_outputs.items():
                    return {str(key): normalize_one(value)}
            return {str(key): normalize_one(value) for key, value in raw_outputs.items()}
        if "outputs" in result:
            raise ValueError("script return field 'outputs' must be a dict")
        raise ValueError("script dict return must include an 'outputs' dict")

    if not ports.has_out_port:
        return {}
    return {"out": normalize_one(result)}
