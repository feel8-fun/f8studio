from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from enum import Enum
from typing import Any, Protocol

from .script_runtime_values import PyScriptStatesView


class RuntimeStateSpec(Protocol):
    name: object
    access: object


def collect_readable_state_names(states: Iterable[RuntimeStateSpec]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for state in list(states):
        name = str(state.name or "").strip()
        access_raw = state.access
        if not name or name in seen:
            continue
        access_value = access_raw.value if isinstance(access_raw, Enum) else access_raw
        access = str(access_value or "").strip().lower()
        if access not in ("rw", "ro", "wo"):
            continue
        seen.add(name)
        out.append(name)
    return tuple(out)


class PyScriptStateAccess:
    def __init__(
        self,
        *,
        readable_state_names: tuple[str, ...],
        state_fields: Callable[[], list[str]],
        get_cached: Callable[[str, Any], Any],
        set_state: Callable[[str, Any], Awaitable[None]],
    ) -> None:
        self._readable_state_names = tuple(str(name) for name in readable_state_names)
        self._state_fields = state_fields
        self._get_cached = get_cached
        self._set_state = set_state
        self._self_state_writes: dict[str, Any] = {}

    @property
    def readable_state_names(self) -> tuple[str, ...]:
        return self._readable_state_names

    async def set_runtime_state(self, field: str, value: Any) -> None:
        field_name = str(field)
        self._self_state_writes[field_name] = value
        await self._set_state(field_name, value)

    def is_self_state_write(self, field: str, value: Any) -> bool:
        field_name = str(field)
        return field_name in self._self_state_writes and self._self_state_writes.get(field_name) == value

    def build_states_view(self, state_keys: tuple[str, ...]) -> PyScriptStatesView:
        resolved_keys = [str(key) for key in state_keys if str(key)]
        if not resolved_keys:
            resolved_keys = [str(key) for key in self._state_fields() if str(key)]
        if not resolved_keys:
            resolved_keys = [str(key) for key in self._readable_state_names if str(key)]
        unique_keys = tuple(sorted({key for key in resolved_keys if key}))
        snapshot: dict[str, Any] = {}
        for key in unique_keys:
            snapshot[str(key)] = self._get_cached(str(key), None)
        return PyScriptStatesView(snapshot)


__all__ = ["PyScriptStateAccess", "RuntimeStateSpec", "collect_readable_state_names"]
