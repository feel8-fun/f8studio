import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

from f8pyscript.state_access import PyScriptStateAccess, collect_readable_state_names


class _Access(Enum):
    rw = "rw"
    ro = "ro"
    wo = "wo"
    hidden = "hidden"


@dataclass
class _StateSpec:
    name: object
    access: object


async def _set_state(field: str, value: Any) -> None:
    del field, value


def test_collect_readable_state_names_keeps_declared_state_access() -> None:
    names = collect_readable_state_names(
        [
            _StateSpec("rw_state", _Access.rw),
            _StateSpec("ro_state", _Access.ro),
            _StateSpec("wo_state", _Access.wo),
            _StateSpec("hidden_state", _Access.hidden),
            _StateSpec("rw_state", _Access.rw),
            _StateSpec("", _Access.rw),
        ]
    )

    assert names == ("rw_state", "ro_state", "wo_state")


def test_state_access_marks_self_writes_and_builds_view() -> None:
    values = {"rw_state": 7, "wo_state": None}
    access = PyScriptStateAccess(
        readable_state_names=("rw_state", "wo_state"),
        state_fields=lambda: [],
        get_cached=lambda field, default: values.get(field, default),
        set_state=_set_state,
    )

    view = access.build_states_view(())

    assert view.get("rw_state") == 7
    assert "wo_state" in view
    assert view.get("wo_state") is None
    assert not access.is_self_state_write("rw_state", 7)

    asyncio.run(access.set_runtime_state("rw_state", 7))

    assert access.is_self_state_write("rw_state", 7)
