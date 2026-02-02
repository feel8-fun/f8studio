from __future__ import annotations

from typing import Any, Iterable

from f8pysdk import F8Command, F8DataPortSpec, F8OperatorSpec, F8ServiceSpec, F8StateSpec


def _assign_or_copy(model: Any, *, apply: callable) -> Any:
    """
    Apply mutation to a pydantic model instance.

    Some specs in the project appear mutable, others effectively behave as immutable.
    This helper writes in-place when possible, otherwise deep-copies and returns the copy.
    """
    try:
        apply(model)
        return model
    except Exception:
        m2 = model.model_copy(deep=True)
        apply(m2)
        return m2


def replace_state_field(spec: Any, *, old_name: str, new_field: F8StateSpec) -> Any:
    old = str(old_name or "").strip()
    fields = list(getattr(spec, "stateFields", None) or [])
    out: list[F8StateSpec] = []
    replaced = False
    for f in fields:
        if str(getattr(f, "name", "") or "").strip() == old:
            out.append(new_field)
            replaced = True
        else:
            out.append(f)
    if not replaced:
        out.append(new_field)
    return _assign_or_copy(spec, apply=lambda s: setattr(s, "stateFields", out))


def add_state_field(spec: Any, *, field: F8StateSpec) -> Any:
    fields = list(getattr(spec, "stateFields", None) or [])
    fields.append(field)
    return _assign_or_copy(spec, apply=lambda s: setattr(s, "stateFields", fields))


def delete_state_field(spec: Any, *, name: str) -> Any:
    n = str(name or "").strip()
    fields = [f for f in list(getattr(spec, "stateFields", None) or []) if str(getattr(f, "name", "") or "").strip() != n]
    return _assign_or_copy(spec, apply=lambda s: setattr(s, "stateFields", fields))


def add_command(spec: F8ServiceSpec, *, cmd: F8Command) -> F8ServiceSpec:
    cmds = list(getattr(spec, "commands", None) or [])
    cmds.append(cmd)
    return _assign_or_copy(spec, apply=lambda s: setattr(s, "commands", cmds))


def replace_command(spec: F8ServiceSpec, *, name: str, cmd: F8Command) -> F8ServiceSpec:
    n = str(name or "").strip()
    cmds = list(getattr(spec, "commands", None) or [])
    out: list[F8Command] = []
    replaced = False
    for c in cmds:
        if str(getattr(c, "name", "") or "").strip() == n:
            out.append(cmd)
            replaced = True
        else:
            out.append(c)
    if not replaced:
        out.append(cmd)
    return _assign_or_copy(spec, apply=lambda s: setattr(s, "commands", out))


def delete_command(spec: F8ServiceSpec, *, name: str) -> F8ServiceSpec:
    n = str(name or "").strip()
    cmds = [c for c in list(getattr(spec, "commands", None) or []) if str(getattr(c, "name", "") or "").strip() != n]
    return _assign_or_copy(spec, apply=lambda s: setattr(s, "commands", cmds))


def set_ports(
    spec: Any,
    *,
    data_in: Iterable[F8DataPortSpec],
    data_out: Iterable[F8DataPortSpec],
    exec_in: Iterable[str] | None = None,
    exec_out: Iterable[str] | None = None,
) -> Any:
    data_in_l = list(data_in)
    data_out_l = list(data_out)
    exec_in_l = list(exec_in or [])
    exec_out_l = list(exec_out or [])

    def _apply(s: Any) -> None:
        setattr(s, "dataInPorts", data_in_l)
        setattr(s, "dataOutPorts", data_out_l)
        if isinstance(s, F8OperatorSpec):
            setattr(s, "execInPorts", exec_in_l)
            setattr(s, "execOutPorts", exec_out_l)

    return _assign_or_copy(spec, apply=_apply)


def is_service_spec(spec: Any) -> bool:
    return isinstance(spec, F8ServiceSpec)


def is_operator_spec(spec: Any) -> bool:
    return isinstance(spec, F8OperatorSpec)

