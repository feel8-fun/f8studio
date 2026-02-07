from __future__ import annotations

from typing import Any, Iterable

from f8pysdk import F8Command, F8DataPortSpec, F8OperatorSpec, F8ServiceSpec, F8StateSpec


def _mutate_or_copy(model: Any, *, mutate: callable, update: dict[str, Any] | None = None) -> Any:
    """
    Apply mutation to a pydantic model instance.

    Some specs in the project appear mutable, others effectively behave as immutable.
    This helper writes in-place when possible, otherwise deep-copies and returns the copy.
    """
    try:
        mutate(model)
        return model
    except Exception:
        try:
            if update is not None:
                return model.model_copy(deep=True, update=update)
        except Exception:
            pass
        m2 = model.model_copy(deep=True)
        mutate(m2)
        return m2


def replace_state_field(spec: F8ServiceSpec | F8OperatorSpec, *, old_name: str, new_field: F8StateSpec) -> Any:
    old = str(old_name or "").strip()
    fields = list(spec.stateFields or [])
    out: list[F8StateSpec] = []
    replaced = False
    for f in fields:
        if str(f.name or "").strip() == old:
            out.append(new_field)
            replaced = True
        else:
            out.append(f)
    if not replaced:
        out.append(new_field)

    def _mutate(s: Any) -> None:
        s.stateFields = out

    return _mutate_or_copy(spec, mutate=_mutate, update={"stateFields": out})


def add_state_field(spec: F8ServiceSpec | F8OperatorSpec, *, field: F8StateSpec) -> Any:
    fields = list(spec.stateFields or [])
    fields.append(field)

    def _mutate(s: Any) -> None:
        s.stateFields = fields

    return _mutate_or_copy(spec, mutate=_mutate, update={"stateFields": fields})


def delete_state_field(spec: F8ServiceSpec | F8OperatorSpec, *, name: str) -> Any:
    n = str(name or "").strip()
    fields = [f for f in list(spec.stateFields or []) if str(f.name or "").strip() != n]

    def _mutate(s: Any) -> None:
        s.stateFields = fields

    return _mutate_or_copy(spec, mutate=_mutate, update={"stateFields": fields})


def add_command(spec: F8ServiceSpec, *, cmd: F8Command) -> F8ServiceSpec:
    cmds = list(spec.commands or [])
    cmds.append(cmd)

    def _mutate(s: Any) -> None:
        s.commands = cmds

    return _mutate_or_copy(spec, mutate=_mutate, update={"commands": cmds})


def replace_command(spec: F8ServiceSpec, *, name: str, cmd: F8Command) -> F8ServiceSpec:
    n = str(name or "").strip()
    cmds = list(spec.commands or [])
    out: list[F8Command] = []
    replaced = False
    for c in cmds:
        if str(c.name or "").strip() == n:
            out.append(cmd)
            replaced = True
        else:
            out.append(c)
    if not replaced:
        out.append(cmd)

    def _mutate(s: Any) -> None:
        s.commands = out

    return _mutate_or_copy(spec, mutate=_mutate, update={"commands": out})


def delete_command(spec: F8ServiceSpec, *, name: str) -> F8ServiceSpec:
    n = str(name or "").strip()
    cmds = [c for c in list(spec.commands or []) if str(c.name or "").strip() != n]

    def _mutate(s: Any) -> None:
        s.commands = cmds

    return _mutate_or_copy(spec, mutate=_mutate, update={"commands": cmds})


def set_ports(
    spec: F8ServiceSpec | F8OperatorSpec,
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

    def _mutate(s: Any) -> None:
        s.dataInPorts = data_in_l
        s.dataOutPorts = data_out_l
        if isinstance(s, F8OperatorSpec):
            s.execInPorts = exec_in_l
            s.execOutPorts = exec_out_l

    update: dict[str, Any] = {"dataInPorts": data_in_l, "dataOutPorts": data_out_l}
    if isinstance(spec, F8OperatorSpec):
        update["execInPorts"] = exec_in_l
        update["execOutPorts"] = exec_out_l

    return _mutate_or_copy(spec, mutate=_mutate, update=update)


def is_service_spec(spec: Any) -> bool:
    return isinstance(spec, F8ServiceSpec)


def is_operator_spec(spec: Any) -> bool:
    return isinstance(spec, F8OperatorSpec)

