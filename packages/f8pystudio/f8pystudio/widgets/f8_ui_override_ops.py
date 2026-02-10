from __future__ import annotations

from typing import Any, Protocol

from f8pysdk import F8OperatorSpec, F8ServiceSpec, F8StateSpec


class _UiOverrideNode(Protocol):
    def ui_overrides(self) -> dict[str, object]: ...

    def set_ui_overrides(self, value: dict[str, object] | None, *, rebuild: bool = True) -> None: ...


def get_ui_overrides(node: _UiOverrideNode) -> dict[str, Any]:
    ui = node.ui_overrides()
    return dict(ui) if isinstance(ui, dict) else {}


def set_ui_overrides(node: _UiOverrideNode, ui: dict[str, Any], *, rebuild: bool) -> None:
    node.set_ui_overrides(ui, rebuild=bool(rebuild))


def _diff_state_ui(base: F8StateSpec, edited: F8StateSpec) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    if edited.showOnNode != base.showOnNode:
        patch["showOnNode"] = edited.showOnNode
    if edited.uiControl != base.uiControl:
        patch["uiControl"] = edited.uiControl
    if edited.uiLanguage != base.uiLanguage:
        patch["uiLanguage"] = edited.uiLanguage
    if edited.label != base.label:
        patch["label"] = edited.label
    if edited.description != base.description:
        patch["description"] = edited.description
    return patch


def set_state_field_ui_override(node: _UiOverrideNode, *, field_name: str, base: F8StateSpec, edited: F8StateSpec) -> None:
    """
    Persist UI-only overrides for a state field.

    Stores only diffs; if there are no diffs, removes the override entry.
    """
    name = str(field_name or "").strip()
    if not name:
        return
    patch = _diff_state_ui(base, edited)

    ui = get_ui_overrides(node)
    state_over = ui.get("stateFields")
    if not isinstance(state_over, dict):
        state_over = {}
    if patch:
        state_over[name] = patch
    else:
        state_over.pop(name, None)
    if state_over:
        ui["stateFields"] = state_over
    else:
        ui.pop("stateFields", None)
    set_ui_overrides(node, ui, rebuild=True)


def set_command_show_on_node_override(
    node: _UiOverrideNode,
    *,
    name: str,
    show_on_node: bool,
    base_show_on_node: bool,
) -> None:
    """
    Persist UI-only overrides for a command (currently only showOnNode).

    Stores only diffs; if value matches base spec, removes the override entry.
    """
    n = str(name or "").strip()
    if not n:
        return
    ui = get_ui_overrides(node)
    cmd_over = ui.get("commands")
    if not isinstance(cmd_over, dict):
        cmd_over = {}
    if bool(show_on_node) == bool(base_show_on_node):
        cmd_over.pop(n, None)
    else:
        cmd_over[n] = {"showOnNode": bool(show_on_node)}
    if cmd_over:
        ui["commands"] = cmd_over
    else:
        ui.pop("commands", None)
    set_ui_overrides(node, ui, rebuild=True)


def base_command_show_on_node(spec: F8ServiceSpec | None, *, name: str) -> bool:
    if spec is None:
        return False
    n = str(name or "").strip()
    if not n:
        return False
    for c in list(spec.commands or []):
        if str(c.name or "").strip() == n:
            return bool(c.showOnNode)
    return False


def base_data_port_show_on_node(spec: F8ServiceSpec | F8OperatorSpec | None, *, name: str, is_in: bool) -> bool:
    n = str(name or "").strip()
    if not n:
        return True
    if spec is None:
        return True
    ports = list(spec.dataInPorts or []) if bool(is_in) else list(spec.dataOutPorts or [])
    for p in ports:
        if str(p.name or "").strip() == n:
            return bool(p.showOnNode)
    return True


def set_data_port_show_on_node_override(
    node: _UiOverrideNode,
    *,
    name: str,
    is_in: bool,
    show_on_node: bool,
    base_show_on_node: bool,
) -> None:
    """
    Persist UI-only overrides for a data port (currently only showOnNode).

    Stores only diffs; if value matches base spec, removes the override entry.
    """
    n = str(name or "").strip()
    if not n:
        return
    ui = get_ui_overrides(node)
    ports_over = ui.get("dataPorts")
    if not isinstance(ports_over, dict):
        ports_over = {}
    key = "in" if bool(is_in) else "out"
    dir_over = ports_over.get(key)
    if not isinstance(dir_over, dict):
        dir_over = {}

    if bool(show_on_node) == bool(base_show_on_node):
        dir_over.pop(n, None)
    else:
        dir_over[n] = {"showOnNode": bool(show_on_node)}

    if dir_over:
        ports_over[key] = dir_over
    else:
        ports_over.pop(key, None)

    if ports_over:
        ui["dataPorts"] = ports_over
    else:
        ui.pop("dataPorts", None)

    set_ui_overrides(node, ui, rebuild=True)


def find_base_state_field(spec: F8ServiceSpec | F8OperatorSpec | None, *, name: str) -> F8StateSpec | None:
    n = str(name or "").strip()
    if not n or spec is None:
        return None
    fields = list(spec.stateFields or [])
    for f in fields:
        if str(f.name or "").strip() == n:
            return f
    return None
