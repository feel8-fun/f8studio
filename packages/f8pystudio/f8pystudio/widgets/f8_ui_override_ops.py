from __future__ import annotations

from typing import Any

from f8pysdk import F8Command, F8ServiceSpec, F8StateSpec

STATEFIELD_UI_KEYS: tuple[str, ...] = ("showOnNode", "uiControl", "uiLanguage", "label", "description")


def get_ui_overrides(node: Any) -> dict[str, Any]:
    try:
        return dict(node.ui_overrides() or {})
    except Exception:
        return {}


def set_ui_overrides(node: Any, ui: dict[str, Any], *, rebuild: bool) -> None:
    try:
        node.set_ui_overrides(ui, rebuild=bool(rebuild))
    except Exception:
        return


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


def set_state_field_ui_override(node: Any, *, field_name: str, base: F8StateSpec, edited: F8StateSpec) -> None:
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
    node: Any,
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


def base_command_show_on_node(spec: Any, *, name: str) -> bool:
    if not isinstance(spec, F8ServiceSpec):
        return False
    n = str(name or "").strip()
    for c in list(spec.commands or []):
        try:
            if str(c.name or "").strip() == n:
                return bool(c.showOnNode)
        except Exception:
            continue
    return False


def find_base_state_field(spec: Any, *, name: str) -> F8StateSpec | None:
    n = str(name or "").strip()
    try:
        fields = list(spec.stateFields or [])
    except Exception:
        fields = []
    for f in fields:
        try:
            if str(f.name or "").strip() == n and isinstance(f, F8StateSpec):
                return f
        except Exception:
            continue
    return None

