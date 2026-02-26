from __future__ import annotations

import copy
import json
from typing import Any

from NodeGraphQt import BaseNode
from NodeGraphQt.nodes.base_node import NodeBaseWidget
from NodeGraphQt.errors import NodeWidgetError

from f8pysdk import F8OperatorSpec, F8ServiceSpec

from .node_model import F8StudioNodeModel


class F8StudioBaseNode(BaseNode):
    """
    Studio base node that persists framework system data in the NodeGraphQt
    session via default node model fields (not custom properties).
    """

    # Class-level spec template for building instance `spec`.
    SPEC_TEMPLATE: F8OperatorSpec | F8ServiceSpec | dict | None = None

    def __init__(self, qgraphics_item=None):
        super().__init__(qgraphics_item=qgraphics_item)
        # NodeGraphQt calls `node.update()` inside `set_model()`. Initialize
        # any fields used by `update()` before attaching the model.
        self._last_spec_obj: F8OperatorSpec | F8ServiceSpec | None = None
        self._last_ui_serial: str = ""
        self.set_model(F8StudioNodeModel())
        # Allow model-level property setters (used by session deserialization) to
        # trigger a spec/UI-driven rebuild before restoring custom properties.
        self.model._owner_node = self  # type: ignore[attr-defined]

        template = type(self).SPEC_TEMPLATE
        if template is None:
            raise RuntimeError(f"{self.__class__.__name__} must define `SPEC_TEMPLATE`.")

        if isinstance(template, F8OperatorSpec):
            spec = F8OperatorSpec.model_validate(template.model_dump(mode="json"))
        elif isinstance(template, F8ServiceSpec):
            spec = F8ServiceSpec.model_validate(template.model_dump(mode="json"))
        elif isinstance(template, dict):
            if "operatorClass" in template:
                spec = F8OperatorSpec.model_validate(template)
            else:
                spec = F8ServiceSpec.model_validate(template)
        else:
            spec = copy.deepcopy(template)

        self.set_spec(spec, rebuild=False)
        self._last_ui_serial = self._ui_serial()

    @property
    def svcId(self) -> Any:
        return self.model.svcId

    @svcId.setter
    def svcId(self, value: Any) -> None:
        self.model.svcId = value

    def update_model(self):
        """
        Extend NodeGraphQt model update so `spec` + system fields persist in
        session JSON.

        Also avoid writing values for ephemeral embedded widgets that are not
        backed by a registered node property.
        """
        for name, val in self.view.properties.items():
            if name in ["inputs", "outputs"]:
                continue
            if name not in self.model.properties and name not in self.model.custom_properties:
                continue
            self.model.set_property(name, val)

        for name, widget in self.view.widgets.items():
            if name not in self.model.properties and name not in self.model.custom_properties:
                continue
            self.model.set_property(name, widget.get_value())

        if not isinstance(self.model.f8_sys, dict):
            self.model.f8_sys = {}
        if not isinstance(self.model.f8_ui, dict):
            self.model.f8_ui = {}

    def add_ephemeral_widget(self, widget: NodeBaseWidget) -> None:
        """
        Add an embedded node widget without creating/persisting a custom node
        property for it.

        Use this for render-only UI panes whose state is persisted through
        explicit state fields instead of NodeGraphQt widget properties.
        """
        if not isinstance(widget, NodeBaseWidget):
            raise NodeWidgetError("'widget' must be an instance of a NodeBaseWidget")
        widget._node = self  # type: ignore[attr-defined]
        self.view.add_widget(widget)
        self.view.draw_node()
        widget.parent()

    def ui_overrides(self) -> dict[str, object]:
        return self.model.f8_ui if isinstance(self.model.f8_ui, dict) else {}

    def set_ui_overrides(self, value: dict[str, object] | None, *, rebuild: bool = True) -> None:
        self.model.set_property("f8_ui", value or {})
        self._last_ui_serial = self._ui_serial()
        if rebuild:
            self.sync_from_spec()

    def effective_state_fields(self):
        """
        Return state fields with UI overrides applied (showOnNode/uiControl/etc).
        """
        spec = self.spec
        fields = list(spec.stateFields or [])
        ui = self.ui_overrides()
        state_over = ui.get("stateFields") if isinstance(ui, dict) else None
        if not isinstance(state_over, dict) or not state_over or not fields:
            return fields

        allowed_keys = {"showOnNode", "uiControl", "uiLanguage", "label", "description"}
        out = []
        for f in fields:
            name = str(f.name or "").strip()
            ov = state_over.get(name) if name else None
            if not isinstance(ov, dict) or not ov:
                out.append(f)
                continue
            patch = {k: ov.get(k) for k in allowed_keys if k in ov}
            out.append(f.model_copy(update=patch))
        return out

    def effective_commands(self):
        """
        Return service commands with UI overrides applied.

        Currently only supports overriding `showOnNode` when `editableCommands`
        is false (UI-only customization).
        """
        spec = self.spec
        cmds = list(spec.commands or [])
        if not cmds:
            return cmds
        ui = self.ui_overrides()
        cmd_over = ui.get("commands") if isinstance(ui, dict) else None
        if not isinstance(cmd_over, dict) or not cmd_over:
            return cmds

        allowed_keys = {"showOnNode"}
        out = []
        for c in cmds:
            name = str(c.name or "").strip()
            ov = cmd_over.get(name) if name else None
            if not isinstance(ov, dict) or not ov:
                out.append(c)
                continue
            patch = {k: ov.get(k) for k in allowed_keys if k in ov}
            out.append(c.model_copy(update=patch))
        return out

    def data_port_show_on_node(self, name: str, *, is_in: bool) -> bool:
        """
        True if the data port should be rendered on the node body.

        Priority: UI override > spec field (if present) > default True.
        """
        n = str(name or "").strip()
        if not n:
            return True

        ui = self.ui_overrides()
        ports_over = ui.get("dataPorts") if isinstance(ui, dict) else None
        if isinstance(ports_over, dict):
            key = "in" if bool(is_in) else "out"
            dir_over = ports_over.get(key)
            if isinstance(dir_over, dict):
                ov = dir_over.get(n)
                if isinstance(ov, dict) and "showOnNode" in ov:
                    return bool(ov.get("showOnNode"))

        spec = self.spec
        ports = list(spec.dataInPorts or []) if bool(is_in) else list(spec.dataOutPorts or [])
        for p in ports:
            if str(p.name or "").strip() == n:
                return bool(p.showOnNode)

        return True

    def _ui_serial(self) -> str:
        try:
            ui = self.model.f8_ui if isinstance(self.model.f8_ui, dict) else {}
            return json.dumps(ui, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return ""

    def set_spec(self, value: F8OperatorSpec | F8ServiceSpec | dict, *, rebuild: bool = True) -> None:
        """
        Update the persistent spec stored on the model.

        `rebuild` controls whether `sync_from_spec()` is called after update.
        """
        self.model.set_property("f8_spec", value)
        self._last_spec_obj = self.model.f8_spec

        if rebuild:
            self.sync_from_spec()

    @property
    def spec(self) -> F8OperatorSpec | F8ServiceSpec:
        """
        Runtime view of the persisted model spec (pydantic object).
        """
        spec = self.model.f8_spec
        if not isinstance(spec, (F8OperatorSpec, F8ServiceSpec)):
            raise RuntimeError(f"{self.__class__.__name__} model is missing `f8_spec`.")
        return spec

    @spec.setter
    def spec(self, value: F8OperatorSpec | F8ServiceSpec | dict) -> None:
        self.set_spec(value, rebuild=True)

    def update(self):
        current = self.model.f8_spec
        has_spec = isinstance(current, (F8OperatorSpec, F8ServiceSpec))
        if has_spec and current is not self._last_spec_obj:
            self._last_spec_obj = current
            self.sync_from_spec()

        # During NodeGraphQt's `set_model()`, `update()` is called before we
        # have a spec. Avoid calling `sync_from_spec()` until `f8_spec` exists.
        ui_serial = self._ui_serial()
        if not has_spec:
            self._last_ui_serial = ui_serial
        else:
            last_ui_serial = self._last_ui_serial
            if ui_serial != last_ui_serial:
                self._last_ui_serial = ui_serial
                self.sync_from_spec()
        super().update()

    def sync_from_spec(self) -> None:
        """
        Hook for subclasses to rebuild ports/properties derived from `self.spec`.
        """
        return

    def is_missing_locked(self) -> bool:
        model = self.model
        if not isinstance(model.f8_sys, dict):
            model.f8_sys = {}
        return bool(model.f8_sys.get("missingLocked"))

    def missing_type(self) -> str:
        model = self.model
        if not isinstance(model.f8_sys, dict):
            model.f8_sys = {}
        return str(model.f8_sys.get("missingType") or "").strip()

    def missing_reason(self) -> str:
        model = self.model
        if not isinstance(model.f8_sys, dict):
            model.f8_sys = {}
        return str(model.f8_sys.get("missingReason") or "").strip()
