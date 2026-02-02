from __future__ import annotations

import json
import copy
from typing import Any

from NodeGraphQt import BaseNode

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

        template = getattr(self.__class__, "SPEC_TEMPLATE", None)
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
        """
        super().update_model()

        if not isinstance(getattr(self.model, "f8_sys", None), dict):
            self.model.f8_sys = {}
        if not isinstance(getattr(self.model, "f8_ui", None), dict):
            self.model.f8_ui = {}

    def ui_overrides(self) -> dict[str, object]:
        try:
            v = getattr(self.model, "f8_ui", None)
            if isinstance(v, dict):
                return v
        except Exception:
            pass
        return {}

    def set_ui_overrides(self, value: dict[str, object] | None, *, rebuild: bool = True) -> None:
        self.model.set_property("f8_ui", value or {})
        self._last_ui_serial = self._ui_serial()
        if rebuild:
            self.sync_from_spec()

    def effective_state_fields(self):
        """
        Return state fields with UI overrides applied (showOnNode/uiControl/etc).
        """
        spec = getattr(self, "spec", None)
        fields = list(getattr(spec, "stateFields", None) or []) if spec is not None else []
        ui = self.ui_overrides()
        state_over = ui.get("stateFields") if isinstance(ui, dict) else None
        if not isinstance(state_over, dict) or not state_over or not fields:
            return fields

        allowed_keys = {"showOnNode", "uiControl", "uiLanguage", "label", "description"}
        out = []
        for f in fields:
            name = str(getattr(f, "name", "") or "").strip()
            ov = state_over.get(name) if name else None
            if not isinstance(ov, dict) or not ov:
                out.append(f)
                continue
            patch = {k: ov.get(k) for k in allowed_keys if k in ov}
            try:
                out.append(f.model_copy(update=patch))
            except Exception:
                # Best-effort: fallback to original if copy fails.
                out.append(f)
        return out

    def effective_commands(self):
        """
        Return service commands with UI overrides applied.

        Currently only supports overriding `showOnNode` when `editableCommands`
        is false (UI-only customization).
        """
        spec = getattr(self, "spec", None)
        cmds = list(getattr(spec, "commands", None) or []) if spec is not None else []
        if not cmds:
            return cmds
        ui = self.ui_overrides()
        cmd_over = ui.get("commands") if isinstance(ui, dict) else None
        if not isinstance(cmd_over, dict) or not cmd_over:
            return cmds

        allowed_keys = {"showOnNode"}
        out = []
        for c in cmds:
            name = str(getattr(c, "name", "") or "").strip()
            ov = cmd_over.get(name) if name else None
            if not isinstance(ov, dict) or not ov:
                out.append(c)
                continue
            patch = {k: ov.get(k) for k in allowed_keys if k in ov}
            try:
                out.append(c.model_copy(update=patch))
            except Exception:
                out.append(c)
        return out

    def _ui_serial(self) -> str:
        try:
            ui = getattr(self.model, "f8_ui", None)
            if not isinstance(ui, dict):
                ui = {}
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
        spec = getattr(self.model, "f8_spec", None)
        if not isinstance(spec, (F8OperatorSpec, F8ServiceSpec)):
            raise RuntimeError(f"{self.__class__.__name__} model is missing `f8_spec`.")
        return spec

    @spec.setter
    def spec(self, value: F8OperatorSpec | F8ServiceSpec | dict) -> None:
        self.set_spec(value, rebuild=True)

    def update(self):
        current = getattr(self.model, "f8_spec", None)
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
            last_ui_serial = getattr(self, "_last_ui_serial", "")
            if ui_serial != last_ui_serial:
                self._last_ui_serial = ui_serial
                self.sync_from_spec()
        super().update()

    def sync_from_spec(self) -> None:
        """
        Hook for subclasses to rebuild ports/properties derived from `self.spec`.
        """
        return
