from __future__ import annotations

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
        self.set_model(F8StudioNodeModel())

        self._last_spec_obj: F8OperatorSpec | F8ServiceSpec | None = None

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
        if isinstance(current, (F8OperatorSpec, F8ServiceSpec)) and current is not self._last_spec_obj:
            self._last_spec_obj = current
            self.sync_from_spec()
        super().update()

    def sync_from_spec(self) -> None:
        """
        Hook for subclasses to rebuild ports/properties derived from `self.spec`.
        """
        return
