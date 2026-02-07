from __future__ import annotations
import json

from NodeGraphQt.base.model import NodeModel

from f8pysdk import F8OperatorSpec, F8ServiceSpec


class F8StudioNodeModel(NodeModel):
    """
    Studio node model that persists framework system data in the NodeGraphQt
    session via default node model fields (not custom properties).
    """

    f8_spec: F8OperatorSpec | F8ServiceSpec | None
    f8_sys: dict[str, object]
    f8_ui: dict[str, object]

    def __init__(self):
        super().__init__()
        self.f8_spec = None
        self.f8_sys = {}
        self.f8_ui = {}

    @staticmethod
    def _coerce_spec(value: object) -> F8OperatorSpec | F8ServiceSpec | None:
        if value is None:
            return None
        if isinstance(value, (F8OperatorSpec, F8ServiceSpec)):
            return value
        if isinstance(value, dict):
            if "operatorClass" in value:
                return F8OperatorSpec.model_validate(value)
            return F8ServiceSpec.model_validate(value)
        raise TypeError(f"Unsupported `f8_spec` type: {type(value)!r}")

    def set_property(self, name, value):
        if name == "f8_spec":
            self.f8_spec = self._coerce_spec(value)
            return
        if name == "f8_ui":
            if isinstance(value, dict):
                self.f8_ui = value
            elif value is None:
                self.f8_ui = {}
            else:
                raise TypeError(f"Unsupported `f8_ui` type: {type(value)!r}")
            return
        return super().set_property(name, value)

    @property
    def to_dict(self):
        """
        Override serialization to:
          1) omit port restore definitions (ports are derived from spec)
          2) serialize pydantic spec to plain JSON dict
        """
        data = super().to_dict
        ((node_id, node_dict),) = data.items()

        spec = node_dict.get("f8_spec")
        if isinstance(spec, (F8OperatorSpec, F8ServiceSpec)):
            node_dict["f8_spec"] = spec.model_dump(mode="json")

        if isinstance(self.f8_ui, dict) and self.f8_ui:
            node_dict["f8_ui"] = self.f8_ui

        return {node_id: node_dict}

    @property
    def serial(self):
        """
        Serialize model information to a string.

        Returns:
            str: serialized JSON string.
        """
        model_dict = self.to_dict

        # We never want NodeGraphQt to restore port definitions from session.
        model_dict[self.id].pop("port_deletion_allowed", None)
        model_dict[self.id].pop("input_ports", None)
        model_dict[self.id].pop("output_ports", None)
        return json.dumps(model_dict)

    @property
    def svcId(self) -> object | None:
        if not isinstance(self.f8_sys, dict):
            self.f8_sys = {}
        return self.f8_sys.get("svcId")

    @svcId.setter
    def svcId(self, value: object | None) -> None:
        if not isinstance(self.f8_sys, dict):
            self.f8_sys = {}
        if value is None:
            self.f8_sys.pop("svcId", None)
        else:
            self.f8_sys["svcId"] = value
