from __future__ import annotations
import json
import logging

from NodeGraphQt.base.model import NodeModel
from NodeGraphQt.errors import NodePropertyError

from f8pysdk import F8OperatorSpec, F8ServiceSpec


logger = logging.getLogger(__name__)


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
        self._owner_node: object | None = None

    _unknown_prop_logged: set[str] = set()

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
            old = self.f8_spec
            self.f8_spec = self._coerce_spec(value)
            # Important: during node construction, the template spec is set once.
            # Subclasses build ports/properties in their __init__. We must NOT
            # trigger sync_from_spec() on this first assignment, otherwise ports
            # may be registered twice (causing PortRegistrationError).
            if old is not None and self.f8_spec is not None:
                owner = self._owner_node
                if owner is not None:
                    try:
                        owner.sync_from_spec()  # type: ignore[attr-defined]
                    except Exception:
                        logger.exception("Failed to sync node after f8_spec update.")
            return
        if name == "f8_ui":
            if isinstance(value, dict):
                self.f8_ui = value
            elif value is None:
                self.f8_ui = {}
            else:
                raise TypeError(f"Unsupported `f8_ui` type: {type(value)!r}")
            # UI changes affect effective state fields/ports, so resync.
            if self.f8_spec is not None:
                owner = self._owner_node
                if owner is not None:
                    try:
                        owner.sync_from_spec()  # type: ignore[attr-defined]
                    except Exception:
                        logger.exception("Failed to sync node after f8_ui update.")
            return
        try:
            return super().set_property(name, value)
        except NodePropertyError:
            # Session compatibility: older sessions may reference properties that no longer exist
            # after node spec/schema changes (eg. a state field label was used as a property name).
            # Ignore these so sessions still load, and log once per property name.
            prop = str(name)
            if prop not in self._unknown_prop_logged:
                self._unknown_prop_logged.add(prop)
                logger.warning("Ignoring unknown node property during session load: %r", name)
            return None

    @property
    def to_dict(self):
        """
        Override serialization to:
          1) omit port restore definitions (ports are derived from spec)
          2) serialize pydantic spec to plain JSON dict
        """
        data = super().to_dict
        ((node_id, node_dict),) = data.items()

        # Never persist runtime-only helpers into session JSON.
        node_dict.pop("_owner_node", None)
        node_dict.pop("_unknown_prop_logged", None)

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
