from __future__ import annotations

import uuid
from collections.abc import Iterable

from NodeGraphQt.nodes.backdrop_node import BackdropNode

from f8pysdk import F8ServiceSpec

from ..services.service_registry import ServiceSpecRegistry
from ..services.builtin import ENGINE_SERVICE_CLASS
from .generic import GenericNode


class EngineServiceNode(BackdropNode):  # type: ignore[misc]
    """
    Engine service node (Backdrop-style grouping).

    This node is intentionally NOT a SubGraph/GroupNode: it stays in the same
    graph view and visually "boxes" a set of operator nodes to indicate they
    run under the same engine service instance (serviceId == this node id).
    """

    __identifier__ = "feel8.service"
    NODE_NAME = "Engine"

    SPEC_KEY: str = ENGINE_SERVICE_CLASS

    service_spec: F8ServiceSpec

    def __init__(self) -> None:
        super().__init__()
        stable_id = uuid.uuid4().hex
        try:
            self.model.id = stable_id  # type: ignore[attr-defined]
            self.view.id = stable_id  # type: ignore[attr-defined]
        except Exception:
            pass
        self.service_spec = ServiceSpecRegistry.instance().get(self.SPEC_KEY)
        try:
            self.set_name(self.service_spec.label)  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            self.model.color = (70, 90, 130, 255)  # type: ignore[attr-defined]
        except Exception:
            pass

    def operator_nodes(self) -> list[GenericNode]:
        try:
            nodes = self.nodes()  # BackdropNode.nodes()
        except Exception:
            nodes = []
        return [n for n in nodes if isinstance(n, GenericNode)]

    def wrap_operator_nodes(self, nodes: Iterable[GenericNode]) -> None:
        try:
            self.wrap_nodes(list(nodes))  # type: ignore[arg-type]
        except Exception:
            pass
