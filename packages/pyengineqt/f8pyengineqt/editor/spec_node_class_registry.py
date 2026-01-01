from __future__ import annotations


from NodeGraphQt import BaseNode, NodeGraph
from ..renderers.generic import GenericNode
from ..renderers.renderer_registry import OperatorRendererRegistry
from ..operators.operator_registry import OperatorSpecRegistry

import logging
import re

logger = logging.getLogger(__name__)


class SpecNodeClassRegistry:
    """Registry for renderer classes keyed by rendererClass."""

    @staticmethod
    def instance() -> "SpecNodeClassRegistry":
        """Get the global singleton instance of the registry."""
        global _GLOBAL_SPEC_NODE_CLASS_REGISTRY
        try:
            return _GLOBAL_SPEC_NODE_CLASS_REGISTRY
        except NameError:
            _GLOBAL_SPEC_NODE_CLASS_REGISTRY = SpecNodeClassRegistry()
            return _GLOBAL_SPEC_NODE_CLASS_REGISTRY

    def __init__(self) -> None:
        self._spec_node_cls_map: dict[str, type[GenericNode]] = {}
        self.update()

    def __item__(self, key) -> type[GenericNode]:
        return self.get(key)

    def update(self):
        spec_registry = OperatorSpecRegistry.instance()
        renderer_registry = OperatorRendererRegistry.instance()

        n = 0
        for spec in spec_registry.all():

            renderer_key = spec.rendererClass or "default"
            operator_class = spec.operatorClass

            base_cls = renderer_registry.get(renderer_key)

            # Keep the node type namespace consistent with the operator namespace to keep the palette tidy.
            namespace = ".".join(operator_class.split(".")[:-1]) or operator_class
            class_name = "Op_" + re.sub(r"[^0-9a-zA-Z_]", "_", operator_class)
            if class_name[0].isdigit():
                class_name = f"Op_{class_name}"

            node_cls = type(
                class_name,
                (base_cls,),
                {
                    "__identifier__": namespace,
                    "NODE_NAME": spec.label,
                    "SPEC_KEY": operator_class,
                },
            )

            self._spec_node_cls_map[operator_class] = node_cls
            n += 1

        logger.debug(f"Registered {n} spec node classes into SpecNodeClassRegistry")

    def apply(self, node_graph: NodeGraph) -> None:
        """Register all known spec node classes into the given NodeGraph."""
        registered_types = set(node_graph.registered_nodes())

        for operator_class, node_cls in self._spec_node_cls_map.items():
            if node_cls.type_ not in registered_types:
                node_graph.register_node(node_cls, alias=operator_class)

    def get(self, key: str) -> type[GenericNode]:
        if key not in self._spec_node_cls_map:
            raise KeyError(f'Node class for spec "{key}" not found')
        return self._spec_node_cls_map[key]

    def keys(self) -> list[str]:
        return list(self._spec_node_cls_map.keys())
