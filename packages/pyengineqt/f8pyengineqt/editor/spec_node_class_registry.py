from __future__ import annotations


from NodeGraphQt import BaseNode, NodeGraph
from ..renderers.generic import GenericNode
from ..renderers.renderer_registry import OperatorRendererRegistry
from ..renderers.service_node import ServiceNode
from ..operators.operator_registry import OperatorSpecRegistry
from ..services.service_registry import ServiceSpecRegistry
from ..services.builtin import ENGINE_SERVICE_CLASS
from ..renderers.service_engine import EngineServiceNode

import logging
import re

logger = logging.getLogger(__name__)


def _spec_key_for(spec: object) -> str:
    from f8pysdk import operator_key

    service_class = str(getattr(spec, "serviceClass", "") or "").strip()
    operator_class = str(getattr(spec, "operatorClass", "") or "").strip()
    return operator_key(service_class, operator_class)


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
        self._service_node_cls_map: dict[str, type[BaseNode]] = {}
        self.update()

    def __item__(self, key) -> type[GenericNode]:
        return self.get(key)

    def update(self):
        spec_registry = OperatorSpecRegistry.instance()
        renderer_registry = OperatorRendererRegistry.instance()
        service_registry = ServiceSpecRegistry.instance()

        n = 0
        for spec in spec_registry.all():

            renderer_key = spec.rendererClass or "default"
            service_class = str(getattr(spec, "serviceClass", "") or "")
            operator_class = str(getattr(spec, "operatorClass", "") or "")
            spec_key = _spec_key_for(spec)

            base_cls = renderer_registry.get(renderer_key)

            # Palette grouping: by serviceClass (operatorClass is local and not unique).
            namespace = service_class or "svc.unknown"
            class_name = "Op_" + re.sub(r"[^0-9a-zA-Z_]", "_", spec_key)
            if class_name[0].isdigit():
                class_name = f"Op_{class_name}"

            node_cls = type(
                class_name,
                (base_cls,),
                {
                    "__identifier__": namespace,
                    "NODE_NAME": spec.label,
                    "SPEC_KEY": spec_key,
                },
            )

            self._spec_node_cls_map[spec_key] = node_cls
            n += 1

        s = 0
        self._service_node_cls_map = {}
        for svc in service_registry.all():
            service_class = svc.serviceClass

            # Engine is a special-case: it renders as a backdrop that groups operators.
            # All other services render as a normal node (GenericNode-like layout).
            if service_class == ENGINE_SERVICE_CLASS:
                base_cls: type[BaseNode] = EngineServiceNode
            else:
                base_cls = ServiceNode

            namespace = ".".join(service_class.split(".")[:-1]) or service_class
            class_name = "Svc_" + re.sub(r"[^0-9a-zA-Z_]", "_", service_class)
            if class_name[0].isdigit():
                class_name = f"Svc_{class_name}"

            node_cls = type(
                class_name,
                (base_cls,),
                {
                    "__identifier__": namespace,
                    "NODE_NAME": svc.label,
                    "SPEC_KEY": service_class,
                },
            )
            self._service_node_cls_map[service_class] = node_cls
            s += 1

        logger.debug(f"Registered {n} operator + {s} service node classes into SpecNodeClassRegistry")

    def apply(self, node_graph: NodeGraph) -> None:
        """Register all known spec node classes into the given NodeGraph."""
        # Registries can be updated at runtime (eg. when builtin specs are registered on app start).
        try:
            self.update()
        except Exception:
            pass
        registered_types = set(node_graph.registered_nodes())

        for spec_key, node_cls in self._spec_node_cls_map.items():
            if node_cls.type_ not in registered_types:
                node_graph.register_node(node_cls, alias=spec_key)

        for service_class, node_cls in self._service_node_cls_map.items():
            if node_cls.type_ not in registered_types:
                node_graph.register_node(node_cls, alias=service_class)

    def get(self, key: str) -> type[GenericNode]:
        if key not in self._spec_node_cls_map:
            raise KeyError(f'Node class for spec "{key}" not found')
        return self._spec_node_cls_map[key]

    def keys(self) -> list[str]:
        return [*self._spec_node_cls_map.keys(), *self._service_node_cls_map.keys()]
