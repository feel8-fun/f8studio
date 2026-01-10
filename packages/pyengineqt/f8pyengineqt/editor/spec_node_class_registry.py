from __future__ import annotations


from NodeGraphQt import BaseNode, NodeGraph
from ..renderers.generic import GenericNode
from ..renderers.renderer_registry import RendererRegistry
from ..renderers.service_node import ServiceNode
from ..services.service_operator_registry import ServiceOperatorSpecRegistry
from ..services.service_registry import ServiceSpecRegistry
from ..services.builtin import ENGINE_SERVICE_CLASS
from ..renderers.service_engine import EngineServiceNode

import logging
import re
import hashlib

logger = logging.getLogger(__name__)

from f8pysdk import operator_key, OPERATOR_KEY_SEP

# def _spec_key_for(spec: object) -> str:
#     from f8pysdk import operator_key

#     service_class = str(getattr(spec, "serviceClass", "") or "").strip()
#     operator_class = str(getattr(spec, "operatorClass", "") or "").strip()
#     return operator_key(service_class, operator_class)


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
        renderer_registry = RendererRegistry.instance()

        spec_registry = ServiceOperatorSpecRegistry.instance()
        service_registry = ServiceSpecRegistry.instance()

        def _safe_identifier(text: str) -> str:
            """
            NodeGraphQt drag/drop parses node ids with regex `node:([\\w\\.]+)`,
            so `node.type_` must only contain [A-Za-z0-9_\\.] characters.
            """
            s = str(text or "").strip()
            s = re.sub(r"[^0-9A-Za-z_\\.]+", "_", s)
            s = re.sub(r"\\.+", ".", s)
            s = s.strip("._")
            return s or "f8"

        def _safe_class_name(text: str, *, prefix: str) -> str:
            s = str(text or "").strip()
            base = re.sub(r"[^0-9A-Za-z_]+", "_", s).strip("_")
            if not base:
                base = prefix
            if base[0].isdigit():
                base = f"{prefix}_{base}"
            base = base[:48]
            h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]
            return f"{base}_{h}"

        num_ops = 0
        for opSpec in spec_registry.all():

            renderer_key = (opSpec.rendererClass or "default").strip()
            spec_key = operator_key(opSpec.serviceClass, opSpec.operatorClass)

            base_cls = renderer_registry.get(renderer_key)
            node_class_name = _safe_class_name(f"{opSpec.serviceClass}{OPERATOR_KEY_SEP}{opSpec.operatorClass}", prefix="Op")
            node_cls = type(
                node_class_name,
                (base_cls,),
                {
                    "__identifier__": _safe_identifier(opSpec.serviceClass),
                    "NODE_NAME": opSpec.label,
                    "SPEC_KEY": spec_key,
                },
            )

            self._spec_node_cls_map[spec_key] = node_cls
            num_ops += 1

        num_svcs = 0
        self._service_node_cls_map = {}
        for svcSpec in service_registry.all():

            spec_key = svcSpec.serviceClass

            renderer_key = (svcSpec.rendererClass or "default").strip()
            if renderer_key == "backdrop":
                base_cls = EngineServiceNode
            else:
                base_cls = ServiceNode

            node_class_name = _safe_class_name(svcSpec.serviceClass, prefix="Svc")
            node_cls = type(
                node_class_name,
                (base_cls,),
                {
                    "__identifier__": _safe_identifier(svcSpec.serviceClass),
                    "NODE_NAME": svcSpec.label,
                    "SPEC_KEY": svcSpec.serviceClass,
                },
            )
            self._service_node_cls_map[spec_key] = node_cls
            num_svcs += 1

        logger.debug(f"Registered {num_ops} operator + {num_svcs} service node classes into SpecNodeClassRegistry")

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
                # Allow editors to create operator nodes by canonical operator key.
                node_graph.register_node(node_cls, alias=str(spec_key))

        for service_class, node_cls in self._service_node_cls_map.items():
            if node_cls.type_ not in registered_types:
                # Allow editors to create service nodes by serviceClass.
                node_graph.register_node(node_cls, alias=str(service_class))

    def get(self, key: str) -> type[GenericNode]:
        if key not in self._spec_node_cls_map:
            raise KeyError(f'Node class for spec "{key}" not found')
        return self._spec_node_cls_map[key]

    def keys(self) -> list[str]:
        return [*self._spec_node_cls_map.keys(), *self._service_node_cls_map.keys()]
