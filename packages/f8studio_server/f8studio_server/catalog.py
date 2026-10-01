from __future__ import annotations

from f8studio_server.errors import InvalidRequestError

from collections.abc import Callable, Sequence
from pathlib import Path
from threading import RLock

import msgspec

from f8pysdk.service_runtime_tools.inventory import ServiceCatalog, load_discovery_into_catalog
from f8pysdk.service_runtime_tools.inventory.index import load_index_into_catalog
from f8pysdk.specs import F8OperatorSpec, F8ServiceDescribe, F8ServiceSpec
from f8studio_core.graph import NodeCatalog
from f8studio_core.graph.models import GraphNode, ServiceNode

from .models import CreateCatalogNodeRequest


class CatalogSnapshot(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    services: tuple[F8ServiceSpec, ...]
    operators: tuple[F8OperatorSpec, ...]


class CatalogService:
    def __init__(
        self,
        *,
        roots: Sequence[Path] | None = None,
        builtins: Sequence[F8ServiceDescribe] = (),
        extension_indexes: Callable[[], tuple[Path, ...]] | None = None,
    ) -> None:
        self._catalog = ServiceCatalog()
        self._roots = None if roots is None else tuple(roots)
        self._builtins = tuple(builtins)
        self._extension_indexes = extension_indexes
        self._lock = RLock()
        self._discovered_service_classes: tuple[str, ...] = ()
        self.refresh()

    def refresh(self, *, force_dynamic_service_classes: Sequence[str] = ()) -> CatalogSnapshot:
        updated = ServiceCatalog()
        discovered: list[str] = []
        if self._roots is None and self._extension_indexes is not None:
            for index in self._extension_indexes():
                discovered.extend(load_index_into_catalog(
                    path=index, catalog=updated, force_dynamic_service_classes=force_dynamic_service_classes,
                ))
        else:
            discovered = load_discovery_into_catalog(
                roots=None if self._roots is None else list(self._roots),
                catalog=updated,
                force_dynamic_service_classes=force_dynamic_service_classes,
            )
        for describe in self._builtins:
            updated.register_service(describe.service)
            operators = () if isinstance(describe.operators, msgspec.UnsetType) else describe.operators
            updated.register_operators(operators)
        with self._lock:
            missing = [service_class for service_class in force_dynamic_service_classes
                       if self._catalog.services.has(service_class) and not updated.services.has(service_class)]
            if missing:
                raise RuntimeError(f"Live service description failed for: {', '.join(missing)}")
            self._catalog = updated
            self._discovered_service_classes = tuple(sorted(discovered))
            return self.snapshot()

    @property
    def sdk_catalog(self) -> ServiceCatalog:
        return self._catalog

    @property
    def discovered_service_classes(self) -> tuple[str, ...]:
        return self._discovered_service_classes

    def snapshot(self) -> CatalogSnapshot:
        with self._lock:
            services = tuple(sorted(self._catalog.services.all(), key=lambda spec: str(spec.serviceClass)))
            operators = tuple(
                sorted(
                    self._catalog.operators.all(),
                    key=lambda spec: (str(spec.serviceClass), str(spec.operatorClass)),
                )
            )
        return CatalogSnapshot(services=services, operators=operators)

    def create_node(self, request: CreateCatalogNodeRequest) -> GraphNode:
        snapshot = self.snapshot()
        catalog = NodeCatalog(services=snapshot.services, operators=snapshot.operators)
        if request.kind == "service":
            try:
                return catalog.create_service_node(
                    node_id=request.node_id,
                    service_class=request.service_class,
                    name=request.name,
                )
            except KeyError as exc:
                raise InvalidRequestError(f"unknown serviceClass: {request.service_class}") from exc
        if request.operator_class is None or request.service_id is None:
            raise InvalidRequestError("operatorClass and serviceId are required for operator nodes")
        try:
            return catalog.create_operator_node(
                node_id=request.node_id,
                service_id=request.service_id,
                service_class=request.service_class,
                operator_class=request.operator_class,
                name=request.name,
            )
        except KeyError as exc:
            raise InvalidRequestError(
                f"unknown operator: {request.service_class}/{request.operator_class}"
            ) from exc

    def spec_for_node(self, node: GraphNode) -> F8ServiceSpec | F8OperatorSpec:
        with self._lock:
            if isinstance(node, ServiceNode):
                return self._catalog.services.get(node.service_class)
            return self._catalog.operators.get(node.service_class, node.operator_class)


__all__ = ["CatalogService", "CatalogSnapshot"]
