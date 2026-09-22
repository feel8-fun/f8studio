from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import msgspec

from f8pysdk.service_runtime_tools.inventory import ServiceCatalog, load_discovery_into_catalog
from f8pysdk.specs import F8OperatorSpec, F8ServiceDescribe, F8ServiceSpec
from f8studio_core.graph import NodeCatalog
from f8studio_core.graph.models import GraphNode

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
    ) -> None:
        self._catalog = ServiceCatalog()
        discovered = load_discovery_into_catalog(
            roots=None if roots is None else list(roots),
            catalog=self._catalog,
        )
        for describe in builtins:
            self._catalog.register_service(describe.service)
            operators = () if isinstance(describe.operators, msgspec.UnsetType) else describe.operators
            self._catalog.register_operators(operators)
        self._discovered_service_classes = tuple(sorted(discovered))

    @property
    def sdk_catalog(self) -> ServiceCatalog:
        return self._catalog

    @property
    def discovered_service_classes(self) -> tuple[str, ...]:
        return self._discovered_service_classes

    def snapshot(self) -> CatalogSnapshot:
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
                raise ValueError(f"unknown serviceClass: {request.service_class}") from exc
        if request.operator_class is None or request.service_id is None:
            raise ValueError("operatorClass and serviceId are required for operator nodes")
        try:
            return catalog.create_operator_node(
                node_id=request.node_id,
                service_id=request.service_id,
                service_class=request.service_class,
                operator_class=request.operator_class,
                name=request.name,
            )
        except KeyError as exc:
            raise ValueError(
                f"unknown operator: {request.service_class}/{request.operator_class}"
            ) from exc


__all__ = ["CatalogService", "CatalogSnapshot"]
