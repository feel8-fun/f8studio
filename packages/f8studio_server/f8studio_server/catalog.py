from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import msgspec

from f8pysdk.service_runtime_tools.inventory import ServiceCatalog, load_discovery_into_catalog
from f8pysdk.specs import F8OperatorSpec, F8ServiceDescribe, F8ServiceSpec


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


__all__ = ["CatalogService", "CatalogSnapshot"]
