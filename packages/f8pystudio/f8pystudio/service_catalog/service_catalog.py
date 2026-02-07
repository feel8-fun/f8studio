from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

from f8pysdk import F8OperatorSpec, F8ServiceSpec

logger = logging.getLogger(__name__)

from .operator_registry import OperatorSpecRegistry
from .service_registry import ServiceSpecRegistry


class ServiceCatalog:
    """
    Unified registry facade for (service spec + operator specs).

    Notes:
    - Services are identified by `serviceClass`.
    - Operators are uniquely identified by `(serviceClass, operatorClass)`.
    - Internally this delegates to two registries to keep their lifecycles independent,
      but provides a single entry-point for callers that treat them as a bundle.
    """

    _instance: ClassVar["ServiceCatalog | None"] = None

    @staticmethod
    def instance() -> "ServiceCatalog":
        # Singleton instance accessor.
        if ServiceCatalog._instance is None:
            ServiceCatalog._instance = ServiceCatalog()
        return ServiceCatalog._instance

    def __init__(self) -> None:
        self.services = ServiceSpecRegistry.instance()
        self.operators = OperatorSpecRegistry.instance()
        self._service_entry_paths: dict[str, Path] = {}

    def service_entry_path(self, service_class: str) -> Path | None:
        return self._service_entry_paths.get(str(service_class or "").strip())

    def service_entry_paths(self) -> dict[str, Path]:
        return dict(self._service_entry_paths)

    def register_service(
        self,
        spec: F8ServiceSpec,
        *,
        service_entry_path: Path | None = None,
    ) -> F8ServiceSpec:
        registered = self.services.register(spec)
        if service_entry_path is not None:
            self._service_entry_paths[str(registered.serviceClass)] = Path(service_entry_path).resolve()
        return registered

    def register_services(self, specs: Iterable[F8ServiceSpec]) -> list[F8ServiceSpec]:
        return self.services.register_many(specs)

    def register_operator(self, spec: F8OperatorSpec) -> F8OperatorSpec:
        return self.operators.register(spec)

    def register_operators(self, specs: Iterable[F8OperatorSpec]) -> list[F8OperatorSpec]:
        return self.operators.register_many(specs)
