from __future__ import annotations

from collections.abc import Iterable

from f8pysdk import F8OperatorSpec, F8ServiceSpec

from .service_operator_registry import ServiceOperatorSpecRegistry
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

    @staticmethod
    def instance() -> "ServiceCatalog":
        global _GLOBAL_SERVICE_CATALOG
        try:
            return _GLOBAL_SERVICE_CATALOG
        except NameError:
            _GLOBAL_SERVICE_CATALOG = ServiceCatalog()
            return _GLOBAL_SERVICE_CATALOG

    def __init__(self) -> None:
        self.services = ServiceSpecRegistry.instance()
        self.operators = ServiceOperatorSpecRegistry.instance()

    def register_service(self, spec: F8ServiceSpec, *, overwrite: bool = False) -> F8ServiceSpec:
        return self.services.register(spec, overwrite=overwrite)

    def register_services(self, specs: Iterable[F8ServiceSpec], *, overwrite: bool = False) -> list[F8ServiceSpec]:
        return self.services.register_many(specs, overwrite=overwrite)

    def register_operator(self, service_class: str, spec: F8OperatorSpec, *, overwrite: bool = False) -> F8OperatorSpec:
        return self.operators.register(service_class, spec, overwrite=overwrite)

    def register_operators(
        self, service_class: str, specs: Iterable[F8OperatorSpec], *, overwrite: bool = False
    ) -> list[F8OperatorSpec]:
        return self.operators.register_many(service_class, specs, overwrite=overwrite)

