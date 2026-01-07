from __future__ import annotations

from collections.abc import Iterable

from ..operators.operator_registry import OperatorSpecRegistry
from f8pysdk import F8OperatorSpec


class RegistryError(Exception):
    """Base class for service/operator registry failures."""


class ServiceNotRegistered(RegistryError):
    """Raised when a serviceClass has no operator registry configured."""


class OperatorAlreadyRegistered(RegistryError):
    """Raised when an operatorClass is already registered for the same serviceClass."""


class ServiceOperatorSpecRegistry:
    """
    Per-service operator spec registries.

    Goals:
    - each serviceClass owns its own operator palette (OperatorSpecRegistry)
    - keep a global union registry for NodeGraphQt node construction
    """

    @staticmethod
    def instance() -> "ServiceOperatorSpecRegistry":
        global _GLOBAL_SERVICE_OPERATOR_SPEC_REGISTRY
        try:
            return _GLOBAL_SERVICE_OPERATOR_SPEC_REGISTRY
        except NameError:
            _GLOBAL_SERVICE_OPERATOR_SPEC_REGISTRY = ServiceOperatorSpecRegistry()
            return _GLOBAL_SERVICE_OPERATOR_SPEC_REGISTRY

    def __init__(self) -> None:
        self._by_service: dict[str, OperatorSpecRegistry] = {}

    def ensure_service(self, service_class: str) -> OperatorSpecRegistry:
        service_class = str(service_class or "").strip()
        if not service_class:
            raise ValueError("service_class must be non-empty")
        reg = self._by_service.get(service_class)
        if reg is None:
            reg = OperatorSpecRegistry()
            self._by_service[service_class] = reg
        return reg

    def services(self) -> list[str]:
        return sorted(self._by_service.keys())

    def operators_for_service(self, service_class: str) -> list[F8OperatorSpec]:
        service_class = str(service_class or "").strip()
        if service_class not in self._by_service:
            raise ServiceNotRegistered(service_class)
        return self._by_service[service_class].all()

    def register(self, service_class: str, spec: F8OperatorSpec, *, overwrite: bool = False) -> F8OperatorSpec:
        service_class = str(service_class or "").strip()
        if not service_class:
            raise ValueError("service_class must be non-empty")

        # Enforce that specs are owned by exactly one service (operator key is (serviceClass, operatorClass)).
        spec_service = str(getattr(spec, "serviceClass", "") or "").strip()
        if spec_service and spec_service != service_class:
            raise ValueError(f"spec.serviceClass ({spec_service}) must match service_class ({service_class})")
        if not spec_service:
            try:
                spec = F8OperatorSpec.model_validate({**spec.model_dump(mode="json"), "serviceClass": service_class})
            except Exception:
                spec = F8OperatorSpec.model_validate({"serviceClass": service_class, **spec.model_dump(mode="json")})

        reg = self.ensure_service(service_class)
        validated = reg.register(spec, overwrite=overwrite)

        # Keep the global registry in sync for NodeGraphQt node class generation.
        OperatorSpecRegistry.instance().register(validated, overwrite=True)
        return validated

    def register_many(
        self, service_class: str, specs: Iterable[F8OperatorSpec], *, overwrite: bool = False
    ) -> list[F8OperatorSpec]:
        return [self.register(service_class, spec, overwrite=overwrite) for spec in specs]
