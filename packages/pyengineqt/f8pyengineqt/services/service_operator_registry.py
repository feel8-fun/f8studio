from __future__ import annotations

from collections.abc import Iterable

from pydantic import ValidationError

from f8pysdk import F8OperatorSpec, operator_key


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
    - each serviceClass owns its own operator palette
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
        self._by_service: dict[str, dict[str, F8OperatorSpec]] = {}
        self._all: dict[str, F8OperatorSpec] = {}

    def services(self) -> list[str]:
        return sorted(self._by_service.keys())

    def has(self, spec_key: str) -> bool:
        return str(spec_key) in self._all

    def get(self, spec_key: str) -> F8OperatorSpec:
        key = str(spec_key)
        if key not in self._all:
            raise KeyError(f'Operator spec "{key}" not found')
        return self._all[key].model_copy(deep=True)

    def all(self) -> list[F8OperatorSpec]:
        return [spec.model_copy(deep=True) for spec in self._all.values()]

    def operators_for_service(self, service_class: str) -> list[F8OperatorSpec]:
        service_class = str(service_class or "").strip()
        if service_class not in self._by_service:
            raise ServiceNotRegistered(service_class)
        return [spec.model_copy(deep=True) for spec in self._by_service[service_class].values()]

    def _validate(self, service_class: str, spec: F8OperatorSpec) -> tuple[str, F8OperatorSpec]:
        try:
            validated = F8OperatorSpec.model_validate(spec)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

        if validated.schemaVersion != "f8operator/1":
            raise ValueError('schemaVersion must be "f8operator/1"')

        spec_service = str(getattr(validated, "serviceClass", "") or "").strip()
        if spec_service and spec_service != service_class:
            raise ValueError(f"spec.serviceClass ({spec_service}) must match service_class ({service_class})")

        if not spec_service:
            payload = validated.model_dump(mode="json")
            payload["serviceClass"] = service_class
            validated = F8OperatorSpec.model_validate(payload)

        key = operator_key(str(validated.serviceClass), str(validated.operatorClass))
        return key, validated

    def register(self, service_class: str, spec: F8OperatorSpec, *, overwrite: bool = False) -> F8OperatorSpec:
        service_class = str(service_class or "").strip()
        if not service_class:
            raise ValueError("service_class must be non-empty")

        key, validated = self._validate(service_class, spec)

        svc_bucket = self._by_service.setdefault(service_class, {})
        exists = key in svc_bucket
        if exists and not overwrite:
            raise OperatorAlreadyRegistered(key)

        svc_bucket[key] = validated
        self._all[key] = validated
        return validated.model_copy(deep=True)

    def register_many(
        self, service_class: str, specs: Iterable[F8OperatorSpec], *, overwrite: bool = False
    ) -> list[F8OperatorSpec]:
        return [self.register(service_class, spec, overwrite=overwrite) for spec in specs]
