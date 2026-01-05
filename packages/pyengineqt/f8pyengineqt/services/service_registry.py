from __future__ import annotations

from collections.abc import Iterable

from pydantic import ValidationError

from f8pysdk import F8ServiceSpec


class RegistryError(Exception):
    """Base class for registry failures."""


class ServiceAlreadyRegistered(RegistryError):
    """Raised when attempting to register a duplicate serviceClass without overwrite."""


class ServiceNotFound(RegistryError):
    """Raised when a requested serviceClass is missing."""


class InvalidServiceSpec(RegistryError):
    """Raised when a spec payload cannot be validated."""


class ServiceSpecRegistry:
    """In-memory registry for validated F8ServiceSpec templates."""

    @staticmethod
    def instance() -> "ServiceSpecRegistry":
        global _GLOBAL_SERVICE_SPEC_REGISTRY
        try:
            return _GLOBAL_SERVICE_SPEC_REGISTRY
        except NameError:
            _GLOBAL_SERVICE_SPEC_REGISTRY = ServiceSpecRegistry()
            return _GLOBAL_SERVICE_SPEC_REGISTRY

    def __init__(self) -> None:
        self._specs: dict[str, F8ServiceSpec] = {}

    def register(self, spec: F8ServiceSpec, *, overwrite: bool = False) -> F8ServiceSpec:
        try:
            validated = F8ServiceSpec.model_validate(spec)
        except ValidationError as exc:
            raise InvalidServiceSpec(str(exc)) from exc

        if validated.schemaVersion != "f8service/1":
            raise InvalidServiceSpec('schemaVersion must be "f8service/1"')

        exists = validated.serviceClass in self._specs
        if exists and not overwrite:
            raise ServiceAlreadyRegistered(validated.serviceClass)

        self._specs[validated.serviceClass] = validated
        return validated

    def register_many(self, specs: Iterable[F8ServiceSpec], *, overwrite: bool = False) -> list[F8ServiceSpec]:
        return [self.register(spec, overwrite=overwrite) for spec in specs]

    def unregister(self, service_class: str) -> None:
        self._specs.pop(service_class, None)

    def has(self, service_class: str) -> bool:
        return service_class in self._specs

    def get(self, service_class: str) -> F8ServiceSpec:
        if service_class not in self._specs:
            raise ServiceNotFound(service_class)
        return self._specs[service_class].model_copy(deep=True)

    def all(self) -> list[F8ServiceSpec]:
        return [spec.model_copy(deep=True) for spec in self._specs.values()]

