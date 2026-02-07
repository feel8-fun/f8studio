from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from pydantic import ValidationError

from f8pysdk import F8ServiceSpec, F8ServiceSchemaVersion


class ServiceSpecRegistry:
    """In-memory registry for validated F8ServiceSpec templates."""

    _instance: ClassVar["ServiceSpecRegistry | None"] = None

    @staticmethod
    def instance() -> "ServiceSpecRegistry":
        # Singleton instance accessor.
        if ServiceSpecRegistry._instance is None:
            ServiceSpecRegistry._instance = ServiceSpecRegistry()
        return ServiceSpecRegistry._instance

    def __init__(self) -> None:
        self._specs: dict[str, F8ServiceSpec] = {}

    def register(self, spec: F8ServiceSpec) -> F8ServiceSpec:
        try:
            validated = F8ServiceSpec.model_validate(spec)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
    
        if validated.schemaVersion != F8ServiceSchemaVersion.f8service_1:
            raise ValueError(f"schemaVersion must be {F8ServiceSchemaVersion.f8service_1}")

        self._specs[validated.serviceClass] = validated
        return validated

    def register_many(self, specs: Iterable[F8ServiceSpec]) -> list[F8ServiceSpec]:
        return [self.register(spec) for spec in specs]

    def unregister(self, service_class: str) -> None:
        self._specs.pop(service_class, None)

    def clear(self) -> None:
        self._specs.clear()

    def has(self, service_class: str) -> bool:
        return service_class in self._specs

    def get(self, service_class: str) -> F8ServiceSpec:
        if service_class not in self._specs:
            raise KeyError(f'Service "{service_class}" not found')
        return self._specs[service_class].model_copy(deep=True)

    def all(self) -> list[F8ServiceSpec]:
        return [spec.model_copy(deep=True) for spec in self._specs.values()]
