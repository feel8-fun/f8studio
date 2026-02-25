from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

from pydantic import ValidationError

from f8pysdk.generated import F8OperatorSchemaVersion, F8OperatorSpec, F8ServiceSchemaVersion, F8ServiceSpec


class ServiceSpecRegistry:
    """In-memory registry for validated F8ServiceSpec templates."""

    _instance: ClassVar["ServiceSpecRegistry | None"] = None

    @staticmethod
    def instance() -> "ServiceSpecRegistry":
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


class OperatorSpecRegistry:
    """
    Per-service operator spec registries.

    Keyed by `(serviceClass, operatorClass)`.
    """

    _instance: ClassVar["OperatorSpecRegistry | None"] = None

    @staticmethod
    def instance() -> "OperatorSpecRegistry":
        if OperatorSpecRegistry._instance is None:
            OperatorSpecRegistry._instance = OperatorSpecRegistry()
        return OperatorSpecRegistry._instance

    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], F8OperatorSpec] = {}

    def register(self, spec: F8OperatorSpec) -> F8OperatorSpec:
        try:
            validated = F8OperatorSpec.model_validate(spec)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

        if validated.schemaVersion != F8OperatorSchemaVersion.f8operator_1:
            raise ValueError(f"schemaVersion must be {F8OperatorSchemaVersion.f8operator_1}")

        self._specs[(validated.serviceClass, validated.operatorClass)] = validated
        return validated

    def register_many(self, specs: Iterable[F8OperatorSpec]) -> list[F8OperatorSpec]:
        return [self.register(spec) for spec in specs]

    def unregister(self, service_class: str, operator_class: str) -> None:
        self._specs.pop((service_class, operator_class), None)

    def clear(self) -> None:
        self._specs.clear()

    def has(self, service_class: str, operator_class: str) -> bool:
        return (service_class, operator_class) in self._specs

    def get(self, service_class: str, operator_class: str) -> F8OperatorSpec:
        key = (service_class, operator_class)
        if key not in self._specs:
            raise KeyError(f'Operator spec "{key}" not found')
        return self._specs[key].model_copy(deep=True)

    def query(self, service_class: str | None) -> list[F8OperatorSpec]:
        if service_class is None:
            return [spec.model_copy(deep=True) for spec in self._specs.values()]
        return [
            spec.model_copy(deep=True)
            for (svc, _), spec in self._specs.items()
            if str(svc) == str(service_class)
        ]

    def all(self) -> list[F8OperatorSpec]:
        return [spec.model_copy(deep=True) for spec in self._specs.values()]


class ServiceCatalog:
    """
    Unified registry facade for (service spec + operator specs + discovery entry paths).
    """

    _instance: ClassVar["ServiceCatalog | None"] = None

    @staticmethod
    def instance() -> "ServiceCatalog":
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

    def register_service(self, spec: F8ServiceSpec, *, service_entry_path: Path | None = None) -> F8ServiceSpec:
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

    def clear(self) -> None:
        self.services.clear()
        self.operators.clear()
        self._service_entry_paths.clear()
