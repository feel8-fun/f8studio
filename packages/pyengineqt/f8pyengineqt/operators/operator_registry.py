from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

from pydantic import ValidationError

from f8pysdk import F8OperatorSpec


class RegistryError(Exception):
    """Base class for registry failures."""


class OperatorAlreadyRegistered(RegistryError):
    """Raised when attempting to register a duplicate operatorClass without overwrite."""


class OperatorNotFound(RegistryError):
    """Raised when a requested operatorClass is missing."""


class InvalidOperatorSpec(RegistryError):
    """Raised when a spec payload cannot be validated."""


class OperatorSpecRegistry:
    """In-memory registry for validated OperatorSpec templates."""

    @staticmethod
    def instance() -> "OperatorSpecRegistry":
        """Get the global singleton instance of the registry."""
        global _GLOBAL_SPEC_REGISTRY
        try:
            return _GLOBAL_SPEC_REGISTRY
        except NameError:
            _GLOBAL_SPEC_REGISTRY = OperatorSpecRegistry()
            return _GLOBAL_SPEC_REGISTRY

    def __init__(self) -> None:
        self._specs: dict[str, F8OperatorSpec] = {}

    def register(self, spec: F8OperatorSpec, *, overwrite: bool = False) -> F8OperatorSpec:
        try:
            validated = F8OperatorSpec.model_validate(spec)
        except ValidationError as exc:
            raise InvalidOperatorSpec(str(exc)) from exc

        if validated.schemaVersion != "f8operator/1":
            raise InvalidOperatorSpec('schemaVersion must be "f8operator/1"')

        exists = validated.operatorClass in self._specs
        if exists and not overwrite:
            raise OperatorAlreadyRegistered(validated.operatorClass)

        self._specs[validated.operatorClass] = validated
        return validated

    def register_many(self, specs: Iterable[F8OperatorSpec], *, overwrite: bool = False) -> list[F8OperatorSpec]:
        return [self.register(spec, overwrite=overwrite) for spec in specs]

    def unregister(self, operator_class: str) -> None:
        self._specs.pop(operator_class, None)

    def has(self, operator_class: str) -> bool:
        return operator_class in self._specs

    def get(self, operator_class: str) -> F8OperatorSpec:
        if operator_class not in self._specs:
            raise OperatorNotFound(operator_class)
        return self._specs[operator_class].model_copy(deep=True)

    def query(
        self,
        *,
        tags: set[str] | None = None,
        text: str | None = None,
        predicate: Callable[[F8OperatorSpec], bool] | None = None,
    ) -> list[F8OperatorSpec]:
        tags = set(tags or [])
        text_lower = text.lower() if text else None

        def matches(spec: F8OperatorSpec) -> bool:
            if tags and not tags.issubset(set(spec.tags or [])):
                return False
            if text_lower:
                haystack = " ".join(
                    filter(
                        None,
                        [
                            spec.operatorClass,
                            spec.label,
                            spec.description,
                        ],
                    )
                ).lower()
                if text_lower not in haystack:
                    return False
            if predicate and not predicate(spec):
                return False
            return True

        return [spec.model_copy(deep=True) for spec in self._specs.values() if matches(spec)]

    def all(self) -> list[F8OperatorSpec]:
        return [spec.model_copy(deep=True) for spec in self._specs.values()]
