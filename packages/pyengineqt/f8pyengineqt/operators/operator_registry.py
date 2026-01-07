from __future__ import annotations

from collections.abc import Iterable
from typing import Callable

from pydantic import ValidationError

from f8pysdk import F8OperatorSpec, operator_key


class RegistryError(Exception):
    """Base class for registry failures."""


class OperatorAlreadyRegistered(RegistryError):
    """Raised when attempting to register a duplicate operatorClass without overwrite."""


class OperatorNotFound(RegistryError):
    """Raised when a requested operatorClass is missing."""


class InvalidOperatorSpec(RegistryError):
    """Raised when a spec payload cannot be validated."""


class OperatorSpecRegistry:
    """In-memory registry for validated F8OperatorSpec templates."""

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
        # Keyed by canonical operator key: "{serviceClass}:{operatorClass}"
        self._specs: dict[str, F8OperatorSpec] = {}

    def register(self, spec: F8OperatorSpec, *, overwrite: bool = False) -> F8OperatorSpec:
        try:
            validated = F8OperatorSpec.model_validate(spec)
        except ValidationError as exc:
            raise InvalidOperatorSpec(str(exc)) from exc

        if validated.schemaVersion != "f8operator/1":
            raise InvalidOperatorSpec('schemaVersion must be "f8operator/1"')

        try:
            key = operator_key(str(validated.serviceClass), str(validated.operatorClass))
        except Exception as exc:
            raise InvalidOperatorSpec(str(exc)) from exc

        exists = key in self._specs
        if exists and not overwrite:
            raise OperatorAlreadyRegistered(key)

        self._specs[key] = validated
        return validated

    def register_many(self, specs: Iterable[F8OperatorSpec], *, overwrite: bool = False) -> list[F8OperatorSpec]:
        return [self.register(spec, overwrite=overwrite) for spec in specs]

    def unregister(self, key: str) -> None:
        self._specs.pop(str(key), None)

    def has(self, key: str) -> bool:
        return str(key) in self._specs

    def get(self, key: str) -> F8OperatorSpec:
        k = str(key)
        if k not in self._specs:
            raise OperatorNotFound(k)
        return self._specs[k].model_copy(deep=True)

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
                            getattr(spec, "serviceClass", None),
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
