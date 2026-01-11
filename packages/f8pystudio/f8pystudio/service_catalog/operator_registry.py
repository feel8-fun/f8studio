from __future__ import annotations

from collections.abc import Iterable

from pydantic import ValidationError

from f8pysdk import F8OperatorSpec, F8OperatorSchemaVersion


class OperatorSpecRegistry:
    """
    Per-service operator spec registries.

    Goals:
    - each serviceClass owns its own operator palette
    - keep a global union registry for NodeGraphQt node construction
    """

    @staticmethod
    def instance() -> "OperatorSpecRegistry":
        # Singleton instance accessor.
        if not hasattr(OperatorSpecRegistry, "_instance"):
            OperatorSpecRegistry._instance = OperatorSpecRegistry()
        return OperatorSpecRegistry._instance

    def __init__(self) -> None:
        self._specs: dict[(str, str), F8OperatorSpec] = {}

    def register(self, spec: F8OperatorSpec):

        try:
            validated = F8OperatorSpec.model_validate(spec)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

        if validated.schemaVersion != F8OperatorSchemaVersion.f8operator_1:
            raise ValueError(f"schemaVersion must be {F8OperatorSchemaVersion.f8operator_1}")

        self._specs[(validated.serviceClass, validated.operatorClass)] = validated

    def register_many(self, specs: Iterable[F8OperatorSpec]) -> list[F8OperatorSpec]:
        return [self.register(spec) for spec in specs]

    def unregister(self, serviceClass: str, operatorClass: str) -> None:
        key = (serviceClass, operatorClass)
        self._specs.pop(key, None)

    def unregister_by_service(self, serviceClass: str) -> None:
        keys_to_remove = [key for key in self._specs if key[0] == serviceClass]
        for key in keys_to_remove:
            self._specs.pop(key)

    def clear(self) -> None:
        self._specs.clear()

    def query(self, serviceClass: str | None) -> list[F8OperatorSpec]:

        filtered = list(self._specs.items())

        if serviceClass is not None:
            filtered = filter(lambda x: x[0] == serviceClass, filtered)

        return [spec.model_copy(deep=True) for _, spec in filtered]

    def has(self, serviceClass, operatorClass) -> bool:
        return (serviceClass, operatorClass) in self._specs

    def get(self, serviceClass, operatorClass) -> F8OperatorSpec:
        key = (serviceClass, operatorClass)
        if key not in self._specs:
            raise KeyError(f'Operator spec "{key}" not found')
        return self._specs[key].model_copy(deep=True)

    def all(self) -> list[F8OperatorSpec]:
        return [spec.model_copy(deep=True) for spec in self._specs.values()]
