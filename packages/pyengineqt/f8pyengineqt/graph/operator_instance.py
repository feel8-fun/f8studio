from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from f8pysdk import F8OperatorSpec, F8StateAccess, F8StateSpec, F8DataTypeSchema, schema_default

@dataclass
class OperatorInstance:
    """
    Runtime node instance that owns a mutable spec copy plus state/context.

    - spec: deep copy of the template spec so per-node tweaks are isolated.
    - state: persistable values keyed by StateField.name.
    - ctx: ephemeral runtime bag, not persisted.
    """

    id: str
    spec: F8OperatorSpec
    state: dict[str, Any] = field(default_factory=dict)
    ctx: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spec(
        cls,
        template: F8OperatorSpec,
        *,
        id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> "OperatorInstance":
        instance_id = id or template.operatorClass
        mutable_spec = template.model_copy(deep=True)
        default_state = cls._build_default_state(mutable_spec)
        if state:
            default_state.update(state)
        return cls(id=instance_id, spec=mutable_spec, state=default_state)

    @staticmethod
    def _build_default_state(spec: F8OperatorSpec) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for field_def in spec.states or []:
            default_value = schema_default(field_def.valueSchema)
            if default_value is not None:
                defaults[field_def.name] = default_value
            elif field_def.required:
                defaults[field_def.name] = None
        return defaults
    
    @property
    def operator_class(self) -> str:
        return self.spec.operatorClass

    def get_state_field(self, name: str) -> F8StateSpec | None:
        for field_def in self.spec.states or []:
            if field_def.name == name:
                return field_def
        return None

    def set_state(self, name: str, value: Any, *, allow_readonly: bool = False) -> None:
        field_def = self.get_state_field(name)
        if not field_def:
            raise KeyError(f"Unknown state field: {name}")
        if field_def.access == F8StateAccess.ro and not allow_readonly:
            raise ValueError(f"State field {name} is read-only")
        self.state[name] = value

    def get_state(self, name: str) -> Any:
        return self.state.get(name)

    def reset_ctx(self) -> None:
        self.ctx.clear()

    def persistable(self) -> dict[str, Any]:
        """Serialize the instance for storage (ctx is intentionally omitted)."""
        return {
            "id": self.id,
            "operatorClass": self.operator_class,
            "spec": self.spec.model_dump(mode="json"),
            "state": self.state,
        }
