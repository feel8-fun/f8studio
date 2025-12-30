from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..generated.operator_spec import Access, OperatorSpec, StateField


@dataclass
class OperatorInstance:
    """
    Runtime node instance with mutable spec and state/context buckets.

    - spec: a deep copy of the template that users can mutate (ports, commands, etc.).
    - state: persistable data keyed by StateField.name.
    - ctx: ephemeral, non-persisted runtime context (connections, handles, etc.).
    """

    id: str
    spec: OperatorSpec
    state: dict[str, Any] = field(default_factory=dict)
    ctx: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spec(
        cls,
        template: OperatorSpec,
        *,
        id: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> 'OperatorInstance':
        """Create an instance from a template spec with defaults applied to state."""
        instance_id = id or template.operatorClass
        mutable_spec = template.model_copy(deep=True)
        default_state = cls._build_default_state(mutable_spec)
        if state:
            default_state.update(state)
        return cls(id=instance_id, spec=mutable_spec, state=default_state)

    @staticmethod
    def _build_default_state(spec: OperatorSpec) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for field_def in spec.states or []:
            if field_def.default is not None:
                defaults[field_def.name] = field_def.default
            elif field_def.required:
                defaults[field_def.name] = None
        return defaults

    @property
    def operator_class(self) -> str:
        return self.spec.operatorClass

    def get_state_field(self, name: str) -> StateField | None:
        for field_def in self.spec.states or []:
            if field_def.name == name:
                return field_def
        return None

    def set_state(self, name: str, value: Any, *, allow_readonly: bool = False) -> None:
        field_def = self.get_state_field(name)
        if not field_def:
            raise KeyError(f'Unknown state field: {name}')

        if field_def.access == Access.ro and not allow_readonly:
            raise ValueError(f'State field {name} is read-only')
        self.state[name] = value

    def get_state(self, name: str) -> Any:
        return self.state.get(name)

    def reset_ctx(self) -> None:
        self.ctx.clear()

    def persistable(self) -> dict[str, Any]:
        """Serialize the instance for storage (ctx is intentionally omitted)."""
        return {
            'id': self.id,
            'operatorClass': self.operator_class,
            'spec': self.spec.model_dump(mode='json'),
            'state': self.state,
        }
