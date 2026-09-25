from __future__ import annotations

from typing import Literal

import msgspec

from ..generated import (
    F8CollectionEditPolicy,
    F8OperatorSpec,
    F8ServiceSpec,
    F8SpecEditPolicy,
    F8StateFieldEditPolicy,
    F8StateSpec,
)


EditableCollectionName = Literal[
    "stateFields", "commands", "dataInPorts", "dataOutPorts", "execInPorts", "execOutPorts"
]
SpecLike = F8ServiceSpec | F8OperatorSpec


def default_collection_edit_policy() -> F8CollectionEditPolicy:
    return F8CollectionEditPolicy(canAdd=False, canDelete=False, canEditExisting=False)


def editable_collection_edit_policy() -> F8CollectionEditPolicy:
    return F8CollectionEditPolicy(canAdd=True, canDelete=True, canEditExisting=True)


def default_spec_edit_policy() -> F8SpecEditPolicy:
    return F8SpecEditPolicy(
        stateFields=default_collection_edit_policy(),
        commands=default_collection_edit_policy(),
        dataInPorts=default_collection_edit_policy(),
        dataOutPorts=default_collection_edit_policy(),
        execInPorts=default_collection_edit_policy(),
        execOutPorts=default_collection_edit_policy(),
    )


def _policy_or_default(value: F8SpecEditPolicy | msgspec.UnsetType | None) -> F8SpecEditPolicy:
    if value is None or isinstance(value, msgspec.UnsetType):
        return default_spec_edit_policy()
    return value


def _collection_or_default(value: F8CollectionEditPolicy | msgspec.UnsetType | None) -> F8CollectionEditPolicy:
    if value is None or isinstance(value, msgspec.UnsetType):
        return default_collection_edit_policy()
    return value


def spec_edit_policy(spec: SpecLike) -> F8SpecEditPolicy:
    return _policy_or_default(spec.editPolicy)


def collection_edit_policy(spec: SpecLike, collection: EditableCollectionName) -> F8CollectionEditPolicy:
    policy = spec_edit_policy(spec)
    if collection == "stateFields":
        return _collection_or_default(policy.stateFields)
    if collection == "commands":
        return _collection_or_default(policy.commands)
    if collection == "dataInPorts":
        return _collection_or_default(policy.dataInPorts)
    if collection == "dataOutPorts":
        return _collection_or_default(policy.dataOutPorts)
    if collection == "execInPorts":
        return _collection_or_default(policy.execInPorts)
    return _collection_or_default(policy.execOutPorts)


def can_add(spec: SpecLike, collection: EditableCollectionName) -> bool:
    return bool(collection_edit_policy(spec, collection).canAdd)


def can_delete(spec: SpecLike, collection: EditableCollectionName) -> bool:
    return bool(collection_edit_policy(spec, collection).canDelete)


def can_edit_existing(spec: SpecLike, collection: EditableCollectionName) -> bool:
    return bool(collection_edit_policy(spec, collection).canEditExisting)


def is_value_required_state_field(field: F8StateSpec) -> bool:
    return bool(field.valueRequired)


def _state_field_edit_policy_or_none(field: F8StateSpec) -> F8StateFieldEditPolicy | None:
    policy = field.editPolicy
    if policy is None or isinstance(policy, msgspec.UnsetType):
        return None
    return policy


def _policy_bool(value: bool | msgspec.UnsetType | None) -> bool | None:
    if value is None or isinstance(value, msgspec.UnsetType):
        return None
    return bool(value)


def can_rename_state_field(field: F8StateSpec) -> bool:
    policy = _state_field_edit_policy_or_none(field)
    if policy is None:
        return True
    override = _policy_bool(policy.canRename)
    if override is None:
        return True
    return override


def can_edit_state_field_access(field: F8StateSpec) -> bool:
    policy = _state_field_edit_policy_or_none(field)
    if policy is None:
        return True
    override = _policy_bool(policy.canEditAccess)
    if override is None:
        return True
    return override


def can_edit_state_field_value_required(field: F8StateSpec) -> bool:
    policy = _state_field_edit_policy_or_none(field)
    if policy is None:
        return True
    override = _policy_bool(policy.canEditValueRequired)
    if override is None:
        return True
    return override


def can_edit_state_field_value_schema(field: F8StateSpec) -> bool:
    policy = _state_field_edit_policy_or_none(field)
    if policy is None:
        return True
    override = _policy_bool(policy.canEditValueSchema)
    if override is None:
        return True
    return override


def can_delete_state_field(field: F8StateSpec) -> bool:
    return can_rename_state_field(field)


def can_edit_state_field_structure(field: F8StateSpec) -> bool:
    return bool(
        can_rename_state_field(field)
        and can_edit_state_field_access(field)
        and can_edit_state_field_value_required(field)
    )


__all__ = [
    "EditableCollectionName",
    "SpecLike",
    "can_add",
    "can_delete_state_field",
    "can_delete",
    "can_edit_existing",
    "can_edit_state_field_access",
    "can_edit_state_field_value_required",
    "can_edit_state_field_structure",
    "can_edit_state_field_value_schema",
    "can_rename_state_field",
    "collection_edit_policy",
    "default_collection_edit_policy",
    "default_spec_edit_policy",
    "editable_collection_edit_policy",
    "is_value_required_state_field",
    "spec_edit_policy",
]
