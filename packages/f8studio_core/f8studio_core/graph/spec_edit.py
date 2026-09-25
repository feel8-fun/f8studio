from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import msgspec

from f8pysdk._specs.edit_policy import (
    EditableCollectionName,
    can_add,
    can_delete,
    can_delete_state_field,
    can_edit_existing,
)
from f8pysdk.specs import F8Command, F8DataPortSpec, F8ExecPortSpec, F8OperatorSpec, F8ServiceSpec, F8StateSpec

Spec = F8ServiceSpec | F8OperatorSpec
SpecItem = F8StateSpec | F8Command | F8DataPortSpec | F8ExecPortSpec


def _items(value: Sequence[SpecItem] | msgspec.UnsetType) -> Sequence[SpecItem]:
    return [] if isinstance(value, msgspec.UnsetType) else value


def _item_name(item: SpecItem) -> str:
    return item.name


def _check_collection(
    previous: Spec,
    collection: EditableCollectionName,
    before: Sequence[SpecItem] | msgspec.UnsetType,
    after: Sequence[SpecItem] | msgspec.UnsetType,
) -> None:
    old = _items(before)
    new = _items(after)
    old_names = [_item_name(item) for item in old]
    new_names = [_item_name(item) for item in new]
    if len(set(new_names)) != len(new_names):
        raise ValueError(f"duplicate {collection} name")
    old_by_name = dict(zip(old_names, old, strict=True))
    new_by_name = dict(zip(new_names, new, strict=True))
    added = set(new_by_name) - set(old_by_name)
    removed = set(old_by_name) - set(new_by_name)
    if added and not can_add(previous, collection):
        raise ValueError(f"{collection} does not allow adding: {', '.join(sorted(added))}")
    if removed and not can_delete(previous, collection):
        raise ValueError(f"{collection} does not allow deleting: {', '.join(sorted(removed))}")
    if collection == "stateFields":
        for name in removed:
            field = old_by_name[name]
            assert isinstance(field, F8StateSpec)
            if not can_delete_state_field(field):
                raise ValueError(f"protected state field cannot be deleted or renamed: {name}")
    if collection in ("dataInPorts", "dataOutPorts", "commands", "execInPorts", "execOutPorts"):
        for name in removed:
            item = old_by_name[name]
            if isinstance(item, (F8DataPortSpec, F8Command, F8ExecPortSpec)) and item.definitionProtected is True:
                raise ValueError(f"protected {collection} entry cannot be deleted: {name}")
        for name in set(old_by_name) & set(new_by_name):
            old_item = old_by_name[name]
            new_item = new_by_name[name]
            if isinstance(old_item, (F8DataPortSpec, F8Command, F8ExecPortSpec)) and isinstance(new_item, type(old_item)):
                if old_item.definitionProtected is True and new_item.definitionProtected is not True:
                    raise ValueError(f"protected {collection} entry cannot be unlocked: {name}")
    changed = [
        name for name in set(old_by_name) & set(new_by_name)
        if msgspec.to_builtins(old_by_name[name]) != msgspec.to_builtins(new_by_name[name])
    ]
    if changed and not can_edit_existing(previous, collection):
        raise ValueError(f"{collection} does not allow editing: {', '.join(sorted(changed))}")
    if collection == "stateFields":
        for name in changed:
            old_field = old_by_name[name]
            new_field = new_by_name[name]
            assert isinstance(old_field, F8StateSpec) and isinstance(new_field, F8StateSpec)
            if old_field.editPolicy != new_field.editPolicy:
                raise ValueError(f"state field edit policy is locked: {name}")
            policy = old_field.editPolicy
            if isinstance(policy, msgspec.UnsetType):
                continue
            if policy.canEditAccess is False and old_field.access != new_field.access:
                raise ValueError(f"state field access is locked: {name}")
            if policy.canEditValueRequired is False and old_field.valueRequired != new_field.valueRequired:
                raise ValueError(f"state field value-required flag is locked: {name}")
            if policy.canEditValueSchema is False and msgspec.to_builtins(old_field.valueSchema) != msgspec.to_builtins(new_field.valueSchema):
                raise ValueError(f"state field value schema is locked: {name}")


def validate_spec_edit(previous: Spec, proposed: Spec) -> None:
    if type(previous) is not type(proposed):
        raise ValueError("node spec kind cannot change")
    old = cast(dict[str, object], msgspec.to_builtins(previous))
    new = cast(dict[str, object], msgspec.to_builtins(proposed))
    collections = ("stateFields", "commands", "dataInPorts", "dataOutPorts", "execInPorts", "execOutPorts")
    for name in collections:
        old.pop(name, None)
        new.pop(name, None)
    if old != new:
        raise ValueError("spec identity, metadata and edit policy are defined by the installed descriptor")
    _check_collection(previous, "stateFields", previous.stateFields, proposed.stateFields)
    _check_collection(previous, "commands", previous.commands, proposed.commands)
    _check_collection(previous, "dataInPorts", previous.dataInPorts, proposed.dataInPorts)
    _check_collection(previous, "dataOutPorts", previous.dataOutPorts, proposed.dataOutPorts)
    if isinstance(previous, F8OperatorSpec) and isinstance(proposed, F8OperatorSpec):
        _check_collection(previous, "execInPorts", previous.execInPorts, proposed.execInPorts)
        _check_collection(previous, "execOutPorts", previous.execOutPorts, proposed.execOutPorts)
