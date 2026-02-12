from __future__ import annotations

from typing import Protocol, TypeAlias, TypeGuard

from f8pysdk import F8OperatorSpec, F8ServiceSpec


HIDDEN_NODE_TAGS: frozenset[str] = frozenset({"__hidden__", "__missing__"})
SpecTemplate: TypeAlias = F8OperatorSpec | F8ServiceSpec


class SupportsSpecTemplate(Protocol):
    SPEC_TEMPLATE: SpecTemplate


def _has_typed_spec_template(obj: object) -> TypeGuard[SupportsSpecTemplate]:
    if not hasattr(obj, "SPEC_TEMPLATE"):
        return False
    spec = getattr(obj, "SPEC_TEMPLATE")
    return isinstance(spec, (F8OperatorSpec, F8ServiceSpec))


def typed_spec_template_or_none(node_cls: object) -> SpecTemplate | None:
    if not _has_typed_spec_template(node_cls):
        return None
    return node_cls.SPEC_TEMPLATE


def _extract_tags(spec: SpecTemplate) -> list[str]:
    tags_any = spec.tags
    return [str(tag).strip().lower() for tag in list(tags_any or [])]


def is_hidden_spec_node_class(node_cls: object) -> bool:
    """
    Return True when a node class has `SPEC_TEMPLATE.tags` that marks it hidden.
    """
    spec = typed_spec_template_or_none(node_cls)
    if spec is None:
        return False
    tags = _extract_tags(spec)
    if not tags:
        return False
    return any(tag in HIDDEN_NODE_TAGS for tag in tags)
