from __future__ import annotations

import re


VARIANT_NODE_TYPE_PREFIX = "__variant__."
_LEGACY_VARIANT_NODE_TYPE_PREFIX = "__variant__:"
_SAFE_VARIANT_ID_RE = re.compile(r"[^A-Za-z0-9_]")


def _normalize_variant_id(variant_id: str) -> str:
    value = str(variant_id or "").strip()
    return _SAFE_VARIANT_ID_RE.sub("_", value)


def build_variant_node_type(variant_id: str) -> str:
    return f"{VARIANT_NODE_TYPE_PREFIX}{_normalize_variant_id(variant_id)}"


def is_variant_node_type(node_type: str) -> bool:
    value = str(node_type or "").strip()
    return value.startswith(VARIANT_NODE_TYPE_PREFIX) or value.startswith(_LEGACY_VARIANT_NODE_TYPE_PREFIX)


def parse_variant_node_type(node_type: str) -> str | None:
    value = str(node_type or "").strip()
    if value.startswith(VARIANT_NODE_TYPE_PREFIX):
        raw = value[len(VARIANT_NODE_TYPE_PREFIX) :]
        return raw or None
    if value.startswith(_LEGACY_VARIANT_NODE_TYPE_PREFIX):
        raw = value[len(_LEGACY_VARIANT_NODE_TYPE_PREFIX) :]
        return _normalize_variant_id(raw) or None
    return None
