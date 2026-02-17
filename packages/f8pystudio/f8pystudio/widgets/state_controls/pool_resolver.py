from __future__ import annotations

import json
from typing import Any, Callable

from ..f8_editor_widgets import parse_multiselect_pool as _parse_multiselect_pool
from ..f8_editor_widgets import parse_select_pool as _parse_select_pool


def resolve_pool_items(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    if isinstance(value, str):
        raw = str(value or "").strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, (list, tuple)):
            return []
        out: list[str] = []
        for item in parsed:
            if isinstance(item, str):
                token = item.strip()
                if token:
                    out.append(token)
                continue
            if isinstance(item, dict):
                token = str(item.get("id") or "").strip()
                if token:
                    out.append(token)
                continue
            token = str(item).strip()
            if token:
                out.append(token)
        return out

    return []


def build_node_pool_resolver(node: Any) -> Callable[[str], list[str]]:
    def _resolver(field_name: str) -> list[str]:
        field = str(field_name or "").strip()
        if not field:
            return []
        try:
            value = node.get_property(field)
        except (AttributeError, KeyError, TypeError):
            return []
        return resolve_pool_items(value)

    return _resolver


def parse_select_pool(ui_control: str) -> str:
    return str(_parse_select_pool(ui_control))


def parse_multiselect_pool(ui_control: str) -> str:
    return str(_parse_multiselect_pool(ui_control))
