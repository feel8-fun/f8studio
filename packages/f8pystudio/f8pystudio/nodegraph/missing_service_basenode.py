from __future__ import annotations

import json
import logging
from typing import Any

from NodeGraphQt.constants import NodePropWidgetEnum

from .container_basenode import F8StudioContainerBaseNode

logger = logging.getLogger(__name__)


class F8StudioMissingServiceBaseNode(F8StudioContainerBaseNode):
    """
    Placeholder service/container node used when a session references an unregistered node type.

    Session loader sets `custom` values:
      - missingType: original `type_` string
      - missingSpec: JSON string of the original `f8_spec` (best-effort)
    """

    def __init__(self, qgraphics_item=None):
        super().__init__(qgraphics_item=qgraphics_item)
        self._ensure_missing_property("missingType", default="")
        self._ensure_missing_property("missingSpec", default="{}", multiline=True)

    def _ensure_missing_property(self, name: str, *, default: Any, multiline: bool = False) -> None:
        prop_name = str(name or "").strip()
        if not prop_name:
            return
        try:
            if self.has_property(prop_name):  # type: ignore[attr-defined]
                return
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            widget = NodePropWidgetEnum.QTEXT_EDIT.value if multiline else NodePropWidgetEnum.QLINE_EDIT.value
            self.create_property(
                prop_name,
                default,
                widget_type=widget,
                widget_tooltip="Session referenced an unregistered node type; this is a placeholder.",
                tab="Node",
            )
        except Exception:
            logger.exception("Failed to create missing-service property: %s", prop_name)

    def sync_from_spec(self) -> None:
        super().sync_from_spec()

        try:
            missing_type = str(self.get_property("missingType") or "").strip()
        except Exception:
            missing_type = ""
        if missing_type:
            try:
                self.set_name(f"[Missing] {missing_type}")
            except (AttributeError, RuntimeError, TypeError):
                pass

        try:
            raw = self.get_property("missingSpec")
        except Exception:
            raw = None
        if isinstance(raw, dict):
            try:
                self.set_property("missingSpec", json.dumps(raw, ensure_ascii=False, indent=2), push_undo=False)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

