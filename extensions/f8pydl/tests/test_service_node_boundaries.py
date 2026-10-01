from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (PKG_PYDL,):
    if path not in sys.path:
        sys.path.insert(0, path)

from f8pydl.service_node import OnnxVisionServiceNode


class _MetadataNode(OnnxVisionServiceNode):
    def __init__(self) -> None:
        super().__init__(
            node_id="detector",
            node=SimpleNamespace(stateFields=[]),
            initial_state=None,
            service_class="f8.dl.detector",
            service_task="detector",
            output_port="detections",
            allowed_tasks={"yolo_det"},
        )
        self.state_updates: list[tuple[str, Any, bool]] = []

    def _resolve_model_yaml(self) -> Path:
        raise FileNotFoundError("model yaml missing")

    async def set_state(
        self,
        field: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        force_publish: bool = False,
    ) -> None:
        _ = ts_ms
        self.state_updates.append((str(field), value, bool(force_publish)))


class ServiceNodeBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_publish_selected_model_metadata_clears_fields_on_missing_yaml(self) -> None:
        node = _MetadataNode()

        await node._publish_selected_model_metadata(force_publish=True)

        self.assertEqual(
            node.state_updates,
            [
                ("modelClasses", [], True),
                ("enabledClasses", [], True),
            ],
        )


if __name__ == "__main__":
    unittest.main()
