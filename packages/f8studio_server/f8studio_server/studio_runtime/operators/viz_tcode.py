from __future__ import annotations

from ...presentation_models import TCodeSnapshot

import msgspec

import time
import re
from typing import Any

from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry
from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    integer_schema,
    string_schema,
)

from ..identifiers import SERVICE_CLASS
from ..presentation import PresentationOutlet
from .categories import PALETTE_CATEGORY_VIZ
from ._viz_base import viz_sampling_state_fields


class VizTCodeRuntimeNode(OperatorNode):
    presentation: PresentationOutlet

    SPEC = F8OperatorSpec(
        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
        serviceClass=SERVICE_CLASS,
        paletteCategory=PALETTE_CATEGORY_VIZ,
        operatorClass="f8.viz.tcode",
        version="0.1.0",
        label="TCode Viz",
        description="Visualize OSR TCode streams in the locally bundled Web renderer.",
        tags=["viz", "tcode", "osr", "device"],
        rendererClass="viz_tcode",
        dataInPorts=[F8DataPortSpec(name="tcode", valueSchema=string_schema(), definitionProtected=True)],
        dataOutPorts=[],
        stateFields=[
            F8StateSpec(
                name="model",
                label="Model",
                valueSchema=string_schema(default="SR6", enum=["OSR2", "SR6", "SSR1"]),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=True,
            ),
            F8StateSpec(
                name="maxLineLength",
                label="Max Line Length",
                valueSchema=integer_schema(default=4096, minimum=32, maximum=65536),
                access=F8StateAccess.rw,
                valueRequired=True,
                showOnNode=False,
            ),
            *viz_sampling_state_fields(),
        ],
    )

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=["tcode"],
            data_out_ports=[],
            state_fields=[field.name for field in (node.stateFields or [])],
        )
        state = initial_state or {}
        self._model = self._model_value(state.get("model"))
        self._max_line_length = self._line_limit(state.get("maxLineLength"))
        self._model_sent = False
        self._channels: dict[str, int] = {}
        self._reset_version = 0

    async def close(self) -> None:
        self.presentation.emit(self.node_id, "viz.tcode.detach", {}, ts_ms=int(time.time() * 1000))

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        if port != "tcode" or not isinstance(value, str):
            return
        timestamp = int(ts_ms) if ts_ms is not None else int(time.time() * 1000)
        self._emit_model(timestamp)
        line = value.replace("\r", "")[: self._max_line_length]
        if not line:
            return
        if not line.endswith("\n"):
            line += "\n"
        for match in re.finditer(r"(?:^|\s)([LRVA][0-9])(\d{1,4})(?=[IS\s]|$)", line):
            self._channels[match.group(1)] = min(9999, int(match.group(2)))
        self._emit_snapshot(line, timestamp)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        timestamp = int(ts_ms) if ts_ms is not None else int(time.time() * 1000)
        if field == "model":
            model = self._model_value(value)
            if model != self._model or not self._model_sent:
                self._model = model
                self._model_sent = False
                self._emit_model(timestamp)
                self._channels.clear()
                self._reset_version += 1
                self._emit_snapshot("", timestamp)
        elif field == "maxLineLength":
            self._max_line_length = self._line_limit(value)

    def _emit_model(self, timestamp: int) -> None:
        if self._model_sent:
            return
        self._emit_snapshot("", timestamp)
        self._model_sent = True

    def _emit_snapshot(self, line: str, timestamp: int) -> None:
        self.presentation.emit(
            self.node_id,
            "viz.tcode.snapshot",
            msgspec.to_builtins(
                TCodeSnapshot(
                    model=self._model,
                    line=line,
                    channels=dict(self._channels),
                    resetVersion=self._reset_version,
                )
            ),
            ts_ms=timestamp,
        )

    @staticmethod
    def _model_value(value: object) -> str:
        model = str(value or "SR6").upper()
        return model if model in {"OSR2", "SR6", "SSR1"} else "SR6"

    @staticmethod
    def _line_limit(value: object) -> int:
        if value is not None and not isinstance(value, (str, int, float)):
            return 4096
        try:
            parsed = int(value) if value is not None else 4096
        except (TypeError, ValueError):
            return 4096
        return max(32, min(65536, parsed))


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(VizTCodeRuntimeNode.SPEC, VizTCodeRuntimeNode, overwrite=True)
    return registry


__all__ = ["VizTCodeRuntimeNode", "register_operator"]
