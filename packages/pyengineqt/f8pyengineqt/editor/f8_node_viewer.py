from __future__ import annotations

from typing import Any

from NodeGraphQt.constants import PortTypeEnum
from NodeGraphQt.widgets.viewer import NodeViewer

from .f8_pipe_item import F8PipeItem, pipe_style_for_kind
from ..schema.compat import (
    PORT_KIND_DATA_KEY,
    PORT_SCHEMA_SIG_DATA_KEY,
    infer_port_kind,
    schema_is_superset,
)


class F8NodeViewer(NodeViewer):
    """
    Custom viewer that rejects incompatible connections during interactive wiring.

    Rules:
    - exec/data/state cannot be cross-connected.
    - for data/state, output schema must be a superset of input schema.
      (if either side is `any`, treat as compatible)
    """

    def _port_kind(self, port_item: Any) -> str | None:
        try:
            kind = port_item.data(PORT_KIND_DATA_KEY)
        except Exception:
            kind = None
        if kind:
            return str(kind)
        try:
            return infer_port_kind(port_item.name)
        except Exception:
            return None

    def _port_schema_sig(self, port_item: Any) -> dict[str, Any] | None:
        try:
            sig = port_item.data(PORT_SCHEMA_SIG_DATA_KEY)
        except Exception:
            sig = None
        return sig if isinstance(sig, dict) else None

    def _validate_accept_connection(self, from_port: Any, to_port: Any) -> bool:
        if not super()._validate_accept_connection(from_port, to_port):
            return False

        try:
            if from_port.port_type == to_port.port_type:
                return False
        except Exception:
            pass

        try:
            out_item = from_port if from_port.port_type == PortTypeEnum.OUT.value else to_port
            in_item = from_port if from_port.port_type == PortTypeEnum.IN.value else to_port
        except Exception:
            return True

        out_kind = self._port_kind(out_item)
        in_kind = self._port_kind(in_item)
        if out_kind and in_kind and out_kind != in_kind:
            return False

        kind = out_kind or in_kind
        if kind not in ("data", "state"):
            return True

        out_sig = self._port_schema_sig(out_item)
        in_sig = self._port_schema_sig(in_item)
        return schema_is_superset(out_sig, in_sig)

    def start_live_connection(self, selected_port: Any) -> None:
        super().start_live_connection(selected_port)
        try:
            kind = self._port_kind(selected_port)
            color, width, _ = pipe_style_for_kind(kind)
            self._LIVE_PIPE.set_pipe_styling(color=color, width=max(int(width), 2), style=self._LIVE_PIPE.style)
        except Exception:
            pass

    def establish_connection(self, start_port: Any, end_port: Any) -> None:
        try:
            out_item = start_port if start_port.port_type == PortTypeEnum.OUT.value else end_port
        except Exception:
            out_item = start_port
        try:
            kind = self._port_kind(out_item)
        except Exception:
            kind = None

        pipe = F8PipeItem(kind=kind)
        self.scene().addItem(pipe)
        pipe.set_connections(start_port, end_port)
        pipe.draw_path(pipe.input_port, pipe.output_port)
        if start_port.node.selected or end_port.node.selected:
            pipe.highlight()
        if not start_port.node.visible or not end_port.node.visible:
            pipe.hide()
