from NodeGraphQt import NodeObject, BaseNode
from NodeGraphQt.base.node import _ClassProperty

from f8pysdk import F8OperatorSpec, F8ServiceSpec
from f8pysdk.schema_helpers import schema_default

from .port_painter import draw_exec_port, draw_square_port

EXEC_PORT_COLOR = (230, 230, 230)
DATA_PORT_COLOR = (150, 150, 150)
STATE_PORT_COLOR = (200, 200, 50)


class F8BaseRenderNode(BaseNode):

    spec: F8OperatorSpec | F8ServiceSpec

    def __init__(self, qgraphics_item=None):
        super().__init__(qgraphics_item=qgraphics_item)

        self._build_exec_port()
        self._build_data_port()
        self._build_state_port()
        self._build_state_properties()

    def _build_exec_port(self):
        if not isinstance(self.spec, F8OperatorSpec):
            return

        for p in self.spec.execInPorts:
            self.add_input(
                f"[E]{p}",
                color=EXEC_PORT_COLOR,
                painter_func=draw_exec_port,
            )

        for p in self.spec.execOutPorts:
            self.add_output(
                f"{p}[E]",
                color=EXEC_PORT_COLOR,
                painter_func=draw_exec_port,
            )

    def _build_data_port(self):

        for p in self.spec.dataInPorts:
            self.add_input(
                f"[D]{p.name}",
                color=DATA_PORT_COLOR,
            )

        for p in self.spec.dataOutPorts:
            self.add_output(
                f"{p.name}[D]",
                color=DATA_PORT_COLOR,
            )

    def _build_state_port(self):

        for s in self.spec.stateFields:
            if not s.showOnNode:
                continue
            self.add_input(
                f"[S]{s.name}",
                color=STATE_PORT_COLOR,
                painter_func=draw_square_port,
            )

            self.add_output(
                f"{s.name}[S]",
                color=STATE_PORT_COLOR,
                painter_func=draw_square_port,
            )

    def _build_state_properties(self) -> None:
        for s in self.spec.stateFields or []:
            name = str(getattr(s, "name", "") or "").strip()
            if not name:
                continue
            try:
                if self.has_property(name):  # type: ignore[attr-defined]
                    continue
            except Exception:
                pass
            try:
                default_value = schema_default(s.valueSchema)
            except Exception:
                default_value = None
            try:
                self.create_property(name, default_value)
            except Exception:
                continue
