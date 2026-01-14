from NodeGraphQt import NodeObject, BaseNode
from NodeGraphQt.base.node import _ClassProperty
from NodeGraphQt.constants import NodePropWidgetEnum

from f8pysdk import F8OperatorSpec, F8ServiceSpec
from f8pysdk.schema_helpers import schema_default, schema_type

from .port_painter import draw_exec_port, draw_square_port

EXEC_PORT_COLOR = (230, 230, 230)
DATA_PORT_COLOR = (150, 150, 150)
STATE_PORT_COLOR = (200, 200, 50)


class F8BaseRenderNode(BaseNode):

    spec: F8OperatorSpec | F8ServiceSpec

    def __init__(self, qgraphics_item=None):
        super().__init__(qgraphics_item=qgraphics_item)
        # Spec editing requires rebuilding ports at runtime.
        self.set_port_deletion_allowed(True)

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
            widget_type, items, prop_range = self._state_widget_for_schema(getattr(s, "valueSchema", None))
            tooltip = str(getattr(s, "description", "") or "").strip() or None
            try:
                self.create_property(
                    name,
                    default_value,
                    items=items,
                    range=prop_range,
                    widget_type=widget_type,
                    widget_tooltip=tooltip,
                    tab="State",
                )
            except Exception:
                continue

    @staticmethod
    def _state_widget_for_schema(value_schema) -> tuple[int, list[str] | None, tuple[float, float] | None]:
        """
        Best-effort mapping from F8DataTypeSchema -> NodeGraphQt property widget.
        """
        if value_schema is None:
            return NodePropWidgetEnum.QTEXT_EDIT.value, None, None
        try:
            t = schema_type(value_schema)
        except Exception:
            t = ""

        # enum choice.
        try:
            enum_items = list(getattr(getattr(value_schema, "root", None), "enum", None) or [])
        except Exception:
            enum_items = []
        if enum_items:
            return NodePropWidgetEnum.QCOMBO_BOX.value, [str(x) for x in enum_items], None

        if t == "boolean":
            return NodePropWidgetEnum.QCHECK_BOX.value, None, None
        if t == "integer":
            # Avoid QSpinBox widgets due to PySide6 incompatibilities in NodeGraphQt's PropSpinBox.
            return NodePropWidgetEnum.QLINE_EDIT.value, None, None
        if t == "number":
            # Avoid QDoubleSpinBox widgets due to PySide6 incompatibilities in NodeGraphQt's PropDoubleSpinBox.
            return NodePropWidgetEnum.QLINE_EDIT.value, None, None
        if t == "string":
            return NodePropWidgetEnum.QLINE_EDIT.value, None, None

        # object/array/any (and unknowns) edited as JSON-ish text.
        return NodePropWidgetEnum.QTEXT_EDIT.value, None, None

    def sync_from_spec(self) -> None:
        """
        Rebuild runtime aspects derived from `self.spec`:
        - ports (exec/data/state)
        - state properties (adds any missing fields)
        """
        try:
            if not self.port_deletion_allowed():
                self.set_port_deletion_allowed(True)
        except Exception:
            pass

        # Rebuild ports (best-effort; may drop connections).
        try:
            for port in list(self._inputs):
                try:
                    self.delete_input(port)
                except Exception:
                    pass
            for port in list(self._outputs):
                try:
                    self.delete_output(port)
                except Exception:
                    pass
        except Exception:
            pass

        self._build_exec_port()
        self._build_data_port()
        self._build_state_port()
        self._build_state_properties()

        try:
            self.view.draw_node()
        except Exception:
            pass
