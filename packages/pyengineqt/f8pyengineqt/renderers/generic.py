from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
import uuid

from f8pysdk import (
    F8PrimitiveTypeEnum,
    F8StateAccess,
    F8DataTypeSchema,
    F8OperatorSpec,
    F8ServiceSpec,
    schema_default,
    schema_type,
)

from ..services.service_operator_registry import ServiceOperatorSpecRegistry
from ..schema.compat import PORT_KIND_DATA_KEY, PORT_SCHEMA_SIG_DATA_KEY, schema_signature

from NodeGraphQt import BaseNode


from NodeGraphQt.qgraphics.node_base import NodeItem
from NodeGraphQt.constants import NodeEnum, NodePropWidgetEnum, PortEnum
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget

from .internal.port_painter import draw_exec_port, draw_square_port

from qtpy import QtWidgets

EMPTY_PORT_COLOR = (0, 0, 0, 0)
EXEC_PORT_COLOR = (230, 230, 230)
DATA_PORT_COLOR = (150, 150, 150)
STATE_PORT_COLOR = (200, 200, 50)

PORT_ROW_DATA_KEY = 10001
WIDGET_ROW_DATA_KEY = 10002

PORT_KIND_EXEC = "exec"
PORT_KIND_DATA = "data"
PORT_KIND_STATE = "state"


def _clean_port_label(port_name: str) -> str:
    name = port_name
    for prefix in ("[E]", "[D]", "[S]"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    for suffix in ("[E]", "[D]", "[S]"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


class _InlineStateRowWidget(NodeBaseWidget):
    def __init__(
        self,
        parent: Any,
        *,
        name: str,
        label_text: str,
        schema: F8DataTypeSchema,
        value: Any,
        show_control: bool,
        read_only: bool,
        enum_values: list[Any] | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        step: float | None = None,
    ) -> None:
        super().__init__(parent, name, "")
        self._schema = schema
        self._schema_t = schema_type(schema)
        self._control: QtWidgets.QWidget | None = None

        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        label = QtWidgets.QLabel(label_text)
        label.setStyleSheet("color: rgba(200,200,200,140); font-size: 8pt;")
        layout.addWidget(label, 1)

        if show_control:
            self._control = self._build_control(
                enum_values=enum_values,
                minimum=minimum,
                maximum=maximum,
                step=step,
            )
            if self._control is not None:
                layout.addWidget(self._control, 0)
                self.set_value(value)
                self._control.setEnabled(not read_only)

        self.set_custom_widget(container)

    def _build_control(
        self,
        *,
        enum_values: list[Any] | None,
        minimum: float | None,
        maximum: float | None,
        step: float | None,
    ) -> QtWidgets.QWidget | None:
        t = self._schema_t
        if t == F8PrimitiveTypeEnum.boolean:
            w = QtWidgets.QCheckBox()
            w.stateChanged.connect(self.on_value_changed)
            return w
        if t == F8PrimitiveTypeEnum.integer:
            w = QtWidgets.QSpinBox()
            w.setMinimumWidth(80)
            min_v = int(minimum) if minimum is not None else -(2**31)
            max_v = int(maximum) if maximum is not None else 2**31 - 1
            w.setRange(min_v, max_v)
            if step is not None:
                w.setSingleStep(int(step))
            w.valueChanged.connect(self.on_value_changed)
            return w
        if t == F8PrimitiveTypeEnum.number:
            w = QtWidgets.QDoubleSpinBox()
            w.setDecimals(6)
            w.setMinimumWidth(100)
            min_v = float(minimum) if minimum is not None else -1e18
            max_v = float(maximum) if maximum is not None else 1e18
            w.setRange(min_v, max_v)
            if step is not None:
                try:
                    w.setSingleStep(float(step))
                except Exception:
                    pass
            w.valueChanged.connect(self.on_value_changed)
            return w
        if t == F8PrimitiveTypeEnum.string:
            if enum_values:
                w = QtWidgets.QComboBox()
                w.setMinimumWidth(110)
                w.addItems([str(v) for v in enum_values])
                w.currentIndexChanged.connect(self.on_value_changed)
                return w
            w = QtWidgets.QLineEdit()
            w.setMinimumWidth(110)
            w.editingFinished.connect(self.on_value_changed)
            return w
        return None

    def get_value(self) -> Any:
        if self._control is None:
            return None
        if isinstance(self._control, QtWidgets.QCheckBox):
            return bool(self._control.isChecked())
        if isinstance(self._control, QtWidgets.QSpinBox):
            return int(self._control.value())
        if isinstance(self._control, QtWidgets.QDoubleSpinBox):
            return float(self._control.value())
        if isinstance(self._control, QtWidgets.QComboBox):
            return str(self._control.currentText())
        if isinstance(self._control, QtWidgets.QLineEdit):
            return self._control.text()
        return None

    def set_value(self, value: Any) -> None:
        if self._control is None:
            return
        if isinstance(self._control, QtWidgets.QCheckBox):
            self._control.setChecked(bool(value))
        elif isinstance(self._control, QtWidgets.QSpinBox):
            self._control.setValue(int(value or 0))
        elif isinstance(self._control, QtWidgets.QDoubleSpinBox):
            self._control.setValue(float(value or 0.0))
        elif isinstance(self._control, QtWidgets.QComboBox):
            idx = self._control.findText("" if value is None else str(value))
            if idx >= 0:
                self._control.setCurrentIndex(idx)
        elif isinstance(self._control, QtWidgets.QLineEdit):
            self._control.setText("" if value is None else str(value))


class GridNodeItem(NodeItem):  # type: ignore[misc]
    """
    A custom NodeGraphQt node item that aligns ports by "row index" metadata.

    This avoids injecting spacer ports just to keep left/right ports aligned.
    """

    def __init__(self) -> None:
        super().__init__()
        self._suspend_draw = False
        self._pending_draw = False

    def begin_transaction(self) -> None:
        self._suspend_draw = True

    def end_transaction(self) -> None:
        self._suspend_draw = False
        if self._pending_draw:
            self._pending_draw = False
            super().draw_node()

    def draw_node(self) -> None:
        if getattr(self, "_suspend_draw", False):
            self._pending_draw = True
            return
        self._pending_draw = False
        super().draw_node()

    def _port_row(self, port_item: Any, *, fallback: int) -> int:
        try:
            row = port_item.data(PORT_ROW_DATA_KEY)
        except Exception:
            row = None
        if row is None:
            return fallback
        try:
            return int(row)
        except Exception:
            return fallback

    def _row_count(self) -> int:
        ports = [p for p in [*self.inputs, *self.outputs] if p.isVisible()]
        if not ports:
            return 0
        max_row = -1
        for idx, port in enumerate(ports):
            row = self._port_row(port, fallback=idx)
            if row > max_row:
                max_row = row
        return max_row + 1

    def _port_metrics(self) -> tuple[float, float]:
        ports = [p for p in [*self.inputs, *self.outputs] if p.isVisible()]
        if not ports:
            return (0.0, 0.0)
        rect = ports[0].boundingRect()
        return (rect.width(), rect.height())

    def _port_area_height(self) -> float:
        _, port_h = self._port_metrics()
        if not port_h:
            return 0.0
        rows = self._row_count()
        if rows <= 0:
            return 0.0
        spacing = 1.0
        return rows * (port_h + spacing)

    def _calc_size_horizontal(self) -> tuple[float, float]:
        text_w = self._text_item.boundingRect().width()

        port_w = 0.0
        p_input_text_w = 0.0
        p_output_text_w = 0.0
        for port, text in self._input_items.items():
            if not port.isVisible():
                continue
            if not port_w:
                port_w = port.boundingRect().width()
            t_width = text.boundingRect().width()
            if text.isVisible() and t_width > p_input_text_w:
                p_input_text_w = t_width
        for port, text in self._output_items.items():
            if not port.isVisible():
                continue
            if not port_w:
                port_w = port.boundingRect().width()
            t_width = text.boundingRect().width()
            if text.isVisible() and t_width > p_output_text_w:
                p_output_text_w = t_width

        inline_widget_w = 0.0
        bottom_widget_w = 0.0
        bottom_widget_h = 0.0
        for widget in self._widgets.values():
            if not widget.isVisible():
                continue
            w_width = widget.boundingRect().width()
            w_height = widget.boundingRect().height()
            try:
                is_inline = widget.data(WIDGET_ROW_DATA_KEY) is not None
            except Exception:
                is_inline = False

            if is_inline:
                if w_width > inline_widget_w:
                    inline_widget_w = w_width
            else:
                if w_width > bottom_widget_w:
                    bottom_widget_w = w_width
                bottom_widget_h += w_height

        side_padding = 20.0
        width = max(
            NodeEnum.WIDTH.value,
            (port_w * 2)
            + max(text_w, p_input_text_w + p_output_text_w, inline_widget_w, bottom_widget_w)
            + side_padding,
        )

        height = self._port_area_height() + (bottom_widget_h + 6.0 if bottom_widget_h else 0.0)
        height *= 1.05
        return width, height

    def _align_ports_horizontal(self, v_offset: float) -> None:
        width = self._width
        spacing = 1.0
        txt_offset = PortEnum.CLICK_FALLOFF.value - 2

        port_w, port_h = self._port_metrics()
        if not port_h:
            return

        left_x = (port_w / 2) * -1
        right_x = width - (port_w / 2)

        visible_inputs = [p for p in self.inputs if p.isVisible()]
        for idx, port in enumerate(visible_inputs):
            row = self._port_row(port, fallback=idx)
            port.setPos(left_x, v_offset + row * (port_h + spacing))

        visible_outputs = [p for p in self.outputs if p.isVisible()]
        for idx, port in enumerate(visible_outputs):
            row = self._port_row(port, fallback=idx)
            port.setPos(right_x, v_offset + row * (port_h + spacing))

        for port, text in self._input_items.items():
            if not port.isVisible():
                continue
            try:
                text.setPlainText(_clean_port_label(port.name))
            except Exception:
                pass
            txt_x = (port.boundingRect().width() / 2) - txt_offset
            text.setPos(txt_x, port.y() - 1.5)

        for port, text in self._output_items.items():
            if not port.isVisible():
                continue
            try:
                text.setPlainText(_clean_port_label(port.name))
            except Exception:
                pass
            txt_w = text.boundingRect().width() - txt_offset
            txt_x = port.x() - txt_w
            text.setPos(txt_x, port.y() - 1.5)

    def _align_widgets_horizontal(self, v_offset: float) -> None:
        if not self._widgets:
            return
        rect = self.boundingRect()

        port_w, port_h = self._port_metrics()
        spacing = 1.0

        max_in_txt = 0.0
        for port, text in self._input_items.items():
            if port.isVisible() and text.isVisible():
                max_in_txt = max(max_in_txt, text.boundingRect().width())
        max_out_txt = 0.0
        for port, text in self._output_items.items():
            if port.isVisible() and text.isVisible():
                max_out_txt = max(max_out_txt, text.boundingRect().width())

        # Inline widgets (per-row).
        for widget in self._widgets.values():
            if not widget.isVisible():
                continue
            try:
                row = widget.data(WIDGET_ROW_DATA_KEY)
            except Exception:
                row = None
            if row is None:
                continue
            try:
                row = int(row)
            except Exception:
                continue

            widget_rect = widget.boundingRect()
            min_x = rect.left() + (port_w * 0.5) + 6.0 + max_in_txt + 10.0
            max_x = rect.right() - (port_w * 0.5) - 6.0 - max_out_txt - 10.0 - widget_rect.width()
            if max_x < min_x:
                x = rect.center().x() - (widget_rect.width() / 2)
            else:
                x = min_x + (max_x - min_x) * 0.5
            y = rect.y() + v_offset + row * (port_h + spacing) - 2.0
            try:
                widget.widget().setTitleAlign("center")
            except Exception:
                pass
            widget.setPos(x, y)

        # Bottom widgets (stacked).
        y = rect.y() + v_offset + self._port_area_height() + 6.0
        for widget in self._widgets.values():
            if not widget.isVisible():
                continue
            try:
                if widget.data(WIDGET_ROW_DATA_KEY) is not None:
                    continue
            except Exception:
                pass
            widget_rect = widget.boundingRect()
            x = rect.center().x() - (widget_rect.width() / 2)
            try:
                widget.widget().setTitleAlign("center")
            except Exception:
                pass
            widget.setPos(x, y)
            y += widget_rect.height()

@dataclass
class PortHandles:
    exec_in: dict[str, Any] = field(default_factory=dict)
    exec_out: dict[str, Any] = field(default_factory=dict)
    data_in: dict[str, Any] = field(default_factory=dict)
    data_out: dict[str, Any] = field(default_factory=dict)
    state_in: dict[str, Any] = field(default_factory=dict)
    state_out: dict[str, Any] = field(default_factory=dict)


class GenericNode(BaseNode):  # type: ignore[misc]
    """
    Base NodeGraphQt node for editing OperatorSpecs in NodeGraphQt.

    This node is editor-only: it builds ports + properties from an F8OperatorSpec
    template fetched from `ServiceOperatorSpecRegistry` using `SPEC_KEY`.
    """

    __identifier__ = "feel8.renderer"
    NODE_NAME = "Generic"

    SPEC_KEY: str = ""

    LABEL: str = ""

    spec: F8OperatorSpec | F8ServiceSpec

    def __init__(self) -> None:
        super().__init__(qgraphics_item=GridNodeItem)
        stable_id = uuid.uuid4().hex
        try:
            self.model.id = stable_id  # type: ignore[attr-defined]
            self.view.id = stable_id  # type: ignore[attr-defined]
        except Exception:
            pass

        spec_key = self.SPEC_KEY
        if ServiceOperatorSpecRegistry.instance().has(spec_key):
            self.spec = ServiceOperatorSpecRegistry.instance().get(spec_key)
        else:
            raise ValueError(f"Spec [{spec_key}] not found.")

        self.port_handles = PortHandles()
        self._state_rows: dict[str, int] = {}

        # Required for dynamic port rebuilds (`delete_input`/`delete_output`).
        # NodeGraphQt raises if port deletion is not allowed on the node model.
        self.set_port_deletion_allowed(True)  # type: ignore[attr-defined]

        self.set_name(self.spec.label)  # type: ignore[attr-defined]
        self._build_ports()
        self._apply_state_properties()
        self._apply_inline_state_widgets()

    @contextmanager
    def _atomic_node_update(self):
        viewer = None
        viewport = None
        try:
            graph = getattr(self, "graph", None)
            if graph is not None:
                try:
                    viewer = graph.viewer()
                    viewport = viewer.viewport() if viewer is not None else None
                except Exception:
                    viewer = None
                    viewport = None

            if viewer is not None:
                try:
                    viewer.setUpdatesEnabled(False)
                except Exception:
                    pass
            if viewport is not None:
                try:
                    viewport.setUpdatesEnabled(False)
                except Exception:
                    pass

            try:
                self.view.begin_transaction()  # type: ignore[attr-defined]
            except Exception:
                pass

            yield
        finally:
            try:
                self.view.end_transaction()  # type: ignore[attr-defined]
            except Exception:
                try:
                    self.view.draw_node()
                except Exception:
                    pass

            if viewport is not None:
                try:
                    viewport.setUpdatesEnabled(True)
                except Exception:
                    pass
            if viewer is not None:
                try:
                    viewer.setUpdatesEnabled(True)
                    viewer.update()
                except Exception:
                    pass

    def _tag_port_row(self, handle: Any, row: int) -> None:
        try:
            handle.view.setData(PORT_ROW_DATA_KEY, int(row))
        except Exception:
            pass

    def _tag_port_meta(self, handle: Any, *, kind: str, schema: Any | None = None) -> None:
        try:
            handle.view.setData(PORT_KIND_DATA_KEY, str(kind))
        except Exception:
            pass
        try:
            handle.view.setData(PORT_SCHEMA_SIG_DATA_KEY, schema_signature(schema))
        except Exception:
            pass

    def _clear_inline_widgets(self) -> None:
        try:
            widgets = dict(getattr(self.view, "_widgets", {}) or {})
        except Exception:
            return
        for name, widget in list(widgets.items()):
            if not (str(name).startswith("__inline_state__") or str(name).startswith("__inline_state_row__")):
                continue
            try:
                self.view._widgets.pop(name, None)
            except Exception:
                pass
            try:
                widget.setParentItem(None)
            except Exception:
                pass
            try:
                widget.deleteLater()
            except Exception:
                pass

    def _apply_inline_state_widgets(self) -> None:
        """
        Add per-row inline widgets for state fields.

        - Always shows a label in the row.
        - Shows a control widget only when `showOnNode=True`.
        - Values are bound to node properties (state name).
        """
        self._clear_inline_widgets()

        for field in self.spec.states or []:
            row = self._state_rows.get(field.name)
            if row is None:
                continue

            try:
                current_value = self.get_property(field.name)
            except Exception:
                current_value = None
            if current_value is None:
                current_value = schema_default(field.valueSchema)

            read_only = field.access == F8StateAccess.ro
            show_control = bool(getattr(field, "showOnNode", False))

            widget_name = f"__inline_state_row__{field.name}"
            label_text = field.label or field.name
            enum_values = getattr(field.valueSchema, "enum", None)
            minimum = getattr(field.valueSchema, "minimum", None)
            maximum = getattr(field.valueSchema, "maximum", None)
            step = getattr(field.valueSchema, "multipleOf", None)

            widget = _InlineStateRowWidget(
                self.view,
                name=widget_name,
                label_text=label_text,
                schema=field.valueSchema,
                value=current_value,
                show_control=show_control,
                read_only=read_only,
                enum_values=enum_values,
                minimum=minimum,
                maximum=maximum,
                step=step,
            )

            try:
                widget.setData(WIDGET_ROW_DATA_KEY, int(row))
            except Exception:
                pass

            try:
                widget._node = self
            except Exception:
                pass

            try:
                widget.widget().setEnabled(not read_only)
            except Exception:
                pass

            widget.value_changed.connect(lambda _k, v, _name=field.name: self.set_property(_name, v, push_undo=False))
            try:
                self.view.add_widget(widget)
            except Exception:
                continue

        try:
            self.view.draw_node()
        except Exception:
            pass

    def ensure_state_properties(self) -> None:
        """
        Ensure NodeGraphQt properties exist for every state field.

        The custom inspector writes via `set_property`, so missing properties
        would otherwise silently drop edits.
        """
        self._apply_state_properties()

    def apply_spec(self, spec: F8OperatorSpec | F8ServiceSpec) -> None:
        """
        Replace the per-node spec copy and rebuild ports/properties.

        Note: rebuilding ports may disconnect existing links (NodeGraphQt
        deletes ports). Callers should confirm with the user.
        """
        self._validate_spec_for_ports(spec)
        old_spec = getattr(self, "spec", None)
        edge_snapshots = self._snapshot_edges()
        with self._atomic_node_update():
            try:
                self.spec = spec
                self.set_name(self.spec.label)  # type: ignore[attr-defined]
                self._build_ports()
                self._apply_state_properties()
                self._apply_inline_state_widgets()
                self._restore_edges(edge_snapshots)
                try:
                    graph = getattr(self, "graph", None)
                    if graph is not None:
                        graph.property_changed.emit(self, "__spec_changed__", self.spec.operatorClass)
                except Exception:
                    pass
            except Exception:
                if old_spec is not None:
                    try:
                        self.spec = old_spec
                        self.set_name(self.spec.label)  # type: ignore[attr-defined]
                        self._build_ports()
                        self._apply_state_properties()
                        self._apply_inline_state_widgets()
                        self._restore_edges(edge_snapshots)
                    except Exception:
                        pass
                raise

    def _spec_port_signature(self, spec: F8OperatorSpec | F8ServiceSpec) -> dict[str, Any]:
        """
        Build a signature map for ports owned by this node.

        Used to decide whether an existing connection can be restored after
        a spec rebuild (eg. if schema type changed, skip reconnect).
        """
        sig: dict[str, Any] = {}

        for port in getattr(spec, "dataInPorts", None) or []:
            try:
                sig[f"[D]{port.name}"] = schema_type(port.valueSchema)
            except Exception:
                sig[f"[D]{port.name}"] = None
        for port in getattr(spec, "dataOutPorts", None) or []:
            try:
                sig[f"{port.name}[D]"] = schema_type(port.valueSchema)
            except Exception:
                sig[f"{port.name}[D]"] = None

        for field in getattr(spec, "states", None) or []:
            access = getattr(field, "access", None) or F8StateAccess.ro
            try:
                st = schema_type(field.valueSchema)
            except Exception:
                st = None
            if access != F8StateAccess.ro:
                sig[f"[S]{field.name}"] = st
            if access != F8StateAccess.wo:
                sig[f"{field.name}[S]"] = st

        return sig

    def _port_signature_for_node(self, node: Any, port_name: str) -> Any:
        """
        Best-effort signature for a NodeGraphQt port name on the given node.

        Returns `None` if the node does not expose a compatible spec.
        """
        try:
            spec = getattr(node, "spec", None)
            if spec is None:
                return None
            if hasattr(node, "_spec_port_signature"):
                mapping = node._spec_port_signature(spec)  # type: ignore[attr-defined]
                return mapping.get(port_name)
        except Exception:
            return None
        return None

    def _snapshot_edges(self) -> set[tuple[str, str, str, str, Any, Any]]:
        """
        Snapshot all pipe connections touching this node as:
        (src_id, src_port, dst_id, dst_port, src_sig, dst_sig)

        The tuple is always oriented output->input.
        """
        edges: set[tuple[str, str, str, str]] = set()
        try:
            ports = [*list(self.input_ports()), *list(self.output_ports())]  # type: ignore[attr-defined]
        except Exception:
            ports = []

        for port in ports:
            try:
                connected = list(port.connected_ports())
            except Exception:
                connected = []
            for other in connected:
                try:
                    a_is_out = port.type_() == "out"
                    b_is_out = other.type_() == "out"
                    if a_is_out and not b_is_out:
                        edges.add((port.node().id, port.name(), other.node().id, other.name()))
                    elif b_is_out and not a_is_out:
                        edges.add((other.node().id, other.name(), port.node().id, port.name()))
                except Exception:
                    continue

        snapshots: set[tuple[str, str, str, str, Any, Any]] = set()
        try:
            graph = self.graph
        except Exception:
            graph = None
        for src_id, src_port, dst_id, dst_port in edges:
            if graph is None:
                snapshots.add((src_id, src_port, dst_id, dst_port, None, None))
                continue
            try:
                src_node = graph.get_node_by_id(src_id)
                dst_node = graph.get_node_by_id(dst_id)
            except Exception:
                src_node = None
                dst_node = None
            src_sig = self._port_signature_for_node(src_node, src_port) if src_node is not None else None
            dst_sig = self._port_signature_for_node(dst_node, dst_port) if dst_node is not None else None
            snapshots.add((src_id, src_port, dst_id, dst_port, src_sig, dst_sig))
        return snapshots

    def _restore_edges(self, edge_snapshots: set[tuple[str, str, str, str, Any, Any]]) -> None:
        """
        Try to restore connections after ports have been rebuilt.

        Only reconnects edges where ports still exist and both ends
        have matching port signatures (when available).
        """
        try:
            graph = self.graph
        except Exception:
            return

        for src_id, src_port_name, dst_id, dst_port_name, src_sig, dst_sig in edge_snapshots:
            try:
                src_node = graph.get_node_by_id(src_id)
                dst_node = graph.get_node_by_id(dst_id)
            except Exception:
                continue
            if src_node is None or dst_node is None:
                continue

            if src_sig is not None:
                current_src_sig = self._port_signature_for_node(src_node, src_port_name)
                if current_src_sig != src_sig:
                    continue
            if dst_sig is not None:
                current_dst_sig = self._port_signature_for_node(dst_node, dst_port_name)
                if current_dst_sig != dst_sig:
                    continue

            try:
                out_port = src_node.outputs().get(src_port_name)
                in_port = dst_node.inputs().get(dst_port_name)
            except Exception:
                continue
            if out_port is None or in_port is None:
                continue

            try:
                out_port.connect_to(in_port)
            except Exception:
                pass

    @staticmethod
    def _find_duplicates(values: list[str]) -> list[str]:
        seen: set[str] = set()
        dupes: set[str] = set()
        for value in values:
            if value in seen:
                dupes.add(value)
            else:
                seen.add(value)
        return sorted(dupes)

    def _validate_spec_for_ports(self, spec: F8OperatorSpec | F8ServiceSpec) -> None:
        exec_in = [str(p).strip() for p in (getattr(spec, "execInPorts", None) or [])]
        exec_out = [str(p).strip() for p in (getattr(spec, "execOutPorts", None) or [])]
        data_in = [str(p.name).strip() for p in (getattr(spec, "dataInPorts", None) or [])]
        data_out = [str(p.name).strip() for p in (getattr(spec, "dataOutPorts", None) or [])]
        states = [str(s.name).strip() for s in (getattr(spec, "states", None) or [])]

        errors: list[str] = []
        if any(not p for p in exec_in):
            errors.append("execInPorts contains an empty name.")
        if any(not p for p in exec_out):
            errors.append("execOutPorts contains an empty name.")
        if any(not p for p in data_in):
            errors.append("dataInPorts contains an empty name.")
        if any(not p for p in data_out):
            errors.append("dataOutPorts contains an empty name.")
        if any(not p for p in states):
            errors.append("states contains an empty name.")

        dup = self._find_duplicates(exec_in)
        if dup:
            errors.append(f"Duplicate execInPorts: {', '.join(dup)}")
        dup = self._find_duplicates(exec_out)
        if dup:
            errors.append(f"Duplicate execOutPorts: {', '.join(dup)}")
        dup = self._find_duplicates(data_in)
        if dup:
            errors.append(f"Duplicate dataInPorts: {', '.join(dup)}")
        dup = self._find_duplicates(data_out)
        if dup:
            errors.append(f"Duplicate dataOutPorts: {', '.join(dup)}")
        dup = self._find_duplicates(states)
        if dup:
            errors.append(f"Duplicate states: {', '.join(dup)}")

        if errors:
            raise ValueError("Invalid spec for port rebuild:\n" + "\n".join(errors))

    # Spacer ports are intentionally removed; port alignment is handled by GridNodeItem.

    def port(self, kind: str, direction: str, name: str) -> Any:
        mapping = {
            ("exec", "in"): self.port_handles.exec_in,
            ("exec", "out"): self.port_handles.exec_out,
            ("data", "in"): self.port_handles.data_in,
            ("data", "out"): self.port_handles.data_out,
            ("state", "in"): self.port_handles.state_in,
            ("state", "out"): self.port_handles.state_out,
        }.get((kind, direction))
        if mapping is None:
            return None
        return mapping.get(name)

    # Builders ------------------------------------------------------------
    def _build_ports(self) -> None:
        self._clear_ports()
        row = 0
        row = self._build_exec_ports(row)
        row = self._build_data_ports(row)
        self._build_state_ports(row)

    def _clear_ports(self) -> None:
        try:
            self.set_port_deletion_allowed(True)  # type: ignore[attr-defined]
        except Exception:
            pass

        def force_disconnect(port: Any) -> None:
            try:
                connected = list(port.connected_ports())
            except Exception:
                connected = []
            if not connected:
                return

            try:
                if port.locked():
                    port.set_locked(False, connected_ports=False, push_undo=False)
            except Exception:
                pass

            for other in connected:
                other_was_locked = False
                try:
                    other_was_locked = bool(other.locked())
                except Exception:
                    other_was_locked = False
                if other_was_locked:
                    try:
                        other.set_locked(False, connected_ports=False, push_undo=False)
                    except Exception:
                        other_was_locked = False

                try:
                    port.disconnect_from(other, push_undo=False, emit_signal=False)
                except Exception:
                    pass
                finally:
                    if other_was_locked:
                        try:
                            other.set_locked(True, connected_ports=False, push_undo=False)
                        except Exception:
                            pass

        for port in list(self.input_ports()):  # type: ignore[attr-defined]
            try:
                if port.locked():
                    port.set_locked(False, connected_ports=False, push_undo=False)
            except Exception:
                pass
            try:
                force_disconnect(port)
            except Exception:
                pass
            self.delete_input(port)
        for port in list(self.output_ports()):  # type: ignore[attr-defined]
            try:
                if port.locked():
                    port.set_locked(False, connected_ports=False, push_undo=False)
            except Exception:
                pass
            try:
                force_disconnect(port)
            except Exception:
                pass
            self.delete_output(port)

        self.port_handles = PortHandles()

    def _build_exec_ports(self, row_offset: int) -> int:
        exec_in = list(self.spec.execInPorts or [])
        exec_out = list(self.spec.execOutPorts or [])
        rows = max(len(exec_in), len(exec_out))

        for idx, port in enumerate(exec_in):
            handle = self.add_input(f"[E]{port}", color=EXEC_PORT_COLOR, multi_input=False, painter_func=draw_exec_port)  # type: ignore[attr-defined]
            self._tag_port_row(handle, row_offset + idx)
            self._tag_port_meta(handle, kind=PORT_KIND_EXEC)
            self.port_handles.exec_in[port] = handle
        for idx, port in enumerate(exec_out):
            handle = self.add_output(f"{port}[E]", color=EXEC_PORT_COLOR, multi_output=False, painter_func=draw_exec_port)  # type: ignore[attr-defined]
            self._tag_port_row(handle, row_offset + idx)
            self._tag_port_meta(handle, kind=PORT_KIND_EXEC)
            self.port_handles.exec_out[port] = handle

        return row_offset + rows

    def _build_data_ports(self, row_offset: int) -> int:
        data_in = list(self.spec.dataInPorts or [])
        data_out = list(self.spec.dataOutPorts or [])
        rows = max(len(data_in), len(data_out))

        for idx, port in enumerate(data_in):
            handle = self.add_input(f"[D]{port.name}", color=DATA_PORT_COLOR)  # type: ignore[attr-defined]
            self._tag_port_row(handle, row_offset + idx)
            self._tag_port_meta(handle, kind=PORT_KIND_DATA, schema=port.valueSchema)
            self.port_handles.data_in[port.name] = handle
        for idx, port in enumerate(data_out):
            handle = self.add_output(f"{port.name}[D]", color=DATA_PORT_COLOR)  # type: ignore[attr-defined]
            self._tag_port_row(handle, row_offset + idx)
            self._tag_port_meta(handle, kind=PORT_KIND_DATA, schema=port.valueSchema)
            self.port_handles.data_out[port.name] = handle

        return row_offset + rows

    def _build_state_ports(self, row_offset: int) -> int:
        fields = list(self.spec.states or [])
        self._state_rows = {}
        for idx, field in enumerate(fields):
            access = field.access or F8StateAccess.ro
            row = row_offset + idx
            self._state_rows[field.name] = row

            if access != F8StateAccess.ro:
                handle = self.add_input(
                    name=f"[S]{field.name}",
                    color=STATE_PORT_COLOR,
                    display_name=False,
                    painter_func=draw_square_port,
                )  # type: ignore[attr-defined]
                self._tag_port_row(handle, row)
                self._tag_port_meta(handle, kind=PORT_KIND_STATE, schema=field.valueSchema)
                self.port_handles.state_in[field.name] = handle

            if access != F8StateAccess.wo:
                handle = self.add_output(
                    name=f"{field.name}[S]",
                    color=STATE_PORT_COLOR,
                    display_name=False,
                    painter_func=draw_square_port,
                )  # type: ignore[attr-defined]
                self._tag_port_row(handle, row)
                self._tag_port_meta(handle, kind=PORT_KIND_STATE, schema=field.valueSchema)
                self.port_handles.state_out[field.name] = handle

        return row_offset + len(fields)

    def _apply_state_properties(self) -> None:
        # Create embedded widgets for state fields and wire them to node properties.

        for field in self.spec.states or []:
            schema = field.valueSchema
            # access = field.access
            # label = field.name
            default_value = schema_default(schema)

            field_type = schema_type(schema)

            if self.has_property(field.name):  # type: ignore[attr-defined]
                continue

            if field_type == F8PrimitiveTypeEnum.boolean:
                self.create_property(field.name, default_value, widget_type=NodePropWidgetEnum.QCHECK_BOX.value)
            elif field_type == F8PrimitiveTypeEnum.integer:
                minimum = getattr(schema, "minimum", None)
                maximum = getattr(schema, "maximum", None)
                if minimum is not None and maximum is not None:
                    self.create_property(
                        field.name,
                        default_value,
                        widget_type=NodePropWidgetEnum.SLIDER.value,
                        range=(int(minimum), int(maximum)),
                    )
                else:
                    self.create_property(field.name, default_value, widget_type=NodePropWidgetEnum.QLINE_EDIT.value)
            elif field_type == F8PrimitiveTypeEnum.number:

                minimum = getattr(schema, "minimum", None)
                maximum = getattr(schema, "maximum", None)
                if minimum is not None and maximum is not None:
                    self.create_property(
                        field.name,
                        default_value,
                        widget_type=NodePropWidgetEnum.DOUBLE_SLIDER.value,
                        range=(float(minimum), float(maximum)),
                    )
                else:
                    self.create_property(field.name, default_value, widget_type=NodePropWidgetEnum.QLINE_EDIT.value)

            elif field_type == F8PrimitiveTypeEnum.string:
                enum_values = getattr(schema, "enum", None) if schema else None
                code_language = getattr(field, "language", None) if field else None
                if enum_values:
                    items = [str(v) for v in enum_values]
                    self.create_property(
                        field.name,
                        default_value,
                        widget_type=NodePropWidgetEnum.QCOMBO_BOX.value,
                        items=items,
                    )
                elif code_language:
                    self.create_property(field.name, default_value, widget_type=NodePropWidgetEnum.QTEXT_EDIT.value)
                else:
                    self.create_property(field.name, default_value, widget_type=NodePropWidgetEnum.QLINE_EDIT.value)
            else:
                # object/array/any: keep it as a raw python value (dict/list/etc).
                self.create_property(field.name, default_value, widget_type=NodePropWidgetEnum.QLINE_EDIT.value)


class UiOperatorNode(GenericNode):  # type: ignore[misc]
    """
    UI-focused renderer node variant (placeholder for style differences).

    Keeping it separate allows rendererClass/rendererId to select node visuals.
    """

    __identifier__ = "fun.feel8.op.renderer.ui"
