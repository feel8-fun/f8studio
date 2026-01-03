from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from f8pysdk import (
    F8PrimitiveTypeEnum,
    F8StateAccess,
    F8DataTypeSchema,
    F8OperatorSpec,
    F8ServiceSpec,
    schema_default,
    schema_type,
)

from ..operators.operator_registry import OperatorSpecRegistry

from NodeGraphQt import BaseNode


from NodeGraphQt.qgraphics.node_base import NodeItem
from NodeGraphQt.constants import NodeEnum, NodePropWidgetEnum, PortEnum
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget

from .port_painter import draw_exec_port, draw_square_port

from qtpy import QtCore, QtWidgets

EMPTY_PORT_COLOR = (0, 0, 0, 0)
EXEC_PORT_COLOR = (230, 230, 230)
DATA_PORT_COLOR = (150, 150, 150)
STATE_PORT_COLOR = (200, 200, 50)

PORT_ROW_DATA_KEY = 10001
WIDGET_ROW_DATA_KEY = 10002


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


class _NodeSpinBox(NodeBaseWidget):
    def __init__(
        self,
        parent: Any = None,
        *,
        name: str,
        label: str = "",
        minimum: int | None = None,
        maximum: int | None = None,
        step: int | None = None,
        value: int | None = None,
    ) -> None:
        super().__init__(parent, name, label)
        widget = QtWidgets.QSpinBox()
        widget.setMinimumWidth(90)
        widget.setRange(minimum if minimum is not None else -(2**31), maximum if maximum is not None else 2**31 - 1)
        if step is not None:
            widget.setSingleStep(int(step))
        if value is not None:
            widget.setValue(int(value))
        widget.valueChanged.connect(self.on_value_changed)
        widget.clearFocus()
        self.set_custom_widget(widget)

    def get_value(self) -> int:
        return int(self.get_custom_widget().value())

    def set_value(self, value: int | None = None) -> None:
        widget = self.get_custom_widget()
        next_value = int(value or 0)
        if next_value != widget.value():
            widget.setValue(next_value)


class _NodeDoubleSpinBox(NodeBaseWidget):
    def __init__(
        self,
        parent: Any = None,
        *,
        name: str,
        label: str = "",
        minimum: float | None = None,
        maximum: float | None = None,
        step: float | None = None,
        value: float | None = None,
    ) -> None:
        super().__init__(parent, name, label)
        widget = QtWidgets.QDoubleSpinBox()
        widget.setDecimals(6)
        widget.setMinimumWidth(110)
        widget.setRange(minimum if minimum is not None else -1e18, maximum if maximum is not None else 1e18)
        if step is not None:
            widget.setSingleStep(float(step))
        if value is not None:
            widget.setValue(float(value))
        widget.valueChanged.connect(self.on_value_changed)
        widget.clearFocus()
        self.set_custom_widget(widget)

    def get_value(self) -> float:
        return float(self.get_custom_widget().value())

    def set_value(self, value: float | None = None) -> None:
        widget = self.get_custom_widget()
        next_value = float(value or 0.0)
        if next_value != widget.value():
            widget.setValue(next_value)


class GridNodeItem(NodeItem):  # type: ignore[misc]
    """
    A custom NodeGraphQt node item that aligns ports by "row index" metadata.

    This avoids injecting spacer ports just to keep left/right ports aligned.
    """

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
    template fetched from `OperatorSpecRegistry` using `OPERATOR_CLASS`.
    """

    __identifier__ = "feel8.renderer"
    NODE_NAME = "Generic"

    SPEC_KEY: str = ""

    spec: F8OperatorSpec | F8ServiceSpec

    def __init__(self) -> None:
        super().__init__(qgraphics_item=GridNodeItem)

        spec_key = self.SPEC_KEY
        # Lookup order: OperatorSpecRegistry -> ServiceSpecRegistry -> UserSpecsLibrary.

        if OperatorSpecRegistry.instance().has(spec_key):
            self.spec = OperatorSpecRegistry.instance().get(spec_key)
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

    def _tag_port_row(self, handle: Any, row: int) -> None:
        try:
            handle.view.setData(PORT_ROW_DATA_KEY, int(row))
        except Exception:
            pass

    def _clear_inline_widgets(self) -> None:
        try:
            widgets = dict(getattr(self.view, "_widgets", {}) or {})
        except Exception:
            return
        for name, widget in list(widgets.items()):
            if not str(name).startswith("__inline_state__"):
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
        Add per-row inline widgets for state fields with `showOnNode=True`.

        Values are bound to node properties (state name).
        """
        self._clear_inline_widgets()

        for field in self.spec.states or []:
            if not getattr(field, "showOnNode", False):
                continue

            row = self._state_rows.get(field.name)
            if row is None:
                continue

            try:
                current_value = self.get_property(field.name)
            except Exception:
                current_value = None
            if current_value is None:
                current_value = schema_default(field.valueSchema)

            schema_t = schema_type(field.valueSchema)
            widget: NodeBaseWidget | None = None
            widget_name = f"__inline_state__{field.name}"
            read_only = field.access == F8StateAccess.ro

            if schema_t == F8PrimitiveTypeEnum.boolean:
                w = NodeBaseWidget(None, widget_name, "")
                cbox = QtWidgets.QCheckBox("")
                cbox.setChecked(bool(current_value))
                cbox.stateChanged.connect(w.on_value_changed)
                w.set_custom_widget(cbox)
                widget = w
            elif schema_t == F8PrimitiveTypeEnum.integer:
                minimum = getattr(field.valueSchema, "minimum", None)
                maximum = getattr(field.valueSchema, "maximum", None)
                step = getattr(field.valueSchema, "multipleOf", None)
                widget = _NodeSpinBox(
                    None,
                    name=widget_name,
                    minimum=int(minimum) if minimum is not None else None,
                    maximum=int(maximum) if maximum is not None else None,
                    step=int(step) if step is not None else None,
                    value=int(current_value or 0),
                )
            elif schema_t == F8PrimitiveTypeEnum.number:
                minimum = getattr(field.valueSchema, "minimum", None)
                maximum = getattr(field.valueSchema, "maximum", None)
                step = getattr(field.valueSchema, "multipleOf", None)
                widget = _NodeDoubleSpinBox(
                    None,
                    name=widget_name,
                    minimum=float(minimum) if minimum is not None else None,
                    maximum=float(maximum) if maximum is not None else None,
                    step=float(step) if step is not None else None,
                    value=float(current_value or 0.0),
                )
            elif schema_t == F8PrimitiveTypeEnum.string:
                enum_values = getattr(field.valueSchema, "enum", None)
                if enum_values:
                    from NodeGraphQt.widgets.node_widgets import NodeComboBox

                    widget = NodeComboBox(None, widget_name, "", items=[str(v) for v in enum_values])
                    try:
                        widget.set_value("" if current_value is None else str(current_value))
                    except Exception:
                        pass
                else:
                    from NodeGraphQt.widgets.node_widgets import NodeLineEdit

                    widget = NodeLineEdit(
                        None, widget_name, "", text="" if current_value is None else str(current_value)
                    )

            if widget is None:
                continue

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

            widget.value_changed.connect(
                lambda _k, v, _name=field.name: self.set_property(_name, v, push_undo=False)
            )
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
        old_spec = getattr(self, "spec", None)
        try:
            self._validate_spec_for_ports(spec)
            self.spec = spec
            self.set_name(self.spec.label)  # type: ignore[attr-defined]
            self._build_ports()
            self._apply_state_properties()
            self._apply_inline_state_widgets()
        except Exception:
            if old_spec is not None:
                try:
                    self.spec = old_spec
                    self.set_name(self.spec.label)  # type: ignore[attr-defined]
                    self._build_ports()
                    self._apply_state_properties()
                    self._apply_inline_state_widgets()
                except Exception:
                    pass
            raise

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

        for port in list(self.input_ports()):  # type: ignore[attr-defined]
            try:
                if port.locked():
                    port.set_locked(False, connected_ports=False, push_undo=False)
            except Exception:
                pass
            try:
                port.clear_connections(push_undo=False, emit_signal=False)
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
                port.clear_connections(push_undo=False, emit_signal=False)
            except Exception:
                pass
            self.delete_output(port)

        self.port_handles = PortHandles()

    def _build_exec_ports(self, row_offset: int) -> int:
        exec_in = list(self.spec.execInPorts or [])
        exec_out = list(self.spec.execOutPorts or [])
        rows = max(len(exec_in), len(exec_out))

        for idx, port in enumerate(exec_in):
            handle = self.add_input(f"[E]{port}", color=EXEC_PORT_COLOR, painter_func=draw_exec_port)  # type: ignore[attr-defined]
            self._tag_port_row(handle, row_offset + idx)
            self.port_handles.exec_in[port] = handle
        for idx, port in enumerate(exec_out):
            handle = self.add_output(f"{port}[E]", color=EXEC_PORT_COLOR, painter_func=draw_exec_port)  # type: ignore[attr-defined]
            self._tag_port_row(handle, row_offset + idx)
            self.port_handles.exec_out[port] = handle

        return row_offset + rows

    def _build_data_ports(self, row_offset: int) -> int:
        data_in = list(self.spec.dataInPorts or [])
        data_out = list(self.spec.dataOutPorts or [])
        rows = max(len(data_in), len(data_out))

        for idx, port in enumerate(data_in):
            handle = self.add_input(f"[D]{port.name}", color=DATA_PORT_COLOR)  # type: ignore[attr-defined]
            self._tag_port_row(handle, row_offset + idx)
            self.port_handles.data_in[port.name] = handle
        for idx, port in enumerate(data_out):
            handle = self.add_output(f"{port.name}[D]", color=DATA_PORT_COLOR)  # type: ignore[attr-defined]
            self._tag_port_row(handle, row_offset + idx)
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
                    display_name=True,
                    painter_func=draw_square_port,
                )  # type: ignore[attr-defined]
                self._tag_port_row(handle, row)
                self.port_handles.state_in[field.name] = handle

            if access != F8StateAccess.wo:
                handle = self.add_output(
                    name=f"{field.name}[S]",
                    color=STATE_PORT_COLOR,
                    display_name=True,
                    painter_func=draw_square_port,
                )  # type: ignore[attr-defined]
                self._tag_port_row(handle, row)
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
