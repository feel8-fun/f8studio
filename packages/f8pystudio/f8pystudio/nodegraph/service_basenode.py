from __future__ import annotations

import enum
import logging
import json
from dataclasses import dataclass
from typing import Any

from .node_base import F8StudioBaseNode

from f8pysdk import F8ServiceSpec, F8StateAccess

from collections import OrderedDict

from f8pysdk.schema_helpers import schema_default, schema_type

from qtpy import QtCore, QtGui, QtWidgets

from NodeGraphQt.constants import (
    ICON_NODE_BASE,
    ITEM_CACHE_MODE,
    Z_VAL_NODE,
    LayoutDirectionEnum,
    NodeEnum,
    PortEnum,
    PortTypeEnum,
    NodePropWidgetEnum,
)
from NodeGraphQt.errors import NodeWidgetError
from NodeGraphQt.qgraphics.node_abstract import AbstractNodeItem
from NodeGraphQt.qgraphics.node_overlay_disabled import XDisabledItem
from NodeGraphQt.qgraphics.node_text_item import NodeTextItem
from NodeGraphQt.qgraphics.port import CustomPortItem, PortItem

from .port_painter import draw_exec_port, draw_square_port, EXEC_PORT_COLOR, DATA_PORT_COLOR, STATE_PORT_COLOR
from .service_process_toolbar import ServiceProcessToolbar
from .service_bridge_protocol import ServiceBridge
from .viewer import F8StudioNodeViewer
from ..widgets.f8_editor_widgets import (
    F8ImageB64Editor,
    F8MultiSelect,
    F8OptionCombo,
    F8Switch,
    F8ValueBar,
    parse_multiselect_pool,
    parse_select_pool,
)
from ..widgets.f8_prop_value_widgets import open_code_editor_dialog, open_code_editor_window
from ..command_ui_protocol import CommandUiHandler, CommandUiSource
import qtawesome as qta

logger = logging.getLogger(__name__)


def _port_name(port: Any) -> str:
    """
    NodeGraphQt Port exposes `name()` (method).
    """
    try:
        return str(port.name() or "")
    except Exception:
        pass
    try:
        return str(port.name or "")
    except Exception:
        return ""


def _model_extra(obj: Any) -> dict[str, Any]:
    """
    Best-effort access to pydantic v2 extra fields without RTTI (`getattr`).
    """
    try:
        extra = obj.model_extra
        return extra if isinstance(extra, dict) else {}
    except Exception:
        pass
    try:
        extra = obj.__pydantic_extra__
        return extra if isinstance(extra, dict) else {}
    except Exception:
        return {}


def _service_exec_ports(spec: F8ServiceSpec) -> tuple[list[str], list[str]]:
    """
    F8ServiceSpec doesn't declare exec ports, but it allows extra fields.
    """
    extra = _model_extra(spec)
    in_raw = extra.get("execInPorts")
    out_raw = extra.get("execOutPorts")
    exec_in = [str(x) for x in list(in_raw or [])] if isinstance(in_raw, (list, tuple)) else []
    exec_out = [str(x) for x in list(out_raw or [])] if isinstance(out_raw, (list, tuple)) else []
    return exec_in, exec_out


@dataclass(frozen=True)
class _StateFieldInfo:
    name: str
    label: str
    tooltip: str
    show_on_node: bool
    access: Any
    access_str: str
    required: bool
    ui_control: str
    ui_language: str
    value_schema: Any


def _state_field_info(field: Any) -> _StateFieldInfo | None:
    if isinstance(field, dict):
        name = str(field.get("name") or "").strip()
    else:
        try:
            name = str(field.name or "").strip()
        except Exception:
            return None
    if not name:
        return None

    if isinstance(field, dict):
        show_on_node = bool(field.get("showOnNode") or False)
    else:
        try:
            show_on_node = bool(field.showOnNode)
        except Exception:
            show_on_node = False

    if isinstance(field, dict):
        label = str(field.get("label") or "").strip() or name
        tooltip = str(field.get("description") or "").strip() or name
        ui_control = str(field.get("uiControl") or "").strip()
        ui_language = str(field.get("uiLanguage") or "")
        value_schema = field.get("valueSchema")
        access = field.get("access")
        required = bool(field.get("required") or False)
    else:
        try:
            label = str(field.label or "").strip() or name
        except Exception:
            label = name
        try:
            tooltip = str(field.description or "").strip() or name
        except Exception:
            tooltip = name
        try:
            ui_control = str(field.uiControl or "").strip()
        except Exception:
            ui_control = ""
        try:
            ui_language = str(field.uiLanguage or "")
        except Exception:
            ui_language = ""
        try:
            value_schema = field.valueSchema
        except Exception:
            value_schema = None
        try:
            access = field.access
        except Exception:
            access = None
        try:
            required = bool(field.required)
        except Exception:
            required = False

    if isinstance(access, enum.Enum):
        access_value = access.value
    else:
        access_value = access if access is not None else ""
    access_str = str(access_value or "").strip().lower()

    return _StateFieldInfo(
        name=name,
        label=label,
        tooltip=tooltip,
        show_on_node=show_on_node,
        access=access,
        access_str=access_str,
        required=required,
        ui_control=ui_control,
        ui_language=ui_language,
        value_schema=value_schema,
    )


class _F8ElideToolButton(QtWidgets.QToolButton):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self._full_text = ""

    def setFullText(self, text: str) -> None:
        self._full_text = str(text or "")
        self._apply_elide()

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_elide()

    def event(self, event):  # type: ignore[override]
        # Tooltips on embedded widgets inside a QGraphicsProxyWidget can pick up
        # an unexpected palette/style (showing as a black box). Force the
        # tooltip to be shown with the global/default styling by passing
        # widget=None.
        try:
            if event.type() == QtCore.QEvent.ToolTip:
                tip = str(self.toolTip() or "").strip()
                if not tip:
                    return True
                pos = None
                try:
                    pos = event.globalPos()
                except Exception:
                    try:
                        pos = event.globalPosition().toPoint()
                    except Exception:
                        pos = None
                if pos is not None:
                    QtWidgets.QToolTip.showText(pos, tip, None)
                    return True
        except Exception:
            pass
        return super().event(event)

    def _apply_elide(self) -> None:
        try:
            fm = QtGui.QFontMetrics(self.font())
            # Leave room for the arrow icon.
            w = max(10, int(self.width() - 24))
            self.setText(fm.elidedText(self._full_text, QtCore.Qt.ElideRight, w))
        except RuntimeError:
            self.setText(self._full_text)


class _F8ForceGlobalToolTipFilter(QtCore.QObject):
    """
    Force tooltip display via `QToolTip.showText(..., widget=None)` to avoid
    dark/black tooltip palette issues when widgets are embedded in a
    `QGraphicsProxyWidget`.
    """

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # type: ignore[override]
        if event.type() != QtCore.QEvent.ToolTip:
            return super().eventFilter(watched, event)
        if not isinstance(watched, QtWidgets.QWidget):
            return True
        tip = str(watched.toolTip() or "").strip()
        if not tip:
            return True
        try:
            pos = event.globalPos()  # type: ignore[attr-defined]
        except Exception:
            try:
                pos = event.globalPosition().toPoint()  # type: ignore[attr-defined]
            except Exception:
                return True
        QtWidgets.QToolTip.showText(pos, tip, None)
        return True


class F8StudioServiceBaseNode(F8StudioBaseNode):
    """
    Base class for all single-node service (nodes that are intended to live without
    a container).

    This class is intentionally small: container binding is orchestrated by
    `F8StudioGraph`, while the view-level `_container_item` link is managed by
    the container item.
    """

    svcId: Any

    def __init__(self, qgraphics_item=None):
        _nodeitem_cls = qgraphics_item or F8StudioServiceNodeItem
        assert issubclass(
            _nodeitem_cls, F8StudioServiceNodeItem
        ), "F8StudioServiceBaseNode requires a F8StudioServiceNodeItem or subclass."
        super().__init__(qgraphics_item=_nodeitem_cls)
        assert isinstance(self.spec, F8ServiceSpec), "F8StudioServiceBaseNode requires F8ServiceSpec"

        self.set_port_deletion_allowed(True)

        self._build_exec_port()
        self._build_data_port()
        self._build_state_port()
        self._build_state_properties()

    def _build_exec_port(self) -> None:
        """
        Services may optionally expose exec ports (extra schema fields).
        """
        exec_in, exec_out = _service_exec_ports(self.spec)
        for p in exec_in:
            self.add_input(
                f"[E]{p}",
                color=EXEC_PORT_COLOR,
                painter_func=draw_exec_port,
            )

        for p in exec_out:
            self.add_output(
                f"{p}[E]",
                color=EXEC_PORT_COLOR,
                painter_func=draw_exec_port,
            )

    def _build_data_port(self):

        for p in self.spec.dataInPorts:
            if not self.data_port_show_on_node(str(p.name or ""), is_in=True):
                continue
            self.add_input(
                f"[D]{p.name}",
                color=DATA_PORT_COLOR,
            )

        for p in self.spec.dataOutPorts:
            if not self.data_port_show_on_node(str(p.name or ""), is_in=False):
                continue
            self.add_output(
                f"{p.name}[D]",
                color=DATA_PORT_COLOR,
            )

    def _build_state_port(self):

        for s in self.effective_state_fields():
            info = _state_field_info(s)
            if info is None or not info.show_on_node:
                continue

            if info.access in [F8StateAccess.rw, F8StateAccess.wo] or info.access_str in {"rw", "wo"}:
                self.add_input(
                    f"[S]{info.name}",
                    color=STATE_PORT_COLOR,
                    painter_func=draw_square_port,
                )

            if info.access in [F8StateAccess.rw, F8StateAccess.ro] or info.access_str in {"rw", "ro"}:
                self.add_output(
                    f"{info.name}[S]",
                    color=STATE_PORT_COLOR,
                    painter_func=draw_square_port,
                )

    def _build_state_properties(self) -> None:
        for s in self.effective_state_fields() or []:
            info = _state_field_info(s)
            if info is None:
                continue
            if self.has_property(info.name):  # type: ignore[attr-defined]
                continue
            try:
                default_value = schema_default(info.value_schema)
            except Exception:
                default_value = None
            widget_type, items, prop_range = self._state_widget_for_schema(info.value_schema)
            tooltip = info.tooltip or None
            self.create_property(
                info.name,
                default_value,
                items=items,
                range=prop_range,
                widget_type=widget_type,
                widget_tooltip=tooltip,
                tab="State",
            )

    @staticmethod
    def _state_widget_for_schema(value_schema) -> tuple[int, list[str] | None, tuple[float, float] | None]:
        """
        Best-effort mapping from F8DataTypeSchema -> NodeGraphQt property widget.
        """
        if value_schema is None:
            return NodePropWidgetEnum.QTEXT_EDIT.value, None, None
        t = schema_type(value_schema) or ""

        # enum choice.
        try:
            root = value_schema.root
            enum_items = list(root.enum or [])
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
        if not self.port_deletion_allowed():
            self.set_port_deletion_allowed(True)

        # Sync ports from spec.
        #
        # Important: NodeGraphQt `delete_input/delete_output` does not clear
        # pipes. If ports are removed while still connected, NodeGraphQt can
        # leave "dangling" pipes in the scene, crashing during paint.
        desired_inputs: dict[str, dict[str, Any]] = {}
        desired_outputs: dict[str, dict[str, Any]] = {}

        exec_in, exec_out = _service_exec_ports(self.spec)
        for p in exec_in:
            desired_inputs[f"[E]{p}"] = {"color": EXEC_PORT_COLOR, "painter_func": draw_exec_port}
        for p in exec_out:
            desired_outputs[f"{p}[E]"] = {"color": EXEC_PORT_COLOR, "painter_func": draw_exec_port}

        def _port_has_connections(port: Any) -> bool:
            if port is None:
                return False
            try:
                return bool(port.connected_ports())
            except Exception:
                try:
                    return bool(port.connected_ports)
                except Exception:
                    return False

        for p in list(self.spec.dataInPorts or []):
            try:
                n = str(p.name or "").strip()
            except Exception:
                n = ""
            if not n:
                continue
            port_name = f"[D]{n}"
            show_on_node = self.data_port_show_on_node(n, is_in=True)
            if not show_on_node:
                try:
                    if _port_has_connections(self.get_input(port_name)):
                        show_on_node = True
                except Exception:
                    pass
            if show_on_node:
                desired_inputs[port_name] = {"color": DATA_PORT_COLOR}

        for p in list(self.spec.dataOutPorts or []):
            try:
                n = str(p.name or "").strip()
            except Exception:
                n = ""
            if not n:
                continue
            port_name = f"{n}[D]"
            show_on_node = self.data_port_show_on_node(n, is_in=False)
            if not show_on_node:
                try:
                    if _port_has_connections(self.get_output(port_name)):
                        show_on_node = True
                except Exception:
                    pass
            if show_on_node:
                desired_outputs[port_name] = {"color": DATA_PORT_COLOR}

        for s in list(self.effective_state_fields() or []):
            info = _state_field_info(s)
            if info is None or not info.show_on_node:
                continue
            if info.access in [F8StateAccess.rw, F8StateAccess.wo] or info.access_str in {"rw", "wo"}:
                desired_inputs[f"[S]{info.name}"] = {"color": STATE_PORT_COLOR, "painter_func": draw_square_port}
            if info.access in [F8StateAccess.rw, F8StateAccess.ro] or info.access_str in {"rw", "ro"}:
                desired_outputs[f"{info.name}[S]"] = {"color": STATE_PORT_COLOR, "painter_func": draw_square_port}

        # Remove ports that no longer exist in spec (disconnect first).
        current_input_names = set(self.inputs().keys())
        current_output_names = set(self.outputs().keys())
        desired_input_names = set(desired_inputs.keys())
        desired_output_names = set(desired_outputs.keys())

        for name in sorted(current_input_names - desired_input_names):
            try:
                port = self.get_input(name)
                if port is not None:
                    try:
                        port.clear_connections(push_undo=False, emit_signal=False)
                    except (AttributeError, RuntimeError, TypeError):
                        pass
                self.delete_input(name)
            except Exception as e:
                logger.warning("Failed to delete input port %r: %s", name, e)

        for name in sorted(current_output_names - desired_output_names):
            try:
                port = self.get_output(name)
                if port is not None:
                    try:
                        port.clear_connections(push_undo=False, emit_signal=False)
                    except (AttributeError, RuntimeError, TypeError):
                        pass
                self.delete_output(name)
            except Exception as e:
                logger.warning("Failed to delete output port %r: %s", name, e)

        # Add new ports from spec.
        current_input_names = set(self.inputs().keys())
        current_output_names = set(self.outputs().keys())

        for name in sorted(desired_input_names - current_input_names):
            meta = desired_inputs.get(name) or {}
            try:
                self.add_input(name, color=meta.get("color"), painter_func=meta.get("painter_func"))
            except Exception as e:
                logger.warning("Failed to add input port %r: %s", name, e)

        for name in sorted(desired_output_names - current_output_names):
            meta = desired_outputs.get(name) or {}
            try:
                self.add_output(name, color=meta.get("color"), painter_func=meta.get("painter_func"))
            except Exception as e:
                logger.warning("Failed to add output port %r: %s", name, e)

        # Best-effort cleanup for any orphaned port items left on the QGraphics node.
        try:
            view = self.view
            valid_in_views = {p.view for p in self.input_ports()}
            valid_out_views = {p.view for p in self.output_ports()}

            try:
                input_items = view._input_items
            except Exception:
                input_items = None
            if isinstance(input_items, dict):
                for port_item in list(input_items.keys()):
                    if port_item in valid_in_views:
                        continue
                    text_item = input_items.pop(port_item, None)
                    if text_item is None:
                        continue
                    try:
                        port_item.setParentItem(None)
                        text_item.setParentItem(None)
                        if view.scene() is not None:
                            view.scene().removeItem(port_item)
                            view.scene().removeItem(text_item)
                    except RuntimeError:
                        pass

            try:
                output_items = view._output_items
            except Exception:
                output_items = None
            if isinstance(output_items, dict):
                for port_item in list(output_items.keys()):
                    if port_item in valid_out_views:
                        continue
                    text_item = output_items.pop(port_item, None)
                    if text_item is None:
                        continue
                    try:
                        port_item.setParentItem(None)
                        text_item.setParentItem(None)
                        if view.scene() is not None:
                            view.scene().removeItem(port_item)
                            view.scene().removeItem(text_item)
                    except RuntimeError:
                        pass
        except Exception:
            logger.debug("sync_from_spec orphan port-item cleanup failed", exc_info=True)

        self._build_state_properties()

        try:
            self.view.draw_node()
        except Exception:
            logger.debug("sync_from_spec draw_node failed", exc_info=True)


class F8StudioServiceNodeItem(AbstractNodeItem):
    """
    Base Node item.

    Args:
        name (str): name displayed on the node.
        parent (QtWidgets.QGraphicsItem): parent item.
    """

    def __init__(self, name="node", parent=None):
        super(F8StudioServiceNodeItem, self).__init__(name, parent)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsScenePositionChanges, True)

        pixmap = QtGui.QPixmap(ICON_NODE_BASE)
        if pixmap.size().height() > NodeEnum.ICON_SIZE.value:
            pixmap = pixmap.scaledToHeight(NodeEnum.ICON_SIZE.value, QtCore.Qt.SmoothTransformation)
        self._properties["icon"] = ICON_NODE_BASE
        self._icon_item = QtWidgets.QGraphicsPixmapItem(pixmap, self)
        self._icon_item.setTransformationMode(QtCore.Qt.SmoothTransformation)
        self._text_item = NodeTextItem(self.name, self)
        self._x_item = XDisabledItem(self, "DISABLED")
        self._input_items = OrderedDict()
        self._output_items = OrderedDict()
        self._widgets = OrderedDict()
        self._proxy_mode = False
        self._proxy_mode_threshold = 70
        self._state_inline_proxies: OrderedDict[str, QtWidgets.QGraphicsProxyWidget] = OrderedDict()
        self._state_inline_controls: OrderedDict[str, QtWidgets.QWidget] = OrderedDict()
        self._state_inline_updaters: OrderedDict[str, Any] = OrderedDict()
        self._state_inline_toggles: OrderedDict[str, QtWidgets.QToolButton] = OrderedDict()
        self._state_inline_headers: OrderedDict[str, QtWidgets.QWidget] = OrderedDict()
        self._state_inline_bodies: OrderedDict[str, QtWidgets.QWidget] = OrderedDict()
        self._state_inline_expanded: dict[str, bool] = {}
        self._state_inline_option_pools: dict[str, str] = {}
        self._state_row_y: dict[str, tuple[float, float]] = {}
        self._graph_prop_hooked: bool = False
        self._bridge_proc_hooked: bool = False
        self._state_inline_ctrl_serial: dict[str, str] = {}
        self._cmd_serial: str = ""
        self._cmd_proxy: QtWidgets.QGraphicsProxyWidget | None = None
        self._cmd_widget: QtWidgets.QWidget | None = None
        self._cmd_buttons: list[QtWidgets.QAbstractButton] = []
        self._tooltip_filters: list[QtCore.QObject] = []
        self._svc_toolbar_proxy: QtWidgets.QGraphicsProxyWidget | None = None
        self._ports_end_y: float | None = None
        self._open_code_editors: list[QtWidgets.QDialog] = []

    def _backend_node(self) -> Any | None:
        """
        Best-effort access to the backing BaseNode object for this view item.
        """
        g = self._graph()
        if g is None:
            return None
        try:
            node_id = str(self.id or "").strip()
        except Exception:
            node_id = ""
        if not node_id:
            return None
        try:
            return g.get_node_by_id(node_id)
        except KeyError:
            return None

    def _inline_state_input_is_connected(self, field_name: str) -> bool:
        """
        True if the state field is upstream-driven via a state-edge.
        """
        name = str(field_name or "").strip()
        if not name:
            return False
        node = self._backend_node()
        if node is None:
            return False
        p = node.get_input(f"[S]{name}")
        if p is None:
            return False
        return bool(p.connected_ports())

    @staticmethod
    def _set_inline_state_control_read_only(control: QtWidgets.QWidget, *, read_only: bool) -> None:
        """
        Best-effort toggle for inline state controls hosted in the node item.

        Inline controls are created here (not via the property panel), so we
        avoid relying on NodeGraphQt property widgets.
        """
        if isinstance(control, F8OptionCombo):
            control.set_read_only(bool(read_only))
            return
        if isinstance(control, F8Switch):
            control.setEnabled(not bool(read_only))
            return
        if isinstance(control, F8ValueBar):
            control.setEnabled(not bool(read_only))
            return
        if isinstance(control, QtWidgets.QLineEdit):
            control.setEnabled(True)
            control.setReadOnly(bool(read_only))
            return
        if isinstance(control, QtWidgets.QPlainTextEdit):
            control.setEnabled(True)
            control.setReadOnly(bool(read_only))
            return
        if isinstance(control, QtWidgets.QTextEdit):
            control.setEnabled(True)
            control.setReadOnly(bool(read_only))
            if read_only:
                control.setTextInteractionFlags(
                    QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
                    | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
                )
            else:
                control.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextEditorInteraction)
            return
        # Fallback: disable to prevent edits.
        control.setEnabled(not bool(read_only))

    def refresh_inline_state_read_only(self) -> None:
        """
        Refresh readonly state for already-built inline state controls.

        This is used when a state-edge is connected/disconnected: we want to
        avoid rebuilding widgets (which can cause flicker / temporary layout
        glitches) and only toggle editability.
        """
        node = self._backend_node()
        if node is None:
            return
        fields = list(node.effective_state_fields() or [])

        for f in fields:
            info = _state_field_info(f)
            if info is None or not info.show_on_node:
                continue
            name = info.name
            ctrl = self._state_inline_controls.get(name)
            if ctrl is None:
                continue
            read_only = info.access_str == "ro" or self._inline_state_input_is_connected(name)
            self._set_inline_state_control_read_only(ctrl, read_only=bool(read_only))

    def _graph(self) -> Any | None:
        viewer = self._viewer_safe()
        if not isinstance(viewer, F8StudioNodeViewer):
            return None
        return viewer.f8_graph

    def _viewer_safe(self) -> Any | None:
        try:
            return self.viewer()
        except RuntimeError:
            return None

    def _ensure_graph_property_hook(self) -> None:
        if self._graph_prop_hooked:
            return
        g = self._graph()
        if g is None:
            return
        try:
            g.property_changed.connect(self._on_graph_property_changed)  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError, TypeError):
            self._graph_prop_hooked = False
            return
        self._graph_prop_hooked = True

    def _select_node_from_embedded_widget(self) -> None:
        """
        Ensure node selection + property panel update when clicking embedded widgets.

        QGraphicsProxyWidget-hosted controls (eg. state inline toggle headers)
        can consume mouse events so the viewer never emits `node_selected`.
        This makes parts of the node unselectable, and the properties panel will
        not update. Call this from embedded widget handlers to keep behavior
        consistent: clicking anywhere on the node selects it and updates props.
        """
        node = self._backend_node()
        g = self._graph()
        if node is None or g is None:
            return
        scene = None
        try:
            scene = self.scene()
        except Exception:
            scene = None

        try:
            mods = QtWidgets.QApplication.keyboardModifiers()
        except Exception:
            mods = QtCore.Qt.NoModifier
        multi = bool(mods & (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier))

        if scene is not None and not multi:
            try:
                scene.clearSelection()
            except Exception:
                pass
        try:
            self.setSelected(True)
        except Exception:
            pass

        # Drive the studio properties panel (listens to graph signals).
        try:
            g.node_selected.emit(node)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            g.node_selection_changed.emit([node], [])  # type: ignore[attr-defined]
        except Exception:
            pass

    def _bridge(self) -> ServiceBridge | None:
        g = self._graph()
        try:
            return g.service_bridge if g is not None else None
        except Exception:
            return None

    def _ensure_bridge_process_hook(self) -> None:
        if self._bridge_proc_hooked:
            return
        bridge = self._bridge()
        if bridge is None:
            return
        try:
            bridge.service_process_state.connect(self._on_bridge_service_process_state)  # type: ignore[attr-defined]
        except Exception:
            self._bridge_proc_hooked = False
            return
        self._bridge_proc_hooked = True

    def _is_service_running(self) -> bool:
        bridge = self._bridge()
        sid = self._service_id()
        if bridge is None or not sid:
            return False
        try:
            return bool(bridge.is_service_running(sid))
        except Exception:
            return False

    def _on_bridge_service_process_state(self, service_id: str, running: bool) -> None:
        if str(service_id or "").strip() != self._service_id():
            return
        enabled = bool(running)
        for b in list(self._cmd_buttons):
            try:
                b.setEnabled(enabled)
                if not enabled:
                    b.setToolTip(
                        (b.toolTip() or "").strip()
                        + ("\nService not running" if b.toolTip() else "Service not running")
                    )
            except Exception:
                continue
        try:
            QtCore.QTimer.singleShot(0, self.draw_node)
        except Exception:
            pass

    def _service_id(self) -> str:
        # For service nodes, nodeId == serviceId.
        try:
            return str(self.id or "").strip()
        except Exception:
            return ""

    def _invoke_command(self, cmd: Any) -> None:
        """
        Invoke a command declared on the service spec.

        - no params: fire immediately
        - has params: show dialog to collect args
        """
        if isinstance(cmd, dict):
            call = str(cmd.get("name") or "").strip()
        else:
            try:
                call = str(cmd.name or "").strip()
            except Exception:
                call = ""
        if not call:
            return
        bridge = self._bridge()
        if bridge is None:
            return
        sid = self._service_id()
        if not sid:
            return
        if not self._is_service_running():
            return

        # Allow a node to intercept command invocation with custom UI logic.
        try:
            node = self._backend_node()
        except Exception:
            node = None
        if isinstance(node, CommandUiHandler):
            parent = None
            try:
                v = self.viewer()
                parent = v.window() if v is not None else None
            except Exception:
                parent = None
            try:
                if bool(node.handle_command_ui(cmd, parent=parent, source=CommandUiSource.NODEGRAPH)):
                    return
            except Exception:
                node_id = ""
                try:
                    node_id = str(self.id or "").strip()
                except Exception:
                    node_id = ""
                logger.exception("handle_command_ui failed nodeId=%s", node_id)
        if isinstance(cmd, dict):
            params = list(cmd.get("params") or [])
        else:
            try:
                params = list(cmd.params or [])
            except Exception:
                params = []

        if not params:
            try:
                bridge.invoke_remote_command(sid, call, {})
            except Exception:
                logger.exception("invoke_remote_command failed serviceId=%s call=%s", sid, call)
            return

        args = self._prompt_command_args(cmd)
        if args is None:
            return
        try:
            bridge.invoke_remote_command(sid, call, args)
        except Exception:
            logger.exception("invoke_remote_command failed serviceId=%s call=%s", sid, call)

    def _prompt_command_args(self, cmd: Any) -> dict[str, Any] | None:
        if isinstance(cmd, dict):
            call = str(cmd.get("name") or "").strip() or "Command"
            params = list(cmd.get("params") or [])
        else:
            try:
                call = str(cmd.name or "").strip() or "Command"
            except Exception:
                call = "Command"
            try:
                params = list(cmd.params or [])
            except Exception:
                params = []
        if not params:
            return {}

        v = self.viewer()
        parent = v.window() if v is not None else None

        dlg = QtWidgets.QDialog(parent)
        dlg.setWindowTitle(call)
        dlg.setModal(True)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)

        editors: dict[str, tuple[QtWidgets.QWidget, callable]] = {}

        for p in params:
            if isinstance(p, dict):
                name = str(p.get("name") or "").strip()
                required = bool(p.get("required") or False)
                ui_raw = str(p.get("uiControl") or "").strip()
                ui = ui_raw.lower()
                schema = p.get("valueSchema")
                desc_raw = p.get("description") or ""
            else:
                try:
                    name = str(p.name or "").strip()
                except Exception:
                    name = ""
                try:
                    required = bool(p.required)
                except Exception:
                    required = False
                try:
                    ui_raw = str(p.uiControl or "").strip()
                    ui = ui_raw.lower()
                except Exception:
                    ui_raw = ""
                    ui = ""
                try:
                    schema = p.valueSchema
                except Exception:
                    schema = None
                try:
                    desc_raw = p.description or ""
                except Exception:
                    desc_raw = ""
            if not name:
                continue
            t = schema_type(schema) if schema is not None else ""
            t = t or ""

            enum_items = self._schema_enum_items(schema)
            lo, hi = self._schema_numeric_range(schema)
            try:
                default_value = schema_default(schema)
            except Exception:
                default_value = None

            label = f"{name} *" if required else name
            tooltip = str(desc_raw or "").strip()

            def _with_tooltip(w: QtWidgets.QWidget) -> QtWidgets.QWidget:
                if tooltip:
                    w.setToolTip(tooltip)
                return w

            pool_field = parse_select_pool(ui_raw)
            if enum_items or pool_field or ui in {"select", "dropdown", "dropbox", "combo", "combobox"}:
                combo = F8OptionCombo()
                if pool_field:
                    node = self._backend_node()
                    items = []
                    if node is not None:
                        try:
                            v = node.get_property(pool_field)
                            if isinstance(v, (list, tuple)):
                                items = [str(x) for x in v]
                        except Exception:
                            items = []
                else:
                    items = list(enum_items)
                combo.set_options(items, labels=items)
                if tooltip:
                    combo.set_context_tooltip(tooltip)
                if default_value is not None:
                    combo.set_value(str(default_value))

                def _get() -> Any:
                    v = combo.value()
                    return None if v is None else str(v)

                editors[name] = (_with_tooltip(combo), _get)
                form.addRow(label, combo)
                continue

            if t == "boolean" or ui in {"switch", "toggle"}:
                sw = F8Switch()
                sw.set_labels("True", "False")
                if default_value is not None:
                    sw.set_value(bool(default_value))

                def _get() -> Any:
                    return bool(sw.value())

                editors[name] = (_with_tooltip(sw), _get)
                form.addRow(label, sw)
                continue

            if t in {"integer", "number"} and ui == "slider":
                is_int = t == "integer"
                bar = F8ValueBar(integer=is_int, minimum=0.0, maximum=1.0)
                bar.set_range(lo, hi)
                if default_value is not None:
                    bar.set_value(default_value)

                def _get() -> Any:
                    v = bar.value()
                    return int(v) if is_int else float(v)

                editors[name] = (_with_tooltip(bar), _get)
                form.addRow(label, bar)
                continue

            if t == "integer" or ui in {"spinbox", "int"}:
                w = QtWidgets.QSpinBox()
                if lo is not None:
                    w.setMinimum(int(lo))
                if hi is not None:
                    w.setMaximum(int(hi))
                if default_value is not None:
                    try:
                        w.setValue(int(default_value))
                    except (TypeError, ValueError):
                        pass

                def _get() -> Any:
                    return int(w.value())

                editors[name] = (_with_tooltip(w), _get)
                form.addRow(label, w)
                continue

            if t == "number" or ui in {"doublespinbox", "float"}:
                w = QtWidgets.QDoubleSpinBox()
                w.setDecimals(6)
                if lo is not None:
                    w.setMinimum(float(lo))
                if hi is not None:
                    w.setMaximum(float(hi))
                if default_value is not None:
                    try:
                        w.setValue(float(default_value))
                    except (TypeError, ValueError):
                        pass

                def _get() -> Any:
                    return float(w.value())

                editors[name] = (_with_tooltip(w), _get)
                form.addRow(label, w)
                continue

            if t in {"object", "array", "any"}:
                w = QtWidgets.QPlainTextEdit()
                w.setMinimumHeight(90)
                if default_value is not None:
                    try:
                        w.setPlainText(json.dumps(default_value, ensure_ascii=False, indent=2))
                    except Exception:
                        w.setPlainText(str(default_value))

                def _get() -> Any:
                    txt = str(w.toPlainText() or "").strip()
                    if not txt:
                        return None
                    try:
                        return json.loads(txt)
                    except Exception:
                        return txt

                editors[name] = (_with_tooltip(w), _get)
                form.addRow(label, w)
                continue

            w = QtWidgets.QLineEdit()
            if default_value is not None:
                w.setText("" if default_value is None else str(default_value))

            def _get() -> Any:
                return str(w.text() or "")

            editors[name] = (_with_tooltip(w), _get)
            form.addRow(label, w)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.addLayout(form)
        layout.addWidget(buttons)

        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        while True:
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                return None
            args: dict[str, Any] = {}
            missing: list[str] = []
            for p in params:
                if isinstance(p, dict):
                    pname = str(p.get("name") or "").strip()
                    required = bool(p.get("required") or False)
                else:
                    try:
                        pname = str(p.name or "").strip()
                    except Exception:
                        pname = ""
                    try:
                        required = bool(p.required)
                    except Exception:
                        required = False
                if not pname or pname not in editors:
                    continue
                _w, getter = editors[pname]
                try:
                    v = getter()
                except Exception:
                    v = None
                # normalize empties
                if isinstance(v, str) and v.strip() == "":
                    v = None
                if required and v is None:
                    missing.append(pname)
                    continue
                if v is not None:
                    args[pname] = v
            if missing:
                QtWidgets.QMessageBox.warning(dlg, "Missing required fields", "Please fill: " + ", ".join(missing))
                continue
            return args

    def _ensure_inline_command_widget(self) -> None:
        self._ensure_bridge_process_hook()
        node = self._backend_node()
        if node is None:
            return
        try:
            spec = node.spec
        except Exception:
            spec = None

        try:
            cmds = list(node.effective_commands() or [])
        except Exception:
            if spec is None:
                cmds = []
            else:
                try:
                    cmds = list(spec.commands or [])
                except Exception:
                    cmds = []

        visible_cmds: list[Any] = []
        for c in cmds:
            if isinstance(c, dict):
                show = bool(c.get("showOnNode") or False)
            else:
                try:
                    show = bool(c.showOnNode)
                except Exception:
                    show = False
            if show:
                visible_cmds.append(c)
        enabled = self._is_service_running()

        # Rebuild only when command list / enabled state changes.
        try:

            def _cmd_name_desc(cmd: Any) -> tuple[str, str]:
                if isinstance(cmd, dict):
                    return str(cmd.get("name") or ""), str(cmd.get("description") or "")
                try:
                    return str(cmd.name or ""), str(cmd.description or "")
                except Exception:
                    return "", ""

            serial = json.dumps(
                {
                    "cmds": [
                        {
                            "name": _cmd_name_desc(c)[0],
                            "desc": _cmd_name_desc(c)[1],
                        }
                        for c in visible_cmds
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        except Exception:
            serial = ""

        # Remove if no commands to show.
        if not visible_cmds:
            if self._cmd_proxy is not None:
                old = None
                try:
                    old = self._cmd_proxy.widget()
                except Exception:
                    old = None
                try:
                    self._cmd_proxy.setWidget(None)
                except RuntimeError:
                    pass
                if old is not None:
                    try:
                        old.setParent(None)
                    except Exception:
                        pass
                    try:
                        old.deleteLater()
                    except Exception:
                        pass
                try:
                    self._cmd_proxy.setParentItem(None)
                    if self.scene() is not None:
                        self.scene().removeItem(self._cmd_proxy)
                except RuntimeError:
                    pass
                self._cmd_proxy = None
                self._cmd_widget = None
                self._cmd_buttons = []
            return

        if self._cmd_proxy is not None and serial and serial == str(self._cmd_serial or ""):
            # Keep enable state in sync (service running can change without spec changes).
            for b in list(self._cmd_buttons or []):
                try:
                    b.setEnabled(bool(enabled))
                except Exception:
                    continue
            return

        self._cmd_serial = serial

        # Build widget (only when changed).
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        w.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        w.setStyleSheet("background: transparent;")

        self._cmd_buttons = []
        for i, c in enumerate(visible_cmds):
            try:
                btn_label = str(c.name or "")
            except Exception:
                btn_label = ""
            b = QtWidgets.QPushButton(btn_label)
            flt = _F8ForceGlobalToolTipFilter(b)
            b.installEventFilter(flt)
            self._tooltip_filters.append(flt)
            b.setMinimumHeight(24)
            b.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            b.setEnabled(bool(enabled))
            b.setStyleSheet(
                """
                QPushButton {
                    color: rgb(235, 235, 235);
                    background: rgba(0, 0, 0, 35);
                    border: 1px solid rgba(120, 200, 255, 85);
                    border-radius: 6px;
                    padding: 6px 10px;
                    text-align: center;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background: rgba(120, 200, 255, 22);
                    border-color: rgba(120, 200, 255, 140);
                }
                QPushButton:pressed {
                    background: rgba(120, 200, 255, 35);
                    border-color: rgba(120, 200, 255, 160);
                }
                QPushButton:disabled {
                    color: rgba(235, 235, 235, 110);
                    background: rgba(0, 0, 0, 20);
                    border-color: rgba(255, 255, 255, 18);
                }
                """
            )
            try:
                desc = str(c.description or "").strip()
            except Exception:
                desc = ""
            if not enabled:
                b.setToolTip((desc + "\n" if desc else "") + "Service not running")
            elif desc:
                b.setToolTip(desc)
            b.clicked.connect(lambda _=False, _c=c: self._invoke_command(_c))  # type: ignore[attr-defined]
            lay.addWidget(b)
            self._cmd_buttons.append(b)

        if self._cmd_proxy is None:
            proxy = QtWidgets.QGraphicsProxyWidget(self)
            proxy.setWidget(w)
            proxy.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
            self._cmd_proxy = proxy
        else:
            old = None
            try:
                old = self._cmd_proxy.widget()
            except Exception:
                old = None
            self._cmd_proxy.setWidget(w)
            if old is not None and old is not w:
                try:
                    old.setParent(None)
                except Exception:
                    pass
                try:
                    old.deleteLater()
                except Exception:
                    pass
        self._cmd_widget = w

    def _on_graph_property_changed(self, node: Any, name: str, value: Any) -> None:
        """
        Keep inline state widgets in sync with NodeGraphQt properties.

        The inspector already tracks these through NodeGraphQt's own property
        widgets; since our inline widgets are custom QWidgets, we mirror updates
        here to get the same "two-way binding" behavior.
        """
        try:
            if str(node.id or "") != str(self.id or ""):
                return
        except Exception:
            return
        key = str(name or "").strip()
        if not key:
            return
        updater = self._state_inline_updaters.get(key)
        if not updater:
            # may still be a pool update affecting dependent option widgets.
            self._refresh_option_pool_for_changed_field(key)
            return
        try:
            updater(value)
        except Exception:
            try:
                node_id = str(self.id or "")
            except Exception:
                node_id = ""
            logger.exception("inline state updater failed nodeId=%s key=%s", node_id, key)
        self._refresh_option_pool_for_changed_field(key)

    def _refresh_option_pool_for_changed_field(self, changed_field: str) -> None:
        """
        If `changed_field` is used as an option-pool, refresh all dependent option controls.
        """
        pool = str(changed_field or "").strip()
        if not pool:
            return
        if pool not in set(self._state_inline_option_pools.values()):
            return
        node = self._backend_node()
        if node is None:
            return
        try:
            pool_value = node.get_property(pool)
        except Exception:
            pool_value = None
        if isinstance(pool_value, (list, tuple)):
            items = [str(x) for x in pool_value]
        else:
            items = []
        for field, pool_name in list(self._state_inline_option_pools.items()):
            if pool_name != pool:
                continue
            ctrl = self._state_inline_controls.get(field)
            if not isinstance(ctrl, (F8OptionCombo, F8MultiSelect)):
                continue
            try:
                cur = ctrl.value()
                ctrl.set_options(items, labels=items)
                ctrl.set_value(cur)
            except Exception:
                continue

    def _on_state_toggle(self, name: str, expanded: bool) -> None:
        name = str(name)
        # When collapsing, node height shrinks. With partial viewport updates this
        # can leave stale pixels from the old bounding rect. Track the old rect
        # and explicitly request a scene update covering both old+new bounds.
        old_scene_rect = None
        try:
            old_scene_rect = self.mapToScene(self.boundingRect()).boundingRect()
        except RuntimeError:
            old_scene_rect = None

        self._state_inline_expanded[name] = bool(expanded)
        # Persist expand state in the node's UI overrides so it survives reloads.
        node = self._backend_node()
        if node is not None:
            try:
                ui = dict(node.ui_overrides() or {})
                store = ui.get("stateInlineExpanded")
                if not isinstance(store, dict):
                    store = {}
                store[name] = bool(expanded)
                ui["stateInlineExpanded"] = store
                node.set_ui_overrides(ui, rebuild=False)
            except AttributeError:
                logger.exception("node missing ui_overrides/set_ui_overrides; cannot persist expand state")
        btn = self._state_inline_toggles.get(str(name))
        if btn is not None:
            try:
                btn.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
            except RuntimeError:
                pass
        body = self._state_inline_bodies.get(str(name))
        if body is not None:
            try:
                body.setVisible(bool(expanded))
            except RuntimeError:
                pass

        def _redraw_and_invalidate() -> None:
            self.draw_node()
            new_scene_rect = self.mapToScene(self.boundingRect()).boundingRect()
            r = new_scene_rect
            if old_scene_rect is not None:
                r = old_scene_rect.united(new_scene_rect)
            r = r.adjusted(-6, -6, 6, 6)
            sc = self.scene()
            if sc is not None:
                sc.update(r)
            v = self.viewer()
            if v is not None:
                v.viewport().update()

        try:
            QtCore.QTimer.singleShot(0, _redraw_and_invalidate)
        except RuntimeError:
            _redraw_and_invalidate()

    @staticmethod
    def _port_group(name: str) -> str:
        n = str(name or "")
        if n.startswith("[E]") or n.endswith("[E]"):
            return "exec"
        if n.startswith("[D]") or n.endswith("[D]"):
            return "data"
        if n.startswith("[S]") or n.endswith("[S]"):
            return "state"
        return "other"

    @staticmethod
    def _display_port_label(name: str, *, max_chars: int | None = None) -> str:
        """
        Display-friendly label for port text items.

        Strip `[E]/[D]/[S]` markers (color already conveys kind), and optionally
        elide to keep the state-field area compact.
        """
        n = str(name or "")
        if n.startswith("[E]"):
            n = n[3:]
        elif n.endswith("[E]"):
            n = n[:-3]
        elif n.startswith("[D]"):
            n = n[3:]
        elif n.endswith("[D]"):
            n = n[:-3]
        elif n.startswith("[S]"):
            n = n[3:]
        elif n.endswith("[S]"):
            n = n[:-3]
        n = n.strip()
        if max_chars is not None and max_chars > 0 and len(n) > max_chars:
            return n[: max(1, max_chars - 1)] + "..."
        return n

    @staticmethod
    def _schema_enum_items(value_schema: Any) -> list[str]:
        if value_schema is None:
            return []
        try:
            root = value_schema.root
            enum_items = list(root.enum or [])
        except Exception:
            enum_items = []
        return [str(x) for x in enum_items]

    @staticmethod
    def _schema_numeric_range(value_schema: Any) -> tuple[float | None, float | None]:
        if value_schema is None:
            return None, None
        try:
            root = value_schema.root
        except Exception:
            return None, None
        mins: list[float] = []
        maxs: list[float] = []
        try:
            if root.minimum is not None:
                mins.append(float(root.minimum))
        except Exception:
            pass
        try:
            if root.exclusiveMinimum is not None:
                mins.append(float(root.exclusiveMinimum))
        except Exception:
            pass
        try:
            if root.maximum is not None:
                maxs.append(float(root.maximum))
        except Exception:
            pass
        try:
            if root.exclusiveMaximum is not None:
                maxs.append(float(root.exclusiveMaximum))
        except Exception:
            pass
        lo = min(mins) if mins else None
        hi = max(maxs) if maxs else None
        return lo, hi

    def _make_state_inline_control(self, state_field: _StateFieldInfo) -> QtWidgets.QWidget:
        name = state_field.name
        ui_raw = state_field.ui_control
        ui = str(ui_raw or "").strip().lower()
        schema = state_field.value_schema
        access_s = state_field.access_str
        t = (schema_type(schema) or "") if schema is not None else ""

        enum_items = self._schema_enum_items(schema)
        lo, hi = self._schema_numeric_range(schema)
        select_pool_field = parse_select_pool(ui_raw)
        multiselect_pool_field = parse_multiselect_pool(ui_raw)
        field_tooltip = state_field.tooltip if state_field.tooltip != name else ""

        def _common_style(w: QtWidgets.QWidget) -> None:
            # Make controls readable on dark node themes.
            w.setStyleSheet(
                """
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {
                    color: rgb(235, 235, 235);
                    background: rgba(0, 0, 0, 45);
                    border: 1px solid rgba(255, 255, 255, 55);
                    border-radius: 3px;
                    padding: 1px 4px;
                }
                QPlainTextEdit, QTextEdit {
                    selection-background-color: rgb(80, 130, 180);
                }
                QComboBox::drop-down { border: 0px; }
                QComboBox QAbstractItemView {
                    color: rgb(235, 235, 235);
                    background: rgb(35, 35, 35);
                    selection-background-color: rgb(80, 130, 180);
                }
                QCheckBox { color: rgb(235, 235, 235); }
                QCheckBox::indicator {
                    width: 13px;
                    height: 13px;
                    border: 1px solid rgba(255, 255, 255, 90);
                    background: rgba(0, 0, 0, 35);
                    border-radius: 2px;
                }
                QCheckBox::indicator:checked { background: rgba(120, 200, 255, 90); }
                """
            )

        def _set_node_value(value: Any, *, push_undo: bool) -> None:
            node = self._backend_node()
            if node is None or not name:
                return
            try:
                node.set_property(name, value, push_undo=push_undo)
            except TypeError:
                node.set_property(name, value)

        def _get_node_value() -> Any:
            node = self._backend_node()
            if node is None or not name:
                return None
            try:
                return node.get_property(name)
            except KeyError:
                return None

        def _pool_items(pool_field: str | None) -> list[str]:
            if not pool_field:
                return []
            node = self._backend_node()
            if node is None:
                return []
            try:
                v = node.get_property(pool_field)
            except Exception:
                return []
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
            # Allow pools stored as JSON strings (eg. "[]", ["a","b"]).
            if isinstance(v, str):
                try:
                    import json as _json

                    parsed = _json.loads(v)
                except Exception:
                    return []
                if isinstance(parsed, (list, tuple)):
                    out: list[str] = []
                    for x in parsed:
                        if isinstance(x, str):
                            s = x.strip()
                            if s:
                                out.append(s)
                            continue
                        if isinstance(x, dict):
                            # Accept [{id,name,...}] and use id.
                            s = str(x.get("id") or "").strip()
                            if s:
                                out.append(s)
                            continue
                        s = str(x).strip()
                        if s:
                            out.append(s)
                    return out
            return []

        # Create control.
        read_only = access_s == "ro" or self._inline_state_input_is_connected(name)

        if ui in {"wrapline"}:

            class _InlineWrapLineEdit(QtWidgets.QPlainTextEdit):
                def __init__(self, parent: QtWidgets.QWidget | None = None):
                    super().__init__(parent)
                    self._prev = ""

                @staticmethod
                def _normalize(s: str) -> str:
                    if "\n" not in s and "\r" not in s:
                        return s.strip()
                    parts = [p.strip() for p in s.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
                    return " ".join([p for p in parts if p]).strip()

                def focusInEvent(self, event):  # type: ignore[override]
                    super().focusInEvent(event)
                    self._prev = str(self.toPlainText() or "")

                def focusOutEvent(self, event):  # type: ignore[override]
                    super().focusOutEvent(event)
                    txt_raw = str(self.toPlainText() or "")
                    txt = self._normalize(txt_raw)
                    if txt != txt_raw:
                        with QtCore.QSignalBlocker(self):
                            self.setPlainText(txt)
                    if txt != self._prev:
                        self._prev = txt
                        _set_node_value(txt, push_undo=True)

                def keyPressEvent(self, event):  # type: ignore[override]
                    if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                        # Commit on enter; do not insert newlines.
                        txt_raw = str(self.toPlainText() or "")
                        txt = self._normalize(txt_raw)
                        if txt != txt_raw:
                            with QtCore.QSignalBlocker(self):
                                self.setPlainText(txt)
                        if txt != self._prev:
                            self._prev = txt
                            _set_node_value(txt, push_undo=True)
                        self.clearFocus()
                        event.accept()
                        return
                    super().keyPressEvent(event)

                def insertFromMimeData(self, source: QtCore.QMimeData) -> None:  # type: ignore[override]
                    if source is None or not source.hasText():
                        return super().insertFromMimeData(source)
                    txt = self._normalize(str(source.text() or ""))
                    if txt:
                        self.textCursor().insertText(txt)
                    return

            edit = _InlineWrapLineEdit()
            edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
            edit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            edit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            edit.setTabStopDistance(4 * edit.fontMetrics().horizontalAdvance(" "))
            try:
                font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
                edit.setFont(font)
            except Exception:
                pass
            edit.setMinimumWidth(160)
            edit.setMinimumHeight(38)
            edit.setMaximumHeight(64)
            _common_style(edit)
            edit.document().setDocumentMargin(4.0)
            if field_tooltip:
                edit.setToolTip(field_tooltip)

            def _apply_value(v: Any) -> None:
                s = "" if v is None else str(v)
                s2 = _InlineWrapLineEdit._normalize(s)
                with QtCore.QSignalBlocker(edit):
                    edit.setPlainText(s2)
                try:
                    edit._prev = s2  # type: ignore[attr-defined]
                except Exception:
                    pass

            _apply_value(_get_node_value())
            self._state_inline_updaters[name] = _apply_value
            if read_only:
                edit.setReadOnly(True)
            return edit

        if ui in {"code_inline", "multiline"}:

            class _InlineExprEdit(QtWidgets.QPlainTextEdit):
                def __init__(self, parent: QtWidgets.QWidget | None = None):
                    super().__init__(parent)
                    self._prev = ""

                def focusInEvent(self, event):  # type: ignore[override]
                    super().focusInEvent(event)
                    self._prev = str(self.toPlainText() or "")

                def focusOutEvent(self, event):  # type: ignore[override]
                    super().focusOutEvent(event)
                    txt = str(self.toPlainText() or "")
                    if txt != self._prev:
                        self._prev = txt
                        _set_node_value(txt, push_undo=True)

                def keyPressEvent(self, event):  # type: ignore[override]
                    if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter) and bool(
                        event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
                    ):
                        txt = str(self.toPlainText() or "")
                        if txt != self._prev:
                            self._prev = txt
                            _set_node_value(txt, push_undo=True)
                        event.accept()
                        return
                    super().keyPressEvent(event)

            edit = _InlineExprEdit()
            edit.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
            edit.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            edit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            edit.setTabStopDistance(4 * edit.fontMetrics().horizontalAdvance(" "))
            try:
                font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
                edit.setFont(font)
            except Exception:
                pass
            edit.setMinimumWidth(160)
            edit.setMinimumHeight(44)
            edit.setMaximumHeight(88)
            _common_style(edit)
            edit.document().setDocumentMargin(4.0)
            if field_tooltip:
                edit.setToolTip(field_tooltip)

            def _apply_value(v: Any) -> None:
                s = "" if v is None else str(v)
                with QtCore.QSignalBlocker(edit):
                    edit.setPlainText(s)
                try:
                    edit._prev = s  # type: ignore[attr-defined]
                except Exception:
                    pass

            _apply_value(_get_node_value())
            self._state_inline_updaters[name] = _apply_value
            if read_only:
                edit.setReadOnly(True)
            return edit

        if ui in {"code"}:
            btn = QtWidgets.QToolButton()
            btn.setAutoRaise(True)
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            btn.setText("Edit...")
            try:
                btn.setIcon(qta.icon("fa5s.code", color="white"))
            except Exception:
                pass
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            if field_tooltip:
                btn.setToolTip(field_tooltip)

            def _apply_value(v: Any) -> None:
                s = "" if v is None else str(v)
                n = len(s.splitlines()) if s else 0
                tip = field_tooltip or ""
                if n:
                    tip2 = f"{n} line" if n == 1 else f"{n} lines"
                    btn.setToolTip((tip + "\n" if tip else "") + tip2)

            def _on_click() -> None:
                current = _get_node_value()

                def _on_saved(updated: str) -> None:
                    _set_node_value(updated, push_undo=True)

                try:
                    dlg = open_code_editor_window(
                        None,
                        title=f"{self.name} - {state_field.label}",
                        code="" if current is None else str(current),
                        language=state_field.ui_language or "plaintext",
                        on_saved=_on_saved,
                    )
                    self._open_code_editors.append(dlg)

                    def _cleanup() -> None:
                        alive: list[QtWidgets.QDialog] = []
                        for w in self._open_code_editors:
                            if w is None:
                                continue
                            try:
                                _ = w.isVisible()
                                alive.append(w)
                            except RuntimeError:
                                continue
                        self._open_code_editors = alive

                    dlg.destroyed.connect(_cleanup)  # type: ignore[attr-defined]
                except Exception:
                    updated = open_code_editor_dialog(
                        None,
                        title=f"{self.name} - {state_field.label}",
                        code="" if current is None else str(current),
                        language=state_field.ui_language or "plaintext",
                    )
                    if updated is None:
                        return
                    _set_node_value(updated, push_undo=True)

            btn.clicked.connect(_on_click)  # type: ignore[attr-defined]
            _apply_value(_get_node_value())
            self._state_inline_updaters[name] = _apply_value
            if read_only:
                btn.setDisabled(True)
            return btn

        is_image_b64 = t == "string" and (ui in {"image", "image_b64", "img"} or "b64" in name.lower())
        if is_image_b64:
            img = F8ImageB64Editor()

            def _apply_value(v: Any) -> None:
                img.set_value("" if v is None else str(v))

            img.valueChanged.connect(lambda v: _set_node_value(str(v or ""), push_undo=True))  # type: ignore[attr-defined]
            _apply_value(_get_node_value())
            self._state_inline_updaters[name] = _apply_value
            if read_only:
                img.set_disabled(True)
            return img

        if multiselect_pool_field or ui in {"multiselect", "multi_select", "multi-select"}:
            multi = F8MultiSelect()
            if field_tooltip:
                multi.set_context_tooltip(field_tooltip)

            items = _pool_items(multiselect_pool_field) if multiselect_pool_field else list(enum_items)
            multi.set_options(items, labels=items)

            def _apply_value(v: Any) -> None:
                multi.set_value(v)

            multi.valueChanged.connect(lambda v: _set_node_value(list(v or []), push_undo=True))  # type: ignore[attr-defined]
            _apply_value(_get_node_value())
            self._state_inline_updaters[name] = _apply_value
            if multiselect_pool_field:
                self._state_inline_option_pools[name] = multiselect_pool_field
            if read_only:
                multi.set_read_only(True)
            return multi

        if enum_items or select_pool_field or ui in {"select", "dropdown", "dropbox", "combo", "combobox"}:
            combo = F8OptionCombo()
            _common_style(combo)

            items = _pool_items(select_pool_field) if select_pool_field else list(enum_items)
            combo.set_options(items, labels=items)
            if field_tooltip:
                combo.set_context_tooltip(field_tooltip)

            def _apply_value(v: Any) -> None:
                combo.set_value("" if v is None else str(v))

            combo.valueChanged.connect(  # type: ignore[attr-defined]
                lambda v: _set_node_value("" if v is None else str(v), push_undo=True)
            )
            _apply_value(_get_node_value())
            self._state_inline_updaters[name] = _apply_value
            if select_pool_field:
                self._state_inline_option_pools[name] = select_pool_field
            if read_only:
                combo.set_read_only(True)
            return combo

        if t == "boolean" or ui in {"switch", "toggle"}:
            sw = F8Switch()
            sw.set_labels("True", "False")
            if field_tooltip:
                sw.setToolTip(field_tooltip)

            def _apply_value(v: Any) -> None:
                with QtCore.QSignalBlocker(sw):
                    sw.set_value(bool(v) if v is not None else False)

            sw.valueChanged.connect(lambda v: _set_node_value(bool(v), push_undo=True))  # type: ignore[attr-defined]
            _apply_value(_get_node_value())
            self._state_inline_updaters[name] = _apply_value
            if read_only:
                sw.setDisabled(True)
            return sw

        if t in {"integer", "number"} and ui == "slider":
            is_int = t == "integer"
            bar = F8ValueBar(integer=is_int, minimum=0.0, maximum=1.0)
            bar.set_range(lo, hi)

            def _apply_value(v: Any) -> None:
                bar.set_value(v)

            bar.valueChanging.connect(lambda v: _set_node_value(v, push_undo=False))  # type: ignore[attr-defined]
            bar.valueCommitted.connect(lambda v: _set_node_value(v, push_undo=True))  # type: ignore[attr-defined]
            _apply_value(_get_node_value())
            self._state_inline_updaters[name] = _apply_value
            if read_only:
                bar.setDisabled(True)
            return bar

        if t == "integer" or ui in {"spinbox", "int"}:
            line = QtWidgets.QLineEdit()
            _common_style(line)
            line.setMinimumWidth(90)
            vmin = int(lo) if lo is not None else -(2**31)
            vmax = int(hi) if hi is not None else (2**31 - 1)
            line.setValidator(QtGui.QIntValidator(vmin, vmax, line))

            def _apply_value(v: Any) -> None:
                s = "" if v is None else str(int(v))
                with QtCore.QSignalBlocker(line):
                    line.setText(s)

            def _commit() -> None:
                txt = str(line.text() or "").strip()
                if not txt:
                    _set_node_value(None, push_undo=True)
                    return
                try:
                    _set_node_value(int(txt), push_undo=True)
                except ValueError:
                    return

            _apply_value(_get_node_value())
            self._state_inline_updaters[name] = _apply_value
            if read_only:
                line.setReadOnly(True)
            else:
                line.editingFinished.connect(_commit)  # type: ignore[attr-defined]
            return line

        if t == "number" or ui in {"doublespinbox", "float"}:
            line = QtWidgets.QLineEdit()
            _common_style(line)
            line.setMinimumWidth(90)
            vmin = float(lo) if lo is not None else -1.0e18
            vmax = float(hi) if hi is not None else 1.0e18
            dv = QtGui.QDoubleValidator(vmin, vmax, 6, line)
            try:
                dv.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
            except Exception:
                pass
            line.setValidator(dv)

            def _apply_value(v: Any) -> None:
                s = "" if v is None else str(float(v))
                with QtCore.QSignalBlocker(line):
                    line.setText(s)

            def _commit() -> None:
                txt = str(line.text() or "").strip()
                if not txt:
                    _set_node_value(None, push_undo=True)
                    return
                try:
                    _set_node_value(float(txt), push_undo=True)
                except ValueError:
                    return

            _apply_value(_get_node_value())
            self._state_inline_updaters[name] = _apply_value
            if read_only:
                line.setReadOnly(True)
            else:
                line.editingFinished.connect(_commit)  # type: ignore[attr-defined]
            return line

        # default: text input.
        line = QtWidgets.QLineEdit()
        line.setMinimumWidth(90)
        _common_style(line)

        def _apply_value(v: Any) -> None:
            s = "" if v is None else str(v)
            with QtCore.QSignalBlocker(line):
                line.setText(s)

        _apply_value(_get_node_value())
        self._state_inline_updaters[name] = _apply_value
        if read_only:
            line.setReadOnly(True)
        else:
            line.editingFinished.connect(lambda: _set_node_value(line.text(), push_undo=True))
        return line

    def _ensure_inline_state_widgets(self) -> None:
        self._ensure_graph_property_hook()
        node = self._backend_node()
        if node is None:
            return
        try:
            fields = list(node.effective_state_fields() or [])
        except Exception:
            try:
                spec = node.spec
            except Exception:
                spec = None
            if spec is None:
                fields = []
            else:
                try:
                    fields = list(spec.stateFields or [])
                except Exception:
                    fields = []

        show: list[_StateFieldInfo] = []
        for f in fields:
            info = _state_field_info(f)
            if info is None or not info.show_on_node:
                continue
            show.append(info)

        desired = [info.name for info in show]

        # delete stale widgets.
        for n in list(self._state_inline_proxies.keys()):
            if n in desired:
                continue
            proxy = self._state_inline_proxies.pop(n, None)
            self._state_inline_controls.pop(n, None)
            self._state_inline_updaters.pop(n, None)
            self._state_inline_toggles.pop(n, None)
            self._state_inline_headers.pop(n, None)
            self._state_inline_bodies.pop(n, None)
            self._state_inline_expanded.pop(n, None)
            self._state_inline_option_pools.pop(n, None)
            self._state_inline_ctrl_serial.pop(n, None)
            if proxy is None:
                continue
            old = None
            try:
                old = proxy.widget()
            except Exception:
                old = None
            try:
                proxy.setWidget(None)
            except RuntimeError:
                pass
            if old is not None:
                try:
                    old.setParent(None)
                except Exception:
                    pass
                try:
                    old.deleteLater()
                except Exception:
                    pass
            try:
                proxy.setParentItem(None)
                if self.scene() is not None:
                    self.scene().removeItem(proxy)
            except RuntimeError:
                pass

        def _ctrl_serial(info: _StateFieldInfo) -> str:
            """
            Signature for deciding when the control widget must be rebuilt.
            (Exclude label/description; those can be updated in-place.)
            """
            try:
                vs = info.value_schema
                enum_items = self._schema_enum_items(vs)
                return json.dumps(
                    {
                        "access": info.access_str,
                        "required": info.required,
                        "uiControl": info.ui_control,
                        "uiLanguage": info.ui_language,
                        "schemaType": str(schema_type(vs) or ""),
                        "enum": [str(x) for x in enum_items],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            except Exception:
                return ""

        for info in show:
            # Always keep label/tooltip up to date without rebuilding.
            n = info.name
            label = info.label or n
            tip = info.tooltip or n
            btn_existing = self._state_inline_toggles.get(n)
            if btn_existing is not None:
                try:
                    btn_existing.setFullText(label)
                except Exception:
                    pass
                try:
                    btn_existing.setToolTip(tip)
                except Exception:
                    pass

            ctrl_sig = _ctrl_serial(info)
            if n in self._state_inline_proxies and ctrl_sig and ctrl_sig == self._state_inline_ctrl_serial.get(n, ""):
                continue

            # Default collapsed; restore persisted expand state from ui overrides.
            expanded = False
            ui = node.ui_overrides() or {}
            store = ui.get("stateInlineExpanded") if isinstance(ui, dict) else None
            if isinstance(store, dict) and n in store:
                expanded = bool(store.get(n))
            expanded = bool(self._state_inline_expanded.get(n, expanded))
            control = self._make_state_inline_control(info)

            # Header: toggle button (state name).
            header = QtWidgets.QWidget()
            header_lay = QtWidgets.QHBoxLayout(header)
            header_lay.setContentsMargins(0, 0, 0, 0)
            header_lay.setSpacing(6)
            header.setAttribute(QtCore.Qt.WA_StyledBackground, True)
            header.setStyleSheet("background: transparent;")

            btn = _F8ElideToolButton()
            btn.setCheckable(True)
            btn.setChecked(expanded)
            btn.setAutoRaise(True)
            btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            btn.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)

            btn.setFullText(label)
            btn.setToolTip(tip)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            btn.setStyleSheet(
                """
                QToolButton {
                    color: rgb(235, 235, 235);
                    background: transparent;
                    border: 1px solid rgba(255, 255, 255, 18);
                    border-radius: 4px;
                    padding: 2px 8px;
                    text-align: left;
                }
                QToolButton:hover { background: transparent; }
                QToolButton:checked { background: transparent; }
                """
            )

            header_lay.addWidget(btn, 1)

            # Body: control widget (collapsed by default).
            body = QtWidgets.QWidget()
            body_lay = QtWidgets.QVBoxLayout(body)
            body_lay.setContentsMargins(8, 0, 8, 6)
            body_lay.setSpacing(0)
            body_lay.addWidget(control)
            body.setVisible(expanded)
            body.setStyleSheet(
                """
                QWidget {
                    background: transparent;
                    border: 0px;
                }
                """
            )

            panel = QtWidgets.QWidget()
            panel_lay = QtWidgets.QVBoxLayout(panel)
            panel_lay.setContentsMargins(0, 0, 0, 0)
            panel_lay.setSpacing(0)
            panel_lay.addWidget(header)
            panel_lay.addWidget(body)
            panel.setProperty("_f8_state_panel", True)
            panel.setAttribute(QtCore.Qt.WA_StyledBackground, True)
            panel.setStyleSheet("background: transparent;")

            # Connect toggle.
            btn.toggled.connect(lambda v, _n=n: self._on_state_toggle(_n, bool(v)))  # type: ignore[attr-defined]
            btn.pressed.connect(self._select_node_from_embedded_widget)  # type: ignore[attr-defined]

            # Install/replace proxy.
            proxy = self._state_inline_proxies.get(n)
            if proxy is None:
                proxy = QtWidgets.QGraphicsProxyWidget(self)
                proxy.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
                self._state_inline_proxies[n] = proxy

            old = None
            try:
                old = proxy.widget()
            except Exception:
                old = None
            proxy.setWidget(panel)
            if old is not None and old is not panel:
                try:
                    old.setParent(None)
                except Exception:
                    pass
                try:
                    old.deleteLater()
                except Exception:
                    pass

            self._state_inline_controls[n] = control
            self._state_inline_toggles[n] = btn
            self._state_inline_headers[n] = header
            self._state_inline_bodies[n] = body
            self._state_inline_expanded[n] = expanded
            if ctrl_sig:
                self._state_inline_ctrl_serial[n] = ctrl_sig

    def post_init(self, viewer=None, pos=None):
        """
        Called after node has been added into the scene.

        Args:
            viewer (NodeGraphQt.widgets.viewer.NodeViewer): main viewer
            pos (tuple): the cursor pos if node is called with tab search.
        """
        if self.layout_direction == LayoutDirectionEnum.VERTICAL.value:
            font = QtGui.QFont()
            font.setPointSize(15)
            self.text_item.setFont(font)

            # hide port text items for vertical layout.
            if self.layout_direction is LayoutDirectionEnum.VERTICAL.value:
                for text_item in self._input_items.values():
                    text_item.setVisible(False)
                for text_item in self._output_items.values():
                    text_item.setVisible(False)

    def _paint_horizontal(self, painter, option, widget):
        painter.save()
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtCore.Qt.NoBrush)

        # base background.
        margin = 1.0
        rect = self.boundingRect()
        rect = QtCore.QRectF(
            rect.left() + margin, rect.top() + margin, rect.width() - (margin * 2), rect.height() - (margin * 2)
        )

        radius = 4.0
        painter.setBrush(QtGui.QColor(*self.color))
        painter.drawRoundedRect(rect, radius, radius)

        # light overlay on background when selected.
        if self.selected:
            painter.setBrush(QtGui.QColor(*NodeEnum.SELECTED_COLOR.value))
            painter.drawRoundedRect(rect, radius, radius)

        # node name background.
        padding = 3.0, 2.0
        text_rect = self._text_item.boundingRect()
        text_rect = QtCore.QRectF(
            text_rect.x() + padding[0],
            rect.y() + padding[1],
            rect.width() - padding[0] - margin,
            text_rect.height() - (padding[1] * 2),
        )
        if self.selected:
            painter.setBrush(QtGui.QColor(*NodeEnum.SELECTED_COLOR.value))
        else:
            painter.setBrush(QtGui.QColor(0, 0, 0, 80))
        painter.drawRoundedRect(text_rect, 3.0, 3.0)

        # node border
        if self.selected:
            border_width = 1.2
            border_color = QtGui.QColor(*NodeEnum.SELECTED_BORDER_COLOR.value)
        else:
            border_width = 0.8
            border_color = QtGui.QColor(*self.border_color)

        border_rect = QtCore.QRectF(rect.left(), rect.top(), rect.width(), rect.height())

        pen = QtGui.QPen(border_color, border_width)
        v = self._viewer_safe()
        zoom = None
        try:
            zoom = float(v.get_zoom()) if v is not None else None
        except Exception:
            zoom = None
        pen.setCosmetic(bool(zoom is not None and zoom < 0.0))
        path = QtGui.QPainterPath()
        path.addRoundedRect(border_rect, radius, radius)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(pen)
        painter.drawPath(path)

        painter.restore()

    def _paint_vertical(self, painter, option, widget):
        painter.save()
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtCore.Qt.NoBrush)

        # base background.
        margin = 1.0
        rect = self.boundingRect()
        rect = QtCore.QRectF(
            rect.left() + margin, rect.top() + margin, rect.width() - (margin * 2), rect.height() - (margin * 2)
        )

        radius = 4.0
        painter.setBrush(QtGui.QColor(*self.color))
        painter.drawRoundedRect(rect, radius, radius)

        # light overlay on background when selected.
        if self.selected:
            painter.setBrush(QtGui.QColor(*NodeEnum.SELECTED_COLOR.value))
            painter.drawRoundedRect(rect, radius, radius)

        # top & bottom edge background.
        padding = 2.0
        height = 10
        if self.selected:
            painter.setBrush(QtGui.QColor(*NodeEnum.SELECTED_COLOR.value))
        else:
            painter.setBrush(QtGui.QColor(0, 0, 0, 80))
        for y in [rect.y() + padding, rect.height() - height - 1]:
            edge_rect = QtCore.QRectF(rect.x() + padding, y, rect.width() - (padding * 2), height)
            painter.drawRoundedRect(edge_rect, 3.0, 3.0)

        # node border
        border_width = 0.8
        border_color = QtGui.QColor(*self.border_color)
        if self.selected:
            border_width = 1.2
            border_color = QtGui.QColor(*NodeEnum.SELECTED_BORDER_COLOR.value)
        border_rect = QtCore.QRectF(rect.left(), rect.top(), rect.width(), rect.height())

        pen = QtGui.QPen(border_color, border_width)
        v = self._viewer_safe()
        zoom = None
        try:
            zoom = float(v.get_zoom()) if v is not None else None
        except Exception:
            zoom = None
        pen.setCosmetic(bool(zoom is not None and zoom < 0.0))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(pen)
        painter.drawRoundedRect(border_rect, radius, radius)

        painter.restore()

    def paint(self, painter, option, widget):
        """
        Draws the node base not the ports.

        Args:
            painter (QtGui.QPainter): painter used for drawing the item.
            option (QtGui.QStyleOptionGraphicsItem):
                used to describe the parameters needed to draw.
            widget (QtWidgets.QWidget): not used.
        """
        self.auto_switch_mode()
        if self.layout_direction is LayoutDirectionEnum.HORIZONTAL.value:
            self._paint_horizontal(painter, option, widget)
        elif self.layout_direction is LayoutDirectionEnum.VERTICAL.value:
            self._paint_vertical(painter, option, widget)
        else:
            raise RuntimeError("Node graph layout direction not valid!")

    def mousePressEvent(self, event):
        """
        Re-implemented to ignore event if LMB is over port collision area.

        Args:
            event (QtWidgets.QGraphicsSceneMouseEvent): mouse event.
        """
        if event.button() == QtCore.Qt.LeftButton:
            for p in self._input_items.keys():
                if p.hovered:
                    event.ignore()
                    return
            for p in self._output_items.keys():
                if p.hovered:
                    event.ignore()
                    return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Re-implemented to ignore event if Alt modifier is pressed.

        Args:
            event (QtWidgets.QGraphicsSceneMouseEvent): mouse event.
        """
        if event.modifiers() == QtCore.Qt.AltModifier:
            event.ignore()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """
        Re-implemented to emit "node_double_clicked" signal.

        Args:
            event (QtWidgets.QGraphicsSceneMouseEvent): mouse event.
        """
        if event.button() == QtCore.Qt.LeftButton:
            if not self.disabled:
                # enable text item edit mode.
                items = self.scene().items(event.scenePos())
                if self._text_item in items:
                    self._text_item.set_editable(True)
                    self._text_item.setFocus()
                    event.ignore()
                    return

            viewer = self.viewer()
            if viewer:
                viewer.node_double_clicked.emit(self.id)
        super().mouseDoubleClickEvent(event)

    def _tooltip_disable(self, state):
        """
        Updates the node tooltip when the node is enabled/disabled.

        Args:
            state (bool): node disable state.
        """
        tooltip = "<b>{}</b>".format(self.name)
        if state:
            tooltip += ' <font color="red"><b>(DISABLED)</b></font>'
        tooltip += "<br/>{}<br/>".format(self.type_)
        self.setToolTip(tooltip)

    def _set_base_size(self, add_w=0.0, add_h=0.0):
        """
        Sets the initial base size for the node.

        Args:
            add_w (float): add additional width.
            add_h (float): add additional height.
        """
        old_rect = None
        try:
            old_rect = self.boundingRect()
        except Exception:
            old_rect = None

        w, h = self.calc_size(add_w, add_h)
        if w < NodeEnum.WIDTH.value:
            w = NodeEnum.WIDTH.value
        if h < NodeEnum.HEIGHT.value:
            h = NodeEnum.HEIGHT.value

        changed = True
        try:
            changed = bool(abs(float(w) - float(self._width)) > 0.01 or abs(float(h) - float(self._height)) > 0.01)
        except Exception:
            changed = True

        if changed:    
            self.prepareGeometryChange()
        
        self._width, self._height = float(w), float(h)

        if not changed or old_rect is None:
            return

        new_rect = self.boundingRect()

        old_scene = self.mapToScene(old_rect).boundingRect()
        new_scene = self.mapToScene(new_rect).boundingRect()
        dirty = old_scene.united(new_scene).adjusted(-6, -6, 6, 6)
        sc = self.scene()
        if sc is not None:
            sc.update(dirty)
        v = self.viewer()
        if v is not None:
            v.viewport().update()

    def _set_text_color(self, color):
        """
        set text color.

        Args:
            color (tuple): color value in (r, g, b, a).
        """
        text_color = QtGui.QColor(*color)
        for port, text in self._input_items.items():
            text.setDefaultTextColor(text_color)
        for port, text in self._output_items.items():
            text.setDefaultTextColor(text_color)
        self._text_item.setDefaultTextColor(text_color)

    def activate_pipes(self):
        """
        active pipe color.
        """
        ports = self.inputs + self.outputs
        for port in ports:
            for pipe in port.connected_pipes:
                pipe.activate()

    def highlight_pipes(self):
        """
        Highlight pipe color.
        """
        ports = self.inputs + self.outputs
        for port in ports:
            for pipe in port.connected_pipes:
                pipe.highlight()

    def reset_pipes(self):
        """
        Reset all the pipe colors.
        """
        ports = self.inputs + self.outputs
        for port in ports:
            for pipe in port.connected_pipes:
                pipe.reset()

    def _calc_size_horizontal(self):
        # width, height from node name text.
        text_w = self._text_item.boundingRect().width()
        text_h = self._text_item.boundingRect().height()

        # width, height from node ports (grouped rows).
        port_width = 0.0
        p_input_text_width = 0.0
        p_output_text_width = 0.0
        p_input_height = 0.0
        p_output_height = 0.0
        port_height = 0.0
        spacing = 1.0
        group_gap = 6.0

        for port, text in self._input_items.items():
            if not port.isVisible():
                continue
            if not port_width:
                port_width = port.boundingRect().width()
            if not port_height:
                port_height = port.boundingRect().height()
            # State labels are displayed via the collapsible header button, not port text.
            if self._port_group(_port_name(port)) == "state":
                continue
            t_width = text.boundingRect().width()
            if text.isVisible() and t_width > p_input_text_width:
                p_input_text_width = text.boundingRect().width()
        for port, text in self._output_items.items():
            if not port.isVisible():
                continue
            if not port_width:
                port_width = port.boundingRect().width()
            if not port_height:
                port_height = port.boundingRect().height()
            if self._port_group(_port_name(port)) == "state":
                continue
            t_width = text.boundingRect().width()
            if text.isVisible() and t_width > p_output_text_width:
                p_output_text_width = text.boundingRect().width()

        # Determine grouped row count using current ports (fallback when backend node isn't available).
        def _names_for(kind: str, *, is_in: bool) -> list[str]:
            items = self._input_items if is_in else self._output_items
            out = []
            for p in items.keys():
                try:
                    if not p.isVisible():
                        continue
                    pname = _port_name(p)
                    if self._port_group(pname) == kind:
                        out.append(pname)
                except Exception:
                    continue
            return out

        exec_in = _names_for("exec", is_in=True)
        exec_out = _names_for("exec", is_in=False)
        data_in = _names_for("data", is_in=True)
        data_out = _names_for("data", is_in=False)
        state_in = _names_for("state", is_in=True)
        state_out = _names_for("state", is_in=False)
        other_in = _names_for("other", is_in=True)
        other_out = _names_for("other", is_in=False)

        state_names: list[str] = [n for n, p in self._state_inline_proxies.items() if p.isVisible()]
        if not state_names:
            # Infer state row order from port names (best-effort).
            tmp: list[str] = []
            for n in state_in:
                if n.startswith("[S]"):
                    tmp.append(n[3:])
            for n in state_out:
                if n.endswith("[S]"):
                    tmp.append(n[:-3])
            state_names = [x for x in list(OrderedDict.fromkeys(tmp).keys()) if x]

        rows_exec = max(len(exec_in), len(exec_out))
        rows_data = max(len(data_in), len(data_out))
        rows_other = max(len(other_in), len(other_out))

        # Calculate port area height with expandable state panels.
        ports_h = 0.0
        if port_height:

            def _add_group_rows(rows: int) -> None:
                nonlocal ports_h
                if rows <= 0:
                    return
                if ports_h > 0:
                    ports_h += group_gap
                ports_h += (rows * port_height) + (max(0, rows - 1) * spacing)

            _add_group_rows(rows_exec)
            _add_group_rows(rows_data)

            # State: each row has a header (ports+toggle) and optional expanded body.
            if state_names:
                if ports_h > 0:
                    ports_h += group_gap
                for i, sname in enumerate(state_names):
                    header_h = port_height
                    try:
                        header = self._state_inline_headers.get(sname)
                        if header is not None:
                            header_h = float(max(port_height, header.sizeHint().height()))
                    except Exception:
                        header_h = port_height
                    # Size hint for the expanded body depends on width (options wrap).
                    # Use the proxy widget bounding rect after forcing a best-effort width.
                    panel_h = header_h
                    try:
                        proxy = self._state_inline_proxies.get(sname)
                        if proxy is not None and proxy.isVisible():
                            try:
                                w = proxy.widget()
                                if w is not None:
                                    rect_w = max(10, int(self.boundingRect().width() - 8.0))
                                    w.setFixedWidth(rect_w)
                                    w.adjustSize()
                            except Exception:
                                pass
                            try:
                                panel_h = float(max(header_h, proxy.boundingRect().height()))
                            except Exception:
                                panel_h = header_h
                    except Exception:
                        panel_h = header_h
                    ports_h += panel_h + spacing
                ports_h = max(0.0, ports_h - spacing)  # remove trailing row spacing

            _add_group_rows(rows_other)

            p_input_height = ports_h
            p_output_height = ports_h

        port_text_width = p_input_text_width + p_output_text_width

        # width, height from node embedded widgets.
        widget_width = 0.0
        widget_height = 0.0
        # Ensure state inline widgets exist so we can account for width.
        try:
            self._ensure_inline_state_widgets()
        except Exception:
            pass
        try:
            self._ensure_inline_command_widget()
        except Exception:
            pass
        for widget in self._widgets.values():
            if not widget.isVisible():
                continue
            w_width = widget.boundingRect().width()
            w_height = widget.boundingRect().height()
            if w_width > widget_width:
                widget_width = w_width
            widget_height += w_height
        # State panels span the node width; they should not participate in width calculation.
        # Command widget spans the node width; it should not participate in width calculation.

        side_padding = 0.0
        if all([widget_width, p_input_text_width, p_output_text_width]):
            port_text_width = max([p_input_text_width, p_output_text_width])
            port_text_width *= 2
        elif widget_width:
            side_padding = 10

        width = port_width + max([text_w, port_text_width]) + side_padding

        port_area_height = max(p_input_height, p_output_height)
        height = max([text_h, port_area_height, widget_height])
        if widget_width:
            # add additional width for node widget.
            width += widget_width
        if widget_height:
            # add bottom margin for node widget.
            height += 4.0

        # Commands: compute height using the final width (flow wrap depends on width).
        if self._cmd_proxy is not None:
            try:
                if self._cmd_proxy.isVisible() and port_height:
                    rect_w = max(10, int(width - 8.0))
                    if self._cmd_widget is not None:
                        self._cmd_widget.setFixedWidth(rect_w)
                        self._cmd_widget.adjustSize()
                    cmd_h = float(self._cmd_proxy.boundingRect().height())
                    if cmd_h > 0:
                        height = max(height, port_area_height + cmd_h + 10.0)
            except Exception:
                pass
        height *= 1.05
        return width, height

    def _calc_size_vertical(self):
        p_input_width = 0.0
        p_output_width = 0.0
        p_input_height = 0.0
        p_output_height = 0.0
        for port in self._input_items.keys():
            if port.isVisible():
                p_input_width += port.boundingRect().width()
                if not p_input_height:
                    p_input_height = port.boundingRect().height()
        for port in self._output_items.keys():
            if port.isVisible():
                p_output_width += port.boundingRect().width()
                if not p_output_height:
                    p_output_height = port.boundingRect().height()

        widget_width = 0.0
        widget_height = 0.0
        for widget in self._widgets.values():
            if not widget.isVisible():
                continue
            if widget.boundingRect().width() > widget_width:
                widget_width = widget.boundingRect().width()
            widget_height += widget.boundingRect().height()

        width = max([p_input_width, p_output_width, widget_width])
        height = p_input_height + p_output_height + widget_height
        return width, height

    def calc_size(self, add_w=0.0, add_h=0.0):
        """
        Calculates the minimum node size.

        Args:
            add_w (float): additional width.
            add_h (float): additional height.

        Returns:
            tuple(float, float): width, height.
        """
        if self.layout_direction is LayoutDirectionEnum.HORIZONTAL.value:
            width, height = self._calc_size_horizontal()
        elif self.layout_direction is LayoutDirectionEnum.VERTICAL.value:
            width, height = self._calc_size_vertical()
        else:
            raise RuntimeError("Node graph layout direction not valid!")

        # additional width, height.
        width += add_w
        height += add_h
        return width, height

    def _align_icon_horizontal(self, h_offset, v_offset):
        icon_rect = self._icon_item.boundingRect()
        text_rect = self._text_item.boundingRect()
        x = self.boundingRect().left() + 2.0
        y = text_rect.center().y() - (icon_rect.height() / 2)
        self._icon_item.setPos(x + h_offset, y + v_offset)

    def _align_icon_vertical(self, h_offset, v_offset):
        center_y = self.boundingRect().center().y()
        icon_rect = self._icon_item.boundingRect()
        text_rect = self._text_item.boundingRect()
        x = self.boundingRect().right() + h_offset
        y = center_y - text_rect.height() - (icon_rect.height() / 2) + v_offset
        self._icon_item.setPos(x, y)

    def align_icon(self, h_offset=0.0, v_offset=0.0):
        """
        Align node icon to the default top left of the node.

        Args:
            v_offset (float): additional vertical offset.
            h_offset (float): additional horizontal offset.
        """
        if self.layout_direction is LayoutDirectionEnum.HORIZONTAL.value:
            self._align_icon_horizontal(h_offset, v_offset)
        elif self.layout_direction is LayoutDirectionEnum.VERTICAL.value:
            self._align_icon_vertical(h_offset, v_offset)
        else:
            raise RuntimeError("Node graph layout direction not valid!")

    def _align_label_horizontal(self, h_offset, v_offset):
        rect = self.boundingRect()
        text_rect = self._text_item.boundingRect()
        x = rect.center().x() - (text_rect.width() / 2)
        self._text_item.setPos(x + h_offset, rect.y() + v_offset)

    def _align_label_vertical(self, h_offset, v_offset):
        rect = self._text_item.boundingRect()
        x = self.boundingRect().right() + h_offset
        y = self.boundingRect().center().y() - (rect.height() / 2) + v_offset
        self.text_item.setPos(x, y)

    def align_label(self, h_offset=0.0, v_offset=0.0):
        """
        Center node label text to the top of the node.

        Args:
            v_offset (float): vertical offset.
            h_offset (float): horizontal offset.
        """
        if self.layout_direction is LayoutDirectionEnum.HORIZONTAL.value:
            self._align_label_horizontal(h_offset, v_offset)
        elif self.layout_direction is LayoutDirectionEnum.VERTICAL.value:
            self._align_label_vertical(h_offset, v_offset)
        else:
            raise RuntimeError("Node graph layout direction not valid!")

    def _align_widgets_horizontal(self, v_offset):
        rect = self.boundingRect()
        inputs = [p for p in self.inputs if p.isVisible()]
        outputs = [p for p in self.outputs if p.isVisible()]

        # Command buttons are placed below the ports area and should span the full node width.
        cmd_bottom = None
        if self._cmd_proxy is not None and self._cmd_proxy.isVisible():
            try:
                rect = self.boundingRect()
                y = float(self._ports_end_y or (rect.y() + v_offset))
                # Force the underlying QWidget to take the full available width.
                try:
                    if self._cmd_widget is not None:
                        self._cmd_widget.setFixedWidth(max(10, int(rect.width() - 8.0)))
                        self._cmd_widget.adjustSize()
                except Exception:
                    pass
                w_rect = self._cmd_proxy.boundingRect()
                x = rect.left() + 4.0
                self._cmd_proxy.setPos(x, y + 6.0)
                cmd_bottom = y + 6.0 + w_rect.height()
            except Exception:
                cmd_bottom = None

        if not self._widgets:
            return
        rect = self.boundingRect()
        # Place regular NodeGraphQt embedded widgets below the ports area (and below
        # command area if present). This prevents custom widgets from overlapping
        # the ports/state region.
        base_y = float(self._ports_end_y or (rect.y() + v_offset))
        y = base_y + 6.0
        if cmd_bottom is not None:
            y = max(y, cmd_bottom + 6.0)
        for widget in self._widgets.values():
            if not widget.isVisible():
                continue
            widget_rect = widget.boundingRect()
            if not inputs:
                x = rect.left() + 10
                widget.widget().setTitleAlign("left")
            elif not outputs:
                x = rect.right() - widget_rect.width() - 10
                widget.widget().setTitleAlign("right")
            else:
                x = rect.center().x() - (widget_rect.width() / 2)
                widget.widget().setTitleAlign("center")
            widget.setPos(x, y)
            y += widget_rect.height()

    def _align_widgets_vertical(self, v_offset):
        if not self._widgets:
            return
        rect = self.boundingRect()
        y = rect.center().y() + v_offset
        widget_height = 0.0
        for widget in self._widgets.values():
            if not widget.isVisible():
                continue
            widget_rect = widget.boundingRect()
            widget_height += widget_rect.height()
        y -= widget_height / 2

        for widget in self._widgets.values():
            if not widget.isVisible():
                continue
            widget_rect = widget.boundingRect()
            x = rect.center().x() - (widget_rect.width() / 2)
            widget.widget().setTitleAlign("center")
            widget.setPos(x, y)
            y += widget_rect.height()

    def align_widgets(self, v_offset=0.0):
        """
        Align node widgets to the default center of the node.

        Args:
            v_offset (float): vertical offset.
        """
        if self.layout_direction is LayoutDirectionEnum.HORIZONTAL.value:
            self._align_widgets_horizontal(v_offset)
        elif self.layout_direction is LayoutDirectionEnum.VERTICAL.value:
            self._align_widgets_vertical(v_offset)
        else:
            raise RuntimeError("Node graph layout direction not valid!")

    def _align_ports_horizontal(self, v_offset):
        width = self._width
        txt_offset = PortEnum.CLICK_FALLOFF.value - 2
        spacing = 1.0
        group_gap = 6.0

        # Ensure inline widgets exist before aligning so sizing + rows match.
        try:
            self._ensure_inline_state_widgets()
        except Exception:
            pass

        node = self._backend_node()
        if node is None:
            spec = None
        else:
            try:
                spec = node.spec
            except Exception:
                spec = None
        try:
            eff_states = list(node.effective_state_fields() or []) if node is not None else []
        except Exception:
            if spec is None:
                eff_states = []
            else:
                try:
                    eff_states = list(spec.stateFields or [])
                except Exception:
                    eff_states = []

        # Build ordered port name lists per group.
        exec_in_names: list[str] = []
        exec_out_names: list[str] = []
        if isinstance(spec, F8ServiceSpec):
            exec_in, exec_out = _service_exec_ports(spec)
            for p in exec_in:
                exec_in_names.append(f"[E]{p}")
            for p in exec_out:
                exec_out_names.append(f"{p}[E]")

        data_in_names: list[str] = []
        data_out_names: list[str] = []
        if node is not None:
            try:
                existing_in = {_port_name(p) for p in self._input_items.keys()}
                existing_out = {_port_name(p) for p in self._output_items.keys()}
                for p in list(spec.dataInPorts or []):
                    port_name = f"[D]{p.name}"
                    if node.data_port_show_on_node(str(p.name or ""), is_in=True) or port_name in existing_in:
                        data_in_names.append(port_name)
                for p in list(spec.dataOutPorts or []):
                    port_name = f"{p.name}[D]"
                    if node.data_port_show_on_node(str(p.name or ""), is_in=False) or port_name in existing_out:
                        data_out_names.append(port_name)
            except Exception:
                data_in_names = []
                data_out_names = []

        state_names: list[str] = []
        for s in eff_states:
            info = _state_field_info(s)
            if info is None or not info.show_on_node:
                continue
            if info.name:
                state_names.append(info.name)

        # Fallback when spec is unavailable: keep insertion order but grouped.
        if not exec_in_names:
            exec_in_names = [
                _port_name(p) for p in self._input_items.keys() if self._port_group(_port_name(p)) == "exec"
            ]
        if not exec_out_names:
            exec_out_names = [
                _port_name(p) for p in self._output_items.keys() if self._port_group(_port_name(p)) == "exec"
            ]
        if not data_in_names:
            data_in_names = [
                _port_name(p) for p in self._input_items.keys() if self._port_group(_port_name(p)) == "data"
            ]
        if not data_out_names:
            data_out_names = [
                _port_name(p) for p in self._output_items.keys() if self._port_group(_port_name(p)) == "data"
            ]
        if not state_names:
            # Infer state rows from existing ports.
            tmp: list[str] = []
            for p in self._input_items.keys():
                n = _port_name(p)
                if n.startswith("[S]"):
                    tmp.append(n[3:])
            for p in self._output_items.keys():
                n = _port_name(p)
                if n.endswith("[S]"):
                    tmp.append(n[:-3])
            state_names = [x for x in list(OrderedDict.fromkeys(tmp).keys()) if x]

        other_in_names = [
            _port_name(p) for p in self._input_items.keys() if self._port_group(_port_name(p)) == "other"
        ]
        other_out_names = [
            _port_name(p) for p in self._output_items.keys() if self._port_group(_port_name(p)) == "other"
        ]

        inputs_by_name = {_port_name(p): p for p in self.inputs if p.isVisible()}
        outputs_by_name = {_port_name(p): p for p in self.outputs if p.isVisible()}

        # Determine base port geometry.
        port_width = 0.0
        port_height = 0.0
        for p in list(inputs_by_name.values()) + list(outputs_by_name.values()):
            try:
                port_width = float(p.boundingRect().width())
                port_height = float(p.boundingRect().height())
                break
            except Exception:
                continue

        in_x = (port_width / 2.0) * -1.0
        out_x = width - (port_width / 2.0)

        rect = self.boundingRect()
        inner_x = rect.left() + 4.0
        inner_w = max(10.0, rect.width() - 8.0)

        def place_row(in_name: str | None, out_name: str | None, *, y: float):
            if in_name:
                p = inputs_by_name.get(in_name)
                if p is not None:
                    p.setPos(in_x, y)
            if out_name:
                p = outputs_by_name.get(out_name)
                if p is not None:
                    p.setPos(out_x, y)

        y = float(v_offset)
        groups: list[tuple[str, list[str], list[str]]] = [
            ("exec", exec_in_names, exec_out_names),
            ("data", data_in_names, data_out_names),
            ("state", [f"[S]{n}" for n in state_names], [f"{n}[S]" for n in state_names]),
            ("other", other_in_names, other_out_names),
        ]

        for gi, (gname, ins, outs) in enumerate(groups):
            if gname == "state":
                rows = len(state_names)
            else:
                rows = max(len(ins), len(outs))
            if rows <= 0:
                continue
            for i in range(rows):
                in_name = ins[i] if i < len(ins) else None
                out_name = outs[i] if i < len(outs) else None

                if gname != "state":
                    place_row(in_name, out_name, y=y)
                    y += port_height + spacing
                    continue

                # State row: place collapsible panel + ports aligned to header line.
                state_key = state_names[i] if i < len(state_names) else None
                panel_proxy = self._state_inline_proxies.get(state_key) if state_key else None
                header_h = port_height
                body_h = 0.0
                if state_key and panel_proxy is not None:
                    # Ensure width is up to date before measuring heights (option rows wrap by width).
                    try:
                        w = panel_proxy.widget()
                        if w is not None:
                            w.setFixedWidth(int(inner_w))
                            w.adjustSize()
                    except Exception:
                        pass
                    try:
                        if self._state_inline_headers.get(state_key) is not None:
                            header_h = float(
                                max(port_height, self._state_inline_headers[state_key].sizeHint().height())
                            )
                    except Exception:
                        header_h = port_height
                    try:
                        body_w = self._state_inline_bodies.get(state_key)
                        if body_w is not None and body_w.isVisible():
                            body_h = float(max(0.0, body_w.sizeHint().height()))
                    except Exception:
                        body_h = 0.0
                    try:
                        # Center panels using their *actual* width. Some controls
                        # can enforce minimum sizes that override our target width,
                        # causing asymmetric margins if we anchor at `inner_x`.
                        w = panel_proxy.widget()
                        if w is None:
                            panel_proxy.setPos(inner_x, y)
                        else:
                            panel_w = float(w.width() or 0)
                            if panel_w <= 0:
                                panel_w = float(panel_proxy.boundingRect().width() or 0)
                            if panel_w <= 0:
                                panel_proxy.setPos(inner_x, y)
                            else:
                                panel_x = rect.left() + (rect.width() - panel_w) / 2.0
                                # Clamp inside the node content area so the right edge
                                # never gets clipped by the node boundary.
                                min_x = float(inner_x)
                                max_x = float(rect.right() - 4.0 - panel_w)
                                if max_x < min_x:
                                    panel_x = min_x
                                else:
                                    panel_x = max(min_x, min(panel_x, max_x))
                                panel_proxy.setPos(panel_x, y)
                    except Exception:
                        pass

                port_y = y + (header_h - port_height) / 2.0
                place_row(in_name, out_name, y=port_y)
                y += header_h + spacing
                if body_h > 0.0:
                    y += body_h + spacing
            # group gap (except after last visible group)
            # determine if any later group has rows.
            has_later = False
            for _g2, ins2, outs2 in groups[gi + 1 :]:
                if _g2 == "state":
                    if len(state_names) > 0:
                        has_later = True
                        break
                else:
                    if max(len(ins2), len(outs2)) > 0:
                        has_later = True
                        break
            if has_later:
                y += group_gap
        self._ports_end_y = y

        # adjust input text position
        for port, text in self._input_items.items():
            if port.isVisible():
                txt_x = port.boundingRect().width() / 2 - txt_offset
                text.setPos(txt_x, port.y() - 1.5)

        # adjust output text position
        for port, text in self._output_items.items():
            if port.isVisible():
                txt_width = text.boundingRect().width() - txt_offset
                txt_x = port.x() - txt_width
                text.setPos(txt_x, port.y() - 1.5)

    def _align_ports_vertical(self, v_offset):
        # adjust input position
        inputs = [p for p in self.inputs if p.isVisible()]
        if inputs:
            port_width = inputs[0].boundingRect().width()
            port_height = inputs[0].boundingRect().height()
            half_width = port_width / 2
            delta = self._width / (len(inputs) + 1)
            port_x = delta
            port_y = (port_height / 2) * -1
            for port in inputs:
                port.setPos(port_x - half_width, port_y)
                port_x += delta

        # adjust output position
        outputs = [p for p in self.outputs if p.isVisible()]
        if outputs:
            port_width = outputs[0].boundingRect().width()
            port_height = outputs[0].boundingRect().height()
            half_width = port_width / 2
            delta = self._width / (len(outputs) + 1)
            port_x = delta
            port_y = self._height - (port_height / 2)
            for port in outputs:
                port.setPos(port_x - half_width, port_y)
                port_x += delta

    def align_ports(self, v_offset=0.0):
        """
        Align input, output ports in the node layout.

        Args:
            v_offset (float): port vertical offset.
        """
        if self.layout_direction is LayoutDirectionEnum.HORIZONTAL.value:
            self._align_ports_horizontal(v_offset)
        elif self.layout_direction is LayoutDirectionEnum.VERTICAL.value:
            self._align_ports_vertical(v_offset)
        else:
            raise RuntimeError("Node graph layout direction not valid!")

    def _draw_node_horizontal(self):
        try:
            self._ensure_inline_state_widgets()
        except Exception:
            pass
        try:
            self._ensure_inline_command_widget()
        except Exception:
            pass
        height = self._text_item.boundingRect().height() + 4.0

        # update port text items in visibility.
        for port, text in self._input_items.items():
            if port.isVisible():
                if self._port_group(_port_name(port)) == "state":
                    text.setVisible(False)
                else:
                    text.setVisible(port.display_name)
        for port, text in self._output_items.items():
            if port.isVisible():
                if self._port_group(_port_name(port)) == "state":
                    text.setVisible(False)
                else:
                    text.setVisible(port.display_name)

        # setup initial base size.
        self._set_base_size(add_h=height)
        # set text color when node is initialized.
        self._set_text_color(self.text_color)
        # set the tooltip
        self._tooltip_disable(self.disabled)

        # --- set the initial node layout ---
        # (do all the graphic item layout offsets here)

        # align label text
        self.align_label()
        # align icon
        self.align_icon(h_offset=2.0, v_offset=1.0)
        # arrange input and output ports.
        self.align_ports(v_offset=height)
        # arrange node widgets
        self.align_widgets(v_offset=height)

        self.update()

    def _draw_node_vertical(self):
        # hide the port text items in vertical layout.
        for port, text in self._input_items.items():
            text.setVisible(False)
        for port, text in self._output_items.items():
            text.setVisible(False)

        # setup initial base size.
        self._set_base_size()
        # set text color when node is initialized.
        self._set_text_color(self.text_color)
        # set the tooltip
        self._tooltip_disable(self.disabled)

        # --- setup node layout ---
        # (do all the graphic item layout offsets here)

        # align label text
        self.align_label(h_offset=6)
        # align icon
        self.align_icon(h_offset=6, v_offset=4)
        # arrange input and output ports.
        self.align_ports()
        # arrange node widgets
        self.align_widgets()

        self.update()

    def draw_node(self):
        """
        Re-draw the node item in the scene with proper
        calculated size and widgets aligned.
        """
        if self.layout_direction is LayoutDirectionEnum.HORIZONTAL.value:
            self._draw_node_horizontal()
        elif self.layout_direction is LayoutDirectionEnum.VERTICAL.value:
            self._draw_node_vertical()
        else:
            raise RuntimeError("Node graph layout direction not valid!")
        self._position_service_toolbar()

    def post_init(self, viewer=None, pos=None):
        """
        Called after node has been added into the scene.
        Adjust the node layout and form after the node has been added.

        Args:
            viewer (NodeGraphQt.widgets.viewer.NodeViewer): not used
            pos (tuple): cursor position.
        """
        self.draw_node()
        self._ensure_service_toolbar(viewer)
        self._position_service_toolbar()

        # set initial node position.
        if pos:
            self.xy_pos = pos
            self._position_service_toolbar()

    def _ensure_service_toolbar(self, viewer: Any | None) -> None:
        if self._svc_toolbar_proxy is not None:
            return
        try:
            service_id = str(self.id or "").strip()
        except Exception:
            service_id = ""
        if not service_id:
            return

        def _resolve_graph() -> Any | None:
            # Prefer the viewer passed by NodeGraphQt (more reliable than self.viewer() during init).
            try:
                if isinstance(viewer, F8StudioNodeViewer) and viewer.f8_graph is not None:
                    return viewer.f8_graph
            except Exception:
                pass
            return self._graph()

        def _get_bridge() -> ServiceBridge | None:
            g = _resolve_graph()
            try:
                return g.service_bridge if g is not None else None
            except Exception:
                return None

        def _get_node() -> Any | None:
            g = _resolve_graph()
            if g is None:
                return None
            try:
                return g.get_node_by_id(service_id)
            except Exception:
                return None

        def _get_service_class() -> str:
            try:
                n = _get_node() or self._backend_node()
                if n is None:
                    return ""
                spec = n.spec
                return str(spec.serviceClass or "")
            except Exception:
                return ""

        def _get_compiled_graphs() -> Any | None:
            try:
                g = _resolve_graph() or self._graph()
                if g is None:
                    return None
                from .runtime_compiler import compile_runtime_graphs_from_studio

                return compile_runtime_graphs_from_studio(g)
            except Exception:
                return None

        try:
            w = ServiceProcessToolbar(
                service_id=service_id,
                get_bridge=_get_bridge,
                get_node=_get_node,
                get_service_class=_get_service_class,
                get_compiled_graphs=_get_compiled_graphs,
            )
            proxy = QtWidgets.QGraphicsProxyWidget(self)
            proxy.setWidget(w)
            proxy.setZValue(10_000)
            proxy.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
            self._svc_toolbar_proxy = proxy
        except Exception:
            self._svc_toolbar_proxy = None

    def _position_service_toolbar(self) -> None:
        proxy = self._svc_toolbar_proxy
        if proxy is None:
            return
        try:
            rect = self.boundingRect()
            w = float(proxy.size().width() or 0.0)
            h = float(proxy.size().height() or 0.0)
        except Exception:
            return

        try:
            proxy.setPos(rect.right() - w, rect.top() - h)
        except Exception:
            pass

    def auto_switch_mode(self):
        """
        Decide whether to draw the node with proxy mode.
        (this is called at the start in the "self.paint()" function.)
        """
        if ITEM_CACHE_MODE is QtWidgets.QGraphicsItem.ItemCoordinateCache:
            return

        v = self._viewer_safe()
        if v is None:
            return

        rect = self.sceneBoundingRect()
        l = v.mapToGlobal(v.mapFromScene(rect.topLeft()))
        r = v.mapToGlobal(v.mapFromScene(rect.topRight()))
        # width is the node width in screen
        width = r.x() - l.x()

        self.set_proxy_mode(width < self._proxy_mode_threshold)

    def set_proxy_mode(self, mode):
        """
        Set whether to draw the node with proxy mode.
        (proxy mode toggles visibility for some qgraphic items in the node.)

        Args:
            mode (bool): true to enable proxy mode.
        """
        if mode is self._proxy_mode:
            return
        self._proxy_mode = mode

        visible = not mode

        # disable overlay item.
        self._x_item.proxy_mode = self._proxy_mode

        # node widget visibility.
        for w in self._widgets.values():
            w.widget().setVisible(visible)
        for p in self._state_inline_proxies.values():
            try:
                p.setVisible(visible)
            except Exception:
                pass
        if self._cmd_proxy is not None:
            try:
                self._cmd_proxy.setVisible(visible)
            except Exception:
                pass

        # port text is not visible in vertical layout.
        if self.layout_direction is LayoutDirectionEnum.VERTICAL.value:
            port_text_visible = False
        else:
            port_text_visible = visible

        # input port text visibility.
        for port, text in self._input_items.items():
            try:
                is_state = self._port_group(_port_name(port)) == "state"
            except Exception:
                is_state = False
            should_show = bool(port_text_visible and port.display_name and not is_state)
            text.setVisible(should_show)

        # output port text visibility.
        for port, text in self._output_items.items():
            try:
                is_state = self._port_group(_port_name(port)) == "state"
            except Exception:
                is_state = False
            should_show = bool(port_text_visible and port.display_name and not is_state)
            text.setVisible(should_show)

        self._text_item.setVisible(visible)
        self._icon_item.setVisible(visible)

    @property
    def icon(self):
        return self._properties["icon"]

    @icon.setter
    def icon(self, path=None):
        self._properties["icon"] = path
        path = path or ICON_NODE_BASE
        pixmap = QtGui.QPixmap(path)
        if pixmap.size().height() > NodeEnum.ICON_SIZE.value:
            pixmap = pixmap.scaledToHeight(NodeEnum.ICON_SIZE.value, QtCore.Qt.SmoothTransformation)
        if pixmap.size().width() > NodeEnum.ICON_SIZE.value:
            pixmap = pixmap.scaledToWidth(NodeEnum.ICON_SIZE.value, QtCore.Qt.SmoothTransformation)
        self._icon_item.setPixmap(pixmap)
        if self.scene():
            self.post_init()

        self.update()

    @AbstractNodeItem.layout_direction.setter
    def layout_direction(self, value=0):
        AbstractNodeItem.layout_direction.fset(self, value)
        self.draw_node()

    @AbstractNodeItem.width.setter
    def width(self, width=0.0):
        w, h = self.calc_size()
        width = width if width > w else w
        AbstractNodeItem.width.fset(self, width)

    @AbstractNodeItem.height.setter
    def height(self, height=0.0):
        w, h = self.calc_size()
        h = 70 if h < 70 else h
        height = height if height > h else h
        AbstractNodeItem.height.fset(self, height)

    @AbstractNodeItem.disabled.setter
    def disabled(self, state=False):
        AbstractNodeItem.disabled.fset(self, state)
        for n, w in self._widgets.items():
            w.widget().setDisabled(state)
        self._tooltip_disable(state)
        self._x_item.setVisible(state)

    @AbstractNodeItem.selected.setter
    def selected(self, selected=False):
        AbstractNodeItem.selected.fset(self, selected)
        if selected:
            self.highlight_pipes()

    @AbstractNodeItem.name.setter
    def name(self, name=""):
        AbstractNodeItem.name.fset(self, name)
        if name == self._text_item.toPlainText():
            return
        self._text_item.setPlainText(name)
        if self.scene():
            self.align_label()
        self.update()

    @AbstractNodeItem.color.setter
    def color(self, color=(100, 100, 100, 255)):
        AbstractNodeItem.color.fset(self, color)
        if self.scene():
            self.scene().update()
        self.update()

    @AbstractNodeItem.border_color.setter
    def border_color(self, color=(100, 100, 100, 255)):
        AbstractNodeItem.border_color.fset(self, color)
        if self.scene():
            self.scene().update()
        self.update()

    @AbstractNodeItem.text_color.setter
    def text_color(self, color=(100, 100, 100, 255)):
        AbstractNodeItem.text_color.fset(self, color)
        self._set_text_color(color)
        self.update()

    @property
    def text_item(self):
        """
        Get the node name text qgraphics item.

        Returns:
            NodeTextItem: node text object.
        """
        return self._text_item

    @property
    def icon_item(self):
        """
        Get the node icon qgraphics item.

        Returns:
            QtWidgets.QGraphicsPixmapItem: node icon object.
        """
        return self._icon_item

    @property
    def inputs(self):
        """
        Returns:
            list[PortItem]: input port graphic items.
        """
        return list(self._input_items.keys())

    @property
    def outputs(self):
        """
        Returns:
            list[PortItem]: output port graphic items.
        """
        return list(self._output_items.keys())

    def _add_port(self, port):
        """
        Adds a port qgraphics item into the node.

        Args:
            port (PortItem): port item.

        Returns:
            PortItem: port qgraphics item.
        """
        full_name = str(port.name or "")
        group = self._port_group(full_name)
        # Aggressively elide state port labels to reduce width usage.
        max_chars = 10 if group == "state" else 18
        label = self._display_port_label(full_name, max_chars=max_chars)
        text = QtWidgets.QGraphicsTextItem(label, self)
        text.font().setPointSize(8)
        text.setFont(text.font())
        text.setVisible(port.display_name)
        text.setCacheMode(ITEM_CACHE_MODE)
        try:
            text.setToolTip(full_name)
        except Exception:
            pass
        if port.port_type == PortTypeEnum.IN.value:
            self._input_items[port] = text
        elif port.port_type == PortTypeEnum.OUT.value:
            self._output_items[port] = text
        if self.scene():
            self.post_init()
        return port

    def add_input(self, name="input", multi_port=False, display_name=True, locked=False, painter_func=None):
        """
        Adds a port qgraphics item into the node with the "port_type" set as
        IN_PORT.

        Args:
            name (str): name for the port.
            multi_port (bool): allow multiple connections.
            display_name (bool): display the port name.
            locked (bool): locked state.
            painter_func (function): custom paint function.

        Returns:
            PortItem: input port qgraphics item.
        """
        if painter_func:
            port = CustomPortItem(self, painter_func)
        else:
            port = PortItem(self)
        port.name = name
        port.port_type = PortTypeEnum.IN.value
        port.multi_connection = multi_port
        port.display_name = display_name
        port.locked = locked
        return self._add_port(port)

    def add_output(self, name="output", multi_port=False, display_name=True, locked=False, painter_func=None):
        """
        Adds a port qgraphics item into the node with the "port_type" set as
        OUT_PORT.

        Args:
            name (str): name for the port.
            multi_port (bool): allow multiple connections.
            display_name (bool): display the port name.
            locked (bool): locked state.
            painter_func (function): custom paint function.

        Returns:
            PortItem: output port qgraphics item.
        """
        if painter_func:
            port = CustomPortItem(self, painter_func)
        else:
            port = PortItem(self)
        port.name = name
        port.port_type = PortTypeEnum.OUT.value
        port.multi_connection = multi_port
        port.display_name = display_name
        port.locked = locked
        return self._add_port(port)

    def _delete_port(self, port, text):
        """
        Removes port item and port text from node.

        Args:
            port (PortItem): port object.
            text (QtWidgets.QGraphicsTextItem): port text object.
        """
        port.setParentItem(None)
        text.setParentItem(None)
        scene = self.scene()
        if scene is not None:
            scene.removeItem(port)
            scene.removeItem(text)
        del port
        del text

    def delete_input(self, port):
        """
        Remove input port from node.

        Args:
            port (PortItem): port object.
        """
        self._delete_port(port, self._input_items.pop(port))

    def delete_output(self, port):
        """
        Remove output port from node.

        Args:
            port (PortItem): port object.
        """
        self._delete_port(port, self._output_items.pop(port))

    def get_input_text_item(self, port_item):
        """
        Args:
            port_item (PortItem): port item.

        Returns:
            QGraphicsTextItem: graphic item used for the port text.
        """
        return self._input_items[port_item]

    def get_output_text_item(self, port_item):
        """
        Args:
            port_item (PortItem): port item.

        Returns:
            QGraphicsTextItem: graphic item used for the port text.
        """
        return self._output_items[port_item]

    @property
    def widgets(self):
        return self._widgets.copy()

    def add_widget(self, widget):
        self._widgets[widget.get_name()] = widget

    def get_widget(self, name):
        widget = self._widgets.get(name)
        if widget:
            return widget
        raise NodeWidgetError('node has no widget "{}"'.format(name))

    def has_widget(self, name):
        return name in self._widgets.keys()

    def from_dict(self, node_dict):
        super().from_dict(node_dict)
        custom_prop = node_dict.get("custom") or {}
        for prop_name, value in custom_prop.items():
            prop_widget = self._widgets.get(prop_name)
            if prop_widget:
                prop_widget.set_value(value)
