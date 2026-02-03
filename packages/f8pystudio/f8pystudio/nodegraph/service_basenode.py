from __future__ import annotations

import logging
import json
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
from ..widgets.f8_editor_widgets import F8ExclusiveToggleRow, F8ImageB64Editor, F8ValueBar, parse_select_pool

logger = logging.getLogger(__name__)


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

    def _apply_elide(self) -> None:
        try:
            fm = QtGui.QFontMetrics(self.font())
            # Leave room for the arrow icon.
            w = max(10, int(self.width() - 24))
            self.setText(fm.elidedText(self._full_text, QtCore.Qt.ElideRight, w))
        except RuntimeError:
            self.setText(self._full_text)


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
        for p in list(getattr(self.spec, "execInPorts", None) or []):
            self.add_input(
                f"[E]{p}",
                color=EXEC_PORT_COLOR,
                painter_func=draw_exec_port,
            )

        for p in list(getattr(self.spec, "execOutPorts", None) or []):
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

        for s in self.effective_state_fields():
            if not s.showOnNode:
                continue

            if getattr(s, "access", None) in [F8StateAccess.rw, F8StateAccess.wo]:
                self.add_input(
                    f"[S]{s.name}",
                    color=STATE_PORT_COLOR,
                    painter_func=draw_square_port,
                )

            if getattr(s, "access", None) in [F8StateAccess.rw, F8StateAccess.ro]:
                self.add_output(
                    f"{s.name}[S]",
                    color=STATE_PORT_COLOR,
                    painter_func=draw_square_port,
                )

    def _build_state_properties(self) -> None:
        for s in self.effective_state_fields() or []:
            name = str(getattr(s, "name", "") or "").strip()
            if not name:
                continue
            if self.has_property(name):  # type: ignore[attr-defined]
                continue
            try:
                default_value = schema_default(s.valueSchema)
            except Exception:
                default_value = None
            widget_type, items, prop_range = self._state_widget_for_schema(getattr(s, "valueSchema", None))
            tooltip = str(getattr(s, "description", "") or "").strip() or None
            self.create_property(
                name,
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
        root = getattr(value_schema, "root", None)
        enum_items = list(getattr(root, "enum", None) or []) if root is not None else []
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

        for p in list(getattr(self.spec, "execInPorts", None) or []):
            desired_inputs[f"[E]{p}"] = {"color": EXEC_PORT_COLOR, "painter_func": draw_exec_port}
        for p in list(getattr(self.spec, "execOutPorts", None) or []):
            desired_outputs[f"{p}[E]"] = {"color": EXEC_PORT_COLOR, "painter_func": draw_exec_port}

        for p in list(getattr(self.spec, "dataInPorts", None) or []):
            desired_inputs[f"[D]{p.name}"] = {"color": DATA_PORT_COLOR}
        for p in list(getattr(self.spec, "dataOutPorts", None) or []):
            desired_outputs[f"{p.name}[D]"] = {"color": DATA_PORT_COLOR}

        for s in list(self.effective_state_fields() or []):
            if not getattr(s, "showOnNode", False):
                continue
            name = str(getattr(s, "name", "") or "").strip()
            if not name:
                continue
            access = getattr(s, "access", None)
            if access in [F8StateAccess.rw, F8StateAccess.wo]:
                desired_inputs[f"[S]{name}"] = {"color": STATE_PORT_COLOR, "painter_func": draw_square_port}
            if access in [F8StateAccess.rw, F8StateAccess.ro]:
                desired_outputs[f"{name}[S]"] = {"color": STATE_PORT_COLOR, "painter_func": draw_square_port}

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

            input_items = getattr(view, "_input_items", None)
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

            output_items = getattr(view, "_output_items", None)
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
        self._ports_end_y: float | None = None

    def _backend_node(self) -> Any | None:
        """
        Best-effort access to the backing BaseNode object for this view item.
        """
        g = self._graph()
        if g is None:
            return None
        node_id = str(getattr(self, "id", "") or "").strip()
        if not node_id:
            return None
        try:
            return g.get_node_by_id(node_id)
        except KeyError:
            return None

    def _graph(self) -> Any | None:
        try:
            viewer = self.viewer()
        except RuntimeError:
            return None
        return getattr(viewer, "_f8_graph", None) if viewer is not None else None

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

    def _bridge(self) -> Any | None:
        g = self._graph()
        return getattr(g, "service_bridge", None) if g is not None else None

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
        fn = getattr(bridge, "is_service_running", None)
        if not callable(fn):
            return False
        try:
            return bool(fn(sid))
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
                    b.setToolTip((b.toolTip() or "").strip() + ("\nService not running" if b.toolTip() else "Service not running"))
            except Exception:
                continue
        try:
            QtCore.QTimer.singleShot(0, self.draw_node)
        except Exception:
            pass

    def _service_id(self) -> str:
        # For service nodes, nodeId == serviceId.
        return str(getattr(self, "id", "") or "").strip()

    def _invoke_command(self, cmd: Any) -> None:
        """
        Invoke a command declared on the service spec.

        - no params: fire immediately
        - has params: show dialog to collect args
        """
        call = str(getattr(cmd, "name", "") or "").strip()
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
        params = list(getattr(cmd, "params", None) or [])

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
        call = str(getattr(cmd, "name", "") or "").strip() or "Command"
        params = list(getattr(cmd, "params", None) or [])
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
            name = str(getattr(p, "name", "") or "").strip()
            if not name:
                continue
            required = bool(getattr(p, "required", False))
            ui = str(getattr(p, "uiControl", "") or "").strip().lower()
            schema = getattr(p, "valueSchema", None)
            t = schema_type(schema) if schema is not None else ""
            t = t or ""

            enum_items = self._schema_enum_items(schema)
            lo, hi = self._schema_numeric_range(schema)
            try:
                default_value = schema_default(schema)
            except Exception:
                default_value = None

            label = f"{name} *" if required else name
            tooltip = str(getattr(p, "description", "") or "").strip()

            def _with_tooltip(w: QtWidgets.QWidget) -> QtWidgets.QWidget:
                if tooltip:
                    w.setToolTip(tooltip)
                return w

            pool_field = parse_select_pool(ui)
            if enum_items or pool_field or ui in {"select", "dropdown", "dropbox", "combo", "combobox"}:
                row = F8ExclusiveToggleRow()
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
                row.set_options(items, labels=items)
                if tooltip:
                    row.set_context_tooltip(tooltip)
                if default_value is not None:
                    row.set_value(str(default_value))

                def _get() -> Any:
                    v = row.value()
                    return None if v is None else str(v)

                editors[name] = (_with_tooltip(row), _get)
                form.addRow(label, row)
                continue

            if t == "boolean" or ui in {"switch", "toggle"}:
                row = F8ExclusiveToggleRow()
                row.set_options([True, False], labels=["True", "False"])
                row.set_value(bool(default_value) if default_value is not None else None)

                def _get() -> Any:
                    v = row.value()
                    return None if v is None else bool(v)

                editors[name] = (_with_tooltip(row), _get)
                form.addRow(label, row)
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
                pname = str(getattr(p, "name", "") or "").strip()
                if not pname or pname not in editors:
                    continue
                required = bool(getattr(p, "required", False))
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
        spec = getattr(node, "spec", None)
        eff = getattr(node, "effective_commands", None)
        if callable(eff):
            cmds = list(eff() or [])
        else:
            cmds = list(getattr(spec, "commands", None) or []) if spec is not None else []
        visible_cmds = [c for c in cmds if bool(getattr(c, "showOnNode", False))]
        enabled = self._is_service_running()

        # Rebuild only when command list / enabled state changes.
        try:
            serial = json.dumps(
                {
                    "cmds": [
                        {
                            "name": str(getattr(c, "name", "") or ""),
                            "desc": str(getattr(c, "description", "") or ""),
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

        if self._cmd_proxy is not None and serial and serial == str(getattr(self, "_cmd_serial", "") or ""):
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
            b = QtWidgets.QPushButton(str(getattr(c, "name", "") or ""))
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
            desc = str(getattr(c, "description", "") or "").strip()
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
        if str(getattr(node, "id", "") or "") != str(getattr(self, "id", "") or ""):
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
            logger.exception("inline state updater failed nodeId=%s key=%s", getattr(self, "id", ""), key)
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
            if not isinstance(ctrl, F8ExclusiveToggleRow):
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
            return n[: max(1, max_chars - 1)] + "…"
        return n

    @staticmethod
    def _schema_enum_items(value_schema: Any) -> list[str]:
        if value_schema is None:
            return []
        root = getattr(value_schema, "root", None)
        enum = getattr(root, "enum", None)
        if not enum:
            return []
        return [str(x) for x in list(enum)]

    @staticmethod
    def _schema_numeric_range(value_schema: Any) -> tuple[float | None, float | None]:
        if value_schema is None:
            return None, None
        mins: list[float] = []
        maxs: list[float] = []
        for k in ("minimum", "exclusiveMinimum"):
            v = getattr(value_schema, k, None)
            if v is not None:
                mins.append(float(v))
        for k in ("maximum", "exclusiveMaximum"):
            v = getattr(value_schema, k, None)
            if v is not None:
                maxs.append(float(v))
        lo = min(mins) if mins else None
        hi = max(maxs) if maxs else None
        return lo, hi

    def _make_state_inline_control(self, state_field: Any) -> QtWidgets.QWidget:
        name = str(getattr(state_field, "name", "") or "").strip()
        ui = str(getattr(state_field, "uiControl", "") or "").strip().lower()
        schema = getattr(state_field, "valueSchema", None)
        access = getattr(state_field, "access", None)
        access_s = str(getattr(access, "value", access) or "").strip().lower()
        t = (schema_type(schema) or "") if schema is not None else ""

        enum_items = self._schema_enum_items(schema)
        lo, hi = self._schema_numeric_range(schema)
        pool_field = parse_select_pool(ui)
        field_tooltip = str(getattr(state_field, "description", "") or "").strip()

        def _common_style(w: QtWidgets.QWidget) -> None:
            # Make controls readable on dark node themes.
            w.setStyleSheet(
                """
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    color: rgb(235, 235, 235);
                    background: rgba(0, 0, 0, 45);
                    border: 1px solid rgba(255, 255, 255, 55);
                    border-radius: 3px;
                    padding: 1px 4px;
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

        # Create control.
        read_only = access_s == "ro"

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

        if enum_items or pool_field or ui in {"select", "dropdown", "dropbox", "combo", "combobox"}:
            row = F8ExclusiveToggleRow()

            def _pool_items() -> list[str]:
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
                # Allow pools stored as JSON strings (eg. "[]", ["a","b"]). This is common for
                # runtime-updated status fields where schemas are declared as strings.
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

            items = _pool_items() if pool_field else list(enum_items)
            row.set_options(items, labels=items)
            if field_tooltip:
                row.set_context_tooltip(field_tooltip)

            def _apply_value(v: Any) -> None:
                row.set_value("" if v is None else str(v))

            row.valueChanged.connect(lambda v: _set_node_value("" if v is None else str(v), push_undo=True))  # type: ignore[attr-defined]
            _apply_value(_get_node_value())
            self._state_inline_updaters[name] = _apply_value
            if pool_field:
                self._state_inline_option_pools[name] = pool_field
            if read_only:
                row.set_disabled(True)
            return row

        if t == "boolean" or ui in {"switch", "toggle"}:
            row = F8ExclusiveToggleRow()
            row.set_options([True, False], labels=["True", "False"], tooltips=["Set True", "Set False"])
            if field_tooltip:
                row.set_context_tooltip(field_tooltip)

            def _apply_value(v: Any) -> None:
                row.set_value(bool(v) if v is not None else None)

            row.valueChanged.connect(lambda v: _set_node_value(bool(v), push_undo=True))  # type: ignore[attr-defined]
            _apply_value(_get_node_value())
            self._state_inline_updaters[name] = _apply_value
            if read_only:
                row.set_disabled(True)
            return row

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
            spin = QtWidgets.QSpinBox()
            _common_style(spin)
            spin.setMinimumWidth(90)
            if lo is not None:
                spin.setMinimum(int(lo))
            if hi is not None:
                spin.setMaximum(int(hi))
            spin.valueChanged.connect(lambda v: _set_node_value(int(v), push_undo=False))
            spin.editingFinished.connect(lambda: _set_node_value(int(spin.value()), push_undo=True))
            def _apply_value(v: Any) -> None:
                if v is None:
                    return
                with QtCore.QSignalBlocker(spin):
                    try:
                        spin.setValue(int(v))
                    except (TypeError, ValueError):
                        return

            _apply_value(_get_node_value())
            self._state_inline_updaters[name] = _apply_value
            if read_only:
                spin.setDisabled(True)
            return spin

        if t == "number" or ui in {"doublespinbox", "float"}:
            dsp = QtWidgets.QDoubleSpinBox()
            _common_style(dsp)
            dsp.setMinimumWidth(90)
            dsp.setDecimals(6)
            if lo is not None:
                dsp.setMinimum(float(lo))
            if hi is not None:
                dsp.setMaximum(float(hi))
            dsp.valueChanged.connect(lambda v: _set_node_value(float(v), push_undo=False))
            dsp.editingFinished.connect(lambda: _set_node_value(float(dsp.value()), push_undo=True))
            def _apply_value(v: Any) -> None:
                if v is None:
                    return
                with QtCore.QSignalBlocker(dsp):
                    try:
                        dsp.setValue(float(v))
                    except (TypeError, ValueError):
                        return

            _apply_value(_get_node_value())
            self._state_inline_updaters[name] = _apply_value
            if read_only:
                dsp.setDisabled(True)
            return dsp

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
        line.editingFinished.connect(lambda: _set_node_value(line.text(), push_undo=True))
        if read_only:
            line.setDisabled(True)
        return line

    def _ensure_inline_state_widgets(self) -> None:
        self._ensure_graph_property_hook()
        node = self._backend_node()
        if node is None:
            return
        try:
            fields = list(node.effective_state_fields() or [])
        except AttributeError:
            fields = list(getattr(getattr(node, "spec", None), "stateFields", None) or [])

        show: list[tuple[str, Any]] = []
        for f in fields:
            if not getattr(f, "showOnNode", False):
                continue
            n = str(getattr(f, "name", "") or "").strip()
            if n:
                show.append((n, f))

        desired = [n for n, _f in show]

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

        def _ctrl_serial(f: Any) -> str:
            """
            Signature for deciding when the control widget must be rebuilt.
            (Exclude label/description; those can be updated in-place.)
            """
            try:
                vs = getattr(f, "valueSchema", None)
                root = getattr(vs, "root", None) if vs is not None else None
                enum_items = list(getattr(root, "enum", None) or []) if root is not None else []
                return json.dumps(
                    {
                        "access": str(getattr(getattr(f, "access", None), "value", "") or ""),
                        "required": bool(getattr(f, "required", False)),
                        "uiControl": str(getattr(f, "uiControl", "") or ""),
                        "uiLanguage": str(getattr(f, "uiLanguage", "") or ""),
                        "schemaType": str(schema_type(vs) or ""),
                        "enum": [str(x) for x in enum_items],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            except Exception:
                return ""

        for n, f in show:
            # Always keep label/tooltip up to date without rebuilding.
            label = str(getattr(f, "label", "") or "").strip() or n
            tip = str(getattr(f, "description", "") or "").strip() or n
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

            ctrl_sig = _ctrl_serial(f)
            if n in self._state_inline_proxies and ctrl_sig and ctrl_sig == self._state_inline_ctrl_serial.get(n, ""):
                continue

            # Default collapsed; restore persisted expand state from ui overrides.
            expanded = False
            ui = node.ui_overrides() or {}
            store = ui.get("stateInlineExpanded") if isinstance(ui, dict) else None
            if isinstance(store, dict) and n in store:
                expanded = bool(store.get(n))
            expanded = bool(self._state_inline_expanded.get(n, expanded))
            control = self._make_state_inline_control(f)

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
        pen.setCosmetic(self.viewer().get_zoom() < 0.0)
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
        pen.setCosmetic(self.viewer().get_zoom() < 0.0)
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
        self._width, self._height = self.calc_size(add_w, add_h)
        if self._width < NodeEnum.WIDTH.value:
            self._width = NodeEnum.WIDTH.value
        if self._height < NodeEnum.HEIGHT.value:
            self._height = NodeEnum.HEIGHT.value

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
            if self._port_group(str(getattr(port, "name", "") or "")) == "state":
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
            if self._port_group(str(getattr(port, "name", "") or "")) == "state":
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
                    if self._port_group(p.name) == kind:
                        out.append(str(p.name))
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
        # Place regular NodeGraphQt embedded widgets below the command area.
        y = rect.y() + v_offset
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
        spec = getattr(node, "spec", None) if node is not None else None
        try:
            eff_states = list(node.effective_state_fields() or []) if node is not None else []
        except Exception:
            eff_states = list(getattr(spec, "stateFields", None) or []) if spec is not None else []

        # Build ordered port name lists per group.
        exec_in_names: list[str] = []
        exec_out_names: list[str] = []
        if spec is not None:
            for p in list(getattr(spec, "execInPorts", None) or []):
                exec_in_names.append(f"[E]{p}")
            for p in list(getattr(spec, "execOutPorts", None) or []):
                exec_out_names.append(f"{p}[E]")

        data_in_names: list[str] = []
        data_out_names: list[str] = []
        if spec is not None:
            for p in list(getattr(spec, "dataInPorts", None) or []):
                try:
                    data_in_names.append(f"[D]{p.name}")
                except Exception:
                    continue
            for p in list(getattr(spec, "dataOutPorts", None) or []):
                try:
                    data_out_names.append(f"{p.name}[D]")
                except Exception:
                    continue

        state_names: list[str] = []
        for s in eff_states:
            try:
                if not getattr(s, "showOnNode", False):
                    continue
                nm = str(getattr(s, "name", "") or "").strip()
                if nm:
                    state_names.append(nm)
            except Exception:
                continue

        # Fallback when spec is unavailable: keep insertion order but grouped.
        if not exec_in_names:
            exec_in_names = [str(p.name) for p in self._input_items.keys() if self._port_group(p.name) == "exec"]
        if not exec_out_names:
            exec_out_names = [str(p.name) for p in self._output_items.keys() if self._port_group(p.name) == "exec"]
        if not data_in_names:
            data_in_names = [str(p.name) for p in self._input_items.keys() if self._port_group(p.name) == "data"]
        if not data_out_names:
            data_out_names = [str(p.name) for p in self._output_items.keys() if self._port_group(p.name) == "data"]
        if not state_names:
            # Infer state rows from existing ports.
            tmp: list[str] = []
            for p in self._input_items.keys():
                n = str(p.name)
                if n.startswith("[S]"):
                    tmp.append(n[3:])
            for p in self._output_items.keys():
                n = str(p.name)
                if n.endswith("[S]"):
                    tmp.append(n[:-3])
            state_names = [x for x in list(OrderedDict.fromkeys(tmp).keys()) if x]

        other_in_names = [str(p.name) for p in self._input_items.keys() if self._port_group(p.name) == "other"]
        other_out_names = [str(p.name) for p in self._output_items.keys() if self._port_group(p.name) == "other"]

        inputs_by_name = {str(p.name): p for p in self.inputs if p.isVisible()}
        outputs_by_name = {str(p.name): p for p in self.outputs if p.isVisible()}

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
                            header_h = float(max(port_height, self._state_inline_headers[state_key].sizeHint().height()))
                    except Exception:
                        header_h = port_height
                    try:
                        body_w = self._state_inline_bodies.get(state_key)
                        if body_w is not None and body_w.isVisible():
                            body_h = float(max(0.0, body_w.sizeHint().height()))
                    except Exception:
                        body_h = 0.0
                    try:
                        panel_proxy.setPos(inner_x, y)
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
                if self._port_group(str(getattr(port, "name", "") or "")) == "state":
                    text.setVisible(False)
                else:
                    text.setVisible(port.display_name)
        for port, text in self._output_items.items():
            if port.isVisible():
                if self._port_group(str(getattr(port, "name", "") or "")) == "state":
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
        if getattr(self, "_svc_toolbar_proxy", None) is not None:
            return
        service_id = str(getattr(self, "id", "") or "").strip()
        if not service_id:
            return

        def _resolve_graph() -> Any | None:
            # Prefer the viewer passed by NodeGraphQt (more reliable than self.viewer() during init).
            try:
                g = getattr(viewer, "_f8_graph", None) if viewer is not None else None
                if g is not None:
                    return g
            except Exception:
                pass
            try:
                v = self.viewer()
            except Exception:
                v = None
            try:
                return getattr(v, "_f8_graph", None) if v is not None else None
            except Exception:
                return None

        def _get_bridge() -> Any | None:
            g = _resolve_graph()
            try:
                return getattr(g, "service_bridge", None) if g is not None else None
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
                spec = getattr(n, "spec", None)
                return str(getattr(spec, "serviceClass", "") or "")
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
        proxy = getattr(self, "_svc_toolbar_proxy", None)
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

        rect = self.sceneBoundingRect()
        l = self.viewer().mapToGlobal(self.viewer().mapFromScene(rect.topLeft()))
        r = self.viewer().mapToGlobal(self.viewer().mapFromScene(rect.topRight()))
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
            if port.display_name:
                text.setVisible(port_text_visible)

        # output port text visibility.
        for port, text in self._output_items.items():
            if port.display_name:
                text.setVisible(port_text_visible)

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
