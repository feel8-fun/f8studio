from __future__ import annotations

import logging
from typing import Any

from .node_base import F8StudioBaseNode

from f8pysdk import F8OperatorSpec, F8StateAccess
from f8pysdk.schema_helpers import schema_default, schema_type

from qtpy import QtCore, QtWidgets

from NodeGraphQt.constants import (
    Z_VAL_NODE,
    NodePropWidgetEnum,
)

from .items.operator_legacy_node_item import _LegacyF8StudioOperatorNodeItem
from .port_painter import draw_exec_port, draw_square_port, EXEC_PORT_COLOR, DATA_PORT_COLOR, STATE_PORT_COLOR
from .service_basenode import F8StudioServiceNodeItem

logger = logging.getLogger(__name__)


class F8StudioOperatorBaseNode(F8StudioBaseNode):
    """
    Base class for all operator nodes (nodes that are intended to live inside
    a container).

    This class is intentionally small: container binding is orchestrated by
    `F8StudioGraph`, while the view-level `_container_item` link is managed by
    the container item.
    """

    svcId: Any

    def __init__(self, qgraphics_item=None):
        _nodeitem_cls = qgraphics_item or F8StudioOperatorNodeItem
        assert issubclass(
            _nodeitem_cls, F8StudioOperatorNodeItem
        ), "F8StudioOperatorBaseNode requires a F8StudioOperatorNodeItem or subclass."
        super().__init__(qgraphics_item=_nodeitem_cls)
        assert isinstance(self.spec, F8OperatorSpec), "F8StudioOperatorBaseNode requires F8OperatorSpec"

        self.set_port_deletion_allowed(True)

        self._build_exec_port()
        self._build_data_port()
        self._build_state_port()
        self._build_state_properties()

    def _build_exec_port(self):
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
            name = str(s.name or "").strip()
            if not name or not bool(s.showOnNode):
                continue

            if s.access in (F8StateAccess.rw, F8StateAccess.wo):
                self.add_input(
                    f"[S]{name}",
                    color=STATE_PORT_COLOR,
                    painter_func=draw_square_port,
                )

            if s.access in (F8StateAccess.rw, F8StateAccess.ro):
                self.add_output(
                    f"{name}[S]",
                    color=STATE_PORT_COLOR,
                    painter_func=draw_square_port,
                )

    def _build_state_properties(self) -> None:
        for s in self.effective_state_fields() or []:
            name = str(s.name or "").strip()
            if not name:
                continue
            try:
                if self.has_property(name):  # type: ignore[attr-defined]
                    continue
            except (AttributeError, RuntimeError, TypeError):
                pass
            try:
                default_value = schema_default(s.valueSchema)
            except Exception:
                default_value = None
            widget_type, items, prop_range = self._state_widget_for_schema(s.valueSchema)
            tooltip = str(s.description or "").strip() or None
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
            except Exception as exc:
                logger.warning("Failed to create operator state property '%s': %s", name, exc)
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
        try:
            if not self.port_deletion_allowed():
                self.set_port_deletion_allowed(True)
        except (AttributeError, RuntimeError, TypeError):
            pass

        # Sync ports from spec.
        #
        # Important: NodeGraphQt `delete_input/delete_output` does not clear
        # pipes. If ports are removed while still connected, NodeGraphQt can
        # leave "dangling" pipes in the scene, crashing during paint.
        desired_inputs: dict[str, dict[str, Any]] = {}
        desired_outputs: dict[str, dict[str, Any]] = {}

        for p in list(self.spec.execInPorts or []):
            desired_inputs[f"[E]{p}"] = {"color": EXEC_PORT_COLOR, "painter_func": draw_exec_port}
        for p in list(self.spec.execOutPorts or []):
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
                except (AttributeError, RuntimeError, TypeError):
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
                except (AttributeError, RuntimeError, TypeError):
                    pass
            if show_on_node:
                desired_outputs[port_name] = {"color": DATA_PORT_COLOR}

        for s in list(self.effective_state_fields() or []):
            name = str(s.name or "").strip()
            if not name or not bool(s.showOnNode):
                continue
            if s.access in (F8StateAccess.rw, F8StateAccess.wo):
                desired_inputs[f"[S]{name}"] = {"color": STATE_PORT_COLOR, "painter_func": draw_square_port}
            if s.access in (F8StateAccess.rw, F8StateAccess.ro):
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
                    except (AttributeError, RuntimeError, TypeError):
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
                    except (AttributeError, RuntimeError, TypeError):
                        pass
        except Exception:
            logger.exception("Failed cleanup for orphaned operator port graphics")

        self._build_state_properties()

        try:
            self.view.draw_node()
        except Exception:
            logger.exception("Failed to redraw operator node after sync_from_spec")

class F8StudioOperatorNodeItem(F8StudioServiceNodeItem):
    """
    Operator node item: reuse the service-node layout (grouped ports + inline collapsible
    state widgets + persisted expand state), but without service process controls.

    This intentionally disables:
    - service process toolbar
    - service command buttons
    """

    def __init__(self, name="node", parent=None):
        super().__init__(name, parent)
        # Operator nodes may be canvas-managed (no container). Containers bind by
        # setting `view._container_item`; keep it always defined to avoid crashes
        # during interactive moves before binding.
        self._container_item = None

    def itemChange(self, change, value):  # type: ignore[override]
        """
        Keep legacy operator behaviors:
        - highlight pipes on selection
        - clamp operator nodes to container bounds while dragging
        """
        if change == QtWidgets.QGraphicsItem.ItemSelectedChange and self.scene():
            try:
                self.reset_pipes()
                if value:
                    self.highlight_pipes()
            except (AttributeError, RuntimeError, TypeError):
                pass
            try:
                self.setZValue(Z_VAL_NODE)
                if not self.selected:
                    self.setZValue(Z_VAL_NODE + 1)
            except (AttributeError, RuntimeError, TypeError):
                pass

        if change == QtWidgets.QGraphicsItem.ItemPositionChange and self.scene():
            return self._clamp_pos_to_container(value)

        return super().itemChange(change, value)

    def _clamp_pos_to_container(self, proposed_pos: QtCore.QPointF) -> QtCore.QPointF:
        """
        Clamp the node's top-left position so the entire node stays within
        its service container bounds (in scene coordinates).
        """
        container = self._container_item
        if container is None:
            return proposed_pos

        container_rect = container.mapToScene(container.boundingRect()).boundingRect()
        padding = 2.0
        container_rect = container_rect.adjusted(padding, padding, -padding, -padding)

        brect = self.boundingRect()
        node_w = brect.width()
        node_h = brect.height()

        x_min = container_rect.left()
        y_min = container_rect.top()
        x_max = container_rect.right() - node_w
        y_max = container_rect.bottom() - node_h

        x_new = x_min if x_max < x_min else min(max(proposed_pos.x(), x_min), x_max)
        y_new = y_min if y_max < y_min else min(max(proposed_pos.y(), y_min), y_max)

        return QtCore.QPointF(x_new, y_new)

    def _ensure_service_toolbar(self, viewer: Any | None) -> None:  # type: ignore[override]
        return

    def _position_service_toolbar(self) -> None:  # type: ignore[override]
        return

    def _ensure_inline_command_widget(self) -> None:  # type: ignore[override]
        # Operators don't expose service commands; ensure any previous command proxy is removed.
        proxy = self._cmd_proxy
        if proxy is not None:
            try:
                proxy.setWidget(None)
            except (AttributeError, RuntimeError, TypeError):
                pass
            try:
                proxy.setParentItem(None)
                if self.scene() is not None:
                    self.scene().removeItem(proxy)
            except (AttributeError, RuntimeError, TypeError):
                pass
        self._cmd_proxy = None
        self._cmd_widget = None
        self._cmd_buttons = []
