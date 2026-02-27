from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Generic

from qtpy import QtCore, QtWidgets, QtGui
from NodeGraphQt import NodeGraph, BaseNode
from NodeGraphQt.constants import PortTypeEnum
from NodeGraphQt.errors import NodeCreationError, NodeDeletionError
from NodeGraphQt.base.commands import NodeAddedCmd, NodeMovedCmd, NodesRemovedCmd, PortConnectedCmd
from NodeGraphQt.base.port import Port as NGPort
import shortuuid
import logging

from f8pysdk import F8OperatorSpec, F8ServiceSpec
from f8pysdk.nats_naming import ensure_token
from .container_basenode import F8StudioContainerBaseNode
from .operator_basenode import F8StudioOperatorBaseNode
from .service_basenode import F8StudioServiceNodeItem

from .viewer import F8StudioNodeViewer
from .service_bridge_protocol import ServiceBridge
from .session import last_session_path
from .spec_visibility import is_hidden_spec_node_class
from ..session_migration import extract_layout as _extract_session_layout
from ..session_migration import wrap_layout_for_save as _wrap_layout_for_save
from ..ui_notifications import show_info, show_warning
from ..variants.variant_ids import build_variant_node_type, parse_variant_node_type
from ..variants.variant_repository import load_library
from ..variants.variant_compose import build_variant_record_from_node
from ..variants.variant_repository import upsert_variant

from ..constants import SERVICE_CLASS as _CANVAS_SERVICE_CLASS_
from ..constants import STUDIO_SERVICE_ID
from .edge_rules import (
    EdgeRuleNodeInfo,
    layout_node_info,
    normalize_edge_kind,
    port_view_name,
    validate_layout_connection,
    validate_runtime_connection,
)

_BASE_OPERATOR_CLS_ = F8StudioOperatorBaseNode
_BASE_CONTAINER_CLS_ = F8StudioContainerBaseNode
MISSING_SERVICE_NODE_TYPE = "svc.f8.missing.service"
MISSING_OPERATOR_NODE_TYPE = "svc.f8.missing.operator"
logger = logging.getLogger(__name__)


def _scene_rect(node: BaseNode) -> QtCore.QRectF | None:
    return node.view.sceneBoundingRect()


def _rect_at_pos(item: QtWidgets.QGraphicsItem, pos: list[float] | tuple[float, float]) -> QtCore.QRectF:
    """
    Compute a "scene-like" rect for an item positioned at `pos` (top-left),
    without requiring the item to be in a scene.
    """
    brect = item.boundingRect()
    return QtCore.QRectF(float(pos[0]), float(pos[1]), brect.width(), brect.height())


class F8StudioGraph(NodeGraph):
    """Main F8PyStudio controller class."""
    node_placement_changed = QtCore.Signal(bool, str)

    def __init__(self, parent=None, **kwargs):
        """
        Args:
            parent (object): object parent.
            **kwargs (dict): Used for overriding internal objects at init time.
        """
        # Use a custom viewer to support keyboard shortcuts (Tab search, Delete).
        undo_stack = kwargs.get("undo_stack") or QtGui.QUndoStack(parent)
        viewer = kwargs.get("viewer") or F8StudioNodeViewer(undo_stack=undo_stack)

        kwargs["undo_stack"] = undo_stack
        kwargs["viewer"] = viewer
        super().__init__(parent, **kwargs)
        viewer.set_graph(self)
        viewer.node_placement_changed.connect(self._on_viewer_node_placement_changed)  # type: ignore[attr-defined]

        self.uuid_length = kwargs.get("uuid_length", 4)
        self.uuid_generator = shortuuid.ShortUUID()
        self._loading_session = False
        # Tab search sends a selected "node type" string. We map display aliases
        # back to actual factory node type ids so menu category paths can be custom.
        self._tab_search_node_type_aliases: dict[str, str] = {}
        self._variant_menu_node_types: set[str] = set()
        self._identity_menu_node_types: set[str] = set()

        self.property_changed.connect(self._on_property_changed)  # type: ignore[attr-defined]

        # Optional bridge used by UI widgets to control local service processes.
        self._service_bridge: ServiceBridge | None = None
        # Debounced reclaim timers for removed service instances (serviceId -> QTimer).
        self._reclaim_timers: dict[str, QtCore.QTimer] = {}

        # NodeGraphQt exposes `nodes_deleted` (list[str]), not `node_deleted`.
        self.nodes_deleted.connect(self._on_nodes_deleted)  # type: ignore[attr-defined]

        # Keep inline state widgets in sync with upstream bindings.
        self.port_connected.connect(self._on_port_connected)  # type: ignore[attr-defined]
        self.port_disconnected.connect(self._on_port_disconnected)  # type: ignore[attr-defined]

    def _on_viewer_node_placement_changed(self, active: bool, label: str) -> None:
        self.node_placement_changed.emit(bool(active), str(label or ""))

    def _notification_parent(self) -> QtWidgets.QWidget | None:
        viewer = self.viewer()
        if viewer is None:
            return None
        window = viewer.window()
        if isinstance(window, QtWidgets.QWidget):
            return window
        return viewer

    def _prompt_variant_metadata(
        self,
        *,
        default_name: str,
        default_description: str,
        default_tags: list[str],
    ) -> tuple[str, str, list[str]] | None:
        dialog = QtWidgets.QDialog(None)
        dialog.setWindowTitle("Save Node As Variant")
        dialog.resize(520, 220)

        name_edit = QtWidgets.QLineEdit(default_name, dialog)
        desc_edit = QtWidgets.QLineEdit(default_description, dialog)
        tags_edit = QtWidgets.QLineEdit(", ".join(default_tags), dialog)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", name_edit)
        form.addRow("Description", desc_edit)
        form.addRow("Tags (comma-separated)", tags_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)  # type: ignore[attr-defined]
        buttons.rejected.connect(dialog.reject)  # type: ignore[attr-defined]

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addLayout(form)
        layout.addWidget(buttons)

        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None
        name = str(name_edit.text() or "").strip()
        if not name:
            return None
        description = str(desc_edit.text() or "").strip()
        tags = [s.strip() for s in str(tags_edit.text() or "").split(",")]
        return name, description, [t for t in tags if t]

    def _save_node_as_variant(self, node: Any) -> None:
        if node is None:
            return
        try:
            spec = node.spec
        except AttributeError:
            return
        if not isinstance(spec, (F8OperatorSpec, F8ServiceSpec)):
            return
        node_display_name = ""
        try:
            node_display_name = str(node.name() or "").strip()
        except (AttributeError, RuntimeError, TypeError):
            node_display_name = ""
        default_name = str(node_display_name or node.NODE_NAME or spec.label or "").strip() or "Variant"
        default_desc = str(spec.description or "").strip()
        default_tags = [str(t) for t in list(spec.tags or []) if str(t).strip()]
        values = self._prompt_variant_metadata(
            default_name=default_name,
            default_description=default_desc,
            default_tags=default_tags,
        )
        if values is None:
            return
        name, description, tags = values
        record = build_variant_record_from_node(node=node, name=name, description=description, tags=tags)
        upsert_variant(record)
        show_info(self._notification_parent(), "Variant Saved", f"Saved variant:\n{name}")

    def _on_save_variant_menu_action(self, graph: Any, node: Any) -> None:
        _ = graph
        self._save_node_as_variant(node)

    def install_variant_context_menu_for_nodes(self, node_classes: list[type]) -> None:
        nodes_menu = self.context_nodes_menu()
        if nodes_menu is None:
            return
        for node_cls in list(node_classes or []):
            node_type = str(node_cls.type_ or "")
            if not node_type or node_type in self._variant_menu_node_types:
                continue
            nodes_menu.add_command(
                "Save As Variant...",
                func=self._on_save_variant_menu_action,
                node_type=node_type,
            )
            self._variant_menu_node_types.add(node_type)

    def install_identity_context_menu_for_nodes(self, node_classes: list[type]) -> None:
        nodes_menu = self.context_nodes_menu()
        if nodes_menu is None:
            return
        for node_cls in list(node_classes or []):
            node_type = str(node_cls.type_ or "")
            if not node_type or node_type in self._identity_menu_node_types:
                continue
            nodes_menu.add_command(
                self.tr("Rename Id..."),
                func=self._on_rename_id_menu_action,
                node_type=node_type,
            )
            self._identity_menu_node_types.add(node_type)

    def _on_rename_id_menu_action(self, graph: Any, node: Any) -> None:
        _ = graph
        if not isinstance(node, BaseNode):
            return
        try:
            spec = node.spec  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError, TypeError):
            return
        if not isinstance(spec, (F8OperatorSpec, F8ServiceSpec)):
            return

        ok_stop, stop_msg = self._is_node_rename_allowed_when_stopped(node)
        if not ok_stop:
            show_warning(self._notification_parent(), self.tr("Rename Id Failed"), stop_msg)
            return

        title = self.tr("Rename ServiceId") if isinstance(spec, F8ServiceSpec) else self.tr("Rename OperatorId")
        label = self.tr("Id Name:")
        current_id = str(node.id or "").strip()
        new_id = self._prompt_id_rename_dialog(title=title, label=label, value=current_id)
        if new_id is None:
            return

        ok_id, id_msg = self._validate_new_node_id(node=node, new_id=new_id)
        if not ok_id:
            show_warning(self._notification_parent(), self.tr("Rename Id Failed"), id_msg)
            return

        ok_rename, rename_msg = self._rename_node_identity(node=node, new_id=new_id)
        if not ok_rename:
            show_warning(self._notification_parent(), self.tr("Rename Id Failed"), rename_msg)

    @staticmethod
    def _prompt_id_rename_dialog(*, title: str, label: str, value: str) -> str | None:
        dialog = QtWidgets.QDialog(None)
        dialog.setWindowTitle(str(title or "Rename Id"))
        dialog.resize(420, 100)

        id_edit = QtWidgets.QLineEdit(str(value or ""), dialog)
        form = QtWidgets.QFormLayout()
        form.addRow(str(label or "Id名字:"), id_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)  # type: ignore[attr-defined]
        buttons.rejected.connect(dialog.reject)  # type: ignore[attr-defined]

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addLayout(form)
        layout.addWidget(buttons)

        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return None
        out = str(id_edit.text() or "").strip()
        if not out:
            return None
        return out

    def _validate_new_node_id(self, *, node: BaseNode, new_id: str) -> tuple[bool, str]:
        candidate = str(new_id or "").strip()
        if not candidate:
            return False, self.tr("Id cannot be empty.")
        try:
            ensure_token(candidate, label="node_id")
        except ValueError as exc:
            return False, self.tr("Invalid Id format: {}").format(exc)

        old_id = str(node.id or "").strip()
        if candidate == old_id:
            return False, self.tr("New Id is the same as the current Id.")

        try:
            spec = node.spec  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError, TypeError):
            spec = None
        if isinstance(spec, F8ServiceSpec) and old_id == STUDIO_SERVICE_ID:
            return False, self.tr("Built-in service Id `{}` cannot be renamed.").format(STUDIO_SERVICE_ID)

        existing = self.get_node_by_id(candidate)
        if existing is not None and existing is not node:
            return False, self.tr("Id `{}` already exists.").format(candidate)
        return True, ""

    def _is_node_rename_allowed_when_stopped(self, node: BaseNode) -> tuple[bool, str]:
        try:
            spec = node.spec  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError, TypeError):
            return False, self.tr("This node type does not support Id rename.")
        bridge = self._service_bridge
        if bridge is None:
            return True, ""

        if isinstance(spec, F8ServiceSpec):
            service_id = str(node.id or "").strip()
            if not service_id:
                return False, self.tr("Current service node is missing serviceId.")
            try:
                running = bool(bridge.is_service_running(service_id))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                running = False
            if running:
                return False, self.tr("Service `{}` is running. Stop it before renaming Id.").format(service_id)
            return True, ""

        if isinstance(spec, F8OperatorSpec):
            try:
                service_id = str(node.svcId or "").strip()  # type: ignore[attr-defined]
            except (AttributeError, RuntimeError, TypeError):
                service_id = ""
            if not service_id:
                return False, self.tr("Operator is not bound to a serviceId.")
            try:
                running = bool(bridge.is_service_running(service_id))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                running = False
            if running:
                return False, self.tr("Operator service `{}` is running. Stop it before renaming Id.").format(
                    service_id
                )
            return True, ""

        return False, self.tr("This node type does not support Id rename.")

    def _update_node_id_mapping(self, *, node: BaseNode, old_id: str, new_id: str) -> None:
        model_nodes = self.model.nodes
        if old_id:
            model_nodes.pop(old_id, None)
        model_nodes[str(new_id)] = node

    def _rewrite_connected_port_node_id_references(self, *, old_id: str, new_id: str) -> None:
        src = str(old_id or "").strip()
        dst = str(new_id or "").strip()
        if not src or not dst or src == dst:
            return
        for node in list(self.all_nodes() or []):
            ports = list(node.input_ports() or []) + list(node.output_ports() or [])
            for port in ports:
                connected = port.model.connected_ports
                if src not in connected:
                    continue
                moved = list(connected.pop(src) or [])
                if not moved:
                    continue
                dst_ports = connected.setdefault(dst, [])
                for name in moved:
                    if name not in dst_ports:
                        dst_ports.append(name)

    def repair_stale_port_connection_refs(self) -> int:
        """
        Drop connected-port references whose node_id is no longer present in graph.

        Returns:
            int: number of stale refs removed.
        """
        valid_node_ids = {str(n.id or "").strip() for n in list(self.all_nodes() or []) if str(n.id or "").strip()}
        removed = 0
        for node in list(self.all_nodes() or []):
            ports = list(node.input_ports() or []) + list(node.output_ports() or [])
            for port in ports:
                connected = port.model.connected_ports
                stale_ids = [nid for nid in list(connected.keys()) if str(nid or "") not in valid_node_ids]
                for stale_id in stale_ids:
                    removed += len(list(connected.pop(stale_id, []) or []))
        if removed:
            logger.warning("Repaired %s stale port connection reference(s).", removed)
        return removed

    def _cascade_service_id_to_bound_operators(
        self,
        *,
        old_service_id: str,
        new_service_id: str,
        service_class: str,
    ) -> None:
        old_sid = str(old_service_id or "").strip()
        new_sid = str(new_service_id or "").strip()
        expected_service_class = str(service_class or "")
        if not old_sid or not new_sid or old_sid == new_sid:
            return
        for n in list(self.all_nodes() or []):
            if not self._is_operator_node(n):
                continue
            try:
                op_spec = n.spec
            except (AttributeError, RuntimeError, TypeError):
                continue
            if not isinstance(op_spec, F8OperatorSpec):
                continue
            try:
                op_svc_id = str(n.svcId or "").strip()
            except (AttributeError, RuntimeError, TypeError):
                op_svc_id = ""
            if op_svc_id != old_sid:
                continue
            if str(op_spec.serviceClass or "") != expected_service_class:
                continue
            try:
                n.svcId = new_sid  # type: ignore[attr-defined]
            except (AttributeError, RuntimeError, TypeError):
                continue
            try:
                if "svcId" in n.model.properties or "svcId" in n.model.custom_properties:
                    n.set_property("svcId", new_sid, push_undo=False)
            except (AttributeError, RuntimeError, TypeError):
                pass

    @staticmethod
    def _refresh_service_identity_bindings(node: BaseNode) -> None:
        view = node.view
        try:
            refresh = view.refresh_service_identity_bindings  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError, TypeError):
            refresh = None
        if callable(refresh):
            try:
                refresh()
            except (AttributeError, RuntimeError, TypeError):
                pass
        try:
            view.draw_node()
        except (AttributeError, RuntimeError, TypeError):
            try:
                view.update()
            except (AttributeError, RuntimeError, TypeError):
                pass

    def _rename_node_identity(self, *, node: BaseNode, new_id: str) -> tuple[bool, str]:
        old_id = str(node.id or "").strip()
        nid = str(new_id or "").strip()
        if not old_id or not nid:
            return False, self.tr("Invalid node Id.")
        if old_id == nid:
            return False, self.tr("New Id is the same as the current Id.")

        try:
            spec = node.spec  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError, TypeError):
            return False, self.tr("This node type does not support Id rename.")

        # NodeGraphQt stores connections as {node_id: [port_name]} on each port model.
        # Rewrite these refs before replacing the model node-id mapping.
        self._rewrite_connected_port_node_id_references(old_id=old_id, new_id=nid)
        self._update_node_id_mapping(node=node, old_id=old_id, new_id=nid)
        node.model.id = nid
        node.view.id = nid
        self.repair_stale_port_connection_refs()

        if isinstance(spec, F8OperatorSpec):
            try:
                if "operatorId" in node.model.properties or "operatorId" in node.model.custom_properties:
                    node.set_property("operatorId", nid, push_undo=False)
            except (AttributeError, RuntimeError, TypeError):
                pass
            return True, ""

        if isinstance(spec, F8ServiceSpec):
            try:
                node.svcId = nid  # type: ignore[attr-defined]
            except (AttributeError, RuntimeError, TypeError):
                pass
            try:
                if "svcId" in node.model.properties or "svcId" in node.model.custom_properties:
                    node.set_property("svcId", nid, push_undo=False)
            except (AttributeError, RuntimeError, TypeError):
                pass
            self._cascade_service_id_to_bound_operators(
                old_service_id=old_id,
                new_service_id=nid,
                service_class=str(spec.serviceClass or ""),
            )
            old_timer = self._reclaim_timers.pop(old_id, None)
            if old_timer is not None:
                remaining_ms = -1
                try:
                    remaining_ms = int(old_timer.remainingTime())
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    remaining_ms = -1
                try:
                    old_timer.stop()
                    old_timer.deleteLater()
                except (AttributeError, RuntimeError, TypeError):
                    pass
                new_timer = QtCore.QTimer(self)
                new_timer.setSingleShot(True)
                new_timer.timeout.connect(lambda _sid=nid: self._reclaim_service_if_unreferenced(_sid))  # type: ignore[attr-defined]
                self._reclaim_timers[nid] = new_timer
                if remaining_ms > 0:
                    try:
                        new_timer.start(remaining_ms)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass
            self._refresh_service_identity_bindings(node)
            return True, ""

        return False, self.tr("This node type does not support Id rename.")

    def _on_port_connected(self, in_port: NGPort, out_port: NGPort) -> None:
        self._on_port_connection_changed(in_port=in_port, out_port=out_port)

    def _on_port_disconnected(self, in_port: NGPort, out_port: NGPort) -> None:
        self._on_port_connection_changed(in_port=in_port, out_port=out_port)

    def set_edge_kind_visible(self, kind: str, visible: bool) -> None:
        normalized = normalize_edge_kind(kind)
        if normalized is None:
            raise ValueError(f"unknown edge kind: {kind}")
        viewer = self._viewer
        if not isinstance(viewer, F8StudioNodeViewer):
            return
        viewer.set_edge_kind_visible(normalized, bool(visible))

    def edge_kind_visible(self, kind: str) -> bool:
        normalized = normalize_edge_kind(kind)
        if normalized is None:
            raise ValueError(f"unknown edge kind: {kind}")
        viewer = self._viewer
        if not isinstance(viewer, F8StudioNodeViewer):
            return True
        return bool(viewer.edge_kind_visible(normalized))

    @staticmethod
    def _ordered_port_views(port_a: Any, port_b: Any) -> tuple[Any, Any] | None:
        if port_a.port_type == PortTypeEnum.OUT.value and port_b.port_type == PortTypeEnum.IN.value:
            return port_a, port_b
        if port_b.port_type == PortTypeEnum.OUT.value and port_a.port_type == PortTypeEnum.IN.value:
            return port_b, port_a
        return None

    def _connection_views_allowed(self, port_a: Any, port_b: Any) -> tuple[bool, str]:
        ordered = self._ordered_port_views(port_a, port_b)
        if ordered is None:
            return False, "connection must be between output and input ports"
        out_view, in_view = ordered
        out_node_id = str(out_view.node.id or "").strip()
        in_node_id = str(in_view.node.id or "").strip()
        if not out_node_id or not in_node_id:
            return False, "connection endpoints are missing node ids"

        try:
            out_node = self.get_node_by_id(out_node_id)
            in_node = self.get_node_by_id(in_node_id)
        except (AttributeError, KeyError, RuntimeError, TypeError):
            return False, "connection endpoint nodes not found"
        if out_node is None or in_node is None:
            return False, "connection endpoint nodes not found"

        return validate_runtime_connection(
            out_port_name=port_view_name(out_view),
            in_port_name=port_view_name(in_view),
            out_node=out_node,
            in_node=in_node,
        )

    def _on_connection_changed(self, disconnected, connected):  # type: ignore[override]
        if not (disconnected or connected):
            return

        valid_connected = []
        rejected_count = 0
        for pair in list(connected or []):
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                rejected_count += 1
                continue
            allowed, reason = self._connection_views_allowed(pair[0], pair[1])
            if not allowed:
                rejected_count += 1
                logger.warning("Rejected invalid connection: %s", reason)
                continue
            valid_connected.append(pair)

        if rejected_count:
            logger.warning("Rejected %s invalid connection(s) by studio edge rules.", rejected_count)

        valid_disconnected = list(disconnected or [])
        if list(connected or []):
            if not valid_connected:
                valid_disconnected = []
            else:
                endpoints: set[Any] = set()
                for a, b in valid_connected:
                    endpoints.add(a)
                    endpoints.add(b)
                filtered = []
                for pair in list(disconnected or []):
                    if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                        continue
                    if pair[0] in endpoints or pair[1] in endpoints:
                        filtered.append(pair)
                valid_disconnected = filtered
        super()._on_connection_changed(valid_disconnected, valid_connected)

    def begin_node_placement(self, node_type: str, node_label: str) -> None:
        viewer = self._viewer
        if isinstance(viewer, F8StudioNodeViewer):
            viewer.begin_node_placement(node_type=node_type, node_label=node_label)

    def cancel_node_placement(self) -> None:
        viewer = self._viewer
        if isinstance(viewer, F8StudioNodeViewer):
            viewer.cancel_node_placement()

    @staticmethod
    def _is_state_port(port: NGPort) -> bool:
        name = str(port.name() or "")
        return name.startswith("[S]") or name.endswith("[S]")

    def _on_port_connection_changed(self, *, in_port: NGPort, out_port: NGPort) -> None:
        """
        Refresh inline state read-only state when state edges are connected/disconnected.

        When a state field is upstream-bound via a state edge, Studio should treat it as
        read-only in the node UI (inline controls).
        """
        if not (self._is_state_port(in_port) or self._is_state_port(out_port)):
            return

        for p in (in_port, out_port):
            if not self._is_state_port(p):
                continue
            node = p.node()
            view = node.view
            if not isinstance(view, F8StudioServiceNodeItem):
                continue
            view.refresh_inline_state_read_only()
            view.update()

    def set_service_bridge(self, bridge: ServiceBridge | None) -> None:
        self._service_bridge = bridge

    @property
    def service_bridge(self) -> ServiceBridge | None:
        return self._service_bridge

    @staticmethod
    def _strip_port_restore_data(layout_data: dict) -> dict:
        """
        NodeGraphQt session format stores `port_deletion_allowed` plus
        `input_ports`/`output_ports` when ports are removable.

        Loading then calls `node.set_ports(...)`, which rebuilds ports without
        our custom styling (color / custom port painter). Studio nodes already
        define their ports from `spec` in `__init__`, so we strip these keys and
        let nodes rebuild themselves via the node factory.
        """
        nodes = layout_data.get("nodes")
        if not isinstance(nodes, dict):
            return layout_data
        for n_data in nodes.values():
            if not isinstance(n_data, dict):
                continue
            n_data.pop("port_deletion_allowed", None)
            n_data.pop("input_ports", None)
            n_data.pop("output_ports", None)
        return layout_data

    @staticmethod
    def _strip_invalid_connections(layout_data: dict) -> dict:
        """
        Remove connections that reference ports not defined by the node spec.

        This prevents NodeGraphQt from creating "dangling" pipes that later crash
        during paint when ports/nodes are missing (eg. when a state field changes
        `showOnNode` and ports are no longer created).
        """
        nodes = layout_data.get("nodes")
        conns = layout_data.get("connections")
        if not isinstance(nodes, dict) or not isinstance(conns, list):
            return layout_data

        def _coerce_spec(v: object) -> F8OperatorSpec | F8ServiceSpec | None:
            if v is None:
                return None
            if isinstance(v, (F8OperatorSpec, F8ServiceSpec)):
                return v
            if isinstance(v, dict):
                try:
                    if "operatorClass" in v:
                        return F8OperatorSpec.model_validate(v)
                    return F8ServiceSpec.model_validate(v)
                except Exception:
                    return None
            return None

        port_sets: dict[str, set[str] | None] = {}
        node_info_by_id: dict[str, EdgeRuleNodeInfo | None] = {}
        for node_id, node_data in nodes.items():
            node_id_str = str(node_id)
            if not isinstance(node_data, dict):
                port_sets[node_id_str] = None
                node_info_by_id[node_id_str] = None
                continue
            node_info_by_id[node_id_str] = layout_node_info(node_id_str, node_data)
            spec = _coerce_spec(node_data.get("f8_spec"))
            if spec is None:
                port_sets[node_id_str] = None
                continue

            # Apply UI overrides (eg. showOnNode) so we can strip connections
            # referencing ports that will not be created.
            state_fields = list(spec.stateFields or [])
            ui = node_data.get("f8_ui")
            state_ui = None
            if isinstance(ui, dict):
                state_ui = ui.get("stateFields")
            if isinstance(state_ui, dict) and state_ui and state_fields:
                allowed_keys = {"showOnNode", "uiControl", "uiLanguage", "label", "description"}
                patched = []
                for f in state_fields:
                    name = str(f.name or "").strip()
                    ov = state_ui.get(name) if name else None
                    if not isinstance(ov, dict) or not ov:
                        patched.append(f)
                        continue
                    patch = {k: ov.get(k) for k in allowed_keys if k in ov}
                    try:
                        patched.append(f.model_copy(update=patch))
                    except Exception:
                        patched.append(f)
                state_fields = patched

            ports: set[str] = set()
            if isinstance(spec, F8OperatorSpec):
                for p in list(spec.execInPorts or []):
                    ports.add(f"[E]{p}")
                for p in list(spec.execOutPorts or []):
                    ports.add(f"{p}[E]")
            for p in list(spec.dataInPorts or []):
                try:
                    ports.add(f"[D]{p.name}")
                except (AttributeError, TypeError):
                    continue
            for p in list(spec.dataOutPorts or []):
                try:
                    ports.add(f"{p.name}[D]")
                except (AttributeError, TypeError):
                    continue
            for s in state_fields:
                try:
                    if not bool(s.showOnNode):
                        continue
                    name = str(s.name or "").strip()
                    if not name:
                        continue
                    ports.add(f"[S]{name}")
                    ports.add(f"{name}[S]")
                except (AttributeError, TypeError):
                    continue
            port_sets[node_id_str] = ports

        kept: list[dict[str, Any]] = []
        dropped = 0
        rule_dropped = 0
        for c in conns:
            if not isinstance(c, dict):
                dropped += 1
                continue
            out_ref = c.get("out")
            in_ref = c.get("in")
            if not (isinstance(out_ref, (list, tuple)) and len(out_ref) == 2 and isinstance(in_ref, (list, tuple)) and len(in_ref) == 2):
                dropped += 1
                continue
            out_nid, out_port = str(out_ref[0]), str(out_ref[1])
            in_nid, in_port = str(in_ref[0]), str(in_ref[1])
            if out_nid not in nodes or in_nid not in nodes:
                dropped += 1
                continue
            out_ports = port_sets.get(out_nid)
            in_ports = port_sets.get(in_nid)
            if out_ports is not None and out_port not in out_ports:
                dropped += 1
                continue
            if in_ports is not None and in_port not in in_ports:
                dropped += 1
                continue

            allowed, _reason = validate_layout_connection(
                out_node_id=out_nid,
                out_port_name=out_port,
                in_node_id=in_nid,
                in_port_name=in_port,
                node_info_by_id=node_info_by_id,
            )
            if not allowed:
                dropped += 1
                rule_dropped += 1
                continue
            kept.append(c)

        if dropped:
            logger.warning(
                "Stripped %s invalid session connection(s) (%s rule-violating).",
                dropped,
                rule_dropped,
            )
        layout_data["connections"] = kept
        return layout_data

    def _merge_session_specs(self, layout_data: dict) -> dict:
        """
        Merge session-stored `f8_spec` with the library default spec template.

        - Reject load when identity fields conflict (eg. operatorClass/serviceClass).
        - Allow session to override *editable* lists (eg. stateFields) when the
          template enables editing for that category.
        """
        nodes = layout_data.get("nodes")
        if not isinstance(nodes, dict):
            return layout_data

        def _coerce_spec(v: object) -> F8OperatorSpec | F8ServiceSpec | None:
            if v is None:
                return None
            if isinstance(v, (F8OperatorSpec, F8ServiceSpec)):
                return v
            if isinstance(v, dict):
                try:
                    if "operatorClass" in v:
                        return F8OperatorSpec.model_validate(v)
                    return F8ServiceSpec.model_validate(v)
                except Exception:
                    return None
            return None

        def _enum_str(v: object) -> str | None:
            if v is None:
                return None
            try:
                import enum

                if isinstance(v, enum.Enum):
                    return str(v.value)
                return str(v)
            except Exception:
                return None

        errors: list[str] = []

        for node_id, node_data in nodes.items():
            if not isinstance(node_data, dict):
                continue
            f8_sys = node_data.get("f8_sys")
            if isinstance(f8_sys, dict) and bool(f8_sys.get("missingLocked")):
                continue

            node_type = node_data.get("type_")
            if not isinstance(node_type, str) or not node_type.strip():
                continue

            node_cls = self._node_factory.nodes.get(node_type.strip())
            if node_cls is None:
                continue

            try:
                template_spec = _coerce_spec(node_cls.SPEC_TEMPLATE)  # type: ignore[attr-defined]
            except Exception:
                template_spec = None
            session_spec_raw = node_data.get("f8_spec")
            if not isinstance(session_spec_raw, dict) or template_spec is None:
                continue

            # Reject when spec kind mismatches the node class.
            session_is_operator = "operatorClass" in session_spec_raw
            template_is_operator = isinstance(template_spec, F8OperatorSpec)
            if session_is_operator != template_is_operator:
                errors.append(
                    f"nodeId={node_id}: session spec kind mismatch for node type {node_type!r} "
                    f"(template={'operator' if template_is_operator else 'service'}, "
                    f"session={'operator' if session_is_operator else 'service'})"
                )
                continue

            # Identity fields must match.
            if isinstance(template_spec, F8OperatorSpec):
                sess_service_class = str(session_spec_raw.get("serviceClass") or "").strip()
                sess_operator_class = str(session_spec_raw.get("operatorClass") or "").strip()
                if sess_service_class and sess_service_class != str(template_spec.serviceClass):
                    errors.append(
                        f"nodeId={node_id}: serviceClass mismatch (session={sess_service_class!r}, template={template_spec.serviceClass!r})"
                    )
                    continue
                if sess_operator_class and sess_operator_class != str(template_spec.operatorClass):
                    errors.append(
                        f"nodeId={node_id}: operatorClass mismatch (session={sess_operator_class!r}, template={template_spec.operatorClass!r})"
                    )
                    continue
                sess_sv = _enum_str(session_spec_raw.get("schemaVersion"))
                tmpl_sv = _enum_str(template_spec.schemaVersion)
                if sess_sv and tmpl_sv and sess_sv != tmpl_sv:
                    errors.append(
                        f"nodeId={node_id}: schemaVersion mismatch (session={sess_sv!r}, template={tmpl_sv!r})"
                    )
                    continue
            else:
                sess_service_class = str(session_spec_raw.get("serviceClass") or "").strip()
                if sess_service_class and sess_service_class != str(template_spec.serviceClass):
                    errors.append(
                        f"nodeId={node_id}: serviceClass mismatch (session={sess_service_class!r}, template={template_spec.serviceClass!r})"
                    )
                    continue
                sess_sv = _enum_str(session_spec_raw.get("schemaVersion"))
                tmpl_sv = _enum_str(template_spec.schemaVersion)
                if sess_sv and tmpl_sv and sess_sv != tmpl_sv:
                    errors.append(
                        f"nodeId={node_id}: schemaVersion mismatch (session={sess_sv!r}, template={tmpl_sv!r})"
                    )
                    continue

            merged = template_spec.model_dump(mode="json")

            def _maybe_override_bool(key: str) -> None:
                if key in session_spec_raw:
                    merged[key] = session_spec_raw.get(key)

            def _maybe_override_list(key: str, allow: bool) -> None:
                if not allow:
                    # warn (non-fatal) if the session attempted to override a non-editable list.
                    if key in session_spec_raw and session_spec_raw.get(key) != merged.get(key):
                        logger.warning(
                            "Ignoring non-editable session override: nodeId=%s key=%s (template wins).",
                            node_id,
                            key,
                        )
                    return
                if key in session_spec_raw:
                    merged[key] = session_spec_raw.get(key)

            if isinstance(template_spec, F8OperatorSpec):
                _maybe_override_bool("editableStateFields")
                _maybe_override_bool("editableExecInPorts")
                _maybe_override_bool("editableExecOutPorts")
                _maybe_override_bool("editableDataInPorts")
                _maybe_override_bool("editableDataOutPorts")

                # Keep user metadata from persisted snapshots/variants.
                if "label" in session_spec_raw:
                    merged["label"] = session_spec_raw.get("label")
                if "description" in session_spec_raw:
                    merged["description"] = session_spec_raw.get("description")
                if "tags" in session_spec_raw:
                    merged["tags"] = session_spec_raw.get("tags")
                _maybe_override_list("stateFields", bool(merged.get("editableStateFields", False)))
                _maybe_override_list("execInPorts", bool(merged.get("editableExecInPorts", False)))
                _maybe_override_list("execOutPorts", bool(merged.get("editableExecOutPorts", False)))
                _maybe_override_list("dataInPorts", bool(merged.get("editableDataInPorts", False)))
                _maybe_override_list("dataOutPorts", bool(merged.get("editableDataOutPorts", False)))
                try:
                    node_data["f8_spec"] = F8OperatorSpec.model_validate(merged).model_dump(mode="json")
                except Exception as e:
                    errors.append(f"nodeId={node_id}: failed to merge operator spec: {e}")
            else:
                _maybe_override_bool("editableStateFields")
                _maybe_override_bool("editableCommands")
                _maybe_override_bool("editableDataInPorts")
                _maybe_override_bool("editableDataOutPorts")

                # Keep user metadata from persisted snapshots/variants.
                if "label" in session_spec_raw:
                    merged["label"] = session_spec_raw.get("label")
                if "description" in session_spec_raw:
                    merged["description"] = session_spec_raw.get("description")
                if "tags" in session_spec_raw:
                    merged["tags"] = session_spec_raw.get("tags")
                _maybe_override_list("stateFields", bool(merged.get("editableStateFields", False)))
                _maybe_override_list("commands", bool(merged.get("editableCommands", False)))
                _maybe_override_list("dataInPorts", bool(merged.get("editableDataInPorts", False)))
                _maybe_override_list("dataOutPorts", bool(merged.get("editableDataOutPorts", False)))
                try:
                    node_data["f8_spec"] = F8ServiceSpec.model_validate(merged).model_dump(mode="json")
                except Exception as e:
                    errors.append(f"nodeId={node_id}: failed to merge service spec: {e}")

        if errors:
            for msg in errors:
                logger.warning("Session spec mismatch: %s", msg)
            preview = "; ".join(errors[:3])
            raise NodeCreationError(
                "Cannot load session due to spec mismatch. "
                f"{preview}. Fix the session file or remove the conflicting nodes."
            )

        return layout_data

    @staticmethod
    def _inject_node_ids(layout_data: dict) -> None:
        """
        NodeGraphQt stores node ids as keys under `nodes`, but does not include
        them in each node dict. We inject `id` so deserialization restores
        stable ids (instead of the default `0x...`).
        """
        nodes = layout_data.get("nodes")
        if not isinstance(nodes, dict):
            return
        for node_id, node_data in nodes.items():
            if isinstance(node_data, dict) and "id" not in node_data:
                node_data["id"] = node_id

    @staticmethod
    def _coerce_layout_spec(spec_obj: object) -> dict[str, Any] | None:
        if isinstance(spec_obj, dict):
            return spec_obj
        if isinstance(spec_obj, (F8OperatorSpec, F8ServiceSpec)):
            return spec_obj.model_dump(mode="json")
        return None

    @staticmethod
    def _strip_missing_lock_for_save(node_data: dict[str, Any]) -> None:
        f8_sys_obj = node_data.get("f8_sys")
        if not isinstance(f8_sys_obj, dict):
            return
        keys_to_remove = (
            "missingLocked",
            "missingType",
            "missingReason",
            "missingRendererFallback",
            "missingSpec",
            "missingOriginalName",
        )
        for key in keys_to_remove:
            f8_sys_obj.pop(key, None)

    def _coerce_missing_session_nodes(self, layout_data: dict) -> dict:
        """
        Convert unknown session nodes to missing placeholders while preserving ids/spec/connections.
        """
        nodes = layout_data.get("nodes")
        if not isinstance(nodes, dict):
            return layout_data

        converted = 0
        for node_id, node_data in nodes.items():
            if not isinstance(node_data, dict):
                continue

            raw_type = str(node_data.get("type_") or "").strip()
            if not raw_type:
                raise NodeCreationError(f"Cannot load session node '{node_id}': missing `type_`.")
            if raw_type in self._node_factory.nodes:
                continue

            spec_payload = self._coerce_layout_spec(node_data.get("f8_spec"))
            if spec_payload is None:
                raise NodeCreationError(
                    f"Cannot load unknown node type '{raw_type}' (nodeId={node_id}): missing or invalid `f8_spec`."
                )
            is_operator = "operatorClass" in spec_payload
            placeholder_type = MISSING_OPERATOR_NODE_TYPE if is_operator else MISSING_SERVICE_NODE_TYPE
            if placeholder_type not in self._node_factory.nodes:
                raise NodeCreationError(f"Missing placeholder node class '{placeholder_type}' is not registered.")

            f8_sys_obj = node_data.get("f8_sys")
            if isinstance(f8_sys_obj, dict):
                f8_sys = dict(f8_sys_obj)
            else:
                f8_sys = {}
            f8_sys["missingLocked"] = True
            f8_sys["missingType"] = raw_type
            f8_sys["missingReason"] = f"unregistered node type '{raw_type}'"
            f8_sys["missingRendererFallback"] = bool(f8_sys.get("missingRendererFallback", False))
            f8_sys["missingSpec"] = dict(spec_payload)
            raw_name = str(node_data.get("name") or "").strip()
            if raw_name and not raw_name.endswith("[Missing]"):
                f8_sys["missingOriginalName"] = raw_name
                node_data["name"] = f"{raw_name} [Missing]"
            node_data["f8_sys"] = f8_sys
            node_data["type_"] = placeholder_type
            node_data["f8_spec"] = spec_payload
            converted += 1

        if converted:
            logger.warning("Recovered %s missing node(s) as placeholders.", converted)
        return layout_data

    def _restore_missing_session_nodes(self, layout_data: dict) -> dict:
        """
        Restore original type/spec for sessions that accidentally persisted placeholder node types.
        """
        nodes = layout_data.get("nodes")
        if not isinstance(nodes, dict):
            return layout_data

        restored = 0
        for _node_id, node_data in nodes.items():
            if not isinstance(node_data, dict):
                continue
            node_type = str(node_data.get("type_") or "").strip()
            if node_type not in {MISSING_OPERATOR_NODE_TYPE, MISSING_SERVICE_NODE_TYPE}:
                continue
            f8_sys = node_data.get("f8_sys")
            if not isinstance(f8_sys, dict):
                continue
            missing_type = str(f8_sys.get("missingType") or "").strip()
            missing_spec = self._coerce_layout_spec(f8_sys.get("missingSpec"))
            if not missing_type or missing_spec is None:
                continue
            node_data["type_"] = missing_type
            node_data["f8_spec"] = missing_spec
            restored += 1
        if restored:
            logger.warning("Restored %s placeholder node(s) back to original type/spec from session metadata.", restored)
        return layout_data

    @staticmethod
    def _strip_unknown_session_custom_properties(layout_data: dict) -> dict:
        """
        Drop session custom properties that are not present in the persisted spec.

        NodeGraphQt requires that every key under `custom` is a registered node
        property.
        """
        nodes = layout_data.get("nodes")
        if not isinstance(nodes, dict):
            return layout_data

        for node_id, node_data in nodes.items():
            if not isinstance(node_data, dict):
                continue
            custom = node_data.get("custom")
            if not isinstance(custom, dict) or not custom:
                continue
            raw_spec = node_data.get("f8_spec")
            if not isinstance(raw_spec, dict):
                continue

            allowed: set[str] = set()
            for sf in list(raw_spec.get("stateFields") or []):
                if not isinstance(sf, dict):
                    continue
                name = str(sf.get("name") or "").strip()
                if name:
                    allowed.add(name)

            if not allowed:
                node_data["custom"] = {}
                continue

            kept: dict[str, Any] = {}
            for k, v in custom.items():
                key = str(k)
                if key in allowed:
                    kept[key] = v

            if kept != custom:
                node_data["custom"] = kept

        return layout_data

    def toggle_node_search(self):
        """
        Open node search (tab search menu).

        NodeGraphQt's default implementation only opens when the viewer is
        under the mouse; for keyboard shortcuts we want it to open when the
        viewer has focus.
        """
        names = self._node_factory.names
        nodes = self._node_factory.nodes

        self._tab_search_node_type_aliases = {}
        alias_counts: dict[str, int] = {}
        filtered_names: dict[str, list[str]] = {}
        for node_name, node_types in dict(names or {}).items():
            kept_types: list[str] = []
            for node_type in list(node_types or []):
                node_type_id = str(node_type)
                node_cls = nodes.get(node_type_id)
                if node_cls is not None and self._is_hidden_node_class(node_cls):
                    continue
                category = self._tab_search_category_for_node(node_cls=node_cls, node_type_id=node_type_id)
                node_leaf = node_type_id.split(".")[-1] if "." in node_type_id else node_type_id
                alias_base = f"{category}.{node_leaf}"
                count = int(alias_counts.get(alias_base, 0)) + 1
                alias_counts[alias_base] = count
                alias_id = alias_base if count == 1 else f"{alias_base}_{count}"
                self._tab_search_node_type_aliases[alias_id] = node_type_id
                kept_types.append(alias_id)
            if kept_types:
                filtered_names[str(node_name)] = kept_types

        self._viewer.tab_search_set_nodes(filtered_names)
        self._viewer.tab_search_toggle()

    @staticmethod
    def _tab_search_category_for_node(*, node_cls: Any | None, node_type_id: str) -> str:
        if node_cls is not None:
            identifier = str(node_cls.__identifier__ or "").strip()
            if identifier:
                return identifier

        if "." in node_type_id:
            return ".".join(node_type_id.split(".")[:-1])
        return "uncategorized"

    def _on_search_triggered(self, node_type: str, pos: tuple[float, float]) -> None:
        """
        Resolve tab-search aliases to real node types before creating nodes.
        """
        node_type_id = self._tab_search_node_type_aliases.get(str(node_type), str(node_type))
        self.create_node(node_type_id, pos=pos)

    @staticmethod
    def _is_hidden_node_class(node_cls: Any) -> bool:
        """
        Hide nodes tagged with `__hidden__` from tab search while keeping them registered.
        """
        return is_hidden_spec_node_class(node_cls)

    def save_last_session(self) -> str:
        """
        Save the current session to `~/.f8/studio/lastSession.json`.
        """
        path = last_session_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_session(str(path))
        return str(path)

    def load_last_session(self) -> str | None:
        """
        Load `~/.f8/studio/lastSession.json` if it exists.
        """
        path = last_session_path()
        if not path.is_file():
            return None
        self.load_session(str(path))
        return str(path)

    def serialize_session(self):
        data = super().serialize_session()
        stripped_layout = self._strip_port_restore_data(data)
        nodes = stripped_layout.get("nodes")
        if isinstance(nodes, dict):
            for _node_id, node_data in nodes.items():
                if not isinstance(node_data, dict):
                    continue
                f8_sys_obj = node_data.get("f8_sys")
                if isinstance(f8_sys_obj, dict) and bool(f8_sys_obj.get("missingLocked")):
                    missing_type = str(f8_sys_obj.get("missingType") or "").strip()
                    missing_spec = self._coerce_layout_spec(f8_sys_obj.get("missingSpec"))
                    if missing_type and missing_spec is not None:
                        node_data["type_"] = missing_type
                        node_data["f8_spec"] = missing_spec
                        missing_original_name = str(f8_sys_obj.get("missingOriginalName") or "").strip()
                        if missing_original_name:
                            node_data["name"] = missing_original_name
                self._strip_missing_lock_for_save(node_data)
        return _wrap_layout_for_save(stripped_layout)

    def load_session(self, file_path: str) -> None:
        """
        Load a NodeGraphQt session file.

        We temporarily disable studio constraints during deserialization, then
        rebuild container/operator bindings based on geometry.
        """
        file_path = file_path.strip()
        if not os.path.isfile(file_path):
            raise IOError(f"file does not exist: {file_path}")

        self._loading_session = True
        try:
            self.clear_session()
            with open(file_path) as data_file:
                payload = json.load(data_file)
            layout_data = _extract_session_layout(payload)
            self._inject_node_ids(layout_data)
            layout_data = self._restore_missing_session_nodes(layout_data)
            layout_data = self._coerce_missing_session_nodes(layout_data)
            layout_data = self._merge_session_specs(layout_data)
            layout_data = self._strip_port_restore_data(layout_data)
            layout_data = self._strip_unknown_session_custom_properties(layout_data)
            layout_data = self._strip_invalid_connections(layout_data)
            super().deserialize_session(layout_data, clear_session=False, clear_undo_stack=True)
            self._model.session = file_path
            self.session_changed.emit(file_path)
        finally:
            self._loading_session = False
        self._rebind_container_children()
        # Session load restores connections after nodes are created/drawn, which can
        # leave inline state widgets with stale editability until the user forces a refresh.
        # Do a post-load pass to apply the "state-edge => readonly" rule.
        QtCore.QTimer.singleShot(0, self._refresh_all_inline_state_read_only)

    def _refresh_all_inline_state_read_only(self) -> None:
        """
        Apply inline readonly state for all nodes (best-effort).

        Needed after session load because NodeGraphQt can restore connections
        without triggering interactive port connect signals in our UI layer.
        """
        nodes = list(self.all_nodes() or [])
        for n in nodes:
            view = n.view
            if not isinstance(view, F8StudioServiceNodeItem):
                continue
            view.refresh_inline_state_read_only()
            view.update()

    def _assign_node_id(self, node: BaseNode) -> BaseNode:
        new_nid = self.new_unique_node_id()
        node.model.id = new_nid
        node.view.id = new_nid
        # Seed identity state into UI properties early (these are runtime-owned readonly
        # fields; the runtime compiler skips ro values so they won't be deployed).
        try:
            spec = node.spec  # type: ignore[attr-defined]
        except Exception:
            spec = None
        try:
            if isinstance(spec, F8OperatorSpec):
                if "operatorId" in node.model.properties or "operatorId" in node.model.custom_properties:
                    node.set_property("operatorId", str(new_nid), push_undo=False)
            elif isinstance(spec, F8ServiceSpec):
                if "svcId" in node.model.properties or "svcId" in node.model.custom_properties:
                    node.set_property("svcId", str(new_nid), push_undo=False)
        except (AttributeError, RuntimeError, TypeError):
            pass
        return node

    @staticmethod
    def _variant_record(variant_id: str) -> dict[str, Any] | None:
        vid = str(variant_id or "").strip()
        if not vid:
            return None
        lib = load_library()
        for v in lib.variants:
            if str(v.variantId) == vid:
                return v.model_dump(mode="json")
        return None

    @staticmethod
    def _coerce_variant_spec(value: dict[str, Any]) -> F8OperatorSpec | F8ServiceSpec:
        if "operatorClass" in value:
            return F8OperatorSpec.model_validate(value)
        return F8ServiceSpec.model_validate(value)

    def _apply_variant_to_node(
        self,
        *,
        node: BaseNode,
        variant_id: str,
        variant_name: str,
        variant_spec_json: dict[str, Any],
    ) -> None:
        spec = self._coerce_variant_spec(variant_spec_json)
        node.spec = spec  # type: ignore[attr-defined]
        node.set_ui_overrides({}, rebuild=False)  # type: ignore[attr-defined]
        node.sync_from_spec()  # type: ignore[attr-defined]
        if not isinstance(node.model.f8_sys, dict):
            node.model.f8_sys = {}
        node.model.f8_sys["variantId"] = str(variant_id)
        node.model.f8_sys["variantName"] = str(variant_name or "")

    def create_variant_node(
        self,
        variant_id: str,
        *,
        pos: tuple[float, float] | None = None,
        selected: bool = True,
        push_undo: bool = True,
    ) -> BaseNode | None:
        node_type = build_variant_node_type(variant_id)
        return self.create_node(node_type, pos=pos, selected=selected, push_undo=push_undo)

    def create_node(self, node_type, name=None, selected=True, color=None, text_color=None, pos=None, push_undo=True):
        """
        Create a new node in the node graph.

        See Also:
            To list all node types :meth:`NodeGraph.registered_nodes`

        Args:
            node_type (str): node instance type.
            name (str): set name of the node.


            selected (bool): set created node to be selected.
            color (tuple or str): node color ``(255, 255, 255)`` or ``"#FFFFFF"``.
            text_color (tuple or str): text color ``(255, 255, 255)`` or ``"#FFFFFF"``.
            pos (list[int, int]): initial x, y position for the node (default: ``(0, 0)``).
            push_undo (bool): register the command to the undo stack. (default: True)

        Returns:
            BaseNode: the created instance of the node.
        """
        variant_id = parse_variant_node_type(str(node_type))
        if variant_id:
            record = self._variant_record(variant_id)
            if record is None:
                raise NodeCreationError(f'Can\'t find variant: "{variant_id}"')
            base_node_type = str(record.get("baseNodeType") or "").strip()
            if not base_node_type:
                raise NodeCreationError(f'Variant "{variant_id}" has empty baseNodeType')
            variant_name = str(record.get("name") or "").strip()
            variant_spec_json = record.get("spec")
            if not isinstance(variant_spec_json, dict):
                raise NodeCreationError(f'Variant "{variant_id}" has invalid spec')
            node = self.create_node(
                base_node_type,
                name=name or variant_name or None,
                selected=selected,
                color=color,
                text_color=text_color,
                pos=pos,
                push_undo=push_undo,
            )
            if node is None:
                return None
            self._apply_variant_to_node(
                node=node,
                variant_id=variant_id,
                variant_name=variant_name,
                variant_spec_json=variant_spec_json,
            )
            return node

        node = self._node_factory.create_node_instance(node_type)
        if node:
            node = self._assign_node_id(node)

            node._graph = self
            node.model._graph_model = self.model

            wid_types = node.model.__dict__.pop("_TEMP_property_widget_types")
            prop_attrs = node.model.__dict__.pop("_TEMP_property_attrs")

            if self.model.get_node_common_properties(node.type_) is None:
                node_attrs = {node.type_: {n: {"widget_type": wt} for n, wt in wid_types.items()}}
                for pname, pattrs in prop_attrs.items():
                    node_attrs[node.type_][pname].update(pattrs)
                self.model.set_node_common_properties(node_attrs)

            accept_types = node.model.__dict__.pop("_TEMP_accept_connection_types")
            for ptype, pdata in accept_types.get(node.type_, {}).items():
                for pname, accept_data in pdata.items():
                    for accept_ntype, accept_ndata in accept_data.items():
                        for accept_ptype, accept_pnames in accept_ndata.items():
                            for accept_pname in accept_pnames:
                                self._model.add_port_accept_connection_type(
                                    port_name=pname,
                                    port_type=ptype,
                                    node_type=node.type_,
                                    accept_pname=accept_pname,
                                    accept_ptype=accept_ptype,
                                    accept_ntype=accept_ntype,
                                )
            reject_types = node.model.__dict__.pop("_TEMP_reject_connection_types")
            for ptype, pdata in reject_types.get(node.type_, {}).items():
                for pname, reject_data in pdata.items():
                    for reject_ntype, reject_ndata in reject_data.items():
                        for reject_ptype, reject_pnames in reject_ndata.items():
                            for reject_pname in reject_pnames:
                                self._model.add_port_reject_connection_type(
                                    port_name=pname,
                                    port_type=ptype,
                                    node_type=node.type_,
                                    reject_pname=reject_pname,
                                    reject_ptype=reject_ptype,
                                    reject_ntype=reject_ntype,
                                )

            node.NODE_NAME = self.get_unique_name(name or node.NODE_NAME)
            node.model.name = node.NODE_NAME
            node.model.selected = selected

            def format_color(clr):
                if isinstance(clr, str):
                    clr = clr.strip("#")
                    return tuple(int(clr[i : i + 2], 16) for i in (0, 2, 4))
                return clr

            if color:
                node.model.color = format_color(color)
            if text_color:
                node.model.text_color = format_color(text_color)
            if pos:
                node.model.pos = [float(pos[0]), float(pos[1])]

            # initial node direction layout.
            node.model.layout_direction = self.layout_direction()

            node.update()

            if not self._loading_session:
                ok, msg = self._ensure_operator_in_container(node, pos=pos)
                if not ok:
                    if msg:
                        show_warning(self._notification_parent(), "Container required", msg)
                    return None

            undo_cmd = NodeAddedCmd(self, node, pos=node.model.pos, emit_signal=True)
            if push_undo:
                undo_label = 'create node: "{}"'.format(node.NODE_NAME)
                self._undo_stack.beginMacro(undo_label)
                for n in self.selected_nodes():
                    n.set_property("selected", False, push_undo=True)
                self._undo_stack.push(undo_cmd)
                self._undo_stack.endMacro()
            else:
                for n in self.selected_nodes():
                    n.set_property("selected", False, push_undo=False)
                undo_cmd.redo()

            return node

        raise NodeCreationError('Can\'t find node: "{}"'.format(node_type))

    def add_node(self, node, pos=None, selected=True, push_undo=True, inherite_graph_style=True):
        """Add an existing node to the graph.
        Args:
            node (BaseNode): node instance to add.
            pos (list[int, int]): initial x, y position for the node (default: ``(0, 0)``).
            selected (bool): set created node to be selected.
            push_undo (bool): register the command to the undo stack. (default: True)
            inherite_graph_style (bool): whether to inherite the graph style settings.
        """

        if not self._loading_session:
            node = self._assign_node_id(node)
        if pos:
            node.model.pos = [float(pos[0]), float(pos[1])]
            node.view.xy_pos = [float(pos[0]), float(pos[1])]

        if not self._loading_session:
            ok, msg = self._ensure_operator_in_container(node, pos=pos)
            if not ok:
                if msg:
                    show_warning(self._notification_parent(), "Container required", msg)
                return

        super().add_node(
            node, pos=pos, selected=selected, push_undo=push_undo, inherite_graph_style=inherite_graph_style
        )

    def delete_node(self, node, push_undo=True):
        """
        Delete a node from the graph.

        Note: deleting a service container also deletes its bound operators.
        """
        nodes = self._expand_delete_nodes([node])
        return self._delete_nodes_expanded(nodes, push_undo=push_undo)

    def delete_nodes(self, nodes, push_undo=True):
        """
        Delete multiple nodes from the graph.

        Note: deleting any service container also deletes its bound operators.
        """
        nodes = self._expand_delete_nodes(list(nodes or []))
        return self._delete_nodes_expanded(nodes, push_undo=push_undo)

    def _delete_nodes_expanded(self, nodes: list[Any], *, push_undo: bool = True) -> Any:
        """
        Delete nodes where the list is already expanded (ie. includes container children).
        """
        if not nodes:
            return
        self.repair_stale_port_connection_refs()

        service_ids: set[str] = set()
        for n in list(nodes or []):
            try:
                # Reclaim applies to *service instance nodes* (containers + standalone services).
                spec = n.spec
                if not isinstance(spec, F8ServiceSpec):
                    continue
                sid = str(n.id or "").strip()
                svc_class = str(spec.serviceClass or "")
                if sid and sid != STUDIO_SERVICE_ID and svc_class != _CANVAS_SERVICE_CLASS_:
                    service_ids.add(sid)
            except (AttributeError, TypeError):
                continue

        # NodeGraphQt's `delete_nodes([single])` calls back into `self.delete_node(...)`,
        # which we override. Avoid recursion by using `delete_node` directly for the
        # single-node case, and manually emitting `nodes_deleted` to keep our cleanup
        # hooks consistent with multi-node delete behavior.
        if len(nodes) == 1:
            node = nodes[0]
            node_id = ""
            try:
                node_id = str(node.id or "")
            except Exception:
                node_id = ""
            r = super().delete_node(node, push_undo=push_undo)
            if node_id:
                try:
                    self.nodes_deleted.emit([node_id])  # type: ignore[attr-defined]
                except (AttributeError, RuntimeError, TypeError):
                    pass
        else:
            r = super().delete_nodes(nodes, push_undo=push_undo)

        for sid in sorted(service_ids):
            self._schedule_service_reclaim(sid, delay_ms=3000)
        return r

    def clear_session(self, *args, **kwargs) -> None:
        """
        Clear the current canvas. Any removed service instances are reclaimed
        after a short debounce (so undo / immediate re-add won't kill processes).
        """
        before: set[str] = set()
        for n in self.all_nodes():
            try:
                spec = n.spec
                if not isinstance(spec, F8ServiceSpec):
                    continue
                sid = str(n.id or "").strip()
                svc_class = str(spec.serviceClass or "")
                if sid and sid != STUDIO_SERVICE_ID and svc_class != _CANVAS_SERVICE_CLASS_:
                    before.add(sid)
            except (AttributeError, TypeError):
                continue
        super().clear_session(*args, **kwargs)
        for sid in sorted({s for s in before if s and s != STUDIO_SERVICE_ID}):
            self._schedule_service_reclaim(sid, delay_ms=3000)

    def _is_service_referenced(self, service_id: str) -> bool:
        """
        True if the serviceId is still referenced by the current canvas.
        """
        sid = str(service_id or "").strip()
        if not sid:
            return False
        try:
            n = self.get_node_by_id(sid)
            # Any service instance node with this id keeps the service alive.
            if n is not None and isinstance(n.spec, F8ServiceSpec):
                return True
        except (AttributeError, RuntimeError, TypeError):
            pass
        # If any operator still points at this svcId, keep the service alive.
        for n in self.all_nodes():
            if not self._is_operator_node(n):
                continue
            try:
                if str(n.svcId or "") == sid:
                    return True
            except (AttributeError, TypeError):
                continue
        return False

    def _schedule_service_reclaim(self, service_id: str, *, delay_ms: int = 3000) -> None:
        sid = str(service_id or "").strip()
        if not sid or sid == STUDIO_SERVICE_ID:
            return
        # Reset debounce timer.
        t = self._reclaim_timers.get(sid)
        if t is None:
            t = QtCore.QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(lambda _sid=sid: self._reclaim_service_if_unreferenced(_sid))  # type: ignore[attr-defined]
            self._reclaim_timers[sid] = t
        try:
            if t.isActive():
                t.stop()
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            t.start(max(1, int(delay_ms)))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    def _reclaim_service_if_unreferenced(self, service_id: str) -> None:
        sid = str(service_id or "").strip()
        if not sid or sid == STUDIO_SERVICE_ID:
            return
        if self._is_service_referenced(sid):
            return
        bridge = self._service_bridge
        if bridge is None:
            return
        try:
            bridge.reclaim_service(sid)
        except (AttributeError, RuntimeError, TypeError):
            return

    def new_unique_node_id(self) -> str:
        """Generate a new unique node ID."""
        uuid = self.uuid_generator.random(self.uuid_length)
        while self.get_node_by_id(uuid) is not None:
            uuid = self.uuid_generator.random(self.uuid_length)
        return uuid

    @staticmethod
    def _is_operator_node(node: Any) -> bool:
        try:
            return isinstance(node.spec, F8OperatorSpec)
        except Exception:
            return False

    @staticmethod
    def _is_container_node(node: Any) -> bool:
        return isinstance(node, _BASE_CONTAINER_CLS_)

    def _container_at_node(self, node: Any) -> _BASE_CONTAINER_CLS_ | None:
        r_node = _scene_rect(node)
        if r_node is None:
            return None
        return self._container_at_rect(r_node)

    def _container_at_rect(self, rect: QtCore.QRectF) -> _BASE_CONTAINER_CLS_ | None:
        for container in self.all_nodes():
            if not self._is_container_node(container):
                continue
            r_run = _scene_rect(container)
            if r_run is None:
                continue
            if r_run.intersects(rect):
                return container
        return None

    def _bind_operator_to_container(self, operator: _BASE_OPERATOR_CLS_, container: _BASE_CONTAINER_CLS_) -> bool:
        if not self._is_operator_node(operator):
            logger.warning("Cannot bind non-operator node to container")
            return False
        if not self._is_container_node(container):
            logger.warning("Cannot bind operator node to non-container node")
            return False
        if operator.spec.serviceClass != container.spec.serviceClass:
            logger.warning(
                f"Operator serviceClass '{operator.spec.serviceClass}' does not match container serviceClass '{container.spec.serviceClass}'"
            )
            return False

        sid = container.id
        if not sid:
            logger.error("Container node has no ID")
            return False
        operator.svcId = sid  # type: ignore[attr-defined]
        try:
            if "svcId" in operator.model.properties or "svcId" in operator.model.custom_properties:
                operator.set_property("svcId", str(sid), push_undo=False)
        except (AttributeError, RuntimeError, TypeError):
            pass
        container.add_child(operator)
        return True

    def _container_bound_nodes(self, container: _BASE_CONTAINER_CLS_) -> list[BaseNode]:
        """
        Return nodes that are bound to the container (best-effort).
        """
        out: list[BaseNode] = []

        # Prefer the node objects tracked by _BASE_CONTAINER_CLS_.
        for child in container._child_nodes:
            nid = child.id
            n = self.get_node_by_id(nid)
            if n is not None:
                out.append(n)

        # Fallback: view-level tracking.
        for view in container.view._child_views:
            nid = view.id
            n = self.get_node_by_id(nid)
            if n is not None:
                out.append(n)

        # Dedupe by id.
        return list({n.id: n for n in out}.values())

    def _expand_delete_nodes(self, nodes: list[Any]) -> list[Any]:
        """
        Expand delete list so deleting a container cascades to its child operators.
        """
        if not nodes:
            return []

        out: list[Any] = []
        seen: set[str] = set()

        def add_node_obj(n: Any) -> None:
            nid = n.id
            if nid in seen:
                return
            seen.add(nid)
            out.append(n)

            if self._is_container_node(n):
                for child in self._container_bound_nodes(n):
                    add_node_obj(child)

        for n in nodes:
            if n is None:
                continue
            add_node_obj(n)

        return out

    def _on_property_changed(self, node: Any, name: str, value: Any) -> None:
        """
        Optional hook for reacting to property changes.
        """
        return

    def _on_nodes_moving(self, node_data: Any) -> None:
        """
        Optional hook for continuous move events during dragging.

        Only used when the viewer implements a `moving_nodes` signal.
        """
        return

    def _ensure_operator_in_container(
        self,
        node: Any,
        *,
        pos: list[float] | tuple[float, float] | None,
    ) -> tuple[bool, str | None]:
        """
        Enforce:
        - operator nodes must be placed within a service container (unless canvas-managed)
        - bind `svcId` and container child relationship

        Returns (ok, message). If ok is False, caller should not keep/add the node.
        """
        if not self._is_operator_node(node):
            return True, None

        if node.spec.serviceClass == _CANVAS_SERVICE_CLASS_:
            node.svcId = STUDIO_SERVICE_ID  # type: ignore[attr-defined]
            try:
                if "svcId" in node.model.properties or "svcId" in node.model.custom_properties:
                    node.set_property("svcId", STUDIO_SERVICE_ID, push_undo=False)
            except (AttributeError, RuntimeError, TypeError):
                pass
            return True, None

        in_scene = node.view.scene() is not None
        node_rect = _scene_rect(node) if in_scene else _rect_at_pos(node.view, pos or node.model.pos)

        container = self._container_at_rect(node_rect)
        if container is None:
            return False, "Operator nodes must be placed within a service container."

        if not self._bind_operator_to_container(node, container):
            return False, "Operator nodes must be placed within a compatible service container."

        return True, None

    def _on_nodes_deleted(self, node_ids: list[str]) -> None:
        """
        Keep container child lists clean when nodes are deleted (including undo).
        """
        if not node_ids:
            return
        dead = set(node_ids)
        for container in self.all_nodes():
            if not self._is_container_node(container):
                continue
            container._child_nodes = [n for n in container._child_nodes if n.id not in dead]

            kept = []
            for view in container.view._child_views:
                vid = view.id
                if vid in dead:
                    view._container_item = None
                    continue
                kept.append(view)
            container.view._child_views = kept

    def _rebind_container_children(self) -> None:
        """
        Rebuild container -> operator bindings from geometry.

        This is used after session load to restore:
        - container dragging moves operators
        - operator drag clamping (via `view._container_item`)
        """
        containers: list[_BASE_CONTAINER_CLS_] = []
        operators: list[BaseNode] = []
        for node in self.all_nodes():
            if self._is_container_node(node):
                containers.append(node)
            elif self._is_operator_node(node):
                operators.append(node)

        # Clear container child lists.
        for container in containers:
            container._child_nodes = []
            container.view._child_views = []

        # Clear operator back-references.
        for op in operators:
            op.view._container_item = None

        # Rebind operators based on intersecting container geometry.
        for op in operators:
            if op.spec.serviceClass == _CANVAS_SERVICE_CLASS_:
                # Studio (editor-local) operators belong to the built-in PyStudio service.
                # They are not bound to a container instance, but still need a stable svcId.
                op.svcId = STUDIO_SERVICE_ID  # type: ignore[attr-defined]
                try:
                    if "svcId" in op.model.properties or "svcId" in op.model.custom_properties:
                        op.set_property("svcId", STUDIO_SERVICE_ID, push_undo=False)
                except (AttributeError, RuntimeError, TypeError):
                    pass
                continue

            container = self._container_at_node(op)
            if container is None:
                # Leave as orphan so user can fix placement manually.
                op.svcId = ""  # type: ignore[attr-defined]
                try:
                    if "svcId" in op.model.properties or "svcId" in op.model.custom_properties:
                        op.set_property("svcId", "", push_undo=False)
                except (AttributeError, RuntimeError, TypeError):
                    pass
                logger.warning('Operator "%s" is not inside any container after load.', op.name())
                continue

            self._bind_operator_to_container(op, container)
