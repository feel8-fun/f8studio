from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Generic

from qtpy import QtCore, QtWidgets, QtGui
from NodeGraphQt import NodeGraph, BaseNode
from NodeGraphQt.errors import NodeCreationError, NodeDeletionError
from NodeGraphQt.base.commands import NodeAddedCmd, NodeMovedCmd, NodesRemovedCmd, PortConnectedCmd
import shortuuid
import logging

from f8pysdk import F8OperatorSpec, F8ServiceSpec
from .container_basenode import F8StudioContainerBaseNode
from .operator_basenode import F8StudioOperatorBaseNode

from .viewer import F8StudioNodeViewer
from .session import last_session_path

from ..service_host import ServiceHostRegistry
from ..service_host.service_host_registry import STUDIO_SERVICE_ID

_BASE_OPERATOR_CLS_ = F8StudioOperatorBaseNode
_BASE_CONTAINER_CLS_ = F8StudioContainerBaseNode
_CANVAS_SERVICE_CLASS_ = ServiceHostRegistry.instance().serviceClass

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

        self.uuid_length = kwargs.get("uuid_length", 4)
        self.uuid_generator = shortuuid.ShortUUID()
        self._loading_session = False
        # self._move = _MoveState()

        self.property_changed.connect(self._on_property_changed)  # type: ignore[attr-defined]

        # NodeGraphQt exposes `nodes_deleted` (list[str]), not `node_deleted`.
        self.nodes_deleted.connect(self._on_nodes_deleted)  # type: ignore[attr-defined]

        # Continuous move events during dragging.
        if hasattr(self._viewer, "moving_nodes"):
            self._viewer.moving_nodes.connect(self._on_nodes_moving)  # type: ignore[attr-defined]

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

    def _validate_session_node_types(self, layout_data: dict) -> None:
        """
        Validate that node classes referenced by the session are registered.

        We intentionally do NOT auto-register node classes from session data.
        Node registration should come from service discovery.
        """
        nodes = layout_data.get("nodes")
        if not isinstance(nodes, dict):
            return

        missing_types: set[str] = set()
        missing_type_field = 0

        for node_data in nodes.values():
            if not isinstance(node_data, dict):
                continue

            node_type = node_data.get("type_")
            if not isinstance(node_type, str) or not node_type.strip():
                missing_type_field += 1
                continue

            node_type = node_type.strip()
            if node_type not in self._node_factory.nodes:
                missing_types.add(node_type)

        if missing_type_field or missing_types:
            parts = []
            if missing_type_field:
                parts.append(f"{missing_type_field} node(s) missing `type_` in session data")
            if missing_types:
                missing_list = ", ".join(sorted(missing_types))
                parts.append(f"unregistered node type(s): {missing_list}")
            msg = "Cannot load session: " + "; ".join(parts) + "."
            raise NodeCreationError(msg)

    def toggle_node_search(self):
        """
        Open node search (tab search menu).

        NodeGraphQt's default implementation only opens when the viewer is
        under the mouse; for keyboard shortcuts we want it to open when the
        viewer has focus.
        """
        self._viewer.tab_search_set_nodes(self._node_factory.names)
        self._viewer.tab_search_toggle()

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
        return self._strip_port_restore_data(data)

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
                layout_data = json.load(data_file)
            self._inject_node_ids(layout_data)
            self._validate_session_node_types(layout_data)
            layout_data = self._strip_port_restore_data(layout_data)
            super().deserialize_session(layout_data, clear_session=False, clear_undo_stack=True)
            self._model.session = file_path
            self.session_changed.emit(file_path)
        finally:
            self._loading_session = False
        self._rebind_container_children()

    def _assign_node_id(self, node: BaseNode) -> BaseNode:
        new_nid = self.new_unique_node_id()
        node.model.id = new_nid
        node.view.id = new_nid
        return node

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
                        QtWidgets.QMessageBox.warning(None, "Container required", msg)
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
                    QtWidgets.QMessageBox.warning(None, "Container required", msg)
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
        if len(nodes) <= 1:
            return super().delete_node(node, push_undo=push_undo)
        return super().delete_nodes(nodes, push_undo=push_undo)

    def delete_nodes(self, nodes, push_undo=True):
        """
        Delete multiple nodes from the graph.

        Note: deleting any service container also deletes its bound operators.
        """
        nodes = self._expand_delete_nodes(list(nodes or []))
        return super().delete_nodes(nodes, push_undo=push_undo)

    def new_unique_node_id(self) -> str:
        """Generate a new unique node ID."""
        uuid = self.uuid_generator.random(self.uuid_length)
        while self.get_node_by_id(uuid) is not None:
            uuid = self.uuid_generator.random(self.uuid_length)
        return uuid

    @staticmethod
    def _is_operator_node(node: Any) -> bool:
        return hasattr(node, "spec") and isinstance(node.spec, F8OperatorSpec)  # type: ignore[attr-defined]

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
                continue

            container = self._container_at_node(op)
            if container is None:
                # Leave as orphan so user can fix placement manually.
                op.svcId = ""  # type: ignore[attr-defined]
                logger.warning('Operator "%s" is not inside any container after load.', op.name())
                continue

            self._bind_operator_to_container(op, container)
