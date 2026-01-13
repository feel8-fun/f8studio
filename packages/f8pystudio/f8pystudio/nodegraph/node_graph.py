from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qtpy import QtCore, QtWidgets
from Qt import QtWidgets as QtNgQtWidgets
from NodeGraphQt import NodeGraph, BaseNode
from NodeGraphQt.errors import NodeCreationError, NodeDeletionError
from NodeGraphQt.base.commands import (NodeAddedCmd, NodeMovedCmd,
                                       NodesRemovedCmd, PortConnectedCmd)
import shortuuid

from f8pysdk import F8OperatorSpec, F8ServiceSpec

from ..renderNodes.operator_runner import OperatorRunnerRenderNode
from .viewer import F8NodeViewer


@dataclass
class _MoveState:
    last_ok_pos: dict[str, list[float]] = field(default_factory=dict)
    last_ok_runner_geom: dict[str, dict[str, Any]] = field(default_factory=dict)
    runner_drag_child_start: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    updating: bool = False


def _node_id(node: Any) -> str:
    try:
        return str(getattr(node, "id", "") or "")
    except Exception:
        return ""


def _scene_rect(node: Any) -> QtCore.QRectF | None:
    try:
        view = getattr(node, "view", None)
        if view is None:
            return None
        return view.sceneBoundingRect()
    except Exception:
        return None


def _clamp_delta(node_rect: QtCore.QRectF, container_rect: QtCore.QRectF) -> QtCore.QPointF:
    dx = 0.0
    dy = 0.0

    # If the item is larger than the container we can't fully contain it.
    # In that case we align the top-left edge to keep behavior predictable.
    if node_rect.width() > container_rect.width():
        dx = container_rect.left() - node_rect.left()
    elif node_rect.left() < container_rect.left():
        dx = container_rect.left() - node_rect.left()
    elif node_rect.right() > container_rect.right():
        dx = container_rect.right() - node_rect.right()

    if node_rect.height() > container_rect.height():
        dy = container_rect.top() - node_rect.top()
    elif node_rect.top() < container_rect.top():
        dy = container_rect.top() - node_rect.top()
    elif node_rect.bottom() > container_rect.bottom():
        dy = container_rect.bottom() - node_rect.bottom()

    return QtCore.QPointF(dx, dy)


class F8StudioGraph(NodeGraph):
    """Main F8PyStudio controller class."""

    def __init__(self, parent=None, **kwargs):
        """
        Args:
            parent (object): object parent.
            **kwargs (dict): Used for overriding internal objects at init time.
        """
        # We need a custom viewer to get continuous node-move events while dragging.
        undo_stack = kwargs.pop("undo_stack", None)
        viewer = kwargs.pop("viewer", None)
        if viewer is None:
            if undo_stack is None:
                undo_stack = QtNgQtWidgets.QUndoStack(parent)
            viewer = F8NodeViewer(undo_stack=undo_stack)
        super().__init__(parent, undo_stack=undo_stack, viewer=viewer, **kwargs)

        self.uuid_length = kwargs.get("uuid_length", 4)
        self.uuid_generator = shortuuid.ShortUUID()
        self._move = _MoveState()

        try:
            self.property_changed.connect(self._on_property_changed)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            self.node_deleted.connect(self._on_node_deleted)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            # Continuous move events during dragging.
            getattr(self._viewer, "moving_nodes").connect(self._on_nodes_moving)  # type: ignore[attr-defined]
        except Exception:
            pass

    
    def create_node(self, node_type, name=None, selected=True, color=None,
                    text_color=None, pos=None, push_undo=True):
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
            node._graph = self
            node.model._graph_model = self.model

            # Create a unique node id.
            node.model.id = self.new_unique_node_id()
            node.view.id = node.model.id

            wid_types = node.model.__dict__.pop('_TEMP_property_widget_types')
            prop_attrs = node.model.__dict__.pop('_TEMP_property_attrs')

            if self.model.get_node_common_properties(node.type_) is None:
                node_attrs = {node.type_: {
                    n: {'widget_type': wt} for n, wt in wid_types.items()
                }}
                for pname, pattrs in prop_attrs.items():
                    node_attrs[node.type_][pname].update(pattrs)
                self.model.set_node_common_properties(node_attrs)

            accept_types = node.model.__dict__.pop(
                '_TEMP_accept_connection_types'
            )
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
                                    accept_ntype=accept_ntype
                                )
            reject_types = node.model.__dict__.pop(
                '_TEMP_reject_connection_types'
            )
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
                                    reject_ntype=reject_ntype
                                )

            node.NODE_NAME = self.get_unique_name(name or node.NODE_NAME)
            node.model.name = node.NODE_NAME
            node.model.selected = selected

            def format_color(clr):
                if isinstance(clr, str):
                    clr = clr.strip('#')
                    return tuple(int(clr[i:i + 2], 16) for i in (0, 2, 4))
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

            undo_cmd = NodeAddedCmd(
                self, node, pos=node.model.pos, emit_signal=True
            )
            if push_undo:
                undo_label = 'create node: "{}"'.format(node.NODE_NAME)
                self._undo_stack.beginMacro(undo_label)
                for n in self.selected_nodes():
                    n.set_property('selected', False, push_undo=True)
                self._undo_stack.push(undo_cmd)
                self._undo_stack.endMacro()
            else:
                for n in self.selected_nodes():
                    n.set_property('selected', False, push_undo=False)
                undo_cmd.redo()

            try:
                self._on_node_created(node)
            except Exception:
                pass
            return node

        raise NodeCreationError('Can\'t find node: "{}"'.format(node_type))

    def new_unique_node_id(self) -> str:
        """Generate a new unique node ID."""
        uuid = self.uuid_generator.random(self.uuid_length)
        while self.get_node_by_id(uuid) is not None:
            uuid = self.uuid_generator.random(self.uuid_length)
        return uuid

    # ---- OperatorRunner constraints ------------------------------------
    def _operator_runners(self) -> list[OperatorRunnerRenderNode]:
        runners: list[OperatorRunnerRenderNode] = []
        try:
            nodes = list(self.all_nodes() or [])
        except Exception:
            nodes = []
        for n in nodes:
            if isinstance(n, OperatorRunnerRenderNode):
                runners.append(n)
        return runners

    @staticmethod
    def _is_operator_node(node: Any) -> bool:
        return isinstance(getattr(node, "spec", None), F8OperatorSpec)

    @staticmethod
    def _is_runner_node(node: Any) -> bool:
        return isinstance(node, OperatorRunnerRenderNode) or isinstance(getattr(node, "spec", None), F8ServiceSpec)

    def _runner_at_node(self, node: Any) -> OperatorRunnerRenderNode | None:
        r_node = _scene_rect(node)
        if r_node is None:
            return None
        for runner in self._operator_runners():
            r_run = _scene_rect(runner)
            if r_run is None:
                continue
            if r_run.contains(r_node.center()):
                return runner
        return None

    def _runner_for_service_id(self, service_id: str | None) -> OperatorRunnerRenderNode | None:
        sid = str(service_id or "").strip()
        if not sid:
            return None
        for runner in self._operator_runners():
            rid = _node_id(runner).replace(".", "_")
            if rid == sid:
                return runner
        return None

    def _bind_operator_to_runner(self, node: Any, runner: OperatorRunnerRenderNode) -> None:
        sid = _node_id(runner).replace(".", "_")
        if not sid:
            return
        try:
            setattr(node, "serviceId", sid)
        except Exception:
            pass
        try:
            node.create_property("serviceId", sid)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            node.set_property("serviceId", sid, push_undo=False)  # type: ignore[attr-defined]
        except Exception:
            try:
                node.set_property("serviceId", sid)  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            setattr(node, "parentNodeId", _node_id(runner))
        except Exception:
            pass
        try:
            runner.add_child(node)
        except Exception:
            pass
        # Do NOT auto-wrap nodes: runner geometry is user-controlled.

    def _on_node_created(self, node: Any) -> None:
        if not self._is_operator_node(node):
            return

        runner = self._runner_at_node(node)
        if runner is None:
            try:
                QtWidgets.QMessageBox.warning(
                    None,
                    "必须指定 Operator Runner",
                    "创建 Operator 时必须放在某个 Operator Runner（服务容器）内部。",
                )
            except Exception:
                pass
            try:
                self.delete_nodes([node])  # type: ignore[attr-defined]
            except Exception:
                pass
            return

        self._bind_operator_to_runner(node, runner)
        try:
            self._move.last_ok_pos[_node_id(node)] = list(node.pos())  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            rid = _node_id(runner)
            if rid and rid not in self._move.last_ok_runner_geom:
                self._move.last_ok_runner_geom[rid] = {
                    "pos": list(runner.pos()),
                    "width": runner.get_property("width"),
                    "height": runner.get_property("height"),
                }
        except Exception:
            pass

    def _on_property_changed(self, node: Any, name: str, value: Any) -> None:
        if self._move.updating:
            return
        prop = str(name)

        # Clamp runner resize so it always contains its children.
        # (Runner movement is handled in `_on_nodes_moved` so we can clamp while dragging.)
        if isinstance(node, OperatorRunnerRenderNode) and prop in ("width", "height"):
            rid = _node_id(node)
            if not rid:
                return
            runner_rect = _scene_rect(node)
            if runner_rect is None:
                return
            try:
                children = list(node.contained_nodes() or [])
            except Exception:
                children = []
            bounds: QtCore.QRectF | None = None
            for c in children:
                r = _scene_rect(c)
                if r is None:
                    continue
                bounds = r if bounds is None else bounds.united(r)
            if bounds is None:
                return

            if runner_rect.contains(bounds):
                return

            # Ensure the runner can never be resized smaller than its children.
            req_w = max(0.0, float(bounds.right() - runner_rect.left()))
            req_h = max(0.0, float(bounds.bottom() - runner_rect.top()))
            try:
                cur_w = float(node.get_property("width") or 0.0)
                cur_h = float(node.get_property("height") or 0.0)
            except Exception:
                cur_w = float(getattr(getattr(node, "view", None), "width", 0.0) or 0.0)
                cur_h = float(getattr(getattr(node, "view", None), "height", 0.0) or 0.0)
            next_w = max(cur_w, req_w)
            next_h = max(cur_h, req_h)
            if next_w == cur_w and next_h == cur_h:
                return

            self._move.updating = True
            try:
                # Avoid creating another undo entry: this is a constraint clamp.
                try:
                    node.set_property("width", next_w, push_undo=False)  # type: ignore[attr-defined]
                    node.set_property("height", next_h, push_undo=False)  # type: ignore[attr-defined]
                except Exception:
                    try:
                        node.view.width = next_w  # type: ignore[attr-defined]
                        node.view.height = next_h  # type: ignore[attr-defined]
                        node.model.width = next_w  # type: ignore[attr-defined]
                        node.model.height = next_h  # type: ignore[attr-defined]
                    except Exception:
                        pass
            finally:
                self._move.updating = False
            return

        if prop != "pos":
            return
        if not self._is_operator_node(node):
            return

        # Operators are not allowed to move between runners. Prefer the explicit
        # parent pointer; fallback to `serviceId` lookup only when missing.
        runner: OperatorRunnerRenderNode | None = None
        try:
            parent_id = str(getattr(node, "parentNodeId", "") or "").strip()
        except Exception:
            parent_id = ""
        if parent_id:
            try:
                parent = self.get_node_by_id(parent_id)
            except Exception:
                parent = None
            if isinstance(parent, OperatorRunnerRenderNode):
                runner = parent

        if runner is None:
            sid = None
            try:
                sid = getattr(node, "serviceId", None)
            except Exception:
                sid = None
            if not sid:
                try:
                    sid = node.get_property("serviceId")  # type: ignore[attr-defined]
                except Exception:
                    sid = None
            runner = self._runner_for_service_id(sid)

        if runner is None:
            self._revert_pos(node)
            return

        r_node = _scene_rect(node)
        r_run = _scene_rect(runner)
        if r_node is None or r_run is None:
            return

        if r_run.contains(r_node):
            try:
                self._move.last_ok_pos[_node_id(node)] = list(value)
            except Exception:
                pass
            return
        delta = _clamp_delta(r_node, r_run)
        try:
            cur = list(node.pos())  # type: ignore[attr-defined]
        except Exception:
            cur = list(value) if isinstance(value, (list, tuple)) else [0.0, 0.0]
        next_pos = [float(cur[0]) + float(delta.x()), float(cur[1]) + float(delta.y())]
        self._move.updating = True
        try:
            try:
                node.set_pos(next_pos)  # type: ignore[attr-defined]
            except Exception:
                try:
                    node.set_property("pos", next_pos, push_undo=False)  # type: ignore[attr-defined]
                except Exception:
                    node.set_property("pos", next_pos)  # type: ignore[attr-defined]
            self._move.last_ok_pos[_node_id(node)] = list(next_pos)
        finally:
            self._move.updating = False

    def _revert_pos(self, node: Any) -> None:
        nid = _node_id(node)
        prev = self._move.last_ok_pos.get(nid)
        if not prev:
            return
        self._move.updating = True
        try:
            try:
                node.set_pos(prev)  # type: ignore[attr-defined]
            except Exception:
                try:
                    node.set_property("pos", prev, push_undo=False)  # type: ignore[attr-defined]
                except Exception:
                    node.set_property("pos", prev)  # type: ignore[attr-defined]
        finally:
            self._move.updating = False

    def _on_node_deleted(self, node: Any) -> None:
        if not self._is_operator_node(node):
            return
        try:
            parent_id = str(getattr(node, "parentNodeId", "") or "").strip()
        except Exception:
            parent_id = ""
        if not parent_id:
            return
        try:
            parent = self.get_node_by_id(parent_id)
        except Exception:
            parent = None
        if isinstance(parent, OperatorRunnerRenderNode):
            parent.remove_child(_node_id(node))

    def _assigned_runner_for_operator(self, node: Any) -> OperatorRunnerRenderNode | None:
        try:
            parent_id = str(getattr(node, "parentNodeId", "") or "").strip()
        except Exception:
            parent_id = ""
        if parent_id:
            try:
                parent = self.get_node_by_id(parent_id)
            except Exception:
                parent = None
            if isinstance(parent, OperatorRunnerRenderNode):
                return parent

        sid = None
        try:
            sid = getattr(node, "serviceId", None)
        except Exception:
            sid = None
        if not sid:
            try:
                sid = node.get_property("serviceId")  # type: ignore[attr-defined]
            except Exception:
                sid = None
        return self._runner_for_service_id(sid)

    def _set_node_xy(self, node: Any, pos: list[float]) -> None:
        try:
            node.view.xy_pos = pos  # type: ignore[attr-defined]
            node.model.pos = pos  # type: ignore[attr-defined]
        except Exception:
            try:
                node.set_property("pos", pos, push_undo=False)  # type: ignore[attr-defined]
            except Exception:
                try:
                    node.set_pos(pos)  # type: ignore[attr-defined]
                except Exception:
                    pass

    def _on_nodes_moved(self, node_data: dict[Any, Any]) -> None:
        """
        Enforce OperatorRunner constraints after a drag move.

        - Operators: clamp to their assigned runner bounds (can't leave or switch runner).
        - Runners: dragging a runner moves its children together.
        """
        if self._move.updating:
            return

        moved_node_ids = self._moved_node_ids(node_data)
        extra_undo = self._runner_child_undo_entries(node_data, moved_node_ids)

        # 2) Clamp moved operators as a group per runner.
        ops_by_runner: dict[str, tuple[OperatorRunnerRenderNode, list[Any]]] = {}
        for node_view, prev_pos in (node_data or {}).items():
            try:
                node = self._model.nodes.get(node_view.id)  # type: ignore[attr-defined]
            except Exception:
                node = None
            if node is None or not self._is_operator_node(node):
                continue

            runner = self._assigned_runner_for_operator(node)
            if runner is None:
                # Unbound operators should not exist; revert to previous position.
                try:
                    prev = list(prev_pos)
                    self._move.updating = True
                    self._set_node_xy(node, [float(prev[0]), float(prev[1])])
                except Exception:
                    pass
                finally:
                    self._move.updating = False
                continue

            rid = _node_id(runner)
            if not rid:
                continue
            if rid not in ops_by_runner:
                ops_by_runner[rid] = (runner, [])
            ops_by_runner[rid][1].append(node)

        for rid, (runner, nodes) in ops_by_runner.items():
            runner_rect = _scene_rect(runner)
            if runner_rect is None:
                continue
            bounds: QtCore.QRectF | None = None
            for n in nodes:
                r = _scene_rect(n)
                if r is None:
                    continue
                bounds = r if bounds is None else bounds.united(r)
            if bounds is None or runner_rect.contains(bounds):
                continue

            delta = _clamp_delta(bounds, runner_rect)
            if delta.x() == 0.0 and delta.y() == 0.0:
                continue

            self._move.updating = True
            try:
                for n in nodes:
                    cur = list(n.pos())
                    next_pos = [float(cur[0]) + float(delta.x()), float(cur[1]) + float(delta.y())]
                    self._set_node_xy(n, next_pos)
                self._move.last_ok_pos[_node_id(n)] = list(next_pos)
            finally:
                self._move.updating = False

        # 3) Register undo for the final (post-clamp) node positions.
        self._undo_stack.beginMacro("move nodes")
        try:
            for node_view, prev_pos in (node_data or {}).items():
                try:
                    node = self._model.nodes.get(node_view.id)  # type: ignore[attr-defined]
                except Exception:
                    node = None
                if node is None:
                    continue
                try:
                    cur = list(node.pos())
                except Exception:
                    continue
                if list(prev_pos) == cur:
                    continue
                self._undo_stack.push(NodeMovedCmd(node, cur, list(prev_pos)))

            for node, new_pos, prev_pos in extra_undo:
                if new_pos == prev_pos:
                    continue
                self._undo_stack.push(NodeMovedCmd(node, list(new_pos), list(prev_pos)))
        finally:
            self._undo_stack.endMacro()

        # Clear per-drag state for moved runners.
        for node_view in (node_data or {}).keys():
            try:
                rid = str(getattr(node_view, "id", "") or "")
            except Exception:
                rid = ""
            if rid:
                self._move.runner_drag_child_start.pop(rid, None)

    def _moved_node_ids(self, node_data: dict[Any, Any]) -> set[str]:
        out: set[str] = set()
        for node_view in (node_data or {}).keys():
            try:
                out.add(str(getattr(node_view, "id", "") or ""))
            except Exception:
                pass
        return out

    def _runner_child_undo_entries(
        self, node_data: dict[Any, Any], moved_node_ids: set[str]
    ) -> list[tuple[Any, list[float], list[float]]]:
        """
        Build undo entries for children moved by runner-drag.

        Children positions are updated continuously in `_on_nodes_moving`.
        Here we only record undo based on stored drag-start positions.
        """
        out: list[tuple[Any, list[float], list[float]]] = []
        for node_view in (node_data or {}).keys():
            try:
                runner = self._model.nodes.get(node_view.id)  # type: ignore[attr-defined]
            except Exception:
                runner = None
            if not isinstance(runner, OperatorRunnerRenderNode):
                continue

            rid = _node_id(runner)
            if not rid:
                continue

            start_map = self._move.runner_drag_child_start.get(rid) or {}
            if not start_map:
                continue

            for cid, c_prev in start_map.items():
                if not cid or cid in moved_node_ids:
                    continue
                try:
                    child = self.get_node_by_id(cid)
                except Exception:
                    child = None
                if child is None:
                    continue
                try:
                    c_cur = list(child.pos())
                except Exception:
                    continue
                out.append((child, list(c_cur), list(c_prev)))
        return out

    def _on_nodes_moving(self, node_data: dict[Any, Any]) -> None:
        """
        Continuous enforcement while dragging (no undo).

        - Runner drag: move children together in real-time.
        - Operator drag: clamp to runner in real-time.
        """
        if self._move.updating:
            return

        moved_node_ids = self._moved_node_ids(node_data)

        # 1) Runner drag: move children with the runner (real-time).
        self._move.updating = True
        try:
            for node_view, prev_pos in (node_data or {}).items():
                try:
                    runner = self._model.nodes.get(node_view.id)  # type: ignore[attr-defined]
                except Exception:
                    runner = None
                if not isinstance(runner, OperatorRunnerRenderNode):
                    continue

                rid = _node_id(runner)
                if not rid:
                    continue

                try:
                    cur = list(runner.pos())
                    prev = list(prev_pos)
                    dx = float(cur[0]) - float(prev[0])
                    dy = float(cur[1]) - float(prev[1])
                except Exception:
                    continue

                if dx == 0.0 and dy == 0.0:
                    continue

                start_map = self._move.runner_drag_child_start.get(rid)
                if start_map is None:
                    start_map = {}
                    try:
                        children = list(runner.contained_nodes() or [])
                    except Exception:
                        children = []
                    for child in children:
                        cid = _node_id(child)
                        if not cid:
                            continue
                        try:
                            start_map[cid] = list(child.pos())  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    self._move.runner_drag_child_start[rid] = start_map

                for cid, c_start in (start_map or {}).items():
                    if not cid or cid in moved_node_ids:
                        continue
                    try:
                        child = self.get_node_by_id(cid)
                    except Exception:
                        child = None
                    if child is None:
                        continue
                    c_next = [float(c_start[0]) + dx, float(c_start[1]) + dy]
                    self._set_node_xy(child, c_next)
                    self._move.last_ok_pos[cid] = list(c_next)
        finally:
            self._move.updating = False

        # 2) Operator drag: clamp inside its assigned runner (real-time).
        ops_by_runner: dict[str, tuple[OperatorRunnerRenderNode, list[Any]]] = {}
        for node_view, prev_pos in (node_data or {}).items():
            try:
                node = self._model.nodes.get(node_view.id)  # type: ignore[attr-defined]
            except Exception:
                node = None
            if node is None or not self._is_operator_node(node):
                continue
            runner = self._assigned_runner_for_operator(node)
            if runner is None:
                continue
            rid = _node_id(runner)
            if not rid:
                continue
            if rid not in ops_by_runner:
                ops_by_runner[rid] = (runner, [])
            ops_by_runner[rid][1].append(node)

        for rid, (runner, nodes) in ops_by_runner.items():
            runner_rect = _scene_rect(runner)
            if runner_rect is None:
                continue
            bounds: QtCore.QRectF | None = None
            for n in nodes:
                r = _scene_rect(n)
                if r is None:
                    continue
                bounds = r if bounds is None else bounds.united(r)
            if bounds is None or runner_rect.contains(bounds):
                continue

            delta = _clamp_delta(bounds, runner_rect)
            if delta.x() == 0.0 and delta.y() == 0.0:
                continue

            self._move.updating = True
            try:
                for n in nodes:
                    cur = list(n.pos())
                    next_pos = [float(cur[0]) + float(delta.x()), float(cur[1]) + float(delta.y())]
                    self._set_node_xy(n, next_pos)
            finally:
                self._move.updating = False
