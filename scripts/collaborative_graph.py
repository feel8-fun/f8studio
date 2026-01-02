from __future__ import annotations

import sys
import uuid
from typing import Any, Iterable

import pycrdt
from pycrdt._base import BaseType
from NodeGraphQt import BaseNode, NodeGraph
from qtpy import QtCore, QtWidgets


MSG_UPDATE = b"\x01"
MSG_SYNC_REQUEST = b"\x02"


class MockNetwork(QtCore.QObject):
    """
    一个最简单的“网络层”模拟：把 update(bytes) 广播给同进程里的其它客户端。

    真实网络实现时，只需要把 `broadcast(sender_id, update)` 换成 websocket / webrtc / tcp 即可。
    """

    message_broadcast = QtCore.Signal(str, bytes)  # (sender_id, payload)

    def broadcast(self, sender_id: str, payload: bytes) -> None:
        if not payload:
            print(f"[MockNetwork][{sender_id}]: empty 0 bytes")
            self.message_broadcast.emit(sender_id, payload)
            return
        msg_type = payload[:1]
        data_len = max(0, len(payload) - 1)
        if msg_type == MSG_UPDATE:
            kind = "update"
        elif msg_type == MSG_SYNC_REQUEST:
            kind = "sync_req"
        else:
            kind = f"unknown({msg_type!r})"
        print(f"[MockNetwork][{sender_id}]: {kind} {data_len} bytes")
        self.message_broadcast.emit(sender_id, payload)


GLOBAL_NETWORK = MockNetwork()


class SyncNode(BaseNode):
    __identifier__ = "f8.collab"
    NODE_NAME = "Sync Node"

    def __init__(self):
        super().__init__()
        self.add_input("in")
        self.add_output("out")
        self.create_property("label_text", "Sync Me")


def _is_primitive(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str, bytes))


def _to_crdt_value(value: Any) -> Any:
    """
    将 Python 值转换成 pycrdt 可写入 Map/Array 的值。

    - primitive: 直接写入
    - list/tuple: 转 Array
    - dict: 转 Map
    """
    if _is_primitive(value):
        return value
    if isinstance(value, (list, tuple)):
        return pycrdt.Array([_to_crdt_value(v) for v in value])
    if isinstance(value, dict):
        return pycrdt.Map({str(k): _to_crdt_value(v) for k, v in value.items()})
    if isinstance(value, BaseType):
        return value
    raise TypeError(f"Unsupported value type for CRDT: {type(value)!r}")


def _from_crdt_value(value: Any) -> Any:
    if isinstance(value, BaseType):
        return value.to_py()
    return value


def _edge_id(src_node_id: str, src_port: str, dst_node_id: str, dst_port: str) -> str:
    return f"{src_node_id}:{src_port}->{dst_node_id}:{dst_port}"


class CollaborativeGraphManager(QtCore.QObject):
    """
    用 pycrdt 做 NodeGraphQt 图数据的协同：

    CRDT schema (Map 为主，使用 Doc 根类型，避免把子 Map 作为 value 写入导致 LWW 覆盖):
      nodes (Map[node_id -> Map])
        node Map:
          type: str
          pos: Array[float, float]
          props: Map[str -> primitive/Array/Map]   # 只同步 custom property
      edges (Map[edge_id -> Map])
        edge Map:
          src: Map{node:str, port:str}   # output
          dst: Map{node:str, port:str}   # input

    说明：NodeGraphQt 内部的 `node.id` / `name` 有自己的约束与唯一化规则，
    协同层使用 `crdt_id` 自己管理稳定 ID，并用 `crdt_id` 派生一个稳定的 node name（避免跨端 name 冲突）。
    """

    def __init__(
        self,
        client_id: str,
        graph: NodeGraph,
        *,
        network: MockNetwork = GLOBAL_NETWORK,
        ydoc: pycrdt.Doc | None = None,
        synced_custom_properties: set[str] | None = None,
        broadcast_debounce_ms: int = 30,
    ) -> None:
        super().__init__()
        self.client_id = client_id
        self.graph = graph
        self.network = network
        self.synced_custom_properties = synced_custom_properties
        self._broadcast_debounce_ms = max(0, int(broadcast_debounce_ms))

        self.ydoc = ydoc or pycrdt.Doc()
        self.ynodes: pycrdt.Map = self.ydoc.get("nodes", type=pycrdt.Map)
        self.yedges: pycrdt.Map = self.ydoc.get("edges", type=pycrdt.Map)

        self._applying_crdt = False
        self._suppress_broadcast = False
        self._broadcast_timer: QtCore.QTimer | None = None
        self._pending_updates: list[bytes] = []

        self._nodes_by_crdt_id: dict[str, Any] = {}
        self._crdt_id_by_nodegraph_id: dict[str, str] = {}

        self._subscription_nodes = self.ynodes.observe_deep(self._on_crdt_events)
        self._subscription_edges = self.yedges.observe_deep(self._on_crdt_events)
        self._subscription_doc = self.ydoc.observe(self._on_doc_transaction)

        self.graph.node_created.connect(self._on_node_created)
        self.graph.nodes_deleted.connect(self._on_nodes_deleted)
        self.graph.property_changed.connect(self._on_property_changed)
        self.graph.port_connected.connect(self._on_port_connected)
        self.graph.port_disconnected.connect(self._on_port_disconnected)
        self.graph.viewer().moved_nodes.connect(self._on_nodes_moved)

        self.network.message_broadcast.connect(self._on_network_message)

        self._rebuild_node_index()
        self.sync_now()
        self.request_sync()

    # ----- Public API -----
    def sync_now(self) -> None:
        self._apply_crdt_to_graph()

    # ----- Network -----
    def _enqueue_broadcast(self, update: bytes) -> None:
        if not update:
            return
        payload = MSG_UPDATE + update
        if self._broadcast_debounce_ms == 0:
            self.network.broadcast(self.client_id, payload)
            return
        self._pending_updates.append(payload)
        if self._broadcast_timer is None:
            self._broadcast_timer = QtCore.QTimer(self)
            self._broadcast_timer.setSingleShot(True)
            self._broadcast_timer.timeout.connect(self._flush_pending_updates)
        if not self._broadcast_timer.isActive():
            self._broadcast_timer.start(self._broadcast_debounce_ms)

    def _flush_pending_updates(self) -> None:
        updates = self._pending_updates
        self._pending_updates = []
        for payload in updates:
            self.network.broadcast(self.client_id, payload)

    def _on_doc_transaction(self, event: Any) -> None:
        """
        通过 Doc.observe 拿到“本次 transaction 的增量 update”，而不是 get_update() 的全量（从 state=0 开始会不断增长）。
        """
        if self._suppress_broadcast:
            return
        update = getattr(event, "update", b"")
        if isinstance(update, (bytes, bytearray)) and update:
            self._enqueue_broadcast(bytes(update))

    def request_sync(self) -> None:
        """
        新加入（或重连）时调用：广播本端 state vector，让其他 peer 计算“你缺的 update”发回来。
        """
        state = self.ydoc.get_state()
        self.network.broadcast(self.client_id, MSG_SYNC_REQUEST + state)

    def _on_network_message(self, sender_id: str, payload: bytes) -> None:
        if sender_id == self.client_id:
            return
        if not payload:
            return

        msg_type = payload[:1]
        data = payload[1:]

        if msg_type == MSG_UPDATE:
            self._suppress_broadcast = True
            try:
                self.ydoc.apply_update(data)
            finally:
                self._suppress_broadcast = False
            return

        if msg_type == MSG_SYNC_REQUEST:
            peer_state = data
            update = self.ydoc.get_update(peer_state)
            if update:
                self.network.broadcast(self.client_id, MSG_UPDATE + update)
            return

        return

    # ----- Local graph -> CRDT -----
    def _ensure_node_crdt_id(self, node: Any) -> str:
        if node.model.is_custom_property("crdt_id"):
            crdt_id = node.get_property("crdt_id")
            if isinstance(crdt_id, str) and crdt_id:
                return crdt_id

        crdt_id = uuid.uuid4().hex
        node.create_property("crdt_id", crdt_id)

        node.set_property("name", f"node_{crdt_id[:8]}", push_undo=False)
        return crdt_id

    def _collect_synced_custom_props(self, node: Any) -> dict[str, Any]:
        props: dict[str, Any] = {}
        for key, value in node.model.custom_properties.items():
            if key == "crdt_id":
                continue
            if self.synced_custom_properties is not None and key not in self.synced_custom_properties:
                continue
            try:
                props[key] = _to_crdt_value(value)
            except TypeError:
                continue
        return props

    def _on_node_created(self, node: Any) -> None:
        if self._applying_crdt:
            return

        crdt_id = self._ensure_node_crdt_id(node)
        self._nodes_by_crdt_id[crdt_id] = node
        self._crdt_id_by_nodegraph_id[node.id] = crdt_id

        pos = list(node.pos())
        props = self._collect_synced_custom_props(node)

        with self.ydoc.transaction():
            # 先把 Map 插入到 Doc 里（integrate），再写入字段，避免 “Not integrated in a document yet”。
            if crdt_id not in self.ynodes:
                self.ynodes[crdt_id] = pycrdt.Map()
            node_map = self.ynodes.get(crdt_id)
            if not isinstance(node_map, pycrdt.Map):
                raise TypeError(f"CRDT schema mismatch: nodes[{crdt_id!r}] is not a Map")
            node_map["type"] = node.type_
            node_map["pos"] = pycrdt.Array([float(pos[0]), float(pos[1])])
            node_map["props"] = pycrdt.Map(props)

    def _on_nodes_deleted(self, node_ids: list[str]) -> None:
        if self._applying_crdt:
            return

        removed_crdt_ids = [self._crdt_id_by_nodegraph_id.get(nid) for nid in node_ids]
        removed_crdt_ids = [cid for cid in removed_crdt_ids if cid]
        if not removed_crdt_ids:
            return

        with self.ydoc.transaction():
            for crdt_id in removed_crdt_ids:
                if crdt_id in self.ynodes:
                    del self.ynodes[crdt_id]
                self._nodes_by_crdt_id.pop(crdt_id, None)

            to_delete: list[str] = []
            for eid, emap in self.yedges.items():
                src = emap.get("src")
                dst = emap.get("dst")
                if not isinstance(src, pycrdt.Map) or not isinstance(dst, pycrdt.Map):
                    continue
                src_node = _from_crdt_value(src.get("node"))
                dst_node = _from_crdt_value(dst.get("node"))
                if src_node in removed_crdt_ids or dst_node in removed_crdt_ids:
                    to_delete.append(eid)
            for eid in to_delete:
                if eid in self.yedges:
                    del self.yedges[eid]

        for nid in node_ids:
            self._crdt_id_by_nodegraph_id.pop(nid, None)

    def _on_nodes_moved(self, node_data: dict[Any, Any]) -> None:
        if self._applying_crdt:
            return

        updates: list[tuple[str, list[float]]] = []
        for node_view in node_data.keys():
            node = self.graph.get_node_by_id(node_view.id)
            if not node or not node.model.is_custom_property("crdt_id"):
                continue
            crdt_id = node.get_property("crdt_id")
            if not isinstance(crdt_id, str) or not crdt_id:
                continue
            pos = list(node.pos())
            updates.append((crdt_id, [float(pos[0]), float(pos[1])]))

        if not updates:
            return

        with self.ydoc.transaction():
            for crdt_id, pos in updates:
                node_map = self.ynodes.get(crdt_id)
                if not isinstance(node_map, pycrdt.Map):
                    continue
                node_map["pos"] = pycrdt.Array(pos)

    def _on_property_changed(self, node: Any, prop_name: str, prop_value: Any) -> None:
        if self._applying_crdt:
            return
        if prop_name == "crdt_id":
            return
        if not node.model.is_custom_property("crdt_id"):
            return

        crdt_id = node.get_property("crdt_id")
        if not isinstance(crdt_id, str) or not crdt_id:
            return

        if prop_name != "pos" and not node.model.is_custom_property(prop_name):
            return
        if self.synced_custom_properties is not None and prop_name != "pos":
            if prop_name not in self.synced_custom_properties:
                return

        with self.ydoc.transaction():
            node_map = self.ynodes.get(crdt_id)
            if not isinstance(node_map, pycrdt.Map):
                return
            if prop_name == "pos":
                pos = list(prop_value)
                node_map["pos"] = pycrdt.Array([float(pos[0]), float(pos[1])])
            else:
                props_map = node_map.get("props")
                if not isinstance(props_map, pycrdt.Map):
                    props_map = pycrdt.Map()
                    node_map["props"] = props_map
                try:
                    props_map[prop_name] = _to_crdt_value(prop_value)
                except TypeError:
                    return

    def _on_port_connected(self, in_port: Any, out_port: Any) -> None:
        if self._applying_crdt:
            return

        out_node = out_port.node()
        in_node = in_port.node()
        if not (out_node.model.is_custom_property("crdt_id") and in_node.model.is_custom_property("crdt_id")):
            return

        src_id = out_node.get_property("crdt_id")
        dst_id = in_node.get_property("crdt_id")
        if not (isinstance(src_id, str) and isinstance(dst_id, str)):
            return

        eid = _edge_id(src_id, out_port.name(), dst_id, in_port.name())

        with self.ydoc.transaction():
            if eid not in self.yedges:
                self.yedges[eid] = pycrdt.Map()
            edge_map = self.yedges.get(eid)
            if not isinstance(edge_map, pycrdt.Map):
                raise TypeError(f"CRDT schema mismatch: edges[{eid!r}] is not a Map")
            edge_map["src"] = pycrdt.Map({"node": src_id, "port": out_port.name()})
            edge_map["dst"] = pycrdt.Map({"node": dst_id, "port": in_port.name()})

    def _on_port_disconnected(self, in_port: Any, out_port: Any) -> None:
        if self._applying_crdt:
            return

        out_node = out_port.node()
        in_node = in_port.node()
        if not (out_node.model.is_custom_property("crdt_id") and in_node.model.is_custom_property("crdt_id")):
            return

        src_id = out_node.get_property("crdt_id")
        dst_id = in_node.get_property("crdt_id")
        if not (isinstance(src_id, str) and isinstance(dst_id, str)):
            return

        eid = _edge_id(src_id, out_port.name(), dst_id, in_port.name())
        with self.ydoc.transaction():
            if eid in self.yedges:
                del self.yedges[eid]

    # ----- CRDT -> Local graph -----
    def _on_crdt_events(self, _events: list[Any]) -> None:
        self._apply_crdt_to_graph()

    def _rebuild_node_index(self) -> None:
        self._nodes_by_crdt_id.clear()
        self._crdt_id_by_nodegraph_id.clear()
        for node in self.graph.all_nodes():
            if not node.model.is_custom_property("crdt_id"):
                continue
            crdt_id = node.get_property("crdt_id")
            if not isinstance(crdt_id, str) or not crdt_id:
                continue
            self._nodes_by_crdt_id[crdt_id] = node
            self._crdt_id_by_nodegraph_id[node.id] = crdt_id

    def _iter_local_edges(self) -> Iterable[tuple[str, Any, Any]]:
        for node in self.graph.all_nodes():
            if not isinstance(node, BaseNode):
                continue
            if not node.model.is_custom_property("crdt_id"):
                continue
            src_id = node.get_property("crdt_id")
            if not isinstance(src_id, str) or not src_id:
                continue
            for out_port in node.output_ports():
                for in_port in out_port.connected_ports():
                    in_node = in_port.node()
                    if not isinstance(in_node, BaseNode):
                        continue
                    if not in_node.model.is_custom_property("crdt_id"):
                        continue
                    dst_id = in_node.get_property("crdt_id")
                    if not isinstance(dst_id, str) or not dst_id:
                        continue
                    eid = _edge_id(src_id, out_port.name(), dst_id, in_port.name())
                    yield eid, out_port, in_port

    def _apply_crdt_to_graph(self) -> None:
        self._applying_crdt = True
        try:
            self._rebuild_node_index()

            # --- nodes ---
            desired_node_ids = set(self.ynodes.keys())
            local_node_ids = set(self._nodes_by_crdt_id.keys())

            for crdt_id in sorted(local_node_ids - desired_node_ids):
                node = self._nodes_by_crdt_id.get(crdt_id)
                if node:
                    self.graph.delete_node(node, push_undo=False)

            for crdt_id in sorted(desired_node_ids - local_node_ids):
                node_map = self.ynodes.get(crdt_id)
                if not isinstance(node_map, pycrdt.Map):
                    continue
                node_type = _from_crdt_value(node_map.get("type"))
                if not isinstance(node_type, str) or not node_type:
                    continue

                pos_val = node_map.get("pos")
                pos_py = _from_crdt_value(pos_val) or [0.0, 0.0]
                try:
                    pos = [float(pos_py[0]), float(pos_py[1])]
                except Exception:
                    pos = [0.0, 0.0]

                node = self.graph.node_factory.create_node_instance(node_type)
                node.NODE_NAME = f"node_{crdt_id[:8]}"
                if not node.model.is_custom_property("crdt_id"):
                    node.create_property("crdt_id", crdt_id)

                self.graph.add_node(node, pos=pos, selected=False, push_undo=False)

            self._rebuild_node_index()

            for crdt_id in desired_node_ids:
                node = self._nodes_by_crdt_id.get(crdt_id)
                node_map = self.ynodes.get(crdt_id)
                if not node or not isinstance(node_map, pycrdt.Map):
                    continue

                pos_py = _from_crdt_value(node_map.get("pos"))
                if isinstance(pos_py, list) and len(pos_py) >= 2:
                    new_pos = [float(pos_py[0]), float(pos_py[1])]
                    if list(node.pos()) != new_pos:
                        node.set_property("pos", new_pos, push_undo=False)

                props_map = node_map.get("props")
                if isinstance(props_map, pycrdt.Map):
                    for key, val in props_map.items():
                        if self.synced_custom_properties is not None and key not in self.synced_custom_properties:
                            continue
                        py_val = _from_crdt_value(val)
                        if node.model.is_custom_property(key) and node.get_property(key) != py_val:
                            node.set_property(key, py_val, push_undo=False)

            # --- edges ---
            desired_edges: dict[str, tuple[str, str, str, str]] = {}
            for eid, emap in self.yedges.items():
                if not isinstance(emap, pycrdt.Map):
                    continue
                src = emap.get("src")
                dst = emap.get("dst")
                if not isinstance(src, pycrdt.Map) or not isinstance(dst, pycrdt.Map):
                    continue
                src_node = _from_crdt_value(src.get("node"))
                src_port = _from_crdt_value(src.get("port"))
                dst_node = _from_crdt_value(dst.get("node"))
                dst_port = _from_crdt_value(dst.get("port"))
                if not all(isinstance(x, str) and x for x in [src_node, src_port, dst_node, dst_port]):
                    continue
                desired_edges[eid] = (src_node, src_port, dst_node, dst_port)

            local_edges: dict[str, tuple[Any, Any]] = {}
            for eid, out_port, in_port in self._iter_local_edges():
                local_edges[eid] = (out_port, in_port)

            for eid in sorted(set(local_edges.keys()) - set(desired_edges.keys())):
                out_port, in_port = local_edges[eid]
                out_port.disconnect_from(in_port, push_undo=False, emit_signal=False)

            for eid in sorted(set(desired_edges.keys()) - set(local_edges.keys())):
                src_node, src_port, dst_node, dst_port = desired_edges[eid]
                src = self._nodes_by_crdt_id.get(src_node)
                dst = self._nodes_by_crdt_id.get(dst_node)
                if not (isinstance(src, BaseNode) and isinstance(dst, BaseNode)):
                    continue
                out_port = src.outputs().get(src_port)
                in_port = dst.inputs().get(dst_port)
                if not out_port or not in_port:
                    continue
                out_port.connect_to(in_port, push_undo=False, emit_signal=False)

        finally:
            self._applying_crdt = False


class DualWindowDemo(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pycrdt + NodeGraphQt (Dynamic Clients + Late Join)")
        self.resize(1400, 700)

        self._clients: list[tuple[str, NodeGraph, CollaborativeGraphManager]] = []

        root = QtWidgets.QVBoxLayout(self)

        top_bar = QtWidgets.QHBoxLayout()
        self.btn_add_client = QtWidgets.QPushButton("Add Client")
        self.btn_add_client.clicked.connect(self._on_add_client_clicked)
        top_bar.addWidget(self.btn_add_client)
        top_bar.addStretch(1)
        root.addLayout(top_bar)

        self._clients_container = QtWidgets.QWidget()
        self._clients_layout = QtWidgets.QHBoxLayout(self._clients_container)
        self._clients_layout.setContentsMargins(0, 0, 0, 0)
        self._clients_layout.setSpacing(10)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._clients_container)
        root.addWidget(scroll)

        self._add_client("Client_A")
        self._add_client("Client_B")
        self._init_demo_state()

    def _init_demo_state(self) -> None:
        graph_a = self._get_client_graph("Client_A")
        if graph_a is None:
            return

        n1 = graph_a.create_node("f8.collab.SyncNode", name="A", pos=[-200, 0])
        n2 = graph_a.create_node("f8.collab.SyncNode", name="B", pos=[200, 0])
        n1.set_property("label_text", "Move me / late join should catch up", push_undo=False)
        n1.outputs()["out"].connect_to(n2.inputs()["in"])

    def _get_client_graph(self, client_id: str) -> NodeGraph | None:
        for cid, graph, _mgr in self._clients:
            if cid == client_id:
                return graph
        return None

    def _next_client_id(self) -> str:
        existing = {cid for cid, _g, _m in self._clients}
        for i in range(0, 1000):
            cid = f"Client_{chr(ord('A') + i)}"
            if cid not in existing:
                return cid
        return f"Client_{uuid.uuid4().hex[:6]}"

    def _add_client(self, client_id: str) -> None:
        graph = NodeGraph()
        graph.register_node(SyncNode)
        manager = CollaborativeGraphManager(
            client_id,
            graph,
            synced_custom_properties={"label_text", "crdt_id"},
            network=GLOBAL_NETWORK,
        )
        self._clients.append((client_id, graph, manager))

        group = QtWidgets.QGroupBox(client_id)
        box = QtWidgets.QVBoxLayout(group)
        box.addWidget(graph.widget)
        self._clients_layout.addWidget(group)

    def _on_add_client_clicked(self) -> None:
        self._add_client(self._next_client_id())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Qt6 下该 attribute 已标记 deprecated；仅在 Qt5 时启用。
    try:
        qt_major = int(QtCore.qVersion().split(".")[0])
    except Exception:
        qt_major = 6
    if qt_major < 6 and hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    demo = DualWindowDemo()
    demo.show()

    exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
    sys.exit(exec_fn())
