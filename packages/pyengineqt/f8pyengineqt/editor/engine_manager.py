from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from typing import Any

from qtpy import QtCore

from ..renderers.generic import GenericNode
from ..engine.nats_naming import ensure_token, kv_bucket_for_service
from .operator_graph_export import export_operator_graph
from .service_runtime_bridge import ServiceRuntimeBridge, ServiceRuntimeBridgeConfig


def _now_ms() -> int:
    return int(QtCore.QDateTime.currentMSecsSinceEpoch())


def _kv_key(node_id: str, field: str) -> str:
    node_id = ensure_token(str(node_id), label="node_id")
    # field may contain ".", parse uses join of remaining tokens.
    return f"nodes.{node_id}.state.{field}"


def _parse_kv_key(key: str) -> tuple[str, str] | None:
    parts = str(key).strip(".").split(".")
    # nodes.<nodeId>.state.<field...>
    if len(parts) < 4:
        return None
    if parts[0] != "nodes" or parts[2] != "state":
        return None
    node_id = parts[1]
    if not node_id or "." in node_id:
        return None
    field = ".".join(parts[3:])
    if not field:
        return None
    return node_id, field


def _actor_id() -> str:
    aid = os.environ.get("F8_ACTOR_ID")
    return aid.strip() if aid and aid.strip() else uuid.uuid4().hex


def _service_id() -> str:
    """
    Service instance id (eg. an engine block node id).

    Design: serviceId == service node id (uuid), so we don't store a separate id.
    """
    sid = os.environ.get("F8_SERVICE_ID")
    if sid and sid.strip():
        return ensure_token(sid.strip(), label="service_id")
    # Backward compatibility (deprecated): older builds used F8_ENGINE_ID.
    legacy = os.environ.get("F8_ENGINE_ID")
    if legacy and legacy.strip():
        return ensure_token(legacy.strip(), label="service_id")
    return ensure_token(uuid.uuid4().hex, label="service_id")


class EngineManager(QtCore.QObject):
    """
    First-version engine bridge:
    - publish full rungraph snapshot (includes per-node specs) to NATS KV.
    - publish state KV entries to NATS KV and apply remote KV updates to nodes.
    """

    statusChanged = QtCore.Signal(str)

    def __init__(
        self,
        editor: Any,
        *,
        service_id: str | None = None,
        actor_id: str | None = None,
        debounce_ms: int = 60,
        node_filter: Callable[[GenericNode], bool] | None = None,
        edge_meta: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self._editor = editor
        graph = getattr(editor, "node_graph", None)
        self._graph = graph if graph is not None else editor
        self._service_id = (service_id or "").strip() or _service_id()
        self._actor_id = (actor_id or "").strip() or _actor_id()
        self._node_filter = node_filter
        self._edge_meta = edge_meta
        if hasattr(editor, "to_operator_graph"):
            self._export_rungraph = lambda: editor.to_operator_graph().to_dict(include_ctx=False)  # type: ignore[attr-defined]
        else:
            self._export_rungraph = lambda: export_operator_graph(
                self._graph, service_id=self._service_id, node_filter=self._node_filter, edge_meta=self._edge_meta
            ).to_dict(include_ctx=False)

        self._applying_engine = False
        self._pending_rungraph = False
        self._pending_kv: dict[str, Any] = {}
        self._nats: ServiceRuntimeBridge | None = None
        self._seeded_nodes: set[str] = set()

        self._rungraph_timer = QtCore.QTimer(self)
        self._rungraph_timer.setSingleShot(True)
        self._rungraph_timer.timeout.connect(self._flush_rungraph)

        self._kv_timer = QtCore.QTimer(self)
        self._kv_timer.setSingleShot(True)
        self._kv_timer.timeout.connect(self._flush_kv)

        self._debounce_ms = max(0, int(debounce_ms))

        if self._graph is not None:
            self._graph.node_created.connect(self._on_node_created)
            self._graph.nodes_deleted.connect(lambda *_a: self.schedule_rungraph_sync())
            self._graph.port_connected.connect(lambda *_a: self.schedule_rungraph_sync())
            self._graph.port_disconnected.connect(lambda *_a: self.schedule_rungraph_sync())
            self._graph.property_changed.connect(self._on_property_changed)
            try:
                self._graph.viewer().moved_nodes.connect(lambda *_a: self.schedule_rungraph_sync())
            except Exception:
                pass
            try:
                self._graph.viewer().node_backdrop_updated.connect(lambda *_a: self.schedule_rungraph_sync())
            except Exception:
                pass

        app = QtCore.QCoreApplication.instance()
        if app is not None:
            try:
                app.aboutToQuit.connect(self.stop)
            except Exception:
                pass

    def start(self) -> None:
        if self._nats is not None:
            return

        nats_url = (os.environ.get("F8_NATS_URL") or "nats://127.0.0.1:4222").strip()
        # One service instance -> one KV bucket (do not mix across processes).
        bucket = (os.environ.get("F8_NATS_BUCKET") or "").strip() or kv_bucket_for_service(self._service_id)
        self._nats = ServiceRuntimeBridge(
            ServiceRuntimeBridgeConfig(service_id=self._service_id, nats_url=nats_url, kv_bucket=bucket, actor_id=self._actor_id)
        )
        self._nats.statusChanged.connect(self.statusChanged)
        self._nats.stateUpdated.connect(self._on_state_updated)
        self._nats.start()
        self.statusChanged.emit(f"service: nats mode serviceId={self._service_id} bucket={bucket}")
        self._seed_existing_nodes()
        self.schedule_rungraph_sync()

    def runtime_bridge(self) -> ServiceRuntimeBridge | None:
        return self._nats

    def _seed_existing_nodes(self) -> None:
        graph = self._graph
        if graph is None:
            return
        try:
            nodes = list(graph.all_nodes())
        except Exception:
            nodes = []
        for node in nodes:
            if not isinstance(node, GenericNode):
                continue
            if self._node_filter is not None and not self._node_filter(node):
                continue
            node_id = str(node.id)
            if node_id in self._seeded_nodes:
                continue
            try:
                fields = [s.name for s in (node.spec.stateFields or [])]
            except Exception:
                fields = []
            for name in fields:
                try:
                    self._pending_kv[_kv_key(node.id, name)] = node.get_property(name)
                except Exception:
                    continue
            self._seeded_nodes.add(node_id)
        if self._pending_kv:
            if self._debounce_ms == 0:
                self._flush_kv()
            elif not self._kv_timer.isActive():
                self._kv_timer.start(self._debounce_ms)

    def stop(self) -> None:
        self._rungraph_timer.stop()
        self._kv_timer.stop()

        if self._nats is not None:
            try:
                self._nats.stop()
            except Exception:
                pass
        self._nats = None
        self._seeded_nodes.clear()

    def schedule_rungraph_sync(self) -> None:
        # When scoping by a node filter, nodes can move into scope without a
        # "node_created" event; ensure initial KV keys exist for newly-in-scope nodes.
        try:
            self._seed_existing_nodes()
        except Exception:
            pass
        self._pending_rungraph = True
        if self._debounce_ms == 0:
            self._flush_rungraph()
            return
        if not self._rungraph_timer.isActive():
            self._rungraph_timer.start(self._debounce_ms)

    def _flush_rungraph(self) -> None:
        if not self._pending_rungraph:
            return
        self._pending_rungraph = False
        if self._nats is None:
            self.start()
        if self._nats is None:
            return
        try:
            payload = self._export_rungraph()
        except Exception as exc:
            self.statusChanged.emit(f"engine: rungraph export failed ({exc})")
            return

        try:
            self._nats.put_rungraph(payload)
        except Exception as exc:
            self.statusChanged.emit(f"nats: put rungraph failed ({exc})")

    def _on_property_changed(self, node: Any, name: str, value: Any) -> None:
        if self._applying_engine:
            return
        # Layout edits (node positions, backdrops resizing) should update rungraph.
        if name in ("pos", "width", "height"):
            self.schedule_rungraph_sync()
            return
        if not isinstance(node, GenericNode):
            return
        if self._node_filter is not None and not self._node_filter(node):
            return
        if name.startswith("__"):
            # internal emit (eg. spec changed) -> rungraph sync
            self.schedule_rungraph_sync()
            return
        try:
            state_fields = {s.name: s for s in (node.spec.stateFields or [])}
        except Exception:
            state_fields = {}
        if name not in state_fields:
            return

        self._pending_kv[_kv_key(node.id, name)] = value
        if self._debounce_ms == 0:
            self._flush_kv()
            return
        if not self._kv_timer.isActive():
            self._kv_timer.start(self._debounce_ms)

    def _flush_kv(self) -> None:
        if not self._pending_kv:
            return
        if self._nats is None:
            self.start()
        if self._nats is None:
            return
        pending = self._pending_kv
        self._pending_kv = {}

        for key, value in pending.items():
            parsed = _parse_kv_key(key)
            if not parsed:
                continue
            node_id, field = parsed
            try:
                self._nats.put_state(node_id, field, value, source="editor", ts_ms=_now_ms())
            except Exception:
                continue

    def _on_state_updated(self, node_id: str, field: str, value: object, ts_ms: object, _meta: object) -> None:
        try:
            ts = int(ts_ms) if ts_ms is not None else _now_ms()
        except Exception:
            ts = _now_ms()
        self._apply_engine_kv_updates(
            [
                {
                    "key": _kv_key(node_id, field),
                    "value": value,
                    "source": "engine",
                    "ts": ts,
                }
            ]
        )

    def _on_node_created(self, node: Any) -> None:
        self.schedule_rungraph_sync()
        if not isinstance(node, GenericNode):
            return
        if self._node_filter is not None and not self._node_filter(node):
            return
        try:
            fields = [s.name for s in (node.spec.stateFields or [])]
        except Exception:
            fields = []
        for name in fields:
            try:
                self._pending_kv[_kv_key(node.id, name)] = node.get_property(name)
            except Exception:
                continue
        if self._pending_kv:
            if self._debounce_ms == 0:
                self._flush_kv()
            elif not self._kv_timer.isActive():
                self._kv_timer.start(self._debounce_ms)

    def _apply_engine_kv_updates(self, writes: list[dict[str, Any]]) -> None:
        graph = self._graph
        if graph is None:
            return
        self._applying_engine = True
        try:
            for w in writes:
                key = str(w.get("key") or "")
                parsed = _parse_kv_key(key)
                if not parsed:
                    continue
                node_id, field = parsed
                try:
                    node = graph.get_node_by_id(node_id)
                except Exception:
                    node = None
                if not isinstance(node, GenericNode):
                    continue
                if self._node_filter is not None and not self._node_filter(node):
                    continue
                try:
                    if node.get_property(field) == w.get("value"):
                        continue
                except Exception:
                    pass
                try:
                    node.set_property(field, w.get("value"), push_undo=False)
                except Exception:
                    pass
        finally:
            self._applying_engine = False
