from __future__ import annotations

from pathlib import Path
import os
from typing import Any, cast

from NodeGraphQt import NodeGraph
from qtpy import QtCore, QtWidgets

from ..renderers.generic import GenericNode
from .engine_manager import EngineManager
from .f8_node_viewer import F8NodeViewer
from .operator_graph_export import export_operator_graph
from .spec_node_class_registry import SpecNodeClassRegistry
from ..renderers.renderer_registry import RendererRegistry
from ..services.builtin import ENGINE_SERVICE_CLASS, EDITOR_SERVICE_CLASS
from ..services.service_registry import ServiceSpecRegistry
from ..services.discovery_loader import load_discovery_into_registries
from ..renderers.service_engine import EngineServiceNode
from f8pysdk import F8EdgeStrategyEnum
from .service_process_manager import ServiceProcessConfig, ServiceProcessManager

BASE_PATH = Path(__file__).parent
EDITOR_SERVICE_ID = "editor"


class ServiceGraphEditor:
    """
    Service graph editor (v1).

    - Root graph: contains service nodes (engine is a Backdrop-style grouping node).
    - Operators live in the same graph view (no sub-graph tabs).
    - Each engine service node runs its own EngineManager scoped to the operator
      nodes enclosed by the engine backdrop (serviceId == engine node id).
    """

    def __init__(self) -> None:
        self.node_graph = NodeGraph(viewer=F8NodeViewer())
        self.node_graph._node_factory.clear_registered_nodes()
        self._engine_sync: dict[str, EngineManager] = {}
        self._engine_geom: dict[str, dict[str, Any]] = {}
        self._operator_owner: dict[str, str] = {}
        self._enforcing = False
        self._svc_procs = ServiceProcessManager()
        self._svc_procs.statusChanged.connect(lambda s: print(s))
        self._editor_mgr: EngineManager | None = None
        self._editor_pollers: dict[str, QtCore.QTimer] = {}
        self._editor_pending: dict[str, object] = {}
        self._editor_last: dict[str, str] = {}
        self._editor_buffers: dict[str, list[float]] = {}

        hotkey_path = BASE_PATH / "hotkeys" / "hotkeys.json"
        self.node_graph.set_context_menu_from_file(str(hotkey_path), "graph")

        RendererRegistry.instance()
        load_discovery_into_registries()

        # Root palette supports both service nodes and operator nodes (operators are typically created inside engines).
        SpecNodeClassRegistry.instance().apply(self.node_graph)
        # Engine service nodes are registered via SpecNodeClassRegistry (ServiceSpecRegistry).

        self.node_graph.node_created.connect(self._on_node_created)
        self.node_graph.nodes_deleted.connect(self._on_nodes_deleted)
        self.node_graph.property_changed.connect(self._on_property_changed)
        try:
            viewer = self.node_graph.viewer()
            viewer.moved_nodes.connect(self._on_viewer_moved_nodes)
            viewer.node_backdrop_updated.connect(self._on_viewer_backdrop_updated)
        except Exception:
            pass

        app = QtCore.QCoreApplication.instance()
        if app is not None:
            try:
                app.aboutToQuit.connect(self.stop)
            except Exception:
                pass

        self._ensure_editor_manager()

    def stop(self) -> None:
        for mgr in list(self._engine_sync.values()):
            try:
                mgr.stop()
            except Exception:
                pass
        self._engine_sync.clear()
        self._engine_geom.clear()
        self._operator_owner.clear()
        for t in list(self._editor_pollers.values()):
            try:
                t.stop()
            except Exception:
                pass
        self._editor_pollers.clear()
        self._editor_pending.clear()
        self._editor_last.clear()
        self._editor_buffers.clear()
        if self._editor_mgr is not None:
            try:
                self._editor_mgr.stop()
            except Exception:
                pass
        self._editor_mgr = None
        try:
            self._svc_procs.stop_all()
        except Exception:
            pass

    def _engine_nodes(self) -> list[EngineServiceNode]:
        return [n for n in self.node_graph.all_nodes() if isinstance(n, EngineServiceNode)]

    @staticmethod
    def _is_editor_node(node: GenericNode) -> bool:
        try:
            spec = getattr(node, "spec", None)
            svc = str(getattr(spec, "serviceClass", "") or "").strip()
            if svc:
                return svc == EDITOR_SERVICE_CLASS
            op = str(getattr(spec, "operatorClass", "") or "")
            if op.startswith("feel8.editor."):
                return True
            tags = set(getattr(spec, "tags", None) or [])
            return "editor" in tags
        except Exception:
            return False

    def _service_id_for_node(self, node: GenericNode) -> str:
        if self._is_editor_node(node):
            return EDITOR_SERVICE_ID
        eng = self._engine_for_operator(node)
        return str(eng.id) if eng is not None else ""

    def _scene_rect(self, node: object) -> QtCore.QRectF | None:
        try:
            view = getattr(node, "view", None)
            if view is None:
                return None
            if not isinstance(view, QtWidgets.QGraphicsItem):
                return None
            return cast(QtCore.QRectF, view.sceneBoundingRect())
        except Exception:
            return None

    def _remember_engine_geom(self, engine: EngineServiceNode) -> None:
        sid = str(engine.id)
        try:
            pos = engine.get_property("pos")
        except Exception:
            pos = None
        try:
            width = engine.get_property("width")
        except Exception:
            width = None
        try:
            height = engine.get_property("height")
        except Exception:
            height = None
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            pos_val = [float(pos[0]), float(pos[1])]
        else:
            rect = self._scene_rect(engine)
            pos_val = [float(rect.x()), float(rect.y())] if rect is not None else [0.0, 0.0]
        self._engine_geom[sid] = {
            "pos": pos_val,
            "width": float(width) if width is not None else None,
            "height": float(height) if height is not None else None,
        }

    def _engine_rect_proposed(self, *, engine: EngineServiceNode, geom: dict[str, Any] | None = None) -> QtCore.QRectF | None:
        if geom:
            try:
                pos = geom.get("pos")
                width = geom.get("width")
                height = geom.get("height")
                if isinstance(pos, (list, tuple)) and len(pos) == 2 and width and height:
                    return QtCore.QRectF(float(pos[0]), float(pos[1]), float(width), float(height))
            except Exception:
                pass
        return self._scene_rect(engine)

    def _operator_owner_engine(self, node: GenericNode) -> EngineServiceNode | None:
        owner = self._operator_owner.get(str(node.id))
        if owner:
            try:
                return next(e for e in self._engine_nodes() if str(e.id) == owner)
            except Exception:
                return None
        return None

    def _is_inside_engine(self, engine: EngineServiceNode, node: GenericNode, *, engine_rect: QtCore.QRectF | None = None) -> bool:
        rect_engine = engine_rect or self._engine_rect_proposed(engine=engine)
        rect_node = self._scene_rect(node)
        if rect_engine is None or rect_node is None:
            return False
        padded = rect_engine.adjusted(1.0, 1.0, -1.0, -1.0)
        return bool(padded.contains(rect_node))

    def _engines_overlap(
        self,
        a: EngineServiceNode,
        b: EngineServiceNode,
        *,
        a_rect: QtCore.QRectF | None = None,
        b_rect: QtCore.QRectF | None = None,
    ) -> bool:
        ra = a_rect or self._engine_rect_proposed(engine=a)
        rb = b_rect or self._engine_rect_proposed(engine=b)
        if ra is None or rb is None:
            return False
        inter = ra.intersected(rb)
        return bool(inter.isValid() and inter.width() > 2.0 and inter.height() > 2.0)

    def _closest_engine(self, node: GenericNode) -> EngineServiceNode | None:
        rect_node = self._scene_rect(node)
        if rect_node is None:
            return None
        c = rect_node.center()
        engines = self._engine_nodes()
        if not engines:
            return None

        def _dist2(e: EngineServiceNode) -> float:
            r = self._engine_rect_proposed(engine=e)
            if r is None:
                return float("inf")
            d = r.center() - c
            return float(d.x() * d.x() + d.y() * d.y())

        return sorted(engines, key=_dist2)[0]

    def _engine_for_operator(self, node: GenericNode) -> EngineServiceNode | None:
        if self._is_editor_node(node):
            return None
        try:
            svc = str(getattr(getattr(node, "spec", None), "serviceClass", "") or "").strip()
            if svc and svc != ENGINE_SERVICE_CLASS:
                return None
        except Exception:
            pass
        owned = self._operator_owner_engine(node)
        if owned is not None:
            return owned
        candidates: list[EngineServiceNode] = []
        for engine in self._engine_nodes():
            try:
                if any(n is node for n in engine.operator_nodes()):
                    candidates.append(engine)
            except Exception:
                continue
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        def _area(e: EngineServiceNode) -> float:
            try:
                w, h = e.size()
                return float(w) * float(h)
            except Exception:
                return float("inf")

        return sorted(candidates, key=_area)[0]

    def _assign_operator_to_engine(self, node: GenericNode) -> EngineServiceNode | None:
        if self._is_editor_node(node):
            return None
        try:
            svc = str(getattr(getattr(node, "spec", None), "serviceClass", "") or "").strip()
            if svc and svc != ENGINE_SERVICE_CLASS:
                return None
        except Exception:
            pass
        engines = self._engine_nodes()
        containing = [e for e in engines if self._is_inside_engine(e, node)]
        if containing:
            chosen = self._engine_for_operator(node) or containing[0]
            self._operator_owner[str(node.id)] = str(chosen.id)
            return chosen

        if not engines:
            engine = self.node_graph.create_node(
                ENGINE_SERVICE_CLASS,
                name="Engine",
                pos=list(map(float, node.pos())),
                push_undo=False,
                selected=False,
            )
            if isinstance(engine, EngineServiceNode):
                try:
                    engine.wrap_operator_nodes([node])
                except Exception:
                    pass
                self._operator_owner[str(node.id)] = str(engine.id)
                return engine
            return None

        engine = self._closest_engine(node) or engines[0]
        prev_geom = self._engine_geom.get(str(engine.id))
        if prev_geom is None:
            self._remember_engine_geom(engine)
            prev_geom = self._engine_geom.get(str(engine.id))

        try:
            engine.wrap_operator_nodes([*engine.operator_nodes(), node])
        except Exception:
            pass

        # If expanding causes overlap, revert and clamp node inside the chosen engine.
        if any(self._engines_overlap(engine, other) for other in engines if other is not engine):
            if prev_geom:
                self._restore_engine_geom(engine, prev_geom)
            rect_engine = self._engine_rect_proposed(engine=engine)
            rect_node = self._scene_rect(node)
            if rect_engine is not None and rect_node is not None:
                p = rect_engine.adjusted(6.0, 26.0, -6.0, -6.0)
                nx = min(max(rect_node.x(), p.left()), p.right() - rect_node.width())
                ny = min(max(rect_node.y(), p.top()), p.bottom() - rect_node.height())
                try:
                    node.set_property("pos", [float(nx), float(ny)], push_undo=False)
                except Exception:
                    pass

        self._operator_owner[str(node.id)] = str(engine.id)
        return engine

    def _ensure_engine_manager(self, engine: EngineServiceNode) -> None:
        sid = str(engine.id)
        if sid in self._engine_sync:
            return

        def _filter(n: GenericNode) -> bool:
            try:
                return self._engine_for_operator(n) is engine
            except Exception:
                return False

        def _edge_meta(src: GenericNode, dst: GenericNode, kind: str, local_side: str | None = None) -> dict[str, Any]:
            a_sid = self._service_id_for_node(src)
            b_sid = self._service_id_for_node(dst)
            meta: dict[str, Any] = {"fromServiceId": a_sid, "toServiceId": b_sid}
            cross = bool(a_sid and b_sid and a_sid != b_sid)
            if kind == "data" and cross:
                # If the receiver is an editor-ui node, default to queue so GUI polling doesn't see intermittent None.
                if self._is_editor_node(dst):
                    meta["strategy"] = F8EdgeStrategyEnum.queue
                    meta["timeoutMs"] = 2500
                    meta["queueSize"] = 64
                else:
                    meta["strategy"] = F8EdgeStrategyEnum.latest
            return meta

        mgr = EngineManager(self.node_graph, service_id=sid, node_filter=_filter, edge_meta=_edge_meta)
        mgr.start()
        self._engine_sync[sid] = mgr
        self._remember_engine_geom(engine)
        # Ensure ownership map covers all currently wrapped operator nodes.
        for op in engine.operator_nodes():
            try:
                if not isinstance(op, GenericNode) or self._is_editor_node(op):
                    continue
                self._operator_owner[str(op.id)] = sid
            except Exception:
                continue

        # Spawn the engine service process for this engine backdrop.
        nats_url = (os.environ.get("F8_NATS_URL") or "nats://127.0.0.1:4222").strip()
        try:
            launch = ServiceSpecRegistry.instance().get(ENGINE_SERVICE_CLASS).launch
        except Exception:
            launch = None
        self._svc_procs.start_service(
            ServiceProcessConfig(
                service_id=sid,
                service_class=ENGINE_SERVICE_CLASS,
                nats_url=nats_url,
                launch=launch,
                operator_runtime_modules=(os.environ.get("F8_OPERATOR_RUNTIME_MODULES") or "").strip() or None,
            )
        )

    def _on_node_created(self, node: object) -> None:
        if isinstance(node, EngineServiceNode):
            self._ensure_engine_manager(node)
            return
        if isinstance(node, GenericNode):
            if self._is_editor_node(node):
                self._ensure_editor_manager()
                self._start_editor_poller(node)
                return
            engine = self._assign_operator_to_engine(node)
            if engine is not None:
                self._ensure_engine_manager(engine)

    def _on_nodes_deleted(self, nodes: list[object]) -> None:
        for node in nodes:
            if not isinstance(node, EngineServiceNode):
                continue
            sid = str(node.id)
            mgr = self._engine_sync.pop(sid, None)
            if mgr is None:
                continue
            try:
                mgr.stop()
            except Exception:
                pass
            self._engine_geom.pop(sid, None)
            try:
                self._svc_procs.stop_service(sid)
            except Exception:
                pass

        for node in nodes:
            if isinstance(node, GenericNode):
                self._operator_owner.pop(str(node.id), None)
                self._stop_editor_poller(str(node.id))

    def _on_property_changed(self, node: object, name: str, _value: object) -> None:
        """
        Enforce "operators cannot leave engine backdrop" even for non-mouse moves
        (eg. keyboard nudges).
        """
        if self._enforcing:
            return
        if isinstance(node, GenericNode) and self._is_editor_node(node) and name == "refreshMs":
            self._update_editor_poller(node)
            return
        if name != "pos":
            return

        if isinstance(node, GenericNode):
            if self._is_editor_node(node):
                return
            engine = self._operator_owner_engine(node) or self._engine_for_operator(node)
            if engine is None:
                engine = self._assign_operator_to_engine(node)
            if engine is None:
                return
            if self._is_inside_engine(engine, node):
                return
            rect_engine = self._engine_rect_proposed(engine=engine)
            rect_node = self._scene_rect(node)
            if rect_engine is None or rect_node is None:
                return
            p = rect_engine.adjusted(6.0, 26.0, -6.0, -6.0)
            nx = min(max(rect_node.x(), p.left()), p.right() - rect_node.width())
            ny = min(max(rect_node.y(), p.top()), p.bottom() - rect_node.height())
            try:
                self._enforcing = True
                node.set_property("pos", [float(nx), float(ny)], push_undo=False)  # type: ignore[attr-defined]
            finally:
                self._enforcing = False
            return

        if isinstance(node, EngineServiceNode):
            # Prevent moving an engine backdrop into another engine.
            sid = str(node.id)
            prev = self._engine_geom.get(sid)
            if prev is None:
                self._remember_engine_geom(node)
                prev = self._engine_geom.get(sid)
            for other in self._engine_nodes():
                if other is node:
                    continue
                if self._engines_overlap(node, other):
                    if prev is not None:
                        self._restore_engine_geom(node, prev)
                    break
            else:
                self._remember_engine_geom(node)

    def _revert_node_pos(self, node: object, *, old_xy: Any) -> None:
        try:
            if not isinstance(old_xy, (list, tuple)) or len(old_xy) != 2:
                return
            pos = [float(old_xy[0]), float(old_xy[1])]
            node.set_property("pos", pos, push_undo=False)  # type: ignore[attr-defined]
        except Exception:
            return

    def _on_viewer_moved_nodes(self, moved: object) -> None:
        if not isinstance(moved, dict):
            return

        moved_engines: dict[str, tuple[EngineServiceNode, Any]] = {}
        moved_ops: dict[str, tuple[GenericNode, Any]] = {}
        for item, old_xy in moved.items():
            node_id = getattr(item, "id", None)
            if not node_id:
                continue
            try:
                ctrl = self.node_graph.get_node_by_id(str(node_id))
            except Exception:
                continue
            if isinstance(ctrl, EngineServiceNode):
                moved_engines[str(ctrl.id)] = (ctrl, old_xy)
            elif isinstance(ctrl, GenericNode):
                moved_ops[str(ctrl.id)] = (ctrl, old_xy)

        engines = self._engine_nodes()
        for sid, (engine, old_xy) in moved_engines.items():
            overlaps = any(self._engines_overlap(engine, other) for other in engines if other is not engine)
            if overlaps:
                self._revert_node_pos(engine, old_xy=old_xy)
                for op_id, (op, op_old) in moved_ops.items():
                    if self._operator_owner.get(op_id) == sid:
                        self._revert_node_pos(op, old_xy=op_old)
                continue
            # Prevent engine moves that would leave any owned operator outside.
            rect = self._engine_rect_proposed(engine=engine)
            if rect is not None:
                any_outside = False
                for op_id, owner_id in list(self._operator_owner.items()):
                    if owner_id != sid:
                        continue
                    try:
                        op = self.node_graph.get_node_by_id(str(op_id))
                    except Exception:
                        continue
                    if not isinstance(op, GenericNode) or self._is_editor_node(op):
                        continue
                    if not self._is_inside_engine(engine, op, engine_rect=rect):
                        any_outside = True
                        break
                if any_outside:
                    self._revert_node_pos(engine, old_xy=old_xy)
                    for op_id, (op, op_old) in moved_ops.items():
                        if self._operator_owner.get(op_id) == sid:
                            self._revert_node_pos(op, old_xy=op_old)
                    continue
            self._remember_engine_geom(engine)

        for op_id, (op, old_xy) in moved_ops.items():
            engine = self._operator_owner_engine(op) or self._engine_for_operator(op)
            if engine is None:
                engine = self._assign_operator_to_engine(op)
            if engine is None:
                continue
            self._operator_owner[op_id] = str(engine.id)
            if not self._is_inside_engine(engine, op):
                self._revert_node_pos(op, old_xy=old_xy)

        for mgr in self._engine_sync.values():
            try:
                mgr.schedule_rungraph_sync()
            except Exception:
                pass

    def _on_viewer_backdrop_updated(self, node_id: str, _update_prop: str, value: object) -> None:
        try:
            engine = self.node_graph.get_node_by_id(str(node_id))
        except Exception:
            return
        if not isinstance(engine, EngineServiceNode):
            return

        sid = str(engine.id)
        prev = self._engine_geom.get(sid) or {}
        proposed: dict[str, Any] | None = None
        if isinstance(value, dict) and all(k in value for k in ("pos", "width", "height")):
            proposed = {"pos": value.get("pos"), "width": value.get("width"), "height": value.get("height")}

        rect_new = self._engine_rect_proposed(engine=engine, geom=proposed)
        if rect_new is None:
            return

        for other in self._engine_nodes():
            if other is engine:
                continue
            if self._engines_overlap(engine, other, a_rect=rect_new):
                self._restore_engine_geom(engine, prev)
                return

        # Ensure all operators *owned* by this engine remain inside the resized rect.
        for op_id, owner_id in list(self._operator_owner.items()):
            if owner_id != sid:
                continue
            try:
                op = self.node_graph.get_node_by_id(str(op_id))
            except Exception:
                continue
            if not isinstance(op, GenericNode) or self._is_editor_node(op):
                continue
            if not self._is_inside_engine(engine, op, engine_rect=rect_new):
                self._restore_engine_geom(engine, prev)
                return

        self._remember_engine_geom(engine)
        mgr = self._engine_sync.get(sid)
        if mgr is not None:
            try:
                mgr.schedule_rungraph_sync()
            except Exception:
                pass

    def _restore_engine_geom(self, engine: EngineServiceNode, geom: dict[str, Any]) -> None:
        try:
            pos = geom.get("pos")
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                engine.set_property("pos", [float(pos[0]), float(pos[1])], push_undo=False)
        except Exception:
            pass
        try:
            if geom.get("width") is not None:
                engine.set_property("width", float(geom["width"]), push_undo=False)
            if geom.get("height") is not None:
                engine.set_property("height", float(geom["height"]), push_undo=False)
        except Exception:
            pass

    def to_operator_graph(self) -> Any:
        """
        Export a full operator graph snapshot for the service graph.

        Edges are annotated with:
        - cross-service edges if `fromServiceId != toServiceId`
        - data edges default to `strategy=latest` (or `queue` for editor pull nodes)
        """

        def _edge_meta(src: GenericNode, dst: GenericNode, kind: str, _local_side: str | None = None) -> dict[str, Any]:
            a_sid = self._service_id_for_node(src)
            b_sid = self._service_id_for_node(dst)
            meta: dict[str, Any] = {"fromServiceId": a_sid, "toServiceId": b_sid}
            cross = bool(a_sid and b_sid and a_sid != b_sid)
            if kind == "data" and cross:
                if self._is_editor_node(dst):
                    meta["strategy"] = F8EdgeStrategyEnum.queue
                else:
                    meta["strategy"] = F8EdgeStrategyEnum.latest
            return meta

        return export_operator_graph(self.node_graph, service_id=EDITOR_SERVICE_ID, edge_meta=_edge_meta)

    def _ensure_editor_manager(self) -> None:
        if self._editor_mgr is not None:
            return

        def _filter(n: GenericNode) -> bool:
            return self._is_editor_node(n)

        def _edge_meta(src: GenericNode, dst: GenericNode, kind: str, local_side: str | None = None) -> dict[str, Any]:
            a_sid = self._service_id_for_node(src)
            b_sid = self._service_id_for_node(dst)
            meta: dict[str, Any] = {"fromServiceId": a_sid, "toServiceId": b_sid}
            cross = bool(a_sid and b_sid and a_sid != b_sid)
            if kind == "data" and cross and self._is_editor_node(dst):
                meta["strategy"] = F8EdgeStrategyEnum.queue
                meta["timeoutMs"] = 2500
                meta["queueSize"] = 64
            return meta

        self._editor_mgr = EngineManager(self.node_graph, service_id=EDITOR_SERVICE_ID, node_filter=_filter, edge_meta=_edge_meta)
        self._editor_mgr.start()

        bridge = self._editor_mgr.runtime_bridge()
        if bridge is not None:
            try:
                bridge.dataReceived.connect(self._on_editor_data_received)
            except Exception:
                pass

    def _start_editor_poller(self, node: GenericNode) -> None:
        node_id = str(node.id)
        if node_id in self._editor_pollers:
            return
        self._ensure_editor_manager()

        timer = QtCore.QTimer(self.node_graph)
        timer.setSingleShot(False)

        def _tick() -> None:
            self._flush_editor_node(node_id)

        timer.timeout.connect(_tick)
        self._editor_pollers[node_id] = timer
        self._update_editor_poller(node)
        timer.start()

    def _stop_editor_poller(self, node_id: str) -> None:
        t = self._editor_pollers.pop(str(node_id), None)
        if t is not None:
            try:
                t.stop()
            except Exception:
                pass
        self._editor_pending.pop(str(node_id), None)
        self._editor_last.pop(str(node_id), None)
        self._editor_buffers.pop(str(node_id), None)

    def _update_editor_poller(self, node: GenericNode) -> None:
        node_id = str(node.id)
        t = self._editor_pollers.get(node_id)
        if t is None:
            return
        try:
            ms = int(node.get_property("refreshMs") or 200)
        except Exception:
            ms = 200
        ms = max(16, min(5000, ms))
        try:
            t.setInterval(ms)
        except Exception:
            pass

    def _on_editor_data_received(self, node_id: str, port: str, value: object, _ts_ms: object) -> None:
        if port != "in":
            return
        try:
            node = self.node_graph.get_node_by_id(str(node_id))
        except Exception:
            return
        if not isinstance(node, GenericNode) or not self._is_editor_node(node):
            return
        if value is None:
            return

        op = ""
        try:
            op = str(getattr(node.spec, "operatorClass", "") or "")
        except Exception:
            op = ""

        key = str(node_id)
        if op.endswith(".oscilloscope"):
            try:
                buf = self._editor_buffers.setdefault(key, [])
                buf.append(float(value))  # type: ignore[arg-type]
            except Exception:
                return
            return

        self._editor_pending[key] = value

    def _flush_editor_node(self, node_id: str) -> None:
        try:
            node = self.node_graph.get_node_by_id(str(node_id))
        except Exception:
            return
        if not isinstance(node, GenericNode) or not self._is_editor_node(node):
            return

        op = ""
        try:
            op = str(getattr(node.spec, "operatorClass", "") or "")
        except Exception:
            op = ""

        key = str(node_id)
        prev = self._editor_last.get(key)

        if op.endswith(".oscilloscope"):
            buf = self._editor_buffers.get(key) or []
            if not buf:
                return
            try:
                n = int(node.get_property("window") or 240)
            except Exception:
                n = 240
            if n <= 0:
                n = 240
            if len(buf) > n:
                del buf[0 : len(buf) - n]
            cur = f"n={len(buf)} last={buf[-1]:.6g} min={min(buf):.6g} max={max(buf):.6g}"
        else:
            if key not in self._editor_pending:
                return
            cur = str(self._editor_pending.get(key))

        if prev == cur:
            return
        self._editor_last[key] = cur

        if op.endswith(".log"):
            try:
                if bool(node.get_property("print")):
                    print(f"[editor.log][{node_id}] {cur}")
            except Exception:
                pass
            try:
                if hasattr(node, "append_log"):
                    node.append_log(cur)  # type: ignore[attr-defined]
            except Exception:
                pass
            return

        # Show the latest value as an inline widget on the node (local-only, avoids KV spam).
        try:
            from ..renderers.generic import WIDGET_ROW_DATA_KEY, PORT_ROW_DATA_KEY  # type: ignore
            from NodeGraphQt.widgets.node_widgets import NodeBaseWidget
            from qtpy import QtWidgets

            widget_name = "__editor_value__"
            w = getattr(node, "_editor_value_widget", None)
            if w is None:
                row = None
                try:
                    handle = node.port_handles.data_in.get("in")
                    if handle is not None:
                        row = handle.view.data(PORT_ROW_DATA_KEY)
                except Exception:
                    row = None

                class _ValueWidget(NodeBaseWidget):
                    def __init__(self, parent: Any, name: str) -> None:
                        super().__init__(parent, name, "")
                        lab = QtWidgets.QLabel("")
                        lab.setMinimumWidth(180)
                        lab.setStyleSheet("color: rgba(230,230,230,200); font-size: 8pt;")
                        self.set_custom_widget(lab)

                    def set_text(self, text: str) -> None:
                        try:
                            self.widget().setText(text)  # type: ignore[attr-defined]
                        except Exception:
                            pass

                w = _ValueWidget(node.view, widget_name)
                try:
                    if row is not None:
                        w.setData(WIDGET_ROW_DATA_KEY, int(row))
                except Exception:
                    pass
                try:
                    node.view.add_widget(w)
                except Exception:
                    pass
                setattr(node, "_editor_value_widget", w)

            try:
                w.set_text(cur)
            except Exception:
                pass
            try:
                node.view.draw_node()
            except Exception:
                pass
        except Exception:
            pass

    def show(self) -> None:
        if hasattr(self.node_graph, "widget"):
            self.node_graph.widget.show()
        elif hasattr(self.node_graph, "show"):
            self.node_graph.show()

    def widget(self) -> Any:
        return getattr(self.node_graph, "widget", self.node_graph)
