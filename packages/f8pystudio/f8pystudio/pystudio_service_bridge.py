from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import nats  # type: ignore[import-not-found]
from qtpy import QtCore

from f8pysdk import F8Edge, F8EdgeKindEnum, F8EdgeStrategyEnum, F8RuntimeGraph, F8RuntimeNode, F8StateAccess
from f8pysdk.nats_naming import cmd_channel_subject, ensure_token, kv_bucket_for_service, new_id, svc_endpoint_subject, svc_micro_name
from f8pysdk.nats_transport import NatsTransport, NatsTransportConfig
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.service_ready import wait_service_ready
from .nodegraph.runtime_compiler import CompiledRuntimeGraphs
from .pystudio_service import PyStudioService, PyStudioServiceConfig
from .service_process_manager import ServiceProcessConfig, ServiceProcessManager
from .pystudio_node_registry import SERVICE_CLASS, STUDIO_SERVICE_ID


_MONITOR_NODE_ID = "monitor"


def _encode_remote_state_key(*, service_id: str, node_id: str, field: str) -> str:
    # Keep this stable and human-readable; avoid "." (it can be confusing inside KV keys).
    return f"{service_id}|{node_id}|{field}"


class _AsyncThread:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._run, name="pystudio-async", daemon=True)
        self._ready = threading.Event()
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(timeout=5)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_until_complete(self._main())
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    async def _main(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(0.05)

    def submit(self, coro: Any) -> "asyncio.Future[Any]":
        if self._loop is None:
            raise RuntimeError("async thread not started")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore[arg-type]

    def stop(self) -> None:
        self._stop.set()
        # Do not call loop.stop() directly: it can abort `run_until_complete(...)`
        # and raise "Event loop stopped before Future completed."
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(lambda: None)
            except Exception:
                pass
        try:
            self._thread.join(timeout=2)
        except Exception:
            pass


@dataclass(frozen=True)
class PyStudioServiceBridgeConfig:
    nats_url: str = "nats://127.0.0.1:4222"
    studio_service_id: str = STUDIO_SERVICE_ID


class PyStudioServiceBridge(QtCore.QObject):
    """
    Orchestrate:
    - singleton studio presence (NATS micro ping/info)
    - start service processes
    - deploy per-service rungraphs
    - monitor remote state via cross-state edges into a local monitor node
    """

    # Note: Qt `int` is typically 32-bit; use `object` for ts_ms (ms timestamps exceed 2^31).
    ui_command = QtCore.Signal(object)  # UiCommand
    service_output = QtCore.Signal(str, str)  # serviceId, line
    log = QtCore.Signal(str)
    service_process_state = QtCore.Signal(str, bool)  # serviceId, running

    def __init__(self, config: PyStudioServiceBridgeConfig, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._cfg = config
        self._async = _AsyncThread()
        self._proc_mgr = ServiceProcessManager()
        self._managed_service_ids: set[str] = set()
        self._managed_service_classes: dict[str, str] = {}  # serviceId -> serviceClass
        self._managed_active: bool = True
        self._pending_proc_actions: dict[str, dict[str, Any]] = {}  # serviceId -> {action, deadline, serviceClass?}
        self._pending_proc_timers: dict[str, QtCore.QTimer] = {}  # serviceId -> timer
        self._service_status_cache: dict[str, tuple[bool | None, float]] = {}  # serviceId -> (active, monotonic_ts)
        self._service_alive_cache: dict[str, tuple[bool, float]] = {}  # serviceId -> (alive, monotonic_ts)
        self._service_status_inflight: set[str] = set()
        self._service_status_req_s: dict[str, float] = {}
        self._last_compiled: CompiledRuntimeGraphs | None = None

        self._svc: PyStudioService | None = None
        self._nc: Any = None
        self._last_nats_error_log_s: float = 0.0

    @property
    def studio_service_id(self) -> str:
        return ensure_token(self._cfg.studio_service_id, label="studio_service_id")

    @property
    def managed_active(self) -> bool:
        return bool(self._managed_active)

    @QtCore.Slot(bool)
    def set_managed_active(self, active: bool) -> None:
        """
        Activate/deactivate all managed service instances (via command channel).

        This is the lifecycle control described in `docs/design/pysdk-runtime.md`.
        """
        try:
            self._managed_active = bool(active)
        except Exception:
            self._managed_active = True
        try:
            self._async.submit(self._set_managed_active_async(bool(self._managed_active)))
        except Exception:
            return

    def start(self) -> None:
        self._async.start()
        self._async.submit(self._start_async())

    def stop(self) -> None:
        try:
            fut = self._async.submit(self._stop_async())
            try:
                fut.result(timeout=2)
            except Exception:
                pass
        except Exception:
            pass
        self._async.stop()

        # Best-effort stop all launched processes.
        try:
            for sid in list(getattr(self._proc_mgr, "_procs", {}).keys()):
                self._proc_mgr.stop(sid)
        except Exception:
            pass

    def deploy_run_and_monitor(self, compiled: CompiledRuntimeGraphs) -> None:
        """
        Starts service processes (if not running), deploys per-service graphs,
        and installs a monitoring graph into the studio bus.
        """
        # 1) start processes (sync)
        self._last_compiled = compiled
        managed: set[str] = set()
        managed_classes: dict[str, str] = {}
        for svc in list(compiled.global_graph.services or []):
            try:
                sid = ensure_token(str(svc.serviceId), label="service_id")
                if sid == self.studio_service_id or str(svc.serviceClass) == SERVICE_CLASS:
                    continue
                managed.add(sid)
                managed_classes[sid] = str(svc.serviceClass)
                # Use the public helper so we dedup against already-running services
                # (including ones started outside this Studio process).
                self.start_service(sid, service_class=str(svc.serviceClass))
            except Exception as exc:
                self.log.emit(f"start service failed: {exc}")
        self._managed_service_ids = managed
        self._managed_service_classes = managed_classes

        # 2) deploy + install monitoring (async)
        self._async.submit(self._deploy_and_monitor_async(compiled))
        # Deploy implies global active by default.
        self.set_managed_active(True)

    @QtCore.Slot(str)
    def unmanage_service(self, service_id: str) -> None:
        """
        Remove a serviceId from the studio's managed set (UI bookkeeping only).

        This does not stop the process by itself.
        """
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return
        if sid == self.studio_service_id:
            return

        try:
            self._managed_service_ids.discard(sid)
        except Exception:
            pass
        try:
            self._managed_service_classes.pop(sid, None)
        except Exception:
            pass
        try:
            self._service_status_cache.pop(sid, None)
        except Exception:
            pass

        # Cancel any pending process actions/timers for this service.
        try:
            self._pending_proc_actions.pop(sid, None)
        except Exception:
            pass
        try:
            t = self._pending_proc_timers.pop(sid, None)
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
        except Exception:
            pass

    @QtCore.Slot(str)
    def reclaim_service(self, service_id: str) -> None:
        """
        Best-effort reclamation for a serviceId that was removed from the canvas:
        terminate the process and drop it from the managed set.
        """
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return
        if sid == self.studio_service_id:
            return
        try:
            self.log.emit(f"reclaim service process serviceId={sid}")
        except Exception:
            pass
        try:
            self.stop_service(sid)
        except Exception:
            pass
        self.unmanage_service(sid)

    @QtCore.Slot(str)
    def deploy_service_rungraph(self, service_id: str, *, compiled: CompiledRuntimeGraphs | None = None) -> None:
        """
        Deploy the current per-service rungraph to a running service instance (best-effort).
        """
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return
        if sid == self.studio_service_id:
            return
        if not self.is_service_running(sid):
            try:
                self.log.emit(f"deploy skipped (service not running) serviceId={sid}")
            except Exception:
                pass
            return

        async def _do() -> None:
            await self._install_monitor_graph_async(compiled=compiled)
            await self._deploy_service_rungraph_async(sid, compiled=compiled)

        try:
            self._async.submit(_do())
        except Exception:
            pass

    async def _ensure_studio_runtime_async(self, *, timeout_s: float = 6.0) -> bool:
        """
        Best-effort wait for the in-process studio runtime (ServiceRuntime) to be ready.
        """
        deadline = time.monotonic() + float(timeout_s or 0.0)
        while True:
            if self._svc is not None and getattr(self._svc, "bus", None) is not None:
                return True
            if time.monotonic() >= deadline:
                try:
                    self.log.emit("studio runtime not ready (timeout)")
                except Exception:
                    pass
                return False
            await asyncio.sleep(0.08)

    @staticmethod
    def _pick_compiled(
        compiled: CompiledRuntimeGraphs | None, fallback: CompiledRuntimeGraphs | None
    ) -> CompiledRuntimeGraphs | None:
        return compiled if compiled is not None else fallback

    def _build_studio_monitor_graph(self, compiled: CompiledRuntimeGraphs) -> F8RuntimeGraph:
        mon_edges: list[F8Edge] = []
        for n in list(compiled.global_graph.nodes or []):
            try:
                from_sid = str(getattr(n, "serviceId", "") or "")
                if not from_sid:
                    continue
                from_sid = ensure_token(from_sid, label="fromServiceId")
                from_nid = ensure_token(str(getattr(n, "nodeId", "") or ""), label="nodeId")
            except Exception:
                continue

            fields: list[str] = []
            for sf in list(getattr(n, "stateFields", None) or []):
                try:
                    field_name = str(getattr(sf, "name", "") or "").strip()
                except Exception:
                    field_name = ""
                if not field_name:
                    continue
                try:
                    access = getattr(sf, "access", None)
                except Exception:
                    access = None
                # Only monitor readable state (rw/ro). Write-only has no fanout by design.
                if access not in (F8StateAccess.rw, F8StateAccess.ro):
                    continue
                fields.append(field_name)

            # Always monitor identity fields even if a node spec/stateFields list is stale.
            if "svcId" not in fields:
                fields.append("svcId")
            try:
                op_class = str(getattr(n, "operatorClass", "") or "").strip()
            except Exception:
                op_class = ""
            if op_class and "operatorId" not in fields:
                fields.append("operatorId")

            for field_name in fields:
                mon_edges.append(
                    F8Edge(
                        edgeId=f"mon_{from_sid}_{from_nid}_{field_name}".replace(".", "_"),
                        fromServiceId=from_sid,
                        fromOperatorId=from_nid,
                        fromPort=field_name,
                        toServiceId=self.studio_service_id,
                        toOperatorId=_MONITOR_NODE_ID,
                        toPort=_encode_remote_state_key(service_id=from_sid, node_id=from_nid, field=field_name),
                        kind=F8EdgeKindEnum.state,
                        strategy=F8EdgeStrategyEnum.latest,
                    )
                )

        studio_sub = compiled.per_service.get(self.studio_service_id)
        base_nodes: list[F8RuntimeNode] = list(getattr(studio_sub, "nodes", None) or []) if studio_sub is not None else []
        base_edges: list[F8Edge] = list(getattr(studio_sub, "edges", None) or []) if studio_sub is not None else []

        return F8RuntimeGraph(
            graphId=str(getattr(compiled.global_graph, "graphId", "") or "studio_monitor"),
            revision=str(getattr(compiled.global_graph, "revision", "") or "1"),
            meta={"source": "studio"},
            services=[],
            nodes=[
                *base_nodes,
                F8RuntimeNode(
                    nodeId=_MONITOR_NODE_ID,
                    serviceId=self.studio_service_id,
                    serviceClass=SERVICE_CLASS,
                    operatorClass="f8.monitor_state",
                ),
            ],
            edges=[*base_edges, *mon_edges],
        )

    def is_service_running(self, service_id: str) -> bool:
        sid = str(service_id or "").strip()
        if not sid:
            return False
        try:
            if bool(self._proc_mgr.is_running(str(sid))):
                return True
        except Exception:
            pass
        # If the service wasn't launched by this studio process, fall back to
        # a best-effort "alive" cache (refreshed via status endpoint).
        v = self._service_alive_cache.get(sid)
        if not v:
            return False
        alive, ts = v
        if not alive:
            return False
        # Consider alive cache fresh for a short window to avoid UI flicker.
        return (time.monotonic() - float(ts)) < 0.9

    def get_service_class(self, service_id: str) -> str:
        """
        Best-effort service identity lookup for UI display (eg log tabs).
        """
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return ""
        try:
            return str(self._managed_service_classes.get(sid, "") or "")
        except Exception:
            return ""

    def _cache_service_active(self, service_id: str, active: bool | None) -> None:
        sid = str(service_id or "").strip()
        if not sid:
            return
        self._service_status_cache[sid] = (active, time.monotonic())

    def _cache_service_alive(self, service_id: str, alive: bool) -> None:
        sid = str(service_id or "").strip()
        if not sid:
            return
        self._service_alive_cache[sid] = (bool(alive), time.monotonic())

    def get_cached_service_active(self, service_id: str) -> bool | None:
        """
        Return last known remote service active state (best-effort).
        """
        sid = str(service_id or "").strip()
        if not sid:
            return None
        v = self._service_status_cache.get(sid)
        if not v:
            return None
        return v[0]

    async def _request_service_status_async(self, service_id: str) -> dict[str, Any] | None:
        sid = ""
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return None
        nc = await self._ensure_nc()
        if nc is None:
            return None

        payload = json.dumps({"reqId": new_id(), "args": {}, "meta": {"actor": "studio", "cmd": "status"}}, ensure_ascii=False).encode(
            "utf-8"
        )
        try:
            msg = await nc.request(svc_endpoint_subject(sid, "status"), payload, timeout=0.4)
        except Exception:
            return None
        raw = bytes(getattr(msg, "data", b"") or b"")
        if not raw:
            return None
        try:
            resp = json.loads(raw.decode("utf-8"))
        except Exception:
            resp = {}
        if not (isinstance(resp, dict) and resp.get("ok") is True):
            return None
        result = resp.get("result") if isinstance(resp.get("result"), dict) else {}
        if not isinstance(result, dict):
            return None
        out: dict[str, Any] = {"alive": True}
        if "active" in result:
            try:
                out["active"] = bool(result.get("active"))
            except Exception:
                out["active"] = None
        return out

    def request_service_status(self, service_id: str) -> None:
        """
        Trigger a best-effort status refresh (async).
        """
        sid = ""
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return
        try:
            now = time.monotonic()
            last = float(self._service_status_req_s.get(sid, 0.0))
            if (now - last) < 0.25:
                return
            if sid in self._service_status_inflight:
                return
            self._service_status_inflight.add(sid)
            self._service_status_req_s[sid] = now
        except Exception:
            pass

        async def _do() -> None:
            try:
                status = await self._request_service_status_async(sid)
                if not isinstance(status, dict):
                    # Fast "down" signal: if status endpoint doesn't respond, treat service
                    # as not running for UI purposes (it will flip back to True on next success).
                    self._cache_service_alive(sid, False)
                    self._cache_service_active(sid, None)
                    return
                self._cache_service_alive(sid, True)
                if "active" in status:
                    self._cache_service_active(sid, status.get("active"))
            finally:
                try:
                    self._service_status_inflight.discard(sid)
                except Exception:
                    pass

        try:
            self._async.submit(_do())
        except Exception:
            try:
                self._service_status_inflight.discard(sid)
            except Exception:
                pass
            return

    async def _set_service_active_async(self, service_id: str, active: bool) -> bool:
        sid = ""
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return False
        nc = await self._ensure_nc()
        if nc is None:
            return False

        cmd = "activate" if bool(active) else "deactivate"
        payload = json.dumps(
            {"reqId": new_id(), "args": {"active": bool(active)}, "meta": {"actor": "studio", "cmd": cmd}},
            ensure_ascii=False,
        ).encode("utf-8")
        for _ in range(2):
            try:
                msg = await nc.request(svc_endpoint_subject(sid, cmd), payload, timeout=0.5)
                data = bytes(getattr(msg, "data", b"") or b"")
                if data:
                    try:
                        resp = json.loads(data.decode("utf-8"))
                    except Exception:
                        resp = {}
                    if isinstance(resp, dict) and resp.get("ok") is True:
                        self._cache_service_active(sid, bool(active))
                        return True
            except Exception:
                await asyncio.sleep(0.15)
                continue
        return False

    @QtCore.Slot(str, bool)
    def set_service_active(self, service_id: str, active: bool) -> None:
        sid = ""
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return
        if sid == self.studio_service_id:
            return

        try:
            self._async.submit(self._set_service_active_async(sid, bool(active)))
        except Exception:
            return

    async def _request_service_terminate_async(self, service_id: str) -> bool:
        """
        Ask a running service process to exit itself (graceful).

        This is best-effort and may fail if the service isn't connected to NATS yet.
        """
        sid = ""
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return False

        nc = await self._ensure_nc()
        if nc is None:
            return False

        subject = svc_endpoint_subject(sid, "terminate")
        payload = json.dumps({"reqId": new_id(), "args": {}, "meta": {"actor": "studio", "cmd": "terminate"}}, ensure_ascii=False).encode(
            "utf-8"
        )
        for _ in range(2):
            try:
                msg = await nc.request(subject, payload, timeout=0.4)
                raw = bytes(getattr(msg, "data", b"") or b"")
                if not raw:
                    continue
                try:
                    resp = json.loads(raw.decode("utf-8"))
                except Exception:
                    resp = {}
                if isinstance(resp, dict) and resp.get("ok") is True:
                    return True
                return False
            except Exception:
                await asyncio.sleep(0.15)
                continue
        return False

    def _ensure_proc_action_timer(self, service_id: str) -> QtCore.QTimer:
        sid = str(service_id)
        existing = self._pending_proc_timers.get(sid)
        if existing is not None:
            return existing
        t = QtCore.QTimer(self)
        t.setInterval(120)
        t.timeout.connect(lambda _sid=sid: self._poll_proc_action(_sid))
        self._pending_proc_timers[sid] = t
        return t

    def _clear_proc_action(self, service_id: str) -> None:
        sid = str(service_id)
        t = self._pending_proc_timers.pop(sid, None)
        if t is not None:
            try:
                t.stop()
            except Exception:
                pass
            try:
                t.deleteLater()
            except Exception:
                pass
        self._pending_proc_actions.pop(sid, None)

    def _poll_proc_action(self, service_id: str) -> None:
        sid = str(service_id)
        action = self._pending_proc_actions.get(sid)
        if not action:
            self._clear_proc_action(sid)
            return

        try:
            if not self.is_service_running(sid):
                self._clear_proc_action(sid)
                try:
                    self.service_process_state.emit(sid, False)
                except Exception:
                    pass

                if action.get("action") == "restart":
                    svc_class = str(action.get("serviceClass") or "").strip() or None
                    self.start_service(sid, service_class=svc_class)
                return
        except Exception:
            pass

        deadline = float(action.get("deadline") or 0.0)
        if deadline and time.monotonic() < deadline:
            return

        # Grace period expired: fall back to local hard stop (taskkill / kill-tree).
        try:
            ok = bool(self._proc_mgr.stop(sid))
        except Exception as exc:
            self.log.emit(f"stop_service failed: {exc}")
            ok = False

        still_running = bool(self.is_service_running(sid))
        if not ok and still_running:
            self.log.emit(f"stop_service incomplete (process still running): serviceId={sid}")
            # Keep timer running; do not clear, to avoid allowing duplicates.
            self._pending_proc_actions[sid]["deadline"] = time.monotonic() + 1.0
            return

        self._clear_proc_action(sid)
        try:
            self.service_process_state.emit(sid, still_running)
        except Exception:
            pass

        if not still_running and action.get("action") == "restart":
            svc_class = str(action.get("serviceClass") or "").strip() or None
            self.start_service(sid, service_class=svc_class)

    @QtCore.Slot(str)
    def start_service(self, service_id: str, *, service_class: str | None = None) -> None:
        sid = ""
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return
        if sid == self.studio_service_id:
            return

        # Dedup: if Studio already believes the service is alive (via local proc tracking or a fresh
        # status ping), do not spawn another process on repeated deploy (e.g. repeated F5).
        try:
            if self.is_service_running(sid):
                self.log.emit(f"start_service ignored (already running): serviceId={sid}")
                return
        except Exception:
            pass

        svc_class = self._managed_service_classes.get(sid, "") or str(service_class or "")
        if not svc_class:
            self.log.emit(f"start_service ignored (unknown serviceClass): serviceId={sid}")
            return
        try:
            self._proc_mgr.start(
                ServiceProcessConfig(service_class=str(svc_class), service_id=sid, nats_url=self._cfg.nats_url),
                on_output=lambda _sid, line, _sid2=sid: self.service_output.emit(_sid2, str(line)),
            )
        except Exception as exc:
            self.log.emit(f"start_service failed: {exc}")
            return
        try:
            self._managed_service_ids.add(sid)
            if svc_class:
                self._managed_service_classes[sid] = str(svc_class)
        except Exception:
            pass
        try:
            self.service_process_state.emit(sid, bool(self.is_service_running(sid)))
        except Exception:
            pass

    @QtCore.Slot(str)
    def stop_service(self, service_id: str) -> None:
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return
        if sid == self.studio_service_id:
            return

        if not self.is_service_running(sid):
            try:
                self.service_process_state.emit(sid, False)
            except Exception:
                pass
            return

        # 1) Ask the service to terminate itself (best for GUI apps / child process trees).
        try:
            self._async.submit(self._request_service_terminate_async(sid))
        except Exception:
            pass

        # 2) Poll for graceful exit, then fall back to local kill-tree.
        self._pending_proc_actions[sid] = {"action": "stop", "deadline": time.monotonic() + 2.2}
        t = self._ensure_proc_action_timer(sid)
        try:
            if not t.isActive():
                t.start()
        except Exception:
            pass
        self._poll_proc_action(sid)

    @QtCore.Slot(str)
    def restart_service(self, service_id: str, *, service_class: str | None = None) -> None:
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return
        if sid == self.studio_service_id:
            return

        svc_class = self._managed_service_classes.get(sid, "") or str(service_class or "")

        if not self.is_service_running(sid):
            self.start_service(sid, service_class=svc_class or None)
            return

        try:
            self._async.submit(self._request_service_terminate_async(sid))
        except Exception:
            pass

        self._pending_proc_actions[sid] = {"action": "restart", "deadline": time.monotonic() + 2.2, "serviceClass": svc_class}
        t = self._ensure_proc_action_timer(sid)
        try:
            if not t.isActive():
                t.start()
        except Exception:
            pass
        self._poll_proc_action(sid)

    async def _deploy_service_rungraph_async(self, service_id: str, *, compiled: CompiledRuntimeGraphs | None = None) -> None:
        """
        Deploy the last compiled per-service rungraph for a single service (best-effort).
        """
        compiled = self._pick_compiled(compiled, self._last_compiled)
        if compiled is None:
            return
        sid = ""
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return
        g = compiled.per_service.get(sid)
        if g is None:
            return

        nats_url = str(self._cfg.nats_url).strip() or "nats://127.0.0.1:4222"
        bucket = kv_bucket_for_service(sid)
        tr = NatsTransport(NatsTransportConfig(url=nats_url, kv_bucket=bucket))
        try:
            await tr.connect()

            # Wait for `ready=true` with periodic warnings (services may take time to boot).
            deadline = time.monotonic() + 30.0
            last_log_s: float = 0.0
            while True:
                try:
                    await wait_service_ready(tr, timeout_s=3.0)
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        try:
                            self.log.emit(f"service not ready (timeout) serviceId={sid}")
                        except Exception:
                            pass
                        return
                    now = time.monotonic()
                    if (now - last_log_s) >= 3.0:
                        last_log_s = now
                        try:
                            remain = max(0.0, deadline - now)
                            self.log.emit(f"waiting service ready... serviceId={sid} (timeout in {remain:.0f}s)")
                        except Exception:
                            pass
                    await asyncio.sleep(0.4)
            payload = g.model_dump(mode="json", by_alias=True)
            meta = dict(payload.get("meta") or {})
            if not str(meta.get("source") or "").strip():
                meta["source"] = "studio"
            payload["meta"] = meta
            req = {"reqId": new_id(), "args": {"graph": payload}, "meta": {"source": "studio"}}
            req_bytes = json.dumps(req, ensure_ascii=False, default=str).encode("utf-8")
            _ = await tr.request(
                svc_endpoint_subject(sid, "set_rungraph"),
                req_bytes,
                timeout=2.0,
                raise_on_error=True,
            )
        except Exception as exc:
            try:
                self.log.emit(f"deploy service rungraph failed serviceId={sid}: {exc}")
            except Exception:
                pass
        finally:
            try:
                await tr.close()
            except Exception:
                pass

    async def _install_monitor_graph_async(self, *, compiled: CompiledRuntimeGraphs | None = None) -> None:
        """
        Reinstall the studio monitor graph from the last compiled graphs (best-effort).
        """
        compiled = self._pick_compiled(compiled, self._last_compiled)
        if compiled is None:
            return
        if not await self._ensure_studio_runtime_async():
            return
        try:
            if self._svc is None or self._svc.bus is None:
                return
            studio_graph = self._build_studio_monitor_graph(compiled)
            await self._svc.bus.set_rungraph(studio_graph)
        except Exception as exc:
            try:
                self.log.emit(f"install monitor graph failed: {exc}")
            except Exception:
                pass

    @QtCore.Slot(str)
    def start_service_and_deploy(
        self, service_id: str, *, service_class: str | None = None, compiled: CompiledRuntimeGraphs | None = None
    ) -> None:
        """
        Start service process (if needed) and deploy last compiled rungraph for it (best-effort).
        """
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return
        if sid == self.studio_service_id:
            return
        self.start_service(sid, service_class=service_class)

        async def _do() -> None:
            await self._install_monitor_graph_async(compiled=compiled)
            await self._deploy_service_rungraph_async(sid, compiled=compiled)
            await self._set_service_active_async(sid, True)

        try:
            self._async.submit(_do())
        except Exception:
            pass

    @QtCore.Slot(str)
    def restart_service_and_deploy(
        self, service_id: str, *, service_class: str | None = None, compiled: CompiledRuntimeGraphs | None = None
    ) -> None:
        """
        Restart service (terminate -> start) and deploy last compiled rungraph for it (best-effort).
        """
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except Exception:
            return
        if sid == self.studio_service_id:
            return

        # Reuse existing restart flow; deploy will happen once process is back.
        self.restart_service(sid, service_class=service_class)

        async def _do() -> None:
            # Give restart a moment to come back; readiness wait inside deploy handles most cases.
            await asyncio.sleep(0.3)
            await self._install_monitor_graph_async(compiled=compiled)
            await self._deploy_service_rungraph_async(sid, compiled=compiled)
            await self._set_service_active_async(sid, True)

        try:
            self._async.submit(_do())
        except Exception:
            pass

    def set_local_state(self, node_id: str, field: str, value: Any) -> None:
        """
        Set state in the local studio service KV (best-effort).
        """
        node_id = ensure_token(node_id, label="node_id")
        field = str(field or "").strip()
        if not field:
            return

        async def _do() -> None:
            if self._svc is None or self._svc.bus is None:
                return
            try:
                await self._svc.bus.set_state(node_id, field, value)
            except Exception:
                return

        try:
            self._async.submit(_do())
        except Exception:
            return

    def set_remote_state(self, service_id: str, node_id: str, field: str, value: Any) -> None:
        """
        Set state in a managed remote service via its `set_state` endpoint (best-effort).

        This is used to propagate UI property edits into the runtime so the
        running node behavior matches the values shown in Node Properties.
        """
        try:
            service_id = ensure_token(str(service_id), label="service_id")
            node_id = ensure_token(str(node_id), label="node_id")
        except Exception:
            return
        field = str(field or "").strip()
        if not field:
            return

        def _coerce_json_value(v: Any) -> Any:
            if v is None or isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, (list, tuple)):
                return [_coerce_json_value(x) for x in v]
            if isinstance(v, dict):
                return {str(k): _coerce_json_value(x) for k, x in v.items()}
            try:
                model_dump = getattr(v, "model_dump", None)
                if callable(model_dump):
                    return _coerce_json_value(model_dump(mode="json"))
            except Exception:
                pass
            try:
                if hasattr(v, "root"):
                    return _coerce_json_value(getattr(v, "root"))
            except Exception:
                pass
            return v

        value_json = _coerce_json_value(value)

        async def _do() -> None:
            nc = await self._ensure_nc()
            if nc is None:
                return
            subject = svc_endpoint_subject(service_id, "set_state")
            payload = json.dumps(
                {
                    "reqId": new_id(),
                    "args": {"nodeId": node_id, "field": field, "value": value_json},
                    "meta": {"actor": "studio", "source": "ui"},
                },
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
            for _ in range(3):
                try:
                    msg = await nc.request(subject, payload, timeout=0.5)
                    raw = bytes(getattr(msg, "data", b"") or b"")
                    if not raw:
                        continue
                    try:
                        resp = json.loads(raw.decode("utf-8"))
                    except Exception:
                        resp = {}
                    if isinstance(resp, dict) and resp.get("ok") is True:
                        return
                    # Validation errors are expected; surface them.
                    if isinstance(resp, dict) and resp.get("ok") is False:
                        err = resp.get("error") if isinstance(resp.get("error"), dict) else {}
                        code = str(err.get("code") or "")
                        msg_s = str(err.get("message") or "")
                        if code or msg_s:
                            self.log.emit(f"set_state rejected serviceId={service_id} nodeId={node_id} field={field} code={code} msg={msg_s}")
                        return
                except Exception:
                    await asyncio.sleep(0.1)
                    continue

        try:
            self._async.submit(_do())
        except Exception:
            return

    def invoke_remote_command(self, service_id: str, call: str, args: dict[str, Any] | None = None) -> None:
        """
        Invoke a user-defined command on a remote service via the reserved cmd channel.

        Request is sent to `cmd_channel_subject(service_id)` with a JSON envelope
        (reqId/call/args/meta). This matches the service control plane `cmd` endpoint.
        """
        try:
            service_id = ensure_token(str(service_id), label="service_id")
        except Exception:
            return
        call = str(call or "").strip()
        if not call or service_id == self.studio_service_id:
            return

        def _coerce_json_value(v: Any) -> Any:
            if v is None or isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, (list, tuple)):
                return [_coerce_json_value(x) for x in v]
            if isinstance(v, dict):
                return {str(k): _coerce_json_value(x) for k, x in v.items()}
            try:
                model_dump = getattr(v, "model_dump", None)
                if callable(model_dump):
                    return _coerce_json_value(model_dump(mode="json"))
            except Exception:
                pass
            try:
                if hasattr(v, "root"):
                    return _coerce_json_value(getattr(v, "root"))
            except Exception:
                pass
            return v

        args_json = _coerce_json_value(args or {})

        async def _do() -> None:
            nc = await self._ensure_nc()
            if nc is None:
                return
            subject = cmd_channel_subject(service_id)
            payload = json.dumps(
                {
                    "reqId": new_id(),
                    "call": call,
                    "args": args_json,
                    "meta": {"actor": "studio", "source": "ui"},
                },
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
            try:
                msg = await nc.request(subject, payload, timeout=1.5)
                raw = bytes(getattr(msg, "data", b"") or b"")
                if not raw:
                    self.log.emit(f"command {call} failed serviceId={service_id}: empty response")
                    return
                try:
                    resp = json.loads(raw.decode("utf-8"))
                except Exception:
                    resp = {}
                if isinstance(resp, dict) and resp.get("ok") is True:
                    return
                msg = ""
                try:
                    msg = str((resp.get("error") or {}).get("message") or "")
                except Exception:
                    msg = ""
                self.log.emit(f"command {call} failed serviceId={service_id}: {msg or 'rejected'}")
            except Exception as exc:
                self.log.emit(f"command {call} failed serviceId={service_id}: {type(exc).__name__}: {exc}")

        try:
            self._async.submit(_do())
        except Exception:
            return

    async def _start_async(self) -> None:
        nats_url = str(self._cfg.nats_url).strip() or "nats://127.0.0.1:4222"

        # Singleton guard (best-effort): if any existing studio ServiceBus micro responds, do not start.
        try:
            async def _err_cb(exc: Exception) -> None:
                now = time.monotonic()
                if (now - self._last_nats_error_log_s) < 2.0:
                    return
                self._last_nats_error_log_s = now
                try:
                    self.log.emit(f"NATS not reachable at {nats_url!r} (will retry): {type(exc).__name__}: {exc}")
                except Exception:
                    pass

            self._nc = await nats.connect(servers=[nats_url], connect_timeout=2, error_cb=_err_cb)
            try:
                await self._nc.request(f"$SRV.PING.{svc_micro_name(self.studio_service_id)}", b"", timeout=0.2)
                self.log.emit("Another PyStudio instance is already running (micro service ping responded).")
                await self._nc.close()
                self._nc = None
                return
            except Exception:
                pass
        except Exception:
            self._nc = None

        try:
            cfg = PyStudioServiceConfig(nats_url=nats_url, studio_service_id=self.studio_service_id)
            self._svc = PyStudioService(cfg, registry=RuntimeNodeRegistry.instance())
            await self._svc.start(
                on_ui_command=lambda cmd: self.ui_command.emit(cmd),
            )
        except Exception as exc:
            self.log.emit(f"studio runtime start failed: {exc}")
            self._svc = None

        # Re-apply current desired lifecycle to any already-known managed services.
        try:
            await self._set_managed_active_async(bool(self._managed_active))
        except Exception:
            pass

    async def _stop_async(self) -> None:
        try:
            if self._svc is not None:
                await self._svc.stop()
        except Exception:
            pass
        self._svc = None

        try:
            if self._nc is not None:
                await self._nc.close()
        except Exception:
            pass
        self._nc = None

    async def _deploy_and_monitor_async(self, compiled: CompiledRuntimeGraphs) -> None:
        nats_url = str(self._cfg.nats_url).strip() or "nats://127.0.0.1:4222"

        # Deploy per-service rungraphs.
        for sid, g in compiled.per_service.items():
            service_id = ensure_token(str(sid), label="service_id")
            if str(service_id) == self.studio_service_id:
                continue
            bucket = kv_bucket_for_service(service_id)
            tr = NatsTransport(NatsTransportConfig(url=nats_url, kv_bucket=bucket))
            try:
                await tr.connect()
                try:
                    await wait_service_ready(tr, timeout_s=6.0)
                except Exception as exc:
                    raise RuntimeError(f"service not ready: {exc}")
                payload = g.model_dump(mode="json", by_alias=True)
                meta = dict(payload.get("meta") or {})
                if not str(meta.get("source") or "").strip():
                    meta["source"] = "studio"
                payload["meta"] = meta
                # Endpoint-only mode: deploy via service endpoint (allows validation/rejection).
                req = {"reqId": new_id(), "args": {"graph": payload}, "meta": {"source": "studio"}}
                req_bytes = json.dumps(req, ensure_ascii=False, default=str).encode("utf-8")
                resp_raw = await tr.request(
                    svc_endpoint_subject(service_id, "set_rungraph"),
                    req_bytes,
                    timeout=2.0,
                    raise_on_error=True,
                )
                if not resp_raw:
                    raise RuntimeError("set_rungraph request failed: empty response")
                try:
                    resp = json.loads(resp_raw.decode("utf-8"))
                except Exception:
                    resp = {}
                if isinstance(resp, dict) and resp.get("ok") is True:
                    pass
                elif isinstance(resp, dict) and resp.get("ok") is False:
                    msg = ""
                    try:
                        msg = str((resp.get("error") or {}).get("message") or "")
                    except Exception:
                        msg = ""
                    raise RuntimeError(msg or "set_rungraph rejected")
                else:
                    raise RuntimeError("invalid set_rungraph response")
            except Exception as exc:
                self.log.emit(f"deploy failed serviceId={service_id}: {exc}")
            finally:
                try:
                    await tr.close()
                except Exception:
                    pass

        # Install monitoring graph into studio serviceId.
        if self._svc is None or self._svc.bus is None:
            return
        try:
            studio_graph = self._build_studio_monitor_graph(compiled)
            await self._svc.bus.set_rungraph(studio_graph)
        except Exception as exc:
            self.log.emit(f"install monitor graph failed: {exc}")

    async def _ensure_nc(self) -> Any | None:
        """
        Ensure a NATS connection exists for command channel requests.
        """
        if self._nc is not None:
            return self._nc
        nats_url = str(self._cfg.nats_url).strip() or "nats://127.0.0.1:4222"
        try:
            async def _err_cb(exc: Exception) -> None:
                now = time.monotonic()
                if (now - self._last_nats_error_log_s) < 2.0:
                    return
                self._last_nats_error_log_s = now
                try:
                    self.log.emit(f"NATS not reachable at {nats_url!r} (will retry): {type(exc).__name__}: {exc}")
                except Exception:
                    pass

            self._nc = await nats.connect(servers=[nats_url], connect_timeout=2, error_cb=_err_cb)
        except Exception:
            self._nc = None
        return self._nc

    async def _set_managed_active_async(self, active: bool) -> None:
        nc = await self._ensure_nc()
        if nc is None:
            return
        service_ids = sorted({sid for sid in self._managed_service_ids if sid and sid != self.studio_service_id})
        if not service_ids:
            return

        cmd = "activate" if active else "deactivate"
        payload = json.dumps(
            {"reqId": new_id(), "args": {"active": bool(active)}, "meta": {"actor": "studio", "cmd": cmd}},
            ensure_ascii=False,
        ).encode("utf-8")

        for sid in service_ids:
            subject = svc_endpoint_subject(sid, cmd)
            ok = False
            for _ in range(3):
                try:
                    msg = await nc.request(subject, payload, timeout=0.5)
                    data = bytes(getattr(msg, "data", b"") or b"")
                    if data:
                        try:
                            resp = json.loads(data.decode("utf-8"))
                        except Exception:
                            resp = {}
                        if isinstance(resp, dict) and resp.get("ok") is True:
                            ok = True
                            break
                except Exception:
                    await asyncio.sleep(0.2)
                    continue
            if not ok:
                try:
                    self.log.emit(f"lifecycle {cmd} failed serviceId={sid}")
                except Exception:
                    pass
