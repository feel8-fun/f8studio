from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import nats  # type: ignore[import-not-found]
from qtpy import QtCore

from f8pysdk import F8Edge, F8RuntimeGraph, F8RuntimeNode
from f8pysdk.nats_naming import ensure_token, new_id, svc_endpoint_subject, svc_micro_name
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.service_bus import StateWriteOrigin
from f8pysdk.service_bus.state_write import StateWriteError
from .bridge.async_runtime import AsyncRuntimeThread
from .bridge.command_client import CommandRequest, NatsCommandGateway
from .bridge.json_codec import coerce_json_value
from .bridge.process_lifecycle import (
    LocalServiceProcessGateway,
    StartServiceRequest,
    StopServiceRequest,
)
from .bridge.remote_state_sync import ApplyWatchTargetsRequest, RemoteStateGatewayAdapter
from .bridge.rungraph_deployer import (
    NatsRungraphGateway,
    RungraphDeployConfig,
    RungraphDeployRequest,
)
from .error_reporting import ExceptionLogOnce, report_exception
from f8pysdk.nats_server_bootstrap import ensure_nats_server
from .nodegraph.runtime_compiler import CompiledRuntimeGraphs
from .pystudio_service import PyStudioService, PyStudioServiceConfig
from .service_process_manager import ServiceProcessConfig, ServiceProcessManager
from .pystudio_node_registry import SERVICE_CLASS, STUDIO_SERVICE_ID
from .remote_state_watcher import RemoteStateWatcher, WatchTarget
from .ui_bus import UiCommand

logger = logging.getLogger(__name__)


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
    - monitor remote state via Studio-side KV watches (UI reflection)
    """

    # Note: Qt `int` is typically 32-bit; use `object` for ts_ms (ms timestamps exceed 2^31).
    ui_command = QtCore.Signal(object)  # UiCommand
    service_output = QtCore.Signal(str, str)  # serviceId, line
    log = QtCore.Signal(str)
    service_process_state = QtCore.Signal(str, bool)  # serviceId, running
    _remote_command_response = QtCore.Signal(str, object, object)  # reqId, result, err

    def __init__(self, config: PyStudioServiceBridgeConfig, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._cfg = config
        self._async = AsyncRuntimeThread()
        self._proc_mgr = ServiceProcessManager()
        self._process_gateway = LocalServiceProcessGateway(self._proc_mgr)
        self._rungraph_gateway = NatsRungraphGateway(RungraphDeployConfig(nats_url=self._cfg.nats_url))
        self._command_gateway = NatsCommandGateway(nats_url=self._cfg.nats_url)
        self._exception_log_once = ExceptionLogOnce()
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
        self._local_state_fields_by_node: dict[str, tuple[str, ...]] = {}
        self._shutting_down: bool = False

        self._svc: PyStudioService | None = None
        self._remote_state_watcher: RemoteStateWatcher | None = None
        self._remote_state_gateway: RemoteStateGatewayAdapter | None = None
        self._watch_targets_cache: tuple[WatchTarget, ...] | None = None
        self._nc: Any = None
        self._last_nats_error_log_s: float = 0.0
        self._pending_remote_command_cbs: dict[str, Callable[[dict[str, Any] | None, str | None], None]] = {}

        try:
            self._remote_command_response.connect(self._on_remote_command_response)  # type: ignore[attr-defined]
        except Exception as exc:
            self._report_exception("connect remote_command_response failed", exc)

    def _emit_log_line(self, line: str) -> None:
        try:
            self.log.emit(str(line))
        except Exception:
            logger.exception("bridge.log.emit failed")

    def _report_exception(self, context: str, exc: BaseException) -> None:
        report_exception(
            self._emit_log_line,
            context=str(context or "").strip(),
            exc=exc,
            log_once=self._exception_log_once,
        )
        try:
            logger.error("%s", str(context or "").strip(), exc_info=exc)
        except Exception:
            # Logging must never crash the bridge.
            pass

    def _emit_remote_command_response_safe(self, req_id: str, result: object, err: object) -> None:
        try:
            self._remote_command_response.emit(str(req_id), result, err)
        except RuntimeError as exc:
            self._report_exception("emit remote command response failed", exc)

    def _submit_async(self, coro: Any, *, context: str) -> bool:
        if self._shutting_down or (not self._async.is_accepting_submissions()):
            if asyncio.iscoroutine(coro):
                coro.close()
            return False
        try:
            self._async.submit(coro)
            return True
        except RuntimeError as exc:
            if asyncio.iscoroutine(coro):
                coro.close()
            if self._shutting_down or (not self._async.is_accepting_submissions()):
                return False
            self._report_exception(context, exc)
            return False
        except Exception as exc:
            self._report_exception(context, exc)
            return False

    @staticmethod
    def _message_data_bytes(message: Any) -> bytes:
        try:
            data = message.data
        except AttributeError:
            return b""
        try:
            return bytes(data or b"")
        except (TypeError, ValueError):
            return b""

    @staticmethod
    def _decode_json_object(raw: bytes) -> dict[str, Any] | None:
        if not raw:
            return None
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if isinstance(decoded, dict):
            return decoded
        return None

    @QtCore.Slot(str, object, object)
    def _on_remote_command_response(self, req_id: str, result: object, err: object) -> None:
        cb = self._pending_remote_command_cbs.pop(str(req_id), None)
        if cb is None:
            return
        try:
            cb(result if isinstance(result, dict) else None, str(err) if err else None)
        except Exception as exc:
            self._report_exception("remote command response callback failed", exc)

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
        self._managed_active = bool(active)
        self._submit_async(
            self._set_managed_active_async(bool(self._managed_active)),
            context="submit set_managed_active failed",
        )

    def start(self) -> None:
        self._shutting_down = False
        self._async.start()
        self._submit_async(self._start_async(), context="submit start failed")

    def stop(self) -> None:
        self._shutting_down = True
        try:
            fut = self._async.submit(self._stop_async())
            try:
                fut.result(timeout=2)
            except concurrent.futures.TimeoutError:
                self._emit_log_line("bridge stop timeout; continue shutdown")
        except Exception as exc:
            self._report_exception("submit stop failed", exc)
        self._async.stop()

        # Best-effort stop all launched processes.
        for sid in list(self._process_gateway.service_ids()):
            try:
                self._process_gateway.stop(StopServiceRequest(service_id=sid))
            except Exception as exc:
                self._report_exception(f"stop service process failed serviceId={sid}", exc)

    def deploy(self, compiled: CompiledRuntimeGraphs) -> None:
        """
        Starts service processes (if not running), deploys per-service graphs,
        installs the studio runtime graph, and enables remote state monitoring.
        """
        # 1) start processes (sync)
        self._last_compiled = compiled
        self._local_state_fields_by_node = self._build_local_state_field_index(compiled)
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
                self._emit_log_line(f"start service failed: {exc}")
        self._managed_service_ids = managed
        self._managed_service_classes = managed_classes

        # 2) deploy + install monitoring (async)
        self._submit_async(self._deploy_and_monitor_async(compiled), context="submit deploy_and_monitor failed")
        # Preserve the current global lifecycle preference across repeated deploys.
        # Only enforce deactivate here when globally paused; avoid forcing activate
        # on every F5, which can override rungraph/state-edge driven inactive states.
        if not bool(self._managed_active):
            self.set_managed_active(False)

    @QtCore.Slot(str)
    def unmanage_service(self, service_id: str) -> None:
        """
        Remove a serviceId from the studio's managed set (UI bookkeeping only).

        This does not stop the process by itself.
        """
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except ValueError:
            return
        if sid == self.studio_service_id:
            return

        self._managed_service_ids.discard(sid)
        self._managed_service_classes.pop(sid, None)
        self._service_status_cache.pop(sid, None)

        # Cancel any pending process actions/timers for this service.
        self._pending_proc_actions.pop(sid, None)
        timer = self._pending_proc_timers.pop(sid, None)
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError as exc:
                self._report_exception(f"stop pending process timer failed serviceId={sid}", exc)

    @QtCore.Slot(str)
    def reclaim_service(self, service_id: str) -> None:
        """
        Best-effort reclamation for a serviceId that was removed from the canvas:
        terminate the process and drop it from the managed set.
        """
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except ValueError:
            return
        if sid == self.studio_service_id:
            return
        self._emit_log_line(f"reclaim service process serviceId={sid}")
        self.stop_service(sid)
        self.unmanage_service(sid)

    @QtCore.Slot(str)
    def deploy_service_rungraph(self, service_id: str, *, compiled: CompiledRuntimeGraphs | None = None) -> None:
        """
        Deploy the current per-service rungraph to a running service instance (best-effort).
        """
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except ValueError:
            return
        if sid == self.studio_service_id:
            return
        if not self.is_service_running(sid):
            self._emit_log_line(f"deploy skipped (service not running) serviceId={sid}")
            return

        async def _do() -> None:
            await self._install_studio_graph_async(compiled=compiled)
            await self._deploy_service_rungraph_async(sid, compiled=compiled)

        self._submit_async(_do(), context=f"submit deploy_service_rungraph failed serviceId={sid}")

    async def _ensure_studio_runtime_async(self, *, timeout_s: float = 6.0) -> bool:
        """
        Best-effort wait for the in-process studio runtime (ServiceRuntime) to be ready.
        """
        deadline = time.monotonic() + float(timeout_s or 0.0)
        while True:
            svc = self._svc
            if svc is not None and svc.bus is not None:
                return True
            if time.monotonic() >= deadline:
                self._emit_log_line("studio runtime not ready (timeout)")
                return False
            await asyncio.sleep(0.08)

    @staticmethod
    def _pick_compiled(
        compiled: CompiledRuntimeGraphs | None, fallback: CompiledRuntimeGraphs | None
    ) -> CompiledRuntimeGraphs | None:
        return compiled if compiled is not None else fallback

    def _build_studio_runtime_graph(self, compiled: CompiledRuntimeGraphs) -> F8RuntimeGraph:
        """
        Build the studio runtime graph without installing a monitor node.

        Remote state monitoring is handled by `RemoteStateWatcher` (Studio-side KV subscription).
        """
        studio_sub = compiled.per_service.get(self.studio_service_id)
        if studio_sub is None:
            base_nodes = []
            base_edges = []
        else:
            try:
                base_nodes = list(studio_sub.nodes or [])
            except Exception:
                base_nodes = []
            try:
                base_edges = list(studio_sub.edges or [])
            except Exception:
                base_edges = []
        try:
            graph_id = str(compiled.global_graph.graphId or "studio")
        except Exception:
            graph_id = "studio"
        try:
            revision = str(compiled.global_graph.revision or "1")
        except Exception:
            revision = "1"
        return F8RuntimeGraph(
            graphId=graph_id,
            revision=revision,
            meta={"source": "studio"},
            services=[],
            nodes=[*base_nodes],
            edges=[*base_edges],
        )

    @staticmethod
    def _dedupe_fields(fields: list[str]) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for field in fields:
            if field in seen:
                continue
            seen.add(field)
            ordered.append(field)
        return tuple(ordered)

    def _build_remote_watch_targets(self, compiled: CompiledRuntimeGraphs) -> tuple[WatchTarget, ...]:
        targets: list[WatchTarget] = []
        try:
            nodes = list(compiled.global_graph.nodes or [])
        except Exception:
            nodes = []
        for n in nodes:
            try:
                sid = ensure_token(str(n.serviceId or ""), label="service_id")
                nid = ensure_token(str(n.nodeId or ""), label="node_id")
            except ValueError as exc:
                self._emit_log_line(f"skip invalid remote watch target: {type(exc).__name__}: {exc}")
                continue

            candidates: list[str] = []
            try:
                state_fields = list(n.stateFields or [])
            except Exception:
                state_fields = []
            for sf in state_fields:
                try:
                    name = str(sf.name or "").strip()
                except Exception:
                    name = ""
                if name:
                    candidates.append(name)

            if "svcId" not in candidates:
                candidates.append("svcId")
            op_class = str(n.operatorClass or "").strip()
            if op_class and "operatorId" not in candidates:
                candidates.append("operatorId")

            targets.append(
                WatchTarget(
                    service_id=sid,
                    node_id=nid,
                    fields=self._dedupe_fields(candidates),
                )
            )
        return tuple(sorted(targets, key=lambda t: (t.service_id, t.node_id, t.fields)))

    def _build_local_state_field_index(self, compiled: CompiledRuntimeGraphs) -> dict[str, tuple[str, ...]]:
        studio_graph = compiled.per_service.get(self.studio_service_id)
        if studio_graph is None:
            return {}
        out: dict[str, tuple[str, ...]] = {}
        for node in list(studio_graph.nodes or []):
            node_id = str(node.nodeId or "").strip()
            if not node_id:
                continue
            field_names: list[str] = []
            for field_spec in list(node.stateFields or []):
                name = str(field_spec.name or "").strip()
                if name:
                    field_names.append(name)
            out[node_id] = self._dedupe_fields(field_names)
        return out

    async def _apply_remote_state_watches_async(self, compiled: CompiledRuntimeGraphs) -> None:
        gateway = self._remote_state_gateway
        if gateway is None:
            return
        targets_sorted = self._build_remote_watch_targets(compiled)
        if self._watch_targets_cache == targets_sorted:
            return
        await gateway.apply_targets(ApplyWatchTargetsRequest(targets=targets_sorted))
        self._watch_targets_cache = targets_sorted

    def is_service_running(self, service_id: str) -> bool:
        sid = str(service_id or "").strip()
        if not sid:
            return False
        try:
            if bool(self._process_gateway.is_running(str(sid))):
                return True
        except Exception as exc:
            self._report_exception(f"check process running failed serviceId={sid}", exc)
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
        except ValueError:
            return ""
        return str(self._managed_service_classes.get(sid, "") or "")

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
        except ValueError:
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
        raw = self._message_data_bytes(msg)
        if not raw:
            return None
        resp = self._decode_json_object(raw)
        if resp is None:
            return None
        if not (isinstance(resp, dict) and resp.get("ok") is True):
            return None
        result = resp.get("result") if isinstance(resp.get("result"), dict) else {}
        if not isinstance(result, dict):
            return None
        out: dict[str, Any] = {"alive": True}
        if "active" in result:
            out["active"] = bool(result.get("active"))
        return out

    def request_service_status(self, service_id: str) -> None:
        """
        Trigger a best-effort status refresh (async).
        """
        sid = ""
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except ValueError:
            return
        now = time.monotonic()
        last = float(self._service_status_req_s.get(sid, 0.0))
        if (now - last) < 0.25:
            return
        if sid in self._service_status_inflight:
            return
        self._service_status_inflight.add(sid)
        self._service_status_req_s[sid] = now

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
                self._service_status_inflight.discard(sid)

        submitted = self._submit_async(_do(), context=f"submit request_service_status failed serviceId={sid}")
        if not submitted:
            self._service_status_inflight.discard(sid)

    async def _set_service_active_async(self, service_id: str, active: bool) -> bool:
        sid = ""
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except ValueError:
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
                data = self._message_data_bytes(msg)
                if data:
                    resp = self._decode_json_object(data) or {}
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
        except ValueError:
            return
        if sid == self.studio_service_id:
            return

        self._submit_async(
            self._set_service_active_async(sid, bool(active)),
            context=f"submit set_service_active failed serviceId={sid}",
        )

    async def _request_service_terminate_async(self, service_id: str) -> bool:
        """
        Ask a running service process to exit itself (graceful).

        This is best-effort and may fail if the service isn't connected to NATS yet.
        """
        sid = ""
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except ValueError:
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
                raw = self._message_data_bytes(msg)
                if not raw:
                    continue
                resp = self._decode_json_object(raw) or {}
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
        timer = self._pending_proc_timers.pop(sid, None)
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError as exc:
                self._report_exception(f"stop proc-action timer failed serviceId={sid}", exc)
            try:
                timer.deleteLater()
            except RuntimeError as exc:
                self._report_exception(f"delete proc-action timer failed serviceId={sid}", exc)
        self._pending_proc_actions.pop(sid, None)

    def _poll_proc_action(self, service_id: str) -> None:
        sid = str(service_id)
        action = self._pending_proc_actions.get(sid)
        if not action:
            self._clear_proc_action(sid)
            return

        if not self.is_service_running(sid):
            self._clear_proc_action(sid)
            try:
                self.service_process_state.emit(sid, False)
            except RuntimeError as exc:
                self._report_exception(f"emit service process state failed serviceId={sid}", exc)

            if action.get("action") == "restart":
                svc_class = str(action.get("serviceClass") or "").strip() or None
                self.start_service(sid, service_class=svc_class)
            return

        deadline = float(action.get("deadline") or 0.0)
        if deadline and time.monotonic() < deadline:
            return

        # Grace period expired: fall back to local hard stop (taskkill / kill-tree).
        try:
            stop_result = self._process_gateway.stop(StopServiceRequest(service_id=sid))
            ok = bool(stop_result.success)
        except Exception as exc:
            self._emit_log_line(f"stop_service failed: {exc}")
            ok = False

        still_running = bool(self.is_service_running(sid))
        if not ok and still_running:
            self._emit_log_line(f"stop_service incomplete (process still running): serviceId={sid}")
            # Keep timer running; do not clear, to avoid allowing duplicates.
            self._pending_proc_actions[sid]["deadline"] = time.monotonic() + 1.0
            return

        self._clear_proc_action(sid)
        try:
            self.service_process_state.emit(sid, still_running)
        except RuntimeError as exc:
            self._report_exception(f"emit service process state failed serviceId={sid}", exc)

        if not still_running and action.get("action") == "restart":
            svc_class = str(action.get("serviceClass") or "").strip() or None
            self.start_service(sid, service_class=svc_class)

    @QtCore.Slot(str)
    def start_service(self, service_id: str, *, service_class: str | None = None) -> None:
        sid = ""
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except ValueError:
            return
        if sid == self.studio_service_id:
            return

        # Dedup: if Studio already believes the service is alive (via local proc tracking or a fresh
        # status ping), do not spawn another process on repeated deploy (e.g. repeated F5).
        if self.is_service_running(sid):
            self._emit_log_line(f"start_service ignored (already running): serviceId={sid}")
            return

        svc_class = self._managed_service_classes.get(sid, "") or str(service_class or "")
        if not svc_class:
            self._emit_log_line(f"start_service ignored (unknown serviceClass): serviceId={sid}")
            return
        try:
            self._process_gateway.start(
                StartServiceRequest(
                    config=ServiceProcessConfig(service_class=str(svc_class), service_id=sid, nats_url=self._cfg.nats_url),
                    on_output=lambda _sid, line, _sid2=sid: self.service_output.emit(_sid2, str(line)),
                )
            )
        except Exception as exc:
            self._emit_log_line(f"start_service failed: {exc}")
            return
        self._managed_service_ids.add(sid)
        if svc_class:
            self._managed_service_classes[sid] = str(svc_class)
        try:
            self.service_process_state.emit(sid, bool(self.is_service_running(sid)))
        except RuntimeError as exc:
            self._report_exception(f"emit service process state failed serviceId={sid}", exc)

    @QtCore.Slot(str)
    def stop_service(self, service_id: str) -> None:
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except ValueError:
            return
        if sid == self.studio_service_id:
            return

        if not self.is_service_running(sid):
            try:
                self.service_process_state.emit(sid, False)
            except RuntimeError as exc:
                self._report_exception(f"emit service process state failed serviceId={sid}", exc)
            return

        # 1) Ask the service to terminate itself (best for GUI apps / child process trees).
        self._submit_async(
            self._request_service_terminate_async(sid),
            context=f"submit request_service_terminate failed serviceId={sid}",
        )

        # 2) Poll for graceful exit, then fall back to local kill-tree.
        self._pending_proc_actions[sid] = {"action": "stop", "deadline": time.monotonic() + 2.2}
        t = self._ensure_proc_action_timer(sid)
        if not t.isActive():
            t.start()
        self._poll_proc_action(sid)

    @QtCore.Slot(str)
    def restart_service(self, service_id: str, *, service_class: str | None = None) -> None:
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except ValueError:
            return
        if sid == self.studio_service_id:
            return

        svc_class = self._managed_service_classes.get(sid, "") or str(service_class or "")

        if not self.is_service_running(sid):
            self.start_service(sid, service_class=svc_class or None)
            return

        self._submit_async(
            self._request_service_terminate_async(sid),
            context=f"submit request_service_terminate failed serviceId={sid}",
        )

        self._pending_proc_actions[sid] = {"action": "restart", "deadline": time.monotonic() + 2.2, "serviceClass": svc_class}
        t = self._ensure_proc_action_timer(sid)
        if not t.isActive():
            t.start()
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
        except ValueError:
            return
        g = compiled.per_service.get(sid)
        if g is None:
            return

        try:
            result = await self._rungraph_gateway.deploy_runtime_graph(
                RungraphDeployRequest(
                    service_id=sid,
                    graph=g,
                    source="studio",
                )
            )
            if not result.success:
                raise RuntimeError(result.error_message or "set_rungraph rejected")
        except Exception as exc:
            self._emit_log_line(f"deploy service rungraph failed serviceId={sid}: {exc}")

    async def _install_studio_graph_async(self, *, compiled: CompiledRuntimeGraphs | None = None) -> None:
        """
        Reinstall the studio runtime graph from the last compiled graphs (best-effort).
        """
        compiled = self._pick_compiled(compiled, self._last_compiled)
        if compiled is None:
            return
        self._local_state_fields_by_node = self._build_local_state_field_index(compiled)
        if not await self._ensure_studio_runtime_async():
            return
        try:
            if self._svc is None or self._svc.bus is None:
                return
            studio_graph = self._build_studio_runtime_graph(compiled)
            await self._svc.bus.set_rungraph(studio_graph)
        except Exception as exc:
            self._emit_log_line(f"install studio graph failed: {exc}")
        try:
            await self._apply_remote_state_watches_async(compiled)
        except Exception as exc:
            self._report_exception("apply remote state watches failed", exc)

    @QtCore.Slot(str)
    def start_service_and_deploy(
        self, service_id: str, *, service_class: str | None = None, compiled: CompiledRuntimeGraphs | None = None
    ) -> None:
        """
        Start service process (if needed) and deploy last compiled rungraph for it (best-effort).
        """
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except ValueError:
            return
        if sid == self.studio_service_id:
            return
        self.start_service(sid, service_class=service_class)

        async def _do() -> None:
            await self._install_studio_graph_async(compiled=compiled)
            await self._deploy_service_rungraph_async(sid, compiled=compiled)
            await self._set_service_active_async(sid, True)

        self._submit_async(_do(), context=f"submit start_service_and_deploy failed serviceId={sid}")

    @QtCore.Slot(str)
    def restart_service_and_deploy(
        self, service_id: str, *, service_class: str | None = None, compiled: CompiledRuntimeGraphs | None = None
    ) -> None:
        """
        Restart service (terminate -> start) and deploy last compiled rungraph for it (best-effort).
        """
        try:
            sid = ensure_token(str(service_id), label="service_id")
        except ValueError:
            return
        if sid == self.studio_service_id:
            return

        # Reuse existing restart flow; deploy will happen once process is back.
        self.restart_service(sid, service_class=service_class)

        async def _do() -> None:
            # Give restart a moment to come back; readiness wait inside deploy handles most cases.
            await asyncio.sleep(0.3)
            await self._install_studio_graph_async(compiled=compiled)
            await self._deploy_service_rungraph_async(sid, compiled=compiled)
            await self._set_service_active_async(sid, True)

        self._submit_async(_do(), context=f"submit restart_service_and_deploy failed serviceId={sid}")

    def set_local_state(self, node_id: str, field: str, value: Any) -> None:
        """
        Set state in the local studio service KV (best-effort).
        """
        node_id = ensure_token(node_id, label="node_id")
        field = str(field or "").strip()
        if not field:
            return
        allowed_fields = self._local_state_fields_by_node.get(node_id)
        if allowed_fields is None or field not in allowed_fields:
            return

        async def _do() -> None:
            if self._svc is None or self._svc.bus is None:
                return
            try:
                await self._svc.bus.publish_state_external(node_id, field, value, source="pystudio")
            except StateWriteError as exc:
                if "unknown state field" in str(exc):
                    logger.warning("Skip local state publish for unknown field: %s.%s", node_id, field)
                    return
                self._report_exception("publish local state failed", exc)
            except Exception as exc:
                self._report_exception("publish local state failed", exc)

        self._submit_async(_do(), context=f"submit set_local_state failed nodeId={node_id}")

    def set_remote_state(self, service_id: str, node_id: str, field: str, value: Any) -> None:
        """
        Set state in a managed remote service via its `set_state` endpoint (best-effort).

        This is used to propagate UI property edits into the runtime so the
        running node behavior matches the values shown in Node Properties.
        """
        try:
            service_id = ensure_token(str(service_id), label="service_id")
            node_id = ensure_token(str(node_id), label="node_id")
        except ValueError:
            return
        field = str(field or "").strip()
        if not field:
            return

        value_json = coerce_json_value(value)

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
                    raw = self._message_data_bytes(msg)
                    if not raw:
                        continue
                    resp = self._decode_json_object(raw) or {}
                    if isinstance(resp, dict) and resp.get("ok") is True:
                        return
                    # Validation errors are expected; surface them.
                    if isinstance(resp, dict) and resp.get("ok") is False:
                        err = resp.get("error") if isinstance(resp.get("error"), dict) else {}
                        code = str(err.get("code") or "")
                        msg_s = str(err.get("message") or "")
                        if code or msg_s:
                            self._emit_log_line(
                                f"set_state rejected serviceId={service_id} nodeId={node_id} field={field} code={code} msg={msg_s}"
                            )
                        return
                except Exception:
                    await asyncio.sleep(0.1)
                    continue

        self._submit_async(
            _do(),
            context=f"submit set_remote_state failed serviceId={service_id} nodeId={node_id} field={field}",
        )

    def request_remote_command(
        self,
        service_id: str,
        call: str,
        args: dict[str, Any] | None,
        cb: Callable[[dict[str, Any] | None, str | None], None],
        *,
        timeout_s: float = 2.0,
    ) -> None:
        """
        Invoke a user-defined command on a remote service and return the parsed `result`.

        Callback is always delivered on the Qt main thread as:
        - cb(result_dict, None) on success (result may be any JSON object; non-dict becomes {"value": ...})
        - cb(None, "error message") on failure
        """
        req_id = new_id()
        self._pending_remote_command_cbs[str(req_id)] = cb

        try:
            service_id = ensure_token(str(service_id), label="service_id")
        except ValueError as exc:
            self._emit_remote_command_response_safe(str(req_id), None, str(exc))
            return
        call = str(call or "").strip()
        if not call or service_id == self.studio_service_id:
            self._emit_remote_command_response_safe(str(req_id), None, "invalid call/service_id")
            return

        async def _do() -> None:
            try:
                response = await self._command_gateway.request_command(
                    CommandRequest(
                        service_id=service_id,
                        call=call,
                        args=args or {},
                        timeout_s=float(timeout_s),
                        source="ui",
                        actor="studio",
                    )
                )
                if response.ok:
                    self._emit_remote_command_response_safe(str(req_id), dict(response.result), None)
                    return
                self._emit_remote_command_response_safe(str(req_id), None, str(response.error_message or "rejected"))
            except Exception as exc:
                self._emit_remote_command_response_safe(str(req_id), None, f"{type(exc).__name__}: {exc}")

        submitted = self._submit_async(_do(), context=f"submit request_remote_command failed serviceId={service_id}")
        if not submitted:
            self._emit_remote_command_response_safe(str(req_id), None, "submit failed")

    def invoke_remote_command(self, service_id: str, call: str, args: dict[str, Any] | None = None) -> None:
        """
        Invoke a user-defined command on a remote service via the reserved cmd channel.

        Request is sent to `cmd_channel_subject(service_id)` with a JSON envelope
        (reqId/call/args/meta). This matches the service control plane `cmd` endpoint.
        """
        try:
            service_id = ensure_token(str(service_id), label="service_id")
        except ValueError:
            return
        call = str(call or "").strip()
        if not call or service_id == self.studio_service_id:
            return

        async def _do() -> None:
            try:
                response = await self._command_gateway.request_command(
                    CommandRequest(
                        service_id=service_id,
                        call=call,
                        args=args or {},
                        timeout_s=1.5,
                        source="ui",
                        actor="studio",
                    )
                )
                if response.ok:
                    return
                self._emit_log_line(f"command {call} failed serviceId={service_id}: {response.error_message or 'rejected'}")
            except Exception as exc:
                self._emit_log_line(f"command {call} failed serviceId={service_id}: {type(exc).__name__}: {exc}")

        self._submit_async(_do(), context=f"submit invoke_remote_command failed serviceId={service_id}")

    async def _start_async(self) -> None:
        nats_url = str(self._cfg.nats_url).strip() or "nats://127.0.0.1:4222"
        try:
            await asyncio.to_thread(ensure_nats_server, nats_url, log_cb=self._emit_log_line)
        except Exception as exc:
            self._report_exception("ensure nats server failed", exc)

        # Singleton guard (best-effort): if any existing studio ServiceBus micro responds, do not start.
        try:
            async def _err_cb(exc: Exception) -> None:
                now = time.monotonic()
                if (now - self._last_nats_error_log_s) < 2.0:
                    return
                self._last_nats_error_log_s = now
                self._emit_log_line(f"NATS not reachable at {nats_url!r} (will retry): {type(exc).__name__}: {exc}")

            self._nc = await nats.connect(servers=[nats_url], connect_timeout=2, error_cb=_err_cb)
            try:
                await self._nc.request(f"$SRV.PING.{svc_micro_name(self.studio_service_id)}", b"", timeout=0.2)
                self._emit_log_line("Another PyStudio instance is already running (micro service ping responded).")
                await self._nc.close()
                self._nc = None
                return
            except Exception as exc:
                exc_name = type(exc).__name__
                if exc_name not in {"TimeoutError", "NoRespondersError"}:
                    self._report_exception("singleton ping failed", exc)
        except Exception as exc:
            self._report_exception("connect nats for singleton guard failed", exc)
            self._nc = None

        try:
            cfg = PyStudioServiceConfig(nats_url=nats_url, studio_service_id=self.studio_service_id)
            self._svc = PyStudioService(cfg, registry=RuntimeNodeRegistry.instance())
            await self._svc.start(
                on_ui_command=lambda cmd: self.ui_command.emit(cmd),
            )
        except Exception as exc:
            self._emit_log_line(f"studio runtime start failed: {exc}")
            self._svc = None

        # Studio-side remote KV watcher (monitors all remote node state and mirrors into UI).
        if self._remote_state_watcher is None:
            async def _on_state(
                service_id: str,
                node_id: str,
                field: str,
                value: Any,
                ts_ms: int,
                meta: dict[str, Any],
            ) -> None:
                _ = meta
                try:
                    self.ui_command.emit(
                        UiCommand(
                            node_id=str(node_id),
                            command="state.update",
                            payload={"serviceId": str(service_id), "field": str(field), "value": value},
                            ts_ms=int(ts_ms),
                        )
                    )
                except RuntimeError as exc:
                    self._report_exception("emit ui state.update failed", exc)

            try:
                self._remote_state_watcher = RemoteStateWatcher(
                    nats_url=nats_url,
                    studio_service_id=self.studio_service_id,
                    on_state=_on_state,
                )
                self._remote_state_gateway = RemoteStateGatewayAdapter(self._remote_state_watcher)
                await self._remote_state_gateway.start()
            except Exception as exc:
                self._report_exception("start remote state watcher failed", exc)
                self._remote_state_watcher = None
                self._remote_state_gateway = None

        # Re-apply current desired lifecycle to any already-known managed services.
        try:
            await self._set_managed_active_async(bool(self._managed_active))
        except Exception as exc:
            self._report_exception("re-apply managed active failed", exc)

    async def _stop_async(self) -> None:
        try:
            if self._remote_state_gateway is not None:
                await self._remote_state_gateway.stop()
            elif self._remote_state_watcher is not None:
                await self._remote_state_watcher.stop()
        except Exception as exc:
            self._report_exception("stop remote state watcher failed", exc)
        self._remote_state_gateway = None
        self._remote_state_watcher = None
        self._watch_targets_cache = None
        try:
            if self._svc is not None:
                await self._svc.stop()
        except Exception as exc:
            self._report_exception("stop studio service failed", exc)
        self._svc = None

        try:
            await self._command_gateway.close()
        except Exception as exc:
            self._report_exception("close command gateway failed", exc)

        try:
            if self._nc is not None:
                await self._nc.close()
        except Exception as exc:
            self._report_exception("close nats connection failed", exc)
        self._nc = None

    async def _deploy_and_monitor_async(self, compiled: CompiledRuntimeGraphs) -> None:
        # Deploy per-service rungraphs.
        for sid, g in compiled.per_service.items():
            service_id = ensure_token(str(sid), label="service_id")
            if str(service_id) == self.studio_service_id:
                continue
            try:
                result = await self._rungraph_gateway.deploy_runtime_graph(
                    RungraphDeployRequest(
                        service_id=service_id,
                        graph=g,
                        source="studio",
                    )
                )
                if not result.success:
                    raise RuntimeError(result.error_message or "set_rungraph rejected")
            except Exception as exc:
                self._emit_log_line(f"deploy failed serviceId={service_id}: {exc}")

        # Install studio runtime graph (studio operators + edges).
        if self._svc is None or self._svc.bus is None:
            return
        try:
            studio_graph = self._build_studio_runtime_graph(compiled)
            await self._svc.bus.set_rungraph(studio_graph)
        except Exception as exc:
            self._emit_log_line(f"install studio graph failed: {exc}")

        try:
            await self._apply_remote_state_watches_async(compiled)
        except Exception as exc:
            self._report_exception("apply remote state watches failed", exc)

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
                self._emit_log_line(f"NATS not reachable at {nats_url!r} (will retry): {type(exc).__name__}: {exc}")

            self._nc = await nats.connect(servers=[nats_url], connect_timeout=2, error_cb=_err_cb)
        except Exception as exc:
            self._report_exception("ensure nats connection failed", exc)
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
                    data = self._message_data_bytes(msg)
                    if data:
                        resp = self._decode_json_object(data) or {}
                        if isinstance(resp, dict) and resp.get("ok") is True:
                            ok = True
                            break
                except Exception:
                    await asyncio.sleep(0.2)
                    continue
            if not ok:
                self._emit_log_line(f"lifecycle {cmd} failed serviceId={sid}")
