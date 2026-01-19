from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from typing import Any, Callable

import nats  # type: ignore[import-not-found]
from nats.micro import ServiceConfig, add_service  # type: ignore[import-not-found]
from qtpy import QtCore

from f8pysdk import F8Edge, F8EdgeKindEnum, F8EdgeStrategyEnum, F8RuntimeGraph, F8RuntimeNode
from f8pysdk.nats_naming import ensure_token, kv_bucket_for_service, kv_key_rungraph
from f8pysdk.nats_transport import NatsTransport, NatsTransportConfig
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.service_bus import ServiceBus, ServiceBusConfig
from .nodegraph.runtime_compiler import CompiledRuntimeGraphs
from .service_process_manager import ServiceProcessConfig, ServiceProcessManager
from .service_host.service_host_registry import SERVICE_CLASS, STUDIO_SERVICE_ID


_STUDIO_MICRO_SERVICE_NAME = "f8.pystudio"
_STUDIO_MICRO_SERVICE_VERSION = "0.0.1"
_MONITOR_NODE_ID = "monitor"


def _encode_remote_state_key(*, service_id: str, node_id: str, field: str) -> str:
    # Keep this stable and human-readable; avoid "." (it can be confusing inside KV keys).
    return f"{service_id}|{node_id}|{field}"


def _decode_remote_state_key(encoded: str) -> tuple[str, str, str] | None:
    try:
        sid, nid, field = str(encoded).split("|", 2)
    except Exception:
        return None
    sid = sid.strip()
    nid = nid.strip()
    field = field.strip()
    if not sid or not nid or not field:
        return None
    return sid, nid, field


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


class _StateMonitorNode(RuntimeNode):
    def __init__(self, *, on_state_cb: Callable[[str, Any, int | None], None]) -> None:
        super().__init__(node_id=_MONITOR_NODE_ID)
        self._on_state_cb = on_state_cb

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        try:
            self._on_state_cb(str(field), value, ts_ms)
        except Exception:
            return


@dataclass(frozen=True)
class StudioRuntimeConfig:
    nats_url: str = "nats://127.0.0.1:4222"
    studio_service_id: str = STUDIO_SERVICE_ID


class StudioRuntime(QtCore.QObject):
    """
    Orchestrate:
    - singleton studio presence (NATS micro ping/info)
    - start service processes
    - deploy per-service rungraphs
    - monitor remote state via cross-state edges into a local monitor node
    """

    # Note: Qt `int` is typically 32-bit; use `object` for ts_ms (ms timestamps exceed 2^31).
    state_updated = QtCore.Signal(str, str, str, object, object)  # serviceId, nodeId, field, value, ts_ms
    data_updated = QtCore.Signal(str, str, object, object)  # nodeId, port, value, ts_ms
    service_output = QtCore.Signal(str, str)  # serviceId, line
    log = QtCore.Signal(str)

    def __init__(self, config: StudioRuntimeConfig, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._cfg = config
        self._async = _AsyncThread()
        self._proc_mgr = ServiceProcessManager()

        self._bus: ServiceBus | None = None
        self._nc: Any = None
        self._micro: Any = None
        self._monitor_node: _StateMonitorNode | None = None

    @property
    def studio_service_id(self) -> str:
        return ensure_token(self._cfg.studio_service_id, label="studio_service_id")

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
        for svc in list(compiled.global_graph.services or []):
            try:
                sid = ensure_token(str(svc.serviceId), label="service_id")
                if sid == self.studio_service_id or str(svc.serviceClass) == SERVICE_CLASS:
                    continue
                self._proc_mgr.start(
                    ServiceProcessConfig(service_class=str(svc.serviceClass), service_id=sid, nats_url=self._cfg.nats_url),
                    on_output=lambda _sid, line, _sid2=sid: self.service_output.emit(_sid2, str(line)),
                )
            except Exception as exc:
                self.log.emit(f"start service failed: {exc}")

        # 2) deploy + install monitoring (async)
        self._async.submit(self._deploy_and_monitor_async(compiled))

    def set_local_state(self, node_id: str, field: str, value: Any) -> None:
        """
        Set state in the local studio service KV (best-effort).
        """
        node_id = ensure_token(node_id, label="node_id")
        field = str(field or "").strip()
        if not field:
            return

        async def _do() -> None:
            if self._bus is None:
                return
            try:
                await self._bus.set_state(node_id, field, value)
            except Exception:
                return

        try:
            self._async.submit(_do())
        except Exception:
            return

    async def _start_async(self) -> None:
        nats_url = str(self._cfg.nats_url).strip() or "nats://127.0.0.1:4222"

        # Singleton guard (best-effort): if any existing studio micro service responds, do not start.
        try:
            self._nc = await nats.connect(servers=[nats_url], connect_timeout=2)
            try:
                await self._nc.request(f"$SRV.PING.{_STUDIO_MICRO_SERVICE_NAME}", b"", timeout=0.2)
                self.log.emit("Another PyStudio instance is already running (micro service ping responded).")
                await self._nc.close()
                self._nc = None
                return
            except Exception:
                pass

            try:
                self._micro = await add_service(
                    self._nc,
                    ServiceConfig(name=_STUDIO_MICRO_SERVICE_NAME, version=_STUDIO_MICRO_SERVICE_VERSION),
                )
            except Exception:
                self._micro = None
        except Exception:
            self._nc = None

        # Start studio bus.
        self._bus = ServiceBus(
            ServiceBusConfig(
                service_id=self.studio_service_id,
                nats_url=nats_url,
                publish_all_data=False,
            )
        )

        def _on_state(field: str, value: Any, ts_ms: int | None) -> None:
            decoded = _decode_remote_state_key(field)
            if decoded is None:
                return
            sid, nid, f = decoded
            try:
                self.state_updated.emit(str(sid), str(nid), str(f), value, ts_ms)
            except Exception:
                return

        async def _on_local_state(node_id: str, field: str, value: Any, ts: int, _meta: dict[str, Any]) -> None:
            if str(node_id) == _MONITOR_NODE_ID:
                _on_state(field, value, ts)
                return
            try:
                self.state_updated.emit(self.studio_service_id, str(node_id), str(field), value, ts)
            except Exception:
                return

        self._bus.add_state_listener(_on_local_state)
        self._monitor_node = _StateMonitorNode(on_state_cb=lambda *_: None)
        self._bus.register_node(self._monitor_node)

        try:
            await self._bus.start()
        except Exception as exc:
            self.log.emit(f"studio bus start failed: {exc}")

    async def _stop_async(self) -> None:
        try:
            if self._bus is not None:
                await self._bus.stop()
        except Exception:
            pass
        self._bus = None

        try:
            if self._micro is not None:
                await self._micro.stop()
        except Exception:
            pass
        self._micro = None

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
                payload = g.model_dump(mode="json", by_alias=True)
                await tr.kv_put(kv_key_rungraph(), json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
            except Exception as exc:
                self.log.emit(f"deploy failed serviceId={service_id}: {exc}")
            finally:
                try:
                    await tr.close()
                except Exception:
                    pass

        # Install monitoring graph into studio serviceId.
        if self._bus is None:
            return

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

            for sf in list(getattr(n, "stateFields", None) or []):
                try:
                    field_name = str(getattr(sf, "name", "") or "").strip()
                except Exception:
                    field_name = ""
                if not field_name:
                    continue
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

        try:
            studio_sub = compiled.per_service.get(self.studio_service_id)
        except Exception:
            studio_sub = None
        base_nodes: list[F8RuntimeNode] = []
        base_edges: list[F8Edge] = []
        if studio_sub is not None:
            base_nodes = list(getattr(studio_sub, "nodes", None) or [])
            base_edges = list(getattr(studio_sub, "edges", None) or [])

        studio_graph = F8RuntimeGraph(
            graphId=str(getattr(compiled.global_graph, "graphId", "") or "studio_monitor"),
            revision=str(getattr(compiled.global_graph, "revision", "") or "1"),
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

        try:
            await self._bus.set_rungraph(studio_graph)
        except Exception as exc:
            self.log.emit(f"install monitor graph failed: {exc}")

    def request_pull(self, node_id: str, port: str) -> None:
        """
        Request the latest buffered data for a node input port (best-effort).
        """
        node_id = ensure_token(node_id, label="node_id")
        port = str(port or "").strip()
        if not port:
            return

        async def _do() -> None:
            if self._bus is None:
                return
            try:
                v = await self._bus.pull_data(node_id, port)
            except Exception:
                return
            try:
                self.data_updated.emit(node_id, port, v, None)
            except Exception:
                return

        try:
            self._async.submit(_do())
        except Exception:
            return
