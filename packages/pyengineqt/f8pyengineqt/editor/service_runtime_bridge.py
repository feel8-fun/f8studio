from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

from qtpy import QtCore

from ..engine.nats_naming import ensure_token, kv_bucket_for_service
from ..runtime import ServiceRuntime, ServiceRuntimeConfig, ServiceRuntimeNode


class _BridgeRuntimeNode(ServiceRuntimeNode):
    def __init__(
        self,
        *,
        bridge: "ServiceRuntimeBridge",
        node_id: str,
        data_in_ports: list[str],
        data_out_ports: list[str],
        state_fields: list[str],
    ) -> None:
        super().__init__(node_id=node_id, data_in_ports=data_in_ports, data_out_ports=data_out_ports, state_fields=state_fields)
        self._bridge = bridge

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        try:
            ts = int(ts_ms) if ts_ms is not None else int(QtCore.QDateTime.currentMSecsSinceEpoch())
        except Exception:
            ts = int(QtCore.QDateTime.currentMSecsSinceEpoch())
        try:
            self._bridge.dataReceived.emit(str(self.node_id), str(port), value, ts)
        except Exception:
            return


@dataclass(frozen=True)
class ServiceRuntimeBridgeConfig:
    service_id: str
    nats_url: str
    kv_bucket: str | None = None
    actor_id: str | None = None


class ServiceRuntimeBridge(QtCore.QObject):
    """
    Qt-friendly bridge that hosts an asyncio `ServiceRuntime` in a background thread.

    Intended for the editor process (rungraph/state publisher + remote state consumer).
    """

    statusChanged = QtCore.Signal(str)
    # `ts_ms` is a millisecond epoch timestamp (~1e12), which overflows Qt's 32-bit `int`.
    stateUpdated = QtCore.Signal(str, str, object, object, object)  # node_id, field, value, ts_ms, meta
    dataReceived = QtCore.Signal(str, str, object, object)  # node_id, port, value, ts_ms
    dataPulled = QtCore.Signal(str, str, object, object)  # node_id, port, value, ts_ms

    def __init__(self, config: ServiceRuntimeBridgeConfig) -> None:
        super().__init__()
        self._config = config

        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

        self._runtime: ServiceRuntime | None = None
        self._pending_rungraph: dict[str, Any] | None = None
        self._pending_state: list[tuple[str, str, Any, str, int | None]] = []
        self._pending_pulls: set[tuple[str, str]] = set()
        self._known_nodes: set[str] = set()

    @property
    def service_id(self) -> str:
        return ensure_token(self._config.service_id, label="service_id")

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._thread_main, name="ServiceRuntimeBridge", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(lambda: None)
            except Exception:
                pass
        thread = self._thread
        self._thread = None
        if thread is not None:
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass

    def put_rungraph(self, payload: dict[str, Any]) -> None:
        rt = self._runtime
        loop = self._loop
        if rt is None or loop is None:
            self._pending_rungraph = payload
            return

        async def _do() -> None:
            await rt.set_rungraph(payload)

        try:
            asyncio.run_coroutine_threadsafe(_do(), loop)
        except Exception:
            pass

    def put_state(self, node_id: str, field: str, value: Any, *, source: str = "editor", ts_ms: int | None = None) -> None:
        rt = self._runtime
        loop = self._loop
        if rt is None or loop is None:
            self._pending_state.append((str(node_id), str(field), value, str(source), ts_ms))
            return

        async def _do() -> None:
            await rt.set_state_with_meta(node_id, field, value, ts_ms=ts_ms, source=source)

        try:
            asyncio.run_coroutine_threadsafe(_do(), loop)
        except Exception:
            pass

    def request_pull_data(self, node_id: str, port: str) -> None:
        """
        Request a single pull of buffered input data for (node_id, port).

        Result is delivered via `dataPulled` signal.
        """
        rt = self._runtime
        loop = self._loop
        node_id_s = str(node_id)
        port_s = str(port)
        key = (node_id_s, port_s)
        if rt is None or loop is None:
            return
        if key in self._pending_pulls:
            return
        self._pending_pulls.add(key)

        async def _do() -> None:
            try:
                v = await rt.pull_data(node_id_s, port_s)
            except Exception:
                v = None
            ts = int(QtCore.QDateTime.currentMSecsSinceEpoch())
            try:
                self.dataPulled.emit(node_id_s, port_s, v, ts)
            except Exception:
                pass
            self._pending_pulls.discard(key)

        try:
            asyncio.run_coroutine_threadsafe(_do(), loop)
        except Exception:
            self._pending_pulls.discard(key)

    # ---- internals -----------------------------------------------------
    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run())
        finally:
            try:
                loop.stop()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
            self._loop = None

    async def _run(self) -> None:
        service_id = ensure_token(self._config.service_id, label="service_id")
        bucket = (self._config.kv_bucket or "").strip() or kv_bucket_for_service(service_id)
        actor_id = (self._config.actor_id or "").strip() or None
        nats_url = str(self._config.nats_url).strip()

        self.statusChanged.emit(f"nats: connecting {nats_url} bucket={bucket}")
        rt = ServiceRuntime(ServiceRuntimeConfig(service_id=service_id, nats_url=nats_url, kv_bucket=bucket, actor_id=actor_id))
        self._runtime = rt

        async def _on_state(node_id: str, field: str, value: Any, ts_ms: int, meta: dict[str, Any]) -> None:
            self.stateUpdated.emit(str(node_id), str(field), value, ts_ms, meta)

        rt.add_state_listener(_on_state)

        async def _on_rungraph(graph: Any) -> None:
            # Register local nodes so cross half-edges can subscribe/buffer.
            try:
                nodes = getattr(graph, "nodes", None) or {}
                want = set(map(str, nodes.keys()))
            except Exception:
                want = set()

            for node_id in sorted(self._known_nodes - want):
                try:
                    rt.unregister_node(node_id)
                except Exception:
                    pass
                self._known_nodes.discard(node_id)

            for node_id in sorted(want - self._known_nodes):
                try:
                    inst = nodes.get(node_id)
                    spec = getattr(inst, "spec", None)
                    data_in = [p.name for p in (getattr(spec, "dataInPorts", None) or [])]
                    data_out = [p.name for p in (getattr(spec, "dataOutPorts", None) or [])]
                    state_fields = [s.name for s in (getattr(spec, "states", None) or [])]
                except Exception:
                    data_in, data_out, state_fields = [], [], []

                try:
                    rt.register_node(
                        _BridgeRuntimeNode(
                            bridge=self,
                            node_id=str(node_id),
                            data_in_ports=data_in,
                            data_out_ports=data_out,
                            state_fields=state_fields,
                        )
                    )
                    self._known_nodes.add(str(node_id))
                except Exception:
                    continue

        rt.add_rungraph_listener(_on_rungraph)

        try:
            await rt.start()
        except Exception as exc:
            self.statusChanged.emit(f"nats: start failed ({exc})")
            self._runtime = None
            return

        self.statusChanged.emit(f"nats: ready serviceId={service_id} bucket={bucket}")

        pending_rungraph = self._pending_rungraph
        self._pending_rungraph = None
        if pending_rungraph is not None:
            try:
                await rt.set_rungraph(pending_rungraph)
            except Exception:
                pass

        pending_state = list(self._pending_state)
        self._pending_state.clear()
        for node_id, field, value, source, ts_ms in pending_state:
            try:
                await rt.set_state_with_meta(node_id, field, value, ts_ms=ts_ms, source=source)
            except Exception:
                continue

        try:
            while not self._stop_evt.is_set():
                await asyncio.sleep(0.2)
        finally:
            try:
                await rt.stop()
            except Exception:
                pass
            self._known_nodes.clear()
            self._runtime = None
            self.statusChanged.emit("nats: stopped")
