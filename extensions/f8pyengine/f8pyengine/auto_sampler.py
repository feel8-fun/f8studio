from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from f8pysdk.capabilities import ComputableNode
from f8pysdk.codec import coerce_flag, coerce_int
from f8pysdk.f8_naming import ensure_token
from f8pysdk.specs import F8AutoSampleRequest, F8RuntimeGraph
from f8pysdk.service_bus.data.emit import DataEmitOptions
from f8pysdk.service_bus.data.flow import emit_data as emit_data_with_options

from .constants import SERVICE_CLASS

if TYPE_CHECKING:
    from f8pysdk.bus import ServiceBus


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AutoSampleConfig:
    source_node_id: str
    source_port: str
    interval_ms: int
    deliver_local: bool
    publish_cross_service: bool


class AutoSamplerManager:
    """
    Service-level periodic sampler for runtime graph auto-sampling requests.

    This keeps the sampling implementation inside pyengine while the compiler
    only provides typed sampling intent via the rungraph.
    """

    def __init__(self, bus: "ServiceBus") -> None:
        self._bus = bus
        self._active = bool(bus.active)
        self._configs: dict[tuple[str, str], _AutoSampleConfig] = {}
        self._tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._last_log_ms_by_sig: dict[str, int] = {}

    async def sync_rungraph(self, graph: F8RuntimeGraph) -> None:
        desired = self._configs_from_graph(graph)
        current = dict(self._configs)

        keys_to_stop = [key for key, cfg in current.items() if desired.get(key) != cfg]
        for key in keys_to_stop:
            await self._stop_task(key)

        self._configs = desired
        if not self._active:
            return
        for key in list(self._configs.keys()):
            self._start_task(key)

    async def set_active(self, active: bool) -> None:
        self._active = bool(active)
        if not self._active:
            await self._stop_all_tasks()
            return
        for key in list(self._configs.keys()):
            self._start_task(key)

    async def close(self) -> None:
        self._configs = {}
        await self._stop_all_tasks()

    def _configs_from_graph(self, graph: F8RuntimeGraph) -> dict[tuple[str, str], _AutoSampleConfig]:
        service_id = str(self._bus.service_id or "").strip()
        runtime_service = None
        for service in list(graph.services or []):
            if str(service.serviceId or "").strip() == service_id:
                runtime_service = service
                break
        if runtime_service is None:
            return {}
        if str(runtime_service.serviceClass or "").strip() != SERVICE_CLASS:
            return {}

        configs: dict[tuple[str, str], _AutoSampleConfig] = {}
        for request in list(runtime_service.autoSampleRequests or []):
            cfg = self._normalize_request(request)
            if cfg is None:
                continue
            key = (cfg.source_node_id, cfg.source_port)
            existing = configs.get(key)
            if existing is None:
                configs[key] = cfg
                continue
            configs[key] = _AutoSampleConfig(
                source_node_id=cfg.source_node_id,
                source_port=cfg.source_port,
                interval_ms=min(int(existing.interval_ms), int(cfg.interval_ms)),
                deliver_local=bool(existing.deliver_local or cfg.deliver_local),
                publish_cross_service=bool(existing.publish_cross_service or cfg.publish_cross_service),
            )
        return configs

    def _normalize_request(self, request: F8AutoSampleRequest) -> _AutoSampleConfig | None:
        try:
            source_node_id = ensure_token(str(request.sourceNodeId or "").strip(), label="sourceNodeId")
            source_port = ensure_token(str(request.sourcePort or "").strip(), label="sourcePort")
        except ValueError as exc:
            now_ms = int(time.time() * 1000.0)
            sig = f"invalid_request:{type(exc).__name__}:{exc}"
            if self._should_log(sig, now_ms=now_ms):
                logger.warning(
                    "pyengine auto sampler skipped invalid request serviceId=%s error=%s",
                    self._bus.service_id,
                    exc,
                )
            return None

        interval_ms = coerce_int(request.intervalMs, default=100, minimum=8, maximum=5000)
        deliver_local = coerce_flag(request.deliverLocal, default=False)
        publish_cross_service = coerce_flag(request.publishCrossService, default=True)
        if not deliver_local and not publish_cross_service:
            return None
        return _AutoSampleConfig(
            source_node_id=source_node_id,
            source_port=source_port,
            interval_ms=interval_ms,
            deliver_local=deliver_local,
            publish_cross_service=publish_cross_service,
        )

    def _start_task(self, key: tuple[str, str]) -> None:
        if not self._active:
            return
        task = self._tasks.get(key)
        if task is not None and not task.done():
            return
        cfg = self._configs.get(key)
        if cfg is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.exception(
                "pyengine auto sampler failed to start task serviceId=%s source=%s.%s",
                self._bus.service_id,
                cfg.source_node_id,
                cfg.source_port,
            )
            return
        self._tasks[key] = loop.create_task(
            self._run_config(key, cfg),
            name=f"pyengine:auto_sample:{cfg.source_node_id}:{cfg.source_port}",
        )

    async def _stop_task(self, key: tuple[str, str]) -> None:
        task = self._tasks.pop(key, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _stop_all_tasks(self) -> None:
        for key in list(self._tasks.keys()):
            await self._stop_task(key)

    async def _run_config(self, key: tuple[str, str], cfg: _AutoSampleConfig) -> None:
        while self._active and self._configs.get(key) == cfg:
            await self._sample_once(cfg)
            await asyncio.sleep(float(cfg.interval_ms) / 1000.0)

    async def _sample_once(self, cfg: _AutoSampleConfig) -> None:
        node = self._bus.get_node(cfg.source_node_id)
        now_ms = int(time.time() * 1000.0)
        if node is None:
            sig = f"node_missing:{cfg.source_node_id}.{cfg.source_port}"
            if self._should_log(sig, now_ms=now_ms):
                logger.warning(
                    "pyengine auto sampler source node missing serviceId=%s source=%s.%s",
                    self._bus.service_id,
                    cfg.source_node_id,
                    cfg.source_port,
                )
            return
        if not isinstance(node, ComputableNode):
            sig = f"not_computable:{cfg.source_node_id}.{cfg.source_port}"
            if self._should_log(sig, now_ms=now_ms):
                logger.warning(
                    "pyengine auto sampler source is not computable serviceId=%s source=%s.%s",
                    self._bus.service_id,
                    cfg.source_node_id,
                    cfg.source_port,
                )
            return

        exec_id = int(time.time() * 1000.0)
        try:
            value = await node.compute_output(cfg.source_port, ctx_id=exec_id)
        except Exception as exc:
            sig = f"compute_failed:{type(exc).__name__}:{exc}:{cfg.source_node_id}.{cfg.source_port}"
            if self._should_log(sig, now_ms=exec_id):
                logger.exception(
                    "pyengine auto sampler compute failed serviceId=%s source=%s.%s",
                    self._bus.service_id,
                    cfg.source_node_id,
                    cfg.source_port,
                )
            return
        if value is None:
            return

        try:
            await emit_data_with_options(
                cast(Any, self._bus),
                cfg.source_node_id,
                cfg.source_port,
                value,
                ts_ms=exec_id,
                options=DataEmitOptions(
                    deliver_local=cfg.deliver_local,
                    publish_cross_service=cfg.publish_cross_service,
                ),
            )
        except Exception as exc:
            sig = f"emit_failed:{type(exc).__name__}:{exc}:{cfg.source_node_id}.{cfg.source_port}"
            if self._should_log(sig, now_ms=exec_id):
                logger.exception(
                    "pyengine auto sampler emit failed serviceId=%s source=%s.%s",
                    self._bus.service_id,
                    cfg.source_node_id,
                    cfg.source_port,
                )

    def _should_log(self, sig: str, *, now_ms: int, interval_ms: int = 2000) -> bool:
        last_ms = int(self._last_log_ms_by_sig.get(sig, 0))
        if last_ms != 0 and (int(now_ms) - last_ms) < int(interval_ms):
            return False
        self._last_log_ms_by_sig[sig] = int(now_ms)
        return True
