from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from ..bus import ServiceBus
from ..capabilities import BusAttachableNode, ComputableNode, EntrypointNode, ExecutableNode
from ..generated import F8EdgeDirection, F8EdgeKindEnum, F8RuntimeGraph
from ..f8_naming import ensure_token
from ..time_utils import now_ms

logger = logging.getLogger(__name__)

_ENTRYPOINT_CANCEL_ERRORS = (RuntimeError, TypeError, ValueError)
_ENTRYPOINT_LIFECYCLE_ERRORS = (RuntimeError, TypeError, ValueError)
_EXEC_CALLBACK_ERRORS = (LookupError, OSError, RuntimeError, TypeError, ValueError)
_EXEC_FLOW_RECOVERY_ERRORS = (RuntimeError, TypeError, ValueError)
_HALF_EDGE_ERRORS = (LookupError, OSError, RuntimeError, TypeError, ValueError)
_NODE_LOOKUP_ERRORS = (TypeError, ValueError)
_TRIGGER_WORKER_STOP_ERRORS = (RuntimeError, TypeError, ValueError)


def _entrypoint_node_ids(graph: F8RuntimeGraph, *, service_id: str) -> tuple[str, ...]:
    entrypoint_ids: list[str] = []
    for node in list(graph.nodes or []):
        if str(node.serviceId or "") != str(service_id):
            continue
        in_n = len(list(node.execInPorts or []))
        out_n = len(list(node.execOutPorts or []))
        if in_n == 0 and out_n > 0:
            entrypoint_ids.append(str(node.nodeId))
    return tuple(sorted(entrypoint_ids))


def validate_exec_topology_or_raise(
    graph: F8RuntimeGraph,
    *,
    service_id: str,
) -> dict[tuple[str, str], tuple[str, str]]:
    """
    Validate exec topology for one service and return the exec route map.
    """
    sid = ensure_token(service_id, label="service_id")
    out_map: dict[tuple[str, str], tuple[str, str]] = {}
    in_seen: set[tuple[str, str]] = set()
    adj: dict[str, set[str]] = {}
    for edge in list(graph.edges or []):
        if edge.kind != F8EdgeKindEnum.exec:
            continue
        if str(edge.fromServiceId or "") != sid or str(edge.toServiceId or "") != sid:
            continue
        if not edge.fromOperatorId or not edge.toOperatorId:
            continue
        from_key = (str(edge.fromOperatorId), str(edge.fromPort))
        to_val = (str(edge.toOperatorId), str(edge.toPort))
        to_key = (to_val[0], to_val[1])

        if from_key in out_map:
            raise ValueError(f"exec out port must be single-connected: {from_key} (edgeId={edge.edgeId})")
        if to_key in in_seen:
            raise ValueError(f"exec in port must be single-connected: {to_key} (edgeId={edge.edgeId})")

        out_map[from_key] = to_val
        in_seen.add(to_key)
        adj.setdefault(from_key[0], set()).add(to_key[0])

    ExecFlowExecutor._ensure_exec_acyclic(adj)
    for node_id in _entrypoint_node_ids(graph, service_id=sid):
        ensure_token(node_id, label="node_id")
    return out_map


@dataclass(frozen=True)
class _ExecTrigger:
    seq: int
    node_id: str
    out_port: str
    exec_id: str | int
    done: asyncio.Future[None] | None


@dataclass
class EntrypointContext:
    """
    Engine-managed context for an entrypoint node (timer/event based).

    An entrypoint node uses this to:
    - spawn cancellable tasks
    - emit exec triggers into the executor
    """

    executor: "ExecFlowExecutor"
    node_id: str
    _tasks: set[asyncio.Task[object]] = field(default_factory=set, init=False, repr=False)

    def create_task(self, coro: Any, *, name: str | None = None) -> asyncio.Task[object]:
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(lambda t: self._tasks.discard(t))
        return task

    async def emit_exec(self, out_port: str, *, exec_id: str | int) -> None:
        """
        Emit an exec trigger out of this entrypoint node.

        `exec_id` is a per-trigger execution id (used as an evaluation/cache key across a single propagation).
        """
        await self.executor.trigger_exec_nowait(self.node_id, out_port, exec_id=exec_id)

    async def cancel(self) -> None:
        for t in list(self._tasks):
            try:
                t.cancel()
            except _ENTRYPOINT_CANCEL_ERRORS as exc:
                logger.exception(
                    "entrypoint task cancel failed service=%s node=%s task=%s",
                    self.executor.service_id,
                    self.node_id,
                    t.get_name(),
                    exc_info=exc,
                )
        if not self._tasks:
            return
        await asyncio.gather(*list(self._tasks), return_exceptions=True)
        self._tasks.clear()


class ExecFlowExecutor:
    """
    In-process executor for exec edges.

    Constraints:
    - Exec edges are strictly intra-process: only when `fromServiceId == toServiceId == bus.service_id`.
    - Cross-process "triggering" must be modeled via data/state edges and handled in nodes.
    - Multiple entrypoint nodes are allowed per graph activation.
    - Exec ports are single-connection (UE-style): each exec in/out port can connect to at most 1 edge.
    - Trigger delivery is serialized through a global FIFO queue per service.
    - Scheduling is depth-first (LIFO stack) to keep branching order predictable (e.g., Sequence).
    """

    def __init__(self, bus: ServiceBus) -> None:
        self._bus = bus
        self._service_id = ensure_token(bus.service_id, label="service_id")
        self._active = True

        self._graph: F8RuntimeGraph | None = None
        self._nodes: dict[str, Any] = {}

        self._exec_out: dict[tuple[str, str], tuple[str, str]] = {}
        self._entrypoint_ctx_by_node_id: dict[str, EntrypointContext] = {}
        self._half_out_ports: dict[str, set[str]] = {}
        self._trigger_queue: asyncio.Queue[_ExecTrigger] = asyncio.Queue()
        self._trigger_worker_task: asyncio.Task[None] | None = None
        self._trigger_seq = 0

    @property
    def service_id(self) -> str:
        return self._service_id

    @property
    def active(self) -> bool:
        return bool(self._active)

    async def set_active(self, active: bool) -> None:
        """
        Activate/deactivate exec processing.

        When inactive:
        - entrypoints are stopped (best-effort)
        - new exec triggers are ignored
        """
        active = bool(active)
        if active == self._active:
            return
        self._active = active
        if not active:
            await self._stop_trigger_worker()
            await self.stop_all_entrypoints()
            self._drain_trigger_queue()
            return
        graph = self._graph
        if graph is not None:
            try:
                await self._start_trigger_worker()
                await self._start_entrypoints_for_graph(graph)
            except _EXEC_FLOW_RECOVERY_ERRORS:
                await self.stop_all_entrypoints()
                await self._stop_trigger_worker()
                self._drain_trigger_queue()
                raise

    # ---- node registry --------------------------------------------------
    def register_node(self, node: BusAttachableNode) -> None:
        node_id = ensure_token(str(node.node_id), label="node_id")
        self._nodes[node_id] = node

    def unregister_node(self, node_id: str) -> None:
        node_id = ensure_token(node_id, label="node_id")
        self._nodes.pop(node_id, None)

    def get_registered_node(self, node_id: str) -> Any | None:
        try:
            node_id = ensure_token(node_id, label="node_id")
        except _NODE_LOOKUP_ERRORS:
            return None
        return self._nodes.get(node_id)

    def current_entrypoint_node_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entrypoint_ctx_by_node_id.keys()))

    # ---- rungraph -------------------------------------------------------
    async def apply_rungraph(self, graph: F8RuntimeGraph) -> None:
        self._graph = graph
        await self._stop_trigger_worker()
        await self.stop_all_entrypoints()
        self._drain_trigger_queue()
        self._rebuild_exec_routes(graph)
        self._rebuild_half_out_ports(graph)
        if self._active:
            try:
                await self._start_trigger_worker()
                await self._start_entrypoints_for_graph(graph)
            except _EXEC_FLOW_RECOVERY_ERRORS:
                await self.stop_all_entrypoints()
                await self._stop_trigger_worker()
                self._drain_trigger_queue()
                raise

    def _rebuild_half_out_ports(self, graph: F8RuntimeGraph) -> None:
        out: dict[str, set[str]] = {}
        for edge in graph.edges if isinstance(graph.edges, list) else []:
            if edge.kind != F8EdgeKindEnum.data:
                continue
            if edge.direction != F8EdgeDirection.out:
                continue
            if str(edge.fromServiceId) != self._service_id:
                continue
            if not edge.fromOperatorId:
                continue
            out.setdefault(str(edge.fromOperatorId), set()).add(str(edge.fromPort))
        self._half_out_ports = out

    async def _emit_half_edge_outputs(self, node_id: str, *, exec_id: str | int) -> None:
        if not self._active:
            return
        if not self._bus.active:
            return
        ports = self._half_out_ports.get(str(node_id)) or set()
        if not ports:
            return
        node = self._nodes.get(str(node_id))
        if node is None or not isinstance(node, ComputableNode):
            return

        for port in sorted(ports):
            try:
                v = await node.compute_output(str(port), ctx_id=exec_id)  # type: ignore[misc]
            except _HALF_EDGE_ERRORS as exc:
                self._report_exec_error(
                    str(node_id),
                    "EXEC_HALF_EDGE_COMPUTE",
                    f"half-edge compute failed node={node_id} port={port} exec_id={exec_id}: {exc}",
                    exc,
                    fingerprint=f"exec_flow:half_compute:{node_id}:{port}:{type(exc).__name__}:{exc}",
                )
                continue
            if v is None:
                continue
            try:
                await self._bus.emit_data(str(node_id), str(port), v, ts_ms=now_ms())
            except _HALF_EDGE_ERRORS as exc:
                self._report_exec_error(
                    str(node_id),
                    "EXEC_HALF_EDGE_EMIT",
                    f"half-edge emit failed node={node_id} port={port} exec_id={exec_id}: {exc}",
                    exc,
                    fingerprint=f"exec_flow:half_emit:{node_id}:{port}:{type(exc).__name__}:{exc}",
                )
                continue

    def _rebuild_exec_routes(self, graph: F8RuntimeGraph) -> None:
        self._exec_out = validate_exec_topology_or_raise(graph, service_id=self._service_id)

    def _report_exec_error(
        self,
        node_id: str,
        code: str,
        message: str,
        exc: BaseException,
        *,
        fingerprint: str,
    ) -> None:
        logger.exception(message, exc_info=exc)
        try:
            self._bus.report_error(
                str(node_id),
                code,
                message,
                severity="error",
                fingerprint=fingerprint,
            )
        except _EXEC_FLOW_RECOVERY_ERRORS as report_exc:
            logger.exception(
                "exec flow monitor error report failed service=%s node=%s code=%s",
                self._service_id,
                node_id,
                code,
                exc_info=report_exc,
            )

    @staticmethod
    def _ensure_exec_acyclic(adj: dict[str, set[str]]) -> None:
        """
        Ensure the exec topology is acyclic (UE-style), so propagation always terminates.
        """

        visiting: set[str] = set()
        visited: set[str] = set()
        stack: list[str] = []

        def _visit(n: str) -> None:
            if n in visited:
                return
            if n in visiting:
                try:
                    i = stack.index(n)
                except ValueError:
                    i = 0
                cycle = stack[i:] + [n]
                raise ValueError(f"exec graph has a cycle: {' -> '.join(cycle)}")
            visiting.add(n)
            stack.append(n)
            for m in sorted(adj.get(n, set())):
                _visit(m)
            stack.pop()
            visiting.remove(n)
            visited.add(n)

        for n in sorted(adj.keys()):
            _visit(n)

    async def _start_entrypoints_for_graph(self, graph: F8RuntimeGraph) -> None:
        entrypoint_ids = _entrypoint_node_ids(graph, service_id=self._service_id)
        if not entrypoint_ids:
            return
        for node_id in entrypoint_ids:
            node = self._nodes.get(node_id)
            if node is None:
                raise KeyError(f"entrypoint node not registered: {node_id}")
            if not isinstance(node, EntrypointNode):
                raise ValueError(f"exec entrypoint node must implement EntrypointNode: {node_id}")
        for node_id in entrypoint_ids:
            await self.start_entrypoint(node_id)

    # ---- source lifecycle ----------------------------------------------
    async def start_entrypoint(self, node_id: str) -> None:
        node_id = ensure_token(node_id, label="node_id")
        if not self._active:
            return
        if node_id in self._entrypoint_ctx_by_node_id:
            return
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"entrypoint node not registered: {node_id}")
        if not isinstance(node, EntrypointNode):
            raise ValueError(f"exec entrypoint node must implement EntrypointNode: {node_id}")
        ctx = EntrypointContext(executor=self, node_id=node_id)
        self._entrypoint_ctx_by_node_id[node_id] = ctx
        try:
            await node.start_entrypoint(ctx)  # type: ignore[misc]
        except _ENTRYPOINT_LIFECYCLE_ERRORS:
            await ctx.cancel()
            self._entrypoint_ctx_by_node_id.pop(node_id, None)
            raise

    async def stop_all_entrypoints(self) -> None:
        for node_id in self.current_entrypoint_node_ids():
            await self._stop_entrypoint(node_id)

    async def _stop_entrypoint(self, node_id: str) -> None:
        ctx = self._entrypoint_ctx_by_node_id.pop(node_id, None)
        if ctx is None:
            return
        try:
            node = self._nodes.get(ctx.node_id)
            if node is not None and isinstance(node, EntrypointNode):
                try:
                    await node.stop_entrypoint()  # type: ignore[misc]
                except _ENTRYPOINT_LIFECYCLE_ERRORS as exc:
                    logger.exception("stop_entrypoint failed for node=%s", node_id, exc_info=exc)
        finally:
            await ctx.cancel()

    # ---- trigger worker -------------------------------------------------
    async def _start_trigger_worker(self) -> None:
        task = self._trigger_worker_task
        if task is not None and not task.done():
            return
        self._trigger_worker_task = asyncio.create_task(
            self._trigger_worker_loop(),
            name=f"exec_flow:trigger_worker:{self._service_id}",
        )

    async def _stop_trigger_worker(self) -> None:
        task = self._trigger_worker_task
        self._trigger_worker_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
        except _TRIGGER_WORKER_STOP_ERRORS as exc:
            logger.exception("trigger worker stop failed for service=%s", self._service_id, exc_info=exc)

    def _drain_trigger_queue(self) -> None:
        while True:
            try:
                trigger = self._trigger_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if trigger.done is not None and not trigger.done.done():
                trigger.done.set_result(None)
            self._trigger_queue.task_done()

    async def _trigger_worker_loop(self) -> None:
        while True:
            trigger = await self._trigger_queue.get()
            try:
                if self._active and self._bus.active:
                    await self._propagate_exec_dfs(
                        trigger.node_id,
                        trigger.out_port,
                        exec_id=trigger.exec_id,
                    )
            finally:
                if trigger.done is not None and not trigger.done.done():
                    trigger.done.set_result(None)
                self._trigger_queue.task_done()

    async def trigger_exec_nowait(
        self,
        node_id: str,
        out_port: str,
        *,
        exec_id: str | int,
    ) -> None:
        if not self._active:
            return
        if (node_id, out_port) not in self._exec_out:
            return
        if self._trigger_worker_task is None or self._trigger_worker_task.done():
            await self._start_trigger_worker()
        self._trigger_seq += 1
        trigger = _ExecTrigger(
            seq=int(self._trigger_seq),
            node_id=node_id,
            out_port=out_port,
            exec_id=exec_id,
            done=None,
        )
        await self._trigger_queue.put(trigger)

    # ---- triggering -----------------------------------------------------
    async def trigger_exec(
        self,
        node_id: str,
        out_port: str,
        *,
        exec_id: str | int,
    ) -> None:
        """
        Enqueue an exec trigger from (node_id, out_port) for serialized propagation.
        """
        # `exec_id` is a per-trigger execution id (used as an evaluation/cache key across a single propagation).

        if not self._active:
            return
        node_id = ensure_token(node_id, label="node_id")
        out_port = str(out_port)
        if (node_id, out_port) not in self._exec_out:
            return
        if self._trigger_worker_task is None or self._trigger_worker_task.done():
            await self._start_trigger_worker()
        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()
        self._trigger_seq += 1
        trigger = _ExecTrigger(
            seq=int(self._trigger_seq),
            node_id=node_id,
            out_port=out_port,
            exec_id=exec_id,
            done=done,
        )
        await self._trigger_queue.put(trigger)
        await done

    async def _propagate_exec_dfs(
        self,
        node_id: str,
        out_port: str,
        *,
        exec_id: str | int,
    ) -> None:
        stack: list[tuple[str, str]] = []
        nxt = self._exec_out.get((node_id, out_port))
        if nxt is not None:
            stack.append(nxt)

        while stack:
            to_node, in_port = stack.pop()
            node = self._nodes.get(to_node)
            if node is None:
                continue
            if not isinstance(node, ExecutableNode):
                continue
            try:
                out_ports = await node.on_exec(exec_id, str(in_port))  # type: ignore[misc]
            except _EXEC_CALLBACK_ERRORS as exc:
                self._report_exec_error(
                    str(to_node),
                    "EXEC_NODE_CALLBACK",
                    f"exec node callback failed node={to_node} in_port={in_port} exec_id={exec_id}: {exc}",
                    exc,
                    fingerprint=f"exec_flow:on_exec:{to_node}:{in_port}:{type(exc).__name__}:{exc}",
                )
                continue

            # Tick-driven cross-service publishing for outgoing half-edges (direction=out).
            # This bridges pull-based compute into runtime data subjects so remote services can subscribe.
            await self._emit_half_edge_outputs(to_node, exec_id=exec_id)

            # DFS scheduling: push in reverse so earlier ports run first.
            if isinstance(out_ports, (list, tuple)):
                route_ports = out_ports
            elif out_ports is None:
                route_ports = ()
            else:
                route_ports = tuple(out_ports)
            for p in reversed(route_ports):
                nxt = self._exec_out.get((to_node, str(p)))
                if nxt is not None:
                    stack.append(nxt)
