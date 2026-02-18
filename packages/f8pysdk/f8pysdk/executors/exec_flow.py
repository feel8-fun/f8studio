from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ..generated import F8EdgeDirection, F8EdgeKindEnum, F8RuntimeGraph
from ..capabilities import BusAttachableNode, ComputableNode, EntrypointNode, ExecutableNode
from ..nats_naming import ensure_token
from ..service_bus.bus import ServiceBus
from ..time_utils import now_ms


def _entrypoint_node_id_or_raise(graph: F8RuntimeGraph, *, service_id: str) -> str | None:
    entrypoint_ids: list[str] = []
    for node in list(graph.nodes or []):
        if str(node.serviceId or "") != str(service_id):
            continue
        in_n = len(list(node.execInPorts or []))
        out_n = len(list(node.execOutPorts or []))
        if in_n == 0 and out_n > 0:
            entrypoint_ids.append(str(node.nodeId))
    if not entrypoint_ids:
        return None
    if len(entrypoint_ids) > 1:
        raise ValueError(f"graph has multiple exec entrypoints: {entrypoint_ids}")
    return entrypoint_ids[0]


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
    _entrypoint_node_id_or_raise(graph, service_id=sid)
    return out_map


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
        await self.executor.trigger_exec(self.node_id, out_port, exec_id=exec_id)

    async def cancel(self) -> None:
        for t in list(self._tasks):
            try:
                t.cancel()
            except Exception:
                pass
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
    - Exactly one entrypoint node is allowed per graph activation.
    - Exec ports are single-connection (UE-style): each exec in/out port can connect to at most 1 edge.
    - Scheduling is depth-first (LIFO stack) to keep branching order predictable (e.g., Sequence).
    """

    def __init__(self, bus: ServiceBus) -> None:
        self._bus = bus
        self._service_id = ensure_token(bus.service_id, label="service_id")
        self._active = True

        self._graph: F8RuntimeGraph | None = None
        self._nodes: dict[str, Any] = {}

        self._exec_out: dict[tuple[str, str], tuple[str, str]] = {}
        self._entrypoint_ctx: EntrypointContext | None = None
        self._half_out_ports: dict[str, set[str]] = {}

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
        - entrypoint is stopped (best-effort)
        - new exec triggers are ignored
        """
        active = bool(active)
        if active == self._active:
            return
        self._active = active
        if not active:
            await self.stop_entrypoint()
            return
        graph = self._graph
        if graph is not None:
            await self._restart_entrypoint_if_needed(graph)

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
        except Exception:
            return None
        return self._nodes.get(node_id)

    def current_entrypoint_node_id(self) -> str | None:
        return self._entrypoint_ctx.node_id if self._entrypoint_ctx else None

    # ---- rungraph -------------------------------------------------------
    async def apply_rungraph(self, graph: F8RuntimeGraph) -> None:
        self._graph = graph
        self._rebuild_exec_routes(graph)
        self._rebuild_half_out_ports(graph)
        if self._active:
            await self._restart_entrypoint_if_needed(graph)

    def _rebuild_half_out_ports(self, graph: F8RuntimeGraph) -> None:
        out: dict[str, set[str]] = {}
        for edge in graph.edges:
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
            except Exception:
                continue
            if v is None:
                continue
            try:
                await self._bus.emit_data(str(node_id), str(port), v, ts_ms=now_ms())
            except Exception:
                continue

    def _rebuild_exec_routes(self, graph: F8RuntimeGraph) -> None:
        self._exec_out = validate_exec_topology_or_raise(graph, service_id=self._service_id)

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

    async def _restart_entrypoint_if_needed(self, graph: F8RuntimeGraph) -> None:
        new_source = _entrypoint_node_id_or_raise(graph, service_id=self._service_id)
        cur = self._entrypoint_ctx.node_id if self._entrypoint_ctx else None
        if new_source == cur:
            return
        await self.stop_entrypoint()
        if new_source:
            await self.start_entrypoint(new_source)

    # ---- source lifecycle ----------------------------------------------
    async def start_entrypoint(self, node_id: str) -> None:
        node_id = ensure_token(node_id, label="node_id")
        if not self._active:
            return
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"entrypoint node not registered: {node_id}")
        if not isinstance(node, EntrypointNode):
            return
        ctx = EntrypointContext(executor=self, node_id=node_id)
        self._entrypoint_ctx = ctx
        try:
            await node.start_entrypoint(ctx)  # type: ignore[misc]
        except Exception:
            await ctx.cancel()
            self._entrypoint_ctx = None
            raise

    async def stop_entrypoint(self) -> None:
        ctx = self._entrypoint_ctx
        self._entrypoint_ctx = None
        if ctx is None:
            return
        try:
            node = self._nodes.get(ctx.node_id)
            if node is not None and isinstance(node, EntrypointNode):
                try:
                    await node.stop_entrypoint()  # type: ignore[misc]
                except Exception:
                    pass
        finally:
            await ctx.cancel()

    # ---- triggering -----------------------------------------------------
    async def trigger_exec(
        self,
        node_id: str,
        out_port: str,
        *,
        exec_id: str | int,
    ) -> None:
        """
        Inject an exec trigger from (node_id, out_port) and propagate intra-service exec edges.
        """
        # `exec_id` is a per-trigger execution id (used as an evaluation/cache key across a single propagation).

        if not self._active:
            return
        node_id = ensure_token(node_id, label="node_id")
        out_port = str(out_port)

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
            except Exception:
                continue

            # Tick-driven cross-service publishing for outgoing half-edges (direction=out).
            # This bridges pull-based compute into NATS data subjects so remote services can subscribe.
            await self._emit_half_edge_outputs(to_node, exec_id=exec_id)

            # DFS scheduling: push in reverse so earlier ports run first.
            for p in reversed(list(out_ports or [])):
                nxt = self._exec_out.get((to_node, str(p)))
                if nxt is not None:
                    stack.append(nxt)
