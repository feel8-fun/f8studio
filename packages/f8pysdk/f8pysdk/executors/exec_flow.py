from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ..generated import F8EdgeKindEnum, F8RuntimeGraph
from ..capabilities import EntrypointNode, ExecutableNode
from ..nats_naming import ensure_token
from ..service_bus import ServiceBus


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

    async def emit_exec(self, out_port: str, *, ctx_id: str | int) -> None:
        await self.executor.trigger_exec(self.node_id, out_port, ctx_id=ctx_id)

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
    - Exec edges are strictly intra-process: only when `fromServiceId == toServiceId == runtime.service_id`.
    - Cross-process "triggering" must be modeled via data/state edges and handled in nodes.
    - Exactly one entrypoint node is allowed per graph activation.
    """

    def __init__(self, runtime: ServiceBus) -> None:
        self._runtime = runtime
        self._service_id = ensure_token(runtime.service_id, label="service_id")
        self._active = True

        self._graph: F8RuntimeGraph | None = None
        self._nodes: dict[str, Any] = {}

        self._exec_out: dict[tuple[str, str], list[tuple[str, str]]] = {}
        self._entrypoint_ctx: EntrypointContext | None = None

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
        try:
            active = bool(active)
        except Exception:
            active = True
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
    def register_node(self, node: Any) -> None:
        node_id = ensure_token(getattr(node, "node_id", ""), label="node_id")
        self._nodes[node_id] = node

    def unregister_node(self, node_id: str) -> None:
        node_id = ensure_token(node_id, label="node_id")
        self._nodes.pop(node_id, None)

    # ---- rungraph -------------------------------------------------------
    async def apply_rungraph(self, graph: F8RuntimeGraph) -> None:
        self._graph = graph
        self._rebuild_exec_routes(graph)
        if self._active:
            await self._restart_entrypoint_if_needed(graph)

    def _rebuild_exec_routes(self, graph: F8RuntimeGraph) -> None:
        out_map: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for edge in graph.edges:
            if edge.kind != F8EdgeKindEnum.exec:
                continue
            if str(edge.fromServiceId) != self._service_id or str(edge.toServiceId) != self._service_id:
                continue
            if not edge.fromOperatorId or not edge.toOperatorId:
                continue
            out_map.setdefault((str(edge.fromOperatorId), str(edge.fromPort)), []).append(
                (str(edge.toOperatorId), str(edge.toPort))
            )
        self._exec_out = out_map

    @staticmethod
    def _entrypoint_node_id(graph: F8RuntimeGraph) -> str | None:
        entrypoint_ids: list[str] = []
        for n in list(graph.nodes or []):
            node_id = str(getattr(n, "nodeId", "") or "")
            in_n = len(list(getattr(n, "execInPorts", None) or []))
            out_n = len(list(getattr(n, "execOutPorts", None) or []))
            if in_n == 0 and out_n > 0:
                entrypoint_ids.append(node_id)
        if not entrypoint_ids:
            return None
        if len(entrypoint_ids) > 1:
            raise ValueError(f"graph has multiple exec entrypoints: {entrypoint_ids}")
        return entrypoint_ids[0]

    async def _restart_entrypoint_if_needed(self, graph: F8RuntimeGraph) -> None:
        new_source = self._entrypoint_node_id(graph)
        cur = getattr(self._entrypoint_ctx, "node_id", None) if self._entrypoint_ctx else None
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
    async def run_once(self, *, max_steps: int = 1024) -> None:
        if not self._active:
            return
        graph = self._graph
        if graph is None:
            raise RuntimeError("no graph applied")
        src = self._entrypoint_node_id(graph)
        if not src:
            raise RuntimeError("graph has no exec entrypoint")
        await self.trigger_exec(src, "__source__", ctx_id=int(time.time() * 1000), max_steps=max_steps)

    async def trigger_exec(self, node_id: str, out_port: str, *, ctx_id: str | int, max_steps: int = 1024) -> None:
        """
        Inject an exec trigger from (node_id, out_port) and propagate intra-service exec edges.
        """
        if not self._active:
            return
        node_id = ensure_token(node_id, label="node_id")
        out_port = str(out_port)

        if max_steps <= 0:
            return

        # The root trigger is a virtual exec out on the source.
        queue: list[tuple[str, str]] = []
        if out_port == "__source__":
            # Use all declared exec out ports from the runtime graph, if available.
            graph = self._graph
            outs: list[str] = []
            if graph is not None:
                for n in graph.nodes:
                    if str(getattr(n, "nodeId", "")) == node_id:
                        outs = list(getattr(n, "execOutPorts", None) or [])
                        break
            for p in outs:
                queue.extend(self._exec_out.get((node_id, str(p)), []))
        else:
            queue.extend(self._exec_out.get((node_id, out_port), []))

        steps = 0
        while queue and steps < max_steps:
            steps += 1
            to_node, in_port = queue.pop(0)
            node = self._nodes.get(to_node)
            if node is None:
                continue
            if not isinstance(node, ExecutableNode):
                continue
            try:
                out_ports = await node.on_exec(ctx_id, str(in_port))  # type: ignore[misc]
            except Exception:
                continue
            for p in list(out_ports or []):
                for nxt in self._exec_out.get((to_node, str(p)), []):
                    queue.append(nxt)

