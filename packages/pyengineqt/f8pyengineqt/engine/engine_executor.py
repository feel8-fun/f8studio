from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from f8pysdk import F8EdgeKindEnum

from ..graph.operator_graph import OperatorGraph
from ..runtime.service_runtime import ServiceRuntime
from .operator_runtime import OperatorRuntimeNode
from .operator_runtime_registry import OperatorRuntimeRegistry


@dataclass(frozen=True)
class EngineExecutorConfig:
    max_steps: int = 10_000


class EngineExecutor:
    def __init__(self, runtime: ServiceRuntime, *, config: EngineExecutorConfig | None = None) -> None:
        self._runtime = runtime
        self._config = config or EngineExecutorConfig()
        self._graph: OperatorGraph | None = None
        self._nodes: dict[str, OperatorRuntimeNode] = {}

    @property
    def graph(self) -> OperatorGraph | None:
        return self._graph

    async def apply_rungraph(self, graph: OperatorGraph) -> None:
        """
        Create/update runtime nodes from the latest rungraph.
        """
        self._graph = graph

        want_ids = set(graph.nodes.keys())
        for node_id in list(self._nodes.keys()):
            if node_id in want_ids:
                continue
            self._runtime.unregister_node(node_id)
            self._nodes.pop(node_id, None)

        reg = OperatorRuntimeRegistry.instance()
        for node_id, inst in graph.nodes.items():
            if node_id in self._nodes:
                continue
            try:
                node = reg.create(node_id=str(node_id), spec=inst.spec, initial_state=inst.state)
            except Exception:
                continue
            self._nodes[str(node_id)] = node
            self._runtime.register_node(node)

        await self._seed_state_defaults(graph)

    async def _seed_state_defaults(self, graph: OperatorGraph) -> None:
        """
        Reconcile rungraph-provided state values into KV.

        If KV already has a value and differs, prefer the rungraph value and write it back
        with a fresh timestamp (current time).
        """
        for node_id, inst in graph.nodes.items():
            for k, v in (inst.state or {}).items():
                try:
                    existing = await self._runtime.get_state(str(node_id), str(k))
                except Exception:
                    existing = None
                if existing is not None and existing == v:
                    continue
                try:
                    await self._runtime.set_state_with_meta(
                        str(node_id),
                        str(k),
                        v,
                        source="rungraph",
                        meta={"rungraphReconcile": True},
                    )
                except Exception:
                    continue

    async def run_once(self) -> None:
        graph = self._graph
        if graph is None:
            return

        entry_nodes: list[tuple[str, list[str]]] = []
        for node_id, inst in graph.nodes.items():
            ins = list(inst.spec.execInPorts or [])
            outs = list(inst.spec.execOutPorts or [])
            if ins:
                continue
            if not outs:
                continue
            entry_nodes.append((str(node_id), outs))

        visited_steps = 0
        seen: set[tuple[str, str]] = set()

        async def _fire(from_node_id: str, out_port: str) -> None:
            nonlocal visited_steps
            if visited_steps >= self._config.max_steps:
                return
            key = (from_node_id, out_port)
            if key in seen:
                return
            seen.add(key)
            visited_steps += 1

            edges = [
                e
                for e in graph.exec_edges
                if e.kind == F8EdgeKindEnum.exec
                and str(e.fromServiceId) == self._runtime.service_id
                and str(e.toServiceId) == self._runtime.service_id
                and str(e.fromOperatorId) == from_node_id
                and str(e.fromPort) == out_port
            ]
            for edge in edges:
                if not edge.toOperatorId:
                    continue
                await _exec_node(str(edge.toOperatorId), str(edge.toPort))

        async def _exec_node(node_id: str, in_port: str) -> None:
            nonlocal visited_steps
            if visited_steps >= self._config.max_steps:
                return
            visited_steps += 1

            node = self._nodes.get(node_id)
            if node is None:
                return
            try:
                outs = await node.on_exec(in_port)
            except Exception:
                outs = list(getattr(node.ctx.spec, "execOutPorts", None) or [])
            for out_port in outs:
                await _fire(node_id, str(out_port))

        for node_id, out_ports in entry_nodes:
            for out_port in out_ports:
                await _fire(node_id, out_port)

        await self._flush_sinks(graph)

    async def _flush_sinks(self, graph: OperatorGraph) -> None:
        """
        Flush sink nodes (nodes with data inputs but no data outputs).
        """
        for node_id, node in list(self._nodes.items()):
            try:
                inst = graph.nodes[node_id]
            except Exception:
                continue
            if not (inst.spec.dataInPorts or []):
                continue
            has_outgoing_edge = any(
                e.kind == F8EdgeKindEnum.data
                and str(e.fromServiceId) == self._runtime.service_id
                and str(e.fromOperatorId) == str(node_id)
                for e in graph.data_edges
            )
            if has_outgoing_edge:
                continue
            try:
                await node.flush()
            except Exception:
                continue
