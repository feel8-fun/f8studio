from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nats.js.api import StorageType  # type: ignore[import-not-found]

from f8pysdk import F8RuntimeGraph
from f8pysdk.runtime import ServiceRuntime, ServiceRuntimeConfig, ensure_token

from f8pyengine.engine_executor import EngineExecutor
from f8pyengine.engine_host import EngineHost, EngineHostConfig
from f8pyengine.runtime_registry import register_pyengine_runtimes


@dataclass(frozen=True)
class _RunningService:
    graph_path: Path
    graph: F8RuntimeGraph
    runtime: ServiceRuntime
    host: EngineHost
    executor: EngineExecutor


def _load_graph(path: Path) -> F8RuntimeGraph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return F8RuntimeGraph.model_validate(payload)


def _rewrite_single_service_id(graph: F8RuntimeGraph, new_service_id: str) -> F8RuntimeGraph:
    """
    Best-effort rewrite for running a single-service graph under a different serviceId.
    """
    new_service_id = ensure_token(new_service_id, label="service_id")
    if not graph.services:
        raise ValueError("graph.services is empty; cannot infer old serviceId")
    if len(graph.services) != 1:
        raise ValueError("can only rewrite serviceId for graphs with exactly 1 service")
    old = ensure_token(str(graph.services[0].serviceId), label="service_id")
    if old == new_service_id:
        return graph

    def r(v: Any) -> Any:
        return new_service_id if str(v) == old else v

    g = graph.model_copy(deep=True)
    for s in g.services:
        s.serviceId = str(r(s.serviceId))
    for n in g.nodes:
        n.serviceId = str(r(getattr(n, "serviceId", "")))
        if str(getattr(n, "nodeId", "")) == old:
            n.nodeId = new_service_id
    for e in g.edges:
        e.fromServiceId = str(r(e.fromServiceId))
        e.toServiceId = str(r(e.toServiceId))
        if getattr(e, "fromOperatorId", None) and str(e.fromOperatorId) == old:
            e.fromOperatorId = new_service_id
        if getattr(e, "toOperatorId", None) and str(e.toOperatorId) == old:
            e.toOperatorId = new_service_id
    return g


async def _run_one_graph(*, graph_path: Path, graph: F8RuntimeGraph, nats_url: str) -> _RunningService:
    if not graph.services:
        raise ValueError(f"{graph_path}: graph.services is empty")
    if len(graph.services) != 1:
        raise ValueError(f"{graph_path}: expected per-service graph with exactly 1 service")

    service_id = ensure_token(str(graph.services[0].serviceId), label="service_id")

    runtime = ServiceRuntime(ServiceRuntimeConfig(service_id=service_id, nats_url=str(nats_url)))
    executor = EngineExecutor(runtime)
    host = EngineHost(runtime, executor, config=EngineHostConfig(service_class="f8.pyengine"))

    async def _on_topology(g: F8RuntimeGraph) -> None:
        await host.apply_topology(g)
        await executor.apply_topology(g)

    runtime.add_topology_listener(_on_topology)

    await runtime.start()
    await runtime.set_topology(graph)

    return _RunningService(graph_path=graph_path, graph=graph, runtime=runtime, host=host, executor=executor)


async def _amain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Debug runner: load per-service F8RuntimeGraph JSON and run PyEngine.")
    parser.add_argument("--graph", action="append", required=True, help="Path to per-service runtime graph JSON.")
    parser.add_argument("--nats-url", default="nats://127.0.0.1:4222", help="NATS server URL.")
    parser.add_argument(
        "--service-id",
        default="",
        help="Override serviceId (only valid when a single --graph is provided).",
    )
    parser.add_argument(
        "--kv-storage",
        default="memory",
        choices=["memory", "file"],
        help="JetStream KV bucket storage backend (default: memory).",
    )
    parser.add_argument(
        "--delete-bucket-on-exit",
        action="store_true",
        help="Delete this serviceId's KV bucket on shutdown (unsafe if other processes share the same serviceId).",
    )
    args = parser.parse_args(argv)

    register_pyengine_runtimes()

    graph_paths = [Path(p).expanduser().resolve() for p in (args.graph or [])]
    graphs = [(_p, _load_graph(_p)) for _p in graph_paths]

    if args.service_id:
        if len(graphs) != 1:
            raise SystemExit("--service-id override only supported with a single --graph")
        gp, g = graphs[0]
        graphs = [(gp, _rewrite_single_service_id(g, str(args.service_id).strip()))]

    kv_storage: StorageType = StorageType.MEMORY
    if str(args.kv_storage or "").strip() == "memory":
        kv_storage = StorageType.MEMORY
    elif str(args.kv_storage or "").strip() == "file":
        kv_storage = StorageType.FILE

    running: list[_RunningService] = []
    try:
        for gp, g in graphs:
            if not g.services:
                raise ValueError(f"{gp}: graph.services is empty")
            if len(g.services) != 1:
                raise ValueError(f"{gp}: expected per-service graph with exactly 1 service")

            sid = ensure_token(str(g.services[0].serviceId), label="service_id")

            runtime = ServiceRuntime(
                ServiceRuntimeConfig(
                    service_id=sid,
                    nats_url=str(args.nats_url),
                    kv_storage=kv_storage,
                    delete_bucket_on_stop=bool(args.delete_bucket_on_exit),
                )
            )
            executor = EngineExecutor(runtime)
            host = EngineHost(runtime, executor, config=EngineHostConfig(service_class="f8.pyengine"))

            async def _on_topology(gr: F8RuntimeGraph) -> None:
                await host.apply_topology(gr)
                await executor.apply_topology(gr)

            runtime.add_topology_listener(_on_topology)
            await runtime.start()
            await runtime.set_topology(g)

            running.append(_RunningService(graph_path=gp, graph=g, runtime=runtime, host=host, executor=executor))
        await asyncio.Event().wait()
    finally:
        for r in running:
            try:
                await r.executor.stop_source()
            except Exception:
                pass
        for r in running:
            try:
                await r.runtime.stop()
            except Exception:
                pass

    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_amain(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
