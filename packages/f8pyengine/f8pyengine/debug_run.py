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

    async def _on_rungraph(g: F8RuntimeGraph) -> None:
        await host.apply_rungraph(g)
        await executor.apply_rungraph(g)

    runtime.add_rungraph_listener(_on_rungraph)

    await runtime.start()
    await runtime.set_rungraph(graph)

    return _RunningService(graph_path=graph_path, graph=graph, runtime=runtime, host=host, executor=executor)


async def _amain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Debug runner: load per-service F8RuntimeGraph JSON and run PyEngine.")
    parser.add_argument("--graph", required=True, help="Path to per-service runtime graph JSON.")
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
    parser.add_argument(
        "--delete-bucket-on-start",
        action="store_true",
        help="Delete this serviceId's KV bucket on startup (unsafe if other processes share the same serviceId).",
    )
    args = parser.parse_args(argv)

    register_pyengine_runtimes()

    graph_path = Path(args.graph).expanduser().resolve()
    graph = _load_graph(graph_path)

    if args.service_id:
        graph = _rewrite_single_service_id(graph, str(args.service_id).strip())

    if not bool(args.delete_bucket_on_start):
        try:
            has_state_values = any(bool(getattr(n, "stateValues", None)) for n in (graph.nodes or []))
        except Exception:
            has_state_values = False
        if has_state_values:
            print(
                "debug_run: note: this run will reconcile rungraph stateValues into KV; "
                "use --delete-bucket-on-start if you want a fully clean state bucket."
            )

    kv_storage: StorageType = StorageType.MEMORY
    if str(args.kv_storage or "").strip() == "memory":
        kv_storage = StorageType.MEMORY
    elif str(args.kv_storage or "").strip() == "file":
        kv_storage = StorageType.FILE

    running: _RunningService | None = None
    try:
        if not graph.services:
            raise ValueError(f"{graph_path}: graph.services is empty")
        if len(graph.services) != 1:
            raise ValueError(f"{graph_path}: expected per-service graph with exactly 1 service")

        sid = ensure_token(str(graph.services[0].serviceId), label="service_id")

        runtime = ServiceRuntime(
            ServiceRuntimeConfig(
                service_id=sid,
                nats_url=str(args.nats_url),
                kv_storage=kv_storage,
                delete_bucket_on_start=bool(args.delete_bucket_on_start),
                delete_bucket_on_stop=bool(args.delete_bucket_on_exit),
            )
        )
        executor = EngineExecutor(runtime)
        host = EngineHost(runtime, executor, config=EngineHostConfig(service_class="f8.pyengine"))

        async def _on_rungraph(gr: F8RuntimeGraph) -> None:
            await host.apply_rungraph(gr)
            await executor.apply_rungraph(gr)

        runtime.add_rungraph_listener(_on_rungraph)
        await runtime.start()
        await runtime.set_rungraph(graph)

        running = _RunningService(graph_path=graph_path, graph=graph, runtime=runtime, host=host, executor=executor)
        await asyncio.Event().wait()
    finally:
        if running is not None:
            try:
                await running.executor.stop_source()
            except Exception:
                pass
            try:
                await running.runtime.stop()
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
