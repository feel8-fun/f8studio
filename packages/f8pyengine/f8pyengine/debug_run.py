from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nats.js.api import StorageType  # type: ignore[import-not-found]

from f8pysdk import F8RuntimeGraph
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.service_app import ServiceApp, ServiceAppConfig

from f8pysdk.executors.exec_flow import ExecFlowExecutor as EngineExecutor
from f8pyengine.engine_binder import EngineBinder
from f8pyengine.pyengine_node_registry import register_pyengine_runtimes


@dataclass(frozen=True)
class _RunningService:
    graph_path: Path
    graph: F8RuntimeGraph
    app: ServiceApp
    executor: EngineExecutor
    binder: EngineBinder


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
        n.serviceId = str(r(n.serviceId))
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


async def _amain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Debug runner: load per-service F8RuntimeGraph JSON and run PyEngine with the latest SDK runtime (ServiceBus)."
        )
    )
    parser.add_argument("--graph", required=True, help="Path to per-service runtime graph JSON.")
    parser.add_argument("--nats-url", default="nats://127.0.0.1:4222", help="NATS server URL.")
    parser.add_argument("--service-class", default="f8.pyengine", help="Local serviceClass filter for node instantiation.")
    parser.add_argument(
        "--runtime-modules",
        default="",
        help=(
            "Comma-separated Python modules to import for registering additional runtime nodes "
            "(via RuntimeNodeRegistry.instance().register(...))."
        ),
    )
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

    registry = RuntimeNodeRegistry.instance()
    register_pyengine_runtimes(registry)
    modules = [m.strip() for m in str(args.runtime_modules or "").split(",") if m.strip()]
    if modules:
        registry.load_modules(modules)

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
        service_class = str(args.service_class or "").strip() or "f8.pyengine"

        app = ServiceApp(
            ServiceAppConfig(
                service_id=sid,
                service_class=service_class,
                nats_url=str(args.nats_url),
                kv_storage=kv_storage,
                delete_bucket_on_start=bool(args.delete_bucket_on_start),
                delete_bucket_on_stop=bool(args.delete_bucket_on_exit),
            ),
            registry=registry,
        )
        executor = EngineExecutor(app.bus)
        binder = EngineBinder(bus=app.bus, executor=executor, service_class=service_class)

        async def _on_lifecycle(active: bool, _meta: dict[str, object]) -> None:
            await executor.set_active(active)

        app.bus.add_lifecycle_listener(_on_lifecycle)

        await app.start()
        await app.bus.set_rungraph(graph)

        print(f"debug_run: serviceId={sid} serviceClass={service_class} nats={str(args.nats_url).strip()}")
        print("debug_run: running (Ctrl+C to stop)")

        running = _RunningService(graph_path=graph_path, graph=graph, app=app, executor=executor, binder=binder)
        await asyncio.Event().wait()
    finally:
        if running is not None:
            try:
                await running.executor.stop_entrypoint()
            except Exception:
                pass
            try:
                await running.app.stop()
            except Exception:
                pass

    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_amain(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    main()
