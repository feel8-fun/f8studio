from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid

from f8pysdk import (
    F8DataPortSpec,
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8ServiceDescribe,
    F8StateAccess,
    F8StateSpec,
)
from f8pysdk.schema_helpers import integer_schema, number_schema, string_schema
from f8pysdk.runtime import ServiceRuntime, ServiceRuntimeConfig

from f8pyengine.engine_executor import EngineExecutor
from f8pyengine.engine_host import EngineHost, EngineHostConfig
from f8pyengine.runtime_registry import register_pyengine_runtimes
from f8pyengine.service_host import ServiceHostRegistry


def _env_or(default: str, key: str) -> str:
    v = os.environ.get(key)
    return v.strip() if v and v.strip() else default


async def _run_service(*, service_id: str, nats_url: str) -> None:
    register_pyengine_runtimes()

    runtime = ServiceRuntime(ServiceRuntimeConfig(service_id=service_id, nats_url=nats_url))
    executor = EngineExecutor(runtime)
    host = EngineHost(runtime, executor, config=EngineHostConfig(service_class="f8.pyengine"))

    async def _on_rungraph(graph) -> None:
        await host.apply_rungraph(graph)
        await executor.apply_rungraph(graph)

    runtime.add_rungraph_listener(_on_rungraph)

    await runtime.start()

    # Optional demo: 10 fps tick drives a 1 Hz sine, printed by a sink node.
    try:
        if os.environ.get("F8_PYENGINE_DEMO", "").strip().lower() in ("1", "true", "yes", "demo_sine"):
            await runtime.set_rungraph(_demo_sine_graph(service_id))
    except Exception:
        pass

    try:
        await asyncio.Event().wait()
    finally:
        try:
            await executor.stop_source()
        except Exception:
            pass
        await runtime.stop()


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F8PyEngine")
    parser.add_argument("--describe", action="store_true", help="Output the service description in JSON format")
    parser.add_argument("--service-id", default=_env_or("", "F8_SERVICE_ID"), help="Service instance id (required)")
    parser.add_argument("--nats-url", default=_env_or("nats://127.0.0.1:4222", "F8_NATS_URL"), help="NATS server URL")
    parser.add_argument(
        "--demo-sine",
        action="store_true",
        help="Publish a demo graph (10fps tick -> 1Hz sine -> print) into this serviceId.",
    )
    args = parser.parse_args(argv)

    if args.describe:
        describe = F8ServiceDescribe(
            service=ServiceHostRegistry.instance().service_spec(),
            operators=ServiceHostRegistry.instance().operator_specs(),
        ).model_dump(mode="json")
        print(json.dumps(describe, ensure_ascii=False, indent=1))
        return 0

    service_id = str(args.service_id or "").strip()
    if not service_id:
        raise SystemExit("Missing --service-id (or env F8_SERVICE_ID)")

    if args.demo_sine:
        os.environ["F8_PYENGINE_DEMO"] = "demo_sine"

    asyncio.run(_run_service(service_id=service_id, nats_url=str(args.nats_url).strip()))
    return 0


def _demo_sine_graph(service_id: str) -> F8RuntimeGraph:
    service_id = str(service_id).strip()
    tick_id = "tick"
    sine_id = "sine"
    print_id = "print"

    return F8RuntimeGraph(
        graphId="demo_sine",
        revision=str(int(time.time() * 1000)),
        nodes=[
            F8RuntimeNode(
                nodeId=tick_id,
                serviceClass="f8.pyengine",
                operatorClass="f8.tick",
                execInPorts=[],
                execOutPorts=["exec"],
                stateFields=[
                    F8StateSpec(
                        name="tickMs",
                        label="Tick (ms)",
                        valueSchema=integer_schema(default=100, minimum=16, maximum=5000),
                        access=F8StateAccess.rw,
                        showOnNode=True,
                    )
                ],
                stateValues={"tickMs": 100},
            ),
            F8RuntimeNode(
                nodeId=sine_id,
                serviceClass="f8.pyengine",
                operatorClass="f8.sine",
                execInPorts=["exec"],
                execOutPorts=[],
                dataOutPorts=[F8DataPortSpec(name="value", description="sine output", valueSchema=number_schema())],
                stateFields=[
                    F8StateSpec(
                        name="hz",
                        label="Hz",
                        valueSchema=number_schema(default=1.0, minimum=0.0, maximum=100.0),
                        access=F8StateAccess.rw,
                        showOnNode=True,
                    ),
                    F8StateSpec(
                        name="amp",
                        label="Amp",
                        valueSchema=number_schema(default=1.0, minimum=0.0, maximum=1000.0),
                        access=F8StateAccess.rw,
                        showOnNode=True,
                    ),
                ],
                stateValues={"hz": 1.0, "amp": 1.0},
            ),
            F8RuntimeNode(
                nodeId=print_id,
                serviceClass="f8.pyengine",
                operatorClass="f8.print",
                dataInPorts=[F8DataPortSpec(name="value", description="value to print", valueSchema=number_schema())],
                stateFields=[
                    F8StateSpec(
                        name="label",
                        label="Label",
                        valueSchema=string_schema(default="sine"),
                        access=F8StateAccess.rw,
                        showOnNode=True,
                    )
                ],
                stateValues={"label": "sine"},
            ),
        ],
        edges=[
            F8Edge(
                edgeId=uuid.uuid4().hex,
                fromServiceId=service_id,
                fromOperatorId=tick_id,
                fromPort="exec",
                toServiceId=service_id,
                toOperatorId=sine_id,
                toPort="exec",
                kind=F8EdgeKindEnum.exec,
                strategy=F8EdgeStrategyEnum.latest,
            ),
            F8Edge(
                edgeId=uuid.uuid4().hex,
                fromServiceId=service_id,
                fromOperatorId=sine_id,
                fromPort="value",
                toServiceId=service_id,
                toOperatorId=print_id,
                toPort="value",
                kind=F8EdgeKindEnum.data,
                strategy=F8EdgeStrategyEnum.latest,
            ),
        ],
    )


if __name__ == "__main__":
    raise SystemExit(_main())
