from __future__ import annotations

import argparse
import asyncio
import os
import sys
from time import perf_counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.generated import (  # noqa: E402
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
)
from f8pysdk.schema_helpers import string_schema  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402


def _state_field(name: str) -> F8StateSpec:
    return F8StateSpec(name=name, valueSchema=string_schema(), access=F8StateAccess.rw)


async def bench_publish_state(iterations: int) -> tuple[float, float]:
    harness = ServiceBusHarness()
    bus = harness.create_bus("svcA")
    bus._state_access_by_node_field[("opA", "out")] = F8StateAccess.rw

    t0 = perf_counter()
    for i in range(iterations):
        await bus.publish_state_runtime("opA", "out", i, ts_ms=i + 1)
    elapsed = perf_counter() - t0
    throughput = float(iterations) / elapsed if elapsed > 0 else 0.0
    return elapsed, throughput


async def bench_cross_state_sync(iterations: int) -> tuple[float, float]:
    harness = ServiceBusHarness()
    bus_a = harness.create_bus("svcA")
    bus_b = harness.create_bus("svcB")

    node_a = F8RuntimeNode(
        nodeId="opA",
        serviceId="svcA",
        serviceClass="svcA",
        operatorClass="OpA",
        stateFields=[_state_field("out")],
    )
    node_b = F8RuntimeNode(
        nodeId="opB",
        serviceId="svcB",
        serviceClass="svcB",
        operatorClass="OpB",
        stateFields=[_state_field("input")],
    )
    edge = F8Edge(
        edgeId="e1",
        fromServiceId="svcA",
        fromOperatorId="opA",
        fromPort="out",
        toServiceId="svcB",
        toOperatorId="opB",
        toPort="input",
        kind=F8EdgeKindEnum.state,
        strategy=F8EdgeStrategyEnum.latest,
    )
    graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[node_a, node_b], edges=[edge])

    await bus_a.set_rungraph(graph)
    await bus_b.set_rungraph(graph)

    total = 0.0
    for i in range(iterations):
        t0 = perf_counter()
        await bus_a.publish_state_runtime("opA", "out", f"v{i}", ts_ms=i + 1)

        matched = False
        for _ in range(1000):
            current = await bus_b.get_state("opB", "input")
            if current.found and current.value == f"v{i}":
                matched = True
                break
            await asyncio.sleep(0)
        if not matched:
            raise RuntimeError(f"cross-state sync timeout at iteration={i}")

        total += perf_counter() - t0

    avg_ms = (total / float(iterations)) * 1000.0 if iterations > 0 else 0.0
    return total, avg_ms


async def bench_set_rungraph(node_count: int) -> float:
    harness = ServiceBusHarness()
    bus = harness.create_bus("svcA")

    nodes: list[F8RuntimeNode] = []
    edges: list[F8Edge] = []
    for i in range(node_count):
        node_id = f"op{i}"
        nodes.append(
            F8RuntimeNode(
                nodeId=node_id,
                serviceId="svcA",
                serviceClass="svcA",
                operatorClass="OpBench",
                stateFields=[_state_field("value")],
                stateValues={"value": f"seed-{i}"},
            )
        )
        if i > 0:
            edges.append(
                F8Edge(
                    edgeId=f"e{i}",
                    fromServiceId="svcA",
                    fromOperatorId=f"op{i-1}",
                    fromPort="value",
                    toServiceId="svcA",
                    toOperatorId=node_id,
                    toPort="value",
                    kind=F8EdgeKindEnum.state,
                    strategy=F8EdgeStrategyEnum.latest,
                )
            )

    graph = F8RuntimeGraph(graphId="g-bench", revision="r1", nodes=nodes, edges=edges)
    t0 = perf_counter()
    await bus.set_rungraph(graph)
    return (perf_counter() - t0) * 1000.0


async def main_async(args: argparse.Namespace) -> None:
    elapsed, throughput = await bench_publish_state(args.publish_n)
    total_cross, avg_cross_ms = await bench_cross_state_sync(args.cross_n)
    set_rungraph_ms = await bench_set_rungraph(args.graph_nodes)

    print(f"publish_state: n={args.publish_n} elapsed={elapsed:.4f}s throughput={throughput:.1f} ops/s")
    print(f"cross_state_sync: n={args.cross_n} total={total_cross:.4f}s avg={avg_cross_ms:.3f}ms")
    print(f"set_rungraph: nodes={args.graph_nodes} apply={set_rungraph_ms:.3f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="ServiceBus micro-benchmarks")
    parser.add_argument("--publish-n", type=int, default=5000, help="state publish iterations")
    parser.add_argument("--cross-n", type=int, default=1000, help="cross-state sync iterations")
    parser.add_argument("--graph-nodes", type=int, default=200, help="node count for set_rungraph benchmark")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
