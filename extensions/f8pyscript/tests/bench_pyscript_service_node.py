from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from time import perf_counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode
from f8pysdk.host import ServiceHost, ServiceHostConfig
from f8pysdk.testing import ServiceBusHarness

from f8pyscript.constants import SERVICE_CLASS
from f8pyscript.script_node_registry import create_pyscript_registry
from f8pyscript.script_service_node import PythonScriptServiceNode


@dataclass(frozen=True)
class BenchResult:
    mode: str
    name: str
    iterations: int
    elapsed_s: float

    @property
    def ops_per_sec(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return float(self.iterations) / self.elapsed_s

    @property
    def us_per_op(self) -> float:
        if self.iterations <= 0:
            return 0.0
        return (self.elapsed_s * 1_000_000.0) / float(self.iterations)


def _service_node(code: str) -> F8RuntimeNode:
    desc = create_pyscript_registry().describe(SERVICE_CLASS)
    spec = desc.service
    return F8RuntimeNode(
        nodeId="svcA",
        serviceId="svcA",
        serviceClass=SERVICE_CLASS,
        operatorClass=None,
        dataInPorts=list(spec.dataInPorts or []),
        dataOutPorts=list(spec.dataOutPorts or []),
        stateFields=list(spec.stateFields or []),
        stateValues={"code": code, "tickEnabled": False, "tickMs": 100},
    )


async def _build_node(code: str) -> PythonScriptServiceNode:
    harness = ServiceBusHarness()
    bus = harness.create_bus("svcA")
    reg = create_pyscript_registry()
    _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)
    graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[_service_node(code)], edges=[])
    await bus.set_rungraph(graph)
    node = bus.get_node("svcA")
    if not isinstance(node, PythonScriptServiceNode):
        raise TypeError("runtime node is not PythonScriptServiceNode")
    return node


async def _bench_on_data(mode: str, iterations: int, warmup: int) -> BenchResult:
    code = (
        "def onData(ctx, port, value, ts_ms=None):\n"
        "    return {'outputs': {'out': value}}\n"
    )
    node = await _build_node(code)
    await node.on_state("code", code, ts_ms=1)

    for i in range(warmup):
        await node.on_data("in", i, ts_ms=i)

    t0 = perf_counter()
    for i in range(iterations):
        await node.on_data("in", i, ts_ms=i)
    elapsed = perf_counter() - t0
    await node.close()
    return BenchResult(mode=mode, name="on_data", iterations=iterations, elapsed_s=elapsed)


async def _bench_on_command(mode: str, iterations: int, warmup: int) -> BenchResult:
    code = (
        "async def onCommand(ctx, name, args, meta=None):\n"
        "    if name != 'inc':\n"
        "        return {'ok': False}\n"
        "    return {'value': int(args.get('x') or 0) + 1}\n"
    )
    node = await _build_node(code)
    await node.on_state("code", code, ts_ms=1)

    for i in range(warmup):
        reply = await node.on_command("inc", {"x": i})
        out = (reply or {}).get("result") if isinstance(reply, dict) else None
        if not isinstance(out, dict) or int(out.get("value") or 0) != (i + 1):
            raise AssertionError("warmup command output mismatch")

    t0 = perf_counter()
    for i in range(iterations):
        reply = await node.on_command("inc", {"x": i})
        out = (reply or {}).get("result") if isinstance(reply, dict) else None
        if not isinstance(out, dict) or int(out.get("value") or 0) != (i + 1):
            raise AssertionError("command output mismatch")
    elapsed = perf_counter() - t0
    await node.close()
    return BenchResult(mode=mode, name="on_command", iterations=iterations, elapsed_s=elapsed)


def _print_result(result: BenchResult) -> None:
    print(
        f"{result.mode}:{result.name} "
        f"n={result.iterations} "
        f"elapsed={result.elapsed_s:.4f}s "
        f"throughput={result.ops_per_sec:,.0f} ops/s "
        f"latency={result.us_per_op:.2f} us/op"
    )


async def _run_mode(mode: str, iterations: int, warmup: int) -> list[BenchResult]:
    on_data_res = await _bench_on_data(mode, iterations=iterations, warmup=warmup)
    on_cmd_res = await _bench_on_command(mode, iterations=iterations, warmup=warmup)
    return [on_data_res, on_cmd_res]


async def main_async(args: argparse.Namespace) -> None:
    print(f"iterations={args.iterations} warmup={args.warmup}")

    current_results = await _run_mode("current", iterations=args.iterations, warmup=args.warmup)
    for item in current_results:
        _print_result(item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark f8.pyscript invoke-path optimizations")
    parser.add_argument("--iterations", type=int, default=100_000, help="iterations per benchmark")
    parser.add_argument("--warmup", type=int, default=2_000, help="warmup iterations")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
