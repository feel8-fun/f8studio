from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from time import perf_counter
from types import MethodType
from typing import Any, Awaitable, Callable

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
PYSCRIPT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pyscript"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)
if PYSCRIPT_ROOT not in sys.path:
    sys.path.insert(0, PYSCRIPT_ROOT)

from f8pysdk.specs import F8DataPortSpec, F8RuntimeNode, any_schema, number_schema, string_schema  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.data_expr import DataExprRuntimeNode  # noqa: E402
from f8pyengine.operators.python_script import PythonScriptRuntimeNode  # noqa: E402
from f8pyengine.operators.tcode import AXES, TCodeRuntimeNode  # noqa: E402
from f8pyscript.script_service_node import PythonScriptServiceNode  # noqa: E402


SCRIPT_CODE_DOT = (
    "import math\n"
    "AXES = ('L0','L1','L2','R0','R1','R2','V0','V1','A0','A1')\n"
    "def _coerce_number(value):\n"
    "    if value is None or isinstance(value, bool):\n"
    "        return None\n"
    "    try:\n"
    "        f = float(value)\n"
    "    except Exception:\n"
    "        return None\n"
    "    if math.isnan(f) or math.isinf(f):\n"
    "        return None\n"
    "    return f\n"
    "def _js_round(value):\n"
    "    if value >= 0:\n"
    "        return int(math.floor(value + 0.5))\n"
    "    return -int(math.floor(abs(value) + 0.5))\n"
    "def onMsg(ctx, inputs):\n"
    "    interval = _coerce_number(inputs.intervalMs)\n"
    "    if interval is None:\n"
    "        interval = 20\n"
    "    interval_i = max(1, _js_round(float(interval)))\n"
    "    commands = []\n"
    "    values = [\n"
    "        ('L0', inputs.L0), ('L1', inputs.L1), ('L2', inputs.L2),\n"
    "        ('R0', inputs.R0), ('R1', inputs.R1), ('R2', inputs.R2),\n"
    "        ('V0', inputs.V0), ('V1', inputs.V1), ('A0', inputs.A0), ('A1', inputs.A1),\n"
    "    ]\n"
    "    for axis, raw in values:\n"
    "        numeric = _coerce_number(raw)\n"
    "        if numeric is None:\n"
    "            continue\n"
    "        clamped = min(1.0, max(0.0, float(numeric)))\n"
    "        payload = _js_round(clamped * 9999.0)\n"
    "        commands.append(f'{axis}{payload:04d}I{interval_i:03d}')\n"
    "    if not commands:\n"
    "        return {'outputs': {'tcode': ''}}\n"
    "    return {'outputs': {'tcode': ' '.join(commands) + '\\n'}}\n"
)

SCRIPT_CODE_MAPPING = (
    "import math\n"
    "AXES = ('L0','L1','L2','R0','R1','R2','V0','V1','A0','A1')\n"
    "def _coerce_number(value):\n"
    "    if value is None or isinstance(value, bool):\n"
    "        return None\n"
    "    try:\n"
    "        f = float(value)\n"
    "    except Exception:\n"
    "        return None\n"
    "    if math.isnan(f) or math.isinf(f):\n"
    "        return None\n"
    "    return f\n"
    "def _js_round(value):\n"
    "    if value >= 0:\n"
    "        return int(math.floor(value + 0.5))\n"
    "    return -int(math.floor(abs(value) + 0.5))\n"
    "def onMsg(ctx, inputs):\n"
    "    interval = _coerce_number(inputs.get('intervalMs'))\n"
    "    if interval is None:\n"
    "        interval = 20\n"
    "    interval_i = max(1, _js_round(float(interval)))\n"
    "    commands = []\n"
    "    values = [\n"
    "        ('L0', inputs.get('L0')), ('L1', inputs.get('L1')), ('L2', inputs.get('L2')),\n"
    "        ('R0', inputs.get('R0')), ('R1', inputs.get('R1')), ('R2', inputs.get('R2')),\n"
    "        ('V0', inputs.get('V0')), ('V1', inputs.get('V1')), ('A0', inputs.get('A0')), ('A1', inputs.get('A1')),\n"
    "    ]\n"
    "    for axis, raw in values:\n"
    "        numeric = _coerce_number(raw)\n"
    "        if numeric is None:\n"
    "            continue\n"
    "        clamped = min(1.0, max(0.0, float(numeric)))\n"
    "        payload = _js_round(clamped * 9999.0)\n"
    "        commands.append(f'{axis}{payload:04d}I{interval_i:03d}')\n"
    "    if not commands:\n"
    "        return {'outputs': {'tcode': ''}}\n"
    "    return {'outputs': {'tcode': ' '.join(commands) + '\\n'}}\n"
)

PYSCRIPT_SERVICE_CODE = (
    "import math\n"
    "AXES = ('L0','L1','L2','R0','R1','R2','V0','V1','A0','A1')\n"
    "def _coerce_number(value):\n"
    "    if value is None or isinstance(value, bool):\n"
    "        return None\n"
    "    try:\n"
    "        f = float(value)\n"
    "    except Exception:\n"
    "        return None\n"
    "    if math.isnan(f) or math.isinf(f):\n"
    "        return None\n"
    "    return f\n"
    "def _js_round(value):\n"
    "    if value >= 0:\n"
    "        return int(math.floor(value + 0.5))\n"
    "    return -int(math.floor(abs(value) + 0.5))\n"
    "def onData(ctx, port, value, ts_ms=None):\n"
    "    if value is None:\n"
    "        return {'outputs': {'tcode': ''}}\n"
    "    interval = _coerce_number(value.get('intervalMs'))\n"
    "    if interval is None:\n"
    "        interval = 20\n"
    "    interval_i = max(1, _js_round(float(interval)))\n"
    "    commands = []\n"
    "    for axis in AXES:\n"
    "        numeric = _coerce_number(value.get(axis))\n"
    "        if numeric is None:\n"
    "            continue\n"
    "        clamped = min(1.0, max(0.0, float(numeric)))\n"
    "        payload = _js_round(clamped * 9999.0)\n"
    "        commands.append(f'{axis}{payload:04d}I{interval_i:03d}')\n"
    "    if not commands:\n"
    "        return {'outputs': {'tcode': ''}}\n"
    "    return {'outputs': {'tcode': ' '.join(commands) + '\\n'}}\n"
)

EXPR_CODE = (
    "round(min(1,max(0,float(L0)))*9999)+"
    "round(min(1,max(0,float(L1)))*9999)+"
    "round(min(1,max(0,float(L2)))*9999)+"
    "round(min(1,max(0,float(R0)))*9999)+"
    "round(min(1,max(0,float(R1)))*9999)+"
    "round(min(1,max(0,float(R2)))*9999)+"
    "round(min(1,max(0,float(V0)))*9999)+"
    "round(min(1,max(0,float(V1)))*9999)+"
    "round(min(1,max(0,float(A0)))*9999)+"
    "round(min(1,max(0,float(A1)))*9999)+"
    "int(max(1,round(float(intervalMs))))"
)


@dataclass(frozen=True)
class BenchResult:
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


def _build_ports() -> tuple[list[F8DataPortSpec], list[F8DataPortSpec]]:
    data_in_ports: list[F8DataPortSpec] = []
    for axis in AXES:
        data_in_ports.append(
            F8DataPortSpec(name=axis, description=f"Axis {axis}", valueSchema=number_schema(), definitionProtected=False)
        )
    data_in_ports.append(
        F8DataPortSpec(name="intervalMs", description="Interval", valueSchema=number_schema(), definitionProtected=False)
    )
    data_out_ports = [F8DataPortSpec(name="tcode", description="TCode", valueSchema=string_schema(), definitionProtected=False)]
    return data_in_ports, data_out_ports


def _build_tcode_node() -> TCodeRuntimeNode:
    data_in_ports, data_out_ports = _build_ports()
    node_desc = F8RuntimeNode(
        nodeId="tcode",
        serviceId="svcA",
        serviceClass=SERVICE_CLASS,
        operatorClass=TCodeRuntimeNode.SPEC.operatorClass,
        dataInPorts=data_in_ports,
        dataOutPorts=data_out_ports,
        stateFields=list(TCodeRuntimeNode.SPEC.stateFields or []),
        stateValues={"intervalMs": 20},
    )
    return TCodeRuntimeNode(node_id="tcode", node=node_desc, initial_state={"intervalMs": 20})


def _build_python_script_node(*, node_id: str, code: str, input_mode: str) -> PythonScriptRuntimeNode:
    data_in_ports, data_out_ports = _build_ports()
    node_desc = F8RuntimeNode(
        nodeId=node_id,
        serviceId="svcA",
        serviceClass=SERVICE_CLASS,
        operatorClass=PythonScriptRuntimeNode.SPEC.operatorClass,
        dataInPorts=data_in_ports,
        dataOutPorts=data_out_ports,
        stateFields=list(PythonScriptRuntimeNode.SPEC.stateFields or []),
        stateValues={"code": code, "inputMode": input_mode},
    )
    return PythonScriptRuntimeNode(node_id=node_id, node=node_desc, initial_state={"code": code, "inputMode": input_mode})


def _build_data_expr_node() -> DataExprRuntimeNode:
    data_in_ports: list[F8DataPortSpec] = []
    for axis in AXES:
        data_in_ports.append(F8DataPortSpec(name=axis, description=f"Axis {axis}", valueSchema=any_schema(), definitionProtected=False))
    data_in_ports.append(F8DataPortSpec(name="intervalMs", description="Interval", valueSchema=any_schema(), definitionProtected=False))
    data_out_ports = [F8DataPortSpec(name="out", description="Expr output", valueSchema=any_schema(), definitionProtected=False)]
    node_desc = F8RuntimeNode(
        nodeId="expr",
        serviceId="svcA",
        serviceClass=SERVICE_CLASS,
        operatorClass=DataExprRuntimeNode.SPEC.operatorClass,
        dataInPorts=data_in_ports,
        dataOutPorts=data_out_ports,
        stateFields=list(DataExprRuntimeNode.SPEC.stateFields or []),
        stateValues={"code": EXPR_CODE, "allowNumpy": False, "unpackDictOutputs": False},
    )
    return DataExprRuntimeNode(
        node_id="expr",
        node=node_desc,
        initial_state={"code": EXPR_CODE, "allowNumpy": False, "unpackDictOutputs": False},
    )


def _build_pyscript_service_node() -> PythonScriptServiceNode:
    node_desc = F8RuntimeNode(
        nodeId="pyscript_service",
        serviceId="svcA",
        serviceClass="f8.pyscript",
        operatorClass=None,
        dataInPorts=[F8DataPortSpec(name="in", description="Input packet", valueSchema=any_schema(), definitionProtected=False)],
        dataOutPorts=[F8DataPortSpec(name="tcode", description="TCode output", valueSchema=string_schema(), definitionProtected=False)],
        stateFields=[],
        stateValues={"code": PYSCRIPT_SERVICE_CODE, "tickEnabled": False, "tickMs": 100},
    )
    return PythonScriptServiceNode(
        node_id="pyscript_service",
        node=node_desc,
        initial_state={"code": PYSCRIPT_SERVICE_CODE, "tickEnabled": False, "tickMs": 100},
    )


def _attach_fixed_inputs(node: TCodeRuntimeNode | PythonScriptRuntimeNode | DataExprRuntimeNode, inputs: dict[str, Any]) -> None:
    async def pull_impl(self: Any, port: str, *, ctx_id: str | int | None = None) -> Any:
        del ctx_id
        return inputs.get(str(port))

    async def get_state_value_impl(self: Any, field: str) -> Any:
        if str(field) == "intervalMs":
            return 20
        return None

    node.pull = MethodType(pull_impl, node)
    node.get_state_value = MethodType(get_state_value_impl, node)


async def _run_bench(
    *,
    name: str,
    compute_one: Callable[[int], Awaitable[Any]],
    iterations: int,
    warmup: int,
    expected: Any,
) -> BenchResult:
    for i in range(warmup):
        output = await compute_one(i)
        if output != expected:
            raise AssertionError(f"{name} warmup output mismatch at i={i}")

    t0 = perf_counter()
    for i in range(iterations):
        output = await compute_one(i + warmup)
        if output != expected:
            raise AssertionError(f"{name} output mismatch at i={i}")
    elapsed = perf_counter() - t0
    return BenchResult(name=name, iterations=iterations, elapsed_s=elapsed)


async def _run_bench_concurrent(
    *,
    name: str,
    compute_one: Callable[[int], Awaitable[Any]],
    total_iterations: int,
    warmup: int,
    concurrency: int,
    expected: Any,
) -> BenchResult:
    for i in range(warmup):
        output = await compute_one(i)
        if output != expected:
            raise AssertionError(f"{name} warmup output mismatch at i={i}")

    worker_count = max(1, int(concurrency))
    iterations_per_worker = total_iterations // worker_count
    remainder = total_iterations % worker_count

    async def worker(worker_id: int, count: int) -> None:
        start = worker_id * iterations_per_worker
        for i in range(count):
            ctx = 10_000_000 + start + i
            output = await compute_one(ctx)
            if output != expected:
                raise AssertionError(f"{name} concurrent output mismatch at worker={worker_id} i={i}")

    tasks: list[asyncio.Task[None]] = []
    for worker_id in range(worker_count):
        count = iterations_per_worker + (1 if worker_id < remainder else 0)
        if count <= 0:
            continue
        tasks.append(asyncio.create_task(worker(worker_id, count), name=f"bench-{name}-w{worker_id}"))

    t0 = perf_counter()
    await asyncio.gather(*tasks)
    elapsed = perf_counter() - t0
    return BenchResult(name=name, iterations=total_iterations, elapsed_s=elapsed)


def _print_result(result: BenchResult) -> None:
    print(
        f"{result.name}: "
        f"n={result.iterations} "
        f"elapsed={result.elapsed_s:.4f}s "
        f"throughput={result.ops_per_sec:,.0f} ops/s "
        f"latency={result.us_per_op:.2f} us/op"
    )


def _print_ratio(base: BenchResult, test: BenchResult, *, label: str) -> None:
    if base.elapsed_s <= 0:
        print(f"{label}: ratio unavailable (base elapsed <= 0)")
        return
    ratio = test.elapsed_s / base.elapsed_s
    delta_pct = (ratio - 1.0) * 100.0
    print(f"{label}: slowdown={ratio:.2f}x ({delta_pct:+.1f}%)")


async def main_async(args: argparse.Namespace) -> None:
    inputs: dict[str, Any] = {
        "L0": 0.10,
        "L1": 0.20,
        "L2": 0.30,
        "R0": 0.40,
        "R1": 0.50,
        "R2": 0.60,
        "V0": 0.70,
        "V1": 0.80,
        "A0": 0.90,
        "A1": 1.00,
        "intervalMs": 20,
    }

    tcode_node = _build_tcode_node()
    pyscript_dot_input_view_node = _build_python_script_node(
        node_id="pyscript_dot_input_view",
        code=SCRIPT_CODE_DOT,
        input_mode="input_view",
    )
    pyscript_dot_msgspec_node = _build_python_script_node(
        node_id="pyscript_dot_msgspec",
        code=SCRIPT_CODE_DOT,
        input_mode="msgspec_struct",
    )
    pyscript_mapping_raw_node = _build_python_script_node(
        node_id="pyscript_mapping_raw",
        code=SCRIPT_CODE_MAPPING,
        input_mode="raw_dict",
    )
    expr_node = _build_data_expr_node()
    pyscript_service_node = _build_pyscript_service_node()
    _attach_fixed_inputs(tcode_node, inputs)
    _attach_fixed_inputs(pyscript_dot_input_view_node, inputs)
    _attach_fixed_inputs(pyscript_dot_msgspec_node, inputs)
    _attach_fixed_inputs(pyscript_mapping_raw_node, inputs)
    _attach_fixed_inputs(expr_node, inputs)

    expected = await tcode_node.compute_output("tcode", ctx_id=-1)
    if not isinstance(expected, str):
        raise AssertionError("tcode expected output is not string")

    pyscript_dot_input_view_once = await pyscript_dot_input_view_node.compute_output("tcode", ctx_id=-1)
    if pyscript_dot_input_view_once != expected:
        raise AssertionError("python_script(dot/input_view) output does not match tcode output")
    pyscript_dot_msgspec_once = await pyscript_dot_msgspec_node.compute_output("tcode", ctx_id=-1)
    if pyscript_dot_msgspec_once != expected:
        raise AssertionError("python_script(dot/msgspec) output does not match tcode output")
    pyscript_mapping_raw_once = await pyscript_mapping_raw_node.compute_output("tcode", ctx_id=-1)
    if pyscript_mapping_raw_once != expected:
        raise AssertionError("python_script(mapping/raw_dict) output does not match tcode output")

    async def compute_tcode(ctx: int) -> Any:
        return await tcode_node.compute_output("tcode", ctx_id=ctx)

    async def compute_pyscript_dot_input_view(ctx: int) -> Any:
        return await pyscript_dot_input_view_node.compute_output("tcode", ctx_id=ctx)

    async def compute_pyscript_dot_msgspec(ctx: int) -> Any:
        return await pyscript_dot_msgspec_node.compute_output("tcode", ctx_id=ctx)

    async def compute_pyscript_mapping_raw(ctx: int) -> Any:
        return await pyscript_mapping_raw_node.compute_output("tcode", ctx_id=ctx)

    expected_expr = await expr_node.compute_output("out", ctx_id=-1)

    async def compute_expr(ctx: int) -> Any:
        return await expr_node.compute_output("out", ctx_id=ctx)

    service_payload = dict(inputs)
    service_capture: dict[str, Any] = {"tcode": None}

    async def service_emit_impl(self: Any, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        del self, ts_ms
        if str(port) == "tcode":
            service_capture["tcode"] = value

    pyscript_service_node.emit = MethodType(service_emit_impl, pyscript_service_node)
    await pyscript_service_node.on_data("in", service_payload, ts_ms=0)
    expected_service = service_capture.get("tcode")
    if expected_service != expected:
        raise AssertionError("pyscript service output does not match tcode output")

    async def compute_pyscript_service(ctx: int) -> Any:
        await pyscript_service_node.on_data("in", service_payload, ts_ms=ctx)
        return service_capture.get("tcode")

    print(f"inputs: {len(AXES)} axis + intervalMs, output='tcode'")
    print(f"warmup={args.warmup} serial_iterations={args.iterations}")

    serial_tcode = await _run_bench(
        name="tcode(serial)",
        compute_one=compute_tcode,
        iterations=args.iterations,
        warmup=args.warmup,
        expected=expected,
    )
    serial_pyscript_dot_input_view = await _run_bench(
        name="python_script_dot_input_view(serial)",
        compute_one=compute_pyscript_dot_input_view,
        iterations=args.iterations,
        warmup=args.warmup,
        expected=expected,
    )
    serial_pyscript_dot_msgspec = await _run_bench(
        name="python_script_dot_msgspec(serial)",
        compute_one=compute_pyscript_dot_msgspec,
        iterations=args.iterations,
        warmup=args.warmup,
        expected=expected,
    )
    serial_pyscript_mapping_raw = await _run_bench(
        name="python_script_mapping_raw(serial)",
        compute_one=compute_pyscript_mapping_raw,
        iterations=args.iterations,
        warmup=args.warmup,
        expected=expected,
    )

    _print_result(serial_tcode)
    _print_result(serial_pyscript_dot_input_view)
    _print_ratio(serial_tcode, serial_pyscript_dot_input_view, label="serial(dot/input_view vs tcode)")
    _print_result(serial_pyscript_dot_msgspec)
    _print_ratio(serial_tcode, serial_pyscript_dot_msgspec, label="serial(dot/msgspec vs tcode)")
    _print_result(serial_pyscript_mapping_raw)
    _print_ratio(serial_tcode, serial_pyscript_mapping_raw, label="serial(mapping/raw_dict vs tcode)")
    serial_expr = await _run_bench(
        name="expr(serial)",
        compute_one=compute_expr,
        iterations=args.iterations,
        warmup=args.warmup,
        expected=expected_expr,
    )
    _print_result(serial_expr)
    _print_ratio(serial_tcode, serial_expr, label="serial(expr vs tcode)")
    serial_pyscript_service = await _run_bench(
        name="pyscript_service(serial)",
        compute_one=compute_pyscript_service,
        iterations=args.iterations,
        warmup=args.warmup,
        expected=expected_service,
    )
    _print_result(serial_pyscript_service)
    _print_ratio(serial_tcode, serial_pyscript_service, label="serial(pyscript_service vs tcode)")

    if args.concurrent_iterations > 0:
        print(
            f"concurrent_iterations={args.concurrent_iterations} "
            f"concurrency={args.concurrency}"
        )
        concurrent_tcode = await _run_bench_concurrent(
            name="tcode(concurrent)",
            compute_one=compute_tcode,
            total_iterations=args.concurrent_iterations,
            warmup=max(1, args.warmup // 2),
            concurrency=args.concurrency,
            expected=expected,
        )
        concurrent_pyscript_dot_input_view = await _run_bench_concurrent(
            name="python_script_dot_input_view(concurrent)",
            compute_one=compute_pyscript_dot_input_view,
            total_iterations=args.concurrent_iterations,
            warmup=max(1, args.warmup // 2),
            concurrency=args.concurrency,
            expected=expected,
        )
        concurrent_pyscript_dot_msgspec = await _run_bench_concurrent(
            name="python_script_dot_msgspec(concurrent)",
            compute_one=compute_pyscript_dot_msgspec,
            total_iterations=args.concurrent_iterations,
            warmup=max(1, args.warmup // 2),
            concurrency=args.concurrency,
            expected=expected,
        )
        concurrent_pyscript_mapping_raw = await _run_bench_concurrent(
            name="python_script_mapping_raw(concurrent)",
            compute_one=compute_pyscript_mapping_raw,
            total_iterations=args.concurrent_iterations,
            warmup=max(1, args.warmup // 2),
            concurrency=args.concurrency,
            expected=expected,
        )
        _print_result(concurrent_tcode)
        _print_result(concurrent_pyscript_dot_input_view)
        _print_ratio(concurrent_tcode, concurrent_pyscript_dot_input_view, label="concurrent(dot/input_view vs tcode)")
        _print_result(concurrent_pyscript_dot_msgspec)
        _print_ratio(concurrent_tcode, concurrent_pyscript_dot_msgspec, label="concurrent(dot/msgspec vs tcode)")
        _print_result(concurrent_pyscript_mapping_raw)
        _print_ratio(concurrent_tcode, concurrent_pyscript_mapping_raw, label="concurrent(mapping/raw_dict vs tcode)")
        concurrent_expr = await _run_bench_concurrent(
            name="expr(concurrent)",
            compute_one=compute_expr,
            total_iterations=args.concurrent_iterations,
            warmup=max(1, args.warmup // 2),
            concurrency=args.concurrency,
            expected=expected_expr,
        )
        _print_result(concurrent_expr)
        _print_ratio(concurrent_tcode, concurrent_expr, label="concurrent(expr vs tcode)")
        concurrent_pyscript_service = await _run_bench_concurrent(
            name="pyscript_service(concurrent)",
            compute_one=compute_pyscript_service,
            total_iterations=args.concurrent_iterations,
            warmup=max(1, args.warmup // 2),
            concurrency=args.concurrency,
            expected=expected_service,
        )
        _print_result(concurrent_pyscript_service)
        _print_ratio(
            concurrent_tcode,
            concurrent_pyscript_service,
            label="concurrent(pyscript_service vs tcode)",
        )

    await pyscript_service_node.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark: f8.tcode vs f8.python_script (dot/msgspec and mapping/raw variants) "
            "+ f8.data_expr overhead reference (not functionally equivalent to TCode)."
        )
    )
    parser.add_argument("--iterations", type=int, default=200_000, help="serial benchmark iterations")
    parser.add_argument("--warmup", type=int, default=2_000, help="warmup iterations")
    parser.add_argument(
        "--concurrent-iterations",
        type=int,
        default=200_000,
        help="total concurrent benchmark iterations (0 to skip)",
    )
    parser.add_argument("--concurrency", type=int, default=32, help="concurrent worker count")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
