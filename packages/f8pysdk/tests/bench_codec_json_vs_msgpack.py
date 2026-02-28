from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from time import perf_counter_ns
from typing import Any

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
from f8pysdk.schema_helpers import any_schema, string_schema  # noqa: E402
from f8pysdk.service_bus.codec import decode_obj as decode_msgpack_obj  # noqa: E402
from f8pysdk.service_bus.codec import encode_obj as encode_msgpack_obj  # noqa: E402


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return float(values[0])
    if q >= 1.0:
        return float(values[-1])
    index = int(math.ceil(q * len(values))) - 1
    index = max(0, min(index, len(values) - 1))
    return float(values[index])


def _payload_small() -> dict[str, Any]:
    return {
        "value": {
            "x": 1.25,
            "y": -3.0,
            "id": "op1",
            "ok": True,
            "tags": ["pose", "realtime", "cam"],
        },
        "ts": 1730000000123,
        "meta": {"source": "bench", "serviceId": "svcA", "nodeId": "op1"},
    }


def _payload_medium() -> dict[str, Any]:
    random.seed(42)
    people: list[dict[str, Any]] = []
    for person_i in range(8):
        nodes: list[dict[str, Any]] = []
        for node_i in range(33):
            nodes.append(
                {
                    "id": node_i,
                    "x": random.random(),
                    "y": random.random(),
                    "z": (random.random() - 0.5) * 2.0,
                    "score": random.random(),
                }
            )
        people.append({"personId": person_i, "nodes": nodes})
    return {"value": {"frameId": 1203, "people": people}, "ts": 1730000000456}


def _payload_large() -> dict[str, Any]:
    random.seed(7)
    frames: list[dict[str, Any]] = []
    for frame_i in range(40):
        points: list[dict[str, Any]] = []
        for point_i in range(160):
            points.append(
                {
                    "i": point_i,
                    "x": random.random(),
                    "y": random.random(),
                    "v": random.random(),
                }
            )
        frames.append({"frameId": frame_i, "points": points})
    return {"value": {"sessionId": "bench-large", "frames": frames}, "ts": 1730000000999}


def _state_spec(name: str) -> F8StateSpec:
    return F8StateSpec(name=name, valueSchema=string_schema(), access=F8StateAccess.rw)


def _build_rungraph_payload(node_count: int) -> dict[str, Any]:
    nodes: list[F8RuntimeNode] = []
    edges: list[F8Edge] = []
    for i in range(node_count):
        node_id = f"op{i}"
        nodes.append(
            F8RuntimeNode(
                nodeId=node_id,
                serviceId="svcA",
                serviceClass="bench.service",
                operatorClass="BenchOp",
                stateFields=[_state_spec("value"), F8StateSpec(name="raw", valueSchema=any_schema(), access=F8StateAccess.rw)],
                stateValues={"value": f"seed-{i}", "raw": {"k": i, "v": [i, i + 1, i + 2]}},
            )
        )
        if i > 0:
            edges.append(
                F8Edge(
                    edgeId=f"e{i}",
                    fromServiceId="svcA",
                    fromOperatorId=f"op{i - 1}",
                    fromPort="value",
                    toServiceId="svcA",
                    toOperatorId=node_id,
                    toPort="value",
                    kind=F8EdgeKindEnum.state,
                    strategy=F8EdgeStrategyEnum.latest,
                )
            )
    graph = F8RuntimeGraph(graphId="g-bench", revision="r1", nodes=nodes, edges=edges)
    return graph.model_dump(mode="json", by_alias=True)


def _bench_codec(
    name: str,
    payload: dict[str, Any],
    *,
    iterations: int,
    warmup: int,
) -> dict[str, Any]:
    json_enc_ns: list[float] = []
    json_dec_ns: list[float] = []
    mp_enc_ns: list[float] = []
    mp_dec_ns: list[float] = []
    json_sizes: list[int] = []
    mp_sizes: list[int] = []

    for i in range(iterations + warmup):
        t0 = perf_counter_ns()
        json_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        t1 = perf_counter_ns()
        _ = json.loads(json_bytes.decode("utf-8"))
        t2 = perf_counter_ns()

        t3 = perf_counter_ns()
        mp_bytes = encode_msgpack_obj(payload)
        t4 = perf_counter_ns()
        _ = decode_msgpack_obj(mp_bytes)
        t5 = perf_counter_ns()

        if i < warmup:
            continue

        json_enc_ns.append(float(t1 - t0))
        json_dec_ns.append(float(t2 - t1))
        mp_enc_ns.append(float(t4 - t3))
        mp_dec_ns.append(float(t5 - t4))
        json_sizes.append(len(json_bytes))
        mp_sizes.append(len(mp_bytes))

    json_total_ns = [a + b for a, b in zip(json_enc_ns, json_dec_ns)]
    mp_total_ns = [a + b for a, b in zip(mp_enc_ns, mp_dec_ns)]

    def _stat_row(v: list[float]) -> dict[str, float]:
        v2 = sorted(v)
        return {
            "avg_us": statistics.fmean(v2) / 1000.0,
            "p50_us": _quantile(v2, 0.50) / 1000.0,
            "p95_us": _quantile(v2, 0.95) / 1000.0,
            "p99_us": _quantile(v2, 0.99) / 1000.0,
        }

    return {
        "name": name,
        "json_encode": _stat_row(json_enc_ns),
        "json_decode": _stat_row(json_dec_ns),
        "json_total": _stat_row(json_total_ns),
        "msgpack_encode": _stat_row(mp_enc_ns),
        "msgpack_decode": _stat_row(mp_dec_ns),
        "msgpack_total": _stat_row(mp_total_ns),
        "json_size_avg": int(round(statistics.fmean(json_sizes))),
        "msgpack_size_avg": int(round(statistics.fmean(mp_sizes))),
    }


def _print_one(result: dict[str, Any]) -> None:
    name = str(result["name"])
    js = int(result["json_size_avg"])
    ms = int(result["msgpack_size_avg"])
    shrink = (1.0 - (float(ms) / float(js))) * 100.0 if js > 0 else 0.0
    print(f"\n[{name}]")
    print(f"  size(bytes): json={js} msgpack={ms} delta={shrink:+.1f}%")
    print(
        "  encode(us): json(avg/p95)={:.1f}/{:.1f}  msgpack(avg/p95)={:.1f}/{:.1f}".format(
            result["json_encode"]["avg_us"],
            result["json_encode"]["p95_us"],
            result["msgpack_encode"]["avg_us"],
            result["msgpack_encode"]["p95_us"],
        )
    )
    print(
        "  decode(us): json(avg/p95)={:.1f}/{:.1f}  msgpack(avg/p95)={:.1f}/{:.1f}".format(
            result["json_decode"]["avg_us"],
            result["json_decode"]["p95_us"],
            result["msgpack_decode"]["avg_us"],
            result["msgpack_decode"]["p95_us"],
        )
    )
    print(
        "  total(us):  json(avg/p95)={:.1f}/{:.1f}  msgpack(avg/p95)={:.1f}/{:.1f}".format(
            result["json_total"]["avg_us"],
            result["json_total"]["p95_us"],
            result["msgpack_total"]["avg_us"],
            result["msgpack_total"]["p95_us"],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare JSON vs MsgPack codec overhead for f8 runtime-like payloads.")
    parser.add_argument("--iterations", type=int, default=2000, help="measured iterations per scenario")
    parser.add_argument("--warmup", type=int, default=300, help="warmup iterations per scenario")
    parser.add_argument("--rungraph-nodes", type=int, default=200, help="node count for rungraph scenario")
    args = parser.parse_args()

    scenarios: list[tuple[str, dict[str, Any]]] = [
        ("small-data-message", _payload_small()),
        ("medium-pose-message", _payload_medium()),
        ("large-batch-message", _payload_large()),
        ("rungraph-model-dump", _build_rungraph_payload(int(args.rungraph_nodes))),
    ]

    print("JSON vs MsgPack benchmark")
    print(f"iterations={int(args.iterations)} warmup={int(args.warmup)} rungraph_nodes={int(args.rungraph_nodes)}")

    for name, payload in scenarios:
        result = _bench_codec(name, payload, iterations=int(args.iterations), warmup=int(args.warmup))
        _print_one(result)


if __name__ == "__main__":
    main()
