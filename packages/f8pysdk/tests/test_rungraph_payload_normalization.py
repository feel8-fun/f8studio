from __future__ import annotations

from typing import Any

import msgspec

from f8pysdk.specs import F8SetRungraphRequest, F8RuntimeGraph, F8RuntimeNode
from f8pysdk.codec import dump_json, validate_as


def _contains_unset(value: Any) -> bool:
    if isinstance(value, msgspec.UnsetType):
        return True
    if isinstance(value, dict):
        return any(_contains_unset(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_unset(item) for item in value)
    if isinstance(value, tuple):
        return any(_contains_unset(item) for item in value)
    return False


def test_dump_json_strips_msgspec_unset_values() -> None:
    graph = F8RuntimeGraph(
        graphId="g1",
        revision="r1",
        nodes=[F8RuntimeNode(nodeId="svc1", serviceId="svc1", serviceClass="svc.a")],
        edges=[],
    )
    payload = dump_json(graph)
    assert isinstance(payload, dict)
    assert _contains_unset(payload) is False


def test_set_rungraph_request_decode_accepts_typed_graph_payload() -> None:
    raw_request = {
        "reqId": "r1",
        "args": {
            "graph": {
                "graphId": "g1",
                "revision": "r1",
                "nodes": [{"nodeId": "svc1", "serviceId": "svc1", "serviceClass": "svc.a"}],
                "edges": [],
            }
        },
        "meta": {"source": "test"},
    }

    request = validate_as(F8SetRungraphRequest, raw_request)
    assert request.reqId == "r1"
    assert request.args.graph.graphId == "g1"
