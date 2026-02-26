from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from f8pysdk.generated import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
)
from f8pysdk.schema_helpers import any_schema

from f8pystudio.nodegraph.node_graph import (
    F8StudioGraph,
    _MISSING_CONNECTION_SNAPSHOTS_KEY,
    _MISSING_OPERATOR_NODE_TYPE,
    _MISSING_SERVICE_NODE_TYPE,
)


def _new_graph_with_registry(registered_types: dict[str, Any]) -> F8StudioGraph:
    graph = F8StudioGraph.__new__(F8StudioGraph)
    graph._node_factory = SimpleNamespace(nodes=dict(registered_types))
    graph._missing_connection_snapshots = []
    return graph


def _service_spec_payload(service_class: str) -> dict[str, Any]:
    return F8ServiceSpec(
        schemaVersion=F8ServiceSchemaVersion.f8service_1,
        serviceClass=service_class,
        version="0.0.1",
        label="Service",
    ).model_dump(mode="json")


def _operator_spec_payload(
    service_class: str,
    operator_class: str,
    *,
    exec_in: list[str] | None = None,
    exec_out: list[str] | None = None,
    data_in: list[str] | None = None,
    data_out: list[str] | None = None,
) -> dict[str, Any]:
    return F8OperatorSpec(
        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
        serviceClass=service_class,
        operatorClass=operator_class,
        version="0.0.1",
        label="Operator",
        execInPorts=list(exec_in or []),
        execOutPorts=list(exec_out or []),
        dataInPorts=[F8DataPortSpec(name=n, valueSchema=any_schema()) for n in list(data_in or [])],
        dataOutPorts=[F8DataPortSpec(name=n, valueSchema=any_schema()) for n in list(data_out or [])],
    ).model_dump(mode="json")


def test_restore_missing_placeholder_node_when_original_type_is_registered() -> None:
    graph = _new_graph_with_registry({"svc.real.operator": object()})
    layout = {
        "nodes": {
            "n1": {
                "type_": _MISSING_OPERATOR_NODE_TYPE,
                "f8_spec": {"serviceClass": "f8.missing", "operatorClass": "f8.missing.operator"},
                "f8_sys": {
                    "missingType": "svc.real.operator",
                    "missingSpec": {"serviceClass": "f8.real", "operatorClass": "f8.real.operator"},
                },
                "custom": {"missingType": "svc.real.operator", "missingSpec": "{}"},
            }
        }
    }

    out = graph._restore_missing_session_nodes(layout)
    node = out["nodes"]["n1"]
    assert node["type_"] == "svc.real.operator"
    assert node["f8_spec"] == {"serviceClass": "f8.real", "operatorClass": "f8.real.operator"}
    assert node["custom"] == {}


def test_merge_and_save_missing_connection_snapshots_roundtrip() -> None:
    graph = _new_graph_with_registry({})
    graph._missing_connection_snapshots = [{"out": ["m1", "a[D]"], "in": ["n1", "[D]b"]}]

    save_layout = {
        "nodes": {
            "m1": {"type_": _MISSING_OPERATOR_NODE_TYPE},
            "n1": {"type_": "svc.real.node"},
        },
        "connections": [],
    }
    save_out = graph._attach_missing_connection_snapshots_for_save(save_layout)
    assert _MISSING_CONNECTION_SNAPSHOTS_KEY in save_out
    assert save_out[_MISSING_CONNECTION_SNAPSHOTS_KEY] == [{"out": ["m1", "a[D]"], "in": ["n1", "[D]b"]}]

    load_layout = {
        "nodes": {
            "m1": {"type_": _MISSING_OPERATOR_NODE_TYPE},
            "n1": {"type_": "svc.real.node"},
        },
        "connections": [],
        _MISSING_CONNECTION_SNAPSHOTS_KEY: [{"out": ["m1", "a[D]"], "in": ["n1", "[D]b"]}],
    }
    load_out = graph._merge_missing_connection_snapshots(load_layout)
    assert load_out["connections"] == [{"out": ["m1", "a[D]"], "in": ["n1", "[D]b"]}]
    assert _MISSING_CONNECTION_SNAPSHOTS_KEY not in load_out
    assert graph._missing_connection_snapshots == [{"out": ["m1", "a[D]"], "in": ["n1", "[D]b"]}]


def test_merge_snapshots_clears_cached_missing_connections_when_no_missing_nodes() -> None:
    graph = _new_graph_with_registry({})
    layout = {
        "nodes": {
            "n1": {"type_": "svc.real.node"},
            "n2": {"type_": "svc.real.node2"},
        },
        "connections": [],
        _MISSING_CONNECTION_SNAPSHOTS_KEY: [{"out": ["n1", "x[D]"], "in": ["n2", "[D]y"]}],
    }

    out = graph._merge_missing_connection_snapshots(layout)
    assert out["connections"] == [{"out": ["n1", "x[D]"], "in": ["n2", "[D]y"]}]
    assert graph._missing_connection_snapshots == []


def test_save_clears_snapshot_field_when_layout_has_no_missing_nodes() -> None:
    graph = _new_graph_with_registry({})
    graph._missing_connection_snapshots = [{"out": ["x", "a[D]"], "in": ["y", "[D]b"]}]
    layout = {
        "nodes": {
            "n1": {"type_": "svc.real.node"},
        },
        "connections": [],
    }

    out = graph._attach_missing_connection_snapshots_for_save(layout)
    assert _MISSING_CONNECTION_SNAPSHOTS_KEY not in out
    assert graph._missing_connection_snapshots == []


def test_coerce_missing_operator_uses_unified_missing_service_class() -> None:
    graph = _new_graph_with_registry({_MISSING_OPERATOR_NODE_TYPE: object(), _MISSING_SERVICE_NODE_TYPE: object()})
    # Simulate an unregistered operator type from session.
    layout = {
        "nodes": {
            "n1": {
                "type_": "svc.unknown.realnode",
                "f8_spec": {
                    "serviceClass": "svc.unknown",
                    "operatorClass": "unknown.op",
                },
            }
        }
    }
    # Hide all real types to force coercion.
    graph._node_factory = SimpleNamespace(nodes={})

    out = graph._coerce_missing_session_nodes(layout)
    node = out["nodes"]["n1"]
    assert node["type_"] == _MISSING_OPERATOR_NODE_TYPE
    assert node["f8_spec"]["serviceClass"] == "f8.missing"


def test_strip_invalid_connections_drops_cross_service_exec_and_mixed_kind() -> None:
    layout = {
        "nodes": {
            "svc1": {"type_": "svc.f8.pyengine", "f8_spec": _service_spec_payload("f8.pyengine")},
            "svc2": {"type_": "svc.f8.pyengine", "f8_spec": _service_spec_payload("f8.pyengine")},
            "op1": {
                "type_": "svc.f8.pyengine.op",
                "f8_spec": _operator_spec_payload(
                    "f8.pyengine",
                    "f8.pyengine.op1",
                    exec_out=["next"],
                    data_out=["out"],
                ),
                "custom": {"svcId": "svc1"},
            },
            "op2": {
                "type_": "svc.f8.pyengine.op",
                "f8_spec": _operator_spec_payload(
                    "f8.pyengine",
                    "f8.pyengine.op2",
                    exec_in=["in"],
                    data_in=["din"],
                ),
                "custom": {"svcId": "svc2"},
            },
            "op3": {
                "type_": "svc.f8.pyengine.op",
                "f8_spec": _operator_spec_payload(
                    "f8.pyengine",
                    "f8.pyengine.op3",
                    exec_in=["in"],
                    data_in=["din"],
                ),
                "custom": {"svcId": "svc1"},
            },
        },
        "connections": [
            {"out": ["op1", "next[E]"], "in": ["op2", "[E]in"]},
            {"out": ["op1", "next[E]"], "in": ["op3", "[D]din"]},
            {"out": ["op1", "next[E]"], "in": ["op3", "[E]in"]},
            {"out": ["op1", "out[D]"], "in": ["op2", "[D]din"]},
        ],
    }
    out = F8StudioGraph._strip_invalid_connections(layout)
    assert out["connections"] == [
        {"out": ["op1", "next[E]"], "in": ["op3", "[E]in"]},
        {"out": ["op1", "out[D]"], "in": ["op2", "[D]din"]},
    ]
