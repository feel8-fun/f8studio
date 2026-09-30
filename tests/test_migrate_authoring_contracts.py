from __future__ import annotations

import json
from pathlib import Path

from f8studio_core.graph import decode_document, export_graph, import_graph
from scripts.migrate_authoring_contracts import migrate


def test_migration_preserves_state_values_and_edit_policies() -> None:
    document = {
        "nodes": [{
            "spec": {"serviceClass": "demo", "operatorClass": "op", "schemaVersion": "f8operator/1",
                     "execOutPorts": ["exec"], "editPolicy": {"execOutPorts": {"canAdd": True}},
                     "dataOutPorts": [{"name": "out", "valueSchema": {"type": "number"}}]},
            "stateValues": {"dataOutPorts": [{"name": "user", "valueSchema": "opaque"}]},
        }],
    }
    migrate(document)
    node = document["nodes"][0]
    assert node["spec"]["execOutPorts"] == [{"name": "exec"}]
    assert node["spec"]["editPolicy"] == {"execOutPorts": {"canAdd": True}}
    assert node["stateValues"] == {"dataOutPorts": [{"name": "user", "valueSchema": "opaque"}]}


def test_exchange_migration_rehashes_definitions_and_is_idempotent() -> None:
    example = Path(__file__).parents[1] / "packages/f8studio_core/examples/basic-pipeline.f8studio.json"
    document = decode_document(example.read_bytes())
    exchange = json.loads(export_graph(document))
    for spec in exchange["definitions"]["operators"].values():
        for port in [*spec.get("dataInPorts", []), *spec.get("dataOutPorts", [])]:
            port["valueSchema"] = port.pop("payload")["valueSchema"]
    migrate(exchange)
    restored = import_graph(json.dumps(exchange))
    assert {node.node_id for node in restored.nodes} == {node.node_id for node in document.nodes}
    first = json.dumps(exchange, sort_keys=True)
    migrate(exchange)
    assert json.dumps(exchange, sort_keys=True) == first
