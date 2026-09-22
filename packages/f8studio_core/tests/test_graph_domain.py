from __future__ import annotations

from pathlib import Path

import msgspec
import pytest

from f8pysdk.command import command_input_state_field
from f8pysdk.specs import (
    F8Command,
    F8EdgeDirection,
    F8EdgeKindEnum,
    F8OperatorSpec,
    F8ServiceDescribe,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    json_data_port,
    number_schema,
    string_schema,
    video_frame_port,
)
from f8studio_core import compile_document, semantic_graph_revision
from f8studio_core.graph import (
    ConnectEdgeOp,
    CreateNodeOp,
    EdgeStrategy,
    GraphEdge,
    GraphEdgeKind,
    GraphStore,
    GraphValidationError,
    HistoryRequest,
    IdempotencyConflictError,
    InsertFragmentOp,
    NodeCatalog,
    NodeLayout,
    OperatorNode,
    PatchRequest,
    PortDirection,
    PortKind,
    RenameNodeOp,
    ReplaceNodeOp,
    RevisionConflictError,
    ServiceNode,
    SetNodeLayoutOp,
    SetNodeStateOp,
    StudioDocument,
    decode_document,
    encode_document,
    new_document,
    replace_node_spec,
)


def build_catalog() -> NodeCatalog:
    return NodeCatalog(
        services=[
            F8ServiceSpec(serviceClass="f8.pyengine", label="Python Engine"),
            F8ServiceSpec(serviceClass="f8.pystudio", label="Web Studio"),
        ],
        operators=[
            F8OperatorSpec(
                serviceClass="f8.pyengine",
                operatorClass="test.source",
                label="Source",
                execOutPorts=["next"],
                dataOutPorts=[json_data_port(name="out", value_schema=number_schema())],
                stateFields=[
                    F8StateSpec(
                        name="gain",
                        valueSchema=number_schema(default=1.0),
                        access=F8StateAccess.rw,
                    )
                ],
            ),
            F8OperatorSpec(
                serviceClass="f8.pyengine",
                operatorClass="test.sink",
                label="Sink",
                execInPorts=["run"],
                dataInPorts=[json_data_port(name="input", value_schema=number_schema())],
                stateFields=[
                    F8StateSpec(
                        name="gain",
                        valueSchema=number_schema(default=0.0),
                        access=F8StateAccess.rw,
                    )
                ],
                commands=[F8Command(name="Reset")],
            ),
            F8OperatorSpec(
                serviceClass="f8.pyengine",
                operatorClass="f8.patch_hub",
                label="Patch Hub",
                dataInPorts=[json_data_port(name="signal", value_schema=number_schema())],
                dataOutPorts=[json_data_port(name="signal", value_schema=number_schema())],
            ),
            F8OperatorSpec(
                serviceClass="f8.pystudio",
                operatorClass="f8.viz.text",
                label="Text",
                dataInPorts=[json_data_port(name="input", value_schema=number_schema())],
                stateFields=[
                    F8StateSpec(
                        name="upstreamSamplingMode",
                        valueSchema=string_schema(default="auto"),
                        access=F8StateAccess.rw,
                    ),
                    F8StateSpec(
                        name="upstreamSampleIntervalMs",
                        valueSchema=number_schema(default=100),
                        access=F8StateAccess.rw,
                    ),
                ],
            ),
        ],
    )


def find_port_id(node: ServiceNode | OperatorNode, *, name: str, kind: PortKind, direction: PortDirection) -> str:
    for port in node.ports:
        if port.name == name and port.kind == kind and port.direction == direction:
            return port.port_id
    raise AssertionError(f"port not found: {name}/{kind}/{direction}")


def base_nodes() -> tuple[NodeCatalog, ServiceNode, OperatorNode, OperatorNode]:
    catalog = build_catalog()
    service = catalog.create_service_node(node_id="engine", service_class="f8.pyengine")
    source = catalog.create_operator_node(
        node_id="source",
        service_id="engine",
        service_class="f8.pyengine",
        operator_class="test.source",
        state_values={"gain": 2.0},
    )
    sink = catalog.create_operator_node(
        node_id="sink",
        service_id="engine",
        service_class="f8.pyengine",
        operator_class="test.sink",
        state_values={"gain": 3.0},
    )
    return catalog, service, source, sink


def test_document_codec_preserves_tagged_node_types() -> None:
    _, service, source, _ = base_nodes()
    document = StudioDocument(
        schema_version="f8studio-document/1",
        project_id="project1",
        graph_id="graph1",
        graph_revision=1,
        layout_revision=0,
        nodes=(service, source),
    )

    decoded = decode_document(encode_document(document))

    assert decoded == document
    assert isinstance(decoded.nodes[0], ServiceNode)
    assert isinstance(decoded.nodes[1], OperatorNode)


def test_document_decoder_rejects_semantically_invalid_documents() -> None:
    document = new_document(project_id="project1", graph_id="graph1")
    invalid = msgspec.structs.replace(document, schema_version="f8studio-document/999")

    with pytest.raises(GraphValidationError) as error:
        decode_document(encode_document(invalid))
    assert error.value.code == "unsupported_document_version"


def test_repository_example_loads_and_compiles() -> None:
    example_path = Path(__file__).parents[1] / "examples" / "basic-pipeline.f8studio.json"

    document = decode_document(example_path.read_bytes())
    compiled = compile_document(document)

    assert document.project_id == "example_project"
    assert [edge.edgeId for edge in compiled.global_graph.edges] == ["source_to_sink"]
    assert set(compiled.per_service) == {"engine"}


def test_graph_store_applies_atomically_and_keeps_revisions_separate() -> None:
    _, service, source, sink = base_nodes()
    store = GraphStore(new_document(project_id="project1", graph_id="graph1"))
    created = store.apply(
        PatchRequest(
            request_id="create",
            expected_graph_revision=0,
            expected_layout_revision=0,
            operations=(
                CreateNodeOp(node=service, layout=NodeLayout(node_id="engine", x=0, y=0)),
                CreateNodeOp(node=source, layout=NodeLayout(node_id="source", x=100, y=0)),
                CreateNodeOp(node=sink, layout=NodeLayout(node_id="sink", x=200, y=0)),
            ),
        )
    )
    assert (created.document.graph_revision, created.document.layout_revision) == (1, 1)

    moved = store.apply(
        PatchRequest(
            request_id="move",
            expected_graph_revision=1,
            expected_layout_revision=1,
            operations=(SetNodeLayoutOp(layout=NodeLayout(node_id="source", x=150, y=20)),),
        )
    )
    assert (moved.document.graph_revision, moved.document.layout_revision) == (1, 2)

    invalid_edge = GraphEdge(
        edge_id="broken",
        from_node_id="source",
        from_port_id="data:output:missing",
        to_node_id="sink",
        to_port_id=find_port_id(sink, name="input", kind=PortKind.data, direction=PortDirection.input),
        kind=GraphEdgeKind.data,
    )
    with pytest.raises(GraphValidationError, match="missing port"):
        store.apply(
            PatchRequest(
                request_id="invalid",
                expected_graph_revision=1,
                expected_layout_revision=2,
                operations=(ConnectEdgeOp(edge=invalid_edge),),
            )
        )
    assert store.snapshot() == moved.document


def test_graph_store_does_not_commit_when_persistence_callback_fails() -> None:
    _, service, _, _ = base_nodes()
    initial = new_document(project_id="project1", graph_id="graph1")
    store = GraphStore(initial)

    def fail_persistence(_result: object) -> None:
        raise OSError("disk unavailable")

    with pytest.raises(OSError, match="disk unavailable"):
        store.apply_with_commit(
            PatchRequest(
                request_id="create",
                expected_graph_revision=0,
                expected_layout_revision=0,
                operations=(CreateNodeOp(node=service),),
            ),
            before_commit=fail_persistence,
        )

    assert store.snapshot() == initial
    recovered = store.apply(
        PatchRequest(
            request_id="create",
            expected_graph_revision=0,
            expected_layout_revision=0,
            operations=(CreateNodeOp(node=service),),
        )
    )
    assert recovered.document.graph_revision == 1


def test_graph_store_idempotency_conflicts_and_history_are_explicit() -> None:
    _, service, _, _ = base_nodes()
    store = GraphStore(new_document(project_id="project1", graph_id="graph1"))
    request = PatchRequest(
        request_id="create",
        expected_graph_revision=0,
        expected_layout_revision=0,
        operations=(CreateNodeOp(node=service),),
    )
    first = store.apply(request)
    assert store.apply(request) == first

    with pytest.raises(IdempotencyConflictError):
        store.apply(
            PatchRequest(
                request_id="create",
                expected_graph_revision=1,
                expected_layout_revision=0,
                operations=(RenameNodeOp(node_id="engine", name="Changed"),),
            )
        )
    with pytest.raises(RevisionConflictError):
        store.apply(
            PatchRequest(
                request_id="stale",
                expected_graph_revision=0,
                expected_layout_revision=0,
                operations=(RenameNodeOp(node_id="engine", name="Changed"),),
            )
        )

    undone = store.undo(HistoryRequest(request_id="undo1", expected_graph_revision=1, expected_layout_revision=0))
    assert undone.document.nodes == ()
    assert undone.document.graph_revision == 2
    redone = store.redo(HistoryRequest(request_id="redo1", expected_graph_revision=2, expected_layout_revision=0))
    assert len(redone.document.nodes) == 1
    assert redone.document.graph_revision == 3


def test_compiler_maps_all_edge_kinds_and_is_deterministic() -> None:
    _, service, source, sink = base_nodes()
    edges = (
        GraphEdge(
            edge_id="data_edge",
            from_node_id="source",
            from_port_id=find_port_id(source, name="out", kind=PortKind.data, direction=PortDirection.output),
            to_node_id="sink",
            to_port_id=find_port_id(sink, name="input", kind=PortKind.data, direction=PortDirection.input),
            kind=GraphEdgeKind.data,
            strategy=EdgeStrategy.queue,
            queue_size=4,
        ),
        GraphEdge(
            edge_id="exec_edge",
            from_node_id="source",
            from_port_id=find_port_id(source, name="next", kind=PortKind.exec, direction=PortDirection.output),
            to_node_id="sink",
            to_port_id=find_port_id(sink, name="run", kind=PortKind.exec, direction=PortDirection.input),
            kind=GraphEdgeKind.exec,
        ),
        GraphEdge(
            edge_id="state_edge",
            from_node_id="source",
            from_port_id=find_port_id(source, name="gain", kind=PortKind.state, direction=PortDirection.output),
            to_node_id="sink",
            to_port_id=find_port_id(sink, name="gain", kind=PortKind.state, direction=PortDirection.input),
            kind=GraphEdgeKind.state,
        ),
        GraphEdge(
            edge_id="command_edge",
            from_node_id="source",
            from_port_id=find_port_id(source, name="gain", kind=PortKind.state, direction=PortDirection.output),
            to_node_id="sink",
            to_port_id=find_port_id(sink, name="Reset", kind=PortKind.command, direction=PortDirection.input),
            kind=GraphEdgeKind.state,
        ),
    )
    document = StudioDocument(
        schema_version="f8studio-document/1",
        project_id="project1",
        graph_id="graph1",
        graph_revision=8,
        layout_revision=12,
        nodes=(sink, service, source),
        edges=tuple(reversed(edges)),
    )

    compiled = compile_document(document)
    compiled_again = compile_document(msgspec.structs.replace(document, layout_revision=99))

    assert compiled.global_graph.revision == semantic_graph_revision(document)
    assert compiled_again.global_graph.revision == compiled.global_graph.revision
    assert [edge.edgeId for edge in compiled.global_graph.edges] == sorted(edge.edge_id for edge in edges)
    command_edge = next(edge for edge in compiled.global_graph.edges if edge.edgeId == "command_edge")
    assert command_edge.toPort == command_input_state_field("Reset")
    sink_runtime = next(node for node in compiled.global_graph.nodes if node.nodeId == "sink")
    assert isinstance(sink_runtime.stateValues, msgspec.UnsetType)


def test_semantic_revision_ignores_collection_and_mapping_insertion_order() -> None:
    catalog = NodeCatalog(
        services=[F8ServiceSpec(serviceClass="f8.pyengine", label="Engine")],
        operators=[
            F8OperatorSpec(
                serviceClass="f8.pyengine",
                operatorClass="test.configured",
                label="Configured",
                stateFields=[
                    F8StateSpec(name="first", valueSchema=number_schema(), access=F8StateAccess.rw),
                    F8StateSpec(name="second", valueSchema=number_schema(), access=F8StateAccess.rw),
                ],
            )
        ],
    )
    service = catalog.create_service_node(node_id="engine", service_class="f8.pyengine")
    first = catalog.create_operator_node(
        node_id="configured",
        service_id="engine",
        service_class="f8.pyengine",
        operator_class="test.configured",
        state_values={"first": 1.0, "second": 2.0},
    )
    second = msgspec.structs.replace(first, state_values={"second": 2.0, "first": 1.0})
    document = StudioDocument(
        schema_version="f8studio-document/1",
        project_id="project1",
        graph_id="graph1",
        graph_revision=1,
        layout_revision=1,
        nodes=(service, first),
    )
    reordered = msgspec.structs.replace(document, nodes=(second, service), graph_revision=99, layout_revision=99)

    assert semantic_graph_revision(document) == semantic_graph_revision(reordered)


def test_compiler_uses_repository_service_catalog_fixture() -> None:
    describe_path = Path(__file__).parents[3] / "services" / "f8" / "engine" / "describe.json"
    describe = msgspec.json.decode(describe_path.read_bytes(), type=F8ServiceDescribe)
    catalog = NodeCatalog(services=[describe.service], operators=describe.operators)
    service = catalog.create_service_node(node_id="engine", service_class="f8.pyengine")
    phase = catalog.create_operator_node(
        node_id="phase",
        service_id="engine",
        service_class="f8.pyengine",
        operator_class="f8.phase",
    )
    cosine = catalog.create_operator_node(
        node_id="cosine",
        service_id="engine",
        service_class="f8.pyengine",
        operator_class="f8.cosine",
    )
    document = StudioDocument(
        schema_version="f8studio-document/1",
        project_id="catalog_fixture",
        graph_id="phase_to_cosine",
        graph_revision=0,
        layout_revision=0,
        nodes=(service, phase, cosine),
        edges=(
            GraphEdge(
                edge_id="phase_value",
                from_node_id="phase",
                from_port_id=find_port_id(
                    phase, name="phase", kind=PortKind.data, direction=PortDirection.output
                ),
                to_node_id="cosine",
                to_port_id=find_port_id(
                    cosine, name="phase", kind=PortKind.data, direction=PortDirection.input
                ),
                kind=GraphEdgeKind.data,
            ),
        ),
    )

    compiled = compile_document(document)

    assert [service.serviceClass for service in compiled.global_graph.services] == ["f8.pyengine"]
    assert {node.nodeId for node in compiled.global_graph.nodes} == {"engine", "phase", "cosine"}


def test_compiler_lowers_patch_hub_and_splits_cross_service_edge() -> None:
    catalog, service, source, sink = base_nodes()
    studio_service = catalog.create_service_node(node_id="studio", service_class="f8.pystudio")
    viz = catalog.create_operator_node(
        node_id="viz",
        service_id="studio",
        service_class="f8.pystudio",
        operator_class="f8.viz.text",
        state_values={"upstreamSamplingMode": "auto", "upstreamSampleIntervalMs": 25.0},
    )
    hub = catalog.create_operator_node(
        node_id="hub",
        service_id="engine",
        service_class="f8.pyengine",
        operator_class="f8.patch_hub",
    )
    document = StudioDocument(
        schema_version="f8studio-document/1",
        project_id="project1",
        graph_id="graph1",
        graph_revision=1,
        layout_revision=0,
        nodes=(service, source, hub, sink, studio_service, viz),
        edges=(
            GraphEdge(
                edge_id="to_hub",
                from_node_id="source",
                from_port_id=find_port_id(source, name="out", kind=PortKind.data, direction=PortDirection.output),
                to_node_id="hub",
                to_port_id=find_port_id(hub, name="signal", kind=PortKind.data, direction=PortDirection.input),
                kind=GraphEdgeKind.data,
            ),
            GraphEdge(
                edge_id="from_hub",
                from_node_id="hub",
                from_port_id=find_port_id(hub, name="signal", kind=PortKind.data, direction=PortDirection.output),
                to_node_id="sink",
                to_port_id=find_port_id(sink, name="input", kind=PortKind.data, direction=PortDirection.input),
                kind=GraphEdgeKind.data,
            ),
            GraphEdge(
                edge_id="to_studio",
                from_node_id="source",
                from_port_id=find_port_id(source, name="out", kind=PortKind.data, direction=PortDirection.output),
                to_node_id="viz",
                to_port_id=find_port_id(viz, name="input", kind=PortKind.data, direction=PortDirection.input),
                kind=GraphEdgeKind.data,
            ),
        ),
    )

    compiled = compile_document(document)

    assert all(node.nodeId != "hub" for node in compiled.global_graph.nodes)
    assert len(compiled.global_graph.edges) == 2
    engine_requests = list(compiled.per_service["engine"].services[0].autoSampleRequests)
    assert len(engine_requests) == 1
    assert engine_requests[0].intervalMs == 25
    outgoing = next(edge for edge in compiled.per_service["engine"].edges if edge.edgeId == "to_studio")
    incoming = next(edge for edge in compiled.per_service["studio"].edges if edge.edgeId == "to_studio")
    assert outgoing.direction == F8EdgeDirection.out
    assert incoming.direction == F8EdgeDirection.in_
    assert outgoing.kind == F8EdgeKindEnum.data


def test_fragment_insertion_and_dynamic_spec_replacement_are_atomic() -> None:
    _, service, source, _ = base_nodes()
    store = GraphStore(new_document(project_id="project1", graph_id="graph1"))
    inserted = store.apply(
        PatchRequest(
            request_id="fragment",
            expected_graph_revision=0,
            expected_layout_revision=0,
            operations=(
                InsertFragmentOp(
                    nodes=(service, source),
                    layout=(
                        NodeLayout(node_id="engine", x=0, y=0),
                        NodeLayout(node_id="source", x=120, y=0),
                    ),
                ),
            ),
        )
    )
    assert (inserted.document.graph_revision, inserted.document.layout_revision) == (1, 1)

    replacement_spec = F8OperatorSpec(
        serviceClass="f8.pyengine",
        operatorClass="test.source",
        label="Source",
        execOutPorts=["next", "alternate"],
        dataOutPorts=[
            json_data_port(name="out", value_schema=number_schema()),
            json_data_port(name="debug", value_schema=string_schema()),
        ],
        stateFields=list(source.spec.stateFields),
    )
    replacement = replace_node_spec(source, replacement_spec)
    updated = store.apply(
        PatchRequest(
            request_id="replace",
            expected_graph_revision=1,
            expected_layout_revision=1,
            operations=(ReplaceNodeOp(node=replacement),),
        )
    )
    updated_source = next(node for node in updated.document.nodes if node.node_id == "source")
    assert find_port_id(
        updated_source,
        name="debug",
        kind=PortKind.data,
        direction=PortDirection.output,
    )
    assert updated_source.state_values == {"gain": 2.0}


def test_invalid_cross_service_exec_is_rejected() -> None:
    catalog = build_catalog()
    service_a = catalog.create_service_node(node_id="engineA", service_class="f8.pyengine")
    service_b = catalog.create_service_node(node_id="engineB", service_class="f8.pyengine")
    source = catalog.create_operator_node(
        node_id="source",
        service_id="engineA",
        service_class="f8.pyengine",
        operator_class="test.source",
    )
    sink = catalog.create_operator_node(
        node_id="sink",
        service_id="engineB",
        service_class="f8.pyengine",
        operator_class="test.sink",
    )
    document = StudioDocument(
        schema_version="f8studio-document/1",
        project_id="project1",
        graph_id="graph1",
        graph_revision=0,
        layout_revision=0,
        nodes=(service_a, service_b, source, sink),
        edges=(
            GraphEdge(
                edge_id="exec_edge",
                from_node_id="source",
                from_port_id=find_port_id(source, name="next", kind=PortKind.exec, direction=PortDirection.output),
                to_node_id="sink",
                to_port_id=find_port_id(sink, name="run", kind=PortKind.exec, direction=PortDirection.input),
                kind=GraphEdgeKind.exec,
            ),
        ),
    )

    with pytest.raises(GraphValidationError) as error:
        GraphStore(document)
    assert error.value.code == "cross_service_exec"


def test_cyclic_state_edges_are_rejected() -> None:
    _, service, source, sink = base_nodes()
    document = StudioDocument(
        schema_version="f8studio-document/1",
        project_id="project1",
        graph_id="graph1",
        graph_revision=0,
        layout_revision=0,
        nodes=(service, source, sink),
        edges=(
            GraphEdge(
                edge_id="forward",
                from_node_id="source",
                from_port_id=find_port_id(source, name="gain", kind=PortKind.state, direction=PortDirection.output),
                to_node_id="sink",
                to_port_id=find_port_id(sink, name="gain", kind=PortKind.state, direction=PortDirection.input),
                kind=GraphEdgeKind.state,
            ),
            GraphEdge(
                edge_id="backward",
                from_node_id="sink",
                from_port_id=find_port_id(sink, name="gain", kind=PortKind.state, direction=PortDirection.output),
                to_node_id="source",
                to_port_id=find_port_id(source, name="gain", kind=PortKind.state, direction=PortDirection.input),
                kind=GraphEdgeKind.state,
            ),
        ),
    )

    with pytest.raises(GraphValidationError) as error:
        GraphStore(document)
    assert error.value.code == "state_cycle"


def test_data_payload_kind_mismatch_is_rejected() -> None:
    catalog = NodeCatalog(
        services=[F8ServiceSpec(serviceClass="f8.pyengine", label="Engine")],
        operators=[
            F8OperatorSpec(
                serviceClass="f8.pyengine",
                operatorClass="video.source",
                label="Video Source",
                dataOutPorts=[video_frame_port(name="video")],
            ),
            F8OperatorSpec(
                serviceClass="f8.pyengine",
                operatorClass="json.sink",
                label="JSON Sink",
                dataInPorts=[json_data_port(name="input", value_schema=number_schema())],
            ),
        ],
    )
    service = catalog.create_service_node(node_id="engine", service_class="f8.pyengine")
    source = catalog.create_operator_node(
        node_id="source",
        service_id="engine",
        service_class="f8.pyengine",
        operator_class="video.source",
    )
    sink = catalog.create_operator_node(
        node_id="sink",
        service_id="engine",
        service_class="f8.pyengine",
        operator_class="json.sink",
    )
    document = StudioDocument(
        schema_version="f8studio-document/1",
        project_id="project1",
        graph_id="graph1",
        graph_revision=0,
        layout_revision=0,
        nodes=(service, source, sink),
        edges=(
            GraphEdge(
                edge_id="bad_data",
                from_node_id="source",
                from_port_id=find_port_id(source, name="video", kind=PortKind.data, direction=PortDirection.output),
                to_node_id="sink",
                to_port_id=find_port_id(sink, name="input", kind=PortKind.data, direction=PortDirection.input),
                kind=GraphEdgeKind.data,
            ),
        ),
    )

    with pytest.raises(GraphValidationError) as error:
        GraphStore(document)
    assert error.value.code == "data_payload_mismatch"


def test_invalid_spec_replacement_rolls_back_with_existing_edges() -> None:
    _, service, source, sink = base_nodes()
    edge = GraphEdge(
        edge_id="source_to_sink",
        from_node_id="source",
        from_port_id=find_port_id(source, name="out", kind=PortKind.data, direction=PortDirection.output),
        to_node_id="sink",
        to_port_id=find_port_id(sink, name="input", kind=PortKind.data, direction=PortDirection.input),
        kind=GraphEdgeKind.data,
    )
    initial = StudioDocument(
        schema_version="f8studio-document/1",
        project_id="project1",
        graph_id="graph1",
        graph_revision=4,
        layout_revision=2,
        nodes=(service, source, sink),
        edges=(edge,),
    )
    store = GraphStore(initial)
    spec_without_output = msgspec.structs.replace(source.spec, dataOutPorts=[])
    replacement = replace_node_spec(source, spec_without_output)

    with pytest.raises(GraphValidationError) as error:
        store.apply(
            PatchRequest(
                request_id="remove-connected-port",
                expected_graph_revision=4,
                expected_layout_revision=2,
                operations=(ReplaceNodeOp(node=replacement),),
            )
        )

    assert error.value.code == "missing_port"
    assert store.snapshot() == initial


def test_disabled_nodes_are_excluded_and_disabled_services_fail_explicitly() -> None:
    _, service, source, sink = base_nodes()
    disabled_source = msgspec.structs.replace(source, enabled=False)
    document = StudioDocument(
        schema_version="f8studio-document/1",
        project_id="project1",
        graph_id="graph1",
        graph_revision=0,
        layout_revision=0,
        nodes=(service, disabled_source, sink),
    )

    compiled = compile_document(document)
    assert {node.nodeId for node in compiled.global_graph.nodes} == {"engine", "sink"}

    disabled_service = msgspec.structs.replace(service, enabled=False)
    invalid = msgspec.structs.replace(document, nodes=(disabled_service, source, sink))
    with pytest.raises(ValueError, match="disabled or missing service"):
        compile_document(invalid)
