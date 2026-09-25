from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

import msgspec

from f8pysdk.specs import F8JsonValue

from .codec import canonical_json_bytes, clone_document
from .models import (
    BindOperatorServiceOp,
    ConnectEdgeOp,
    CreateNodeOp,
    DeleteNodeOp,
    DisconnectEdgeOp,
    GraphEdge,
    GraphEdgeKind,
    GraphNode,
    GraphOperation,
    InsertFragmentOp,
    NodeLayout,
    OperatorNode,
    PatchRequest,
    RenameNodeOp,
    ReplaceNodeOp,
    ServiceNode,
    SetNodeEnabledOp,
    SetNodeLayoutOp,
    SetNodeStateOp,
    StudioDocument,
)
from .validation import validate_document


class GraphStoreError(RuntimeError):
    code = "graph_store_error"


class RevisionConflictError(GraphStoreError):
    code = "revision_conflict"


class IdempotencyConflictError(GraphStoreError):
    code = "idempotency_conflict"


class OperationTargetError(GraphStoreError):
    code = "operation_target_error"


class PatchResult(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    request_id: str
    document: StudioDocument
    graph_changed: bool
    layout_changed: bool
    runtime_errors: tuple[str, ...] = ()


class HistoryRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    request_id: str
    expected_graph_revision: int
    expected_layout_revision: int


@dataclass(frozen=True)
class _ProcessedRequest:
    fingerprint: str
    result: PatchResult


CommitCallback = Callable[[PatchResult], None]


def _fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _clone_result(result: PatchResult) -> PatchResult:
    return PatchResult(
        request_id=result.request_id,
        document=clone_document(result.document),
        graph_changed=result.graph_changed,
        layout_changed=result.layout_changed,
    )


def _replace_service_node(
    node: ServiceNode,
    *,
    name: str | None = None,
    state_values: dict[str, F8JsonValue] | None = None,
    enabled: bool | None = None,
) -> ServiceNode:
    return ServiceNode(
        node_id=node.node_id,
        name=node.name if name is None else name,
        service_id=node.service_id,
        service_class=node.service_class,
        spec=node.spec,
        ports=node.ports,
        state_values=node.state_values if state_values is None else state_values,
        enabled=node.enabled if enabled is None else enabled,
    )


def _replace_operator_node(
    node: OperatorNode,
    *,
    name: str | None = None,
    service_id: str | None = None,
    state_values: dict[str, F8JsonValue] | None = None,
    enabled: bool | None = None,
) -> OperatorNode:
    return OperatorNode(
        node_id=node.node_id,
        name=node.name if name is None else name,
        service_id=node.service_id if service_id is None else service_id,
        service_class=node.service_class,
        operator_class=node.operator_class,
        spec=node.spec,
        ports=node.ports,
        state_values=node.state_values if state_values is None else state_values,
        enabled=node.enabled if enabled is None else enabled,
    )


def _replace_node(document: StudioDocument, node: GraphNode) -> StudioDocument:
    found = False
    nodes: list[GraphNode] = []
    for current in document.nodes:
        if current.node_id == node.node_id:
            nodes.append(node)
            found = True
        else:
            nodes.append(current)
    if not found:
        raise OperationTargetError(f"node not found: {node.node_id}")
    return msgspec.structs.replace(document, nodes=tuple(nodes))


def _upsert_layout(layouts: tuple[NodeLayout, ...], layout: NodeLayout) -> tuple[NodeLayout, ...]:
    replaced = False
    result: list[NodeLayout] = []
    for current in layouts:
        if current.node_id == layout.node_id:
            result.append(layout)
            replaced = True
        else:
            result.append(current)
    if not replaced:
        result.append(layout)
    return tuple(result)


def _apply_operation(document: StudioDocument, operation: GraphOperation) -> StudioDocument:
    if isinstance(operation, CreateNodeOp):
        if any(node.node_id == operation.node.node_id for node in document.nodes):
            raise OperationTargetError(f"node already exists: {operation.node.node_id}")
        layout = document.layout
        if operation.layout is not None:
            if operation.layout.node_id != operation.node.node_id:
                raise OperationTargetError("created node layout must reference the created node")
            layout = (*layout, operation.layout)
        return msgspec.structs.replace(document, nodes=(*document.nodes, operation.node), layout=layout)

    if isinstance(operation, DeleteNodeOp):
        target = next((node for node in document.nodes if node.node_id == operation.node_id), None)
        if target is None:
            raise OperationTargetError(f"node not found: {operation.node_id}")
        removed_node_ids = {target.node_id}
        if isinstance(target, ServiceNode):
            removed_node_ids.update(
                node.node_id
                for node in document.nodes
                if isinstance(node, OperatorNode) and node.service_id == target.service_id
            )
        return msgspec.structs.replace(
            document,
            nodes=tuple(node for node in document.nodes if node.node_id not in removed_node_ids),
            edges=tuple(
                edge
                for edge in document.edges
                if edge.from_node_id not in removed_node_ids and edge.to_node_id not in removed_node_ids
            ),
            layout=tuple(item for item in document.layout if item.node_id not in removed_node_ids),
        )

    if isinstance(operation, ConnectEdgeOp):
        if any(edge.edge_id == operation.edge.edge_id for edge in document.edges):
            raise OperationTargetError(f"edge already exists: {operation.edge.edge_id}")
        return msgspec.structs.replace(document, edges=(*document.edges, operation.edge))

    if isinstance(operation, DisconnectEdgeOp):
        if not any(edge.edge_id == operation.edge_id for edge in document.edges):
            raise OperationTargetError(f"edge not found: {operation.edge_id}")
        return msgspec.structs.replace(
            document,
            edges=tuple(edge for edge in document.edges if edge.edge_id != operation.edge_id),
        )

    if isinstance(operation, SetNodeStateOp):
        node = next((item for item in document.nodes if item.node_id == operation.node_id), None)
        if node is None:
            raise OperationTargetError(f"node not found: {operation.node_id}")
        values = dict(node.state_values)
        values[operation.field] = operation.value
        replacement: GraphNode
        if isinstance(node, ServiceNode):
            replacement = _replace_service_node(node, state_values=values)
        else:
            replacement = _replace_operator_node(node, state_values=values)
        return _replace_node(document, replacement)

    if isinstance(operation, RenameNodeOp):
        node = next((item for item in document.nodes if item.node_id == operation.node_id), None)
        if node is None:
            raise OperationTargetError(f"node not found: {operation.node_id}")
        replacement = (
            _replace_service_node(node, name=operation.name)
            if isinstance(node, ServiceNode)
            else _replace_operator_node(node, name=operation.name)
        )
        return _replace_node(document, replacement)

    if isinstance(operation, ReplaceNodeOp):
        return _replace_node(document, operation.node)

    if isinstance(operation, BindOperatorServiceOp):
        node = next((item for item in document.nodes if item.node_id == operation.node_id), None)
        if node is None:
            raise OperationTargetError(f"node not found: {operation.node_id}")
        if not isinstance(node, OperatorNode):
            raise OperationTargetError(f"only operators can be rebound: {operation.node_id}")
        rebound = _replace_node(document, _replace_operator_node(node, service_id=operation.service_id))
        rebound_nodes = {item.node_id: item for item in rebound.nodes}

        def valid_edge(edge: GraphEdge) -> bool:
            if edge.kind != GraphEdgeKind.exec:
                return True
            source = rebound_nodes[edge.from_node_id]
            target = rebound_nodes[edge.to_node_id]
            return (
                isinstance(source, OperatorNode)
                and isinstance(target, OperatorNode)
                and source.service_id == target.service_id
            )

        return msgspec.structs.replace(rebound, edges=tuple(edge for edge in rebound.edges if valid_edge(edge)))

    if isinstance(operation, SetNodeEnabledOp):
        node = next((item for item in document.nodes if item.node_id == operation.node_id), None)
        if node is None:
            raise OperationTargetError(f"node not found: {operation.node_id}")
        replacement = (
            _replace_service_node(node, enabled=operation.enabled)
            if isinstance(node, ServiceNode)
            else _replace_operator_node(node, enabled=operation.enabled)
        )
        return _replace_node(document, replacement)

    if isinstance(operation, SetNodeLayoutOp):
        return msgspec.structs.replace(document, layout=_upsert_layout(document.layout, operation.layout))

    assert isinstance(operation, InsertFragmentOp)
    existing_nodes = {node.node_id for node in document.nodes}
    existing_edges = {edge.edge_id for edge in document.edges}
    inserted_nodes = {node.node_id for node in operation.nodes}
    if len(inserted_nodes) != len(operation.nodes) or existing_nodes & inserted_nodes:
        raise OperationTargetError("inserted fragment contains duplicate node ids")
    inserted_edges = {edge.edge_id for edge in operation.edges}
    if len(inserted_edges) != len(operation.edges) or existing_edges & inserted_edges:
        raise OperationTargetError("inserted fragment contains duplicate edge ids")
    layout = document.layout
    for item in operation.layout:
        layout = _upsert_layout(layout, item)
    return msgspec.structs.replace(
        document,
        nodes=(*document.nodes, *operation.nodes),
        edges=(*document.edges, *operation.edges),
        layout=layout,
    )


def _operation_changes_layout(operation: GraphOperation) -> bool:
    if isinstance(operation, SetNodeLayoutOp):
        return True
    if isinstance(operation, CreateNodeOp):
        return operation.layout is not None
    if isinstance(operation, DeleteNodeOp):
        return True
    if isinstance(operation, InsertFragmentOp):
        return bool(operation.layout)
    return False


def _operation_changes_graph(operation: GraphOperation) -> bool:
    return not isinstance(operation, SetNodeLayoutOp)


class GraphStore:
    def __init__(self, document: StudioDocument, *, request_history_limit: int = 2048) -> None:
        if request_history_limit < 1:
            raise ValueError("request_history_limit must be positive")
        validate_document(document)
        self._document = clone_document(document)
        self._request_history_limit = request_history_limit
        self._processed: dict[str, _ProcessedRequest] = {}
        self._processed_order: list[str] = []
        self._undo: list[StudioDocument] = []
        self._redo: list[StudioDocument] = []
        self._lock = RLock()

    def snapshot(self) -> StudioDocument:
        with self._lock:
            return clone_document(self._document)

    def apply(self, request: PatchRequest) -> PatchResult:
        return self.apply_with_commit(request)

    def apply_with_commit(
        self,
        request: PatchRequest,
        *,
        before_commit: CommitCallback | None = None,
    ) -> PatchResult:
        return self._apply_request(
            request,
            fingerprint=_fingerprint(request),
            before_commit=before_commit,
        )

    def _apply_request(
        self,
        request: PatchRequest,
        *,
        fingerprint: str,
        before_commit: CommitCallback | None,
    ) -> PatchResult:
        with self._lock:
            previous = self._processed.get(request.request_id)
            if previous is not None:
                if previous.fingerprint != fingerprint:
                    raise IdempotencyConflictError(f"request id reused with different content: {request.request_id}")
                return _clone_result(previous.result)
            self._check_revisions(request.expected_graph_revision, request.expected_layout_revision)

            before = self._document
            candidate = before
            for operation in request.operations:
                candidate = _apply_operation(candidate, operation)
            validate_document(candidate)

            graph_requested = any(_operation_changes_graph(operation) for operation in request.operations)
            layout_requested = any(_operation_changes_layout(operation) for operation in request.operations)
            graph_changed = graph_requested and (candidate.nodes != before.nodes or candidate.edges != before.edges)
            layout_changed = layout_requested and candidate.layout != before.layout
            committed = msgspec.structs.replace(
                candidate,
                graph_revision=before.graph_revision + int(graph_changed),
                layout_revision=before.layout_revision + int(layout_changed),
            )
            final_document = committed if graph_changed or layout_changed else before
            result = PatchResult(
                request_id=request.request_id,
                document=clone_document(final_document),
                graph_changed=graph_changed,
                layout_changed=layout_changed,
            )
            if before_commit is not None:
                before_commit(_clone_result(result))
            if graph_changed or layout_changed:
                self._undo.append(before)
                self._redo.clear()
                self._document = clone_document(committed)
            self._remember(request.request_id, fingerprint, result)
            return _clone_result(result)

    def undo(self, request: HistoryRequest) -> PatchResult:
        return self.undo_with_commit(request)

    def undo_with_commit(
        self,
        request: HistoryRequest,
        *,
        before_commit: CommitCallback | None = None,
    ) -> PatchResult:
        with self._lock:
            fingerprint = _fingerprint(("undo", request))
            previous = self._processed.get(request.request_id)
            if previous is not None:
                if previous.fingerprint != fingerprint:
                    raise IdempotencyConflictError(f"request id reused with different content: {request.request_id}")
                return _clone_result(previous.result)
            self._check_revisions(request.expected_graph_revision, request.expected_layout_revision)
            if not self._undo:
                raise OperationTargetError("undo history is empty")
            target = self._undo[-1]
            current = self._document
            result = self._history_result(request.request_id, current, target)
            if before_commit is not None:
                before_commit(_clone_result(result))
            self._undo.pop()
            self._redo.append(current)
            self._document = clone_document(result.document)
            self._remember(request.request_id, fingerprint, result)
            return _clone_result(result)

    def redo(self, request: HistoryRequest) -> PatchResult:
        return self.redo_with_commit(request)

    def redo_with_commit(
        self,
        request: HistoryRequest,
        *,
        before_commit: CommitCallback | None = None,
    ) -> PatchResult:
        with self._lock:
            fingerprint = _fingerprint(("redo", request))
            previous = self._processed.get(request.request_id)
            if previous is not None:
                if previous.fingerprint != fingerprint:
                    raise IdempotencyConflictError(f"request id reused with different content: {request.request_id}")
                return _clone_result(previous.result)
            self._check_revisions(request.expected_graph_revision, request.expected_layout_revision)
            if not self._redo:
                raise OperationTargetError("redo history is empty")
            target = self._redo[-1]
            current = self._document
            result = self._history_result(request.request_id, current, target)
            if before_commit is not None:
                before_commit(_clone_result(result))
            self._redo.pop()
            self._undo.append(current)
            self._document = clone_document(result.document)
            self._remember(request.request_id, fingerprint, result)
            return _clone_result(result)

    def _history_result(
        self,
        request_id: str,
        current: StudioDocument,
        target: StudioDocument,
    ) -> PatchResult:
        graph_changed = current.nodes != target.nodes or current.edges != target.edges
        layout_changed = current.layout != target.layout
        restored = StudioDocument(
            schema_version=current.schema_version,
            project_id=current.project_id,
            graph_id=current.graph_id,
            graph_revision=current.graph_revision + int(graph_changed),
            layout_revision=current.layout_revision + int(layout_changed),
            nodes=target.nodes,
            edges=target.edges,
            layout=target.layout,
        )
        validate_document(restored)
        return PatchResult(
            request_id=request_id,
            document=clone_document(restored),
            graph_changed=graph_changed,
            layout_changed=layout_changed,
        )

    def _check_revisions(self, graph_revision: int, layout_revision: int) -> None:
        if graph_revision != self._document.graph_revision or layout_revision != self._document.layout_revision:
            raise RevisionConflictError(
                "revision conflict: "
                f"expected graph/layout {graph_revision}/{layout_revision}, "
                f"current {self._document.graph_revision}/{self._document.layout_revision}"
            )

    def _remember(self, request_id: str, fingerprint: str, result: PatchResult) -> None:
        self._processed[request_id] = _ProcessedRequest(fingerprint=fingerprint, result=_clone_result(result))
        self._processed_order.append(request_id)
        while len(self._processed_order) > self._request_history_limit:
            expired = self._processed_order.pop(0)
            self._processed.pop(expired, None)
