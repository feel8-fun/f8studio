from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from uuid import uuid4

import msgspec
from f8pysdk.generated import F8StateAccess
from f8studio_core.graph import (
    GraphNode,
    NodeCatalog,
    OperatorNode,
    PatchRequest,
    PatchResult,
    RevisionConflictError,
    SetNodeStateOp,
    StudioDocument,
)

from f8studio_server.errors import InvalidRequestError, NotFoundError

from ..automation_tools import StudioAutomationTools
from ..editor import CreateEditorSessionRequest, EditorAnalysis, EditorSessionService
from ..editor_context import editor_support_files
from ..events import EventJournal
from ..local_integration import (
    ApplyUnityInstallRequest,
    DetectModdingTargetRequest,
    LocalIntegrationService,
    PreviewUnityInstallRequest,
    VerifySkeletonUdpRequest,
)
from ..models import DeployProjectRequest
from ..project_repository import utc_now_text
from .evidence import (
    arguments_hash,
    catalog_evidence,
    catalog_index,
    deploy_evidence,
    document_evidence,
    graph_outline,
    json_value,
    monitor_evidence,
    patch_evidence,
    tool_text,
)
from .evidence import catalog_operator as read_catalog_operator
from .evidence import catalog_search as search_catalog
from .execution import AgentToolExecution
from .graph_edits import GraphChanges, build_patch
from .models import (
    AgentArtifact,
    AgentSessionRecord,
)
from .sessions import AgentSessions
from .skills import AgentSkillLibrary


class AgentModelTools:
    """Bind model tools with preview and proposal state isolated to one run."""

    def __init__(self, *, tools: StudioAutomationTools, execution: AgentToolExecution, sessions: AgentSessions, editor: EditorSessionService, local: LocalIntegrationService, skills: AgentSkillLibrary, events: EventJournal) -> None:
        self._tools = tools
        self._execution = execution
        self._sessions = sessions
        self._editor = editor
        self._local = local
        self._skills = skills
        self._events = events

    def _code_target(self, project_id: str, node_id: str) -> tuple[StudioDocument, GraphNode, str]:
        document = self._tools.document(project_id)
        node = next((item for item in document.nodes if item.node_id == node_id), None)
        if node is None:
            raise NotFoundError(f"code node not found: {node_id}")
        editor_support_files(node, "code")
        state_fields = node.spec.stateFields
        fields = () if isinstance(state_fields, msgspec.UnsetType) else state_fields
        field = next(item for item in fields if item.name == "code")
        control = field.control
        if isinstance(control, msgspec.UnsetType) or control.language != "python":
            raise InvalidRequestError(f"code field is not Python: {node_id}")
        if field.access is not F8StateAccess.rw:
            raise InvalidRequestError(f"code field is not writable: {node_id}")
        incoming = {port.port_id for port in node.ports if port.kind.value == "state" and port.name == "code"}
        if any(edge.to_node_id == node_id and edge.to_port_id in incoming for edge in document.edges):
            raise InvalidRequestError(f"code field is driven by an incoming state connection: {node_id}")
        value = node.state_values.get("code", "")
        if not isinstance(value, str):
            raise TypeError(f"code field is not text: {node_id}")
        return document, node, value

    def _analyze_code(self, project_id: str, node_id: str, code: str) -> EditorAnalysis:
        _, node, _ = self._code_target(project_id, node_id)
        session = self._editor.create(
            CreateEditorSessionRequest(
                language="python",
                text=code,
                filename="state.py",
                support_files=editor_support_files(node, "code"),
                project_id=project_id,
                node_id=node_id,
                field_name="code",
            )
        )
        try:
            return self._editor.analyze(session.session_id)
        finally:
            self._editor.close_session(session.session_id)

    def bind(self, record: AgentSessionRecord) -> tuple[Callable[..., Awaitable[str]], ...]:
        project_id = record.project_id
        previewed_patches: set[str] = set()
        proposals: dict[str, PatchRequest] = {}
        previewed_unity_plans: set[str] = set()

        async def catalog_read() -> str:
            """List installed service and operator IDs/labels. Use catalog_search and catalog_operator to inspect relevant nodes."""
            result = await self._execution.run(
                record, tool_name="catalog.read", arguments={}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(self._tools.catalog), result_encoder=catalog_evidence,
            )
            return tool_text(catalog_index(result))

        async def catalog_search(query: str) -> str:
            """Find operators by name, class, or description; returns matching IDs for catalog_operator."""
            result = await self._execution.run(
                record, tool_name="catalog.search", arguments={"query": query}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(search_catalog, self._tools.catalog(), query),
            )
            return tool_text(result)

        async def catalog_operator(service_class: str, operator_class: str) -> str:
            """Read exact operator specification: behavior, ports, state fields, defaults, and constraints."""
            result = await self._execution.run(
                record, tool_name="catalog.operator",
                arguments={"serviceClass": service_class, "operatorClass": operator_class}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(read_catalog_operator, self._tools.catalog(), service_class, operator_class),
            )
            return tool_text(result)

        async def catalog_create_node(node_id: str, service_class: str, operator_class: str,
                                      service_id: str, name: str) -> str:
            """Create the exact operator-node JSON for a createNode patch; does not edit the graph."""
            def create() -> GraphNode:
                snapshot = self._tools.catalog()
                catalog = NodeCatalog(services=snapshot.services, operators=snapshot.operators)
                document = self._tools.document(project_id)
                if any(node.node_id == node_id for node in document.nodes):
                    raise InvalidRequestError(f"Node ID already exists: {node_id}")
                if not any(node.service_id == service_id and node.service_class == service_class
                           and not isinstance(node, OperatorNode) for node in document.nodes):
                    raise InvalidRequestError(f"Service instance not found: {service_id} ({service_class})")
                return catalog.create_operator_node(
                    node_id=node_id, service_id=service_id, service_class=service_class,
                    operator_class=operator_class, name=name,
                )
            result = await self._execution.run(
                record, tool_name="catalog.create_node",
                arguments={"nodeId": node_id, "serviceClass": service_class,
                           "operatorClass": operator_class, "serviceId": service_id, "name": name},
                target_graph_revision=None, operation=lambda: asyncio.to_thread(create),
                result_encoder=lambda node: {"nodeId": node.node_id, "operatorClass": operator_class},
            )
            return tool_text(result)

        async def skills_list() -> str:
            """List available Studio workflow skills, including locally installed game skills."""
            result = await self._execution.run(
                record, tool_name="skills.list", arguments={}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(self._skills.list),
            )
            return tool_text(result)

        async def skill_read(skill_id: str) -> str:
            """Read an available Studio skill by ID before using its workflow guidance."""
            result = await self._execution.run(
                record, tool_name="skills.read", arguments={"skillId": skill_id}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(self._skills.read, skill_id),
            )
            return result

        async def graph_read() -> str:
            """Read current graph revisions, node IDs, state values, ports, edges, and layout; use graph_node for full node JSON."""
            result = await self._execution.run(
                record, tool_name="graph.read", arguments={"projectId": project_id}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(self._tools.document, project_id), result_encoder=document_evidence,
            )
            return tool_text(graph_outline(result))

        async def graph_node(node_id: str) -> str:
            """Read one existing graph node with its full spec and port IDs for patch construction."""
            def read() -> GraphNode:
                document = self._tools.document(project_id)
                node = next((item for item in document.nodes if item.node_id == node_id), None)
                if node is None:
                    raise InvalidRequestError(f"Node not found: {node_id}")
                return node
            result = await self._execution.run(
                record, tool_name="graph.node", arguments={"nodeId": node_id}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(read),
                result_encoder=lambda node: {"nodeId": node.node_id},
            )
            return tool_text(result)

        async def graph_preview_patch(patch_json: str) -> str:
            """Validate a JSON PatchRequest against the current project without changing it."""
            patch = msgspec.json.decode(patch_json, type=PatchRequest)
            result = await self._execution.run(
                record, tool_name="graph.preview_patch", arguments={"projectId": project_id, "patch": json_value(patch)},
                target_graph_revision=patch.expected_graph_revision,
                operation=lambda: asyncio.to_thread(self._tools.preview_patch, project_id, patch),
                result_encoder=patch_evidence,
            )
            previewed_patches.add(arguments_hash({"patch": json_value(patch)}))
            await self._sessions.append_artifact(record.session_id, AgentArtifact(
                artifact_id=uuid4().hex, kind="graph_patch", title="Proposed graph patch",
                payload={"patch": json_value(patch), "afterGraphRevision": result.document.graph_revision},
                created_at=utc_now_text(),
            ))
            return tool_text(patch_evidence(result))

        async def graph_apply_patch(patch_json: str) -> str:
            """Apply a previously previewed JSON PatchRequest after human approval."""
            patch = msgspec.json.decode(patch_json, type=PatchRequest)
            if any(isinstance(operation, SetNodeStateOp) and operation.field == "code" for operation in patch.operations):
                raise InvalidRequestError("use code_read, code_analyze, and code_write to edit Python node code")
            fingerprint = arguments_hash({"patch": json_value(patch)})
            if fingerprint not in previewed_patches:
                raise InvalidRequestError("graph patch must be previewed in this run before applying")
            result = await self._execution.approved(
                record, tool_name="graph.apply_patch", arguments={"projectId": project_id, "patch": json_value(patch)},
                target_graph_revision=patch.expected_graph_revision,
                target_layout_revision=patch.expected_layout_revision,
                operation=lambda: self._tools.apply_patch(project_id, patch), result_encoder=patch_evidence,
            )
            previewed_patches.discard(fingerprint)
            return tool_text(patch_evidence(result))

        async def graph_propose_changes(changes_json: str) -> str:
            """Build and preview a graph patch from compact JSON; does not change the graph.

            Prefer this over copying full specs into graph_preview_patch. JSON shape:
            {"expectedGraphRevision": 0, "expectedLayoutRevision": 0,
             "nodes": [{"nodeId": "phase", "serviceClass": "f8.pyengine", "serviceId": "engine",
                        "operatorClass": "f8.phase", "name": "Phase", "stateValues": {"hz": 1}, "x": 100, "y": 100}],
             "connections": [{"fromNodeId": "phase", "fromPort": "phase", "toNodeId": "cosine", "toPort": "phase", "kind": "data"}],
             "stateUpdates": [{"nodeId": "wave", "field": "upstreamSampleIntervalMs", "value": 20}]}
            Nodes, connections, and stateUpdates are optional arrays. Connections use port names, not IDs.
            To create a service omit operatorClass/serviceId. Reuse existing services and nodes.
            Returns proposalId; immediately call graph_apply_proposal to show the approval UI.
            """
            changes = msgspec.json.decode(changes_json, type=GraphChanges)
            proposal_id = uuid4().hex

            def prepare() -> tuple[PatchRequest, PatchResult]:
                document = self._tools.document(project_id)
                patch = build_patch(document, self._tools.catalog(), changes, request_id=f"agent:{proposal_id}")
                preview = self._tools.preview_patch(project_id, patch)
                self._tools.validate_document(preview.document)
                return patch, preview

            patch, preview = await self._execution.run(
                record, tool_name="graph.propose_changes", arguments={"changes": json_value(changes)},
                target_graph_revision=changes.expected_graph_revision,
                operation=lambda: asyncio.to_thread(prepare),
                result_encoder=lambda result: {"proposalId": proposal_id, "operationCount": len(result[0].operations),
                                               "preview": patch_evidence(result[1])},
            )
            proposals[proposal_id] = patch
            await self._sessions.append_artifact(record.session_id, AgentArtifact(
                artifact_id=proposal_id, kind="graph_patch", title="Proposed graph changes",
                payload={"patch": json_value(patch), "changes": json_value(changes),
                         "afterGraphRevision": preview.document.graph_revision}, created_at=utc_now_text(),
            ))
            return tool_text({"proposalId": proposal_id, "operationCount": len(patch.operations),
                               "nextAction": "Call graph_apply_proposal with this proposalId to request approval."})

        async def graph_apply_proposal(proposal_id: str) -> str:
            """Request user approval and apply the exact patch prepared by graph_propose_changes.

            Calling this tool opens the approval UI and waits for the user's decision. Do not ask
            for approval in chat instead. No graph mutation happens before approval.
            """
            patch = proposals.get(proposal_id)
            if patch is None:
                raise InvalidRequestError("Unknown proposalId; call graph_propose_changes in this run first")
            if any(isinstance(operation, SetNodeStateOp) and operation.field == "code" for operation in patch.operations):
                raise InvalidRequestError("use code_read, code_analyze, and code_write to edit Python node code")
            result = await self._execution.approved(
                record, tool_name="graph.apply_patch", arguments={"projectId": project_id, "patch": json_value(patch)},
                target_graph_revision=patch.expected_graph_revision,
                target_layout_revision=patch.expected_layout_revision,
                operation=lambda: self._tools.apply_patch(project_id, patch), result_encoder=patch_evidence,
            )
            proposals.pop(proposal_id)
            return tool_text(patch_evidence(result))

        async def code_read(node_id: str) -> str:
            """Read the Python code in one node, its graph revision, and its SHA-256 content hash."""
            document, node, code = await self._execution.run(
                record, tool_name="code.read", arguments={"nodeId": node_id}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(self._code_target, project_id, node_id),
                result_encoder=lambda result: {"nodeId": result[1].node_id, "graphRevision": result[0].graph_revision},
            )
            return tool_text({
                "nodeId": node.node_id, "nodeName": node.name, "graphRevision": document.graph_revision,
                "codeSha256": hashlib.sha256(code.encode("utf-8")).hexdigest(), "code": code,
            })

        async def code_analyze(node_id: str, code: str) -> str:
            """Analyze proposed Python node code with its generated Studio API support files."""
            result = await self._execution.run(
                record, tool_name="code.analyze", arguments={"nodeId": node_id, "code": code},
                target_graph_revision=None,
                operation=lambda: asyncio.to_thread(self._analyze_code, project_id, node_id, code),
            )
            return tool_text(result)

        async def code_write(node_id: str, expected_graph_revision: int, expected_code_sha256: str, code: str) -> str:
            """Write Python code to one node after human approval; requires the read revision and content hash."""
            compile(code, f"{node_id}.py", "exec")
            document, _, current = await asyncio.to_thread(self._code_target, project_id, node_id)
            digest = hashlib.sha256(current.encode("utf-8")).hexdigest()
            if document.graph_revision != expected_graph_revision or digest != expected_code_sha256:
                raise RevisionConflictError(f"code changed since read: {node_id}; read the node again")
            await self._sessions.append_artifact(record.session_id, AgentArtifact(
                artifact_id=uuid4().hex, kind="text", title=f"Proposed code change: {node_id}",
                payload={"nodeId": node_id, "beforeSha256": digest, "before": current, "after": code},
                created_at=utc_now_text(),
            ))
            patch = PatchRequest(
                request_id=f"agent-code:{record.session_id}:{uuid4().hex}",
                expected_graph_revision=expected_graph_revision,
                expected_layout_revision=document.layout_revision,
                operations=(SetNodeStateOp(node_id=node_id, field="code", value=code),),
            )
            async def apply_code() -> PatchResult:
                _, _, latest = await asyncio.to_thread(self._code_target, project_id, node_id)
                if hashlib.sha256(latest.encode("utf-8")).hexdigest() != expected_code_sha256:
                    raise RevisionConflictError(f"code changed during approval: {node_id}")
                return await self._tools.apply_patch(project_id, patch)

            result = await self._execution.approved(
                record, tool_name="code.write",
                arguments={"nodeId": node_id, "expectedGraphRevision": expected_graph_revision,
                           "expectedCodeSha256": expected_code_sha256, "code": code},
                target_graph_revision=expected_graph_revision, target_layout_revision=patch.expected_layout_revision,
                operation=apply_code, result_encoder=patch_evidence,
            )
            return tool_text({
                "graphRevision": result.document.graph_revision,
                "layoutRevision": result.document.layout_revision,
                "runtimeErrors": list(result.runtime_errors),
            })

        async def graph_validate() -> str:
            """Validate the current graph after edits."""
            document = await asyncio.to_thread(self._tools.document, project_id)
            await self._execution.run(
                record, tool_name="graph.validate", arguments={"graphRevision": document.graph_revision},
                target_graph_revision=document.graph_revision,
                operation=lambda: asyncio.to_thread(self._tools.validate_document, document),
            )
            return tool_text({"valid": True, "graphRevision": document.graph_revision})

        async def project_deploy() -> str:
            """Deploy the current graph after human approval and wait for the deployment result."""
            document = await asyncio.to_thread(self._tools.document, project_id)
            request = DeployProjectRequest(
                request_id=f"agent-deploy:{record.session_id}:{uuid4().hex}",
                expected_graph_revision=document.graph_revision,
            )
            result = await self._execution.approved(
                record, tool_name="project.deploy", arguments={"projectId": project_id, "request": json_value(request)},
                target_graph_revision=document.graph_revision,
                operation=lambda: self._execution.deploy_and_wait(project_id, request), result_encoder=deploy_evidence,
            )
            return tool_text(result)

        async def runtime_observe() -> str:
            """Read the project's current runtime monitor samples after deployment."""
            result = await self._execution.run(
                record, tool_name="runtime.observe", arguments={"projectId": project_id}, target_graph_revision=None,
                operation=lambda: self._tools.monitor_snapshot(project_id), result_encoder=monitor_evidence,
            )
            return tool_text(result)

        async def logs_read(limit: int = 50) -> str:
            """Read recent Studio logs for debugging; limit must be between 1 and 100."""
            if not 1 <= limit <= 100:
                raise InvalidRequestError("log limit must be between 1 and 100")
            result = await self._execution.run(
                record, tool_name="logs.read", arguments={"limit": limit}, target_graph_revision=None,
                operation=lambda: self._events.recent_logs(limit=limit),
            )
            return tool_text(result)

        async def modding_detect_target(target_path: str) -> str:
            """Detect the game engine and whether Studio has a supported installer for this local target."""
            result = await self._execution.run(
                record, tool_name="modding.detect_target", arguments={"targetPath": target_path}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(
                    self._local.detect_modding_target, DetectModdingTargetRequest(target_path=target_path)
                ),
            )
            return tool_text(result)

        async def modding_preview_unity_install(target_path: str) -> str:
            """Preview exact Unity exporter installation actions and files; does not write to the game."""
            result = await self._execution.run(
                record, tool_name="modding.preview_unity_install", arguments={"targetPath": target_path},
                target_graph_revision=None,
                operation=lambda: asyncio.to_thread(
                    self._local.preview_unity_install, PreviewUnityInstallRequest(target_path=target_path)
                ),
            )
            previewed_unity_plans.add(result.plan_id)
            await self._sessions.append_artifact(record.session_id, AgentArtifact(
                artifact_id=uuid4().hex, kind="text", title="Unity installation preview",
                payload=json_value(result), created_at=utc_now_text(),
            ))
            return tool_text(result)

        async def modding_apply_unity_install(plan_id: str) -> str:
            """Install the exact previewed Unity plan after human approval; never use for Unreal."""
            if plan_id not in previewed_unity_plans:
                raise InvalidRequestError("Unity installation plan must be previewed in this run before applying")
            document = await asyncio.to_thread(self._tools.document, project_id)
            result = await self._execution.approved(
                record, tool_name="modding.apply_unity_install", arguments={"planId": plan_id},
                target_graph_revision=document.graph_revision,
                operation=lambda: asyncio.to_thread(
                    self._local.apply_unity_install, ApplyUnityInstallRequest(plan_id=plan_id, confirm=True)
                ),
            )
            previewed_unity_plans.discard(plan_id)
            return tool_text(result)

        async def modding_verify_udp(port: int = 39540) -> str:
            """Verify a complete decoded skeleton frame from the game exporter on a UDP port."""
            result = await self._execution.run(
                record, tool_name="modding.verify_udp", arguments={"port": port}, target_graph_revision=None,
                operation=lambda: self._local.verify_skeleton_udp(VerifySkeletonUdpRequest(port=port)),
            )
            return tool_text(result)

        return (
            skills_list, skill_read, catalog_read, catalog_search, catalog_operator, catalog_create_node,
            graph_read, graph_node, graph_preview_patch, graph_apply_patch,
            graph_propose_changes, graph_apply_proposal,
            code_read, code_analyze, code_write, graph_validate, project_deploy,
            runtime_observe, logs_read, modding_detect_target, modding_preview_unity_install,
            modding_apply_unity_install, modding_verify_udp,
        )
