from __future__ import annotations

import asyncio
from typing import cast
from uuid import uuid4

from f8pysdk.specs import F8JsonValue
from f8studio_core.graph import (
    CreateNodeOp,
    NodeCatalog,
    NodeLayout,
    OperatorNode,
    PatchRequest,
    StudioDocument,
)

from ..automation_tools import StudioAutomationTools
from ..catalog import CatalogSnapshot
from ..models import DeployProjectRequest
from ..project_repository import utc_now_text
from .evidence import catalog_evidence, deploy_evidence, document_evidence, json_value, monitor_evidence, patch_evidence
from .execution import AgentToolExecution
from .models import (
    AgentArtifact,
    AgentSessionRecord,
)
from .sessions import AgentSessions


class DeterministicWorkflow:
    """Run offline graph workflows through the same audit and approval boundary."""

    def __init__(self, tools: StudioAutomationTools, execution: AgentToolExecution, sessions: AgentSessions) -> None:
        self._tools = tools
        self._execution = execution
        self._sessions = sessions

    async def run(self, record: AgentSessionRecord, prompt: str) -> None:
        catalog = await self._execution.run(
            record,
            tool_name="catalog.read",
            arguments={},
            target_graph_revision=None,
            operation=lambda: asyncio.to_thread(self._tools.catalog),
            result_encoder=catalog_evidence,
        )
        record = await asyncio.to_thread(self._sessions.get, record.session_id)
        document = await self._execution.run(
            record,
            tool_name="graph.read",
            arguments={"projectId": record.project_id},
            target_graph_revision=None,
            operation=lambda: asyncio.to_thread(self._tools.document, record.project_id),
            result_encoder=document_evidence,
        )
        if "diagnos" in prompt.lower() or "诊断" in prompt:
            await self._run_diagnostics(record, document)
        else:
            await self._run_graph_build(record, catalog, document)

    async def _run_diagnostics(self, record: AgentSessionRecord, document: StudioDocument) -> None:
        await self._execution.run(
            record,
            tool_name="graph.validate",
            arguments={"projectId": record.project_id, "graphRevision": document.graph_revision},
            target_graph_revision=document.graph_revision,
            operation=lambda: asyncio.to_thread(self._tools.validate_document, document),
        )
        latest = await asyncio.to_thread(self._sessions.get, record.session_id)
        monitors = await self._execution.run(
            latest,
            tool_name="runtime.observe",
            arguments={"projectId": record.project_id},
            target_graph_revision=document.graph_revision,
            operation=lambda: self._tools.monitor_snapshot(record.project_id),
            result_encoder=monitor_evidence,
        )
        artifact = AgentArtifact(
            artifact_id=uuid4().hex,
            kind="diagnostics",
            title="Graph diagnostics",
            payload={
                "valid": True,
                "graphRevision": document.graph_revision,
                "nodeCount": len(document.nodes),
                "edgeCount": len(document.edges),
                "monitors": monitors,
            },
            created_at=utc_now_text(),
        )
        await self._sessions.append_artifact(record.session_id, artifact)

    async def _run_graph_build(self, record: AgentSessionRecord, catalog: CatalogSnapshot, document: StudioDocument) -> None:
        operations = self._build_value_stepper_operations(catalog, document)
        if not operations:
            artifact = AgentArtifact(
                artifact_id=uuid4().hex,
                kind="diagnostics",
                title="Graph already satisfies goal",
                payload={"graphRevision": document.graph_revision, "changed": False},
                created_at=utc_now_text(),
            )
            await self._sessions.append_artifact(record.session_id, artifact)
            return
        patch = PatchRequest(
            request_id=f"agent:{record.session_id}:{uuid4().hex}",
            expected_graph_revision=document.graph_revision,
            expected_layout_revision=document.layout_revision,
            operations=operations,
        )
        patch_json = cast(dict[str, F8JsonValue], json_value(patch))
        latest = await asyncio.to_thread(self._sessions.get, record.session_id)
        preview = await self._execution.run(
            latest,
            tool_name="graph.preview_patch",
            arguments={"projectId": record.project_id, "patch": patch_json},
            target_graph_revision=document.graph_revision,
            operation=lambda: asyncio.to_thread(self._tools.preview_patch, record.project_id, patch),
            result_encoder=patch_evidence,
        )
        artifact = AgentArtifact(
            artifact_id=uuid4().hex,
            kind="graph_patch",
            title="Proposed graph patch",
            payload={
                "patch": patch_json,
                "beforeGraphRevision": document.graph_revision,
                "afterGraphRevision": preview.document.graph_revision,
                "operationCount": len(operations),
            },
            created_at=utc_now_text(),
        )
        await self._sessions.append_artifact(record.session_id, artifact)
        latest = await asyncio.to_thread(self._sessions.get, record.session_id)
        applied = await self._execution.approved(
            latest,
            tool_name="graph.apply_patch",
            arguments={"projectId": record.project_id, "patch": patch_json},
            target_graph_revision=document.graph_revision,
            target_layout_revision=patch.expected_layout_revision,
            operation=lambda: self._tools.apply_patch(record.project_id, patch),
            result_encoder=patch_evidence,
        )
        latest = await asyncio.to_thread(self._sessions.get, record.session_id)
        await self._execution.run(
            latest,
            tool_name="graph.validate",
            arguments={"projectId": record.project_id, "graphRevision": applied.document.graph_revision},
            target_graph_revision=applied.document.graph_revision,
            operation=lambda: asyncio.to_thread(self._tools.validate_document, applied.document),
        )
        deploy_request = DeployProjectRequest(
            request_id=f"agent-deploy:{record.session_id}:{uuid4().hex}",
            expected_graph_revision=applied.document.graph_revision,
        )
        deploy_json = cast(dict[str, F8JsonValue], json_value(deploy_request))
        latest = await asyncio.to_thread(self._sessions.get, record.session_id)
        job = await self._execution.approved(
            latest,
            tool_name="project.deploy",
            arguments={"projectId": record.project_id, "request": deploy_json},
            target_graph_revision=applied.document.graph_revision,
            operation=lambda: self._execution.deploy_and_wait(record.project_id, deploy_request),
            result_encoder=deploy_evidence,
        )
        await self._sessions.append_artifact(
            record.session_id,
            AgentArtifact(
                artifact_id=uuid4().hex,
                kind="deployment",
                title="Deployment result",
                payload=json_value(job),
                created_at=utc_now_text(),
            ),
        )
        latest = await asyncio.to_thread(self._sessions.get, record.session_id)
        monitors = await self._execution.run(
            latest,
            tool_name="runtime.observe",
            arguments={"projectId": record.project_id},
            target_graph_revision=applied.document.graph_revision,
            operation=lambda: self._tools.monitor_snapshot(record.project_id),
            result_encoder=monitor_evidence,
        )
        await self._sessions.append_artifact(
            record.session_id,
            AgentArtifact(
                artifact_id=uuid4().hex,
                kind="monitor",
                title="Runtime monitor evidence",
                payload=monitors,
                created_at=utc_now_text(),
            ),
        )

    @staticmethod
    def _build_value_stepper_operations(catalog: CatalogSnapshot, document: StudioDocument) -> tuple[CreateNodeOp, ...]:
        node_catalog = NodeCatalog(services=catalog.services, operators=catalog.operators)
        service = next(
            (node for node in document.nodes if not isinstance(node, OperatorNode) and node.service_class == "f8.pystudio"),
            None,
        )
        operations: list[CreateNodeOp] = []
        if service is None:
            service = node_catalog.create_service_node(node_id="studio", service_class="f8.pystudio", name="Studio")
            operations.append(
                CreateNodeOp(node=service, layout=NodeLayout(node_id=service.node_id, x=80.0, y=80.0, width=524.0, height=300.0))
            )
        existing_stepper = next(
            (
                node
                for node in document.nodes
                if isinstance(node, OperatorNode) and node.operator_class == "f8.value_stepper"
            ),
            None,
        )
        if existing_stepper is None:
            node_id = "agent_value_stepper"
            occupied = {node.node_id for node in document.nodes}
            suffix = 2
            while node_id in occupied:
                node_id = f"agent_value_stepper_{suffix}"
                suffix += 1
            operator = node_catalog.create_operator_node(
                node_id=node_id,
                service_id=service.service_id,
                service_class="f8.pystudio",
                operator_class="f8.value_stepper",
                name="AI Value Stepper",
            )
            operations.append(
                CreateNodeOp(node=operator, layout=NodeLayout(node_id=node_id, x=120.0, y=150.0, width=240.0))
            )
        return tuple(operations)
