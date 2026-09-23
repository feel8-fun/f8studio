from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Self

from f8pysdk.specs import F8JsonValue
from f8studio_server import mcp_server
from f8studio_server.api_client import StudioApiClient


class RecordingStudioApiClient(StudioApiClient):
    calls: list[tuple[str, object]] = []

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        return None

    def apply_patch(self, project_id: str, patch: Mapping[str, object]) -> F8JsonValue:
        patch_object = dict(patch)
        self.calls.append(("apply_patch", {"projectId": project_id, "patch": patch_object}))
        return {"projectId": project_id, "accepted": True}

    def resolve_agent_approval(
        self,
        session_id: str,
        approval_id: str,
        *,
        approved: bool,
        arguments_hash: str,
    ) -> F8JsonValue:
        self.calls.append(
            (
                "resolve_agent_approval",
                {
                    "sessionId": session_id,
                    "approvalId": approval_id,
                    "approved": approved,
                    "argumentsHash": arguments_hash,
                },
            )
        )
        return {"sessionId": session_id, "accepted": True}


def test_mcp_registers_tools_and_forwards_exact_mutation_arguments() -> None:
    RecordingStudioApiClient.calls = []
    server = mcp_server.create_mcp_server(
        studio_url="http://studio.invalid",
        host="127.0.0.1",
        port=8765,
        path="/mcp",
        client_factory=RecordingStudioApiClient,
    )
    patch = {
        "requestId": "mcp-test",
        "expectedGraphRevision": 4,
        "expectedLayoutRevision": 7,
        "operations": [],
    }

    async def invoke_tools() -> None:
        names = {tool.name for tool in await server.list_tools()}
        assert {
            "graph_apply_patch",
            "graph_validate",
            "agent_approval_resolve",
            "agent_run_cancel",
            "runtime_observe",
        } <= names

        await server.call_tool("graph_apply_patch", {"project_id": "project-1", "patch": patch})
        await server.call_tool(
            "agent_approval_resolve",
            {
                "session_id": "session-1",
                "approval_id": "approval-1",
                "arguments_hash": "abc123",
                "approved": True,
            },
        )

    asyncio.run(invoke_tools())

    assert RecordingStudioApiClient.calls == [
        ("apply_patch", {"projectId": "project-1", "patch": patch}),
        (
            "resolve_agent_approval",
            {
                "sessionId": "session-1",
                "approvalId": "approval-1",
                "approved": True,
                "argumentsHash": "abc123",
            },
        ),
    ]
