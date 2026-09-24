from __future__ import annotations

import enum
from typing import Literal

import msgspec

from f8pysdk.specs import F8JsonValue


class AgentRunStatus(str, enum.Enum):
    idle = "idle"
    running = "running"
    waiting_for_approval = "waiting_for_approval"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class ToolCallStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    waiting_for_approval = "waiting_for_approval"
    succeeded = "succeeded"
    failed = "failed"
    denied = "denied"
    cancelled = "cancelled"


class ApprovalStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    denied = "denied"
    expired = "expired"
    invalidated = "invalidated"
    cancelled = "cancelled"


class AgentProviderSummary(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    provider_id: str
    display_name: str
    models: tuple[str, ...]
    configured: bool
    deterministic: bool = False


class AgentMessage(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    message_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str


class AgentArtifact(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    artifact_id: str
    kind: Literal["graph_patch", "diagnostics", "deployment", "monitor", "text"]
    title: str
    payload: F8JsonValue
    created_at: str


class AgentToolCall(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    tool_call_id: str
    tool_name: str
    arguments: dict[str, F8JsonValue]
    arguments_hash: str
    target_graph_revision: int | None
    status: ToolCallStatus
    created_at: str
    updated_at: str
    result: F8JsonValue = None
    error_message: str = ""
    traceback_id: str = ""


class AgentApproval(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    approval_id: str
    tool_call_id: str
    tool_name: str
    arguments_hash: str
    target_graph_revision: int
    expires_at: str
    status: ApprovalStatus
    resolved_at: str | None = None


class AgentSessionRecord(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    session_id: str
    project_id: str
    title: str
    provider_id: str
    model_id: str
    status: AgentRunStatus
    created_at: str
    updated_at: str
    messages: tuple[AgentMessage, ...] = ()
    tool_calls: tuple[AgentToolCall, ...] = ()
    artifacts: tuple[AgentArtifact, ...] = ()
    approval: AgentApproval | None = None
    error_message: str = ""
    traceback_id: str = ""


class AgentSessionSummary(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    session_id: str
    project_id: str
    title: str
    provider_id: str
    model_id: str
    status: AgentRunStatus
    updated_at: str
    message_count: int


class CreateAgentSessionRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    project_id: str
    title: str = "New agent session"
    provider_id: str = "deterministic"
    model_id: str = "graph-builder-v1"


class StartAgentRunRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    prompt: str


class ResolveAgentApprovalRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    approved: bool
    arguments_hash: str


__all__ = [
    "AgentApproval",
    "AgentArtifact",
    "AgentMessage",
    "AgentProviderSummary",
    "AgentRunStatus",
    "AgentSessionRecord",
    "AgentSessionSummary",
    "AgentToolCall",
    "ApprovalStatus",
    "CreateAgentSessionRequest",
    "ResolveAgentApprovalRequest",
    "StartAgentRunRequest",
    "ToolCallStatus",
]
