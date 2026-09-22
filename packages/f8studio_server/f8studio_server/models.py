from __future__ import annotations

import enum
from typing import Literal

import msgspec

from f8pysdk.specs import F8JsonValue
from f8studio_core.graph import StudioDocument


class BrowserIceServer(msgspec.Struct, frozen=True, kw_only=True, rename="camel", omit_defaults=True):
    urls: tuple[str, ...]
    username: str | None = None
    credential: str | None = None


class BrowserRtcConfiguration(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    ice_servers: tuple[BrowserIceServer, ...] = ()
    ice_transport_policy: Literal["all", "relay"] = "all"


class ProjectSummary(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    project_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    graph_revision: int
    layout_revision: int


class ProjectRecord(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    project_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    document: StudioDocument


class CreateProjectRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    name: str = "Untitled Project"
    description: str = ""
    project_id: str | None = None


class UpdateProjectRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    name: str
    description: str = ""


class ValidateDocumentRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    document: StudioDocument


class DeployProjectRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    request_id: str
    expected_graph_revision: int
    force_apply: bool = False


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    partially_failed = "partially_failed"
    failed = "failed"
    cancelled = "cancelled"


class ServiceDeployResult(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    service_id: str
    success: bool
    error_message: str = ""


class RuntimeActionResult(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    success: bool
    result: F8JsonValue = None
    error_message: str = ""


class ServiceRuntimeStatus(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    service_id: str
    service_class: str
    runtime_instance_id: str
    active: bool
    rungraph_graph_id: str = ""
    rungraph_revision: str = ""
    rungraph_fingerprint: str = ""


class DeployJob(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    job_id: str
    request_id: str
    project_id: str
    source_graph_revision: int
    source_semantic_revision: str
    status: JobStatus
    created_at: str
    updated_at: str
    service_results: tuple[ServiceDeployResult, ...] = ()
    error_message: str = ""


class ServiceStartRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    service_class: str


class ServiceActiveRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    active: bool


class ServiceStateRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    node_id: str
    field: str
    value: F8JsonValue


class ServiceCommandRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    call: str
    params: dict[str, F8JsonValue] = msgspec.field(default_factory=dict)


__all__ = [
    "CreateProjectRequest",
    "DeployJob",
    "DeployProjectRequest",
    "JobStatus",
    "ProjectRecord",
    "ProjectSummary",
    "RuntimeActionResult",
    "ServiceActiveRequest",
    "ServiceCommandRequest",
    "ServiceDeployResult",
    "ServiceRuntimeStatus",
    "ServiceStartRequest",
    "ServiceStateRequest",
    "UpdateProjectRequest",
    "ValidateDocumentRequest",
]
