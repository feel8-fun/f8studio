"""Mapping between graph-visible Studio IDs and isolated runtime IDs."""
from __future__ import annotations

from dataclasses import dataclass

from f8pysdk.f8_naming import ensure_token

from .studio_runtime.identifiers import STUDIO_SERVICE_ID


@dataclass(frozen=True)
class StudioRuntimeIdentity:
    private_service_id: str | None = None

    def __post_init__(self) -> None:
        if self.private_service_id is not None:
            ensure_token(self.private_service_id, label="studio_service_id")

    def to_runtime(self, identifier: str) -> str:
        if identifier == STUDIO_SERVICE_ID and self.private_service_id is not None:
            return self.private_service_id
        return identifier

    def to_public(self, service_id: str, node_id: str) -> tuple[str, str] | None:
        """Return a public endpoint, or discard traffic from another Studio."""
        if service_id.startswith("studio_"):
            if service_id != self.private_service_id:
                return None
            return STUDIO_SERVICE_ID, STUDIO_SERVICE_ID if node_id == service_id else node_id
        if service_id == STUDIO_SERVICE_ID and self.private_service_id is not None:
            return None
        return service_id, node_id
