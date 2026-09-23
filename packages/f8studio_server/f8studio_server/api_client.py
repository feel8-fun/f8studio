from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import quote

import httpx

from f8pysdk.specs import F8JsonValue


class StudioApiError(RuntimeError):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class StudioApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8260", *, timeout_s: float = 30.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> StudioApiClient:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def health(self) -> F8JsonValue:
        return self._request("GET", "/api/health")

    def projects(self) -> F8JsonValue:
        return self._request("GET", "/api/projects")

    def catalog(self) -> F8JsonValue:
        return self._request("GET", "/api/catalog")

    def graph(self, project_id: str) -> F8JsonValue:
        return self._request("GET", f"/api/projects/{project_id}")

    def preview_patch(self, project_id: str, patch: Mapping[str, Any]) -> F8JsonValue:
        return self._request("POST", f"/api/projects/{project_id}/patch:preview", json=dict(patch))

    def apply_patch(self, project_id: str, patch: Mapping[str, Any]) -> F8JsonValue:
        return self._request("POST", f"/api/projects/{project_id}/patch", json=dict(patch))

    def validate(self, project_id: str, document: Mapping[str, Any]) -> F8JsonValue:
        return self._request(
            "POST",
            f"/api/projects/{project_id}/validate",
            json={"document": dict(document)},
        )

    def deploy(self, project_id: str, *, graph_revision: int, request_id: str) -> F8JsonValue:
        return self._request(
            "POST",
            f"/api/projects/{project_id}/deploy",
            json={"requestId": request_id, "expectedGraphRevision": graph_revision},
        )

    def monitors(self, project_id: str | None = None) -> F8JsonValue:
        path = "/api/runtime/monitors"
        if project_id is not None:
            path = f"{path}?project_id={quote(project_id, safe='')}"
        return self._request("GET", path)

    def agent_providers(self) -> F8JsonValue:
        return self._request("GET", "/api/agents/providers")

    def create_agent_session(
        self,
        *,
        project_id: str,
        title: str,
        provider_id: str,
        model_id: str,
    ) -> F8JsonValue:
        return self._request(
            "POST",
            "/api/agents/sessions",
            json={
                "projectId": project_id,
                "title": title,
                "providerId": provider_id,
                "modelId": model_id,
            },
        )

    def agent_session(self, session_id: str) -> F8JsonValue:
        return self._request("GET", f"/api/agents/sessions/{session_id}")

    def start_agent_run(self, session_id: str, *, prompt: str) -> F8JsonValue:
        return self._request(
            "POST",
            f"/api/agents/sessions/{session_id}/runs",
            json={"prompt": prompt},
        )

    def cancel_agent_run(self, session_id: str) -> F8JsonValue:
        return self._request("DELETE", f"/api/agents/sessions/{session_id}/runs/current")

    def resolve_agent_approval(
        self,
        session_id: str,
        approval_id: str,
        *,
        approved: bool,
        arguments_hash: str,
    ) -> F8JsonValue:
        return self._request(
            "POST",
            f"/api/agents/sessions/{session_id}/approvals/{approval_id}",
            json={"approved": approved, "argumentsHash": arguments_hash},
        )

    def _request(self, method: str, path: str, *, json: object | None = None) -> F8JsonValue:
        response = self._client.request(method, path, json=json)
        try:
            payload: object = response.json()
        except ValueError as exc:
            raise StudioApiError(
                f"Studio returned non-JSON HTTP {response.status_code}",
                status_code=response.status_code,
            ) from exc
        if response.is_error:
            message = f"Studio request failed with HTTP {response.status_code}"
            if isinstance(payload, dict):
                payload_object = cast(dict[str, object], payload)
                detail = payload_object.get("detail")
                if isinstance(detail, str):
                    message = detail
                elif isinstance(detail, dict):
                    detail_object = cast(dict[str, object], detail)
                    detail_message = detail_object.get("message")
                    if isinstance(detail_message, str):
                        message = detail_message
            raise StudioApiError(message, status_code=response.status_code)
        return cast(F8JsonValue, payload)


__all__ = ["StudioApiClient", "StudioApiError"]
