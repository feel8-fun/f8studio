from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Sequence
from typing import TypeVar, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import msgspec

from f8studio_core.graph import CreateNodeOp, NodeCatalog, PatchRequest, PatchResult
from f8studio_server.catalog import CatalogSnapshot
from f8studio_server.models import (
    CreateProjectRequest,
    DeployJob,
    DeployProjectRequest,
    JobStatus,
    ProjectRecord,
    ServiceStartRequest,
)
from f8studio_server.processes import ManagedProcessResult
from probe_webrtc_media import run_probe as run_media_probe


T = TypeVar("T")


def _request(
    base_url: str,
    method: str,
    path: str,
    *,
    response_type: type[T],
    body: object | None = None,
) -> T:
    encoded = None if body is None else msgspec.json.encode(body)
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=encoded,
        headers={} if encoded is None else {"content-type": "application/json"},
        method=method,
    )
    try:
        with urlopen(request, timeout=20.0) as response:
            return msgspec.json.decode(response.read(), type=response_type)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc


def _wait_for_job(base_url: str, job_id: str) -> DeployJob:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        job = _request(base_url, "GET", f"/api/jobs/{job_id}", response_type=DeployJob)
        if job.status not in {JobStatus.queued, JobStatus.running}:
            return job
        time.sleep(0.1)
    raise TimeoutError(f"deploy job did not finish: {job_id}")


async def run_probe(base_url: str) -> None:
    suffix = uuid4().hex[:10]
    service_id = f"capture{suffix}"
    project_id = f"video{suffix}"
    started = False
    try:
        catalog_snapshot = await asyncio.to_thread(
            _request, base_url, "GET", "/api/catalog", response_type=CatalogSnapshot
        )
        catalog = NodeCatalog(services=catalog_snapshot.services, operators=catalog_snapshot.operators)
        service = catalog.create_service_node(
            node_id=service_id,
            service_class="f8.screencap",
            state_values={"mode": "display", "fps": 30.0},
        )
        process = await asyncio.to_thread(
            _request,
            base_url,
            "POST",
            f"/api/runtime/services/{service_id}/start",
            response_type=ManagedProcessResult,
            body=ServiceStartRequest(service_class="f8.screencap"),
        )
        if not process.running:
            raise RuntimeError("managed screen capture process did not remain running")
        started = True
        # Process creation precedes complete C++ control-endpoint registration.
        await asyncio.sleep(1.0)

        await asyncio.to_thread(
            _request,
            base_url,
            "POST",
            "/api/projects",
            response_type=ProjectRecord,
            body=CreateProjectRequest(project_id=project_id, name="P3 Real Video Probe"),
        )
        patch = await asyncio.to_thread(
            _request,
            base_url,
            "POST",
            f"/api/projects/{project_id}/patch",
            response_type=PatchResult,
            body=PatchRequest(
                request_id=f"patch{suffix}",
                expected_graph_revision=0,
                expected_layout_revision=0,
                operations=(CreateNodeOp(node=service),),
            ),
        )
        submitted = await asyncio.to_thread(
            _request,
            base_url,
            "POST",
            f"/api/projects/{project_id}/deploy",
            response_type=DeployJob,
            body=DeployProjectRequest(
                request_id=f"deploy{suffix}", expected_graph_revision=patch.document.graph_revision
            ),
        )
        deployed = await asyncio.to_thread(_wait_for_job, base_url, submitted.job_id)
        if deployed.status != JobStatus.succeeded:
            raise RuntimeError(f"screen capture deployment failed: {deployed.status.value}: {deployed.error_message}")

        source = f"f8/svc/{service_id}/nodes/{service_id}/data/video"
        await run_media_probe(base_url, "main", source)
        print(f"Real video probe passed: service={service_id} project={project_id} source={source}")
    finally:
        if started:
            stopped = await asyncio.to_thread(
                _request,
                base_url,
                "POST",
                f"/api/runtime/services/{service_id}/stop",
                response_type=ManagedProcessResult,
            )
            if stopped.running:
                raise RuntimeError(f"managed screen capture still running after stop: {service_id}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy real screen capture and receive it through WebRTC.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8210")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    asyncio.run(run_probe(cast(str, args.base_url)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
