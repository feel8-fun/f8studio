from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Sequence
from typing import TypeVar, cast
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

import msgspec
from websockets.asyncio.client import connect
from websockets.typing import Origin

from f8studio_core.graph import CreateNodeOp, NodeCatalog, PatchRequest, PatchResult
from f8studio_server.catalog import CatalogSnapshot
from f8studio_server.models import (
    CreateProjectRequest,
    DeployJob,
    DeployProjectRequest,
    JobStatus,
    ProjectRecord,
    RuntimeActionResult,
    ServiceStateRequest,
)


T = TypeVar("T")


class ProbeEvent(msgspec.Struct, rename="camel"):
    type: str
    payload: object


class ProbePresentation(msgspec.Struct, rename="camel"):
    node_id: str
    command: str
    payload: object


def _request(base_url: str, method: str, path: str, *, response_type: type[T], body: object | None = None) -> T:
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
    project_id = f"presentation{suffix}"
    catalog_snapshot = await asyncio.to_thread(
        _request,
        base_url,
        "GET",
        "/api/catalog",
        response_type=CatalogSnapshot,
    )
    catalog = NodeCatalog(services=catalog_snapshot.services, operators=catalog_snapshot.operators)
    studio = catalog.create_service_node(node_id="studio", service_class="f8.pystudio")
    three_d = catalog.create_operator_node(
        node_id="three1",
        service_id="studio",
        service_class="f8.pystudio",
        operator_class="f8.viz.three_d",
        state_values={"worldUp": "+y", "throttleMs": 0},
    )

    parsed = urlparse(base_url)
    websocket_url = f"ws://{parsed.netloc}/api/events"
    async with connect(websocket_url, origin=Origin(base_url)) as websocket:
        snapshot_raw = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        snapshot = msgspec.json.decode(snapshot_raw, type=ProbeEvent)
        if snapshot.type != "stream.snapshot":
            raise RuntimeError(f"expected initial event snapshot, received {snapshot!r}")

        await asyncio.to_thread(
            _request,
            base_url,
            "POST",
            "/api/projects",
            response_type=ProjectRecord,
            body=CreateProjectRequest(project_id=project_id, name="P2 Studio Presentation Probe"),
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
                operations=(CreateNodeOp(node=studio), CreateNodeOp(node=three_d)),
            ),
        )
        submitted = await asyncio.to_thread(
            _request,
            base_url,
            "POST",
            f"/api/projects/{project_id}/deploy",
            response_type=DeployJob,
            body=DeployProjectRequest(
                request_id=f"deploy{suffix}",
                expected_graph_revision=patch.document.graph_revision,
            ),
        )
        deployed = await asyncio.to_thread(_wait_for_job, base_url, submitted.job_id)
        if deployed.status != JobStatus.succeeded:
            raise RuntimeError(f"Studio deployment failed: {deployed.status.value}: {deployed.error_message}")

        state_result = await asyncio.to_thread(
            _request,
            base_url,
            "POST",
            "/api/runtime/services/studio/state",
            response_type=RuntimeActionResult,
            body=ServiceStateRequest(node_id="three1", field="worldUp", value="-z"),
        )
        if not state_result.success:
            raise RuntimeError(f"Studio state write failed: {state_result.error_message}")

        deadline = asyncio.get_running_loop().time() + 10.0
        while asyncio.get_running_loop().time() < deadline:
            raw_event = await asyncio.wait_for(websocket.recv(), timeout=deadline - asyncio.get_running_loop().time())
            event = msgspec.json.decode(raw_event, type=ProbeEvent)
            if event.type != "presentation.command":
                continue
            try:
                payload = msgspec.convert(event.payload, type=ProbePresentation)
            except (msgspec.ValidationError, TypeError):
                continue
            if payload.node_id != "three1" or payload.command != "viz.three_d.world_up":
                continue
            command_payload = payload.payload
            if command_payload != {"worldUp": "-z"}:
                raise RuntimeError(f"unexpected presentation payload: {command_payload!r}")
            print(
                "Studio presentation probe passed: "
                f"project={project_id} job={deployed.job_id} command=viz.three_d.world_up worldUp=-z"
            )
            return
    raise TimeoutError("presentation.command was not observed within 10 seconds")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy an internal Studio graph and observe a real WebSocket presentation event.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8231")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    asyncio.run(run_probe(cast(str, args.base_url)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
