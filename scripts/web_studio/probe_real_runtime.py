from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from typing import TypeVar
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import msgspec

from f8studio_core.graph import CreateNodeOp, NodeCatalog, PatchRequest, PatchResult
from f8studio_core.graph import ConnectEdgeOp, GraphEdge, GraphEdgeKind
from f8pysdk.specs import F8MonitorSnapshot
from f8studio_server.catalog import CatalogSnapshot
from f8studio_server.models import (
    CreateProjectRequest,
    DeployJob,
    DeployProjectRequest,
    JobStatus,
    ProjectRecord,
    RuntimeActionResult,
    ServiceActiveRequest,
    ServiceRuntimeStatus,
    ServiceStartRequest,
    ServiceStateRequest,
)
from f8studio_server.processes import ManagedProcessResult


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


def _wait_for_status(base_url: str, service_id: str) -> ServiceRuntimeStatus:
    deadline = time.monotonic() + 15.0
    last_error = ""
    while time.monotonic() < deadline:
        try:
            return _request(
                base_url,
                "GET",
                f"/api/runtime/services/{service_id}/status",
                response_type=ServiceRuntimeStatus,
            )
        except RuntimeError as exc:
            last_error = str(exc)
            time.sleep(0.2)
    raise TimeoutError(f"service did not become ready: {last_error}")


def _wait_for_job(base_url: str, job_id: str) -> DeployJob:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        job = _request(base_url, "GET", f"/api/jobs/{job_id}", response_type=DeployJob)
        if job.status not in {JobStatus.queued, JobStatus.running}:
            return job
        time.sleep(0.1)
    raise TimeoutError(f"deploy job did not finish: {job_id}")


def _wait_for_monitor_activity(base_url: str, service_id: str) -> F8MonitorSnapshot:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        snapshots = _request(
            base_url,
            "GET",
            "/api/runtime/monitors",
            response_type=tuple[F8MonitorSnapshot, ...],
        )
        for snapshot in snapshots:
            processed = snapshot.frame.processed
            if (
                str(snapshot.serviceId) == service_id
                and not isinstance(processed, msgspec.UnsetType)
                and processed > 0
            ):
                return snapshot
        time.sleep(0.2)
    raise TimeoutError(f"runtime monitor did not report processed activity: {service_id}")


def run_probe(base_url: str) -> None:
    suffix = uuid4().hex[:10]
    service_id = f"engine{suffix}"
    project_id = f"probe{suffix}"
    started = False
    try:
        catalog_snapshot = _request(base_url, "GET", "/api/catalog", response_type=CatalogSnapshot)
        catalog = NodeCatalog(
            services=catalog_snapshot.services,
            operators=catalog_snapshot.operators,
        )
        service = catalog.create_service_node(
            node_id=service_id,
            service_class="f8.pyengine",
        )
        tick = catalog.create_operator_node(
            node_id="tick1",
            service_id=service_id,
            service_class="f8.pyengine",
            operator_class="f8.tick",
            state_values={"tickMs": 100, "hiResTimer": False},
        )
        printer = catalog.create_operator_node(
            node_id="print1",
            service_id=service_id,
            service_class="f8.pyengine",
            operator_class="f8.print",
            state_values={"strip": True},
        )
        tick_to_print = GraphEdge(
            edge_id="tick-to-print",
            from_node_id="tick1",
            from_port_id="exec:output:exec",
            to_node_id="print1",
            to_port_id="exec:input:exec",
            kind=GraphEdgeKind.exec,
        )

        process = _request(
            base_url,
            "POST",
            f"/api/runtime/services/{service_id}/start",
            response_type=ManagedProcessResult,
            body=ServiceStartRequest(service_class="f8.pyengine"),
        )
        if not process.running:
            raise RuntimeError("managed PyEngine process did not remain running")
        started = True
        initial_status = _wait_for_status(base_url, service_id)

        _request(
            base_url,
            "POST",
            "/api/projects",
            response_type=ProjectRecord,
            body=CreateProjectRequest(project_id=project_id, name="P2 Real Runtime Probe"),
        )
        patch = _request(
            base_url,
            "POST",
            f"/api/projects/{project_id}/patch",
            response_type=PatchResult,
            body=PatchRequest(
                request_id=f"patch{suffix}",
                expected_graph_revision=0,
                expected_layout_revision=0,
                operations=(
                    CreateNodeOp(node=service),
                    CreateNodeOp(node=tick),
                    CreateNodeOp(node=printer),
                    ConnectEdgeOp(edge=tick_to_print),
                ),
            ),
        )
        submitted = _request(
            base_url,
            "POST",
            f"/api/projects/{project_id}/deploy",
            response_type=DeployJob,
            body=DeployProjectRequest(
                request_id=f"deploy{suffix}",
                expected_graph_revision=patch.document.graph_revision,
            ),
        )
        deployed = _wait_for_job(base_url, submitted.job_id)
        if deployed.status != JobStatus.succeeded:
            raise RuntimeError(f"deployment failed: {deployed.status.value}: {deployed.error_message}")

        deployed_status = _wait_for_status(base_url, service_id)
        if deployed_status.rungraph_graph_id != project_id:
            raise RuntimeError(
                f"runtime reported graph {deployed_status.rungraph_graph_id!r}, expected {project_id!r}"
            )
        monitor = _wait_for_monitor_activity(base_url, service_id)
        state_result = _request(
            base_url,
            "POST",
            f"/api/runtime/services/{service_id}/state",
            response_type=RuntimeActionResult,
            body=ServiceStateRequest(node_id="tick1", field="tickMs", value=125),
        )
        if not state_result.success:
            raise RuntimeError(f"state write failed: {state_result.error_message}")
        for active in (False, True):
            action = _request(
                base_url,
                "POST",
                f"/api/runtime/services/{service_id}/active",
                response_type=RuntimeActionResult,
                body=ServiceActiveRequest(active=active),
            )
            if not action.success:
                raise RuntimeError(f"active={active} failed: {action.error_message}")

        print(
            "real runtime probe passed: "
            f"service={service_id} runtime={initial_status.runtime_instance_id} "
            f"graph={deployed_status.rungraph_graph_id} revision={deployed_status.rungraph_revision} "
            f"observed={monitor.frame.observed} processed={monitor.frame.processed}"
        )
    finally:
        if started:
            stopped = _request(
                base_url,
                "POST",
                f"/api/runtime/services/{service_id}/stop",
                response_type=ManagedProcessResult,
            )
            if stopped.running:
                raise RuntimeError(f"managed service still running after stop: {service_id}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise the Web Studio API against a real PyEngine process.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8231")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_probe(str(args.base_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
