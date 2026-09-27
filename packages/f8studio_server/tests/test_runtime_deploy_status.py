from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from f8pysdk.rungraph_fingerprint import build_rungraph_deploy_fingerprint
from f8pysdk.service_runtime_tools.deploy.readiness import RungraphDeployStatusTimeout
from f8pysdk.specs import F8RuntimeGraph
from f8studio_server.models import ServiceRuntimeStatus
from f8studio_server.runtime import ZenohRuntimeGateway


def _status(graph: F8RuntimeGraph, fingerprint: str) -> ServiceRuntimeStatus:
    return ServiceRuntimeStatus(
        service_id="engine", service_class="f8.pyengine", runtime_instance_id="instance-1", active=True,
        rungraph_graph_id=str(graph.graphId), rungraph_revision=str(graph.revision),
        rungraph_fingerprint=fingerprint,
    )


def test_deploy_accepts_verified_rungraph_when_confirmation_is_missed() -> None:
    graph = F8RuntimeGraph(graphId="graph-1", revision="revision-1", services=[], nodes=[], edges=[])
    gateway = ZenohRuntimeGateway()
    fingerprint = build_rungraph_deploy_fingerprint(graph)
    with (
        patch.object(gateway, "_wait_until_ready", new=AsyncMock(return_value=_status(graph, "old"))),
        patch.object(gateway, "_request", new=AsyncMock(return_value=b"reply")),
        patch.object(gateway, "_connected_transport", new=AsyncMock(return_value=object())),
        patch.object(gateway, "status", new=AsyncMock(return_value=_status(graph, fingerprint))),
        patch("f8studio_server.runtime.decode_as", return_value=SimpleNamespace(ok=True)),
        patch("f8studio_server.runtime.wait_rungraph_deploy_status", new=AsyncMock(
            side_effect=RungraphDeployStatusTimeout("confirmation missing")
        )),
    ):
        result = asyncio.run(gateway.deploy(service_id="engine", graph=graph, force_apply=False))

    assert result.success


def test_deploy_does_not_accept_an_unverified_or_forced_rungraph() -> None:
    graph = F8RuntimeGraph(graphId="graph-1", revision="revision-1", services=[], nodes=[], edges=[])
    for fingerprint, force_apply in (("other", False), (build_rungraph_deploy_fingerprint(graph), True)):
        gateway = ZenohRuntimeGateway()
        with (
            patch.object(gateway, "_wait_until_ready", new=AsyncMock(return_value=_status(graph, "old"))),
            patch.object(gateway, "_request", new=AsyncMock(return_value=b"reply")),
            patch.object(gateway, "_connected_transport", new=AsyncMock(return_value=object())),
            patch.object(gateway, "status", new=AsyncMock(return_value=_status(graph, fingerprint))),
            patch("f8studio_server.runtime.decode_as", return_value=SimpleNamespace(ok=True)),
            patch("f8studio_server.runtime.wait_rungraph_deploy_status", new=AsyncMock(
                side_effect=RungraphDeployStatusTimeout("confirmation missing")
            )),
        ):
            result = asyncio.run(gateway.deploy(service_id="engine", graph=graph, force_apply=force_apply))

        assert not result.success
        assert result.error_message == "confirmation missing"
