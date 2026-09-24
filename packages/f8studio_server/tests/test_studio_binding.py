import asyncio
from typing import cast
from unittest.mock import AsyncMock

from f8pysdk.specs import F8Edge, F8EdgeKindEnum, F8RuntimeGraph, F8RuntimeNode, F8RuntimeService
from f8studio_server.models import ServiceDeployResult, ServiceRuntimeStatus
from f8studio_server.runtime import RuntimeGateway, StudioBoundRuntimeGateway


def test_studio_runtime_binding_isolates_deployments_and_preserves_logical_ids() -> None:
    async def scenario() -> None:
        remote = AsyncMock()
        remote.deploy.return_value = ServiceDeployResult(service_id="studio_local", success=True)
        remote.status.return_value = ServiceRuntimeStatus(
            service_id="studio_local", service_class="f8.pystudio", runtime_instance_id="instance", active=True,
        )
        gateway = StudioBoundRuntimeGateway(cast(RuntimeGateway, remote), studio_service_id="studio_local")
        graph = F8RuntimeGraph(
            graphId="video", revision="r1",
            services=[F8RuntimeService(serviceId="studio", serviceClass="f8.pystudio")],
            nodes=[
                F8RuntimeNode(nodeId="studio", serviceId="studio", serviceClass="f8.pystudio"),
                F8RuntimeNode(nodeId="viewer", serviceId="studio", serviceClass="f8.pystudio", operatorClass="f8.viz.video"),
            ],
            edges=[F8Edge(
                edgeId="video", fromServiceId="player", fromOperatorId="player", fromPort="video",
                toServiceId="studio", toOperatorId="viewer", toPort="video", kind=F8EdgeKindEnum.data,
            )],
        )

        result = await gateway.deploy(service_id="studio", graph=graph, force_apply=False)
        assert result.service_id == "studio"
        bound = remote.deploy.await_args.kwargs["graph"]
        assert remote.deploy.await_args.kwargs["service_id"] == "studio_local"
        assert bound.services[0].serviceId == "studio_local"
        assert [(node.nodeId, node.serviceId) for node in bound.nodes] == [
            ("studio_local", "studio_local"), ("viewer", "studio_local"),
        ]
        assert (bound.edges[0].fromServiceId, bound.edges[0].toServiceId) == ("player", "studio_local")
        assert bound.edges[0].toOperatorId == "viewer"
        assert (await gateway.status("studio")).service_id == "studio"
        remote.status.assert_awaited_with("studio_local")

        await gateway.set_state("studio", node_id="studio", field="tickMs", value=100)
        remote.set_state.assert_awaited_with("studio_local", node_id="studio_local", field="tickMs", value=100)
        await gateway.terminate("studio")
        assert remote.deploy.await_args.kwargs["service_id"] == "studio_local"
        assert remote.deploy.await_args.kwargs["graph"].nodes == []
        remote.terminate.assert_not_awaited()

    asyncio.run(scenario())
