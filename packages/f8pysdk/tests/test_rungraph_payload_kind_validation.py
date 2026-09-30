from __future__ import annotations
from f8pysdk.specs import F8DataPayloadSpec, F8DataPortPayloadKind

import pytest

from f8pysdk.rungraph_validation import validate_data_edges_or_raise
from f8pysdk.specs import (
    F8DataPortSpec,
    F8Edge,
    F8EdgeKindEnum,
    F8RuntimeGraph,
    F8RuntimeNode,
    any_schema,
    video_frame_port,
)


def test_data_edge_rejects_video_frame_stream_into_json_port() -> None:
    graph = F8RuntimeGraph(
        graphId="payload-kind",
        revision="r1",
        nodes=[
            F8RuntimeNode(
                nodeId="player",
                serviceId="svc_player",
                serviceClass="f8.implayer",
                operatorClass="f8.implayer",
                dataOutPorts=[video_frame_port(name="video")],
            ),
            F8RuntimeNode(
                nodeId="text",
                serviceId="studio",
                serviceClass="f8.pystudio",
                operatorClass="f8.viz.text",
                dataInPorts=[F8DataPortSpec(name="inputData", payload=F8DataPayloadSpec(kind=F8DataPortPayloadKind.json, valueSchema=any_schema()))],
            ),
        ],
        edges=[
            F8Edge(
                edgeId="e1",
                fromServiceId="svc_player",
                fromOperatorId="player",
                fromPort="video",
                toServiceId="studio",
                toOperatorId="text",
                toPort="inputData",
                kind=F8EdgeKindEnum.data,
            )
        ],
    )

    with pytest.raises(ValueError, match="payload kind mismatch"):
        validate_data_edges_or_raise(graph)


def test_data_edge_allows_video_frame_stream_into_video_frame_port() -> None:
    graph = F8RuntimeGraph(
        graphId="payload-kind",
        revision="r1",
        nodes=[
            F8RuntimeNode(
                nodeId="player",
                serviceId="svc_player",
                serviceClass="f8.implayer",
                operatorClass="f8.implayer",
                dataOutPorts=[video_frame_port(name="video")],
            ),
            F8RuntimeNode(
                nodeId="viewer",
                serviceId="studio",
                serviceClass="f8.pystudio",
                operatorClass="f8.viz.video",
                dataInPorts=[video_frame_port(name="video")],
            ),
        ],
        edges=[
            F8Edge(
                edgeId="e1",
                fromServiceId="svc_player",
                fromOperatorId="player",
                fromPort="video",
                toServiceId="studio",
                toOperatorId="viewer",
                toPort="video",
                kind=F8EdgeKindEnum.data,
            )
        ],
    )

    validate_data_edges_or_raise(graph)
