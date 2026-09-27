from __future__ import annotations

from f8pysdk.codec import decode_as, dump_json, encode_obj
from f8pysdk.rungraph_fingerprint import build_rungraph_deploy_fingerprint, build_rungraph_deploy_snapshot
from f8pysdk.specs import (
    F8DataPayloadSpec,
    F8DataPortDelivery,
    F8DataPortPayloadKind,
    F8DataPortSpec,
    F8DataStreamCongestion,
    F8DataStreamPriority,
    F8DataStreamReliability,
    F8DataStreamSpec,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8StateAccess,
    F8StateFieldEditPolicy,
    F8StateSpec,
    number_schema,
    string_schema,
)


def test_deploy_fingerprint_matches_payload_after_json_roundtrip() -> None:
    graph = F8RuntimeGraph(
        graphId="g1",
        revision="r1",
        nodes=[
            F8RuntimeNode(
                nodeId="svc",
                serviceId="svc",
                serviceClass="svc.video",
                dataOutPorts=[
                    F8DataPortSpec(
                        name="frame",
                        valueSchema={"type": "object"},
                        payload=F8DataPayloadSpec(
                            kind=F8DataPortPayloadKind.video_frame,
                            metadataSchema={"type": "object", "required": ["width", "height"]},
                            schemaVersion=1,
                            formats=["bgra32"],
                        ),
                        stream=F8DataStreamSpec(
                            delivery=F8DataPortDelivery.latest,
                            reliability=F8DataStreamReliability.best_effort,
                            congestion=F8DataStreamCongestion.drop,
                            priority=F8DataStreamPriority.real_time,
                        ),
                        payloadKind=F8DataPortPayloadKind.video_frame,
                        delivery=F8DataPortDelivery.latest,
                    )
                ],
                stateFields=[
                    F8StateSpec(
                        name="url",
                        valueSchema=string_schema(),
                        access=F8StateAccess.rw,
                        editPolicy=F8StateFieldEditPolicy(canRename=False, canEditValueSchema=False),
                    )
                ],
            )
        ],
        edges=[],
    )
    graph_payload = dump_json(graph, mode="json", by_alias=True)

    assert build_rungraph_deploy_fingerprint(graph) == build_rungraph_deploy_fingerprint(graph_payload)
    snapshot = build_rungraph_deploy_snapshot(graph)
    node = snapshot["nodes"][0]
    port = node["dataOutPorts"][0]
    state = node["stateFields"][0]
    assert port["payload"]["kind"] == "video_frame"
    assert port["stream"]["delivery"] == "latest"
    assert port["payloadKind"] == "video_frame"
    assert port["delivery"] == "latest"
    assert state["editPolicy"]["canRename"] is False


def test_deploy_fingerprint_survives_runtime_wire_defaults() -> None:
    graph = F8RuntimeGraph(
        graphId="g-defaults", revision="r1", services=[], edges=[],
        nodes=[F8RuntimeNode(
            nodeId="source", serviceId="svc", serviceClass="svc.test",
            dataInPorts=[F8DataPortSpec(name="input", valueSchema=number_schema())],
            stateFields=[F8StateSpec(name="label", valueSchema=string_schema(), access=F8StateAccess.rw)],
        )],
    )

    assert build_rungraph_deploy_fingerprint(graph) == build_rungraph_deploy_fingerprint(
        decode_as(encode_obj(graph), F8RuntimeGraph)
    )
