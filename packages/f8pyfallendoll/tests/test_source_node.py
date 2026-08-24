from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from f8pysdk.specs import F8RuntimeNode

from f8pyfallendoll.constants import SERVICE_CLASS
from f8pyfallendoll.main import build_app
from f8pyfallendoll.node_registry import FallenDollSourceNode, register_specs
from f8pysdk.registry import Registry


class _RecordingSourceNode(FallenDollSourceNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> None:
        super().__init__(node_id=node_id, node=node, initial_state=initial_state)
        self.emitted: list[tuple[str, Any, int | None]] = []
        self.states: dict[str, Any] = {}

    async def emit(
        self,
        port: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        ctx_id: str | int | None = None,
    ) -> None:
        del ctx_id
        self.emitted.append((port, value, ts_ms))

    async def set_state(
        self,
        field: str,
        value: Any,
        *,
        ts_ms: int | None = None,
        force_publish: bool = False,
    ) -> None:
        del ts_ms, force_publish
        self.states[field] = value


def _packet(*, role: str, preferred_bone: str, timestamp_ms: int = 1000) -> dict[str, Any]:
    return {
        "type": "skeleton_binary",
        "schema": "fallen-doll-ue-world-v1",
        "modelName": role,
        "stableKey": f"fallen-doll:{role}:0",
        "timestampMs": timestamp_ms,
        "bones": [{"name": preferred_bone, "pos": [0.0, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]}],
        "trailer": {
            "profileId": "fallen-doll",
            "hanimeActive": True,
            "hanimeId": "Hand02",
            "hanimeAsset": "/Game/HAnime/Hand02",
            "hanimeCategory": "hand",
            "role": role,
            "roleIndex": 0,
            "participantPriority": 0,
            "preferredBones": [preferred_bone],
        },
    }


def _runtime_node() -> F8RuntimeNode:
    registry = Registry()
    register_specs(registry)
    spec = registry.describe(SERVICE_CLASS).service
    return F8RuntimeNode(
        nodeId="fallen-doll",
        serviceId="fallen-doll",
        serviceClass=SERVICE_CLASS,
        dataInPorts=list(spec.dataInPorts or []),
        dataOutPorts=list(spec.dataOutPorts or []),
        stateFields=list(spec.stateFields or []),
    )


def test_describe_exposes_standard_source_ports() -> None:
    service = build_app().describe_json()["service"]
    assert service["serviceClass"] == SERVICE_CLASS
    assert [port["name"] for port in service["dataOutPorts"][:6]] == [
        "skeletons",
        "referenceSkeleton",
        "targetSkeleton",
        "referenceBone",
        "targetBone",
        "status",
    ]


def test_poll_emits_selected_frame_then_one_stale_frame(tmp_path: Path) -> None:
    spool = tmp_path / "fd-skeleton.ndjson"
    packets = [_packet(role="male", preferred_bone="Penis02"), _packet(role="female", preferred_bone="R_Hand")]
    spool.write_text("".join(json.dumps(packet) + "\n" for packet in packets), encoding="utf-8")
    node = _RecordingSourceNode(
        node_id="fallen-doll",
        node=_runtime_node(),
        initial_state={"runtimeDir": str(tmp_path), "staleAfterMs": 250},
    )

    async def run_case() -> None:
        await node._poll_once(2000)
        outputs = {port: value for port, value, _timestamp in node.emitted}
        assert node.states["connected"] is True
        assert outputs["referenceBone"]["name"] == "Penis02"
        assert outputs["targetBone"]["name"] == "R_Hand"
        assert outputs["targetBone"]["basis"] == {"up": "+local_z", "right": "-local_y"}
        assert outputs["status"]["valid"] is True

        emitted_before_stale = len(node.emitted)
        await node._poll_once(2300)
        assert len(node.emitted) == emitted_before_stale + 6
        assert node.emitted[-1][1]["reason"] == "stale"

        await node._poll_once(2600)
        assert len(node.emitted) == emitted_before_stale + 6

    asyncio.run(run_case())
