import os
import sys
import unittest
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.generated import F8RuntimeNode  # noqa: E402

from f8pyengine.operators.udp_skeleton import UdpSkeletonRuntimeNode  # noqa: E402


def _mk_payload(frame_id: int, chunk_index: int, chunk_count: int, bones: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "skeleton_binary",
        "modelName": "Model_A",
        "timestampMs": 1000,
        "schema": "f8.skeleton.v1",
        "boneCount": len(bones),
        "bones": bones,
        "trailer": {
            "magic": "LMEX",
            "extVersion": 1,
            "frameId": frame_id,
            "chunkIndex": chunk_index,
            "chunkCount": chunk_count,
            "totalBoneCount": 4,
            "characterId": 7,
        },
    }


class UdpSkeletonChunkReassemblyTests(unittest.TestCase):
    def _new_node(self) -> UdpSkeletonRuntimeNode:
        node = F8RuntimeNode(
            nodeId="udp1",
            serviceId="svcA",
            serviceClass="f8.pyengine",
            operatorClass=UdpSkeletonRuntimeNode.SPEC.operatorClass,
            stateFields=list(UdpSkeletonRuntimeNode.SPEC.stateFields or []),
            stateValues={},
        )
        return UdpSkeletonRuntimeNode(node_id="udp1", node=node, initial_state={})

    def test_reassembles_chunks_before_publish(self) -> None:
        runtime_node = self._new_node()
        part0 = _mk_payload(
            frame_id=10,
            chunk_index=0,
            chunk_count=2,
            bones=[{"name": "b0"}, {"name": "b1"}],
        )
        part1 = _mk_payload(
            frame_id=10,
            chunk_index=1,
            chunk_count=2,
            bones=[{"name": "b2"}, {"name": "b3"}],
        )

        result0 = runtime_node._merge_or_defer_chunk_payload(key="Model_A", payload=part0, rx_ts_ms=1000)
        self.assertIsNone(result0)

        result1 = runtime_node._merge_or_defer_chunk_payload(key="Model_A", payload=part1, rx_ts_ms=1001)
        self.assertIsInstance(result1, dict)
        assert isinstance(result1, dict)
        self.assertEqual(int(result1["boneCount"]), 4)
        self.assertEqual([b["name"] for b in result1["bones"]], ["b0", "b1", "b2", "b3"])
        self.assertEqual(int(result1["trailer"]["assembledChunkCount"]), 2)
        self.assertEqual(int(result1["trailer"]["chunkCount"]), 1)

    def test_ignores_older_frame_after_completion(self) -> None:
        runtime_node = self._new_node()
        part0 = _mk_payload(frame_id=8, chunk_index=0, chunk_count=2, bones=[{"name": "b0"}])
        part1 = _mk_payload(frame_id=8, chunk_index=1, chunk_count=2, bones=[{"name": "b1"}])
        _ = runtime_node._merge_or_defer_chunk_payload(key="Model_A", payload=part0, rx_ts_ms=1000)
        _ = runtime_node._merge_or_defer_chunk_payload(key="Model_A", payload=part1, rx_ts_ms=1001)

        old_part = _mk_payload(frame_id=7, chunk_index=0, chunk_count=2, bones=[{"name": "old"}])
        old_result = runtime_node._merge_or_defer_chunk_payload(key="Model_A", payload=old_part, rx_ts_ms=1002)
        self.assertIsNone(old_result)

    def test_non_chunk_payload_passthrough(self) -> None:
        runtime_node = self._new_node()
        payload = _mk_payload(frame_id=20, chunk_index=0, chunk_count=1, bones=[{"name": "single"}])
        result = runtime_node._merge_or_defer_chunk_payload(key="Model_A", payload=payload, rx_ts_ms=1000)
        self.assertIsInstance(result, dict)
        assert isinstance(result, dict)
        self.assertEqual(int(result["boneCount"]), 1)
        self.assertEqual(result["bones"][0]["name"], "single")


if __name__ == "__main__":
    unittest.main()
