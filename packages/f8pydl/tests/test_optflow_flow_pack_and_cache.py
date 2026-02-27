import os
import sys
import unittest


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_PYDL, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)


from f8pydl.optflow_service_node import (  # noqa: E402
    OptflowFramePairCache,
    PreparedFlowFrame,
    pack_flow2_f16_payload,
)


class OptflowPackAndCacheTests(unittest.TestCase):
    def test_pack_flow2_f16_payload_roundtrip(self) -> None:
        import numpy as np  # type: ignore

        flow = np.asarray(
            [
                [[1.25, -2.5], [0.0, 3.125]],
                [[-4.0, 5.5], [6.0, -7.75]],
            ],
            dtype=np.float32,
        )
        pitch, payload = pack_flow2_f16_payload(flow)
        self.assertEqual(pitch, 8)
        self.assertEqual(len(payload), 16)

        decoded = np.frombuffer(payload, dtype=np.float16).reshape((2, 2, 2)).astype(np.float32)
        self.assertTrue(np.allclose(decoded, flow, atol=1e-2))

    def test_frame_pair_cache_reuses_previous_tensor(self) -> None:
        cache = OptflowFramePairCache()
        tensor1 = object()
        tensor2 = object()
        tensor3 = object()
        f1 = PreparedFlowFrame(frame_id=1, width=640, height=480, tensor=tensor1)
        f2 = PreparedFlowFrame(frame_id=2, width=640, height=480, tensor=tensor2)
        f3 = PreparedFlowFrame(frame_id=3, width=640, height=480, tensor=tensor3)

        pair1 = cache.push_and_get_pair(f1)
        self.assertIsNone(pair1)

        pair2 = cache.push_and_get_pair(f2)
        assert pair2 is not None
        self.assertIs(pair2[0].tensor, tensor1)
        self.assertIs(pair2[1].tensor, tensor2)

        pair3 = cache.push_and_get_pair(f3)
        assert pair3 is not None
        self.assertIs(pair3[0].tensor, tensor2)
        self.assertIs(pair3[1].tensor, tensor3)

    def test_frame_pair_cache_resets_on_resolution_change(self) -> None:
        cache = OptflowFramePairCache()
        f1 = PreparedFlowFrame(frame_id=1, width=640, height=480, tensor=object())
        f2 = PreparedFlowFrame(frame_id=2, width=320, height=240, tensor=object())
        f3 = PreparedFlowFrame(frame_id=3, width=320, height=240, tensor=object())

        self.assertIsNone(cache.push_and_get_pair(f1))
        self.assertIsNone(cache.push_and_get_pair(f2))
        self.assertIsNotNone(cache.push_and_get_pair(f3))


if __name__ == "__main__":
    unittest.main()
