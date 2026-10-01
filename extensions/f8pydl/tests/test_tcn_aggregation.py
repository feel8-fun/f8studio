import os
import sys
import unittest


PKG_PYDL = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_PYDL not in sys.path:
    sys.path.insert(0, PKG_PYDL)


from f8pydl.tcnwave_service_node import DelayedAverageAggregator  # noqa: E402


class TcnAggregationTests(unittest.TestCase):
    def test_k1_outputs_current_frame_without_extra_delay(self) -> None:
        agg = DelayedAverageAggregator()
        for frame_index in range(4):
            _ = agg.register_frame(frame_id=100 + frame_index, ts_ms=1000 + frame_index * 33)
            _ = agg.apply_window(window_end_index=frame_index, values=[float(frame_index + 1)])
            ready = agg.pop_ready(latest_window_end_index=frame_index, output_length=1)
            self.assertEqual(len(ready), 1)
            self.assertEqual(ready[0].frame_index, frame_index)
            self.assertEqual(ready[0].frame_id, 100 + frame_index)
            self.assertIsInstance(ready[0].value, float)
            self.assertAlmostEqual(ready[0].value, float(frame_index + 1), places=6)

    def test_k10_delayed_average_matches_overlap_semantics(self) -> None:
        agg = DelayedAverageAggregator()
        for frame_index in range(11):
            _ = agg.register_frame(frame_id=200 + frame_index, ts_ms=2000 + frame_index * 33)

        values0 = [float(v) for v in range(10)]  # maps to frame 0..9
        _ = agg.apply_window(window_end_index=9, values=values0)
        ready0 = agg.pop_ready(latest_window_end_index=9, output_length=10)
        self.assertEqual(len(ready0), 1)
        self.assertEqual(ready0[0].frame_index, 0)
        self.assertAlmostEqual(ready0[0].value, 0.0, places=6)

        values1 = [float(v) for v in range(10, 20)]  # maps to frame 1..10
        _ = agg.apply_window(window_end_index=10, values=values1)
        ready1 = agg.pop_ready(latest_window_end_index=10, output_length=10)
        self.assertEqual(len(ready1), 1)
        self.assertEqual(ready1[0].frame_index, 1)
        self.assertAlmostEqual(ready1[0].value, 5.5, places=6)  # (1 + 10) / 2

    def test_duplicate_and_gapped_frame_ids_do_not_break_order(self) -> None:
        agg = DelayedAverageAggregator()
        ids = [7, 7, 15, 21]
        for frame_index, frame_id in enumerate(ids):
            _ = agg.register_frame(frame_id=frame_id, ts_ms=3000 + frame_index * 10)
            _ = agg.apply_window(window_end_index=frame_index, values=[1.0])
            ready = agg.pop_ready(latest_window_end_index=frame_index, output_length=1)
            self.assertEqual(ready[0].frame_index, frame_index)
            self.assertEqual(ready[0].frame_id, frame_id)


if __name__ == "__main__":
    unittest.main()
