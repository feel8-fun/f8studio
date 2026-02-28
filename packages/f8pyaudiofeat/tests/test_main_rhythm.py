from __future__ import annotations

import os
import sys
import unittest


PKG_AUDIOFEAT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_AUDIOFEAT, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)

from f8pyaudiofeat.main_rhythm import AudioFeatureRhythmService  # noqa: E402


class AudioFeatureRhythmServiceTests(unittest.TestCase):
    def test_program_defaults_data_delivery_to_both(self) -> None:
        program = AudioFeatureRhythmService()
        cfg = program.build_runtime_config(service_id="svcA", nats_url="mem://")
        self.assertEqual(str(cfg.bus.data_delivery), "both")


if __name__ == "__main__":
    unittest.main()
