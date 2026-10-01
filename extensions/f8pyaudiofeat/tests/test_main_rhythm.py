from __future__ import annotations

import os
import sys
import unittest


PKG_AUDIOFEAT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (PKG_AUDIOFEAT,):
    if p not in sys.path:
        sys.path.insert(0, p)

from f8pyaudiofeat.main_rhythm import build_app  # noqa: E402


class AudioFeatureRhythmServiceTests(unittest.TestCase):
    def test_program_defaults_data_delivery_to_callback(self) -> None:
        app = build_app()
        cfg = app.build_runtime_config(service_id="svcA")
        self.assertEqual(str(cfg.bus.data_delivery), "callback")


if __name__ == "__main__":
    unittest.main()
