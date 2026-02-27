from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PKG_AUDIOFEAT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_SDK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
for p in (PKG_AUDIOFEAT, PKG_SDK):
    if p not in sys.path:
        sys.path.insert(0, p)

from f8pyaudiofeat.feature_math import (  # noqa: E402
    compute_core_features,
    compute_pulse_clarity,
    compute_tempo_bpm,
    librosa_available,
)


@unittest.skipUnless(librosa_available(), "librosa is required")
class FeatureMathTests(unittest.TestCase):
    def test_silence_core_features(self) -> None:
        sr = 48_000
        window = 36_864
        hop = 3_072
        mono = np.zeros((window,), dtype=np.float32)
        rms, centroid, onset_env = compute_core_features(
            mono=mono,
            sample_rate=sr,
            window_length=window,
            hop_length=hop,
        )
        self.assertLess(abs(rms), 1e-8)
        self.assertLess(abs(centroid), 1e-6)
        self.assertGreaterEqual(onset_env.size, 1)
        self.assertLess(abs(float(onset_env[-1])), 1e-6)

    def test_sine_centroid_is_near_tone(self) -> None:
        sr = 48_000
        window = 36_864
        hop = 3_072
        hz = 1000.0
        t = np.arange(window, dtype=np.float32) / float(sr)
        mono = (0.4 * np.sin(2.0 * np.pi * hz * t)).astype(np.float32)
        rms, centroid, onset_env = compute_core_features(
            mono=mono,
            sample_rate=sr,
            window_length=window,
            hop_length=hop,
        )
        self.assertGreater(rms, 0.05)
        self.assertGreater(centroid, 500.0)
        self.assertLess(centroid, 3000.0)
        self.assertGreaterEqual(onset_env.size, 1)

    def test_pulse_onset_strength_high(self) -> None:
        sr = 48_000
        window = 36_864
        hop = 3_072
        mono = np.zeros((window,), dtype=np.float32)
        pulse_spacing = 3_000
        pulse_width = 400
        i = 0
        while i < window:
            mono[i : i + pulse_width] = 1.0
            i += pulse_spacing

        _, _, onset_env = compute_core_features(
            mono=mono,
            sample_rate=sr,
            window_length=window,
            hop_length=hop,
        )
        self.assertGreater(float(np.max(onset_env)), 0.1)

    def test_tempo_and_clarity(self) -> None:
        sr = 48_000
        hop = 512
        hops_per_second = float(sr) / float(hop)
        bpm_target = 120.0
        beat_hops = int(round((60.0 / bpm_target) * hops_per_second))

        onset = np.zeros((1024,), dtype=np.float32)
        idx = 0
        while idx < onset.size:
            onset[idx] = 1.0
            idx += beat_hops

        bpm = compute_tempo_bpm(onset_envelope=onset, sample_rate=sr, hop_length=hop)
        clarity = compute_pulse_clarity(onset)
        self.assertGreater(bpm, 80.0)
        self.assertLess(bpm, 160.0)
        self.assertGreater(clarity, 0.1)

        rng = np.random.default_rng(42)
        noise = rng.normal(loc=0.0, scale=1.0, size=1024).astype(np.float32)
        noise_clarity = compute_pulse_clarity(noise)
        self.assertLess(noise_clarity, 0.8)


if __name__ == "__main__":
    unittest.main()
