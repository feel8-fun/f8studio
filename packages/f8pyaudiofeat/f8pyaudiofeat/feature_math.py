from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import librosa
except ModuleNotFoundError:
    librosa = None  # type: ignore[assignment]


def librosa_available() -> bool:
    return librosa is not None


def compute_core_features(
    *,
    mono: np.ndarray,
    sample_rate: int,
    window_length: int,
    hop_length: int,
) -> tuple[float, float, np.ndarray]:
    if librosa is None:
        raise RuntimeError("librosa is not installed")
    if mono.ndim != 1:
        raise ValueError("mono must be 1D")
    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be positive")
    if int(window_length) <= 0:
        raise ValueError("window_length must be positive")
    if int(hop_length) <= 0:
        raise ValueError("hop_length must be positive")
    if int(mono.size) < int(window_length):
        raise ValueError("mono length must be >= window_length")

    window = mono[-int(window_length) :]
    rms_arr = librosa.feature.rms(y=window, frame_length=int(window_length), hop_length=int(hop_length), center=False)
    centroid_arr = librosa.feature.spectral_centroid(
        y=window,
        sr=int(sample_rate),
        n_fft=int(window_length),
        hop_length=int(hop_length),
        center=False,
    )
    onset_env = librosa.onset.onset_strength(y=window, sr=int(sample_rate), hop_length=int(hop_length), center=False)
    rms_value = float(rms_arr[0, -1]) if rms_arr.size > 0 else 0.0
    centroid_value = float(centroid_arr[0, -1]) if centroid_arr.size > 0 else 0.0
    onset_out = np.asarray(onset_env, dtype=np.float32)
    return rms_value, centroid_value, onset_out


def compute_tempo_bpm(*, onset_envelope: np.ndarray, sample_rate: int, hop_length: int) -> float:
    if librosa is None:
        raise RuntimeError("librosa is not installed")
    if onset_envelope.ndim != 1:
        raise ValueError("onset_envelope must be 1D")
    if onset_envelope.size < 4:
        return 0.0
    tempo_arr = librosa.feature.tempo(onset_envelope=onset_envelope, sr=int(sample_rate), hop_length=int(hop_length))
    if tempo_arr.size <= 0:
        return 0.0
    return float(tempo_arr[0])


def compute_pulse_clarity(onset_envelope: np.ndarray) -> float:
    if onset_envelope.ndim != 1:
        raise ValueError("onset_envelope must be 1D")
    if onset_envelope.size < 8:
        return 0.0

    x = np.asarray(onset_envelope, dtype=np.float32)
    x = x - float(np.mean(x))
    std = float(np.std(x))
    if std <= 1e-8:
        return 0.0
    x = x / std

    ac = np.correlate(x, x, mode="full")
    center = int(ac.size // 2)
    positive = ac[center + 1 :]
    if positive.size < 4:
        return 0.0

    max_lag = max(4, int(positive.size // 2))
    usable = positive[:max_lag]
    if usable.size <= 0:
        return 0.0

    peak = float(np.max(usable))
    floor = float(np.mean(usable))
    denom = float(ac[center])
    if denom <= 1e-8:
        return 0.0

    clarity = (peak - floor) / denom
    if clarity < 0.0:
        return 0.0
    if clarity > 1.0:
        return 1.0
    return float(clarity)


def select_recent_onset(envelope: Sequence[float], *, hops: int) -> np.ndarray:
    if int(hops) <= 0:
        return np.asarray([], dtype=np.float32)
    arr = np.asarray(list(envelope), dtype=np.float32)
    if arr.size <= int(hops):
        return arr
    return arr[-int(hops) :]
