"""
audio_analysis.py
-----------------
Responsible for ONE thing only:
    Load audio and return a normalized energy array.

No execution, no plotting, no cutting.
Just pure functions that main.py calls.
"""

import numpy as np
import librosa


# ── Constants (used as defaults, can be overridden by caller) ─────────────────
FRAME_LENGTH = 2048
HOP_LENGTH   = 512


def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """
    Load a .wav file using librosa.

    Args:
        audio_path: Path to the .wav file.

    Returns:
        (y, sr) — audio time series and sample rate.
    """
    print(f"[Audio] Loading: {audio_path}")
    y, sr = librosa.load(audio_path)
    print(f"[Audio] Loaded — duration: {len(y) / sr:.1f}s  |  sample rate: {sr}Hz")
    return y, sr


def compute_energy(
    y:            np.ndarray,
    sr:           int,
    frame_length: int = FRAME_LENGTH,
    hop_length:   int = HOP_LENGTH,
) -> np.ndarray:
    """
    Compute normalized short-time RMS energy from an audio signal.

    Args:
        y:            Audio time series (from load_audio).
        sr:           Sample rate (from load_audio).
        frame_length: FFT frame size in samples.
        hop_length:   Hop size between frames in samples.

    Returns:
        1-D numpy array of normalized energy values in [0, 1].
    """
    energy = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    # normalize to [0, 1] so threshold logic is consistent across different videos
    energy /= energy.max()

    print(f"[Audio] Energy frames: {len(energy)}")
    print(f"[Audio] max={energy.max():.3f}  mean={energy.mean():.3f}  min={energy.min():.3f}")

    return energy


def compute_threshold(energy: np.ndarray, sensitivity: float = 0.5) -> float:
    """
    Compute a dynamic threshold based on mean + (sensitivity * std).

    Higher sensitivity → lower threshold → more peaks detected.
    Lower sensitivity  → higher threshold → only the loudest moments.

    Args:
        energy:      Normalized energy array (from compute_energy).
        sensitivity: Multiplier on std deviation. Default 0.5 works well for streams.

    Returns:
        Float threshold value.
    """
    threshold = float(energy.mean() + sensitivity * energy.std())
    print(f"[Audio] Threshold: {threshold:.3f}  (sensitivity={sensitivity})")
    return threshold