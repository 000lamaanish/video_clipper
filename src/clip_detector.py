import numpy as np


def detect_peaks(
    energy: np.ndarray,
    threshold: float = 0.3,
    min_gap: int = 10,
) -> list[tuple[int, int]]:
    """
    Return (start, end) frame pairs where energy exceeds `threshold`
    for at least `min_gap` consecutive frames.
    """
    peaks = []
    i = 0
    n = len(energy)

    while i < n:
        if energy[i] > threshold:
            start = i

            # advance until energy drops back below threshold
            while i < n and energy[i] > threshold:
                i += 1

            end = i  # exclusive, consistent with Python slice convention

            if (end - start) >= min_gap:
                peaks.append((start, end))
            # do NOT increment i here — it already points past this peak

        else:
            i += 1  # only step forward when we're in a quiet region

    return peaks


def merge_segments(
    segments: list[tuple[int, int]],
    max_gap: int = 10,          # ← frames, not seconds (fix unit mismatch)
) -> list[tuple[int, int]]:
    """
    Merge consecutive segments whose inter-segment gap is within `max_gap` frames.

    Args:
        segments: Sorted list of (start_frame, end_frame) pairs.
        max_gap:  Maximum allowed gap in **frames** between two segments
                  before they are kept separate.
    """
    if not segments:
        return []

    # be defensive: sort so callers don't have to guarantee order
    segments = sorted(segments)
    merged = [segments[0]]

    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]

        if start <= prev_end + max_gap:
            # extend the last segment instead of appending a new one
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged