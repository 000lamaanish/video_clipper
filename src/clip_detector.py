import numpy as np

def detect_peaks(energy, threshold=0.3, min_gap=10):
    peaks = []

    i = 0
    while i < len(energy):
        if energy[i] > threshold:
            start = i

            while i < len(energy) and energy[i] > threshold:
                i += 1

            end = i

            if end - start > min_gap:
                peaks.append((start, end))
        i += 1

    return peaks
def merge_segments(segments, max_gap=30):
    if not segments:
        return []

    merged = [segments[0]]

    for current in segments[1:]:
        last = merged[-1]

        # if close → merge
        if current[0] - last[1] <= max_gap:
            merged[-1] = (last[0], current[1])
        else:
            merged.append(current)

    return merged