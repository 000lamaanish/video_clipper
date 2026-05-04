"""
scorer.py
---------
Responsible for ONE thing only:
    Take energy peaks + whisper segments and return
    the top N (start_sec, end_sec) timestamps.

Key fix: diversity enforcement — no two returned clips
can overlap or be within MIN_SPREAD_S of each other.
"""

import numpy as np
from dataclasses import dataclass
from transcriber import TranscribedSegment


# ── Config defaults ───────────────────────────────────────────────────────────
WINDOW_FRAMES  = 968
MIN_DURATION_S = 8.0
MAX_CLIPS      = 6
WEIGHT_ENERGY  = 0.5
WEIGHT_KEYWORD = 0.5
MIN_SPREAD_S   = 60.0   # minimum seconds between any two selected clips


@dataclass
class ScoredSegment:
    """A single ranked highlight segment ready for video cutting."""
    start_sec:      float
    end_sec:        float
    combined_score: float
    energy_score:   float
    keyword_score:  float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


# ── Internal helpers ──────────────────────────────────────────────────────────

def _frames_to_sec(frame: int, hop_length: int, sr: int) -> float:
    return frame * hop_length / sr


def _energy_score(energy: np.ndarray, start_frame: int, end_frame: int) -> float:
    duration = end_frame - start_frame
    if duration <= 0:
        return 0.0
    return float(energy[start_frame:end_frame].mean()) * (duration ** 1.3)


def _keyword_score(
    start_sec:     float,
    end_sec:       float,
    hype_segments: list[TranscribedSegment],
) -> float:
    return float(sum(
        seg.keyword_score
        for seg in hype_segments
        if seg.start < end_sec and seg.end > start_sec
    ))


def _expand_peak(
    start_frame: int,
    end_frame:   int,
    half_win:    int,
    max_frame:   int,
) -> tuple[int, int]:
    centre = (start_frame + end_frame) // 2
    return max(0, centre - half_win), min(max_frame, centre + half_win)


def _overlaps_any(
    start_s:  float,
    end_s:    float,
    selected: list[ScoredSegment],
    spread_s: float,
) -> bool:
    """
    Return True if this segment overlaps or is too close to
    any already-selected segment.
    """
    for sel in selected:
        # check if intervals are within spread_s of each other
        if start_s < sel.end_sec + spread_s and end_s > sel.start_sec - spread_s:
            return True
    return False


# ── Public API ────────────────────────────────────────────────────────────────

def rank_segments(
    energy:         np.ndarray,
    peaks:          list[tuple[int, int]],
    hype_segments:  list[TranscribedSegment],
    hop_length:     int,
    sr:             int,
    max_clips:      int   = MAX_CLIPS,
    window_frames:  int   = WINDOW_FRAMES,
    min_duration_s: float = MIN_DURATION_S,
    weight_energy:  float = WEIGHT_ENERGY,
    weight_keyword: float = WEIGHT_KEYWORD,
    min_spread_s:   float = MIN_SPREAD_S,
) -> list[ScoredSegment]:
    """
    Score all peaks, then greedily select top N clips
    that are at least min_spread_s apart from each other.
    """
    max_frame = len(energy) - 1

    # ── score every peak ──────────────────────────────────────
    all_scored = []

    raw_energy_scores = [
        _energy_score(energy, *_expand_peak(s, e, window_frames, max_frame))
        for s, e in peaks if (e - s) >= 5
    ]
    max_e = max(raw_energy_scores) if raw_energy_scores else 1.0

    for start_f, end_f in peaks:
        if (end_f - start_f) < 5:
            continue

        exp_start, exp_end = _expand_peak(start_f, end_f, window_frames, max_frame)
        start_s = _frames_to_sec(exp_start, hop_length, sr)
        end_s   = _frames_to_sec(exp_end,   hop_length, sr)

        if (end_s - start_s) < min_duration_s:
            continue

        e_score = _energy_score(energy, exp_start, exp_end)
        k_score = _keyword_score(start_s, end_s, hype_segments)

        e_norm = e_score / max_e
        k_norm = min(k_score / 5.0, 1.0)

        combined = (weight_energy * e_norm) + (weight_keyword * k_norm)

        all_scored.append(ScoredSegment(
            start_sec      = start_s,
            end_sec        = end_s,
            combined_score = combined,
            energy_score   = e_norm,
            keyword_score  = k_norm,
        ))

    # sort best first
    all_scored.sort(key=lambda x: x.combined_score, reverse=True)

    # ── greedy diversity selection ────────────────────────────
    # pick the best clip, then only add clips that are far enough away
    selected = []

    for seg in all_scored:
        if len(selected) >= max_clips:
            break
        if not _overlaps_any(seg.start_sec, seg.end_sec, selected, min_spread_s):
            selected.append(seg)

    # sort final selection by start time for cleaner output
    selected.sort(key=lambda x: x.start_sec)

    print(f"\n[Scorer] {len(all_scored)} segments scored → {len(selected)} diverse clips selected:")
    for i, seg in enumerate(selected):
        print(
            f"  [{i:02d}] {seg.start_sec:7.2f}s → {seg.end_sec:7.2f}s  "
            f"({seg.duration:.1f}s)  "
            f"combined={seg.combined_score:.3f}  "
            f"energy={seg.energy_score:.3f}  "
            f"keywords={seg.keyword_score:.3f}"
        )

    return selected


def to_timestamps(segments: list[ScoredSegment]) -> list[tuple[float, float]]:
    return [(seg.start_sec, seg.end_sec) for seg in segments]