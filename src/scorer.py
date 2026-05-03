"""
scorer.py
---------
Responsible for ONE thing only:
    Take energy peaks + whisper segments and return
    the top N (start_sec, end_sec) timestamps.

No audio loading, no transcription, no video cutting.
Just scoring and ranking.
"""

import numpy as np
from dataclasses import dataclass
from transcriber import TranscribedSegment


# ── Config defaults ───────────────────────────────────────────────────────────
WINDOW_FRAMES  = 150    # half-width around each peak centre (~5–10s at hop=512)
MIN_DURATION_S = 8.0    # drop clips shorter than this (seconds)
MAX_CLIPS      = 6      # maximum number of clips to return
WEIGHT_ENERGY  = 0.5    # weight for audio energy score  (must sum to 1.0)
WEIGHT_KEYWORD = 0.5    # weight for keyword score


@dataclass
class ScoredSegment:
    """A single ranked highlight segment ready for video cutting."""
    start_sec:     float
    end_sec:       float
    combined_score: float
    energy_score:  float
    keyword_score: float

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


# ── Internal helpers ──────────────────────────────────────────────────────────

def _frames_to_sec(frame: int, hop_length: int, sr: int) -> float:
    return frame * hop_length / sr


def _energy_score(
    energy:      np.ndarray,
    start_frame: int,
    end_frame:   int,
) -> float:
    """
    Score a frame window by average energy weighted by duration.
    Longer high-energy windows score higher.
    """
    duration = end_frame - start_frame
    if duration <= 0:
        return 0.0
    avg_energy = float(energy[start_frame:end_frame].mean())
    return avg_energy * (duration ** 1.3)


def _keyword_score(
    start_sec:      float,
    end_sec:        float,
    hype_segments:  list[TranscribedSegment],
) -> float:
    """
    Count total hype keyword hits from Whisper segments
    that overlap with the given time window.
    """
    total = 0
    for seg in hype_segments:
        overlaps = seg.start < end_sec and seg.end > start_sec
        if overlaps:
            total += seg.keyword_score
    return float(total)


def _expand_peak(
    start_frame: int,
    end_frame:   int,
    half_win:    int,
    max_frame:   int,
) -> tuple[int, int]:
    """
    Expand a peak outward from its centre by half_win frames.
    Clamps to [0, max_frame].
    """
    centre = (start_frame + end_frame) // 2
    return max(0, centre - half_win), min(max_frame, centre + half_win)


# ── Public API ────────────────────────────────────────────────────────────────

def rank_segments(
    energy:         np.ndarray,
    peaks:          list[tuple[int, int]],    # from clip_detector.merge_segments()
    hype_segments:  list[TranscribedSegment], # from transcriber.get_high_value_segments()
    hop_length:     int,
    sr:             int,
    max_clips:      int   = MAX_CLIPS,
    window_frames:  int   = WINDOW_FRAMES,
    min_duration_s: float = MIN_DURATION_S,
    weight_energy:  float = WEIGHT_ENERGY,
    weight_keyword: float = WEIGHT_KEYWORD,
) -> list[ScoredSegment]:
    """
    Score, rank, and return the top highlight segments.

    Args:
        energy:         Normalized energy array from audio_analysis.compute_energy().
        peaks:          Merged frame-space peaks from clip_detector.merge_segments().
        hype_segments:  Whisper segments with keyword hits from transcriber.
        hop_length:     Hop length used during energy computation.
        sr:             Audio sample rate.
        max_clips:      Maximum number of clips to return.
        window_frames:  Half-width (in frames) to expand each peak.
        min_duration_s: Minimum clip duration in seconds.
        weight_energy:  Weight for energy score (0.0–1.0).
        weight_keyword: Weight for keyword score (0.0–1.0).

    Returns:
        List of ScoredSegment sorted by combined_score descending.
    """
    max_frame = len(energy) - 1
    scored    = []

    # max raw energy score across all peaks — used for normalization
    raw_energy_scores = [
        _energy_score(energy, *_expand_peak(s, e, window_frames, max_frame))
        for s, e in peaks if (e - s) >= 5
    ]
    max_energy_score = max(raw_energy_scores) if raw_energy_scores else 1.0

    for start_f, end_f in peaks:
        # drop trivially short detections
        if (end_f - start_f) < 5:
            continue

        # expand around centre
        exp_start, exp_end = _expand_peak(start_f, end_f, window_frames, max_frame)

        # convert to seconds for keyword lookup + duration check
        start_s = _frames_to_sec(exp_start, hop_length, sr)
        end_s   = _frames_to_sec(exp_end,   hop_length, sr)

        if (end_s - start_s) < min_duration_s:
            continue

        # compute individual scores
        e_score = _energy_score(energy, exp_start, exp_end)
        k_score = _keyword_score(start_s, end_s, hype_segments)

        # normalize both to [0, 1]
        e_score_norm = e_score / max_energy_score
        k_score_norm = min(k_score / 5.0, 1.0)   # cap at 5 keyword hits

        combined = (weight_energy * e_score_norm) + (weight_keyword * k_score_norm)

        scored.append(ScoredSegment(
            start_sec      = start_s,
            end_sec        = end_s,
            combined_score = combined,
            energy_score   = e_score_norm,
            keyword_score  = k_score_norm,
        ))

    # sort by combined score, take top N
    scored.sort(key=lambda x: x.combined_score, reverse=True)
    top = scored[:max_clips]

    # fallback: if strict filter removed everything, relax it
    if not top:
        print("⚠️  No segments passed scoring — check threshold or audio quality.")

    print(f"\n[Scorer] {len(scored)} segments scored, returning top {len(top)}:")
    for i, seg in enumerate(top):
        print(
            f"  [{i:02d}] {seg.start_sec:7.2f}s → {seg.end_sec:7.2f}s  "
            f"({seg.duration:.1f}s)  "
            f"combined={seg.combined_score:.3f}  "
            f"energy={seg.energy_score:.3f}  "
            f"keywords={seg.keyword_score:.3f}"
        )

    return top


def to_timestamps(segments: list[ScoredSegment]) -> list[tuple[float, float]]:
    """
    Convert ScoredSegment list to plain (start_sec, end_sec) tuples
    ready to pass into cut_video().

    Args:
        segments: Output of rank_segments().

    Returns:
        List of (start_sec, end_sec) tuples.
    """
    return [(seg.start_sec, seg.end_sec) for seg in segments]