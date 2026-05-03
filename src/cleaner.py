"""
cleaner.py
----------
Responsible for ONE thing only:
    Take raw scored segments and return clean, valid,
    non-overlapping (start_sec, end_sec) tuples ready
    for video cutting.

Rules applied (in order):
    1. Enforce min/max duration
    2. Resolve overlaps by trimming (keep both clips)
    3. Apply padding (±N seconds around each clip)
    4. Clamp to video bounds
    5. Final duration check after padding
"""

from dataclasses import dataclass
from scorer import ScoredSegment


# ── Config defaults ───────────────────────────────────────────────────────────
MIN_DURATION_S  = 30.0    # shortest allowed clip (seconds)
MAX_DURATION_S  = 60.0    # longest allowed clip (seconds)
PADDING_S       = 3.0     # seconds added before start and after end
MIN_GAP_S       = 2.0     # minimum gap to enforce between two clips after trimming


@dataclass
class CleanSegment:
    """A validated, padded, overlap-resolved clip ready for cutting."""
    start_sec:      float
    end_sec:        float
    original_score: float   # preserved from ScoredSegment for reference

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    def __repr__(self) -> str:
        return (
            f"CleanSegment({self.start_sec:.2f}s → {self.end_sec:.2f}s  "
            f"duration={self.duration:.1f}s  score={self.original_score:.3f})"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def clean_segments(
    segments:       list[ScoredSegment],
    video_duration: float,                  # total video length in seconds
    min_duration_s: float = MIN_DURATION_S,
    max_duration_s: float = MAX_DURATION_S,
    padding_s:      float = PADDING_S,
    min_gap_s:      float = MIN_GAP_S,
) -> list[CleanSegment]:
    """
    Clean, validate, and resolve overlaps for a list of scored segments.

    Args:
        segments:       Output of scorer.rank_segments().
        video_duration: Full video length in seconds (for clamping).
        min_duration_s: Drop clips shorter than this after padding.
        max_duration_s: Trim clips longer than this (from the end).
        padding_s:      Seconds to add before start and after end.
        min_gap_s:      Minimum gap between two clips after overlap trimming.

    Returns:
        List of CleanSegment sorted by start time, ready for cut_video().
    """
    if not segments:
        print("[Cleaner] ⚠️  No segments to clean.")
        return []

    print(f"\n[Cleaner] Cleaning {len(segments)} segments …")

    # ── 1. Apply padding + enforce max duration ───────────────────────────────
    padded = []
    for seg in segments:
        start = max(0.0, seg.start_sec - padding_s)
        end   = min(video_duration, seg.end_sec + padding_s)

        # if padding made it too long, trim from the end
        if (end - start) > max_duration_s:
            end = start + max_duration_s

        padded.append(CleanSegment(
            start_sec      = start,
            end_sec        = end,
            original_score = seg.combined_score,
        ))

    # ── 2. Sort by start time (required for overlap detection) ────────────────
    padded.sort(key=lambda s: s.start_sec)

    # ── 3. Enforce min duration (after padding) ───────────────────────────────
    duration_filtered = [s for s in padded if s.duration >= min_duration_s]

    dropped = len(padded) - len(duration_filtered)
    if dropped:
        print(f"[Cleaner] Dropped {dropped} clip(s) shorter than {min_duration_s}s")

    if not duration_filtered:
        print(f"[Cleaner] ⚠️  All segments dropped by duration filter.")
        print(f"[Cleaner] Tip: lower MIN_DURATION_S in config (currently {min_duration_s}s)")
        return []

    # ── 4. Resolve overlaps by trimming (keep both clips) ────────────────────
    resolved = [duration_filtered[0]]

    for current in duration_filtered[1:]:
        prev = resolved[-1]

        overlap = prev.end_sec - current.start_sec

        if overlap > 0:
            # trim: split the overlap evenly between the two clips
            trim = overlap / 2.0 + min_gap_s / 2.0

            # shorten previous clip's end
            new_prev_end = prev.end_sec - trim

            # push current clip's start forward
            new_curr_start = current.start_sec + trim

            # update previous in place
            resolved[-1] = CleanSegment(
                start_sec      = prev.start_sec,
                end_sec        = new_prev_end,
                original_score = prev.original_score,
            )

            current = CleanSegment(
                start_sec      = new_curr_start,
                end_sec        = current.end_sec,
                original_score = current.original_score,
            )

        resolved.append(current)

    # ── 5. Final duration check after overlap trimming ────────────────────────
    final = [s for s in resolved if s.duration >= min_duration_s]

    dropped_after_trim = len(resolved) - len(final)
    if dropped_after_trim:
        print(f"[Cleaner] Dropped {dropped_after_trim} clip(s) too short after overlap trim")

    # ── 6. Report ─────────────────────────────────────────────────────────────
    print(f"[Cleaner] ✅ {len(final)} clean clip(s) ready:")
    for i, seg in enumerate(final):
        print(
            f"  [{i:02d}] {seg.start_sec:7.2f}s → {seg.end_sec:7.2f}s  "
            f"({seg.duration:.1f}s)  score={seg.original_score:.3f}"
        )

    return final


def to_timestamps(segments: list[CleanSegment]) -> list[tuple[float, float]]:
    """
    Convert CleanSegment list to plain (start_sec, end_sec) tuples
    ready to pass into cut_video().

    Args:
        segments: Output of clean_segments().

    Returns:
        List of (start_sec, end_sec) tuples.
    """
    return [(seg.start_sec, seg.end_sec) for seg in segments]
