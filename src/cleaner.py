"""
cleaner.py
----------
Responsible for ONE thing only:
    Take raw scored segments and return clean, valid,
    non-overlapping (start_sec, end_sec) tuples ready
    for video cutting.

Rules applied (in order):
    1. Apply padding (±N seconds around each clip)
    2. Enforce max duration (trim from end)
    3. Enforce min duration
    4. Resolve overlaps by trimming (keep both clips)
    5. Final duration check (relaxed — only drop if truly unusable)
"""

from dataclasses import dataclass
from scorer import ScoredSegment


# ── Config defaults ───────────────────────────────────────────────────────────
MIN_DURATION_S      = 30.0
MAX_DURATION_S      = 60.0
PADDING_S           = 3.0
MIN_GAP_S           = 2.0
POST_TRIM_MIN_S     = 15.0   # relaxed minimum after overlap trimming


@dataclass
class CleanSegment:
    """A validated, padded, overlap-resolved clip ready for cutting."""
    start_sec:      float
    end_sec:        float
    original_score: float

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
    segments:        list[ScoredSegment],
    video_duration:  float,
    min_duration_s:  float = MIN_DURATION_S,
    max_duration_s:  float = MAX_DURATION_S,
    padding_s:       float = PADDING_S,
    min_gap_s:       float = MIN_GAP_S,
    post_trim_min_s: float = POST_TRIM_MIN_S,
) -> list[CleanSegment]:
    """
    Clean, validate, and resolve overlaps for scored segments.

    Since scorer.py now guarantees diversity (no two clips are close),
    overlaps here are rare — this is a safety net, not the main filter.
    """
    if not segments:
        print("[Cleaner] ⚠️  No segments to clean.")
        return []

    print(f"\n[Cleaner] Cleaning {len(segments)} segments …")

    # ── 1. Apply padding + clamp to video bounds ──────────────────────────────
    padded = []
    for seg in segments:
        start = max(0.0, seg.start_sec - padding_s)
        end   = min(video_duration, seg.end_sec + padding_s)

        # if padding pushed over max duration, trim from end
        if (end - start) > max_duration_s:
            end = start + max_duration_s

        padded.append(CleanSegment(
            start_sec      = start,
            end_sec        = end,
            original_score = seg.combined_score,
        ))

    # ── 2. Sort by start time ─────────────────────────────────────────────────
    padded.sort(key=lambda s: s.start_sec)

    # ── 3. Enforce min duration ───────────────────────────────────────────────
    duration_filtered = [s for s in padded if s.duration >= min_duration_s]
    dropped = len(padded) - len(duration_filtered)
    if dropped:
        print(f"[Cleaner] Dropped {dropped} clip(s) shorter than {min_duration_s}s")

    if not duration_filtered:
        print(f"[Cleaner] ⚠️  All segments dropped. Lowering min to {post_trim_min_s}s as fallback.")
        duration_filtered = [s for s in padded if s.duration >= post_trim_min_s]

    if not duration_filtered:
        print("[Cleaner] ⚠️  Still no segments. Check WINDOW_FRAMES in config.")
        return []

    # ── 4. Resolve overlaps by trimming (keep both) ───────────────────────────
    resolved = [duration_filtered[0]]

    for current in duration_filtered[1:]:
        prev    = resolved[-1]
        overlap = prev.end_sec - current.start_sec

        if overlap > 0:
            trim = overlap / 2.0 + min_gap_s / 2.0
            resolved[-1] = CleanSegment(
                start_sec      = prev.start_sec,
                end_sec        = prev.end_sec - trim,
                original_score = prev.original_score,
            )
            current = CleanSegment(
                start_sec      = current.start_sec + trim,
                end_sec        = current.end_sec,
                original_score = current.original_score,
            )

        resolved.append(current)

    # ── 5. Final check — use relaxed minimum after trimming ───────────────────
    # scorer already spread clips apart so trimming should be minimal here
    final = [s for s in resolved if s.duration >= post_trim_min_s]

    dropped_after = len(resolved) - len(final)
    if dropped_after:
        print(f"[Cleaner] Dropped {dropped_after} clip(s) too short after overlap trim "
              f"(post-trim minimum: {post_trim_min_s}s)")

    # ── 6. Report ─────────────────────────────────────────────────────────────
    print(f"[Cleaner] ✅ {len(final)} clean clip(s) ready:")
    for i, seg in enumerate(final):
        print(
            f"  [{i:02d}] {seg.start_sec:7.2f}s → {seg.end_sec:7.2f}s  "
            f"({seg.duration:.1f}s)  score={seg.original_score:.3f}"
        )

    return final


def to_timestamps(segments: list[CleanSegment]) -> list[tuple[float, float]]:
    return [(seg.start_sec, seg.end_sec) for seg in segments]