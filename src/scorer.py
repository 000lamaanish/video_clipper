"""
scorer.py
---------
Responsible for ONE thing only:
    Take energy peaks + whisper segments + emotion results
    and return the top N (start_sec, end_sec) timestamps.

Scoring signals (5 total):
    1. Audio energy       — how loud/intense the moment is
    2. Keyword score      — hype words detected by Whisper
    3. Emotion score      — surprise/fear/sadness from transformer model
    4. Speech rate        — fast talking = excitement
    5. Diversity          — no two clips from the same moment
"""

import numpy as np
from dataclasses import dataclass
from transcriber       import TranscribedSegment
from emotion_detector  import EmotionResult, get_emotion_score_for_window, get_speech_rate_for_window


# ── Config defaults ───────────────────────────────────────────────────────────
WINDOW_FRAMES  = 968
MIN_DURATION_S = 8.0
MAX_CLIPS      = 6
MIN_SPREAD_S   = 60.0

# Default weights — tuned for Movies / TV
# emotion is highest because it's the most reliable signal for viral moments
WEIGHT_ENERGY   = 0.20
WEIGHT_KEYWORD  = 0.20
WEIGHT_EMOTION  = 0.35
WEIGHT_SPEECH   = 0.25


@dataclass
class ScoredSegment:
    """A single ranked highlight segment ready for video cutting."""
    start_sec:      float
    end_sec:        float
    combined_score: float
    energy_score:   float
    keyword_score:  float
    emotion_score:  float
    speech_score:   float

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
    for sel in selected:
        if start_s < sel.end_sec + spread_s and end_s > sel.start_sec - spread_s:
            return True
    return False


# ── Public API ────────────────────────────────────────────────────────────────

def rank_segments(
    energy:          np.ndarray,
    peaks:           list[tuple[int, int]],
    hype_segments:   list[TranscribedSegment],
    hop_length:      int,
    sr:              int,
    all_segments:    list[TranscribedSegment] = None,   # for speech rate
    emotion_results: list[EmotionResult]      = None,   # from emotion_detector
    max_clips:       int   = MAX_CLIPS,
    window_frames:   int   = WINDOW_FRAMES,
    min_duration_s:  float = MIN_DURATION_S,
    weight_energy:   float = WEIGHT_ENERGY,
    weight_keyword:  float = WEIGHT_KEYWORD,
    weight_emotion:  float = WEIGHT_EMOTION,
    weight_speech:   float = WEIGHT_SPEECH,
    min_spread_s:    float = MIN_SPREAD_S,
) -> list[ScoredSegment]:
    """
    Score all peaks using up to 4 signals, then greedily select
    top N clips that are at least min_spread_s apart.

    Args:
        energy:          Normalized energy array.
        peaks:           Merged frame-space peaks from clip_detector.
        hype_segments:   Whisper hype segments with keyword hits.
        hop_length:      Hop length used during energy computation.
        sr:              Audio sample rate.
        all_segments:    All transcribed segments (for speech rate scoring).
                         If None, speech rate scoring is skipped.
        emotion_results: Output of emotion_detector.analyze_emotions().
                         If None, emotion scoring is skipped.
        max_clips:       Maximum clips to return.
        window_frames:   Half-width in frames to expand each peak.
        min_duration_s:  Minimum clip duration in seconds.
        weight_*:        Scoring weights. Should sum to 1.0.
        min_spread_s:    Minimum seconds between any two selected clips.

    Returns:
        List of ScoredSegment sorted by start time.
    """
    max_frame    = len(energy) - 1
    all_scored   = []

    # normalize weights so they always sum to 1.0 even if some signals missing
    use_emotion = emotion_results is not None and len(emotion_results) > 0
    use_speech  = all_segments   is not None and len(all_segments)   > 0

    if not use_emotion:
        weight_energy  += weight_emotion / 2
        weight_keyword += weight_emotion / 2
        weight_emotion  = 0.0

    if not use_speech:
        weight_energy  += weight_speech / 2
        weight_keyword += weight_speech / 2
        weight_speech   = 0.0

    # raw energy scores for normalization
    raw_e_scores = [
        _energy_score(energy, *_expand_peak(s, e, window_frames, max_frame))
        for s, e in peaks if (e - s) >= 5
    ]
    max_e = max(raw_e_scores) if raw_e_scores else 1.0

    for start_f, end_f in peaks:
        if (end_f - start_f) < 5:
            continue

        exp_start, exp_end = _expand_peak(start_f, end_f, window_frames, max_frame)
        start_s = _frames_to_sec(exp_start, hop_length, sr)
        end_s   = _frames_to_sec(exp_end,   hop_length, sr)

        if (end_s - start_s) < min_duration_s:
            continue

        # ── compute each signal ───────────────────────────────
        e_score = _energy_score(energy, exp_start, exp_end) / max_e
        k_score = min(_keyword_score(start_s, end_s, hype_segments) / 5.0, 1.0)

        em_score = (
            get_emotion_score_for_window(start_s, end_s, emotion_results)
            if use_emotion else 0.0
        )
        sp_score = (
            get_speech_rate_for_window(start_s, end_s, all_segments)
            if use_speech else 0.0
        )

        combined = (
            weight_energy  * e_score  +
            weight_keyword * k_score  +
            weight_emotion * em_score +
            weight_speech  * sp_score
        )

        all_scored.append(ScoredSegment(
            start_sec      = start_s,
            end_sec        = end_s,
            combined_score = combined,
            energy_score   = e_score,
            keyword_score  = k_score,
            emotion_score  = em_score,
            speech_score   = sp_score,
        ))

    # sort best first
    all_scored.sort(key=lambda x: x.combined_score, reverse=True)

    # greedy diversity selection
    selected = []
    for seg in all_scored:
        if len(selected) >= max_clips:
            break
        if not _overlaps_any(seg.start_sec, seg.end_sec, selected, min_spread_s):
            selected.append(seg)

    # sort by start time for readable output
    selected.sort(key=lambda x: x.start_sec)

    print(f"\n[Scorer] {len(all_scored)} segments scored → {len(selected)} diverse clips:")
    for i, seg in enumerate(selected):
        print(
            f"  [{i:02d}] {seg.start_sec:7.2f}s → {seg.end_sec:7.2f}s  "
            f"({seg.duration:.1f}s)  "
            f"combined={seg.combined_score:.3f}  "
            f"energy={seg.energy_score:.3f}  "
            f"keywords={seg.keyword_score:.3f}  "
            f"emotion={seg.emotion_score:.3f}  "
            f"speech={seg.speech_score:.3f}"
        )

    return selected


def to_timestamps(segments: list[ScoredSegment]) -> list[tuple[float, float]]:
    return [(seg.start_sec, seg.end_sec) for seg in segments]