"""
emotion_detector.py
--------------------
Responsible for ONE thing only:
    Take transcribed segments and return an emotion score
    for each one using a pretrained transformer model.

Tuned for Movies / TV shows where viral moments are:
    - Surprise  (plot twists, reveals)
    - Fear      (tension, danger)
    - Sadness   (emotional peaks, deaths)
    - Anger     (confrontations, betrayals)
    - Joy       (victories, reunions)

Model used:
    j-hartmann/emotion-english-distilroberta-base
    - Fast, lightweight, runs on CPU
    - 7 emotion labels: anger, disgust, fear, joy, neutral, sadness, surprise
"""

from __future__ import annotations
import torch
from transformers import pipeline
from dataclasses import dataclass
from transcriber import TranscribedSegment


# ── Emotion weights tuned for Movies / TV viral moments ───────────────────────
# Higher = more likely to be a shareable/viral moment
EMOTION_WEIGHTS = {
    "surprise": 1.00,   # plot twists, reveals — highest viral potential
    "fear":     0.85,   # tension, danger, horror moments
    "sadness":  0.75,   # emotional peaks, deaths, farewells
    "anger":    0.65,   # confrontations, betrayals
    "joy":      0.60,   # victories, reunions, celebrations
    "disgust":  0.30,   # rarely viral on its own
    "neutral":  0.00,   # boring, skip
}


@dataclass
class EmotionResult:
    """Emotion analysis result for a single transcribed segment."""
    segment:       TranscribedSegment
    emotion:       str      # dominant emotion label
    confidence:    float    # model confidence (0–1)
    emotion_score: float    # weighted viral score (0–1)

    def __repr__(self) -> str:
        return (
            f"EmotionResult({self.emotion:8s}  "
            f"conf={self.confidence:.2f}  "
            f"score={self.emotion_score:.2f}  "
            f"| {self.segment.text[:60]})"
        )


# ── Model loader ──────────────────────────────────────────────────────────────

def load_emotion_model() -> pipeline:
    """
    Load the emotion classification pipeline.
    Downloads ~300MB on first run, cached after that.

    Returns:
        HuggingFace pipeline ready for inference.
    """
    print("[Emotion] Loading emotion model …")
    print("[Emotion] (Downloads ~300MB on first run — cached after)")

    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"[Emotion] Running on: {device_name}")

    classifier = pipeline(
        "text-classification",
        model  = "j-hartmann/emotion-english-distilroberta-base",
        device = device,
        top_k  = 1,        # only return the top emotion per segment
    )

    print("[Emotion] Model ready.")
    return classifier


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_emotions(
    segments:   list[TranscribedSegment],
    classifier: pipeline,
    batch_size: int = 16,
) -> list[EmotionResult]:
    """
    Run emotion classification on a list of transcribed segments.

    Args:
        segments:   Output of transcriber.get_high_value_segments() or
                    transcriber.transcribe_segments(). Works on all segments,
                    not just hype ones — catches emotional moments without keywords.
        classifier: Output of load_emotion_model().
        batch_size: How many segments to process at once. 16 is safe for CPU.

    Returns:
        List of EmotionResult, one per segment, sorted by emotion_score descending.
    """
    if not segments:
        print("[Emotion] No segments to analyze.")
        return []

    print(f"[Emotion] Analyzing {len(segments)} segments …")

    # extract text for batch inference
    texts = [seg.text for seg in segments]

    # run in batches to avoid memory issues on CPU
    results = []
    for i in range(0, len(texts), batch_size):
        batch        = texts[i : i + batch_size]
        predictions  = classifier(batch, truncation=True, max_length=512)
        results.extend(predictions)

    emotion_results = []

    for seg, pred in zip(segments, results):
        # pred is a list with one dict: [{"label": "surprise", "score": 0.91}]
        top         = pred[0] if isinstance(pred, list) else pred
        emotion     = top["label"].lower()
        confidence  = float(top["score"])
        weight      = EMOTION_WEIGHTS.get(emotion, 0.0)
        emotion_score = weight * confidence

        emotion_results.append(EmotionResult(
            segment       = seg,
            emotion       = emotion,
            confidence    = confidence,
            emotion_score = emotion_score,
        ))

    emotion_results.sort(key=lambda x: x.emotion_score, reverse=True)

    print(f"[Emotion] Done. Top results:")
    for r in emotion_results[:10]:
        print(f"  {r}")

    return emotion_results


def get_emotion_score_for_window(
    start_s:        float,
    end_s:          float,
    emotion_results: list[EmotionResult],
) -> float:
    """
    Get the max emotion score from any segment overlapping a time window.
    Used by scorer.py to add emotion signal to clip ranking.

    Args:
        start_s:         Window start in seconds.
        end_s:           Window end in seconds.
        emotion_results: Output of analyze_emotions().

    Returns:
        Float in [0, 1] — highest emotion score in that window.
    """
    scores = [
        r.emotion_score
        for r in emotion_results
        if r.segment.start < end_s and r.segment.end > start_s
    ]
    return max(scores, default=0.0)


def get_speech_rate_score(segment: TranscribedSegment) -> float:
    """
    Score a segment by how fast the speaker is talking.
    Fast speech = excitement = higher viral potential.

    Normal speech  ~130 wpm → score ~0.5
    Excited speech ~200 wpm → score ~1.0
    Slow speech     ~80 wpm → score ~0.3

    Args:
        segment: A TranscribedSegment from transcriber.py.

    Returns:
        Float in [0, 1].
    """
    duration = segment.end - segment.start
    if duration <= 0:
        return 0.0
    words = len(segment.text.split())
    wpm   = (words / duration) * 60
    # normalize: 200 wpm = score of 1.0
    return min(wpm / 200.0, 1.0)


def get_speech_rate_for_window(
    start_s:   float,
    end_s:     float,
    segments:  list[TranscribedSegment],
) -> float:
    """
    Get average speech rate score for all segments in a time window.

    Args:
        start_s:  Window start in seconds.
        end_s:    Window end in seconds.
        segments: All transcribed segments (from transcriber.py).

    Returns:
        Float in [0, 1].
    """
    window_segs = [
        s for s in segments
        if s.start < end_s and s.end > start_s
    ]
    if not window_segs:
        return 0.0
    scores = [get_speech_rate_score(s) for s in window_segs]
    return sum(scores) / len(scores)