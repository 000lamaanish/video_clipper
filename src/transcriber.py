"""
transcriber.py
--------------
Responsible for ONE thing only:
    Transcribe audio segments using Whisper and return
    hype keyword hits with timestamps.

Cloud fix: audio is loaded via librosa and passed to Whisper
as a numpy array instead of a file path — avoids Whisper's
internal FFmpeg dependency which breaks on Streamlit Cloud.
"""

import os
import whisper
import numpy as np
import librosa
from dataclasses import dataclass


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_SIZE = "base"
LANGUAGE   = "en"

HYPE_KEYWORDS = {
    # excitement
    "wow", "insane", "crazy", "unbelievable", "no way",
    "let's go", "let's gooo", "let's goooo",
    # gaming / stream specific
    "clip it", "clip", "clutch", "ace", "destroyed", "obliterated",
    "nasty", "cracked", "goated", "bussin",
    # reactions
    "oh my god", "omg", "bro", "dude", "are you kidding",
    "i can't believe", "that's actually", "holy",
}


@dataclass
class TranscribedSegment:
    """A single transcribed sentence with timing and scoring info."""
    text:           str
    start:          float   # seconds (relative to original audio)
    end:            float   # seconds (relative to original audio)
    keyword_score:  int     # number of hype keywords found
    avg_confidence: float   # whisper avg log-prob


def load_model(model_size: str = MODEL_SIZE) -> whisper.Whisper:
    """Load and return the Whisper model."""
    print(f"[Whisper] Loading '{model_size}' model …")
    model = whisper.load_model(model_size)
    print("[Whisper] Model ready.")
    return model


def transcribe_segments(
    audio_path:  str,
    peak_times:  list[tuple[float, float]],
    model:       whisper.Whisper,
    temp_dir:    str = None,   # kept for API compatibility, no longer used
) -> list[TranscribedSegment]:
    """
    Transcribe ONLY the peak regions instead of the full audio.

    Cloud-safe: loads audio via librosa and passes numpy arrays
    directly to Whisper — no FFmpeg dependency, no temp files.

    Args:
        audio_path:  Path to full .wav file.
        peak_times:  List of (start_sec, end_sec) for each energy peak.
        model:       Loaded Whisper model.
        temp_dir:    Ignored — kept for backward compatibility.

    Returns:
        List of TranscribedSegment with timestamps relative to original audio.
    """
    # load full audio once using librosa (no FFmpeg needed)
    print(f"[Whisper] Loading audio via librosa: {audio_path}")
    y, sr = librosa.load(audio_path, sr=16000)   # Whisper expects 16kHz
    print(f"[Whisper] Audio loaded — {len(y)/sr:.1f}s at {sr}Hz")

    all_segments = []
    print(f"[Whisper] Transcribing {len(peak_times)} peak regions …")

    for i, (start_s, end_s) in enumerate(peak_times):
        # slice audio chunk directly from numpy array — no temp file needed
        start_sample = int(start_s * sr)
        end_sample   = int(end_s   * sr)
        chunk        = y[start_sample:end_sample]

        if len(chunk) == 0:
            continue

        # convert to float32 as Whisper expects
        chunk = chunk.astype(np.float32)

        # pass numpy array directly — bypasses Whisper's internal FFmpeg call
        result = model.transcribe(
            chunk,
            language        = LANGUAGE,
            verbose         = False,
            fp16            = False,
            task            = "transcribe",
        )

        # collect segments, adjusting timestamps back to original audio time
        for seg in result["segments"]:
            text  = seg["text"].strip()
            start = start_s + seg["start"]
            end   = start_s + seg["end"]

            all_segments.append(TranscribedSegment(
                text           = text,
                start          = start,
                end            = end,
                keyword_score  = _count_keywords(text),
                avg_confidence = seg.get("avg_logprob", -1.0),
            ))

        print(f"  [{i+1:02d}/{len(peak_times)}] {start_s:.1f}s → {end_s:.1f}s  transcribed")

    print(f"[Whisper] Done — {len(all_segments)} segments transcribed.")
    return all_segments


def get_high_value_segments(
    segments:       list[TranscribedSegment],
    min_keywords:   int   = 1,
    min_confidence: float = -1.0,
) -> list[TranscribedSegment]:
    """
    Filter to segments containing hype keywords above confidence threshold.
    """
    filtered = [
        s for s in segments
        if s.keyword_score  >= min_keywords
        and s.avg_confidence >= min_confidence
    ]
    filtered.sort(key=lambda s: s.keyword_score, reverse=True)

    print(f"[Whisper] Hype segments found: {len(filtered)}")
    for s in filtered[:10]:
        print(f"  [{s.start:.1f}s → {s.end:.1f}s]  kw={s.keyword_score}  | {s.text}")

    return filtered


# ── Internal helpers ──────────────────────────────────────────────────────────

def _count_keywords(text: str) -> int:
    lower = text.lower()
    return sum(1 for kw in HYPE_KEYWORDS if kw in lower)