"""
transcriber.py
--------------
Responsible for ONE thing only:
    Transcribe audio segments using Whisper and return
    hype keyword hits with timestamps.

Key improvement: transcribes only peak regions, not the full audio.
This reduces transcription time from ~7 min to ~30-60 sec.
"""

import os
import whisper
import numpy as np
import librosa
import soundfile as sf
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
    avg_confidence: float   # whisper avg log-prob (-1.0 = ok, lower = gibberish)


def load_model(model_size: str = MODEL_SIZE) -> whisper.Whisper:
    """Load and return the Whisper model."""
    print(f"[Whisper] Loading '{model_size}' model …")
    model = whisper.load_model(model_size)
    print("[Whisper] Model ready.")
    return model


def transcribe_segments(
    audio_path:   str,
    peak_times:   list[tuple[float, float]],  # (start_sec, end_sec) from energy peaks
    model:        whisper.Whisper,
    temp_dir:     str = "output/temp_chunks",
) -> list[TranscribedSegment]:
    """
    Transcribe ONLY the peak regions instead of the full audio.
    This is the key speed improvement — avoids processing silence/background.

    Args:
        audio_path:  Path to full .wav file.
        peak_times:  List of (start_sec, end_sec) for each energy peak.
        model:       Loaded Whisper model.
        temp_dir:    Directory for temporary audio chunks (auto-cleaned).

    Returns:
        List of TranscribedSegment with timestamps relative to original audio.
    """
    os.makedirs(temp_dir, exist_ok=True)

    # load full audio once
    y, sr = librosa.load(audio_path, sr=None)

    all_segments = []

    print(f"[Whisper] Transcribing {len(peak_times)} peak regions …")

    for i, (start_s, end_s) in enumerate(peak_times):
        # slice the audio chunk
        start_sample = int(start_s * sr)
        end_sample   = int(end_s   * sr)
        chunk        = y[start_sample:end_sample]

        if len(chunk) == 0:
            continue

        # write temp chunk
        chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
        sf.write(chunk_path, chunk, sr)

        # transcribe
        result = model.transcribe(
            chunk_path,
            language=LANGUAGE,
            verbose=False,
            fp16=False,
        )

        # collect segments, adjusting timestamps back to original audio time
        for seg in result["segments"]:
            text  = seg["text"].strip()
            start = start_s + seg["start"]   # offset back to original timeline
            end   = start_s + seg["end"]

            all_segments.append(TranscribedSegment(
                text           = text,
                start          = start,
                end            = end,
                keyword_score  = _count_keywords(text),
                avg_confidence = seg.get("avg_logprob", -1.0),
            ))

        # clean up temp chunk
        os.remove(chunk_path)
        print(f"  [{i+1:02d}/{len(peak_times)}] {start_s:.1f}s → {end_s:.1f}s  transcribed")

    # clean up temp dir if empty
    if not os.listdir(temp_dir):
        os.rmdir(temp_dir)

    print(f"[Whisper] Done — {len(all_segments)} segments transcribed.")
    return all_segments


def get_high_value_segments(
    segments:       list[TranscribedSegment],
    min_keywords:   int   = 1,
    min_confidence: float = -1.0,
) -> list[TranscribedSegment]:
    """
    Filter to segments containing hype keywords above confidence threshold.

    Args:
        segments:       Output of transcribe_segments().
        min_keywords:   Minimum hype keyword hits to keep a segment.
        min_confidence: Minimum Whisper confidence (avg log-prob).

    Returns:
        Filtered list sorted by keyword_score descending.
    """
    filtered = [
        s for s in segments
        if s.keyword_score  >= min_keywords
        and s.avg_confidence >= min_confidence
    ]
    filtered.sort(key=lambda s: s.keyword_score, reverse=True)

    print(f"[Whisper] Hype segments found: {len(filtered)}")
    for s in filtered[:10]:   # show top 10
        print(f"  [{s.start:.1f}s → {s.end:.1f}s]  kw={s.keyword_score}  | {s.text}")

    return filtered


# ── Internal helpers ──────────────────────────────────────────────────────────

def _count_keywords(text: str) -> int:
    lower = text.lower()
    return sum(1 for kw in HYPE_KEYWORDS if kw in lower)