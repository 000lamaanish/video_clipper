"""
main.py — AI Video Clipper
===========================
Orchestrator only. No logic lives here.
All logic lives in src/.

Pipeline:
    1. Extract audio from video         (moviepy)
    2. Load audio + compute energy      (src/audio_analysis.py)
    3. Detect + merge energy peaks      (src/clip_detector.py)
    4. Transcribe PEAK REGIONS only     (src/transcriber.py)  ← fast now
    5. Score + rank segments            (src/scorer.py)
    6. Cut video into highlight clips   (src/video_cutter.py)
"""

import os
import sys
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# make src/ importable from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from audio_analysis import load_audio, compute_energy, compute_threshold
from clip_detector  import detect_peaks, merge_segments
from transcriber    import load_model, transcribe_segments, get_high_value_segments
from scorer         import rank_segments, to_timestamps
from video_cutter   import cut_video


# ═══════════════════════════════════════════════════════════════
# CONFIG — only edit this block
# ═══════════════════════════════════════════════════════════════
INPUT_VIDEO    = "input/Avatar1.mkv"
AUDIO_PATH     = "output/audio.wav"
OUTPUT_FOLDER  = "output/clips"

HOP_LENGTH     = 512
FRAME_LENGTH   = 2048
SENSITIVITY    = 0.5    # higher = more peaks detected
MAX_CLIPS      = 6
WINDOW_FRAMES  = 220    # ← increased from 150 (was causing 0 clips bug)
                        #   220 * 512 / 22050 = ~5.1s per side = ~10.2s clips
MIN_DURATION_S = 8.0

WEIGHT_ENERGY  = 0.5    # must sum to 1.0
WEIGHT_KEYWORD = 0.5

SAVE_PLOT      = True


# ═══════════════════════════════════════════════════════════════
# STEP 1 — Extract audio
# ═══════════════════════════════════════════════════════════════
print("\n── Step 1: Audio Extraction ──────────────────────────")

os.makedirs("output", exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if not os.path.exists(INPUT_VIDEO):
    print(f"❌ Video not found: {INPUT_VIDEO}")
    sys.exit(1)

if os.path.exists(AUDIO_PATH):
    print("✅ Audio already exists — skipping extraction.")
else:
    print("Extracting audio …")
    video = VideoFileClip(INPUT_VIDEO)
    video.audio.write_audiofile(AUDIO_PATH)
    video.close()
    print("✅ Audio extracted.")


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Load audio + compute energy
# ═══════════════════════════════════════════════════════════════
print("\n── Step 2: Energy Analysis ───────────────────────────")

y, sr      = load_audio(AUDIO_PATH)
energy     = compute_energy(y, sr, FRAME_LENGTH, HOP_LENGTH)
threshold  = compute_threshold(energy, SENSITIVITY)


# ═══════════════════════════════════════════════════════════════
# STEP 3 — Detect + merge energy peaks
# ═══════════════════════════════════════════════════════════════
print("\n── Step 3: Peak Detection ────────────────────────────")

peaks       = detect_peaks(energy, threshold=threshold, min_gap=10)
clean_peaks = merge_segments(peaks)

print(f"[Peaks] Raw peaks: {len(peaks)}  |  After merge: {len(clean_peaks)}")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — Whisper: transcribe PEAK REGIONS only (fast)
# ═══════════════════════════════════════════════════════════════
print("\n── Step 4: Whisper Transcription (peaks only) ────────")

# convert top peaks to seconds for Whisper
# we only send the top 30 peaks to Whisper to keep it fast
def frames_to_sec(frame: int) -> float:
    return frame * HOP_LENGTH / sr

# score peaks by energy to pick the best candidates for transcription
peak_scores = sorted(
    clean_peaks,
    key=lambda p: energy[p[0]:p[1]].mean(),
    reverse=True
)
top_peaks_for_whisper = peak_scores[:30]   # transcribe top 30 peaks max

peak_times = [
    (frames_to_sec(s), frames_to_sec(e))
    for s, e in top_peaks_for_whisper
]

print(f"[Whisper] Sending {len(peak_times)} peak regions for transcription")
print(f"[Whisper] Total audio to transcribe: {sum(e-s for s,e in peak_times):.1f}s "
      f"(vs {y.shape[0]/sr:.1f}s full audio)")

model         = load_model()
all_segments  = transcribe_segments(AUDIO_PATH, peak_times, model)
hype_segments = get_high_value_segments(all_segments)

print(f"[Whisper] Total segments: {len(all_segments)}  |  Hype segments: {len(hype_segments)}")


# ═══════════════════════════════════════════════════════════════
# STEP 5 — Score + rank
# ═══════════════════════════════════════════════════════════════
print("\n── Step 5: Scoring ───────────────────────────────────")

top_segments = rank_segments(
    energy         = energy,
    peaks          = clean_peaks,
    hype_segments  = hype_segments,
    hop_length     = HOP_LENGTH,
    sr             = sr,
    max_clips      = MAX_CLIPS,
    window_frames  = WINDOW_FRAMES,
    min_duration_s = MIN_DURATION_S,
    weight_energy  = WEIGHT_ENERGY,
    weight_keyword = WEIGHT_KEYWORD,
)

final_timestamps = to_timestamps(top_segments)


# ═══════════════════════════════════════════════════════════════
# STEP 6 — Debug plot
# ═══════════════════════════════════════════════════════════════
print("\n── Step 6: Debug Plot ────────────────────────────────")

plt.figure(figsize=(14, 4))
plt.plot(energy, linewidth=0.8, color="steelblue")

for seg in top_segments:
    s_frame = int(seg.start_sec * sr / HOP_LENGTH)
    e_frame = int(seg.end_sec   * sr / HOP_LENGTH)
    plt.axvspan(s_frame, e_frame, alpha=0.3, color="orange", label="selected clip")

for seg in hype_segments:
    s_frame = int(seg.start * sr / HOP_LENGTH)
    e_frame = int(seg.end   * sr / HOP_LENGTH)
    plt.axvspan(s_frame, e_frame, alpha=0.2, color="green", label="whisper hit")

plt.title("Energy — orange: selected clips | green: whisper keyword hits")
plt.xlabel("Frame")
plt.ylabel("Normalised Energy")
plt.tight_layout()

if SAVE_PLOT:
    plot_path = "output/energy_plot.png"
    plt.savefig(plot_path, dpi=120)
    print(f"✅ Plot saved: {plot_path}")

plt.show()


# ═══════════════════════════════════════════════════════════════
# STEP 7 — Cut video
# ═══════════════════════════════════════════════════════════════
print("\n── Step 7: Cutting Video ─────────────────────────────")

created = cut_video(
    input_video   = INPUT_VIDEO,
    segments      = final_timestamps,
    output_folder = OUTPUT_FOLDER,
)

print(f"\n🚀 Done — {len(created)}/{len(final_timestamps)} clip(s) saved to '{OUTPUT_FOLDER}'")