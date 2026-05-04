"""
main.py — AI Video Clipper
===========================
Orchestrator only. No logic lives here.
All logic lives in src/.

Pipeline:
    1. Extract audio from video         (moviepy)
    2. Load audio + compute energy      (src/audio_analysis.py)
    3. Detect + merge energy peaks      (src/clip_detector.py)
    4. Transcribe peak regions only     (src/transcriber.py)
    5. Score + rank segments            (src/scorer.py)
    6. Clean + validate clips           (src/cleaner.py)
    7. Cut video into highlight clips   (src/video_cutter.py)
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
from scorer         import rank_segments
from cleaner        import clean_segments, to_timestamps
from video_cutter   import cut_video


# ═══════════════════════════════════════════════════════════════
# CONFIG — only edit this block
# ═══════════════════════════════════════════════════════════════
INPUT_VIDEO    = "input/Avatar1.mkv"
AUDIO_PATH     = "output/audio.wav"
OUTPUT_FOLDER  = "output/clips"

HOP_LENGTH     = 512
FRAME_LENGTH   = 2048
SENSITIVITY    = 0.5
MAX_CLIPS      = 6

# Clip length control
# Formula: clip_duration ≈ (WINDOW_FRAMES * 2 * HOP_LENGTH) / sr
#   650  frames → ~30s clips
#   968  frames → ~45s clips
#   1300 frames → ~60s clips
WINDOW_FRAMES  = 968

WEIGHT_ENERGY  = 0.5    # must sum to 1.0
WEIGHT_KEYWORD = 0.5

# Diversity — minimum seconds between any two selected clips
# Prevents scorer from picking 6 clips from the same moment
MIN_SPREAD_S   = 120.0  # at least 2 minutes apart

# Clip quality settings
MIN_DURATION_S  = 30.0  # shortest allowed clip
MAX_DURATION_S  = 60.0  # longest allowed clip
PADDING_S       = 3.0   # padding around each clip
MIN_GAP_S       = 2.0   # gap between two clips after overlap trim
POST_TRIM_MIN_S = 15.0  # relaxed minimum after overlap trimming

SAVE_PLOT       = True

# Whisper settings
WHISPER_CONTEXT_S = 45.0   # seconds of audio around each peak sent to Whisper
WHISPER_TOP_PEAKS = 20     # number of peaks to transcribe


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

y, sr          = load_audio(AUDIO_PATH)
energy         = compute_energy(y, sr, FRAME_LENGTH, HOP_LENGTH)
threshold      = compute_threshold(energy, SENSITIVITY)
video_duration = len(y) / sr

print(f"[Audio] Video duration: {video_duration:.1f}s  ({video_duration/60:.1f} min)")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — Detect + merge energy peaks
# ═══════════════════════════════════════════════════════════════
print("\n── Step 3: Peak Detection ────────────────────────────")

peaks       = detect_peaks(energy, threshold=threshold, min_gap=10)
clean_peaks = merge_segments(peaks)

print(f"[Peaks] Raw: {len(peaks)}  |  After merge: {len(clean_peaks)}")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — Whisper: transcribe peak regions with context window
# ═══════════════════════════════════════════════════════════════
print("\n── Step 4: Whisper Transcription (peaks only) ────────")

def frames_to_sec(frame: int) -> float:
    return frame * HOP_LENGTH / sr

# rank peaks by avg energy, take top N
peak_scores = sorted(
    clean_peaks,
    key=lambda p: energy[p[0]:p[1]].mean(),
    reverse=True,
)
top_peaks = peak_scores[:WHISPER_TOP_PEAKS]

# build wider time windows around each peak centre
peak_times = []
for start_f, end_f in top_peaks:
    centre_s  = (frames_to_sec(start_f) + frames_to_sec(end_f)) / 2
    win_start = max(0.0, centre_s - WHISPER_CONTEXT_S / 2)
    win_end   = min(video_duration, centre_s + WHISPER_CONTEXT_S / 2)
    peak_times.append((win_start, win_end))

total_s = sum(e - s for s, e in peak_times)
print(f"[Whisper] {len(peak_times)} regions  |  ~{total_s:.0f}s to transcribe  "
      f"(vs {video_duration:.0f}s full audio)")

model         = load_model()
all_segments  = transcribe_segments(AUDIO_PATH, peak_times, model)
hype_segments = get_high_value_segments(all_segments)

print(f"[Whisper] Segments: {len(all_segments)}  |  Hype: {len(hype_segments)}")


# ═══════════════════════════════════════════════════════════════
# STEP 5 — Score + rank with diversity enforcement
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
    min_duration_s = 8.0,
    weight_energy  = WEIGHT_ENERGY,
    weight_keyword = WEIGHT_KEYWORD,
    min_spread_s   = MIN_SPREAD_S,
)


# ═══════════════════════════════════════════════════════════════
# STEP 6 — Clean + validate clips
# ═══════════════════════════════════════════════════════════════
print("\n── Step 6: Cleaning Clips ────────────────────────────")

clean = clean_segments(
    segments        = top_segments,
    video_duration  = video_duration,
    min_duration_s  = MIN_DURATION_S,
    max_duration_s  = MAX_DURATION_S,
    padding_s       = PADDING_S,
    min_gap_s       = MIN_GAP_S,
    post_trim_min_s = POST_TRIM_MIN_S,
)

final_timestamps = to_timestamps(clean)

if not final_timestamps:
    print("\n❌ No valid clips after cleaning.")
    print(f"   → Try lowering MIN_DURATION_S (currently {MIN_DURATION_S}s)")
    print(f"   → Or increase WINDOW_FRAMES (currently {WINDOW_FRAMES})")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# STEP 7 — Debug plot
# ═══════════════════════════════════════════════════════════════
print("\n── Step 7: Debug Plot ────────────────────────────────")

plt.figure(figsize=(14, 4))
plt.plot(energy, linewidth=0.8, color="steelblue")

for seg in clean:
    s_frame = int(seg.start_sec * sr / HOP_LENGTH)
    e_frame = int(seg.end_sec   * sr / HOP_LENGTH)
    plt.axvspan(s_frame, e_frame, alpha=0.3, color="orange")

for seg in hype_segments:
    s_frame = int(seg.start * sr / HOP_LENGTH)
    e_frame = int(seg.end   * sr / HOP_LENGTH)
    plt.axvspan(s_frame, e_frame, alpha=0.2, color="green")

plt.title("Energy — orange: final clips | green: whisper keyword hits")
plt.xlabel("Frame")
plt.ylabel("Normalised Energy")
plt.tight_layout()

if SAVE_PLOT:
    plot_path = "output/energy_plot.png"
    plt.savefig(plot_path, dpi=120)
    print(f"✅ Plot saved: {plot_path}")

plt.show()


# ═══════════════════════════════════════════════════════════════
# STEP 8 — Cut video
# ═══════════════════════════════════════════════════════════════
print("\n── Step 8: Cutting Video ─────────────────────────────")

created = cut_video(
    input_video   = INPUT_VIDEO,
    segments      = final_timestamps,
    output_folder = OUTPUT_FOLDER,
)

print(f"\n🚀 Done — {len(created)}/{len(final_timestamps)} clip(s) saved to '{OUTPUT_FOLDER}'")