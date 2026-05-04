"""
main.py — AI Video Clipper CLI
================================
Usage examples:

    # Basic usage (all defaults)
    python main.py --input input/Avatar1.mkv

    # Custom output folder and clip count
    python main.py --input input/Avatar1.mkv --output output/clips --clips 5

    # Full control
    python main.py \
        --input input/Avatar1.mkv \
        --output output/clips \
        --clips 6 \
        --min-duration 30 \
        --max-duration 60 \
        --sensitivity 0.5 \
        --spread 120 \
        --weight-energy 0.5 \
        --weight-keyword 0.5 \
        --whisper-model base \
        --whisper-peaks 20 \
        --whisper-context 45

    # Skip Whisper (energy-only, much faster)
    python main.py --input input/Avatar1.mkv --no-transcribe

    # Skip plot
    python main.py --input input/Avatar1.mkv --no-plot
"""

import os
import sys
import argparse
import time
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
# CLI ARGUMENT DEFINITIONS
# ═══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-clipper",
        description="🎬 AI Video Clipper — automatically generate highlight clips from long videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py --input input/stream.mkv
  python main.py --input input/stream.mkv --clips 5 --min-duration 20 --max-duration 45
  python main.py --input input/stream.mkv --no-transcribe --no-plot
        """,
    )

    # ── Required ──────────────────────────────────────────────
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="PATH",
        help="Path to input video file (e.g. input/stream.mkv)",
    )

    # ── Output paths ──────────────────────────────────────────
    parser.add_argument(
        "--output", "-o",
        default="output/clips",
        metavar="FOLDER",
        help="Folder to save generated clips (default: output/clips)",
    )
    parser.add_argument(
        "--audio",
        default="output/audio.wav",
        metavar="PATH",
        help="Path for extracted audio file (default: output/audio.wav)",
    )

    # ── Clip settings ─────────────────────────────────────────
    parser.add_argument(
        "--clips", "-n",
        type=int,
        default=6,
        metavar="N",
        help="Number of highlight clips to generate (default: 6)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Minimum clip duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Maximum clip duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=3.0,
        metavar="SECONDS",
        help="Seconds of padding added before/after each clip (default: 3)",
    )

    # ── Detection settings ────────────────────────────────────
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Peak detection sensitivity 0.0–2.0. Higher = more peaks (default: 0.5)",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=120.0,
        metavar="SECONDS",
        help="Minimum seconds between two selected clips (default: 120)",
    )

    # ── Scoring weights ───────────────────────────────────────
    parser.add_argument(
        "--weight-energy",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Weight for audio energy score, 0.0–1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--weight-keyword",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Weight for Whisper keyword score, 0.0–1.0 (default: 0.5)",
    )

    # ── Whisper settings ──────────────────────────────────────
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium"],
        metavar="MODEL",
        help="Whisper model size: tiny/base/small/medium (default: base)",
    )
    parser.add_argument(
        "--whisper-peaks",
        type=int,
        default=20,
        metavar="N",
        help="Number of peak regions to send to Whisper (default: 20)",
    )
    parser.add_argument(
        "--whisper-context",
        type=float,
        default=45.0,
        metavar="SECONDS",
        help="Seconds of audio context around each peak for Whisper (default: 45)",
    )

    # ── Flags ─────────────────────────────────────────────────
    parser.add_argument(
        "--no-transcribe",
        action="store_true",
        help="Skip Whisper transcription — energy-only mode, much faster",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the debug energy plot",
    )
    parser.add_argument(
        "--force-audio",
        action="store_true",
        help="Re-extract audio even if audio.wav already exists",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument values and exit with helpful errors."""
    errors = []

    if not os.path.exists(args.input):
        errors.append(f"Input video not found: {args.input}")

    if args.clips < 1:
        errors.append("--clips must be at least 1")

    if args.min_duration >= args.max_duration:
        errors.append("--min-duration must be less than --max-duration")

    if not (0.0 <= args.sensitivity <= 2.0):
        errors.append("--sensitivity should be between 0.0 and 2.0")

    w_sum = args.weight_energy + args.weight_keyword
    if abs(w_sum - 1.0) > 0.01:
        errors.append(
            f"--weight-energy + --weight-keyword must sum to 1.0 (got {w_sum:.2f})"
        )

    if errors:
        print("\n❌ Invalid arguments:")
        for e in errors:
            print(f"   • {e}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> None:
    total_start  = time.time()
    HOP_LENGTH   = 512
    FRAME_LENGTH = 2048

    # derive WINDOW_FRAMES from target clip midpoint
    # (recalculated with real sr after audio load)
    target_duration = (args.min_duration + args.max_duration) / 2
    WINDOW_FRAMES   = int((target_duration / 2) * 22050 / HOP_LENGTH)  # temp estimate

    print(f"\n{'═'*55}")
    print(f"  🎬 AI Video Clipper")
    print(f"{'═'*55}")
    print(f"  Input       : {args.input}")
    print(f"  Output      : {args.output}")
    print(f"  Clips       : {args.clips}")
    print(f"  Duration    : {args.min_duration}s – {args.max_duration}s")
    print(f"  Sensitivity : {args.sensitivity}")
    print(f"  Spread      : {args.spread}s minimum between clips")
    print(f"  Transcribe  : {'No (energy only)' if args.no_transcribe else f'Yes (whisper-{args.whisper_model})'}")
    print(f"{'═'*55}\n")

    # ── Step 1: Extract audio ──────────────────────────────────
    print("── Step 1: Audio Extraction ──────────────────────────")
    os.makedirs(os.path.dirname(args.audio) or "output", exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    if os.path.exists(args.audio) and not args.force_audio:
        print("✅ Audio already exists — skipping. (--force-audio to re-extract)")
    else:
        print("Extracting audio …")
        video = VideoFileClip(args.input)
        video.audio.write_audiofile(args.audio)
        video.close()
        print("✅ Audio extracted.")

    # ── Step 2: Energy analysis ────────────────────────────────
    print("\n── Step 2: Energy Analysis ───────────────────────────")
    y, sr          = load_audio(args.audio)
    energy         = compute_energy(y, sr, FRAME_LENGTH, HOP_LENGTH)
    threshold      = compute_threshold(energy, args.sensitivity)
    video_duration = len(y) / sr

    # recalculate with real sr now
    WINDOW_FRAMES = int((target_duration / 2) * sr / HOP_LENGTH)
    print(f"[Config] Target clip ~{target_duration:.0f}s → WINDOW_FRAMES={WINDOW_FRAMES}")

    # ── Step 3: Peak detection ─────────────────────────────────
    print("\n── Step 3: Peak Detection ────────────────────────────")
    peaks       = detect_peaks(energy, threshold=threshold, min_gap=10)
    clean_peaks = merge_segments(peaks)
    print(f"[Peaks] Raw: {len(peaks)}  |  After merge: {len(clean_peaks)}")

    # ── Step 4: Whisper transcription ─────────────────────────
    hype_segments = []

    if not args.no_transcribe:
        print("\n── Step 4: Whisper Transcription (peaks only) ────────")

        def frames_to_sec(frame: int) -> float:
            return frame * HOP_LENGTH / sr

        peak_scores = sorted(
            clean_peaks,
            key=lambda p: energy[p[0]:p[1]].mean(),
            reverse=True,
        )
        top_peaks = peak_scores[:args.whisper_peaks]

        peak_times = []
        for start_f, end_f in top_peaks:
            centre_s  = (frames_to_sec(start_f) + frames_to_sec(end_f)) / 2
            win_start = max(0.0, centre_s - args.whisper_context / 2)
            win_end   = min(video_duration, centre_s + args.whisper_context / 2)
            peak_times.append((win_start, win_end))

        total_s = sum(e - s for s, e in peak_times)
        print(f"[Whisper] {len(peak_times)} regions  |  ~{total_s:.0f}s to transcribe")

        model         = load_model(args.whisper_model)
        all_segments  = transcribe_segments(args.audio, peak_times, model)
        hype_segments = get_high_value_segments(all_segments)

        print(f"[Whisper] Segments: {len(all_segments)}  |  Hype: {len(hype_segments)}")
    else:
        print("\n── Step 4: Whisper Transcription ─────────────────────")
        print("[Whisper] Skipped (--no-transcribe)")

    # ── Step 5: Scoring ────────────────────────────────────────
    print("\n── Step 5: Scoring ───────────────────────────────────")
    top_segments = rank_segments(
        energy         = energy,
        peaks          = clean_peaks,
        hype_segments  = hype_segments,
        hop_length     = HOP_LENGTH,
        sr             = sr,
        max_clips      = args.clips,
        window_frames  = WINDOW_FRAMES,
        min_duration_s = 8.0,
        weight_energy  = args.weight_energy,
        weight_keyword = args.weight_keyword,
        min_spread_s   = args.spread,
    )

    # ── Step 6: Clean clips ────────────────────────────────────
    print("\n── Step 6: Cleaning Clips ────────────────────────────")
    clean = clean_segments(
        segments        = top_segments,
        video_duration  = video_duration,
        min_duration_s  = args.min_duration,
        max_duration_s  = args.max_duration,
        padding_s       = args.padding,
        min_gap_s       = 2.0,
        post_trim_min_s = 15.0,
    )

    final_timestamps = to_timestamps(clean)

    if not final_timestamps:
        print("\n❌ No valid clips after cleaning.")
        print("   → Try: --min-duration 15")
        print("   → Or:  --sensitivity 0.8")
        sys.exit(1)

    # ── Step 7: Debug plot ─────────────────────────────────────
    if not args.no_plot:
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

        plot_path = os.path.join(
            os.path.dirname(args.output) or "output", "energy_plot.png"
        )
        plt.savefig(plot_path, dpi=120)
        print(f"✅ Plot saved: {plot_path}")
        plt.show()

    # ── Step 8: Cut video ──────────────────────────────────────
    print("\n── Step 8: Cutting Video ─────────────────────────────")
    created = cut_video(
        input_video   = args.input,
        segments      = final_timestamps,
        output_folder = args.output,
    )

    # ── Summary ────────────────────────────────────────────────
    elapsed    = time.time() - total_start
    mins, secs = divmod(int(elapsed), 60)

    print(f"\n{'═'*55}")
    print(f"  🚀 Done in {mins}m {secs}s")
    print(f"  {len(created)}/{len(final_timestamps)} clip(s) → '{args.output}'")
    for i, path in enumerate(created):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"    [{i:02d}] {os.path.basename(path)}  ({size_mb:.1f} MB)")
    print(f"{'═'*55}\n")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    validate_args(args)
    run(args)