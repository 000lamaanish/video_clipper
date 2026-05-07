"""
main.py — AI Video Clipper CLI
================================
Usage examples:

    # Basic
    python main.py --input input/Avatar1.mkv

    # Skip emotion detection (faster)
    python main.py --input input/Avatar1.mkv --no-emotion

    # Skip both Whisper and emotion (fastest, energy only)
    python main.py --input input/Avatar1.mkv --no-transcribe --no-plot

    # Full control
    python main.py \
        --input input/Avatar1.mkv \
        --clips 6 \
        --min-duration 30 \
        --max-duration 60 \
        --weight-energy 0.20 \
        --weight-keyword 0.20 \
        --weight-emotion 0.35 \
        --weight-speech 0.25
"""

import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from audio_analysis    import load_audio, compute_energy, compute_threshold
from clip_detector     import detect_peaks, merge_segments
from transcriber       import load_model, transcribe_segments, get_high_value_segments
from emotion_detector  import load_emotion_model, analyze_emotions
from scorer            import rank_segments
from cleaner           import clean_segments, to_timestamps
from video_cutter      import cut_video


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
  python main.py --input input/stream.mkv --no-emotion --clips 5
  python main.py --input input/stream.mkv --no-transcribe --no-plot
        """,
    )

    # ── Required ──────────────────────────────────────────────
    parser.add_argument("--input", "-i", required=True, metavar="PATH",
                        help="Path to input video file")

    # ── Output ────────────────────────────────────────────────
    parser.add_argument("--output", "-o", default="output/clips", metavar="FOLDER",
                        help="Folder to save clips (default: output/clips)")
    parser.add_argument("--audio", default="output/audio.wav", metavar="PATH",
                        help="Path for extracted audio (default: output/audio.wav)")

    # ── Clip settings ─────────────────────────────────────────
    parser.add_argument("--clips", "-n", type=int, default=6, metavar="N",
                        help="Number of clips to generate (default: 6)")
    parser.add_argument("--min-duration", type=float, default=30.0, metavar="SECONDS",
                        help="Minimum clip duration in seconds (default: 30)")
    parser.add_argument("--max-duration", type=float, default=60.0, metavar="SECONDS",
                        help="Maximum clip duration in seconds (default: 60)")
    parser.add_argument("--padding", type=float, default=3.0, metavar="SECONDS",
                        help="Padding around each clip in seconds (default: 3)")

    # ── Detection ─────────────────────────────────────────────
    parser.add_argument("--sensitivity", type=float, default=0.5, metavar="FLOAT",
                        help="Peak detection sensitivity 0.0–2.0 (default: 0.5)")
    parser.add_argument("--spread", type=float, default=120.0, metavar="SECONDS",
                        help="Minimum seconds between clips (default: 120)")

    # ── Scoring weights ───────────────────────────────────────
    parser.add_argument("--weight-energy",  type=float, default=0.20, metavar="FLOAT",
                        help="Weight for audio energy (default: 0.20)")
    parser.add_argument("--weight-keyword", type=float, default=0.20, metavar="FLOAT",
                        help="Weight for keyword score (default: 0.20)")
    parser.add_argument("--weight-emotion", type=float, default=0.35, metavar="FLOAT",
                        help="Weight for emotion score (default: 0.35)")
    parser.add_argument("--weight-speech",  type=float, default=0.25, metavar="FLOAT",
                        help="Weight for speech rate (default: 0.25)")

    # ── Whisper ───────────────────────────────────────────────
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium"], metavar="MODEL",
                        help="Whisper model size (default: base)")
    parser.add_argument("--whisper-peaks", type=int, default=20, metavar="N",
                        help="Peak regions to transcribe (default: 20)")
    parser.add_argument("--whisper-context", type=float, default=45.0, metavar="SECONDS",
                        help="Audio context around each peak for Whisper (default: 45)")

    # ── Flags ─────────────────────────────────────────────────
    parser.add_argument("--no-transcribe", action="store_true",
                        help="Skip Whisper — energy only mode")
    parser.add_argument("--no-emotion", action="store_true",
                        help="Skip emotion detection (faster)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the debug energy plot")
    parser.add_argument("--force-audio", action="store_true",
                        help="Re-extract audio even if it already exists")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    errors = []
    if not os.path.exists(args.input):
        errors.append(f"Input video not found: {args.input}")
    if args.clips < 1:
        errors.append("--clips must be at least 1")
    if args.min_duration >= args.max_duration:
        errors.append("--min-duration must be less than --max-duration")
    w_sum = args.weight_energy + args.weight_keyword + args.weight_emotion + args.weight_speech
    if abs(w_sum - 1.0) > 0.01:
        errors.append(f"Weights must sum to 1.0 (got {w_sum:.2f})")
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
    target_duration = (args.min_duration + args.max_duration) / 2

    print(f"\n{'═'*58}")
    print(f"  🎬 AI Video Clipper")
    print(f"{'═'*58}")
    print(f"  Input       : {args.input}")
    print(f"  Output      : {args.output}")
    print(f"  Clips       : {args.clips}")
    print(f"  Duration    : {args.min_duration}s – {args.max_duration}s")
    print(f"  Transcribe  : {'No' if args.no_transcribe else f'Yes (whisper-{args.whisper_model})'}")
    print(f"  Emotion     : {'No' if args.no_emotion or args.no_transcribe else 'Yes'}")
    print(f"  Weights     : energy={args.weight_energy}  keyword={args.weight_keyword}  "
          f"emotion={args.weight_emotion}  speech={args.weight_speech}")
    print(f"{'═'*58}\n")

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
    WINDOW_FRAMES  = int((target_duration / 2) * sr / HOP_LENGTH)
    print(f"[Config] Target clip ~{target_duration:.0f}s → WINDOW_FRAMES={WINDOW_FRAMES}")

    # ── Step 3: Peak detection ─────────────────────────────────
    print("\n── Step 3: Peak Detection ────────────────────────────")
    peaks       = detect_peaks(energy, threshold=threshold, min_gap=10)
    clean_peaks = merge_segments(peaks)
    print(f"[Peaks] Raw: {len(peaks)}  |  After merge: {len(clean_peaks)}")

    # ── Step 4: Whisper transcription ─────────────────────────
    all_segments  = []
    hype_segments = []

    if not args.no_transcribe:
        print("\n── Step 4: Whisper Transcription ─────────────────────")

        def frames_to_sec(frame): return frame * HOP_LENGTH / sr

        peak_scores = sorted(clean_peaks, key=lambda p: energy[p[0]:p[1]].mean(), reverse=True)
        top_peaks   = peak_scores[:args.whisper_peaks]
        peak_times  = []
        for sf, ef in top_peaks:
            c = (frames_to_sec(sf) + frames_to_sec(ef)) / 2
            peak_times.append((max(0.0, c - args.whisper_context/2),
                               min(video_duration, c + args.whisper_context/2)))

        total_s = sum(e - s for s, e in peak_times)
        print(f"[Whisper] {len(peak_times)} regions  |  ~{total_s:.0f}s to transcribe")

        whisper_model = load_model(args.whisper_model)
        all_segments  = transcribe_segments(args.audio, peak_times, whisper_model)
        hype_segments = get_high_value_segments(all_segments)
        print(f"[Whisper] Segments: {len(all_segments)}  |  Hype: {len(hype_segments)}")
    else:
        print("\n── Step 4: Whisper Transcription ─────────────────────")
        print("[Whisper] Skipped (--no-transcribe)")

    # ── Step 5: Emotion detection ──────────────────────────────
    emotion_results = []

    if not args.no_transcribe and not args.no_emotion and all_segments:
        print("\n── Step 5: Emotion Detection ─────────────────────────")
        emotion_model   = load_emotion_model()
        emotion_results = analyze_emotions(all_segments, emotion_model)
        print(f"[Emotion] {len(emotion_results)} segments analyzed.")
    else:
        print("\n── Step 5: Emotion Detection ─────────────────────────")
        reason = "--no-transcribe" if args.no_transcribe else "--no-emotion" if args.no_emotion else "no segments"
        print(f"[Emotion] Skipped ({reason})")

    # ── Step 6: Scoring ────────────────────────────────────────
    print("\n── Step 6: Scoring ───────────────────────────────────")
    top_segments = rank_segments(
        energy          = energy,
        peaks           = clean_peaks,
        hype_segments   = hype_segments,
        hop_length      = HOP_LENGTH,
        sr              = sr,
        all_segments    = all_segments    or None,
        emotion_results = emotion_results or None,
        max_clips       = args.clips,
        window_frames   = WINDOW_FRAMES,
        min_duration_s  = 8.0,
        weight_energy   = args.weight_energy,
        weight_keyword  = args.weight_keyword,
        weight_emotion  = args.weight_emotion,
        weight_speech   = args.weight_speech,
        min_spread_s    = args.spread,
    )

    # ── Step 7: Clean clips ────────────────────────────────────
    print("\n── Step 7: Cleaning Clips ────────────────────────────")
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
        print("\n❌ No valid clips. Try: --min-duration 15  or  --sensitivity 0.8")
        sys.exit(1)

    # ── Step 8: Debug plot ─────────────────────────────────────
    if not args.no_plot:
        print("\n── Step 8: Debug Plot ────────────────────────────────")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(energy, linewidth=0.8, color="steelblue")

        for seg in clean:
            ax.axvspan(int(seg.start_sec * sr / HOP_LENGTH),
                       int(seg.end_sec   * sr / HOP_LENGTH),
                       alpha=0.3, color="orange")

        for seg in hype_segments:
            ax.axvspan(int(seg.start * sr / HOP_LENGTH),
                       int(seg.end   * sr / HOP_LENGTH),
                       alpha=0.2, color="green")

        # mark high-emotion moments in red
        for r in emotion_results:
            if r.emotion_score > 0.5:
                ax.axvspan(int(r.segment.start * sr / HOP_LENGTH),
                           int(r.segment.end   * sr / HOP_LENGTH),
                           alpha=0.2, color="red")

        ax.set_title("Energy — orange: clips | green: keywords | red: high emotion")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Normalised Energy")
        fig.tight_layout()
        plot_path = os.path.join(os.path.dirname(args.output) or "output", "energy_plot.png")
        fig.savefig(plot_path, dpi=120)
        plt.show()
        print(f"✅ Plot saved: {plot_path}")

    # ── Step 9: Cut video ──────────────────────────────────────
    print("\n── Step 9: Cutting Video ─────────────────────────────")
    created = cut_video(
        input_video   = args.input,
        segments      = final_timestamps,
        output_folder = args.output,
    )

    # ── Summary ────────────────────────────────────────────────
    elapsed    = time.time() - total_start
    mins, secs = divmod(int(elapsed), 60)

    print(f"\n{'═'*58}")
    print(f"  🚀 Done in {mins}m {secs}s")
    print(f"  {len(created)}/{len(final_timestamps)} clip(s) → '{args.output}'")
    for i, path in enumerate(created):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"    [{i:02d}] {os.path.basename(path)}  ({size_mb:.1f} MB)")
    print(f"{'═'*58}\n")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    validate_args(args)
    run(args)