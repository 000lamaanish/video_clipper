"""
app.py — AI Video Clipper Streamlit UI
========================================
Run with:
    streamlit run app.py

To raise the upload limit permanently, create .streamlit/config.toml:
    [server]
    maxUploadSize = 2048
"""

import os
import sys
import time
import tempfile
import matplotlib.pyplot as plt
import streamlit as st

# make src/ importable from root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from audio_analysis import load_audio, compute_energy, compute_threshold
from clip_detector  import detect_peaks, merge_segments
from transcriber    import load_model, transcribe_segments, get_high_value_segments
from scorer         import rank_segments
from cleaner        import clean_segments, to_timestamps
from video_cutter   import cut_video


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI Video Clipper",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 AI Video Clipper")
st.caption("Automatically detect and extract highlight clips from long videos.")


# ═══════════════════════════════════════════════════════════════
# SIDEBAR — CONFIG
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Clip Settings")
    n_clips      = st.slider("Number of clips",       1,  10,  6)
    min_duration = st.slider("Min clip duration (s)", 10, 60,  30)
    max_duration = st.slider("Max clip duration (s)", 20, 120, 60)
    padding      = st.slider("Padding (s)",           0,  10,  3)

    st.subheader("Detection")
    sensitivity = st.slider(
        "Sensitivity", 0.1, 2.0, 0.5, step=0.1,
        help="Higher = more peaks detected",
    )
    spread = st.slider(
        "Min spread between clips (s)", 30, 300, 120, step=10,
        help="Prevents picking clips from the same moment",
    )

    st.subheader("Scoring Weights")
    weight_energy  = st.slider("Energy weight", 0.0, 1.0, 0.5, step=0.1)
    weight_keyword = round(1.0 - weight_energy, 1)
    st.caption(f"Keyword weight: **{weight_keyword}** (auto)")

    st.subheader("Whisper")
    use_transcribe  = st.toggle("Enable Whisper transcription", value=True)
    whisper_model   = st.selectbox("Model", ["tiny", "base", "small", "medium"], index=1)
    whisper_peaks   = st.slider("Peaks to transcribe", 5, 40, 20)
    whisper_context = st.slider("Context window (s)", 15, 90, 45)

    st.divider()
    st.subheader("Output")
    output_folder = st.text_input("Output folder", value="output/clips")
    audio_path    = st.text_input("Audio path",    value="output/audio.wav")


# ═══════════════════════════════════════════════════════════════
# VIDEO INPUT — UPLOAD OR PATH
# ═══════════════════════════════════════════════════════════════

st.subheader("📁 Input Video")

input_tab1, input_tab2 = st.tabs(["Upload File", "File Path"])

video_path    = None
uploaded_file = None

with input_tab1:
    st.caption("⚠️ Large files (>200MB): use the **File Path** tab instead.")
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mkv", "avi", "mov", "webm"],
    )
    if uploaded_file:
        st.video(uploaded_file)

with input_tab2:
    manual_path = st.text_input(
        "Enter full or relative path to video file",
        placeholder="e.g. input/Avatar1.mkv",
    )
    if manual_path and os.path.exists(manual_path):
        size_gb = os.path.getsize(manual_path) / 1024 / 1024 / 1024
        st.success(f"✅ Found: {manual_path}  ({size_gb:.2f} GB)")
        video_path = manual_path
    elif manual_path:
        st.error(f"❌ File not found: {manual_path}")


# ═══════════════════════════════════════════════════════════════
# RUN BUTTON
# ═══════════════════════════════════════════════════════════════

st.divider()
run_button = st.button("🚀 Generate Clips", type="primary", use_container_width=True)

if run_button:

    # ── Resolve input path ─────────────────────────────────────
    temp_video_path = None

    if uploaded_file:
        suffix = os.path.splitext(uploaded_file.name)[-1]
        tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.read())
        tmp.close()
        temp_video_path = tmp.name
        video_path      = temp_video_path

    if not video_path:
        st.error("❌ Please upload a video or enter a valid file path.")
        st.stop()

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.dirname(audio_path) or "output", exist_ok=True)

    # ── Clean up old clips to avoid rename conflicts ───────────
    for f in os.listdir(output_folder):
        if f.endswith(".mp4"):
            os.remove(os.path.join(output_folder, f))

    total_start  = time.time()
    HOP_LENGTH   = 512
    FRAME_LENGTH = 2048

    # ── Step 1: Audio Extraction ───────────────────────────────
    with st.expander("🎧 Step 1 — Audio Extraction", expanded=True):
        if os.path.exists(audio_path):
            st.info("Audio already exists — skipping extraction.")
        else:
            with st.spinner("Extracting audio …"):
                from moviepy.editor import VideoFileClip
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(audio_path)
                video.close()
            st.success("✅ Audio extracted.")

    # ── Step 2: Energy Analysis ────────────────────────────────
    with st.expander("📊 Step 2 — Energy Analysis", expanded=True):
        with st.spinner("Analyzing audio energy …"):
            y, sr          = load_audio(audio_path)
            energy         = compute_energy(y, sr, FRAME_LENGTH, HOP_LENGTH)
            threshold      = compute_threshold(energy, sensitivity)
            video_duration = len(y) / sr

            target_duration = (min_duration + max_duration) / 2
            WINDOW_FRAMES   = int((target_duration / 2) * sr / HOP_LENGTH)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Duration",    f"{video_duration/60:.1f} min")
        col2.metric("Max Energy",  f"{energy.max():.3f}")
        col3.metric("Mean Energy", f"{energy.mean():.3f}")
        col4.metric("Threshold",   f"{threshold:.3f}")
        st.success("✅ Energy computed.")

    # ── Step 3: Peak Detection ─────────────────────────────────
    with st.expander("🔍 Step 3 — Peak Detection", expanded=True):
        with st.spinner("Detecting peaks …"):
            peaks       = detect_peaks(energy, threshold=threshold, min_gap=10)
            clean_peaks = merge_segments(peaks)

        col1, col2 = st.columns(2)
        col1.metric("Raw peaks",   len(peaks))
        col2.metric("After merge", len(clean_peaks))
        st.success("✅ Peaks detected.")

    # ── Step 4: Whisper Transcription ─────────────────────────
    hype_segments = []

    with st.expander("🧠 Step 4 — Whisper Transcription", expanded=True):
        if not use_transcribe:
            st.info("Whisper disabled — running in energy-only mode.")
        else:
            def frames_to_sec(frame: int) -> float:
                return frame * HOP_LENGTH / sr

            peak_scores = sorted(
                clean_peaks,
                key=lambda p: energy[p[0]:p[1]].mean(),
                reverse=True,
            )
            top_peaks  = peak_scores[:whisper_peaks]
            peak_times = []
            for start_f, end_f in top_peaks:
                centre_s  = (frames_to_sec(start_f) + frames_to_sec(end_f)) / 2
                win_start = max(0.0, centre_s - whisper_context / 2)
                win_end   = min(video_duration, centre_s + whisper_context / 2)
                peak_times.append((win_start, win_end))

            total_s = sum(e - s for s, e in peak_times)
            st.info(f"Transcribing {len(peak_times)} regions (~{total_s:.0f}s of audio) …")

            with st.spinner("Running Whisper …"):
                model         = load_model(whisper_model)
                all_segments  = transcribe_segments(audio_path, peak_times, model)
                hype_segments = get_high_value_segments(all_segments)

            col1, col2 = st.columns(2)
            col1.metric("Total segments", len(all_segments))
            col2.metric("Hype segments",  len(hype_segments))

            if hype_segments:
                st.markdown("**Top keyword hits:**")
                for seg in hype_segments[:8]:
                    st.markdown(
                        f"- `{seg.start:.1f}s → {seg.end:.1f}s` "
                        f"**kw={seg.keyword_score}** — _{seg.text}_"
                    )
            st.success("✅ Transcription complete.")

    # ── Step 5: Scoring ────────────────────────────────────────
    with st.expander("🏆 Step 5 — Scoring & Ranking", expanded=True):
        with st.spinner("Scoring segments …"):
            top_segments = rank_segments(
                energy         = energy,
                peaks          = clean_peaks,
                hype_segments  = hype_segments,
                hop_length     = HOP_LENGTH,
                sr             = sr,
                max_clips      = n_clips,
                window_frames  = WINDOW_FRAMES,
                min_duration_s = 8.0,
                weight_energy  = weight_energy,
                weight_keyword = weight_keyword,
                min_spread_s   = spread,
            )

        if top_segments:
            import pandas as pd
            df = pd.DataFrame([{
                "Start (s)": f"{s.start_sec:.1f}",
                "End (s)":   f"{s.end_sec:.1f}",
                "Duration":  f"{s.duration:.1f}s",
                "Score":     f"{s.combined_score:.3f}",
                "Energy":    f"{s.energy_score:.3f}",
                "Keywords":  f"{s.keyword_score:.3f}",
            } for s in top_segments])
            st.dataframe(df, use_container_width=True)
        st.success(f"✅ {len(top_segments)} segments ranked.")

    # ── Step 6: Cleaning ───────────────────────────────────────
    with st.expander("🧹 Step 6 — Cleaning Clips", expanded=True):
        with st.spinner("Cleaning …"):
            clean = clean_segments(
                segments        = top_segments,
                video_duration  = video_duration,
                min_duration_s  = min_duration,
                max_duration_s  = max_duration,
                padding_s       = padding,
                min_gap_s       = 2.0,
                post_trim_min_s = 15.0,
            )
            final_timestamps = to_timestamps(clean)

        if not final_timestamps:
            st.error("❌ No valid clips after cleaning. Try lowering Min clip duration.")
            st.stop()

        import pandas as pd
        df2 = pd.DataFrame([{
            "Clip":       f"clip_{i:03d}",
            "Start (s)":  f"{s:.1f}",
            "End (s)":    f"{e:.1f}",
            "Duration":   f"{e-s:.1f}s",
        } for i, (s, e) in enumerate(final_timestamps)])
        st.dataframe(df2, use_container_width=True)
        st.success(f"✅ {len(final_timestamps)} clean clip(s) ready.")

    # ── Step 7: Energy Plot ────────────────────────────────────
    with st.expander("📈 Step 7 — Energy Plot", expanded=True):
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(energy, linewidth=0.8, color="steelblue")

        for seg in clean:
            s_f = int(seg.start_sec * sr / HOP_LENGTH)
            e_f = int(seg.end_sec   * sr / HOP_LENGTH)
            ax.axvspan(s_f, e_f, alpha=0.3, color="orange")

        for seg in hype_segments:
            s_f = int(seg.start * sr / HOP_LENGTH)
            e_f = int(seg.end   * sr / HOP_LENGTH)
            ax.axvspan(s_f, e_f, alpha=0.2, color="green")

        ax.set_title("Energy — orange: selected clips | green: whisper hits")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Normalised Energy")
        fig.tight_layout()
        st.pyplot(fig)

    # ── Step 8: Cut Video ──────────────────────────────────────
    with st.expander("✂️ Step 8 — Cutting Video", expanded=True):
        progress = st.progress(0, text="Starting …")
        created  = []

        for i, (start, end) in enumerate(final_timestamps):
            progress.progress(
                int((i / len(final_timestamps)) * 100),
                text=f"Cutting clip {i+1}/{len(final_timestamps)} …",
            )

            # write directly to the correct final filename
            # avoids any rename — cut_video writes clip_000.mp4 etc.
            clip_path = os.path.join(output_folder, f"clip_{i:03d}.mp4")

            import subprocess
            command = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", video_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-avoid_negative_ts", "make_zero",
                clip_path,
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if result.returncode == 0 and os.path.exists(clip_path):
                created.append(clip_path)
            else:
                st.warning(f"⚠️ FFmpeg failed for clip {i}")

        progress.progress(100, text="Done!")
        st.success(f"✅ {len(created)} clip(s) saved to `{output_folder}`")

    # ── Results: Preview all clips ─────────────────────────────
    st.divider()
    st.subheader("🎥 Generated Clips")

    if created:
        cols = st.columns(min(len(created), 3))
        for i, clip_path in enumerate(created):
            col = cols[i % 3]
            with col:
                if os.path.exists(clip_path):
                    size_mb = os.path.getsize(clip_path) / 1024 / 1024
                    st.caption(f"clip_{i:03d}.mp4  ({size_mb:.1f} MB)")
                    st.video(clip_path)
                    with open(clip_path, "rb") as f:
                        st.download_button(
                            label     = f"⬇️ Download clip_{i:03d}",
                            data      = f,
                            file_name = f"clip_{i:03d}.mp4",
                            mime      = "video/mp4",
                            key       = f"dl_{i}",
                        )

    # ── Final summary ──────────────────────────────────────────
    elapsed    = time.time() - total_start
    mins, secs = divmod(int(elapsed), 60)

    st.divider()
    st.success(f"🚀 Done in {mins}m {secs}s — {len(created)} clip(s) generated.")

    # cleanup temp file if uploaded
    if temp_video_path and os.path.exists(temp_video_path):
        os.remove(temp_video_path)