"""
video_cutter.py
---------------
Responsible for ONE thing only:
    Cut a video into segments using FFmpeg.

Cloud-safe: automatically finds FFmpeg binary using shutil.which()
and falls back to common install paths if not on system PATH.
"""

import os
import shutil
import subprocess


def _find_ffmpeg() -> str:
    """
    Find FFmpeg binary path.
    Checks PATH first, then common install locations on Linux/Cloud.

    Returns:
        Full path to ffmpeg binary.

    Raises:
        FileNotFoundError if ffmpeg cannot be found anywhere.
    """
    # 1. check system PATH first (works locally)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    # 2. common locations on Linux / Streamlit Cloud
    fallback_paths = [
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",      # macOS homebrew
        "/usr/share/ffmpeg",
    ]
    for path in fallback_paths:
        if os.path.isfile(path):
            return path

    # 3. try imageio-ffmpeg (installed as Python package — no system install needed)
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg and os.path.isfile(ffmpeg):
            return ffmpeg
    except ImportError:
        pass

    raise FileNotFoundError(
        "FFmpeg not found. Install it with:\n"
        "  Linux/Cloud: add 'ffmpeg' to packages.txt\n"
        "  Or add 'imageio-ffmpeg' to requirements.txt"
    )


def cut_video(
    input_video:   str,
    segments:      list[tuple[float, float]],
    output_folder: str,
) -> list[str]:
    """
    Cut a video into segments using FFmpeg.

    Args:
        input_video:   Path to the source video file.
        segments:      List of (start_seconds, end_seconds) tuples.
        output_folder: Directory where clips will be saved.

    Returns:
        List of output file paths that were successfully created.
    """
    os.makedirs(output_folder, exist_ok=True)
    created = []

    # resolve ffmpeg once before the loop
    try:
        ffmpeg_path = _find_ffmpeg()
        print(f"[FFmpeg] Using: {ffmpeg_path}")
    except FileNotFoundError as e:
        print(f"[FFmpeg] ❌ {e}")
        return []

    for i, (start, end) in enumerate(segments):
        if end <= start:
            print(f"⚠️  Skipping segment {i}: end ({end}s) must be after start ({start}s)")
            continue

        output_path = os.path.join(output_folder, f"clip_{i:03d}.mp4")

        command = [
            ffmpeg_path, "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", input_video,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            print(f"❌ FFmpeg failed for clip {i}:\n{result.stderr.decode()}")
        else:
            print(f"✅ Created: {output_path}")
            created.append(output_path)

    return created