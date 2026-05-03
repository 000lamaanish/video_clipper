import os
import subprocess


def cut_video(input_video: str, segments: list[tuple[float, float]], output_folder: str) -> list[str]:
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

    for i, (start, end) in enumerate(segments):
        if end <= start:
            print(f"⚠️  Skipping segment {i}: end ({end}s) must be after start ({start}s)")
            continue

        output_path = os.path.join(output_folder, f"clip_{i:03d}.mp4")

        command = [
            "ffmpeg", "-y",
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