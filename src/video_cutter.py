import os
import subprocess

def cut_video(input_video, segments, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for i, (start, end) in enumerate(segments):
        duration = end - start

        output_path = os.path.join(output_folder, f"clip_{i}.mp4")

        command = [
            "ffmpeg",
            "-y",
            "-i", input_video,
            "-ss", str(start),
            "-t", str(duration),
            "-c", "copy",
            output_path
        ]

        subprocess.run(command)

        print(f"✅ Created: {output_path}")