from moviepy.editor import VideoFileClip
import os

video_path = "input/2026-04-17 11-00-37.mkv"
audio_path = "Output/audio.wav"

os.makedirs("Output", exist_ok=True)

if not os.path.exists(video_path):
    print("❌ Video not found")
    exit()

video = VideoFileClip(video_path)
video.audio.write_audiofile(audio_path)

print("✅ Audio extracted successfully!")