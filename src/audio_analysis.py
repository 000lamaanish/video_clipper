import librosa
import numpy as np
import matplotlib.pyplot as plt
from clip_detector import detect_peaks

# Path (since you're running from project root)
audio_path = "../Output/audio.wav"

# Load audio
y, sr = librosa.load(audio_path)

# Frame settings
frame_length = 2048
hop_length = 512

# Compute energy over time
energy = np.array([
    np.sum(np.abs(y[i:i+frame_length])**2)
    for i in range(0, len(y), hop_length)
])

# Normalize
energy = energy / np.max(energy)

# Debug info
print("Energy extracted:", len(energy))
print("Max value:", np.max(energy))
print("Mean energy:", np.mean(energy))
print("Min energy:", np.min(energy))

# Dynamic threshold (important)
threshold = np.mean(energy) + 0.5 * np.std(energy)
print("Using threshold:", threshold)

# Detect peaks
peaks = detect_peaks(energy, threshold=threshold, min_gap=10)

print("\n🔥 Detected highlight segments (frames):")
from clip_detector import merge_segments

clean_peaks = merge_segments(peaks)

print("\n🎯 Clean highlight segments:")
print(clean_peaks)

# Plot graph
plt.figure(figsize=(12, 4))
plt.plot(energy)

# Highlight detected segments
for start, end in peaks:
    plt.axvspan(start, end, alpha=0.3)

plt.title("Audio Energy with Detected Highlights")
plt.xlabel("Time Frames")
plt.ylabel("Normalized Energy")
plt.show()