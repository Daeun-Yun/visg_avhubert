import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import librosa
import librosa.display

# --- CONFIGURATION ---
video_path = "/home/aristosp/datasets/LRS3/roi/test/0Fi83BHQsMA/00004.mp4"
audio_path = "/home/aristosp/datasets/LRS3/audio/test/0Fi83BHQsMA/00004.wav"
num_frames_to_plot = 3
frame_size = (96, 96)  # (width, height)

# --- LOAD VIDEO ---
clip = VideoFileClip(video_path)
fps = clip.fps
duration = clip.duration
total_frames = int(fps * duration)

# Choose evenly spaced frames
frame_indices = np.linspace(0, total_frames - 1, num_frames_to_plot, dtype=int)

frames = []
for idx in frame_indices:
    frame_time = idx / fps
    frame = clip.get_frame(frame_time)
    # frame = cv2.resize(frame, frame_size)
    frames.append(frame)

# --- LOAD AUDIO ---
y, sr = librosa.load(audio_path, sr=None)
audio_duration = librosa.get_duration(y=y, sr=sr)

# --- PLOT ---
fig, axes = plt.subplots(2, num_frames_to_plot, figsize=(4 * num_frames_to_plot, 6), dpi=300)


for i, (frame, idx) in enumerate(zip(frames, frame_indices)):
    frame_time = idx / fps

    # Frame
    axes[0, i].imshow(frame.astype(np.uint8))
    axes[0, i].axis("off")

    # Audio segment
    start_sample = int(frame_time / audio_duration * len(y))
    segment_len = int(0.5 * sr)
    end_sample = min(start_sample + segment_len, len(y))
    segment = y[start_sample:end_sample]

    librosa.display.waveshow(segment, sr=sr, ax=axes[1, i], color="gray")
    axes[1, i].set_xlim([0, len(segment) / sr])
    axes[1, i].set_xlabel("")  # remove "Time (s)"
    axes[1, i].set_ylabel("")  # remove amplitude label (if any)
    axes[1, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Keep clean black box border
    for spine in axes[1, i].spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)



plt.tight_layout()
plt.show()