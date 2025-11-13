import cv2
import pandas as pd
import matplotlib.pyplot as plt
import random
import os


video_path = "/home/aristosp/datasets/LRS3/roi/test/0Fi83BHQsMA/00004.mp4"
audio_path = "/home/aristosp/datasets/LRS3/audio/test/0Fi83BHQsMA/00004.wav"
csv_path = "/home/aristosp/datasets/LRS3/audio/aligned/test/0Fi83BHQsMA/00004_viseme_lee.csv"
num_frames = 5

df = pd.read_csv(csv_path)
required_cols = {'Begin', 'End', 'Viseme', 'Label'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")

# --- Pick a viseme label ---
viseme_labels = df['Viseme'].unique().tolist()
phonemes = df['Label'].unique().tolist()
# phoneme = random.choice(phonemes)
phoneme = 'P'
# if viseme_label is None:
# viseme_label = random.choice(viseme_labels)
print(f"Selected phoneme: {phoneme}")

# --- Open the video ---
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- Find intervals for the chosen viseme ---
phoneme_rows = df[df['Label'] == phoneme][['Begin', 'End', 'Viseme']]


# --- Sample timestamps within those intervals ---
begins, ends = [], []
visemes = []
for _, row in phoneme_rows.iterrows():
    b, e, viseme = row['Begin'], row['End'], row['Viseme']  # ← reads df['Label'] explicitly
    visemes.append(viseme)
    begins.append(b)
    ends.append(e)



# --- Read and collect frames ---
frames = []

for begin, end, viseme in zip(begins, ends, visemes):  # use viseme, not phonemes
    start_frame_idx = int(begin * fps)
    end_frame_idx = int(end * fps)

    for frame_idx in range(start_frame_idx, end_frame_idx + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {frame_idx}")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = frame_idx / fps

        # Store frame with its corresponding phoneme and viseme
        frames.append((frame_idx, ts, frame, phoneme, viseme))

cap.release()

# --- Plot frames ---
n = len(frames)
plt.figure(figsize=(4 * n, 5))
for i, (f_idx, ts, frame, phoneme, viseme) in enumerate(frames):
    ax = plt.subplot(1, n, i + 1)
    ax.imshow(frame)
    ax.set_title(f"Frame: {f_idx}", fontsize=14)
    ax.axis("off")

    # Subtitle below image
    subtitle = f"Phoneme: {phoneme}\nViseme: {viseme}\nt = {ts:.2f}s"
    ax.text(
        0.5, -0.01, subtitle, 
        ha='center', va='top', 
        transform=ax.transAxes,
        fontsize=14
    )

plt.tight_layout()
plt.show()

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Plot frames with the same viseme label")
#     parser.add_argument("video", help="Path to .mp4 video file")
#     parser.add_argument("csv", help="Path to alignment CSV file")
#     parser.add_argument("--label", help="Viseme label to plot (optional)")
#     parser.add_argument("--num_frames", type=int, default=5, help="Number of frames to show")

#     args = parser.parse_args()
#     plot_viseme_frames(args.video, args.csv, args.label, args.num_frames)
