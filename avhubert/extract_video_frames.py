"""
MP4 비디오에서 grayscale 프레임을 미리 추출해 .npy로 저장.
학습 시 실시간 MP4 디코딩 병목을 제거하기 위한 1회성 전처리.

사용법:
    python extract_video_frames.py --tsv_dir /DB/lrs3/433h_data --num_workers 32
"""

import argparse
import os
import glob
import numpy as np
import cv2
from multiprocessing import Pool
from tqdm import tqdm


def collect_video_paths(tsv_dir):
    paths = set()
    for tsv_file in glob.glob(os.path.join(tsv_dir, "*.tsv")):
        with open(tsv_file) as f:
            f.readline()  # skip root line
            for line in f:
                items = line.strip().split("\t")
                if len(items) >= 2:
                    paths.add(items[1])  # absolute video path
    return sorted(paths)


def extract_one(mp4_path):
    npy_path = mp4_path.replace(".mp4", ".npy")
    if os.path.exists(npy_path):
        return "skip"
    try:
        cap = cv2.VideoCapture(mp4_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        if not frames:
            return f"empty: {mp4_path}"
        np.save(npy_path, np.stack(frames))
        return "ok"
    except Exception as e:
        return f"error: {mp4_path} — {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_dir", default="/DB/lrs3/433h_data")
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    print(f"Collecting video paths from {args.tsv_dir} ...")
    paths = collect_video_paths(args.tsv_dir)
    print(f"Total unique videos: {len(paths)}")

    already_done = sum(1 for p in paths if os.path.exists(p.replace(".mp4", ".npy")))
    print(f"Already extracted: {already_done} / {len(paths)}")

    todo = [p for p in paths if not os.path.exists(p.replace(".mp4", ".npy"))]
    if not todo:
        print("All done.")
        return

    print(f"Extracting {len(todo)} videos with {args.num_workers} workers ...")
    errors = []
    with Pool(args.num_workers) as pool:
        for result in tqdm(pool.imap_unordered(extract_one, todo), total=len(todo)):
            if result not in ("ok", "skip"):
                errors.append(result)

    print(f"\nDone. Errors: {len(errors)}")
    if errors:
        for e in errors[:10]:
            print(" ", e)


if __name__ == "__main__":
    main()
