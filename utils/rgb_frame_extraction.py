import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2
from pathlib import Path


def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a video file at a specified frame rate.
    frame_rate is then umber of frames to extract per second (1 FPS).
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    interval = max(1, fps // frame_rate)

    frame_count = 0
    saved_frame_count = 0

    success, frame = cap.read()
    while success:
        if frame_count % interval == 0:
            output_path = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_frame_count += 1

        frame_count += 1
        success, frame = cap.read()

    cap.release()
    print(f"Extracted {saved_frame_count} frames from {video_path} to {output_dir}")


def extract_frames_from_directory(video_dir, output_root, frame_rate=1):
    """
    Extract frames from all videos in a directory.
    """
    video_dir = Path(video_dir)
    output_root = Path(output_root)

    if not video_dir.exists():
        print(f"Error: Video directory {video_dir} does not exist.")
        return

    for video_file in video_dir.glob("*.avi"):
        video_name = video_file.stem  # Get video name without extension
        output_dir = output_root / video_name
        extract_frames(str(video_file), str(output_dir), frame_rate)


if __name__ == "__main__":
    video_data_path = os.path.join(BASE_DIR, "data", "video_data")
    resulting_frames_path = os.path.join(BASE_DIR, "data", "extracted_rgb_frames")

    frame_rate = 1

    extract_frames_from_directory(
        video_dir=str(video_data_path),
        output_root=str(resulting_frames_path),
        frame_rate=frame_rate,
    )
