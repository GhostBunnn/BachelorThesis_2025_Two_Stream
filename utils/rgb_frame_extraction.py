import os
import sys

BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)

import cv2
from pathlib import Path


def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extracts frames from a video file at a specified frame rate.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
        frame_rate (int): Number of frames to extract per second (default is 1 FPS).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
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

    Args:
        video_dir (str): Directory containing video files.
        output_root (str): Root directory to save extracted frames for each video.
        frame_rate (int): Number of frames to extract per second (default is 1 FPS).
    """
    video_dir = Path(video_dir)
    output_root = Path(output_root)

    if not video_dir.exists():
        print(f"Error: Video directory {video_dir} does not exist.")
        return

    for video_file in video_dir.glob("*.avi"):  # Change extension if needed
        video_name = video_file.stem  # Get video name without extension
        output_dir = output_root / video_name  # Directory for this video's frames
        extract_frames(str(video_file), str(output_dir), frame_rate)


if __name__ == "__main__":
    # SPATIAL_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "spatial", f"{args.run_id}_spatial_pruned_{args.prune_amount}.pth")
    VIDEO_DATA_PATH = os.path.join(BASE_DIR, "data", "video_data")
    EXTRACTED_FRAMES_PATH = os.path.join(BASE_DIR, "data", "extracted_rgb_frames")

    FRAME_RATE = 1

    extract_frames_from_directory(
        video_dir=str(VIDEO_DATA_PATH),
        output_root=str(EXTRACTED_FRAMES_PATH),
        frame_rate=FRAME_RATE,
    )
