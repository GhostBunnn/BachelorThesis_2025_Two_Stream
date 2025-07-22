import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(BASE_DIR)

import shutil

def flatten_ucf101_dataset(source_dir, target_dir, copy=False):
    """
    Moves or copies all video files from class subdirectories into one target directory.

    Args:
        source_dir (str): Path to the UCF101 dataset root.
        target_dir (str): Path to the destination folder where all videos will go.
        copy (bool): If True, files are copied. If False, files are moved.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    video_count = 0
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.avi'):
                src_path = os.path.join(root, file)
                
                # Optionally, rename to avoid name collisions (e.g., class_filename.avi)
                class_name = os.path.basename(root)
                new_filename = f"{class_name}_{file}"
                dst_path = os.path.join(target_dir, new_filename)

                if copy:
                    shutil.copy2(src_path, dst_path)
                else:
                    shutil.move(src_path, dst_path)

                video_count += 1

    print(f"Processed {video_count} video files into '{target_dir}'")

# Example usage:
source_directory = os.path.join(BASE_DIR, "UCF101")
target_directory = os.path.join(BASE_DIR, "data", "video_data")

flatten_ucf101_dataset(source_directory, target_directory, copy=False)
