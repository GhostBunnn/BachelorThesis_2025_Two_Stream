import os
from collections import defaultdict
import matplotlib.pyplot as plt
import csv

def analyze_class_distribution(data_dir):
    """
    Analyze and visualize the distribution of classes in the dataset.

    Args:
        data_dir (str): Path to the root dataset directory.

    Returns:
        dict: A dictionary with class names as keys and video counts as values.
    """
    # Dictionary to store class counts
    class_counts = defaultdict(int)

    # Group videos by class
    for video_dir in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, video_dir)):
            # Extract class name (e.g., "ApplyEyeMakeup" from "v_ApplyEyeMakeup_g01_c01")
            class_name = video_dir.split("_")[1]
            class_counts[class_name] += 1

    # Print the class distribution
    print("Class Distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

    # Visualize the class distribution
    visualize_distribution(class_counts)

    # Save the distribution to a CSV file
    save_distribution_to_csv(class_counts, "class_distribution.csv")

    return class_counts


def visualize_distribution(class_counts, output_file="class_distribution.png"):
    """
    Visualize the class distribution as a bar chart and save it to a file.

    Args:
        class_counts (dict): Dictionary with class names as keys and counts as values.
        output_file (str): Path to save the bar chart image.
    """
    # Sort classes by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    class_names = [item[0] for item in sorted_classes]
    counts = [item[1] for item in sorted_classes]

    # Plot the distribution
    plt.figure(figsize=(15, 6))
    plt.bar(class_names, counts, color='skyblue')
    plt.xlabel("Classes")
    plt.ylabel("Number of Videos")
    plt.title("Class Distribution in Dataset")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(output_file)
    print(f"Class distribution plot saved to {output_file}")


def save_distribution_to_csv(class_counts, output_file):
    """
    Save the class distribution to a CSV file.

    Args:
        class_counts (dict): Dictionary with class names as keys and counts as values.
        output_file (str): Path to the output CSV file.
    """
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class Name", "Count"])
        for class_name, count in class_counts.items():
            writer.writerow([class_name, count])
    print(f"Class distribution saved to {output_file}")


def analyze_frame_distribution(data_dir):
    """
    Analyze and display the distribution of frame counts across all video subdirectories.

    Args:
        data_dir (str): Path to the root directory containing video subdirectories.

    Returns:
        dict: A dictionary where keys are frame counts and values are the number of videos with that frame count.
    """
    # Dictionary to store frame count distribution
    frame_distribution = defaultdict(int)

    # Iterate over all video subdirectories
    for video_dir in os.listdir(data_dir):
        video_path = os.path.join(data_dir, video_dir)
        if os.path.isdir(video_path):
            # Count the number of frames (files) in the subdirectory
            num_frames = len([f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))])
            frame_distribution[num_frames] += 1

    # Print the frame distribution
    print("Frame Distribution:")
    for frame_count, video_count in sorted(frame_distribution.items()):
        print(f"{video_count} videos have {frame_count} frames")

    # Save the distribution to a CSV file
    save_frame_distribution_to_csv(frame_distribution, "frame_distribution.csv")

    # Plot the frame distribution
    plot_frame_distribution(frame_distribution, "frame_distribution.png")

    return frame_distribution


def save_frame_distribution_to_csv(frame_distribution, output_file):
    """
    Save the frame distribution to a CSV file.

    Args:
        frame_distribution (dict): Dictionary with frame counts as keys and video counts as values.
        output_file (str): Path to the output CSV file.
    """
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame Count", "Number of Videos"])
        for frame_count, video_count in sorted(frame_distribution.items()):
            writer.writerow([frame_count, video_count])
    print(f"Frame distribution saved to {output_file}")


def plot_frame_distribution(frame_distribution, output_file):
    """
    Plot the frame distribution and save it as an image.

    Args:
        frame_distribution (dict): Dictionary with frame counts as keys and video counts as values.
        output_file (str): Path to save the plot image.
    """
    # Sort data by frame count
    frame_counts = sorted(frame_distribution.keys())
    video_counts = [frame_distribution[frame_count] for frame_count in frame_counts]

    # Create the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(frame_counts, video_counts, color='blue', alpha=0.7)
    plt.xlabel("Number of Frames")
    plt.ylabel("Number of Videos")
    plt.title("Frame Distribution Across Videos")
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(output_file)
    print(f"Frame distribution plot saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    # Path to your dataset
    DATA_DIR = "../data/extracted_rgb_frames"

    # Analyze and display frame distribution
    analyze_frame_distribution(DATA_DIR)