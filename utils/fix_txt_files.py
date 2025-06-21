import os
import json

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
SPLIT_DIR = os.path.join(BASE_DIR, "data", "splits")
RGB_DIR = os.path.join(BASE_DIR, "data", "extracted_rgb_frames")

# List of split files to process
split_files = [
    os.path.join(SPLIT_DIR, "trainlist01.txt"),   # modify with 01, 02 or 03 at the end
    os.path.join(SPLIT_DIR, "vallist01.txt"),
    os.path.join(SPLIT_DIR, "testlist01.txt"),
]

# Path to save class-to-index mapping
CLASS_MAPPING_PATH = os.path.join(BASE_DIR, "scripts", "class_mapping.json")


def generate_class_mapping(rgb_dir):
    """
    Generate a class-to-index mapping based on the folder names in the RGB directory.

    Args:
        rgb_dir (str): Path to the RGB frames directory.

    Returns:
        dict: A mapping of class names to indices.
    """
    # Get directory names inside extracted_rgb_frames as classes
    class_dirs = sorted([d for d in os.listdir(rgb_dir) if os.path.isdir(os.path.join(rgb_dir, d))])
    # Extract unique class names (e.g., ApplyEyeMakeup) from directory names
    unique_classes = sorted(set(d.split('_')[1] for d in class_dirs))
    class_mapping = {class_name: idx for idx, class_name in enumerate(unique_classes)}
    print(f"Generated class mapping: {class_mapping}")
    return class_mapping


def process_split_file(split_file, class_to_idx):
    """
    Process a split file to update labels based on class mapping and remove folder paths.

    Args:
        split_file (str): Path to the split file.
        class_to_idx (dict): Mapping of class names to indices.
    """
    processed_lines = []

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        line_parts = line.strip().split(" ")
        video_path = line_parts[0]  # Extract the video path
        folder_name = os.path.basename(video_path).replace('.avi', '')  # Remove .avi extension
        class_name = folder_name.split('_')[1]  # Extract class name (e.g., ApplyEyeMakeup)

        # Debug: Log class_name and check against the class mapping
        if class_name not in class_to_idx:
            print(f"Warning: Class '{class_name}' not found in class mapping. Skipping.")
            continue

        label = class_to_idx[class_name]  # Get the label from the mapping
        processed_lines.append(f"{folder_name} {label}\n")  # Format as: v_ClassName_gXX_cYY N

    # Save the processed lines to a new file
    processed_file = split_file.replace(".txt", "_processed.txt")
    with open(processed_file, "w") as f:
        f.writelines(processed_lines)

    print(f"Processed split file saved to: {processed_file}")


# Generate class-to-index mapping
print("Generating class mapping from RGB frames directory...")
class_to_idx = generate_class_mapping(RGB_DIR)

# Save the mapping to a JSON file
with open(CLASS_MAPPING_PATH, "w") as f:
    json.dump(class_to_idx, f, indent=4)
print(f"Class mapping saved to: {CLASS_MAPPING_PATH}")

# Process each split file
print("\nProcessing split files...")
for split_file in split_files:
    process_split_file(split_file, class_to_idx)
