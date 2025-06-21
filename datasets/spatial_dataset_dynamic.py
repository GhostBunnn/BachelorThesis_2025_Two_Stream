import os
from torch.utils.data import Dataset
from PIL import Image
import json


class RGBDataset(Dataset):
    def __init__(self, video_dirs, transform=None):
        """
        Args:
            video_dirs (list): List of video directory paths.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.video_dirs = video_dirs
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Build class-to-index mapping and collect samples
        self.build_samples()

    def build_samples(self):
        """
        Build the list of samples and a mapping of class names to numeric indices.
        """
        for video_dir in self.video_dirs:
            # Skip if the directory is empty
            if not os.listdir(video_dir):
                print(f"Skipping empty directory: {video_dir}")
                continue

            # Extract class name from video directory name
            class_name = self.extract_class_name(video_dir)
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)

            class_idx = self.class_to_idx[class_name]

            for frame in os.listdir(video_dir):
                frame_path = os.path.join(video_dir, frame)
                if not os.path.isfile(frame_path):
                    continue
                try:
                    # Validate that the file is a readable image
                    Image.open(frame_path).verify()
                    self.samples.append((frame_path, class_idx))
                except Exception as e:
                    print(f"Skipping invalid frame: {frame_path} ({e})")

        # Save class mapping to JSON for consistency
        mapping_path = "class_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(self.class_to_idx, f, indent=4)
        print(f"Saved class mapping to {mapping_path}")

    def extract_class_name(self, video_dir):
        """
        Extract the class name from a video directory name.
        Assumes naming convention is `v_<ClassName>_<Details>`.
        """
        return os.path.basename(video_dir).split("_")[1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: Transformed image and class index.
        """
        frame_path, label = self.samples[idx]

        # Load image
        image = Image.open(frame_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
