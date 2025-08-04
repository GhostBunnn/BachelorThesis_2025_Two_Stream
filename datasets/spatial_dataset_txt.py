import os
from torch.utils.data import Dataset
from PIL import Image


class RGBDataset(Dataset):
    def __init__(self, data_dir, split_file, transform):
        """
        Args:
            data_dir (str): Base directory where extracted RGB frames are stored.
            split_file (str): Path to the .txt file specifying the split (train, val, or test).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.split_file = split_file
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Load samples from split file
        self.load_samples()

    def load_samples(self):
        """
        Load samples from the split file and generate a class-to-index mapping.
        """
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        class_names = set()

        with open(self.split_file, "r") as f:
            for line in f:
                try:
                    folder_name, label = line.strip().split()
                    class_name = folder_name.split("_")[1]  # Extract the class name
                    class_names.add(class_name)

                    folder_path = os.path.join(self.data_dir, folder_name)
                    if not os.path.isdir(folder_path):
                        print(f"Warning: Folder not found: {folder_path}")
                        continue

                    # Collect valid image files from the folder
                    frames = [
                        os.path.join(folder_path, frame)
                        for frame in os.listdir(folder_path)
                        if frame.endswith((".jpg", ".png"))
                    ]

                    if not frames:
                        print(f"Warning: No valid image files found in {folder_path}")
                        continue

                    # Add samples to the dataset
                    for frame_path in frames:
                        self.samples.append((frame_path, int(label)))

                except ValueError:
                    print(f"Invalid line in split file: {line.strip()}")

        # Create a class-to-index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
        print(f"Loaded {len(self.samples)} samples from {self.split_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        """
        frame_path, label = self.samples[idx]

        # Load image
        image = Image.open(frame_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
