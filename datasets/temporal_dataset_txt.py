import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class TemporalDataset(Dataset):
    def __init__(self, flow_dir, split_file, num_frames, transform):
        """
        Args:
            flow_dir (str): Path to the directory containing optical flow folders (per video).
            split_file (str): Path to the .txt file specifying the split (train, val, or test).
            num_frames (int): Number of flow frames (x and y) to stack.
            transform (callable, optional): Transform to apply to individual frames.
        """
        self.flow_dir = flow_dir
        self.split_file = split_file
        self.num_frames = num_frames
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Load samples from the split file
        self.load_samples()

    def load_samples(self):
        """
        Load samples from the split file and validate their existence.
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
                    
                    folder_path = os.path.join(self.flow_dir, folder_name)
                    if not os.path.isdir(folder_path):
                        print(f"Warning: Video folder not found: {folder_path}")
                        continue

                    # Validate frame existence
                    flow_x_frames = [
                        os.path.join(folder_path, f"flow_x_{i:04d}.jpg")
                        for i in range(self.num_frames)
                    ]
                    flow_y_frames = [
                        os.path.join(folder_path, f"flow_y_{i:04d}.jpg")
                        for i in range(self.num_frames)
                    ]

                    if all(os.path.isfile(f) for f in flow_x_frames + flow_y_frames):
                        self.samples.append((flow_x_frames, flow_y_frames, int(label)))
                    else:
                        print(f"Warning: Missing flow frames in folder {folder_path}")

                except ValueError:
                    print(f"Invalid line in split file: {line.strip()}")
        # Create a class-to-index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
        print(f"Loaded {len(self.samples)} samples from {self.split_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flow_x_frames, flow_y_frames, label = self.samples[idx]

        # Load and transform flow_x and flow_y frames individually
        flow_x = [self._process_frame(frame) for frame in flow_x_frames]
        flow_y = [self._process_frame(frame) for frame in flow_y_frames]

        # Stack flow_x and flow_y frames into a 20-channel tensor
        stacked_flow = np.concatenate(flow_x + flow_y, axis=0)  # Shape: (20, H, W)
        stacked_flow = torch.tensor(stacked_flow, dtype=torch.float32)  # Convert to Tensor

        return stacked_flow, label

    def _process_frame(self, frame_path):
        """
        Load a single optical flow frame, apply transformations, and convert to NumPy array.

        Args:
            frame_path (str): Path to the optical flow frame.

        Returns:
            np.ndarray: Transformed frame as a NumPy array.
        """
        frame = Image.open(frame_path).convert("L")  # Load as grayscale
        if self.transform:
            frame = self.transform(frame)
        return np.array(frame)
