import torch.nn as nn
from torchvision import models
import os
import json

# Base directory dynamically determined for consistent path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MAPPING_PATH = os.path.join(BASE_DIR, "scripts", "class_mapping_temporal.json")


def load_class_mapping(mapping_path=DEFAULT_MAPPING_PATH):
    """
    Load the class-to-index mapping from a JSON file.

    Args:
        mapping_path (str): Path to the class mapping JSON file.

    Returns:
        dict: A dictionary mapping class names to indices.

    Raises:
        FileNotFoundError: If the mapping file does not exist.
    """
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Class mapping file not found: {mapping_path}")

    with open(mapping_path, "r") as f:
        class_mapping = json.load(f)
    
    if not isinstance(class_mapping, dict):
        raise ValueError("Invalid class mapping format. Expected a dictionary.")

    print(f"Loaded class mapping with {len(class_mapping)} classes from {mapping_path}")
    return class_mapping


def count_classes(mapping_path=DEFAULT_MAPPING_PATH):
    """
    Count the number of classes from the saved class mapping file.

    Args:
        mapping_path (str): Path to the class mapping JSON file.

    Returns:
        int: Number of classes.
    """
    class_mapping = load_class_mapping(mapping_path)
    return len(class_mapping)


class TemporalModel(nn.Module):
    def __init__(self, num_classes=None, mapping_path=DEFAULT_MAPPING_PATH):
        """
        Initialize the Temporal Model with ResNet-101 architecture.

        Args:
            num_classes (int, optional): Number of output classes. If None, it is inferred from the class mapping.
            mapping_path (str): Path to the class mapping JSON file.
        """
        super(TemporalModel, self).__init__()
        
        if num_classes is None:
            num_classes = count_classes(mapping_path)

        # Load ResNet101 model pre-trained on ImageNet
        self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

        # Modify the input layer to accept 20 channels (10 x + 10 y frames)
        self.model.conv1 = nn.Conv2d(20, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Final fully connected layer for classification
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        print(f"Initialized Temporal Model with {num_classes} output classes")

    def forward(self, x):
        """
        Forward pass through the Temporal Model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 20, H, W).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        return self.model(x)
