import torch.nn as nn
import torchvision.models as models
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MAPPING_PATH = os.path.join(BASE_DIR, "scripts", "class_mapping.json")


def load_class_mapping(mapping_path=DEFAULT_MAPPING_PATH):
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Class mapping file not found: {mapping_path}")

    with open(mapping_path, "r") as f:
        class_mapping = json.load(f)
    
    if not isinstance(class_mapping, dict):
        raise ValueError("Invalid class mapping format. Expected a dictionary.")

    print(f"Loaded class mapping with {len(class_mapping)} classes from {mapping_path}")
    return class_mapping


def count_classes(mapping_path=DEFAULT_MAPPING_PATH):
    class_mapping = load_class_mapping(mapping_path)
    return len(class_mapping)


def load_spatial_model(num_classes, freeze_base=True):
    if num_classes <= 0:
        raise ValueError(f"Invalid number of classes: {num_classes}. Must be > 0.")

    # Load the ResNet-101 model pre-trained on ImageNet
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

    # Freeze base layers if transfer learning
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer to match the number of classes
    model.fc = model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.fc.in_features, num_classes)
        )

    print(f"Loaded ResNet101 model with {num_classes} output classes")
    return model
