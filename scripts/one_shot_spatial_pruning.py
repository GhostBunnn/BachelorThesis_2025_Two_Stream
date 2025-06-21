import sys
import os

# Dynamically add the base project directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import torch.nn as nn
from torch.nn.utils import prune
from models.spatial_model import load_spatial_model
from datasets.spatial_dataset_txt import RGBDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "best_spatial_model.pth")
PRUNED_MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "pruned_spatial_model.pth")
DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_rgb_frames")
TEST_SPLIT = os.path.join(BASE_DIR, "data", "splits", "testlist01_processed.txt")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define channel importance computation
def compute_channel_importance(layer):
    return torch.norm(layer.weight.data.view(layer.weight.size(0), -1), p=1, dim=1)

def create_adaptive_mask(layer, prune_percentage):
    """
    Create a mask for pruning the output channels of a convolutional layer.

    Args:
        layer (nn.Conv2d): The convolutional layer to be pruned.
        prune_percentage (float): The percentage of output channels to prune.

    Returns:
        torch.Tensor: A boolean mask for the output channels.
    """
    num_output_channels = layer.weight.size(0)  # Number of output channels
    num_prune = int(prune_percentage * num_output_channels)

    importance = compute_channel_importance(layer)
    _, indices = importance.sort()  # Sort importance scores in ascending order
    prune_indices = indices[:num_prune]  # Least important channels

    mask = torch.ones(num_output_channels, dtype=torch.bool, device=layer.weight.device)
    mask[prune_indices] = False  # Mark channels to prune as False

    return mask

def replace_layer_with_pruned_version(layer, mask):
    pruned_layer = nn.Conv2d(
        in_channels=layer.in_channels,
        out_channels=mask.sum().item(),
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=layer.bias is not None
    ).to(layer.weight.device)  # Move to the same device as the original layer

    pruned_layer.weight.data = layer.weight.data[mask, :, :, :].to(layer.weight.device)
    if layer.bias is not None:
        pruned_layer.bias.data = layer.bias.data[mask].to(layer.bias.device)

    return pruned_layer

def replace_bn_with_pruned_version(bn, mask):
    """
    Replace a BatchNorm2D layer with a pruned version.

    Args:
        bn (nn.BatchNorm2d): The BatchNorm layer to replace.
        mask (torch.Tensor): Mask for the pruned channels.
    
    Returns:
        nn.BatchNorm2d: A new BatchNorm layer with updated features.
    """
    # Create a new BatchNorm2D layer with pruned features
    pruned_bn = nn.BatchNorm2d(num_features=mask.sum().item())
    pruned_bn.weight.data = bn.weight.data[mask]
    pruned_bn.bias.data = bn.bias.data[mask]
    pruned_bn.running_mean = bn.running_mean[mask]
    pruned_bn.running_var = bn.running_var[mask]
    return pruned_bn

def prune_layer(layer, mask, next_bn=None):
    """
    Replace the convolutional layer and optionally the BatchNorm layer with pruned versions.

    Args:
        layer (nn.Conv2d): The convolutional layer to prune.
        mask (torch.Tensor): A boolean mask for output channels.
        next_bn (nn.BatchNorm2d, optional): The BatchNorm layer to update.
    """
    # Replace the convolutional layer
    pruned_layer = replace_layer_with_pruned_version(layer, mask)

    if next_bn is not None:
        # Replace the BatchNorm layer
        pruned_bn = replace_bn_with_pruned_version(next_bn, mask)
        return pruned_layer, pruned_bn

    return pruned_layer, None

def prune_channels(model, prune_percentage):
    """
    Prune channels from a model based on channel importance.

    Args:
        model (nn.Module): The model to be pruned.
        prune_percentage (float): Percentage of channels to prune.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Compute the mask for the current convolutional layer
            mask = create_adaptive_mask(module, prune_percentage)
            print(f"Pruning {name}: {module.weight.shape} -> Mask: {mask.shape}")

            # Check if the next layer is a BatchNorm layer
            next_bn = None
            for next_name, next_module in model.named_modules():
                if name in next_name and next_name != name and isinstance(next_module, nn.BatchNorm2d):
                    next_bn = next_module
                    break

            # Prune the layer and update associated BatchNorm, if applicable
            pruned_layer, pruned_bn = prune_layer(module, mask, next_bn)

            # Replace the original layers in the model
            module = pruned_layer
            if next_bn is not None:
                next_bn = pruned_bn
            
def load_spatial_model_with_dropout(num_classes):
    """
    Load a ResNet-101 model with Dropout layers for regularization.
    """
    model = load_spatial_model(num_classes).to(device)  # Original ResNet-101 model
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # Dropout before the final layer
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

def test_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", unit="batch") as test_bar:
            for inputs, labels in test_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                test_bar.set_postfix(loss=f"{loss.item():.4f}")

    test_accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    # Load the pre-trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    model = load_spatial_model_with_dropout(num_classes=101)  # Adjust num_classes to your dataset
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Apply pruning
    prune_percentage = 0.2  # Fraction of filters to prune (e.g., 20%)
    print("Applying channel pruning...")
    prune_channels(model, prune_percentage)
    model.to(device)
    print("Channel pruning completed.")

    # Save the pruned model
    torch.save(model.state_dict(), PRUNED_MODEL_SAVE_PATH)
    print(f"Pruned model saved to {PRUNED_MODEL_SAVE_PATH}")
    
    # Prepare the test dataset and loader
    test_dataset = RGBDataset(DATA_DIR, TEST_SPLIT, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"Testing samples: {len(test_dataset)}")

    # Test the pruned model
    print("Evaluating the pruned model on the test set...")
    test_model(model, test_loader, device)
